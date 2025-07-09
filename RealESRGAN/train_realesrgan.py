# train_realesrgan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Import your model architecture
from .config import train_config, train_config_minimal
from .rrdbnet_arch import RRDBNet
from .data_preparation import RealESRGANDataset

class RealESRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=config['num_feat'],
            num_block=config['num_block'],
            num_grow_ch=config['num_grow_ch'],
            scale=config['scale']
        ).to(self.device)
        
        # Load pretrained weights if available
        pretrained_path = os.path.join(config['weights_dir'], f'RealESRGAN_x{config["scale"]}.pth')
        if os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)
        
        # Loss functions
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = self.setup_perceptual_loss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.99)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config['lr_milestones'],
            gamma=config['lr_gamma']
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def setup_perceptual_loss(self):
        """Setup VGG-based perceptual loss"""
        from torchvision.models import vgg19
        
        vgg = vgg19(pretrained=True).features[:35].to(self.device)
        for param in vgg.parameters():
            param.requires_grad = False
        
        def perceptual_loss_fn(pred, target):
            pred_features = vgg(pred)
            target_features = vgg(target)
            return nn.functional.mse_loss(pred_features, target_features)
        
        return perceptual_loss_fn
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        
        for lr_imgs, hr_imgs in tqdm(train_loader, desc="Training"):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            sr_imgs = self.model(lr_imgs)
            
            # Calculate losses
            pixel_loss = self.pixel_loss(sr_imgs, hr_imgs)
            perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
            
            # Combined loss
            total_loss = pixel_loss + 0.1 * perceptual_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                sr_imgs = self.model(lr_imgs)
                loss = self.pixel_loss(sr_imgs, hr_imgs)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            
        # Save model weights only (for inference)
        weights_path = os.path.join(self.config['weights_dir'], f'RealESRGAN_x{self.config["scale"]}.pth')
        torch.save({
            'params': self.model.state_dict(),
            'scale': self.config['scale']
        }, weights_path)
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print progress
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
            
            # Save sample images
            if (epoch + 1) % self.config['sample_freq'] == 0:
                self.save_sample_images(val_loader, epoch + 1)
    
    def save_sample_images(self, val_loader, epoch):
        """Save sample images for visual inspection"""
        self.model.eval()
        with torch.no_grad():
            lr_imgs, hr_imgs = next(iter(val_loader))
            lr_imgs = lr_imgs[:4].to(self.device)  # First 4 images
            hr_imgs = hr_imgs[:4].to(self.device)
            
            sr_imgs = self.model(lr_imgs)
            
            # Create comparison grid
            comparison = torch.cat([lr_imgs, sr_imgs, hr_imgs], dim=0)
            save_path = os.path.join(self.config['sample_dir'], f'epoch_{epoch}_comparison.png')
            save_image(comparison, save_path, nrow=4, normalize=True)
    
    def load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights for transfer learning"""
        pretrained = torch.load(pretrained_path)
        
        if 'params' in pretrained:
            pretrained_dict = pretrained['params']
        else:
            pretrained_dict = pretrained
        
        model_dict = self.model.state_dict()
        
        # Filter out unnecessary keys and size mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.size() == model_dict[k].size()}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)} layers from pretrained model")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    return epoch, loss

# Training configuration
config = {
    'scale': 4,
    'num_feat': 64,
    'num_block': 23,
    'num_grow_ch': 32,
    'learning_rate': 2e-4,
    'lr_milestones': [50000, 100000, 200000, 300000],
    'lr_gamma': 0.5,
    'batch_size': 16,
    'patch_size': 192,
    'num_epochs': 500,
    'save_freq': 10,
    'sample_freq': 5,
    'checkpoint_dir': 'checkpoints',
    'weights_dir': 'weights',
    'sample_dir': 'samples'
}

# Main training function
def main():
    # Create directories
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
    os.makedirs(train_config['weights_dir'], exist_ok=True)
    os.makedirs(train_config['sample_dir'], exist_ok=True)
    
    # Create datasets
    train_dataset = RealESRGANDataset(
        hr_dir='training_data/high_res',
        scale=train_config['scale'],
        patch_size=train_config['patch_size'],
        augment=True
    )
    
    val_dataset = RealESRGANDataset(
        hr_dir='training_data/validation/high_res',
        scale=train_config['scale'],
        patch_size=train_config['patch_size'],
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = RealESRGANTrainer(train_config)
    trainer = RealESRGANTrainer(train_config_minimal)
    
    # RESUME FROM CHECKPOINT (if needed)
    start_epoch = 0
    checkpoint_path = 'checkpoints/checkpoint_epoch_[the checkpoint num].pth'  # Change this the checkpoint desired
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, _ = load_checkpoint(
            trainer.model, 
            trainer.optimizer, 
            trainer.scheduler, 
            checkpoint_path
        )
    
    # Start training (adjust epochs if resuming)
    remaining_epochs = config['num_epochs'] - start_epoch
    if remaining_epochs > 0:
        trainer.train(train_loader, val_loader, remaining_epochs)
    else:
        print("Training already completed!")

if __name__ == "__main__":
    main()