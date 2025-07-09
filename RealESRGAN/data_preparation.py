# data_preparation.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class RealESRGANDataset(Dataset):
    def __init__(self, hr_dir, scale=4, patch_size=192, augment=True):
        self.hr_dir = hr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        
        # Get all image files
        self.hr_images = [f for f in os.listdir(hr_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Augmentation transforms
        if augment:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
            ])
        else:
            self.transforms = None
    
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        # Load HR image
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Convert to numpy
        hr_array = np.array(hr_img)
        
        # Random crop to patch_size
        h, w = hr_array.shape[:2]
        if h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            hr_patch = hr_array[top:top+self.patch_size, left:left+self.patch_size]
        else:
            hr_patch = hr_array
        
        # Generate LR image using degradation
        lr_patch = self.degrade_image(hr_patch)
        
        # Apply augmentations
        if self.transforms:
            # Convert to PIL for transforms
            hr_pil = Image.fromarray(hr_patch)
            lr_pil = Image.fromarray(lr_patch)
            
            # Apply same transform to both
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            hr_pil = self.transforms(hr_pil)
            torch.manual_seed(seed)
            lr_pil = self.transforms(lr_pil)
            
            hr_patch = np.array(hr_pil)
            lr_patch = np.array(lr_pil)
        
        # Convert to tensors and normalize
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor
    
    def degrade_image(self, hr_image):
        """Apply realistic degradation to create LR image"""
        # Resize down then up to create basic LR
        h, w = hr_image.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        
        # Downscale with random kernel
        if np.random.random() < 0.5:
            # Bicubic downscaling
            lr_img = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Area downscaling
            lr_img = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, np.random.uniform(0, 25), lr_img.shape)
            lr_img = np.clip(lr_img + noise, 0, 255)
        
        # Add JPEG compression artifacts
        if np.random.random() < 0.4:
            quality = np.random.randint(30, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', lr_img, encode_param)
            lr_img = cv2.imdecode(encimg, 1)
        
        return lr_img.astype(np.uint8)