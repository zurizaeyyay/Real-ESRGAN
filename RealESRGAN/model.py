import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import time
import cv2
from huggingface_hub import hf_hub_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image
from .progress import ProgressTracker
from .config import HF_MODELS


class RealESRGAN:
    def __init__(self, device, scale=4, use_attention=False, resample_mode='bicubic'):
        self.device = device
        self.scale = scale
        self.use_attention = use_attention
        self.resample_mode=resample_mode
        
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=scale,
            use_attention=use_attention,
            resample_mode=resample_mode
        )
        
    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_name = os.path.basename(model_path)
            # Downloads cache and weight data to same folder
            downloaded_path = hf_hub_download(repo_id=config['repo_id'],
                            filename=config['filename'], 
                            cache_dir=cache_dir, 
                            local_dir=cache_dir,
                            resume_download=True,  # Resume interrupted downloads
                            force_download=False   # Use cached if available
                            )
            print(f'Weights downloaded to: {downloaded_path}')
        
        try:
            if self.device.type == 'cuda':
                loadnet = torch.load(model_path, map_location=self.device)
            else:
                loadnet = torch.load(model_path, map_location='cpu')
            
            if 'params' in loadnet:
                pretrained_dict = loadnet['params']
            elif 'params_ema' in loadnet:
                pretrained_dict = loadnet['params_ema']
            else:
                pretrained_dict = loadnet
            
            # Check if pretrained model has attention layers
            has_attention = any('attention' in k for k in pretrained_dict.keys())
            
            
            # fall back code for old weights and model incompatibility
            if has_attention and not self.use_attention:
                print("Warning: Pretrained weights has no attention layers but current model does")
            elif not has_attention and self.use_attention:
                print("Warning: Current model has attention but pretrained weights doesn't. Attention layers will be randomly initialized.")
            
            # Smart loading based on architecture compatibility
            if has_attention == self.use_attention:
                # Perfect match - load normally
                self.model.load_state_dict(pretrained_dict, strict=True)
                print("Loaded weights with perfect architecture match")
            else:
                # Partial match - load compatible layers only
                model_dict = self.model.state_dict()
                compatible_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and v.size() == model_dict[k].size()}
                
                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict, strict=False)
                print(f"Loaded {len(compatible_dict)}/{len(pretrained_dict)} compatible weights")
                
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            raise    
            
        self.model.eval()
        self.model.to(self.device)
        
    def _get_autocast_kwargs(self):
        """Get appropriate autocast kwargs based on device"""
        if self.device.type == 'cuda':
            return {'device_type': 'cuda', 'dtype': torch.float16, 'enabled': True}
        elif self.device.type == 'mps':  # Apple Silicon
            return {'device_type': 'cpu', 'dtype': torch.float16, 'enabled': False}
        else:  # CPU
            return {'device_type': 'cpu', 'dtype': torch.float32, 'enabled': False}
    
    def set_resample_mode(self, resample_mode):
        """Update the resampling mode"""
        valid_modes = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area']
        if resample_mode not in valid_modes:
            raise ValueError(f"Invalid resample mode: {resample_mode}. Valid modes: {valid_modes}")
        
        self.resample_mode = resample_mode
        print(f"Resample mode updated to: {resample_mode}")
    
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        with torch.amp.autocast(**self._get_autocast_kwargs()):
            scale = self.scale
            device = self.device
            lr_image = np.array(lr_image)
            lr_image = pad_reflect(lr_image, pad_size)

            patches, p_shape = split_image_into_overlapping_patches(
                lr_image, patch_size=patches_size, padding_size=padding
            )
            img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()

            with torch.no_grad():
                res = self.model(img[0:batch_size])
                for i in range(batch_size, img.shape[0], batch_size):
                    res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

            sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
            np_sr_image = sr_image.numpy()

            padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
            scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
            np_sr_image = stich_together(
                np_sr_image, padded_image_shape=padded_size_scaled, 
                target_shape=scaled_image_shape, padding_size=padding * scale
            )
            sr_img = (np_sr_image*255).astype(np.uint8)
            sr_img = unpad_image(sr_img, pad_size*scale)
            sr_img = Image.fromarray(sr_img)

            return sr_img
        
    async def predict_with_progress(self, lr_image, batch_size=4, patches_size=192,
                            padding=24, pad_size=15, progress_callback=None):
        """
        Predict with progress tracking
        
        Args:
            progress_callback: Function that takes (progress_float, message_string)
        """
        with torch.amp.autocast(**self._get_autocast_kwargs()):
            if progress_callback is None:
                return self.predict(lr_image, batch_size, patches_size, padding, pad_size)
            
            # Initialize progress tracker
            tracker = ProgressTracker(progress_callback)
            
            scale = self.scale
            device = self.device
            lr_image = np.array(lr_image)
            
            # Stage 1: Preparation
            await tracker.update(0.0, 0, 0, "preparing")
            
            lr_image = pad_reflect(lr_image, pad_size)
            patches, p_shape = split_image_into_overlapping_patches(
                lr_image, patch_size=patches_size, padding_size=padding
            )
            
            total_patches = patches.shape[0]
            img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()
            
            # Stage 2: Processing patches
            await tracker.update(0.1, 0, total_patches, "processing")
            
            processed_patches = 0
            
            with torch.no_grad():
                # Process first batch
                res = self.model(img[0:batch_size])
                processed_patches = min(batch_size, total_patches)
                
                progress = 0.1 + (processed_patches / total_patches) * 0.7
                await tracker.update(progress, processed_patches, total_patches, "processing")
                
                # Process remaining batches
                for i in range(batch_size, img.shape[0], batch_size):
                    res = torch.cat((res, self.model(img[i:i+batch_size])), 0)
                    processed_patches = min(i + batch_size, total_patches)
                    
                    progress = 0.1 + (processed_patches / total_patches) * 0.7
                    remaining = tracker.calculate_remaining_time(processed_patches, total_patches)
                    await tracker.update(progress, processed_patches, total_patches, "processing", remaining)

            # Stage 3: Reconstruction
            await tracker.update(0.85, total_patches, total_patches, "reconstructing")
            
            sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
            np_sr_image = sr_image.numpy()

            padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
            scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
            np_sr_image = stich_together(
                np_sr_image, padded_image_shape=padded_size_scaled, 
                target_shape=scaled_image_shape, padding_size=padding * scale
            )
            
            # Stage 4: Finalizing
            await tracker.update(0.95, total_patches, total_patches, "finalizing")
            
            sr_img = (np_sr_image*255).astype(np.uint8)
            sr_img = unpad_image(sr_img, pad_size*scale)
            sr_img = Image.fromarray(sr_img)
            
            # Complete
            total_time = time.time() - tracker.start_time
            await tracker.update(1.0, total_patches, total_patches, "complete", elapsed=total_time)
            
            return sr_img
        
    
class EnhancedRealESRGAN(RealESRGAN):
    def __init__(self, device, scale=4, model_variant='standard'):
        self.model_variant = model_variant
        super().__init__(device, scale)
        
        # Different model configurations
        if model_variant == 'lightweight':
            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=32, 
                num_block=12, num_grow_ch=16, scale=scale
            )
        elif model_variant == 'enhanced':
            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=96, 
                num_block=32, num_grow_ch=48, scale=scale
            )
        else:  # standard
            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, 
                num_block=23, num_grow_ch=32, scale=scale
            )
    