import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import time
import cv2
from huggingface_hub import hf_hub_url, cached_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image
from .progress import ProgressTracker

HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=scale
        )
        
    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_filename = os.path.basename(model_path)
            config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
            cached_download(config_file_url, cache_dir=cache_dir, force_filename=local_filename)
            print('Weights downloaded to:', os.path.join(cache_dir, local_filename))
        
        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        self.model.to(self.device)
        
    @torch.cuda.amp.autocast()
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
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
    
    
    @torch.cuda.amp.autocast()
    def predict_with_progress(self, lr_image, batch_size=4, patches_size=192,
                            padding=24, pad_size=15, progress_callback=None):
        """
        Predict with progress tracking
        
        Args:
            progress_callback: Function that takes (progress_float, message_string)
        """
        if progress_callback is None:
            return self.predict(lr_image, batch_size, patches_size, padding, pad_size)
        
        # Initialize progress tracker
        tracker = ProgressTracker(progress_callback)
        
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        
        # Stage 1: Preparation
        tracker.update(0.0, 0, 0, "preparing")
        
        lr_image = pad_reflect(lr_image, pad_size)
        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        
        total_patches = patches.shape[0]
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()
        
        # Stage 2: Processing patches
        tracker.update(0.1, 0, total_patches, "processing")
        
        processed_patches = 0
        
        with torch.no_grad():
            # Process first batch
            res = self.model(img[0:batch_size])
            processed_patches = min(batch_size, total_patches)
            
            progress = 0.1 + (processed_patches / total_patches) * 0.7
            tracker.update(progress, processed_patches, total_patches, "processing")
            
            # Process remaining batches
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)
                processed_patches = min(i + batch_size, total_patches)
                
                progress = 0.1 + (processed_patches / total_patches) * 0.7
                remaining = tracker.calculate_remaining_time(processed_patches, total_patches)
                tracker.update(progress, processed_patches, total_patches, "processing", remaining)

        # Stage 3: Reconstruction
        tracker.update(0.85, total_patches, total_patches, "reconstructing")
        
        sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled, 
            target_shape=scaled_image_shape, padding_size=padding * scale
        )
        
        # Stage 4: Finalizing
        tracker.update(0.95, total_patches, total_patches, "finalizing")
        
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size*scale)
        sr_img = Image.fromarray(sr_img)
        
        # Complete
        total_time = time.time() - tracker.start_time
        tracker.update(1.0, total_patches, total_patches, "complete", elapsed=total_time)
        
        return sr_img