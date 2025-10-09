"""
MAE Loss Functions for Medical Images
Based on models_mae.py
"""
import torch
import torch.nn as nn


class MAELoss(nn.Module):
    """
    Masked Autoencoder Loss
    Computes reconstruction loss on masked patches only
    """
    def __init__(self, patch_size=16, in_chans=1, norm_pix_loss=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
    
    def patchify(self, imgs):
        """
        Convert images to patches
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        c = imgs.shape[1]  # Get channel count dynamically
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x
    
    def forward(self, imgs, pred, mask):
        """
        Compute reconstruction loss on masked patches
        Args:
            imgs: [N, C, H, W] original images
            pred: [N, L, p*p*C] predicted pixel values
            mask: [N, L] binary mask (0=keep, 1=remove)
        Returns:
            loss: scalar reconstruction loss
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


class MAELossWithVisualization(MAELoss):
    """
    MAE Loss with additional outputs for visualization
    """
    def __init__(self, patch_size=16, in_chans=1, norm_pix_loss=False):
        super().__init__(patch_size, in_chans, norm_pix_loss)
    
    def unpatchify(self, x):
        """
        Convert patches back to image
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        c = self.in_chans
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, imgs, pred, mask, return_visualization=False):
        """
        Compute loss and optionally return visualization data
        Args:
            imgs: [N, C, H, W] original images
            pred: [N, L, p*p*C] predicted pixel values
            mask: [N, L] binary mask
            return_visualization: if True, return additional data for visualization
        Returns:
            loss: scalar reconstruction loss
            (optional) dict with visualization data
        """
        loss = super().forward(imgs, pred, mask)
        
        if return_visualization:
            # Create visualization data
            y = self.unpatchify(pred)  # [N, C, H, W]
            
            # Expand mask to image dimensions
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2 * self.in_chans)
            mask_img = self.unpatchify(mask_expanded)  # [N, C, H, W]
            
            # Create masked image and reconstruction
            im_masked = imgs * (1 - mask_img)
            im_paste = imgs * (1 - mask_img) + y * mask_img
            
            viz_data = {
                'original': imgs,
                'masked': im_masked,
                'reconstructed': y,
                'reconstruction_paste': im_paste,
                'mask': mask_img
            }
            return loss, viz_data
        
        return loss

