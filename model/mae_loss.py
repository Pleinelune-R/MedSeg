"""
3D MAE Loss Functions for Medical Volumes
"""
import torch
import torch.nn as nn


class MAELoss(nn.Module):
    """
    3D Masked Autoencoder Loss
    Computes reconstruction loss on masked patches only
    """
    def __init__(self, patch_size=16, in_chans=1, norm_pix_loss=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
    
    def patchify(self, volumes):
        """
        Convert 3D volumes to patches
        """
        p = self.patch_size
        N, C, D, H, W = volumes.shape
        assert D % p == 0 and H % p == 0 and W % p == 0, f"Volume dimensions must be divisible by patch_size"
        
        g1, g2, g3 = D // p, H // p, W // p
        
        x = volumes.reshape(shape=(N, C, g1, p, g2, p, g3, p))
        x = torch.einsum('ncaxbydz->nabdxyzc', x)
        x = x.reshape(shape=(N, g1*g2*g3, p**3 * C))
        
        return x
    
    def forward(self, volumes, pred, mask):
        """
        Compute reconstruction loss on masked patches
        """
        target = self.patchify(volumes)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        loss = (loss * mask).sum() / mask.sum()
        return loss


class MAELossWithVisualization(MAELoss):
    """
    3D MAE Loss with additional outputs for visualization
    Extracts 2D slices from 3D volumes for visualization
    """
    def __init__(self, patch_size=16, in_chans=1, norm_pix_loss=False):
        super().__init__(patch_size, in_chans, norm_pix_loss)
    
    def unpatchify(self, x, grid_size):
        """
        Convert 3D patches back to volume
        """
        p = self.patch_size
        c = self.in_chans
        g1, g2, g3 = grid_size
        
        assert g1 * g2 * g3 == x.shape[1], f"Expected {g1 * g2 * g3} patches, got {x.shape[1]}"
        
        x = x.reshape(shape=(x.shape[0], g1, g2, g3, p, p, p, c))
        x = torch.einsum('nabdxyzc->ncaxbydz', x)
        volumes = x.reshape(shape=(x.shape[0], c, g1 * p, g2 * p, g3 * p))
        
        return volumes
    
    def forward(self, volumes, pred, mask, return_visualization=False):
        """
        Compute loss and optionally return visualization data
        """
        # Re-implement forward to capture mean/var for denormalization
        target = self.patchify(volumes)
        
        mean = None
        var = None
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        if return_visualization:
            # Calculate grid_size dynamically from the input volume
            grid_size = tuple([d // self.patch_size for d in volumes.shape[2:]])
            
            # Denormalize prediction if needed
            pred_viz = pred
            if self.norm_pix_loss and mean is not None:
                pred_viz = pred * (var + 1.e-6)**.5 + mean
            
            y = self.unpatchify(pred_viz, grid_size)
            
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, self.patch_size**3 * self.in_chans)
            mask_vol = self.unpatchify(mask_expanded, grid_size)
            
            vol_masked = volumes * (1 - mask_vol)
            vol_paste = volumes * (1 - mask_vol) + y * mask_vol
            
            mid_slice = volumes.shape[2] // 2
            
            viz_data = {
                'original': volumes[:, :, mid_slice, :, :],
                'masked': vol_masked[:, :, mid_slice, :, :],
                'reconstructed': y[:, :, mid_slice, :, :],
                'reconstruction_paste': vol_paste[:, :, mid_slice, :, :],
                'mask': mask_vol[:, :, mid_slice, :, :],
                'original_3d': volumes,
                'reconstructed_3d': y,
                'reconstruction_paste_3d': vol_paste
            }
            return loss, viz_data
        
        return loss
