"""
MAE Decoder for Medical Images
Based on simple_mae_medical.py and models_mae.py
"""
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

from logger import get_logger

logger = get_logger("mae_decoder")

try:
    from .pos_embed import get_2d_sincos_pos_embed
except Exception:
    logger.warning("Could not import local pos_embed utility")
    get_2d_sincos_pos_embed = None


class MAEDecoder(nn.Module):
    """
    MAE Decoder with VisionTransformer backbone
    Reconstructs masked patches from latent representations
    """
    def __init__(
        self,
        num_patches,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=None,
        **kwargs
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # Decoder embedding
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Position embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        
        # Store parameters
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        # Initialize position embeddings with sin-cos embedding
        if get_2d_sincos_pos_embed is not None:
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.num_patches**.5),
                cls_token=True
            )
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
            )
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, ids_restore):
        """
        Forward pass to reconstruct masked patches
        Args:
            x: [N, L_keep+1, D] encoded latent features (with cls token)
            ids_restore: [N, L] indices to restore original order
        Returns:
            pred: [N, L, patch_size**2 * in_chans] predicted pixel values
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add position embeddings
        x = x + self.decoder_pos_embed
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
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

