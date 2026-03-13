"""
3D MAE Decoder for Medical Volume Data
"""
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

from logger import get_logger
from model.tools.pos_embed import get_sincos_pos_embed

logger = get_logger("mae_decoder")


class MAEDecoder(nn.Module):
    """
    3D MAE Decoder with VisionTransformer backbone
    Reconstructs masked 3D patches from latent representations
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
        
        if isinstance(num_patches, int):
            n_cubic = int(round(num_patches ** (1/3)))
            self.grid_size = (n_cubic, n_cubic, n_cubic)
            num_patches_total = num_patches
        elif isinstance(num_patches, (tuple, list)):
            self.grid_size = tuple(num_patches)
            num_patches_total = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            raise ValueError('num_patches must be an int or a tuple/list of length 3')

        logger.info(f"MAE Decoder 3D:")
        logger.info(f"  Number of patches: {num_patches_total}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Patch size: {patch_size}³")
        logger.info(f"  Decoder embed dim: {decoder_embed_dim}")
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_total + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Standard MAE Linear Prediction
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True)
        
        self.num_patches = num_patches_total
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        decoder_pos_embed = get_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.grid_size,
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        logger.info("Initialized 3D decoder positional embeddings")
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        
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
        """
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x = x + self.decoder_pos_embed
        
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
        Convert 3D patches back to volume
        x: (N, L, patch_size**3 * C)
        volumes: (N, C, D, H, W)
        """
        p = self.patch_size
        g = self.grid_size
        c = self.in_chans
        
        assert x.shape[1] == g[0]*g[1]*g[2], f"Expected {g[0]*g[1]*g[2]} patches, got {x.shape[1]}"
        
        # (N, L, p*p*p*c) -> (N, d, h, w, p, p, p, c)
        x = x.reshape(shape=(x.shape[0], g[0], g[1], g[2], p, p, p, c))
        
        # (N, d, h, w, p, p, p, c) -> (N, c, d, p, h, p, w, p)
        x = torch.einsum('ndhwpqrc->ncdphqwr', x)
        
        # (N, c, d*p, h*p, w*p)
        volumes = x.reshape(shape=(x.shape[0], c, g[0] * p, g[1] * p, g[2] * p))
        
        return volumes