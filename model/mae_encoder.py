"""
3D MAE Encoder for Medical Volume Data
"""
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

from logger import get_logger
from .patch_embed import PatchEmbed
from .pos_embed import get_sincos_pos_embed

logger = get_logger("mae_encoder")


class MAEEncoder3D(nn.Module):
    """
    3D MAE Encoder with VisionTransformer backbone
    Handles 3D medical volume data (D×H×W)
    """
    def __init__(
        self,
        volume_size=160,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        norm_layer=None, drop_rate=0.,
        **kwargs
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # 确保volume_size和patch_size为tuple
        if isinstance(volume_size, int):
            volume_size = (volume_size, volume_size, volume_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_embed = PatchEmbed(volume_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        logger.info(f"MAE Encoder 3D:")
        logger.info(f"  Volume size: {volume_size}")
        logger.info(f"  Patch size: {patch_size}³")
        logger.info(f"  Grid size: {self.patch_embed.grid_size}")
        logger.info(f"  Number of patches: {num_patches}")
        logger.info(f"  Embed dim: {embed_dim}")
        logger.info(f"  Dropout: {drop_rate}")
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, proj_drop=drop_rate, attn_drop=drop_rate)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.volume_size = volume_size
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        pos_embed = get_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        logger.info("Initialized 3D positional embeddings")
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ratio=0.75):
        """
        Forward pass with masking and skip connections for U-Net
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # Apply masking only if mask_ratio > 0
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            # If not masking, the "mask" is all ones (no patches removed), and ids_restore is not needed.
            N, L, D = x.shape
            mask = torch.ones(N, L, device=x.device)
            ids_restore = None

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        skip_connections = []
        num_blocks = len(self.blocks)
        # Collect skip connections after each quarter of blocks (4 stages)
        skip_indices = [num_blocks // 4, num_blocks // 2, 3 * num_blocks // 4, num_blocks]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i + 1) in skip_indices:
                # Only collect skip connections if not masking (full volume)
                if mask_ratio == 0:
                    # Remove cls token, reshape for CNN decoder
                    skip = x[:, 1:, :]
                    d, h, w = self.patch_embed.grid_size
                    skip = skip.reshape(skip.shape[0], d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
                    skip_connections.append(skip)
                else:
                    skip_connections.append(None)
        x = self.norm(x)
        return x, mask, ids_restore, skip_connections

    def unpatchify(self, x):
        """
        Convert patch tokens back to 3D feature volume for CNN decoder
        x: (N, num_patches, embed_dim)
        Returns: (N, embed_dim, D, H, W)
        """
        # 去除cls token（如果有）
        if x.shape[1] != self.patch_embed.num_patches:
            x = x[:, -self.patch_embed.num_patches:, :]
        grid_size = self.patch_embed.grid_size
        embed_dim = x.shape[-1]
        N = x.shape[0]
        D, H, W = grid_size
        x = x.transpose(1, 2)  # (N, embed_dim, num_patches)
        x = x.reshape(N, embed_dim, D, H, W)
        return x
