from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep
from torch import nn
from .vit_moe import ViT_MoE_Encoder

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels: int = 4,
        embed_dim: int = 96,
        patch_size: int = 16,
        depth: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 384,
        num_modalities: int = 4,
        dropout: float = 0.1,
         **kwargs
    ):
        super().__init__()
        self.vit_moe = ViT_MoE_Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,  # Each modality processed separately
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_modalities=num_modalities,
            dropout=dropout
        )

    def forward(self, x):
        # x: (B, 4, D, H, W)
        out = self.vit_moe(x)  # (B, 4, C, d, h, w)
        return out
