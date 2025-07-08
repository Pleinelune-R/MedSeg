import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from typing import Optional, List

class MLP_MoE(nn.Module):
    """
    Mixture of Experts (MOE) Feed-Forward Network for ViT block.
    Each modality uses its own MOE instance.
    """
    def __init__(self, embed_dim, ffn_dim, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        # Gating network: simple softmax over experts
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x, modality_idx):
        # x: (B, N, C), modality_idx: int in [0, num_experts-1]
        # Only use the expert for the given modality
        return self.experts[modality_idx](x)

class TransformerEncoderLayerMoE(nn.Module):
    """
    ViT Transformer Encoder Layer with MOE FFN.
    Attention and LayerNorm are shared; MOE is per-modality.
    """
    def __init__(self, embed_dim, num_heads, ffn_dim, num_modalities=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.moe_ffn = nn.ModuleList([
            MLP_MoE(embed_dim, ffn_dim, num_experts=1, dropout=dropout) for _ in range(num_modalities)
        ])
        self.dropout = nn.Dropout(dropout)
        self.num_modalities = num_modalities

    def forward(self, x, modality_idx):
        # x: (B, N, C)
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = h + self.dropout(x)
        h = x
        x = self.norm2(x)
        # MOE: select the correct modality's FFN
        x = self.moe_ffn[modality_idx](x, 0)
        x = h + self.dropout(x)
        return x

class ViT_MoE_Encoder(nn.Module):
    """
    Vision Transformer Encoder with shared transformer layers and per-modality MOE FFN.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, ffn_dim, num_modalities=4, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.patch_embed = nn.Conv3d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerMoE(embed_dim, num_heads, ffn_dim, num_modalities, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

    def forward(self, x):
        # x: (B, 4, D, H, W)
        outputs = []
        for m in range(self.num_modalities):
            xm = x[:, m:m+1]  # (B, 1, D, H, W)
            x_patch = self.patch_embed(xm)  # (B, C, d, h, w)
            # flatten spatial dims to sequence
            B, C, d, h, w = x_patch.shape
            x_patch_seq = x_patch.flatten(2).transpose(1, 2)  # (B, N, C)
            x_patch_seq = x_patch_seq + self.pos_embed
            for layer in self.transformer_layers:
                x_patch_seq = layer(x_patch_seq, m)
            x_patch_seq = self.norm(x_patch_seq)
            # reshape back to (B, C, d, h, w)
            x_patch_out = x_patch_seq.transpose(1, 2).reshape(B, C, d, h, w)
            outputs.append(x_patch_out.unsqueeze(1))  # (B,1,C,d,h,w)
        # Concatenate along modality (channel) dimension
        out = torch.cat(outputs, dim=1)  # (B, 4, C, d, h, w)
        return out

class ViT_Decoder(nn.Module):
    """
    Simple ViT-based decoder for segmentation.
    """
    def __init__(self, in_channels, embed_dim, patch_size, img_size, depth, num_heads, ffn_dim, out_channels=4, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_channels)
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, 4, D, H, W)
        x_patch = self.patch_embed(x)  # (B, C, d, h, w)
        x_patch = x_patch.flatten(2).transpose(1, 2)  # (B, N, C)
        x_patch = x_patch + self.pos_embed
        for layer in self.transformer_layers:
            x_patch = layer(x_patch)
        x_patch = self.norm(x_patch)
        # Project to output channels
        x_patch = self.head(x_patch)  # (B, N, 4)
        # Reshape back to (B, 4, D, H, W)
        B = x.shape[0]
        d, h, w = [s // self.patch_size for s in self.img_size]
        x_patch = x_patch.transpose(1, 2).reshape(B, self.out_channels, d, h, w)
        return x_patch 