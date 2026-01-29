"""
3D Patch Embedding for Medical Volume Data
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    3D Image to Patch Embedding for volumetric data
    Converts 3D volume into a sequence of patch embeddings
    """
    def __init__(self, volume_size=(160, 160, 160), patch_size=16, in_chans=1, embed_dim=768):
        """
        Args:
            volume_size: Input volume size (D, H, W) tuple.
            patch_size: Patch size.
            in_chans: Input channels.
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size

        # Number of patches in each direction
        if isinstance(volume_size, int):
            self.grid_size = tuple([volume_size // patch_size] * 3)
        elif isinstance(patch_size, int):
            self.grid_size = tuple([v // patch_size for v in volume_size])
        else:
            self.grid_size = tuple([v // p for v, p in zip(volume_size, patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) input volume data
        Returns:
            patches: (B, N, embed_dim) N=num_patches
        """
        B, C, D, H, W = x.shape
        if isinstance(self.volume_size, int):
            expected_shape = (self.volume_size, self.volume_size, self.volume_size)
        else:
            expected_shape = self.volume_size
        assert (D, H, W) == expected_shape, f"Input data shape ({D},{H},{W}) does not match configured volume size {expected_shape}"
        
        # Project to (B, embed_dim, g1, g2, g3)
        x = self.proj(x)
        # Flatten (B, embed_dim, g1, g2, g3) -> (B, embed_dim, N)
        x = x.flatten(2)
        # Transpose (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.transpose(1, 2)
        return x
