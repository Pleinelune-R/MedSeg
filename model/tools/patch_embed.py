"""
Generic Patch Embedding Module for Multi-Dimensional Data
Supports 1D, 2D, and 3D patch embeddings
"""
import torch
import torch.nn as nn
from typing import Union, Tuple, List


class PatchEmbed(nn.Module):
    """
    Generic Patch Embedding for N-dimensional volumetric/spatial data.
    Converts N-D data into a sequence of patch embeddings.
    
    Supports:
    - 2D images: input (B, C, H, W) -> (B, N, embed_dim)
    - 3D volumes: input (B, C, D, H, W) -> (B, N, embed_dim)
    """
    
    def __init__(
        self,
        input_size: Union[int, Tuple[int, ...], List[int]],
        patch_size: Union[int, Tuple[int, ...], List[int]],
        in_chans: int = 1,
        embed_dim: int = 768,
        ndim: int = None
    ):
        """
        Args:
            input_size: Input spatial dimensions. Can be:
                - int: single dimension (will be replicated for all spatial dims)
                - tuple/list of ints: dimensions for each spatial axis
            patch_size: Patch size for each spatial dimension. Can be:
                - int: single size (will be replicated for all spatial dims)
                - tuple/list of ints: patch size for each spatial axis
            in_chans: Number of input channels
            embed_dim: Output embedding dimension
            ndim: Number of spatial dimensions (1, 2, or 3). If None, inferred from input_size
        """
        super().__init__()
        
        # Normalize input_size
        if isinstance(input_size, int):
            # Infer ndim if not provided
            if ndim is None:
                raise ValueError("Must specify ndim when input_size is a scalar")
            input_size = (input_size,) * ndim
        else:
            input_size = tuple(input_size)
            if ndim is None:
                ndim = len(input_size)
        
        # Normalize patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * ndim
        else:
            patch_size = tuple(patch_size)
        
        assert len(input_size) == len(patch_size) == ndim, \
            f"input_size and patch_size must have the same length as ndim={ndim}"
        
        self.ndim = ndim
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Calculate grid size (number of patches in each dimension)
        self.grid_size = tuple(s // p for s, p in zip(input_size, patch_size))
        self.num_patches = 1
        for g in self.grid_size:
            self.num_patches *= g
        
        # Create convolutional projection layer
        # Use appropriate Conv layer based on ndim
        if ndim == 1:
            conv_cls = nn.Conv1d
            kernel_size = tuple(patch_size)
            stride = tuple(patch_size)
        elif ndim == 2:
            conv_cls = nn.Conv2d
            kernel_size = tuple(patch_size)
            stride = tuple(patch_size)
        elif ndim == 3:
            conv_cls = nn.Conv3d
            kernel_size = tuple(patch_size)
            stride = tuple(patch_size)
        else:
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3")
        
        self.proj = conv_cls(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, *spatial_dims)
               For 1D: (B, C, L)
               For 2D: (B, C, H, W)
               For 3D: (B, C, D, H, W)
        
        Returns:
            Patches of shape (B, num_patches, embed_dim)
        """
        B = x.shape[0]
        C = x.shape[1]
        spatial_shape = x.shape[2:]
        
        assert spatial_shape == self.input_size, \
            f"Input spatial shape {spatial_shape} does not match configured size {self.input_size}"
        
        # Project patches
        x = self.proj(x)  # (B, embed_dim, *grid_size)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        
        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


# Backward compatibility aliases for 3D case
class PatchEmbed3D(PatchEmbed):
    """Backward compatibility alias for 3D PatchEmbed"""
    def __init__(
        self,
        volume_size: Union[int, Tuple[int, ...], List[int]] = (160, 160, 160),
        patch_size: Union[int, Tuple[int, ...], List[int]] = 16,
        in_chans: int = 1,
        embed_dim: int = 768
    ):
        super().__init__(
            input_size=volume_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            ndim=3
        )
        # Preserve old attribute names for backward compatibility
        self.volume_size = self.input_size


# Backward compatibility aliases for 2D case
class PatchEmbed2D(PatchEmbed):
    """Backward compatibility alias for 2D PatchEmbed"""
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int], List[int]] = (224, 224),
        patch_size: Union[int, Tuple[int, int], List[int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        super().__init__(
            input_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            ndim=2
        )
        self.image_size = self.input_size
