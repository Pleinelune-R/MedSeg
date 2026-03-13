"""
Generic Positional Embeddings for Vision Transformers
Supports 1D, 2D, and 3D sinusoidal positional encoding
"""
import numpy as np
import torch
from typing import Union, Tuple, List


def get_sincos_pos_embed(embed_dim, grid_size, cls_token=False, ndim=None):
    """
    Generate N-dimensional sinusoidal positional embeddings.
    
    Args:
        embed_dim (int): Embedding dimension
        grid_size (int, tuple, or list): Spatial dimensions of the patch grid
            - int: will be repeated ndim times (assumes cubic grid)
            - tuple/list: dimensions for each axis (ndim will be inferred)
        cls_token (bool): Whether to prepend a cls token embedding
        ndim (int): Number of spatial dimensions (1, 2, or 3). 
            If None, inferred from grid_size
    
    Returns:
        numpy array of shape (num_patches [+ 1 if cls_token], embed_dim)
    """
    # Normalize grid_size and infer ndim
    if isinstance(grid_size, int):
        if ndim is None:
            raise ValueError("Must specify ndim when grid_size is a scalar")
        grid_size = (grid_size,) * ndim
    else:
        grid_size = tuple(grid_size)
        if ndim is None:
            ndim = len(grid_size)
    
    assert len(grid_size) == ndim, \
        f"grid_size length {len(grid_size)} does not match ndim={ndim}"
    
    assert embed_dim % 2 == 0, "embed_dim must be even"
    
    num_patches = 1
    for g in grid_size:
        num_patches *= g
    
    # Create position arrays for each dimension
    pos_arrays = []
    for g in grid_size:
        pos = np.arange(g, dtype=np.float32)
        # Normalize to [0, 1]
        pos = pos / max(g - 1, 1)
        pos_arrays.append(pos)
    
    # Create meshgrid
    if ndim == 1:
        grids = [pos_arrays[0].reshape(-1, 1)]
    elif ndim == 2:
        grid0, grid1 = np.meshgrid(pos_arrays[0], pos_arrays[1], indexing='ij')
        grids = [grid0.reshape(-1, 1), grid1.reshape(-1, 1)]
    elif ndim == 3:
        grid0, grid1, grid2 = np.meshgrid(
            pos_arrays[0], pos_arrays[1], pos_arrays[2], indexing='ij'
        )
        grids = [grid0.reshape(-1, 1), grid1.reshape(-1, 1), grid2.reshape(-1, 1)]
    else:
        raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3")
    
    # Compute frequency scaling (temperature)
    embed_per_dim = embed_dim // (2 * ndim)
    dim_t = np.arange(0, embed_dim, 2 * ndim, dtype=np.float32)
    dim_t = 10000.0 ** (dim_t / embed_dim)
    
    # Build positional embedding
    pos_embed = np.zeros((num_patches, embed_dim), dtype=np.float32)
    
    # Fill embeddings: sin/cos for each dimension
    idx = 0
    for dim_idx, grid in enumerate(grids):
        for freq_idx in range(embed_per_dim):
            pos_embed[:, idx] = np.sin(grid[:, 0] * dim_t[freq_idx])
            pos_embed[:, idx + 1] = np.cos(grid[:, 0] * dim_t[freq_idx])
            idx += 2
    
    # Handle leftover dimensions if embed_dim is not perfectly divisible
    if idx < embed_dim:
        remaining = embed_dim - idx
        for i in range(remaining):
            dim_idx = i % ndim
            pos_embed[:, idx + i] = np.sin(grids[dim_idx][:, 0] * dim_t[i % len(dim_t)])
    
    # Prepend cls token embedding (zero vector)
    if cls_token:
        cls_embed = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)
    
    return pos_embed


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    3D version of sinusoidal positional embeddings.
    
    Args:
        embed_dim (int): Embedding dimension
        grid_size (tuple or list): (D, H, W) - spatial dimensions
        cls_token (bool): Whether to prepend cls token
    
    Returns:
        numpy array of shape (num_patches [+ 1], embed_dim)
    """
    return get_sincos_pos_embed(embed_dim, grid_size, cls_token=cls_token, ndim=3)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w=None, cls_token=False):
    """
    2D version of sinusoidal positional embeddings.
    
    Args:
        embed_dim (int): Embedding dimension
        grid_h (int or tuple): Height dimension. If tuple, treated as (H, W)
        grid_w (int): Width dimension (ignored if grid_h is tuple)
        cls_token (bool): Whether to prepend cls token
    
    Returns:
        numpy array of shape (num_patches [+ 1], embed_dim)
    """
    # Support both get_2d_sincos_pos_embed(dim, h, w) and get_2d_sincos_pos_embed(dim, (h, w))
    if isinstance(grid_h, (tuple, list)):
        grid_size = tuple(grid_h)
    else:
        if grid_w is None:
            grid_w = grid_h  # Assume square grid
        grid_size = (grid_h, grid_w)
    
    return get_sincos_pos_embed(embed_dim, grid_size, cls_token=cls_token, ndim=2)


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    1D version of sinusoidal positional embeddings.
    
    Args:
        embed_dim (int): Embedding dimension
        grid_size (int): Sequence length
        cls_token (bool): Whether to prepend cls token
    
    Returns:
        numpy array of shape (grid_size [+ 1], embed_dim)
    """
    return get_sincos_pos_embed(embed_dim, grid_size, cls_token=cls_token, ndim=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Legacy function for backward compatibility.
    Generates 2D positional embeddings from precomputed grids.
    
    Args:
        embed_dim (int): Embedding dimension
        grid (np.ndarray): Grid array of shape (2, 1, H, W)
    
    Returns:
        np.ndarray: Positional embeddings of shape (H*W, embed_dim)
    """
    assert embed_dim % 2 == 0
    
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Legacy function for backward compatibility.
    Generates 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim (int): Output dimension for each position
        pos (np.ndarray): Positions to encode, shape (M,)
    
    Returns:
        np.ndarray: Shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
