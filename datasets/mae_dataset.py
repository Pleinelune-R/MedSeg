"""
MAE Dataset for Medical Images
"""
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import random
from torch.utils.data import Dataset
from pathlib import Path
from logger import get_logger

logger = get_logger("mae_dataset")


def center_crop(img, crop_ratio=0.3):
    """
    Args:
        img: 2D numpy array medical image
        crop_ratio: ratio of height/width to keep (e.g., 0.3 keeps 30%)
    Returns:
        cropped_img: center-cropped image
    """
    h, w = img.shape
    crop_h = max(1, int(h * crop_ratio))
    crop_w = max(1, int(w * crop_ratio))
    start_h = max(0, (h - crop_h) // 2)
    start_w = max(0, (w - crop_w) // 2)
    end_h = min(h, start_h + crop_h)
    end_w = min(w, start_w + crop_w)
    return img[start_h:end_h, start_w:end_w]


class MedicalMAEDataset(Dataset):
    """
    Simplified MR Dataset - randomly select from modalities 0,1,2,3 and skip all-black samples
    
    Fast strategy: randomly select modality, if all-black then skip to next sample.
    Prioritizes speed over avoiding dummy data.
    """
    
    def __init__(self, h5_dir, img_size=None, crop_ratio=None, config=None):
        self.h5_dir = Path(h5_dir)
        # Pull defaults from config if provided
        img_size = getattr(config, 'img_size', None)
        crop_ratio = getattr(config, 'crop_ratio', None)
        # Fallback hard defaults
        self.img_size = img_size if img_size is not None else 256
        self.crop_ratio = crop_ratio if crop_ratio is not None else 0.3
        
        self.h5_files = list(self.h5_dir.glob("*.h5"))
        self.h5_files = sorted(self.h5_files)
        
        # Simplified: only keep basic file handle cache
        self._file_cache = {}
        
        # Calculate valid slice indices per file to avoid runtime retries
        self.file_slice_mapping = []  # [(file_idx, slice_idx), ...] only valid slices
        self.total_slices = 0
        
        logger.info("Scanning H5 files, calculating total slice count...")
        for file_idx, h5_file in enumerate(self.h5_files):
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Assume structure: f['images'][modality_idx].shape = (num_slices, H, W)
                    # Use modality 0 to get slice count (assuming all modalities have same slice count)
                    modality_data = f['images'][0]
                    num_slices = modality_data.shape[0]
                    
                    # Create index for valid slices (based on modality 0 quick check)
                    valid_count = 0
                    for slice_idx in range(num_slices):
                        sl = modality_data[slice_idx]
                        if sl.size == 0:
                            continue
                        vmin, vmax = float(sl.min()), float(sl.max())
                        if vmax > vmin and vmax > 0.0:
                            self.file_slice_mapping.append((file_idx, slice_idx))
                            valid_count += 1
                    self.total_slices += valid_count
            except Exception as e:
                logger.error(f"Error reading file {h5_file}: {e}")
        
        logger.info(f"Dataset statistics:")
        logger.info(f"   - H5 file count: {len(self.h5_files)}")
        logger.info(f"   - Total slice count: {self.total_slices}")
    
    def __len__(self):
        return self.total_slices
    
    def _get_file_handle(self, file_path):
        """Get file handle with worker-level caching"""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        cache_key = (worker_id, str(file_path))
        
        if cache_key not in self._file_cache:
            self._file_cache[cache_key] = h5py.File(file_path, "r")
        return self._file_cache[cache_key]
    
    def __getitem__(self, idx):
        """Fast getitem without retries; uses pre-filtered valid slices and torch interpolate."""
        if len(self.file_slice_mapping) == 0:
            dummy_img = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            return dummy_img, 0
        if idx >= len(self.file_slice_mapping):
            idx = idx % len(self.file_slice_mapping)

        file_idx, slice_idx = self.file_slice_mapping[idx]
        h5_path = self.h5_files[file_idx]

        try:
            f = self._get_file_handle(h5_path)
            # Use modality 0 for speed and consistency with filtering
            modality_img = f['images'][0]
            slice_img_np = modality_img[slice_idx]

            # Center crop on numpy then convert to torch
            cropped_np = center_crop(slice_img_np, crop_ratio=self.crop_ratio)
            img_tensor = torch.from_numpy(cropped_np).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
            # Resize with torch (bilinear)
            img_tensor = F.interpolate(img_tensor, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
            img_tensor = img_tensor.squeeze(0)  # (1,H,W)
            # Normalize to [0,1]
            tmin = torch.amin(img_tensor)
            tmax = torch.amax(img_tensor)
            if (tmax - tmin) > 1e-6:
                img_tensor = (img_tensor - tmin) / (tmax - tmin)
            else:
                img_tensor = torch.zeros_like(img_tensor)

            return img_tensor, 0
        except Exception:
            dummy_img = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            return dummy_img, 0
    
    def __del__(self):
        """Clean up file handles"""
        for f in self._file_cache.values():
            if hasattr(f, 'close'):
                f.close()


class MedicalSegmentationDataset(Dataset):
    """
    Medical Image Segmentation Dataset
    Loads images and corresponding segmentation masks from H5 files
    
    H5 Structure:
        file.h5
        ├── images/
        │   ├── [0] - modality 0: (num_slices, H, W)
        │   ├── [1] - modality 1: (num_slices, H, W)
        │   ├── [2] - modality 2: (num_slices, H, W)
        │   └── [3] - modality 3: (num_slices, H, W)
        └── labels/     - segmentation labels: (num_slices, H, W)
    """
    
    def __init__(self, h5_dir, img_size=None, crop_ratio=None, num_classes=4, config=None):
        self.h5_dir = Path(h5_dir)
        # Pull defaults from config if provided
        if config is not None:
            if img_size is None:
                img_size = getattr(config, 'img_size', None)
            if crop_ratio is None:
                crop_ratio = getattr(config, 'crop_ratio', None)
        # Fallback hard defaults
        self.img_size = img_size if img_size is not None else 256
        self.crop_ratio = crop_ratio if crop_ratio is not None else 0.3
        self.num_classes = num_classes
        
        self.h5_files = list(self.h5_dir.glob("*.h5"))
        self.h5_files = sorted(self.h5_files)
        
        self._file_cache = {}
        
        # Build index mapping
        self.file_slice_mapping = []
        self.total_slices = 0
        
        logger.info("Scanning H5 files for segmentation...")
        for file_idx, h5_file in enumerate(self.h5_files):
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Check if labels exist
                    if 'labels' not in f:
                        logger.warning(f"Skipping {h5_file.name}: no 'labels' dataset")
                        continue
                    
                    modality_data = f['images'][0]
                    num_slices = modality_data.shape[0]
                    
                    for slice_idx in range(num_slices):
                        self.file_slice_mapping.append((file_idx, slice_idx))
                    
                    self.total_slices += num_slices
            except Exception as e:
                logger.error(f"Error reading file {h5_file}: {e}")
        
        logger.info(f"Segmentation dataset:")
        logger.info(f"   - H5 files: {len(self.h5_files)}")
        logger.info(f"   - Total slices: {self.total_slices}")
    
    def __len__(self):
        return self.total_slices
    
    def _get_file_handle(self, file_path):
        """Get file handle with worker-level caching"""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        cache_key = (worker_id, str(file_path))
        
        if cache_key not in self._file_cache:
            self._file_cache[cache_key] = h5py.File(file_path, "r")
        return self._file_cache[cache_key]
    
    def __getitem__(self, idx):
        """Get image and segmentation mask"""
        if idx >= len(self.file_slice_mapping):
            idx = idx % len(self.file_slice_mapping)
        
        max_attempts = 10
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.file_slice_mapping)
            file_idx, slice_idx = self.file_slice_mapping[current_idx]
            h5_path = self.h5_files[file_idx]
            
            try:
                f = self._get_file_handle(h5_path)
                
                # Load random modality
                random_modality_idx = random.randint(0, 3)
                modality_img = f['images'][random_modality_idx]
                slice_img = modality_img[slice_idx].copy()
                
                # Load mask (labels)
                mask_data = f['labels']
                slice_mask = mask_data[slice_idx].copy()
                
                # Check if valid
                if slice_img.max() == 0 or (slice_img.max() - slice_img.min()) < 1e-6:
                    continue
                
                # Apply center crop and resize (image and mask), then normalize image
                cropped_img = center_crop(slice_img, crop_ratio=self.crop_ratio)
                cropped_mask = center_crop(slice_mask, crop_ratio=self.crop_ratio)
                
                slice_img = transform.resize(
                    cropped_img, (self.img_size, self.img_size),
                    preserve_range=True
                ).astype(np.float32)
                
                # Resize mask with nearest neighbor to preserve labels
                slice_mask = transform.resize(
                    cropped_mask, (self.img_size, self.img_size),
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(np.int64)
                
                # Normalize image to [0,1]
                slice_min, slice_max = slice_img.min(), slice_img.max()
                if slice_max > slice_min:
                    slice_img = (slice_img - slice_min) / (slice_max - slice_min)
                else:
                    continue
                
                # Convert to tensors
                img_tensor = torch.from_numpy(slice_img[np.newaxis, ...]).float()  # (1, H, W)
                
                # Convert mask to one-hot encoding
                mask_tensor = torch.from_numpy(slice_mask).long()  # (H, W)
                mask_onehot = torch.nn.functional.one_hot(
                    mask_tensor, num_classes=self.num_classes
                ).permute(2, 0, 1).float()  # (C, H, W)
                
                return img_tensor, mask_onehot
                
            except Exception as e:
                # Silently skip on error
                continue
        
        # Return zero data if no valid sample found
        dummy_img = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        dummy_mask = torch.zeros((self.num_classes, self.img_size, self.img_size), dtype=torch.float32)
        dummy_mask[0] = 1.0  # Set background class
        return dummy_img, dummy_mask
    
    def __del__(self):
        """Clean up file handles"""
        for f in self._file_cache.values():
            if hasattr(f, 'close'):
                f.close()