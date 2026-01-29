"""
MAE Dataset for 3D Medical Volumes (Preprocessed Data Only)

Note: Use preprocess_data.py with --use_3d flag to prepare your 3D data before using these datasets.

Supports 3D volume-based data loading only.
"""
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import random
import json
from torch.utils.data import Dataset
from pathlib import Path
from logger import get_logger

logger = get_logger("mae_dataset_3d")


class MedicalMAEDataset(Dataset):
    """
    Medical MAE Dataset for 3D volumes - loads preprocessed 3D data
    
    Expected preprocessed H5 structure:
        - images: (1, D, H, W) float32, normalized to [0,1] - single modality per file
        - attrs['mode']: 'mae_3d'
    
    Each file contains one modality from one patient.
    Multiple modalities are saved as separate files (e.g., 001_mod0.h5, 001_mod1.h5).
    """
    
    def __init__(self, h5_dir, volume_size=None, config=None):
        self._file_cache = {}
        self.h5_dir = Path(h5_dir)
        self.volume_size = getattr(config, 'volume_size', volume_size) if config else (volume_size or 160)
        self.config = config
        
        # Directly scan for h5 files - each file is one preprocessed volume sample
        self.h5_files = sorted(list(self.h5_dir.glob("*.h5")))
        self.total_volumes = len(self.h5_files)
        
        if self.total_volumes == 0:
            raise FileNotFoundError(
                f"No H5 files found in {h5_dir}\n"
                f"Please run preprocess_data.py first to generate preprocessed 3D data."
            )
        
        logger.info(f"MAE 3D Dataset:")
        logger.info(f"   - H5 files: {len(self.h5_files)}")
        logger.info(f"   - Total volumes: {self.total_volumes}")
    
    def __len__(self):
        return self.total_volumes
    
    def _get_file_handle(self, file_path):
        """
        Cache file handles per worker to avoid repeated open/close operations.
        Each DataLoader worker maintains its own cache.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        cache_key = (worker_id, str(file_path))
        
        if cache_key not in self._file_cache:
            self._file_cache[cache_key] = h5py.File(file_path, "r")
        return self._file_cache[cache_key]
    
    def __getitem__(self, idx):
        """Load preprocessed 3D volume (already cropped, resized, normalized)
        Each file contains a single modality.
        """
        idx = idx % len(self.h5_files)
        
        f = self._get_file_handle(self.h5_files[idx])
        # images shape: (1, D, H, W) - single modality
        volume = torch.from_numpy(f['images'][:]).float()  # (1, D, H, W)
        
        # Crop to config.crop_ratio (default 0.4)
        crop_ratio = getattr(self.config, 'crop_ratio', 1.0) if self.config else 1.0
        
        def crop_center(tensor, crop_ratio_val):
            shape = tensor.shape
            crop_shape = [int(s * crop_ratio_val) for s in shape[-3:]]
            start = [(s - c) // 2 for s, c in zip(shape[-3:], crop_shape)]
            end = [start[i] + crop_shape[i] for i in range(3)]
            if tensor.dim() == 4:
                return tensor[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            elif tensor.dim() == 3:
                return tensor[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            else:
                return tensor

        volume = crop_center(volume, crop_ratio)

        # Resize back to volume_size
        target_shape = self.volume_size if isinstance(self.volume_size, (tuple, list)) else (self.volume_size, self.volume_size, self.volume_size)
        # Robust conversion to tuple of ints
        new_shape = []
        for x in target_shape:
            if isinstance(x, (list, tuple)):
                new_shape.append(int(x[0]))
            else:
                new_shape.append(int(x))
        target_shape = tuple(new_shape)
        
        volume = F.interpolate(volume.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=False).squeeze(0)
        
        return volume, 0
    
    def __del__(self):
        """Clean up file handles"""
        if hasattr(self, '_file_cache'):
            for f in self._file_cache.values():
                if hasattr(f, 'close'):
                    f.close()


class MedicalSegmentationDataset3D(Dataset):
    """
    Medical Segmentation Dataset for 3D volumes - loads preprocessed 3D data
    
    Expected preprocessed H5 structure:
        - images: (n, D, H, W) float32 for n modalities, normalized to [0,1]
        - labels: (D, H, W) int64
        - attrs['mode']: 'seg_3d'
    """
    
    def __init__(self, h5_dir, volume_size=None, num_classes=4, config=None, patient_ids=None, augment=False):
        self._file_cache = {}
        self.h5_dir = Path(h5_dir)
        self.volume_size = getattr(config, 'volume_size', volume_size) if config else (volume_size or (160, 160, 160)) # Ensure volume_size is tuple
        self.num_classes = num_classes
        self.config = config # Store config for access to in_chans
        self.augment = augment
        
        if patient_ids is not None:
            self.patient_ids = patient_ids
        else:
            # Group files by patient ID
            patient_files = {}
            for f_path in self.h5_dir.glob("*.h5"):
                patient_id = f_path.stem.split('_mod')[0]
                if patient_id not in patient_files:
                    patient_files[patient_id] = []
                patient_files[patient_id].append(f_path)
            
            self.patient_ids = sorted(list(patient_files.keys()))
            
        self.total_volumes = len(self.patient_ids) # Each patient is one sample

        if self.total_volumes == 0:
            raise FileNotFoundError(
                f"No H5 files found in {h5_dir}\n"
                f"Please run preprocess_data.py first to generate preprocessed 3D data."
            )
        
        logger.info(f"Segmentation 3D Dataset:")
        logger.info(f"   - Patients: {len(self.patient_ids)}")
        logger.info(f"   - Total volumes (samples): {self.total_volumes}")
        logger.info(f"   - Expected in_chans: {self.config.in_chans if self.config else 1}")
        logger.info(f"   - Augmentation: {self.augment}")
    
    def __len__(self):
        return self.total_volumes
    
    def _get_file_handle(self, file_path):
        """
        Cache file handles per worker to avoid repeated open/close operations.
        Each DataLoader worker maintains its own cache.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        cache_key = (worker_id, str(file_path))
        
        if cache_key not in self._file_cache:
            self._file_cache[cache_key] = h5py.File(file_path, "r")
        return self._file_cache[cache_key]
    
    def __getitem__(self, idx):
        """Load preprocessed 3D volume and mask for all modalities, with zero-padding for missing modalities"""
        patient_id = self.patient_ids[idx]
        
        target_in_chans = self.config.in_chans if self.config else 1
        modalities_data = [None] * target_in_chans # Initialize with None placeholders
        labels = None
        
        # Load all modalities for the current patient, or zero-pad if missing
        # Determine a spatial shape from the first available modality or config
        sample_volume_shape = None
        
        # First pass to find labels and determine sample_volume_shape
        for i in range(target_in_chans):
            modality_file = self.h5_dir / f"{patient_id}_mod{i}.h5"
            if modality_file.exists():
                f = self._get_file_handle(modality_file)
                # Assume images are (1, D, H, W) and labels are (D, H, W)
                if sample_volume_shape is None:
                    sample_volume_shape = f['images'].shape[1:] # (D, H, W)
                if labels is None:
                    labels = torch.from_numpy(f['labels'][:]).long() # (D, H, W)
            if labels is not None and sample_volume_shape is not None: # Break early if labels and shape found
                break
        
        if labels is None or sample_volume_shape is None:
            # Fallback to config volume_size if no file found or labels missing
            logger.warning(f"Labels or initial volume shape not found for patient {patient_id}. Using config.volume_size for padding.")
            effective_volume_size = self.volume_size if isinstance(self.volume_size, tuple) else (self.volume_size, self.volume_size, self.volume_size)
            sample_volume_shape = effective_volume_size # Use target size for zero padding
            # Create a dummy label mask if not found, to avoid errors downstream
            if labels is None:
                labels = torch.zeros(sample_volume_shape, dtype=torch.long)
            
        # Second pass to load actual modalities or create zero-padded tensors
        for i in range(target_in_chans):
            modality_file = self.h5_dir / f"{patient_id}_mod{i}.h5"
            if modality_file.exists():
                f = self._get_file_handle(modality_file)
                mod_volume = torch.from_numpy(f['images'][:]).float()  # (1, D, H, W)
                modalities_data[i] = mod_volume
            else:
                logger.warning(f"Modality file {modality_file} not found for patient {patient_id}. Using zero-padding.")
                # Create a zero-padded tensor for missing modality
                modalities_data[i] = torch.zeros((1,) + sample_volume_shape, dtype=torch.float)
        
        # Concatenate all modalities (loaded or zero-padded)
        volume = torch.cat(modalities_data, dim=0) # (target_in_chans, D, H, W)
        mask = labels # (D, H, W)

        # Crop to 0.3 of original size (center crop)
        def crop_center(tensor, crop_ratio_val):
            shape = tensor.shape
            crop_shape = [int(s * crop_ratio_val) for s in shape[-3:]]
            start = [(s - c) // 2 for s, c in zip(shape[-3:], crop_shape)]
            end = [start[i] + crop_shape[i] for i in range(3)]
            if tensor.dim() == 4:
                return tensor[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            elif tensor.dim() == 3:
                return tensor[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            else:
                return tensor
        
        # Use config's crop_ratio if available, otherwise default to 1.0
        crop_ratio = getattr(self.config, 'crop_ratio', 1.0) if self.config else 1.0
        volume = crop_center(volume, crop_ratio)
        mask = crop_center(mask, crop_ratio)

        # Upsample to target volume size
        target_shape = self.volume_size if isinstance(self.volume_size, tuple) else (self.volume_size, self.volume_size, self.volume_size)
        # volume: (in_chans, D, H, W) -> interpolate
        volume = F.interpolate(volume.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=False).squeeze(0)
        # mask: (D, H, W) -> (1, D, H, W) -> interpolate (nearest)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_shape, mode='nearest').squeeze(0).long().squeeze(0)

        # Convert mask to one-hot: (D, H, W) -> (C, D, H, W)
        if self.num_classes == 1:
            # Manual one-hot for binary to avoid "Class values must be smaller than num_classes" error
            # mask is (D, H, W) with 0 and 1
            bg = (mask == 0).float()
            fg = (mask == 1).float()
            mask_onehot = torch.stack([bg, fg], dim=0) # (2, D, H, W)
        else:
            mask_onehot = F.one_hot(mask, num_classes=self.num_classes).permute(3, 0, 1, 2).float()  # (C, D, H, W)
            
        if self.augment:
            # Random Flip
            if random.random() > 0.5:
                volume = torch.flip(volume, [1]) # Flip D
                mask_onehot = torch.flip(mask_onehot, [1])
            if random.random() > 0.5:
                volume = torch.flip(volume, [2]) # Flip H
                mask_onehot = torch.flip(mask_onehot, [2])
            if random.random() > 0.5:
                volume = torch.flip(volume, [3]) # Flip W
                mask_onehot = torch.flip(mask_onehot, [3])
            
            # Random Rotation (90 degrees)
            k = random.randint(0, 3)
            if k > 0:
                volume = torch.rot90(volume, k, [2, 3]) # Rotate in H-W plane
                mask_onehot = torch.rot90(mask_onehot, k, [2, 3])
                
        return volume, mask_onehot

    def __del__(self):
        """Clean up file handles"""
        if hasattr(self, '_file_cache'):
            for f in self._file_cache.values():
                if hasattr(f, 'close'):
                    f.close()
