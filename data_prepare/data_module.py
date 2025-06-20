import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data_prepare.load_file import collect_sample_paths
import torch
import SimpleITK as SpITK
import h5py
import os
import numpy as np
import time

from logger import get_logger

logger = get_logger("data_module")

# H5Dataset: 用于直接读取h5文件
class H5Dataset(Dataset):
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = indices
        self._file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        i = self.indices[idx]
        image = torch.tensor(self._file["images"][i], dtype=torch.float32)
        label = torch.tensor(self._file["labels"][i], dtype=torch.float32)
        return image, label

    def __del__(self):
        if self._file is not None:
            self._file.close()

# Helper function to create a PyTorch Dataset for MR images
def make_mr_dataset(samples, resize, process_fn):
    class _Dataset(Dataset):
        def __len__(self):
            # Return the number of samples
            return len(samples)
        def __getitem__(self, idx):
            # Process and return a single sample
            return process_fn(samples[idx], resize)
    return _Dataset()

# LightningDataModule for MR image segmentation
class MRDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "/home/Datasets/bionet/Dataset/lsj_MICCAI_BraTS2020_TrainingData/",
                 batch_size=32, 
                 num_workers=8,
                 train_val_split=0.8,
                 h5_path="preprocessed_dataset.h5"):
        super().__init__()
        self.data_dir = data_dir  # Path to dataset
        self.batch_size = batch_size  # Batch size for dataloaders
        self.num_workers = num_workers  # Number of workers for dataloaders
        self.train_val_split = train_val_split  # Train/val split ratio
        self.target_size = (256, 256)  # Target image size
        self.h5_path = h5_path

    @staticmethod
    def process_mr_sample(paths, resize):
        # Load and preprocess MR images and label for a single sample
        images = []
        for key in ['flair', 't1', 't2', 't1ce']:
            # Read image using SimpleITK
            img = SpITK.GetArrayFromImage(SpITK.ReadImage(paths[key]))
            # Center crop to 128 slices
            start_idx = (img.shape[0] - 128) // 2
            img = img[start_idx:start_idx+128]
            # Convert to torch tensor and resize
            img = torch.tensor(img, dtype=torch.float32)
            img = resize(img.unsqueeze(0)).squeeze(0)
            # Normalize to [0, 1]
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-8)
            images.append(img)

        # Stack all modalities into a single tensor
        image_tensor = torch.stack(images, dim=0)
        # Process label
        label = SpITK.GetArrayFromImage(SpITK.ReadImage(paths['label']))
        start_idx = (label.shape[0] - 128) // 2
        label = label[start_idx:start_idx+128]
        label = torch.tensor(label, dtype=torch.int64)
        label = resize(label.unsqueeze(0)).squeeze(0).long()

        # One-hot encode label for 4 classes (0, 1, 2, 4)
        num_classes = 4
        label_tensor = torch.zeros((num_classes, 128, 256, 256), dtype=torch.float32)
        label_tensor[0] = (label == 0).float()
        label_tensor[1] = (label == 1).float()
        label_tensor[2] = (label == 2).float()
        label_tensor[3] = (label == 4).float()
        return image_tensor, label_tensor

    def setup(self, stage: str = None):
        if os.path.exists(self.h5_path):
            # 用 H5Dataset
            with h5py.File(self.h5_path, "r") as f:
                n_total = f["images"].shape[0]
            n_train = int(n_total * self.train_val_split)
            indices = np.arange(n_total)
            np.random.seed(42)
            np.random.shuffle(indices)
            self.train_dataset = H5Dataset(self.h5_path, indices[:n_train])
            self.val_dataset = H5Dataset(self.h5_path, indices[n_train:])
            self.test_dataset = self.val_dataset
        else:
            # 原有逻辑
            samples = collect_sample_paths(self.data_dir)
            n_total = len(samples)
            n_train = int(n_total * self.train_val_split)
            n_val = n_total - n_train
            train_samples, val_samples = torch.utils.data.random_split(samples, [n_train, n_val], generator=torch.Generator().manual_seed(42))
            resize = transforms.Resize(self.target_size)
            self.train_dataset = make_mr_dataset([samples[i] for i in train_samples.indices], resize, self.process_mr_sample)
            self.val_dataset = make_mr_dataset([samples[i] for i in val_samples.indices], resize, self.process_mr_sample)
            self.test_dataset = make_mr_dataset([samples[i] for i in val_samples.indices], resize, self.process_mr_sample)

    def train_dataloader(self):
        # Return DataLoader for training set
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Return DataLoader for validation set
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Return DataLoader for test set
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)