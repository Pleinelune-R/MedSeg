import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data_prepare.load_file import collect_sample_paths
import torch
import SimpleITK as SpITK

def make_mr_dataset(samples, resize, process_fn):
    class _Dataset(Dataset):
        def __len__(self):
            return len(samples)
        def __getitem__(self, idx):
            return process_fn(samples[idx], resize)
    return _Dataset()

class MRDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "/home/Datasets/bionet/Dataset/lsj_MICCAI_BraTS2020_TrainingData/",
                 batch_size: int = 32, 
                 num_workers: int = 4,
                 train_val_split: float = 0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.target_size = (256, 256)

    @staticmethod
    def process_mr_sample(paths, resize, target_size=(256, 256)):
        images = []
        for key in ['flair', 't1', 't2', 't1ce']:
            img = SpITK.GetArrayFromImage(SpITK.ReadImage(paths[key]))
            start_idx = (img.shape[0] - 128) // 2
            img = img[start_idx:start_idx+128]
            img = torch.tensor(img, dtype=torch.float32)
            img = resize(img.unsqueeze(0)).squeeze(0)
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-8)
            images.append(img)
        image_tensor = torch.stack(images, dim=0)
        label = SpITK.GetArrayFromImage(SpITK.ReadImage(paths['label']))
        start_idx = (label.shape[0] - 128) // 2
        label = label[start_idx:start_idx+128]
        label = torch.tensor(label, dtype=torch.int64)
        label = resize(label.unsqueeze(0)).squeeze(0).long()
        num_classes = 4
        label_tensor = torch.zeros((num_classes, 128, 256, 256), dtype=torch.float32)
        label_tensor[0] = (label == 0).float()
        label_tensor[1] = (label == 1).float()
        label_tensor[2] = (label == 2).float()
        label_tensor[3] = (label == 4).float()
        return image_tensor, label_tensor

    def setup(self, stage: str = None):
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)