import torch 
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt 
from data_prepare.logger import MyLogger
from .load_file import generate_dataset, read_nii_files
from torchvision.transforms import Resize 
 
logger = MyLogger("data_iter") 

## TODO: 分离不同序列作为多模态输入；

import torch 
from torch.utils.data  import Dataset 
from torchvision.transforms  import Resize 
import logging 
 
# 假设的 SpITK 模块，实际使用时需要替换为正确的模块 
import SimpleITK as SpITK 
import os 
from torch.utils.data  import DataLoader 
import pytorch_lightning as pl 
from pytorch_lightning.loggers  import TensorBoardLogger 
 
class MRDataset(Dataset): 
    def __init__(self, images, labels, augment=False): 
        self.all_images   = [] 
        self.all_labels   = [] 
        target_size = (256, 256) 
        resize = Resize(target_size)  # change interpolation method here 
 
        for img_list, label in zip(images, labels): 
            num_slices = img_list[0].shape[0] 
            for slice_idx in range(num_slices): 
                slice_images = [] 
                for img in img_list: 
                    img_slice = img[slice_idx] 
                    # to tensor 
                    img_tensor = torch.tensor(img_slice,   dtype=torch.float32).unsqueeze(0)    # shape: [1,H,W] 
                    # squeeze num_samples and num_instances 
                    resized_img = resize(img_tensor).squeeze(0)  # shape: [C,H,W] 
                    # Min-Max Scaling to [0,1] 
                    img_min, img_max = resized_img.min(),   resized_img.max()   
                    normalized_img = (resized_img - img_min) / (img_max - img_min + 1e-8) 
                    slice_images.append(normalized_img)   
 
                label_slice = label[slice_idx] 
                lbl_tensor = torch.tensor(label_slice,   dtype=torch.long).unsqueeze(0)   
                resized_lbl = resize(lbl_tensor).squeeze(0) 
                lbl_max, lbl_min = resized_lbl.max(),   resized_lbl.min()   
                normalized_lbl = resized_lbl / (lbl_max - lbl_min + 1e-8) 
 
                slice_images = torch.stack(slice_images,   dim=0)  # shape: [4, H, W] 
 
                self.all_images.append(slice_images)   
                self.all_labels.append(normalized_lbl)   
 
        self.augment   = augment 
 
    def __len__(self): 
        return len(self.all_images)   
 
    def __getitem__(self, idx): 
        image = self.all_images[idx]   
        label = self.all_labels[idx]   
        return image, label 
def show_data_iter(dataset, batch_size=4):
    images = [sample['image'] for sample in dataset]
    labels = [sample['label'] for sample in dataset]
 
    medical_dataset = MRDataset(images, labels) 
    dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=False) 
 
    for images, labels in dataloader: 
        fig, axes = plt.subplots(len(images),  2, figsize=(12, 6 * len(images))) 
        for i in range(len(images)): 
            # remove extra dimension
            img = images[i].squeeze(0).numpy() 
            lbl = labels[i].squeeze(0).numpy() 
            axes[i, 0].imshow(img) 
            axes[i, 1].imshow(lbl) 
        plt.tight_layout()    
        plt.show()    
 
if __name__ == '__main__': 
    MRdatasets = read_nii_files("C://Users/ForRiver/OneDrive/Desktop/Pre/BIONET/MRImg/BraTS020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    show_data_iter(MRdatasets, batch_size=4)