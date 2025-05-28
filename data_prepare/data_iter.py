import torch 
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt 
from logger import get_logger
import os
# 禁用 TensorFlow 日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # 只显示错误信息

from .load_file import read_nii_files
from torchvision.transforms import Resize 
 
logger = get_logger("data_iter") 

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
        self.all_images = []
        self.all_labels = []
        target_size = (256, 256)  # 目标大小
        resize = Resize(target_size)
        
        # 假设 images 的形状是 [n, 155, 240, 240]，其中 n 是样本数量
        num_samples = len(images)
        for sample_idx in range(num_samples):
            # 获取完整的体积数据
            img_volume = images[sample_idx]  # [155, 240, 240]
            label_volume = labels[sample_idx]  # [155, 240, 240]
            
            # 从中间部分截取128个切片
            start_idx = (img_volume.shape[0] - 128) // 2
            end_idx = start_idx + 128
            img_volume = img_volume[start_idx:end_idx]  # [128, 240, 240]
            label_volume = label_volume[start_idx:end_idx]  # [128, 240, 240]
            
            # 转换为tensor
            img_tensor = torch.tensor(img_volume, dtype=torch.float32)  # [128, 240, 240]
            lbl_tensor = torch.tensor(label_volume, dtype=torch.float32)  # [128, 240, 240]
            
            # 调整大小到 256x256
            # 需要添加通道维度进行resize，然后再移除
            img_tensor = resize(img_tensor.unsqueeze(1)).squeeze(1)  # [128, 256, 256]
            lbl_tensor = resize(lbl_tensor.unsqueeze(1)).squeeze(1)  # [128, 256, 256]
            
            # Min-Max 归一化图像
            img_min, img_max = img_tensor.min(), img_tensor.max()
            normalized_img = (img_tensor - img_min) / (img_max - img_min + 1e-8)
            
            # 保持标签的原始值，不进行二值化
            # BraTS数据集中的标签：
            # 0: 背景
            # 1: 坏死核心/非增强肿瘤核心
            # 2: 水肿
            # 4: 增强肿瘤
            self.all_images.append(normalized_img)
            self.all_labels.append(lbl_tensor)
        
        self.augment = augment
    
    def __len__(self): 
        return len(self.all_images)
    
    def __getitem__(self, idx): 
        image = self.all_images[idx]  # [128, 256, 256]
        label = self.all_labels[idx]  # [128, 256, 256]
        return image, label

def show_data_iter(dataset, batch_size=4):
    images = dataset['images']  # 直接使用字典中的数据
    labels = dataset['labels']
 
    medical_dataset = MRDataset(images, labels) 
    dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=False) 
 
    for images, labels in dataloader: 
        print(f"Batch shape: {images.shape}")  # 应该是 [batch_size, 128, 256, 256]
        fig, axes = plt.subplots(len(images), 2, figsize=(12, 6 * len(images))) 
        for i in range(len(images)): 
            # 显示中间的切片
            middle_slice = images[i, images.shape[1]//2].numpy()
            middle_label = labels[i, labels.shape[1]//2].numpy()
            axes[i, 0].imshow(middle_slice) 
            axes[i, 1].imshow(middle_label) 
        plt.tight_layout()    
        plt.show()    
 
if __name__ == '__main__': 
    MRdatasets = read_nii_files("C://Users/ForRiver/OneDrive/Desktop/Pre/BIONET/MRImg/BraTS020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    show_data_iter(MRdatasets, batch_size=4)