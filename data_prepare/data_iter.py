import torch 
from torch.utils.data  import Dataset, DataLoader 
import matplotlib.pyplot  as plt 
from logger import get_logger 
import os 
# 禁用 TensorFlow 日志输出 
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR 
import tensorflow as tf 
tf.get_logger().setLevel('ERROR')   # 只显示错误信息 
 
from .load_file import read_nii_files 
from torchvision.transforms  import Resize 
 
logger = get_logger("data_iter") 
 
# 假设的 SpITK 模块，实际使用时需要替换为正确的模块 
import SimpleITK as SpITK 
import os 
import pytorch_lightning as pl 
from pytorch_lightning.loggers  import TensorBoardLogger 
 
class MRDataset(Dataset): 
    def __init__(self, images, labels, augment=False): 
        self.all_images  = [] 
        self.all_labels  = [] 
        target_size = (256, 256)  # 目标大小 
        resize = Resize(target_size) 
 
        # 处理 images 
        num_samples = len(images) 
        for sample_idx in range(num_samples): 
            # 获取完整的体积数据 
            img_volume = images[sample_idx]  # [155, 240, 240] 
 
            # 从中间部分截取128个切片 
            start_idx = (img_volume.shape[0]  - 128) // 2 
            end_idx = start_idx + 128 
            img_volume = img_volume[start_idx:end_idx]  # [128, 240, 240] 
 
            # 转换为tensor 
            img_tensor = torch.tensor(img_volume,  dtype=torch.float32)   # [128, 240, 240] 
 
            # 调整大小到 256x256 
            # 需要添加通道维度进行resize，然后再移除 
            img_tensor = resize(img_tensor.unsqueeze(1)).squeeze(1)   # [128, 256, 256] 
 
            # Min-Max 归一化图像 
            img_min, img_max = img_tensor.min(),  img_tensor.max()  
            normalized_img = (img_tensor - img_min) / (img_max - img_min + 1e-8) 
 
            self.all_images.append(normalized_img) 
 
        # 处理 labels 
        num_samples = labels.shape[1]   # 25 
        num_channels = labels.shape[0]   # 3 
        for sample_idx in range(num_samples): 
            label_channels = [] 
            for channel_idx in range(num_channels): 
                # 获取完整的体积数据 
                label_volume = labels[channel_idx, sample_idx]  # [155, 240, 240] 
 
                # 从中间部分截取128个切片 
                start_idx = (label_volume.shape[0]  - 128) // 2 
                end_idx = start_idx + 128 
                label_volume = label_volume[start_idx:end_idx]  # [128, 240, 240] 
 
                # 转换为tensor 
                lbl_tensor = torch.tensor(label_volume,  dtype=torch.float32)   # [128, 240, 240] 
 
                # 调整大小到 256x256 
                lbl_tensor = resize(lbl_tensor)  # [128, 256, 256] 
 
                label_channels.append(lbl_tensor)  
 
            label_tensor = torch.stack(label_channels,  dim=0)  # [3, 128, 256, 256] 
            self.all_labels.append(label_tensor)  
 
        self.augment  = augment 
 
    def __len__(self): 
        return len(self.all_images)  
 
    def __getitem__(self, idx): 
        image = self.all_images[idx]   # [1, 128, 256, 256] 
        label = self.all_labels[idx]   # [3, 128, 256, 256] 
        return image, label 
 
def show_data_iter(dataset, batch_size=4): 
    images = dataset['images']  # 直接使用字典中的数据 
    labels = dataset['labels'] 
 
    medical_dataset = MRDataset(images, labels) 
    dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=False) 
 
    for images, labels in dataloader: 
        print(f"Batch shape: {images.shape}")   # 应该是 [batch_size, 1, 128, 256, 256] 
        fig, axes = plt.subplots(len(images),  2, figsize=(12, 6 * len(images))) 
        for i in range(len(images)): 
            # 显示中间的切片 
            middle_slice = images[i, 0, images.shape[2]//2].numpy()  
            middle_label = labels[i, 0, labels.shape[2]//2].numpy()  
            axes[i, 0].imshow(middle_slice) 
            axes[i, 1].imshow(middle_label) 
        plt.tight_layout()  
        plt.show()  
 
if __name__ == '__main__': 
    MRdatasets = read_nii_files("C://Users/ForRiver/OneDrive/Desktop/Pre/BIONET/MRImg/BraTS020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData") 
    show_data_iter(MRdatasets, batch_size=4) 