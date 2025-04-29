import torch 
from torch.utils.data  import Dataset, DataLoader 
import matplotlib.pyplot  as plt 
from logger import MyLogger 
from load_file import generate_dataset 
from monai.transforms  import Resize 
 
logger = MyLogger("data_iter") 

## 还要修改iter维度（现在是num_sample，num_instance, H, W)
## 注释和命名部分需要修改
 
# Define the MedicalDataset class, inheriting from torch.utils.data.Dataset  
class MedicalDataset(Dataset): 
    # Initialization function that accepts a list of images, a list of labels, and a flag for data augmentation 
    def __init__(self, images, labels, augment=False, target_size=None): 
        # Save the list of images 
        self.images  = images 
        # Save the list of labels 
        self.labels  = labels 
        # Save the flag for data augmentation 
        self.augment  = augment 
        # Save the target size for resizing 
        self.target_size  = target_size 
        # Initialize the resizer if target size is provided 
        if target_size is not None: 
            self.resizer  = Resize(spatial_size=target_size) 
 
    # Return the length of the dataset 
    def __len__(self): 
        return len(self.images)  
 
    # Get a sample from the dataset according to the index 
    def __getitem__(self, idx): 
        # Get the image at the specified index 
        image = self.images[idx]  
        # Get the label at the specified index 
        label = self.labels[idx]  
 
        # If data augmentation is required 
        if self.augment:  
            # Data augmentation code can be added here 
            pass 
 
        # Resize the image and label if target size is provided 
        if self.target_size  is not None: 
            image = self.resizer(torch.tensor(image,  dtype=torch.float32).unsqueeze(0)).squeeze(0)  
            label = self.resizer(torch.tensor(label,  dtype=torch.long).unsqueeze(0)).squeeze(0)  
        else: 
            # Convert the image and label to torch.Tensor type 
            image = torch.tensor(image,  dtype=torch.float32)  
            label = torch.tensor(label,  dtype=torch.long)  
 
        return image, label 
 
def data_iter(dataset, batch_size=4, target_size=None): 
    # Extract images and labels from the dataset 
    images = [sample['image'] for sample in dataset] 
    labels = [sample['label'] for sample in dataset] 
 
    # Create Dataset and DataLoader 
    medical_dataset = MedicalDataset(images, labels, target_size=target_size) 
    dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
 
    # Iterate over the dataloader 
    for images, labels in dataloader: 
        num_samples = images.shape[0]  
        # 创建一个大图，用于放置所有子图 
        fig, axes = plt.subplots(num_samples,  2, figsize=(12, 6 * num_samples)) 
 
        for i in range(num_samples): 
            # 提取当前图像和标签 
            current_image = images[i, 25, :, :].numpy() 
            current_label = labels[i, 25, :, :].numpy() 
 
            # 绘制图像 
            axes[i, 0].imshow(current_image) 
            axes[i, 0].set_title('Image') 
            axes[i, 0].axis('off') 
 
            # 绘制标签 
            axes[i, 1].imshow(current_label) 
            axes[i, 1].set_title('Label') 
            axes[i, 1].axis('off') 
 
        plt.tight_layout()  
        plt.show()  
 
 
if __name__ == '__main__': 
    folder_path = ".\\data" 
    output_dir = ".\\data\\output" 
    dataset = generate_dataset(folder_path, output_dir) 
    # Set the target size to the desired shape 
    target_size = (40, 1120, 1120) 
    data_iter(dataset, batch_size=4, target_size=target_size) 