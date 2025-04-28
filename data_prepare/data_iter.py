import torch 
from torch.utils.data  import Dataset, DataLoader 
import matplotlib.pyplot  as plt 
from logger import MyLogger 
from load_file import generate_dataset 
from monai.transforms  import Resize 
 
logger = MyLogger("data_iter") 


# 现在还存在问题，一个文件夹里有不同STRUCT文件，会生成同样的文件
# 目前要求不同患者现在放在./data的子文件夹

 
# Define the MedicalDataset class, inheriting from torch.utils.data.Dataset  
class MedicalDataset(Dataset): 
    # Initialization function that accepts a list of images, a list of labels, and a flag for data augmentation 
    def __init__(self, images, labels, augment=False): 
        # Save the list of images 
        self.images  = images 
        # Save the list of labels 
        self.labels  = labels 
        # Save the flag for data augmentation 
        self.augment  = augment 
 
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
 
        # Convert the image and label to torch.Tensor type 
        image = torch.tensor(image,  dtype=torch.float32)  
        label = torch.tensor(label,  dtype=torch.long)  
 
        return image, label 
 
def data_iter(dataset, batch_size=4, patch_size=4): 
    # Create Dataset and DataLoader 
    medical_dataset = MedicalDataset(dataset) 
    dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
 
    # Iterate through batch data 
    for batch_idx, (images, labels) in enumerate(dataloader): 
        logger.info(f"Batch  {batch_idx + 1}") 
        # Print metadata for each sample 
        for i in range(len(images)): 
            sample = dataset[batch_idx * batch_size + i] 
            logger.info(f"Sample  {batch_idx * batch_size + i + 1} Metadata:") 
            logger.info(f"    SeriesNumber: {sample['SeriesNumber']}") 
            logger.info(f"    StudyDescription: {sample['StudyDescription']}") 
            logger.info(f"    SeriesDescription: {sample['SeriesDescription']}") 
            logger.info(f"    spacing: {sample['spacing']}") 
            logger.info(f"    spatial_shape: {sample['spatial_shape']}") 
            logger.info(f"    space: {sample['space']}") 
 
        # Visualize images and labels 
        fig, axes = plt.subplots(2,  patch_size, figsize=(15, 5)) 
        for i in range(patch_size): 
            if i >= images.shape[0]:   # Prevent index out of bounds 
                break 
            # Show the middle slice 
            slice_idx = images.shape[2]  // 2 
            axes[0, i].imshow(images[i, 0, slice_idx].cpu().numpy(), cmap='gray') 
            axes[0, i].axis('off') 
            axes[1, i].imshow(labels[i, 0, slice_idx].cpu().numpy(), cmap='jet', alpha=0.5) 
            axes[1, i].axis('off') 
        plt.suptitle(f"Batch  {batch_idx + 1}") 
        plt.show()  
 
 
if __name__ == '__main__': 
    folder_path = ".\\data" 
    output_dir = ".\\data\\output" 
    dataset = generate_dataset(folder_path, output_dir) 
    data_iter(dataset, batch_size=2, patch_size=2)