import torch 
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt 
from logger import get_logger 
from torchvision.transforms import Resize 

logger = get_logger("data_iter") 

class MRDataset(Dataset): 
    """
    Custom Dataset class for handling medical imaging data (MRI scans)
    Processes 3D volumes of medical images and their corresponding segmentation labels
    """
    def __init__(self, images, labels, augment=False): 
        self.all_images = [] 
        self.all_labels = [] 
        target_size = (256, 256)  # Target size for resizing
        resize = Resize(target_size) 

        # Process input images
        num_samples = len(images) 
        for sample_idx in range(num_samples): 
            # Get full volume data
            img_volume = images[sample_idx]  # Shape: [155, 240, 240] 

            # Extract 128 slices from the middle portion
            start_idx = (img_volume.shape[0] - 128) // 2 
            end_idx = start_idx + 128 
            img_volume = img_volume[start_idx:end_idx]  # Shape: [128, 240, 240] 

            # Convert to tensor
            img_tensor = torch.tensor(img_volume, dtype=torch.float32)   # Shape: [128, 240, 240] 

            # Resize to 256x256
            # Add channel dimension for resize, then remove it
            img_tensor = resize(img_tensor.unsqueeze(1)).squeeze(1)   # Shape: [128, 256, 256] 

            # Min-Max normalization of the image
            img_min, img_max = img_tensor.min(), img_tensor.max()  
            normalized_img = (img_tensor - img_min) / (img_max - img_min + 1e-8) 

            self.all_images.append(normalized_img) 

        # Process segmentation labels
        num_samples = labels.shape[1]   # Number of samples (25)
        num_channels = labels.shape[0]   # Number of label channels (3)
        for sample_idx in range(num_samples): 
            label_channels = [] 
            for channel_idx in range(num_channels): 
                # Get full volume data
                label_volume = labels[channel_idx, sample_idx]  # Shape: [155, 240, 240] 

                # Extract 128 slices from the middle portion
                start_idx = (label_volume.shape[0] - 128) // 2 
                end_idx = start_idx + 128 
                label_volume = label_volume[start_idx:end_idx]  # Shape: [128, 240, 240] 

                # Convert to tensor
                lbl_tensor = torch.tensor(label_volume, dtype=torch.float32)   # Shape: [128, 240, 240] 

                # Resize to 256x256
                lbl_tensor = resize(lbl_tensor)  # Shape: [128, 256, 256] 

                label_channels.append(lbl_tensor)  

            # Stack label channels
            label_tensor = torch.stack(label_channels, dim=0)  # Shape: [3, 128, 256, 256] 
            self.all_labels.append(label_tensor)  

        self.augment = augment 

    def __len__(self): 
        """Return the total number of samples in the dataset"""
        return len(self.all_images)  

    def __getitem__(self, idx): 
        """Return a single sample (image and its corresponding label)"""
        image = self.all_images[idx]   # Shape: [1, 128, 256, 256] 
        label = self.all_labels[idx]   # Shape: [3, 128, 256, 256] 
        return image, label 

def show_data_iter(dataset, batch_size=4): 
    """
    Visualize the dataset by displaying sample images and their labels
    Args:
        dataset: Dictionary containing 'images' and 'labels'
        batch_size: Number of samples to display
    """
    images = dataset['images']  # Get data from dictionary
    labels = dataset['labels'] 

    medical_dataset = MRDataset(images, labels) 
    dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=False) 

    for images, labels in dataloader: 
        print(f"Batch shape: {images.shape}")   # Expected: [batch_size, 1, 128, 256, 256] 
        fig, axes = plt.subplots(len(images), 2, figsize=(12, 6 * len(images))) 
        for i in range(len(images)): 
            # Display middle slice
            middle_slice = images[i, 0, images.shape[2]//2].numpy()  
            middle_label = labels[i, 0, labels.shape[2]//2].numpy()  
            axes[i, 0].imshow(middle_slice) 
            axes[i, 1].imshow(middle_label) 
        plt.tight_layout()  
        plt.show()  

