import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from logger import get_logger

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
        # images shape: [4, 369, 155, 240, 240]
        num_samples = images.shape[1]  # 369 samples
        for sample_idx in range(num_samples):
            # Get full volume data for all channels
            img_volume = images[:, sample_idx]  # Shape: [4, 155, 240, 240] for 4 channels

            # Extract 128 slices from the middle portion for each channel
            start_idx = (img_volume.shape[1] - 128) // 2
            end_idx = start_idx + 128
            img_volume = img_volume[:, start_idx:end_idx]  # Shape: [4, 128, 240, 240] 

            # Convert to tensor
            img_tensor = torch.tensor(img_volume, dtype=torch.float32)  # Shape: [4, 128, 240, 240]

            # Resize to 256x256 for each channel
            resized_channels = []
            for channel in range(img_tensor.shape[0]):
                # Resize each channel separately
                resized_channel = resize(img_tensor[channel].unsqueeze(0)).squeeze(0)  # Shape: [128, 256, 256]
                resized_channels.append(resized_channel)

            # Stack all channels
            img_tensor = torch.stack(resized_channels, dim=0)  # Shape: [4, 128, 256, 256]

            # Min-Max normalization of each channel separately
            normalized_channels = []
            for channel in range(img_tensor.shape[0]):
                channel_min, channel_max = img_tensor[channel].min(), img_tensor[channel].max()
                normalized_channel = (img_tensor[channel] - channel_min) / (channel_max - channel_min + 1e-8)
                normalized_channels.append(normalized_channel)

            normalized_img = torch.stack(normalized_channels, dim=0)  # Shape: [4, 128, 256, 256]
            self.all_images.append(normalized_img)

            # Process segmentation labels
        # labels shape: [3, 369, 155, 240, 240]
        num_samples = labels.shape[1]  # 369 samples
        num_channels = labels.shape[0]  # 3 label channels
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
                lbl_tensor = torch.tensor(label_volume, dtype=torch.float32)  # Shape: [128, 240, 240]

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
        image = self.all_images[idx]  # Shape: [4, 128, 256, 256] for 4 channels
        label = self.all_labels[idx]  # Shape: [3, 128, 256, 256]
        return image, label

    # def show_data_iter(dataset, batch_size=4):
#     """
#     Visualize the dataset by displaying sample images and their labels
#     Args:
#         dataset: Dictionary containing 'images' and 'labels'
#         batch_size: Number of samples to display
#     """
#     images = dataset['images']  # Shape: [4, n, 155, 240, 240]
#     labels = dataset['labels']  # Shape: [3, n, 155, 240, 240]

#     medical_dataset = MRDataset(images, labels) 
#     dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=False) 

#     for batch_images, batch_labels in dataloader: 
#         print(f"Batch images shape: {batch_images.shape}")   # Expected: [batch_size, 4, 128, 256, 256]
#         print(f"Batch labels shape: {batch_labels.shape}")   # Expected: [batch_size, 3, 128, 256, 256]

#         # Create figure with appropriate number of rows and columns
#         num_samples = len(batch_images)
#         fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))  # 5 columns: 4 channels + 1 label

#         # Handle single sample case
#         if num_samples == 1:
#             axes = axes.reshape(1, -1)

#         for i in range(num_samples): 
#             # Display middle slice for each channel
#             middle_slice_idx = batch_images.shape[2] // 2

#             # Display all 4 image channels
#             for channel in range(4):
#                 middle_slice = batch_images[i, channel, middle_slice_idx].cpu().numpy()
#                 axes[i, channel].imshow(middle_slice, cmap='gray')
#                 axes[i, channel].set_title(f'Channel {channel+1}')
#                 axes[i, channel].axis('off')

#             # Display first label channel (Whole Tumor mask)
#             middle_label = batch_labels[i, 0, middle_slice_idx].cpu().numpy()
#             axes[i, 4].imshow(middle_label, cmap='gray')
#             axes[i, 4].set_title('Whole Tumor Mask')
#             axes[i, 4].axis('off')

#         plt.tight_layout()  
#         plt.show()
#         break  # Only show first batch
