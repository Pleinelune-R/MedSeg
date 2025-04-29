import torch 
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt 
from logger import MyLogger 
from load_file import generate_dataset 
from torchvision.transforms import Resize 
 
logger = MyLogger("data_iter") 

## TODO: 分离不同序列作为多模态输入；

class MRDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images     = []
        self.labels     = []
        target_size = labels[0].shape[-2:] if labels else (256, 256)
        resize = Resize(target_size)
        
        for img, lbl in zip(images, labels):
            # to tensor
            img_tensor = torch.tensor(img,  dtype=torch.float32).unsqueeze(0)   # shape: [1,H,W]
            lbl_tensor = torch.tensor(lbl,  dtype=torch.long).unsqueeze(0) 
            
            # squeeze num_samples and num_instances
            resized_img = resize(img_tensor).squeeze(0)  # shape: [C,H,W]
            resized_lbl = resize(lbl_tensor).squeeze(0)
            
            # Min-Max Scaling to [0,1]
            img_min, img_max = resized_img.min(),  resized_img.max() 
            lbl_max, lbl_min = resized_lbl.max(),  resized_lbl.min()
            normalized_img = (resized_img - img_min) / (img_max - img_min + 1e-8) 
            normalized_lbl = resized_lbl / (lbl_max - lbl_min + 1e-8)
            
            self.images.append(normalized_img) 
            self.labels.append(normalized_lbl) 
            
        self.augment  = augment 
 
    def __len__(self): 
        return len(self.images)    
 
    def __getitem__(self, idx): 
        image = self.images[idx]    
        label = self.labels[idx]    
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
    dataset = generate_dataset(".\\data", ".\\data\\output") 
    show_data_iter(dataset, batch_size=4) 