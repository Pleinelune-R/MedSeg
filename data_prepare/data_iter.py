import torch 
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt 
from logger import MyLogger 
from load_file import generate_dataset 
from torchvision.transforms import Resize 
 
logger = MyLogger("data_iter") 

## TODO: 分离不同序列作为多模态输入；归一化及增强数据；

class MRDataset(Dataset): 
    def __init__(self, images, labels, augment=False):  
        self.images    = [] 
        self.labels    = [] 
        # resize to label size
        if labels: 
            target_size = labels[0].shape[-2:] 
        else: 
            target_size = (256, 256)  # default size 
        resize = Resize(target_size) 
        for img, lbl in zip(images, labels): 
            img_tensor = torch.tensor(img,    dtype=torch.float32).unsqueeze(0)    
            lbl_tensor = torch.tensor(lbl,    dtype=torch.long).unsqueeze(0)    
            # squeeze to remove extra dimension(num_instance)
            resized_img = resize(img_tensor).squeeze(0) 
            resized_lbl = resize(lbl_tensor).squeeze(0) 
            self.images.append(resized_img)    
            self.labels.append(resized_lbl)    
        self.augment    = augment 
 
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
    dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=True) 
 
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