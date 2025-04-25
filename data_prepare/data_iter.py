import torch 
from torch.utils.data  import DataLoader 
from load_file import CustomDataset
import matplotlib.pyplot as plt

file_path = "./data/MR00061837-LinLiChai" 
 
dataset = CustomDataset(file_path) 
 
batch_size = 4 
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
 
# iter
for images, series_numbers, instance_numbers, series_descriptions in train_loader: 
    plt.imshow(images[0,:, :, 0],'gray')
    plt.show()

