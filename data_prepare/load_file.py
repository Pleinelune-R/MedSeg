import logging 
import os 
import glob 
from collections import defaultdict 
import numpy as np 
import matplotlib.pyplot  as plt 
import torch 
from monai.data  import ITKReader 
from monai.transforms  import LoadImage 
from torchvision.transforms  import Resize 
 
from logger import MyLogger 
 
logger = MyLogger("data_prepare") 
SERIES_UID_KEY = '0020|000e' 
 
def group_dicoms_by_series(folder_path): 
    """Get all DICOM files in folder""" 
    files = sorted(glob.glob(os.path.join(folder_path,   "*.dcm"))) 
    data_lists = [] 
    loader = LoadImage(image_only=False, reader=ITKReader())  # return (image_array, metadata_dict) 
 
    for f in files: 
        try: 
            img_obj = loader(f) 
            meta_data = img_obj[1]  # metadata_dict 
 
            # Initialize metadata 
            extracted_data = { 
                "SeriesUID": "",                 # Tag: 0020|000e 
                "SeriesNumber": 0,               # Tag: 0020|0011 
                "InstanceNumber": 0,             # Tag: 0020|0013 
                "StudyDescription": "Unnamed",   # Tag: 0008|1030 
                "SeriesDescription": "Unnamed",  # Tag: 0008|103e 
                "spacing": [1.0, 1.0, 1.0], 
                "spatial_shape": [0, 0, 0], 
                "space": "RAS"                   # Coordinate system 
            } 
 
            for key, value in meta_data.items():     
                try: 
                    # Match the key to extract the corresponding metadata 
                    if key == "0020|000e": 
                        extracted_data["SeriesUID"] = str(value) 
                    elif key == "0020|0011": 
                        extracted_data["SeriesNumber"] = int(value) 
                    elif key == "0020|0013": 
                        extracted_data["InstanceNumber"] = int(value) 
                    elif key == "0008|1030": 
                        extracted_data["StudyDescription"] = str(value) 
                    elif key == "0008|103e": 
                        extracted_data["SeriesDescription"] = str(value) 
                    elif key == "spacing": 
                        extracted_data["spacing"] = [float(v) for v in value[:3]] 
                    elif key == "spatial_shape": 
                        extracted_data["spatial_shape"] = [int(v) for v in value] 
                    elif key == "space": 
                        extracted_data["space"] = str(value).upper()  # normalize RAS/LPS 
                except Exception as e: 
                    # Log a warning if an error occurs while processing the metadata key 
                    logger.warning(f"Error   processing metadata key '{key}': {str(e)}") 
 
            data_lists.append({     
                "path": f, 
                **extracted_data  # You can change the structure of the data_list 
            }) 
 
        except Exception as e: 
            # Skip the file if an error occurs while loading it 
            print(f"Skipping file {os.path.basename(f)}:   {str(e)}") 
 
    # Sort the data list first by SeriesNumber and then by InstanceNumber 
    sorted_files = sorted(data_lists, key=lambda x: (x["SeriesNumber"], x["InstanceNumber"])) 
 
    # Count the numbers of per series 
    series_slice_count = {} 
    series_descriptions = {} 
    for item in sorted_files: 
        series_number = item["SeriesNumber"] 
        series_desc = item["SeriesDescription"] 
        series_descriptions[series_number] = series_desc 
        if series_number not in series_slice_count: 
            series_slice_count[series_number] = 1 
        else: 
            series_slice_count[series_number] += 1 
 
    # Generate a single info message 
    info_message = "Slice information :\n" 
    for series_number, slice_count in series_slice_count.items():     
        series_desc = series_descriptions[series_number] 
        info_message += f"Series {series_number} {series_desc} has {slice_count} slices.\n" 
 
    logger.info(info_message)     
    return sorted_files 
 
def analyze_conversion(original, converted): 
    """Analyzes precision loss during conversion""" 
    original = torch.tensor(original)     
    converted = torch.tensor(converted)     
    original_float = original.float()        
    abs_diff = torch.abs(original_float   - converted)  
    rel_diff = abs_diff / (original_float.abs()   + 1e-6)  # Relative error with numerical stability 
    
    return { 
        'max_abs': abs_diff.max().item(),           # Maximum absolute error 
        'mean_abs': abs_diff.mean().item(),         # Mean absolute error 
        'max_rel': rel_diff.max().item(),           # Maximum relative error 
        'hist': torch.histc(abs_diff,   bins=50)  # Error distribution histogram 
    } 
 
def load_images_series(data_lists): 
    loader = LoadImage(image_only=True, reader=ITKReader()) 
    image_lists = [] 
    resize = Resize((1024, 1024))  # resize
    
    for files in data_lists: 
        raw_data = loader(files['path']) 
        # MetaTensor  -> np.ndarray  
        if hasattr(raw_data, 'numpy'): 
            raw_data = raw_data.numpy()  
        if raw_data.ndim  < 3: 
            # add dimension
            raw_data = np.expand_dims(raw_data,  axis=-1) 
        raw_data = torch.from_numpy(raw_data).permute(2,  0, 1) 
        resized_data = resize(raw_data).permute(1, 2, 0).numpy()  # resize
        img_info = { 
            'SeriesNumber': files['SeriesNumber'], 
            'InstanceNumber': files['InstanceNumber'], 
            'SeriesDescription': files['SeriesDescription'], 
            'ImageData': resized_data 
        } 
        image_lists.append(img_info)      
    return image_lists 
 
def get_image_data(img_lists, series_num, instance_num): 
    """Display a single slice with the specified SeriesNumber and InstanceNumber""" 
    # Filter the matching slice 
    target_slice = [ 
        img for img in img_lists 
        if img['SeriesNumber'] == series_num 
        and img['InstanceNumber'] == instance_num 
    ] 
 
    if not target_slice: 
        print(f"Slice {instance_num} of series {series_num} not found") 
        return 
 
    # Visualization configuration 
    img_info = target_slice[0] 
    if img_info is None: 
        return 
    return img_info 
 
def plot_single_image(img_info): 
    if not img_info: 
        print(f"Image not found.") 
        return 
    img_data = img_info['ImageData'] 
    if img_data.ndim  == 4:  # (Batch, Channel, H, W) 
        img_data = img_data.squeeze(0)     
        img_data = np.squeeze(img_data)     # resize 
 
    if img_data.ndim  not in [2, 3]: 
        raise ValueError(f"Invalid image shape {img_data.shape}.   Expected 2D (H,W) or 3D (H,W,C)") 
 
    plt.figure(figsize=(8,   6)) 
    plt.imshow(img_data,   cmap="gray", 
               vmin=np.percentile(img_data,   1), 
               vmax=np.percentile(img_data,   99)) 
    plt.title(f"   Series {img_info['SeriesNumber']} | Slice {img_info['InstanceNumber']}\n{img_info['SeriesDescription']}") 
    plt.axis('off')     
    plt.tight_layout()     
    plt.show()     
 
class CustomDataset(): 
    def __init__(self, folder_path): 
        self.data_lists   = group_dicoms_by_series(folder_path) 
        self.img_lists   = load_images_series(self.data_lists)     
 
    def __len__(self): 
        return len(self.img_lists)     
 
    def __getitem__(self, idx): 
        img_info = self.img_lists[idx]     
        image = img_info['ImageData'] 
        series_number = img_info['SeriesNumber'] 
        instance_number = img_info['InstanceNumber'] 
        series_description = img_info['SeriesDescription'] 
 
        # Convert image to torch tensor 
        image = image.detach().clone()   if isinstance(image, torch.Tensor) else torch.tensor(image)    
 
        return image, series_number, instance_number, series_description 