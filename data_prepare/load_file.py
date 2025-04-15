import logging
import os
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from monai.data import ITKReader
from monai.transforms import LoadImage
import torch

from logger import MyLogger

logger = MyLogger("data_prepare")
SERIES_UID_KEY = '0020|000e'


def group_dicoms_by_series(folder_path): 
    # Get all DICOM files in folder
    files = sorted(glob.glob(os.path.join(folder_path,   "*.dcm"))) 
    data_lists = [] 
    # nii、nii.gz -> NibabelReader 
    # png、jpg、bmp -> PILReader 
    # npz、npy -> NumpyReader
    # others -> ITKReader
    loader = LoadImage(image_only=False, reader=ITKReader())  # return (image_array, metadata_dict) 
 
    for f in files: 
        try: 
            img_obj = loader(f) 
            meta_data = img_obj[1] #metadata_dict
 
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
                    match key: 
                        case "0020|000e": 
                            extracted_data["SeriesUID"] = str(value) 
                        case "0020|0011": 
                            extracted_data["SeriesNumber"] = int(value) 
                        case "0020|0013": 
                            extracted_data["InstanceNumber"] = int(value) 
                        case "0008|1030": 
                            extracted_data["StudyDescription"] = str(value) 
                        case "0008|103e": 
                            extracted_data["SeriesDescription"] = str(value) 
                        case "spacing": 
                            extracted_data["spacing"] = [float(v) for v in value[:3]] 
                        case "spatial_shape": 
                            extracted_data["spatial_shape"] = [int(v) for v in value] 
                        case "space": 
                            extracted_data["space"] = str(value).upper()  # normalize RAS/LPS 
                except Exception as e: 
                    # Log a warning if an error occurs while processing the metadata key 
                    logger.warning(f"Error  processing metadata key '{key}': {str(e)}") 
 
            data_lists.append({  
                "path": f, 
                **extracted_data  # You can change the structure of the data_list 
            }) 
 
        except Exception as e: 
            # Skip the file if an error occurs while loading it 
            print(f"Skipping file {os.path.basename(f)}:  {str(e)}") 
 
    # Sort the data list first by SeriesNumber and then by InstanceNumber 
    sorted_files = sorted(data_lists, key=lambda x: (x["SeriesNumber"], x["InstanceNumber"])) 
    return sorted_files 

folder_path = './MONAI_DATA_DIRECTORY/MR00061837-LinLiChai'
sorted_files = group_dicoms_by_series(folder_path)
print(sorted_files[1]["SeriesNumber"])
def analyze_conversion(original, converted):
    """新增：量化转换精度损失"""
    original_float = original.float()
    abs_diff = torch.abs(original_float - converted)
    rel_diff = abs_diff / (original_float.abs() + 1e-6)

    return {
        'max_abs': abs_diff.max().item(),
        'mean_abs': abs_diff.mean().item(),
        'max_rel': rel_diff.max().item(),
        'hist': torch.histc(abs_diff, bins=50)
    }

def visualize_multi_series(series_dict, slices_per_series=3):
    loader = LoadImage(image_only=True, reader=ITKReader())

    for series_id, files in series_dict.items():
        fig, axes = plt.subplots(2, slices_per_series, figsize=(15, 8))
        fig.suptitle(f"Series  {files[0]['SeriesNumber']}: {files[0]['SeriesDescription']}", fontsize=12)

        for i, f_info in enumerate(files[:slices_per_series]):
            # 原始数据 
            img_orig = loader(f_info["path"])
            axes[0, i].imshow(img_orig, cmap="gray",
                              vmin=np.percentile(img_orig, 1),
                              vmax=np.percentile(img_orig, 99))
            axes[0, i].set_title(f"Original\nSlice {i + 1}")

            # TODO ：不需要针对CT来，到时候CT一个函数MRI一个函数，成像原理不一样，不需要兼容。
            # 转换后数据（自动选择CT或常规转换）
            if "CT" in files[0]['SeriesDescription'].upper():
                img_converted = img_orig.float() * f_info["RescaleSlope"] + f_info["RescaleIntercept"]
            else:
                img_converted = img_orig.float()

            axes[1, i].imshow(img_converted, cmap="gray",
                              vmin=np.percentile(img_converted, 1),
                              vmax=np.percentile(img_converted, 99))
            axes[1, i].set_title(f"Converted\nMax err: {analyze_conversion(img_orig, img_converted)['max_abs']:.2f}")

            for ax in axes[:, i]:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

def load_dicom_series(folder_path, visualize=True, convert_type='auto'):
    """
    convert_type: 'auto'|'ct'|'raw' 选择转换方式 
    """
    series_dict = group_dicoms_by_series(folder_path)
    loader = LoadImage(image_only=True, reader=ITKReader())
    torch_series_data = {}
    global_stats = []  # 新增：全局统计记录 

    logger.info(f"发现 {len(series_dict)} 个序列:")
    for uid, files in series_dict.items():
        print(f"→ 序列 {files[0]['SeriesNumber']}: {files[0]['SeriesDescription']} (共 {len(files)} 张切片)")

    if visualize:
        visualize_multi_series(series_dict)

    for series_id, files in series_dict.items():
        series_tensors = []
        series_stats = []

        for f_info in files:
            img = loader(f_info["path"])

            # 根据类型转换 
            if convert_type == 'ct' or (convert_type == 'auto' and "CT" in files[0]['SeriesDescription'].upper()):
                converted = img.float() * f_info["RescaleSlope"] + f_info["RescaleIntercept"]
            else:
                converted = img.float()

                # 记录统计
            stats = analyze_conversion(img, converted)
            series_stats.append(stats)
            series_tensors.append(converted)

        # 打印本序列统计
        # 可以用logger整体输出，不需要多个print看着难受
        print(f"\n序列 {files[0]['SeriesNumber']} 转换精度:")
        print(f"最大绝对误差: {max(s['max_abs'] for s in series_stats):.4f}")
        print(f"平均相对误差: {np.mean([s['mean_abs'] for s in series_stats]):.4f}")
        global_stats.extend(series_stats)

        # 堆叠3D张量 
        if series_tensors:
            torch_series_data[series_id] = {
                'data_prepare': torch.stack(series_tensors, dim=0),
                'info': {
                    **{k: files[0][k] for k in ['SeriesNumber', 'SeriesDescription']},
                    'SliceCount': len(files),
                    'ConversionStats': series_stats  # 新增统计信息 
                }
            }

    print("\n全局转换精度摘要:")
    print(f"总切片数: {len(global_stats)}")
    print(f"最大绝对误差: {max(s['max_abs'] for s in global_stats):.4f}")
    print(f"平均绝对误差: {np.mean([s['mean_abs'] for s in global_stats]):.4f}")

    plt.figure(figsize=(10, 5))
    all_errors = torch.cat(
        [s['hist'] for s in torch_series_data[next(iter(torch_series_data))]['info']['ConversionStats']])
    plt.hist(all_errors.numpy(), bins=50, log=True)
    plt.title("Global  Absolute Error Distribution")
    plt.xlabel("Pixel  Value Difference")
    plt.show()

    return torch_series_data
