import os 
import glob 
from collections import defaultdict 
import numpy as np 
import matplotlib.pyplot  as plt 
from monai.data  import ITKReader 
from monai.transforms  import LoadImage 
import torch 
 
def group_dicoms_by_series(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path,  "*.dcm")))
    series_dict = defaultdict(list)
    loader = LoadImage(image_only=False, reader=ITKReader())
    
    for f in files:
        try:
            img_obj = loader(f)
            meta_data = img_obj[1]
            
            series_uid = None 
            for key in ['series_instance_uid', '0020|000e', 'SeriesInstanceUID']:
                if key in meta_data:
                    series_uid = meta_data[key]
                    break 
            
            if not series_uid:
                print(f"警告: 无法从文件 {os.path.basename(f)}  中获取SeriesInstanceUID")
                continue 
                
            # 获取DICOM斜率/截距
            slope = 1.0 
            intercept = 0.0 
            for key in ['rescale_slope', '0028|1053', 'RescaleSlope']:
                if key in meta_data:
                    slope = float(meta_data[key])
                    break 
            for key in ['rescale_intercept', '0028|1052', 'RescaleIntercept']:
                if key in meta_data:
                    intercept = float(meta_data[key])
                    break 
 
            series_number = 0 
            for key in ['series_number', '0020|0011', 'SeriesNumber']:
                if key in meta_data:
                    try:
                        series_number = int(meta_data[key])
                    except (ValueError, TypeError):
                        series_number = 0 
                    break 
            
            series_desc = "Unnamed"
            for key in ['series_description', '0008|103e', 'SeriesDescription']:
                if key in meta_data:
                    series_desc = meta_data[key]
                    break 
     
            series_dict[series_uid].append({
                "path": f,
                "SeriesNumber": series_number,
                "SeriesDescription": series_desc,
                "RescaleSlope": slope,    # 新增元数据 
                "RescaleIntercept": intercept 
            })
        except Exception as e:
            print(f"跳过文件 {os.path.basename(f)}:  {str(e)}")
            continue 
 
    return {k: sorted(v, key=lambda x: x["path"])
            for k, v in sorted(series_dict.items(),  
                              key=lambda x: int(x[1][0]["SeriesNumber"]) if x[1][0]["SeriesNumber"] else 0)}
 
def analyze_conversion(original, converted):
    """新增：量化转换精度损失"""
    original_float = original.float() 
    abs_diff = torch.abs(original_float  - converted)
    rel_diff = abs_diff / (original_float.abs()  + 1e-6)
    
    return {
        'max_abs': abs_diff.max().item(), 
        'mean_abs': abs_diff.mean().item(), 
        'max_rel': rel_diff.max().item(), 
        'hist': torch.histc(abs_diff,  bins=50)
    }
 
def visualize_multi_series(series_dict, slices_per_series=3):
    loader = LoadImage(image_only=True, reader=ITKReader())
 
    for series_id, files in series_dict.items():  
        fig, axes = plt.subplots(2,  slices_per_series, figsize=(15, 8))
        fig.suptitle(f"Series  {files[0]['SeriesNumber']}: {files[0]['SeriesDescription']}", fontsize=12)
        
        for i, f_info in enumerate(files[:slices_per_series]):
            # 原始数据 
            img_orig = loader(f_info["path"])
            axes[0,i].imshow(img_orig, cmap="gray", 
                           vmin=np.percentile(img_orig,  1),
                           vmax=np.percentile(img_orig,  99))
            axes[0,i].set_title(f"Original\nSlice {i+1}")
            
            # 转换后数据（自动选择CT或常规转换）
            if "CT" in files[0]['SeriesDescription'].upper():
                img_converted = img_orig.float()  * f_info["RescaleSlope"] + f_info["RescaleIntercept"]
            else:
                img_converted = img_orig.float() 
            
            axes[1,i].imshow(img_converted, cmap="gray",
                           vmin=np.percentile(img_converted,  1),
                           vmax=np.percentile(img_converted,  99))
            axes[1,i].set_title(f"Converted\nMax err: {analyze_conversion(img_orig, img_converted)['max_abs']:.2f}")
            
            for ax in axes[:,i]:
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
 
    print(f"发现 {len(series_dict)} 个序列:")
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
                converted = img.float()  * f_info["RescaleSlope"] + f_info["RescaleIntercept"]
            else:
                converted = img.float() 
            
            # 记录统计 
            stats = analyze_conversion(img, converted)
            series_stats.append(stats) 
            series_tensors.append(converted) 
        
        # 打印本序列统计 
        print(f"\n序列 {files[0]['SeriesNumber']} 转换精度:")
        print(f"最大绝对误差: {max(s['max_abs'] for s in series_stats):.4f}")
        print(f"平均相对误差: {np.mean([s['mean_abs']  for s in series_stats]):.4f}")
        global_stats.extend(series_stats) 
        
        # 堆叠3D张量 
        if series_tensors:
            torch_series_data[series_id] = {
                'data': torch.stack(series_tensors,  dim=0),
                'info': {
                    **{k: files[0][k] for k in ['SeriesNumber', 'SeriesDescription']},
                    'SliceCount': len(files),
                    'ConversionStats': series_stats  # 新增统计信息 
                }
            }
    
    print("\n全局转换精度摘要:")
    print(f"总切片数: {len(global_stats)}")
    print(f"最大绝对误差: {max(s['max_abs'] for s in global_stats):.4f}")
    print(f"平均绝对误差: {np.mean([s['mean_abs']  for s in global_stats]):.4f}")

    plt.figure(figsize=(10,5)) 
    all_errors = torch.cat([s['hist']  for s in torch_series_data[next(iter(torch_series_data))]['info']['ConversionStats']])
    plt.hist(all_errors.numpy(),  bins=50, log=True)
    plt.title("Global  Absolute Error Distribution")
    plt.xlabel("Pixel  Value Difference")
    plt.show() 
    
    return torch_series_data 
 
if __name__ == "__main__":
    folder_path = "./MONAI_DATA_DIRECTORY/MR00061837-LinLiChai"
    series_data = load_dicom_series(folder_path, convert_type='auto')
    
