import os 
import glob 
from collections import defaultdict 
import numpy as np 
import matplotlib.pyplot as plt 
from monai.data import ITKReader
from monai.transforms import LoadImage
import torch

# 1. 分组多序列 DICOM 文件 
def group_dicoms_by_series(folder_path):
    """按 SeriesInstanceUID 自动分组 DICOM 文件"""
    files = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
    series_dict = defaultdict(list)
    
    # nii、nii.gz -> NibabelReader
    # png、jpg、bmp -> PILReader
    # npz、npy -> NumpyReader
    # 其他 -> ITKReader
    # 切Reader先放着以后再说

    loader = LoadImage(image_only=False, reader=ITKReader())
    
    for f in files:
        try:
            img_obj = loader(f)
            meta_data = img_obj[1]  # LoadImage： (image_data, meta_data) 
            
            series_uid = None
            for key in ['series_instance_uid', '0020|000e', 'SeriesInstanceUID']:
                if key in meta_data:
                    series_uid = meta_data[key]
                    break
            
            if not series_uid:
                print(f"警告: 无法从文件 {os.path.basename(f)} 中获取SeriesInstanceUID")
                continue
                
            # SeriesNumber
            series_number = 0
            for key in ['series_number', '0020|0011', 'SeriesNumber']:
                if key in meta_data:
                    try:
                        series_number = int(meta_data[key])
                    except (ValueError, TypeError):
                        series_number = 0
                    break
            
            # SeriesDescription
            series_desc = "Unnamed"
            for key in ['series_description', '0008|103e', 'SeriesDescription']:
                if key in meta_data:
                    series_desc = meta_data[key]
                    break
     
            series_dict[series_uid].append({
                "path": f,
                "SeriesNumber": series_number,
                "SeriesDescription": series_desc 
            })
        except Exception as e:
            # 处理没有像素数据的文件（如RTSTRUCT）
            print(f"跳过文件 {os.path.basename(f)}: {str(e)}")
            continue
 
    # 按 SeriesNumber 排序 
    if not series_dict:
        print("警告: 未找到有效的DICOM序列")
        return {}
        
    return {k: sorted(v, key=lambda x: x["path"])
            for k, v in sorted(series_dict.items(), 
                              key=lambda x: int(x[1][0]["SeriesNumber"]) if x[1][0]["SeriesNumber"] else 0)}
 
# 2. 多序列可视化函数 
def visualize_multi_series(series_dict, slices_per_series=3):
    """显示每个序列的前 N 个切片"""

    # 转换格式后同样要修改
    loader = LoadImage(image_only=True, reader=ITKReader())
 
    for series_id, files in series_dict.items(): 
        imgs = []
        for f_info in files[:slices_per_series]:
            f_path = f_info["path"]
            img_tensor = loader(f_path)
            # 转换为 NumPy 数组 
            img_np = img_tensor.numpy() 
            imgs.append(img_np) 
 
        # 创建子图 
        fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
        if len(imgs) == 1:  # 处理只有一个切片的情况
            axes = [axes]
            
        fig.suptitle(f"Series {files[0]['SeriesNumber']}: {files[0]['SeriesDescription']}",
                     fontsize=12, y=1.05)
 
        for i, (img, ax) in enumerate(zip(imgs, axes)):
            ax.imshow(img, cmap="gray",
                     vmin=np.percentile(img, 1),
                     vmax=np.percentile(img, 99))
            ax.set_title(f"Slice {i+1}/{len(files)}")
            ax.axis("off") 
 
        plt.tight_layout() 
        plt.show() 

# 3. 封装函数：加载DICOM文件并返回torch数据
def load_dicom_series(folder_path, visualize=True, slices_per_series=3):
    """
    加载DICOM序列并返回torch格式的数据
    
    参数:
        folder_path (str): DICOM文件所在文件夹路径
        visualize (bool): 是否可视化显示序列，默认为True
        slices_per_series (int): 每个序列显示的切片数，默认为3
        
    返回:
        dict: 包含每个序列的torch张量数据，格式为 {series_id: {'data': torch_tensor, 'info': series_info}}
    """
    # 步骤1: 自动分组多序列
    series_dict = group_dicoms_by_series(folder_path)
    
    # 打印序列信息
    print(f"发现 {len(series_dict)} 个序列:")
    for uid, files in series_dict.items(): 
        print(f"→ 序列 {files[0]['SeriesNumber']}: {files[0]['SeriesDescription']} (共 {len(files)} 张切片)")
    
    # 步骤2: 可选的可视化
    if visualize:
        visualize_multi_series(series_dict, slices_per_series)
    
    # 步骤3: 加载所有序列数据并转换为torch张量
    loader = LoadImage(image_only=True, reader=ITKReader())
    torch_series_data = {}
    
    for series_id, files in series_dict.items():
        # 收集该序列的所有切片
        series_images = []
        for f_info in files:
            f_path = f_info["path"]
            img_tensor = loader(f_path)
            series_images.append(img_tensor)
        
        # 将所有切片堆叠成一个3D张量 [depth, height, width]
        if series_images:
            stacked_tensor = torch.stack(series_images, dim=0)
            
            # 保存张量数据和序列信息
            torch_series_data[series_id] = {
                'data': stacked_tensor,
                'info': {
                    'SeriesNumber': files[0]['SeriesNumber'],
                    'SeriesDescription': files[0]['SeriesDescription'],
                    'SliceCount': len(files)
                }
            }
    
    return torch_series_data

# 示例用法
if __name__ == "__main__":
    folder_path = "./MONAI_DATA_DIRECTORY/MR00061837-LinLiChai"  # 文件夹路径
    
    # 调用封装函数
    series_data = load_dicom_series(folder_path)
    
    # 展示返回的torch数据信息
    print("\n返回的Torch数据信息:")
    for series_id, data_dict in series_data.items():
        tensor = data_dict['data']
        info = data_dict['info']
        print(f"序列 {info['SeriesNumber']}: {info['SeriesDescription']}")
        print(f"  形状: {tensor.shape}")
        print(f"  数据类型: {tensor.dtype}")
        print(f"  数据范围: [{tensor.min().item():.2f}, {tensor.max().item():.2f}]")