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
    files = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
    series_dict = defaultdict(list)
    loader = LoadImage(image_only=False, reader=ITKReader())  # return ß(image_array, metadata_dict)

    for f in files:
        try:
            img_obj = loader(f)
            meta_data = img_obj[1]
            print(meta_data)
            print(type(meta_data))

            # maybe useful do not del
            # for tag, value in meta_data.items():
            #     print(f"Tag: {tag} | Value: {value}")
            # print("\n" + "-" * 50 + "\n")  # 分隔线

            # TODO:  更换一种匹配方式，可以考虑匹配ID即可，
            #  现在这种方法太过于长了，冗余计算很多，这种批量的参数获取我建议写成类似C的 switch关键字来实现，
            #  python支持 switch，看一下新版的switch来写一下，for循环效率太低，可扩展性非常差
            #  可以参考下其他的人的办法。或者我这个写法，也不太正规，但是简略清晰一点，
            #  麻烦print 换 log，并且用英文写注释

            series_uid = meta_data[SERIES_UID_KEY] if SERIES_UID_KEY in meta_data else None

            if not series_uid:
                logger.info(f"警告: 无法从文件 {os.path.basename(f)}  中获取SeriesInstanceUID")
                continue

            # TODO:  DICOM斜率，截距没什么大用处，下面这些关键字提取一下并保留.
            # Tag: 0008|1030 | Value: npc/yt what's the meaning of ？
            # Tag: 0008|103e | Value: Ax T2W_STIR SENSE
            # Tag: spacing | Value: [0.23392858 0.23392858 6.        ] spacing info
            # Tag: spatial_shape | Value: [1120 1120    1] #image shape
            # Tag: space | Value: RAS
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
                "RescaleSlope": slope,  # 新增元数据
                "RescaleIntercept": intercept
            })
        except Exception as e:
            print(f"跳过文件 {os.path.basename(f)}:  {str(e)}")
            continue

    # TODO ：这样排序是否正确？理论上应该按切片顺序来排训，一个序列的切片顺序，然后四个序列，应该返回四个一样的数据结构回来
    # 医学逻辑（如解剖位置）决定，而非文件路径
    # 这位置代码写的只能说是逆天，我一眼看不懂，竟然不写注释，这排序太难懂了，这让其他人怎么看
    # 字典顺序是随机的，不建议用字典，这一段重写用list或者其他的数据结构
    # SeriesNumber 每个series_uid到这里的Series保证一样么，x[1][0]["SeriesNumber"]这种可读性太差了，看得太费劲，要不写注释，要不换个方法
    return {k: sorted(v, key=lambda x: x["path"])
            for k, v in
            sorted(series_dict.items(), key=lambda x: int(x[1][0]["SeriesNumber"]) if x[1][0]["SeriesNumber"] else 0)}


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
