from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
 
path1 = './data/MR00061837-LinLiChai/RTSTRUCT_1.2.276.0.7230010.3.1.4.753410915.7152.1718301137.957.dcm'   # rtstruct.dcm
path2 = './data/MR00061837-LinLiChai'               # 对应的图像文件夹
path3 = './data'           # 转换后的nii结果输出文件夹
 
print(list_rt_structs(path1))   # 输出RTSTRUCT文件的信息列表
 
dcmrtstruct2nii(path1, path2, path3)
 
main_dir = "./image"
 
for root, dirs, files in os.walk(main_dir):
    for file in files:
        if file.endswith(".nii.gz"):
            file_path = os.path.join(root, file)
            adjust_window(file_path, 40, 400)


adc_labels_dir = "./data"


for file_name in os.listdir(adc_labels_dir):
    if file_name.endswith(".nii.gz"):
        file_path = os.path.join(adc_labels_dir, file_name)
        img = nib.load(file_path)
        img_data = img.get_fdata()
        unique_values = np.unique(img_data)
        print(f"文件: {file_name}, 唯一值: {unique_values}")
