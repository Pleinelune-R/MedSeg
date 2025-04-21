# MEDSEG
## 项目简介：
MedSeg是一个用于医学影像（MRI、CT）分割任务的项目。
## 项目结构：
```plaintext
MedSeg/
├── setup.py 
├── main.py
├── data/
└── data_prepare/
    ├── __init__.py
    ├── load_file.py 
    └── logger.py 
```
## 安装步骤：
``` git clone git@github.com:Pleinelune-R/MedSeg.git ```  
``` cd MedSeg ```  
## 使用方法：
项目使用了 setuptools 进行包管理，你可以通过``` from 子包名  import 模块名 ```来导入模块，例如：  
``` from data_prepare.load_file  import load_dicom_series ```  
main.py：修改folder_path并运行程序。注意，数据存放的文件夹名应为data。  
group_dicoms_by_series：按系列号和实例号进行排序DICOM文件，记录每个系列的切片数量。  
analyze_conversion：分析数据转换过程中的精度损失。  
load_images_series：加载 DICOM 系列数据，并将其存储在一个列表中。  
get_image_data：根据指定的系列号和实例号，获取对应的图像数据。  
plot_single_image：可视化指定的图像数据。  
## 注意事项：
请确保你的数据文件夹中包含 DICOM 格式的文件，并且路径为data/。  
