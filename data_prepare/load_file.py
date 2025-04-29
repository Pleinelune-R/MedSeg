import os 
import pydicom 
from dcmrtstruct2nii import dcmrtstruct2nii 
import SimpleITK as sitk 
import matplotlib.pyplot  as plt 
import time 
 
from logger import MyLogger 
 
logger = MyLogger("data_prepare") 
SERIES_UID_KEY = '0020|000e' 
 
 
def find_rtstruct_files(folder_path): 
    rtstruct_files = [] 
    for root, _, files in os.walk(folder_path):  
        for file in files: 
            if file.endswith(".dcm"):  
                try: 
                    ds = pydicom.dcmread(os.path.join(root,  file), force=True) 
                    if ds.Modality == "RTSTRUCT": 
                        # check if is already in the list 
                        file_path = os.path.join(root,  file) 
                        if file_path not in rtstruct_files: 
                            rtstruct_files.append(file_path)  
                except Exception as e: 
                    logger.warning(f"Skipping  non - DICOM file: {file}, error: {e}") 
    return rtstruct_files 
 
 
def match_dicom_dir(rtstruct_path): 
    try: 
        # 读取 RTSTRUCT DICOM 文件 
        rtstruct_ds = pydicom.dcmread(rtstruct_path,  force=True) 
        # 获取 RTSTRUCT 的 SeriesNumber 和 StudyInstanceUID 
        series_number = rtstruct_ds.SeriesNumber 
        study_uid = rtstruct_ds.StudyInstanceUID 
        print(f"SeriesNumber: {series_number}, StudyInstanceUID: {study_uid}") 
 
        # 获取 RTSTRUCT 文件所在的基础目录 
        base_dir = os.path.dirname(rtstruct_path)  
 
        # 用于存储所有匹配的 DICOM 文件路径 
        matched_files = [] 
 
        # 遍历基础目录及其子目录 
        for root, _, files in os.walk(base_dir):  
            for file in files: 
                if file.endswith(".dcm"):  
                    try: 
                        # 读取当前 DICOM 文件 
                        ds = pydicom.dcmread(os.path.join(root,  file), force=True) 
                        # 检查 Modality 不是 RTSTRUCT，并且 SeriesNumber 和 StudyInstanceUID 匹配 
                        if ds.Modality != "RTSTRUCT" and ds.SeriesNumber == series_number and ds.StudyInstanceUID == study_uid: 
                            matched_files.append(os.path.join(root,  file)) 
                    except Exception as e: 
                        # 记录非 DICOM 文件的错误信息 
                        logger.warning(f"Skipping  non - DICOM file: {file}, error: {e}") 
 
        # 如果没有找到匹配的文件，记录警告信息 
        if not matched_files: 
            logger.warning(f"No  matching DICOM files found for: {rtstruct_path}") 
            return None 
 
        return matched_files 
    except Exception as e: 
        # 处理读取 RTSTRUCT 文件时的错误 
        logger.error(f"Error  reading RTSTRUCT file: {rtstruct_path}, error: {e}") 
        return None 
 
 
def process_rtstruct(rtstruct_path, dicom_files, output_dir, index): 
    # Generate custom filenames 
    label_filename = f"mask_GTV_{index}.nii.gz"  
    custom_label_path = os.path.join(output_dir,  label_filename) 
 
    # Convert RTStruct to NIfTI 
    dicom_dir = os.path.dirname(dicom_files[0])  
    dcmrtstruct2nii(rtstruct_path, dicom_dir, output_dir, structures=['GTV']) 
 
    # Rename label file 
    default_label_path = os.path.join(output_dir,  "mask_GTV.nii.gz")  
    if os.path.exists(default_label_path):  
        os.rename(default_label_path,  custom_label_path) 
    else: 
        logger.warning(f"Default  label file {default_label_path} not found.") 
 
    # 存储每个 DICOM 文件的信息和图像 
    dataset = [] 
    for dicom_file in dicom_files: 
        try: 
            # 读取 DICOM 文件 
            ds = pydicom.dcmread(dicom_file,  force=True) 
            # 读取 DICOM 图像 
            image_sitk = sitk.ReadImage(dicom_file) 
            spacing = image_sitk.GetSpacing() 
            spatial_shape = image_sitk.GetSize() 
 
            series_number = ds.SeriesNumber 
            study_desc = ds.StudyDescription if 'StudyDescription' in ds else '' 
            series_desc = ds.SeriesDescription if 'SeriesDescription' in ds else '' 
 
            # 读取标签图像 
            if os.path.exists(custom_label_path):  
                label_sitk = sitk.ReadImage(custom_label_path) 
                label = sitk.GetArrayFromImage(label_sitk) 
            else: 
                logger.warning(f"Custom  label file {custom_label_path} not found.") 
                label = None 
 
            # 构建字典 
            sample_dict = { 
                'SeriesNumber': series_number, 
                'StudyDescription': study_desc, 
                'SeriesDescription': series_desc, 
                'spacing': spacing, 
                'spatial_shape': spatial_shape, 
                'space': image_sitk.GetDirection(),  # Spatial direction matrix 
                'label': label, 
                'image': sitk.GetArrayFromImage(image_sitk) 
            } 
            dataset.append(sample_dict)  
        except Exception as e: 
            logger.warning(f"Error  processing DICOM file: {dicom_file}, error: {e}") 
 
    return dataset 
 
 
def generate_dataset(folder_path, output_dir): 
    # Create the output directory 
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
 
    # Filter RTStruct files 
    rtstruct_files = find_rtstruct_files(folder_path) 
 
    # Initialize the dataset list 
    dataset = [] 
 
    # Process each RTStruct 
    for index, rtstruct_path in enumerate(rtstruct_files, start=1): 
        dicom_files = match_dicom_dir(rtstruct_path) 
        if not dicom_files: 
            logger.warning(f"No  matching DICOM files found: {rtstruct_path}") 
            continue 
 
        # Process and get the dictionary data 
        sample_data = process_rtstruct(rtstruct_path, dicom_files, output_dir, index) 
        if sample_data: 
            dataset.extend(sample_data)  
 
    return dataset 
 