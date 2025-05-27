import os 
import pydicom 
from dcmrtstruct2nii import dcmrtstruct2nii 
import SimpleITK as SpITK
 
from logger import get_logger
 
logger = get_logger("data_prepare") 
 
 
def find_rtstruct_files(folder_path):  # find RTSTRUCT files 
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
                    logger.warning(f"Skipping  non-DICOM file: {file}, error: {e}") 
    return rtstruct_files 
 
 
def match_dicom_file(rtstruct_path): 
    try: 
        # read RTSTRUCT DICOM 
        rtstruct_ds = pydicom.dcmread(rtstruct_path,  force=True) 
        # get SeriesNumber and StudyInstanceUID 
        series_number = rtstruct_ds.SeriesNumber 
        study_uid = rtstruct_ds.StudyInstanceUID 
 
        base_dir = os.path.dirname(rtstruct_path)  
 
        matched_files = [] 
 
        # Traverse 
        for root, _, files in os.walk(base_dir):  
            for file in files: 
                if file.endswith(".dcm"):  
                    try: 
                        # Read DICOM file and match SeriesNumber and StudyInstanceUID 
                        ds = pydicom.dcmread(os.path.join(root,  file), force=True) 
                        if ds.Modality != "RTSTRUCT" and ds.SeriesNumber == series_number and ds.StudyInstanceUID == study_uid: 
                            matched_files.append(os.path.join(root,  file)) 
                    except Exception as e: 
                        logger.warning(f"Skipping  non - DICOM file: {file}, error: {e}") 
 
        if not matched_files: 
            logger.warning(f"No  matching DICOM files found for: {rtstruct_path}") 
            return None 
        return matched_files 
    except Exception as e: 
        # Error reading RTSTRUCT file 
        logger.error(f"Error  reading RTSTRUCT file: {rtstruct_path}, error: {e}") 
        return None 
 
 
def process_rtstruct(rtstruct_path, dicom_files, output_dir, index): 
    try: 
        # read RTSTRUCT DICOM to get StudyInstanceUID, SeriesNumber and Patient's name 
        rtstruct_ds = pydicom.dcmread(rtstruct_path,  force=True) 
        series_number = rtstruct_ds.SeriesNumber 
        patient_name = str(rtstruct_ds.PatientName) if 'PatientName' in rtstruct_ds else 'Unknown' 
 
        # Generate custom filenames 
        label_filename = f"mask_{patient_name}_{series_number}.nii.gz"  
        custom_label_path = os.path.join(output_dir,  label_filename) 
 
        # Check if the file already exists 
        if os.path.exists(custom_label_path):  
            logger.warning(f"File  {custom_label_path} already exists, skipping conversion.") 
        else: 
            # Convert RTStruct to NIfTI 
            dicom_dir = os.path.dirname(dicom_files[0])  
            dcmrtstruct2nii(rtstruct_path, dicom_dir, output_dir, structures=['GTV']) 
 
            # Rename label file 
            default_label_path = os.path.join(output_dir,  "mask_GTV.nii.gz")  
            if os.path.exists(default_label_path):  
                os.rename(default_label_path,  custom_label_path) 
            else: 
                logger.warning(f"Default  label file {default_label_path} not found.") 
                return [] 
 
        # Sort DICOM files by InstanceNumber 
        dicom_files_with_instance_number = [] 
        for dicom_file in dicom_files: 
            try: 
                ds = pydicom.dcmread(dicom_file,  force=True) 
                instance_number = ds.InstanceNumber if 'InstanceNumber' in ds else 0 
                dicom_files_with_instance_number.append((instance_number,  dicom_file)) 
            except Exception as e: 
                logger.warning(f"Error  reading DICOM file for sorting: {dicom_file}, error: {e}") 
        dicom_files_with_instance_number.sort()  
        sorted_dicom_files = [file for _, file in dicom_files_with_instance_number] 
 
        dataset = [] 
        # read label file 
        if os.path.exists(custom_label_path):  
            label_sitk = SpITK.ReadImage(custom_label_path)
            label = SpITK.GetArrayFromImage(label_sitk)
        else: 
            logger.warning(f"Custom  label file {custom_label_path} not found.") 
            label = None 
 
        for i, dicom_file in enumerate(sorted_dicom_files): 
            try: 
                # read DICOM image 
                ds = pydicom.dcmread(dicom_file,  force=True) 
                image_sitk = SpITK.ReadImage(dicom_file)
                spacing = image_sitk.GetSpacing() 
                spatial_shape = image_sitk.GetSize() 
 
                series_number = ds.SeriesNumber 
                study_desc = ds.StudyDescription if 'StudyDescription' in ds else '' 
                series_desc = ds.SeriesDescription if 'SeriesDescription' in ds else '' 
 
                # extract the label of the corresponding slice 
                if label is not None: 
                    slice_label = label[i] if i < len(label) else None 
                else: 
                    slice_label = None 
 
                # data dictionary 
                data_dict = { 
                    'SeriesNumber': series_number, 
                    'StudyDescription': study_desc, 
                    'SeriesDescription': series_desc, 
                    'spacing': spacing, 
                    'spatial_shape': spatial_shape, 
                    'space': image_sitk.GetDirection(),  # Spatial direction matrix 
                    'label': slice_label, 
                    'image': SpITK.GetArrayFromImage(image_sitk)
                } 
                dataset.append(data_dict)  
            except Exception as e: 
                logger.warning(f"Error  processing DICOM file: {dicom_file}, error: {e}") 
 
        return dataset 
    except Exception as e: 
        logger.error(f"Error processing RTSTRUCT file: {rtstruct_path}, error: {e}")
        return [] 
 
 
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
        dicom_files = match_dicom_file(rtstruct_path) 
        if not dicom_files: 
            logger.warning(f"No  matching DICOM files found: {rtstruct_path}") 
            continue 
 
        # Process and get the dictionary data 
        sample_data = process_rtstruct(rtstruct_path, dicom_files, output_dir, index) 
        if sample_data: 
            dataset.extend(sample_data)  
 
    return dataset 

 
def read_nii_files(folder_path): 
    dataset = [] 
    # 用于存储标签文件路径的字典，键为文件名主体，值为标签文件路径 
    label_files = {} 
 
    # 第一次遍历，找出所有的标签文件 
    for root, _, files in os.walk(folder_path):  
        for file in files: 
            if file.endswith("_seg.nii")  or file.endswith("_seg.nii.gz"):  
                # 获取文件名主体，例如从 BraTS20_Training_354_seg.nii  得到 BraTS20_Training_354 
                base_name = file.rsplit("_",  1)[0] 
                label_files[base_name] = os.path.join(root,  file) 
 
    sample_data = {} 
    # 第二次遍历，读取图像文件并关联标签 
    for root, _, files in os.walk(folder_path):  
        for file in files: 
            # 检查文件是否为 flair, t1, t2, t1ce 结尾的图像文件 
            if file.endswith(("_flair.nii",  "_flair.nii.gz",  "_t1.nii",  "_t1.nii.gz",  "_t2.nii",  "_t2.nii.gz",  "_t1ce.nii",  "_t1ce.nii.gz")):  
                # 获取文件名主体，用于查找对应的标签文件 
                base_name = file.rsplit("_",  1)[0] 
                # 查找对应的标签文件 
                label_path = label_files.get(base_name)  
                if not label_path: 
                    # 如果没有找到对应的标签文件，跳过该图像文件 
                    logger.warning(f"No  label found for file: {file}, skipping...") 
                    continue 
 
                try: 
                    # 构建文件的完整路径 
                    nii_path = os.path.join(root,  file) 
                    # 读取 NIfTI 图像 
                    image_sitk = SpITK.ReadImage(nii_path) 
                    # 获取图像的间距 
                    spacing = image_sitk.GetSpacing() 
                    # 获取图像的空间形状 
                    spatial_shape = image_sitk.GetSize() 
 
                    # 简单起见，为 SeriesNumber、StudyDescription 和 SeriesDescription 设置默认值 
                    series_number = 0 
                    study_desc = "N/A" 
                    series_desc = "N/A" 
 
                    # 读取标签文件 
                    label_sitk = SpITK.ReadImage(label_path) 
                    label = SpITK.GetArrayFromImage(label_sitk) 
 
                    # 创建包含图像信息的字典 
                    data_dict = { 
                        'SeriesNumber': series_number, 
                        'StudyDescription': study_desc, 
                        'SeriesDescription': series_desc, 
                        'spacing': spacing, 
                        'spatial_shape': spatial_shape, 
                        'space': image_sitk.GetDirection(), 
                        'label': label, 
                        'image': SpITK.GetArrayFromImage(image_sitk) 
                    } 
 
                    if base_name not in sample_data: 
                        sample_data[base_name] = {'images': [], 'label': label} 
                    sample_data[base_name]['images'].append(data_dict['image']) 
 
                except Exception as e: 
                    # 记录读取文件时的错误信息 
                    logger.warning(f"Error  reading NIfTI file: {file}, error: {e}") 
 
    # 整理样本数据 
    for base_name, data in sample_data.items():  
        images = data['images'] 
        label = data['label'] 
        # 确保图像按正确顺序排列 
        sorted_images = sorted(images, key=lambda x: [ 
            "_flair.nii"  in file, 
            "_t1.nii"  in file, 
            "_t2.nii"  in file, 
            "_t1ce.nii"  in file 
        ]) 
        dataset.append({'images':  sorted_images, 'label': label}) 
 
    return dataset 