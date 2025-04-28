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
                    ds = pydicom.dcmread(os.path.join(root,   file), force=True) 
                    if ds.Modality == "RTSTRUCT": 
                        # check if is already in the list 
                        file_path = os.path.join(root,   file) 
                        if file_path not in rtstruct_files: 
                            rtstruct_files.append(file_path)   
                except Exception as e: 
                    logger.warning(f"Skipping   non - DICOM file: {file}, error: {e}") 
    return rtstruct_files 
 
def match_dicom_dir(rtstruct_path): 
    rtstruct_ds = pydicom.dcmread(rtstruct_path,   force=True) 
    study_uid = rtstruct_ds.StudyInstanceUID 
    base_dir = os.path.dirname(rtstruct_path)   
    for root, _, files in os.walk(base_dir):   
        for file in files: 
            if file.endswith(".dcm"):   
                try: 
                    ds = pydicom.dcmread(os.path.join(root,   file), force=True) 
                    if ds.StudyInstanceUID == study_uid and ds.Modality != "RTSTRUCT": 
                        logger.info(f"Matched   DICOM directory: {root} for RTSTRUCT: {rtstruct_path}") 
                        return root 
                except Exception as e: 
                    logger.warning(f"Skipping   non - DICOM file: {file}, error: {e}") 
    logger.warning(f"No   matching DICOM directory found for: {rtstruct_path}") 
    return None 
 
 
def process_rtstruct(rtstruct_path, dicom_dir, output_dir, index): 
    # Generate custom filenames 
    label_filename = f"mask_GTV_{index}.nii.gz"   
    image_filename = f"image_{index}.nii.gz"   
 
    # Convert RTStruct to NIfTI 
    dcmrtstruct2nii(rtstruct_path, dicom_dir, output_dir, structures=['GTV']) 
 
    # Rename files 
    default_label_path = os.path.join(output_dir,   "mask_GTV.nii.gz")   
    custom_label_path = os.path.join(output_dir,   label_filename) 
    if os.path.exists(default_label_path):   
        os.rename(default_label_path,   custom_label_path) 
    else: 
        logger.warning(f"Default   label file {default_label_path} not found.") 
 
    default_image_path = os.path.join(output_dir,   "image.nii.gz")   
    custom_image_path = os.path.join(output_dir,   image_filename) 
    if os.path.exists(default_image_path):   
        os.rename(default_image_path,   custom_image_path) 
    else: 
        logger.warning(f"Default   image file {default_image_path} not found.") 
 
    # Read the metadata of the image and label 
    if os.path.exists(custom_image_path):   
        image_sitk = sitk.ReadImage(custom_image_path) 
        spacing = image_sitk.GetSpacing() 
        spatial_shape = image_sitk.GetSize() 
    else: 
        logger.warning(f"Custom   image file {custom_image_path} not found.") 
        return None 
 
    # Read the first DICOM file to extract metadata 
    sample_dcm = pydicom.dcmread(os.path.join(dicom_dir,   os.listdir(dicom_dir)[0]),   force=True) 
    series_number = sample_dcm.SeriesNumber 
    study_desc = sample_dcm.StudyDescription if 'StudyDescription' in sample_dcm else '' 
    series_desc = sample_dcm.SeriesDescription if 'SeriesDescription' in sample_dcm else '' 
 
    # Build the dictionary 
    sample_dict = { 
        'SeriesNumber': series_number, 
        'StudyDescription': study_desc, 
        'SeriesDescription': series_desc, 
        'spacing': spacing, 
        'spatial_shape': spatial_shape, 
        'space': image_sitk.GetDirection(),  # Spatial direction matrix 
        'label': sitk.GetArrayFromImage(sitk.ReadImage(custom_label_path)), 
        'image': sitk.GetArrayFromImage(image_sitk) 
    } 
    return sample_dict 
 
 
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
        dicom_dir = match_dicom_dir(rtstruct_path) 
        if not dicom_dir: 
            logger.warning(f"No   matching DICOM directory found: {rtstruct_path}") 
            continue 
 
        # Process and get the dictionary data 
        sample_data = process_rtstruct(rtstruct_path, dicom_dir, output_dir, index) 
        if sample_data: 
            dataset.append(sample_data)   
 
    return dataset 