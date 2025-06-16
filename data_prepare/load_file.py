import os 
import SimpleITK as SpITK 
import numpy as np 
from logger import get_logger 
 
logger = get_logger("data_prepare") 
 

def read_nii_files(folder_path): 
    """
    Read and process NIfTI files from the BraTS dataset.
    This function handles both image files (flair, t1, t2, t1ce) and their corresponding segmentation labels.
    
    Args:
        folder_path (str): Path to the directory containing BraTS dataset files
        
    Returns:
        dict: Dictionary containing processed images and labels
            - images: numpy array of shape [4, n, 155, 256, 256] (4 modalities, n samples, 155 slices, 256x256 resolution)
            - labels: numpy array of shape [3, n, 155, 256, 256] (3 segmentation masks, n samples, 155 slices, 256x256 resolution)
    """
    dataset = [] 
    # Dictionary to store label file paths, key is base filename, value is label file path
    label_files = {} 
 
    # First pass: Find all segmentation label files
    for root, _, files in os.walk(folder_path):  
        for file in files: 
            if file.endswith("_seg.nii") or file.endswith("_seg.nii.gz"):  
                # Extract base name (e.g., from 'BraTS20_Training_354_seg.nii' to 'BraTS20_Training_354')
                base_name = file.rsplit("_", 1)[0] 
                label_files[base_name] = os.path.join(root, file) 
 
    sample_data = {} 
    num_samples = 0
    
    # Second pass: Read image files and associate with labels
    for root, _, files in os.walk(folder_path): 
        for file in files: 
            # Check if file is one of the four MRI modalities (flair, t1, t2, t1ce)
            if file.endswith(("_flair.nii", "_flair.nii.gz", "_t1.nii", "_t1.nii.gz", 
                            "_t2.nii", "_t2.nii.gz", "_t1ce.nii", "_t1ce.nii.gz")):  
                base_name = file.rsplit("_", 1)[0] 
                # Find corresponding label file
                label_path = label_files.get(base_name)  
                if not label_path: 
                    logger.warning(f"No label found for file: {file}, skipping...") 
                    continue 
 
                try: 
                    num_samples += 1 
                    print(num_samples) 
                    # Construct full file path
                    nii_path = os.path.join(root, file) 
                    # Read NIfTI image
                    image_sitk = SpITK.ReadImage(nii_path) 
                    spacing = image_sitk.GetSpacing() 
                    spatial_shape = image_sitk.GetSize() 
 
                    # Set default values for metadata
                    series_number = 0 
                    study_desc = "N/A" 
                    series_desc = "N/A" 
 
                    # Read label file
                    label_sitk = SpITK.ReadImage(label_path) 
                    label = SpITK.GetArrayFromImage(label_sitk) 
 
                    # Create dictionary with image information
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
                    logger.warning(f"Error reading NIfTI file: {file}, error: {e}") 
 
    # Organize sample data
    all_images = [] 
    all_labels = []  
    for base_name, data in sample_data.items():  
        images = data['images'] 
        label = data['label'] 
        # Ensure images are in correct order (flair, t1, t2, t1ce)
        sorted_images = sorted(images, key=lambda x: [ 
            "_flair.nii" in file, 
            "_t1.nii" in file, 
            "_t2.nii" in file, 
            "_t1ce.nii" in file 
        ]) 
        # Stack images to desired shape
        stacked_images = np.stack(sorted_images, axis=0)  # [1, 155, 256, 256]
        all_images.append(stacked_images)  
        all_labels.append(label)  

    all_labels = np.stack(all_labels, axis=0) 

    # Process BraTS segmentation labels:
    # 0: Background
    # 1: Necrotic and non-enhancing tumor core
    # 2: Edema
    # 4: Enhancing tumor
    
    # Create three binary masks:
    # 1. Whole Tumor (WT): combines labels 1, 2, and 4
    mask_WT = all_labels.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1

    # 2. Tumor Core (TC): combines labels 1 and 4
    mask_TC = all_labels.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 4] = 1

    # 3. Enhancing Tumor (ET): only label 4
    mask_ET = all_labels.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 4] = 1

    # Stack the three masks
    all_labels = np.stack([mask_WT, mask_TC, mask_ET], axis=0)
    
    # Stack all sample images
    final_images = np.stack(all_images, axis=1)  # [4, n, 155, 256, 256] 
    logger.info("Datasets sorted successfully") 
    print(all_labels.shape)
    print(final_images.shape)
    return {'images': final_images, 'labels': all_labels} 
 
 
# def find_rtstruct_files(folder_path):  # find RTSTRUCT files 
#     rtstruct_files = [] 
#     for root, _, files in os.walk(folder_path):  
#         for file in files: 
#             if file.endswith(".dcm"):  
#                 try: 
#                     ds = pydicom.dcmread(os.path.join(root,  file), force=True) 
#                     if ds.Modality == "RTSTRUCT": 
#                         # check if is already in the list 
#                         file_path = os.path.join(root,  file) 
#                         if file_path not in rtstruct_files: 
#                             rtstruct_files.append(file_path)  
#                 except Exception as e: 
#                     logger.warning(f"Skipping  non-DICOM file: {file}, error: {e}") 
#     return rtstruct_files 
 
 
# def match_dicom_file(rtstruct_path): 
#     try: 
#         # read RTSTRUCT DICOM 
#         rtstruct_ds = pydicom.dcmread(rtstruct_path,  force=True) 
#         # get SeriesNumber and StudyInstanceUID 
#         series_number = rtstruct_ds.SeriesNumber 
#         study_uid = rtstruct_ds.StudyInstanceUID 
 
#         base_dir = os.path.dirname(rtstruct_path)  
 
#         matched_files = [] 
 
#         # Traverse 
#         for root, _, files in os.walk(base_dir):  
#             for file in files: 
#                 if file.endswith(".dcm"):  
#                     try: 
#                         # Read DICOM file and match SeriesNumber and StudyInstanceUID 
#                         ds = pydicom.dcmread(os.path.join(root,  file), force=True) 
#                         if ds.Modality != "RTSTRUCT" and ds.SeriesNumber == series_number and ds.StudyInstanceUID == study_uid: 
#                             matched_files.append(os.path.join(root,  file)) 
#                     except Exception as e: 
#                         logger.warning(f"Skipping  non - DICOM file: {file}, error: {e}") 
 
#         if not matched_files: 
#             logger.warning(f"No  matching DICOM files found for: {rtstruct_path}") 
#             return None 
#         return matched_files 
#     except Exception as e: 
#         # Error reading RTSTRUCT file 
#         logger.error(f"Error  reading RTSTRUCT file: {rtstruct_path}, error: {e}") 
#         return None 
 
 
# def process_rtstruct(rtstruct_path, dicom_files, output_dir, index): 
#     try: 
#         # read RTSTRUCT DICOM to get StudyInstanceUID, SeriesNumber and Patient's name 
#         rtstruct_ds = pydicom.dcmread(rtstruct_path,  force=True) 
#         series_number = rtstruct_ds.SeriesNumber 
#         patient_name = str(rtstruct_ds.PatientName) if 'PatientName' in rtstruct_ds else 'Unknown' 
 
#         # Generate custom filenames 
#         label_filename = f"mask_{patient_name}_{series_number}.nii.gz"  
#         custom_label_path = os.path.join(output_dir,  label_filename) 
 
#         # Check if the file already exists 
#         if os.path.exists(custom_label_path):  
#             logger.warning(f"File  {custom_label_path} already exists, skipping conversion.") 
#         else: 
#             # Convert RTStruct to NIfTI 
#             dicom_dir = os.path.dirname(dicom_files[0])  
#             dcmrtstruct2nii(rtstruct_path, dicom_dir, output_dir, structures=['GTV']) 
 
#             # Rename label file 
#             default_label_path = os.path.join(output_dir,  "mask_GTV.nii.gz")  
#             if os.path.exists(default_label_path):  
#                 os.rename(default_label_path,  custom_label_path) 
#             else: 
#                 logger.warning(f"Default  label file {default_label_path} not found.") 
#                 return [] 
 
#         # Sort DICOM files by InstanceNumber 
#         dicom_files_with_instance_number = [] 
#         for dicom_file in dicom_files: 
#             try: 
#                 ds = pydicom.dcmread(dicom_file,  force=True) 
#                 instance_number = ds.InstanceNumber if 'InstanceNumber' in ds else 0 
#                 dicom_files_with_instance_number.append((instance_number,  dicom_file)) 
#             except Exception as e: 
#                 logger.warning(f"Error  reading DICOM file for sorting: {dicom_file}, error: {e}") 
#         dicom_files_with_instance_number.sort()  
#         sorted_dicom_files = [file for _, file in dicom_files_with_instance_number] 
 
#         dataset = [] 
#         # read label file 
#         if os.path.exists(custom_label_path):  
#             label_sitk = SpITK.ReadImage(custom_label_path)
#             label = SpITK.GetArrayFromImage(label_sitk)
#         else: 
#             logger.warning(f"Custom  label file {custom_label_path} not found.") 
#             label = None 
 
#         for i, dicom_file in enumerate(sorted_dicom_files): 
#             try: 
#                 # read DICOM image 
#                 ds = pydicom.dcmread(dicom_file,  force=True) 
#                 image_sitk = SpITK.ReadImage(dicom_file)
#                 spacing = image_sitk.GetSpacing() 
#                 spatial_shape = image_sitk.GetSize() 
 
#                 series_number = ds.SeriesNumber 
#                 study_desc = ds.StudyDescription if 'StudyDescription' in ds else '' 
#                 series_desc = ds.SeriesDescription if 'SeriesDescription' in ds else '' 
 
#                 # extract the label of the corresponding slice 
#                 if label is not None: 
#                     slice_label = label[i] if i < len(label) else None 
#                 else: 
#                     slice_label = None 
 
#                 # data dictionary 
#                 data_dict = { 
#                     'SeriesNumber': series_number, 
#                     'StudyDescription': study_desc, 
#                     'SeriesDescription': series_desc, 
#                     'spacing': spacing, 
#                     'spatial_shape': spatial_shape, 
#                     'space': image_sitk.GetDirection(),  # Spatial direction matrix 
#                     'label': slice_label, 
#                     'image': SpITK.GetArrayFromImage(image_sitk)
#                 } 
#                 dataset.append(data_dict)  
#             except Exception as e: 
#                 logger.warning(f"Error  processing DICOM file: {dicom_file}, error: {e}") 
 
#         return dataset 
#     except Exception as e: 
#         logger.error(f"Error processing RTSTRUCT file: {rtstruct_path}, error: {e}")
#         return [] 
 
 
# def generate_dataset(folder_path, output_dir): 
#     # Create the output directory 
#     if not os.path.exists(output_dir):  
#         os.makedirs(output_dir)  
 
#     # Filter RTStruct files 
#     rtstruct_files = find_rtstruct_files(folder_path) 
 
#     # Initialize the dataset list 
#     dataset = [] 
 
#     # Process each RTStruct 
#     for index, rtstruct_path in enumerate(rtstruct_files, start=1): 
#         dicom_files = match_dicom_file(rtstruct_path) 
#         if not dicom_files: 
#             logger.warning(f"No  matching DICOM files found: {rtstruct_path}") 
#             continue 
 
#         # Process and get the dictionary data 
#         sample_data = process_rtstruct(rtstruct_path, dicom_files, output_dir, index) 
#         if sample_data: 
#             dataset.extend(sample_data)  
 
#     return dataset 
 

 