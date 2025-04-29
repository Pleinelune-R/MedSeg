import os 
import pydicom 
from dcmrtstruct2nii import dcmrtstruct2nii 
import SimpleITK as sitk 
 
from logger import MyLogger 
 
logger = MyLogger("data_prepare") 
 
 
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
        # read RTSTRUCT DICOM to get StudyInstanceUID 
        rtstruct_ds = pydicom.dcmread(rtstruct_path,  force=True) 
        study_uid = rtstruct_ds.StudyInstanceUID 
 
        # Generate custom filenames 
        label_filename = f"mask_{index}_{study_uid}.nii.gz"  
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
            label_sitk = sitk.ReadImage(custom_label_path) 
            label = sitk.GetArrayFromImage(label_sitk) 
        else: 
            logger.warning(f"Custom  label file {custom_label_path} not found.") 
            label = None 
 
        for i, dicom_file in enumerate(sorted_dicom_files): 
            try: 
                # read DICOM image 
                ds = pydicom.dcmread(dicom_file,  force=True) 
                image_sitk = sitk.ReadImage(dicom_file) 
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
                    'image': sitk.GetArrayFromImage(image_sitk) 
                } 
                dataset.append(data_dict)  
            except Exception as e: 
                logger.warning(f"Error  processing DICOM file: {dicom_file}, error: {e}") 
 
        return dataset 
    except Exception as e: 
        logger.error(f"Error  processing RTSTRUCT file: {rtstruct_path}, error: {e}") 
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
 