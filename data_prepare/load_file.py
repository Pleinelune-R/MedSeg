import os
import SimpleITK as SpITK
import numpy as np
from logger import get_logger

logger = get_logger("data_prepare")

def collect_sample_paths(folder_path, max_samples=None):
    """
    返回 [{'flair': ..., 't1': ..., 't2': ..., 't1ce': ..., 'label': ...}, ...]
    """
    samples = []
    for case in sorted(os.listdir(folder_path)):
        case_dir = os.path.join(folder_path, case)
        if not os.path.isdir(case_dir):
            continue
        flair = os.path.join(case_dir, "flair.nii.gz")
        t1 = os.path.join(case_dir, "t1.nii.gz")
        t2 = os.path.join(case_dir, "t2.nii.gz")
        t1ce = os.path.join(case_dir, "t1ce.nii.gz")
        # 支持label.nii.gz或label.nii
        label = os.path.join(case_dir, "label.nii.gz")
        if not os.path.exists(label):
            label = os.path.join(case_dir, "label.nii")
        if all(os.path.exists(p) for p in [flair, t1, t2, t1ce, label]):
            samples.append({
                'flair': flair,
                't1': t1,
                't2': t2,
                't1ce': t1ce,
                'label': label
            })
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    logger.info(f"collect finish, total samples: {len(samples)}")
    return samples


def read_nii_files(folder_path, max_samples=None):
    """
    Read and process NIfTI files from the BraTS dataset.
    This function handles both image files (flair, t1, t2, t1ce) and their corresponding segmentation labels.
    
    Args:
        folder_path (str): Path to the directory containing BraTS dataset files
        max_samples (int, optional): Maximum number of samples to process. If None, process all available samples.
        
    Returns:
        dict: Dictionary containing processed images and labels
            - images: numpy array of shape [4, n, 155, 240, 240] (4 modalities, n samples, 155 slices, 240x240 resolution)
            - labels: numpy array of shape [3, n, 155, 240, 240] (3 segmentation masks, n samples, 155 slices, 240x240 resolution)
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
        if max_samples is not None and num_samples >= max_samples:  # 如果设置了max_samples且达到限制则停止
            break

        for file in files:
            if max_samples is not None and num_samples >= max_samples:  # 如果设置了max_samples且达到限制则停止
                break

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
                    # Initialize sample data if not exists
                    if base_name not in sample_data:
                        sample_data[base_name] = {
                            'images': {'flair': None, 't1': None, 't2': None, 't1ce': None},
                            'label': None
                        }
                        # Read label file only once per sample
                        label_sitk = SpITK.ReadImage(label_path)
                        sample_data[base_name]['label'] = SpITK.GetArrayFromImage(label_sitk)

                    # Read image file
                    nii_path = os.path.join(root, file)
                    image_sitk = SpITK.ReadImage(nii_path)
                    image = SpITK.GetArrayFromImage(image_sitk)

                    # Store image based on modality
                    if '_flair' in file:
                        sample_data[base_name]['images']['flair'] = image
                    elif '_t1.' in file and not '_t1ce' in file:
                        sample_data[base_name]['images']['t1'] = image
                    elif '_t2' in file:
                        sample_data[base_name]['images']['t2'] = image
                    elif '_t1ce' in file:
                        sample_data[base_name]['images']['t1ce'] = image

                    # Check if we have all modalities for this sample
                    if all(v is not None for v in sample_data[base_name]['images'].values()):
                        num_samples += 1
                        if num_samples % 10 == 0:  # 每处理10个样本打印一次进度
                            if max_samples is not None:
                                print(f"Processing sample {num_samples}/{max_samples}")
                            else:
                                print(f"Processing sample {num_samples}")

                except Exception as e:
                    logger.warning(f"Error reading NIfTI file: {file}, error: {e}")

                    # Organize sample data
    all_images = []
    all_labels = []
    for base_name, data in sample_data.items():
        # Skip incomplete samples
        if not all(v is not None for v in data['images'].values()):
            continue

        # Stack images in correct order (flair, t1, t2, t1ce)
        images = [
            data['images']['flair'],
            data['images']['t1'],
            data['images']['t2'],
            data['images']['t1ce']
        ]
        stacked_images = np.stack(images, axis=0)  # [4, 155, 240, 240]
        all_images.append(stacked_images)
        all_labels.append(data['label'])

    all_labels = np.stack(all_labels, axis=0)

    # Process BraTS segmentation labels:
    # 0: Background
    # 1: Necrotic and non-enhancing tumor core
    # 2: Edema
    # 4: Enhancing tumor

    # Convert to one-hot encoding (4 channels)
    num_classes = 4
    one_hot_labels = np.zeros((num_classes, *all_labels.shape))

    # Set values for each class
    one_hot_labels[0] = (all_labels == 0).astype(np.float32)  # Background
    one_hot_labels[1] = (all_labels == 1).astype(np.float32)  # Necrotic and non-enhancing tumor core
    one_hot_labels[2] = (all_labels == 2).astype(np.float32)  # Edema
    one_hot_labels[3] = (all_labels == 4).astype(np.float32)  # Enhancing tumor

    # Stack all sample images
    final_images = np.stack(all_images, axis=1)  # [4, n, 155, 240, 240] 
    logger.info(f"Datasets sorted successfully. Total samples: {num_samples}")
    print(f"Labels shape: {one_hot_labels.shape}")
    print(f"Images shape: {final_images.shape}")
    return {'images': final_images, 'labels': one_hot_labels}

