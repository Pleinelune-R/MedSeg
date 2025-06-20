import os
import h5py
import numpy as np
from torchvision import transforms
from data_prepare.data_module import MRDataModule

raw_data_dir = "data/lsj_MICCAI_BraTS2020_TrainingData/"
save_path = "preprocessed_dataset.h5"
resize = transforms.Resize((256, 256))

samples = []
for case in sorted(os.listdir(raw_data_dir)):
    case_dir = os.path.join(raw_data_dir, case)
    if not os.path.isdir(case_dir):
        continue
    flair = os.path.join(case_dir, f"{case}_flair.nii")
    t1 = os.path.join(case_dir, f"{case}_t1.nii")
    t1ce = os.path.join(case_dir, f"{case}_t1ce.nii")
    t2 = os.path.join(case_dir, f"{case}_t2.nii")
    label = os.path.join(case_dir, f"{case}_seg.nii")
    if all(os.path.exists(p) for p in [flair, t1, t1ce, t2, label]):
        samples.append({'flair': flair, 't1': t1, 't2': t2, 't1ce': t1ce, 'label': label})

print(f"Found {len(samples)} samples")
if len(samples) == 0:
    raise RuntimeError("No samples found! Please check your data directory and file names.")

first_image, first_label = MRDataModule.process_mr_sample(samples[0], resize)
# image_shape = first_image.shape
# label_shape = first_label.shape
image_shape = (4, 128, 256, 256)
label_shape = (4, 128, 256, 256)

with h5py.File(save_path, "w") as f:
    images_ds = f.create_dataset("images", shape=(len(samples), *image_shape), dtype=np.float32)
    labels_ds = f.create_dataset("labels", shape=(len(samples), *label_shape), dtype=np.float32)
    for i, paths in enumerate(samples):
        image, label = MRDataModule.process_mr_sample(paths, resize)
        images_ds[i] = image.numpy()
        labels_ds[i] = label.numpy()
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(samples)}")
print("All samples preprocessed and saved to HDF5.")