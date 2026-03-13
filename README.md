# MedSeg - Medical Image Segmentation with MAE

A comprehensive framework for 3D medical image pretraining using Masked Autoencoders (MAE) and supervised segmentation.

## Project Structure

```plaintext
MedSeg/
 model/                          # Core model components
   ├── config.py                   # Configuration management
   ├── pretrain_model/             # MAE pretraining models
   │   ├──  mae_encoder.py          # 3D MAE encoder (ViT-based)   
   │   ├── mae_decoder.py          # 3D MAE decoder (reconstruction)
   │   ├── mae_loss.py             # MAE loss functions
   │   └── ...
   └── train_model/                # Supervised segmentation models
       ├── model.py                # ✨ NEW: Model definitions
       └── __init__.py             # Model exports

 train/                          # Training modules
   ├── pretrain.py                 # 🔄 MAE pretraining script
   ├── train.py                    # ✨ NEW: Supervised training script
   │   ├── compute_metrics_3d()    # Evaluation metrics
   │   ├── create_segmentation_grid() # Visualization
   │   ├── SegmentationVisualizationCallback # Training callback
   │   └── train_cnnseg_3d()       # Main training function
   └── __init__.py                 # Training exports

 datasets/
   ├── mae_dataset.py              # Data loading utilities
   └── __init__.py

 run_pretrain.py                 # Entry: MAE pretraining
 run_train.py                    # Entry: Supervised training
 logger.py                        # Logging utilities
 requirements.txt
 README.md

```

## Installation

```bash
git clone <repo-url>
cd MedSeg
pip install -r requirements.txt
```

## Quick Start

### 1. MAE Pretraining

```bash
python run_pretrain.py \
    --data_path ./data/preprocessed_pretrain \
    --save_dir ./checkpoints_pretrain \
    --max_epochs 200 \
    --batch_size 4
```

**Imports:**
```python
from train import pretrain_mae
from train.pretrain import pretrain_mae
```

### 2. Supervised Segmentation Training

```bash
python run_train.py \
    --data_path ./data/preprocessed_train \
    --save_dir ./checkpoints_train \
    --pretrained_path ./checkpoints_pretrain/best_model.ckpt \
    --max_epochs 100 \
    --batch_size 8
```

**Imports:**
```python
from train import train_cnnseg_3d
from train.train import train_cnnseg_3d
```

## Core Components

### Model Architecture

#### Encoder: 3D MAE (Vision Transformer)
- **Location**: `model/pretrain_model/mae_encoder.py`
- **Input**: 3D medical volumes (D×H×W)
- **Output**: Patch embeddings with learnable positional encodings
- **Features**: Multi-head self-attention, patch masking support

#### Decoder: Feature ResUNet
- **Location**: `model/train_model/model.py` → `FeatureResUNet3D`
- **Input**: MAE features (embed_dim channels)
- **Output**: Segmentation logits
- **Architecture**: Encoder-decoder with skip connections

#### Loss Functions
- **Dice Loss**: `DiceLoss3D` - suitable for medical image segmentation
- **Cross-Entropy Loss**: Combined with Dice for multi-class support

### Training Pipeline

#### Pretraining (`train/pretrain.py`)
```python
from train import pretrain_mae
from model.config import ModelConfig

config = ModelConfig(...)
trainer = pretrain_mae(
    config=config,
    mask_ratio=0.75,
    save_dir="./checkpoints_pretrain"
)
```

#### Supervised Training (`train/train.py`)
```python
from train import train_cnnseg_3d
from model.config import ModelConfig

config = ModelConfig(...)
trainer = train_cnnseg_3d(
    config=config,
    save_dir="./checkpoints_train",
    pretrained_path="./checkpoints_pretrain/best_model.ckpt"
)
```

## Data Format

Data should be organized as HDF5 files with the following structure:

```
data/
 preprocessed_pretrain/
   ├── patient_001_mod0.h5        # 4 modalities per patient
   ├── patient_001_mod1.h5
   ├── patient_001_mod2.h5
   ├── patient_001_mod3.h5
 ...   
 preprocessed_train/
    ├── patient_001_mod0.h5
    └── ...
```

Each H5 file contains:
- `volume`: 3D medical image (D×H×W)
- `mask`: Corresponding segmentation mask

## Key Features

| Feature | Details |
|---------|---------|
| **3D Support** | Native support for volumetric medical imaging (CT, MRI) |
| **Separation of Concerns** | Models in `train_model/`, training logic in `train/` |
| **Pre-trained Encoder** | Transfer learning support via pretrained MAE weights |
| **Metrics Computation** | Dice, IoU, HD95 for comprehensive evaluation |
| **Visualization** | Automatic segmentation grid generation during training |
| **PyTorch Lightning** | Modern training framework with callbacks and loggers |

## Architecture Refactoring

This project has been refactored for better code organization:

| Change | Before | After |
|--------|--------|-------|
| **Model Definition** | Mixed in train_model.py | `model/train_model/model.py` |
| **Training Logic** | Mixed in train_model.py | `train/train.py` |
| **Supervised Training** | train.py (duplicate) | Removed (was dead code) |
| **Imports** | Complex paths | Unified via `__init__.py` |

## Configuration

Edit `model/config.py` to adjust:
- `volume_size`: Input volume dimensions (default: (80, 160, 160))
- `patch_size`: Patch size for tokenization (default: 16)
- `embed_dim`: Embedding dimension (default: 768)
- `depth`: Number of transformer blocks (default: 12)
- `batch_size`: Training batch size
- `max_epochs`: Maximum training epochs
- `learning_rate`: Optimizer learning rate

## Evaluation Metrics

- **Dice**: Measures overlap between prediction and ground truth
- **IoU**: Intersection over Union (Jaccard index)
- **HD95**: 95th percentile of Hausdorff distance (boundary-based metric)

## References

- MAE: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- 3D Medical Imaging: Volumetric analysis for CT/MRI segmentation
- Dice Loss: [Generalised Dice overlap as a deep learning loss function](https://arxiv.org/abs/1707.00478)

## Notes

- Ensure HDF5 files contain properly preprocessed volumes
- GPU recommended for training (CUDA support required)
- Tensorboard logs saved in checkpoint directories
- Visualization grids generated every 5 epochs (customizable)
