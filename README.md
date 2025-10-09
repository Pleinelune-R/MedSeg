# MEDSEG

## Overview
MedSeg is a project for medical image pretraining (MAE) and segmentation on MRI/CT.

## Project Structure
```plaintext
MedSeg/
├── main.py                 # Entry: choose pretrain or supervised training
├── model/                  # Core model components (encoder/decoder/loss/embed)
│   ├── mae_encoder.py
│   ├── mae_decoder.py
│   ├── mae_loss.py
│   └── pos_embed.py
├── data/                   # Datasets
│   └── mae_dataset.py
├── train/                  # Training scripts
│   ├── mae_pretrain.py
│   └── mae_supervised.py
├── legacy/                 # Legacy utilities (optional/unused in main flow)
└── requirements.txt
```

## Installation
```bash
git clone <repo-url>
cd MedSeg
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

## Quick Start
- Prepare dataset directory with HDF5 files under your path.
- Launch from the entry script and select mode:
```bash
python main.py
# Choose: 1) MAE unsupervised pretraining, or 2) MAE supervised segmentation
```

Main entry uses:
- Pretraining: `from train.mae_pretrain import pretrain_mae`
- Supervised: `from train.mae_supervised import train_mae_supervised`

## Notes
- Ensure your data folder contains HDF5 files; each file typically holds up to 4 modalities and multiple slices.
