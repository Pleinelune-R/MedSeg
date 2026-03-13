"""
3D MAE Encoder + CNNDecoder Supervised Segmentation Training
"""
import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import random

from logger import get_logger
from model.config import ModelConfig
from model.train_model.model import MAECNNSegModel3D
from datasets.mae_dataset import MedicalSegmentationDataset3D

logger = get_logger("train_supervised")


def compute_metrics_3d(pred, gt):
    """
    Compute Dice, IoU, and HD95 for 3D volumes.
    Assumes binary segmentation (foreground > 0).
    """
    pred_binary = (pred > 0).astype(bool)
    gt_binary = (gt > 0).astype(bool)
    
    # Edge case: both empty
    if not np.any(gt_binary) and not np.any(pred_binary):
        return {'dice': 1.0, 'hd95': 0.0, 'iou': 1.0}
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    sum_pred = pred_binary.sum()
    sum_gt = gt_binary.sum()
    
    dice = 2.0 * intersection / (sum_pred + sum_gt) if (sum_pred + sum_gt) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    
    # HD95
    if not np.any(pred_binary) or not np.any(gt_binary):
        hd95 = 100.0 # Max penalty (arbitrary large value)
    else:
        try:
            # Invert masks for distance transform (dt calculates distance to nearest 0)
            dt_pred = distance_transform_edt(~pred_binary)
            dt_gt = distance_transform_edt(~gt_binary)
            
            dist_pred_to_gt = dt_gt[pred_binary]
            dist_gt_to_pred = dt_pred[gt_binary]
            
            all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
            hd95 = np.percentile(all_dists, 95)
        except Exception as e:
            print(f"HD95 calculation failed: {e}")
            hd95 = -1.0
        
    return {'dice': dice, 'hd95': hd95, 'iou': iou}


def create_segmentation_grid(pl_module, val_loader, device, num_samples=4, save_dir="."):
    """
    Generates and saves a grid of segmentation predictions and ground truths,
    prioritizing samples with non-empty masks.
    """
    pl_module.eval()
    
    # Collect samples, prioritizing those with labels
    valid_samples = []
    empty_samples = []
    
    with torch.no_grad():
        # Ensure we iterate a real DataLoader (not a list)
        loader = val_loader[0] if isinstance(val_loader, (list, tuple)) else val_loader
        for batch in loader:
            volumes, masks = batch
            volumes, masks = volumes.to(device), masks.to(device)
            
            logits = pl_module(volumes)
            if logits.shape[1] == 1:
                preds = (torch.sigmoid(logits) > 0.5).float().squeeze(1)
            else:
                preds = torch.argmax(logits, dim=1)
            masks_indices = torch.argmax(masks, dim=1)
            
            for i in range(volumes.shape[0]):
                vol_i = volumes[i, 0]
                mask_i = masks_indices[i]
                pred_i = preds[i]
                
                # Compute metrics for the whole volume
                metrics = compute_metrics_3d(pred_i.cpu().numpy(), mask_i.cpu().numpy())
                
                # Pick a slice that contains foreground (non-empty)
                # Fallback to center slice if none contain foreground
                D = mask_i.shape[0]
                # Foreground voxel counts per slice
                fg_per_slice = (mask_i > 0).sum(dim=(1, 2))
                if torch.any(fg_per_slice > 0):
                    # Choose slice with max foreground area
                    best_slice = torch.argmax(fg_per_slice).item()
                else:
                    best_slice = D // 2

                sample_data = {
                    'volume': vol_i.cpu().numpy(),
                    'mask': mask_i.cpu().numpy(),
                    'pred': pred_i.cpu().numpy(),
                    'slice_idx': best_slice,
                    'metrics': metrics,
                    'has_fg': bool(torch.any(fg_per_slice > 0).item()),
                }

                # Check if the volume contains any non-zero labels at all
                if sample_data['has_fg']:
                    if len(valid_samples) < num_samples:
                        valid_samples.append(sample_data)
                else:
                    if len(empty_samples) < num_samples:
                        empty_samples.append(sample_data)
            
            # Stop if we have enough valid samples
            if len(valid_samples) >= num_samples:
                break

    # Combine samples: prioritize valid ones, fill with empty ones if needed
    samples = valid_samples + empty_samples
    samples = samples[:num_samples]

    if not samples:
        logger.warning("No samples collected for visualization.")
        pl_module.train()
        return

    # Create a grid plot
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)))
    if len(samples) == 1: # Handle case with single sample
        axes = [axes]
    fig.suptitle('Segmentation Results (Prioritizing Labeled Samples)', fontsize=16)
    
    for i, sample in enumerate(samples):
        slice_idx = int(sample.get('slice_idx', sample['volume'].shape[0] // 2))
        metrics = sample['metrics']
        metrics_lines = [
            f"Dice: {metrics['dice']:.3f}",
            f"IoU:  {metrics['iou']:.3f}",
            f"HD95: {metrics['hd95']:.1f}"
        ]
        metrics_str = "\n".join(metrics_lines)
        
        # Plot Input Volume
        ax_row = axes[i] if len(samples) > 1 else axes
        ax = ax_row[0]
        ax.imshow(sample['volume'][slice_idx], cmap="gray")
        ax.set_title(f"Sample {i+1}: Input")
        ax.axis("off")
        
        # Plot Ground Truth
        ax = ax_row[1]
        ax.imshow(sample['mask'][slice_idx], cmap="jet")
        ax.set_title(f"Sample {i+1}: Ground Truth")
        ax.axis("off")
        
        # Plot Prediction
        ax = ax_row[2]
        ax.imshow(sample['pred'][slice_idx], cmap="jet")
        ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes, 
                color='white', fontsize=9, verticalalignment='top', 
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        ax.set_title(f"Sample {i+1}: Pred")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, "segmentation_grid_prioritized.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    pl_module.train()


class SegmentationVisualizationCallback(pl.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0 or trainer.current_epoch == 0:
            # Handle possible list of val dataloaders
            val_loader = trainer.val_dataloaders
            if isinstance(val_loader, (list, tuple)):
                val_loader = val_loader[0]
            device = pl_module.device
            
            epoch_save_dir = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch + 1}")
            os.makedirs(epoch_save_dir, exist_ok=True)
            
            try:
                create_segmentation_grid(pl_module, val_loader, device, num_samples=4, save_dir=epoch_save_dir)
            except Exception as e:
                logger.error(f"Error generating segmentation visualization: {e}")


def train_cnnseg_3d(config, save_dir="./checkpoints_cnnseg_3d", pretrained_path=None):
    """
    3D MAE Encoder + CNNDecoder 分割训练主流程
    """
    logger.info("=== Starting 3D MAE-CNN Segmentation Training ===")
    
    volume_size = getattr(config, 'volume_size', (80,160,160))
    # Manually split files to enable augmentation only for training
    h5_dir = Path(config.dataset_path)
    
    # Group files by patient ID
    patient_files = {}
    for f_path in h5_dir.glob("*.h5"):
        patient_id = f_path.stem.split('_mod')[0]
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(f_path)
    
    all_patient_ids = sorted(list(patient_files.keys()))
    random.shuffle(all_patient_ids)
    
    total_patients = len(all_patient_ids)
    train_size = int(total_patients * config.train_val_split)
    val_size = int(total_patients * 0.1)
    test_size = total_patients - train_size - val_size
    
    train_ids = all_patient_ids[:train_size]
    val_ids = all_patient_ids[train_size:train_size+val_size]
    test_ids = all_patient_ids[train_size+val_size:]
    
    logger.info(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    train_dataset = MedicalSegmentationDataset3D(
        h5_dir=config.dataset_path,
        volume_size=volume_size,
        num_classes=config.num_classes,
        config=config,
        patient_ids=train_ids,
        augment=True # Enable augmentation for training
    )
    
    val_dataset = MedicalSegmentationDataset3D(
        h5_dir=config.dataset_path,
        volume_size=volume_size,
        num_classes=config.num_classes,
        config=config,
        patient_ids=val_ids,
        augment=False
    )
    
    test_dataset = MedicalSegmentationDataset3D(
        h5_dir=config.dataset_path,
        volume_size=volume_size,
        num_classes=config.num_classes,
        config=config,
        patient_ids=test_ids,
        augment=False
    )
    
    batch_size = config.batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    model = MAECNNSegModel3D(config=config)
    if pretrained_path:
        model.load_pretrained_encoder(pretrained_path)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=config.devices_numbers,
        precision="32",
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                filename='best_cnnseg_model_3d',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=getattr(config, 'early_stopping_patience', 20),
                mode='min'
            ),
            LearningRateMonitor(logging_interval='epoch'),
            SegmentationVisualizationCallback(save_dir=save_dir)
        ],
        logger=TensorBoardLogger(save_dir, name='cnnseg_3d'),
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)
    
    logger.info("3D MAE-CNN segmentation training complete!")
    return trainer
