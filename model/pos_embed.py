"""
3D MAE Encoder + Residual Decoder (ResUNet style) Segmentation Training
"""
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from logger import get_logger
from model.config import ModelConfig
from model.mae_encoder import MAEEncoder3D
from datasets.mae_dataset import MedicalSegmentationDataset3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.ndimage import distance_transform_edt

# Optimize for Tensor Cores
torch.set_float32_matmul_precision('medium')

logger = get_logger("mae_resunet")

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class FeatureResUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, volume_size=(80, 160, 160), grid_size=None, num_patches=None, dropout_rate=0.0):
        super().__init__()
        self.volume_size = volume_size
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.dropout = nn.Dropout3d(p=dropout_rate)
        
        # Encoder Path
        # Input is 768 channels (MAE features)
        self.enc1 = ResBlock3D(in_channels, 256)
        self.down1 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.enc2 = ResBlock3D(512, 512)
        self.down2 = nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1)
        
        # Bridge
        self.bridge = ResBlock3D(1024, 1024)
        
        # Decoder Path
        self.up2 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(512 + 512, 512) # Concat with enc2
        
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(256 + 256, 256) # Concat with enc1
        
        self.final_conv = nn.Conv3d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, N, E) from MAE Encoder
        
        # Reshape to 3D feature map
        if self.num_patches is not None and x.shape[1] == self.num_patches + 1:
            x = x[:, 1:, :]
        if self.grid_size is not None:
            x = x.transpose(1, 2)
            d, h, w = self.grid_size
            x = x.reshape(x.shape[0], x.shape[1], d, h, w)
            
        # Encoder
        e1 = self.enc1(x) # (B, 256, D, H, W)
        d1 = self.down1(e1) # (B, 512, D/2, H/2, W/2)
        d1 = self.dropout(d1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2) # (B, 1024, D/4, H/4, W/4)
        d2 = self.dropout(d2)
        
        # Bridge
        b = self.bridge(d2)
        b = self.dropout(b)
        
        # Decoder
        u2 = self.up2(b)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = nn.functional.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        c2 = torch.cat([u2, e2], dim=1)
        d2_out = self.dec2(c2)
        
        u1 = self.up1(d2_out)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = nn.functional.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        c1 = torch.cat([u1, e1], dim=1)
        d1_out = self.dec1(c1)
        
        out = self.final_conv(d1_out)
        
        # Final Upsample to original volume size
        if out.shape[2:] != self.volume_size:
            out = nn.functional.interpolate(out, size=self.volume_size, mode='trilinear', align_corners=False)
            
        return out

class DiceLoss3D(nn.Module):
    def __init__(self, smooth=1.0, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
    def forward(self, pred, target):
        # Check if binary segmentation (1 channel output)
        if pred.shape[1] == 1:
            pred = torch.sigmoid(pred)
            # target is one-hot encoded: (B, 2, D, H, W) -> take channel 1 (foreground)
            if target.shape[1] == 2:
                target = target[:, 1:2]
            
            pred = pred.float()
            target = target.float()
            intersection = (pred * target).sum(dim=(2,3,4))
            union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
            dice = (2*intersection + self.smooth)/(union + self.smooth)
            return 1.0 - dice.mean()
        else:
            # Multi-class case
            pred = torch.softmax(pred, dim=1)
            if self.ignore_background:
                pred = pred[:, 1:]
                target = target[:, 1:]
            pred = pred.float()
            target = target.float()
            intersection = (pred * target).sum(dim=(2,3,4))
            union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
            dice = (2*intersection + self.smooth)/(union + self.smooth)
            return 1.0 - dice.mean()

class MAECNNSegModel3D(pl.LightningModule):
    def __init__(self, config: ModelConfig, num_classes=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.volume_size = getattr(config, 'volume_size', (80,160,160))
        self.num_classes = num_classes if num_classes is not None else config.num_classes
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.max_epochs = config.max_epochs
        # Encoder
        self.encoder = MAEEncoder3D(
            volume_size=self.volume_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio
        )
        
        # Feature ResUNet Decoder
        # Takes MAE features (embed_dim) as input
        self.decoder = FeatureResUNet3D(
            in_channels=config.embed_dim,
            out_channels=self.num_classes,
            volume_size=self.volume_size,
            grid_size=self.encoder.patch_embed.grid_size,
            num_patches=self.encoder.patch_embed.num_patches,
            dropout_rate=getattr(config, 'dropout', 0.0)
        )
        # Initialize Dice Loss once
        self.dice_loss_fn = DiceLoss3D(ignore_background=True)

    def load_pretrained_encoder(self, ckpt_path):
        logger.info(f"Loading pretrained MAE encoder weights from {ckpt_path}")
        # Fix for PyTorch >=2.6: weights_only=False
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        enc_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                enc_dict[k[len('encoder.'):]] = v
        
        # Handle patch_embed.proj.weight size mismatch for in_chans
        if 'patch_embed.proj.weight' in enc_dict and \
            enc_dict['patch_embed.proj.weight'].shape[1] == 1 and \
            self.encoder.patch_embed.proj.weight.shape[1] > 1:
            logger.info("Detected patch_embed.proj.weight size mismatch. Expanding 1-channel weights to multi-channel.")
            current_in_chans = self.encoder.patch_embed.proj.weight.shape[1]
            original_weight = enc_dict['patch_embed.proj.weight']
            new_weight = torch.zeros(
                original_weight.shape[0],
                current_in_chans,
                *original_weight.shape[2:]
            ).to(original_weight.device)
            new_weight[:, 0:1, :, :, :] = original_weight
            enc_dict['patch_embed.proj.weight'] = new_weight
            
        self.encoder.load_state_dict(enc_dict, strict=False)
        logger.info("✓ Loaded pretrained MAE encoder weights!")

    def forward(self, x):
        # Encoder returns: final latent, mask, ids_restore, skip_connections
        final_latent, _, _, _ = self.encoder(x, mask_ratio=0)
        
        # Pass latent features to the decoder
        logits = self.decoder(final_latent)
        
        return logits

    def compute_loss(self, logits, targets_onehot):
        if logits.shape[1] == 1:
            # Binary case: BCEWithLogits + Dice
            # target: (B, 2, D, H, W) -> take channel 1 (foreground)
            target_fg = targets_onehot[:, 1:2].float()
            
            dice_loss = self.dice_loss_fn(logits, targets_onehot)
            ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, target_fg)
            # Use 0.95 Dice + 0.05 CE to prioritize Dice score
            total_loss = 0.8 * dice_loss + 0.2 * ce_loss
            return total_loss, dice_loss, ce_loss
        else:
            # Multi-class case
            dice_loss = self.dice_loss_fn(logits, targets_onehot)
            ce_loss = nn.functional.cross_entropy(logits, torch.argmax(targets_onehot, dim=1))
            total_loss = 0.8 * dice_loss + 0.2 * ce_loss
            return total_loss, dice_loss, ce_loss

    def on_train_start(self):
        # 保证 encoder 参数可训练
        for param in self.encoder.parameters():
            param.requires_grad = True

    def training_step(self, batch, batch_idx):
        volumes, masks = batch
        logits = self(volumes)
        total_loss, dice_loss, focal_loss = self.compute_loss(logits, masks)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_focal_loss', focal_loss, on_step=True, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        volumes, masks = batch
        logits = self(volumes)
        total_loss, dice_loss, focal_loss = self.compute_loss(logits, masks)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_dice_loss', dice_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_focal_loss', focal_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        volumes, masks = batch
        logits = self(volumes)
        total_loss, dice_loss, focal_loss = self.compute_loss(logits, masks)
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_dice_loss', dice_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_focal_loss', focal_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        # Use same learning rate for encoder and decoder
        encoder_lr = self.learning_rate
        decoder_lr = self.learning_rate
        
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.decoder.parameters(), 'lr': decoder_lr}
        ], weight_decay=self.weight_decay)
        
        # Use CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch",
                "frequency": 1
            }
        }

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
                    'metrics': metrics
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
        metrics = sample['metrics']
        # metrics_str = f"Dice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f} | HD95: {metrics['hd95']:.1f}"
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
        ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes, color='white', fontsize=9, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        ax.set_title(f"Sample {i+1}: Pred")
        # Add metrics text on the image (top-left) with a background
        ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes, 
                color='white', fontsize=9, verticalalignment='top', 
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
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
    3D MAE Encoder + CNNDecoder分割训练主流程
    """
    logger.info("=== Starting 3D MAE-CNN Segmentation Training ===")
    
    volume_size = getattr(config, 'volume_size', (80,160,160))
    # Manually split files to enable augmentation only for training
    from pathlib import Path
    import random
    
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
        # strategy=config.strategy
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)
    logger.info("3D MAE-CNN segmentation training complete!")