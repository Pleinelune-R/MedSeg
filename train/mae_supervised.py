"""
MAE-based Supervised Segmentation
Uses MAE encoder/decoder architecture for supervised segmentation tasks
"""
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from logger import get_logger
from model.mae_encoder import MAEEncoder
from model.mae_decoder import MAEDecoder

logger = get_logger("mae_supervised")


class MAESegmentationHead(nn.Module):
    """
    Segmentation head for MAE decoder output
    Converts decoder output to segmentation masks
    """
    def __init__(self, in_channels, num_classes=4, img_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Simple conv layers to convert to segmentation
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x: (B, C, H, W) from decoder
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        # Upsample to target size if needed
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size), 
                mode='bilinear', align_corners=True
            )
        
        return x


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, C, H, W) one-hot
        pred = torch.softmax(pred, dim=1)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class MAESegmentationModel(pl.LightningModule):
    """
    MAE-based Segmentation Model
    Uses MAE encoder/decoder for supervised segmentation
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        
        # Calculate number of patches
        num_patches = (img_size // patch_size) ** 2
        
        # MAE Encoder (without masking for supervised learning)
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # MAE Decoder
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # Segmentation head
        self.seg_head = MAESegmentationHead(
            in_channels=in_chans,
            num_classes=num_classes,
            img_size=img_size
        )
        
        # Losses
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # For visualization
        self.test_outputs = []
    
    def forward(self, x):
        """
        Forward pass for segmentation
        Args:
            x: (B, C, H, W) input images
        Returns:
            seg_logits: (B, num_classes, H, W) segmentation logits
        """
        # Encode without masking (mask_ratio=0)
        latent, _, ids_restore = self.encoder(x, mask_ratio=0.0)
        
        # Decode
        decoder_out = self.decoder(latent, ids_restore)
        
        # Convert to image
        reconstructed = self.decoder.unpatchify(decoder_out)
        
        # Segmentation head
        seg_logits = self.seg_head(reconstructed)
        
        return seg_logits
    
    def compute_loss(self, logits, targets):
        """
        Compute combined loss
        Args:
            logits: (B, C, H, W) prediction logits
            targets: (B, C, H, W) one-hot targets or (B, H, W) class indices
        """
        # Convert targets if needed
        if targets.dim() == 3:
            # (B, H, W) -> (B, C, H, W) one-hot
            targets_onehot = torch.nn.functional.one_hot(
                targets.long(), num_classes=self.num_classes
            ).permute(0, 3, 1, 2).float()
        else:
            targets_onehot = targets
        
        # Get class indices for CE loss
        target_indices = torch.argmax(targets_onehot, dim=1)
        
        # Compute losses
        dice_loss = self.dice_loss(logits, targets_onehot)
        ce_loss = self.ce_loss(logits, target_indices)
        
        total_loss = 0.5 * dice_loss + 0.5 * ce_loss
        
        return total_loss, dice_loss, ce_loss
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        total_loss, dice_loss, ce_loss = self.compute_loss(logits, masks)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True)
        
        # Prefer Lightning progress bar/TensorBoard over logging within loop
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        total_loss, dice_loss, ce_loss = self.compute_loss(logits, masks)
        
        # Log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        total_loss, dice_loss, ce_loss = self.compute_loss(logits, masks)
        
        # Store for visualization
        pred_masks = torch.argmax(logits, dim=1)
        self.test_outputs.append({
            'images': images.cpu(),
            'true_masks': masks.cpu(),
            'pred_masks': pred_masks.cpu()
        })
        
        self.log('test_loss', total_loss, on_step=False, on_epoch=True)
        self.log('test_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def on_test_epoch_end(self):
        self.visualize_results()
        self.visualize_results()
    
    def visualize_results(self, save_dir='test_results'):
        """Visualize test results"""
        os.makedirs(save_dir, exist_ok=True)
        
        num_samples = min(10, len(self.test_outputs))
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            output = self.test_outputs[i]
            img = output['images'][0, 0].numpy()  # First sample, first channel
            true_mask = output['true_masks'][0].numpy()
            pred_mask = output['pred_masks'][0].numpy()
            
            # Convert one-hot to class indices if needed
            if true_mask.ndim == 3:  # (C, H, W)
                true_mask = np.argmax(true_mask, axis=0)
            
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(true_mask, cmap='tab10', vmin=0, vmax=self.num_classes-1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=self.num_classes-1)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'segmentation_results.png'), dpi=200)
        plt.close()
        
        # Saved visualization
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


def train_mae_supervised(config, save_dir="./checkpoints_supervised"):
    """
    Train MAE-based supervised segmentation
    Args:
        config: ModelConfig with training parameters
        save_dir: Directory to save checkpoints
    """
    # Start supervised training
    
    # Import here to avoid circular dependency
    from data.mae_dataset import MedicalSegmentationDataset
    
    # Create dataset
    dataset = MedicalSegmentationDataset(
        h5_dir=config.dataset_path,
        img_size=getattr(config, 'img_size', 256),
        crop_ratio=getattr(config, 'crop_ratio', 0.3),
        num_classes=4
    )
    
    # Split into train, val, test
    train_size = int(len(dataset) * config.train_val_split)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Dataset split sizes (optional)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Create model
    model = MAESegmentationModel(
        img_size=getattr(config, 'img_size', 256),
        patch_size=16,
        in_chans=1,
        num_classes=4,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=config.devices_numbers,
        precision="32",
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                filename='best_seg_model',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=getattr(config, 'early_stopping_patience', 20),
                mode='min'
            ),
            LearningRateMonitor(logging_interval='epoch')
        ],
        logger=TensorBoardLogger(save_dir, name='mae_supervised'),
        enable_progress_bar=True,
        strategy=config.strategy
    )
    
    # Fit & test
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)
    
    # Done
    return trainer

