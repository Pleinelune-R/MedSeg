"""
MAE Pretraining Module for Medical Images
Based on simple_mae_medical.py
"""
import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from logger import get_logger
from model.pretrain_model.mae_encoder import MAEEncoder
from model.pretrain_model.mae_deocder import MAEDecoder
from model.pretrain_model.mae_loss import MAELossWithVisualization
from datasets.mae_dataset import MedicalMAEDataset

logger = get_logger("pretrain_pretrain")


class MAEPretrainModel(pl.LightningModule):
    """
    MAE Pretraining Model using PyTorch Lightning
    Combines encoder, decoder, and loss for pretraining
    """
    def __init__(
        self,
        volume_size=160,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        mask_ratio=0.75,
        norm_pix_loss=False,
        learning_rate=1.5e-4,
        weight_decay=0.05,
        warmup_epochs=10,
        max_epochs=100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Normalize volume_size to tuple
        if isinstance(volume_size, int):
            volume_size = (volume_size, volume_size, volume_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        
        # Calculate grid size and total patches for 3D volumes
        grid_size = tuple(v // p for v, p in zip(volume_size, patch_size))
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        
        # Initialize encoder with 3D volume size
        self.encoder = MAEEncoder(
            volume_size=volume_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # Initialize decoder with grid_size tuple
        self.decoder = MAEDecoder(
            num_patches=grid_size,
            patch_size=patch_size[0],
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # Initialize loss
        self.loss_fn = MAELossWithVisualization(
            patch_size=patch_size[0],
            in_chans=in_chans,
            norm_pix_loss=norm_pix_loss
        )
        
        # Store parameters
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
    
    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass through encoder and decoder
        Args:
            imgs: [N, C, H, W] input images
            mask_ratio: masking ratio (default: self.mask_ratio)
        Returns:
            loss: reconstruction loss
            pred: predicted pixel values
            mask: binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Encode with masking
        latent, mask, ids_restore, _ = self.encoder(imgs, mask_ratio)
        
        # Decode
        pred = self.decoder(latent, ids_restore)
        
        # Compute loss
        loss = self.loss_fn(imgs, pred, mask)
        
        return loss, pred, mask
    
    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, pred, mask = self.forward(imgs)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Optional: rely on Lightning progress bar/TensorBoard instead of logger spam
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, pred, mask = self.forward(imgs)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.max_epochs - self.warmup_epochs),
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def get_reconstruction_visualization(self, imgs):
        """
        Generate reconstruction visualization
        Args:
            imgs: [N, C, H, W] input images
        Returns:
            dict with visualization data
        """
        self.eval()
        with torch.no_grad():
            latent, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
            pred = self.decoder(latent, ids_restore)
            _, viz_data = self.loss_fn(imgs, pred, mask, return_visualization=True)
        self.train()
        return viz_data


def create_reconstruction_grid(model, dataloader, device, num_samples=4, save_dir="pretrain_results"):
    """Create reconstruction grid display"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if dataset is empty
    if len(dataloader.dataset) == 0:
        logger.warning("Dataset is empty, cannot generate reconstruction result")
        return
    
    with torch.no_grad():
        try:
            # Get some samples
            images, _ = next(iter(dataloader))
            
            B = min(num_samples, images.size(0))
            if B == 0:
                logger.warning("Batch size is 0, cannot generate reconstruction result")
                return
                
            images = images[:B].to(device)
            logger.info(f"Generating {B} samples reconstruction result...")
            
            # Get reconstruction visualization
            viz_data = model.get_reconstruction_visualization(images)
            
            # Convert to numpy for visualization
            def to_display(tensor_img):
                # [N, C, H, W] -> [N, H, W]
                img = tensor_img.cpu().numpy()
                if len(img.shape) == 4:
                    img = img[:, 0, :, :]  # Take first channel
                # Normalize to [0, 1]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                return img
            
            original = to_display(viz_data['original'])
            masked = to_display(viz_data['masked'])
            reconstructed = to_display(viz_data['reconstructed'])
            reconstruction_paste = to_display(viz_data['reconstruction_paste'])
            
            # Create grid layout visualization
            fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
            if B == 1:
                axes = axes.reshape(1, -1)
            
            titles = ['Original', 'Masked', 'Reconstruction', 'Reconstruction + Visible']
            
            for i in range(B):
                images_to_show = [
                    original[i],
                    masked[i],
                    reconstructed[i],
                    reconstruction_paste[i]
                ]
                
                # Display images
                for j, (img, title) in enumerate(zip(images_to_show, titles)):
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title(f'{title} (Sample {i+1})', fontsize=10, fontweight='bold')
                    axes[i, j].axis('off')
            
            plt.tight_layout()
            
            # Save grid image
            grid_save_path = os.path.join(save_dir, f'reconstruction_grid.png')
            plt.savefig(grid_save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Generated {B} reconstruction result grid visualization: {grid_save_path}")
            
        except Exception as e:
            logger.error(f"Error generating reconstruction result: {e}")
            import traceback
            logger.error(traceback.format_exc())


def pretrain_mae(config, mask_ratio=0.75, save_dir="./checkpoints_mae", norm_pix_loss=False):
    """
    MAE pretraining function
    Args:
        config: ModelConfig object with training parameters
        mask_ratio: ratio of patches to mask
        save_dir: directory to save checkpoints
    Returns:
        trainer: PyTorch Lightning trainer
    """
    # Start pretraining
    
    # Create dataset
    dataset = MedicalMAEDataset(
        h5_dir=config.dataset_path,
        volume_size=getattr(config, 'volume_size', 160),
        config=config
    )
    
    # Split into train and validation
    train_size = int(len(dataset) * config.train_val_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Dataset sizes (optional)
    
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
        drop_last=True
    )
    
    # Create model
    volume_size = getattr(config, 'volume_size', 160)
    patch_size = getattr(config, 'patch_size', 16)
    
    model = MAEPretrainModel(
        volume_size=volume_size,
        patch_size=patch_size,
        in_chans=1,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        mask_ratio=mask_ratio,
        norm_pix_loss=norm_pix_loss,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_epochs=getattr(config, 'warmup_epochs', 10),
        max_epochs=config.max_epochs
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=getattr(config, "devices_numbers", [0]),
        precision="32",
        profiler=getattr(config, "profiler", "simple"),
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                filename='pretrain_best_model',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            LearningRateMonitor(logging_interval='epoch')
        ],
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir, name='pretrain_pretrain'),
        enable_progress_bar=True
    )
    
    # Fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Generate final reconstruction visualization
    # Generate final reconstruction visualization
    devices_list = getattr(config, "devices_numbers", [0])
    device = torch.device(f'cuda:{devices_list[0]}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    create_reconstruction_grid(
        model, val_loader, device,
        num_samples=8,
        save_dir=os.path.join(save_dir, "final_pretrain_results")
    )
    
    # Done
    return trainer

