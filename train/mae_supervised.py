"""
3D MAE Pretraining Module for Medical Volumes
Supports 3D volumetric data (D×H×W)
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
from model.config import ModelConfig
from model.mae_encoder import MAEEncoder3D
from model.mae_decoder import MAEDecoder
from model.mae_loss import MAELossWithVisualization
from datasets.mae_dataset import MedicalMAEDataset

logger = get_logger("mae_pretrain")


class ValidationVisualizationCallback(pl.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0 or trainer.current_epoch == 0:
            val_loader = trainer.val_dataloaders
            device = pl_module.device
            
            # Create a directory for the current epoch
            epoch_save_dir = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch + 1}")
            os.makedirs(epoch_save_dir, exist_ok=True)
            
            try:
                create_reconstruction_grid(pl_module, val_loader, device, num_samples=4, save_dir=epoch_save_dir)
            except Exception as e:
                logger.error(f"Error generating epoch visualization: {e}")


class MAEPretrainModel(pl.LightningModule):
    def __init__(self, config: ModelConfig, mask_ratio=0.75, norm_pix_loss=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        volume_size = getattr(config, 'volume_size', (80, 160, 160))
        patch_size = config.patch_size
        
        if isinstance(volume_size, int):
            volume_size_tuple = (volume_size, volume_size, volume_size)
        else:
            volume_size_tuple = tuple(volume_size)
        grid_size = tuple([v // patch_size for v in volume_size_tuple])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        
        logger.info(f"Initializing 3D MAE Model:")
        logger.info(f"  Volume size: {volume_size_tuple}³")
        logger.info(f"  Patch size: {patch_size}³")
        logger.info(f"  Grid size: {grid_size}³")
        logger.info(f"  Number of patches: {num_patches}")
        
        self.encoder = MAEEncoder3D(
            volume_size=volume_size_tuple,
            patch_size=patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio, drop_rate=config.dropout
        )
        
        self.decoder = MAEDecoder(
            num_patches=grid_size,
            patch_size=patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio, drop_rate=config.dropout
        )
        
        self.loss_fn = MAELossWithVisualization(
            patch_size=patch_size,
            in_chans=config.in_chans,
            norm_pix_loss=norm_pix_loss
        )
        self.loss_fn.volume_shape = volume_size_tuple
        
        self.mask_ratio = mask_ratio
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.warmup_epochs = config.warmup_epochs
        self.max_epochs = config.max_epochs
        self.volume_size = volume_size_tuple
    
    def forward(self, volumes, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore, _ = self.encoder(volumes, mask_ratio)
        
        pred = self.decoder(latent, ids_restore)
        
        loss = self.loss_fn(volumes, pred, mask)
        
        return loss, pred, mask
    
    def training_step(self, batch, batch_idx):
        volumes, _ = batch
        loss, pred, mask = self.forward(volumes)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        volumes, _ = batch
        loss, pred, mask = self.forward(volumes)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)
        )
        
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
    
    def get_reconstruction_visualization(self, volumes):
        self.eval()
        self.loss_fn.volume_shape = tuple(volumes.shape[2:])
        with torch.no_grad():
            latent, mask, ids_restore, _ = self.encoder(volumes, self.mask_ratio)
            pred = self.decoder(latent, ids_restore)
            _, viz_data = self.loss_fn(volumes, pred, mask, return_visualization=True)
        self.train()
        return viz_data


def create_reconstruction_grid(model, dataloader, device, num_samples=4, save_dir="mae_results_3d"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    if len(dataloader.dataset) == 0:
        logger.warning("Dataset is empty, cannot generate reconstruction result")
        return
    
    with torch.no_grad():
        try:
            volumes, _ = next(iter(dataloader))
            
            B = min(num_samples, volumes.size(0))
            if B == 0:
                logger.warning("Batch size is 0, cannot generate reconstruction result")
                return
                
            volumes = volumes[:B].to(device)
            logger.info(f"Generating {B} 3D volume reconstruction results...")
            logger.info(f"Volume shape: {volumes.shape}")
            
            viz_data = model.get_reconstruction_visualization(volumes)
            
            def to_display(tensor_img):
                img = tensor_img.cpu().numpy()
                if len(img.shape) == 4:
                    img = img[:, 0, :, :]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                return img
            
            original = to_display(viz_data['original'])
            masked = to_display(viz_data['masked'])
            reconstructed = to_display(viz_data['reconstructed'])
            reconstruction_paste = to_display(viz_data['reconstruction_paste'])
            
            fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
            if B == 1:
                axes = axes.reshape(1, -1)
            
            titles = ['Original (mid slice)', 'Masked', 'Reconstruction', 'Reconstruction + Visible']
            
            for i in range(B):
                images_to_show = [
                    original[i],
                    masked[i],
                    reconstructed[i],
                    reconstruction_paste[i]
                ]
                
                for j, (img, title) in enumerate(zip(images_to_show, titles)):
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title(f'{title} (Sample {i+1})', fontsize=10, fontweight='bold')
                    axes[i, j].axis('off')
            
            plt.tight_layout()
            
            grid_save_path = os.path.join(save_dir, 'reconstruction_grid.png')
            plt.savefig(grid_save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Generated {B} 3D reconstruction result grid visualization: {grid_save_path}")
            
        except Exception as e:
            logger.error(f"Error generating 3D reconstruction result: {e}")
            import traceback
            logger.error(traceback.format_exc())


def pretrain_mae(config, mask_ratio=0.75, norm_pix_loss=False, save_dir="./checkpoints_mae_3d"):
    logger.info("Starting 3D MAE pretraining...")
    
    dataset = MedicalMAEDataset(
        h5_dir=config.dataset_path,
        volume_size=getattr(config, 'volume_size', (80,160,160)),
        config=config
    )
    
    train_size = int(len(dataset) * config.train_val_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Train volumes: {train_size}, Val volumes: {val_size}")
    
    batch_size = min(config.batch_size, 4)
    logger.info(f"Using batch size: {batch_size} (adjusted for 3D data)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        drop_last=True
    )
    
    model = MAEPretrainModel(
        config=config,
        mask_ratio=mask_ratio,
        norm_pix_loss=norm_pix_loss
    )
    model.validation_loader = val_loader
    
    vis_callback = ValidationVisualizationCallback(save_dir=os.path.join(save_dir, "validation_images"))
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=config.devices_numbers,
        precision="16-mixed",
        # profiler=config.profiler,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                filename='mae_3d_best_model',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            LearningRateMonitor(logging_interval='epoch'),
            vis_callback
        ],
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir, name='mae_pretrain'),
        enable_progress_bar=True,
        # strategy=config.strategy
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    device = torch.device(f'cuda:{config.devices_numbers[0]}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    create_reconstruction_grid(
        model, val_loader, device,
        num_samples=4,
        save_dir=os.path.join(save_dir, "final_mae_results_3d")
    )
    
    logger.info("3D MAE pretraining complete!")
    return trainer

