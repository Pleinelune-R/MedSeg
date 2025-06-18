import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader

from data_prepare.load_file import read_nii_files, collect_sample_paths
from logger import get_logger
from .image_deocder import ImageDecoder
from .image_encoder import ImageEncoder
from .loss import MultiClassBCEDiceLoss
from data_prepare.data_module import MRDataModule

logger = get_logger("model")


class ModelConfig:
    """
    Configuration class for model parameters
    """

    def __init__(self,
                 # Image parameters
                 input_size=(128, 256, 256),
                 network_size=(64, 128, 128),
                 in_channels=4,

                 # Training parameters
                 batch_size=16,
                 max_epochs=70,
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 min_lr=1e-4,

                 # Early stopping parameters
                 early_stopping_patience=30,
                 early_stopping_min_delta=0.001,

                 # Data parameters
                 train_val_split=0.8,
                 num_workers=2):
        # Image parameters
        self.input_size = input_size
        self.network_size = network_size
        self.in_channels = in_channels

        # Training parameters
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_lr = min_lr

        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # Data parameters
        self.train_val_split = train_val_split
        self.num_workers = num_workers


class CustomLoggingCallback(pl.Callback):
    """Custom callback for logging training progress"""

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_start(self, trainer, pl_module):
        logger.info(f"\nEpoch {trainer.current_epoch} started")

    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = sum(self.train_losses) / len(self.train_losses)
        logger.info(f"Epoch {trainer.current_epoch} completed - Average Training Loss: {avg_loss:.4f}")
        self.train_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        logger.info(f"Validation completed - Average Validation Loss: {avg_loss:.4f}")
        self.val_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, torch.Tensor):
            self.train_losses.append(outputs.item())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, torch.Tensor):
            self.val_losses.append(outputs.item())


class MainModel(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Use configuration parameters
        self.input_size = config.input_size
        self.network_size = config.network_size
        self.in_channels = config.in_channels

        # ===== Model Components =====
        self.save_hyperparameters()  # Save hyperparameters for reproducibility
        self.test_results = []  # Store results for saving 
        self.test_images = []  # Store test images for visualization
        self.test_true_masks = []  # Store ground truth masks
        self.test_pred_masks = []  # Store predicted masks

        # Initialize encoder and decoder modules
        self.image_encoder = ImageEncoder(
            img_size=self.network_size,
            in_channels=self.in_channels,
            spatial_dims=3
        )
        args = type('Args', (), {'align_score': False, 'n_prompts': 0})
        self.image_decoder = ImageDecoder(args=args)

        # Use combined loss (Dice + BCE)
        self.criterion = MultiClassBCEDiceLoss(
            ce_weight=0.4,
            dice_weight=0.6,
            class_weights=[1.0, 1.0, 1.0, 1.0]
            # Based on data analysis: Background: 0.1, NCR/NET: 3.0, ED: 2.0, ET: 4.0
        )

        # Layer to resize input to network's expected size
        self.resize = torch.nn.Sequential(
            torch.nn.Upsample(size=self.network_size, mode='trilinear', align_corners=True)
        )

    def _build_model(self):
        # Your model building code here
        pass

    def forward(self, x):
        # Remove the unsqueeze since input already has channel dimension
        # x = x.unsqueeze(1)   # Remove this line since input is already [B, 4, 128, 256, 256]
        x = self.resize(x)  # Resize to [B, 4, 64, 128, 128]
        x = self.image_encoder(x)
        x = self.image_decoder(x)
        x = torch.nn.functional.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        return x  # Return [B, 4, 128, 256, 256] - keep channel dimension for multi-class

    def common_step(self, x, y):
        y_hat = self(x)
        mask, ce_loss, dice_loss, loss = self.criterion(y_hat, y)
        return mask, loss, dice_loss, ce_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        mask, loss, dice_loss, ce_loss = self.common_step(x, y)
        # Log training loss using print for progress updates
        if batch_idx % 10 == 0:  # Log every 10 batches to avoid too frequent logging
            print(
                f"\rEpoch {self.current_epoch}, Batch {batch_idx}, Train Loss: {loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}",
                end="")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss, dice_loss, ce_loss = self.common_step(x, y)
        # Log validation loss using print for progress updates
        if batch_idx % 10 == 0:  # Log every 10 batches
            print(
                f"\rEpoch {self.current_epoch}, Batch {batch_idx}, Val Loss: {loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}",
                end="")
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_dice_loss', dice_loss, on_step=True, on_epoch=True)
        self.log('val_ce_loss', ce_loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mask, loss, dice_loss, ce_loss = self.common_step(x, y)
        # Log test loss using print for progress updates
        if batch_idx % 10 == 0:  # Log every 10 batches
            print(
                f"\rTest Batch {batch_idx}, Test Loss: {loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}",
                end="")
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_dice_loss', dice_loss, on_step=True, on_epoch=True)
        self.log('test_ce_loss', ce_loss, on_step=True, on_epoch=True)

        # Store data for visualization
        self.test_images.extend(x.cpu().numpy())
        self.test_true_masks.extend(y.cpu().numpy())

        # Get model predictions - mask already contains the predictions
        # Convert to one-hot encoding for visualization
        pred_masks = torch.nn.functional.one_hot(mask, num_classes=4).permute(0, 4, 1, 2, 3)  # [B, 4, H, W, D]
        self.test_pred_masks.extend(pred_masks.cpu().numpy())

        return loss

    def on_train_epoch_end(self):
        # Print newline at the end of each epoch
        print()  # This will create a new line after the last batch update

    def on_validation_epoch_end(self):
        # Print newline at the end of each validation
        print()  # This will create a new line after the last validation batch update

    def on_test_epoch_end(self):
        # Print newline at the end of testing
        print()  # This will create a new line after the last test batch update

        # Print final evaluation metrics
        avg_loss = self.trainer.callback_metrics['test_loss']
        avg_dice_loss = self.trainer.callback_metrics['test_dice_loss']
        avg_ce_loss = self.trainer.callback_metrics['test_ce_loss']
        logger.info(f'Final Evaluation Metrics:')
        logger.info(f'Total Loss: {avg_loss:.4f}')
        logger.info(f'Dice Loss: {avg_dice_loss:.4f}')
        logger.info(f'CE Loss: {avg_ce_loss:.4f}')

        # Visualize masks after testing
        self.visualize_masks()

    def visualize_masks(self):
        # Visualize up to 20 samples: original image, true mask, predicted mask
        num_samples = min(20, len(self.test_images))  # Visualize up to 20 samples

        # Create a directory for saving individual images
        save_dir = 'test_results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create a combined figure for all samples
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        # Define colors for each class (0: background, 1: NCR/NET, 2: ED, 4: ET)
        colors = ['black', 'red', 'green', 'blue']
        cmap = plt.cm.colors.ListedColormap(colors)
        bounds = [0, 1, 2, 3, 4]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        for i in range(num_samples):
            # Get the middle slice of the 3D volume for visualization
            image = self.test_images[i]  # [4, 155, 240, 240] - 4 modalities
            true_mask = self.test_true_masks[i]  # [4, 155, 240, 240] - 4 one-hot encoded masks
            pred_mask = self.test_pred_masks[i]  # [4, 155, 240, 240] - 4 one-hot encoded masks

            # Select the middle slice and use FLAIR modality for visualization
            middle_slice = image.shape[1] // 2  # 155 // 2
            image_slice = image[0, middle_slice]  # Use FLAIR modality (first channel)

            # Convert one-hot encoded masks to single channel with proper labels
            true_mask_single = np.zeros((true_mask.shape[2], true_mask.shape[3]))
            pred_mask_single = np.zeros((pred_mask.shape[2], pred_mask.shape[3]))

            # Convert true mask (0: background, 1: NCR/NET, 2: ED, 4: ET)
            true_mask_single[true_mask[0, middle_slice] == 1] = 0  # Background
            true_mask_single[true_mask[1, middle_slice] == 1] = 1  # NCR/NET
            true_mask_single[true_mask[2, middle_slice] == 1] = 2  # ED
            true_mask_single[true_mask[3, middle_slice] == 1] = 4  # ET

            # Convert predicted mask (0: background, 1: NCR/NET, 2: ED, 4: ET)
            pred_mask_single[pred_mask[0, middle_slice] == 1] = 0  # Background
            pred_mask_single[pred_mask[1, middle_slice] == 1] = 1  # NCR/NET
            pred_mask_single[pred_mask[2, middle_slice] == 1] = 2  # ED
            pred_mask_single[pred_mask[3, middle_slice] == 1] = 4  # ET

            # Show original image (FLAIR modality)
            axes[i, 0].imshow(image_slice, cmap='gray')
            axes[i, 0].set_title('FLAIR Image')
            axes[i, 0].axis('off')

            # Show ground truth mask
            axes[i, 1].imshow(image_slice, cmap='gray', alpha=0.5)
            axes[i, 1].imshow(true_mask_single, cmap=cmap, norm=norm, alpha=0.5)
            axes[i, 1].set_title('True Mask')
            axes[i, 1].axis('off')

            # Show predicted mask
            axes[i, 2].imshow(image_slice, cmap='gray', alpha=0.5)
            axes[i, 2].imshow(pred_mask_single, cmap=cmap, norm=norm, alpha=0.5)
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')

            # Save individual sample
            fig_single, axes_single = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes_single[0].imshow(image_slice, cmap='gray')
            axes_single[0].set_title('FLAIR Image')
            axes_single[0].axis('off')

            # True mask
            axes_single[1].imshow(image_slice, cmap='gray', alpha=0.5)
            axes_single[1].imshow(true_mask_single, cmap=cmap, norm=norm, alpha=0.5)
            axes_single[1].set_title('True Mask')
            axes_single[1].axis('off')

            # Predicted mask
            axes_single[2].imshow(image_slice, cmap='gray', alpha=0.5)
            axes_single[2].imshow(pred_mask_single, cmap=cmap, norm=norm, alpha=0.5)
            axes_single[2].set_title('Predicted Mask')
            axes_single[2].axis('off')

            # Add colorbar with labels
            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes_single.ravel().tolist())
            cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
            cbar.set_ticklabels(['Background', 'NCR/NET', 'ED', 'ET'])

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{i + 1}.png'), bbox_inches='tight', dpi=300)
            plt.close(fig_single)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'all_samples.png'), bbox_inches='tight', dpi=300)  # Save high-quality image
        plt.close()  # Close figure to free memory

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Optimizer and learning rate scheduler configuration
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)

        # Use CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs,  # Set period to total epochs
            eta_min=self.config.min_lr  # Minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # Monitor validation loss for learning rate scheduling
            }
        }

        # TODO: update parameters


# Training function: loads data, splits, and runs training/testing
def train(dataset_path, devices_numbers, save_dir="./checkpoints"):
    """
    Train the model on the BraTS dataset.
    
    Args:
        dataset_path
        devices_numbers (int or List[int]): Number of GPU devices to use
        save_dir (str): Directory to save model checkpoints and logs
        max_samples (int, optional): Maximum number of samples to use for training. If None, use all available samples.
    """
    # 使用MRDataModule进行数据加载和分割
    data_module = MRDataModule(data_dir=dataset_path, batch_size=8, train_val_split=0.8, num_workers=2)
    data_module.setup()
    logger.info(f"Collected {len(data_module.train_dataset) + len(data_module.val_dataset)} samples.")

    config = ModelConfig(
        # Image parameters
        input_size=(128, 256, 256),
        network_size=(64, 128, 128),
        in_channels=4,  # Changed to 4 channels

        # Training parameters
        batch_size=8,  # Reduced batch size for better gradient updates
        max_epochs=4,  # Increased epochs
        learning_rate=5e-4,  # Slightly lower learning rate
        weight_decay=1e-4,
        min_lr=1e-6,

        # Early stopping parameters
        early_stopping_patience=25,  # Increased patience
        early_stopping_min_delta=0.0005,  # Smaller delta

        # Data parameters
        train_val_split=0.8,
        num_workers=2
    )
    model = MainModel(config)

    # 检查数据维度
    for batch in data_module.train_dataloader():
        x, y = batch
        print(f"Input data shape: {x.shape}")
        print(f"Label data shape: {y.shape}")
        break

    logger.info("Start training")
    trainer = pl.Trainer(
        max_epochs=model.config.max_epochs,
        accelerator='gpu',
        devices=devices_numbers,
        precision="32",
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',  # Monitor validation loss
                patience=model.config.early_stopping_patience,
                min_delta=model.config.early_stopping_min_delta,
                mode='min'
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='best_model',  # Monitor validation loss for model checkpointing
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir, name='test'),
        enable_progress_bar=False  # Disable default progress bar
    )
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
    trainer.test(model, dataloaders=data_module.test_dataloader())
    return trainer
