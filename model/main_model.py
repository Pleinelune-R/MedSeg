import pytorch_lightning as pl 
import torch 
from torch.utils.data  import DataLoader 
from pytorch_lightning.utilities.types  import OptimizerLRScheduler 
from pytorch_lightning.loggers  import TensorBoardLogger 
import matplotlib.pyplot  as plt 
import numpy as np 
 
from data_prepare.data_iter  import MRDataset 
from data_prepare.load_file  import read_nii_files 
from logger import get_logger 
from .image_deocder import ImageDecoder 
from .image_encoder import ImageEncoder 
from .loss import BCEDiceLoss 
 
logger = get_logger("model") 
 
class ModelConfig:
    """
    Configuration class for model parameters
    """
    def __init__(self,
                 # Image parameters
                 input_size=(128, 256, 256),
                 network_size=(64, 128, 128),
                 in_channels=1,
                 
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
        self.criterion = BCEDiceLoss() 

        # Layer to resize input to network's expected size
        self.resize = torch.nn.Sequential(  
            torch.nn.Upsample(size=self.network_size, mode='trilinear', align_corners=True) 
        ) 
 
    def _build_model(self):
        # Your model building code here
        pass
        
    def forward(self, x): 
        # preprocess, encode, decode, and resize output
        x = x.unsqueeze(1)   # Add channel dimension: [B, 1, 128, 256, 256] 
        x = self.resize(x)   # Resize to [B, 1, 64, 128, 128] 
        x = self.image_encoder(x) 
        x = self.image_decoder(x)  
        x = torch.nn.functional.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True) # Restore to original size
        return x.squeeze(1)   # Remove channel dimension, return to original shape
    
    def common_step(self, x, y):
        y_hat = self(x)
        mask, dice_loss, bce_loss, loss = self.criterion(y_hat, y)
        return mask, loss, dice_loss, bce_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        mask, loss, _, _ = self.common_step(x, y)
        # 简化日志记录，移除可能导致问题的参数
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss, _, _ = self.common_step(x, y)
        # 只在验证步骤中显示进度条
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mask, loss, _, _ = self.common_step(x, y)
        # Log detailed metrics only during testing
        self.test_results.append(loss.item())
        self.log('test_loss', loss)
        
        # Store data for visualization
        self.test_images.extend(x.cpu().numpy())
        self.test_true_masks.extend(y.cpu().numpy())
        pred_masks = torch.sigmoid(self(x))
        pred_masks = (pred_masks > 0.5).float()
        self.test_pred_masks.extend(pred_masks.cpu().numpy())
        
        return loss

    def on_test_epoch_end(self):
        # Print final evaluation metrics
        avg_loss = self.trainer.callback_metrics['test_loss']
        logger.info(f'Final Evaluation Metrics:')
        logger.info(f'Total Loss: {avg_loss:.4f}')
        
        # Visualize masks after testing
        self.visualize_masks()

    def visualize_masks(self): 
        # Visualize up to 5 samples: original image, true mask, predicted mask
        num_samples = min(5, len(self.test_images))   # Visualize up to 5 samples 
        fig, axes = plt.subplots(num_samples,  3, figsize=(15, 5 * num_samples)) 
 
        for i in range(num_samples): 
            # Get the middle slice of the 3D volume for visualization
            image = self.test_images[i]   # [128, 256, 256] 
            true_mask = self.test_true_masks[i]   # [3, 128, 256, 256] 
            pred_mask = self.test_pred_masks[i]   # [3, 128, 256, 256] 
 
            # Select the middle slice
            middle_slice = image.shape[0]  // 2 + 1
            image_slice = image[middle_slice] 
            true_mask_slice = true_mask[:, middle_slice]  # [3, 256, 256] 
            pred_mask_slice = pred_mask[:, middle_slice]  # [3, 256, 256] 
 
            # Convert 3-channel mask to single-channel 4-class mask
            true_mask_single = np.zeros((true_mask_slice.shape[1],  true_mask_slice.shape[2]))  
            pred_mask_single = np.zeros((pred_mask_slice.shape[1],  pred_mask_slice.shape[2]))  
 
            # Class 0: 000
            true_mask_single[(true_mask_slice[0] == 0) & (true_mask_slice[1] == 0) & (true_mask_slice[2] == 0)] = 0 
            pred_mask_single[(pred_mask_slice[0] == 0) & (pred_mask_slice[1] == 0) & (pred_mask_slice[2] == 0)] = 0 
            # Class 1: 110
            true_mask_single[(true_mask_slice[0] == 1) & (true_mask_slice[1] == 1) & (true_mask_slice[2] == 0)] = 1 
            pred_mask_single[(pred_mask_slice[0] == 1) & (pred_mask_slice[1] == 1) & (pred_mask_slice[2] == 0)] = 1 
            # Class 2: 100
            true_mask_single[(true_mask_slice[0] == 1) & (true_mask_slice[1] == 0) & (true_mask_slice[2] == 0)] = 2 
            pred_mask_single[(pred_mask_slice[0] == 1) & (pred_mask_slice[1] == 0) & (pred_mask_slice[2] == 0)] = 2 
            # Class 4: 111
            true_mask_single[(true_mask_slice[0] == 1) & (true_mask_slice[1] == 1) & (true_mask_slice[2] == 1)] = 4 
            pred_mask_single[(pred_mask_slice[0] == 1) & (pred_mask_slice[1] == 1) & (pred_mask_slice[2] == 1)] = 4 

            # Show original image
            axes[i, 0].imshow(image_slice, cmap='gray') 
            axes[i, 0].set_title('Original Image') 
            axes[i, 0].axis('off') 
 
            # Show ground truth mask
            axes[i, 1].imshow(image_slice, cmap='gray') 
            axes[i, 1].imshow(true_mask_single, cmap='tab20', vmin=0, vmax=4) 
            axes[i, 1].set_title('True Mask') 
            axes[i, 1].axis('off') 
 
            # Show predicted mask
            axes[i, 2].imshow(image_slice, cmap='gray') 
            axes[i, 2].imshow(pred_mask_single, cmap='tab20', vmin=0, vmax=4) 
            axes[i, 2].set_title('Predicted Mask') 
            axes[i, 2].axis('off') 
 
        plt.tight_layout()  
        plt.savefig('test_results.png',  bbox_inches='tight', dpi=300)  # Save high-quality image
        plt.close()   # Close figure to free memory
 
    def configure_optimizers(self) -> OptimizerLRScheduler: 
        # Optimizer and learning rate scheduler configuration
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay) 

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
    dataset = read_nii_files(dataset_path) 
    images = dataset['images']  # [4, n, 155, 256, 256] 
    labels = dataset['labels']  # [3, n, 155, 256, 256]，确保这里是三通道的 
    logger.info("Datasets loaded successfully") 

    # Only use the nth modality image (here, index 1)
    images = images[1]  # [1, n, 155, 256, 256] 

    medical_dataset = MRDataset(images, labels) 
    config = ModelConfig(
        # Image parameters
        input_size=(128, 256, 256),
        network_size=(64, 128, 128),
        in_channels=1,
        
        # Training parameters
        batch_size=32,  
        max_epochs=80,  # 增加训练轮数
        learning_rate=5e-4,  # 调整学习率
        weight_decay=1e-5,  # 减小权重衰减
        min_lr=5e-7,  # 最小学习率
        
        # Early stopping parameters
        early_stopping_patience=20,  # 增加早停耐心值
        early_stopping_min_delta=0.001,  # 降低最小改善阈值
        
        # Data parameters
        train_val_split=0.8,
        num_workers=2
    )
    model = MainModel(config)
    train_size = int(model.config.train_val_split * len(medical_dataset)) 
    test_size = len(medical_dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(medical_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)) 
    train_loader = DataLoader(train_dataset, batch_size=model.config.batch_size, shuffle=True, num_workers=model.config.num_workers) 
    test_loader = DataLoader(test_dataset, batch_size=model.config.batch_size, shuffle=False, num_workers=model.config.num_workers) 
    # Check data dimensions for debugging
    for batch in train_loader: 
        x, y = batch 
        print(f"Input data shape: {x.shape}")  
        print(f"Label data shape: {y.shape}")   # Ensure label is 3-channel
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
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.ProgressBar()  # 使用默认进度条配置
        ], 
        log_every_n_steps=1, 
        logger=TensorBoardLogger(save_dir, name='test')
    ) 
    trainer.fit(model,  train_dataloaders=train_loader, val_dataloaders=test_loader) 
    trainer.test(model,  dataloaders=test_loader) 
    return trainer 
 