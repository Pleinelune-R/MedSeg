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
 
 
class MainModel(pl.LightningModule): 
    def __init__(self): 
        super().__init__()
        
        # ===== Frequently Modified Parameters =====
        # Image parameters
        self.input_size = (128, 256, 256)  # Original input size
        self.network_size = (64, 128, 128)  # Network expected size
        self.in_channels = 1
        
        # Training parameters
        self.batch_size = 16
        self.max_epochs = 50
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.min_lr = 1e-4
        
        # Early stopping parameters
        self.early_stopping_patience = 30
        self.early_stopping_min_delta = 0.001
        
        # Data parameters
        self.train_val_split = 0.8
        self.num_workers = 2
        
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
 
    def forward(self, x): 
        # preprocess, encode, decode, and resize output
        x = x.unsqueeze(1)   # Add channel dimension: [B, 1, 128, 256, 256] 
        x = self.resize(x)   # Resize to [B, 1, 64, 128, 128] 
        x = self.image_encoder(x) 
        x = self.image_decoder(x)  
        x = torch.nn.functional.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True) # Restore to original size
        return x.squeeze(1)   # Remove channel dimension, return to original shape
 
    def training_step(self, batch, batch_idx): 
        # Training step: forward + loss + log
        x, y = batch 
        y_hat = self(x) 
        mask, loss = self.criterion(y_hat,  y) 
        self.log('train_loss',  loss) 
        return loss 
 
    def validation_step(self, batch, batch_idx): 
        x, y = batch 
        y_hat = self(x) 
        mask, loss = self.criterion(y_hat,  y) 
        self.log('val_loss',  loss, prog_bar=True) 
        return loss 
 
    def test_step(self, batch, batch_idx): 
        # model forward 
        x, y = batch 
        y_hat = self(x) 
        mask, loss = self.criterion(y_hat,  y) 
        self.test_results.append(loss.item())  # Store loss for averaging
        self.log('test_loss',  loss) 
 
        # Store images, true masks, and predicted masks for later visualization
        self.test_images.extend(x.cpu().numpy())  
        self.test_true_masks.extend(y.cpu().numpy())  
        pred_masks = mask
        self.test_pred_masks.extend(pred_masks.cpu().numpy())  
 
    def on_test_epoch_end(self): 
        # Called at the end of test epoch: log average loss and visualize results
        avg_test_loss = sum(self.test_results)  / len(self.test_results)  
        self.log('avg_test_loss',  avg_test_loss) 
        print(f"Average Test Loss: {avg_test_loss}") 
 
        # Visualize images with masks
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) 

        # Use CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  
            optimizer, 
            T_max=self.max_epochs,  # Set period to total epochs
            eta_min=self.min_lr  # Minimum learning rate
        ) 

        return { 
            "optimizer": optimizer, 
            "lr_scheduler": { 
                "scheduler": scheduler, 
                "monitor": "val_loss" 
            } 
        } 
 
    # TODO: update parameters 
 
# Training function: loads data, splits, and runs training/testing
def train(devices_numbers, save_dir): 
    dataset = read_nii_files("/home/Datasets/bionet/Dataset/lsj_MICCAI_BraTS2020_TrainingData/") 
    images = dataset['images']  # [4, n, 155, 256, 256] 
    labels = dataset['labels']  # [3, n, 155, 256, 256]，确保这里是三通道的 
    logger.info("Datasets loaded successfully") 

    # Only use the nth modality image (here, index 1)
    images = images[1]  # [1, n, 155, 256, 256] 

    medical_dataset = MRDataset(images, labels) 
    model = MainModel()
    train_size = int(model.train_val_split * len(medical_dataset)) 
    test_size = len(medical_dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(medical_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)) 
    train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, num_workers=model.num_workers) 
    test_loader = DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False, num_workers=model.num_workers) 
    # Check data dimensions for debugging
    for batch in train_loader: 
        x, y = batch 
        print(f"Input data shape: {x.shape}")  
        print(f"Label data shape: {y.shape}")   # Ensure label is 3-channel
        break 

    logger.info("Start  training") 
    trainer = pl.Trainer( 
        max_epochs=model.max_epochs,
        accelerator='gpu', 
        devices=devices_numbers, 
        precision="16", 
        callbacks=[ 
            pl.callbacks.EarlyStopping(  
                monitor='val_loss', 
                patience=model.early_stopping_patience,
                min_delta=model.early_stopping_min_delta,
                mode='min' 
            ), 
            pl.callbacks.ModelCheckpoint(  
                dirpath=save_dir, 
                filename='MED-{epoch:02d}-{val_loss:.3f}', 
                monitor='val_loss', 
                mode='min', 
                save_top_k=3, 
                save_last=True 
            ), 
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')  
        ], 
        log_every_n_steps=1, 
        logger=TensorBoardLogger(save_dir, name='test') 
    ) 
    trainer.fit(model,  train_dataloaders=train_loader, val_dataloaders=test_loader) 
    trainer.test(model,  dataloaders=test_loader) 
    return trainer 
 