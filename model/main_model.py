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
        self.save_hyperparameters()  
        self.test_results  = []  # Store results for saving 
        self.test_images  = [] 
        self.test_true_masks  = [] 
        self.test_pred_masks  = [] 
 
        # module init 
        self.image_encoder  = ImageEncoder( 
            img_size=(64, 128, 128),  # 将输入尺寸调整为网络期望的大小 
            in_channels=1, 
            spatial_dims=3 
        ) 
        args = type('Args', (), {'align_score': False, 'n_prompts': 0}) 
        self.image_decoder  = ImageDecoder(args=args) 
 
        # 使用组合损失函数（Dice Loss + BCE Loss） 
        self.criterion  = BCEDiceLoss() 
 
        # 添加用于调整输入大小的层 
        self.resize  = torch.nn.Sequential(  
            torch.nn.Upsample(size=(64,  128, 128), mode='trilinear', align_corners=True) 
        ) 
 
    def forward(self, x): 
        # 添加通道维度 
        x = x.unsqueeze(1)   # [B, 1, 128, 256, 256] 
        # 调整输入大小 
        x = self.resize(x)   # [B, 1, 64, 128, 128] 
        x = self.image_encoder(x)  
        x = self.image_decoder(x)  
        # 调整输出大小以匹配原始大小 
        x = torch.nn.functional.interpolate(x,  size=(128, 256, 256), mode='trilinear', align_corners=True) 
        return x.squeeze(1)   # 移除通道维度，返回到原始形状 
 
    def training_step(self, batch, batch_idx): 
        # TODO：model forward 
        x, y = batch 
        y_hat = self(x) 
        mask, loss = self.criterion(y_hat,  y) 
        self.log('train_loss',  loss) 
        return loss 
 
    def validation_step(self, batch, batch_idx): 
        # TODO：model forward 
        x, y = batch 
        y_hat = self(x) 
        mask, loss = self.criterion(y_hat,  y) 
        self.log('val_loss',  loss, prog_bar=True) 
        return loss 
 
    def test_step(self, batch, batch_idx): 
        # TODO：model forward 
        x, y = batch 
        y_hat = self(x) 
        mask, loss = self.criterion(y_hat,  y) 
        # 对每个通道分别应用 sigmoid 和四舍五入 
        self.test_results.append(loss.item())  
        self.log('test_loss',  loss) 
 
        # Store images, true masks, and predicted masks 
        self.test_images.extend(x.cpu().numpy())  
        self.test_true_masks.extend(y.cpu().numpy())  
        pred_masks = mask
        self.test_pred_masks.extend(pred_masks.cpu().numpy())  
 
    def on_test_epoch_end(self): 
        # TODO：save batch results and calculate AUC or other metric 
        avg_test_loss = sum(self.test_results)  / len(self.test_results)  
        self.log('avg_test_loss',  avg_test_loss) 
        print(f"Average Test Loss: {avg_test_loss}") 
 
        # Visualize images with masks 
        self.visualize_masks()  
 
    def visualize_masks(self): 
        num_samples = min(5, len(self.test_images))   # Visualize up to 5 samples 
        fig, axes = plt.subplots(num_samples,  3, figsize=(15, 5 * num_samples)) 
 
        for i in range(num_samples): 
            # 获取3D体积的中间切片 
            image = self.test_images[i]   # [128, 256, 256] 
            true_mask = self.test_true_masks[i]   # [3, 128, 256, 256] 
            pred_mask = self.test_pred_masks[i]   # [3, 128, 256, 256] 
 
            # 选择中间切片 
            middle_slice = image.shape[0]  // 2 + 1
            image_slice = image[middle_slice] 
            true_mask_slice = true_mask[:, middle_slice]  # [3, 256, 256] 
            pred_mask_slice = pred_mask[:, middle_slice]  # [3, 256, 256] 
 
            # 将三通道的掩码转换为单通道的4类掩码 
            true_mask_single = np.zeros((true_mask_slice.shape[1],  true_mask_slice.shape[2]))  
            pred_mask_single = np.zeros((pred_mask_slice.shape[1],  pred_mask_slice.shape[2]))  
 
            # 0类: 000 
            true_mask_single[(true_mask_slice[0] == 0) & (true_mask_slice[1] == 0) & (true_mask_slice[2] == 0)] = 0 
            pred_mask_single[(pred_mask_slice[0] == 0) & (pred_mask_slice[1] == 0) & (pred_mask_slice[2] == 0)] = 0 
            # 1类: 110 
            true_mask_single[(true_mask_slice[0] == 1) & (true_mask_slice[1] == 1) & (true_mask_slice[2] == 0)] = 1 
            pred_mask_single[(pred_mask_slice[0] == 1) & (pred_mask_slice[1] == 1) & (pred_mask_slice[2] == 0)] = 1 
            # 2类: 100 
            true_mask_single[(true_mask_slice[0] == 1) & (true_mask_slice[1] == 0) & (true_mask_slice[2] == 0)] = 2 
            pred_mask_single[(pred_mask_slice[0] == 1) & (pred_mask_slice[1] == 0) & (pred_mask_slice[2] == 0)] = 2 
            # 4类: 111 
            true_mask_single[(true_mask_slice[0] == 1) & (true_mask_slice[1] == 1) & (true_mask_slice[2] == 1)] = 4 
            pred_mask_single[(pred_mask_slice[0] == 1) & (pred_mask_slice[1] == 1) & (pred_mask_slice[2] == 1)] = 4 

 
            # 显示原始图像 
            axes[i, 0].imshow(image_slice, cmap='gray') 
            axes[i, 0].set_title('Original Image') 
            axes[i, 0].axis('off') 
 
            # 显示真实掩码 
            axes[i, 1].imshow(image_slice, cmap='gray') 
            axes[i, 1].imshow(true_mask_single, cmap='tab20', vmin=0, vmax=4) 
            axes[i, 1].set_title('True Mask') 
            axes[i, 1].axis('off') 
 
            # 显示预测掩码 
            axes[i, 2].imshow(image_slice, cmap='gray') 
            axes[i, 2].imshow(pred_mask_single, cmap='tab20', vmin=0, vmax=4) 
            axes[i, 2].set_title('Predicted Mask') 
            axes[i, 2].axis('off') 
 
        plt.tight_layout()  
        plt.savefig('test_results.png',  bbox_inches='tight', dpi=300)  # 保存高质量图像 
        plt.close()   # 关闭图像以释放内存 
 
    def configure_optimizers(self) -> OptimizerLRScheduler: 
        optimizer = torch.optim.AdamW(self.parameters(),  lr=1e-3, weight_decay=1e-4) 
 
        # 使用 CosineAnnealingLR 调度器 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  
            optimizer, 
            T_max=50,  # 周期设为总epoch数 
            eta_min=1e-4  # 最小学习率 
        ) 
 
        return { 
            "optimizer": optimizer, 
            "lr_scheduler": { 
                "scheduler": scheduler, 
                "monitor": "val_loss" 
            } 
        } 
 
    # TODO: update parameters 
 
 
# 在 train 函数中检查数据加载 
def train(devices_numbers, save_dir): 
    dataset = read_nii_files("/home/Datasets/bionet/Dataset/lsj_MICCAI_BraTS2020_TrainingData/") 
    images = dataset['images']  # [4, n, 155, 256, 256] 
    labels = dataset['labels']  # [3, n, 155, 256, 256]，确保这里是三通道的 
    logger.info("Datasets loaded successfully") 
 
    # 只使用第n个模态的图像 
    images = images[1]  # [1, n, 155, 256, 256] 
 
    medical_dataset = MRDataset(images, labels) 
    train_size = int(0.8 * len(medical_dataset)) 
    test_size = len(medical_dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(medical_dataset,  [train_size, test_size], generator=torch.Generator().manual_seed(0)) 
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2) 
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2) 
    # 检查数据维度 
    for batch in train_loader: 
        x, y = batch 
        print(f"Input data shape: {x.shape}")  
        print(f"Label data shape: {y.shape}")   # 确保标签是三通道的 
        break 
 
    logger.info("Start  training") 
    # 根据实际情况调整 in_channels 参数 
    model = MainModel() 
    trainer = pl.Trainer( 
        max_epochs=50,  # 增加到50个epoch 
        accelerator='gpu', 
        devices=devices_numbers, 
        precision="16", 
        callbacks=[ 
            pl.callbacks.EarlyStopping(  
                monitor='val_loss', 
                patience=30,  # 如果10个epoch没有改善就停止 
                min_delta=0.001,  # 减小最小改善阈值 
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
 