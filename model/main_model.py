import pytorch_lightning as pl
import torch
from torch.utils.data  import DataLoader
from pytorch_lightning.utilities.types  import OptimizerLRScheduler
from pytorch_lightning.loggers  import TensorBoardLogger  # 显式导入 Logger
import matplotlib.pyplot  as plt

from data_prepare.data_iter  import MRDataset
from data_prepare.load_file  import generate_dataset
from logger import MyLogger
from .image_deocder import ImageDecoder
from .image_encoder import ImageEncoder

logger = MyLogger("model")


class MainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.test_results  = []  # Store results for saving
        self.test_images  = []
        self.test_true_masks  = []
        self.test_pred_masks  = []
        
        # module init
        self.image_encoder  = ImageEncoder(img_size=(256, 256), in_channels=4, spatial_dims=2)
        args = type('Args', (), {'align_score': False, 'n_prompts': 0})
        self.image_decoder  = ImageDecoder(args = args)
        self.criterion  = torch.nn.MSELoss()

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.image_decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat,  y)
        self.log('train_loss',  loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat,  y)
        self.log('val_loss',  loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat,  y)
        self.test_results.append(loss.item())
        self.log('test_loss',  loss)

        # Store images, true masks, and predicted masks
        self.test_images.extend(x.cpu().numpy())
        self.test_true_masks.extend(y.cpu().numpy())
        self.test_pred_masks.extend(y_hat.cpu().numpy())

    def on_test_epoch_end(self):
        # TODO：save batch results  and calculate AUC or other metric
        avg_test_loss = sum(self.test_results)  / len(self.test_results)
        self.log('avg_test_loss',  avg_test_loss)
        print(f"Average Test Loss: {avg_test_loss}")

        # Visualize images with masks
        self.visualize_masks()

    def visualize_masks(self):
        num_samples = min(5, len(self.test_images))   # Visualize up to 5 samples
        fig, axes = plt.subplots(num_samples,  3, figsize=(15, 5 * num_samples))

        for i in range(num_samples):
            image = self.test_images[i].squeeze()
            true_mask = self.test_true_masks[i].squeeze()
            pred_mask = self.test_pred_masks[i].squeeze()

            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(image, cmap='gray')
            axes[i, 1].imshow(true_mask, alpha=0.5, cmap='jet')
            axes[i, 1].set_title('Original Image + True Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(image, cmap='gray')
            axes[i, 2].imshow(pred_mask, alpha=0.5, cmap='jet')
            axes[i, 2].set_title('Original Image + Predicted Mask')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(),  lr=0.001)  # 这里假设 lr 为 0.001，你可以根据需要修改
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

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
    dataset = generate_dataset(".\\data", ".\\data\\output")
    images = [sample['image'] for sample in dataset]
    labels = [sample['label'] for sample in dataset]
    medical_dataset = MRDataset(images, labels)
    train_size = int(0.8 * len(medical_dataset))
    test_size = len(medical_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(medical_dataset, [train_size, test_size])
    # 增加 num_workers 参数
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=15)

    # 检查数据维度
    for batch in train_loader:
        x, y = batch
        print(f"Input data shape: {x.shape}")  # 检查输入数据的形状
        break

        # 根据实际情况调整 in_channels 参数
    model = MainModel()
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='auto',
        devices=devices_numbers,
        precision="16-mixed",
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                min_delta=0.01,
                mode='min'
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='MED-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir, name='test')
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader)
    return trainer