import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from pytorch_lightning.loggers import TensorBoardLogger  # 显式导入 Logger

from data_prepare.data_iter import MRDataset
from data_prepare.load_file import generate_dataset
from logger import MyLogger
from .image_deocder import ImageDecoder
from .image_encoder import ImageEncoder

logger = MyLogger("model")


class MainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.test_results = []  # Store results for saving

        # module init
        self.image_encoder = ImageEncoder()
        self.image_decoder = ImageDecoder()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.image_decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_results.append(loss.item())
        self.log('test_loss', loss)

    def on_test_epoch_end(self):
        # TODO：save batch results  and calculate AUC or other metric
        avg_test_loss = sum(self.test_results) / len(self.test_results)
        self.log('avg_test_loss', avg_test_loss)
        print(f"Average Test Loss: {avg_test_loss}")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # 这里假设 lr 为 0.001，你可以根据需要修改
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


def train(devices_numbers, save_dir):
    dataset = generate_dataset(".\\data", ".\\data\\output")
    images = [sample['image'] for sample in dataset]
    labels = [sample['label'] for sample in dataset]
    medical_dataset = MRDataset(images, labels)
    train_size = int(0.8 * len(medical_dataset))
    test_size = len(medical_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(medical_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    model = MainModel()
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=devices_numbers,
        precision="bf16-mixed",
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