import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from pytorch_lightning.loggers import TensorBoardLogger  # 显式导入 Logger

from data_prepare.logger import MyLogger
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

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch
        pass

    def validation_step(self, batch, batch_idx):
        # TODO：model forward
        x, y = batch

        # TODO：log
        # logger.info('val_loss', loss, prog_bar=True)
        # logger.info('lr', self.optimizers().defaults['lr'])
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        # TODO：model forward
        pass
        # TODO：add results into test_results
        pass

    def on_test_epoch_end(self):
        # TODO：save batch results  and calculate AUC or other metric
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"}
        }


# TODO: update parameters
def train(devices_numbers, save_dir):
    # devices_number: how many cards to train
    # save_dir: checkpoint dir to save,
    # for example, to use this:
    #     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #     trainer.test(model, dataloaders=test_loader)

    # TODO: init dataset and dataloader
    pass

    # TODO: init model
    model = MainModel()

    # TODO: init trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=devices_numbers,
        precision="bf16-mixed",  # 32-true or https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision
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

    return trainer
