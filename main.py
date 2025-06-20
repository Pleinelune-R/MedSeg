from logger import get_logger
from model.main_model import train, ModelConfig

if __name__ == '__main__':
    logger = get_logger("main")
    config = ModelConfig(                
                 input_size=(128, 256, 256),
                 network_size=(64, 128, 128),
                 in_channels=4,

                 # Training parameters
                 batch_size=8,
                 max_epochs=30,
                 learning_rate=5e-4,
                 weight_decay=1e-5,
                 min_lr=5e-7,

                 # Early stopping parameters
                 early_stopping_patience=25,
                 early_stopping_min_delta=0.0005,

                 # Data parameters
                 train_val_split=0.8,
                 num_workers=16,
                 
                 dataset_path = "./data/lsj_MICCAI_BraTS2020_TrainingData/",
                 devices_numbers = [0,2],
                 strategy="ddp"
                 )
    
    trainer = train(config)
    logger.info("Training completed successfully")


