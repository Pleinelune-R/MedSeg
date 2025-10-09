from logger import get_logger
from model.config import ModelConfig
from train.mae_pretrain import pretrain_mae
from train.mae_supervised import train_mae_supervised


if __name__ == '__main__':
    logger = get_logger("main")

    print("MAE training mode selection")
    print("=" * 50)
    print("1. MAE unsupervised pretraining (reconstruction)")
    print("2. MAE supervised segmentation training")
    print("=" * 50)

    choice = input("Please select mode (1-2): ").strip()

    dataset_path = "/home/Datasets/bionet/Dataset/lsj_NPC_MR_Dataset_v3"

    if choice == "1":
        logger.info("Starting MAE unsupervised pretraining")
        config = ModelConfig(
            img_size=256,
            crop_ratio=0.3,
            batch_size=16,
            max_epochs=100,
            learning_rate=1.5e-4,
            weight_decay=0.05,
            min_lr=1e-6,
            warmup_epochs=10,
            train_val_split=0.9,
            num_workers=8,
            dataset_path=dataset_path,
            devices_numbers=[0],
            profiler="simple",
            input_size=(50, 256, 256),
            network_size=(64, 128, 128),
            in_channels=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_dim=3072,
            dropout=0.1
        )

        pretrain_mae(
            config,
            mask_ratio=0.75,
            save_dir="./checkpoints_mae"
        )
        logger.info("MAE pretraining completed")

    elif choice == "2":
        logger.info("Starting MAE supervised segmentation training")
        config = ModelConfig(
            img_size=256,
            crop_ratio=0.3,
            batch_size=8,
            max_epochs=100,
            learning_rate=1e-4,
            weight_decay=0.01,
            min_lr=1e-6,
            early_stopping_patience=20,
            train_val_split=0.8,
            num_workers=8,
            dataset_path=dataset_path,
            devices_numbers=[0],
            profiler="simple",
            input_size=(50, 256, 256),
            network_size=(64, 128, 128),
            in_channels=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_dim=3072,
            dropout=0.1
        )

        train_mae_supervised(
            config,
            save_dir="./checkpoints_supervised"
        )
        logger.info("MAE supervised training completed")

    else:
        logger.error("Invalid selection")
        print("Invalid selection, please re-run the program")