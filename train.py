"""
Main script for 3D MAE Supervised Segmentation Training
"""
import argparse
from model.config import ModelConfig
from train.mae_cnnseg import train_cnnseg_3d
from logger import get_logger

logger = get_logger("train")


def main():
    parser = argparse.ArgumentParser(description='3D MAE Supervised Segmentation Training')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed 3D segmentation H5 data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_supervised_3d',
                       help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--volume_size', type=int, nargs=3, default=[48, 256, 256],
                       help='Input volume size (default: 48 256 256)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size (default: 16)')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of segmentation classes (default: 4)')
    parser.add_argument('--embed_dim', type=int, default=768,
                       help='Embedding dimension (default: 768)')
    # ... other args ...

    # Training parameters
    parser.add_argument('--devices_numbers', type=int, nargs='+', default=[0],
                       help='GPU device numbers (default: [0])')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained MAE encoder checkpoint')
    parser.add_argument('--patience', type=int, default=25,
                       help='Early stopping patience (default: 25)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate (default: 0.0)')


    args = parser.parse_args()
    
    config = ModelConfig(
        dataset_path=args.data_path,
        volume_size=tuple(args.volume_size),
        devices_numbers=args.devices_numbers,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        early_stopping_patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )
    
    try:
        trainer = train_cnnseg_3d(
            config=config,
            save_dir=args.save_dir,
            pretrained_path=args.pretrained_path
        )
        logger.info("Supervised training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
