"""
Main script for 3D MAE Pretraining
"""
import argparse
from model.config import ModelConfig
from train.mae_pretrain import pretrain_mae
from logger import get_logger

logger = get_logger("pretrain")


def main():
    parser = argparse.ArgumentParser(description='3D MAE Pretraining')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None, help='Path to preprocessed 3D medical volume H5 data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_mae_3d',
                       help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--volume_size', type=int, nargs=3, default=[48, 256, 256],
                       help='Input volume size (default: 48 256 256)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=384,
                       help='Embedding dimension (default: 768, must be divisible by 3)')
    parser.add_argument('--depth', type=int, default=12,
                       help='Number of transformer blocks (default: 12)')
    parser.add_argument('--num_heads', type=int, help='Encoder num_heads')
    parser.add_argument('--decoder_embed_dim', type=int, help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', type=int, default=8,
                       help='Decoder depth (default: 8)')
    parser.add_argument('--decoder_num_heads', type=int, help='Decoder num_heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                       help='MLP ratio (default: 4.0)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=200,
                       help='Maximum training epochs (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay (default: 0.05)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Warmup epochs (default: 10)')
    
    # MAE specific parameters
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Masking ratio (default: 0.75)')
    parser.add_argument('--crop_ratio', type=float, default=0.4, help='Center crop ratio (default: 0.4)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (default: 0.0)')
    parser.add_argument('--norm_pix_loss', action='store_true',
                       help='Use normalized pixel loss')
    
    # Data loading parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--train_val_split', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')
    
    # Device parameters
    parser.add_argument('--profiler', type=str, default='simple', help='Profiler type (default: simple)')
    parser.add_argument('--devices_numbers', type=int, nargs='+', default=[0], help='GPU device IDs (default: [0])')
    
    args = parser.parse_args()
    
    # Separate args for ModelConfig from other script args
    config_args = {}
    trainer_args = {}
    
    # Get all possible arg names from ModelConfig's __init__
    import inspect
    sig = inspect.signature(ModelConfig.__init__)
    model_config_keys = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]

    for key, value in vars(args).items():
        if value is not None:
            # Map command-line names to ModelConfig names
            if key == 'data_path':
                config_args['dataset_path'] = value
            elif key in model_config_keys:
                config_args[key] = value
            else:
                # These are args for the trainer function, not the config object
                trainer_args[key] = value

    # Create config with only valid arguments
    config = ModelConfig(**config_args)

    # Handle defaults for paths if not provided
    if config.dataset_path is None:
        config.dataset_path = './data/preprocessed_pretrain'
        logger.warning(f"--data_path not provided, using default: {config.dataset_path}")
    
    save_dir = trainer_args.get('save_dir', './checkpoints_mae_3d')
    mask_ratio = trainer_args.get('mask_ratio', 0.75) # Default if not passed
    norm_pix_loss = trainer_args.get('norm_pix_loss', False)

    # Start pretraining
    try:
        trainer = pretrain_mae(
            config=config,
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix_loss,
            save_dir=save_dir
        )
        logger.info("Pretraining completed successfully!")
    except Exception as e:
        logger.error(f"Pretraining failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
