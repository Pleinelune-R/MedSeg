"""
Model Configuration
Shared configuration class for both MAE pretraining and supervised training
"""


class ModelConfig:
    """
    Configuration class for model parameters
    """

    def __init__(self,
                     # Training parameters
                     batch_size=4,
                     max_epochs=100,
                     learning_rate=5e-4,
                     weight_decay=1e-5,
                     min_lr=5e-7,

                     # Early stopping parameters
                     early_stopping_patience=50,
                     early_stopping_min_delta=0.0005,

                     # Data parameters
                     train_val_split=0.8,
                     num_workers=8,
                     dataset_path=None,
                     devices_numbers=None,

                     # MAE pre-training parameters
                     volume_size=(48, 256, 256),  # For 3D MAE
                     crop_ratio=1.0,
                     warmup_epochs=10,

                     # Encoder architecture parameters
                     embed_dim=384,
                     patch_size=16,
                     depth=12,
                     num_heads=12,
                     mlp_ratio=4.0,
                     dropout=0,

                     # Decoder architecture parameters (for MAE)
                     decoder_embed_dim=96,  # must be divisible by 3
                     decoder_depth=8,
                     decoder_num_heads=4,

                     # Medical image specific
                     in_chans=1,  # 1 for grayscale, 4 for multi-modal
                     num_classes=1,  # for segmentation (binary)
                     ):
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
        self.dataset_path = dataset_path
        self.devices_numbers = devices_numbers
        
        # MAE pre-training parameters
        self.volume_size = volume_size
        self.crop_ratio = crop_ratio
        self.warmup_epochs = warmup_epochs
        
        # Encoder architecture parameters
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Decoder architecture parameters (for MAE)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        
        # Medical image specific
        self.in_chans = in_chans
        self.num_classes = num_classes
