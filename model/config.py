"""
Model Configuration
Shared configuration class for both MAE pretraining and supervised training
"""


class ModelConfig:
    """
    Configuration class for model parameters
    """

    def __init__(self,
                 # Image parameters
                 input_size=(128, 256, 256),
                 network_size=(64, 128, 128),
                 in_channels=4,

                 # Training parameters
                 batch_size=16,
                 max_epochs=30,
                 learning_rate=5e-4,
                 weight_decay=1e-5,
                 min_lr=5e-7,

                 # Early stopping parameters
                 early_stopping_patience=25,
                 early_stopping_min_delta=0.0005,

                 # Data parameters
                 train_val_split=0.8,
                 num_workers=8,
                 dataset_path=None,
                 devices_numbers=None,
                 strategy="auto", 
                 profiler="simple",
                 
                 # Pre-trained ViT parameters
                 use_pretrained=False,
                 pretrained_model="vit_base_patch16_224",
                 freeze_pretrained=False,
                 multi_scale=False,
                 
                 # MAE pre-training parameters
                 img_size=256,
                 crop_ratio=0.3,
                 warmup_epochs=10,
                 
                 # Model architecture parameters
                 embed_dim=96,
                 patch_size=16,
                 depth=6,
                 num_heads=8,
                 ffn_dim=384,
                 dropout=0.1
                 ):
        # Image parameters
        self.input_size = input_size
        self.network_size = network_size
        self.in_channels = in_channels

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
        self.strategy = strategy
        self.profiler = profiler
        
        # Pre-trained ViT parameters
        self.use_pretrained = use_pretrained
        self.pretrained_model = pretrained_model
        self.freeze_pretrained = freeze_pretrained
        self.multi_scale = multi_scale
        
        # MAE pre-training parameters
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.warmup_epochs = warmup_epochs
        
        # Model architecture parameters
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

