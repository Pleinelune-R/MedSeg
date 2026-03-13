"""
3D MAE Encoder + Residual Decoder (ResUNet style) Segmentation Models
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from logger import get_logger
from model.config import ModelConfig
from model.pretrain_model.mae_encoder import MAEEncoder3D

# Optimize for Tensor Cores
torch.set_float32_matmul_precision('medium')

logger = get_logger("mae_resunet")


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class FeatureResUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, volume_size=(80, 160, 160), grid_size=None, num_patches=None, dropout_rate=0.0):
        super().__init__()
        self.volume_size = volume_size
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.dropout = nn.Dropout3d(p=dropout_rate)
        
        # Encoder Path
        # Input is 768 channels (MAE features)
        self.enc1 = ResBlock3D(in_channels, 256)
        self.down1 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.enc2 = ResBlock3D(512, 512)
        self.down2 = nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1)
        
        # Bridge
        self.bridge = ResBlock3D(1024, 1024)
        
        # Decoder Path
        self.up2 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(512 + 512, 512) # Concat with enc2
        
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(256 + 256, 256) # Concat with enc1
        
        self.final_conv = nn.Conv3d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, N, E) from MAE Encoder
        
        # Reshape to 3D feature map
        if self.num_patches is not None and x.shape[1] == self.num_patches + 1:
            x = x[:, 1:, :]
        if self.grid_size is not None:
            x = x.transpose(1, 2)
            d, h, w = self.grid_size
            x = x.reshape(x.shape[0], x.shape[1], d, h, w)
            
        # Encoder
        e1 = self.enc1(x) # (B, 256, D, H, W)
        d1 = self.down1(e1) # (B, 512, D/2, H/2, W/2)
        d1 = self.dropout(d1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2) # (B, 1024, D/4, H/4, W/4)
        d2 = self.dropout(d2)
        
        # Bridge
        b = self.bridge(d2)
        b = self.dropout(b)
        
        # Decoder
        u2 = self.up2(b)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = nn.functional.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        c2 = torch.cat([u2, e2], dim=1)
        d2_out = self.dec2(c2)
        
        u1 = self.up1(d2_out)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = nn.functional.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        c1 = torch.cat([u1, e1], dim=1)
        d1_out = self.dec1(c1)
        
        out = self.final_conv(d1_out)
        
        # Final Upsample to original volume size
        if out.shape[2:] != self.volume_size:
            out = nn.functional.interpolate(out, size=self.volume_size, mode='trilinear', align_corners=False)
        
        return out


class DiceLoss3D(nn.Module):
    def __init__(self, smooth=1.0, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, pred, target):
        # Check if binary segmentation (1 channel output)
        if pred.shape[1] == 1:
            pred = torch.sigmoid(pred)
            # target is one-hot encoded: (B, 2, D, H, W) -> take channel 1 (foreground)
            if target.shape[1] == 2:
                target = target[:, 1:2]
            
            pred = pred.float()
            target = target.float()
            intersection = (pred * target).sum(dim=(2,3,4))
            union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
            dice = (2*intersection + self.smooth)/(union + self.smooth)
            return 1.0 - dice.mean()
        else:
            # Multi-class case
            pred = torch.softmax(pred, dim=1)
            if self.ignore_background:
                pred = pred[:, 1:]
                target = target[:, 1:]
            pred = pred.float()
            target = target.float()
            intersection = (pred * target).sum(dim=(2,3,4))
            union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
            dice = (2*intersection + self.smooth)/(union + self.smooth)
            return 1.0 - dice.mean()


class MAECNNSegModel3D(pl.LightningModule):
    def __init__(self, config: ModelConfig, num_classes=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.volume_size = getattr(config, 'volume_size', (80,160,160))
        self.num_classes = num_classes if num_classes is not None else config.num_classes
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.max_epochs = config.max_epochs
        # Encoder
        self.encoder = MAEEncoder3D(
            volume_size=self.volume_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio
        )
        
        # Feature ResUNet Decoder
        # Takes MAE features (embed_dim) as input
        self.decoder = FeatureResUNet3D(
            in_channels=config.embed_dim,
            out_channels=self.num_classes,
            volume_size=self.volume_size,
            grid_size=self.encoder.patch_embed.grid_size,
            num_patches=self.encoder.patch_embed.num_patches,
            dropout_rate=getattr(config, 'dropout', 0.0)
        )
        # Initialize Dice Loss once
        self.dice_loss_fn = DiceLoss3D(ignore_background=True)

    def load_pretrained_encoder(self, ckpt_path):
        logger.info(f"Loading pretrained MAE encoder weights from {ckpt_path}")
        # Fix for PyTorch >=2.6: weights_only=False
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        enc_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                enc_dict[k[len('encoder.'):]] = v
        
        # Handle patch_embed.proj.weight size mismatch for in_chans
        if 'patch_embed.proj.weight' in enc_dict and \
            enc_dict['patch_embed.proj.weight'].shape[1] == 1 and \
            self.encoder.patch_embed.proj.weight.shape[1] > 1:
            logger.info("Detected patch_embed.proj.weight size mismatch. Expanding 1-channel weights to multi-channel.")
            current_in_chans = self.encoder.patch_embed.proj.weight.shape[1]
            original_weight = enc_dict['patch_embed.proj.weight']
            new_weight = torch.zeros(
                original_weight.shape[0],
                current_in_chans,
                *original_weight.shape[2:]
            ).to(original_weight.device)
            new_weight[:, 0:1, :, :, :] = original_weight
            enc_dict['patch_embed.proj.weight'] = new_weight
            
        self.encoder.load_state_dict(enc_dict, strict=False)
        logger.info("✓ Loaded pretrained MAE encoder weights!")

    def forward(self, x):
        # Encoder returns: final latent, mask, ids_restore, skip_connections
        final_latent, _, _, _ = self.encoder(x, mask_ratio=0)
        
        # Pass latent features to the decoder
        logits = self.decoder(final_latent)
        
        return logits

    def compute_loss(self, logits, targets_onehot):
        if logits.shape[1] == 1:
            # Binary case: BCEWithLogits + Dice
            # target: (B, 2, D, H, W) -> take channel 1 (foreground)
            target_fg = targets_onehot[:, 1:2].float()
            
            dice_loss = self.dice_loss_fn(logits, targets_onehot)
            ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, target_fg)
            # Use 0.95 Dice + 0.05 CE to prioritize Dice score
            total_loss = 0.8 * dice_loss + 0.2 * ce_loss
            return total_loss, dice_loss, ce_loss
        else:
            # Multi-class case
            dice_loss = self.dice_loss_fn(logits, targets_onehot)
            ce_loss = nn.functional.cross_entropy(logits, torch.argmax(targets_onehot, dim=1))
            total_loss = 0.8 * dice_loss + 0.2 * ce_loss
            return total_loss, dice_loss, ce_loss

    def on_train_start(self):
        # 保证 encoder 参数可训练
        for param in self.encoder.parameters():
            param.requires_grad = True

    def training_step(self, batch, batch_idx):
        volumes, masks = batch
        logits = self(volumes)
        total_loss, dice_loss, focal_loss = self.compute_loss(logits, masks)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_focal_loss', focal_loss, on_step=True, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        volumes, masks = batch
        logits = self(volumes)
        total_loss, dice_loss, focal_loss = self.compute_loss(logits, masks)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_dice_loss', dice_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_focal_loss', focal_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        volumes, masks = batch
        logits = self(volumes)
        total_loss, dice_loss, focal_loss = self.compute_loss(logits, masks)
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_dice_loss', dice_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_focal_loss', focal_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        # Use same learning rate for encoder and decoder
        encoder_lr = self.learning_rate
        decoder_lr = self.learning_rate
        
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.decoder.parameters(), 'lr': decoder_lr}
        ], weight_decay=self.weight_decay)
        
        # Use CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch",
                "frequency": 1
            }
        }
