import pytorch_lightning as pl 
import torch 
import torch.nn  as nn 
from typing import Union 
from collections.abc  import Sequence 
import logging
logger = logging.getLogger(__name__)
 
from monai.networks.blocks.dynunet_block  import UnetBasicBlock, UnetResBlock, get_conv_layer 
from .vit_moe import ViT_Decoder

# maybe can use pl 
 
class ContextUnetrUpBlock(nn.Module): 
    """ 
    An upsampling module that can be used for UNETR: "Hatamizadeh et al., 
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"  
    """ 
 
    def __init__( 
        self, 
        spatial_dims: int, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[Sequence[int], int], 
        upsample_kernel_size: Union[Sequence[int], int], 
        norm_name: tuple | str, 
        res_block: bool = False, 
        add_channels: int = 0, 
    ) -> None: 
        """ 
        Args: 
            spatial_dims: number of spatial dimensions. 
            in_channels: number of input channels. 
            out_channels: number of output channels. 
            kernel_size: convolution kernel size. 
            upsample_kernel_size: convolution kernel size for transposed convolution layers. 
            norm_name: feature normalization type and arguments. 
            res_block: bool argument to determine if residual block is used. 
 
        """ 
 
        super().__init__() 
        upsample_stride = upsample_kernel_size 
        self.transp_conv  = get_conv_layer( 
            spatial_dims, 
            in_channels + add_channels, 
            out_channels, 
            kernel_size=upsample_kernel_size, 
            stride=upsample_stride, 
            conv_only=True, 
            is_transposed=True, 
        ) 
 
        if res_block: 
            self.conv_block  = UnetResBlock( 
                spatial_dims, 
                out_channels + out_channels + add_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=1, 
                norm_name=norm_name, 
            ) 
        else: 
            self.conv_block  = UnetBasicBlock(  # type: ignore 
                spatial_dims, 
                out_channels + out_channels + add_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=1, 
                norm_name=norm_name, 
            ) 
 
    def forward(self, inp, skip): 
        # number of channels for skip should equals to out_channels 
        out = self.transp_conv(inp)  
        out = torch.cat((out,  skip), dim=1) 
        out = self.conv_block(out)  
        return out 

class ImageDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 384,
        embed_dim: int = 96,
        patch_size: int = 16,
        img_size=(64, 128, 128),
        depth: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 384,
        out_channels: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.vit_decoder = ViT_Decoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            out_channels=out_channels,
            dropout=dropout
        )

    def forward(self, x):
        # x: (B, 4, C, d, h, w)
        B, M, C, d, h, w = x.shape
        x = x.reshape(B, M*C, d, h, w)  # (B, 4*C, d, h, w)
        out = self.vit_decoder(x)  # (B, 4, D, H, W)
        return out 
