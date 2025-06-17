import pytorch_lightning as pl 
import torch 
import torch.nn  as nn 
from typing import Union 
from collections.abc  import Sequence 
import logging
logger = logging.getLogger(__name__)
 
from monai.networks.blocks.dynunet_block  import UnetBasicBlock, UnetResBlock, get_conv_layer 
 
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
        feature_size: int = 24, 
        norm_name: tuple | str = "instance", 
        spatial_dims: int = 3, 
        args=None, 
        context=False  
    ) -> None: 
        super().__init__() 
        self.context = context
        
        # Calculate the actual input channels for each decoder block
        add_ch = args.n_prompts if args.align_score else 0 if self.context else 0
        
        # Base channel number
        base_channels = 24
        
        # decoder 
        self.decoder3 = ContextUnetrUpBlock(
            spatial_dims=spatial_dims, 
            in_channels=base_channels * 4,   # 96 = 24 * 4
            out_channels=base_channels * 2,  # 48 = 24 * 2
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            add_channels=0
        ) 
        
        self.decoder2 = ContextUnetrUpBlock(
            spatial_dims=spatial_dims, 
            in_channels=base_channels * 2,   # 48 = 24 * 2
            out_channels=base_channels,      # 24 = 24 * 1
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            add_channels=0
        ) 
 
        self.decoder1 = ContextUnetrUpBlock(
            spatial_dims=spatial_dims, 
            in_channels=base_channels,       # 24 = 24 * 1
            out_channels=base_channels,      # 24 = 24 * 1
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            add_channels=0
        ) 
 
        self.out = nn.Conv3d(base_channels, 4, kernel_size=1) 


    def forward(self, hidden_states_out): 
        # visual decoder 
        dec1 = self.decoder3(hidden_states_out[3], hidden_states_out[2]) 
        dec0 = self.decoder2(dec1, hidden_states_out[1]) 
        out = self.decoder1(dec0, hidden_states_out[0]) 
        out = torch.nn.functional.interpolate(out, size=(out.shape[2]*2, out.shape[3]*2, out.shape[4]*2), mode='trilinear', align_corners=True)
        logger.debug(f"Decoder hidden states shapes:")
        logger.debug(f"dec1: {dec1.shape}")
        logger.debug(f"dec0: {dec0.shape}")
        logger.debug(f"out: {out.shape}")
        logits = self.out(out)  
        return logits 