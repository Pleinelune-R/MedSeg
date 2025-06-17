from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep
from torch import nn

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    def __init__(
            self,
            img_size: Sequence[int] | int,
            in_channels: int,
            feature_size: int = 24,
            norm_name: tuple | str = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            spatial_dims: int = 3,
            context=False,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        self.context = context
        self.normalize = normalize

        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=in_channels,
                                        out_channels=feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=feature_size,
                                        out_channels=feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=feature_size,
                                        out_channels=2 * feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=2 * feature_size,
                                        out_channels=4 * feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

    def forward(self, x_in, report_in=None):
        # TODO : model forward
        hidden_states_out = []

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(enc0)
        enc2 = self.encoder3(enc1)
        enc3 = self.encoder4(enc2)

        hidden_states_out.append(enc0)
        hidden_states_out.append(enc1)
        hidden_states_out.append(enc2)
        hidden_states_out.append(enc3)
        
        logger.debug(f"Encoder hidden states shapes:")
        logger.debug(f"enc0: {hidden_states_out[0].shape}")
        logger.debug(f"enc1: {hidden_states_out[1].shape}")
        logger.debug(f"enc2: {hidden_states_out[2].shape}")
        logger.debug(f"enc3: {hidden_states_out[3].shape}")

        return hidden_states_out
