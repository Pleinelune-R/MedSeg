import pytorch_lightning as pl
from torch import nn
# maybe can use pl


class ImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO : encoder create
        self.conv_transpose1 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)

    def forward(self, x):
        # TODO : model forward
        y = self.conv_transpose1(x)
        return y