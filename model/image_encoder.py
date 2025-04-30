import pytorch_lightning as pl  # maybe can use pl
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO : encoder create
        self.conv1  = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        # TODO : model forward
        x = self.conv1(x)
        x = self.relu(x)
        y = self.pool(x)
        return y
