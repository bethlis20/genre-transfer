import torch
from torch import nn
from models.blocks import Conv2dReLU

class Embedder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.conv_1 = Conv2dReLU(1, 512, kernel_size=(80, 3))
        self.conv_2 = Conv2dReLU(512, 512, kernel_size=(1, 9), stride=(1, 2))
        self.conv_3 = Conv2dReLU(512, 512, kernel_size=(1, 7), stride=(1, 2))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(103936, latent_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
