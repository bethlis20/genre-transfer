import torch
from torch import nn
from models.blocks import Conv2dReLU

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = Conv2dReLU(1, 512, kernel_size=(80, 3), batch_norm=False)
        self.conv_2 = Conv2dReLU(512, 512, kernel_size=(1, 9), stride=(1, 2), batch_norm=False)
        self.conv_3 = Conv2dReLU(512, 512, kernel_size=(1, 7), stride=(1, 2), batch_norm=False)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(333312, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
