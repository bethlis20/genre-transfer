import torch
from torch import nn
from models.blocks import Conv2dReLU

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_0 = Conv2dReLU(1, 64)
        self.down_1 = Conv2dReLU(64, 64)
        self.maxpool_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2 = Conv2dReLU(64, 128)
        self.down_3 = Conv2dReLU(128, 128)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_4 = Conv2dReLU(128, 256)
        self.down_5 = Conv2dReLU(256, 256)

        self.deconv_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_0 = Conv2dReLU(256, 128)
        self.up_1 = Conv2dReLU(128, 128)

        self.deconv_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_2 = Conv2dReLU(128, 64)
        self.up_3 = Conv2dReLU(64, 64)

        # Output layer
        self.out_layer = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.down_0(x)
        x1 = self.down_1(x)
        x = self.maxpool_0(x1)

        x = self.down_2(x)
        x2 = self.down_3(x)
        x = self.maxpool_1(x2)

        x = self.down_4(x)
        x = self.down_5(x)

        x = self.deconv_0(x)
        x = torch.cat([x2, x], dim=1)  

        x = self.up_0(x)
        x = self.up_1(x)

        x = self.deconv_1(x)
        x = torch.cat([x1, x], dim=1)  

        x = self.up_2(x)
        x = self.up_3(x)

        x = self.out_layer(x)

        return x
