import torch
from torch import nn
from torch.nn.utils import spectral_norm

class Conv2dReLU(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect', batch_norm=True):
        super().__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode
        )))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.LeakyReLU(0.2))  # LeakyReLU for better gradient flow
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
