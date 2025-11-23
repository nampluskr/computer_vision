## filename: dcgan.py

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# --------------------------------------------------------------
#  Generator
# --------------------------------------------------------------

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class Generator32(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            DeconvBlock(base*4, base*2),
            DeconvBlock(base*2, base),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return x


class Generator64(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True)
        )  # (N, 512, 4, 4)
        self.blocks = nn.Sequential(
            DeconvBlock(base * 8, base * 4),  # 4x4 → 8x8
            DeconvBlock(base * 4, base * 2),  # 8x8 → 16x16
            DeconvBlock(base * 2, base),      # 16x16 → 32x32
            DeconvBlock(base, base)           # 32x32 → 64x64
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(self.init_weights)
        # if use_spectral_norm:
        #     self.apply(apply_spectral_norm)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return x


# --------------------------------------------------------------
#  Discriminator
# --------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator32(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
        )
        self.final = nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return x.view(-1, 1)


class Discriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 64x64 → 32x32
        self.blocks = nn.Sequential(
            ConvBlock(base, base * 2),        # 32x32 → 16x16
            ConvBlock(base * 2, base * 4),    # 16x16 → 8x8
            ConvBlock(base * 4, base * 8)     # 8x8 → 4x4
        )
        self.final = nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)  # 4x4 → 1x1

        self.apply(self.init_weights)
        if use_spectral_norm:
            self.apply(apply_spectral_norm)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return x.view(-1, 1)
