# filename: resgan.py

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def apply_spectral_norm(m):
    if isinstance(m, nn.Conv2d):
        return spectral_norm(m)
    return m


# --------------------------------------------------------------
#  Generator
# --------------------------------------------------------------

class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ResGenerator32(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True)
        )
        self.blocks = nn.Sequential(
            ResBlockUp(base * 4, base * 2),
            ResBlockUp(base * 2, base),    
            ResBlockUp(base, base)         
        )
        self.final = nn.Sequential(
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.Conv2d(base, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(self.init_weights)
        if use_spectral_norm:
            self.apply(apply_spectral_norm)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # nn.init.orthogonal_(m.weight)
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


class ResGenerator64(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True)
        )  # (N, 512, 4, 4)
        self.blocks = nn.Sequential(
            ResBlockUp(base * 8, base * 4),  # 4x4 → 8x8
            ResBlockUp(base * 4, base * 2),  # 8x8 → 16x16
            ResBlockUp(base * 2, base),      # 16x16 → 32x32
            ResBlockUp(base, base)           # 32x32 → 64x64
        )
        self.final = nn.Sequential(
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.Conv2d(base, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(self.init_weights)
        if use_spectral_norm:
            self.apply(apply_spectral_norm)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # nn.init.orthogonal_(m.weight)
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

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ResDiscriminator32(nn.Module):
    def __init__(self, in_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.blocks = nn.Sequential(
            ResBlockDown(base, base * 2),
            ResBlockDown(base * 2, base * 4),
            ResBlockDown(base * 4, base * 4)
        )
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base * 4, 1, kernel_size=4, stride=1, padding=0, bias=False)  # 4x4 → 1x1
        )
        self.apply(self.init_weights)
        if use_spectral_norm:
            self.apply(apply_spectral_norm)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # nn.init.orthogonal_(m.weight)
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return x.view(-1, 1)


class ResDiscriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False)
        )  # 64x64 → 64x64
        self.blocks = nn.Sequential(
            ResBlockDown(base, base * 2),      # 64x64 → 32x32
            ResBlockDown(base * 2, base * 4),  # 32x32 → 16x16
            ResBlockDown(base * 4, base * 8),  # 16x16 → 8x8
            ResBlockDown(base * 8, base * 8)   # 8x8 → 4x4
        )
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)  # 4x4 → 1x1
        )
        self.apply(self.init_weights)
        if use_spectral_norm:
            self.apply(apply_spectral_norm)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # nn.init.orthogonal_(m.weight)
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return x.view(-1, 1)
