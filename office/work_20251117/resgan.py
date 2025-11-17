import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip = self.skip(self.upsample(x))
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out + skip


class ResGenerator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(base * 4),
            # nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            ResBlockUp(base * 4, base * 2),
            ResBlockUp(base * 2, base),
            ResBlockUp(base, base // 2),
        )
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base // 2, out_channels, kernel_size=3, stride=1, padding=1),
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


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        skip = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.downsample(out)
        return out + skip


def apply_spectral_norm(m):
    if isinstance(m, nn.Conv2d):
        return spectral_norm(m)
    return m


class ResDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.initial = ResBlockDown(in_channels, base // 2)
        self.blocks = nn.Sequential(
            ResBlockDown(base // 2, base),
            ResBlockDown(base, base * 2),
        )
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 2, 1),
        )
        if use_spectral_norm:
            self.apply(apply_spectral_norm)

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
        return x
