import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

from cifar10 import CIFAR10, get_train_loader, get_test_loader
from trainer import fit
from utils import create_images, plot_images, plot_history, set_seed


# --------------------------------------------------------------
#  Self-Attention Module
# --------------------------------------------------------------

class SelfAttention(nn.Module):
    """Self-Attention layer for SAGAN"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 1x1 convolutions for query, key, value
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: (N, C, H, W)
        Returns:
            out: (N, C, H, W)
        """
        batch_size, C, H, W = x.size()

        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, H * W)  # (N, C//8, H*W)
        key = self.key(x).view(batch_size, -1, H * W)      # (N, C//8, H*W)
        value = self.value(x).view(batch_size, -1, H * W)  # (N, C, H*W)

        # Attention map: softmax(Q^T K)
        attention = torch.bmm(query.permute(0, 2, 1), key)  # (N, H*W, H*W)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (N, C, H*W)
        out = out.view(batch_size, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


# --------------------------------------------------------------
#  Spectral Normalization
# --------------------------------------------------------------

def spectral_norm(module, mode=True):
    """Apply spectral normalization to a module"""
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# --------------------------------------------------------------
#  Generator
# --------------------------------------------------------------

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm=False):
        super().__init__()
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.block = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class SAGenerator32(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial projection
        conv = nn.ConvTranspose2d(latent_dim, base*4, kernel_size=4, stride=1, padding=0, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.initial = nn.Sequential(
            conv,
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),
        )  # (N, 256, 4, 4)

        self.block1 = DeconvBlock(base*4, base*2, use_spectral_norm)  # (N, 128, 8, 8)
        self.attention = SelfAttention(base*2)  # Self-attention at 8x8 resolution
        self.block2 = DeconvBlock(base*2, base, use_spectral_norm)    # (N, 64, 16, 16)

        final_conv = nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)

        self.final = nn.Sequential(
            final_conv,
            nn.Tanh(),
        )  # (N, 3, 32, 32)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.initial(z)
        x = self.block1(x)
        x = self.attention(x)  # Apply self-attention
        x = self.block2(x)
        x = self.final(x)
        return x


class SAGenerator64(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim

        conv = nn.ConvTranspose2d(latent_dim, base * 8, kernel_size=4, stride=1, padding=0, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.initial = nn.Sequential(
            conv,
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True)
        )  # (N, 512, 4, 4)

        self.block1 = DeconvBlock(base * 8, base * 4, use_spectral_norm)  # (N, 256, 8, 8)
        self.attention1 = SelfAttention(base * 4)  # Self-attention at 8x8
        self.block2 = DeconvBlock(base * 4, base * 2, use_spectral_norm)  # (N, 128, 16, 16)
        self.attention2 = SelfAttention(base * 2)  # Self-attention at 16x16
        self.block3 = DeconvBlock(base * 2, base, use_spectral_norm)      # (N, 64, 32, 32)

        final_conv = nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)

        self.final = nn.Sequential(
            final_conv,
            nn.Tanh()
        )  # (N, 3, 64, 64)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.initial(z)
        x = self.block1(x)
        x = self.attention1(x)
        x = self.block2(x)
        x = self.attention2(x)
        x = self.block3(x)
        x = self.final(x)
        return x


# --------------------------------------------------------------
#  Discriminator
# --------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm=False):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.block = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SADiscriminator32(nn.Module):
    def __init__(self, in_channels=3, base=64, use_spectral_norm=True):
        super().__init__()

        conv = nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.initial = nn.Sequential(
            conv,
            nn.LeakyReLU(0.2, inplace=True),
        )  # (N, 64, 16, 16)

        self.block1 = ConvBlock(base, base*2, use_spectral_norm)  # (N, 128, 8, 8)
        self.attention = SelfAttention(base*2)  # Self-attention at 8x8
        self.block2 = ConvBlock(base*2, base*4, use_spectral_norm)  # (N, 256, 4, 4)

        final_conv = nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)

        self.final = final_conv  # (N, 1, 1, 1)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.attention(x)  # Apply self-attention
        x = self.block2(x)
        logits = self.final(x).view(-1, 1)
        return logits


class SADiscriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64, use_spectral_norm=True):
        super().__init__()

        conv = nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.initial = nn.Sequential(
            conv,
            nn.LeakyReLU(0.2, inplace=True)
        )  # (N, 64, 32, 32)

        self.block1 = ConvBlock(base, base * 2, use_spectral_norm)        # (N, 128, 16, 16)
        self.attention1 = SelfAttention(base * 2)  # Self-attention at 16x16
        self.block2 = ConvBlock(base * 2, base * 4, use_spectral_norm)    # (N, 256, 8, 8)
        self.attention2 = SelfAttention(base * 4)  # Self-attention at 8x8
        self.block3 = ConvBlock(base * 4, base * 8, use_spectral_norm)    # (N, 512, 4, 4)

        final_conv = nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)

        self.final = final_conv  # (N, 1, 1, 1)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.attention1(x)
        x = self.block2(x)
        x = self.attention2(x)
        x = self.block3(x)
        logits = self.final(x).view(-1, 1)
        return logits


# --------------------------------------------------------------
#  SAGAN with Hinge Loss
# --------------------------------------------------------------

class SAGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = discriminator.to(self.device)
        self.g_model = generator.to(self.device)

        # SAGAN typically uses different learning rates and TTUR (Two Time-Scale Update Rule)
        self.d_optimizer = optim.Adam(self.d_model.parameters(), lr=4e-4, betas=(0.0, 0.9))
        self.g_optimizer = optim.Adam(self.g_model.parameters(), lr=1e-4, betas=(0.0, 0.9))

        self.latent_dim = latent_dim or generator.latent_dim

    def d_hinge_loss(self, real_logits, fake_logits):
        """Hinge loss for discriminator"""
        d_real_loss = torch.mean(F.relu(1.0 - real_logits))
        d_fake_loss = torch.mean(F.relu(1.0 + fake_logits))
        return d_real_loss, d_fake_loss

    def g_hinge_loss(self, fake_logits):
        """Hinge loss for generator"""
        return -torch.mean(fake_logits)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        batch_size = images.size(0)

        # (1) Update Discriminator
        real_logits = self.d_model(images)

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises).detach()
        fake_logits = self.d_model(fake_images)

        d_real_loss, d_fake_loss = self.d_hinge_loss(real_logits, fake_logits)
        d_loss = d_real_loss + d_fake_loss

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises)
        fake_logits = self.d_model(fake_images)

        g_loss = self.g_hinge_loss(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.detach().cpu().item(),
            real_loss=d_real_loss.detach().cpu().item(),
            fake_loss=d_fake_loss.detach().cpu().item(),
            g_loss=g_loss.detach().cpu().item(),
        )


if __name__ == "__main__":

    set_seed(42)

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
    train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train", transform=transform), batch_size=64)

    latent_dim = 100

    # SAGAN with spectral normalization
    discriminator = SADiscriminator32(in_channels=3, base=64, use_spectral_norm=True)
    generator = SAGenerator32(latent_dim=latent_dim, out_channels=3, base=64, use_spectral_norm=False)
    gan = SAGAN(discriminator, generator, latent_dim=latent_dim)

    # Fixed noise for visualization
    noises = np.random.normal(size=(50, latent_dim, 1, 1))

    filename = os.path.splitext(os.path.basename(__file__))[0]
    total_history = {}
    epoch, num_epochs = 0, 5

    for _ in range(4):
        history = fit(gan, train_loader, num_epochs=num_epochs)
        epoch += num_epochs
        for split_name, metrics in history.items():
            total_history.setdefault(split_name, {})
            for metric_name, metric_values in metrics.items():
                total_history[split_name].setdefault(metric_name, [])
                total_history[split_name][metric_name].extend(metric_values)

        images = create_images(generator, noises)
        image_path = f"./outputs/{filename}-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)
