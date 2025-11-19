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
# from ac_dcgan import ACDiscriminator32, CGenerator32


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


class CGenerator32(nn.Module):
    def __init__(self, num_classes, latent_dim=100, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.labels_embedding = nn.Embedding(num_classes, num_classes)

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, base*4, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, noises, labels):
        labels_embedding = self.labels_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([noises, labels_embedding], dim=1)
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return x


class CGenerator64(nn.Module):
    def __init__(self, num_classes, latent_dim=100, out_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.labels_embedding = nn.Embedding(num_classes, num_classes)

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, base * 8, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, noises, labels):
        labels = self.labels_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([noises, labels], dim=1)
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


class ACDiscriminator32(nn.Module):
    def __init__(self, num_classes, in_channels=3, base=64):
        super().__init__()
        self.num_classes = num_classes
        self.labels_embedding = nn.Sequential(
            nn.Embedding(num_classes, num_classes),
            nn.Linear(num_classes, 32 * 32),
            nn.Unflatten(dim=1, unflattened_size=(1, 32, 32))
        )
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + 1, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
        )
        self.final = nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.aux = nn.Conv2d(base*4, num_classes, kernel_size=4, stride=1, padding=0, bias=False)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, images, labels):
        labels = self.labels_embedding(labels)
        x = torch.cat([images, labels], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        aux = self.aux(x).view(-1, self.num_classes)
        return logits, aux


class ACDiscriminator64(nn.Module):
    def __init__(self, num_classes, in_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.num_classes = num_classes
        self.labels_embedding = nn.Sequential(
            nn.Embedding(num_classes, num_classes),
            nn.Linear(num_classes, 64 * 64),
            nn.Unflatten(dim=1, unflattened_size=(1, 64, 64))
        )
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + 1, base, kernel_size=4, stride=2, padding=1, bias=False),
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

    def forward(self, images, labels):
        labels = self.labels_embedding(labels)
        x = torch.cat([images, labels], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        aux = self.aux(x).view(-1, self.num_classes)
        return logits, aux

###########################################################################################
###########################################################################################

class ACGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = discriminator.to(self.device)
        self.g_model = generator.to(self.device)

        self.d_optimizer = optim.Adam(self.d_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.g_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim

    def d_loss_fn(self, real_logits, fake_logits):
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        d_real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        d_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        return d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        real_labels = torch.ones_like(fake_logits)
        return F.binary_cross_entropy_with_logits(fake_logits, real_labels)

    def d_aux_loss_fn(self, real_aux, fake_aux, labels):
        d_real_aux_loss = F.cross_entropy(real_aux, labels)
        d_fake_aux_loss = F.cross_entropy(fake_aux, labels)
        return d_real_aux_loss, d_fake_aux_loss

    def g_aux_loss_fn(self, fake_aux, labels):
        return F.cross_entropy(fake_aux, labels)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        batch_size = images.size(0)

        # (1) Update Discriminator
        real_logits, real_aux = self.d_model(images, labels)
        
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises, labels).detach()
        fake_logits, fake_aux = self.d_model(fake_images, labels)

        d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)
        d_real_aux_loss, d_fake_aux_loss = self.d_aux_loss_fn(real_aux, fake_aux, labels)
        d_loss = d_real_loss + d_fake_loss + d_real_aux_loss + d_fake_aux_loss

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises, labels)
        fake_logits, fake_aux = self.d_model(fake_images, labels)
        
        g_loss = self.g_loss_fn(fake_logits) + self.g_aux_loss_fn(fake_aux, labels)

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
    train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train", transform=transform), batch_size=128)

    num_classes = 10
    discriminator = ACDiscriminator32(num_classes=num_classes, in_channels=3, base=64)
    generator = CGenerator32(num_classes=num_classes, latent_dim=100, out_channels=3, base=64)
    gan = ACGAN(discriminator, generator)
    noises = np.random.normal(size=(50, 100, 1, 1))
    labels = np.tile(np.arange(num_classes), 5)

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

        images = create_images(generator, noises, labels)
        image_path = f"./outputs/{filename}-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)
