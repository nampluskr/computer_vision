""" GAN-B: DCGAN+
- ConvTranspose 기반 Generator
- BCE loss
- BN everywhere => BatchNorm 제거 + SpectralNorm 추가
"""

import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T


from cifar10 import CIFAR10, get_train_loader, get_test_loader
from trainer import fit
from utils import create_images, plot_images, plot_history


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        return self.deconv_block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, z):
        x = self.net(z)
        return torch.tanh(x)

# BatchNorm 제거 + SpectralNorm 추가
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            # nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            # nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(base * 4, 1, kernel_size=4, stride=1, padding=0, bias=False)),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1)
        return torch.sigmoid(x)


class GAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = discriminator.to(self.device)
        self.g_model = generator.to(self.device)

        self.d_model.apply(self.init_weights)
        self.g_model.apply(self.init_weights)

        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim
        self.loss_fn = nn.BCELoss()

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def train_step(self, batch):
        batch_size = batch["image"].shape[0]
        real_labels = torch.ones((batch_size, 1)).to(self.device)
        fake_labels = torch.zeros((batch_size, 1)).to(self.device)

        # Train discriminator
        self.d_optimizer.zero_grad()
        real_images = batch["image"].to(self.device)
        real_preds = self.d_model(real_images)
        d_real_loss = self.loss_fn(real_preds, real_labels)
        d_real_loss.backward()

        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(z).detach()
        fake_preds = self.d_model(fake_images)
        d_fake_loss = self.loss_fn(fake_preds, fake_labels)
        d_fake_loss.backward()
        self.d_optimizer.step()

        # Train generator
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(z)
        fake_preds = self.d_model(fake_images)
        g_loss = self.loss_fn(fake_preds, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return dict(real_loss=d_real_loss, fake_loss=d_fake_loss, gen_loss=g_loss)


if __name__ == "__main__":

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    root_dir = "/mnt/d/datasets/cifar10"
    train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train", transform=transform), batch_size=128)

    discriminator = Discriminator(in_channels=3)
    generator = Generator(latent_dim=100, out_channels=3)
    gan = GAN(discriminator, generator)
    z_sample = np.random.normal(size=(50, 100, 1, 1))

    filename = os.path.splitext(os.path.basename(__file__))[0]
    total_history = {}
    epoch, num_epochs = 0, 5

    for _ in range(10):
        history = fit(gan, train_loader, num_epochs=num_epochs)
        epoch += num_epochs
        for split_name, metrics in history.items():
            total_history.setdefault(split_name, {})
            for metric_name, metric_values in metrics.items():
                total_history[split_name].setdefault(metric_name, [])
                total_history[split_name][metric_name].extend(metric_values)

        images = create_images(generator, z_sample)
        image_path = f"./outputs/{filename}-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)