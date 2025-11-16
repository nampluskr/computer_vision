""" GAN-I: ResDCGAN (Residual Block)
ResBlock Upsample
ResBlock Downsample
CIFAR10에서도 안정성과 품질 상승
"""

import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T


from cifar10 import CIFAR10, get_train_loader, get_test_loader
from trainer import fit
from utils import create_images, plot_images, plot_history


class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        skip = self.skip(self.upsample(x))
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out + skip


class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 2, 0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        skip = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.down(out)
        return out + skip


class ResGenerator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim

        # z: (B, latent_dim, 1, 1) → (B, base*4, 4, 4)
        self.fc = nn.ConvTranspose2d(latent_dim, base * 4, 4, 1, 0)

        # ResBlockUpsample: 4→8, 8→16, 16→32
        self.res1 = ResBlockUp(base * 4, base * 2)   # 4→8
        self.res2 = ResBlockUp(base * 2, base)       # 8→16
        self.res3 = ResBlockUp(base, base // 2)      # 16→32

        self.out_conv = nn.Conv2d(base // 2, out_channels, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.out_conv(nn.LeakyReLU(0.2)(x))
        return torch.tanh(x)


class ResDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()

        # 32→16, 16→8, 8→4
        self.res1 = ResBlockDown(in_channels, base // 2)
        self.res2 = ResBlockDown(base // 2, base)
        self.res3 = ResBlockDown(base, base * 2)

        # global sum pooling 후 linear
        self.linear = nn.Linear(base * 2, 1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # Global Sum Pool
        x = nn.functional.leaky_relu(x, 0.2)
        x = x.sum(dim=[2, 3])  # (B, C)

        return self.linear(x)


class HingeGAN(nn.Module):
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
        # self.loss_fn = nn.BCELoss()

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def train_step(self, batch):
        batch_size = batch["image"].shape[0]
        # real_labels = torch.ones((batch_size, 1)).to(self.device)
        # fake_labels = torch.zeros((batch_size, 1)).to(self.device)

        # Train discriminator
        self.d_optimizer.zero_grad()
        real_images = batch["image"].to(self.device)
        real_preds = self.d_model(real_images)
        # d_real_loss = self.loss_fn(real_preds, real_labels)
        d_real_loss = torch.relu(1.0 - real_preds).mean()
        d_real_loss.backward()

        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(z).detach()
        fake_preds = self.d_model(fake_images)
        # d_fake_loss = self.loss_fn(fake_preds, fake_labels)
        d_fake_loss = torch.relu(1.0 + fake_preds).mean()
        d_fake_loss.backward()
        self.d_optimizer.step()

        # Train generator
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(z)
        fake_preds = self.d_model(fake_images)
        # g_loss = self.loss_fn(fake_preds, real_labels)
        g_loss = -fake_preds.mean()
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

    discriminator = ResDiscriminator(in_channels=3, base=64)
    generator = ResGenerator(latent_dim=100, out_channels=3, base=64)
    gan = HingeGAN(discriminator, generator)
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