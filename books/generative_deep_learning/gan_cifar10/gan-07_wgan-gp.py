""" GAN-F: WGAN (Loss 변경)
- Wasserstein Loss
- Sigmoid 제거
- D는 Critic 역할 수행
- 더 smooth한 gradient 제공
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


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1)
        # return torch.sigmoid(x)
        return x    # No sigmoid


class WGANGP(nn.Module):
    def __init__(self, critic, generator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = critic.to(self.device)
        self.g_model = generator.to(self.device)

        self.d_model.apply(self.init_weights)
        self.g_model.apply(self.init_weights)

        self.c_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=1e-4, betas=(0.0, 0.9))

        self.latent_dim = latent_dim or generator.latent_dim
        # self.loss_fn = nn.BCELoss()
        self.gp_lambda = 10.0
        self.d_steps = 5

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates = interpolates.requires_grad_(True)
        score = self.d_model(interpolates)

        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolates,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gp = torch.mean((gradients_norm - 1.0) ** 2)
        return gp

    def train_step(self, batch):
        batch_size = batch["image"].shape[0]
        real_images = batch["image"].to(self.device)

        # (1) Update Critic (d_steps times)
        for _ in range(self.d_steps):
            z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            real_score = self.d_model(real_images)
            fake_images = self.g_model(z).detach()
            fake_score = self.d_model(fake_images)

            # Wasserstein Critic loss: maximize (real_score - fake_score) == minimize -(real - fake)
            d_loss = -(real_score.mean() - fake_score.mean())
            gp = self.gradient_penalty(real_images, fake_images)
            d_loss_gp = d_loss + self.gp_lambda * gp

            self.c_optimizer.zero_grad()
            d_loss_gp.backward()
            self.c_optimizer.step()

        # (2) Update Generator
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(z)
        fake_score = self.d_model(fake_images)

        # Generator tries to maximize D(fake) == minimize -D(fake)
        g_loss = -torch.mean(fake_score)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(critic_loss=d_loss_gp, gen_loss=g_loss)


if __name__ == "__main__":

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    root_dir = "/mnt/d/datasets/cifar10"
    train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train", transform=transform), batch_size=128)

    discriminator = Discriminator(in_channels=3, base=64)
    generator = Generator(latent_dim=100, out_channels=3, base=64)
    gan = WGANGP(discriminator, generator)
    z_sample = np.random.normal(size=(50, 100, 1, 1))

    filename = os.path.splitext(os.path.basename(__file__))[0]
    total_history = {}
    epoch, num_epochs = 0, 5

    for _ in range(2):
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