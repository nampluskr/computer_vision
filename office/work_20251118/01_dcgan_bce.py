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
from dcgan import Discriminator32, Generator32


class GAN(nn.Module):
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
        d_loss = d_real_loss + d_fake_loss
        return d_loss, d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        real_labels = torch.ones_like(fake_logits)
        return F.binary_cross_entropy_with_logits(fake_logits, real_labels)

    def train_step(self, batch):
        batch_size = batch["image"].size(0)

        # (1) Update Discriminator
        real_images = batch["image"].to(self.device)
        real_logits = self.d_model(real_images)
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises).detach()
        fake_logits = self.d_model(fake_images)
        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises)
        fake_logits = self.d_model(fake_images)
        g_loss = self.g_loss_fn(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.detach().cpu().item(),
            real_loss=d_real_loss.detach().cpu().item(), 
            fake_loss=d_fake_loss.detach().cpu().item(), 
            g_loss=g_loss.detach().cpu().item()
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

    discriminator = Discriminator32(in_channels=3, base=64)
    generator = Generator32(latent_dim=100, out_channels=3, base=64)
    gan = GAN(discriminator, generator)
    z_sample = np.random.normal(size=(50, 100, 1, 1))

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

        images = create_images(generator, z_sample)
        image_path = f"./outputs/{filename}-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)
