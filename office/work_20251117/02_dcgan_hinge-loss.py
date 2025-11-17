import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.nn.utils import spectral_norm

from cifar10 import CIFAR10, get_train_loader, get_test_loader
from trainer import fit
from utils import create_images, plot_images, plot_history, set_seed
from dcgan import Discriminator, Generator


class HingeGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = discriminator.to(self.device)
        self.g_model = generator.to(self.device)

        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim

    def train_step(self, batch):
        batch_size = batch["image"].shape[0]

        # (1) Update Discriminator
        self.d_optimizer.zero_grad()
        real_images = batch["image"].to(self.device)
        real_preds = self.d_model(real_images)
        d_real_loss = torch.relu(1.0 - real_preds).mean()
        d_real_loss.backward()

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises).detach()
        fake_preds = self.d_model(fake_images)
        d_fake_loss = torch.relu(1.0 + fake_preds).mean()
        d_fake_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        self.g_optimizer.zero_grad()
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises)
        fake_preds = self.d_model(fake_images)
        g_loss = -fake_preds.mean()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            real_loss=d_real_loss.detach().cpu().item(), 
            fake_loss=d_fake_loss.detach().cpu().item(), 
            gen_loss=g_loss.detach().cpu().item()
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

    discriminator = Discriminator(in_channels=3, base=64)
    generator = Generator(latent_dim=100, out_channels=3, base=64)
    gan = HingeGAN(discriminator, generator)
    z_sample = np.random.normal(size=(50, 100, 1, 1))

    filename = os.path.splitext(os.path.basename(__file__))[0]
    total_history = {}
    epoch, num_epochs = 0, 5

    for _ in range(1):
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
