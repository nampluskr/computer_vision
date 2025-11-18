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


class RGAN_P(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.D = discriminator.to(self.device)
        self.G = generator.to(self.device)
        self.d_opt = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_opt = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.latent_dim = latent_dim or getattr(self.G, "latent_dim", 100)

    @staticmethod
    def _relativistic_pair_loss(real_logits, fake_logits, for_generator):
        if not for_generator:
            loss_real = -F.logsigmoid(real_logits - fake_logits).mean()
            loss_fake = -F.logsigmoid(fake_logits - real_logits).mean()
            loss = loss_real + loss_fake
        else:
            loss_real = -F.logsigmoid(fake_logits - real_logits).mean()
            loss_fake = -F.logsigmoid(real_logits - fake_logits).mean()
            loss = loss_real + loss_fake
        return loss

    def d_loss_fn(self, real_logits, fake_logits):
        return self._relativistic_pair_loss(real_logits, fake_logits, False)

    def g_loss_fn(self, real_logits, fake_logits):
        return self._relativistic_pair_loss(real_logits, fake_logits, True)

    def train_step(self, batch):
        B = batch["image"].size(0)
        real_imgs = batch["image"].to(self.device)

        real_logits = self.D(real_imgs)
        noises = torch.randn(B, self.latent_dim, 1, 1, device=self.device)
        fake_imgs = self.G(noises).detach()
        fake_logits = self.D(fake_imgs)

        d_loss, _, _ = self.d_loss_fn(real_logits, fake_logits)
        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()

        noises = torch.randn(B, self.latent_dim, 1, 1, device=self.device)
        fake_imgs = self.G(noises)
        fake_logits = self.D(fake_imgs)
        real_logits = self.D(real_imgs)

        g_loss = self.g_loss_fn(real_logits, fake_logits)
        self.g_opt.zero_grad()
        g_loss.backward()
        self.g_opt.step()

        return dict(
            d_loss=d_loss.detach().cpu().item(),
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

    discriminator = Discriminator32(in_channels=3, base=64)
    generator = Generator32(latent_dim=100, out_channels=3, base=64)
    gan = RGAN_P(discriminator, generator)
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
