import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gan import GAN


class CGAN(GAN):
    def train_step(self, batch):
        batch_size = batch["image"].size(0)
        labels = batch["label"].to(self.device)

        # (1) Update Discriminator
        real_images = batch["image"].to(self.device)
        real_logits = self.discriminator(real_images, labels)

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels).detach()
        fake_logits = self.discriminator(fake_images, labels)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels)
        fake_logits = self.discriminator(fake_images, labels)
        g_loss = self.g_loss_fn(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item()
        )
