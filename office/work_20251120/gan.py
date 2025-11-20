import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

from cifar10 import CIFAR10, get_train_loader, get_test_loader
from trainer import train_gan
from utils import set_seed


#####################################################################
# Generator for GAN
#####################################################################

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


class Generator32(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base*4, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, z):
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return x


class Generator64(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 8, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, z):
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return x


#####################################################################
# Discriminator for GAN
#####################################################################

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


class Discriminator32(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
        )
        self.final = nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=0, bias=False)

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
        logits = self.final(x).view(-1, 1)
        return logits


class Discriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64, use_spectral_norm=False):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
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

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


#####################################################################
# Loss functions for GAN
#####################################################################

# 1. BCE Loss (Vanilla GAN)
def bce_d_loss_fn(real_logits, fake_logits):
    d_real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    d_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    d_loss = d_real_loss + d_fake_loss
    return d_loss, d_real_loss, d_fake_loss

def bce_g_loss_fn(fake_logits):
    g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    return g_loss

# 2. MSE Loss (LSGAN)
def mse_d_loss_fn(real_logits, fake_logits):
    d_real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits))
    d_fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
    d_loss = d_real_loss + d_fake_loss
    return d_loss, d_real_loss, d_fake_loss

def mse_g_loss_fn(fake_logits):
    g_loss = F.mse_loss(fake_logits, torch.ones_like(fake_logits))
    return g_loss

# 3. Hinge Loss (StyleGAN)
def hinge_d_loss_fn(real_logits, fake_logits):
    d_real_loss = torch.relu(1.0 - real_logits).mean()
    d_fake_loss = torch.relu(1.0 + fake_logits).mean()
    d_loss = d_real_loss + d_fake_loss
    return d_loss, d_real_loss, d_fake_loss

def hinge_g_loss_fn(fake_logits):
    g_loss = -fake_logits.mean()
    return g_loss


#####################################################################
# GAN
#####################################################################

class GAN(nn.Module):
    def __init__(self, discriminator, generator, loss_type="bce", latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)

        if loss_type == "bce":
            self.d_loss_fn, self.g_loss_fn = bce_d_loss_fn, bce_g_loss_fn
        elif loss_type == "mse":
            self.d_loss_fn, self.g_loss_fn = mse_d_loss_fn, mse_g_loss_fn
        elif loss_type == "hinge":
            self.d_loss_fn, self.g_loss_fn = hinge_d_loss_fn, hinge_g_loss_fn
        else:
             raise ValueError(f"Unknown loss type: {loss_type}")

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim

    def train_step(self, batch):
        batch_size = batch["image"].size(0)

        # (1) Update Discriminator
        real_images = batch["image"].to(self.device)
        real_logits = self.discriminator(real_images)

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises).detach()
        fake_logits = self.discriminator(fake_images)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises)
        fake_logits = self.discriminator(fake_images)
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


class WGAN(nn.Module):
    def __init__(self, critic, generator, latent_dim=None, device=None, use_gp=False, one_sided=False):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = critic.to(self.device)
        self.generator = generator.to(self.device)

        if use_gp:
            self.c_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
        else:
            self.c_optimizer = optim.RMSprop(self.critic.parameters(), lr=5e-5)
            self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=5e-5)

        self.latent_dim = latent_dim or generator.latent_dim
        self.clip_value = 0.01
        self.gp_lambda = 10.0
        self.d_steps = 5
        self.use_gp = use_gp
        self.one_sided = one_sided

    def gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates = interpolates.requires_grad_(True)
        score = self.critic(interpolates)

        gradients = torch.autograd.grad(outputs=score, inputs=interpolates,
            grad_outputs=torch.ones_like(score),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        if self.one_sided:
            penalty =  torch.mean((torch.relu(gradients_norm - 1)) ** 2)
        else:    
            penalty = torch.mean((gradients_norm - 1) ** 2)
        return penalty

    def d_loss_fn(self, real_logits, fake_logits):
        d_real_loss = -real_logits.mean()
        d_fake_loss = fake_logits.mean()
        d_loss = d_real_loss + d_fake_loss
        return d_loss, d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        return -fake_logits.mean()

    def train_step(self, batch):
        batch_size = batch["image"].size(0)

        # (1) Update Critic
        real_images = batch["image"].to(self.device)
        for _ in range(self.d_steps):
            real_logits = self.critic(real_images)

            noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            fake_images = self.generator(noises).detach()
            fake_logits = self.critic(fake_images)

            d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)
            if self.use_gp:
                gp = self.gradient_penalty(real_images, fake_images)
            else:
                gp = torch.tensor(0.0).to(self.device)

            total_d_loss = d_loss + gp * self.gp_lambda

            self.c_optimizer.zero_grad()
            total_d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.c_optimizer.step()

            if not self.use_gp:
                for param in self.critic.parameters():
                    param.data.clamp_(-self.clip_value, self.clip_value)

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises)
        fake_logits = self.critic(fake_images)
        g_loss = self.g_loss_fn(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        results = dict(d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item(),
        )
        if self.use_gp:
            results["gp"] = gp.item()
        return results


# 4. Relativistic BCE loss (RaGAN)
def ra_d_loss_fn(real_logits, fake_logits):
    d_real_loss = F.binary_cross_entropy_with_logits(
        real_logits - fake_logits.mean(dim=0, keepdim=True), torch.ones_like(real_logits))
    d_fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits - real_logits.mean(dim=0, keepdim=True), torch.zeros_like(fake_logits))
    d_loss = d_real_loss + d_fake_loss
    return d_loss, d_real_loss, d_fake_loss

def ra_g_loss_fn(fake_logits, real_logits):
    g_fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits - real_logits.mean(dim=0, keepdim=True), torch.ones_like(fake_logits))
    g_real_loss = F.binary_cross_entropy_with_logits(
        real_logits - fake_logits.mean(dim=0, keepdim=True), torch.zeros_like(real_logits))
    g_loss = g_fake_loss + g_real_loss
    return g_loss

# 4. Relativistic Least Square Loss (RaLSGAN)
def rals_d_loss_fn(real_logits, fake_logits):
    d_real_loss = ((real_logits - fake_logits.mean() - 1) ** 2).mean()
    d_fake_loss = ((fake_logits - real_logits.mean() + 1) ** 2).mean()
    d_loss = (d_real_loss + d_fake_loss) / 2
    return d_loss, d_real_loss, d_fake_loss

def rals_g_loss_fn(fake_logits, real_logits):
    g_loss = ((fake_logits - real_logits.mean() + 1) ** 2).mean()
    return g_loss


class RaGAN(nn.Module):
    def __init__(self, discriminator, generator, loss_type="ra", latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)

        if loss_type == "ra":
            self.d_loss_fn, self.g_loss_fn = ra_d_loss_fn, ra_g_loss_fn
        elif loss_type == "rals":
            self.d_loss_fn, self.g_loss_fn = rals_d_loss_fn, rals_g_loss_fn
        else:
             raise ValueError(f"Unknown loss type: {loss_type}")

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim

    def train_step(self, batch):
        batch_size = batch["image"].size(0)
        real_images = batch["image"].to(self.device)

        # (1) Update Discriminator
        real_logits = self.discriminator(real_images)

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises).detach()
        fake_logits = self.discriminator(fake_images)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises)
        fake_logits = self.discriminator(fake_images)
        real_logits = self.discriminator(real_images).detach()

        g_loss = self.g_loss_fn(fake_logits, real_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item(),
        )


if __name__ == "__main__":

    set_seed(42)

    train_loader = get_train_loader(
        dataset=CIFAR10(
            root_dir="/home/namu/myspace/NAMU/datasets/cifar10",
            split="train",
            transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        ),
        batch_size=128)

    noises = np.random.normal(size=(50, 100, 1, 1))     # num_samples = 50, latent_dim = 100

    # for loss_type in ["bce", "mse", "hinge"]:
    #     print(f"\n>> Loss type: {loss_type}")
    #     discriminator = Discriminator32(in_channels=3, base=64)
    #     generator = Generator32(latent_dim=100, out_channels=3, base=64)
    #     gan = GAN(discriminator, generator, loss_type=loss_type)
    #     train_gan(gan, train_loader, num_epochs=5, total_epochs=20, noises=noises, filename=f"gan-{loss_type}")

    # total_epochs = 5

    # loss_type = "default"
    # print(f"\n>> Loss type: {loss_type}")
    # discriminator = Discriminator32(in_channels=3, base=64)
    # generator = Generator32(latent_dim=100, out_channels=3, base=64)
    # gan = WGAN(discriminator, generator)
    # train_gan(gan, train_loader, num_epochs=5, total_epochs=total_epochs, noises=noises, filename=f"wgan-{loss_type}")

    # loss_type = "gp"
    # print(f"\n>> Loss type: {loss_type}")
    # discriminator = Discriminator32(in_channels=3, base=64)
    # generator = Generator32(latent_dim=100, out_channels=3, base=64)
    # gan = WGAN(discriminator, generator, use_gp=True)
    # train_gan(gan, train_loader, num_epochs=5, total_epochs=total_epochs, noises=noises, filename=f"wgan-{loss_type}")

    # loss_type = "gp_one-sided"
    # print(f"\n>> Loss type: {loss_type}")
    # discriminator = Discriminator32(in_channels=3, base=64)
    # generator = Generator32(latent_dim=100, out_channels=3, base=64)
    # gan = WGAN(discriminator, generator, use_gp=True, one_sided=True)
    # train_gan(gan, train_loader, num_epochs=5, total_epochs=total_epochs, noises=noises, filename=f"wgan-{loss_type}")

    for loss_type in ["ra", "rals"]:
        print(f"\n>> Loss type: {loss_type}")
        discriminator = Discriminator32(in_channels=3, base=64)
        generator = Generator32(latent_dim=100, out_channels=3, base=64)
        gan = RaGAN(discriminator, generator, loss_type=loss_type)
        train_gan(gan, train_loader, num_epochs=5, total_epochs=20, noises=noises, filename=f"ragan-{loss_type}")
