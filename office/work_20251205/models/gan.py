import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#####################################################################
# Loss functions for GAN
#####################################################################

# 1. BCE Loss (Vanilla GAN)
def bce_d_loss_fn(real_logits, fake_logits):
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    d_real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    d_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
    d_loss = d_real_loss + d_fake_loss
    return d_loss, d_real_loss, d_fake_loss

def bce_g_loss_fn(fake_logits):
    real_labels = torch.ones_like(fake_logits)
    g_loss = F.binary_cross_entropy_with_logits(fake_logits, real_labels)
    return g_loss

# 2. MSE Loss (LSGAN)
def mse_d_loss_fn(real_logits, fake_logits):
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    d_real_loss = F.mse_loss(real_logits, real_labels)
    d_fake_loss = F.mse_loss(fake_logits, fake_labels)
    d_loss = d_real_loss + d_fake_loss
    return d_loss, d_real_loss, d_fake_loss

def mse_g_loss_fn(fake_logits):
    real_labels = torch.ones_like(fake_logits)
    g_loss = F.mse_loss(fake_logits, real_labels)
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


class GAN(nn.Module):
    def __init__(self, discriminator, generator, loss_type="hinge", latent_dim=None, device=None):
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
        self.global_epoch = 0

    def set_optimizers(self, d_optimizer, g_optimizer):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

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
            g_loss=g_loss.item(),
        )
