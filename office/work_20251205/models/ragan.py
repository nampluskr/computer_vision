import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#####################################################################
# Loss functions for RaGAN / RaLSGAN
#####################################################################

# Relativistic BCE loss (RaGAN)
def ra_d_loss_fn(real_logits, fake_logits, smooth=0.1):
    d_real_loss = F.binary_cross_entropy_with_logits(
        real_logits - fake_logits.mean(dim=0, keepdim=True), torch.ones_like(real_logits) * (1.0 - smooth))
    d_fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits - real_logits.mean(dim=0, keepdim=True), torch.ones_like(fake_logits) * smooth)
    d_loss = (d_real_loss + d_fake_loss) / 2
    return d_loss, d_real_loss, d_fake_loss


def ra_g_loss_fn(fake_logits, real_logits):
    g_fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits - real_logits.mean(dim=0, keepdim=True), torch.ones_like(fake_logits))
    g_real_loss = F.binary_cross_entropy_with_logits(
        real_logits - fake_logits.mean(dim=0, keepdim=True), torch.zeros_like(real_logits))
    g_loss = (g_fake_loss + g_real_loss) / 2
    return g_loss


# Relativistic Least Square Loss (RaLSGAN)
def rals_d_loss_fn(real_logits, fake_logits):
    d_real_loss = ((real_logits - fake_logits.mean() - 1) ** 2).mean()
    d_fake_loss = ((fake_logits - real_logits.mean() + 1) ** 2).mean()
    d_loss = (d_real_loss + d_fake_loss) / 2
    return d_loss, d_real_loss, d_fake_loss


def rals_g_loss_fn(fake_logits, real_logits):
    g_fake_loss = ((fake_logits - real_logits.mean() - 1) ** 2).mean()
    g_real_loss = ((real_logits - fake_logits.mean() + 1) ** 2).mean()
    g_loss = (g_fake_loss + g_real_loss) / 2
    return g_loss


class RaGAN(nn.Module):
    def __init__(self, discriminator, generator, loss_type="ra", latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)

        if loss_type == "ra":
            self.d_loss_fn, self.g_loss_fn = ra_d_loss_fn, ra_g_loss_fn
            self.d_step = 1
        elif loss_type == "rals":
            self.d_loss_fn, self.g_loss_fn = rals_d_loss_fn, rals_g_loss_fn
            self.d_step = 5
        else:
             raise ValueError(f"Unknown loss type: {loss_type}")

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim

    def train_step(self, batch):
        batch_size = batch["image"].size(0)
        real_images = batch["image"].to(self.device)

        # (1) Update Discriminator
        for _ in range(self.d_step):
            real_logits = self.discriminator(real_images)

            noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            fake_images = self.generator(noises).detach()
            fake_logits = self.discriminator(fake_images)

            d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

            self.d_optimizer.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises)
        fake_logits = self.discriminator(fake_images)
        real_logits = self.discriminator(real_images).detach()

        g_loss = self.g_loss_fn(fake_logits, real_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item(),
        )
