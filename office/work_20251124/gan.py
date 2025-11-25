import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#####################################################################
# Generators for GAN
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


class CGenerator32(nn.Module):
    def __init__(self, num_classes, latent_dim=100, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.labels_embedding = nn.Sequential(
            nn.Embedding(num_classes, num_classes),
            nn.Unflatten(dim=1, unflattened_size=(num_classes, 1, 1))
        )
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, base*4, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, noises, labels):
        labels = self.labels_embedding(labels)
        z = torch.cat([noises, labels], dim=1)
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return x


#####################################################################
# Discriminators for GAN
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

    def forward(self, images):
        x = self.initial(images)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


class CDiscriminator32(nn.Module):
    def __init__(self, num_classes, in_channels=3, base=64):
        super().__init__()
        self.num_classes = num_classes
        self.labels_embedding = nn.Sequential(
            nn.Embedding(num_classes, 32 * 32),
            nn.Unflatten(dim=1, unflattened_size=(1, 32, 32))
        )
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + 1, base, kernel_size=4, stride=2, padding=1, bias=False),
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

    def forward(self, images, labels):
        labels = self.labels_embedding(labels)
        x = torch.cat([images, labels], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


class ACDiscriminator32(nn.Module):
    def __init__(self, num_classes, in_channels=3, base=64):
        super().__init__()
        self.num_classes = num_classes
        self.labels_embedding = nn.Sequential(
            nn.Embedding(num_classes, 32 * 32),
            nn.Unflatten(dim=1, unflattened_size=(1, 32, 32))
        )
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + 1, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
        )
        self.final = nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.aux = nn.Conv2d(base*4, num_classes, kernel_size=4, stride=1, padding=0, bias=False)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, images, labels):
        labels = self.labels_embedding(labels)
        x = torch.cat([images, labels], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        aux = self.aux(x).view(-1, self.num_classes)
        return logits, aux


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
            g_loss=g_loss.item()
        )


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


class ACGAN(GAN):
    def d_aux_loss_fn(self, real_aux, fake_aux, labels):
        d_real_aux_loss = F.cross_entropy(real_aux, labels)
        d_fake_aux_loss = F.cross_entropy(fake_aux, labels)
        return d_real_aux_loss, d_fake_aux_loss

    def g_aux_loss_fn(self, fake_aux, labels):
        return F.cross_entropy(fake_aux, labels)

    def train_step(self, batch):
        batch_size = batch["image"].size(0)
        labels = batch["label"].to(self.device)

        # (1) Update Discriminator
        images = batch["image"].to(self.device)
        real_logits, real_aux = self.discriminator(images, labels)

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels).detach()
        fake_logits, fake_aux = self.discriminator(fake_images, labels)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)
        d_real_aux_loss, d_fake_aux_loss = self.d_aux_loss_fn(real_aux, fake_aux, labels)
        total_d_loss = d_loss + d_real_aux_loss + d_fake_aux_loss

        self.d_optimizer.zero_grad()
        total_d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels)
        fake_logits, fake_aux = self.discriminator(fake_images, labels)

        g_loss = self.g_loss_fn(fake_logits) + self.g_aux_loss_fn(fake_aux, labels)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item()
        )

#####################################################################
# WGAN
#####################################################################

class WGAN(nn.Module):
    def __init__(self, critic, generator, latent_dim=None, device=None, use_gp=True, one_sided=True):
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

        if self.one_sided:
            return torch.mean(torch.relu(gradients_norm - 1) ** 2)
        else:
            return torch.mean((gradients_norm - 1) ** 2)

    def d_loss_fn(self, real_logits, fake_logits):
        d_real_loss = -real_logits.mean()
        d_fake_loss = fake_logits.mean()
        d_loss = d_real_loss + d_fake_loss
        return d_loss, d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        return -fake_logits.mean()

    def train_step(self, batch):
        batch_size = batch["image"].size(0)

        # (1) Update Discriminator (Critic)
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
            d_loss_gp = d_loss + gp * self.gp_lambda

            self.c_optimizer.zero_grad()
            d_loss_gp.backward()
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

#####################################################################
# RaGAN
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
