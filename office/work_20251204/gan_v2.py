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
        self.final = nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
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
        return torch.tanh(x)


class Generator64(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            DeconvBlock(base*8, base*4),
            DeconvBlock(base*4, base*2),
            DeconvBlock(base*2, base),
        )
        self.final = nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
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
        return torch.tanh(x)


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


class Discriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
            ConvBlock(base*4, base*8),
        )
        self.final = nn.Conv2d(base*8, 1, kernel_size=4, stride=1, padding=0, bias=False)
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


#####################################################################
# InfoGAN
#####################################################################

class InfoGenerator64(nn.Module):
    def __init__(self, latent_dim, num_continuous, num_discrete, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete
        self.total_dim = latent_dim + num_continuous + num_discrete
        self.base = base

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(self.total_dim, base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            DeconvBlock(base * 8, base * 4),
            DeconvBlock(base * 4, base * 2),
            DeconvBlock(base * 2, base),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
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

    def forward(self, zc):
        x = self.initial(zc)
        x = self.blocks(x)
        return self.final(x)


class InfoDiscriminator64(nn.Module):
    def __init__(self, num_continuous=1, num_discrete=2, in_channels=3, base=64):
        super().__init__()
        self.num_continuous = num_continuous  # e.g., Smiling intensity
        self.num_discrete = num_discrete      # e.g., Gender: Male/Female
        self.base = base

        # Feature extractor
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 8),
        )  # Output: (B, 512, 4, 4)

        # Discriminator head
        self.d_head = nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)

        # Q network shared layers
        self.q_shared = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base * 4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Continuous code: Gaussian parameters
        self.q_mu = nn.Linear(128, num_continuous)
        self.q_logvar = nn.Linear(128, num_continuous)

        # Discrete code: classification logits
        self.q_logits = nn.Linear(128, num_discrete)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.initial(x)
        h = self.blocks(h)

        real_logits = self.d_head(h).view(-1, 1)
        q_feat = self.q_shared(h)
        mu = self.q_mu(q_feat)
        logvar = self.q_logvar(q_feat)
        logits_discrete = self.q_logits(q_feat)
        return real_logits, mu, logvar, logits_discrete


# https://github.com/hwalsuklee/tensorflow-generative-model-collections
# https://github.com/znxlwm/pytorch-generative-model-collections
class InfoGAN(nn.Module):
    def __init__(self, discriminator, generator, 
                 latent_dim=100, num_continuous=1, num_discrete=2,
                 lambda_continuous=1.0, lambda_discrete=1.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)

        self.latent_dim = latent_dim
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete
        self.lambda_continuous = lambda_continuous
        self.lambda_discrete = lambda_discrete

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.q_optimizer = optim.Adam(
            list(self.discriminator.q_shared.parameters()) +
            list(self.discriminator.q_mu.parameters()) +
            list(self.discriminator.q_logvar.parameters()) +
            list(self.discriminator.q_logits.parameters()),
            lr=1e-3, betas=(0.5, 0.999)
        )

    def d_loss_fn(self, real_logits, fake_logits):
        d_real_loss = torch.relu(1.0 - real_logits).mean()
        d_fake_loss = torch.relu(1.0 + fake_logits).mean()
        return d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        real_labels = torch.ones_like(fake_logits)
        return -fake_logits.mean()

    def mutual_info_loss(self, c_continuous, c_discrete, mu, logvar, logits_discrete):
        c_continuous = c_continuous.squeeze(-1).squeeze(-1)  # (B, num_continuous)
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(logvar)
        continuous_loss = 0.5 * torch.sum(
            logvar + (c_continuous - mu) ** 2 / (std ** 2 + 1e-8), dim=1).mean()

        c_discrete = c_discrete.squeeze(-1).squeeze(-1)  # (B, num_discrete)
        labels = torch.argmax(c_discrete, dim=1)  # one-hot → class index
        discrete_loss = F.cross_entropy(logits_discrete, labels)
        return continuous_loss, discrete_loss

    def sample_codes(self, batch_size):
        # z ~ N(0, 1)
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

        # c_continuous ~ Uniform(-1, 1)
        c_continuous = torch.rand(batch_size, self.num_continuous, 1, 1, device=self.device) * 2 - 1

        # c_discrete: one-hot
        labels = torch.randint(0, self.num_discrete, (batch_size,), device=self.device)
        c_discrete = F.one_hot(labels, num_classes=self.num_discrete).float()
        c_discrete = c_discrete.unsqueeze(-1).unsqueeze(-1)  # (B, num_discrete, 1, 1)
        return z, c_continuous, c_discrete

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        batch_size = images.size(0)

        # (1) Update Discriminator
        real_logits, _, _, _ = self.discriminator(images)
        z, c_cont, c_disc = self.sample_codes(batch_size)
        zc = torch.cat([z, c_cont, c_disc], dim=1)
        fake_images = self.generator(zc).detach()
        fake_logits, _, _, _ = self.discriminator(fake_images)

        d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)
        d_loss = d_real_loss + d_fake_loss

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator + Q Network
        z, c_cont, c_disc = self.sample_codes(batch_size)
        zc = torch.cat([z, c_cont, c_disc], dim=1)
        fake_images = self.generator(zc)
        fake_logits, mu, logvar, logits_discrete = self.discriminator(fake_images)

        # Generator 손실
        g_loss = self.g_loss_fn(fake_logits)
        cont_loss, disc_loss = self.mutual_info_loss(c_cont, c_disc, mu, logvar, logits_discrete)
        total_g_loss = g_loss + self.lambda_continuous * cont_loss + self.lambda_discrete * disc_loss

        self.g_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        total_g_loss.backward()
        self.g_optimizer.step()
        self.q_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            g_loss=g_loss.item(),
            cont_loss=cont_loss.item(),
            disc_loss=disc_loss.item()
        )
