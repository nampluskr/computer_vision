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
            g_loss=g_loss.item()
        )


#####################################################################
# InfoGAN
#####################################################################

class InfoGenerator64(nn.Module):
    def __init__(self, z_dim=100, c_dim=2, out_channels=3, base=64):
        super().__init__()
        self.base = base
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim + c_dim, base*8, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, zc):
        x = self.initial(zc)
        x = self.blocks(x)
        x = self.final(x)
        return torch.tanh(x)


class InfoDiscriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.base = base
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

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        return self.final(x).view(-1, 1)

    def forward_features(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        return x    # [B, base*8, 4, 4]


class InfoGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=100, c_dim_discrete=2, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)
        self.latent_dim = latent_dim
        self.c_dim_discrete = c_dim_discrete
        self.total_latent_dim = latent_dim + c_dim_discrete

        # InfoGAN: Q Network (c 추정)
        self.q_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(discriminator.base * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, c_dim_discrete)  # 출력: [B, 2] (Male, Smiling)
        ).to(self.device)

        # 손실 함수 (Hinge 또는 BCE)
        self.d_loss_fn = self.hinge_d_loss_fn
        self.g_loss_fn = self.hinge_g_loss_fn
        self.q_loss_fn = nn.BCEWithLogitsLoss()

        # Optimizer
        params = list(self.discriminator.parameters()) + list(self.q_network.parameters())
        self.d_optimizer = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Mutual information 정규화 계수
        self.lambda_info = 1.0

    def hinge_d_loss_fn(self, real_logits, fake_logits):
        d_real_loss = torch.relu(1.0 - real_logits).mean()
        d_fake_loss = torch.relu(1.0 + fake_logits).mean()
        return d_real_loss + d_fake_loss, d_real_loss, d_fake_loss

    def hinge_g_loss_fn(self, fake_logits):
        return -fake_logits.mean()

    def sample_latent(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        c_discrete = torch.randint(0, 2, (batch_size, self.c_dim_discrete), 
                                   dtype=torch.float, device=self.device)
        return z, c_discrete

    def reparameterize(self, z, c):
        c = c.view(c.size(0), -1, 1, 1)
        return torch.cat([z, c], dim=1)

    def train_step(self, batch):
        batch_size = batch["image"].size(0)
        real_images = batch["image"].to(self.device)

        # =============================================
        # (1) Discriminator + Q Network 업데이트
        # =============================================
        real_logits = self.discriminator(real_images)

        z, c_discrete = self.sample_latent(batch_size)
        z_input = self.reparameterize(z, c_discrete)
        fake_images = self.generator(z_input).detach()
        fake_logits = self.discriminator(fake_images)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

        # Q 손실: 생성된 이미지에서 c 복원
        features = self.discriminator.forward_features(fake_images)
        q_logits = self.q_network(features)
        q_loss = self.q_loss_fn(q_logits, c_discrete)

        total_d_loss = d_loss + self.lambda_info * q_loss

        self.d_optimizer.zero_grad()
        total_d_loss.backward()
        self.d_optimizer.step()

        # =============================================
        # (2) Generator + Q Network 업데이트
        # =============================================
        z, c_discrete = self.sample_latent(batch_size)
        z_input = self.reparameterize(z, c_discrete)
        fake_images = self.generator(z_input)
        fake_logits = self.discriminator(fake_images)

        g_loss = self.g_loss_fn(fake_logits)

        # Q 손실 추가 (정보 보존 유도)
        features = self.discriminator.forward_features(fake_images)
        q_logits = self.q_network(features)
        q_loss = self.q_loss_fn(q_logits, c_discrete)
        total_g_loss = g_loss + self.lambda_info * q_loss

        self.g_optimizer.zero_grad()
        total_g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            g_loss=g_loss.item(),
            q_loss=q_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item()
        )
