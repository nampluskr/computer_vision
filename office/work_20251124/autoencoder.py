import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from torchmetrics.image import StructuralSimilarityIndexMeasure


#####################################################################
# Simple CNN for CIFAR10 (32, 32)
#####################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder32(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.flatten_dim = 128 * 4 * 4
        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, images):
        z = self.initial(images)
        z = self.blocks(z)
        z = z.view(-1, self.flatten_dim)
        latent = self.fc(z)
        return latent


class VEncoder32(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.flatten_dim = 128 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, images):
        z = self.initial(images)
        z = self.blocks(z)
        z = z.view(-1, self.flatten_dim)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        latent = self.reparameterize(mu, logvar)
        return latent, mu, logvar


#####################################################################
# Decoder: latent vector -> image (32, 32, out_channels)
#####################################################################

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Decoder32(nn.Module):
    def __init__(self, out_channels=3, latent_dim=128):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 128*4*4),
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        )
        self.blocks = nn.Sequential(
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
        )
        self.final = nn.ConvTranspose2d(32, out_channels, 
            kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, noises):
        x = self.initial(noises)
        x = self.blocks(x)
        x = self.final(x)
        return torch.sigmoid(x)


#####################################################################
# Autoencoder
#####################################################################

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.latent_dim = encoder.latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def forward(self, images):
        latent = self.encoder(images)
        recon = self.decoder(latent)
        return recon, latent

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        recon, latent = self.forward(images)
        loss = self.loss_fn(recon, images)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            ssim = self.ssim_metric(recon, images)
        return dict(loss=loss.item(), ssim=ssim.item())

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        recon, latent = self.forward(images)

        loss = self.loss_fn(recon, images)
        ssim = self.ssim_metric(recon, images)
        return dict(loss=loss.item(), ssim=ssim.item())


class VAE(nn.Module):
    def __init__(self, encoder, decoder, device=None, beta=0.1):
        super().__init__()
        self.latent_dim = encoder.latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=1e-3)

        self.recon_loss_fn = nn.MSELoss(reduction="mean")
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        self.beta = beta

    def forward(self, images):
        latent, mu, logvar = self.encoder(images)
        recon = self.decoder(latent)
        return recon, mu, logvar

    def loss_fn(self, recon, images, mu, logvar):
        recon_loss = self.recon_loss_fn(recon, images)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon_loss + self.beta * kld_loss
        return loss, recon_loss, kld_loss

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        recon, mu, logvar = self.forward(images)
        loss, recon_loss, kld_loss = self.loss_fn(recon, images, mu, logvar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            ssim = self.ssim_metric(recon, images)
        return dict(loss=loss.item(), mse=recon_loss.item(), kld=kld_loss.item(), ssim=ssim.item())

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        recon, mu, logvar = self.forward(images)
        loss, recon_loss, kld_loss = self.loss_fn(recon, images, mu, logvar)
        ssim = self.ssim_metric(recon, images)
        return dict(loss=loss.item(), mse=recon_loss.item(), kld=kld_loss.item(), ssim=ssim.item())
