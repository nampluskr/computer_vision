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


# --------------------------------------------------------------
#  Generator
# --------------------------------------------------------------

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


class InfoGenerator32(nn.Module):
    def __init__(self, latent_dim=62, num_continuous=2, num_discrete=10, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete
        total_input_dim = latent_dim + num_continuous + num_discrete

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(total_input_dim, base*4, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, z, c_continuous, c_discrete):
        """
        Args:
            z: (N, latent_dim, 1, 1) - random noise
            c_continuous: (N, num_continuous, 1, 1) - continuous codes
            c_discrete: (N, num_discrete, 1, 1) - discrete codes (one-hot)
        """
        x = torch.cat([z, c_continuous, c_discrete], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return x


class InfoGenerator64(nn.Module):
    def __init__(self, latent_dim=62, num_continuous=2, num_discrete=10, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete
        total_input_dim = latent_dim + num_continuous + num_discrete

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(total_input_dim, base * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True)
        )  # (N, 512, 4, 4)
        self.blocks = nn.Sequential(
            DeconvBlock(base * 8, base * 4),
            DeconvBlock(base * 4, base * 2),
            DeconvBlock(base * 2, base),
            DeconvBlock(base, base)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
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

    def forward(self, z, c_continuous, c_discrete):
        x = torch.cat([z, c_continuous, c_discrete], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return x


# --------------------------------------------------------------
#  Discriminator (with Q Network for mutual information)
# --------------------------------------------------------------

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


class InfoDiscriminator32(nn.Module):
    def __init__(self, num_continuous=2, num_discrete=10, in_channels=3, base=64):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
        )
        
        # Discriminator head
        self.d_head = nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        
        # Q Network head (for mutual information estimation)
        self.q_shared = nn.Sequential(
            nn.Conv2d(base*4, base*2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base*2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Continuous code head (outputs mean and log_std)
        self.q_continuous_mu = nn.Linear(128, num_continuous)
        self.q_continuous_logstd = nn.Linear(128, num_continuous)
        
        # Discrete code head (outputs logits for categorical distribution)
        self.q_discrete = nn.Linear(128, num_discrete)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, images):
        x = self.initial(images)
        x = self.blocks(x)
        
        # Discriminator output
        logits = self.d_head(x).view(-1, 1)
        
        # Q Network outputs
        q_feat = self.q_shared(x)
        
        # Continuous codes: Gaussian distribution parameters
        c_continuous_mu = self.q_continuous_mu(q_feat)
        c_continuous_logstd = self.q_continuous_logstd(q_feat)
        
        # Discrete codes: Categorical distribution logits
        c_discrete_logits = self.q_discrete(q_feat)
        
        return logits, c_continuous_mu, c_continuous_logstd, c_discrete_logits


class InfoDiscriminator64(nn.Module):
    def __init__(self, num_continuous=2, num_discrete=10, in_channels=3, base=64):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 8)
        )
        
        self.d_head = nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        
        self.q_shared = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base * 4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.q_continuous_mu = nn.Linear(128, num_continuous)
        self.q_continuous_logstd = nn.Linear(128, num_continuous)
        self.q_discrete = nn.Linear(128, num_discrete)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, images):
        x = self.initial(images)
        x = self.blocks(x)
        
        logits = self.d_head(x).view(-1, 1)
        
        q_feat = self.q_shared(x)
        c_continuous_mu = self.q_continuous_mu(q_feat)
        c_continuous_logstd = self.q_continuous_logstd(q_feat)
        c_discrete_logits = self.q_discrete(q_feat)
        
        return logits, c_continuous_mu, c_continuous_logstd, c_discrete_logits


# --------------------------------------------------------------
#  InfoGAN
# --------------------------------------------------------------

class InfoGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=62, num_continuous=2, num_discrete=10, 
                 lambda_continuous=1.0, lambda_discrete=1.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = discriminator.to(self.device)
        self.g_model = generator.to(self.device)

        self.d_optimizer = optim.Adam(self.d_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.g_model.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.q_optimizer = optim.Adam(
            list(self.d_model.q_shared.parameters()) + 
            list(self.d_model.q_continuous_mu.parameters()) + 
            list(self.d_model.q_continuous_logstd.parameters()) + 
            list(self.d_model.q_discrete.parameters()), 
            lr=1e-3, betas=(0.5, 0.999)
        )

        self.latent_dim = latent_dim
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete
        self.lambda_continuous = lambda_continuous
        self.lambda_discrete = lambda_discrete

    def d_loss_fn(self, real_logits, fake_logits):
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        d_real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        d_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        return d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        real_labels = torch.ones_like(fake_logits)
        return F.binary_cross_entropy_with_logits(fake_logits, real_labels)

    def mutual_info_loss(self, c_continuous, c_discrete, c_continuous_mu, c_continuous_logstd, c_discrete_logits):
        """
        Compute mutual information loss for InfoGAN
        
        For continuous codes: Negative log-likelihood under Gaussian
        For discrete codes: Cross-entropy loss
        """
        # Continuous code loss (Gaussian NLL)
        c_continuous = c_continuous.squeeze(-1).squeeze(-1)  # (N, num_continuous)
        
        # Clamp log_std to prevent numerical instability
        c_continuous_logstd = torch.clamp(c_continuous_logstd, -10, 10)
        c_continuous_std = torch.exp(c_continuous_logstd)
        
        # Gaussian NLL: 0.5 * log(2*pi*std^2) + (x - mu)^2 / (2*std^2)
        continuous_loss = 0.5 * torch.sum(
            c_continuous_logstd + 
            (c_continuous - c_continuous_mu) ** 2 / (c_continuous_std ** 2 + 1e-8),
            dim=1
        ).mean()
        
        # Discrete code loss (Cross-entropy)
        c_discrete = c_discrete.squeeze(-1).squeeze(-1)  # (N, num_discrete)
        c_discrete_labels = torch.argmax(c_discrete, dim=1)  # Convert one-hot to labels
        discrete_loss = F.cross_entropy(c_discrete_logits, c_discrete_labels)
        
        return continuous_loss, discrete_loss

    def sample_codes(self, batch_size):
        """Sample latent codes"""
        # Random noise z ~ N(0, 1)
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        
        # Continuous codes c ~ Uniform(-1, 1)
        c_continuous = torch.rand(batch_size, self.num_continuous, 1, 1).to(self.device) * 2 - 1
        
        # Discrete codes c ~ Categorical(uniform)
        c_discrete_labels = torch.randint(0, self.num_discrete, (batch_size,)).to(self.device)
        c_discrete = F.one_hot(c_discrete_labels, num_classes=self.num_discrete).float()
        c_discrete = c_discrete.unsqueeze(-1).unsqueeze(-1)  # (N, num_discrete, 1, 1)
        
        return z, c_continuous, c_discrete

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        batch_size = images.size(0)

        # (1) Update Discriminator
        real_logits, _, _, _ = self.d_model(images)
        
        z, c_continuous, c_discrete = self.sample_codes(batch_size)
        fake_images = self.g_model(z, c_continuous, c_discrete).detach()
        fake_logits, _, _, _ = self.d_model(fake_images)

        d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)
        d_loss = d_real_loss + d_fake_loss

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator and Q Network
        z, c_continuous, c_discrete = self.sample_codes(batch_size)
        fake_images = self.g_model(z, c_continuous, c_discrete)
        fake_logits, c_continuous_mu, c_continuous_logstd, c_discrete_logits = self.d_model(fake_images)
        
        g_loss = self.g_loss_fn(fake_logits)
        continuous_loss, discrete_loss = self.mutual_info_loss(
            c_continuous, c_discrete, c_continuous_mu, c_continuous_logstd, c_discrete_logits
        )
        
        # Total generator loss with mutual information
        total_g_loss = g_loss + self.lambda_continuous * continuous_loss + self.lambda_discrete * discrete_loss

        self.g_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        total_g_loss.backward()
        self.g_optimizer.step()
        self.q_optimizer.step()

        return dict(
            d_loss=d_loss.detach().cpu().item(),
            # real_loss=d_real_loss.detach().cpu().item(),
            # fake_loss=d_fake_loss.detach().cpu().item(),
            g_loss=g_loss.detach().cpu().item(),
            continuous_loss=continuous_loss.detach().cpu().item(),
            discrete_loss=discrete_loss.detach().cpu().item(),
        )


@torch.no_grad()
def create_infogan_images(generator, noises, c_continuous, c_discrete):
    """
    Args:
        generator: InfoGAN generator
        noises: (N, latent_dim, 1, 1) numpy array
        c_continuous: (N, num_continuous, 1, 1) numpy array
        c_discrete: (N, num_discrete, 1, 1) numpy array
    
    Returns:
        images: (N, H, W, C) numpy array with values in [0, 1]
    """
    device = next(generator.parameters()).device
    generator.eval()
    
    z = torch.from_numpy(noises).float().to(device)
    c_cont = torch.from_numpy(c_continuous).float().to(device)
    c_disc = torch.from_numpy(c_discrete).float().to(device)
    
    fake_images = generator(z, c_cont, c_disc).cpu()
    
    # Denormalize from [-1, 1] to [0, 1]
    images = (fake_images + 1) / 2
    images = images.permute(0, 2, 3, 1).numpy()
    
    generator.train()
    return images


if __name__ == "__main__":

    set_seed(42)

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
    train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train", transform=transform), batch_size=128)

    latent_dim = 62
    num_continuous = 2
    num_discrete = 10
    
    discriminator = InfoDiscriminator32(num_continuous=num_continuous, num_discrete=num_discrete, in_channels=3, base=64)
    generator = InfoGenerator32(latent_dim=latent_dim, num_continuous=num_continuous, num_discrete=num_discrete, out_channels=3, base=64)
    gan = InfoGAN(discriminator, generator, latent_dim=latent_dim, num_continuous=num_continuous, 
                  num_discrete=num_discrete, lambda_continuous=1.0, lambda_discrete=1.0)
    
    # Create fixed codes for visualization
    # Vary continuous codes while fixing discrete code
    n_samples = 10
    n_variations = 10
    
    # Fixed noise and discrete code
    fixed_z = np.random.normal(size=(n_samples * n_variations, latent_dim, 1, 1))
    fixed_discrete = np.zeros((n_samples * n_variations, num_discrete, 1, 1))
    for i in range(n_samples):
        fixed_discrete[i*n_variations:(i+1)*n_variations, i % num_discrete, 0, 0] = 1
    
    # Vary first continuous code from -2 to 2
    c_continuous = np.zeros((n_samples * n_variations, num_continuous, 1, 1))
    for i in range(n_samples):
        c_continuous[i*n_variations:(i+1)*n_variations, 0, 0, 0] = np.linspace(-2, 2, n_variations)

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

        images = create_infogan_images(generator, fixed_z, c_continuous, fixed_discrete)
        image_path = f"./outputs/{filename}-epoch{epoch}.png"
        plot_images(*images, ncols=n_variations, xunit=1, yunit=1, save_path=image_path)
        
