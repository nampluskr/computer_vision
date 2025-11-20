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
#  Utilities
# --------------------------------------------------------------

class PixelNorm(nn.Module):
    """Pixel-wise feature normalization used in ProGAN"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for discriminator"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Calculate std across batch
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8)
        # Average std across channels and spatial dimensions
        std_mean = std.mean().expand(batch_size, 1, height, width)
        # Concatenate as new channel
        return torch.cat([x, std_mean], dim=1)


class EqualizedConv2d(nn.Module):
    """Equalized learning rate convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # He initialization scale
        self.scale = np.sqrt(2 / (in_channels * kernel_size * kernel_size))

        # Initialize with N(0, 1)
        nn.init.normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1)


class EqualizedConvTranspose2d(nn.Module):
    """Equalized learning rate transposed convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # He initialization scale
        self.scale = np.sqrt(2 / (in_channels * kernel_size * kernel_size))

        # Initialize with N(0, 1)
        nn.init.normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1)


# --------------------------------------------------------------
#  Generator Blocks
# --------------------------------------------------------------

class GBlock(nn.Module):
    """Generator block with upsampling"""
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = EqualizedConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pixelnorm = PixelNorm() if use_pixelnorm else None
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        if self.pixelnorm:
            x = self.pixelnorm(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        if self.pixelnorm:
            x = self.pixelnorm(x)
        x = self.lrelu(x)
        return x


class ToRGB(nn.Module):
    """Convert feature maps to RGB"""
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class ProGenerator(nn.Module):
    """Progressive GAN Generator"""
    def __init__(self, latent_dim=512, base=512, max_resolution=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_resolution = max_resolution

        # Initial 4x4 block
        self.initial = nn.Sequential(
            EqualizedConvTranspose2d(latent_dim, base, kernel_size=4, stride=1, padding=0),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(base, base, kernel_size=3, stride=1, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        )

        # Progressive blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.blocks = nn.ModuleList([
            GBlock(base, base),      # 4x4 -> 8x8
            GBlock(base, base // 2), # 8x8 -> 16x16
            GBlock(base // 2, base // 4), # 16x16 -> 32x32
            GBlock(base // 4, base // 8), # 32x32 -> 64x64
        ])

        # RGB output layers for each resolution
        self.to_rgbs = nn.ModuleList([
            ToRGB(base),          # 4x4
            ToRGB(base),          # 8x8
            ToRGB(base // 2),     # 16x16
            ToRGB(base // 4),     # 32x32
            ToRGB(base // 8),     # 64x64
        ])

    def forward(self, z, alpha=1.0, steps=4):
        """
        Args:
            z: latent vector (N, latent_dim, 1, 1)
            alpha: blending factor for fade-in [0, 1]
            steps: number of resolution steps (0=4x4, 1=8x8, 2=16x16, 3=32x32, 4=64x64)
        """
        x = self.initial(z)

        if steps == 0:
            return self.to_rgbs[0](x)

        # Build up to previous resolution
        for i in range(steps - 1):
            x = self.blocks[i](x)

        # Fade-in: blend current resolution with upsampled previous resolution
        if alpha < 1.0:
            # Previous resolution RGB output
            rgb_prev = self.to_rgbs[steps - 1](x)
            # Upsample to match current resolution
            rgb_prev_upsampled = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')

            # Current resolution
            x_curr = self.blocks[steps - 1](x)
            rgb_curr = self.to_rgbs[steps](x_curr)

            # Blend
            return alpha * rgb_curr + (1 - alpha) * rgb_prev_upsampled
        else:
            # No fade-in, just process normally
            x = self.blocks[steps - 1](x)
            return self.to_rgbs[steps](x)


# --------------------------------------------------------------
#  Discriminator Blocks
# --------------------------------------------------------------

class DBlock(nn.Module):
    """Discriminator block with downsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        return x


class FromRGB(nn.Module):
    """Convert RGB to feature maps"""
    def __init__(self, out_channels, in_channels=3):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class ProDiscriminator(nn.Module):
    """Progressive GAN Discriminator"""
    def __init__(self, base=512, max_resolution=64):
        super().__init__()
        self.max_resolution = max_resolution

        # RGB input layers for each resolution
        self.from_rgbs = nn.ModuleList([
            FromRGB(base),          # 4x4
            FromRGB(base),          # 8x8
            FromRGB(base // 2),     # 16x16
            FromRGB(base // 4),     # 32x32
            FromRGB(base // 8),     # 64x64
        ])

        # Progressive blocks: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.blocks = nn.ModuleList([
            DBlock(base // 8, base // 4), # 64x64 -> 32x32
            DBlock(base // 4, base // 2), # 32x32 -> 16x16
            DBlock(base // 2, base),      # 16x16 -> 8x8
            DBlock(base, base),           # 8x8 -> 4x4
        ])

        # Final block with minibatch std
        self.minibatch_std = MinibatchStdDev()
        self.final = nn.Sequential(
            EqualizedConv2d(base + 1, base, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(base, base, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(base, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, alpha=1.0, steps=4):
        """
        Args:
            x: input image
            alpha: blending factor for fade-in [0, 1]
            steps: number of resolution steps (0=4x4, 1=8x8, 2=16x16, 3=32x32, 4=64x64)
        """
        if steps == 0:
            # 4x4 resolution
            x = self.from_rgbs[0](x)
        elif alpha < 1.0:
            # Fade-in: blend downsampled current with previous resolution
            # Downsampled current resolution
            x_down = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_prev = self.from_rgbs[steps - 1](x_down)

            # Current resolution
            x_curr = self.from_rgbs[steps](x)
            x_curr = self.blocks[4 - steps](x_curr)

            # Blend
            x = alpha * x_curr + (1 - alpha) * x_prev
        else:
            # No fade-in
            x = self.from_rgbs[steps](x)
            x = self.blocks[4 - steps](x)

        # Process remaining blocks down to 4x4
        for i in range(4 - steps + 1, 4):
            x = self.blocks[i](x)

        # Final layers
        x = self.minibatch_std(x)
        logits = self.final(x).view(-1, 1)
        return logits


# --------------------------------------------------------------
#  ProGAN with WGAN-GP Loss
# --------------------------------------------------------------

class ProGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=512, device=None,
                 current_resolution=32, lambda_gp=10.0):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = discriminator.to(self.device)
        self.g_model = generator.to(self.device)

        self.d_optimizer = optim.Adam(self.d_model.parameters(), lr=1e-3, betas=(0.0, 0.99))
        self.g_optimizer = optim.Adam(self.g_model.parameters(), lr=1e-3, betas=(0.0, 0.99))

        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp

        # Progressive training parameters
        self.current_resolution = current_resolution
        self.alpha = 1.0  # Fade-in parameter
        self.steps = self.resolution_to_steps(current_resolution)

    def resolution_to_steps(self, resolution):
        """Convert resolution to step number"""
        res_to_step = {4: 0, 8: 1, 16: 2, 32: 3, 64: 4}
        return res_to_step.get(resolution, 3)

    def set_alpha(self, alpha):
        """Set fade-in parameter"""
        self.alpha = min(max(alpha, 0.0), 1.0)

    def grow_network(self, new_resolution):
        """Transition to higher resolution"""
        self.current_resolution = new_resolution
        self.steps = self.resolution_to_steps(new_resolution)
        self.alpha = 0.0  # Start fade-in from 0

    def gradient_penalty(self, real_images, fake_images):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_images.size(0)

        # Random weight for interpolation
        epsilon = torch.rand(batch_size, 1, 1, 1).to(self.device)

        # Interpolated images
        interpolated = epsilon * real_images + (1 - epsilon) * fake_images
        interpolated.requires_grad_(True)

        # Get discriminator output
        d_interpolated = self.d_model(interpolated, self.alpha, self.steps)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)

        return penalty

    def train_step(self, batch):
        images = batch["image"].to(self.device)

        # Resize images to current resolution if needed
        if images.size(-1) != self.current_resolution:
            images = F.interpolate(images, size=self.current_resolution, mode='bilinear', align_corners=False)

        batch_size = images.size(0)

        # (1) Update Discriminator multiple times
        for _ in range(1):  # Can increase to 5 for better stability
            real_logits = self.d_model(images, self.alpha, self.steps)

            noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            fake_images = self.g_model(noises, self.alpha, self.steps).detach()
            fake_logits = self.d_model(fake_images, self.alpha, self.steps)

            # WGAN loss
            d_real_loss = -torch.mean(real_logits)
            d_fake_loss = torch.mean(fake_logits)

            # Gradient penalty
            gp = self.gradient_penalty(images, fake_images)

            d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gp

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.g_model(noises, self.alpha, self.steps)
        fake_logits = self.d_model(fake_images, self.alpha, self.steps)

        g_loss = -torch.mean(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.detach().cpu().item(),
            real_loss=d_real_loss.detach().cpu().item(),
            fake_loss=d_fake_loss.detach().cpu().item(),
            g_loss=g_loss.detach().cpu().item(),
            gp=gp.detach().cpu().item(),
        )


def create_progan_images(generator, z_sample, alpha=1.0, steps=4):
    """Generate images for ProGAN visualization"""
    device = next(generator.parameters()).device
    generator.eval()

    with torch.no_grad():
        z_tensor = torch.from_numpy(z_sample).float().to(device)
        outputs = generator(z_tensor, alpha=alpha, steps=steps).cpu()

        images = (outputs + 1) / 2
        images = images.permute(0, 2, 3, 1).numpy()

    generator.train()
    return [images[i] for i in range(len(images))]


if __name__ == "__main__":

    set_seed(42)

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
    train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train", transform=transform), batch_size=64)

    latent_dim = 512

    discriminator = ProDiscriminator(base=512, max_resolution=64)
    generator = ProGenerator(latent_dim=latent_dim, base=512, max_resolution=64)

    # Fixed noise for visualization
    noises = np.random.normal(size=(50, latent_dim, 1, 1))

    filename = os.path.splitext(os.path.basename(__file__))[0]
    total_history = {}
    epoch = 0

    # Progressive training schedule
    # Phase 1: 4x4 stabilization
    print("=" * 50)
    print("Phase 1: Training at 4x4 resolution")
    print("=" * 50)
    gan = ProGAN(discriminator, generator, latent_dim=latent_dim, current_resolution=4)
    num_epochs = 3
    for _ in range(1):
        history = fit(gan, train_loader, num_epochs=num_epochs)
        epoch += num_epochs
        for split_name, metrics in history.items():
            total_history.setdefault(split_name, {})
            for metric_name, metric_values in metrics.items():
                total_history[split_name].setdefault(metric_name, [])
                total_history[split_name][metric_name].extend(metric_values)

        images = create_progan_images(generator, noises, alpha=1.0, steps=0)
        image_path = f"./outputs/{filename}-4x4-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)

    # Phase 2: 8x8 fade-in and stabilization
    print("=" * 50)
    print("Phase 2: Growing to 8x8 resolution")
    print("=" * 50)
    gan.grow_network(8)
    num_epochs = 3
    total_iters = len(train_loader) * num_epochs
    for iter_idx in range(1):
        # Gradually increase alpha during fade-in
        for i in range(num_epochs):
            gan.set_alpha((i + 1) / num_epochs)
            history = fit(gan, train_loader, num_epochs=1)
            epoch += 1
            for split_name, metrics in history.items():
                total_history.setdefault(split_name, {})
                for metric_name, metric_values in metrics.items():
                    total_history[split_name].setdefault(metric_name, [])
                    total_history[split_name][metric_name].extend(metric_values)

        images = create_progan_images(generator, noises, alpha=1.0, steps=1)
        image_path = f"./outputs/{filename}-8x8-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)

    # Phase 3: 16x16 fade-in and stabilization
    print("=" * 50)
    print("Phase 3: Growing to 16x16 resolution")
    print("=" * 50)
    gan.grow_network(16)
    num_epochs = 3
    for iter_idx in range(1):
        for i in range(num_epochs):
            gan.set_alpha((i + 1) / num_epochs)
            history = fit(gan, train_loader, num_epochs=1)
            epoch += 1
            for split_name, metrics in history.items():
                total_history.setdefault(split_name, {})
                for metric_name, metric_values in metrics.items():
                    total_history[split_name].setdefault(metric_name, [])
                    total_history[split_name][metric_name].extend(metric_values)

        images = create_progan_images(generator, noises, alpha=1.0, steps=2)
        image_path = f"./outputs/{filename}-16x16-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)

    # Phase 4: 32x32 fade-in and stabilization
    print("=" * 50)
    print("Phase 4: Growing to 32x32 resolution")
    print("=" * 50)
    gan.grow_network(32)
    num_epochs = 5
    for iter_idx in range(2):
        for i in range(num_epochs):
            gan.set_alpha(min((i + 1) / 3, 1.0))  # Fade-in over 3 epochs
            history = fit(gan, train_loader, num_epochs=1)
            epoch += 1
            for split_name, metrics in history.items():
                total_history.setdefault(split_name, {})
                for metric_name, metric_values in metrics.items():
                    total_history[split_name].setdefault(metric_name, [])
                    total_history[split_name][metric_name].extend(metric_values)

        images = create_progan_images(generator, noises, alpha=1.0, steps=3)
        image_path = f"./outputs/{filename}-32x32-epoch{epoch}.png"
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)

    # Plot training history
    # plot_history(total_history, save_path=f"./outputs/{filename}-history.png")
