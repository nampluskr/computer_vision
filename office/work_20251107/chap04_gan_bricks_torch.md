```python
import os
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from utils import set_seed, show_images

IMAGE_SIZE = 64
CHANNELS = 1
BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 300
LOAD_MODEL = False
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
LEARNING_RATE = 0.0002
NOISE_PARAM = 0.1

SEED = 42
set_seed(SEED)

### Data Loading
from datasets import LegoBricks, get_train_loader, get_test_loader

root_dir = "/home/namu/myspace/NAMU/datasets/lego_bricks"
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

train_loader = get_train_loader(dataset=LegoBricks(root_dir, "train", transform=transform), batch_size=BATCH_SIZE)
test_loader = get_test_loader(dataset=LegoBricks(root_dir, "test", transform=transform), batch_size=64)

batch = next(iter(train_loader))
images, labels = batch["image"], batch["label"]
print(f"\ntrain dataset: {len(train_loader.dataset)}, dataloader: {len(train_loader)}")
print(f"train images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")
print(f"train labels: {labels.shape}, {labels.dtype}, {labels.min()}, {labels.max()}")

batch = next(iter(test_loader))
images, labels = batch["image"], batch["label"]
print(f"\ntest dataset: {len(test_loader.dataset)}, dataloader: {len(test_loader)}")
print(f"test  images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")
print(f"test  labels: {labels.shape}, {labels.dtype}, {labels.min()}, {labels.max()}")

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 1)
        return torch.sigmoid(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, 1)  # (B, Z_DIM) -> (B, Z_DIM, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return torch.tanh(x)

from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy

class DCGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.loss_fn = nn.BCELoss()

        self.d_loss_metric = MeanMetric()
        self.d_real_acc_metric = BinaryAccuracy()
        self.d_fake_acc_metric = BinaryAccuracy()
        self.d_acc_metric = BinaryAccuracy()
        self.g_loss_metric = MeanMetric()
        self.g_acc_metric = BinaryAccuracy()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def metrics(self):
        return [self.d_loss_metric, self.d_real_acc_metric, self.d_fake_acc_metric,
                self.d_acc_metric, self.g_loss_metric, self.g_acc_metric]

    def train_step(self, batch, d_optimizer, g_optimizer):
        self.train()
        real_images = batch["image"].to(self.device)
        batch_size = real_images.shape[0]

        # Sample random points in the latent space
        z = torch.randn(batch_size, self.latent_dim).to(device)

        # Generator produces fake images
        fake_images = self.generator(z)

        # Discriminator forward pass
        real_preds = self.discriminator(real_images)
        fake_preds = self.discriminator(fake_images.detach())  # stop gradient

        # Labels
        real_labels = torch.ones_like(real_preds).to(device)
        fake_labels = torch.zeros_like(fake_preds).to(device)

        # Noisy labels (smoothing)
        real_noisy_labels = real_labels + NOISE_PARAM * torch.rand_like(real_preds)
        fake_noisy_labels = fake_labels - NOISE_PARAM * torch.rand_like(fake_preds)

        # Discriminator loss
        d_real_loss = self.loss_fn(real_preds, real_noisy_labels)
        d_fake_loss = self.loss_fn(fake_preds, fake_noisy_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2.0

        # Update discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Generator forward (new pass to build grad graph)
        g_optimizer.zero_grad()
        fake_images_g = self.generator(z)
        fake_preds_g = self.discriminator(fake_images_g)
        g_loss = self.loss_fn(fake_preds_g, real_labels)  # generator wants to fool D
        g_loss.backward()
        g_optimizer.step()

        # Update metrics
        self.d_loss_metric.update(d_loss)
        self.d_real_acc_metric.update(real_preds, real_labels)
        self.d_fake_acc_metric.update(fake_preds, fake_labels)

        all_preds = torch.cat([real_preds, fake_preds], dim=0)
        all_labels = torch.cat([real_labels, fake_labels], dim=0)
        self.d_acc_metric.update(all_preds, all_labels)
        self.g_loss_metric.update(g_loss)
        self.g_acc_metric.update(fake_preds_g, real_labels)

        return {name: metric.compute() for metric in self.metrics}

```
