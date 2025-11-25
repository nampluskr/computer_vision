import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T

from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy

from datasets import CIFAR10, FashionMNIST, MNIST
from datasets import get_train_loader, get_test_loader
from utils import set_seed


#####################################################################
# Encoder or Feature Extractor
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


class Encoder32(nn.Module):
    def __init__(self, in_channels=3, latent_dim=10, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
        )
        self.final = nn.Conv2d(base*4, latent_dim, kernel_size=4, stride=1, padding=0, bias=False)

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
        logits = self.final(x).view(-1, self.latent_dim)
        return logits


#####################################################################
# Classifiers
#####################################################################

class MulticlassClassifier(nn.Module):
    def __init__(self, encoder, device=None):
        super().__init__()
        self.num_classes = encoder.latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

    def train_step(self, batch):
        images = batch["image"].to(self.device)     # (B, in_channels, H, W)
        labels = batch["label"].to(self.device)     # (B,) long
        logits = self.encoder(images)
        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = self.acc_metric(logits, labels)
        return dict(loss=loss.item(), acc=acc.item())

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)     # (B, in_channels, H, W)
        labels = batch["label"].to(self.device)     # (B,) long
        logits = self.encoder(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(logits, labels)
        return dict(loss=loss.item(), acc=acc.item())


class BinaryClassifier(nn.Module):
    def __init__(self, encoder, device=None):
        super().__init__()
        self.num_classes = encoder.latent_dim       # = 1
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_metric = BinaryAccuracy().to(self.device)

    def train_step(self, batch):
        images = batch["image"].to(self.device)     # (B, in_channels, H, W)
        labels = batch["label"].to(self.device)     # (B,) long
        labels = labels.float().unsqueeze(-1)       # (B, 1) float

        logits = self.encoder(images)
        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = self.acc_metric(logits, labels)
        return dict(loss=loss.item(), acc=acc.item())

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)     # (B, in_channels, H, W)
        labels = batch["label"].to(self.device)     # (B,) long
        labels = labels.float().unsqueeze(-1)       # (B, 1) float

        logits = self.encoder(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(logits, labels)
        return dict(loss=loss.item(), acc=acc.item())
