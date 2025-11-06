```python
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from scipy.stats import norm

IMAGE_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 128
NUM_FEATURES = 128
Z_DIM = 200
LEARNING_RATE = 0.0005
EPOCHS = 10
BETA = 2000
LOAD_MODEL = False

### Data Loading
from datasets import CelebA, get_train_loader, get_test_loader

root_dir = "/home/namu/myspace/NAMU/datasets/celeba"
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
])

train_loader = get_train_loader(dataset=CelebA(root_dir, "train", transform=transform), batch_size=BATCH_SIZE)
valid_loader = get_test_loader(dataset=CelebA(root_dir, "valid", transform=transform), batch_size=32)
test_loader = get_test_loader(dataset=CelebA(root_dir, "test", transform=transform), batch_size=32)

batch = next(iter(train_loader))
images, labels = batch["image"], batch["label"]
print(f"\ntrain dataset: {len(train_loader.dataset)}, dataloader: {len(train_loader)}")
print(f"train images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")
print(f"train labels: {labels.shape}, {labels.dtype}, {labels.min()}, {labels.max()}")

batch = next(iter(valid_loader))
images, labels = batch["image"], batch["label"]
print(f"\nvalid dataset: {len(valid_loader.dataset)}, dataloader: {len(valid_loader)}")
print(f"valid images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")
print(f"valid labels: {labels.shape}, {labels.dtype}, {labels.min()}, {labels.max()}")

batch = next(iter(test_loader))
images, labels = batch["image"], batch["label"]
print(f"\ntest dataset: {len(test_loader.dataset)}, dataloader: {len(test_loader)}")
print(f"test  images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")
print(f"test  labels: {labels.shape}, {labels.dtype}, {labels.min()}, {labels.max()}")

class Encoder(nn.Module):
    def __init__(self, latent_dim=Z_DIM, in_channels=CHANNELS):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, NUM_FEATURES, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, NUM_FEATURES, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, NUM_FEATURES, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, NUM_FEATURES, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.flatten_size = NUM_FEATURES * (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16)
        self.fc1 = nn.Linear(self.flatten_size, latent_dim)
        self.fc2 = nn.Linear(self.flatten_size, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.flatten_size)
        mu, logvar = self.fc1(x), self.fc2(x)
        latent = self.reparameterize(mu, logvar)
        return latent, mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=Z_DIM, out_channels=CHANNELS):
        super().__init__()

        self.flatten_size = NUM_FEATURES * (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16)
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(NUM_FEATURES, NUM_FEATURES, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(NUM_FEATURES, NUM_FEATURES, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(NUM_FEATURES, NUM_FEATURES, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(NUM_FEATURES, NUM_FEATURES, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(0.2)
        )
        self.deconv5 = nn.ConvTranspose2d(NUM_FEATURES, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, NUM_FEATURES, IMAGE_SIZE // 16, IMAGE_SIZE // 16)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoder = Encoder(latent_dim=200, in_channels=3).to(device)
# decoder = Decoder(latent_dim=200, out_channels=3).to(device)

# images = torch.randn(100, 3, 32, 32).to(device)
# latent, mu, logvar = encoder(images)
# recon = decoder(latent)

# print(f"images: {images.shape}")
# print(f"latent: {latent.shape}")
# print(f"recon:  {recon.shape}")

### Modeling: Variational Autoencoder

class VAE(nn.Module):
    def __init__(self, encoder, decoder, beta=500):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        # self.bce_loss = nn.BCELoss(reduction='none')
        self.bce_loss = nn.MSELoss(reduction='none')

    def forward(self, images):
        latent, mu, logvar = self.encoder(images)
        recon = self.decoder(latent)
        return recon, latent, mu, logvar

    @property
    def device(self):
        return next(self.parameters()).device

    def loss_fn(self, recon, images, mu, logvar):
        bce_pixel = self.bce_loss(recon, images)
        bce = bce_pixel.view(bce_pixel.size(0), -1).mean(dim=1)    # (B,)
        bce = self.beta * bce.mean() 

        kld_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        kld = kld_sample.mean()

        loss =  bce + kld
        return loss, bce, kld

    def train_step(self, batch, optimizer):
        images = batch["image"].to(self.device)
        optimizer.zero_grad()
        recon, latent, mu, logvar = self.forward(images)
        loss, bce, kld = self.loss_fn(recon, images, mu, logvar)
        loss.backward()
        optimizer.step()
        return dict(loss=loss, bce=bce, kld=kld)

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        recon, latent, mu, logvar = self.forward(images)
        loss, bce, kld = self.loss_fn(recon, images, mu, logvar)
        return dict(loss=loss, bce=bce, kld=kld)

    @torch.no_grad()
    def pred_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"]
        recon, latent, mu, logvar = self.forward(images)
        return dict(image=images, label=labels, latent=latent, recon=recon)

from trainer import fit, evaluate, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(latent_dim=Z_DIM, in_channels=CHANNELS).to(device)
decoder = Decoder(latent_dim=Z_DIM, out_channels=CHANNELS).to(device)
model = VAE(encoder, decoder, beta=2000).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

history = fit(model, train_loader, optimizer, num_epochs=10, valid_loader=valid_loader)
evaluate(model, test_loader)

predictions = predict(model, test_loader)
images = predictions["image"]
labels = predictions["label"]
latent = predictions["latent"]
recon  = predictions["recon"]

def display(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None):
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()

display(images)
display(recon)
```
