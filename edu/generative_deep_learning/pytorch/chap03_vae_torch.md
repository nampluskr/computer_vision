```python
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from torchmetrics import Accuracy
from scipy.stats import norm

### Data Loading
from datasets_pytorch import FashionMNIST, get_train_loader, get_test_loader

root_dir = "/home/namu/myspace/NAMU/datasets/fashion_mnist"
train_loader = get_train_loader(dataset=FashionMNIST(root_dir, "train"), batch_size=100)
test_loader = get_test_loader(dataset=FashionMNIST(root_dir, "test"), batch_size=64)

batch = next(iter(train_loader))
images, labels = batch["image"], batch["label"]
print(images.shape, images.dtype, images.min(), images.max())
print(labels.shape, labels.dtype, labels.min(), labels.max())

batch = next(iter(test_loader))
images, labels = batch["image"], batch["label"]
print(images.shape, images.dtype, images.min(), images.max())
print(labels.shape, labels.dtype, labels.min(), labels.max())

### Modeling

class Encoder(nn.Module):
    def __init__(self, latent_dim=2, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 4 * 4)
        mu, logvar = self.fc1(x), self.fc2(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=2, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.deconv3 = nn.ConvTranspose2d(32, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder, beta=500):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, images):
        z, mu, logvar = self.encoder(images)
        recon = self.decoder(z)
        return recon, z, mu, logvar

    def loss_fn(self, recon, images, mu, logvar):
        bce_per_pixel = self.bce_loss(recon, images)
        bce_loss = bce_per_pixel.view(bce_per_pixel.size(0), -1).mean(dim=1)    # (B,)
        bce_loss = self.beta * bce_loss.mean() 

        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        kld_loss = kl_per_sample.mean()

        total_loss =  bce_loss + kld_loss
        return total_loss, bce_loss, kld_loss

    def train_step(self, images, optimizer):
        self.train()
        optimizer.zero_grad()
        recon, z, mu, logvar = self(images)
        total_loss, bce_loss, kld_loss = self.loss_fn(recon, images, mu, logvar)
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), bce_loss.item(), kld_loss.item()

    @torch.no_grad()
    def test_step(self, images):
        self.eval()
        recon, z, mu, logvar = self(images)
        total_loss, bce_loss, kld_loss = self.loss_fn(recon, images, mu, logvar)
        return total_loss.item(), bce_loss.item(), kld_loss.item()

### Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(latent_dim=2, in_channels=1).to(device)
decoder = Decoder(latent_dim=2, out_channels=1).to(device)
model = VAE(encoder, decoder, beta=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
model.train()
for epoch in range(1, num_epochs + 1):
    results = {"loss": 0.0, "bce": 0.0, "kld": 0.0}
    total = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        batch_size = images.size(0)
        total += batch_size

        total_loss, recon_loss, kl_loss = model.train_step(images, optimizer)

        results["loss"] += total_loss * batch_size
        results["bce"] += recon_loss * batch_size
        results["kld"] += kl_loss * batch_size

    print(f"[{epoch:2d}/{num_epochs}] "
          f"loss={results['loss']/total:.3f}, "
          f"bce={results['bce']/total:.3f}, "
          f"kld={results['kld']/total:.3f}")

@torch.no_grad()
def predict(model, data, labels=None, device=None):
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_images = []
    all_recons = []
    all_labels = []
    all_latents = []

    if isinstance(data, torch.utils.data.DataLoader):
        for batch in data:
            images = batch["image"].to(device)
            labels = batch["label"]

            z_mean, z_log_var, z = model.encoder(images)
            recons = model.decoder(z)

            all_images.append(images.cpu().permute(0, 2, 3, 1).numpy()) # (B, H, W, C)
            all_recons.append(recons.cpu().permute(0, 2, 3, 1).numpy()) # (B, H, W, C)
            all_labels.append(labels.cpu().numpy())                     # (B,)
            all_latents.append(z.cpu().numpy())                         # (B, latent_dim)
    else:
        labels = np.zeros(len(data)) if labels is None else np.array(labels)
        transform = T.Compose([T.ToPILImage(), T.ToTensor()])
        for image, label in zip(data, labels):
            image = transform(image).unsqueeze(0).to(device)
            z_mean, z_log_var, z = model.encoder(image)
            recon = model.decoder(z)

            all_images.append(image.cpu().permute(0, 2, 3, 1).numpy())
            all_recons.append(recon.cpu().permute(0, 2, 3, 1).numpy())
            all_labels.append(np.array([label])) 
            all_latents.append(z.cpu().numpy())

    images_np = np.concatenate(all_images, axis=0)     # (N, H, W, C)
    recons_np = np.concatenate(all_recons, axis=0)     # (N, H, W, C)
    labels_np = np.concatenate(all_labels, axis=0)     # (N,)
    latents_np = np.concatenate(all_latents, axis=0)   # (N, latent_dim)

    images_np = np.clip(images_np, 0, 1)
    recons_np = np.clip(recons_np, 0, 1)
    return images_np, recons_np, labels_np, latents_np

images, recons, labels, z = predict(model, test_loader)

n_to_predict = -1
example_images = images
example_labels = labels
embeddings = z

plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1],
    cmap="rainbow", c=example_labels, alpha=0.8, s=3)
plt.colorbar()
plt.show()

# Get the range of the existing embeddings
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)

# Sample some points in the latent space
grid_width, grid_height = (6, 3)
sample = np.random.uniform(mins, maxs, size=(grid_width * grid_height, 2))

recons = predict(model, sample)
# figsize = 8
# plt.figure(figsize=(figsize, figsize))

# # ... the original embeddings ...
# plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=2)

# # ... and the newly generated points in the latent space
# plt.scatter(sample[:, 0], sample[:, 1], c="#00B0F0", alpha=1, s=40)
# plt.show()

# # Add underneath a grid of the decoded images
# fig = plt.figure(figsize=(figsize, grid_height * 2))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# for i in range(grid_width * grid_height):
#     ax = fig.add_subplot(grid_height, grid_width, i + 1)
#     ax.axis("off")
#     ax.text(0.5, -0.35, str(np.round(sample[i, :], 1)), 
#             fontsize=10, ha="center", transform=ax.transAxes)
#     ax.imshow(recons[i, :, :], cmap="Greys")
# plt.show()

# Colour the embeddings by their label (clothing type - see table)
figsize = 12
grid_size = 15
plt.figure(figsize=(figsize, figsize))
plt.scatter(embeddings[:, 0], embeddings[:, 1],
    cmap="rainbow", c=example_labels, alpha=0.8, s=20)
plt.colorbar()

x = np.linspace(min(embeddings[:, 0]), max(embeddings[:, 0]), grid_size)
y = np.linspace(max(embeddings[:, 1]), min(embeddings[:, 1]), grid_size)
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

plt.scatter(grid[:, 0], grid[:, 1], c="black", alpha=1, s=10)
plt.show()

fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow(recons[i, :, :], cmap="Greys")
plt.show()

grid_width, grid_height = (6, 3)
z_sample = np.random.normal(size=(grid_width * grid_height, 2))
# recons = decoder.predict(grid)
p = norm.cdf(z)
p_sample = norm.cdf(z_sample)

plt.figure(figsize=(8, 8))
plt.scatter(z[:, 0], z[:, 1], c="black", alpha=0.5, s=2)
plt.scatter(z_sample[:, 0], z_sample[:, 1], c="#00B0F0", alpha=1, s=40)
plt.show()

# Add underneath a grid of the decoded images
fig = plt.figure(figsize=(8, grid_height * 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(0.5, -0.35, str(np.round(z_sample[i, :], 1)), fontsize=10,
        ha="center", transform=ax.transAxes)
    ax.imshow(recons[i, :, :], cmap="Greys")

fig = plt.figure(figsize=(8 * 2, 8))
ax = fig.add_subplot(1, 2, 1)
plot_1 = ax.scatter(z[:, 0], z[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=5)
plt.colorbar(plot_1)
ax = fig.add_subplot(1, 2, 2)
plot_2 = ax.scatter(p[:, 0], p[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=5)
plt.show()

figsize = 12
grid_size = 15
plt.figure(figsize=(figsize, figsize))
plt.scatter(p[:, 0], p[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=30)
plt.colorbar()

x = norm.ppf(np.linspace(0, 1, grid_size))
y = norm.ppf(np.linspace(1, 0, grid_size))
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

# reconstructions = decoder.predict(grid)
plt.show()

fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow(recons[i, :, :], cmap="Greys")

plt.show()


```
