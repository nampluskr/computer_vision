```python
import os
import numpy as np
import matplotlib.pyplot as plt
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from torchmetrics import Accuracy

## Data Loading
def get_train_loader(dataset, batch_size, collate_fn=None):
    return DataLoader(dataset, batch_size, shuffle=True, # drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=False)


def get_test_loader(dataset, batch_size, collate_fn=None):
    return DataLoader(dataset, batch_size, shuffle=False, # drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=False)


class MNIST(Dataset):
    CLASS_NAMES = [str(i) for i in range(10)]

    def __init__(self, root_dir, split, transform=None):
        self.num_classes = len(self.CLASS_NAMES)
        self.transform = transform or T.Compose([T.ToPILImage(), T.ToTensor()])

        if split == "train":
            images_path = os.path.join(root_dir, 'train-images-idx3-ubyte.gz')
            labels_path = os.path.join(root_dir, 'train-labels-idx1-ubyte.gz')
        elif split == "test":
            images_path = os.path.join(root_dir, 't10k-images-idx3-ubyte.gz')
            labels_path = os.path.join(root_dir, 't10k-labels-idx1-ubyte.gz')
        else:
            raise ValueError("split must be 'train' or 'test'")

        with gzip.open(images_path, 'rb') as file:
            self.images = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
            self.images = self.images.reshape(-1, 28, 28)
            self.images = np.pad(self.images, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)

        with gzip.open(labels_path, 'rb') as file:
            self.labels = np.frombuffer(file.read(), dtype=np.uint8, offset=8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx]).long()
        class_name = self.CLASS_NAMES[label.item()]
        return dict(image=image, label=label, class_name=class_name)


class FashionMNIST(MNIST):
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

root_dir = "/home/namu/myspace/NAMU/datasets/mnist"
train_loader = get_train_loader(dataset=MNIST(root_dir, "train"), batch_size=100)
test_loader = get_test_loader(dataset=MNIST(root_dir, "test"), batch_size=64)

batch = next(iter(train_loader))
images, labels = batch["image"], batch["label"]
print(images.shape, images.dtype, images.min(), images.max())
print(labels.shape, labels.dtype, labels.min(), labels.max())

batch = next(iter(test_loader))
images, labels = batch["image"], batch["label"]
print(images.shape, images.dtype, images.min(), images.max())
print(labels.shape, labels.dtype, labels.min(), labels.max())
```

```python
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.fc_mean = nn.Linear(128*4*4, latent_dim)
        self.fc_log_var = nn.Linear(128*4*4, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128*4*4)
        self.reshape_height = 4
        self.reshape_width = 4

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, self.reshape_height, self.reshape_width)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder, beta=500):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def forward(self, images):
        z_mean, z_log_var, z = self.encoder(images)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def loss_fn(self, images, z_mean, z_log_var, reconstruction):
        recon_loss = F.binary_cross_entropy(reconstruction, images, reduction='sum') / images.size(0)
        recon_loss *= self.beta
        kl_loss = -0.5 * torch.mean(torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1))
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, images, optimizer):
        self.train()
        optimizer.zero_grad()
        z_mean, z_log_var, reconstruction = self(images)
        total_loss, recon_loss, kl_loss = self.loss_fn(images, z_mean, z_log_var, reconstruction)
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), recon_loss.item(), kl_loss.item()

    @torch.no_grad()
    def test_step(self, images):
        self.eval()
        z_mean, z_log_var, reconstruction = self(images)
        total_loss, recon_loss, kl_loss = self.loss_fn(images, z_mean, z_log_var, reconstruction)
        return total_loss.item(), recon_loss.item(), kl_loss.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(latent_dim=2).to(device)
decoder = Decoder(latent_dim=2).to(device)
model = VAE(encoder, decoder, beta=500).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
model.train()
for epoch in range(1, num_epochs + 1):
    results = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
    total = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        batch_size = images.size(0)
        total += batch_size

        total_loss, recon_loss, kl_loss = model.train_step(images, optimizer)

        results["loss"] += total_loss * batch_size
        results["recon_loss"] += recon_loss * batch_size
        results["kl_loss"] += kl_loss * batch_size

    print(f"[{epoch:2d}/{num_epochs}] "
          f"loss={results['loss']/total:.3f}, "
          f"recon_loss={results['recon_loss']/total:.3f}, "
          f"kl_loss={results['kl_loss']/total:.3f}")

n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]

z_mean, z_var, z = encoder(example_images)

plt.figure(figsize=(8, 8))
plt.scatter(z[:, 0], z[:, 1], c="black", alpha=0.5, s=3)
plt.show()
```
