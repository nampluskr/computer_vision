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

### Data Loading
from datasets import CIFAR10, get_train_loader, get_test_loader

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=64)
test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=32)

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

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.generator(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return torch.sigmoid(x)

import os
import sys
from tqdm import tqdm
from torchvision.utils import save_image
from utils import set_seed

## Hyperparameters
set_seed(42)
n_epochs = 10
learning_rate = 2e-4
latent_dim = 100
step_size = 1

n_outputs = 100
output_name = "cifar10_gan"

## Modeling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelG = Generator(latent_dim).to(device)
modelD = Discriminator().to(device)

loss_fn = nn.BCELoss()
optimizerD = torch.optim.Adam(modelD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(modelG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

## Training
fixed_noises = torch.randn(n_outputs, latent_dim, 1, 1).to(device)
output_dir = './output_gan'
output_path = os.path.join(output_dir, f"{output_name}_0.png")
output_images = modelG(fixed_noises)
os.makedirs(output_dir, exist_ok=True)
save_image(output_images, output_path, nrow=10, normalize=True)

for epoch in range(1, n_epochs + 1):
    with tqdm(train_loader, leave=False, file=sys.stdout, dynamic_ncols=True, ascii=True) as pbar:
        train_loss_r, train_loss_f, train_loss_g = 0, 0, 0
        for i, batch in enumerate(pbar):
            real_images = batch["image"]
            batch_size = len(real_images)
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            real_images = real_images.to(device)
            fake_images = modelG(noise)

            ## Training Discriminator
            pred_r = modelD(real_images)
            loss_r = loss_fn(pred_r, real_labels)
            loss_r.backward()

            pred_f = modelD(fake_images.detach())
            loss_f = loss_fn(pred_f, fake_labels)
            loss_f.backward()

            optimizerD.step()
            optimizerD.zero_grad()

            # Training Generator
            pred_g = modelD(fake_images)
            loss_g = loss_fn(pred_g, real_labels)
            loss_g.backward()

            optimizerG.step()
            optimizerG.zero_grad()

            train_loss_r += loss_r.item()
            train_loss_f += loss_f.item()
            train_loss_g += loss_g.item()

            desc = f"[{epoch:3d}/{n_epochs}] loss_r: {train_loss_r/(i + 1):.2e} " \
                   f"loss_f: {train_loss_f/(i + 1):.2e} loss_g: {train_loss_g/(i + 1):.2e}"

            if i % 10 == 0:
                pbar.set_description(desc)

        if epoch % step_size == 0:
            print(desc)
            output_images = modelG(fixed_noises)
            output_path = os.path.join(output_dir, f"{output_name}_{epoch}.png")
            save_image(output_images, output_path, nrow=10, normalize=True)



# [  1/10] loss_r: 1.05e-01 loss_f: 9.08e-01 loss_g: 6.87e-01
# [  2/10] loss_r: 1.44e-01 loss_f: 9.32e-01 loss_g: 6.35e-01
# [  3/10] loss_r: 1.31e-01 loss_f: 9.15e-01 loss_g: 6.44e-01
# [  4/10] loss_r: 1.21e-01 loss_f: 9.03e-01 loss_g: 6.48e-01
# [  5/10] loss_r: 1.10e-01 loss_f: 8.87e-01 loss_g: 6.56e-01
# [  6/10] loss_r: 9.64e-02 loss_f: 8.66e-01 loss_g: 6.66e-01
# [  7/10] loss_r: 1.16e-01 loss_f: 8.89e-01 loss_g: 6.50e-01
# [  8/10] loss_r: 1.35e-01 loss_f: 9.07e-01 loss_g: 6.35e-01
# [  9/10] loss_r: 1.29e-01 loss_f: 9.03e-01 loss_g: 6.43e-01
# [ 10/10] loss_r: 1.24e-01 loss_f: 9.04e-01 loss_g: 6.51e-01
```

```python
from torchmetrics.classification import BinaryAccuracy

class GAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=None):
        super().__init__()
        self.d_model = discriminator
        self.g_model = generator

        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim
        self.loss_fn = nn.BCELoss()
        self.acc_metric = BinaryAccuracy().to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_optimziers(self, d_optimizer, g_optimizer):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, batch):
        batch_size = batch["image"].shape[0]
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        real_labels = torch.ones((batch_size, 1)).to(self.device)
        fake_labels = torch.zeros((batch_size, 1)).to(self.device)

        # Train discriminator
        self.d_optimizer.zero_grad()
        real_images = batch["image"].to(self.device)
        real_preds = self.d_model(real_images)
        d_real_loss = self.loss_fn(real_preds, real_labels)
        d_real_loss.backward()

        fake_images = self.g_model(z).detach()
        fake_preds = self.d_model(fake_images)
        d_fake_loss = self.loss_fn(fake_preds, fake_labels)
        d_fake_loss.backward()
        self.d_optimizer.step()

        with torch.no_grad():
            d_real_acc = self.acc_metric(real_preds, real_labels)
            d_fake_acc = self.acc_metric(fake_preds, fake_labels)

        # Train generator
        self.g_optimizer.zero_grad()
        fake_images = self.g_model(z)
        fake_preds = self.d_model(fake_images)
        g_loss = self.loss_fn(fake_preds, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        with torch.no_grad():
            g_acc = self.acc_metric(fake_preds, real_labels)

        return dict(real_loss=d_real_loss, fake_loss=d_fake_loss, gen_loss=g_loss,
                    real_acc=d_real_acc, fake_acc=d_fake_acc, gen_acc=g_acc)

    def pred_step(self, z):
        pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator().to(device)
generator = Generator(latent_dim=100).to(device)
gan = GAN(discriminator, generator).to(device)

from trainer import fit

history = fit(gan, train_loader, num_epochs=10)

# [  1/10] real_loss:0.261, fake_loss:0.242, gen_loss:4.856, real_acc:0.903, fake_acc:0.911, gen_acc:0.011                                                 
# [  2/10] real_loss:0.315, fake_loss:0.283, gen_loss:3.395, real_acc:0.877, fake_acc:0.883, gen_acc:0.014                                                 
# [  3/10] real_loss:0.274, fake_loss:0.251, gen_loss:3.572, real_acc:0.899, fake_acc:0.899, gen_acc:0.018                                                 
# [  4/10] real_loss:0.206, fake_loss:0.188, gen_loss:3.922, real_acc:0.935, fake_acc:0.932, gen_acc:0.015                                                 
# [  5/10] real_loss:0.170, fake_loss:0.162, gen_loss:4.145, real_acc:0.946, fake_acc:0.947, gen_acc:0.016                                                 
# [  6/10] real_loss:0.174, fake_loss:0.171, gen_loss:4.087, real_acc:0.943, fake_acc:0.945, gen_acc:0.021                                                 
# [  7/10] real_loss:0.189, fake_loss:0.179, gen_loss:4.072, real_acc:0.941, fake_acc:0.945, gen_acc:0.024                                                 
# [  8/10] real_loss:0.167, fake_loss:0.161, gen_loss:4.140, real_acc:0.945, fake_acc:0.947, gen_acc:0.020                                                 
# [  9/10] real_loss:0.180, fake_loss:0.175, gen_loss:4.037, real_acc:0.942, fake_acc:0.944, gen_acc:0.024                                                 
# [ 10/10] real_loss:0.181, fake_loss:0.181, gen_loss:3.987, real_acc:0.943, fake_acc:0.947, gen_acc:0.022                                                 
```
