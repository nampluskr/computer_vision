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

root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=64)
test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=32)

batch = next(iter(train_loader))
images, labels = batch["image"], batch["label"]
print(f"train images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")
print(f"train labels: {labels.shape}, {labels.dtype}, {labels.min()}, {labels.max()}")

batch = next(iter(test_loader))
images, labels = batch["image"], batch["label"]
print(f"test  images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")
print(f"test  labels: {labels.shape}, {labels.dtype}, {labels.min()}, {labels.max()}")

### Modeling

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.block1 = ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(32, 32, kernel_size=3, stride=2, padding=1)
        self.block3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.block4 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    def train_step(self, batch, optimizer):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        optimizer.zero_grad()
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.softmax(logits, dim=1).argmax(dim=1)
            acc = self.acc_metric(preds, labels)
        return dict(loss=loss, acc=acc)

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        acc = self.acc_metric(preds, labels)
        return dict(loss=loss, acc=acc)
    
    @torch.no_grad()
    def pred_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"]
        logits = self.forward(images)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        return dict(images=images, labels=labels, preds=preds)

from trainer import fit, evaluate, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels=3, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = fit(model, train_loader, optimizer, num_epochs=10, valid_loader=test_loader)

evaluate(model, test_loader)

predictions = predict(model, test_loader)
images = predictions["images"]
labels = predictions["labels"]
preds = predictions["preds"]

class_names = CIFAR10.CLASS_NAMES
num_samples = 10
indices = np.random.choice(len(images), size=num_samples, replace=False)

fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
for i, idx in enumerate(indices):
    axes[i].imshow(images[idx])
    axes[i].set_title(f"{class_names[preds[idx]]}\n({class_names[labels[idx]]})")
    axes[i].axis('off')

fig.tight_layout
plt.show()
```
