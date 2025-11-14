```python
import sys

# COMMON_DIR = "/mnt/d/github/computer_vision/books/generative_deep_learning/pytorch/common"
COMMON_DIR = "/home/namu/myspace/edu/generative_deep_learning/pytorch"
if COMMON_DIR not in sys.path:
    sys.path.append(COMMON_DIR)

from common.datasets import get_train_loader, get_test_loader
from common.utils import set_seed, plot_images, create_images
from common.trainer import fit, evaluate, predict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T

### Data Loading
from common.datasets import CIFAR10

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

class MLP(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.linear1 = nn.Sequential(nn.Linear(32 * 32 * in_channels, 512), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * self.in_channels)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.fc(x)
        return x


from torchmetrics import Accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_channels=1, num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)

num_epochs = 10
model.train()
for epoch in range(1, num_epochs + 1):
    results = {"loss": 0.0, "acc": 0.0}
    total = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        batch_size = images.size(0)
        total += batch_size

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        results["loss"] += loss.item() * batch_size
        with torch.no_grad():
            results["acc"] += acc_metric(preds, labels) * batch_size

    print(f"[{epoch:2d}/{num_epochs}] "
          f"loss={results['loss']/total:.3f}, acc={results['acc']/total:.3f}")


from models.classifier import Classifier

encoder = MLP(in_channels=3, num_classes=10)
model = Classifier(encoder, num_classes=10)
history = fit(model, train_loader, num_epochs=10, valid_loader=test_loader)

from common.utils import plot_history

plot_history(history["train"])
```
