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
from datasets_pytorch import CIFAR10, get_train_loader, get_test_loader

root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=64)
test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=32)

batch = next(iter(train_loader))
images, labels = batch["image"], batch["label"]
print(images.shape, images.dtype, images.min(), images.max())
print(labels.shape, labels.dtype, labels.min(), labels.max())

batch = next(iter(test_loader))
images, labels = batch["image"], batch["label"]
print(images.shape, images.dtype, images.min(), images.max())
print(labels.shape, labels.dtype, labels.min(), labels.max())

@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    all_images = []     # (N, H, W, C)
    all_preds  = []     # (N, 10)
    all_labels = []     # (N,)

    for batch in dataloader:
        images   = batch["image"].to(device)    # (B, C, H, W)
        labels = batch["label"].to(device)      # (B,)

        logits = model(images)                  # (B, 10)
        preds  = F.softmax(logits, dim=1)       # (B, 10)

        all_images.append(images.cpu().permute(0, 2, 3, 1).numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    images_np = np.concatenate(all_images, axis=0)      # (N, H, W, C)
    preds_np  = np.concatenate(all_preds,  axis=0)      # (N, 10)
    preds_np  = preds_np.argmax(axis=1)                 # (N,)
    labels_np = np.concatenate(all_labels, axis=0)      # (N,)
    return images_np, preds_np, labels_np

### Modeling

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

### Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(num_classes=10).to(device)
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

### Evaluation

model.eval()
results = {"loss": 0.0, "acc": 0.0}
total = 0
for batch in test_loader:
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    batch_size = images.size(0)
    total += batch_size

    preds = model(images)
    loss = loss_fn(preds, labels)

    with torch.no_grad():
        results["loss"] += loss.item() * batch_size
        results["acc"] += acc_metric(preds, labels) * batch_size

print(f"loss={results['loss']/total:.3f}, acc={results['acc']/total:.3f}")

### Inference
class_names = CIFAR10.CLASS_NAMES
images, preds, labels = predict(model, test_loader)

n_show = 10
N = len(images_np)
indices = np.random.choice(N, size=n_show, replace=False)
fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(wspace=0.4, hspace=0.4)

for i, idx in enumerate(indices):
    ax = fig.add_subplot(1, n_show, i + 1)
    ax.axis('off')
    ax.text(0.5, -0.35, f'pred = {class_names[preds[idx]]}', 
        fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.70, f'true = {class_names[labels[idx]]}', 
        fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(images_np[idx])
plt.show()
```
