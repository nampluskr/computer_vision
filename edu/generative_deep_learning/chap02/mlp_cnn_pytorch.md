```python
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from torchmetrics import Accuracy
```

```python
def get_train_loader(dataset, batch_size, collate_fn=None):
    return DataLoader(dataset, batch_size, shuffle=True, # drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=False)

def get_test_loader(dataset, batch_size, collate_fn=None):
    return DataLoader(dataset, batch_size, shuffle=False, # drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=False)

class CIFAR10(Dataset):
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root_dir, split, transform=None):
        self.num_classes = len(self.CLASS_NAMES)
        self.transform = transform or T.Compose([T.ToPILImage(), T.ToTensor()])
        self.images = []
        self.labels = []

        data_dir = os.path.join(root_dir, "cifar-10-batches-py")
        if split == "train":
            data_batches = [f"data_batch_{i}" for i in range(1, 6)]
        elif split == "test":
            data_batches = ["test_batch"]
        else:
            raise ValueError("split must be 'train' or 'test'")

        for batch_name in data_batches:
            batch_path = os.path.join(data_dir, batch_name)
            with open(batch_path, "rb") as file:
                batch = pickle.load(file, encoding='bytes')
                self.images.append(batch[b'data'])
                self.labels.extend(batch[b'labels'])

        self.images = np.vstack(self.images)
        self.images = self.images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx]).long()
        class_name = self.CLASS_NAMES[label.item()]
        return dict(image=image, label=label, class_name=class_name)
```

```python
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
```

```python
def show_predictions(model, loader, class_names, n_show=10, seed=42):
    model.eval()
    all_images = []     # (N, H, W, C)
    all_preds  = []     # (N, 10)
    all_labels = []     # (N,)

    with torch.no_grad():
        for batch in loader:
            images   = batch["image"].to(device)    # (B, C, H, W)
            labels = batch["label"].to(device)      # (B,)

            logits = model(images)                  # (B, 10)
            preds  = F.softmax(logits, dim=1)       # (B, 10)

            all_images.append(images.cpu().permute(0, 2, 3, 1).numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    images_np = np.concatenate(all_images, axis=0)  # (N, H, W, C)
    preds_np  = np.concatenate(all_preds,  axis=0)  # (N, 10)
    labels_np = np.concatenate(all_labels, axis=0)  # (N,)

    pred_idx   = np.argmax(preds_np, axis=1)
    pred_names = np.array(class_names)[pred_idx]
    true_names = np.array(class_names)[labels_np]

    N = len(images_np)
    indices = np.random.choice(N, size=n_show, replace=False)
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, n_show, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, f'pred = {pred_names[idx]}', fontsize=10,
            ha='center', transform=ax.transAxes)
        ax.text(0.5, -0.70, f'true = {true_names[idx]}', fontsize=10,
            ha='center', transform=ax.transAxes)
        ax.imshow(images_np[idx])
    plt.show()
```

```python
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
```

```python
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
```

```python
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
```

```python
show_predictions(model, test_loader, CIFAR10.CLASS_NAMES, n_show=10)
```

### CNN

```python
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
    def __init__(self, num_classes=10):
        super().__init__()

        self.block1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(32, 32, kernel_size=3, stride=2, padding=1)
        self.block3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.block4 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc(x)
        logits = self.classifier(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=10).to(device)
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

model.eval()
results = {"loss": 0.0, "acc": 0.0}
total = 0
for batch in test_loader:
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
    results["acc"] += acc_metric(preds, labels) * batch_size

print(f"loss={results['loss']/total:.3f}, acc={results['acc']/total:.3f}")

show_predictions(model, test_loader, CIFAR10.CLASS_NAMES, n_show=10)
```
