import os
import numpy as np
import random
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


#################################################################
## Data Loading
#################################################################

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    data = np.pad(data, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    return data


def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


class MNIST(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        super().__init__()
        if split == "train":
            self.images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
            self.labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
        elif split == "test":
            self.images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
            self.labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
        else:
            raise ValueError(">> split must be train or test!")

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()
        return image, label


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, hidden_channels=(32, 64)):
        super().__init__()
        c1, c2 = hidden_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


import sys
from tqdm import tqdm
from torchmetrics import Accuracy

class Classifier:
    def __init__(self, model, optimizer=None, loss_fn=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(model, lr=0.001)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.acc_metric = Accuracy(task="multiclass", num_classes=10).to(self.device)

    def train_step(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return dict(loss=loss, acc=acc)

    def eval_step(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(logits, labels)
        return dict(loss=loss, acc=acc)


def train(model, dataloader):
    model.model.train()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Train", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for images, labels in progress_bar:
            batch_size = images.shape[0]
            total += batch_size

            outputs = model.train_step(images, labels)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def evaluate(model, dataloader):
    model.model.eval()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Evaluate", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for images, labels in progress_bar:
            batch_size = images.shape[0]
            total += batch_size

            outputs = model.eval_step(images, labels)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def fit(model, train_loader, num_epochs, valid_loader=None):
    history = {"train": {}, "valid": {}}
    for epoch in range(1, num_epochs + 1):
        epoch_info = f"[{epoch:3d}/{num_epochs}]"
        train_results = train(model, train_loader)
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in train_results.items()])

        for name, value in train_results.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)
            valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in valid_results.items()])

            for name, value in valid_results.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)
            print(f"{epoch_info} {train_info} | (val) {valid_info}")
        else:
            print(f"{epoch_info} {train_info}")

    return history


if __name__ == "__main__":

    if 1:
        print("\n** MNIST Classification Using MLP **")
        set_seed(42)

        ## Data Loading
        transform = T.ToTensor()    # float32, [0, 1]
        data_dir = "/mnt/d/datasets/mnist"
        train_dataset = MNIST(data_dir, split="train", transform=transform)
        test_dataset = MNIST(data_dir, split="test", transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        model = MLP(input_size=32*32, hidden_size=100, output_size=10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        clf = Classifier(model, optimizer=optimizer)

        ## Training Model
        print("\n>> Training start ...")
        history = fit(clf, train_loader, num_epochs=10, valid_loader=test_loader)

        ## Evaluation
        results = evaluate(clf, test_loader)
        print(f"\n>> Evaluation: loss:{results['loss']:.3f} acc:{results['acc']:.3f}")

    if 1:
        print("\n** MNIST Classification Using CNN **")
        set_seed(42)

        ## Data Loading
        transform = T.ToTensor()    # float32, [0, 1]
        data_dir = "/mnt/d/datasets/mnist"
        train_dataset = MNIST(data_dir, split="train", transform=transform)
        test_dataset = MNIST(data_dir, split="test", transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        model = CNN(in_channels=1, num_classes=10, hidden_channels=(16, 32))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        clf = Classifier(model, optimizer=optimizer)

        ## Training Model
        print("\n>> Training start ...")
        history = fit(clf, train_loader, num_epochs=10, valid_loader=test_loader)

        ## Evaluation
        results = evaluate(clf, test_loader)
        print(f"\n>> Evaluation: loss:{results['loss']:.3f} acc:{results['acc']:.3f}")
