import platform
import os
from glob import glob
import numpy as np
import random
import pandas as pd
from PIL import Image
import gzip
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def get_dataloader_config():
    # print(f">> OS: {platform.system()}")
    if platform.system() == "Windows":
        return {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    else:  # Linux (NAMU)
        # return {"num_workers": 8, "pin_memory": True, "persistent_workers": True}  # WSL2
        return {"num_workers": 8, "pin_memory": True, "persistent_workers": False}  # NAMU


def get_train_loader(dataset, batch_size, collate_fn=None):
    dataloader_config = get_dataloader_config()
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True, **dataloader_config)


def get_test_loader(dataset, batch_size, collate_fn=None):
    dataloader_config = get_dataloader_config()
    return DataLoader(dataset, batch_size, shuffle=False, drop_last=False, **dataloader_config)


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


class MNIST(Dataset):
    CLASS_NAMES = [str(i) for i in range(10)]

    def __init__(self, root_dir, split, transform=None, binary=False):
        self.binary = binary
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
        if self.binary:
            label = torch.tensor(self.labels[idx] % 2).float().unsqueeze(-1)
            class_name = "digit"
        else:
            label = torch.tensor(self.labels[idx]).long()
            class_name = self.CLASS_NAMES[label.item()]
        return dict(image=image, label=label, class_name=class_name)


class FashionMNIST(MNIST):
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
