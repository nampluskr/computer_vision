"""" filename: test_mnist.py """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchmetrics.classification import MulticlassAccuracy
import gzip
import numpy as np
import os
from time import time
from trainer import BaseTrainer, EarlyStopper


#############################################################
# MNIST Dataset
#############################################################

class MNISTDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform or T.ToTensor()

        if split == "train":
            images_file = os.path.join(root_dir, 'train-images-idx3-ubyte.gz')
            labels_file = os.path.join(root_dir, 'train-labels-idx1-ubyte.gz')
        else:
            images_file = os.path.join(root_dir, 't10k-images-idx3-ubyte.gz')
            labels_file = os.path.join(root_dir, 't10k-labels-idx1-ubyte.gz')

        # images
        with gzip.open(images_file, 'rb') as file:
            file.read(16)
            buffer = file.read()
            images = np.frombuffer(buffer, dtype=np.uint8)
            if split == "train":
                images = images.reshape(60000, 28, 28)
            else:
                images = images.reshape(10000, 28, 28)

        # labels
        with gzip.open(labels_file, 'rb') as file:
            file.read(8)
            buffer = file.read()
            labels = np.frombuffer(buffer, dtype=np.uint8)

        # numpy array로 저장 (H, W) 형태로 유지
        self.images = (images.astype(np.float32) / 255.0)  # (N, 28, 28)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx]).long()
        return dict(image=image, label=label)


#############################################################
# Simple CNN Model
#############################################################

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            # Conv2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


#############################################################
# Classification Trainer
#############################################################

class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, num_classes=10, **kwargs):
        super().__init__(model, **kwargs)
        self.acc_metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)

    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.acc_metric(outputs, labels)
        return dict(loss=loss, acc=acc)

    def validation_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.acc_metric(outputs, labels)
        return dict(loss=loss, acc=acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return dict(optimizer=optimizer, scheduler=scheduler)

    def configure_early_stoppers(self):
        return dict(
            train=EarlyStopper(patience=3, mode='min', min_delta=0.001, monitor='loss'),
            valid=EarlyStopper(patience=5, mode='max', min_delta=0.001, target_value=0.99, monitor='acc')
        )



#############################################################
# Main
#############################################################

def main():

    root_dir = "/mnt/d/datasets/mnist"
    batch_size = 128
    num_epochs = 10
    output_dir = "./logs_mnist"

    print("Loading MNIST dataset...")
    train_dataset = MNISTDataset(root_dir, split="train")
    valid_dataset = MNISTDataset(root_dir, split="test")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCNN(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    trainer = ClassificationTrainer(model, num_classes=10, loss_fn=loss_fn)
    trainer.fit(train_loader, num_epochs, valid_loader=valid_loader, 
                output_dir=output_dir, run_name="mnist_cnn")


if __name__ == "__main__":
    main()