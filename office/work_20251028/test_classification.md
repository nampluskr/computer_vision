```python
import torch
import torch.nn as nn
from datasets import MNIST, CIFAR10, OxfordPets, get_train_loader, get_test_loader
from models import Classifier, EncoderV1, EncoderV2, EncoderV3
from trainers import train, validate
```

### MNIST

```python
root_dir = "/home/namu/myspace/NAMU/datasets/mnist"
train_loader = get_train_loader(dataset=MNIST(root_dir, "train"), batch_size=128)
test_loader = get_test_loader(dataset=MNIST(root_dir, "test"), batch_size=64)

model = Classifier(encoder=EncoderV1(in_channels=1, latent_dim=128), num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train_results = train(model, train_loader, optimizer)
    valid_results = validate(model, test_loader)

    epoch_info = f"[{epoch:2d}/{num_epochs}]"
    train_info = ", ".join(f"{name}={value:.3f}" for name, value in train_results.items())
    valid_info = ", ".join(f"{name}={value:.3f}" for name, value in valid_results.items())
    print(f"{epoch_info} {train_info} | (val) {valid_info}")
```

### CIFAR10

```python
root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=64)
test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=32)

model = Classifier(encoder=EncoderV2(in_channels=3, latent_dim=128), num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
for epoch in range(1, num_epochs + 1):  
    train_results = train(model, train_loader, optimizer)
    valid_results = validate(model, test_loader)

    epoch_info = f"[{epoch:3d}/{num_epochs}]"
    train_info = ", ".join(f"{name}={value:.3f}" for name, value in train_results.items())
    valid_info = ", ".join(f"{name}={value:.3f}" for name, value in valid_results.items())

    if epoch % 5 == 0:
        print(f"{epoch_info} {train_info} | (val) {valid_info}")
```

### Oxford Pets

```python
root_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"
train_loader = get_train_loader(dataset=OxfordPets(root_dir, "train"), batch_size=32)
test_loader = get_test_loader(dataset=OxfordPets(root_dir, "test"), batch_size=16)

model = Classifier(encoder=EncoderV3(in_channels=3, latent_dim=128), num_classes=37)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
for epoch in range(1, num_epochs + 1):  
    train_results = train(model, train_loader, optimizer)
    valid_results = validate(model, test_loader)

    epoch_info = f"[{epoch:3d}/{num_epochs}]"
    train_info = ", ".join(f"{name}={value:.3f}" for name, value in train_results.items())
    valid_info = ", ".join(f"{name}={value:.3f}" for name, value in valid_results.items())

    if epoch % 5 == 0:
        print(f"{epoch_info} {train_info} | (val) {valid_info}")
```
