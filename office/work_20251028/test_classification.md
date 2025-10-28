### MNIST

```python
import torch
import torch.nn as nn

def train(model, train_loader, loss_fn, optimizer, device):
    model.train() 
    results = {"loss": 0.0, "acc": 0.0}
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size

        results["loss"] += loss.item() * batch_size 
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            results["acc"] += (preds == labels).sum().item()

    return {name: value / total_samples for name, value in results.items()}

@torch.no_grad()
def validate(model, train_loader, loss_fn, device):
    model.eval() 
    results = {"loss": 0.0, "acc": 0.0}
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss   = loss_fn(logits, labels)

        batch_size = images.size(0)
        total_samples += batch_size

        results["loss"] += loss.item() * batch_size 
        preds = logits.argmax(dim=1)
        results["acc"] += (preds == labels).sum().item()

    return {name: value / total_samples for name, value in results.items()}
```

```python
import torch
import torch.nn as nn

class EncoderV1(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, latent_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.to_linear(x)
        return x

class DecoderV1(nn.Module):
    def __init__(self, out_channels=1, latent_dim=32):
        super().__init__()
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.Unflatten(1, unflattened_size=(32, 7, 7))
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.from_linear(z)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x
```

```python
class MNISTClassifier(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = EncoderV1(in_channels, latent_dim)
        self.fc = nn.Linear(latent_dim, 10)

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.fc(latent)
        return logits

from datasets import MNIST, get_train_loader, get_test_loader

root_dir = "/home/namu/myspace/NAMU/datasets/mnist"
train_loader = get_train_loader(dataset=MNIST(root_dir, "train"), batch_size=64)
test_loader = get_test_loader(dataset=MNIST(root_dir, "test"), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train_results = train(model, train_loader, loss_fn, optimizer, device)
    print(train_results)
    valid_results = validate(model, test_loader, loss_fn, device)
    print(valid_results)
```

```
{'loss': 0.7384785009168383, 'acc': 0.8029782550693704}
{'loss': 0.3299846656143665, 'acc': 0.9047}
{'loss': 0.31465081834774006, 'acc': 0.9069670490928495}
{'loss': 0.2756643213480711, 'acc': 0.9205}
{'loss': 0.26757812079364235, 'acc': 0.9218416488794023}
{'loss': 0.23825367297828198, 'acc': 0.9303}
{'loss': 0.22018107912615498, 'acc': 0.9365161419423693}
{'loss': 0.1854581127271056, 'acc': 0.947}
{'loss': 0.17443660680669795, 'acc': 0.9496898345784418}
{'loss': 0.14593653291985392, 'acc': 0.9554}
{'loss': 0.13793999456234968, 'acc': 0.9606623532550693}
{'loss': 0.11727944871783257, 'acc': 0.9645}
{'loss': 0.11291639021077972, 'acc': 0.967215848452508}
{'loss': 0.10043203909918666, 'acc': 0.9705}
{'loss': 0.0960552770079868, 'acc': 0.9723685965848452}
{'loss': 0.0854851150766015, 'acc': 0.9744}
{'loss': 0.08386002429398204, 'acc': 0.9753201707577375}
{'loss': 0.08093258517384529, 'acc': 0.975}
{'loss': 0.07566683515240394, 'acc': 0.9777714781216649}
{'loss': 0.0680818398013711, 'acc': 0.9785}
```

### CIFAR10

```python
class EncoderV2(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, latent_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        z = self.to_linear(x)
        return z

class DecoderV2(nn.Module):
    def __init__(self, out_channels=3, latent_dim=128):
        super().__init__()
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 128*4*4),
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nnNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.from_linear(z)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
```

```python
class CIFAR10Classifier(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.encoder = EncoderV2(in_channels, latent_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(latent_dim, 10)

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.dropout(latent)
        logits = self.fc(latent)
        return logits

from datasets import CIFAR10, get_train_loader, get_test_loader

root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=128)
test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10Classifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 20
for epoch in range(1, num_epochs + 1):
    train_results = train(model, train_loader, loss_fn, optimizer, device)
    print(train_results)
    valid_results = validate(model, test_loader, loss_fn, device)
    print(valid_results)
```

```
{'loss': 1.6296982817160777, 'acc': 0.4200520833333333}
{'loss': 1.3805867374420167, 'acc': 0.5114}
{'loss': 1.2844388605692447, 'acc': 0.5445913461538462}
{'loss': 1.227533885192871, 'acc': 0.5633}
{'loss': 1.1414810767540564, 'acc': 0.5975160256410257}
{'loss': 1.1418805681228639, 'acc': 0.593}
{'loss': 1.0431204959368094, 'acc': 0.6338141025641025}
{'loss': 1.0741617532730103, 'acc': 0.6192}
{'loss': 0.9638638684382805, 'acc': 0.6623597756410257}
{'loss': 1.0340501502990722, 'acc': 0.6353}
{'loss': 0.9008583210981809, 'acc': 0.6863782051282051}
{'loss': 1.015845025062561, 'acc': 0.6422}
{'loss': 0.8445059128296681, 'acc': 0.7060096153846154}
{'loss': 1.0026593996047974, 'acc': 0.6433}
{'loss': 0.7951658462866759, 'acc': 0.7249399038461538}
{'loss': 0.9778435571670532, 'acc': 0.6541}
{'loss': 0.7533557440990056, 'acc': 0.7378004807692308}
{'loss': 0.9830520240783691, 'acc': 0.6597}
{'loss': 0.7086528759736281, 'acc': 0.7549679487179487}
{'loss': 0.9750581903457641, 'acc': 0.6611}
{'loss': 0.6718202276871754, 'acc': 0.7707932692307692}
{'loss': 0.9816719265937806, 'acc': 0.6599}
{'loss': 0.635759080067659, 'acc': 0.7841546474358975}
{'loss': 0.9815170744895935, 'acc': 0.6616}
{'loss': 0.6000168334979278, 'acc': 0.7961939102564103}
{'loss': 0.9853920250892639, 'acc': 0.6646}
{'loss': 0.5650644814356779, 'acc': 0.8092347756410256}
{'loss': 1.0169924493789673, 'acc': 0.6616}
{'loss': 0.5352866001618214, 'acc': 0.8213141025641025}
{'loss': 1.038197501373291, 'acc': 0.6558}
{'loss': 0.50302316790972, 'acc': 0.8344951923076923}
{'loss': 1.0392722118377686, 'acc': 0.6585}
{'loss': 0.4733647038539251, 'acc': 0.8448116987179487}
{'loss': 1.0602998303413391, 'acc': 0.657}
{'loss': 0.44482246858951374, 'acc': 0.8551282051282051}
{'loss': 1.0970668395996093, 'acc': 0.6525}
{'loss': 0.4177749480574559, 'acc': 0.8654246794871795}
{'loss': 1.1023150344848633, 'acc': 0.6558}
{'loss': 0.3920073343775211, 'acc': 0.8741386217948718}
{'loss': 1.1263556141853333, 'acc': 0.6593}

```
