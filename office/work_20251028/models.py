import torch
import torch.nn as nn


#####################################################################
# Simple CNN for MNIST (28, 28)
#####################################################################

class EncoderV1(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
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


#####################################################################
# Simple CNN for CIFAR10 (32, 32)
#####################################################################

class EncoderV2(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

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
            nn.BatchNorm2d(32),
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


#####################################################################
# Simple CNN for Oxford III Pets (224, 224)
#####################################################################

class EncoderV3(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

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
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, latent_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        z = self.to_linear(x)
        return z


class DecoderV3(nn.Module):
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14),
            nn.Unflatten(1, unflattened_size=(256, 14, 14))
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.from_linear(z)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x


#####################################################################
# Classifiers
#####################################################################

from torchmetrics import Accuracy

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.latent_dim = encoder.latent_dim
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.latent_dim, num_classes),
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

    def forward(self, images):
        latent = self.encoder(images)
        logits = self.fc(latent)
        return logits

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        with torch.no_grad():
            acc = self.acc_metric(preds.argmax(dim=1), labels)
        return dict(loss=loss, acc=acc)

    @torch.no_grad()
    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        acc = self.acc_metric(preds.argmax(dim=1), labels)
        return dict(loss=loss, acc=acc)


#####################################################################
# Autoencoder
#####################################################################

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.latent_dim = encoder.latent_dim
        self.loss_fn = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(self.device)

    def forward(self, images):
        latent = self.encoder(images)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        reconstructed, latent = self.forward(images)
        loss = self.loss_fn(reconstructed, images)

        with torch.no_grad():
            ssim = self.ssim_metric(reconstructed, images)
            psnr = self.psnr_metric(reconstructed, images)
        return dict(loss=loss, ssim=ssim, psnr=psnr)

    @torch.no_grad()
    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        reconstructed, latent = self.forward(images)

        loss = self.loss_fn(reconstructed, images)
        ssim = self.ssim_metric(reconstructed, images)
        psnr = self.psnr_metric(reconstructed, images)
        return dict(loss=loss, ssim=ssim, psnr=psnr)
