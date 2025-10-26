""" filename: models/autoencoder.py """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.components.trainer import AnomalyTrainer, EarlyStopper


###########################################################
# Conv Block / Deconv Block / Autoencoder
###########################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.deconv_block(x)


class _Autoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        feat_size = img_size // 8
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * feat_size * feat_size, latent_dim),
        )
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * feat_size * feat_size),
            nn.Unflatten(dim=1, unflattened_size=(256, feat_size, feat_size)),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, images):
        latent = self.encoder(images)
        latent = self.to_linear(latent)
        latent = self.from_linear(latent)
        recon = self.decoder(latent)

        if self.training:
            return recon, latent

        anomaly_map = torch.mean((images - recon)**2, dim=1, keepdim=True)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim)
        )
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 16, 16)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, images):
        latent = self.encoder(images)
        latent = self.to_linear(latent)
        latent = self.from_linear(latent)
        recon = self.decoder(latent)

        if self.training:
            return recon, latent

        anomaly_map = torch.mean((images - recon)**2, dim=1, keepdim=True)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


############################################################
# Loss Functions and Metrics
############################################################

class AELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon, original):
        return F.mse_loss(recon, original, reduction=self.reduction)


class AECombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7, ssim_weight=0.3, reduction="mean", data_range: float = 2.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.reduction = reduction
        self.data_range = data_range

        torchmetrics_reduction = {
            "mean": "elementwise_mean",
            "sum": "sum",
            "none": "none",
        }.get(reduction, "elementwise_mean")

        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=self.data_range,
            reduction=torchmetrics_reduction,
        )

    def forward(self, recon, original):
        mse_loss = F.mse_loss(recon, original, reduction=self.reduction)
        ssim_loss = 1.0 - self.ssim_metric(recon, original)
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


class SSIMMetric(nn.Module):
    def __init__(self, data_range=2.0, reduction="mean"):
        super().__init__()
        torchmetrics_reduction = {
            "mean": "elementwise_mean",
            "sum": "sum",
            "none": "none",
        }.get(reduction, "elementwise_mean")

        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            reduction=torchmetrics_reduction,
        )

    def forward(self, preds, targets):
        return self.ssim_metric(preds, targets)


#############################################################
# Trainer for Autoencoder Model
#############################################################

class AutoencoderTrainer(AnomalyTrainer):

    def __init__(self, model, loss_fn=None, device=None, logger=None, learning_rate=1e-3):
        if loss_fn is None:
            loss_fn = AELoss()

        super().__init__(model, loss_fn=loss_fn, device=device, logger=logger)
        
        self.lr = learning_rate
        self.ssim_metric = SSIMMetric(data_range=2.0, reduction='mean').to(self.device)

    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        recon, latent = self.model(images)
        loss = self.loss_fn(recon, images)

        with torch.no_grad():
            ssim_value = self.ssim_metric(recon, images)

        return dict(loss=loss, ssim=ssim_value)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return dict(optimizer=optimizer, scheduler=scheduler)

    def configure_early_stoppers(self):
        return dict(
            train=EarlyStopper(patience=5, mode='min', min_delta=1e-4, monitor='loss'),
            valid=EarlyStopper(patience=5, mode='max', min_delta=1e-3, target_value=0.95, monitor='auroc')
        )