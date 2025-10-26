import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure


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


class Autoencoder(nn.Module):
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
from .components.trainer import BaseTrainer, EarlyStopper

class AutoencoderTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None, backbone_dir=None, 
                 latent_dim=512, img_size=256):

        if model is None:
            super().set_backbone_dir(backbone_dir)
            model = Autoencoder(latent_dim=latent_dim, img_size=img_size)
        if optimizer is None:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
        if loss_fn is None:
            loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # if early_stopper_loss is None:
        #     early_stopper_loss = EarlyStopper(patience=10, min_delta=0.01, mode='min', target_value=0.1)
        # if early_stopper_auroc is None:
        #     early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max', target_value=0.9)
        if metrics is None:
            metrics = {"ssim": SSIMMetric(data_range=2.0, reduction="elementwise_mean")}

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)
        recon, *_ = self.model(images)
        loss = self.loss_fn(recon, images)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            for name, metric_fn in self.metrics.items():
                results[name] = metric_fn(images, recon).item()
        return results