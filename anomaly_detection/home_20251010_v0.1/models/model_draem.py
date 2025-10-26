"""
- DRAEM (2021): A discriminatively trained reconstruction embedding for surface anomaly detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/draem
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/draem.html
  - https://arxiv.org/abs/2108.07610
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2

from kornia.losses import FocalLoss, SSIMLoss

from .components.sspcab import SSPCAB


#############################################################
# anomalib\models\images\stfpm\loss.py
#############################################################

class DRAEMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l2_loss = nn.modules.loss.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)

    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        return l2_loss_val + ssim_loss_val + focal_loss_val


#############################################################
## anomalib/src/anomalib/models/image/draem/torch_model.py
#############################################################

class DraemModel(nn.Module):
    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        reconstruction = self.reconstructive_subnetwork(batch)
        concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
        prediction = self.discriminative_subnetwork(concatenated_inputs)
        if self.training:
            return reconstruction, prediction

        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


class ReconstructiveSubNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_width: int = 128,
        sspcab: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width, sspcab=sspcab)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 64) -> None:
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        return self.decoder_segment(act1, act2, act3, act4, act5, act6)


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        act5 = self.block5(mp4)
        mp5 = self.mp5(act5)
        act6 = self.block6(mp5)
        return act1, act2, act3, act4, act5, act6


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.up_b = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(
        self,
        act1: torch.Tensor,
        act2: torch.Tensor,
        act3: torch.Tensor,
        act4: torch.Tensor,
        act5: torch.Tensor,
        act6: torch.Tensor,
    ) -> torch.Tensor:
        up_b = self.up_b(act6)
        cat_b = torch.cat((up_b, act5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, act4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, act3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, act2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, act1), dim=1)
        db4 = self.db4(cat4)

        return self.fin_out(db4)


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels: int, base_width: int, sspcab: bool = False) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        if sspcab:
            self.block5 = SSPCAB(base_width * 8)
        else:
            self.block5 = nn.Sequential(
                nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        return self.block5(mp4)


class DecoderReconstructive(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, act5: torch.Tensor) -> torch.Tensor:
        up1 = self.up1(act5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        return self.fin_out(db4)


#############################################################
# Trainer for DRAEM Model
#############################################################
import os
from .components.trainer import BaseTrainer, EarlyStopper
from .components.perlin import PerlinAnomalyGenerator


class DraemTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 sspcab=True, dtd_dir=None):

        if model is None:
            model = DraemModel(sspcab=sspcab)
            if dtd_dir is None:
                from dataloader import get_dataset_dir
                dtd_dir = os.path.join(get_dataset_dir(), "dtd")
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        if loss_fn is None:
            loss_fn = DRAEMLoss()
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # if early_stopper_loss is None:
        #     early_stopper_loss = EarlyStopper(patience=10, min_delta=0.001, mode='min')
        # if early_stopper_auroc is None:
        #     early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max')

        super().__init__(model, optimizer, loss_fn, metrics, device, 
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5

        self.dtd_dir = dtd_dir
        self.augmenter = PerlinAnomalyGenerator(anomaly_source_path=self.dtd_dir, blend_factor=(0.1, 1.0))
        self.sspcab = sspcab

        if self.sspcab:
            self.sspcab_activations = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = 0.1

    def setup_sspcab(self) -> None:
        def get_activation(name: str):
            def hook(_, __, output: torch.Tensor) -> None:
                self.sspcab_activations[name] = output
            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)

        # 1. Anomaly generation
        augmented_images, anomaly_masks = self.augmenter(images)
        augmented_images = augmented_images.to(self.device)
        anomaly_masks = anomaly_masks.to(self.device)

        # 2. Forward pass
        self.optimizer.zero_grad()
        reconstruction, prediction = self.model(augmented_images)

        # 3. Loss 계산
        loss = self.loss_fn(images, reconstruction, anomaly_masks, prediction)

        # 4. SSPCAB loss 추가
        if self.sspcab:
            sspcab_loss = self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"]
            )
            loss = loss + self.sspcab_lambda * sspcab_loss

        # 5. Backward pass
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}