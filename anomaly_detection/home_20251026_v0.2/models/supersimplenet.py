"""
- SuperSimpleNet (2024): Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/supersimplenet
  - https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/supersimplenet.html
  - https://github.com/blaz-r/SuperSimpleNet
  - https://arxiv.org/pdf/2408.03143
"""

import math
from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn import Parameter
from torchvision.ops.focal_loss import sigmoid_focal_loss

# from anomalib.data import InferenceBatch
# from anomalib.models.components import GaussianBlur2d, TimmFeatureExtractor
# from anomalib.models.image.supersimplenet.anomaly_generator import AnomalyGenerator

from .components.blur import GaussianBlur2d
from .components.feature_extractor import TimmFeatureExtractor
from .components.perlin import generate_perlin_noise


#####################################################################
# anomalib/src/anomalib/models/image/supersimplenet/anomaly_map.py
#####################################################################

class AnomalyGenerator(nn.Module):
    def __init__(
        self,
        noise_mean: float,
        noise_std: float,
        threshold: float,
    ) -> None:
        super().__init__()

        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.threshold = threshold

    @staticmethod
    def next_power_2(num: int) -> int:
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches: int, height: int, width: int) -> torch.Tensor:
        perlin = []
        for _ in range(batches):
            perlin_height = self.next_power_2(height)
            perlin_width = self.next_power_2(width)

            # keep power of 2 here for reproduction purpose, although this function supports power2 internally
            perlin_noise = generate_perlin_noise(height=perlin_height, width=perlin_width)

            # Rescale with threshold at center if all values are below the threshold
            if not (perlin_noise > self.threshold).any():
                # First normalize to [0,1] range by min-max normalization
                perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
                # Then rescale to [-1, 1] range
                perlin_noise = (perlin_noise * 2) - 1

            # original is power of 2 scale, so fit to our size
            perlin_noise = F.interpolate(
                perlin_noise.reshape(1, 1, perlin_height, perlin_width),
                size=(height, width),
                mode="bilinear",
            )
            # binarize
            thresholded_perlin = torch.where(perlin_noise > self.threshold, 1, 0)

            # 50% of anomaly
            if torch.rand(1).item() > 0.5:
                thresholded_perlin = torch.zeros_like(thresholded_perlin)

            perlin.append(thresholded_perlin)
        return torch.cat(perlin)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = features.shape

        # duplicate
        features = torch.cat((features, features))
        mask = torch.cat((mask, mask))
        labels = torch.cat((labels, labels))

        noise = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=features.shape,
            device=features.device,
            requires_grad=False,
        )

        # mask indicating which regions will have noise applied
        # [B * 2, 1, H, W] initial all masked as anomalous
        noise_mask = torch.ones(
            b * 2,
            1,
            h,
            w,
            device=features.device,
            requires_grad=False,
        )

        # no overlap: don't apply to already anomalous regions (mask=1 -> bad)
        noise_mask = noise_mask * (1 - mask)

        # shape of noise is [B * 2, 1, H, W]
        perlin_mask = self.generate_perlin(b * 2, h, w).to(features.device)
        # only apply where perlin mask is 1
        noise_mask = noise_mask * perlin_mask

        # update gt mask
        mask = mask + noise_mask
        # binarize
        mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

        # make new labels. 1 if any part of mask is 1, 0 otherwise
        new_anomalous = noise_mask.reshape(b * 2, -1).any(dim=1).type(torch.float32)
        labels = labels + new_anomalous
        # binarize
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))

        # apply masked noise
        perturbed = features + noise * noise_mask

        return perturbed, mask, labels

#####################################################################
# anomalib/src/anomalib/models/image/supersimplenet/loss.py
#####################################################################

class SSNLoss(nn.Module):
    def __init__(self, truncation_term: float = 0.5) -> None:
        super().__init__()
        self.focal_loss = partial(sigmoid_focal_loss, alpha=-1, gamma=4.0, reduction="mean")
        self.th = truncation_term

    def trunc_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        normal_scores = pred[target == 0]
        anomalous_scores = pred[target > 0]
        # push normal towards negative numbers
        true_loss = torch.clip(normal_scores + self.th, min=0)
        # push anomalous towards positive numbers
        fake_loss = torch.clip(-anomalous_scores + self.th, min=0)

        true_loss = true_loss.mean() if len(true_loss) else torch.tensor(0)
        fake_loss = fake_loss.mean() if len(fake_loss) else torch.tensor(0)

        return true_loss + fake_loss

    def forward(
        self,
        pred_map: torch.Tensor,
        pred_score: torch.Tensor,
        target_mask: torch.Tensor,
        target_label: torch.Tensor,
    ) -> torch.Tensor:
        map_focal = self.focal_loss(pred_map, target_mask)
        map_trunc_l1 = self.trunc_l1_loss(pred_map, target_mask)
        score_focal = self.focal_loss(pred_score, target_label)

        return map_focal + map_trunc_l1 + score_focal

#####################################################################
# anomalib/src/anomalib/models/image/supersimplenet/torch_model.py
#####################################################################

class SupersimplenetModel(nn.Module):
    def __init__(
        self,
        perlin_threshold: float = 0.2,
        backbone: str = "wide_resnet50_2.tv_in1k",  # IMPORTANT: use .tv weights, not tv2
        layers: list[str] = ["layer2", "layer3"],  # noqa: B006
        stop_grad: bool = True,
        adapt_cls_features: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = UpscalingFeatureExtractor(backbone=backbone, layers=layers)

        channels = self.feature_extractor.get_channels_dim()
        self.adapt_cls_features = adapt_cls_features
        self.adaptor = FeatureAdapter(channels)
        self.segdec = SegmentationDetectionModule(channel_dim=channels, stop_grad=stop_grad)
        self.anomaly_generator = AnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=perlin_threshold)
        self.anomaly_map_generator = AnomalyMapGenerator(sigma=4)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output_size = images.shape[-2:]

        features = self.feature_extractor(images)
        adapted = self.adaptor(features)

        if self.training:
            if masks is None:
                batch_size = images.shape[0]
                feat_h, feat_w = features.shape[-2:]
                masks = torch.zeros(batch_size, 1, feat_h, feat_w, device=images.device, dtype=torch.float32)
            else:
                masks = self.downsample_mask(masks, *features.shape[-2:])

            if labels is None:
                batch_size = images.shape[0]
                labels = torch.zeros(batch_size, device=images.device, dtype=torch.float32)
            else:
                labels = labels.type(torch.float32)

            features, masks, labels = self.anomaly_generator(adapted, masks, labels)
            anomaly_map, anomaly_score = self.segdec(features)
            return anomaly_map, anomaly_score, masks, labels

        anomaly_map, anomaly_score = self.segdec(adapted)
        anomaly_map = self.anomaly_map_generator(anomaly_map, final_size=output_size)

        return dict(anomaly_map=anomaly_map, pred_score=anomaly_score)

    @staticmethod
    def downsample_mask(masks: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        masks = masks.type(torch.float32)
        # best downsampling proposed by DestSeg
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(feat_h, feat_w),
            mode="bilinear",
        )
        return torch.where(
            masks < 0.5,
            torch.zeros_like(masks),
            torch.ones_like(masks),
        )

    # @staticmethod
    # def downsample_mask(masks: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
    #     masks = masks.type(torch.float32)

    #     # Ensure 4D tensor [B, C, H, W]
    #     if masks.ndim == 2:  # [H, W] → [1, 1, H, W]
    #         masks = masks.unsqueeze(0).unsqueeze(0)
    #     elif masks.ndim == 3:  # [B, H, W] → [B, 1, H, W]
    #         masks = masks.unsqueeze(1)
    #     elif masks.ndim == 4:  # Already [B, C, H, W]
    #         if masks.shape[1] != 1:
    #             masks = masks[:, :1, :, :]  # Take first channel only
    #     else:
    #         raise ValueError(f"Unexpected mask dimension: {masks.ndim}, shape: {masks.shape}")

    #     # Downsample
    #     masks = F.interpolate(masks, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
    #     return torch.where(masks < 0.5, torch.zeros_like(masks), torch.ones_like(masks))


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear | nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)


class UpscalingFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: list[str], patch_size: int = 3) -> None:
        super().__init__()

        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            layers=layers,
        )

        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # extract features
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = list(features.values())

        _, _, h, w = features[0].shape
        feature_map = []
        for layer in features:
            # upscale all to 2x the size of the first (largest)
            resized = F.interpolate(
                layer,
                size=(h * 2, w * 2),
                mode="bilinear",
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        return self.pooler(feature_map)

    def get_channels_dim(self) -> int:
        # dryrun
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, 256, 256))
        # sum channels
        return sum(feature.shape[1] for feature in features.values())


class FeatureAdapter(nn.Module):
    def __init__(self, channel_dim: int) -> None:
        super().__init__()
        # linear layer equivalent
        self.projection = nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=1,
            stride=1,
        )
        self.apply(init_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class SegmentationDetectionModule(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        stop_grad: bool = False,
    ) -> None:
        super().__init__()
        self.stop_grad = stop_grad

        # 1x1 conv - linear layer equivalent
        self.seg_head = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=1024,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        # pooling for cls. conv out and map
        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # cls. head conv block
        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_dim + 1,
                out_channels=128,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # cls. head fc block: 128 from dec and 2 from map, * 2 due to max and avg pool
        self.cls_fc = nn.Linear(in_features=128 * 2 + 2, out_features=1)

        self.apply(init_weights)

    def get_params(self) -> tuple[list[Parameter], list[Parameter]]:
        seg_params = list(self.seg_head.parameters())
        dec_params = list(self.cls_conv.parameters()) + list(self.cls_fc.parameters())
        return seg_params, dec_params

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # get anomaly map from seg head
        ano_map = self.seg_head(features)

        map_dec_copy = ano_map
        if self.stop_grad:
            map_dec_copy = map_dec_copy.detach()
        # dec conv layer takes feat + map
        mask_cat = torch.cat((features, map_dec_copy), dim=1)
        dec_out = self.cls_conv(mask_cat)

        # conv block result pooling
        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        # predicted map pooling (and stop grad)
        map_max = self.map_max_pool(ano_map)
        if self.stop_grad:
            map_max = map_max.detach()

        map_avg = self.map_avg_pool(ano_map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        # final dec layer: conv channel max and avg and map max and avg
        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze()
        ano_score = self.cls_fc(dec_cat).squeeze()

        return ano_map, ano_score


class AnomalyMapGenerator(nn.Module):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.smoothing = GaussianBlur2d(kernel_size=kernel_size, sigma=4)

    def forward(self, out_map: torch.Tensor, final_size: tuple[int, int]) -> torch.Tensor:
        # upscale & smooth
        anomaly_map = F.interpolate(out_map, size=final_size, mode="bilinear")
        return self.smoothing(anomaly_map)


#####################################################################
# Trainer for Supersimplenet Model
#####################################################################
import warnings
import torch
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from .components.trainer import AnomalyTrainer, EarlyStopper


class SupersimplenetTrainer(AnomalyTrainer):
    """
    Trainer for SuperSimpleNet model

    Supports both supervised and unsupervised training:
    - Unsupervised: Uses synthetic anomalies (Perlin noise)
    - Supervised: Uses real anomaly masks and labels

    Based on:
    - ICPR 2024: SuperSimpleNet: Unifying Unsupervised and Supervised Learning
      for Fast and Reliable Surface Defect Detection
      https://arxiv.org/pdf/2408.03143
    - JIMS 2025: No Label Left Behind: A Unified Surface Defect Detection Model
      for all Supervision Regimes
      https://link.springer.com/article/10.1007/s10845-025-02680-8

    Example:
        >>> # Unsupervised mode (default)
        >>> trainer = SupersimplenetTrainer(
        ...     backbone="wide_resnet50_2.tv_in1k",
        ...     layers=["layer2", "layer3"],
        ...     supervised=False
        ... )
        >>> trainer.fit(train_loader, num_epochs=100, valid_loader=valid_loader)

        >>> # Supervised mode
        >>> trainer = SupersimplenetTrainer(
        ...     backbone="wide_resnet50_2.tv_in1k",
        ...     supervised=True,
        ...     adapt_cls_features=True  # ICPR version
        ... )
    """

    def __init__(self, model=None, loss_fn=None, device=None, logger=None,
                 perlin_threshold=0.2,
                 backbone="wide_resnet50_2.tv_in1k",
                 layers=None,
                 supervised=False,
                 adapt_cls_features=False,
                 gradient_clip_val=None):
        """
        Args:
            model: SupersimplenetModel instance (if None, creates default)
            loss_fn: Loss function (default: SSNLoss)
            device: Device for training
            logger: Logger instance
            perlin_threshold: Threshold for Perlin noise binarization (default: 0.2)
            backbone: Backbone architecture (default: "wide_resnet50_2.tv_in1k")
                     IMPORTANT: Use .tv_in1k weights, not .tv2_in1k
            layers: Feature extraction layers (default: ["layer2", "layer3"])
            supervised: If True, use real masks; if False, use synthetic anomalies
            adapt_cls_features: Whether to adapt classification features
                               ICPR version: True, JIMS version: False (default)
            gradient_clip_val: Gradient clipping value (default: 1.0 for supervised,
                              0.0 for unsupervised). Set to custom value to override.
        """
        # Set default layers
        if layers is None:
            layers = ["layer2", "layer3"]

        # Create model if not provided
        if model is None:
            # stop_grad: False for supervised, True for unsupervised
            stop_grad = False if supervised else True

            model = SupersimplenetModel(
                perlin_threshold=perlin_threshold,
                backbone=backbone,
                layers=layers,
                stop_grad=stop_grad,
                adapt_cls_features=adapt_cls_features,
            )

            # Freeze feature extractor (pretrained backbone)
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

        # Create loss function if not provided
        if loss_fn is None:
            loss_fn = SSNLoss()

        # Initialize parent
        super().__init__(model, loss_fn=loss_fn, device=device, logger=logger)

        # Store hyperparameters
        self.perlin_threshold = perlin_threshold
        self.backbone = backbone
        self.layers = layers
        self.supervised = supervised
        self.adapt_cls_features = adapt_cls_features

        # Gradient clipping (Lightning: gradient_clip_val in trainer_arguments)
        if gradient_clip_val is None:
            # Auto-configure based on mode
            self.gradient_clip_val = 1.0 if supervised else 0.0
        else:
            # User override
            self.gradient_clip_val = gradient_clip_val

        if self.gradient_clip_val > 0:
            self.logger.info(f"Gradient clipping enabled: {self.gradient_clip_val}")

    def training_step(self, batch, batch_idx):
        """
        Training step for SuperSimpleNet

        Args:
            batch: Dict with 'image', and optionally 'mask', 'label'
            batch_idx: Batch index

        Returns:
            dict: Dictionary with 'loss'
        """
        images = batch['image'].to(self.device)
        batch_size = images.size(0)

        # Get masks and labels
        if self.supervised:
            # Supervised: use real masks and labels from batch
            masks = batch.get('mask', None)
            labels = batch.get('label', None)

            # Process masks to ensure correct shape [B, 1, H, W]
            if masks is not None:
                masks = masks.to(self.device)

                if masks.ndim == 2:  # [H, W]
                    masks = masks.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    if batch_size > 1:
                        masks = masks.repeat(batch_size, 1, 1, 1)  # [B, 1, H, W]
                elif masks.ndim == 3:  # [B, H, W]
                    masks = masks.unsqueeze(1)  # [B, 1, H, W]
                elif masks.ndim == 4:  # [B, C, H, W]
                    if masks.shape[1] > 1:
                        masks = masks[:, :1, :, :]  # Take first channel [B, 1, H, W]

            # Process labels to ensure 1D tensor [B]
            if labels is not None:
                labels = labels.to(self.device)
                if labels.ndim == 0:  # Scalar
                    labels = labels.unsqueeze(0)  # [1]
                elif labels.ndim > 1:
                    labels = labels.squeeze()  # [B]
        else:
            # Unsupervised: model generates synthetic anomalies
            # Pass None to let model create zeros internally
            masks = None
            labels = None

        # Forward pass
        # In training mode, model returns:
        # (anomaly_map, anomaly_score, masks_augmented, labels_augmented)
        anomaly_map, anomaly_score, masks_aug, labels_aug = self.model(
            images=images,
            masks=masks,
            labels=labels
        )

        # Compute loss
        loss = self.loss_fn(
            pred_map=anomaly_map,
            pred_score=anomaly_score,
            target_mask=masks_aug,
            target_label=labels_aug
        )

        return dict(loss=loss)

    def configure_optimizers(self):
        """
        Configure AdamW optimizer with different learning rates for different modules
        Following Lightning implementation
        """
        optimizer = torch.optim.AdamW([
            {"params": self.model.adaptor.parameters(), "lr": 0.0001,},
            {"params": self.model.segdec.parameters(), "lr": 0.0002, "weight_decay": 0.00001},
        ])

        # Scheduler created in on_fit_start
        return dict(optimizer=optimizer)

    def configure_early_stoppers(self):
        """Configure early stopping"""
        return dict(
            train=EarlyStopper(patience=10, mode='min', min_delta=0.001, monitor='loss'),
            valid=EarlyStopper(patience=10, mode='max', min_delta=0.001, target_value=0.995, monitor='auroc')
        )

    def on_fit_start(self):
        super().on_fit_start()

        # Create scheduler (Lightning: milestones at 0.8 and 0.9 of max_epochs)
        if self.num_epochs > 0:
            milestones = [int(self.num_epochs * 0.8), int(self.num_epochs * 0.9)]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=0.4)
            self.logger.info(f"Scheduler: MultiStepLR(milestones={milestones}, gamma=0.4)")

    def _train_epoch(self, train_loader):
        if self.gradient_clip_val > 0:
            # Use custom training loop with gradient clipping
            return self._train_epoch_with_clip(train_loader)
        else:
            # Use default training loop (no clipping)
            return super()._train_epoch(train_loader)

    @torch.enable_grad()
    def _train_epoch_with_clip(self, train_loader):
        accumulated_outputs = {}
        total_images = 0

        with tqdm(train_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f"{self.epoch_info} Training")
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
                total_images += batch_size

                self.on_train_batch_start(batch, batch_idx)

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    outputs = self.training_step(batch, batch_idx)
                    loss = outputs.get('loss')

                    if loss is not None:
                        loss.backward()

                        # Apply gradient clipping (Lightning style)
                        # Clip AFTER backward() but BEFORE optimizer.step()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                        outputs['grad_norm'] = grad_norm.item()

                        self.optimizer.step()
                    self.global_step += 1
                else:
                    outputs = self.training_step(batch, batch_idx)

                for name, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    accumulated_outputs.setdefault(name, 0.0)
                    accumulated_outputs[name] += value * batch_size

                progress_bar.set_postfix({
                    name: f"{value / total_images:.3f}"
                    for name, value in accumulated_outputs.items()
                })
                self.on_train_batch_end(outputs, batch, batch_idx)

        return {name: value / total_images for name, value in accumulated_outputs.items()}
