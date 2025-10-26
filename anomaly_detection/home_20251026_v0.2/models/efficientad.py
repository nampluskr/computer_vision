"""
- EfficientAd (2024): Accurate Visual Anomaly Detection at Millisecond-Level Latencies
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/efficient_ad
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/efficient_ad.html
  - https://arxiv.org/pdf/2303.14535.pdf
"""

import logging
import math
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchvision import transforms


logger = logging.getLogger(__name__)


#####################################################################
# anomalib/src/anomalib/models/image/efficientad/torch_model.py
#####################################################################

def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    return (x - mean) / std


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAdModelSize(str, Enum):
    M = "medium"
    S = "small"


class SmallPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding = padding
        # use ceil to match output shape of PDN
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        return self.decoder(x, image_size)


class EfficientAdModel(nn.Module):
    def __init__(
        self,
        teacher_out_channels: int = 384,
        model_size: EfficientAdModelSize = EfficientAdModelSize.S,
        padding: bool = False,
        pad_maps: bool = True,
    ) -> None:
        super().__init__()

        self.pad_maps = pad_maps
        self.teacher: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork
        self.student: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork

        if model_size == EfficientAdModelSize.M:
            self.teacher = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == EfficientAdModelSize.S:
            self.teacher = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        else:
            msg = f"Unknown model size {model_size}"
            raise ValueError(msg)

        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding)
        self.teacher_out_channels: int = teacher_out_channels

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            },
        )

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            },
        )

    @staticmethod
    def is_set(p_dic: nn.ParameterDict) -> bool:
        return any(value.sum() != 0 for _, value in p_dic.items())

    @staticmethod
    def choose_random_aug_image(image: torch.Tensor) -> torch.Tensor:
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        idx = int(torch.randint(0, len(transform_functions), (1,)).item())
        transform_function = transform_functions[idx]
        return transform_function(image, coefficient)

    def forward(
        self,
        batch: torch.Tensor,
        batch_imagenet: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        if self.training:
            return self.compute_losses(batch, batch_imagenet, distance_st)

        map_st, map_stae = self.compute_maps(batch, student_output, distance_st, normalize)
        anomaly_map = 0.5 * map_st + 0.5 * map_stae
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def compute_student_teacher_distance(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)
        return student_output, distance_st

    def compute_losses(
        self,
        batch: torch.Tensor,
        batch_imagenet: torch.Tensor,
        distance_st: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Student loss
        distance_st = reduce_tensor_elems(distance_st)
        d_hard = torch.quantile(distance_st, 0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])
        student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
        loss_penalty = torch.mean(student_output_penalty**2)
        loss_st = loss_hard + loss_penalty

        # Autoencoder and Student AE Loss
        aug_img = self.choose_random_aug_image(batch)
        ae_output_aug = self.ae(aug_img, batch.shape[-2:])

        with torch.no_grad():
            teacher_output_aug = self.teacher(aug_img)
            if self.is_set(self.mean_std):
                teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

        student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]

        distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
        distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        return (loss_st, loss_ae, loss_stae)

    def compute_maps(
        self,
        batch: torch.Tensor,
        student_output: torch.Tensor,
        distance_st: torch.Tensor,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_size = batch.shape[-2:]
        # Eval mode.
        with torch.no_grad():
            ae_output = self.ae(batch, image_size)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels :]) ** 2,
                dim=1,
                keepdim=True,
            )

        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))
        map_st = F.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = F.interpolate(map_stae, size=image_size, mode="bilinear")

        if self.is_set(self.quantiles) and normalize:
            map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
            map_stae = 0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
        return map_st, map_stae

    def get_maps(self, batch: torch.Tensor, normalize: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        return self.compute_maps(batch, student_output, distance_st, normalize)


#####################################################################
# Trainer for EfficientAd Model
#####################################################################
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from .components.trainer import BaseTrainer, EarlyStopper
from .components.backbone import get_backbone_dir


class EfficientAdTrainer(BaseTrainer):

    def __init__(self, model=None, loss_fn=None, device=None, logger=None,
                 learning_rate=1e-4, model_size="small", teacher_out_channels=384):

        if model is None:
            model = EfficientAdModel(
                model_size=EfficientAdModelSize.S if model_size == "small" else EfficientAdModelSize.M,
                teacher_out_channels=teacher_out_channels,
                padding=False, pad_maps=True)

        super().__init__(model, loss_fn=None, device=device, logger=logger)

        self.lr = learning_rate
        self.model_size = model_size
        self.teature_out_channels = teacher_out_channels

        self.imagenet_dir = os.path.join(os.getenv('DATA_DIR', '/mnt/d/datasets'), 'imagenette2')
        self.imagenet_loader = None
        self.imagenet_iterator = None

        self.backbone_dir = os.getenv('BACKBONE_DIR', '/mnt/d/backbones')

        self.optimizer = self.configure_optimizers()['optimizer']
        self.scheduler_step_ratio = 0.95
        self.scheduler_gamma = 0.1

        self.auroc_metric = BinaryAUROC().to(self.device)
        self.aupr_metric = BinaryAveragePrecision().to(self.device)

    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)

        try:
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)

        loss_st, loss_ae, loss_stae = self.model(batch=images, batch_imagenet=batch_imagenet)
        loss = loss_st + loss_ae + loss_stae
        return dict(loss=loss, loss_st=loss_st, loss_ae=loss_ae, loss_stae=loss_stae)

    def validation_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        # Forward pass (model returns dict with pred_score in eval mode)
        outputs = self.model(images)
        anomaly_scores = outputs['pred_score']

        # Ensure proper shapes
        if anomaly_scores.dim() == 0:
            anomaly_scores = anomaly_scores.unsqueeze(0)
        elif anomaly_scores.dim() > 1:
            anomaly_scores = anomaly_scores.view(anomaly_scores.size(0), -1).mean(dim=1)

        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1:
            labels = labels.view(-1)

        # Update metrics
        self.auroc_metric.update(anomaly_scores.cpu(), labels.cpu())
        self.aupr_metric.update(anomaly_scores.cpu(), labels.cpu())

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                auroc = self.auroc_metric.compute()
                aupr = self.aupr_metric.compute()
        except:
            auroc = torch.tensor(0.0)
            aupr = torch.tensor(0.0)

        return dict(auroc=auroc, aupr=aupr)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        self.auroc_metric.reset()
        self.aupr_metric.reset()

    def on_validation_epoch_end(self, outputs):
        auroc_value = self.auroc_metric.compute()
        aupr_value = self.aupr_metric.compute()

        outputs['auroc'] = auroc_value.item()
        outputs['aupr'] = aupr_value.item()

        self.auroc_metric.reset()
        self.aupr_metric.reset()

        super().on_validation_epoch_end(outputs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # First epoch: perform one-time setup
        if self.epoch == 1:
            self.logger.info("="*70)
            self.logger.info("EfficientAD Initialization")
            self.logger.info("="*70)

            # Validate batch size
            first_batch = next(iter(self._current_train_loader))
            if first_batch['image'].shape[0] != 1:
                raise ValueError(
                    f"train_batch_size for EfficientAD should be 1, "
                    f"got {first_batch['image'].shape[0]}"
                )

            # Get image size
            image_size = first_batch['image'].shape[-2:]
            self.logger.info(f"Image size: {image_size}")

            # 1. Load pretrained teacher model
            self._load_pretrained_teacher()

            # 2. Prepare Imagenet data
            self._prepare_imagenet_data(image_size)

            # 3. Calculate teacher channel statistics
            if not self.model.is_set(self.model.mean_std):
                self.logger.info("Calculating teacher channel statistics...")
                channel_mean_std = self._compute_teacher_statistics(self._current_train_loader)
                self.model.mean_std.update(channel_mean_std)
                self.logger.info("Teacher statistics computed successfully")

            self.logger.info("="*70)

    def on_validation_start(self):
        # Check if this is being called (BaseTrainer doesn't have this hook)
        # If validation loader exists and quantiles not set
        if hasattr(self, '_current_valid_loader') and self._current_valid_loader is not None:
            if not self.model.is_set(self.model.quantiles):
                self.logger.info("="*70)
                self.logger.info("Computing validation quantiles...")
                map_norm_quantiles = self._compute_quantiles(self._current_valid_loader)
                self.model.quantiles.update(map_norm_quantiles)
                self.logger.info("Quantiles computed successfully")
                self.logger.info("="*70)

    def configure_optimizers(self):
        params = list(self.model.student.parameters()) + list(self.model.ae.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        # Scheduler will be created in fit() when num_epochs is known
        return dict(optimizer=optimizer)

    def configure_early_stoppers(self):
        return dict(
            train=EarlyStopper(patience=10, mode='min', min_delta=0.01, monitor='loss'),
            valid=EarlyStopper(patience=10, mode='max', min_delta=0.001, target_value=0.995, monitor='auroc')
        )

    def fit(self, train_loader, num_epochs, valid_loader=None, output_dir=None, run_name=None):
        # Store loaders for access in hooks
        self._current_train_loader = train_loader
        self._current_valid_loader = valid_loader

        # Create scheduler now that we know num_epochs
        # Calculate total training steps
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        step_size = int(self.scheduler_step_ratio * total_steps)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
            step_size=step_size, gamma=self.scheduler_gamma)
        self.logger.info(f"Scheduler: StepLR(step_size={step_size}/{total_steps}, "
                         f"gamma={self.scheduler_gamma})")

        # Manually trigger on_validation_start before first validation
        if valid_loader is not None:
            self.on_validation_start()

        # Call parent fit
        return super().fit(train_loader, num_epochs, valid_loader, output_dir, run_name)

    #############################################################
    # Helper Methods
    #############################################################

    def _load_pretrained_teacher(self):
        backbone_dir = Path(self.backbone_dir)

        if not os.path.isdir(backbone_dir):
            raise RuntimeError(
                f"Backbone directory not found: {backbone_dir}\n"
                f"Please download pretrained weights and place them in this directory."
            )

        model_size_str = self.model_size
        teacher_path = backbone_dir / f"pretrained_teacher_{model_size_str}.pth"

        if not os.path.isfile(teacher_path):
            raise RuntimeError(
                f"Teacher weight file not found: {teacher_path}\n"
                f"Expected files in {backbone_dir}:\n"
                f"  - pretrained_teacher_small.pth\n"
                f"  - pretrained_teacher_medium.pth"
            )

        self.logger.info(f"Loading pretrained teacher from: {teacher_path}")
        state_dict = torch.load(teacher_path, map_location=self.device, weights_only=True)
        self.model.teacher.load_state_dict(state_dict)
        self.logger.info("Teacher model loaded successfully")

    def _prepare_imagenet_data(self, image_size: tuple[int, int]):
        """Prepare Imagenette dataset loader"""
        if not os.path.isdir(self.imagenet_dir):
            raise RuntimeError(
                f"Imagenette directory not found: {self.imagenet_dir}\n"
                f"Please download Imagenette2 dataset from:\n"
                f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
            )
        self.logger.info(f"Loading Imagenette data from: {self.imagenet_dir}")

        # Transforms for Imagenet data (no normalization - done in model)
        data_transforms = T.Compose([
            T.Resize((image_size[0] * 2, image_size[1] * 2)),
            T.RandomGrayscale(p=0.3),
            T.CenterCrop((image_size[0], image_size[1])),
            T.ToTensor(),
        ])

        imagenet_dataset = ImageFolder(self.imagenet_dir, transform=data_transforms)
        self.imagenet_loader = DataLoader(
            imagenet_dataset,
            batch_size=1,  # EfficientAD uses batch_size=1
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.imagenet_iterator = iter(self.imagenet_loader)
        self.logger.info(f"Imagenette dataset loaded: {len(imagenet_dataset)} images")

    @torch.no_grad()
    def _compute_teacher_statistics(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        arrays_defined = False
        n = None
        channel_sum = None
        channel_sum_sqr = None

        for batch in tqdm(dataloader, desc="Computing teacher statistics", ascii=True, leave=False):
            y = self.model.teacher(batch['image'].to(self.device))

            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                channel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                channel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            channel_sum += torch.sum(y, dim=[0, 2, 3])
            channel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        if n is None:
            raise ValueError("No data processed for teacher statistics")

        channel_mean = channel_sum / n
        channel_std = torch.sqrt((channel_sum_sqr / n) - (channel_mean**2))

        return {
            "mean": channel_mean.float()[None, :, None, None],
            "std": channel_std.float()[None, :, None, None]
        }

    @torch.no_grad()
    def _compute_quantiles(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        maps_st = []
        maps_ae = []

        for batch in tqdm(dataloader, desc="Computing quantiles", ascii=True, leave=False):
            for img, label in zip(batch['image'], batch['label'], strict=True):
                if label == 0:  # Only use normal images
                    map_st, map_ae = self.model.get_maps(
                        img.unsqueeze(0).to(self.device),
                        normalize=False
                    )
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)

        if not maps_st:
            self.logger.warning("No normal samples found in validation set!")
            # Return dummy quantiles
            return {
                "qa_st": torch.tensor(0.0).to(self.device),
                "qb_st": torch.tensor(1.0).to(self.device),
                "qa_ae": torch.tensor(0.0).to(self.device),
                "qb_ae": torch.tensor(1.0).to(self.device)
            }

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qb_st": qb_st, "qa_ae": qa_ae, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb
