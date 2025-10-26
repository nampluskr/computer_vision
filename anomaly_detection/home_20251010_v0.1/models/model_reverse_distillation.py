"""
- Reverse Distillation (2022): Anomaly Detection via Reverse Distillation from One-Class Embedding
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/reverse_distillation
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/reverse_distillation.html
  - https://arxiv.org/pdf/2201.10703v2.pdf (2022)
"""

from enum import Enum
from collections.abc import Sequence, Callable
from omegaconf import ListConfig

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

from .components.feature_extractor import TimmFeatureExtractor
from .components.tiler import Tiler
from .components.blur import GaussianBlur2d
from .components.resnet_decoder import get_decoder


#####################################################################
# anomalib/models/image/reverse_distillation/components/bottleneck.py
#####################################################################

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OCBE(nn.Module):
    def __init__(
        self,
        block: Bottleneck | BasicBlock,
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        # self.conv4 and self.bn4 are from the original code:
        # https://github.com/hq-deng/RD4AD/blob/6554076872c65f8784f6ece8cfb39ce77e1aee12/resnet.py#L412
        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[Bottleneck | BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ),
        )
        self.inplanes = planes * block.expansion
        layers.extend(
            [
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
                for _ in range(1, blocks)
            ],
        )

        return nn.Sequential(*layers)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Always assumes that features has length of 3
        feature0 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        feature1 = self.relu(self.bn3(self.conv3(features[1])))
        feature_cat = torch.cat([feature0, feature1, features[2]], 1)
        output = self.bn_layer(feature_cat)

        return output.contiguous()


def get_bottleneck_layer(backbone: str, **kwargs) -> OCBE:
    return OCBE(BasicBlock, 2, **kwargs) if backbone in {"resnet18", "resnet34"} else OCBE(Bottleneck, 3, **kwargs)

#####################################################################
# anomalib/src/anomalib/models/image/reverse_distillation/anomaly_map.py
#####################################################################

class AnomalyMapGenerationMode(str, Enum):
    ADD = "add"
    MULTIPLY = "multiply"


class AnomalyMapGenerator(nn.Module):
    def __init__(
        self,
        image_size: ListConfig | tuple,
        sigma: int = 4,
        mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.MULTIPLY,
    ) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        if mode not in {AnomalyMapGenerationMode.ADD, AnomalyMapGenerationMode.MULTIPLY}:
            msg = f"Found mode {mode}. Only multiply and add are supported."
            raise ValueError(msg)
        self.mode = mode

    def forward(self, student_features: list[torch.Tensor], teacher_features: list[torch.Tensor]) -> torch.Tensor:
        if self.mode == AnomalyMapGenerationMode.MULTIPLY:
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )  # b c h w
        elif self.mode == AnomalyMapGenerationMode.ADD:
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )

        for student_feature, teacher_feature in zip(student_features, teacher_features, strict=True):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)
            if self.mode == AnomalyMapGenerationMode.MULTIPLY:
                anomaly_map *= distance_map
            elif self.mode == AnomalyMapGenerationMode.ADD:
                anomaly_map += distance_map

        gaussian_blur = GaussianBlur2d(
            kernel_size=(self.kernel_size, self.kernel_size),
            sigma=(self.sigma, self.sigma),
        ).to(student_features[0].device)

        return gaussian_blur(anomaly_map)

#####################################################################
# anomalib/src/anomalib/models/image/reverse_distillation/loss.py
#####################################################################

class ReverseDistillationLoss(nn.Module):
    @staticmethod
    def forward(encoder_features: list[torch.Tensor], decoder_features: list[torch.Tensor]) -> torch.Tensor:
        cos_loss = torch.nn.CosineSimilarity()
        loss_sum = 0
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            loss_sum += torch.mean(
                1
                - cos_loss(
                    encoder_feature.view(encoder_feature.shape[0], -1),
                    decoder_feature.view(decoder_feature.shape[0], -1),
                ),
            )
        return loss_sum

#####################################################################
# anomalib/src/anomalib/models/image/reverse_distillation/torch_model.py
#####################################################################

class ReverseDistillationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: Sequence[str],
        anomaly_map_mode: AnomalyMapGenerationMode,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        encoder_backbone = backbone
        self.encoder = TimmFeatureExtractor(backbone=encoder_backbone, pre_trained=pre_trained, layers=layers)
        self.bottleneck = get_bottleneck_layer(backbone)
        self.decoder = get_decoder(backbone)

        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size, mode=anomaly_map_mode)

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]] | dict[str, torch.Tensor]:
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.tiler:
            for i, features in enumerate(encoder_features):
                encoder_features[i] = self.tiler.untile(features)
            for i, features in enumerate(decoder_features):
                decoder_features[i] = self.tiler.untile(features)

        if self.training:
            return encoder_features, decoder_features

        anomaly_map = self.anomaly_map_generator(encoder_features, decoder_features)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

#####################################################################
# Trainer for Reverse Distillation Model
#####################################################################
from .components.trainer import BaseTrainer, EarlyStopper

class ReverseDistillationTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], input_size=(256, 256)):

        if model is None:
            model = ReverseDistillationModel(backbone=backbone, layers=layers, pre_trained=True,
                input_size=input_size, anomaly_map_mode=AnomalyMapGenerationMode.ADD)
        if optimizer is None:
            params = params=list(model.decoder.parameters()) + list(model.bottleneck.parameters())
            optimizer = torch.optim.Adam(params, lr=0.005, betas=(0.5, 0.99))
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        if early_stopper_loss is None:
            early_stopper_loss = EarlyStopper(patience=10, min_delta=0.01, mode='min', target_value=0.1)
        if early_stopper_auroc is None:
            early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max', target_value=0.995)
        if loss_fn is None:
            loss_fn = ReverseDistillationLoss()

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)

        self.optimizer.zero_grad()
        encoder_features, decoder_features = self.model(images)
        loss = self.loss_fn(encoder_features, decoder_features)
        loss.backward()
        self.optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            for name, metric_fn in self.metrics.items():
                results[name] = metric_fn(encoder_features, decoder_features).item()
        return results