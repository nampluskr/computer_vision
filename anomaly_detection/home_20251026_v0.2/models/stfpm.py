""" filename: models/stfpm.py
- STFPM (2021): Student-Teacher Feature Pyramid Matching for anomaly detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/stfpm
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/stfpm.html
  - https://arxiv.org/pdf/2103.04257.pdf
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .components.feature_extractor import TimmFeatureExtractor
from .components.tiler import Tiler


#####################################################################
# anomalib/src/anomalib/models/image/stfpm/anomaly_map.py
#####################################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)

    @staticmethod
    def compute_layer_map(
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        return F.interpolate(layer_map, size=image_size, align_corners=False, mode="bilinear")

    def compute_anomaly_map(
        self,
        teacher_features: dict[str, torch.Tensor],
        student_features: dict[str, torch.Tensor],
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.ones(batch_size, 1, image_size[0], image_size[1])
        for layer in teacher_features:
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer], image_size)
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def forward(self, **kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not ("teacher_features" in kwargs and "student_features" in kwargs):
            msg = f"Expected keys `teacher_features` and `student_features. Found {kwargs.keys()}"
            raise ValueError(msg)

        teacher_features: dict[str, torch.Tensor] = kwargs["teacher_features"]
        student_features: dict[str, torch.Tensor] = kwargs["student_features"]
        image_size: tuple[int, int] | torch.Size = kwargs["image_size"]

        return self.compute_anomaly_map(teacher_features, student_features, image_size)


#####################################################################
# anomalib/src/anomalib/models/image/stfpm/loss.py
#####################################################################

class STFPMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def compute_layer_loss(self, teacher_feats: torch.Tensor, student_feats: torch.Tensor) -> torch.Tensor:
        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        return (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)

    def forward(
        self,
        teacher_features: dict[str, torch.Tensor],
        student_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        layer_losses: list[torch.Tensor] = []
        for layer in teacher_features:
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        return torch.stack(layer_losses).sum()


#####################################################################
# anomalib/src/anomalib/models/image/stfpm/torch_model.py
#####################################################################

class STFPMModel(nn.Module):
    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.teacher_model = TimmFeatureExtractor(backbone=self.backbone, pre_trained=True,
            layers=layers).eval()
        self.student_model = TimmFeatureExtractor(backbone=self.backbone, pre_trained=False,
            layers=layers, requires_grad=True)

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator()

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        output_size = images.shape[-2:]
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: dict[str, torch.Tensor] = self.teacher_model(images)
        student_features: dict[str, torch.Tensor] = self.student_model(images)

        if self.tiler:
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.tiler.untile(data)
            for layer, data in student_features.items():
                student_features[layer] = self.tiler.untile(data)

        if self.training:
            return teacher_features, student_features

        anomaly_map = self.anomaly_map_generator(
            teacher_features=teacher_features,
            student_features=student_features,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


#####################################################################
# Trainer for STFPM Model
#####################################################################
from .components.trainer import AnomalyTrainer, EarlyStopper

class STFPMTrainer(AnomalyTrainer):
    def __init__(self, model=None, loss_fn=None, device=None, logger=None,
                 learning_rate=0.4, backbone="resnet50", layers=["layer1", "layer2", "layer3"]):

        if model is None:
            model = STFPMModel(backbone=backbone, layers=layers)
        if loss_fn is None:
            loss_fn = STFPMLoss()

        super().__init__(model, loss_fn=loss_fn, device=device, logger=logger)
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        teacher_features, student_features = self.model(images)
        loss = self.loss_fn(teacher_features, student_features)
        return dict(loss=loss)

    def configure_optimizers(self):
        params = self.model.student_model.parameters()
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9,
            dampening=0.0, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return dict(optimizer=optimizer, scheduler=scheduler)

    def configure_early_stoppers(self):
        return dict(
            train=EarlyStopper(patience=5, mode='min', min_delta=0.01, monitor='loss'),
            valid=EarlyStopper(patience=5, mode='max', min_delta=0.001, target_value=0.999, monitor='auroc')
        )
