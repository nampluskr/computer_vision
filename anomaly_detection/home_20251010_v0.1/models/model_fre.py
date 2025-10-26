"""
- FRE (2023): A Fast Method For Anomaly Detection And Segmentation
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/fre
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/fre.html
  - https://papers.bmvc2023.org/0614.pdf (2023)
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .components.feature_extractor import TimmFeatureExtractor


#####################################################################
# anomalib/src/anomalib/models/image/fre/torch_model.py
#####################################################################

class TiedAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.weight = nn.Parameter(torch.empty(latent_dim, input_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = F.linear(features, self.weight, self.encoder_bias)
        return F.linear(encoded, self.weight.t(), self.decoder_bias)


class FREModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        layer: str,
        input_dim: int = 65536,
        latent_dim: int = 220,
        pre_trained: bool = True,
        pooling_kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.fre_model = TiedAE(input_dim, latent_dim)
        self.layer = layer
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=[layer],
        ).eval()

    def get_features(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.feature_extractor.eval()
        features_in = self.feature_extractor(batch)[self.layer]
        batch_size = len(features_in)
        if self.pooling_kernel_size > 1:
            features_in = F.avg_pool2d(input=features_in, kernel_size=self.pooling_kernel_size)
        feature_shapes = features_in.shape
        features_in = features_in.view(batch_size, -1).detach()
        features_out = self.fre_model(features_in)
        return features_in, features_out, feature_shapes

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        features_in, features_out, feature_shapes = self.get_features(batch)
        fre = torch.square(features_in - features_out).reshape(feature_shapes)
        anomaly_map = torch.sum(fre, 1)  # NxCxHxW --> NxHxW
        score = torch.sum(anomaly_map, (1, 2))  # NxHxW --> N
        anomaly_map = torch.unsqueeze(anomaly_map, 1)
        anomaly_map = F.interpolate(anomaly_map, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return dict(pred_score=score, anomaly_map=anomaly_map)


#####################################################################
# Trainer for FRE Model
#####################################################################
from .components.trainer import BaseTrainer, EarlyStopper

class FRETrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="resnet50", layer="layer3"):

        if model is None:
            model = FREModel(backbone=backbone, layer=layer, pre_trained=True)
        if optimizer is None:
            params = model.fre_model.parameters()
            optimizer = torch.optim.Adam(params, lr=1e-3)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # if early_stopper_loss is None:
        #     early_stopper_loss = EarlyStopper(patience=10, min_delta=0.01, mode='min', target_value=1.0)
        if early_stopper_auroc is None:
            early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max', target_value=0.995)
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)

        self.optimizer.zero_grad()
        features_in, features_out, _ = self.model.get_features(images)
        loss = self.loss_fn(features_in, features_out)
        loss.backward()
        self.optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            for name, metric_fn in self.metrics.items():
                results[name] = metric_fn(features_in, features_out).item()
        return results