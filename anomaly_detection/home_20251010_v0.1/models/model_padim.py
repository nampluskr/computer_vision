"""
- PaDiM (2020): A Patch Distribution Modeling Framework for Anomaly Detection and Localization
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/padim
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/padim.html
  - https://arxiv.org/abs/2011.08785
"""

from random import sample

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .components.multi_variate_gaussian import MultiVariateGaussian
from .components.feature_extractor import TimmFeatureExtractor, dryrun_find_featuremap_dims
from .components.blur import GaussianBlur2d
from .components.tiler import Tiler


#####################################################################
# anomalib/src/anomalib/models/image/padim/anomaly_map.py
#####################################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(self, sigma: int = 4) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    @staticmethod
    def compute_distance(embedding: torch.Tensor, stats: list[torch.Tensor]) -> torch.Tensor:
        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, 1, height, width)
        return distances.clamp(0).sqrt()

    @staticmethod
    def up_sample(distance: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        return F.interpolate(
            distance,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

    def smooth_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        return self.blur(anomaly_map)

    def compute_anomaly_map(
        self,
        embedding: torch.Tensor,
        mean: torch.Tensor,
        inv_covariance: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        if image_size:
            score_map = self.up_sample(score_map, image_size)
        return self.smooth_anomaly_map(score_map)

    def forward(self, **kwargs) -> torch.Tensor:
        if not ("embedding" in kwargs and "mean" in kwargs and "inv_covariance" in kwargs):
            msg = f"Expected keys `embedding`, `mean` and `covariance`. Found {kwargs.keys()}"
            raise ValueError(msg)

        embedding: torch.Tensor = kwargs["embedding"]
        mean: torch.Tensor = kwargs["mean"]
        inv_covariance: torch.Tensor = kwargs["inv_covariance"]
        image_size: tuple[int, int] | torch.Size = kwargs.get("image_size")

        return self.compute_anomaly_map(embedding, mean, inv_covariance, image_size=image_size)


#####################################################################
# anomalib/src/anomalib/models/image/padim/torch_model.py
#####################################################################

# defaults from the paper
_N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}

def _deduce_dims(
    feature_extractor: TimmFeatureExtractor,
    input_size: tuple[int, int],
    layers: list[str],
) -> tuple[int, int]:
    dimensions_mapping = dryrun_find_featuremap_dims(feature_extractor, input_size, layers)

    # the first layer in `layers` has the largest resolution
    first_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
    n_patches = torch.tensor(first_layer_resolution).prod().int().item()

    # the original embedding size is the sum of the channels of all layers
    n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)  # type: ignore[misc]

    return n_features_original, n_patches


class PadimModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=layers,
            pre_trained=pre_trained,
        ).eval()
        self.n_features_original = sum(self.feature_extractor.out_dims)
        self.n_features = n_features or _N_FEATURES_DEFAULTS.get(self.backbone)
        if self.n_features is None:
            msg = (
                f"n_features must be specified for backbone {self.backbone}. "
                f"Default values are available for: {sorted(_N_FEATURES_DEFAULTS.keys())}"
            )
            raise ValueError(msg)

        if not (0 < self.n_features <= self.n_features_original):
            msg = f"For backbone {self.backbone}, 0 < n_features <= {self.n_features_original}, found {self.n_features}"
            raise ValueError(msg)

        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(self.n_features_original), self.n_features)),
        )
        self.idx: torch.Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator()

        self.gaussian = MultiVariateGaussian()
        self.memory_bank: list[torch.tensor] = []

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            self.memory_bank.append(embeddings)
            return embeddings

        anomaly_map = self.anomaly_map_generator(
            embedding=embeddings,
            mean=self.gaussian.mean,
            inv_covariance=self.gaussian.inv_covariance,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        return torch.index_select(embeddings, 1, idx)

    def fit(self) -> None:
        if len(self.memory_bank) == 0:
            msg = "Memory bank is empty. Cannot perform coreset selection."
            raise ValueError(msg)
        self.memory_bank = torch.vstack(self.memory_bank)

        # fit gaussian
        self.gaussian.fit(self.memory_bank)

        # clear memory bank, redcues gpu usage
        self.memory_bank = []
        
#####################################################################
# Trainer for PatchCore Model
#####################################################################
import os
from tqdm import tqdm
from .components.trainer import BaseTrainer, EarlyStopper

class PadimTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"]):

        if model is None:
            model = PadimModel(backbone=backbone, layers=layers, pre_trained=True, n_features=None)

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 1

    def on_train_start(self, train_loader):
        self.model.train()
        pbar = tqdm(train_loader, desc="Extracting embeddings", leave=False, ascii=True)
        for batch in pbar:
            images = batch["image"].to(self.device)
            _ = self.model(images)

    def on_train_end(self, train_results):
        self.model.fit()
        super().on_train_end(train_results)

    @torch.enable_grad()
    def train_step(self, batch):
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def train_epoch(self, train_loader):
        return {"loss": 0.0}