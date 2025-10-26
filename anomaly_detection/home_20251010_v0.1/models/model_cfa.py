"""
- CFA (2022): Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/cfa
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/cfa.html
  - https://arxiv.org/abs/2206.04325
"""

from einops import rearrange
from sklearn.cluster import KMeans
from tqdm import tqdm

import torch
import torchvision
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.common_types import _size_2_t
from torch.fx.graph_module import GraphModule
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

from .components.blur import GaussianBlur2d
from .components.dynamic_buffer import DynamicBufferMixin
from .components.feature_extractor import dryrun_find_featuremap_dims
from .components.backbone import get_backbone_path


#####################################################################
# anomalib/src/anomalib/models/image/cfa/anomaly_map.py
#####################################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(
        self,
        num_nearest_neighbors: int,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.sigma = sigma

    def compute_score(self, distance: torch.Tensor, scale: tuple[int, int]) -> torch.Tensor:
        distance = torch.sqrt(distance)
        distance = distance.topk(self.num_nearest_neighbors, largest=False).values
        distance = (F.softmin(distance, dim=-1)[:, :, 0]) * distance[:, :, 0]
        distance = distance.unsqueeze(-1)

        score = rearrange(distance, "b (h w) c -> b c h w", h=scale[0], w=scale[1])
        return score.detach()

    def compute_anomaly_map(
        self,
        score: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        anomaly_map = score.mean(dim=1, keepdim=True)
        if image_size is not None:
            anomaly_map = F.interpolate(anomaly_map, size=image_size, mode="bilinear", align_corners=False)

        gaussian_blur = GaussianBlur2d(sigma=self.sigma).to(score.device)
        return gaussian_blur(anomaly_map)  # pylint: disable=not-callable

    def forward(self, **kwargs) -> torch.Tensor:
        if not ("distance" in kwargs and "scale" in kwargs):
            msg = f"Expected keys `distance` and `scale`. Found {kwargs.keys()}"
            raise ValueError(msg)

        distance: torch.Tensor = kwargs["distance"]
        scale: tuple[int, int] = kwargs["scale"]
        image_size: tuple[int, int] | torch.Size | None = kwargs.get("image_size")

        score = self.compute_score(distance=distance, scale=scale)
        return self.compute_anomaly_map(score, image_size=image_size)


#####################################################################
# anomalib/src/anomalib/models/image/cfa/loss.py
#####################################################################

class CfaLoss(nn.Module):
    def __init__(self, num_nearest_neighbors: int, num_hard_negative_features: int, radius: float) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_hard_negative_features = num_hard_negative_features
        self.radius = torch.ones(1, requires_grad=True) * radius

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        num_neighbors = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(num_neighbors, largest=False).values

        score = distance[:, :, : self.num_nearest_neighbors] - (self.radius**2).to(distance.device)
        l_att = torch.mean(torch.max(torch.zeros_like(score), score))

        score = (self.radius**2).to(distance.device) - distance[:, :, self.num_hard_negative_features :]
        l_rep = torch.mean(torch.max(torch.zeros_like(score), score - 0.1))

        return (l_att + l_rep) * 1000


#####################################################################
# anomalib/src/anomalib/models/image/cfa/torch_model.py
#####################################################################

SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")

def get_return_nodes(backbone: str) -> list[str]:
    if backbone == "efficientnet_b5":
        msg = "EfficientNet feature extractor has not implemented yet."
        raise NotImplementedError(msg)

    return_nodes: list[str]
    if backbone in {"resnet18", "wide_resnet50_2"}:
        return_nodes = ["layer1", "layer2", "layer3"]
    elif backbone == "vgg19_bn":
        return_nodes = ["features.25", "features.38", "features.52"]
    else:
        msg = f"Backbone {backbone} is not supported. Supported backbones are {SUPPORTED_BACKBONES}."
        raise ValueError(msg)
    return return_nodes


# TODO(samet-akcay): Replace this with the new torchfx feature extractor.
# CVS-122673
def get_feature_extractor(backbone: str, return_nodes: list[str]) -> GraphModule:
    model = getattr(torchvision.models, backbone)(weights=None)
    weight_path = get_backbone_path(backbone)
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    feature_extractor = create_feature_extractor(model=model, return_nodes=return_nodes)
    feature_extractor.eval()
    return feature_extractor


class CfaModel(DynamicBufferMixin):
    def __init__(
        self,
        backbone: str,
        gamma_c: int,
        gamma_d: int,
        num_nearest_neighbors: int,
        num_hard_negative_features: int,
        radius: float,
    ) -> None:
        super().__init__()
        self.gamma_c = gamma_c
        self.gamma_d = gamma_d

        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_hard_negative_features = num_hard_negative_features

        self.register_buffer("memory_bank", torch.tensor(0.0))
        self.memory_bank: torch.Tensor

        self.backbone = backbone
        return_nodes = get_return_nodes(backbone)
        self.feature_extractor = get_feature_extractor(backbone, return_nodes)

        self.descriptor = Descriptor(self.gamma_d, backbone)
        self.radius = torch.ones(1, requires_grad=True) * radius

        self.anomaly_map_generator = AnomalyMapGenerator(
            num_nearest_neighbors=num_nearest_neighbors,
        )

    def get_scale(self, input_size: tuple[int, int] | torch.Size) -> torch.Size:
        feature_map_metadata = dryrun_find_featuremap_dims(
            feature_extractor=self.feature_extractor,
            input_size=(input_size[0], input_size[1]),
            layers=get_return_nodes(self.backbone),
        )
        # Scale is to get the largest feature map dimensions of different layers
        # of the feature extractor. In a typical feature extractor, the first
        # layer has the highest resolution.
        resolution = next(iter(feature_map_metadata.values()))["resolution"]
        if isinstance(resolution, int):
            scale = (resolution,) * 2
        elif isinstance(resolution, tuple):
            scale = resolution
        else:
            msg = f"Unknown type {type(resolution)} for `resolution`. Expected types are either int or tuple[int, int]."
            raise TypeError(msg)
        return torch.Size(scale)

    def initialize_centroid(self, data_loader: DataLoader) -> None:
        """Initialize the centroid (memory bank) using training data."""
        device = next(self.feature_extractor.parameters()).device

        # Collect all target features
        all_features = []

        with torch.no_grad():
            for data in tqdm(data_loader, desc=" > Initializing centroid", ascii=True, leave=False):
                batch = data["image"].to(device)
                features = self.feature_extractor(batch)
                features = list(features.values())
                target_features = self.descriptor(features)  # [B, C, H, W]
                all_features.append(target_features)

        # Stack and compute mean
        all_features = torch.cat(all_features, dim=0)  # [N, C, H, W]
        memory_bank = all_features.mean(dim=0, keepdim=True)  # [1, C, H, W]

        # Reshape: [1, C, H, W] -> [(H*W), C]
        memory_bank = rearrange(memory_bank, "b c h w -> (b h w) c")

        # Get spatial scale
        scale = self.get_scale(all_features.shape[-2:])

        # Apply K-means clustering if gamma_c > 1
        if self.gamma_c > 1:
            n_clusters = (scale[0] * scale[1]) // self.gamma_c
            print(f" > Applying K-means: {memory_bank.shape[0]} -> {n_clusters} clusters")
            k_means = KMeans(n_clusters=n_clusters, max_iter=3000, random_state=42)
            cluster_centers = k_means.fit(memory_bank.cpu()).cluster_centers_
            memory_bank = torch.tensor(cluster_centers, requires_grad=False).to(device)

        # Transpose: [(H*W), C] -> [C, (H*W)]
        self.memory_bank = rearrange(memory_bank, "h w -> w h")
        print(f" > Memory bank final shape: {self.memory_bank.shape}")

    def compute_distance(self, target_oriented_features: torch.Tensor) -> torch.Tensor:
        if target_oriented_features.ndim == 4:
            target_oriented_features = rearrange(target_oriented_features, "b c h w -> b (h w) c")

        features = target_oriented_features.pow(2).sum(dim=2, keepdim=True)
        centers = self.memory_bank.pow(2).sum(dim=0, keepdim=True).to(features.device)
        f_c = 2 * torch.matmul(target_oriented_features, (self.memory_bank.to(features.device)))
        return features + centers - f_c

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.memory_bank.ndim == 0:
            msg = "Memory bank is not initialized. Run `initialize_centroid` method first."
            raise ValueError(msg)

        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = list(features.values())

        target_features = self.descriptor(features)
        distance = self.compute_distance(target_features)

        if self.training:
            return distance

        anomaly_map = self.anomaly_map_generator(
            distance=distance,
            scale=target_features.shape[-2:],
            image_size=input_tensor.shape[-2:],
        ).squeeze()
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


class Descriptor(nn.Module):
    def __init__(self, gamma_d: int, backbone: str) -> None:
        super().__init__()

        self.backbone = backbone
        if self.backbone not in SUPPORTED_BACKBONES:
            msg = f"Supported backbones are {SUPPORTED_BACKBONES}. Got {self.backbone} instead."
            raise ValueError(msg)

        # TODO(samet-akcay): Automatically infer the number of dims
        # CVS-122673
        backbone_dims = {"vgg19_bn": 1280, "resnet18": 448, "wide_resnet50_2": 1792, "efficientnet_b5": 568}
        dim = backbone_dims[backbone]
        out_channels = 2 * dim // gamma_d if backbone == "efficientnet_b5" else dim // gamma_d

        self.layer = CoordConv2d(in_channels=dim, out_channels=out_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor] | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(features, dict):
            features = list(features.values())

        patch_features: torch.Tensor | None = None
        for feature in features:
            pooled_features = (
                F.avg_pool2d(feature, 3, 1, 1) / feature.size(1)
                if self.backbone == "efficientnet_b5"
                else F.avg_pool2d(feature, 3, 1, 1)
            )
            patch_features = (
                pooled_features
                if patch_features is None
                else torch.cat((patch_features, F.interpolate(feature, patch_features.size(2), mode="bilinear")), dim=1)
            )

        return self.layer(patch_features)


class CoordConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # AddCoord layer.
        self.add_coords = AddCoords(with_r)

        # Create conv layer on top of add_coords layer.
        self.conv2d = nn.Conv2d(
            in_channels=in_channels + 2 + int(with_r),  # 2 for rank-2, 1 for r
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = self.add_coords(input_tensor)
        return self.conv2d(out)


class AddCoords(nn.Module):
    def __init__(self, with_r: bool = False) -> None:
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # NOTE: This is a modified version of the original implementation,
        #   which only supports rank 2 tensors.
        batch, _, x_dim, y_dim = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, y_dim], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, x_dim], dtype=torch.int32)

        xx_range = torch.arange(x_dim, dtype=torch.int32)
        yy_range = torch.arange(y_dim, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # Transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch, 1, 1, 1).to(input_tensor.device)
        yy_channel = yy_channel.repeat(batch, 1, 1, 1).to(input_tensor.device)

        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr_channel = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr_channel], dim=1)

        return out

#####################################################################
# Trainer for CFA Model
#####################################################################
import os
from .components.trainer import BaseTrainer, EarlyStopper

class CfaTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="wide_resnet50_2"):

        if model is None:
            model = CfaModel(backbone=backbone, gamma_c=1, gamma_d=1,
                num_nearest_neighbors=3, num_hard_negative_features=3, radius=1e-5)
        if optimizer is None:
            optimizer =  torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4, amsgrad=True)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # if early_stopper_loss is None:
        #     early_stopper_loss = EarlyStopper(patience=10, min_delta=0.01, mode='min', target_value=1.0)
        # if early_stopper_auroc is None:
        #     early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max', target_value=0.995)
        if loss_fn is None:
            loss_fn = CfaLoss(num_nearest_neighbors=3, num_hard_negative_features=3, radius=1e-5)

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5
        self.centroid_initialized = False

    def on_train_start(self, train_loader):
        if not self.centroid_initialized:
            self.model.initialize_centroid(data_loader=train_loader)
            self.centroid_initialized = True
        self.model.train()

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)

        self.optimizer.zero_grad()
        distance = self.model(images)
        loss = self.loss_fn(distance)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        results = {"loss": loss.item()}
        return results

    def save_model(self, weight_path):
        if weight_path is not None:
            output_dir = os.path.abspath(os.path.dirname(weight_path))
            os.makedirs(output_dir, exist_ok=True)

            checkpoint = {
                "model": self.model.state_dict(),
                "memory_bank": self.model.memory_bank.cpu(),
            }
            if self.optimizer is not None:
                checkpoint["optimizer"] = self.optimizer.state_dict()

            torch.save(checkpoint, weight_path)
            print(f" > CFA model saved to: {weight_path}")
            print(f"   - Memory bank shape: {self.model.memory_bank.shape}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)

            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])

            if "memory_bank" in checkpoint:
                self.model.memory_bank = checkpoint["memory_bank"].to(self.device)
                self.centroid_initialized = True

            if self.optimizer is not None and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            print(f" > Loaded CFA model from: {weight_path}")
            print(f"   - Memory bank shape: {self.model.memory_bank.shape}")
            print()
        else:
            print(f" > No checkpoint found at: {weight_path}\n")

