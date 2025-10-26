"""
- FastFlow (2021): Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/fastflow
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/fastflow.html
  - https://arxiv.org/abs/2111.07677
  - url = "https://dl.fbaipublicfiles.com/deit/cait_m48_448.pth"
  - url = "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth"
  https://huggingface.co/timm/cait_m48_448.fb_dist_in1k/resolve/main/pytorch_model.bin
  https://huggingface.co/timm/deit_base_distilled_patch16_384.fb_in1k/resolve/main/pytorch_model.bin
"""

from collections.abc import Callable

import os
import timm
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer
from FrEIA.framework import SequenceINN
from omegaconf import ListConfig

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .components.all_in_one_block import AllInOneBlock
from .components.backbone import get_backbone_path


#####################################################################
# anomalib/src/anomalib/models/image/fastflow/anomaly_map.py
#####################################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        flow_maps: list[torch.Tensor] = []
        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        return torch.mean(flow_maps, dim=-1)


#####################################################################
# anomalib/src/anomalib/models/image/fastflow/loss.py
#####################################################################

class FastflowLoss(nn.Module):
    @staticmethod
    def forward(hidden_variables: list[torch.Tensor], jacobians: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
        for hidden_variable, jacobian in zip(hidden_variables, jacobians, strict=True):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss


#####################################################################
# anomalib/src/anomalib/models/image/fastflow/torch_model.py
#####################################################################

def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        # NOTE: setting padding="same" in nn.Conv2d breaks the onnx export so manual padding required.
        # TODO(ashwinvaidya17): Use padding="same" in nn.Conv2d once PyTorch v2.1 is released
        # CVS-122671
        padding_dims = (kernel_size // 2 - ((1 + kernel_size) % 2), kernel_size // 2)
        padding = (*padding_dims, *padding_dims)
        return nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.ReLU(),
            nn.ZeroPad2d(padding),
            nn.Conv2d(hidden_channels, out_channels, kernel_size),
        )

    return subnet_conv


def create_fast_flow_block(
    input_dimensions: list[int],
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastflowModel(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        if backbone in {"cait_m48_448", "deit_base_distilled_patch16_384"}:
            self.feature_extractor = timm.create_model(backbone, pretrained=False)
            channels = [768]
            scales = [16]

            if pre_trained:
                weight_path = get_backbone_path(backbone)
                if weight_path and os.path.isfile(weight_path):
                    from safetensors.torch import load_file

                    state_dict = load_file(weight_path)
                    self.feature_extractor.load_state_dict(state_dict)
                    print(f" > [Info] Loaded pretrained weights for {backbone}")
                else:
                    print(f" > [Warning] Pretrained weights not found for {backbone}")

        elif backbone in {"resnet18", "wide_resnet50_2"}:
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=False,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales, strict=True):
                self.norms.append(
                    nn.LayerNorm(
                        [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    ),
                )
            if pre_trained:
                weights_path = get_backbone_path(backbone)
                state_dict = torch.load(weights_path, map_location='cpu')
                self.feature_extractor.load_state_dict(state_dict, strict=False)
        else:
            msg = (
                f"Backbone {backbone} is not supported. List of available backbones are "
                "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
            )
            raise ValueError(msg)

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.fast_flow_blocks = nn.ModuleList()
        for channel, scale in zip(channels, scales, strict=True):
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                ),
            )
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def forward(self, input_tensor: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]] | dict[str, torch.Tensor]:
        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(input_tensor)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        # Compute the hidden variable f: X -> Z and log-likelihood of the jacobian
        # (See Section 3.3 in the paper.)
        # NOTE: output variable has z, and jacobian tuple for each fast-flow blocks.
        hidden_variables: list[torch.Tensor] = []
        log_jacobians: list[torch.Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features, strict=True):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        if self.training:
            return hidden_variables, log_jacobians

        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def _get_cnn_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        features = self.feature_extractor(input_tensor)
        return [self.norms[i](feature) for i, feature in enumerate(features)]

    def _get_cait_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """CaiT feature extraction"""
        feature = self.feature_extractor.patch_embed(input_tensor)
        feature = feature + self.feature_extractor.pos_embed
        feature = self.feature_extractor.pos_drop(feature)
        
        # Block 0-40까지 통과 (총 41개)
        for i in range(41):  
            feature = self.feature_extractor.blocks[i](feature)
        
        # Normalization
        feature = self.feature_extractor.norm(feature)
        
        # Reshape: [B, N, C] -> [B, C, H, W]
        batch_size, num_patches, num_channels = feature.shape
        feature = feature.permute(0, 2, 1)  # [B, C, N]
        
        # 패치 수에서 spatial size 계산
        spatial_size = int(num_patches ** 0.5)
        feature = feature.reshape(batch_size, num_channels, spatial_size, spatial_size)
        
        return [feature]

    def _get_vit_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """DeiT feature extraction"""
        feature = self.feature_extractor.patch_embed(input_tensor)
        cls_token = self.feature_extractor.cls_token.expand(feature.shape[0], -1, -1)
        
        if self.feature_extractor.dist_token is None:
            feature = torch.cat((cls_token, feature), dim=1)
        else:
            feature = torch.cat(
                (
                    cls_token,
                    self.feature_extractor.dist_token.expand(feature.shape[0], -1, -1),
                    feature,
                ),
                dim=1,
            )
        
        feature = self.feature_extractor.pos_drop(feature + self.feature_extractor.pos_embed)
        
        for i in range(8):  # paper Table 6. Block Index = 7
            feature = self.feature_extractor.blocks[i](feature)
        
        feature = self.feature_extractor.norm(feature)
        feature = feature[:, 2:, :]  # Remove cls and dist tokens
        
        batch_size, num_patches, num_channels = feature.shape
        feature = feature.permute(0, 2, 1)
        
        # 패치 수에서 spatial size 계산
        spatial_size = int(num_patches ** 0.5)
        feature = feature.reshape(batch_size, num_channels, spatial_size, spatial_size)
        
        return [feature]


#####################################################################
# Trainer for FastFlow Model
#####################################################################
from .components.trainer import BaseTrainer, EarlyStopper

class FastflowTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="wide_resnet50_2", input_size=(256, 256)):

        if model is None:
            model = FastflowModel(backbone=backbone, input_size=input_size, pre_trained=True)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # if early_stopper_loss is None:
        #     early_stopper_loss = EarlyStopper(patience=10, min_delta=0.01, mode='min', target_value=0.1)
        # if early_stopper_auroc is None:
        #     early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max', target_value=0.995)
        if loss_fn is None:
            loss_fn = FastflowLoss()

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5

        self.model.feature_extractor.eval()

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)
        
        self.optimizer.zero_grad()
        hidden_variables, jacobians = self.model(images)
        loss = self.loss_fn(hidden_variables, jacobians)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}