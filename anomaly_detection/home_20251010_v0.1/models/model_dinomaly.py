"""
- Dinomaly (2025): The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dinomaly
  - https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/dinomaly.html
  - https://github.com/guojiajeremy/Dinomaly
  - https://arxiv.org/abs/2405.14325

- model weights (regular)
  - dinov2_vit_small_14: https://huggingface.co/FoundationVision/unitok_external/blob/main/dinov2_vits14_pretrain.pth
  - dinov2_vit_base_14:  https://huggingface.co/spaces/BoukamchaSmartVisions/FeatureMatching/blob/main/models/dinov2_vitb14_pretrain.pth
  - dinov2_vit_large_14: https://huggingface.co/Cusyoung/CrossEarth/blob/main/dinov2_vitl14_pretrain.pth
- model weights (reg)
  - dinov2reg_vit_small_14: https://dl.fbaipublicfiles.com/dinov2/dinov2_vit_s_14/dinov2_vit_s_14_reg4_pretrain.pth
  - dinov2reg_vit_base_14:  https://dl.fbaipublicfiles.com/dinov2/dinov2_vit_b_14/dinov2_vit_b_14_reg4_pretrain.pth
  - dinov2reg_vit_large_14: https://dl.fbaipublicfiles.com/dinov2/dinov2_vit_l_14/dinov2_vit_l_14_reg4_pretrain.pth
"""


import math
from functools import partial

import torch
from torch.nn.init import trunc_normal_
import torch.nn.functional as F  # noqa: N812
from timm.layers.drop import DropPath
from torch import nn

# from anomalib.data import InferenceBatch
# from anomalib.models.components import GaussianBlur2d
# from anomalib.models.image.dinomaly.components import CosineHardMiningLoss, DinomalyMLP, LinearAttention
# from anomalib.models.image.dinomaly.components import load as load_dinov2_model

from .components.blur import GaussianBlur2d
from .components_dinomaly.loss import CosineHardMiningLoss
from .components_dinomaly.layers import DinomalyMLP, LinearAttention
from .components_dinomaly.dinov2_loader import load as load_dinov2_model


#####################################################################
# anomalib/src/anomalib/models/image/dinomaly/torch_model.py
#####################################################################

# Encoder architecture configurations for DINOv2 models.
# The target layers are the
DINOV2_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
}

# Default fusion layer configurations
# Instead of comparing layer to layer between encoder and decoder, dinomaly uses
# layer groups to fuse features from multiple layers.
# By Default, the first 4 layers and the last 4 layers are fused.
# Note that these are the layer indices of the encoder and decoder layers used for feature extraction.
DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]

# Default values for inference processing
DEFAULT_RESIZE_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_MAX_RATIO = 0.01

# Transformer architecture constants
TRANSFORMER_CONFIG: dict[str, float | bool] = {
    "mlp_ratio": 4.0,
    "layer_norm_eps": 1e-8,
    "qkv_bias": True,
    "attn_drop": 0.0,
}


class DinomalyModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
    ) -> None:
        super().__init__()

        if target_layers is None:
            # 8 middle layers of the encoder are used for feature extraction.
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

        # Instead of comparing layer to layer between encoder and decoder, dinomaly uses
        # layer groups to fuse features from multiple layers.
        if fuse_layer_encoder is None:
            fuse_layer_encoder = DEFAULT_FUSE_LAYERS
        if fuse_layer_decoder is None:
            fuse_layer_decoder = DEFAULT_FUSE_LAYERS

        encoder = load_dinov2_model(encoder_name)

        # Extract architecture configuration based on the model name
        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]
        target_layers = arch_config["target_layers"]

        # Add validation
        if decoder_depth <= 1:
            msg = f"decoder_depth must be greater than 1, got {decoder_depth}"
            raise ValueError(msg)

        bottleneck = []
        bottle_neck_mlp = DinomalyMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            act_layer=nn.GELU,
            drop=bottleneck_dropout,
            bias=False,
            apply_input_dropout=True,  # Apply dropout to input
        )
        bottleneck.append(bottle_neck_mlp)
        bottleneck = nn.ModuleList(bottleneck)

        decoder = []
        for _ in range(decoder_depth):
            # Extract and validate config values for type safety
            mlp_ratio_val = TRANSFORMER_CONFIG["mlp_ratio"]
            assert isinstance(mlp_ratio_val, float)
            qkv_bias_val = TRANSFORMER_CONFIG["qkv_bias"]
            assert isinstance(qkv_bias_val, bool)
            layer_norm_eps_val = TRANSFORMER_CONFIG["layer_norm_eps"]
            assert isinstance(layer_norm_eps_val, float)
            attn_drop_val = TRANSFORMER_CONFIG["attn_drop"]
            assert isinstance(attn_drop_val, float)

            decoder_block = DecoderViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_val,
                qkv_bias=qkv_bias_val,
                norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps_val),  # type: ignore[arg-type]
                attn_drop=attn_drop_val,
                attn=LinearAttention,
            )
            decoder.append(decoder_block)
        decoder = nn.ModuleList(decoder)

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

        # Initialize Gaussian blur for anomaly map smoothing
        self.gaussian_blur = GaussianBlur2d(
            sigma=DEFAULT_GAUSSIAN_SIGMA,
            channels=1,
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        )

        self.loss_fn = CosineHardMiningLoss()

    def get_encoder_decoder_outputs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x = self.encoder.prepare_tokens(x)

        encoder_features = []
        decoder_features = []

        for i, block in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = block(x)
            else:
                continue
            if i in self.target_layers:
                encoder_features.append(x)
        side = int(math.sqrt(encoder_features[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            encoder_features = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in encoder_features]

        x = self._fuse_feature(encoder_features)
        for _i, block in enumerate(self.bottleneck):
            x = block(x)

        # attn_mask is explicitly set to None to disable attention masking.
        # This will not have any effect as it was essentially set to None in the original implementation
        # as well but was configurable to be not None for testing, if required.
        for _i, block in enumerate(self.decoder):
            x = block(x, attn_mask=None)
            decoder_features.append(x)
        decoder_features = decoder_features[::-1]

        en = [self._fuse_feature([encoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([decoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        # Process features for spatial output
        en = self._process_features_for_spatial_output(en, side)
        de = self._process_features_for_spatial_output(de, side)
        return en, de

    def forward(self, batch: torch.Tensor, global_step: int | None = None) -> torch.Tensor | dict[str, torch.Tensor]:
        en, de = self.get_encoder_decoder_outputs(batch)
        image_size = batch.shape[2]

        if self.training:
            if global_step is None:
                error_msg = "global_step must be provided during training"
                raise ValueError(error_msg)

            return self.loss_fn(encoder_features=en, decoder_features=de, global_step=global_step)

        # If inference, calculate anomaly maps, predictions, from the encoder and decoder features.
        anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=image_size)
        anomaly_map_resized = anomaly_map.clone()

        # Resize anomaly map for processing
        if DEFAULT_RESIZE_SIZE is not None:
            anomaly_map = F.interpolate(anomaly_map, size=DEFAULT_RESIZE_SIZE, mode="bilinear", align_corners=False)

        # Apply Gaussian smoothing
        anomaly_map = self.gaussian_blur(anomaly_map)

        # Calculate anomaly score
        if DEFAULT_MAX_RATIO == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            anomaly_map_flat = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][
                :,
                : int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO),
            ]
            sp_score = sp_score.mean(dim=1)
        pred_score = sp_score

        return dict(pred_score=pred_score, anomaly_map=anomaly_map_resized)

    @staticmethod
    def calculate_anomaly_maps(
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)

        anomaly_map_list = []
        for i in range(len(target_feature_maps)):
            fs = source_feature_maps[i]
            ft = target_feature_maps[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            anomaly_map_list.append(a_map)
        anomaly_map = torch.cat(anomaly_map_list, dim=1).mean(dim=1, keepdim=True)
        return anomaly_map, anomaly_map_list

    @staticmethod
    def _fuse_feature(feat_list: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(feat_list, dim=1).mean(dim=1)

    @staticmethod
    def _get_architecture_config(encoder_name: str, target_layers: list[int] | None) -> dict:
        for arch_name, config in DINOV2_ARCHITECTURES.items():
            if arch_name in encoder_name:
                result = config.copy()
                # Override target_layers if explicitly provided
                if target_layers is not None:
                    result["target_layers"] = target_layers
                return result

        msg = f"Architecture not supported. Encoder name must contain one of {list(DINOV2_ARCHITECTURES.keys())}"
        raise ValueError(msg)

    def _process_features_for_spatial_output(
        self,
        features: list[torch.Tensor],
        side: int,
    ) -> list[torch.Tensor]:
        # Remove class token and register tokens if not already removed
        if not self.remove_class_token:
            features = [f[:, 1 + self.encoder.num_register_tokens :, :] for f in features]

        # Reshape to spatial dimensions
        batch_size = features[0].shape[0]
        return [f.permute(0, 2, 1).reshape([batch_size, -1, side, side]).contiguous() for f in features]


class DecoderViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float | None = None,
        qkv_bias: bool | None = None,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn: type[nn.Module] = LinearAttention,
    ) -> None:
        super().__init__()

        # Use default values from TRANSFORMER_CONFIG if not provided
        mlp_ratio_config = TRANSFORMER_CONFIG["mlp_ratio"]
        assert isinstance(mlp_ratio_config, float)
        mlp_ratio = mlp_ratio if mlp_ratio is not None else mlp_ratio_config

        qkv_bias_config = TRANSFORMER_CONFIG["qkv_bias"]
        assert isinstance(qkv_bias_config, bool)
        qkv_bias = qkv_bias if qkv_bias is not None else qkv_bias_config

        attn_drop_config = TRANSFORMER_CONFIG["attn_drop"]
        assert isinstance(attn_drop_config, float)
        attn_drop = attn_drop if attn_drop is not None else attn_drop_config

        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinomalyMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
            apply_input_dropout=False,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through decoder block."""
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


#####################################################################
# Trainer for Dinomaly Model
#####################################################################
import os
from time import time
from typing import Any
from .components.trainer import BaseTrainer, EarlyStopper
from .components_dinomaly.optimizer import StableAdamW, WarmCosineScheduler


# Training constants
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
MAX_STEPS_DEFAULT = 5000

# Default Training hyperparameters
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 2e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-8,
    },
    "scheduler": {
        "base_value": 2e-3,
        "final_value": 2e-4,
        "total_iters": MAX_STEPS_DEFAULT,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_steps": MAX_STEPS_DEFAULT,
    },
}

class DinomalyTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 encoder_name="dinov2_vit_small_14", bottleneck_dropout=0.2, decoder_depth=8):
        if model is None:
            # encoder_name = "dinov2_vit_small_14", "dinov2_vit_base_14", "dinov2_vit_large_14"
            # encoder_name = "dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"
            model = DinomalyModel(encoder_name=encoder_name,
                bottleneck_dropout=bottleneck_dropout, decoder_depth=decoder_depth
            )
            # Only the bottleneck and decoder parameters are trained.
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze bottleneck and decoder
            for param in model.bottleneck.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True

            self.trainable_modules = torch.nn.ModuleList([model.bottleneck, model.decoder])
            self._initialize_trainable_modules(self.trainable_modules)
        if optimizer is None:
            optimizer_config = TRAINING_CONFIG["optimizer"]
            optimizer = StableAdamW([{"params": self.trainable_modules.parameters()}], **optimizer_config)
        if scheduler is None:
            scheduler_config = TRAINING_CONFIG["scheduler"].copy()
            scheduler_config["total_iters"] = -1
            scheduler = WarmCosineScheduler(optimizer, **scheduler_config)

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 1
        self.global_step = 0

    @staticmethod
    def _initialize_trainable_modules(trainable_modules: torch.nn.ModuleList) -> None:
        for m in trainable_modules.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

    def on_fit_start(self):
        super().on_fit_start()
        self.global_step = 0

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(images, global_step=self.global_step)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        results = {"loss": loss.item()}
        return results

    def _step_scheduler(self):
        if self.scheduler is None:
            return

        old_lr = self.optimizer.param_groups[0]["lr"]

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # mode='max', factor=0.5, patience=5, threshold=1e-3, verbose=True
            metric = self.valid_info["auroc"] if self.valid_info else None
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires a metric (e.g., auroc).")
            self.scheduler.step(metric)

        # ② OneCycleLR / CyclicLR 등 step‑based 스케줄러
        elif isinstance(self.scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
            self.scheduler.step()
        else:
            self.scheduler.step()

        new_lr = self.optimizer.param_groups[0]["lr"]
        if abs(old_lr - new_lr) > 1e-10:
            print(f" > Learning rate changed: {old_lr:.3e} -> {new_lr:.3e}\n")

    def on_epoch_end(self):
        if self.valid_info is None:
            elapsed_time = time() - self.epoch_start_time
            epoch_info = f" [{self.epoch:3d}/{self.num_epochs}]"
            print(f" {epoch_info} {self.train_info} ({elapsed_time:.1f}s)")

        self._step_scheduler()

    def on_fit_end(self, weight_path=None):
        elapsed = time() - self.fit_start_time
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        print(f"\n > Training finished in {hrs:02d}:{mins:02d}:{secs:02d} "
              f"({self.global_step} optimizer steps)\n")
        super().on_fit_end(weight_path)

    def save_model(self, weight_path):
        if weight_path is not None:
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            ckpt = {"model": self.model.state_dict(), "global_step": self.global_step}
            if self.optimizer is not None:
                ckpt["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                ckpt["scheduler"] = self.scheduler.state_dict()
            torch.save(ckpt, weight_path)
            print(f" > Model weights saved to: {weight_path}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            ckpt = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            print(f" > Model weights loaded from: {weight_path}")

            if self.optimizer is not None and "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                print(" > Optimizer state loaded.")
            if self.scheduler is not None and "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
                print(" > Scheduler state loaded.")
            if "global_step" in ckpt:
                self.global_step = ckpt["global_step"]
                print(f" > Global step restored to {self.global_step}")
        else:
            print(f" > No model weights found at: {weight_path}\n")
