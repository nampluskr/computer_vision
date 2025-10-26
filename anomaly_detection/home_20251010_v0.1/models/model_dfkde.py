"""
- DFKDE (2022): Deep Feature Kernel Density Estimation
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dfkde
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/dfkde.html
  - github.com/openvinotoolkit/anomalib
"""

from abc import ABC
import logging
import random
from enum import Enum
from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

# from anomalib.data import InferenceBatch
# from anomalib.models.components import TimmFeatureExtractor
# from anomalib.models.components.classification import FeatureScalingMethod, KDEClassifier

from .components.feature_extractor import TimmFeatureExtractor
from .components.kde_classifier import FeatureScalingMethod, KDEClassifier

logger = logging.getLogger(__name__)


###########################################################
# anomalib\models\images\dfkde\torch_model.py
###########################################################

class DfkdeModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        layers: Sequence[str],
        pre_trained: bool = True,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.feature_extractor = TimmFeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers).eval()

        self.classifier = KDEClassifier(
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )
        self.memory_bank: list[torch.tensor] = []

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch)
        for layer in layer_outputs:
            batch_size = len(layer_outputs[layer])
            layer_outputs[layer] = F.adaptive_avg_pool2d(input=layer_outputs[layer], output_size=(1, 1))
            layer_outputs[layer] = layer_outputs[layer].view(batch_size, -1)
        return torch.cat(list(layer_outputs.values())).detach()

    def forward(self, batch: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        # 1. apply feature extraction
        features = self.get_features(batch)
        if self.training:
            self.memory_bank.append(features)
            return features

        # 2. apply density estimation
        scores = self.classifier(features)
        return dict(pred_score=scores)

    def fit(self) -> None:
        if len(self.memory_bank) == 0:
            msg = "Memory bank is empty. Cannot perform coreset selection."
            raise ValueError(msg)
        self.memory_bank = torch.vstack(self.memory_bank)

        # fit gaussian
        self.classifier.fit(self.memory_bank)

        # clear memory bank, redcues gpu size
        self.memory_bank = []


#############################################################
# Trainer for DFKDE Model
#############################################################
import os
from tqdm import tqdm
from .components.trainer import BaseTrainer, EarlyStopper

class DFKDETrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="resnet18", layers=["layer4"], pre_trained=True):

        if model is None:
            model =  DfkdeModel(layers=layers, backbone=backbone, pre_trained=pre_trained,
                n_pca_components=16, feature_scaling_method=FeatureScalingMethod.SCALE,
                max_training_points=40000)

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 1

    def on_train_start(self, train_loader):
        self.model.train()
        self.model.memory_bank = []

        pbar = tqdm(train_loader, desc=" > Extracting features", leave=False, ascii=True)
        for batch in pbar:
            images = batch["image"].to(self.device)
            _ = self.model(images)

        print(f" > Extracted {len(self.model.memory_bank)} batches of features\n")

    def on_train_end(self, train_results):
        print(" > Fitting KDE classifier...")
        self.model.fit()
        super().on_train_end(train_results)

    @torch.enable_grad()
    def train_step(self, batch):
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def train_epoch(self, train_loader):
        return {"loss": 0.0}

    @torch.no_grad()
    def save_maps(self, test_loader, result_dir=None, desc=None, show_image=False,
                  skip_normal=False, skip_anomaly=False, num_max=-1, normalize=True):
        print("\n > DFKDE model does not support anomaly map visualization.")
        print(" > Skipping anomaly map generation.\n")
