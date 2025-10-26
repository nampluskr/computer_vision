"""
- DFM (2019): Deep Feature Modeling (DFM) for anomaly detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dfm
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/dfm.html
  - https://arxiv.org/abs/1909.11786
"""

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .components.pca import PCA
from .components.dynamic_buffer import DynamicBufferMixin
from .components.feature_extractor import TimmFeatureExtractor


#####################################################################
# anomalib/src/anomalib/models/image/dfm/torch_model.py
#####################################################################

class SingleClassGaussian(DynamicBufferMixin):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean_vec", torch.empty(0))
        self.register_buffer("u_mat", torch.empty(0))
        self.register_buffer("sigma_mat", torch.empty(0))

        self.mean_vec: torch.Tensor
        self.u_mat: torch.Tensor
        self.sigma_mat: torch.Tensor

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit a Gaussian model to dataset X."""
        num_samples = dataset.shape[1]
        # self.mean_vec = torch.mean(dataset, dim=1, device=dataset.device)
        self.mean_vec = torch.mean(dataset, dim=1)
        data_centered = (dataset - self.mean_vec.reshape(-1, 1)) / math.sqrt(num_samples)
        self.u_mat, self.sigma_mat, _ = torch.linalg.svd(data_centered, full_matrices=False)

    def score_samples(self, features: torch.Tensor) -> torch.Tensor:
        features_transformed = torch.matmul(features - self.mean_vec, self.u_mat / self.sigma_mat)
        return torch.sum(features_transformed * features_transformed, dim=1) + 2 * torch.sum(torch.log(self.sigma_mat))

    def forward(self, dataset: torch.Tensor) -> None:
        self.fit(dataset)


class DFMModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        layer: str,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        n_comps: float = 0.97,
        score_type: str = "fre",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.n_components = n_comps
        self.pca_model = PCA(n_components=self.n_components)
        self.gaussian_model = SingleClassGaussian()
        self.score_type = score_type
        self.layer = layer
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=[layer],
        ).eval()

        self.memory_bank: list[torch.tensor] = []

    def fit(self) -> None:
        """Fit PCA and Gaussian model to dataset."""
        self.memory_bank = torch.vstack(self.memory_bank)
        self.pca_model.fit(self.memory_bank)
        if self.score_type == "nll":
            features_reduced = self.pca_model.transform(self.memory_bank)
            self.gaussian_model.fit(features_reduced.T)

        # clear memory bank, reduces GPU size
        self.memory_bank = []

    def score(self, features: torch.Tensor, feature_shapes: tuple) -> torch.Tensor:
        batch_size, channels, height, width = feature_shapes
        feats_projected = self.pca_model.transform(features)  # [B, n_components]
        feats_reconstructed = self.pca_model.inverse_transform(feats_projected)  # [B, n_features]
        reconstruction_error = torch.square(features - feats_reconstructed)  # [B, n_features]
        fre_spatial = reconstruction_error.reshape(batch_size, channels, height, width)
        score_map = torch.sum(fre_spatial, dim=1, keepdim=True)

        if self.score_type == "nll":
            # Use Mahalanobis distance in the Gaussian model space
            # Transform to Gaussian model space
            features_transformed = torch.matmul(
                feats_projected - self.gaussian_model.mean_vec,
                self.gaussian_model.u_mat / self.gaussian_model.sigma_mat
            )  # [B, n_components]
            # Compute negative log-likelihood
            # = Mahalanobis distance + log determinant of covariance
            mahalanobis_sq = torch.sum(features_transformed ** 2, dim=1)  # [B]
            log_det_term = 2 * torch.sum(torch.log(self.gaussian_model.sigma_mat))
            score = mahalanobis_sq + log_det_term  # [B]
        elif self.score_type == "fre":
            # Simply sum all reconstruction errors
            score = torch.sum(reconstruction_error, dim=1)  # [B]
        else:
            msg = f"Unsupported score type: {self.score_type}. Use 'fre' or 'nll'."
            raise ValueError(msg)

        return score, score_map

    # def score(self, features: torch.Tensor, feature_shapes: tuple) -> torch.Tensor:
    #     feats_projected = self.pca_model.transform(features)
    #     if self.score_type == "nll":
    #         score = self.gaussian_model.score_samples(feats_projected)
    #     elif self.score_type == "fre":
    #         feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
    #         fre = torch.square(features - feats_reconstructed).reshape(feature_shapes)
    #         score_map = torch.unsqueeze(torch.sum(fre, dim=1), 1)
    #         score = torch.sum(torch.square(features - feats_reconstructed), dim=1)
    #     else:
    #         msg = f"unsupported score type: {self.score_type}"
    #         raise ValueError(msg)

    #     return (score, None) if self.score_type == "nll" else (score, score_map)

    def get_features(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        with torch.no_grad():
            features = self.feature_extractor(batch)[self.layer]
            batch_size = len(features)
            if self.pooling_kernel_size > 1:
                features = F.avg_pool2d(input=features, kernel_size=self.pooling_kernel_size)
            feature_shapes = features.shape
            features = features.view(batch_size, -1)
        return features, feature_shapes

    def forward(self, batch: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        feature_vector, feature_shapes = self.get_features(batch)

        if self.training:
            self.memory_bank.append(feature_vector)
            return feature_vector

        pred_score, anomaly_map = self.score(feature_vector.view(feature_vector.shape[:2]), feature_shapes)
        if anomaly_map is not None:
            anomaly_map = F.interpolate(anomaly_map, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


#####################################################################
# Trainer for CFlow Model
#####################################################################
import os
from tqdm import tqdm
from .components.trainer import BaseTrainer, EarlyStopper

class DFMTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="resnet50", layer="layer3", n_comp=0.97, score_type="fre"):

        if model is None:
            # score_type = fre (feature reconstruction error) or nll (negative log-likelihood).
            model = DFMModel(backbone=backbone, layer=layer, pre_trained=True, score_type=score_type)

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 1
        self.model_fitted = False
        self.score_type = score_type

    @torch.no_grad()
    def _extract_features(self, loader):
        self.model.train()  # Set to train mode to store features
        self.model.memory_bank = []  # Reset memory bank

        print(" > Extracting features from training data...")
        pbar = tqdm(loader, desc=" > Extracting", leave=False, ascii=True)
        for batch in pbar:
            imgs = batch["image"].to(self.device)
            _ = self.model(imgs)  # Features are stored in model.memory_bank

        print(f" > Extracted {len(self.model.memory_bank)} batches of features\n")

    @torch.no_grad()
    def _fit_models(self):
        if self.model_fitted:
            return

        print(" > Fitting PCA and Gaussian models to embeddings...")
        self.model.fit()
        self.model_fitted = True

        # Print model information
        print(f" > PCA n_components: {self.model.pca_model.n_components}")
        if hasattr(self.model.pca_model, 'components') and self.model.pca_model.components is not None:
            print(f" > PCA components shape: {self.model.pca_model.components.shape}")

        if self.score_type == "nll":
            print(f" > Gaussian mean shape: {self.model.gaussian_model.mean_vec.shape}")
            print(f" > Gaussian U matrix shape: {self.model.gaussian_model.u_mat.shape}")
        print()

    def on_train_start(self, train_loader):
        self._extract_features(train_loader)
        self._fit_models()

    @torch.enable_grad()
    def train_step(self, batch):
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def train_epoch(self, train_loader):
        return {"loss": 0.0}

    def save_model(self, weight_path):
        if weight_path is not None:
            output_dir = os.path.abspath(os.path.dirname(weight_path))
            os.makedirs(output_dir, exist_ok=True)

            checkpoint = {
                "score_type": self.model.score_type,
                "n_components": self.model.pca_model.n_components,
            }
            # Save PCA parameters
            if hasattr(self.model.pca_model, 'mean') and self.model.pca_model.mean is not None:
                checkpoint["pca_mean"] = self.model.pca_model.mean.cpu()
            if hasattr(self.model.pca_model, 'components') and self.model.pca_model.components is not None:
                checkpoint["pca_components"] = self.model.pca_model.components.cpu()
            if hasattr(self.model.pca_model, 'singular_values') and self.model.pca_model.singular_values is not None:
                checkpoint["pca_singular_values"] = self.model.pca_model.singular_values.cpu()

            # Save Gaussian model parameters (only for 'nll' score type)
            if self.model.score_type == "nll":
                checkpoint["gaussian_mean_vec"] = self.model.gaussian_model.mean_vec.cpu()
                checkpoint["gaussian_u_mat"] = self.model.gaussian_model.u_mat.cpu()
                checkpoint["gaussian_sigma_mat"] = self.model.gaussian_model.sigma_mat.cpu()

            torch.save(checkpoint, weight_path)
            print(f" > DFM model parameters saved to: {weight_path}")
            print(f"   - Score type: {self.model.score_type}")
            print(f"   - PCA components: {self.model.pca_model.n_components}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)

            # Load basic config
            self.model.score_type = checkpoint["score_type"]
            self.model.pca_model.n_components = checkpoint["n_components"]

            # Load PCA parameters
            if "pca_mean" in checkpoint:
                self.model.pca_model.mean = checkpoint["pca_mean"].to(self.device)
            if "pca_components" in checkpoint:
                self.model.pca_model.components = checkpoint["pca_components"].to(self.device)
            if "pca_singular_values" in checkpoint:
                self.model.pca_model.singular_values = checkpoint["pca_singular_values"].to(self.device)

            # Load Gaussian parameters (only for 'nll' score type)
            if self.model.score_type == "nll":
                if "gaussian_mean_vec" in checkpoint:
                    self.model.gaussian_model.mean_vec = checkpoint["gaussian_mean_vec"].to(self.device)
                if "gaussian_u_mat" in checkpoint:
                    self.model.gaussian_model.u_mat = checkpoint["gaussian_u_mat"].to(self.device)
                if "gaussian_sigma_mat" in checkpoint:
                    self.model.gaussian_model.sigma_mat = checkpoint["gaussian_sigma_mat"].to(self.device)

            self.model_fitted = True
            print(f" > Loaded DFM model parameters from: {weight_path}")
            print(f"   - Score type: {self.model.score_type}")
            print(f"   - PCA components: {self.model.pca_model.n_components}")
            if self.model.score_type == "nll":
                print(f"   - Gaussian mean shape: {self.model.gaussian_model.mean_vec.shape}")
            print()
        else:
            print(f" > No checkpoint found at: {weight_path}\n")