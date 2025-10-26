"""
- PatchCore (2022): Towards Total Recall in Industrial Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/patchcore
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/patchcore.html
  - https://arxiv.org/pdf/2106.08265.pdf
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

# from anomalib.data import InferenceBatch
# from anomalib.models.components import DynamicBufferMixin, KCenterGreedy, TimmFeatureExtractor
# from anomalib.utils import deprecate
# from .anomaly_map import AnomalyMapGenerator

from .components.feature_extractor import TimmFeatureExtractor
from .components.tiler import Tiler
from .components.blur import GaussianBlur2d
from .components.dynamic_buffer import DynamicBufferMixin
from .components.k_center_greedy import KCenterGreedy


#####################################################################
# anomalib/src/anomalib/models/image/patchcore/anomaly_map.py
#####################################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(
        self,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = F.interpolate(patch_scores, size=(image_size[0], image_size[1]))
        return self.blur(anomaly_map)

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        return self.compute_anomaly_map(patch_scores, image_size)


#####################################################################
# anomalib/src/anomalib/models/image/patchcore/torch_model.py
#####################################################################

class PatchcoreModel(DynamicBufferMixin, nn.Module):
    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.num_neighbors = num_neighbors

        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=self.layers,
        ).eval()
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.memory_bank: torch.Tensor
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store: list[torch.tensor] = []

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        # Ensure memory bank is not empty
        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores"
            raise ValueError(msg)

        # apply nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        # get anomaly map
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    # @deprecate(args={"embeddings": None}, since="2.1.0", reason="Use the default memory bank instead.")
    def subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
        if embeddings is not None:
            del embeddings

        if len(self.embedding_store) == 0:
            msg = "Embedding store is empty. Cannot perform coreset selection."
            raise ValueError(msg)

        # Coreset Subsampling
        self.memory_bank = torch.vstack(self.embedding_store)
        self.embedding_store.clear()

        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper


#####################################################################
# Trainer for PatchCore Model
#####################################################################
import os
from tqdm import tqdm
from .components.trainer import BaseTrainer, EarlyStopper

class PatchcoreTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="wide_resnet50_2", layers=["layer2", "layer3"]):

        if model is None:
            model = PatchcoreModel(backbone=backbone, layers=layers, pre_trained=True)

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 1

        self.coreset_sampling_ratio = 0.1
        self.memory_built = False

    @torch.no_grad()
    def _extract_embeddings(self, loader):
        self.model.train()
        self.model.embedding_store = []
        pbar = tqdm(loader, desc="Extracting embeddings", leave=False, ascii=True)
        for batch in pbar:
            imgs = batch["image"].to(self.device)
            _ = self.model(imgs)

    @torch.no_grad()
    def _build_memory_bank(self):
        if self.memory_built:
            return

        print("\n > Building PatchCore memory bank via Coreset sampling...")
        self.model.subsample_embedding(sampling_ratio=self.coreset_sampling_ratio)
        self.memory_built = True
        print(f" > Memory bank size: {self.model.memory_bank.shape[0]} patches\n")

    def on_train_start(self, train_loader):
        self._extract_embeddings(train_loader)
        self._build_memory_bank()

    @torch.enable_grad()
    def train_step(self, batch):
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def train_epoch(self, train_loader):
        return {"loss": 0.0}

    def save_model(self, weight_path):
        if weight_path is not None:
            output_dir = os.path.abspath(os.path.dirname(weight_path))
            os.makedirs(output_dir, exist_ok=True)

            checkpoint = {"memory_bank": self.model.memory_bank.cpu(),
                          "coreset_sampling_ratio": self.coreset_sampling_ratio}
            torch.save(checkpoint, weight_path)
            print(f" > PatchCore memory bank saved to: {weight_path}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.memory_bank = checkpoint["memory_bank"].to(self.device)
            self.memory_built = True
            print(f" > Loaded PatchCore memory bank from: {weight_path}")
            print(f" > Memory bank size: {self.model.memory_bank.shape[0]} patches")
        else:
            print(f" > No checkpoint found at: {weight_path}\n")