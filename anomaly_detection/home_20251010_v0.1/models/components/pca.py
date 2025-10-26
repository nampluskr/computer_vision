#####################################################################
# anomalib/src/anomalib/models/components/dimensionality_reduction/pca.py
#####################################################################

import torch

from .dynamic_buffer import DynamicBufferMixin


class PCA(DynamicBufferMixin):
    def __init__(self, n_components: int | float) -> None:
        super().__init__()
        self.n_components = n_components

        self.register_buffer("singular_vectors", torch.empty(0))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("num_components", torch.empty(0))

        self.singular_vectors: torch.Tensor
        self.singular_values: torch.Tensor
        self.mean: torch.Tensor
        self.num_components: torch.Tensor

    def fit(self, dataset: torch.Tensor) -> None:
        mean = dataset.mean(dim=0)
        dataset -= mean

        _, sig, v_h = torch.linalg.svd(dataset.double(), full_matrices=False)
        num_components: int
        if self.n_components <= 1:
            variance_ratios = torch.cumsum(sig * sig, dim=0) / torch.sum(sig * sig)
            num_components = torch.nonzero(variance_ratios >= self.n_components)[0]
        else:
            num_components = int(self.n_components)

        self.num_components = torch.tensor([num_components], device=dataset.device)

        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components].float()
        self.singular_values = sig[:num_components].float()
        self.mean = mean

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        mean = dataset.mean(dim=0)
        dataset -= mean
        num_components = int(self.n_components)
        self.num_components = torch.tensor([num_components], device=dataset.device)

        v_h = torch.linalg.svd(dataset)[-1]
        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components]
        self.mean = mean

        return torch.matmul(dataset, self.singular_vectors)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        features -= self.mean
        return torch.matmul(features, self.singular_vectors)

    def inverse_transform(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features, self.singular_vectors.transpose(-2, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.transform(features)