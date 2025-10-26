#############################################################
# anomalib/src/anomalib/models/components/stats/multi_variate_gaussian.py
#############################################################

from typing import Any

import torch
from torch import nn

from .dynamic_buffer import DynamicBufferMixin


class MultiVariateGaussian(DynamicBufferMixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inv_covariance", torch.empty(0))

        self.mean: torch.Tensor
        self.inv_covariance: torch.Tensor

    @staticmethod
    def _cov(
        observations: torch.Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: int | None = None,
        aweights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ensure at least 2D
        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            ddof = 1 if bias == 0 else 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        # Determine the normalization
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        x_transposed = observations_m.t() if weights is None else torch.mm(torch.diag(weights), observations_m).t()

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        # Stabilize the covariance matrix by adding a small regularization term
        stabilized_covariance = covariance.permute(2, 0, 1) + 1e-5 * identity

        # Check if the device is MPS and fallback to CPU if necessary
        if device.type == "mps":
            # Move stabilized covariance to CPU for inversion
            self.inv_covariance = torch.linalg.inv(stabilized_covariance.cpu()).to(device)
        else:
            # Calculate inverse covariance as we need only the inverse
            self.inv_covariance = torch.linalg.inv(stabilized_covariance)

        return [self.mean, self.inv_covariance]

    def fit(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        return self.forward(embedding)