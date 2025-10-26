#############################################################
# anomalib/src/anomalib/data/transforms/multi_random_choice.py
#############################################################

from collections.abc import Callable, Sequence

import torch
from torchvision.transforms import v2


class MultiRandomChoice(v2.Transform):
    def __init__(
        self,
        transforms: Sequence[Callable],
        probabilities: list[float] | None = None,
        num_transforms: int = 1,
        fixed_num_transforms: bool = False,
    ) -> None:
        if not isinstance(transforms, Sequence):
            msg = "Argument transforms should be a sequence of callables"
            raise TypeError(msg)

        if probabilities is None:
            probabilities = [1.0] * len(transforms)
        elif len(probabilities) != len(transforms):
            msg = f"Length of p doesn't match the number of transforms: {len(probabilities)} != {len(transforms)}"
            raise ValueError(msg)

        super().__init__()

        self.transforms = transforms
        total = sum(probabilities)
        self.probabilities = [probability / total for probability in probabilities]

        self.num_transforms = num_transforms
        self.fixed_num_transforms = fixed_num_transforms

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # First determine number of transforms to apply
        num_transforms = (
            self.num_transforms if self.fixed_num_transforms else int(torch.randint(self.num_transforms, (1,)) + 1)
        )
        # Get transforms
        idx = torch.multinomial(torch.tensor(self.probabilities), num_transforms).tolist()
        transform = v2.Compose([self.transforms[i] for i in idx])
        return transform(*inputs)