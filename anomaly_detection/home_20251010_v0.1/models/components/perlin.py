#############################################################
# anomalib/src/anomalib/data/utils/generators/perlin.py
#############################################################

from pathlib import Path

import torch
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2

from .multi_random_choice import MultiRandomChoice


def generate_perlin_noise(
    height: int,
    width: int,
    scale: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle scale parameter
    if scale is None:
        min_scale, max_scale = 0, 6
        scalex = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
        scaley = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
    else:
        scalex, scaley = scale

    # Ensure dimensions are powers of 2 for proper noise generation
    def nextpow2(value: int) -> int:
        return int(2 ** torch.ceil(torch.log2(torch.tensor(value))).int().item())

    pad_h = nextpow2(height)
    pad_w = nextpow2(width)

    # Generate base grid
    delta = (scalex / pad_h, scaley / pad_w)
    d = (pad_h // scalex, pad_w // scaley)

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, scalex, delta[0], device=device),
                torch.arange(0, scaley, delta[1], device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        % 1
    )

    # Generate random gradients
    angles = 2 * torch.pi * torch.rand(int(scalex) + 1, int(scaley) + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1: list[int | None], slice2: list[int | None]) -> torch.Tensor:
        return (
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
            .repeat_interleave(int(d[0]), 0)
            .repeat_interleave(int(d[1]), 1)
        )

    def dot(grad: torch.Tensor, shift: list[float]) -> torch.Tensor:
        return (
            torch.stack(
                (grid[:pad_h, :pad_w, 0] + shift[0], grid[:pad_h, :pad_w, 1] + shift[1]),
                dim=-1,
            )
            * grad[:pad_h, :pad_w]
        ).sum(dim=-1)

    # Calculate noise values at grid points
    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    # Interpolate between grid points using quintic curve
    def fade(t: torch.Tensor) -> torch.Tensor:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    t = fade(grid[:pad_h, :pad_w])
    noise = torch.sqrt(torch.tensor(2.0, device=device)) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]),
        torch.lerp(n01, n11, t[..., 0]),
        t[..., 1],
    )

    # Crop to desired dimensions
    return noise[:height, :width]


class PerlinAnomalyGenerator(v2.Transform):
    def __init__(
        self,
        anomaly_source_path: Path | str | None = None,
        probability: float = 0.5,
        blend_factor: float | tuple[float, float] = (0.2, 1.0),
        rotation_range: tuple[float, float] = (-90, 90),
    ) -> None:
        super().__init__()
        self.probability = probability
        self.blend_factor = blend_factor

        # Load anomaly source paths
        self.anomaly_source_paths: list[Path] = []
        if anomaly_source_path is not None:
            for img_ext in IMG_EXTENSIONS:
                self.anomaly_source_paths.extend(Path(anomaly_source_path).rglob("*" + img_ext))

        # Initialize perlin rotation transform
        self.perlin_rotation_transform = v2.RandomAffine(
            degrees=rotation_range,
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=0,
        )

        # Initialize augmenters
        self.augmenters = MultiRandomChoice(
            transforms=[
                v2.ColorJitter(contrast=(0.5, 2.0)),
                v2.RandomPhotometricDistort(
                    brightness=(0.8, 1.2),
                    contrast=(1.0, 1.0),  # No contrast change
                    saturation=(1.0, 1.0),  # No saturation change
                    hue=(0.0, 0.0),  # No hue change
                    p=1.0,
                ),
                v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
                v2.ColorJitter(hue=[-50 / 360, 50 / 360], saturation=[0.5, 1.5]),
                v2.RandomSolarize(threshold=torch.empty(1).uniform_(32 / 255, 128 / 255).item(), p=1.0),
                v2.RandomPosterize(bits=4, p=1.0),
                v2.RandomInvert(p=1.0),
                v2.AutoAugment(),
                v2.RandomEqualize(p=1.0),
                v2.RandomAffine(degrees=(-45, 45), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            ],
            probabilities=None,
            num_transforms=3,
            fixed_num_transforms=True,
        )

    def generate_perturbation(
        self,
        height: int,
        width: int,
        device: torch.device | None = None,
        anomaly_source_path: Path | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate perlin noise
        perlin_noise = generate_perlin_noise(height, width, device=device)

        # Create rotated noise pattern
        perlin_noise = perlin_noise.unsqueeze(0)  # [1, H, W]
        perlin_noise = self.perlin_rotation_transform(perlin_noise).squeeze(0)  # [H, W]

        # Generate binary mask from perlin noise
        mask = torch.where(
            perlin_noise > 0.5,
            torch.ones_like(perlin_noise, device=device),
            torch.zeros_like(perlin_noise, device=device),
        ).unsqueeze(-1)  # [H, W, 1]

        # Generate anomaly source image
        if anomaly_source_path:
            anomaly_source_img = (
                io.read_image(str(anomaly_source_path), mode=io.ImageReadMode.RGB).float().to(device) / 255.0
            )
            if anomaly_source_img.shape[-2:] != (height, width):
                anomaly_source_img = v2.functional.resize(anomaly_source_img, [height, width], antialias=True)
            anomaly_source_img = anomaly_source_img.permute(1, 2, 0)  # [H, W, C]
        else:
            anomaly_source_img = perlin_noise.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, C]
            anomaly_source_img = (anomaly_source_img * 0.5) + 0.25  # Adjust intensity range

        # Apply augmentations to source image
        anomaly_augmented = self.augmenters(anomaly_source_img.permute(2, 0, 1))  # [C, H, W]
        anomaly_augmented = anomaly_augmented.permute(1, 2, 0)  # [H, W, C]

        # Create final perturbation by applying mask
        perturbation = anomaly_augmented * mask

        return perturbation, mask

    def _transform_image(
        self,
        img: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1, device=device) > self.probability:
            return img, torch.zeros((1, h, w), device=device)

        anomaly_source_path = (
            list(self.anomaly_source_paths)[int(torch.randint(len(self.anomaly_source_paths), (1,)).item())]
            if self.anomaly_source_paths
            else None
        )

        perturbation, mask = self.generate_perturbation(h, w, device, anomaly_source_path)
        perturbation = perturbation.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        beta = (
            self.blend_factor
            if isinstance(self.blend_factor, float)
            else torch.rand(1, device=device) * (self.blend_factor[1] - self.blend_factor[0]) + self.blend_factor[0]
            if isinstance(self.blend_factor, tuple)
            # Add type guard
            else torch.tensor(0.5, device=device)  # Fallback value
        )

        if not isinstance(beta, float):
            beta = beta.view(-1, 1, 1).expand_as(img)

        augmented_img = img * (1 - mask) + beta * perturbation + (1 - beta) * img * mask
        return augmented_img, mask

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = img.device
        is_batch = len(img.shape) == 4

        if is_batch:
            batch, _, height, width = img.shape
            # Initialize batch outputs
            batch_augmented = []
            batch_masks = []

            for i in range(batch):
                # Apply transform to each image in batch
                augmented, mask = self._transform_image(img[i], height, width, device)
                batch_augmented.append(augmented)
                batch_masks.append(mask)

            return torch.stack(batch_augmented), torch.stack(batch_masks)

        # Handle single image
        return self._transform_image(img, img.shape[1], img.shape[2], device)