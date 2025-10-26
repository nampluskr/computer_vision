#####################################################################
# anomalib/src/anomalib/data/utils/tiler.py
#####################################################################

from collections.abc import Sequence
from enum import Enum
from itertools import product
from math import ceil

import torch
import torchvision.transforms as T  # noqa: N812
from torch.nn import functional as F  # noqa: N812


class ImageUpscaleMode(str, Enum):
    PADDING = "padding"
    INTERPOLATION = "interpolation"


class StrideSizeError(Exception):
    """Error raised when stride size exceeds tile size."""


def compute_new_image_size(image_size: tuple, tile_size: tuple, stride: tuple) -> tuple:
    def __compute_new_edge_size(edge_size: int, tile_size: int, stride: int) -> int:
        if (edge_size - tile_size) % stride != 0:
            edge_size = (ceil((edge_size - tile_size) / stride) * stride) + tile_size

        return edge_size

    resized_h = __compute_new_edge_size(image_size[0], tile_size[0], stride[0])
    resized_w = __compute_new_edge_size(image_size[1], tile_size[1], stride[1])

    return resized_h, resized_w


def upscale_image(
    image: torch.Tensor,
    size: tuple,
    mode: ImageUpscaleMode = ImageUpscaleMode.PADDING,
) -> torch.Tensor:
    image_h, image_w = image.shape[2:]
    resize_h, resize_w = size

    if mode == ImageUpscaleMode.PADDING:
        pad_h = resize_h - image_h
        pad_w = resize_w - image_w

        image = F.pad(image, [0, pad_w, 0, pad_h])
    elif mode == ImageUpscaleMode.INTERPOLATION:
        image = F.interpolate(input=image, size=(resize_h, resize_w))
    else:
        msg = f"Unknown mode {mode}. Only padding and interpolation is available."
        raise ValueError(msg)

    return image


def downscale_image(
    image: torch.Tensor,
    size: tuple,
    mode: ImageUpscaleMode = ImageUpscaleMode.PADDING,
) -> torch.Tensor:
    input_h, input_w = size
    if mode == ImageUpscaleMode.PADDING:
        image = image[:, :, :input_h, :input_w]
    else:
        image = F.interpolate(input=image, size=(input_h, input_w))

    return image


class Tiler:
    def __init__(
        self,
        tile_size: int | Sequence,
        stride: int | Sequence | None = None,
        remove_border_count: int = 0,
        mode: ImageUpscaleMode = ImageUpscaleMode.PADDING,
    ) -> None:
        self.tile_size_h, self.tile_size_w = self.validate_size_type(tile_size)
        self.random_tile_count = 4

        if stride is not None:
            self.stride_h, self.stride_w = self.validate_size_type(stride)

        self.remove_border_count = remove_border_count
        self.overlapping = not (self.stride_h == self.tile_size_h and self.stride_w == self.tile_size_w)
        self.mode = mode

        if self.stride_h > self.tile_size_h or self.stride_w > self.tile_size_w:
            msg = "Stride size larger than tile size produces unreliable results. Ensure stride size <= tile size."
            raise StrideSizeError(msg)

        if self.mode not in {ImageUpscaleMode.PADDING, ImageUpscaleMode.INTERPOLATION}:
            msg = f"Unknown mode {self.mode}. Available modes: padding and interpolation"
            raise ValueError(msg)

        self.batch_size: int
        self.num_channels: int

        self.input_h: int
        self.input_w: int

        self.pad_h: int
        self.pad_w: int

        self.resized_h: int
        self.resized_w: int

        self.num_patches_h: int
        self.num_patches_w: int

    @staticmethod
    def validate_size_type(parameter: int | Sequence) -> tuple[int, ...]:
        if isinstance(parameter, int):
            output = (parameter, parameter)
        elif isinstance(parameter, Sequence):
            output = (parameter[0], parameter[1])
        else:
            msg = f"Invalid type {type(parameter)} for tile/stride size. Must be int or Sequence."
            raise TypeError(msg)

        if len(output) != 2:
            msg = f"Size must have length 2, got {len(output)}"
            raise ValueError(msg)

        return output

    def __random_tile(self, image: torch.Tensor) -> torch.Tensor:
        return torch.vstack([T.RandomCrop(self.tile_size_h)(image) for i in range(self.random_tile_count)])

    def __unfold(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        batch, channels, image_h, image_w = tensor.shape

        self.num_patches_h = int((image_h - self.tile_size_h) / self.stride_h) + 1
        self.num_patches_w = int((image_w - self.tile_size_w) / self.stride_w) + 1

        tiles = torch.zeros(
            (
                self.num_patches_h,
                self.num_patches_w,
                batch,
                channels,
                self.tile_size_h,
                self.tile_size_w,
            ),
            device=device,
        )

        for (tile_i, tile_j), (loc_i, loc_j) in zip(
            product(range(self.num_patches_h), range(self.num_patches_w)),
            product(
                range(0, image_h - self.tile_size_h + 1, self.stride_h),
                range(0, image_w - self.tile_size_w + 1, self.stride_w),
            ),
            strict=True,
        ):
            tiles[tile_i, tile_j, :] = tensor[
                :,
                :,
                loc_i : (loc_i + self.tile_size_h),
                loc_j : (loc_j + self.tile_size_w),
            ]

        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        return tiles.contiguous().view(-1, channels, self.tile_size_h, self.tile_size_w)

    def __fold(self, tiles: torch.Tensor) -> torch.Tensor:
        _, num_channels, tile_size_h, tile_size_w = tiles.shape
        scale_h, scale_w = (tile_size_h / self.tile_size_h), (tile_size_w / self.tile_size_w)
        device = tiles.device
        reduced_tile_h = tile_size_h - (2 * self.remove_border_count)
        reduced_tile_w = tile_size_w - (2 * self.remove_border_count)
        image_size = (
            self.batch_size,
            num_channels,
            int(self.resized_h * scale_h),
            int(self.resized_w * scale_w),
        )

        tiles = tiles.contiguous().view(
            self.batch_size,
            self.num_patches_h,
            self.num_patches_w,
            num_channels,
            tile_size_h,
            tile_size_w,
        )
        tiles = tiles.permute(0, 3, 1, 2, 4, 5)
        tiles = tiles.contiguous().view(self.batch_size, num_channels, -1, tile_size_h, tile_size_w)
        tiles = tiles.permute(2, 0, 1, 3, 4)

        tiles = tiles[
            :,
            :,
            :,
            self.remove_border_count : reduced_tile_h + self.remove_border_count,
            self.remove_border_count : reduced_tile_w + self.remove_border_count,
        ]

        img = torch.zeros(image_size, device=device)
        lookup = torch.zeros(image_size, device=device)
        ones = torch.ones(reduced_tile_h, reduced_tile_w, device=device)

        for patch, (loc_i, loc_j) in zip(
            tiles,
            product(
                range(
                    self.remove_border_count,
                    int(self.resized_h * scale_h) - reduced_tile_h + 1,
                    int(self.stride_h * scale_h),
                ),
                range(
                    self.remove_border_count,
                    int(self.resized_w * scale_w) - reduced_tile_w + 1,
                    int(self.stride_w * scale_w),
                ),
            ),
            strict=True,
        ):
            img[
                :,
                :,
                loc_i : (loc_i + reduced_tile_h),
                loc_j : (loc_j + reduced_tile_w),
            ] += patch
            lookup[
                :,
                :,
                loc_i : (loc_i + reduced_tile_h),
                loc_j : (loc_j + reduced_tile_w),
            ] += ones

        img = torch.divide(img, lookup)
        img[img != img] = 0  # noqa: PLR0124

        return img

    def tile(self, image: torch.Tensor, use_random_tiling: bool = False) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)

        self.batch_size, self.num_channels, self.input_h, self.input_w = image.shape

        if self.input_h < self.tile_size_h or self.input_w < self.tile_size_w:
            msg = f"Tile size {self.tile_size_h, self.tile_size_w} exceeds image size {self.input_h, self.input_w}"
            raise ValueError(msg)

        self.resized_h, self.resized_w = compute_new_image_size(
            image_size=(self.input_h, self.input_w),
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )

        image = upscale_image(image, size=(self.resized_h, self.resized_w), mode=self.mode)

        return self.__random_tile(image) if use_random_tiling else self.__unfold(image)

    def untile(self, tiles: torch.Tensor) -> torch.Tensor:
        image = self.__fold(tiles)
        return downscale_image(image=image, size=(self.input_h, self.input_w), mode=self.mode)