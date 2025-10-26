""" filename: transforms.py """

import torch
from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PRESETS = {
    'mvtec': {'img_size': 224, 'normalize': True},
    'visa': {'img_size': 224, 'normalize': True},
    'btad': {'img_size': 224, 'normalize': True},
    'custom': {'img_size': 256, 'normalize': True},
}


def get_transform(split='train', img_size=224, normalize=True,
                  for_mask=False, augmentation=True):
    transform_list = []

    # Augmentation (train only, not for mask)
    if split == 'train' and not for_mask and augmentation:
        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])

    # Resize / ToTensor
    transform_list.append(T.Resize((img_size, img_size)))
    transform_list.append(T.ToTensor())

    # Normalize (not for mask)
    if normalize and not for_mask:
        transform_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transform_list)


def get_train_transform(img_size=224, normalize=True, augmentation=True):
    return get_transform(
        split='train',
        img_size=img_size,
        normalize=normalize,
        for_mask=False,
        augmentation=augmentation
    )


def get_test_transform(img_size=224, normalize=True):
    return get_transform(
        split='test',
        img_size=img_size,
        normalize=normalize,
        for_mask=False,
        augmentation=False
    )


def get_mask_transform(img_size=224):
    return get_transform(
        split='test',
        img_size=img_size,
        normalize=False,
        for_mask=True,
        augmentation=False
    )


def get_anomaly_transform(split='train', img_size=224, strong_augmentation=False):
    transform_list = []

    if split == 'train':
        if strong_augmentation:
            # Strong augmentation for limited training data
            transform_list.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            ])
        else:
            # Standard augmentation
            transform_list.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])

    # Resize & ToTensor & Normalize
    transform_list.extend([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return T.Compose(transform_list)


def get_custom_transform(img_size=224, resize_mode='resize',
                        interpolation='bilinear', center_crop=False,
                        normalize=True, **augmentation_kwargs):
    transform_list = []

    # Interpolation mode
    interp_mode = {
        'bilinear': T.InterpolationMode.BILINEAR,
        'bicubic': T.InterpolationMode.BICUBIC,
        'nearest': T.InterpolationMode.NEAREST
    }.get(interpolation, T.InterpolationMode.BILINEAR)

    # Augmentation
    if augmentation_kwargs.get('horizontal_flip', False):
        transform_list.append(T.RandomHorizontalFlip(p=0.5))

    if augmentation_kwargs.get('vertical_flip', False):
        transform_list.append(T.RandomVerticalFlip(p=0.5))

    if 'rotation_degrees' in augmentation_kwargs:
        degrees = augmentation_kwargs['rotation_degrees']
        transform_list.append(T.RandomRotation(degrees=degrees))

    if 'color_jitter' in augmentation_kwargs:
        jitter_params = augmentation_kwargs['color_jitter']
        transform_list.append(T.ColorJitter(**jitter_params))

    if 'gaussian_blur' in augmentation_kwargs:
        blur_params = augmentation_kwargs['gaussian_blur']
        transform_list.append(T.GaussianBlur(**blur_params))

    # Resize
    if resize_mode == 'resize':
        transform_list.append(T.Resize((img_size, img_size), interpolation=interp_mode))
    elif resize_mode == 'resize_shorter':
        transform_list.append(T.Resize(img_size, interpolation=interp_mode))

    # Center crop
    if center_crop:
        transform_list.append(T.CenterCrop(img_size))

    # ToTensor
    transform_list.append(T.ToTensor())

    # Normalize
    if normalize:
        transform_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transform_list)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)

    if tensor.dim() == 4:  # [B, C, H, W]
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def get_preset_transform(dataset_name, split='train'):
    if dataset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {dataset_name}. "
            f"Available: {list(PRESETS.keys())}"
        )
    preset = PRESETS[dataset_name]

    return get_transform(
        split=split,
        img_size=preset['img_size'],
        normalize=preset['normalize']
    )