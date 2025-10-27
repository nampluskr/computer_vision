""" filename: dataloaders.py """

import os
from torch.utils.data import DataLoader


class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, dataset_name, dataset_class, dataset_config=None, dataloader_config=None):
        cls._registry[dataset_name] = {
            'dataset_class': dataset_class,
            'dataset_config': dataset_config or {},
            'dataloader_config': dataloader_config or {}
        }

    @classmethod
    def get(cls, dataset_name):
        if dataset_name not in cls._registry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {cls.list_datasets()}"
            )
        return cls._registry[dataset_name]

    @classmethod
    def get_dataset_class(cls, dataset_name):
        config = cls.get(dataset_name)
        dataset_class = config['dataset_class']

        if isinstance(dataset_class, str):
            module_path, class_name = dataset_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)

        return dataset_class

    @classmethod
    def get_dataset_config(cls, dataset_name):
        return cls.get(dataset_name)['dataset_config']

    @classmethod
    def get_dataloader_config(cls, dataset_name, split):
        config = cls.get(dataset_name)['dataloader_config']
        return config.get(split, {})

    @classmethod
    def list_datasets(cls):
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, dataset_name):
        return dataset_name in cls._registry


# DataLoader factory functions

def get_dataloader(dataset_name, split, category, transform=None, mask_transform=None,
                   batch_size=None, shuffle=None, num_workers=None, **kwargs):
    # Get dataset class
    dataset_class = DatasetRegistry.get_dataset_class(dataset_name)

    # Get default configs
    dataset_config = DatasetRegistry.get_dataset_config(dataset_name)
    dataloader_config = DatasetRegistry.get_dataloader_config(dataset_name, split)

    # Merge dataset configs
    merged_dataset_config = {**dataset_config, 'category': category, **kwargs}

    # Create dataset
    dataset = dataset_class(
        split=split,
        transform=transform,
        mask_transform=mask_transform,
        **merged_dataset_config
    )

    # Set dataloader config (with defaults)
    if batch_size is None:
        batch_size = dataloader_config.get('batch_size', 32 if split == 'train' else 1)

    if shuffle is None:
        shuffle = dataloader_config.get('shuffle', split == 'train')

    if num_workers is None:
        # num_workers = dataloader_config.get('num_workers', 4 if split == 'train' else 1)
        num_workers = 8

    # Create dataloader
    loader_config = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': False
    }

    return DataLoader(dataset, **loader_config)


def get_train_loader(dataset_name, category, transform=None, **kwargs):
    return get_dataloader(
        dataset_name=dataset_name,
        split='train',
        category=category,
        transform=transform,
        mask_transform=None,
        **kwargs
    )


def get_test_loader(dataset_name, category, transform=None, mask_transform=None, **kwargs):
    return get_dataloader(
        dataset_name=dataset_name,
        split='test',
        category=category,
        transform=transform,
        mask_transform=mask_transform,
        **kwargs
    )


def get_dataloaders(dataset_name, category, train_transform=None, test_transform=None,
                    mask_transform=None, **kwargs):
    # Separate kwargs
    train_kwargs = {k[6:]: v for k, v in kwargs.items() if k.startswith('train_')}
    test_kwargs = {k[5:]: v for k, v in kwargs.items() if k.startswith('test_')}
    common_kwargs = {k: v for k, v in kwargs.items()
                    if not k.startswith('train_') and not k.startswith('test_')}

    # Create train loader
    train_loader = get_dataloader(
        dataset_name=dataset_name,
        split='train',
        category=category,
        transform=train_transform,
        mask_transform=None,
        **common_kwargs,
        **train_kwargs
    )

    # Create test loader
    test_loader = get_dataloader(
        dataset_name=dataset_name,
        split='test',
        category=category,
        transform=test_transform,
        mask_transform=mask_transform,
        **common_kwargs,
        **test_kwargs
    )

    return train_loader, test_loader


# Register all datasets

def register_all_datasets():
    data_dir = os.getenv('DATA_DIR', '/mnt/d/datasets')

    # MVTec
    DatasetRegistry.register(
        dataset_name='mvtec',
        dataset_class='datasets.MVTecDataset',
        dataset_config=dict(
            root_dir=os.path.join(data_dir, 'mvtec')
        ),
        dataloader_config=dict(
            train=dict(batch_size=32, shuffle=True, num_workers=4),
            test=dict(batch_size=1, shuffle=False, num_workers=1)
        )
    )

    # VisA
    DatasetRegistry.register(
        dataset_name='visa',
        dataset_class='datasets.VisADataset',
        dataset_config=dict(
            root_dir=os.path.join(data_dir, 'visa')
        ),
        dataloader_config=dict(
            train=dict(batch_size=32, shuffle=True, num_workers=4),
            test=dict(batch_size=1, shuffle=False, num_workers=1)
        )
    )

    # BTAD
    DatasetRegistry.register(
        dataset_name='btad',
        dataset_class='datasets.BTADDataset',
        dataset_config=dict(
            root_dir=os.path.join(data_dir, 'btad')
        ),
        dataloader_config=dict(
            train=dict(batch_size=32, shuffle=True, num_workers=4),
            test=dict(batch_size=1, shuffle=False, num_workers=1)
        )
    )

    # Custom
    DatasetRegistry.register(
        dataset_name='custom',
        dataset_class='datasets.CustomDataset',
        dataset_config=dict(
            root_dir=os.path.join(data_dir, 'custom'),
            use_frequency=True,
            frequency_bands=[3.0, 5.0]
        ),
        dataloader_config=dict(
            train=dict(batch_size=16, shuffle=True, num_workers=4),
            test=dict(batch_size=1, shuffle=False, num_workers=1)
        )
    )


# Auto-register
register_all_datasets()
