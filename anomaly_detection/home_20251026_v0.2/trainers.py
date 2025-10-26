""" filename: trainers.py """

import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_logger(output_dir, run_name):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_formatter = logging.Formatter('%(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, f"{run_name}_training.log"))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, model_name, model_class, trainer_class,
                 loss_fn=None, model_config=None, trainer_config=None):
        cls._registry[model_name] = {
            'model_class': model_class,
            'trainer_class': trainer_class,
            'model_config': model_config or {},
            'trainer_config': trainer_config or {}
        }

    @classmethod
    def get(cls, model_name):
        if model_name not in cls._registry:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {cls.list_models()}"
            )
        return cls._registry[model_name]

    @classmethod
    def _get_class(cls, class_path):
        if isinstance(class_path, str):
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        return class_path

    @classmethod
    def get_model_class(cls, model_name):
        config = cls.get(model_name)
        return cls._get_class(config['model_class'])

    @classmethod
    def get_trainer_class(cls, model_name):
        config = cls.get(model_name)
        return cls._get_class(config['trainer_class'])

    @classmethod
    def list_models(cls):
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_name):
        return model_name in cls._registry


# ============================================================================
# Factory Functions
# ============================================================================

def get_trainer(model_name, logger=None, **kwargs):
    """Create trainer with model using registry"""
    # Get classes (not instances!)
    ModelClass = ModelRegistry.get_model_class(model_name)
    TrainerClass = ModelRegistry.get_trainer_class(model_name)

    # Get configs
    config = ModelRegistry.get(model_name)
    model_config = config['model_config']
    trainer_config = config['trainer_config']

    merged_model_config = {**model_config, **kwargs}
    model = ModelClass(**merged_model_config)
    trainer = TrainerClass(model=model, logger=logger, **trainer_config)

    return trainer


def get_model(model_name, **kwargs):
    """Create model only using registry"""
    ModelClass = ModelRegistry.get_model_class(model_name)
    model_config = ModelRegistry.get(model_name)['model_config']

    return ModelClass(**model_config, **kwargs)


def get_trainer_class(model_name):
    """Get trainer class from registry"""
    return ModelRegistry.get_trainer_class(model_name)


# ============================================================================
# Registration
# ============================================================================

def register_all_models():
    """Register all available models"""

    # Autoencoder
    ModelRegistry.register(
        model_name='autoencoder',
        model_class='models.autoencoder.Autoencoder',
        trainer_class='models.autoencoder.AutoencoderTrainer',
        model_config=dict(latent_dim=256),
        trainer_config=dict(learning_rate=1e-3)
    )

    # STFPM
    ModelRegistry.register(
        model_name='stfpm',
        model_class='models.stfpm.STFPMModel',
        trainer_class='models.stfpm.STFPMTrainer',
        model_config=dict(backbone='resnet50', layers=['layer1', 'layer2', 'layer3']),
        trainer_config=dict(learning_rate=0.4)
    )

    # EfficientAD
    ModelRegistry.register(
        model_name='efficientad',
        model_class='models.EfficientAD',
        trainer_class='models.EfficientadTrainer',
        model_config=dict(model_size='small', teacher_out_channels=384),
        trainer_config=dict(learning_rate=1e-4)
    )


# Auto-register
register_all_models()