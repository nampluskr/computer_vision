"""
Model Registry for Anomaly Detection Framework

This module provides a centralized registry for all anomaly detection models.
Models can be registered with their configurations and retrieved by name.

Categories:
1. Memory-based (3): padim, patchcore, dfkde
2. Normalizing Flow (9): cflow variants, fastflow variants, csflow, uflow variants
3. Knowledge Distillation (6): stfpm, fre, reverse-distillation, efficientad variants
4. Reconstruction (4): autoencoder, ganomaly, draem, dsr
5. Feature Adaptation (2): dfm, cfa
6. Foundation Models (12): dinomaly variants (9), supersimplenet (2), uninet (1)

Classes:
- ModelRegistry: Central registry for all models

Functions:
- register(): Register new model configuration
- get(): Retrieve model configuration
- is_registered(): Check if model is registered
- list_models(): List all registered models
- list_by_category(): List models by category
- get_train_config(): Get training configuration
- get_model_config(): Get model configuration
- get_trainer(): Create trainer instance
- register_all_models(): Register all available models (auto-called on import)

Example:
    >>> from registry import ModelRegistry
    >>> config = ModelRegistry.get("padim")
    >>> models = ModelRegistry.list_models()
"""

import os
from models.components.trainer import EarlyStopper
from dataloader import get_dataset_dir


class ModelRegistry:
    """Register a new model configuration.

    Args:
        model_type: Unique identifier for the model (e.g., "padim", "patchcore")
        trainer_path: Python path to the trainer class (e.g., "models.model_padim.PadimTrainer")
        model_config: Model-specific configuration (backbone, layers, etc.)
        train_config: Training configuration (epochs, batch_size, etc.)

    Example:
        >>> ModelRegistry.register(
        ...     "padim",
        ...     "models.model_padim.PadimTrainer",
        ...     {"backbone": "resnet50", "layers": ["layer1", "layer2"]},
        ...     {"num_epochs": 1, "batch_size": 8}
        ... )
    """
    _registry = {}

    @classmethod
    def register(cls, model_type, trainer_path, model_config, train_config):
        cls._registry[model_type] = {
            "trainer_path": trainer_path,
            "model_config": model_config,
            "train_config": train_config
        }

    @classmethod
    def get(cls, model_type):
        if model_type not in cls._registry:
            available = ', '.join(cls.list_models())
            raise ValueError(
                f"Unknown model_type: '{model_type}'.\n"
                f"Available models: {available}"
            )
        return cls._registry[model_type]

    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        return model_type in cls._registry

    @classmethod
    def list_models(cls):
        return sorted(cls._registry.keys())

    @classmethod
    def list_by_category(cls):
        categories = {
            "Memory-based": [],
            "Normalizing Flow": [],
            "Knowledge Distillation": [],
            "Reconstruction": [],
            "Feature Adaptation": [],
        }
        for model_type in cls._registry.keys():
            if model_type in ["padim", "patchcore"]:
                categories["Memory-based"].append(model_type)
            elif model_type.startswith(("cflow", "fastflow", "csflow", "uflow")):
                categories["Normalizing Flow"].append(model_type)
            elif model_type.startswith(("stfpm", "fre", "efficientad", "reverse-distillation")):
                categories["Knowledge Distillation"].append(model_type)
            elif model_type in ["autoencoder", "draem"]:
                categories["Reconstruction"].append(model_type)
            elif model_type in ["dfm", "cfa"]:
                categories["Feature Adaptation"].append(model_type)
        return categories


def get_train_config(model_type):
    config = ModelRegistry.get(model_type)
    return config["train_config"]


def get_model_config(model_type):
    config = ModelRegistry.get(model_type)
    return config["model_config"]


def get_trainer(model_type, img_size):
    """Create trainer instance for specified model type
    
    Args:
        model_type: Model identifier (e.g., "padim", "stfpm")
        backbone_dir: Path to backbone weights directory
        dataset_dir: Path to dataset directory
        img_size: Input image size
        
    Returns:
        Trainer instance (subclass of BaseTrainer)
        
    Note:
        - Dynamically imports trainer class from models.model_xxx
        - Auto-updates model_config with img_size and dtd_dir
        - Returns initialized trainer ready for training
    """
    config = ModelRegistry.get(model_type)
    module_path, class_name = config["trainer_path"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    TrainerClass = getattr(module, class_name)

    model_config = config["model_config"]
    if 'input_size' in model_config:
        model_config['input_size'] = (img_size, img_size)
    if 'img_size' in model_config:
        model_config['img_size'] = img_size

    return TrainerClass(**model_config)


def register_all_models():
    """Register all available anomaly detection models.

    This function registers all implemented models with their default configurations.
    It should be called once at module import time.
    """

    #############################################################
    # 1. Memory-based: PaDim(2020), PatchCore(2022), DFKDE(2022)
    #############################################################

    ModelRegistry.register("padim", "models.model_padim.PadimTrainer",
        dict(backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"]),
        dict(num_epochs=1, batch_size=4, normalize=True, img_size=256)
    )
    ModelRegistry.register("patchcore", "models.model_patchcore.PatchcoreTrainer",
        dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"]),
        dict(num_epochs=1, batch_size=8, normalize=True, img_size=256)
    )
    ModelRegistry.register("dfkde", "models.model_dfkde.DFKDETrainer",
        dict(backbone="resnet50", layers=["layer4"], pre_trained=True),
        dict(num_epochs=1, batch_size=8, normalize=True, img_size=256)
    )

    #############################################################
    # 2. Nomalizing Flow-based: CFlow(2021), FastFlow(2021), CSFlow(2021), UFlow(2022)
    #############################################################

    ModelRegistry.register("cflow", "models.model_cflow.CflowTrainer",
        dict(backbone="resnet18", layers=["layer2", "layer3"]),
        dict(num_epochs=10, batch_size=4, normalize=True, img_size=256)
    )
    ModelRegistry.register("cflow-resnet18", "models.model_cflow.CflowTrainer",
        dict(backbone="resnet18", layers=["layer2", "layer3"]),
        dict(num_epochs=10, batch_size=4, normalize=True, img_size=256)
    )
    ModelRegistry.register("cflow-resnet50", "models.model_cflow.CflowTrainer",
        dict(backbone="resnet50", layers=["layer2", "layer3"]),
        dict(num_epochs=10, batch_size=4, normalize=True, img_size=256)
    )
    ModelRegistry.register("fastflow", "models.model_fastflow.FastflowTrainer",
        dict(backbone="wide_resnet50_2"),
        dict(num_epochs=20, batch_size=8, normalize=True, img_size=256)
    )
    ModelRegistry.register("fastflow-cait", "models.model_fastflow.FastflowTrainer",
        dict(backbone="cait_m48_448"),
        dict(num_epochs=5, batch_size=4, normalize=True, img_size=448)
    )
    ModelRegistry.register("fastflow-deit", "models.model_fastflow.FastflowTrainer",
        dict(backbone="deit_base_distilled_patch16_384"),
        dict(num_epochs=10, batch_size=8, normalize=True, img_size=384)
    )
    ModelRegistry.register("csflow", "models.model_csflow.CsFlowTrainer",
        dict(num_channels=3),
        dict(num_epochs=10, batch_size=8, normalize=True, img_size=256)
    )
    ModelRegistry.register("uflow", "models.model_uflow.UflowTrainer",
        dict(backbone="wide_resnet50_2"),
        dict(num_epochs=10, batch_size=8, normalize=True, img_size=256)
    )
    ModelRegistry.register("uflow-mcait", "models.model_uflow.UflowTrainer",
        dict(backbone="mcait"),
        dict(num_epochs=10, batch_size=4, normalize=True, img_size=448)
    )

    #############################################################
    # 3. Knowledge Distillation: STFPM(2021), FRE(2023), Reverse Distillation(2022), EfficientAD(2024)
    #############################################################

    ModelRegistry.register("stfpm", "models.model_stfpm.STFPMTrainer",
        dict(backbone="resnet50", layers=["layer1", "layer2", "layer3"]),
        dict(num_epochs=50, batch_size=16, normalize=True, img_size=256)
    )
    ModelRegistry.register("fre", "models.model_fre.FRETrainer",
        dict(backbone="resnet50", layer="layer3"),
        dict(num_epochs=50, batch_size=16, normalize=True, img_size=256)
    )
    ModelRegistry.register("reverse-distillation", "models.model_reverse_distillation.ReverseDistillationTrainer",
        dict(backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"]),
        dict(num_epochs=50, batch_size=8, normalize=True, img_size=256)
    )

    ModelRegistry.register("efficientad-small", "models.model_efficientad.EfficientAdTrainer",
        dict(model_size="small", imagenet_dir=os.path.join(get_dataset_dir(), "imagenette2")),
        dict(num_epochs=20, batch_size=1, normalize=False, img_size=256)
    )
    ModelRegistry.register("efficientad-medium", "models.model_efficientad.EfficientAdTrainer",
        dict(model_size="medium", imagenet_dir=os.path.join(get_dataset_dir(), "imagenette2")),
        dict(num_epochs=20, batch_size=1, normalize=False, img_size=256)
    )
    ModelRegistry.register("efficientad", "models.model_efficientad.EfficientAdTrainer",
        dict(model_size="medium", early_stopper_auroc=EarlyStopper(target_value=0.998)),
        dict(num_epochs=50, batch_size=1, normalize=False, img_size=256)
    )

    #############################################################
    # 4. Reconstruction-based: Autoencoder(Baseline), GANomaly(2018), DRAEM(2021), DSR(2022)
    #############################################################

    ModelRegistry.register("autoencoder", "models.model_autoencoder.AutoencoderTrainer",
        dict(latent_dim=128),
        dict(num_epochs=50, batch_size=16, normalize=False, img_size=256)
    )
    ModelRegistry.register("ganomaly", "models.model_ganomaly.GanomalyTrainer",
        dict(input_size=(256, 256), n_features=64, latent_vec_size=256, gamma=0.01),
        dict(num_epochs=20, batch_size=8, normalize=False, img_size=256)
    )
    ModelRegistry.register("draem", "models.model_draem.DraemTrainer",
        dict(sspcab=True, dtd_dir=os.path.join(get_dataset_dir(), "dtd")),
        dict(num_epochs=10, batch_size=8, normalize=False, img_size=256)
    )
    ModelRegistry.register("dsr", "models.model_dsr.DsrTrainer",
        dict(latent_anomaly_strength=0.2, embedding_dim=128, num_embeddings=4096),
        dict(num_epochs=50, batch_size=8, normalize=False, img_size=256)
    )

    #############################################################
    # 5. Feature Adaptation: DFM(2019), CFA(2022)
    #############################################################

    ModelRegistry.register("dfm", "models.model_dfm.DFMTrainer",
        dict(backbone="resnet50", layer="layer3", score_type="fre"),
        dict(num_epochs=1, batch_size=16, normalize=True, img_size=256)
    )
    ModelRegistry.register("cfa", "models.model_cfa.CfaTrainer",
        dict(backbone="wide_resnet50_2"),
        dict(num_epochs=20, batch_size=16, normalize=True, img_size=256)
    )

    #############################################################
    # 6. Foundation Models: Dinomaly (2025)
    #############################################################

    # dinomaly-small/base/large (img_size = 224)
    ModelRegistry.register("dinomaly-small-224", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_small_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=15, batch_size=32, normalize=True, img_size=224)
    )
    ModelRegistry.register("dinomaly-base-224", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_base_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=15, batch_size=16, normalize=True, img_size=224)
    )
    ModelRegistry.register("dinomaly-large-224", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_large_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=15, batch_size=8, normalize=True, img_size=224)
    )
    # dinomaly-small/base/large (img_size = 392)
    ModelRegistry.register("dinomaly-small-392", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_small_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=10, batch_size=24, normalize=True, img_size=392)
    )
    ModelRegistry.register("dinomaly-base-392", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_base_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=10, batch_size=12, normalize=True, img_size=392)
    )
    ModelRegistry.register("dinomaly-large-392", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_large_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=10, batch_size=6, normalize=True, img_size=392)
    )
    # dinomaly-small/base/large (img_size = 448)
    ModelRegistry.register("dinomaly-small-448", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_small_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=10, batch_size=16, normalize=True, img_size=448)
    )
    ModelRegistry.register("dinomaly-base-448", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_base_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=10, batch_size=8, normalize=True, img_size=448)
    )
    ModelRegistry.register("dinomaly-large-448", "models.model_dinomaly.DinomalyTrainer",
        dict(encoder_name="dinov2_vit_large_14", bottleneck_dropout=0.2, decoder_depth=8),
        dict(num_epochs=10, batch_size=4, normalize=True, img_size=448)
    )
    # supersimplenet (supervised/unsupersived)
    ModelRegistry.register("supersimplenet", "models.model_supersimplenet.SupersimplenetTrainer",
        dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], supervised=False),
        dict(num_epochs=50, batch_size=16, normalize=True, img_size=256)
    )
    ModelRegistry.register("supersimplenet-supervised", "models.model_supersimplenet.SupersimplenetTrainer",
        dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], supervised=True),
        dict(num_epochs=50, batch_size=16, normalize=True, img_size=256)
    )
    # UniNet (temperature=0.5)
    ModelRegistry.register("uninet", "models.model_uninet.UniNetTrainer",
        dict(student_backbone="wide_resnet50_2", teacher_backbone="wide_resnet50_2", temperature=0.4),
        dict(num_epochs=20, batch_size=4, normalize=True, img_size=256)
    )

# Auto-register all models when module is imported
register_all_models()