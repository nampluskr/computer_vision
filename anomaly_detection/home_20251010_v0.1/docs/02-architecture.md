# Architecture - 전체 프레임워크 구조

## 목차

1. [개요](#1-개요)
2. [전체 워크스페이스 구조](#2-전체-워크스페이스-구조)
3. [Project 디렉토리](#3-project-디렉토리)
4. [Datasets 디렉토리](#4-datasets-디렉토리)
5. [Backbones 디렉토리](#5-backbones-디렉토리)
6. [Outputs 디렉토리](#6-outputs-디렉토리)
7. [주요 파일 상세 설명](#7-주요-파일-상세-설명)
8. [Components 아키텍처](#8-components-아키텍처)
9. [모델 구현 패턴](#9-모델-구현-패턴)

---

## 1. 개요

본 프레임워크는 4개의 주요 디렉토리로 구성되어 있으며, 각 디렉토리는 명확한 역할을 가지고 있습니다.

```
workspace/
├── project/              # 코드 및 실행 파일
├── datasets/             # 데이터셋 저장소
├── backbones/            # 사전 학습된 가중치
└── outputs/              # 학습 결과
```

---

## 2. 전체 워크스페이스 구조

```
workspace/
├── project/             # 코드 및 실행 파일
│   ├── main.py
│   ├── train.py
│   ├── registry.py
│   ├── dataloader.py
│   └── models/
│       ├── model_*.py
│       ├── components/
│       └── components_dinomaly/
│
├── datasets/            # 데이터셋 저장소
│   ├── mvtec/           # MVTec AD (15 categories)
│   ├── visa/            # VisA (12 categories)
│   ├── btad/            # BTAD (3 categories)
│   ├── custom/          # Custom datasets
│   ├── dtd/             # Textures (DRAEM)
│   └── imagenette2/     # ImageNet subset (EfficientAD)
│
├── backbones/           # 사전 학습된 가중치
│   ├── *.pth            # CNN backbones
│   ├── *_vit_*/         # Transformer backbones
│   ├── efficientad/     # EfficientAD specific
│   └── dsr/             # DSR specific
│
└── outputs/             # 학습 결과
    ├── mvtec/
    ├── visa/
    ├── btad/
    └── custom/
```

---

## 3. Project 디렉토리

### 3.1. 디렉토리 구조

```
project/
├── main.py                             # 실행 스크립트
├── train.py                            # 학습 관련 함수들
├── registry.py                         # 모델 레지스트리
├── dataloader.py                       # 데이터로더
├── README.md                           # 프로젝트 문서
│
└── models/                             # 모델 디렉토리
    ├── __init__.py
    │
    # ===== Memory-based Models (3) =====
    ├── model_padim.py
    ├── model_patchcore.py
    ├── model_dfkde.py
    │
    # ===== Normalizing Flow Models (4) =====
    ├── model_cflow.py
    ├── model_fastflow.py
    ├── model_csflow.py
    ├── model_uflow.py
    │
    # ===== Knowledge Distillation Models (4) =====
    ├── model_stfpm.py
    ├── model_fre.py
    ├── model_reverse_distillation.py
    ├── model_efficientad.py
    │
    # ===== Reconstruction Models (4) =====
    ├── model_autoencoder.py
    ├── model_ganomaly.py
    ├── model_draem.py
    ├── model_dsr.py
    │
    # ===== Feature Adaptation Models (2) =====
    ├── model_dfm.py
    ├── model_cfa.py
    │
    # ===== Foundation Models (3) =====
    ├── model_dinomaly.py
    ├── model_supersimplenet.py
    ├── model_uninet.py
    │
    # ===== Common Components =====
    ├── components/
    │   ├── __init__.py
    │   ├── trainer.py
    │   ├── backbone.py
    │   ├── feature_extractor.py
    │   ├── tiler.py
    │   ├── blur.py
    │   ├── pca.py
    │   ├── k_center_greedy.py
    │   ├── multi_variate_gaussian.py
    │   ├── dynamic_buffer.py
    │   ├── all_in_one_block.py
    │   ├── perlin.py
    │   ├── sspcab.py
    │   ├── resnet_decoder.py
    │   └── multi_random_choice.py
    │
    └── components_dinomaly/
        ├── __init__.py
        ├── dinov2_loader.py
        ├── download.py
        ├── layers.py
        ├── loss.py
        ├── optimizer.py
        └── vision_transformer.py
```

### 3.2. 핵심 파일 설명

#### main.py
```python
"""
Experiment Execution Script

Features:
- Global configuration setup
- Single model training
- Multi-model training

Usage:
    python main.py
"""
```

실험 실행을 위한 메인 스크립트입니다.

#### train.py
```python
"""
Training Utility Functions

Global Variables:
- DATASET_DIR: Dataset root directory
- BACKBONE_DIR: Pretrained weights directory
- OUTPUT_DIR: Training results directory
- SEED: Random seed
- NUM_WORKERS: DataLoader workers
- PIN_MEMORY: Pin memory for DataLoader
- PERSISTENT_WORKERS: Keep workers alive

Functions:
- set_globals(): Set global configuration
- get_globals(): Get current configuration
- print_globals(): Print current configuration
- set_seed(): Set random seed for reproducibility
- count_parameters(): Count model parameters
- train(): Train single model
- train_models(): Train multiple models on multiple categories
- clear_dataloader(): Clean up dataloader resources
- clear_memory(): GPU memory cleanup
- print_memory(): Print GPU memory status
"""
```

#### registry.py
```python
"""
Model Registry for Anomaly Detection Framework

Total Registered Models: 44 configurations (34 unique models)

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
- register_all_models(): Register all available models
"""
```

#### dataloader.py
```python
"""
Unified DataLoader for Multiple Dataset Types

Supported Datasets:
1. MVTecDataset: MVTec AD benchmark (15 categories)
2. VisADataset: VisA benchmark (12 categories)
3. BTADDataset: BTAD benchmark (3 categories)
4. CustomDataset: User-defined datasets

Key Features:
- Automatic CSV generation for custom datasets
- Support for multiple dataset types
- Support for multiple categories
- Flexible image transforms
- Mask handling for anomaly localization

Functions:
- set_dataset_dir(): Set global dataset directory
- get_data_info(): Parse custom dataset filename
- create_csv(): Auto-generate metadata CSV
- get_dataloaders(): Main function to create train/test loaders

Classes:
- BaseDataset: Base class for all datasets
- MVTecDataset: MVTec AD dataset loader
- VisADataset: VisA dataset loader
- BTADDataset: BTAD dataset loader
- CustomDataset: Custom dataset loader
"""
```

---

## 4. Datasets 디렉토리

### 4.1. Standard Benchmark Datasets

#### MVTec AD Dataset (15 categories)
```
mvtec/
├── bottle/
│   ├── train/
│   │   └── good/            # Normal training images (209 images)
│   ├── test/
│   │   ├── good/            # Normal test images (20 images)
│   │   ├── broken_large/    # Anomaly type 1
│   │   ├── broken_small/    # Anomaly type 2
│   │   └── contamination/   # Anomaly type 3
│   └── ground_truth/
│       ├── broken_large/    # Anomaly masks
│       ├── broken_small/
│       └── contamination/
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
```

#### VisA Dataset (12 categories)
```
visa/
├── candle/
│   ├── Data/
│   │   └── Images/
│   │       ├── Anomaly/
│   │       │   ├── 000/
│   │       │   ├── 001/
│   │       │   └── ...
│   │       └── Normal/
│   ├── image_anno.csv       # Image annotations
│   └── split_csv/
│       ├── 1cls.csv
│       └── 2cls.csv
├── capsules/
├── cashew/
├── chewinggum/
├── fryum/
├── macaroni1/
├── macaroni2/
├── pcb1/
├── pcb2/
├── pcb3/
├── pcb4/
└── pipe_fryum/
```

#### BTAD Dataset (3 categories)
```
btad/
├── 01/
│   ├── train/
│   │   └── ok/              # Normal training images
│   ├── test/
│   │   ├── ok/              # Normal test images
│   │   └── ko/              # Anomaly images
│   └── ground_truth/
│       └── ko/              # Anomaly masks
├── 02/
└── 03/
```

### 4.2. Custom Dataset

```
custom/
└── your_dataset_name/
    └── data_rgb/
        ├── normal/          # Normal images (label=0)
        │   ├── pattern1 60 100.png
        │   ├── pattern1 60 200.png
        │   └── ...
        ├── defect_type1/    # Anomaly type 1 (label=1)
        │   └── ...
        ├── defect_type2/    # Anomaly type 2 (label=1)
        │   └── ...
        └── data_info.csv    # Auto-generated metadata
```

**파일명 규칙:**
```
Format: {category} {freq} {dimming}[_{extra}].png

Valid Examples:
✓ pattern1 60 100.png
✓ pattern2 120 200.png
✓ design_A 80 150_v2.png

Invalid Examples:
✗ image001.png                  # Missing metadata
✗ pattern1_60_100.png           # Wrong separator
✗ pattern1 60.png               # Missing dimming value
```

### 4.3. Auxiliary Datasets

#### DTD (Describable Textures Dataset)
```
dtd/
└── images/
    ├── banded/
    ├── blotchy/
    ├── braided/
    ├── bubbly/
    └── ... (47 texture categories)
```

DRAEM 모델의 이상 시뮬레이션에 사용됩니다.

#### Imagenette2
```
imagenette2/
└── train/
    ├── n01440764/           # tench
    ├── n02102040/           # English springer
    ├── n02979186/           # cassette player
    ├── n03000684/           # chain saw
    ├── n03028079/           # church
    ├── n03394916/           # French horn
    ├── n03417042/           # garbage truck
    ├── n03425413/           # gas pump
    ├── n03445777/           # golf ball
    └── n03888257/           # parachute
```

EfficientAD 모델의 teacher 네트워크 사전 학습에 사용됩니다.

---

## 5. Backbones 디렉토리

### 5.1. CNN Backbones

```
backbones/
├── resnet18.pth                 # ResNet-18
├── resnet50.pth                 # ResNet-50
├── wide_resnet50_2.pth          # Wide ResNet-50-2
└── efficientnet_b5.pth          # EfficientNet-B5
```

### 5.2. Transformer Backbones

```
backbones/
├── cait_m48_448.fb_dist_in1k/
│   └── model.safetensors
├── deit_base_distilled_patch16_384.fb_in1k/
│   └── model.safetensors
└── wide_resnet50_2.tv_in1k/
    └── model.safetensors
```

### 5.3. DINOv2 Backbones

```
backbones/
├── dinov2_vit_small_14/
│   └── dinov2_vits14_pretrain.pth
├── dinov2_vit_base_14/
│   └── dinov2_vitb14_pretrain.pth
└── dinov2_vit_large_14/
    └── dinov2_vitl14_pretrain.pth
```

### 5.4. Model-specific Weights

```
backbones/
├── efficientad/
│   ├── pretrained_teacher_small.pth
│   └── pretrained_teacher_medium.pth
└── dsr/
    └── vq_model_pretrained_128_4096.pckl
```

---

## 6. Outputs 디렉토리

### 6.1. 출력 구조

```
outputs/
├── mvtec/
│   ├── bottle/
│   │   ├── padim/
│   │   │   ├── model_mvtec_bottle_padim_epochs-1.pth
│   │   │   ├── results_mvtec_bottle_padim_thresholds.txt
│   │   │   ├── histogram_mvtec_bottle_padim_scores.png
│   │   │   ├── image_mvtec_bottle_padim_normal_0001.png
│   │   │   └── image_mvtec_bottle_padim_anomaly_0001.png
│   │   ├── stfpm/
│   │   ├── efficientad-small/
│   │   └── dinomaly-base-224/
│   ├── wood/
│   └── grid/
│
├── visa/
│   ├── candle/
│   └── capsules/
│
├── btad/
│   ├── 01/
│   ├── 02/
│   └── 03/
│
└── custom/
    ├── dataset_A/
    │   ├── pattern1/
    │   └── pattern2/
    └── dataset_B/
```

### 6.2. 출력 파일 설명

- **model_*.pth**: 학습된 모델 가중치 및 optimizer 상태
- **results_*_thresholds.txt**: 상세 평가 지표 및 임계값 분석
- **histogram_*_scores.png**: 정상/이상 점수 분포 히스토그램
- **image_*_normal_*.png**: 정상 샘플의 이상 맵 시각화
- **image_*_anomaly_*.png**: 이상 샘플의 이상 맵 시각화

---

## 7. 주요 파일 상세 설명

### 7.1. models/components/trainer.py

```python
"""
Base Trainer and Early Stopping

Classes:
- EarlyStopper: Early stopping implementation
- BaseTrainer: Base trainer class for all models

Key Methods:
- fit(): Main training loop
- train_epoch(): Single epoch training
- validation_epoch(): Validation with metrics
- save_model(): Save model weights
- load_model(): Load model weights
- save_results(): Save evaluation results
- save_histogram(): Save score distribution
- save_maps(): Save anomaly map visualizations

Hooks (Override in subclasses):
- on_fit_start(): Called before training starts
- on_fit_end(): Called after training ends
- on_epoch_start(): Called before each epoch
- on_epoch_end(): Called after each epoch
- on_train_start(): Called before training epoch
- on_train_end(): Called after training epoch
- on_validation_start(): Called before validation
- on_validation_end(): Called after validation
- train_step(): Single training step (MUST override)
- validation_step(): Single validation step (MUST override)
"""
```

### 7.2. models/components/backbone.py

```python
"""
Backbone Weight Path Management

Global Variable:
- BACKBONE_DIR: Directory containing pretrained backbone weights

Dictionary:
- BACKBONE_WEIGHT_FILES: Mapping of backbone names to weight filenames

Functions:
- set_backbone_dir(): Update global BACKBONE_DIR
- get_backbone_dir(): Get current BACKBONE_DIR
- get_backbone_path(): Get full path to backbone weight file
"""
```

### 7.3. models/components/feature_extractor.py

```python
"""
Feature Extraction Components

Classes:
- TimmFeatureExtractor: Feature extractor using timm library
  * Supports ResNet, Wide ResNet, EfficientNet
  * Multi-layer feature extraction
  * Frozen weights for feature extraction

Functions:
- dryrun_find_featuremap_dims(): Calculate feature map dimensions
"""
```

---

## 8. Components 아키텍처

### 8.1. Common Components

| Component | 설명 | 사용 모델 |
|-----------|------|-----------|
| trainer.py | BaseTrainer, EarlyStopper | 모든 모델 |
| backbone.py | Backbone 경로 관리 | 모든 모델 |
| feature_extractor.py | TimmFeatureExtractor | PaDiM, PatchCore, STFPM 등 |
| tiler.py | Image Tiling/Untiling | PatchCore |
| blur.py | GaussianBlur2d | PaDiM, PatchCore, CFA 등 |
| pca.py | PCA Implementation | PaDiM |
| k_center_greedy.py | Coreset Sampling | PatchCore |
| multi_variate_gaussian.py | Multivariate Gaussian | PaDiM |
| dynamic_buffer.py | DynamicBufferMixin | PaDiM, PatchCore, DFM |
| all_in_one_block.py | FrEIA Block | CFlow, FastFlow, CSFlow |
| perlin.py | Perlin Noise Generator | DRAEM, DSR |
| sspcab.py | SSPCAB Module | DRAEM |
| resnet_decoder.py | ResNet Decoder | Reverse Distillation, UniNet |

### 8.2. Dinomaly Components

| Component | 설명 |
|-----------|------|
| dinov2_loader.py | DINOv2 Model Loader |
| download.py | Model Download Utilities |
| layers.py | DinomalyMLP, LinearAttention |
| loss.py | CosineHardMiningLoss |
| optimizer.py | StableAdamW, WarmCosineScheduler |
| vision_transformer.py | Vision Transformer Components |

---

## 9. 모델 구현 패턴

### 9.1. 모델 파일 구조

각 `model_*.py` 파일은 다음 구조를 따릅니다:

```python
# 1. Imports
from .components.trainer import BaseTrainer

# 2. Model Implementation (from Anomalib)
class XXXModel(nn.Module):
    def __init__(...):
        pass
    
    def forward(self, batch):
        # Returns dict with 'pred_score' and 'anomaly_map'
        pass

# 3. Trainer Implementation
class XXXTrainer(BaseTrainer):
    def __init__(self, model=None, ...):
        if model is None:
            model = XXXModel(...)
        super().__init__(model=model, ...)
    
    def train_step(self, batch):
        # Training logic
        pass
    
    def validation_step(self, batch):
        # Validation logic
        pass
```

### 9.2. Registry 등록 패턴

```python
ModelRegistry.register(
    "model_name",
    "models.model_xxx.XXXTrainer",
    dict(
        # Model configuration
        backbone="resnet50",
        layers=["layer1", "layer2"]
    ),
    dict(
        # Training configuration
        num_epochs=50,
        batch_size=16,
        normalize=True,
        img_size=256
    )
)
```

### 9.3. 데이터 흐름

```
DataLoader → Trainer.train_step() → Model.forward() → Loss Calculation
                                                     ↓
                                              Backward & Update
                                                     ↓
Trainer.validation_step() → Model.forward() → Metrics Calculation
                                             ↓
                                    Save Results & Visualizations
```

---

**다음 문서:** [Getting Started](01-getting-started.md)

