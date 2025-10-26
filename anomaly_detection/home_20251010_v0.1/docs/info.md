# 전체 프레임워크 파일 구조

## Vision Anomaly Detection (Anomalib Image Models)

### 1. Memory-Based / Feature Matching

- [x] **PaDiM (2020)**: A Patch Distribution Modeling Framework for Anomaly Detection and Localization
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/padim
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/padim.html
  - https://arxiv.org/abs/2011.08785

- [x] **PatchCore (2022)**: Towards Total Recall in Industrial Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/patchcore
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/patchcore.html
  - https://arxiv.org/pdf/2106.08265.pdf

- [x] **DFKDE (2022)**: Deep Feature Kernel Density Estimation
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dfkde
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/dfkde.html
  - github.com/openvinotoolkit/anomalib


### 2. Normalizing Flow

- [x] **CFLOW (2021)**: Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/cflow
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/cflow.html
  - https://arxiv.org/pdf/2107.12571v1.pdf

- [x] **FastFlow (2021)**: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/fastflow
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/fastflow.html
  - https://arxiv.org/abs/2111.07677

- [x] **CS-Flow (2021)**: Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/csflow
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/csflow.html
  - https://arxiv.org/pdf/2110.02855.pdf

- [x] **U-Flow (2022)**: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uflow
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/uflow.html
  - https://arxiv.org/abs/2211.12353


### 3. Knowledge Distillation

- [x] **STFPM (2021)**: Student-Teacher Feature Pyramid Matching for anomaly detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/stfpm
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/stfpm.html
  - https://arxiv.org/pdf/2103.04257.pdf

- [x] **Reverse Distillation (2022)**: Anomaly Detection via Reverse Distillation from One-Class Embedding
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/reverse_distillation
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/reverse_distillation.html
  - https://arxiv.org/pdf/2201.10703v2.pdf (2022)

- [x] **FRE (2023)**: A Fast Method For Anomaly Detection And Segmentation
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/fre
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/fre.html
  - https://papers.bmvc2023.org/0614.pdf (2023)

- [x] **EfficientAd (2024)**: Accurate Visual Anomaly Detection at Millisecond-Level Latencies
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/efficient_ad
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/efficient_ad.html
  - https://arxiv.org/pdf/2303.14535.pdf


### 4. Reconstruction-Based

- [x] **GANomaly (2018)**: Semi-Supervised Anomaly Detection via Adversarial Training
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/ganomaly
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/ganomaly.html
  - https://arxiv.org/abs/1805.06725

- [x] **DRAEM (2021)**: A discriminatively trained reconstruction embedding for surface anomaly detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/draem
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/draem.html
  - https://arxiv.org/abs/2108.07610

- [x] **DSR (2022)**: A Dual Subspace Re-Projection Network for Surface Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dsr
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/dsr.html
  - https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31


### 5. Feature Adaptation

- [x] **DFM (2019)**: Deep Feature Modeling (DFM) for anomaly detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dfm
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/dfm.html
  - https://arxiv.org/abs/1909.11786

- [x] **CFA (2022)**: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/cfa
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/cfa.html
  - https://arxiv.org/abs/2206.04325

### 6. Foundation Models - latest

- [ ] **WinCLIP (2023)**: Zero-/Few-Shot Anomaly Classification and Segmentation
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/winclip
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/winclip.html
  - https://arxiv.org/pdf/2303.14814.pdf

- [x] **Dinomaly (2025)**: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dinomaly
  - https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/dinomaly.html
  - https://github.com/guojiajeremy/Dinomaly
  - https://arxiv.org/abs/2405.14325

- [ ] **VLM-AD (2024)**: Vision Language Model (VLM) based Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/vlm_ad
  - https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/vlm_ad.html
  - https://arxiv.org/abs/2412.14446

- [x] **SuperSimpleNet (2024)**: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/supersimplenet
  - https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/supersimplenet.html
  - https://github.com/blaz-r/SuperSimpleNet
  - https://arxiv.org/pdf/2408.03143

- [x] **UniNet (2025)**: Unified Contrastive Learning Framework for Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uninet
  - https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/uninet.html
  - https://github.com/pangdatangtt/UniNet
  - https://pangdatangtt.github.io/#poster (2025)


## 1. Project 디렉토리 (코드 및 실행 파일)

```
project/
├── main.py                             # 실행 스크립트 (실험 설정 및 실행)
├── train.py                            # 학습 관련 함수들 (train, train_models, 유틸리티)
├── registry.py                         # 모델 레지스트리 (44개 설정 등록)
├── dataloader.py                       # 데이터로더 (MVTec, VisA, BTAD, Custom)
├── README.md                           # 프로젝트 문서
│
└── models/                             # 모델 디렉토리
    ├── __init__.py
    │
    # ===== 1. Memory-based Models (3) =====
    ├── model_padim.py                  # PaDiM (2020) - Patch Distribution Modeling
    ├── model_patchcore.py              # PatchCore (2022) - Coreset Subsampling
    ├── model_dfkde.py                  # DFKDE (2022) - Deep Feature Kernel Density
    │
    # ===== 2. Normalizing Flow Models (4) =====
    ├── model_cflow.py                  # CFlow (2021) - Conditional Normalizing Flows
    ├── model_fastflow.py               # FastFlow (2021) - 2D Normalizing Flows
    ├── model_csflow.py                 # CS-Flow (2021) - Cross-Scale Flows
    ├── model_uflow.py                  # U-Flow (2022) - U-shaped Normalizing Flow
    │
    # ===== 3. Knowledge Distillation Models (4) =====
    ├── model_stfpm.py                  # STFPM (2021) - Student-Teacher Feature Pyramid
    ├── model_fre.py                    # FRE (2023) - Feature Reconstruction Error
    ├── model_reverse_distillation.py   # Reverse Distillation (2022)
    ├── model_efficientad.py            # EfficientAD (2024) - Millisecond-Level Detection
    │
    # ===== 4. Reconstruction Models (4) =====
    ├── model_autoencoder.py            # Vanilla Autoencoder (Baseline)
    ├── model_ganomaly.py               # GANomaly (2018) - GAN-based Detection
    ├── model_draem.py                  # DRAEM (2021) - Discriminative Reconstruction
    ├── model_dsr.py                    # DSR (2022) - Dual Subspace Re-Projection
    │
    # ===== 5. Feature Adaptation Models (2) =====
    ├── model_dfm.py                    # DFM (2019) - Deep Feature Modeling
    ├── model_cfa.py                    # CFA (2022) - Coupled-hypersphere Adaptation
    │
    # ===== 6. Foundation Models (3) =====
    ├── model_dinomaly.py               # Dinomaly (2025) - DINOv2-based Detection
    ├── model_supersimplenet.py         # SuperSimpleNet (2024) - Fast Surface Defect
    ├── model_uninet.py                 # UniNet (2025) - Unified Contrastive Learning
    │
    # ===== Common Components =====
    ├── components/                     # 공통 컴포넌트 (모든 모델 공유)
    │   ├── __init__.py
    │   ├── trainer.py                  # BaseTrainer + EarlyStopper
    │   ├── backbone.py                 # Backbone 경로 관리 (BACKBONE_DIR, get_backbone_path)
    │   ├── feature_extractor.py        # TimmFeatureExtractor (ResNet, WideResNet, EfficientNet)
    │   ├── tiler.py                    # Image Tiling/Untiling
    │   ├── blur.py                     # GaussianBlur2d
    │   ├── pca.py                      # PCA Implementation
    │   ├── k_center_greedy.py          # Coreset Sampling (PatchCore)
    │   ├── multi_variate_gaussian.py   # Multivariate Gaussian (PaDiM)
    │   ├── dynamic_buffer.py           # DynamicBufferMixin
    │   ├── all_in_one_block.py         # FrEIA Block (Flow Models)
    │   ├── perlin.py                   # Perlin Noise Generator (DRAEM, DSR)
    │   ├── sspcab.py                   # SSPCAB Module (DRAEM)
    │   ├── resnet_decoder.py           # ResNet Decoder (Reverse Distillation)
    │   └── multi_random_choice.py      # Multi Random Choice
    │
    └── components_dinomaly/            # Dinomaly 전용 컴포넌트
        ├── __init__.py
        ├── dinov2_loader.py            # DINOv2 Model Loader (load function)
        ├── download.py                 # Model Download Utilities
        ├── layers.py                   # DinomalyMLP, LinearAttention
        ├── loss.py                     # CosineHardMiningLoss
        ├── optimizer.py                # StableAdamW, WarmCosineScheduler
        └── vision_transformer.py       # Vision Transformer Components
```

---

## 2. Datasets 디렉토리 (데이터 저장소)

```
datasets/
├── README.md                    # 데이터셋 설명 및 다운로드 링크
│
# ===== Standard Benchmark Datasets =====
├── mvtec/                       # MVTec AD Dataset (15 categories)
│   ├── bottle/
│   │   ├── train/
│   │   │   └── good/            # Normal training images (209 images)
│   │   ├── test/
│   │   │   ├── good/            # Normal test images (20 images)
│   │   │   ├── broken_large/    # Anomaly type 1 (9 images)
│   │   │   ├── broken_small/    # Anomaly type 2 (15 images)
│   │   │   └── contamination/   # Anomaly type 3 (6 images)
│   │   └── ground_truth/
│   │       ├── broken_large/    # Anomaly masks (9 images)
│   │       ├── broken_small/
│   │       └── contamination/
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
│
├── visa/                        # VisA Dataset (12 categories)
│   ├── candle/
│   │   ├── Data/
│   │   │   └── Images/
│   │   │       ├── Anomaly/
│   │   │       │   ├── 000/
│   │   │       │   ├── 001/
│   │   │       │   └── ...
│   │   │       └── Normal/
│   │   ├── image_anno.csv       # Image annotations
│   │   └── split_csv/
│   │       ├── 1cls.csv
│   │       └── 2cls.csv
│   ├── capsules/
│   ├── cashew/
│   ├── chewinggum/
│   ├── fryum/
│   ├── macaroni1/
│   ├── macaroni2/
│   ├── pcb1/
│   ├── pcb2/
│   ├── pcb3/
│   ├── pcb4/
│   └── pipe_fryum/
│
├── btad/                        # BTAD Dataset (3 categories)
│   ├── 01/
│   │   ├── train/
│   │   │   └── ok/              # Normal training images
│   │   ├── test/
│   │   │   ├── ok/              # Normal test images
│   │   │   └── ko/              # Anomaly images
│   │   └── ground_truth/
│   │       └── ko/              # Anomaly masks (.png or .bmp)
│   ├── 02/
│   └── 03/
│
# ===== Custom Dataset (User-defined) =====
├── custom/                      # Custom Dataset Root
│   ├── dataset_A/               # Custom dataset A
│   │   └── data_rgb/
│   │       ├── normal/          # Normal images
│   │       │   ├── pattern1 60 100.png
│   │       │   ├── pattern1 60 200.png
│   │       │   ├── pattern2 80 150.png
│   │       │   └── ...
│   │       ├── defect_type1/    # Anomaly type 1
│   │       │   ├── pattern1 60 100.png
│   │       │   └── ...
│   │       ├── defect_type2/    # Anomaly type 2
│   │       │   └── ...
│   │       └── data_info.csv    # Auto-generated metadata
│   │
│   ├── dataset_B/               # Custom dataset B
│   │   └── data_rgb/
│   │       ├── normal/
│   │       ├── scratch/
│   │       ├── crack/
│   │       └── data_info.csv
│   │
│   └── README_custom.md         # Custom dataset format guide
│
# ===== Auxiliary Datasets =====
├── dtd/                         # Describable Textures Dataset (DRAEM용)
│   └── images/
│       ├── banded/
│       ├── blotchy/
│       ├── braided/
│       ├── bubbly/
│       └── ... (47 texture categories)
│
└── imagenette2/                 # ImageNet subset (EfficientAD용)
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

---

## 3. Backbones 디렉토리 (사전 학습된 가중치)

```
backbones/
├── README.md                    # 백본 다운로드 가이드
│
# ===== CNN Backbones =====
├── resnet18.pth                 # ResNet-18 (ImageNet pretrained)
├── resnet50.pth                 # ResNet-50 (ImageNet pretrained)
├── wide_resnet50_2.pth          # Wide ResNet-50-2 (ImageNet pretrained)
├── efficientnet_b5.pth          # EfficientNet-B5 (ImageNet pretrained)
│
# ===== Transformer Backbones =====
├── cait_m48_448.fb_dist_in1k/
│   └── model.safetensors        # CaiT-M48 (448x448)
│
├── deit_base_distilled_patch16_384.fb_in1k/
│   └── model.safetensors        # DeiT-Base (384x384)
│
# ===== DINOv2 Backbones (Foundation Models) =====
├── dinov2_vit_small_14/
│   └── pytorch_model.bin        # DINOv2 ViT-Small
│
├── dinov2_vit_base_14/
│   └── pytorch_model.bin        # DINOv2 ViT-Base
│
├── dinov2_vit_large_14/
│   └── pytorch_model.bin        # DINOv2 ViT-Large
│
# ===== Model-specific Pretrained Weights =====
├── efficientad/
│   ├── pretrained_teacher_small.pth
│   ├── pretrained_teacher_medium.pth
│   └── imagenette2/             # Symlink to datasets/imagenette2/
│       └── train/
│
└── dsr/
    └── vq_model_pretrained_128_4096.pckl
```

---

## 4. Outputs 디렉토리 (학습 결과)

```
outputs/
├── mvtec/
│   ├── bottle/
│   │   ├── padim/
│   │   │   ├── model_mvtec_bottle_padim_epochs-1.pth
│   │   │   ├── results_mvtec_bottle_padim_thresholds.txt
│   │   │   ├── histogram_mvtec_bottle_padim_scores.png
│   │   │   ├── image_mvtec_bottle_padim_normal_0001.png
│   │   │   ├── image_mvtec_bottle_padim_anomaly_0001.png
│   │   │   └── ...
│   │   ├── stfpm/
│   │   ├── efficientad-small/
│   │   └── dinomaly-base-224/
│   ├── wood/
│   ├── grid/
│   └── ...
│
├── visa/
│   ├── candle/
│   ├── capsules/
│   └── ...
│
├── btad/
│   ├── 01/
│   ├── 02/
│   └── 03/
│
└── custom/
    ├── dataset_A/
    │   ├── pattern1/
    │   ├── pattern2/
    │   └── all/                 # When category="all"
    └── dataset_B/
```

---

## 주요 파일 상세 설명

### 1. `main.py`

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

from train import train, train_models, set_globals, print_globals

if __name__ == "__main__":
    set_globals(
        dataset_dir="/mnt/d/datasets",
        backbone_dir="/mnt/d/backbones",
        output_dir="/mnt/d/outputs",
        seed=42,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    print_globals()

    # Example: Single model training
    train("mvtec", "bottle", "stfpm", num_epochs=50)

    # Example: Multi-model training
    train_models("mvtec",
        categories=["bottle", "wood", "grid"],
        models=["padim", "stfpm", "efficientad-small"]
    )
```

---

### 2. `train.py`

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

---

### 3. `registry.py`

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
- register_all_models(): Register all available models (auto-called on import)
"""
```

---

### 4. `dataloader.py`

```python
"""
Unified DataLoader for Multiple Dataset Types

Supported Datasets:
1. MVTecDataset: MVTec AD benchmark (15 categories)
   - Structure: dataset/category/train|test/good|defect_type/
   - Masks: dataset/category/ground_truth/defect_type/

2. VisADataset: VisA benchmark (12 categories)
   - Structure: dataset/category/Data/Images/Normal|Anomaly/
   - CSV: dataset/category/image_anno.csv

3. BTADDataset: BTAD benchmark (3 categories)
   - Structure: dataset/category/train|test/ok|ko/
   - Masks: dataset/category/ground_truth/ko/

4. CustomDataset: User-defined datasets
   - Structure: dataset/data_rgb/normal|defect_type/
   - Filename: {category} {freq} {dimming}[_extra].png
   - CSV: Auto-generated data_info.csv

Key Features:
- Automatic CSV generation for custom datasets
- Support for multiple dataset types (list input)
- Support for multiple categories (list input or "all")
- Flexible image transforms (with/without normalization)
- Mask handling for anomaly localization

Global Variable:
- DATASET_DIR: Default dataset directory

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
- CustomDataset: Custom dataset loader with auto CSV generation
"""
```

---

### 5. `models/components/trainer.py`

```python
"""
Base Trainer and Early Stopping

Classes:
- EarlyStopper: Early stopping implementation
  * Supports 'min' and 'max' modes
  * Patience-based stopping
  * Target value-based stopping

- BaseTrainer: Base trainer class for all models
  * Training loop management
  * Validation with metrics (AUROC, AUPR)
  * Model save/load
  * Results visualization (histogram, anomaly maps)
  * GPU memory management
  * Early stopping integration

Key Methods:
- fit(): Main training loop
- train_epoch(): Single epoch training
- validation_epoch(): Validation with metrics
- save_model(): Save model weights
- load_model(): Load model weights
- save_results(): Save evaluation results to text file
- save_histogram(): Save score distribution plot
- save_maps(): Save anomaly map visualizations

Hooks (Override in subclasses):
- on_fit_start(): Called before training starts (prints start message, records time)
- on_fit_end(weight_path): Called after training ends (prints elapsed time, saves model)
- on_epoch_start(): Called before each epoch (records epoch start time)
- on_epoch_end(): Called after each epoch (handles learning rate scheduler)
- on_train_start(): Called before training epoch (sets model to train mode)
- on_train_end(): Called after training epoch (updates history, checks early stopping)
- on_validation_start(): Called before validation (sets model to eval mode)
- on_validation_end(): Called after validation (prints metrics, checks early stopping)
- train_step(): Single training step (MUST override)
- validation_step(): Single validation step (MUST override)
"""
```

---

## Custom Dataset 상세 가이드

### Custom Dataset 파일명 규칙
```
Format: {category} {freq} {dimming}[_{extra}].png

Components:
- category: Pattern or design name (e.g., pattern1, design_A)
- freq: Frequency parameter (integer)
- dimming: Dimming parameter (integer)
- extra: Optional extra identifier (e.g., v2, test)

Valid Examples:
✓ pattern1 60 100.png
✓ pattern2 120 200.png
✓ design_A 80 150_v2.png
✓ test_pattern 100 50.png

Invalid Examples:
✗ image001.png                  # Missing metadata
✗ pattern1_60_100.png           # Wrong separator (use space)
✗ pattern1 60.png               # Missing dimming value
```

### Custom Dataset 디렉토리 구조
```
custom/
└── your_dataset_name/
    └── data_rgb/
        ├── normal/              # 정상 이미지 (label=0)
        │   ├── pattern1 60 100.png
        │   ├── pattern1 60 200.png
        │   ├── pattern2 80 150.png
        │   └── ...
        ├── defect_type1/        # 결함 타입 1 (label=1)
        │   ├── pattern1 60 100.png
        │   ├── pattern2 80 150.png
        │   └── ...
        ├── defect_type2/        # 결함 타입 2 (label=1)
        │   └── ...
        └── data_info.csv        # 자동 생성됨

data_info.csv 구조:
┌─────────────────────────┬──────────┬──────┬─────────┬─────────────────┬──────────────┬──────────────┬───────┐
│ filename                │ category │ freq │ dimming │ image_path      │ dataset_type │ defect_type  │ label │
├─────────────────────────┼──────────┼──────┼─────────┼─────────────────┼──────────────┼──────────────┼───────┤
│ pattern1 60 100.png     │ pattern1 │ 60   │ 100     │ /full/path/...  │ dataset_name │ normal       │ 0     │
│ pattern1 60 200.png     │ pattern1 │ 60   │ 200     │ /full/path/...  │ dataset_name │ defect_type1 │ 1     │
│ pattern2 80 150.png     │ pattern2 │ 80   │ 150     │ /full/path/...  │ dataset_name │ normal       │ 0     │
└─────────────────────────┴──────────┴──────┴─────────┴─────────────────┴──────────────┴──────────────┴───────┘
```

### Custom Dataset 사용 예시

```python
# 1. Single custom dataset, single category
train("your_dataset_name", "pattern1", "stfpm", num_epochs=50)

# 2. Single custom dataset, multiple categories
train("your_dataset_name", ["pattern1", "pattern2"], "stfpm", num_epochs=50)

# 3. Single custom dataset, all categories
train("your_dataset_name", "all", "dinomaly-base-224", num_epochs=15)

# 4. Multiple custom datasets, specific categories
train(["dataset_A", "dataset_B"], ["pattern1", "pattern2"],
      "efficientad-small", num_epochs=20)

# 5. Multiple custom datasets, all categories
train(["dataset_A", "dataset_B"], "all", "uninet", num_epochs=20)
```

---

## 전체 워크스페이스 구조 요약

```
workspace/
├── project/              # 코드 및 실행 파일
│   ├── main.py
│   ├── train.py
│   ├── registry.py
│   ├── dataloader.py
│   └── models/
│
├── datasets/            # 데이터셋 저장소
│   ├── mvtec/           # MVTec AD (15 categories)
│   ├── visa/            # VisA (12 categories)
│   ├── btad/            # BTAD (3 categories)
│   ├── custom/          # Custom datasets
│   ├── dtd/             # Textures (DRAEM)
│   └── imagenette2/     # ImageNet subset (EfficientAD)
│
├── backbones/            # 사전 학습된 가중치
│   ├── *.pth            # CNN backbones
│   ├── *_vit_*/         # Transformer backbones
│   ├── efficientad/     # EfficientAD specific
│   └── dsr/             # DSR specific
│
└── outputs/              # 학습 결과
    ├── mvtec/
    ├── visa/
    ├── btad/
    └── custom/
```

이 구조는 다음과 같은 특징이 있습니다:
- **명확한 구조**: 코드, 데이터, 백본, 결과 분리
- **Custom Dataset 지원**: 유연한 사용자 데이터셋 처리
- **확장 가능**: 새로운 모델/데이터셋 추가 용이