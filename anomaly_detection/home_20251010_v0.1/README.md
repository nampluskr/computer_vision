# Anomaly Detection Framework

PyTorch 기반의 20개 SOTA 모델을 지원하는 산업용 비전 이상 감지 통합 프레임워크입니다.

## 목차

1. [개요](#1-개요)
2. [주요 기능](#2-주요-기능)
3. [지원 모델](#3-지원-모델)
4. [빠른 시작](#4-빠른-시작)
5. [지원 데이터셋](#5-지원-데이터셋)
6. [문서](#6-문서)
7. [프로젝트 구조](#7-프로젝트-구조)
8. [사용 예시](#8-사용-예시)
9. [출력 결과](#9-출력-결과)
10. [시스템 요구사항](#10-시스템-요구사항)
11. [설정](#11-설정)

---

## 1. 개요

본 프레임워크는 다양한 산업 검사 데이터셋에서 최신 이상 감지 모델을 학습하고 평가하기 위한 통합 인터페이스를 제공합니다. PyTorch 기반으로 구축되었으며, 6개 카테고리에 걸쳐 20개의 고유 모델과 44개의 서로 다른 구성을 지원합니다.

### 1.1. 개발 배경 및 목적

산업 현장에서의 이상 감지는 제품 품질 관리와 불량 검출에 필수적입니다. 그러나 실제 산업 데이터는 다음과 같은 특징을 가집니다:

#### 1.1.1. 실제 산업 데이터의 특성

- **이상 데이터 부족**: 정상 데이터에 비해 불량 데이터가 극히 적음
- **다양한 이상 유형**: 결함 종류에 따라 모델의 감도와 특징 검출 능력이 상이함
- **모델별 성능 차이**: 데이터 특성에 따라 특정 모델이 다른 모델보다 월등한 성능을 보임

#### 1.1.2. 프레임워크 개발 목표

이러한 이유로 **단일 모델에 의존하지 않고 다양한 SOTA 모델을 평가**할 수 있는 통합 프레임워크가 필요합니다. 본 프레임워크는 다음을 목표로 합니다:

1. 표준 벤치마크 데이터셋(MVTec, VisA, BTAD)에서 모델 성능 검증
2. 사용자 정의 데이터셋에 대한 신속한 적용 및 평가
3. 다양한 모델 비교를 통한 최적 모델 선정
4. 최종적으로 3-5개 Top 모델을 활용한 하이브리드 또는 앙상블 모델 구축

### 1.2. 기술적 도전과제

#### 1.2.1. 오프라인 환경 제약

로컬 실행 환경은 보안 정책에 따른 방화벽으로 외부 네트워크 접근이 제한됩니다:

**제약사항:**
- **사전학습 가중치 다운로드 불가**: ResNet, Wide ResNet, DINOv2 등의 백본 가중치를 자동으로 다운로드할 수 없음
- **보조 데이터셋 접근 불가**: DRAEM의 DTD, EfficientAD의 Imagenette2 등 필수 데이터셋 다운로드 불가
- **라이브러리 설치 제한**: PyPI 접근 제한으로 의존성 관리 복잡

**해결 방안:**
- 외부 환경에서 모든 가중치 파일을 사전 다운로드하여 `backbones/` 폴더에 저장
- 보조 데이터셋을 `datasets/` 폴더에 사전 배치
- 최소한의 핵심 라이브러리만 사용하여 의존성 최소화

#### 1.2.2. Lightning 의존성 제거

Anomalib은 PyTorch Lightning을 사용하지만, 로컬 환경에서는 다음 이유로 사용이 어렵습니다:

**문제점:**
- **라이브러리 호환성 문제**: Lightning과 관련 의존성 패키지 간 버전 충돌
- **불필요한 복잡성**: 단순 학습 파이프라인에 과도한 추상화
- **설치 오류**: 오프라인 환경에서 Lightning 설치 실패

**해결 방안:**
- 순수 PyTorch만을 사용한 학습 파이프라인 구현
- BaseTrainer 클래스를 통한 통합 학습 인터페이스 제공
- Hook 패턴을 통한 모델별 커스터마이징 지원

### 1.3. 프로젝트의 기술적 가치

본 프로젝트는 단순한 코드 복사가 아닌, 다음과 같은 독창적인 엔지니어링 결과물입니다:

#### 1.3.1. 아키텍처 재설계

**Lightning → Pure PyTorch 변환:**
- 20개 모델의 Lightning 기반 구현을 순수 PyTorch로 전환
- 통합 학습 인터페이스: BaseTrainer를 통한 일관된 학습/평가 파이프라인
- Hook 기반 확장: 모델별 특성을 유지하면서 공통 인터페이스 제공

#### 1.3.2. 오프라인 환경 최적화

**핵심 기능:**
- **백본 관리 시스템**: `backbone.py`를 통한 로컬 가중치 경로 관리
- **자동 CSV 생성**: Custom 데이터셋의 메타데이터 자동 생성
- **최소 의존성**: 핵심 라이브러리(torch, timm, einops 등)만 사용

#### 1.3.3. 사용자 경험 개선

**간편한 인터페이스:**
- **단일 함수 인터페이스**: 모든 모델을 `train()` 함수 하나로 학습
- **자동 설정 관리**: Registry 패턴을 통한 모델별 최적 설정 자동 적용
- **결과 자동 저장**: 학습 결과, 시각화, 임계값 분석 자동 생성

```python
# 단 한 줄로 모델 학습
train("mvtec", "bottle", "stfpm", num_epochs=50)
```

### 1.4. 개발 과정 및 규모

본 프로젝트는 **1개월 이상**의 집중적인 개발 기간을 거쳐 완성되었습니다:

#### 1.4.1. 주요 개발 작업

**6개 카테고리 20개 모델 구현:**

```
models/
├── # ===== 1. Memory-based Models (3) =====
├── model_padim.py              # PaDiM (2020): Patch Distribution Modeling
├── model_patchcore.py          # PatchCore (2022): Coreset-based Memory Bank
├── model_dfkde.py              # DFKDE (2022): Deep Feature Kernel Density Estimation
│
├── # ===== 2. Normalizing Flow Models (4) =====
├── model_cflow.py              # CFlow (2021): Conditional Normalizing Flows
├── model_fastflow.py           # FastFlow (2021): 2D Normalizing Flows
├── model_csflow.py             # CS-Flow (2021): Cross-Scale Flows
├── model_uflow.py              # U-Flow (2022): U-shaped Normalizing Flow
│
├── # ===== 3. Knowledge Distillation Models (4) =====
├── model_stfpm.py              # STFPM (2021): Student-Teacher Feature Pyramid Matching
├── model_fre.py                # FRE (2023): Feature Reconstruction Error
├── model_reverse_distillation.py  # Reverse Distillation (2022): One-Class Embedding
├── model_efficientad.py        # EfficientAD (2024): Millisecond-Level Anomaly Detection
│
├── # ===== 4. Reconstruction Models (4) =====
├── model_autoencoder.py        # Autoencoder (Baseline): Vanilla Autoencoder
├── model_ganomaly.py           # GANomaly (2018): GAN-based Semi-Supervised Detection
├── model_draem.py              # DRAEM (2021): Discriminatively Trained Reconstruction
├── model_dsr.py                # DSR (2022): Dual Subspace Re-Projection Network
│
├── # ===== 5. Feature Adaptation Models (2) =====
├── model_dfm.py                # DFM (2019): Deep Feature Modeling
├── model_cfa.py                # CFA (2022): Coupled-hypersphere Feature Adaptation
│
├── # ===== 6. Foundation Models (3) =====
├── model_dinomaly.py           # Dinomaly (2025): DINOv2-based Multi-Class Detection
├── model_supersimplenet.py     # SuperSimpleNet (2024): Fast and Reliable Detection
└── model_uninet.py             # UniNet (2025): Unified Contrastive Learning
```

**총 44개 구성:** 백본 선택, 이미지 크기, 모델 크기 등 다양한 설정 조합

**개발 내용:**
- 20개 모델의 Lightning 코드 분석 및 PyTorch 변환
- 6개 카테고리별 학습 파이프라인 구현 및 테스트
- 4가지 데이터셋 포맷 지원을 위한 통합 DataLoader 개발
- 모델별 성능 검증 및 하이퍼파라미터 최적화
- 오프라인 환경에서의 안정성 테스트

#### 1.4.2. 코드 규모

**구현 규모:**
- 약 15,000줄 이상의 Python 코드
- 20개의 모델 구현 파일 (`model_*.py`)
- 30개 이상의 공통 컴포넌트 (`components/`)
- 상세한 문서화 (6개의 주요 문서)

**주요 컴포넌트:**
```
project/
├── main.py                    # Main execution script
├── train.py                   # Training utility functions
├── registry.py                # Model registry system (44 configurations)
├── dataloader.py              # Unified dataset loaders (4 datasets)
└── models/
    ├── model_*.py             # 20 model implementations
    ├── components/            # Shared components
    │   ├── trainer.py         # BaseTrainer (unified interface)
    │   ├── backbone.py        # Backbone weight management
    │   ├── tiler.py           # Tiling utilities
    │   └── ...
    └── components_dinomaly/   # Dinomaly-specific components
```

### 1.5. Anomalib과의 관계

본 프레임워크는 [Anomalib](https://github.com/openvinotoolkit/anomalib)의 모델 아키텍처를 기반으로 하되, 실행 환경에 맞게 전면 재구성하였습니다:

#### 1.5.1. Anomalib 활용 부분

**모델 아키텍처:**
- 각 모델의 `torch_model.py`: Anomalib의 검증된 구현 사용
- 논문 기반 알고리즘: 원 논문의 수식 및 방법론 충실히 구현
- 네트워크 구조: 공식 구현과 동일한 레이어 구성

#### 1.5.2. 독자적 구현 부분

**완전 재구현:**
- **학습 파이프라인**: Lightning 제거, Pure PyTorch 구현
  - `BaseTrainer`: 통합 학습 인터페이스
  - `*Trainer`: 20개 모델별 학습 로직
- **데이터 로더**: 4가지 데이터셋 통합 지원
  - MVTec, VisA, BTAD, Custom 데이터셋
- **오프라인 지원**: 
  - `backbone.py`: 백본 가중치 로컬 관리
  - 최소 의존성 설계
- **사용자 인터페이스**: 
  - `registry.py`: Registry 패턴 기반 모델 관리 (44개 구성)
  - `train.py`: 통합 학습 함수

#### 1.5.3. 프로젝트 기여

**핵심 기여:**
- 오프라인 환경에서 동작 가능한 완전한 프레임워크
- 사용자 친화적 인터페이스 및 자동화
- 상세한 한국어 문서 및 가이드
- Custom 데이터셋 지원 강화

모든 모델 구현은 원 논문 및 Anomalib 구현을 기반으로 하며, 상세한 인용 정보는 [Models 문서](docs/03-models.md)를 참조하시기 바랍니다.

---

## 2. 주요 기능

- **20개의 SOTA 모델 지원**: PaDiM, PatchCore, STFPM, EfficientAD, Dinomaly 등
- **44개의 구성 조합**: 백본, 이미지 크기, 모델 크기별 다양한 설정
- **6개 모델 카테고리**: Memory-based, Normalizing Flow, Knowledge Distillation, Reconstruction, Feature Adaptation, Foundation Models
- **다중 데이터셋 지원**: MVTec AD, VisA, BTAD 및 사용자 정의 데이터셋
- **통합 인터페이스**: 모든 모델과 데이터셋을 위한 단일 API
- **간편한 학습**: 간단한 설정 및 실행
- **풍부한 평가 지표**: AUROC, AUPR, 임계값 분석 및 시각화
- **확장 가능한 구조**: 새로운 모델 및 데이터셋 추가 용이
- **오프라인 환경 지원**: 외부 인터넷이 차단된 환경을 위한 사전 다운로드 가중치 지원

---

## 3. 지원 모델

### 3.1. 카테고리별 모델 (20개 모델, 44개 구성)

| 카테고리 | 모델 수 | 모델 목록 |
|---------|--------|----------|
| **Memory-based** | 3 | PaDiM, PatchCore, DFKDE |
| **Normalizing Flow** | 4 | CFlow, FastFlow, CS-Flow, U-Flow |
| **Knowledge Distillation** | 4 | STFPM, FRE, Reverse Distillation, EfficientAD |
| **Reconstruction** | 4 | Autoencoder, GANomaly, DRAEM, DSR |
| **Feature Adaptation** | 2 | DFM, CFA |
| **Foundation Models** | 3 | Dinomaly, SuperSimpleNet, UniNet |

### 3.2. 주요 모델 특징

#### Memory-based Models (학습 불필요)
- **PaDiM (2020)**: Patch Distribution Modeling
- **PatchCore (2022)**: Coreset-based Memory Bank
- **DFKDE (2022)**: Deep Feature Kernel Density Estimation

#### Normalizing Flow Models (높은 정확도)
- **CFlow (2021)**: Conditional Normalizing Flows (2 variants)
- **FastFlow (2021)**: 2D Normalizing Flows (3 variants)
- **CS-Flow (2021)**: Cross-Scale Flows
- **U-Flow (2022)**: U-shaped Normalizing Flow (2 variants)

#### Knowledge Distillation Models (균형잡힌 성능)
- **STFPM (2021)**: Student-Teacher Feature Pyramid Matching
- **FRE (2023)**: Feature Reconstruction Error
- **Reverse Distillation (2022)**: One-Class Embedding
- **EfficientAD (2024)**: Millisecond-Level Detection (2 variants)

#### Reconstruction Models (기본 접근)
- **Autoencoder (Baseline)**: Vanilla Autoencoder
- **GANomaly (2018)**: GAN-based Semi-Supervised Detection
- **DRAEM (2021)**: Discriminatively Trained Reconstruction
- **DSR (2022)**: Dual Subspace Re-Projection Network

#### Feature Adaptation Models (도메인 적응)
- **DFM (2019)**: Deep Feature Modeling
- **CFA (2022)**: Coupled-hypersphere Feature Adaptation

#### Foundation Models (최신 성능)
- **Dinomaly (2025)**: DINOv2-based Multi-Class Detection (9 variants)
- **SuperSimpleNet (2024)**: Fast and Reliable Detection (2 variants)
- **UniNet (2025)**: Unified Contrastive Learning

---

## 4. 빠른 시작

### 4.1. 설치

```bash
# Install PyTorch (check official site for your system)
pip install torch torchvision

# Install core dependencies
pip install timm einops FrEIA omegaconf safetensors torchmetrics kornia scipy
```

**참고:**
- 표준 라이브러리(tqdm, scikit-learn, matplotlib, numpy, pandas 등)는 별도 설치가 필요할 수 있습니다
- PyTorch 설치는 시스템 환경(CPU/GPU, CUDA 버전)에 따라 [PyTorch 공식 사이트](https://pytorch.org)를 참조하시기 바랍니다

### 4.2. 기본 사용법

```python
from train import train, set_globals

# Configure global paths
set_globals(
    dataset_dir="/path/to/datasets",
    backbone_dir="/path/to/backbones",
    output_dir="/path/to/outputs"
)

# Train single model
train("mvtec", "bottle", "stfpm", num_epochs=50)
```

### 4.3. 다중 모델 학습

```python
from train import train_models

# Train multiple models on multiple categories
train_models(
    dataset_type="mvtec",
    categories=["bottle", "wood", "grid"],
    models=["padim", "stfpm", "efficientad-small"]
)
```

---

## 5. 지원 데이터셋

| 데이터셋 | 카테고리 수 | 특징 | 다운로드 |
|---------|-----------|------|----------|
| **MVTec AD** | 15 | 산업 검사 벤치마크 | [공식 사이트](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **VisA** | 12 | 시각적 이상 감지 | [GitHub](https://github.com/amazon-science/spot-diff) |
| **BTAD** | 3 | 표면 결함 감지 | [공식 사이트](http://avires.dimi.uniud.it/papers/btad/) |
| **Custom** | 무제한 | 사용자 정의 데이터 | - |

**보조 데이터셋:**
- **DTD**: DRAEM 모델용 텍스처 데이터셋
- **Imagenette2**: EfficientAD 모델용 ImageNet 서브셋

---

## 6. 문서

- **[Getting Started](docs/01-getting-started.md)** - 설치 및 설정 가이드
- **[Architecture](docs/02-architecture.md)** - 전체 프레임워크 구조
- **[Models](docs/03-models.md)** - 상세 모델 설명
- **[Datasets](docs/04-datasets.md)** - 데이터셋 준비 가이드
- **[Training](docs/05-training.md)** - 학습 설정 및 팁
- **[Inference](docs/06-inference.md)** - 배포 및 추론

---

## 7. 프로젝트 구조

```
project/
├── main.py              # Main execution script
├── train.py             # Training utility functions
├── registry.py          # Model registry system (44 configurations)
├── dataloader.py        # Unified dataset loaders
└── models/              # Model implementations
    ├── model_padim.py
    ├── model_patchcore.py
    ├── model_stfpm.py
    ├── model_efficientad.py
    ├── model_dinomaly.py
    ├── ... (20 models total)
    ├── components/              # Shared components
    │   ├── trainer.py           # BaseTrainer
    │   ├── backbone.py          # Backbone management
    │   └── tiler.py             # Tiling utilities
    └── components_dinomaly/     # Dinomaly-specific
```

---

## 8. 사용 예시

### 8.1. MVTec 데이터셋 학습

```python
# Quick evaluation (no training required)
train("mvtec", "bottle", "padim", num_epochs=1)
train("mvtec", "bottle", "patchcore", num_epochs=1)

# Standard training
train("mvtec", "bottle", "stfpm", num_epochs=50)
train("mvtec", "bottle", "efficientad-small", num_epochs=20)

# Foundation models with different sizes
train("mvtec", "bottle", "dinomaly-small-224", num_epochs=15)
train("mvtec", "bottle", "dinomaly-base-224", num_epochs=15)
train("mvtec", "bottle", "dinomaly-large-224", num_epochs=15)

# Flow-based models with different backbones
train("mvtec", "tile", "fastflow-resnet50", num_epochs=500)
train("mvtec", "tile", "fastflow-cait", num_epochs=500)
train("mvtec", "tile", "fastflow-deit", num_epochs=500)

# Multiple categories
train("mvtec", ["bottle", "wood", "grid"], "stfpm", num_epochs=50)

# All categories
train("mvtec", "all", "stfpm", num_epochs=50)

# Custom settings
train("mvtec", "bottle", "stfpm", 
      num_epochs=50, batch_size=32, img_size=512)

# Disable normalization (for specific models)
train("mvtec", "bottle", "autoencoder", num_epochs=50, normalize=False)
train("mvtec", "bottle", "efficientad-small", num_epochs=20, normalize=False)
```

### 8.2. 사용자 정의 데이터셋 학습

```python
# Single custom dataset
train("your_dataset_name", "pattern1", "stfpm", num_epochs=50)

# Multiple categories
train("your_dataset_name", ["pattern1", "pattern2"], "stfpm", num_epochs=50)

# All categories
train("your_dataset_name", "all", "dinomaly-base-224", num_epochs=15)

# Multiple datasets
train(["dataset_A", "dataset_B"], ["pattern1", "pattern2"],
      "efficientad-small", num_epochs=20)
```

### 8.3. 배치 학습

```python
from train import train_models

# Train multiple models sequentially
train_models(
    dataset_type="mvtec",
    categories=["bottle", "wood", "grid"],
    models=["padim", "patchcore", "stfpm", "efficientad-small"]
)
```

---

## 9. 출력 결과

### 9.1. 생성 파일 목록

학습 과정에서 다음 항목들이 자동으로 생성됩니다:

- **모델 가중치**: `.pth` 형식의 체크포인트 파일
- **평가 지표**: AUROC, AUPR, accuracy, precision, recall, F1 score
- **시각화**: 점수 분포 히스토그램, 이상 영역 맵
- **임계값 분석**: 4가지 방법(F1-Percentile, F1-Uniform, ROC, Percentile-95%)으로 계산된 임계값

### 9.2. 출력 디렉토리 구조

```
outputs/
└── mvtec/
    └── bottle/
        └── stfpm/
            ├── model_mvtec_bottle_stfpm_epochs-50.pth
            ├── results_mvtec_bottle_stfpm_thresholds.txt
            ├── histogram_mvtec_bottle_stfpm_scores.png
            └── image_mvtec_bottle_stfpm_*.png
```

### 9.3. 결과 파일 예시

**results_mvtec_bottle_stfpm_thresholds.txt:**
```
=== Image-level Results ===
AUROC: 0.987 (98.7%)
AUPR: 0.991 (99.1%)

=== Thresholds ===
F1 (Percentile): 0.234 | F1: 0.915
F1 (Uniform): 0.236 | F1: 0.916
ROC (Youden J): 0.241 | Sensitivity: 0.945, Specificity: 0.898
Percentile (95%): 0.189

=== Confusion Matrix (F1 Threshold) ===
TP=89, FN=5, TN=82, FP=11
Accuracy: 0.912, Precision: 0.887, Recall: 0.945
```

---

## 10. 시스템 요구사항

### 10.1. 하드웨어

**최소 사양:**
- CPU: 4코어 이상
- RAM: 8GB 이상
- GPU: NVIDIA GPU (8GB VRAM 이상)
- 저장공간: 20GB 이상

**권장 사양:**
- CPU: 8코어 이상
- RAM: 16GB 이상
- GPU: NVIDIA RTX 3080 (10GB) 이상 또는 RTX 3090 (24GB)
- 저장공간: 50GB 이상

### 10.2. 소프트웨어

- **운영체제**: Linux, Windows 10/11, macOS
- **Python**: 3.8 이상
- **PyTorch**: 1.10 이상
- **CUDA**: 11.0 이상 (GPU 사용 시)
- **cuDNN**: 8.0 이상 (GPU 사용 시)

### 10.3. 오프라인 환경

- 백본 가중치 사전 다운로드 (`backbones/`)
- 데이터셋 사전 다운로드 (`datasets/`)
- 보조 데이터셋 준비 (DTD, Imagenette2)

---

## 11. 설정

### 11.1. 전역 설정

```python
from train import set_globals

set_globals(
    dataset_dir="/path/to/datasets",      # Dataset root directory
    backbone_dir="/path/to/backbones",    # Pretrained weights directory
    output_dir="/path/to/outputs",        # Results output directory
    seed=42,                              # Random seed for reproducibility
    num_workers=8,                        # Number of DataLoader workers
    pin_memory=True,                      # Enable memory pinning for GPU
    persistent_workers=True,              # Keep workers alive between epochs
    show_globals=True                     # Print configuration
)
```

### 11.2. 모델별 설정

`registry.py`에 등록된 각 모델(44개 구성)은 다음을 설정할 수 있습니다:

**기본 설정:**
- `num_epochs`: 학습 에포크 수 (모델별 기본값)
- `batch_size`: 배치 크기 (모델별 기본값)
- `img_size`: 입력 이미지 크기 (모델별 기본값)
- `normalize`: ImageNet 정규화 여부 (모델별 기본값)

**모델 특화 설정:**
- `backbone`: 백본 아키텍처 (ResNet, Wide ResNet, DINOv2 등)
- `layers`: 특징 추출 레이어
- `model_size`: 모델 크기 (small, medium, large)
- 기타 모델별 하이퍼파라미터

### 11.3. 학습 시 설정 오버라이드

```python
# Override default settings
train(
    dataset_type="mvtec",
    category="bottle",
    model_type="stfpm",
    num_epochs=100,        # Override default
    batch_size=32,         # Override default
    img_size=512,          # Override default
    normalize=True         # Override default
)
```

---

## 라이선스

본 프로젝트는 Anomalib의 모델 구현을 기반으로 하며, Apache License 2.0을 따릅니다.

## 인용

본 프레임워크를 사용하는 경우, 해당 모델의 원 논문을 인용해 주시기 바랍니다. 상세한 인용 정보는 [Models 문서](docs/03-models.md)를 참조하시기 바랍니다.
