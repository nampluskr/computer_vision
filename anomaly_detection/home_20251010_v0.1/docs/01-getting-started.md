# Getting Started - 설치 및 설정 가이드

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [설치](#2-설치)
3. [디렉토리 구조 준비](#3-디렉토리-구조-준비)
4. [백본 가중치 다운로드](#4-백본-가중치-다운로드)
5. [데이터셋 준비](#5-데이터셋-준비)
6. [설정 확인](#6-설정-확인)
7. [첫 번째 실험 실행](#7-첫-번째-실험-실행)
8. [오프라인 환경 설정](#8-오프라인-환경-설정)
9. [문제 해결](#9-문제-해결)

---

## 1. 시스템 요구사항

### 1.1. 하드웨어 요구사항

**최소 사양:**
- **CPU**: 4코어 이상
- **RAM**: 8GB 이상
- **GPU**: NVIDIA GPU (8GB VRAM 이상)
- **저장공간**: 20GB 이상 여유 공간

**권장 사양:**
- **CPU**: 8코어 이상
- **RAM**: 16GB 이상
- **GPU**: NVIDIA RTX 3080 (10GB) 이상 또는 RTX 3090 (24GB)
- **저장공간**: 50GB 이상 여유 공간

**저장공간 상세:**
- 프로젝트 코드: ~100MB
- 백본 가중치: ~5GB
- 데이터셋: ~10GB (MVTec AD 기준)
- 출력 결과: 모델 및 실험 수에 따라 가변

### 1.2. 소프트웨어 요구사항

- **운영체제**: Linux, Windows 10/11, macOS
- **Python**: 3.8 이상
- **CUDA**: 11.0 이상 (GPU 사용 시)
- **cuDNN**: 8.0 이상 (GPU 사용 시)

---

## 2. 설치

### 2.1. 프로젝트 클론

```bash
# Clone repository
git clone <repository-url>
cd project
```

### 2.2. Python 환경 설정

가상환경 생성을 권장합니다:

#### venv 사용

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### conda 사용

```bash
# Create conda environment
conda create -n anomaly python=3.8

# Activate environment
conda activate anomaly
```

### 2.3. PyTorch 설치

시스템 환경에 맞는 PyTorch를 설치합니다.

#### CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### CPU 버전

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**참고:** 정확한 설치 명령어는 [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)에서 확인하시기 바랍니다.

### 2.4. 필수 라이브러리 설치

```bash
# Install core dependencies
pip install timm                    # PyTorch Image Models (feature extractors)
pip install einops                  # Tensor operations
pip install FrEIA                   # Normalizing flow models
pip install omegaconf               # Configuration management
pip install safetensors             # Safe tensor serialization
pip install torchmetrics            # Metrics for PyTorch
pip install kornia                  # Computer vision library for PyTorch
pip install scipy                   # Scientific computing
```

### 2.5. 표준 라이브러리 설치

일부 환경에서는 다음 라이브러리들이 필요할 수 있습니다:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image tqdm
```

### 2.6. 선택적 설치

```bash
# For anomalib components (optional)
pip install anomalib
```

### 2.7. 설치 확인

Python 환경에서 다음 코드를 실행하여 설치를 확인합니다:

```python
import torch
import timm
import einops
import torchmetrics

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"timm version: {timm.__version__}")
```

**예상 출력:**
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU device: NVIDIA GeForce RTX 3080
timm version: 0.9.12
```

---

## 3. 디렉토리 구조 준비

### 3.1. 기본 디렉토리 생성

#### Linux/macOS

```bash
# Create workspace directories
mkdir -p ~/workspace/anomaly_detection/datasets
mkdir -p ~/workspace/anomaly_detection/backbones
mkdir -p ~/workspace/anomaly_detection/outputs

# Navigate to workspace
cd ~/workspace/anomaly_detection
```

#### Windows (PowerShell)

```powershell
# Create workspace directories
New-Item -Path "C:\workspace\anomaly_detection\datasets" -ItemType Directory -Force
New-Item -Path "C:\workspace\anomaly_detection\backbones" -ItemType Directory -Force
New-Item -Path "C:\workspace\anomaly_detection\outputs" -ItemType Directory -Force

# Navigate to workspace
cd C:\workspace\anomaly_detection
```

#### Windows (Command Prompt)

```cmd
mkdir C:\workspace\anomaly_detection\datasets
mkdir C:\workspace\anomaly_detection\backbones
mkdir C:\workspace\anomaly_detection\outputs

cd C:\workspace\anomaly_detection
```

### 3.2. 프로젝트 배치

프로젝트를 workspace 내에 배치합니다:

```bash
# Clone or copy project to workspace
cd ~/workspace/anomaly_detection
git clone <repository-url> project

# Or move existing project
mv /path/to/project ./
```

### 3.3. 최종 디렉토리 구조

```
workspace/anomaly_detection/
├── project/              # Cloned project repository
│   ├── main.py
│   ├── train.py
│   ├── registry.py
│   ├── dataloader.py
│   └── models/
├── datasets/             # Dataset storage (empty initially)
├── backbones/            # Pretrained weights (empty initially)
└── outputs/              # Training results (empty initially)
```

---

## 4. 백본 가중치 다운로드

백본 가중치는 사전 학습된 모델의 가중치 파일입니다. 사용하려는 모델에 따라 필요한 백본만 다운로드하면 됩니다.

### 4.1. 필수 백본 (권장)

대부분의 모델에서 사용되는 백본입니다:

```bash
cd ~/workspace/anomaly_detection/backbones

# ResNet-50 (PaDiM, STFPM, FRE, DFM 등)
wget https://download.pytorch.org/models/resnet50-0676ba61.pth

# Wide ResNet-50-2 (PatchCore, CFA, SuperSimpleNet, UniNet 등)
wget https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
```

### 4.2. 추가 CNN 백본

```bash
# ResNet-18 (CFlow, U-Flow)
wget https://download.pytorch.org/models/resnet18-f37072fd.pth

# EfficientNet-B5 (CSFlow)
# Note: 파일명 변경 필요
wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth
mv efficientnet-b5-b6417697.pth efficientnet_b5_lukemelas-1a07897c.pth
```

### 4.3. Transformer 백본 (FastFlow, U-Flow)

#### CaiT-M48

```bash
mkdir -p cait_m48_448.fb_dist_in1k
cd cait_m48_448.fb_dist_in1k
wget https://huggingface.co/timm/cait_m48_448.fb_dist_in1k/resolve/main/model.safetensors
cd ..
```

#### DeiT-Base

```bash
mkdir -p deit_base_distilled_patch16_384.fb_in1k
cd deit_base_distilled_patch16_384.fb_in1k
wget https://huggingface.co/timm/deit_base_distilled_patch16_384.fb_in1k/resolve/main/model.safetensors
cd ..
```

### 4.4. DINOv2 백본 (Dinomaly)

#### DINOv2 ViT-Small

```bash
mkdir -p dinov2_vit_small_14
wget https://huggingface.co/FoundationVision/unitok_external/resolve/main/dinov2_vits14_pretrain.pth \
     -O dinov2_vit_small_14/dinov2_vits14_pretrain.pth
```

#### DINOv2 ViT-Base

```bash
mkdir -p dinov2_vit_base_14
wget https://huggingface.co/spaces/BoukamchaSmartVisions/FeatureMatching/resolve/main/models/dinov2_vitb14_pretrain.pth \
     -O dinov2_vit_base_14/dinov2_vitb14_pretrain.pth
```

#### DINOv2 ViT-Large

```bash
mkdir -p dinov2_vit_large_14
wget https://huggingface.co/Cusyoung/CrossEarth/resolve/main/dinov2_vitl14_pretrain.pth \
     -O dinov2_vit_large_14/dinov2_vitl14_pretrain.pth
```

### 4.5. 모델별 특수 가중치

#### EfficientAD

EfficientAD는 사전 학습된 teacher 네트워크가 필요합니다. 자세한 내용은 EfficientAD 문서를 참조하시기 바랍니다.

#### DSR

```bash
mkdir -p dsr
# DSR의 VQ 모델 가중치는 별도로 제공됩니다
```

### 4.6. 백본 다운로드 확인

```bash
# List downloaded backbones
ls -lh ~/workspace/anomaly_detection/backbones

# Expected output (example):
# resnet50-0676ba61.pth
# wide_resnet50_2-95faca4d.pth
# dinov2_vit_base_14/
# ...
```

---

## 5. 데이터셋 준비

### 5.1. MVTec AD 데이터셋 다운로드

MVTec AD는 가장 널리 사용되는 산업 이상 감지 벤치마크입니다.

```bash
cd ~/workspace/anomaly_detection/datasets

# Download MVTec AD
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz

# Extract
tar -xf mvtec_anomaly_detection.tar.xz
mv mvtec_anomaly_detection mvtec

# Verify structure
ls mvtec/
# Expected: bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
```

### 5.2. VisA 데이터셋 다운로드

```bash
cd ~/workspace/anomaly_detection/datasets

# Download from official source
# https://github.com/amazon-science/spot-diff

# After download, extract to:
# datasets/visa/
```

### 5.3. BTAD 데이터셋 다운로드

```bash
cd ~/workspace/anomaly_detection/datasets

# Download from official source
# http://avires.dimi.uniud.it/papers/btad/btad.zip

wget http://avires.dimi.uniud.it/papers/btad/btad.zip
unzip btad.zip
mv BTech_Dataset_transformed btad
```

### 5.4. 보조 데이터셋

#### DTD (DRAEM용)

```bash
cd ~/workspace/anomaly_detection/datasets

# Download Describable Textures Dataset
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz
mv dtd/images dtd_images
rm -rf dtd
mv dtd_images dtd
```

#### Imagenette2 (EfficientAD용)

```bash
cd ~/workspace/anomaly_detection/datasets

# Download Imagenette2
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xzf imagenette2.tgz
```

### 5.5. 사용자 정의 데이터셋

사용자 정의 데이터셋을 준비하는 방법은 [Datasets 문서](04-datasets.md)를 참조하시기 바랍니다.

기본 구조:

```bash
cd ~/workspace/anomaly_detection/datasets
mkdir -p custom/my_dataset/data_rgb/normal
mkdir -p custom/my_dataset/data_rgb/defect_type1

# Copy your images following the naming convention:
# {category} {freq} {dimming}.png
```

---

## 6. 설정 확인

### 6.1. 프로젝트 설정 파일 수정

`project/main.py` 파일을 열어 경로를 수정합니다:

```python
from train import train, train_models, set_globals

if __name__ == "__main__":
    set_globals(
        dataset_dir="/home/user/workspace/anomaly_detection/datasets",
        backbone_dir="/home/user/workspace/anomaly_detection/backbones",
        output_dir="/home/user/workspace/anomaly_detection/outputs",
        seed=42,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        show_globals=True
    )
```

**경로를 본인의 환경에 맞게 수정하세요:**
- Linux/macOS: `/home/username/workspace/anomaly_detection/...`
- Windows: `C:/workspace/anomaly_detection/...` 또는 `C:\\workspace\\anomaly_detection\\...`

### 6.2. 설정 확인 스크립트

다음 Python 스크립트로 설정을 확인합니다:

```python
# test_setup.py
import os
import torch
from train import set_globals, print_globals

# Set paths
set_globals(
    dataset_dir="/path/to/datasets",
    backbone_dir="/path/to/backbones",
    output_dir="/path/to/outputs",
    show_globals=True
)

# Check directories
print("\n=== Directory Check ===")
from train import get_globals
config = get_globals()

for key, path in config.items():
    if 'dir' in key:
        exists = os.path.isdir(path)
        print(f"{key}: {path} - {'✓ EXISTS' if exists else '✗ NOT FOUND'}")

# Check CUDA
print("\n=== CUDA Check ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check datasets
print("\n=== Dataset Check ===")
dataset_dir = config['dataset_dir']
datasets = ['mvtec', 'visa', 'btad', 'custom']
for ds in datasets:
    ds_path = os.path.join(dataset_dir, ds)
    exists = os.path.isdir(ds_path)
    print(f"{ds}: {'✓ EXISTS' if exists else '✗ NOT FOUND'}")
    if exists and ds == 'mvtec':
        categories = [d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))]
        print(f"  Categories: {len(categories)}")

# Check backbones
print("\n=== Backbone Check ===")
backbone_dir = config['backbone_dir']
required_backbones = [
    'resnet50-0676ba61.pth',
    'wide_resnet50_2-95faca4d.pth'
]
for bb in required_backbones:
    bb_path = os.path.join(backbone_dir, bb)
    exists = os.path.isfile(bb_path)
    print(f"{bb}: {'✓ EXISTS' if exists else '✗ NOT FOUND'}")

print("\n=== Setup Complete ===")
```

실행:

```bash
cd ~/workspace/anomaly_detection/project
python test_setup.py
```

---

## 7. 첫 번째 실험 실행

### 7.1. 간단한 테스트 실행

PaDiM은 학습이 필요 없는(1 epoch) 메모리 기반 모델로 빠른 테스트에 적합합니다:

```python
# quick_test.py
from train import train, set_globals

set_globals(
    dataset_dir="/path/to/datasets",
    backbone_dir="/path/to/backbones",
    output_dir="/path/to/outputs",
    show_globals=True
)

# Run quick test with PaDiM (no training required)
train("mvtec", "bottle", "padim", num_epochs=1)
```

실행:

```bash
python quick_test.py
```

**예상 소요 시간:** 2-5분

### 7.2. 전체 학습 실행

```python
# first_experiment.py
from train import train, set_globals

set_globals(
    dataset_dir="/path/to/datasets",
    backbone_dir="/path/to/backbones",
    output_dir="/path/to/outputs",
    show_globals=True
)

# Train STFPM model
train("mvtec", "bottle", "stfpm", num_epochs=50)
```

실행:

```bash
python first_experiment.py
```

**예상 소요 시간:** 30-60분 (GPU 사양에 따라 다름)

### 7.3. 결과 확인

학습이 완료되면 다음 위치에 결과가 저장됩니다:

```
outputs/
└── mvtec/
    └── bottle/
        └── padim/  (or stfpm/)
            ├── model_mvtec_bottle_padim_epochs-1.pth
            ├── results_mvtec_bottle_padim_thresholds.txt
            ├── histogram_mvtec_bottle_padim_scores.png
            └── image_mvtec_bottle_padim_*.png
```

**주요 결과 파일:**
- `results_*_thresholds.txt`: AUROC, AUPR 등 평가 지표
- `histogram_*_scores.png`: 점수 분포 시각화
- `image_*_*.png`: 이상 맵 시각화

---

## 8. 오프라인 환경 설정

외부 인터넷이 차단된 환경에서는 다음과 같이 준비합니다.

### 8.1. 온라인 환경에서 준비

```bash
# 1. Download all Python packages
pip download torch torchvision timm einops FrEIA omegaconf safetensors torchmetrics kornia scipy -d packages/

# 2. Download all backbones (Section 4 참조)

# 3. Download all datasets (Section 5 참조)

# 4. Create archive
tar -czf anomaly_detection_offline.tar.gz packages/ backbones/ datasets/ project/
```

### 8.2. 오프라인 환경에서 설치

```bash
# 1. Extract archive
tar -xzf anomaly_detection_offline.tar.gz

# 2. Install packages from local directory
cd packages/
pip install --no-index --find-links . torch torchvision timm einops FrEIA omegaconf safetensors torchmetrics kornia scipy

# 3. Set up directories
cd ..
mv backbones ~/workspace/anomaly_detection/
mv datasets ~/workspace/anomaly_detection/
mv project ~/workspace/anomaly_detection/

# 4. Configure paths in main.py
```

---

## 9. 문제 해결

### 9.1. CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결 방법:**

1. 배치 크기 감소:
```python
train("mvtec", "bottle", "stfpm", num_epochs=50, batch_size=8)  # Default: 16
```

2. 이미지 크기 감소:
```python
train("mvtec", "bottle", "stfpm", num_epochs=50, img_size=224)  # Default: 256
```

3. 더 작은 모델 사용:
```python
train("mvtec", "bottle", "dinomaly-small-224")  # Instead of dinomaly-large
```

### 9.2. 백본 파일을 찾을 수 없음

**증상:**
```
> resnet50 weight not found in /path/to/backbones/resnet50.pth
```

**해결 방법:**

1. 파일 존재 확인:
```bash
ls -l /path/to/backbones/resnet50*.pth
```

2. 파일명 확인:
```bash
# 정확한 파일명: resnet50-0676ba61.pth
```

3. backbone_dir 경로 확인:
```python
from models.components.backbone import get_backbone_dir
print(get_backbone_dir())
```

### 9.3. DataLoader 오류

**증상:**
```
FileNotFoundError: Dataset not found at /path/to/datasets/mvtec/bottle
```

**해결 방법:**

1. 데이터셋 구조 확인:
```bash
ls /path/to/datasets/mvtec/bottle/
# Expected: train/ test/ ground_truth/
```

2. dataset_dir 경로 확인:
```python
from dataloader import DATASET_DIR
print(DATASET_DIR)
```

### 9.4. Import 오류

**증상:**
```
ModuleNotFoundError: No module named 'timm'
```

**해결 방법:**

1. 가상환경 활성화 확인:
```bash
which python
# Should point to virtual environment
```

2. 라이브러리 재설치:
```bash
pip install timm einops FrEIA
```

### 9.5. Permission 오류

**증상:**
```
PermissionError: [Errno 13] Permission denied: '/path/to/outputs'
```

**해결 방법:**

1. 디렉토리 권한 확인:
```bash
ls -ld /path/to/outputs
```

2. 권한 부여:
```bash
chmod 755 /path/to/outputs
```

### 9.6. 성능 문제

**증상:** 학습이 매우 느림

**해결 방법:**

1. num_workers 조정:
```python
set_globals(num_workers=4)  # Reduce if system has limited cores
```

2. pin_memory 비활성화:
```python
set_globals(pin_memory=False)  # If system RAM is limited
```

3. persistent_workers 비활성화:
```python
set_globals(persistent_workers=False)
```

---

**다음 단계:** 설치가 완료되었으면 [Training 가이드](05-training.md)를 참조하여 모델 학습을 시작하시기 바랍니다.