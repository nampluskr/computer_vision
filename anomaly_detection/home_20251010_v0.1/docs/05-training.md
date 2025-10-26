# Training - 학습 설정 및 팁

## 목차

1. [개요](#1-개요)
2. [기본 학습](#2-기본-학습)
3. [고급 학습 설정](#3-고급-학습-설정)
4. [하이퍼파라미터 튜닝](#4-하이퍼파라미터-튜닝)
5. [모델별 학습 가이드](#5-모델별-학습-가이드)
6. [학습 모니터링](#6-학습-모니터링)
7. [Early Stopping](#7-early-stopping)
8. [메모리 최적화](#8-메모리-최적화)
9. [멀티 GPU 학습](#9-멀티-gpu-학습)
10. [실험 관리](#10-실험-관리)
11. [문제 해결](#11-문제-해결)

---

## 1. 개요

본 프레임워크는 통합된 학습 인터페이스를 통해 모든 모델의 학습을 간단하게 수행할 수 있습니다.

### 1.1. 학습 워크플로우

```
설정 → 데이터 로드 → 모델 생성 → 학습 → 검증 → 결과 저장
  ↓         ↓           ↓         ↓       ↓         ↓
Global   DataLoader   Registry  Trainer  Metrics  Outputs
```

### 1.2. 주요 함수

```python
# Single model training
train(dataset_type, category, model_type, **kwargs)

# Multi-model training
train_models(dataset_type, categories, models, **kwargs)

# Global configuration
set_globals(dataset_dir, backbone_dir, output_dir, **kwargs)
```

---

## 2. 기본 학습

### 2.1. 최소 설정

```python
from train import train, set_globals

# Set paths
set_globals(
    dataset_dir="/path/to/datasets",
    backbone_dir="/path/to/backbones",
    output_dir="/path/to/outputs"
)

# Train single model
train("mvtec", "bottle", "stfpm", num_epochs=50)
```

### 2.2. 전체 설정

```python
from train import train, set_globals

# Configure all parameters
set_globals(
    dataset_dir="/path/to/datasets",
    backbone_dir="/path/to/backbones",
    output_dir="/path/to/outputs",
    seed=42,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    show_globals=True
)

# Train with custom parameters
train(
    dataset_type="mvtec",
    category="bottle",
    model_type="stfpm",
    num_epochs=50,
    batch_size=16,
    img_size=256,
    normalize=True
)
```

### 2.3. 학습 파라미터

| 파라미터 | 타입 | 설명 | 기본값 |
|---------|------|------|--------|
| dataset_type | str or list | 데이터셋 이름 | 필수 |
| category | str or list | 카테고리 이름 | 필수 |
| model_type | str | 모델 이름 | 필수 |
| num_epochs | int | 학습 에포크 수 | 모델별 기본값 |
| batch_size | int | 배치 크기 | 모델별 기본값 |
| img_size | int | 입력 이미지 크기 | 모델별 기본값 |
| normalize | bool | ImageNet 정규화 | 모델별 기본값 |

---

## 3. 고급 학습 설정

### 3.1. 다중 카테고리 학습

```python
# Multiple categories in single dataset
train("mvtec", ["bottle", "wood", "grid"], "stfpm", num_epochs=50)

# All categories
train("mvtec", "all", "padim", num_epochs=1)

# Multiple datasets, multiple categories
train(
    ["mvtec", "visa"],
    ["bottle", "candle"],
    "stfpm",
    num_epochs=50
)
```

### 3.2. 배치 학습

```python
from train import train_models

# Train multiple models on multiple categories
train_models(
    dataset_type="mvtec",
    categories=["bottle", "wood", "grid"],
    models=["padim", "patchcore", "stfpm"],
    num_epochs=50
)
```

### 3.3. 학습 재개

```python
# Load checkpoint and continue training
train(
    "mvtec",
    "bottle",
    "stfpm",
    num_epochs=100,
    resume_from="/path/to/checkpoint.pth"
)
```

**참고:** 현재 구현에서는 `resume_from` 파라미터를 직접 지원하지 않습니다. 학습을 재개하려면 `trainer.load_model()`을 사용해야 합니다.

### 3.4. 커스텀 출력 디렉토리

```python
# Change output directory for specific experiment
set_globals(output_dir="/path/to/experiment_v2")

train("mvtec", "bottle", "stfpm", num_epochs=50)
```

---

## 4. 하이퍼파라미터 튜닝

### 4.1. 배치 크기 튜닝

#### 원칙

- **큰 배치**: 안정적 학습, 많은 메모리 필요
- **작은 배치**: 불안정하지만 일반화 성능 향상 가능

#### 권장 설정

```python
# Memory-based models (no training)
train("mvtec", "bottle", "padim", batch_size=4)

# Knowledge distillation models
train("mvtec", "bottle", "stfpm", batch_size=16)

# Flow models (large)
train("mvtec", "bottle", "fastflow-cait", batch_size=2)

# Foundation models
train("mvtec", "bottle", "dinomaly-base-224", batch_size=16)
```

#### GPU 메모리별 권장

| GPU VRAM | 권장 배치 크기 | 모델 예시 |
|----------|---------------|----------|
| 8GB | 4-8 | STFPM, PaDiM |
| 10GB | 8-16 | STFPM, Dinomaly-Small |
| 16GB | 16-32 | Dinomaly-Base, FastFlow |
| 24GB+ | 32+ | Dinomaly-Large, All models |

### 4.2. 이미지 크기 튜닝

#### 원칙

- **큰 이미지**: 세밀한 이상 감지, 많은 메모리 필요
- **작은 이미지**: 빠른 학습, 큰 이상만 감지

#### 권장 설정

```python
# Standard size (most models)
train("mvtec", "bottle", "stfpm", img_size=256)

# Small objects or fine defects
train("mvtec", "bottle", "stfpm", img_size=512)

# Large objects or coarse defects
train("mvtec", "bottle", "stfpm", img_size=224)

# Dinomaly models (fixed sizes)
train("mvtec", "bottle", "dinomaly-base-224", img_size=224)  # Fixed
train("mvtec", "bottle", "dinomaly-base-392", img_size=392)  # Fixed
train("mvtec", "bottle", "dinomaly-base-448", img_size=448)  # Fixed
```

### 4.3. 에포크 수 튜닝

#### 모델별 권장 에포크

| 모델 카테고리 | 권장 에포크 | 예시 |
|--------------|-----------|------|
| Memory-based | 1 | PaDiM, PatchCore |
| Knowledge Distillation | 20-100 | STFPM(50), EfficientAD(20) |
| Normalizing Flow | 250-500 | FastFlow(500), CFlow(250) |
| Reconstruction | 20-100 | Autoencoder(50), DRAEM(10) |
| Foundation | 10-20 | Dinomaly(15), UniNet(20) |

#### 데이터셋 크기별 조정

```python
# Small dataset (< 100 images)
train("mvtec", "toothbrush", "stfpm", num_epochs=30)  # Reduce from 50

# Medium dataset (100-500 images)
train("mvtec", "bottle", "stfpm", num_epochs=50)  # Default

# Large dataset (> 500 images)
train("custom", "my_product", "stfpm", num_epochs=100)  # Increase
```

### 4.4. 학습률 튜닝

학습률은 `registry.py`에서 모델별로 정의되어 있습니다. 변경하려면:

#### 방법 1: Registry 수정

```python
# registry.py
ModelRegistry.register("stfpm", "models.model_stfpm.STFPMTrainer",
    dict(
        backbone="resnet50",
        layers=["layer1", "layer2", "layer3"],
        learning_rate=0.001  # Add custom parameter
    ),
    dict(num_epochs=50, batch_size=16, normalize=True, img_size=256)
)
```

#### 방법 2: Trainer 수정

```python
# models/model_stfpm.py
class STFPMTrainer(BaseTrainer):
    def __init__(self, learning_rate=0.0001, **kwargs):
        # ...
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate  # Use custom learning rate
        )
```

---

## 5. 모델별 학습 가이드

### 5.1. Memory-Based Models

#### PaDiM

```python
# Quick evaluation (no training needed)
train("mvtec", "bottle", "padim", num_epochs=1)

# Adjust batch size for speed
train("mvtec", "bottle", "padim", num_epochs=1, batch_size=8)
```

**학습 팁:**
- 학습 불필요 (1 epoch만 실행)
- 배치 크기는 속도에만 영향
- 메모리 사용량이 높으므로 대규모 이미지 주의

#### PatchCore

```python
# Standard training
train("mvtec", "bottle", "patchcore", num_epochs=1)

# Adjust coreset percentage (modify registry.py)
# Default: 10% of training features
```

**학습 팁:**
- Coreset 샘플링 시간 필요
- 배치 크기 증가로 속도 향상 가능
- 최고 성능을 위해 충분한 학습 데이터 필요

---

### 5.2. Knowledge Distillation Models

#### STFPM

```python
# Standard training
train("mvtec", "bottle", "stfpm", num_epochs=50)

# Fine-tuning with more epochs
train("mvtec", "bottle", "stfpm", num_epochs=100)

# Smaller batch for limited GPU
train("mvtec", "bottle", "stfpm", num_epochs=50, batch_size=8)
```

**학습 팁:**
- 50 epochs로 충분한 성능
- Teacher 네트워크는 frozen (학습 안됨)
- 안정적인 학습 곡선

#### EfficientAD

```python
# Standard training
train("mvtec", "bottle", "efficientad-small", num_epochs=20)

# Medium model for better accuracy
train("mvtec", "bottle", "efficientad-medium", num_epochs=20)

# IMPORTANT: normalize=False required
train("mvtec", "bottle", "efficientad-small", num_epochs=20, normalize=False)
```

**학습 팁:**
- **반드시 `normalize=False` 사용**
- `batch_size=1` 권장
- Imagenette2 데이터셋 필요
- Teacher 네트워크 사전학습 필요

**Teacher 사전학습 (필요시):**

```python
# EfficientAD는 자동으로 teacher를 학습하거나 로드합니다
# 수동 사전학습이 필요한 경우 model_efficientad.py 참조
```

---

### 5.3. Normalizing Flow Models

#### FastFlow

```python
# ResNet-50 backbone
train("mvtec", "bottle", "fastflow-resnet50", num_epochs=500)

# CaiT backbone (higher accuracy, slower)
train("mvtec", "bottle", "fastflow-cait", num_epochs=500, batch_size=2)

# DeiT backbone (balanced)
train("mvtec", "bottle", "fastflow-deit", num_epochs=500, batch_size=2)
```

**학습 팁:**
- 많은 에포크 필요 (500)
- 긴 학습 시간 (수 시간~수 일)
- Transformer 백본은 배치 크기 2-4 권장
- 학습 후 빠른 추론

#### CFlow

```python
# ResNet-18 (faster)
train("mvtec", "bottle", "cflow-resnet18", num_epochs=250)

# ResNet-50 (better accuracy)
train("mvtec", "bottle", "cflow-resnet50", num_epochs=250)
```

**학습 팁:**
- 250 epochs 권장
- 위치 인코딩 사용
- 안정적인 학습

---

### 5.4. Reconstruction Models

#### Autoencoder

```python
# Basic training
train("mvtec", "bottle", "autoencoder", num_epochs=50, normalize=False)

# Increase capacity
train("mvtec", "bottle", "autoencoder", num_epochs=100, normalize=False)

# Adjust latent dimension (modify registry.py)
ModelRegistry.register("autoencoder", ...,
    dict(latent_dim=256),  # Default: 128
    ...
)
```

**학습 팁:**
- **반드시 `normalize=False` 사용**
- Identity mapping 방지를 위해 충분히 작은 latent_dim
- Early stopping 사용 권장
- Overfitting 주의

#### DRAEM

```python
# Standard training (short)
train("mvtec", "bottle", "draem", num_epochs=10, normalize=False)

# With SSPCAB module
train("mvtec", "bottle", "draem", num_epochs=10, normalize=False)
```

**학습 팁:**
- **DTD 데이터셋 필수**
- 10 epochs로 충분
- Perlin noise로 인공 이상 생성
- 빠른 학습

---

### 5.5. Foundation Models

#### Dinomaly

```python
# Small model (fast training)
train("mvtec", "bottle", "dinomaly-small-224", num_epochs=15)

# Base model (recommended)
train("mvtec", "bottle", "dinomaly-base-224", num_epochs=15)

# Large model (best accuracy)
train("mvtec", "bottle", "dinomaly-large-224", num_epochs=15)

# Higher resolution
train("mvtec", "bottle", "dinomaly-base-392", num_epochs=10)
train("mvtec", "bottle", "dinomaly-base-448", num_epochs=10)
```

**학습 팁:**
- DINOv2 인코더는 frozen (학습 안됨)
- 디코더만 학습 (빠름)
- 높은 해상도는 에포크 감소
- 최신 성능

**모델 크기 선택:**

| 모델 | 학습 시간 | 메모리 | 정확도 | 용도 |
|------|----------|--------|--------|------|
| Small | 빠름 | 낮음 | 높음 | 빠른 실험 |
| Base | 중간 | 중간 | 매우 높음 | 권장 |
| Large | 느림 | 높음 | 최고 | 최고 성능 |

---

## 6. 학습 모니터링

### 6.1. 콘솔 출력

학습 중 다음 정보가 출력됩니다:

```
=== Training Configuration ===
  dataset_dir: /path/to/datasets
  backbone_dir: /path/to/backbones
  output_dir: /path/to/outputs
  seed: 42
  ...

> Start training...

[  1/50] loss=0.123 | auroc=0.856, aupr=0.789 (12.3s)
[  2/50] loss=0.098 | auroc=0.892, aupr=0.834 (11.8s)
...
[  5/50] loss=0.067 | auroc=0.934, aupr=0.891 (11.5s)
 > th=0.234, acc=0.912, prec=0.887, recall=0.945, f1=0.915 | TP=89, FN=5, TN=82, FP=11

...

> Training finished... in 00:10:23

> Model weights saved to: /path/to/outputs/mvtec/bottle/stfpm/model_mvtec_bottle_stfpm_epochs-50.pth
```

### 6.2. 학습 곡선 저장

현재 구현에서는 학습 곡선이 자동으로 저장되지 않습니다. 직접 구현하려면:

```python
# train.py에 추가
import matplotlib.pyplot as plt

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    if 'loss' in history:
        axes[0].plot(history['loss'])
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
    
    # Metrics plot
    if 'auroc' in history:
        axes[1].plot(history['auroc'], label='AUROC')
    if 'aupr' in history:
        axes[1].plot(history['aupr'], label='AUPR')
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Use in train() function
history = trainer.fit(train_loader, num_epochs, valid_loader=test_loader, weight_path=weight_path)
plot_training_history(history, os.path.join(result_dir, f"history_{desc}.png"))
```

### 6.3. TensorBoard 통합 (선택)

```python
# Install tensorboard
pip install tensorboard

# Add to trainer.py
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, ..., log_dir=None):
        self.writer = SummaryWriter(log_dir) if log_dir else None
    
    def on_train_end(self, train_results):
        if self.writer:
            for name, value in train_results.items():
                self.writer.add_scalar(f'train/{name}', value, self.epoch)
    
    def on_validation_end(self, valid_results, scores, labels):
        if self.writer:
            for name, value in valid_results.items():
                self.writer.add_scalar(f'val/{name}', value, self.epoch)

# Run tensorboard
tensorboard --logdir=/path/to/outputs/logs
```

---

## 7. Early Stopping

### 7.1. 기본 사용

Early stopping은 `trainer.py`의 `EarlyStopper` 클래스로 구현되어 있습니다.

```python
# registry.py에서 설정
from models.components.trainer import EarlyStopper

ModelRegistry.register("stfpm", "models.model_stfpm.STFPMTrainer",
    dict(
        backbone="resnet50",
        layers=["layer1", "layer2", "layer3"],
        early_stopper_loss=EarlyStopper(patience=10, mode='min'),
        early_stopper_auroc=EarlyStopper(patience=10, mode='max')
    ),
    dict(num_epochs=50, batch_size=16, normalize=True, img_size=256)
)
```

### 7.2. EarlyStopper 파라미터

```python
EarlyStopper(
    patience=10,        # Number of epochs to wait
    min_delta=1e-3,     # Minimum change to qualify as improvement
    mode='max',         # 'max' for metrics (AUROC), 'min' for loss
    target_value=None   # Stop when reaching this value
)
```

### 7.3. 예시

#### Loss 기반

```python
# Stop if loss doesn't improve for 10 epochs
early_stopper_loss = EarlyStopper(patience=10, min_delta=0.001, mode='min')
```

#### AUROC 기반

```python
# Stop if AUROC doesn't improve for 10 epochs
early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max')
```

#### Target 값 도달

```python
# Stop when AUROC reaches 0.99
early_stopper_auroc = EarlyStopper(target_value=0.99, mode='max')
```

---

## 8. 메모리 최적화

### 8.1. GPU 메모리 부족 해결

#### 방법 1: 배치 크기 감소

```python
# Default
train("mvtec", "bottle", "stfpm", batch_size=16)

# Reduced
train("mvtec", "bottle", "stfpm", batch_size=8)
train("mvtec", "bottle", "stfpm", batch_size=4)
```

#### 방법 2: 이미지 크기 감소

```python
# Default
train("mvtec", "bottle", "stfpm", img_size=256)

# Reduced
train("mvtec", "bottle", "stfpm", img_size=224)
train("mvtec", "bottle", "stfpm", img_size=192)
```

#### 방법 3: 더 작은 모델 사용

```python
# Large model
train("mvtec", "bottle", "dinomaly-large-224")

# Smaller alternatives
train("mvtec", "bottle", "dinomaly-base-224")
train("mvtec", "bottle", "dinomaly-small-224")
```

#### 방법 4: Gradient Accumulation (고급)

```python
# Modify trainer.py
class BaseTrainer:
    def train_step(self, batch):
        # Accumulate gradients over multiple batches
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
```

### 8.2. 메모리 정리

연속 학습 시 메모리 누수 방지:

```python
from train import clear_memory

# Train multiple models
for model in ["padim", "patchcore", "stfpm"]:
    train("mvtec", "bottle", model)
    clear_memory()  # Clean up GPU memory
```

### 8.3. DataLoader 최적화

```python
# Reduce memory usage
set_globals(
    num_workers=4,              # Reduce from 8
    pin_memory=False,           # Disable if RAM is limited
    persistent_workers=False    # Don't keep workers alive
)
```

---

## 9. 멀티 GPU 학습

현재 구현은 단일 GPU를 지원합니다. 멀티 GPU 지원을 위해서는 다음과 같이 수정이 필요합니다:

### 9.1. DataParallel 사용

```python
# Modify trainer.py
class BaseTrainer:
    def __init__(self, model, ..., device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        self.model = model.to(self.device)
```

### 9.2. DistributedDataParallel (권장)

더 효율적인 멀티 GPU 학습을 위해 DDP 사용:

```python
# train_ddp.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, dataset_type, category, model_type):
    setup(rank, world_size)
    
    # Create model and move to GPU
    trainer = get_trainer(model_type, ...)
    trainer.model = DDP(trainer.model, device_ids=[rank])
    
    # Training logic
    # ...
    
    cleanup()

# Run with torchrun
# torchrun --nproc_per_node=4 train_ddp.py
```

---

## 10. 실험 관리

### 10.1. 실험 명명 규칙

일관된 명명 규칙 사용:

```
{dataset}_{category}_{model}_{variant}_{date}
```

예시:
```
mvtec_bottle_stfpm_baseline_20250112
mvtec_bottle_stfpm_tuned_20250113
custom_product_a_dinomaly_base_20250114
```

### 10.2. 실험 스크립트

```python
# experiments/exp_001_baseline.py
from train import train, set_globals
from datetime import datetime

# Experiment configuration
EXPERIMENT_NAME = "exp_001_baseline"
DATE = datetime.now().strftime("%Y%m%d")

# Set output directory
set_globals(
    output_dir=f"/path/to/experiments/{EXPERIMENT_NAME}_{DATE}"
)

# Run experiments
models = ["padim", "patchcore", "stfpm"]
categories = ["bottle", "wood", "grid"]

for model in models:
    for category in categories:
        print(f"\n{'='*70}")
        print(f"Training: {model} on {category}")
        print(f"{'='*70}\n")
        
        train("mvtec", category, model)
```

### 10.3. 결과 비교

```python
# compare_results.py
import os
import pandas as pd

def parse_results(result_file):
    """Parse results from text file"""
    with open(result_file, 'r') as f:
        content = f.read()
    
    # Extract AUROC and AUPR
    auroc = float(content.split("AUROC")[1].split(":")[1].split("(")[0].strip())
    aupr = float(content.split("AUPR")[1].split(":")[1].split("(")[0].strip())
    
    return {"AUROC": auroc, "AUPR": aupr}

def compare_experiments(output_dir, models, categories):
    """Compare results across models and categories"""
    results = []
    
    for model in models:
        for category in categories:
            result_file = os.path.join(
                output_dir,
                "mvtec",
                category,
                model,
                f"results_mvtec_{category}_{model}_thresholds.txt"
            )
            
            if os.path.exists(result_file):
                metrics = parse_results(result_file)
                results.append({
                    "Model": model,
                    "Category": category,
                    **metrics
                })
    
    df = pd.DataFrame(results)
    
    # Pivot table for easy comparison
    auroc_table = df.pivot(index="Category", columns="Model", values="AUROC")
    aupr_table = df.pivot(index="Category", columns="Model", values="AUPR")
    
    print("\n=== AUROC Comparison ===")
    print(auroc_table)
    
    print("\n=== AUPR Comparison ===")
    print(aupr_table)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)
    auroc_table.to_csv(os.path.join(output_dir, "auroc_comparison.csv"))
    aupr_table.to_csv(os.path.join(output_dir, "aupr_comparison.csv"))

# Usage
compare_experiments(
    output_dir="/path/to/outputs",
    models=["padim", "patchcore", "stfpm"],
    categories=["bottle", "wood", "grid"]
)
```

---

## 11. 문제 해결

### 11.1. 학습이 시작되지 않음

**증상:**
```
프로그램이 멈춰있음, 아무 출력 없음
```

**해결:**

1. DataLoader 확인:
```python
# Test dataloader separately
from dataloader import get_dataloaders

train_loader, test_loader = get_dataloaders("mvtec", "bottle", 256, 4)
batch = next(iter(train_loader))
print(f"Batch loaded: {batch['image'].shape}")
```

2. num_workers 조정:
```python
set_globals(num_workers=0)  # Single process
```

3. persistent_workers 비활성화:
```python
set_globals(persistent_workers=False)
```

### 11.2. Loss가 NaN

**증상:**
```
[  5/50] loss=nan | auroc=0.500, aupr=0.500
```

**해결:**

1. 학습률 감소:
```python
# Modify model trainer
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)  # From 1e-3
```

2. Gradient clipping 추가:
```python
# In train_step
loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
self.optimizer.step()
```

3. 정규화 확인:
```python
# Ensure correct normalization setting
train("mvtec", "bottle", "stfpm", normalize=True)  # Feature-based
train("mvtec", "bottle", "autoencoder", normalize=False)  # Reconstruction
```

### 11.3. 수렴하지 않음

**증상:**
```
Loss가 감소하지 않거나 매우 느리게 감소
```

**해결:**

1. 에포크 증가:
```python
train("mvtec", "bottle", "stfpm", num_epochs=100)  # From 50
```

2. 배치 크기 조정:
```python
train("mvtec", "bottle", "stfpm", batch_size=32)  # From 16
```

3. 학습률 증가 (주의):
```python
# Modify trainer
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)  # From 1e-4
```

4. 모델 초기화 확인:
```python
# Check if model is properly initialized
print(f"Total parameters: {sum(p.numel() for p in trainer.model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}")
```

### 11.4. Overfitting

**증상:**
```
Train loss가 감소하지만 validation metrics가 감소하지 않거나 악화됨
```

**해결:**

1. Early stopping 사용:
```python
# Add to model config
early_stopper_auroc = EarlyStopper(patience=5, mode='max')
```

2. 에포크 감소:
```python
train("mvtec", "bottle", "stfpm", num_epochs=30)  # From 50
```

3. 데이터 증강 (주의: 이상 감지에서는 제한적):
```python
# Modify dataloader.py with careful augmentation
```

### 11.5. 메모리 누수

**증상:**
```
학습 중 메모리 사용량이 계속 증가
```

**해결:**

1. 메모리 정리 함수 사용:
```python
from train import clear_memory

for epoch in range(num_epochs):
    # Training
    if epoch % 10 == 0:
        clear_memory()
```

2. Gradient 축적 해제:
```python
# Ensure gradients are cleared
self.optimizer.zero_grad()
```

3. 불필요한 변수 삭제:
```python
# After using large tensors
del large_tensor
torch.cuda.empty_cache()
```

### 11.6. 느린 학습 속도

**증상:**
```
학습이 예상보다 훨씬 느림
```

**해결:**

1. num_workers 증가:
```python
set_globals(num_workers=8)  # Match CPU cores
```

2. pin_memory 활성화:
```python
set_globals(pin_memory=True)
```

3. 배치 크기 증가:
```python
train("mvtec", "bottle", "stfpm", batch_size=32)  # From 16
```

4. Mixed precision training (고급):
```python
# Add to trainer.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = self.train_step(batch)

scaler.scale(loss).backward()
scaler.step(self.optimizer)
scaler.update()
```

---

**다음 문서:** [Inference](06-inference.md) - 배포 및 추론