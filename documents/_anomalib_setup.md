# Anomalib 환경 구축 매뉴얼

**WSL2 + Anaconda + Anomalib + CUDA 환경 구축하기**

이미지 이상 감지(Anomaly Detection)를 위한 최신 라이브러리 Anomalib 설치 가이드입니다.

---

## 목차
1. [Anomalib 소개](#1-anomalib-소개)
2. [환경 생성 및 설치](#2-환경-생성-및-설치)
3. [설치 확인 및 테스트](#3-설치-확인-및-테스트)
4. [MVTec AD 데이터셋 설정](#4-mvtec-ad-데이터셋-설정)
5. [간단한 모델 학습 예시](#5-간단한-모델-학습-예시)
6. [문제 해결](#6-문제-해결)

---

## 1. Anomalib 소개

### Anomalib이란?
- Intel에서 개발한 **이상 감지(Anomaly Detection) 전문 라이브러리**
- PyTorch Lightning 기반
- 14+ 최신 이상 감지 알고리즘 제공
- 산업용 품질 검사에 특화

### 지원 모델
- **Reconstruction-based**: AutoEncoder, VAE
- **Embedding-based**: PatchCore, PaDiM, CFlow
- **One-class**: STFPM, DFM, FastFlow
- **Transformer-based**: EfficientAD, WinCLIP
- 그 외: Reverse Distillation, GANomaly 등

---

## 2. 환경 생성 및 설치

### 2-1. Anomalib 환경 생성

```bash
# 환경 비활성화
conda deactivate

# Python 3.10으로 환경 생성
conda create -n anomalib_env python=3.10 -y

# 환경 활성화
conda activate anomalib_env
```

### 2-2. PyTorch 설치 (필수)

Anomalib은 PyTorch를 기반으로 합니다.

```bash
# NumPy 먼저 설치 (MKL 의존성)
conda install numpy -y

# PyTorch 설치 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2-3. Anomalib 설치

**방법 1: pip 설치 (권장)**

```bash
# 최신 stable 버전
pip install anomalib

# 또는 개발 버전 (최신 기능)
# pip install git+https://github.com/openvinotoolkit/anomalib.git
```

**방법 2: 소스 코드에서 설치 (개발자용)**

```bash
# 홈 디렉토리로 이동
cd ~

# Git clone
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib

# editable 모드로 설치
pip install -e .
```

### 2-4. 추가 필수 패키지 설치

```bash
# Lightning (PyTorch Lightning)
pip install lightning

# 이미지 처리
pip install opencv-python pillow albumentations

# 시각화
pip install matplotlib seaborn plotly

# 데이터 처리
conda install -y pandas scipy

# 유틸리티
pip install tqdm rich

# 평가 메트릭
pip install torchmetrics scikit-learn

# 텐서보드
pip install tensorboard

# 추가 도구
pip install omegaconf hydra-core wandb
```

### 2-5. OpenVINO (선택사항 - 추론 최적화)

```bash
# OpenVINO를 사용한 추론 가속 (선택)
pip install openvino openvino-dev
```

---

## 3. 설치 확인 및 테스트

### 3-1. 기본 Import 테스트

```bash
python << EOF
import sys
print("="*50)
print("Anomalib Installation Check")
print("="*50)

# 1. PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Lightning
import lightning as L
print(f"Lightning version: {L.__version__}")

# 3. Anomalib
import anomalib
print(f"Anomalib version: {anomalib.__version__}")

# 4. 주요 모듈 확인
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

print("\n✓ All imports successful!")
print("="*50)
EOF
```

**예상 출력**:
```
==================================================
Anomalib Installation Check
==================================================
PyTorch version: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce GTX 1080 Ti
Lightning version: 2.x.x
Anomalib version: 1.1.x

✓ All imports successful!
==================================================
```

### 3-2. 사용 가능한 모델 확인

```bash
python << EOF
from anomalib import TaskType
from anomalib.models import get_available_models

print("\n=== Available Anomaly Detection Models ===\n")

models = get_available_models()
for i, model in enumerate(models, 1):
    print(f"{i:2d}. {model}")

print(f"\nTotal: {len(models)} models available")
EOF
```

---

## 4. MVTec AD 데이터셋 설정

### 4-1. 데이터셋 디렉토리 구조

MVTec AD 데이터셋은 이상 감지의 표준 벤치마크입니다.

```bash
# 데이터셋 저장 디렉토리 생성
mkdir -p ~/datasets/MVTec
```

**디렉토리 구조**:
```
~/datasets/MVTec/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── broken_large/
│   │   └── broken_small/
│   └── ground_truth/
├── cable/
├── capsule/
└── ... (15개 카테고리)
```

### 4-2. 데이터셋 다운로드

**방법 1: Anomalib 자동 다운로드 (권장)**

```python
from anomalib.data import MVTec

# 자동 다운로드 및 설정
datamodule = MVTec(
    root="~/datasets/MVTec",
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
)

# 데이터 준비
datamodule.prepare_data()
datamodule.setup()
```

**방법 2: 수동 다운로드**

```bash
# MVTec AD 공식 사이트에서 다운로드
# https://www.mvtec.com/company/research/datasets/mvtec-ad

cd ~/datasets/MVTec
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz

# 압축 해제
tar -xf mvtec_anomaly_detection.tar.xz
```

### 4-3. 데이터 확인

```bash
python << EOF
from anomalib.data import MVTec
import matplotlib.pyplot as plt

# 데이터 로드
datamodule = MVTec(
    root="~/datasets/MVTec",
    category="bottle",
)
datamodule.setup()

# 학습 데이터 확인
train_data = datamodule.train_dataloader()
print(f"Train batches: {len(train_data)}")

# 테스트 데이터 확인
test_data = datamodule.test_dataloader()
print(f"Test batches: {len(test_data)}")

print("✓ Dataset loaded successfully!")
EOF
```

---

## 5. 간단한 모델 학습 예시

### 5-1. PatchCore 모델 학습 (빠르고 정확함)

```bash
python << 'EOF'
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch

print("\n=== PatchCore Training Example ===\n")

# 1. 데이터 설정
datamodule = MVTec(
    root="~/datasets/MVTec",
    category="bottle",
    image_size=(224, 224),
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=4,
)

# 2. 모델 생성
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    num_neighbors=9,
)

# 3. 학습 엔진 설정
engine = Engine(
    max_epochs=1,  # PatchCore는 1 epoch만 필요
    accelerator="gpu",
    devices=1,
    logger=False,
    enable_checkpointing=True,
    default_root_dir="./results",
)

# 4. 학습
print("Training PatchCore model...")
engine.fit(model=model, datamodule=datamodule)

# 5. 테스트
print("\nTesting model...")
test_results = engine.test(model=model, datamodule=datamodule)

# 6. 결과 출력
print("\n=== Test Results ===")
for key, value in test_results[0].items():
    if isinstance(value, torch.Tensor):
        value = value.item()
    print(f"{key}: {value:.4f}")

print("\n✓ Training completed!")
EOF
```

### 5-2. 다른 모델 예시

**PaDiM (빠른 학습)**:
```python
from anomalib.models import Padim

model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)
```

**FastFlow (실시간 추론)**:
```python
from anomalib.models import Fastflow

model = Fastflow(
    backbone="resnet18",
    flow_steps=8,
)
```

**EfficientAD (최신 모델)**:
```python
from anomalib.models import EfficientAd

model = EfficientAd(
    teacher_out_channels=384,
    model_size="small",  # "small" or "medium"
)
```

### 5-3. 설정 파일 기반 학습

**config.yaml 생성**:
```bash
cat > ~/config.yaml << 'EOF'
data:
  class_path: anomalib.data.MVTec
  init_args:
    root: ~/datasets/MVTec
    category: bottle
    image_size: [224, 224]
    train_batch_size: 32
    eval_batch_size: 32
    num_workers: 8

model:
  class_path: anomalib.models.Patchcore
  init_args:
    backbone: wide_resnet50_2
    layers: [layer2, layer3]

trainer:
  max_epochs: 1
  accelerator: gpu
  devices: 1
  default_root_dir: ./results
EOF
```

**실행**:
```bash
anomalib fit --config ~/config.yaml
```

### 5-4. CLI로 간단히 학습

```bash
# PatchCore 학습
anomalib train \
  --model Patchcore \
  --data anomalib.data.MVTec \
  --data.root ~/datasets/MVTec \
  --data.category bottle \
  --trainer.max_epochs 1 \
  --trainer.accelerator gpu \
  --trainer.devices 1

# 테스트
anomalib test \
  --model Patchcore \
  --data anomalib.data.MVTec \
  --data.root ~/datasets/MVTec \
  --data.category bottle \
  --ckpt_path results/Patchcore/MVTec/bottle/version_0/checkpoints/last.ckpt
```

---

## 6. 추론 (Inference) 예시

### 6-1. 단일 이미지 추론

```python
from anomalib.deploy import OpenVINOInferencer
from pathlib import Path

# 모델 로드
inferencer = OpenVINOInferencer(
    path="results/Patchcore/MVTec/bottle/version_0/weights/openvino/model.bin",
    metadata="results/Patchcore/MVTec/bottle/version_0/weights/openvino/metadata.json",
)

# 추론
result = inferencer.predict(image="path/to/test/image.png")

print(f"Anomaly score: {result.pred_score}")
print(f"Prediction: {'Anomalous' if result.pred_label else 'Normal'}")
```

### 6-2. 배치 추론

```python
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Patchcore

# 학습된 모델 로드
model = Patchcore.load_from_checkpoint("path/to/checkpoint.ckpt")

# 데이터 준비
datamodule = MVTec(root="~/datasets/MVTec", category="bottle")

# 추론
engine = Engine()
predictions = engine.predict(model=model, datamodule=datamodule)
```

---

## 7. 시각화

### 7-1. 결과 시각화

```python
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
import matplotlib.pyplot as plt

# 모델 및 데이터 설정
datamodule = MVTec(root="~/datasets/MVTec", category="bottle")
model = Patchcore.load_from_checkpoint("path/to/checkpoint.ckpt")

# 추론
engine = Engine()
predictions = engine.predict(model=model, datamodule=datamodule)

# 시각화
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for idx, pred in enumerate(predictions[:10]):
    ax = axes[idx // 5, idx % 5]
    ax.imshow(pred.image)
    ax.set_title(f"Score: {pred.pred_score:.2f}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("anomaly_results.png")
print("Results saved to anomaly_results.png")
```

---

## 8. 벤치마크 - 여러 모델 비교

```bash
python << 'EOF'
from anomalib.data import MVTec
from anomalib.models import Patchcore, Padim, Fastflow
from anomalib.engine import Engine

models = {
    "PatchCore": Patchcore(backbone="wide_resnet50_2"),
    "PaDiM": Padim(backbone="resnet18"),
    "FastFlow": Fastflow(backbone="resnet18"),
}

datamodule = MVTec(root="~/datasets/MVTec", category="bottle")

results = {}
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    engine = Engine(max_epochs=1 if name == "PatchCore" else 100)
    engine.fit(model=model, datamodule=datamodule)
    test_result = engine.test(model=model, datamodule=datamodule)
    results[name] = test_result[0]

print("\n=== Benchmark Results ===")
for name, result in results.items():
    print(f"{name}: AUROC = {result['image_AUROC']:.4f}")
EOF
```

---

## 9. 문제 해결

### 9-1. `ImportError: cannot import name 'XXX'`

**원인**: Lightning 버전 충돌

**해결**:
```bash
pip install --upgrade anomalib lightning
```

### 9-2. CUDA Out of Memory

**해결**:
```python
# 배치 사이즈 줄이기
datamodule = MVTec(
    train_batch_size=16,  # 32 → 16
    eval_batch_size=16,
)

# 또는 더 가벼운 backbone 사용
model = Patchcore(backbone="resnet18")  # wide_resnet50_2 대신
```

### 9-3. 데이터셋 다운로드 실패

**해결**:
```bash
# 수동 다운로드 후 경로 지정
datamodule = MVTec(
    root="/absolute/path/to/MVTec",
    category="bottle",
)
```

### 9-4. OpenVINO 관련 에러

**해결**:
```bash
# OpenVINO 재설치
pip uninstall openvino openvino-dev -y
pip install openvino openvino-dev
```

---

## 10. 환경 관리

### 10-1. 단축 명령 추가

```bash
cat >> ~/.bashrc << 'EOF'

# Anomalib 환경 단축 명령
alias al='conda activate anomalib_env'
EOF

source ~/.bashrc
```

**사용법**: `al` → anomalib_env 활성화

### 10-2. 패키지 백업

```bash
conda activate anomalib_env
conda list --export > ~/anomalib_env_packages.txt
pip freeze > ~/anomalib_requirements.txt
```

---

## 11. 전체 설치 체크리스트

| 단계 | 내용 | 확인 |
|------|------|------|
| 1 | anomalib_env 환경 생성 | ☐ |
| 2 | PyTorch 설치 (CUDA) | ☐ |
| 3 | Anomalib 설치 | ☐ |
| 4 | Lightning 설치 | ☐ |
| 5 | `import anomalib` 성공 | ☐ |
| 6 | GPU 인식 확인 | ☐ |
| 7 | MVTec 데이터셋 다운로드 | ☐ |
| 8 | 모델 학습 테스트 | ☐ |
| 9 | 추론 테스트 | ☐ |
| 10 | 결과 시각화 | ☐ |

---

## 12. 전체 환경 요약

이제 **4개의 독립된 딥러닝 환경**이 완성되었습니다!

| 환경 | Python | 주요 라이브러리 | 용도 |
|------|--------|----------------|------|
| pytorch_env | 3.10 | PyTorch 2.5.1 | 범용 딥러닝 |
| cupy_env | 3.10 | CuPy 13.6.0 | NumPy GPU 가속 |
| tensorflow_env | 3.10 | TensorFlow 2.18.0 | TensorFlow 딥러닝 |
| anomalib_env | 3.10 | Anomalib 1.1.x | 이상 감지 특화 |

### 빠른 환경 전환

```bash
pt   # PyTorch
cu   # CuPy
tf   # TensorFlow
al   # Anomalib
ca   # 비활성화
```

---

## 13. 참고 자료

- **Anomalib 공식 문서**: https://anomalib.readthedocs.io/
- **Anomalib GitHub**: https://github.com/openvinotoolkit/anomalib
- **MVTec AD 데이터셋**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **논문 모음**: https://github.com/hoya012/awesome-anomaly-detection

---

# 14. 사용자 정의 폴더 설정 (Datasets & Backbones)

Anomalib에서 데이터셋과 백본 가중치를 사용자 정의 경로에 저장하고 관리하는 방법입니다.

---

## 14-1. 현재 폴더 구조 확인

```bash
# 기존 데이터셋 위치
ls /mnt/d/datasets/
# 출력: mvtec  visa  btad

# 기존 백본 가중치 위치
ls /mnt/d/backbones/
```

---

## 14-2. 환경 변수 설정 (영구 적용)

### 방법 1: .bashrc에 추가 (권장)

```bash
cat >> ~/.bashrc << 'EOF'

# ============================================
# Anomalib 사용자 정의 경로 설정
# ============================================

# 데이터셋 루트 경로
export ANOMALIB_DATASET_ROOT="/mnt/d/datasets"

# 백본 가중치 저장 경로
export TORCH_HOME="/mnt/d/backbones/torch"
export TRANSFORMERS_CACHE="/mnt/d/backbones/huggingface"
export HF_HOME="/mnt/d/backbones/huggingface"
export TIMM_CACHE_DIR="/mnt/d/backbones/timm"

# OpenVINO 모델 캐시 (선택)
export OPENVINO_HOME="/mnt/d/backbones/openvino"

EOF

# 적용
source ~/.bashrc
```

### 방법 2: Anaconda 환경별 설정

```bash
conda activate anomalib_env

# conda 환경 디렉토리 확인
CONDA_ENV_DIR=$(conda info --base)/envs/anomalib_env

# 환경별 환경 변수 설정
mkdir -p $CONDA_ENV_DIR/etc/conda/activate.d
cat > $CONDA_ENV_DIR/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/sh
export ANOMALIB_DATASET_ROOT="/mnt/d/datasets"
export TORCH_HOME="/mnt/d/backbones/torch"
export TRANSFORMERS_CACHE="/mnt/d/backbones/huggingface"
export HF_HOME="/mnt/d/backbones/huggingface"
export TIMM_CACHE_DIR="/mnt/d/backbones/timm"
EOF

# 환경 비활성화 시 원복
mkdir -p $CONDA_ENV_DIR/etc/conda/deactivate.d
cat > $CONDA_ENV_DIR/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/sh
unset ANOMALIB_DATASET_ROOT
unset TORCH_HOME
unset TRANSFORMERS_CACHE
unset HF_HOME
unset TIMM_CACHE_DIR
EOF

# 재활성화
conda deactivate
conda activate anomalib_env
```

---

## 14-3. 디렉토리 구조 생성

```bash
# 백본 가중치 저장 디렉토리 생성
mkdir -p /mnt/d/backbones/torch
mkdir -p /mnt/d/backbones/huggingface
mkdir -p /mnt/d/backbones/timm
mkdir -p /mnt/d/backbones/openvino

# datasets 하위 구조 (자동 생성될 예정)
mkdir -p /mnt/d/datasets/custom

# 심볼릭 링크 생성 (선택 - 홈에서 쉽게 접근)
ln -s /mnt/d/datasets ~/datasets
ln -s /mnt/d/backbones ~/backbones
```

**최종 디렉토리 구조**:
```
/mnt/d/
├── datasets/
│   ├── mvtec/           # 기존
│   ├── visa/            # 기존
│   ├── btad/            # 기존
│   └── custom/          # 새로운 데이터셋 추가
│       ├── my_dataset1/
│       └── my_dataset2/
└── backbones/
    ├── torch/           # torch.hub 가중치
    ├── huggingface/     # transformers 가중치
    ├── timm/            # timm 가중치
    └── openvino/        # OpenVINO 모델
```

---

## 14-4. 데이터셋 경로 설정

### 방법 1: 환경 변수 사용 (자동)

```python
import os
from anomalib.data import MVTec, Visa, BTech

# 환경 변수에서 자동으로 경로 가져오기
dataset_root = os.getenv("ANOMALIB_DATASET_ROOT", "~/datasets")

# MVTec 데이터셋
mvtec_data = MVTec(
    root=os.path.join(dataset_root, "mvtec"),
    category="bottle",
)

# Visa 데이터셋
visa_data = Visa(
    root=os.path.join(dataset_root, "visa"),
    category="candle",
)

# BTech 데이터셋
btech_data = BTech(
    root=os.path.join(dataset_root, "btad"),
    category="01",
)
```

### 방법 2: Config 파일 사용

**config.yaml 생성**:
```bash
cat > ~/anomalib_config.yaml << 'EOF'
# 기본 경로 설정
defaults:
  - dataset_root: /mnt/d/datasets
  - backbone_root: /mnt/d/backbones

# MVTec 설정
mvtec:
  data:
    class_path: anomalib.data.MVTec
    init_args:
      root: ${defaults.dataset_root}/mvtec
      category: bottle
      image_size: [224, 224]

# Visa 설정
visa:
  data:
    class_path: anomalib.data.Visa
    init_args:
      root: ${defaults.dataset_root}/visa
      category: candle

# BTech 설정
btech:
  data:
    class_path: anomalib.data.BTech
    init_args:
      root: ${defaults.dataset_root}/btad
      category: "01"
EOF
```

---

## 14-5. 사용자 정의 데이터셋 추가

### 커스텀 데이터셋 폴더 구조

```
/mnt/d/datasets/custom/my_product/
├── train/
│   └── good/
│       ├── 001.png
│       ├── 002.png
│       └── ...
├── test/
│   ├── good/
│   │   ├── 001.png
│   │   └── ...
│   ├── defect_type1/
│   │   ├── 001.png
│   │   └── ...
│   └── defect_type2/
└── ground_truth/  (선택사항)
    ├── defect_type1/
    └── defect_type2/
```

### 커스텀 데이터셋 클래스 생성

```python
from anomalib.data import Folder
import os

# 방법 1: Folder 클래스 사용 (간단)
dataset_root = os.getenv("ANOMALIB_DATASET_ROOT", "/mnt/d/datasets")

custom_data = Folder(
    name="my_product",
    root=os.path.join(dataset_root, "custom/my_product"),
    normal_dir="train/good",
    abnormal_dir="test",
    normal_test_dir="test/good",  # 선택사항
    mask_dir="ground_truth",      # 선택사항
    image_size=(256, 256),
    train_batch_size=32,
    eval_batch_size=32,
)
```

### 커스텀 데이터로더 예시

```bash
cat > ~/custom_datamodule.py << 'EOF'
"""사용자 정의 데이터셋 로더"""
from anomalib.data import Folder
from pathlib import Path
import os


class CustomDataModule:
    """커스텀 데이터셋을 위한 통합 로더"""
    
    def __init__(self, dataset_name: str, category: str = None):
        """
        Args:
            dataset_name: 데이터셋 이름 (my_product1, my_product2 등)
            category: 카테고리 (필요한 경우)
        """
        self.dataset_root = os.getenv("ANOMALIB_DATASET_ROOT", "/mnt/d/datasets")
        self.dataset_name = dataset_name
        self.category = category
        
    def get_datamodule(self, **kwargs):
        """데이터모듈 생성"""
        if self.category:
            data_path = Path(self.dataset_root) / "custom" / self.dataset_name / self.category
        else:
            data_path = Path(self.dataset_root) / "custom" / self.dataset_name
            
        return Folder(
            name=self.dataset_name,
            root=str(data_path),
            normal_dir="train/good",
            abnormal_dir="test",
            normal_test_dir="test/good",
            mask_dir="ground_truth",
            **kwargs
        )


# 사용 예시
if __name__ == "__main__":
    loader = CustomDataModule("my_product1")
    datamodule = loader.get_datamodule(
        image_size=(256, 256),
        train_batch_size=32,
    )
    
    datamodule.setup()
    print(f"Train samples: {len(datamodule.train_dataloader())}")
    print(f"Test samples: {len(datamodule.test_dataloader())}")
EOF
```

---

## 14-6. 백본 가중치 관리

### torch.hub 가중치 저장

```python
import torch
import os

# TORCH_HOME이 설정되어 있으면 자동으로 해당 경로에 저장됨
print(f"Torch hub dir: {torch.hub.get_dir()}")

# ResNet 다운로드 예시 (자동으로 /mnt/d/backbones/torch에 저장)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# Wide ResNet 다운로드
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
```

### timm 가중치 저장

```python
import timm
import os

# timm 캐시 경로 확인
print(f"TIMM cache dir: {os.getenv('TIMM_CACHE_DIR')}")

# 백본 다운로드 (자동으로 /mnt/d/backbones/timm에 저장)
model = timm.create_model('resnet50', pretrained=True)
model = timm.create_model('efficientnet_b0', pretrained=True)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

### HuggingFace 가중치 저장

```python
from transformers import AutoModel
import os

# HuggingFace 캐시 경로 확인
print(f"HF cache dir: {os.getenv('HF_HOME')}")

# 모델 다운로드 (자동으로 /mnt/d/backbones/huggingface에 저장)
model = AutoModel.from_pretrained('microsoft/resnet-50')
```

### 백본 가중치 사전 다운로드 스크립트

```bash
cat > ~/download_backbones.py << 'EOF'
"""주요 백본 가중치 사전 다운로드"""
import torch
import timm
from tqdm import tqdm

def download_torch_models():
    """PyTorch Hub 모델 다운로드"""
    models = [
        'resnet18',
        'resnet34', 
        'resnet50',
        'wide_resnet50_2',
        'efficientnet_b0',
    ]
    
    print("\n=== Downloading PyTorch Hub Models ===")
    for model_name in tqdm(models, desc="PyTorch"):
        try:
            torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")


def download_timm_models():
    """TIMM 모델 다운로드"""
    models = [
        'resnet18',
        'resnet50',
        'efficientnet_b0',
        'efficientnet_b4',
        'vit_base_patch16_224',
        'wide_resnet50_2',
    ]
    
    print("\n=== Downloading TIMM Models ===")
    for model_name in tqdm(models, desc="TIMM"):
        try:
            timm.create_model(model_name, pretrained=True)
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")


if __name__ == "__main__":
    print("Starting backbone download...")
    print(f"TORCH_HOME: {torch.hub.get_dir()}")
    
    download_torch_models()
    download_timm_models()
    
    print("\n✓ All backbones downloaded successfully!")
EOF

# 실행
python ~/download_backbones.py
```

---

## 14-7. 경로 검증 및 확인

### 경로 설정 확인 스크립트

```bash
cat > ~/check_paths.py << 'EOF'
"""경로 설정 확인 스크립트"""
import os
import torch
from pathlib import Path

def check_environment():
    """환경 변수 확인"""
    print("="*60)
    print("Environment Variables Check")
    print("="*60)
    
    env_vars = {
        "ANOMALIB_DATASET_ROOT": "/mnt/d/datasets",
        "TORCH_HOME": "/mnt/d/backbones/torch",
        "TRANSFORMERS_CACHE": "/mnt/d/backbones/huggingface",
        "HF_HOME": "/mnt/d/backbones/huggingface",
        "TIMM_CACHE_DIR": "/mnt/d/backbones/timm",
    }
    
    all_ok = True
    for var_name, expected_path in env_vars.items():
        actual_path = os.getenv(var_name, "NOT SET")
        status = "✓" if actual_path == expected_path else "✗"
        print(f"{status} {var_name}: {actual_path}")
        if actual_path != expected_path:
            all_ok = False
    
    return all_ok


def check_directories():
    """디렉토리 존재 확인"""
    print("\n" + "="*60)
    print("Directory Structure Check")
    print("="*60)
    
    required_dirs = [
        "/mnt/d/datasets/mvtec",
        "/mnt/d/datasets/visa",
        "/mnt/d/datasets/btad",
        "/mnt/d/datasets/custom",
        "/mnt/d/backbones/torch",
        "/mnt/d/backbones/huggingface",
        "/mnt/d/backbones/timm",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_weights():
    """백본 가중치 확인"""
    print("\n" + "="*60)
    print("Backbone Weights Check")
    print("="*60)
    
    # Torch Hub
    torch_hub = Path(torch.hub.get_dir())
    torch_weights = list(torch_hub.glob("**/*.pth"))
    print(f"Torch Hub weights: {len(torch_weights)} files")
    
    # TIMM
    timm_dir = Path(os.getenv("TIMM_CACHE_DIR", "~/.cache/timm"))
    if timm_dir.exists():
        timm_weights = list(timm_dir.glob("**/*.pth"))
        print(f"TIMM weights: {len(timm_weights)} files")
    else:
        print("TIMM cache not found")
    
    # HuggingFace
    hf_dir = Path(os.getenv("HF_HOME", "~/.cache/huggingface"))
    if hf_dir.exists():
        hf_models = list((hf_dir / "hub").glob("models--*"))
        print(f"HuggingFace models: {len(hf_models)} models")
    else:
        print("HuggingFace cache not found")


def check_datasets():
    """데이터셋 확인"""
    print("\n" + "="*60)
    print("Datasets Check")
    print("="*60)
    
    dataset_root = Path(os.getenv("ANOMALIB_DATASET_ROOT", "/mnt/d/datasets"))
    
    datasets = {
        "MVTec": dataset_root / "mvtec",
        "Visa": dataset_root / "visa",
        "BTech": dataset_root / "btad",
    }
    
    for name, path in datasets.items():
        if path.exists():
            categories = [d.name for d in path.iterdir() if d.is_dir()]
            print(f"✓ {name}: {len(categories)} categories")
            print(f"  Categories: {', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}")
        else:
            print(f"✗ {name}: Not found")


if __name__ == "__main__":
    env_ok = check_environment()
    dir_ok = check_directories()
    
    check_weights()
    check_datasets()
    
    print("\n" + "="*60)
    if env_ok and dir_ok:
        print("✓ All paths configured correctly!")
    else:
        print("✗ Some paths need configuration")
        print("\nRun: source ~/.bashrc")
        print("Or: conda deactivate && conda activate anomalib_env")
    print("="*60)
EOF

# 실행
python ~/check_paths.py
```

---

## 14-8. 학습 시 경로 사용 예시

### 예시 1: MVTec with Custom Path

```python
import os
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# 환경 변수에서 경로 가져오기
dataset_root = os.getenv("ANOMALIB_DATASET_ROOT", "/mnt/d/datasets")

# 데이터 설정
datamodule = MVTec(
    root=os.path.join(dataset_root, "mvtec"),
    category="bottle",
    image_size=(224, 224),
)

# 모델 생성 (백본 가중치는 자동으로 TORCH_HOME에서 로드)
model = Patchcore(
    backbone="wide_resnet50_2",  # /mnt/d/backbones/torch에서 로드
)

# 학습
engine = Engine(max_epochs=1)
engine.fit(model=model, datamodule=datamodule)
```

### 예시 2: Custom Dataset

```python
from anomalib.data import Folder
import os

dataset_root = os.getenv("ANOMALIB_DATASET_ROOT", "/mnt/d/datasets")

# 커스텀 데이터셋 로드
datamodule = Folder(
    name="my_product",
    root=os.path.join(dataset_root, "custom/my_product"),
    normal_dir="train/good",
    abnormal_dir="test",
)

# 학습 코드 동일
```

### 예시 3: Config 파일로 일괄 관리

```bash
cat > ~/train_config.yaml << 'EOF'
# 환경 변수 사용
data:
  class_path: anomalib.data.MVTec
  init_args:
    root: ${oc.env:ANOMALIB_DATASET_ROOT}/mvtec
    category: bottle
    image_size: [224, 224]

model:
  class_path: anomalib.models.Patchcore
  init_args:
    backbone: wide_resnet50_2

trainer:
  max_epochs: 1
  default_root_dir: ${oc.env:ANOMALIB_DATASET_ROOT}/../results
EOF

# 실행
anomalib fit --config ~/train_config.yaml
```

---

## 14-9. 유틸리티 함수 모음

### 경로 관리 헬퍼 클래스

```bash
cat > ~/anomalib_paths.py << 'EOF'
"""Anomalib 경로 관리 유틸리티"""
from pathlib import Path
import os


class AnomalibPaths:
    """경로 관리 클래스"""
    
    def __init__(self):
        self.dataset_root = Path(os.getenv("ANOMALIB_DATASET_ROOT", "/mnt/d/datasets"))
        self.backbone_root = Path(os.getenv("TORCH_HOME", "/mnt/d/backbones/torch")).parent
    
    def get_dataset_path(self, dataset_name: str, category: str = None) -> Path:
        """데이터셋 경로 반환"""
        if category:
            return self.dataset_root / dataset_name / category
        return self.dataset_root / dataset_name
    
    def get_custom_dataset_path(self, name: str) -> Path:
        """커스텀 데이터셋 경로 반환"""
        return self.dataset_root / "custom" / name
    
    def list_datasets(self) -> dict:
        """사용 가능한 데이터셋 목록"""
        datasets = {}
        for dataset_dir in self.dataset_root.iterdir():
            if dataset_dir.is_dir():
                categories = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
                datasets[dataset_dir.name] = categories
        return datasets
    
    def get_backbone_path(self, source: str = "torch") -> Path:
        """백본 가중치 경로 반환"""
        return self.backbone_root / source
    
    def __repr__(self):
        return f"AnomalibPaths(dataset_root={self.dataset_root}, backbone_root={self.backbone_root})"


# 사용 예시
if __name__ == "__main__":
    paths = AnomalibPaths()
    print(paths)
    print("\nAvailable datasets:")
    for name, categories in paths.list_datasets().items():
        print(f"  {name}: {len(categories)} categories")
EOF
```

---

## 14-10. 체크리스트

| 단계 | 내용 | 확인 |
|------|------|------|
| 1 | .bashrc에 환경 변수 추가 | ☐ |
| 2 | 디렉토리 구조 생성 | ☐ |
| 3 | 기존 데이터셋 경로 확인 (mvtec, visa, btad) | ☐ |
| 4 | 백본 가중치 디렉토리 생성 | ☐ |
| 5 | 환경 변수 적용 (source ~/.bashrc) | ☐ |
| 6 | check_paths.py 실행하여 검증 | ☐ |
| 7 | 백본 가중치 다운로드 테스트 | ☐ |
| 8 | 커스텀 데이터셋 로드 테스트 | ☐ |

---

## 14-11. 요약

### 핵심 환경 변수

```bash
export ANOMALIB_DATASET_ROOT="/mnt/d/datasets"
export TORCH_HOME="/mnt/d/backbones/torch"
export TRANSFORMERS_CACHE="/mnt/d/backbones/huggingface"
export HF_HOME="/mnt/d/backbones/huggingface"
export TIMM_CACHE_DIR="/mnt/d/backbones/timm"
```

### 폴더 구조

```
/mnt/d/
├── datasets/
│   ├── mvtec/          # 기존
│   ├── visa/           # 기존
│   ├── btad/           # 기존
│   └── custom/         # 새 데이터셋 추가
└── backbones/
    ├── torch/          # torch.hub
    ├── huggingface/    # transformers
    └── timm/           # timm
```

### 빠른 검증

```bash
# 환경 변수 확인
echo $ANOMALIB_DATASET_ROOT
echo $TORCH_HOME

# 경로 검증 스크립트 실행
python ~/check_paths.py
```

이제 모든 데이터셋과 백본 가중치가 `/mnt/d/` 아래에서 체계적으로 관리됩니다! 🎉