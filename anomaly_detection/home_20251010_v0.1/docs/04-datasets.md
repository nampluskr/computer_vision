# Datasets - 데이터셋 준비 가이드

## 목차

1. [개요](#1-개요)
2. [MVTec AD 데이터셋](#2-mvtec-ad-데이터셋)
3. [VisA 데이터셋](#3-visa-데이터셋)
4. [BTAD 데이터셋](#4-btad-데이터셋)
5. [Custom 데이터셋](#5-custom-데이터셋)
6. [보조 데이터셋](#6-보조-데이터셋)
7. [데이터셋 검증](#7-데이터셋-검증)
8. [데이터 증강](#8-데이터-증강)
9. [문제 해결](#9-문제-해결)

---

## 1. 개요

본 프레임워크는 4가지 유형의 데이터셋을 지원합니다. 각 데이터셋은 고유한 디렉토리 구조와 특성을 가지고 있습니다.

### 1.1. 지원 데이터셋

| 데이터셋 | 카테고리 수 | 특징 | 용도 |
|---------|-----------|------|------|
| MVTec AD | 15 | 산업 검사 벤치마크 | 표준 평가 |
| VisA | 12 | 다양한 이상 유형 | 복잡한 이상 감지 |
| BTAD | 3 | 실제 산업 데이터 | 실용적 응용 |
| Custom | 무제한 | 사용자 정의 | 특화된 응용 |

### 1.2. 데이터셋 디렉토리 구조

```
datasets/
├── mvtec/              # MVTec AD Dataset
├── visa/               # VisA Dataset
├── btad/               # BTAD Dataset
├── custom/             # Custom Datasets
├── dtd/                # Describable Textures (DRAEM용)
└── imagenette2/        # ImageNet subset (EfficientAD용)
```

---

## 2. MVTec AD 데이터셋

### 2.1. 개요

MVTec Anomaly Detection (MVTec AD)는 산업 이상 감지를 위한 가장 널리 사용되는 벤치마크 데이터셋입니다.

**공식 사이트**: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**특징:**
- 15개 카테고리 (10개 물체, 5개 텍스처)
- 총 5,354개 고해상도 이미지
- 픽셀 수준 이상 영역 주석
- 다양한 이상 유형

### 2.2. 카테고리 목록

#### 물체 카테고리 (10개)
- bottle (병)
- cable (케이블)
- capsule (캡슐)
- hazelnut (헤이즐넛)
- metal_nut (금속 너트)
- pill (알약)
- screw (나사)
- toothbrush (칫솔)
- transistor (트랜지스터)
- zipper (지퍼)

#### 텍스처 카테고리 (5개)
- carpet (카펫)
- grid (격자)
- leather (가죽)
- tile (타일)
- wood (나무)

### 2.3. 다운로드

#### 방법 1: 직접 다운로드

```bash
cd ~/workspace/anomaly_detection/datasets

# Download MVTec AD
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz

# Extract
tar -xf mvtec_anomaly_detection.tar.xz

# Rename directory
mv mvtec_anomaly_detection mvtec

# Remove archive
rm mvtec_anomaly_detection.tar.xz
```

#### 방법 2: Python 스크립트

```python
import os
import urllib.request
import tarfile

dataset_dir = "~/workspace/anomaly_detection/datasets"
os.makedirs(dataset_dir, exist_ok=True)

url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
filename = os.path.join(dataset_dir, "mvtec_anomaly_detection.tar.xz")

print("Downloading MVTec AD...")
urllib.request.urlretrieve(url, filename)

print("Extracting...")
with tarfile.open(filename, "r:xz") as tar:
    tar.extractall(dataset_dir)

os.rename(
    os.path.join(dataset_dir, "mvtec_anomaly_detection"),
    os.path.join(dataset_dir, "mvtec")
)
os.remove(filename)

print("Download complete!")
```

### 2.4. 디렉토리 구조

```
mvtec/
├── bottle/
│   ├── train/
│   │   └── good/                    # 정상 학습 이미지 (209장)
│   │       ├── 000.png
│   │       ├── 001.png
│   │       └── ...
│   ├── test/
│   │   ├── good/                    # 정상 테스트 이미지 (20장)
│   │   ├── broken_large/            # 이상 유형 1 (9장)
│   │   ├── broken_small/            # 이상 유형 2 (15장)
│   │   └── contamination/           # 이상 유형 3 (6장)
│   └── ground_truth/
│       ├── broken_large/            # 이상 마스크
│       │   ├── 000_mask.png
│       │   └── ...
│       ├── broken_small/
│       └── contamination/
├── cable/
├── capsule/
└── ...
```

### 2.5. 데이터 통계

| 카테고리 | 학습(정상) | 테스트(정상) | 테스트(이상) | 이상 유형 |
|---------|-----------|-------------|-------------|----------|
| bottle | 209 | 20 | 63 | 3 |
| cable | 224 | 58 | 92 | 5 |
| capsule | 219 | 23 | 109 | 5 |
| carpet | 280 | 28 | 89 | 5 |
| grid | 264 | 21 | 57 | 5 |
| hazelnut | 391 | 40 | 70 | 5 |
| leather | 245 | 32 | 92 | 5 |
| metal_nut | 220 | 22 | 93 | 4 |
| pill | 267 | 26 | 141 | 7 |
| screw | 320 | 41 | 119 | 5 |
| tile | 230 | 33 | 84 | 5 |
| toothbrush | 60 | 12 | 30 | 1 |
| transistor | 213 | 60 | 40 | 4 |
| wood | 247 | 19 | 60 | 5 |
| zipper | 240 | 32 | 119 | 7 |

### 2.6. 사용 예시

```python
from train import train

# Single category
train("mvtec", "bottle", "stfpm", num_epochs=50)

# Multiple categories
train("mvtec", ["bottle", "wood", "grid"], "stfpm", num_epochs=50)

# All categories
train("mvtec", "all", "padim", num_epochs=1)
```

### 2.7. 검증

```python
import os

dataset_dir = "~/workspace/anomaly_detection/datasets/mvtec"
categories = os.listdir(dataset_dir)

print(f"Found {len(categories)} categories:")
for cat in sorted(categories):
    cat_path = os.path.join(dataset_dir, cat)
    if os.path.isdir(cat_path):
        train_path = os.path.join(cat_path, "train/good")
        test_path = os.path.join(cat_path, "test")
        
        n_train = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
        n_test_dirs = len(os.listdir(test_path)) if os.path.exists(test_path) else 0
        
        print(f"  {cat:15s}: {n_train:3d} train, {n_test_dirs:2d} test types")
```

---

## 3. VisA 데이터셋

### 3.1. 개요

Visual Anomaly (VisA)는 다양한 실제 산업 시나리오를 포함하는 대규모 데이터셋입니다.

**공식 사이트**: [https://github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)

**특징:**
- 12개 카테고리
- 10,821개 이미지
- 복잡한 이상 패턴
- CSV 기반 주석

### 3.2. 카테고리 목록

- candle (양초)
- capsules (캡슐)
- cashew (캐슈넛)
- chewinggum (껌)
- fryum (튀김과자)
- macaroni1 (마카로니 1)
- macaroni2 (마카로니 2)
- pcb1 (PCB 1)
- pcb2 (PCB 2)
- pcb3 (PCB 3)
- pcb4 (PCB 4)
- pipe_fryum (파이프 튀김과자)

### 3.3. 다운로드

```bash
cd ~/workspace/anomaly_detection/datasets

# Clone repository
git clone https://github.com/amazon-science/spot-diff.git

# Download data (follow instructions in repository)
# The dataset must be requested from authors

# After obtaining the data, organize as:
mkdir -p visa
# Extract downloaded files to visa/
```

### 3.4. 디렉토리 구조

```
visa/
├── candle/
│   ├── Data/
│   │   └── Images/
│   │       ├── Anomaly/
│   │       │   ├── 000/
│   │       │   │   ├── 0000.JPG
│   │       │   │   └── ...
│   │       │   ├── 001/
│   │       │   └── ...
│   │       └── Normal/
│   │           ├── 0000.JPG
│   │           └── ...
│   ├── image_anno.csv           # Image annotations
│   └── split_csv/
│       ├── 1cls.csv             # 1-class split
│       └── 2cls.csv             # 2-class split
├── capsules/
├── cashew/
└── ...
```

### 3.5. CSV 구조

#### image_anno.csv
```csv
image,label,split,mask
candle/Data/Images/Normal/0000.JPG,normal,train,
candle/Data/Images/Anomaly/000/0000.JPG,bad,test,candle/Data/Masks/Anomaly/000/0000.png
```

**필드 설명:**
- `image`: 이미지 상대 경로
- `label`: "normal" 또는 "bad"
- `split`: "train" 또는 "test"
- `mask`: 이상 마스크 경로 (이상 이미지만)

### 3.6. 사용 예시

```python
# Single category
train("visa", "candle", "stfpm", num_epochs=50)

# Multiple categories
train("visa", ["candle", "capsules", "cashew"], "padim", num_epochs=1)

# All categories
train("visa", "all", "patchcore", num_epochs=1)
```

---

## 4. BTAD 데이터셋

### 4.1. 개요

BeanTech Anomaly Detection (BTAD)는 실제 산업 응용을 위한 데이터셋입니다.

**공식 사이트**: [http://avires.dimi.uniud.it/papers/btad/btad.zip](http://avires.dimi.uniud.it/papers/btad/btad.zip)

**특징:**
- 3개 산업 제품
- 실제 결함 이미지
- 높은 해상도
- 실용적 시나리오

### 4.2. 카테고리 목록

- 01: 산업 제품 1
- 02: 산업 제품 2
- 03: 산업 제품 3

### 4.3. 다운로드

```bash
cd ~/workspace/anomaly_detection/datasets

# Download BTAD
wget http://avires.dimi.uniud.it/papers/btad/btad.zip

# Extract
unzip btad.zip

# Rename directory
mv BTech_Dataset_transformed btad

# Remove archive
rm btad.zip
```

### 4.4. 디렉토리 구조

```
btad/
├── 01/
│   ├── train/
│   │   └── ok/                      # 정상 학습 이미지
│   │       ├── 000.bmp
│   │       ├── 001.bmp
│   │       └── ...
│   ├── test/
│   │   ├── ok/                      # 정상 테스트 이미지
│   │   └── ko/                      # 이상 테스트 이미지
│   └── ground_truth/
│       └── ko/                      # 이상 마스크
│           ├── 000.bmp
│           └── ...
├── 02/
└── 03/
```

### 4.5. 데이터 통계

| 카테고리 | 학습(정상) | 테스트(정상) | 테스트(이상) |
|---------|-----------|-------------|-------------|
| 01 | 127 | 36 | 22 |
| 02 | 183 | 39 | 38 |
| 03 | 141 | 34 | 33 |

### 4.6. 사용 예시

```python
# Single category
train("btad", "01", "stfpm", num_epochs=50)

# Multiple categories
train("btad", ["01", "02", "03"], "padim", num_epochs=1)

# All categories
train("btad", "all", "patchcore", num_epochs=1)
```

---

## 5. Custom 데이터셋

### 5.1. 개요

사용자 정의 데이터셋을 위한 유연한 구조를 제공합니다. 파일명 규칙을 따르면 자동으로 메타데이터가 생성됩니다.

### 5.2. 디렉토리 구조

```
custom/
└── your_dataset_name/
    └── data_rgb/
        ├── normal/              # 정상 이미지 (label=0)
        │   ├── pattern1 60 100.png
        │   ├── pattern1 60 200.png
        │   ├── pattern2 80 150.png
        │   └── ...
        ├── defect_type1/        # 이상 유형 1 (label=1)
        │   ├── pattern1 60 100.png
        │   ├── pattern2 80 150.png
        │   └── ...
        ├── defect_type2/        # 이상 유형 2 (label=1)
        │   └── ...
        └── data_info.csv        # 자동 생성됨
```

### 5.3. 파일명 규칙

#### 형식
```
{category} {freq} {dimming}[_{extra}].png
```

#### 구성 요소

- **category**: 패턴 또는 디자인 이름 (예: pattern1, design_A)
- **freq**: 주파수 파라미터 (정수)
- **dimming**: 디밍 파라미터 (정수)
- **extra**: 선택적 추가 식별자 (예: v2, test)

#### 유효한 예시

```
✓ pattern1 60 100.png
✓ pattern2 120 200.png
✓ design_A 80 150_v2.png
✓ test_pattern 100 50.png
```

#### 잘못된 예시

```
✗ image001.png                  # 메타데이터 누락
✗ pattern1_60_100.png           # 구분자 오류 (공백 사용)
✗ pattern1 60.png               # dimming 값 누락
```

### 5.4. 데이터셋 준비 단계

#### Step 1: 디렉토리 생성

```bash
cd ~/workspace/anomaly_detection/datasets
mkdir -p custom/my_product/data_rgb/normal
mkdir -p custom/my_product/data_rgb/scratch
mkdir -p custom/my_product/data_rgb/crack
```

#### Step 2: 이미지 복사

```bash
# Copy normal images
cp /path/to/normal/*.png custom/my_product/data_rgb/normal/

# Copy defect images
cp /path/to/scratch/*.png custom/my_product/data_rgb/scratch/
cp /path/to/crack/*.png custom/my_product/data_rgb/crack/
```

#### Step 3: 파일명 확인

```python
import os

data_dir = "~/workspace/anomaly_detection/datasets/custom/my_product/data_rgb"

for defect_type in os.listdir(data_dir):
    defect_path = os.path.join(data_dir, defect_type)
    if os.path.isdir(defect_path):
        files = os.listdir(defect_path)
        print(f"\n{defect_type}:")
        print(f"  Files: {len(files)}")
        print(f"  Sample: {files[0] if files else 'None'}")
```

#### Step 4: CSV 자동 생성

CSV 파일은 첫 학습 시 자동으로 생성됩니다:

```python
train("my_product", "all", "padim", num_epochs=1)
```

### 5.5. data_info.csv 구조

```csv
filename,category,freq,dimming,image_path,dataset_type,defect_type,label
pattern1 60 100.png,pattern1,60,100,/full/path/to/normal/pattern1 60 100.png,my_product,normal,0
pattern1 60 200.png,pattern1,60,200,/full/path/to/scratch/pattern1 60 200.png,my_product,scratch,1
pattern2 80 150.png,pattern2,80,150,/full/path/to/normal/pattern2 80 150.png,my_product,normal,0
```

**필드 설명:**

| 필드 | 타입 | 설명 |
|-----|------|------|
| filename | str | 파일명 |
| category | str | 파일명에서 추출한 카테고리 |
| freq | int | 주파수 파라미터 |
| dimming | int | 디밍 파라미터 |
| image_path | str | 전체 경로 |
| dataset_type | str | 데이터셋 이름 |
| defect_type | str | 결함 유형 (디렉토리 이름) |
| label | int | 0=정상, 1=이상 |

### 5.6. 학습/테스트 분할

Custom 데이터셋은 자동으로 학습/테스트로 분할됩니다:

- **정상 이미지**: 80% 학습, 20% 테스트
- **이상 이미지**: 100% 테스트

```python
# dataloader.py에서 자동 처리
# train_ratio = 0.8 for normal images
# train_ratio = 0.0 for anomaly images
```

### 5.7. 사용 예시

#### 단일 카테고리

```python
# Category: pattern1
train("my_product", "pattern1", "stfpm", num_epochs=50)
```

#### 다중 카테고리

```python
# Categories: pattern1, pattern2
train("my_product", ["pattern1", "pattern2"], "stfpm", num_epochs=50)
```

#### 모든 카테고리

```python
# All categories
train("my_product", "all", "dinomaly-base-224", num_epochs=15)
```

#### 다중 데이터셋

```python
# Multiple datasets, specific categories
train(
    ["product_A", "product_B"],
    ["pattern1", "pattern2"],
    "stfpm",
    num_epochs=50
)

# Multiple datasets, all categories
train(
    ["product_A", "product_B"],
    "all",
    "padim",
    num_epochs=1
)
```

### 5.8. 고급 설정

#### CSV 재생성

기존 CSV를 삭제하고 재실행하면 자동으로 재생성됩니다:

```bash
rm ~/workspace/anomaly_detection/datasets/custom/my_product/data_rgb/data_info.csv
```

```python
train("my_product", "all", "padim", num_epochs=1)
```

#### 커스텀 파일명 파서

파일명 규칙을 변경하려면 `dataloader.py`의 `get_data_info()` 함수를 수정합니다:

```python
def get_data_info(filename: str) -> tuple[str, int, int]:
    """
    Parse custom filename to extract metadata.
    
    Returns:
        category: Pattern name
        freq: Frequency parameter
        dimming: Dimming parameter
    """
    # Custom parsing logic
    pass
```

---

## 6. 보조 데이터셋

### 6.1. DTD (Describable Textures Dataset)

DRAEM 모델의 이상 시뮬레이션을 위한 텍스처 데이터셋입니다.

#### 다운로드

```bash
cd ~/workspace/anomaly_detection/datasets

# Download DTD
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

# Extract
tar -xzf dtd-r1.0.1.tar.gz

# Reorganize
mv dtd/images dtd_images
rm -rf dtd
mv dtd_images dtd

# Remove archive
rm dtd-r1.0.1.tar.gz
```

#### 구조

```
dtd/
├── banded/
├── blotchy/
├── braided/
├── bubbly/
├── bumpy/
└── ... (총 47개 텍스처 카테고리)
```

#### 사용

DRAEM 학습 시 자동으로 로드됩니다:

```python
train("mvtec", "bottle", "draem", num_epochs=10)
```

---

### 6.2. Imagenette2

EfficientAD의 Teacher 네트워크 사전학습을 위한 ImageNet 서브셋입니다.

#### 다운로드

```bash
cd ~/workspace/anomaly_detection/datasets

# Download Imagenette2
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz

# Extract
tar -xzf imagenette2.tgz

# Remove archive
rm imagenette2.tgz
```

#### 구조

```
imagenette2/
└── train/
    ├── n01440764/           # tench (잉어)
    ├── n02102040/           # English springer (개)
    ├── n02979186/           # cassette player
    ├── n03000684/           # chain saw
    ├── n03028079/           # church
    ├── n03394916/           # French horn
    ├── n03417042/           # garbage truck
    ├── n03425413/           # gas pump
    ├── n03445777/           # golf ball
    └── n03888257/           # parachute
```

#### 사용

EfficientAD 학습 시 자동으로 로드됩니다:

```python
train("mvtec", "bottle", "efficientad-small", num_epochs=20)
```

---

## 7. 데이터셋 검증

### 7.1. 자동 검증 스크립트

```python
# validate_datasets.py
import os
from pathlib import Path

def validate_mvtec(dataset_dir):
    """Validate MVTec AD dataset structure"""
    mvtec_dir = Path(dataset_dir) / "mvtec"
    
    if not mvtec_dir.exists():
        print("✗ MVTec directory not found")
        return False
    
    categories = [d.name for d in mvtec_dir.iterdir() if d.is_dir()]
    print(f"\n=== MVTec AD Dataset ===")
    print(f"Found {len(categories)} categories")
    
    for cat in sorted(categories):
        cat_path = mvtec_dir / cat
        train_path = cat_path / "train" / "good"
        test_path = cat_path / "test"
        gt_path = cat_path / "ground_truth"
        
        n_train = len(list(train_path.glob("*.png"))) if train_path.exists() else 0
        n_test_types = len(list(test_path.iterdir())) if test_path.exists() else 0
        has_gt = gt_path.exists()
        
        status = "✓" if n_train > 0 else "✗"
        print(f"{status} {cat:15s}: {n_train:3d} train, {n_test_types:2d} test types, GT: {has_gt}")
    
    return True

def validate_visa(dataset_dir):
    """Validate VisA dataset structure"""
    visa_dir = Path(dataset_dir) / "visa"
    
    if not visa_dir.exists():
        print("✗ VisA directory not found")
        return False
    
    categories = [d.name for d in visa_dir.iterdir() if d.is_dir()]
    print(f"\n=== VisA Dataset ===")
    print(f"Found {len(categories)} categories")
    
    for cat in sorted(categories):
        cat_path = visa_dir / cat
        images_path = cat_path / "Data" / "Images"
        csv_path = cat_path / "image_anno.csv"
        
        has_images = images_path.exists()
        has_csv = csv_path.exists()
        
        status = "✓" if (has_images and has_csv) else "✗"
        print(f"{status} {cat:15s}: Images: {has_images}, CSV: {has_csv}")
    
    return True

def validate_btad(dataset_dir):
    """Validate BTAD dataset structure"""
    btad_dir = Path(dataset_dir) / "btad"
    
    if not btad_dir.exists():
        print("✗ BTAD directory not found")
        return False
    
    categories = [d.name for d in btad_dir.iterdir() if d.is_dir()]
    print(f"\n=== BTAD Dataset ===")
    print(f"Found {len(categories)} categories")
    
    for cat in sorted(categories):
        cat_path = btad_dir / cat
        train_path = cat_path / "train" / "ok"
        test_path = cat_path / "test"
        
        n_train = len(list(train_path.glob("*.bmp"))) if train_path.exists() else 0
        n_test_types = len(list(test_path.iterdir())) if test_path.exists() else 0
        
        status = "✓" if n_train > 0 else "✗"
        print(f"{status} {cat:5s}: {n_train:3d} train, {n_test_types:2d} test types")
    
    return True

def validate_custom(dataset_dir):
    """Validate Custom datasets structure"""
    custom_dir = Path(dataset_dir) / "custom"
    
    if not custom_dir.exists():
        print("✗ Custom directory not found")
        return False
    
    datasets = [d.name for d in custom_dir.iterdir() if d.is_dir()]
    print(f"\n=== Custom Datasets ===")
    print(f"Found {len(datasets)} custom datasets")
    
    for ds in sorted(datasets):
        ds_path = custom_dir / ds / "data_rgb"
        
        if not ds_path.exists():
            print(f"✗ {ds}: data_rgb directory not found")
            continue
        
        types = [d.name for d in ds_path.iterdir() if d.is_dir()]
        csv_path = ds_path / "data_info.csv"
        
        n_normal = len(list((ds_path / "normal").glob("*.png"))) if (ds_path / "normal").exists() else 0
        n_defect_types = len([t for t in types if t != "normal"])
        has_csv = csv_path.exists()
        
        status = "✓" if n_normal > 0 else "✗"
        print(f"{status} {ds:20s}: {n_normal:3d} normal, {n_defect_types:2d} defect types, CSV: {has_csv}")
    
    return True

if __name__ == "__main__":
    dataset_dir = "~/workspace/anomaly_detection/datasets"
    dataset_dir = os.path.expanduser(dataset_dir)
    
    print("="*70)
    print("Dataset Validation")
    print("="*70)
    
    validate_mvtec(dataset_dir)
    validate_visa(dataset_dir)
    validate_btad(dataset_dir)
    validate_custom(dataset_dir)
    
    print("\n" + "="*70)
    print("Validation Complete")
    print("="*70)
```

실행:

```bash
python validate_datasets.py
```

### 7.2. 데이터 로더 테스트

```python
# test_dataloader.py
from dataloader import get_dataloaders
from train import set_globals

set_globals(
    dataset_dir="~/workspace/anomaly_detection/datasets"
)

# Test MVTec
print("Testing MVTec dataloader...")
train_loader, test_loader = get_dataloaders(
    dataset_type="mvtec",
    category="bottle",
    img_size=256,
    batch_size=4
)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# Get one batch
batch = next(iter(train_loader))
print(f"Batch keys: {batch.keys()}")
print(f"Image shape: {batch['image'].shape}")
print(f"Label shape: {batch['label'].shape}")

# Test Custom
print("\nTesting Custom dataloader...")
train_loader, test_loader = get_dataloaders(
    dataset_type="my_product",
    category="all",
    img_size=256,
    batch_size=4
)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
```

---

## 8. 데이터 증강

### 8.1. 기본 변환

프레임워크는 자동으로 다음 변환을 적용합니다:

```python
# Training transforms
transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # If normalize=True
                        std=[0.229, 0.224, 0.225])
])

# Test transforms (no augmentation)
transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # If normalize=True
                        std=[0.229, 0.224, 0.225])
])
```

### 8.2. 커스텀 증강

`dataloader.py`를 수정하여 추가 증강을 적용할 수 있습니다:

```python
# Add to get_dataloaders() function

# Custom augmentation for training
train_transform = transforms.Compose([
    transforms.Resize(img_size + 32),
    transforms.RandomCrop(img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**주의:** 이상 감지는 일반적으로 데이터 증강을 제한적으로 사용합니다. 과도한 증강은 성능을 저하시킬 수 있습니다.

---

## 9. 문제 해결

### 9.1. 디렉토리를 찾을 수 없음

**증상:**
```
FileNotFoundError: Dataset not found at /path/to/datasets/mvtec/bottle
```

**해결:**

1. 경로 확인:
```python
import os
from train import get_globals

config = get_globals()
print(f"Dataset dir: {config['dataset_dir']}")

# Check if directory exists
ds_path = os.path.join(config['dataset_dir'], "mvtec", "bottle")
print(f"Exists: {os.path.exists(ds_path)}")
```

2. 디렉토리 생성:
```bash
ls ~/workspace/anomaly_detection/datasets/mvtec/
```

### 9.2. 이미지를 로드할 수 없음

**증상:**
```
OSError: cannot identify image file
```

**해결:**

1. 이미지 형식 확인:
```python
from PIL import Image

img_path = "/path/to/image.png"
try:
    img = Image.open(img_path)
    print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
except Exception as e:
    print(f"Error: {e}")
```

2. 손상된 파일 검사:
```python
import os
from PIL import Image
from pathlib import Path

def check_images(directory):
    corrupted = []
    for img_path in Path(directory).rglob("*.png"):
        try:
            Image.open(img_path).verify()
        except Exception as e:
            corrupted.append(str(img_path))
    
    if corrupted:
        print(f"Found {len(corrupted)} corrupted images:")
        for path in corrupted:
            print(f"  {path}")
    else:
        print("All images are valid")

check_images("~/workspace/anomaly_detection/datasets/mvtec/bottle")
```

### 9.3. CSV 생성 오류 (Custom 데이터셋)

**증상:**
```
ValueError: Invalid filename format
```

**해결:**

1. 파일명 확인:
```python
import os

data_dir = "~/workspace/anomaly_detection/datasets/custom/my_product/data_rgb/normal"
files = os.listdir(data_dir)

print("Sample filenames:")
for f in files[:5]:
    print(f"  {f}")
    
# Check format
import re
pattern = r"^(.+)\s+(\d+)\s+(\d+)(?:_(.+))?\.png$"
for f in files[:5]:
    match = re.match(pattern, f)
    if match:
        print(f"  ✓ Valid: {f}")
    else:
        print(f"  ✗ Invalid: {f}")
```

2. 파일명 변경 스크립트:
```python
import os
import re

def rename_files(directory):
    """Rename files to match expected format"""
    for filename in os.listdir(directory):
        if not filename.endswith('.png'):
            continue
        
        # Example: convert "image_001.png" to "pattern 60 100.png"
        # Customize this logic for your naming convention
        new_name = f"pattern 60 100_{filename}"
        
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

# Usage
rename_files("~/workspace/anomaly_detection/datasets/custom/my_product/data_rgb/normal")
```

### 9.4. 메모리 부족

**증상:**
```
RuntimeError: DataLoader worker is killed by signal
```

**해결:**

1. num_workers 감소:
```python
set_globals(num_workers=4)  # Default: 8
```

2. batch_size 감소:
```python
train("mvtec", "bottle", "stfpm", batch_size=8)  # Default: 16
```

3. persistent_workers 비활성화:
```python
set_globals(persistent_workers=False)
```

### 9.5. 데이터 불균형

**증상:** 정상 이미지가 너무 많거나 적음

**해결:**

Custom 데이터셋의 경우 수동으로 분할 비율을 조정할 수 있습니다:

```python
# dataloader.py 수정
# Line ~200 in CustomDataset.__init__()

# Adjust train ratio
train_ratio = 0.7  # Default: 0.8 for normal images
```

---

**다음 문서:** [Training](05-training.md) - 학습 설정 및 팁