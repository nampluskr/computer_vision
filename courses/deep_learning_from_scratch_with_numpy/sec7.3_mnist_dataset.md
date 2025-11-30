## 7.3 MNIST Dataset (10 Classes)

MNIST(Modified National Institute of Standards and Technology) 데이터셋은 손글씨 숫자 이미지로 구성된 다중 클래스 분류의 대표적인 벤치마크 데이터셋입니다.

### 7.3.1 Dataset Overview

**기본 정보:**

- **클래스 개수**: 10개 (숫자 0~9)
- **훈련 데이터**: 60,000개 이미지
- **테스트 데이터**: 10,000개 이미지
- **이미지 크기**: 28×28 픽셀 (grayscale)
- **픽셀 값 범위**: 0~255 (uint8)

**데이터셋 구조:**

```
MNIST Dataset
├── Training Set (60,000 samples)
│   ├── Images: (60000, 28, 28)
│   └── Labels: (60000,) - values in {0, 1, 2, ..., 9}
│
└── Test Set (10,000 samples)
    ├── Images: (10000, 28, 28)
    └── Labels: (10000,) - values in {0, 1, 2, ..., 9}
```

**클래스 분포:**

```python
import numpy as np

def analyze_class_distribution():
    """클래스별 샘플 수 분석"""
    
    # 시뮬레이션 (실제로는 데이터 로딩 필요)
    # MNIST는 거의 균등하게 분포
    train_samples_per_class = {
        0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842,
        5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949
    }
    
    test_samples_per_class = {
        0: 980, 1: 1135, 2: 1032, 3: 1010, 4: 982,
        5: 892, 6: 958, 7: 1028, 8: 974, 9: 1009
    }
    
    print("=" * 60)
    print("MNIST Class Distribution")
    print("=" * 60)
    print(f"{'Class':<10} {'Train Samples':<20} {'Test Samples':<20}")
    print("-" * 60)
    
    for cls in range(10):
        print(f"{cls:<10} {train_samples_per_class[cls]:<20} {test_samples_per_class[cls]:<20}")
    
    print("-" * 60)
    print(f"{'Total':<10} {sum(train_samples_per_class.values()):<20} "
          f"{sum(test_samples_per_class.values()):<20}")
    print("=" * 60)

analyze_class_distribution()
```

```
============================================================
MNIST Class Distribution
============================================================
Class      Train Samples        Test Samples        
------------------------------------------------------------
0          5923                 980                 
1          6742                 1135                
2          5958                 1032                
3          6131                 1010                
4          5842                 982                 
5          5421                 892                 
6          5918                 958                 
7          6265                 1028                
8          5851                 974                 
9          5949                 1009                
------------------------------------------------------------
Total      60000                10000               
============================================================
```

### 7.3.2 Data Loading Functions

```python
import os
import gzip
import numpy as np

def load_mnist_images(data_dir, filename):
    """
    MNIST 이미지 파일 로딩
    
    Parameters:
    -----------
    data_dir : str
        데이터 디렉토리 경로
    filename : str
        이미지 파일명 (예: 'train-images-idx3-ubyte.gz')
    
    Returns:
    --------
    images : ndarray, shape (num_samples, 28, 28)
        이미지 데이터
    """
    data_path = os.path.join(data_dir, filename)
    
    with gzip.open(data_path, 'rb') as f:
        # IDX 파일 형식: magic number(4) + num_images(4) + rows(4) + cols(4) + data
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    # (num_samples, 28, 28) 형태로 reshape
    images = data.reshape(-1, 28, 28)
    
    return images


def load_mnist_labels(data_dir, filename):
    """
    MNIST 레이블 파일 로딩
    
    Parameters:
    -----------
    data_dir : str
        데이터 디렉토리 경로
    filename : str
        레이블 파일명 (예: 'train-labels-idx1-ubyte.gz')
    
    Returns:
    --------
    labels : ndarray, shape (num_samples,)
        레이블 데이터 (0~9)
    """
    data_path = os.path.join(data_dir, filename)
    
    with gzip.open(data_path, 'rb') as f:
        # IDX 파일 형식: magic number(4) + num_labels(4) + data
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return data


def get_mnist(data_dir, split="train"):
    """
    MNIST 데이터셋 로딩 (통합 함수)
    
    Parameters:
    -----------
    data_dir : str
        데이터 디렉토리 경로
    split : str
        'train' 또는 'test'
    
    Returns:
    --------
    images : ndarray
        이미지 데이터
    labels : ndarray
        레이블 데이터
    """
    if split == "train":
        images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    elif split == "test":
        images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
    else:
        raise ValueError("split must be 'train' or 'test'!")
    
    return images, labels
```

### 7.3.3 Data Preprocessing

```python
def one_hot(labels, num_classes=10):
    """
    One-hot encoding
    
    Parameters:
    -----------
    labels : ndarray, shape (N,)
        클래스 인덱스 (0~9)
    num_classes : int
        전체 클래스 개수
    
    Returns:
    --------
    onehot : ndarray, shape (N, num_classes)
        One-hot encoded 레이블
    """
    return np.eye(num_classes)[labels]


def preprocess_mnist(images, labels, flatten=True, normalize=True, onehot=True):
    """
    MNIST 데이터 전처리
    
    Parameters:
    -----------
    images : ndarray, shape (N, 28, 28)
        원본 이미지
    labels : ndarray, shape (N,)
        원본 레이블
    flatten : bool
        이미지를 1D 벡터로 평탄화 여부
    normalize : bool
        [0, 1] 범위로 정규화 여부
    onehot : bool
        레이블을 one-hot으로 인코딩 여부
    
    Returns:
    --------
    images_processed : ndarray
        전처리된 이미지
    labels_processed : ndarray
        전처리된 레이블
    """
    # 이미지 전처리
    images_processed = images.astype(np.float32)
    
    if flatten:
        images_processed = images_processed.reshape(-1, 28 * 28)
    
    if normalize:
        images_processed = images_processed / 255.0
    
    # 레이블 전처리
    if onehot:
        labels_processed = one_hot(labels, num_classes=10).astype(np.int64)
    else:
        labels_processed = labels.astype(np.int64)
    
    return images_processed, labels_processed
```

### 7.3.4 Complete Loading Example

```python
# 데이터 로딩 및 전처리 전체 과정
def load_and_preprocess_example():
    """MNIST 로딩 및 전처리 예제"""
    
    # 데이터 디렉토리 설정 (실제 경로로 수정 필요)
    data_dir = "/mnt/d/datasets/mnist"
    
    print("=" * 60)
    print("MNIST Data Loading and Preprocessing")
    print("=" * 60)
    
    # 1. 원본 데이터 로딩
    print("\n[Step 1] Loading raw data...")
    x_train, y_train = get_mnist(data_dir, split="train")
    x_test, y_test = get_mnist(data_dir, split="test")
    
    print(f"  Train images: {x_train.dtype}, {x_train.shape}, "
          f"[{x_train.min()}, {x_train.max()}]")
    print(f"  Train labels: {y_train.dtype}, {y_train.shape}, "
          f"[{y_train.min()}, {y_train.max()}]")
    print(f"  Test images:  {x_test.dtype}, {x_test.shape}, "
          f"[{x_test.min()}, {x_test.max()}]")
    print(f"  Test labels:  {y_test.dtype}, {y_test.shape}, "
          f"[{y_test.min()}, {y_test.max()}]")
    
    # 2. 전처리
    print("\n[Step 2] Preprocessing...")
    x_train, y_train = preprocess_mnist(x_train, y_train)
    x_test, y_test = preprocess_mnist(x_test, y_test)
    
    print(f"  Train images: {x_train.dtype}, {x_train.shape}, "
          f"[{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"  Train labels: {y_train.dtype}, {y_train.shape}, "
          f"[{y_train.min()}, {y_train.max()}]")
    print(f"  Test images:  {x_test.dtype}, {x_test.shape}, "
          f"[{x_test.min():.2f}, {x_test.max():.2f}]")
    print(f"  Test labels:  {y_test.dtype}, {y_test.shape}, "
          f"[{y_test.min()}, {y_test.max()}]")
    
    # 3. 샘플 확인
    print("\n[Step 3] Sample verification...")
    sample_idx = 0
    print(f"  Sample {sample_idx}:")
    print(f"    Image shape: {x_train[sample_idx].shape}")
    print(f"    Label (one-hot): {y_train[sample_idx]}")
    print(f"    Label (class): {y_train[sample_idx].argmax()}")
    
    return x_train, y_train, x_test, y_test

# 실행 (실제 데이터가 있을 때)
# x_train, y_train, x_test, y_test = load_and_preprocess_example()
```

```
============================================================
MNIST Data Loading and Preprocessing
============================================================

[Step 1] Loading raw data...
  Train images: uint8, (60000, 28, 28), [0, 255]
  Train labels: uint8, (60000,), [0, 9]
  Test images:  uint8, (10000, 28, 28), [0, 255]
  Test labels:  uint8, (10000,), [0, 9]

[Step 2] Preprocessing...
  Train images: float32, (60000, 784), [0.00, 1.00]
  Train labels: int64, (60000, 10), [0, 1]
  Test images:  float32, (10000, 784), [0.00, 1.00]
  Test labels:  int64, (10000, 10), [0, 1]

[Step 3] Sample verification...
  Sample 0:
    Image shape: (784,)
    Label (one-hot): [0 0 0 0 0 1 0 0 0 0]
    Label (class): 5
```

### 7.3.5 Data Visualization

```python
def visualize_mnist_samples(images, labels, num_samples=10):
    """
    MNIST 샘플 시각화
    
    Parameters:
    -----------
    images : ndarray, shape (N, 28, 28) or (N, 784)
        이미지 데이터
    labels : ndarray, shape (N,) or (N, 10)
        레이블 데이터
    num_samples : int
        표시할 샘플 개수
    """
    import matplotlib.pyplot as plt
    
    # 레이블 형식 확인
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)
    
    # 이미지 형식 확인
    if images.ndim == 2:
        images = images.reshape(-1, 28, 28)
    
    # 시각화
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nMNIST 샘플 이미지가 'mnist_samples.png'로 저장되었습니다.")


# 사용 예시 (실제 데이터가 있을 때)
# visualize_mnist_samples(x_train[:10], y_train[:10])
```

### 7.3.6 Mini-batch Generation

```python
def create_mini_batches(x, y, batch_size=64, shuffle=True):
    """
    미니배치 생성기
    
    Parameters:
    -----------
    x : ndarray
        입력 데이터
    y : ndarray
        레이블 데이터
    batch_size : int
        배치 크기
    shuffle : bool
        데이터 섞기 여부
    
    Yields:
    -------
    x_batch, y_batch : tuple of ndarrays
        미니배치 데이터
    """
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield x[batch_indices], y[batch_indices]


# 사용 예시
def batch_generation_example():
    """배치 생성 예시"""
    
    # 더미 데이터
    x_dummy = np.random.randn(1000, 784)
    y_dummy = np.eye(10)[np.random.randint(0, 10, 1000)]
    
    print("\n" + "=" * 60)
    print("Mini-batch Generation Example")
    print("=" * 60)
    
    batch_size = 64
    num_batches = 0
    
    for x_batch, y_batch in create_mini_batches(x_dummy, y_dummy, batch_size=batch_size):
        num_batches += 1
        if num_batches <= 3:
            print(f"\nBatch {num_batches}:")
            print(f"  x_batch shape: {x_batch.shape}")
            print(f"  y_batch shape: {y_batch.shape}")
    
    print(f"\nTotal batches: {num_batches}")
    print(f"Expected: {int(np.ceil(1000 / batch_size))}")

batch_generation_example()
```

```
============================================================
Mini-batch Generation Example
============================================================

Batch 1:
  x_batch shape: (64, 784)
  y_batch shape: (64, 10)

Batch 2:
  x_batch shape: (64, 784)
  y_batch shape: (64, 10)

Batch 3:
  x_batch shape: (64, 784)
  y_batch shape: (64, 10)

Total batches: 16
Expected: 16
```

### 7.3.7 Data Statistics

```python
def compute_dataset_statistics(x_train, y_train):
    """데이터셋 통계 계산"""
    
    # 레이블을 클래스 인덱스로 변환
    if y_train.ndim == 2:
        y_train = y_train.argmax(axis=1)
    
    print("\n" + "=" * 60)
    print("MNIST Dataset Statistics")
    print("=" * 60)
    
    # 전체 통계
    print("\n[Image Statistics]")
    print(f"  Mean:     {x_train.mean():.4f}")
    print(f"  Std:      {x_train.std():.4f}")
    print(f"  Min:      {x_train.min():.4f}")
    print(f"  Max:      {x_train.max():.4f}")
    print(f"  Median:   {np.median(x_train):.4f}")
    
    # 클래스별 통계
    print("\n[Class Statistics]")
    print(f"{'Class':<10} {'Count':<10} {'Percentage':<15}")
    print("-" * 60)
    
    for cls in range(10):
        count = np.sum(y_train == cls)
        percentage = (count / len(y_train)) * 100
        print(f"{cls:<10} {count:<10} {percentage:>6.2f}%")
    
    print("=" * 60)


# 사용 예시 (실제 데이터가 있을 때)
# compute_dataset_statistics(x_train, y_train)
```

### 7.3.8 Summary

| 항목 | 훈련 데이터 | 테스트 데이터 |
|------|------------|--------------|
| **샘플 수** | 60,000 | 10,000 |
| **이미지 크기** | 28×28 | 28×28 |
| **픽셀 범위 (원본)** | [0, 255] | [0, 255] |
| **픽셀 범위 (정규화)** | [0.0, 1.0] | [0.0, 1.0] |
| **클래스 개수** | 10 | 10 |
| **레이블 형식** | 0~9 또는 one-hot | 0~9 또는 one-hot |

**전처리 파이프라인:**

| 단계 | 변환 | 목적 |
|------|------|------|
| **1. Type casting** | uint8 → float32 | 연산 정밀도 향상 |
| **2. Normalization** | /255 | [0, 1] 범위로 정규화 |
| **3. Flattening** | (28, 28) → (784,) | MLP 입력 형태 |
| **4. One-hot encoding** | 5 → [0,0,0,0,0,1,0,0,0,0] | 다중 클래스 분류용 |

**주요 함수:**

```python
# 데이터 로딩
images, labels = get_mnist(data_dir, split="train")

# 전처리
x, y = preprocess_mnist(images, labels, 
                        flatten=True, 
                        normalize=True, 
                        onehot=True)

# 배치 생성
for x_batch, y_batch in create_mini_batches(x, y, batch_size=64):
    # 학습 로직
    pass
```

**파일 구조:**

```
data_dir/
├── train-images-idx3-ubyte.gz  (훈련 이미지)
├── train-labels-idx1-ubyte.gz  (훈련 레이블)
├── t10k-images-idx3-ubyte.gz   (테스트 이미지)
└── t10k-labels-idx1-ubyte.gz   (테스트 레이블)
```

**핵심 특징:**

- 균등한 클래스 분포 (각 클래스 약 10%)
- 간단하고 명확한 이미지 (28×28 grayscale)
- 전처리가 용이함 (정규화, 평탄화)
- 빠른 학습 가능 (작은 이미지 크기)
- 다중 클래스 분류의 표준 벤치마크
