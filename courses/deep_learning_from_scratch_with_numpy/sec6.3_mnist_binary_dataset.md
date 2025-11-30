## 6.3 MNIST Binary Dataset (0 vs 1)

MNIST 데이터셋에서 숫자 0과 1만을 선택하여 이진 분류 문제를 구성합니다. 이는 다중 클래스 분류를 이진 분류로 단순화한 것입니다.

### 6.3.1 Dataset Construction

**원본 MNIST에서 추출:**

- **전체 MNIST**: 10개 클래스 (0~9)
- **이진 MNIST**: 2개 클래스 (0, 1)

**데이터 필터링:**

```python
import numpy as np

def create_binary_mnist(images, labels, class_0=0, class_1=1):
    """
    MNIST에서 두 개의 클래스만 선택하여 이진 분류 데이터셋 생성
    
    Parameters:
    -----------
    images : ndarray, shape (N, 28, 28)
        전체 MNIST 이미지
    labels : ndarray, shape (N,)
        전체 MNIST 레이블 (0~9)
    class_0 : int
        음성 클래스 (0으로 매핑)
    class_1 : int
        양성 클래스 (1로 매핑)
    
    Returns:
    --------
    binary_images : ndarray
        필터링된 이미지
    binary_labels : ndarray
        이진 레이블 (0 또는 1)
    """
    # 선택한 두 클래스에 해당하는 인덱스 찾기
    mask = (labels == class_0) | (labels == class_1)
    
    # 이미지 필터링
    binary_images = images[mask]
    
    # 레이블 필터링 및 이진화
    binary_labels = labels[mask]
    binary_labels = (binary_labels == class_1).astype(np.int64)
    
    return binary_images, binary_labels
```

### 6.3.2 Dataset Statistics

```python
def analyze_binary_mnist_distribution(labels):
    """이진 MNIST 데이터셋 통계"""
    
    class_0_count = np.sum(labels == 0)
    class_1_count = np.sum(labels == 1)
    total_count = len(labels)
    
    print("=" * 60)
    print("Binary MNIST Dataset Statistics")
    print("=" * 60)
    print(f"\nTotal samples: {total_count:,}")
    print(f"\nClass distribution:")
    print(f"  Class 0 (digit 0): {class_0_count:,} samples ({class_0_count/total_count*100:.1f}%)")
    print(f"  Class 1 (digit 1): {class_1_count:,} samples ({class_1_count/total_count*100:.1f}%)")
    print(f"\nClass balance ratio: {class_0_count/class_1_count:.3f}")
    print("=" * 60)


# 예상 통계 (실제 MNIST 기준)
# 훈련 데이터: 0 → 5,923개, 1 → 6,742개
# 테스트 데이터: 0 → 980개, 1 → 1,135개
```

```
============================================================
Binary MNIST Dataset Statistics
============================================================

Total samples: 12,665

Class distribution:
  Class 0 (digit 0): 5,923 samples (46.8%)
  Class 1 (digit 1): 6,742 samples (53.2%)

Class balance ratio: 0.878
============================================================
```

### 6.3.3 Complete Data Loading Pipeline

```python
import os
import gzip

def load_mnist_images(data_dir, filename):
    """MNIST 이미지 로딩"""
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(data_dir, filename):
    """MNIST 레이블 로딩"""
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def get_binary_mnist(data_dir, split="train", class_0=0, class_1=1):
    """
    이진 분류용 MNIST 데이터셋 로딩
    
    Parameters:
    -----------
    data_dir : str
        데이터 디렉토리 경로
    split : str
        'train' 또는 'test'
    class_0, class_1 : int
        선택할 두 클래스
    
    Returns:
    --------
    images, labels : tuple of ndarrays
    """
    # 전체 MNIST 로딩
    if split == "train":
        images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    elif split == "test":
        images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
    else:
        raise ValueError("split must be 'train' or 'test'!")
    
    # 이진 분류 데이터셋 생성
    binary_images, binary_labels = create_binary_mnist(
        images, labels, class_0, class_1
    )
    
    return binary_images, binary_labels
```

### 6.3.4 Data Preprocessing

```python
def preprocess_binary_mnist(images, labels, flatten=True, normalize=True):
    """
    이진 MNIST 데이터 전처리
    
    Parameters:
    -----------
    images : ndarray, shape (N, 28, 28)
        원본 이미지
    labels : ndarray, shape (N,)
        이진 레이블 (0 또는 1)
    flatten : bool
        이미지를 1D 벡터로 평탄화 여부
    normalize : bool
        [0, 1] 범위로 정규화 여부
    
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
    
    # 레이블 전처리 (이미 0 또는 1)
    labels_processed = labels.astype(np.float32).reshape(-1, 1)
    
    return images_processed, labels_processed
```

### 6.3.5 Complete Loading Example

```python
def load_binary_mnist_example():
    """이진 MNIST 로딩 및 전처리 전체 예제"""
    
    # 데이터 디렉토리 설정
    data_dir = "/mnt/d/datasets/mnist"
    
    print("\n" + "=" * 60)
    print("Binary MNIST (0 vs 1) Loading and Preprocessing")
    print("=" * 60)
    
    # 1. 원본 데이터 로딩
    print("\n[Step 1] Loading raw binary data...")
    x_train, y_train = get_binary_mnist(data_dir, split="train", 
                                        class_0=0, class_1=1)
    x_test, y_test = get_binary_mnist(data_dir, split="test", 
                                      class_0=0, class_1=1)
    
    print(f"  Train images: {x_train.dtype}, {x_train.shape}, "
          f"[{x_train.min()}, {x_train.max()}]")
    print(f"  Train labels: {y_train.dtype}, {y_train.shape}, "
          f"[{y_train.min()}, {y_train.max()}]")
    print(f"  Test images:  {x_test.dtype}, {x_test.shape}, "
          f"[{x_test.min()}, {x_test.max()}]")
    print(f"  Test labels:  {y_test.dtype}, {y_test.shape}, "
          f"[{y_test.min()}, {y_test.max()}]")
    
    # 2. 클래스 분포 확인
    print("\n[Step 2] Analyzing class distribution...")
    print(f"  Train - Class 0: {np.sum(y_train == 0):,}, "
          f"Class 1: {np.sum(y_train == 1):,}")
    print(f"  Test  - Class 0: {np.sum(y_test == 0):,}, "
          f"Class 1: {np.sum(y_test == 1):,}")
    
    # 3. 전처리
    print("\n[Step 3] Preprocessing...")
    x_train, y_train = preprocess_binary_mnist(x_train, y_train)
    x_test, y_test = preprocess_binary_mnist(x_test, y_test)
    
    print(f"  Train images: {x_train.dtype}, {x_train.shape}, "
          f"[{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"  Train labels: {y_train.dtype}, {y_train.shape}, "
          f"[{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  Test images:  {x_test.dtype}, {x_test.shape}, "
          f"[{x_test.min():.2f}, {x_test.max():.2f}]")
    print(f"  Test labels:  {y_test.dtype}, {y_test.shape}, "
          f"[{y_test.min():.0f}, {y_test.max():.0f}]")
    
    # 4. 샘플 확인
    print("\n[Step 4] Sample verification...")
    for i in range(3):
        digit = 0 if y_train[i] == 0 else 1
        print(f"  Sample {i}: Label = {y_train[i, 0]:.0f} (digit {digit})")
    
    print("=" * 60)
    
    return x_train, y_train, x_test, y_test

# 실행 예제 (실제 데이터가 있을 때)
# x_train, y_train, x_test, y_test = load_binary_mnist_example()
```

```
============================================================
Binary MNIST (0 vs 1) Loading and Preprocessing
============================================================

[Step 1] Loading raw binary data...
  Train images: uint8, (12665, 28, 28), [0, 255]
  Train labels: int64, (12665,), [0, 1]
  Test images:  uint8, (2115, 28, 28), [0, 255]
  Test labels:  int64, (2115,), [0, 1]

[Step 2] Analyzing class distribution...
  Train - Class 0: 5,923, Class 1: 6,742
  Test  - Class 0: 980, Class 1: 1,135

[Step 3] Preprocessing...
  Train images: float32, (12665, 784), [0.00, 1.00]
  Train labels: float32, (12665, 1), [0, 1]
  Test images:  float32, (2115, 784), [0.00, 1.00]
  Test labels:  float32, (2115, 1), [0, 1]

[Step 4] Sample verification...
  Sample 0: Label = 0 (digit 0)
  Sample 1: Label = 0 (digit 0)
  Sample 2: Label = 0 (digit 0)
============================================================
```

### 6.3.6 Data Visualization

```python
def visualize_binary_mnist_samples(images, labels, num_samples=10):
    """
    이진 MNIST 샘플 시각화
    
    Parameters:
    -----------
    images : ndarray, shape (N, 28, 28) or (N, 784)
        이미지 데이터
    labels : ndarray, shape (N,) or (N, 1)
        레이블 데이터
    num_samples : int
        표시할 샘플 개수
    """
    import matplotlib.pyplot as plt
    
    # 레이블 형식 확인
    labels = labels.flatten()
    
    # 이미지 형식 확인
    if images.ndim == 2:
        images = images.reshape(-1, 28, 28)
    
    # 각 클래스에서 샘플 선택
    class_0_indices = np.where(labels == 0)[0][:num_samples//2]
    class_1_indices = np.where(labels == 1)[0][:num_samples//2]
    
    selected_indices = np.concatenate([class_0_indices, class_1_indices])
    
    # 시각화
    fig, axes = plt.subplots(2, num_samples//2, figsize=(12, 5))
    
    for i, idx in enumerate(selected_indices):
        row = i // (num_samples // 2)
        col = i % (num_samples // 2)
        
        axes[row, col].imshow(images[idx], cmap='gray')
        digit = 0 if labels[idx] == 0 else 1
        axes[row, col].set_title(f'Digit: {digit}\nLabel: {int(labels[idx])}', 
                                 fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
    
    axes[0, 0].set_ylabel('Class 0\n(Digit 0)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Class 1\n(Digit 1)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('binary_mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n이진 MNIST 샘플 이미지가 'binary_mnist_samples.png'로 저장되었습니다.")


# 사용 예시
# visualize_binary_mnist_samples(x_train, y_train, num_samples=10)
```

### 6.3.7 Class Balance Check

```python
def check_class_balance(y_train, y_test):
    """클래스 균형 확인 및 시각화"""
    
    import matplotlib.pyplot as plt
    
    # 레이블 평탄화
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # 클래스별 샘플 수
    train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
    test_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train distribution
    axes[0].bar(['Class 0\n(Digit 0)', 'Class 1\n(Digit 1)'], train_counts, 
                color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Training Set Class Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(train_counts):
        axes[0].text(i, count + 100, f'{count:,}\n({count/sum(train_counts)*100:.1f}%)', 
                    ha='center', fontsize=11, fontweight='bold')
    
    # Test distribution
    axes[1].bar(['Class 0\n(Digit 0)', 'Class 1\n(Digit 1)'], test_counts, 
                color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_title('Test Set Class Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(test_counts):
        axes[1].text(i, count + 20, f'{count:,}\n({count/sum(test_counts)*100:.1f}%)', 
                    ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('binary_mnist_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n클래스 분포 그래프가 'binary_mnist_distribution.png'로 저장되었습니다.")
    
    # 불균형 정도 평가
    print("\n" + "=" * 60)
    print("Class Balance Assessment")
    print("=" * 60)
    
    train_ratio = min(train_counts) / max(train_counts)
    test_ratio = min(test_counts) / max(test_counts)
    
    print(f"Training set balance ratio: {train_ratio:.3f}")
    print(f"Test set balance ratio:     {test_ratio:.3f}")
    
    if train_ratio > 0.9:
        print("\nAssessment: Well-balanced dataset ✓")
    elif train_ratio > 0.7:
        print("\nAssessment: Slightly imbalanced (acceptable)")
    else:
        print("\nAssessment: Imbalanced dataset (consider resampling)")
    
    print("=" * 60)


# 사용 예시
# check_class_balance(y_train, y_test)
```

```
클래스 분포 그래프가 'binary_mnist_distribution.png'로 저장되었습니다.

============================================================
Class Balance Assessment
============================================================
Training set balance ratio: 0.878
Test set balance ratio:     0.863

Assessment: Slightly imbalanced (acceptable)
============================================================
```

### 6.3.8 Mini-batch Generation for Binary Classification

```python
def create_binary_mini_batches(x, y, batch_size=64, shuffle=True):
    """
    이진 분류용 미니배치 생성기
    
    Parameters:
    -----------
    x : ndarray, shape (N, 784)
        입력 데이터
    y : ndarray, shape (N, 1) or (N,)
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
        
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        
        # Shape 확인
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)
        
        yield x_batch, y_batch


# 사용 예시
def batch_generation_binary_example():
    """이진 분류 배치 생성 예시"""
    
    # 더미 데이터
    x_dummy = np.random.randn(1000, 784)
    y_dummy = np.random.randint(0, 2, (1000, 1)).astype(np.float32)
    
    print("\n" + "=" * 60)
    print("Binary Classification Mini-batch Generation")
    print("=" * 60)
    
    batch_size = 64
    num_batches = 0
    
    class_0_total = 0
    class_1_total = 0
    
    for x_batch, y_batch in create_binary_mini_batches(x_dummy, y_dummy, 
                                                        batch_size=batch_size):
        num_batches += 1
        
        class_0_in_batch = np.sum(y_batch == 0)
        class_1_in_batch = np.sum(y_batch == 1)
        
        class_0_total += class_0_in_batch
        class_1_total += class_1_in_batch
        
        if num_batches <= 3:
            print(f"\nBatch {num_batches}:")
            print(f"  x_batch shape: {x_batch.shape}")
            print(f"  y_batch shape: {y_batch.shape}")
            print(f"  Class 0: {class_0_in_batch}, Class 1: {class_1_in_batch}")
    
    print(f"\nTotal batches: {num_batches}")
    print(f"Total samples: Class 0 = {class_0_total}, Class 1 = {class_1_total}")

batch_generation_binary_example()
```

```
============================================================
Binary Classification Mini-batch Generation
============================================================

Batch 1:
  x_batch shape: (64, 784)
  y_batch shape: (64, 1)
  Class 0: 31, Class 1: 33

Batch 2:
  x_batch shape: (64, 784)
  y_batch shape: (64, 1)
  Class 0: 28, Class 1: 36

Batch 3:
  x_batch shape: (64, 784)
  y_batch shape: (64, 1)
  Class 0: 35, Class 1: 29

Total batches: 16
Total samples: Class 0 = 502, Class 1 = 498
```

### 6.3.9 Summary

| 항목 | 훈련 데이터 | 테스트 데이터 |
|------|------------|--------------|
| **총 샘플 수** | 12,665 | 2,115 |
| **Class 0 (digit 0)** | 5,923 (46.8%) | 980 (46.3%) |
| **Class 1 (digit 1)** | 6,742 (53.2%) | 1,135 (53.7%) |
| **이미지 크기** | 28×28 | 28×28 |
| **픽셀 범위 (정규화)** | [0.0, 1.0] | [0.0, 1.0] |
| **레이블 형식** | 0 또는 1 (float32) | 0 또는 1 (float32) |

**전처리 파이프라인:**

| 단계 | 변환 | 목적 |
|------|------|------|
| **1. Class filtering** | 10 classes → 2 classes | 이진 분류 문제로 단순화 |
| **2. Label mapping** | {0, 1} → {0, 1} | 이진 레이블 생성 |
| **3. Type casting** | uint8 → float32 | 연산 정밀도 향상 |
| **4. Normalization** | /255 | [0, 1] 범위로 정규화 |
| **5. Flattening** | (28, 28) → (784,) | MLP 입력 형태 |
| **6. Reshape labels** | (N,) → (N, 1) | 이진 분류 출력 형태 |

**주요 함수:**

```python
# 이진 데이터셋 생성
binary_images, binary_labels = create_binary_mnist(images, labels, 
                                                    class_0=0, class_1=1)

# 데이터 로딩
x_train, y_train = get_binary_mnist(data_dir, split="train")

# 전처리
x, y = preprocess_binary_mnist(images, labels, 
                                flatten=True, 
                                normalize=True)

# 배치 생성
for x_batch, y_batch in create_binary_mini_batches(x, y, batch_size=64):
    # 학습 로직
    pass
```

**다중 클래스 MNIST와의 차이점:**

| 특성 | 이진 MNIST (0 vs 1) | 다중 클래스 MNIST |
|------|---------------------|-------------------|
| 클래스 수 | 2개 | 10개 |
| 샘플 수 (train) | 12,665 | 60,000 |
| 샘플 수 (test) | 2,115 | 10,000 |
| 레이블 형식 | 스칼라 (0 또는 1) | One-hot (10차원) |
| 출력 차원 | 1 | 10 |
| 활성화 함수 | Sigmoid | Softmax |
| 손실 함수 | Binary CE | Categorical CE |

**핵심 특징:**

- 균형잡힌 클래스 분포 (약 47% vs 53%)
- 단순한 문제 (두 개의 명확히 구분되는 숫자)
- 빠른 학습 가능 (데이터셋 크기 약 1/5)
- 이진 분류의 기본 개념 학습에 적합
