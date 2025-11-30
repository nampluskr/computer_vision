## 9.2 Version 2 - DataLoader

Version 2에서는 데이터 로딩과 배치 생성을 추상화하여 코드의 재사용성과 유연성을 높입니다. PyTorch의 DataLoader와 유사한 인터페이스를 구현합니다.

### 9.2.1 Overview

**Version 2의 개선 사항:**

- DataLoader 클래스 도입
- 미니배치 생성 자동화
- 데이터 셔플링 캡슐화
- 반복 가능한(iterable) 인터페이스

**Version 1 → Version 2 변화:**

```python
# Version 1: 수동 배치 생성
indices = np.random.permutation(len(x_train))
for i in range(0, len(x_train), batch_size):
    x = x_train[indices[i: i + batch_size]]
    y = y_train[indices[i: i + batch_size]]
    # ...

# Version 2: DataLoader 사용
train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
for x_batch, y_batch in train_loader:
    # ...
```

### 9.2.2 DataLoader Implementation

```python
import numpy as np

class DataLoader:
    """
    PyTorch-style DataLoader
    
    데이터를 미니배치로 나누어 반복 제공
    """
    
    def __init__(self, x, y, batch_size=32, shuffle=True, drop_last=False):
        """
        Parameters:
        -----------
        x : ndarray
            입력 데이터
        y : ndarray
            타겟 데이터
        batch_size : int
            배치 크기
        shuffle : bool
            에포크마다 데이터 셔플 여부
        drop_last : bool
            마지막 불완전한 배치 버릴지 여부
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.num_samples = len(x)
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1
    
    def __iter__(self):
        """반복자(iterator) 초기화"""
        if self.shuffle:
            self.indices = np.random.permutation(self.num_samples)
        else:
            self.indices = np.arange(self.num_samples)
        
        self.current_idx = 0
        return self
    
    def __next__(self):
        """다음 배치 반환"""
        if self.current_idx >= self.num_samples:
            raise StopIteration
        
        # 배치 인덱스 계산
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        # drop_last 처리
        if self.drop_last and end_idx - start_idx < self.batch_size:
            raise StopIteration
        
        # 배치 추출
        batch_indices = self.indices[start_idx:end_idx]
        x_batch = self.x[batch_indices]
        y_batch = self.y[batch_indices]
        
        self.current_idx = end_idx
        
        return x_batch, y_batch
    
    def __len__(self):
        """전체 배치 개수 반환"""
        return self.num_batches


# 사용 예시
if __name__ == "__main__":
    # 더미 데이터
    x = np.random.randn(100, 784)
    y = np.random.randint(0, 10, (100, 10))
    
    # DataLoader 생성
    loader = DataLoader(x, y, batch_size=32, shuffle=True)
    
    print(f"Total samples: {loader.num_samples}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches: {len(loader)}")
    
    # 배치 반복
    print("\nIterating through batches:")
    for i, (x_batch, y_batch) in enumerate(loader):
        print(f"Batch {i+1}: x={x_batch.shape}, y={y_batch.shape}")
```

```
Total samples: 100
Batch size: 32
Number of batches: 4

Iterating through batches:
Batch 1: x=(32, 784), y=(32, 10)
Batch 2: x=(32, 784), y=(32, 10)
Batch 3: x=(32, 784), y=(32, 10)
Batch 4: x=(4, 784), y=(4, 10)
```

### 9.2.3 Complete Version 2 Code

```python
import os
import numpy as np
import gzip

#################################################################
## DataLoader Class
#################################################################

class DataLoader:
    """미니배치 데이터 로더"""
    
    def __init__(self, x, y, batch_size=32, shuffle=True, drop_last=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.num_samples = len(x)
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1
    
    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.num_samples)
        else:
            self.indices = np.arange(self.num_samples)
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        if self.drop_last and end_idx - start_idx < self.batch_size:
            raise StopIteration
        
        batch_indices = self.indices[start_idx:end_idx]
        x_batch = self.x[batch_indices]
        y_batch = self.y[batch_indices]
        
        self.current_idx = end_idx
        return x_batch, y_batch
    
    def __len__(self):
        return self.num_batches


#################################################################
## Data Loading Functions
#################################################################

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def get_mnist(data_dir, split="train"):
    if split == "train":
        images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    elif split == "test":
        images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
    else:
        raise ValueError(">> split must be train or test!")
    return images, labels


#################################################################
## Math Functions
#################################################################

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    if targets.ndim == 1:
        batch_size = preds.shape[0]
        probs = preds[np.arange(batch_size), targets]
    else:
        probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))


def accuracy(preds, targets):
    preds = preds.argmax(axis=1)
    if targets.ndim == 2:
        targets = targets.argmax(axis=1)
    return (preds == targets).mean()


#################################################################
## Training Script with DataLoader
#################################################################

if __name__ == "__main__":
    
    np.random.seed(42)
    
    # Data Loading
    data_dir = "/mnt/d/datasets/mnist"
    x_train, y_train = get_mnist(data_dir, split="train")
    x_test, y_test = get_mnist(data_dir, split="test")
    
    print("\n>> Data before preprocessing:")
    print(f"train images: {x_train.dtype}, {x_train.shape}")
    
    # Preprocessing
    def preprocess(images, labels):
        images = images.astype(np.float32).reshape(-1, 28*28) / 255
        labels = one_hot(labels, num_classes=10).astype(np.int64)
        return images, labels
    
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    
    print("\n>> Data after preprocessing:")
    print(f"train images: {x_train.dtype}, {x_train.shape}")
    
    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(x_train, y_train, batch_size=batch_size, 
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(x_test, y_test, batch_size=batch_size, 
                             shuffle=False, drop_last=False)
    
    print(f"\n>> DataLoaders created:")
    print(f"Train: {len(train_loader)} batches")
    print(f"Test:  {len(test_loader)} batches")
    
    # Network Initialization
    input_size, hidden_size, output_size = 28*28, 100, 10
    
    w1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    w2 = np.random.randn(hidden_size, hidden_size)
    b2 = np.zeros(hidden_size)
    w3 = np.random.randn(hidden_size, output_size)
    b3 = np.zeros(output_size)
    
    # Training Loop with DataLoader
    num_epochs = 10
    learning_rate = 0.01
    
    print("\n>> Training start ...")
    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        for x_batch, y_batch in train_loader:
            batch_size_actual = x_batch.shape[0]
            train_samples += batch_size_actual
            
            # Forward
            z1 = np.dot(x_batch, w1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, w2) + b2
            a2 = sigmoid(z2)
            z3 = np.dot(a2, w3) + b3
            preds = softmax(z3)
            
            loss = cross_entropy(preds, y_batch)
            acc = accuracy(preds, y_batch)
            
            # Backward
            grad_z3 = (preds - y_batch) / y_batch.shape[0]
            grad_w3 = np.dot(a2.T, grad_z3)
            grad_b3 = np.sum(grad_z3, axis=0)
            
            grad_a2 = np.dot(grad_z3, w3.T)
            grad_z2 = a2 * (1 - a2) * grad_a2
            grad_w2 = np.dot(a1.T, grad_z2)
            grad_b2 = np.sum(grad_z2, axis=0)
            
            grad_a1 = np.dot(grad_z2, w2.T)
            grad_z1 = a1 * (1 - a1) * grad_a1
            grad_w1 = np.dot(x_batch.T, grad_z1)
            grad_b1 = np.sum(grad_z1, axis=0)
            
            # Update
            w1 -= learning_rate * grad_w1
            b1 -= learning_rate * grad_b1
            w2 -= learning_rate * grad_w2
            b2 -= learning_rate * grad_b2
            w3 -= learning_rate * grad_w3
            b3 -= learning_rate * grad_b3
            
            train_loss += loss * batch_size_actual
            train_acc += acc * batch_size_actual
        
        print(f"[{epoch:3d}/{num_epochs}] "
              f"loss:{train_loss/train_samples:.3f} "
              f"acc:{train_acc/train_samples:.3f}")
    
    # Evaluation with DataLoader
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    for x_batch, y_batch in test_loader:
        batch_size_actual = x_batch.shape[0]
        test_samples += batch_size_actual
        
        z1 = np.dot(x_batch, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        out = softmax(z3)
        
        loss = cross_entropy(out, y_batch)
        acc = accuracy(out, y_batch)
        
        test_loss += loss * batch_size_actual
        test_acc += acc * batch_size_actual
    
    print(f"\n>> Evaluation: loss:{test_loss/test_samples:.3f} "
          f"acc:{test_acc/test_samples:.3f}")
```

### 9.2.4 Key Improvements

```python
def compare_v1_v2():
    """Version 1과 Version 2 비교"""
    
    print("\n" + "=" * 70)
    print("Version 1 vs Version 2 Comparison")
    print("=" * 70)
    
    comparison = {
        "Batch Generation": {
            "Version 1": "수동으로 인덱스 생성 및 슬라이싱",
            "Version 2": "DataLoader가 자동 처리"
        },
        "Code Repetition": {
            "Version 1": "학습/평가에서 배치 생성 코드 중복",
            "Version 2": "DataLoader 재사용으로 중복 제거"
        },
        "Shuffling": {
            "Version 1": "매 에포크마다 수동 permutation",
            "Version 2": "DataLoader가 자동 처리"
        },
        "Iterator Interface": {
            "Version 1": "for i in range(...) 사용",
            "Version 2": "for x, y in loader 사용 (Pythonic)"
        },
        "Flexibility": {
            "Version 1": "배치 크기 변경 시 여러 곳 수정",
            "Version 2": "DataLoader 파라미터만 변경"
        },
        "Drop Last": {
            "Version 1": "구현되지 않음",
            "Version 2": "drop_last 파라미터로 제어"
        }
    }
    
    for aspect, versions in comparison.items():
        print(f"\n[{aspect}]")
        for version, description in versions.items():
            print(f"  {version}: {description}")
    
    print("=" * 70)

compare_v1_v2()
```

```
======================================================================
Version 1 vs Version 2 Comparison
======================================================================

[Batch Generation]
  Version 1: 수동으로 인덱스 생성 및 슬라이싱
  Version 2: DataLoader가 자동 처리

[Code Repetition]
  Version 1: 학습/평가에서 배치 생성 코드 중복
  Version 2: DataLoader 재사용으로 중복 제거

[Shuffling]
  Version 1: 매 에포크마다 수동 permutation
  Version 2: DataLoader가 자동 처리

[Iterator Interface]
  Version 1: for i in range(...) 사용
  Version 2: for x, y in loader 사용 (Pythonic)

[Flexibility]
  Version 1: 배치 크기 변경 시 여러 곳 수정
  Version 2: DataLoader 파라미터만 변경

[Drop Last]
  Version 1: 구현되지 않음
  Version 2: drop_last 파라미터로 제어
======================================================================
```

### 9.2.5 DataLoader Features

```python
def demonstrate_dataloader_features():
    """DataLoader의 다양한 기능 시연"""
    
    print("\n" + "=" * 70)
    print("DataLoader Features Demonstration")
    print("=" * 70)
    
    # 더미 데이터
    x = np.arange(100).reshape(100, 1)
    y = np.arange(100).reshape(100, 1)
    
    # Feature 1: Shuffling
    print("\n[Feature 1: Shuffling]")
    loader_shuffle = DataLoader(x, y, batch_size=10, shuffle=True)
    loader_no_shuffle = DataLoader(x, y, batch_size=10, shuffle=False)
    
    print("With shuffle=True:")
    for i, (x_batch, _) in enumerate(loader_shuffle):
        if i == 0:
            print(f"  First batch indices: {x_batch.flatten()[:5]}...")
        break
    
    print("With shuffle=False:")
    for i, (x_batch, _) in enumerate(loader_no_shuffle):
        if i == 0:
            print(f"  First batch indices: {x_batch.flatten()[:5]}...")
        break
    
    # Feature 2: Drop Last
    print("\n[Feature 2: Drop Last]")
    loader_keep = DataLoader(x, y, batch_size=30, drop_last=False)
    loader_drop = DataLoader(x, y, batch_size=30, drop_last=True)
    
    print(f"drop_last=False: {len(loader_keep)} batches (keeps last incomplete batch)")
    print(f"drop_last=True:  {len(loader_drop)} batches (drops last incomplete batch)")
    
    # Feature 3: Variable Batch Size
    print("\n[Feature 3: Iteration Example]")
    loader = DataLoader(x, y, batch_size=25, shuffle=False)
    
    for i, (x_batch, y_batch) in enumerate(loader):
        print(f"  Batch {i+1}: size={x_batch.shape[0]}")
    
    # Feature 4: Multiple Epochs
    print("\n[Feature 4: Multiple Epochs with Different Shuffles]")
    loader = DataLoader(x[:20], y[:20], batch_size=20, shuffle=True)
    
    for epoch in range(1, 4):
        for x_batch, _ in loader:
            print(f"  Epoch {epoch}: {x_batch.flatten()[:5]}...")
            break
    
    print("=" * 70)

demonstrate_dataloader_features()
```

```
======================================================================
DataLoader Features Demonstration
======================================================================

[Feature 1: Shuffling]
With shuffle=True:
  First batch indices: [73 42 18 91 56]...
With shuffle=False:
  First batch indices: [0 1 2 3 4]...

[Feature 2: Drop Last]
drop_last=False: 4 batches (keeps last incomplete batch)
drop_last=True:  3 batches (drops last incomplete batch)

[Feature 3: Iteration Example]
  Batch 1: size=25
  Batch 2: size=25
  Batch 3: size=25
  Batch 4: size=25

[Feature 4: Multiple Epochs with Different Shuffles]
  Epoch 1: [15  3 11  7 19]...
  Epoch 2: [ 9 14  2 18 10]...
  Epoch 3: [12  8  5  1 17]...
======================================================================
```

### 9.2.6 Benefits of DataLoader

```python
def analyze_v2_benefits():
    """Version 2의 이점 분석"""
    
    print("\n" + "=" * 70)
    print("Version 2 - Benefits and Improvements")
    print("=" * 70)
    
    benefits = {
        "1. Code Reusability": [
            "✓ 동일한 DataLoader를 학습/평가에서 재사용",
            "✓ 다른 데이터셋에도 쉽게 적용 가능",
            "✓ 배치 생성 로직이 한 곳에 집중"
        ],
        "2. Pythonic Interface": [
            "✓ for x, y in loader: 형태로 간결한 반복",
            "✓ __iter__, __next__ 프로토콜 준수",
            "✓ len(loader)로 배치 수 확인 가능"
        ],
        "3. Flexibility": [
            "✓ batch_size 쉽게 변경",
            "✓ shuffle on/off 간단히 제어",
            "✓ drop_last 옵션으로 마지막 배치 처리"
        ],
        "4. Error Reduction": [
            "✓ 인덱스 계산 오류 가능성 감소",
            "✓ 배치 크기 불일치 자동 처리",
            "✓ 경계 조건 자동 관리"
        ],
        "5. Maintainability": [
            "✓ DataLoader 수정만으로 모든 곳에 적용",
            "✓ 테스트 및 디버깅 용이",
            "✓ 기능 추가 쉬움 (예: multiprocessing)"
        ]
    }
    
    for category, items in benefits.items():
        print(f"\n{category}")
        for item in items:
            print(f"  {item}")
    
    print("=" * 70)

analyze_v2_benefits()
```

```
======================================================================
Version 2 - Benefits and Improvements
======================================================================

1. Code Reusability
  ✓ 동일한 DataLoader를 학습/평가에서 재사용
  ✓ 다른 데이터셋에도 쉽게 적용 가능
  ✓ 배치 생성 로직이 한 곳에 집중

2. Pythonic Interface
  ✓ for x, y in loader: 형태로 간결한 반복
  ✓ __iter__, __next__ 프로토콜 준수
  ✓ len(loader)로 배치 수 확인 가능

3. Flexibility
  ✓ batch_size 쉽게 변경
  ✓ shuffle on/off 간단히 제어
  ✓ drop_last 옵션으로 마지막 배치 처리

4. Error Reduction
  ✓ 인덱스 계산 오류 가능성 감소
  ✓ 배치 크기 불일치 자동 처리
  ✓ 경계 조건 자동 관리

5. Maintainability
  ✓ DataLoader 수정만으로 모든 곳에 적용
  ✓ 테스트 및 디버깅 용이
  ✓ 기능 추가 쉬움 (예: multiprocessing)
======================================================================
```

### 9.2.7 Remaining Problems

```python
def analyze_v2_limitations():
    """Version 2의 남은 문제점"""
    
    print("\n" + "=" * 70)
    print("Version 2 - Remaining Problems")
    print("=" * 70)
    
    problems = {
        "Still Present from V1": [
            "순전파/역전파가 여전히 명시적",
            "네트워크 구조 변경 어려움",
            "파라미터가 개별 변수로 존재",
            "레이어 추상화 없음",
            "옵티마이저 하드코딩"
        ],
        "New Improvements Needed": [
            "레이어를 모듈로 캡슐화",
            "파라미터 관리 자동화",
            "네트워크를 클래스로 구조화",
            "다양한 활성화 함수 지원",
            "모델 저장/로딩 기능"
        ]
    }
    
    for category, items in problems.items():
        print(f"\n[{category}]")
        for item in items:
            print(f"  • {item}")
    
    print("\n" + "=" * 70)
    print("다음 단계 (Version 3): Module 추상화")
    print("  → 레이어를 클래스로 캡슐화")
    print("  → 네트워크를 모듈의 조합으로 구성")
    print("=" * 70)

analyze_v2_limitations()
```

```
======================================================================
Version 2 - Remaining Problems
======================================================================

[Still Present from V1]
  • 순전파/역전파가 여전히 명시적
  • 네트워크 구조 변경 어려움
  • 파라미터가 개별 변수로 존재
  • 레이어 추상화 없음
  • 옵티마이저 하드코딩

[New Improvements Needed]
  • 레이어를 모듈로 캡슐화
  • 파라미터 관리 자동화
  • 네트워크를 클래스로 구조화
  • 다양한 활성화 함수 지원
  • 모델 저장/로딩 기능

======================================================================
다음 단계 (Version 3): Module 추상화
  → 레이어를 클래스로 캡슐화
  → 네트워크를 모듈의 조합으로 구성
======================================================================
```

### 9.2.8 Summary

| 항목 | Version 1 | Version 2 |
|------|----------|----------|
| **코드 라인 수** | ~165 lines | ~185 lines |
| **클래스 수** | 0 | 1 (DataLoader) |
| **배치 생성** | 수동 | 자동 |
| **코드 중복** | 높음 | 낮음 (DataLoader 재사용) |
| **Pythonic** | 보통 | 높음 (iterator protocol) |
| **유연성** | 낮음 | 중간 (DataLoader 설정) |
| **재사용성** | 낮음 | 중간 (DataLoader) |

**주요 변경사항:**

```python
# Before (V1): 수동 배치 생성
indices = np.random.permutation(len(x_train))
for i in range(0, len(x_train), batch_size):
    x_batch = x_train[indices[i: i + batch_size]]
    y_batch = y_train[indices[i: i + batch_size]]

# After (V2): DataLoader 사용
train_loader = DataLoader(x_train, y_train, 
                          batch_size=64, shuffle=True)
for x_batch, y_batch in train_loader:
    # ...
```

**DataLoader 인터페이스:**

```python
class DataLoader:
    def __init__(self, x, y, batch_size, shuffle, drop_last):
        # 초기화
    
    def __iter__(self):
        # 반복자 초기화, 셔플 수행
        return self
    
    def __next__(self):
        # 다음 배치 반환
        return x_batch, y_batch
    
    def __len__(self):
        # 전체 배치 수 반환
        return num_batches
```

**개선 효과:**

1. **코드 간결화**: 배치 생성 로직이 캡슐화됨
2. **재사용성 향상**: 동일한 DataLoader를 여러 곳에서 사용
3. **Pythonic**: for x, y in loader 형태의 자연스러운 반복
4. **유연성**: 파라미터만 변경하여 동작 제어
5. **오류 감소**: 인덱스 계산 오류 가능성 감소

**다음 단계:**

Version 3에서는 레이어와 네트워크를 모듈로 추상화하여 더욱 유연하고 확장 가능한 구조를 만듭니다.
