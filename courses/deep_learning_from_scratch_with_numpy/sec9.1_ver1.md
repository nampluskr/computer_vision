## 9.1 Version 1 - Basic Implementation

이 섹션에서는 가장 기본적인 형태의 MLP 구현을 다룹니다. Chapter 7에서 작성한 `01_mnist_manual.py`를 기반으로, 순수 NumPy만을 사용한 명시적이고 직관적인 구현을 분석합니다.

### 9.1.1 Overview

**Version 1의 특징:**

- 모든 코드가 하나의 파일에 포함
- 명시적인 순전파/역전파 구현
- 최소한의 추상화
- 학습 과정이 명확하게 드러남

**장점:**

- 이해하기 쉬움
- 디버깅이 용이함
- 각 단계를 명확히 파악 가능

**단점:**

- 코드 재사용성 낮음
- 확장성 부족
- 반복적인 코드 많음

### 9.1.2 Complete Code Structure

```python
import os
import numpy as np
import gzip

#################################################################
## Data Loading Functions
#################################################################

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


def get_mnist(data_dir, split="train"):
    """MNIST 데이터셋 로딩"""
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
    """One-hot encoding"""
    return np.eye(num_classes)[x]


def sigmoid(x):
    """시그모이드 함수"""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    """소프트맥스 함수"""
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    """Cross-Entropy Loss"""
    if targets.ndim == 1:
        batch_size = preds.shape[0]
        probs = preds[np.arange(batch_size), targets]
    else:
        probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))


def accuracy(preds, targets):
    """정확도 계산"""
    preds = preds.argmax(axis=1)
    if targets.ndim == 2:
        targets = targets.argmax(axis=1)
    return (preds == targets).mean()


#################################################################
## Main Training Script
#################################################################

if __name__ == "__main__":
    
    np.random.seed(42)
    
    # Data Loading
    data_dir = "/mnt/d/datasets/mnist"
    x_train, y_train = get_mnist(data_dir, split="train")
    x_test, y_test = get_mnist(data_dir, split="test")
    
    print("\n>> Data before preprocessing:")
    print(f"train images: {x_train.dtype}, {x_train.shape}")
    print(f"train labels: {y_train.dtype}, {y_train.shape}")
    
    # Preprocessing
    def preprocess(images, labels):
        images = images.astype(np.float32).reshape(-1, 28*28) / 255
        labels = one_hot(labels, num_classes=10).astype(np.int64)
        return images, labels
    
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    
    print("\n>> Data after preprocessing:")
    print(f"train images: {x_train.dtype}, {x_train.shape}")
    print(f"train labels: {y_train.dtype}, {y_train.shape}")
    
    # Network Initialization
    input_size, hidden_size, output_size = 28*28, 100, 10
    
    w1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    w2 = np.random.randn(hidden_size, hidden_size)
    b2 = np.zeros(hidden_size)
    w3 = np.random.randn(hidden_size, output_size)
    b3 = np.zeros(output_size)
    
    print(f"\n>> Network Architecture:")
    print(f"Layer 1: ({input_size:4d}, {hidden_size:3d})")
    print(f"Layer 2: ({hidden_size:4d}, {hidden_size:3d})")
    print(f"Layer 3: ({hidden_size:4d}, {output_size:3d})")
    
    # Training Loop
    num_epochs = 10
    learning_rate = 0.01
    batch_size = 64
    
    print("\n>> Training start ...")
    for epoch in range(1, num_epochs + 1):
        batch_loss = 0
        batch_acc = 0
        total_size = 0
        
        indices = np.random.permutation(len(x_train))
        for i in range(0, len(x_train), batch_size):
            x = x_train[indices[i: i + batch_size]]
            y = y_train[indices[i: i + batch_size]]
            x_size = x.shape[0]
            total_size += x_size
            
            # Forward propagation
            z1 = np.dot(x, w1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, w2) + b2
            a2 = sigmoid(z2)
            z3 = np.dot(a2, w3) + b3
            preds = softmax(z3)
            
            loss = cross_entropy(preds, y)
            acc = accuracy(preds, y)
            
            # Backward propagation
            grad_z3 = (preds - y) / y.shape[0]
            grad_w3 = np.dot(a2.T, grad_z3)
            grad_b3 = np.sum(grad_z3, axis=0)
            
            grad_a2 = np.dot(grad_z3, w3.T)
            grad_z2 = a2 * (1 - a2) * grad_a2
            grad_w2 = np.dot(a1.T, grad_z2)
            grad_b2 = np.sum(grad_z2, axis=0)
            
            grad_a1 = np.dot(grad_z2, w2.T)
            grad_z1 = a1 * (1 - a1) * grad_a1
            grad_w1 = np.dot(x.T, grad_z1)
            grad_b1 = np.sum(grad_z1, axis=0)
            
            # Update weights and biases
            w1 -= learning_rate * grad_w1
            b1 -= learning_rate * grad_b1
            w2 -= learning_rate * grad_w2
            b2 -= learning_rate * grad_b2
            w3 -= learning_rate * grad_w3
            b3 -= learning_rate * grad_b3
            
            batch_loss += loss * x_size
            batch_acc += acc * x_size
        
        print(f"[{epoch:3d}/{num_epochs}] "
              f"loss:{batch_loss/total_size:.3f} acc:{batch_acc/total_size:.3f}")
    
    # Evaluation
    batch_loss = 0
    batch_acc = 0
    total_size = 0
    
    for i in range(0, len(x_test), batch_size):
        x = x_test[i: i + batch_size]
        y = y_test[i: i + batch_size]
        x_size = x.shape[0]
        total_size += x_size
        
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        out = softmax(z3)
        
        loss = cross_entropy(out, y)
        acc = accuracy(out, y)
        
        batch_loss += loss * x_size
        batch_acc += acc * x_size
    
    print(f"\n>> Evaluation: loss:{batch_loss/total_size:.3f} "
          f"acc:{batch_acc/total_size:.3f}")
```

### 9.1.3 Code Structure Analysis

**구조 분석:**

```python
# 1. 데이터 로딩 함수들 (3개)
- load_mnist_images()
- load_mnist_labels()
- get_mnist()

# 2. 수학 함수들 (5개)
- one_hot()
- sigmoid()
- softmax()
- cross_entropy()
- accuracy()

# 3. 메인 스크립트
- 데이터 로딩
- 전처리
- 네트워크 초기화 (w1, b1, w2, b2, w3, b3)
- 학습 루프
  - 미니배치 생성
  - 순전파 (명시적)
  - 역전파 (명시적)
  - 파라미터 업데이트 (명시적)
- 평가
```

### 9.1.4 Key Components Breakdown

**1. Forward Propagation (명시적):**

```python
# Layer 1
z1 = np.dot(x, w1) + b1
a1 = sigmoid(z1)

# Layer 2
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)

# Layer 3
z3 = np.dot(a2, w3) + b3
preds = softmax(z3)
```

**특징:**
- 각 레이어가 명시적으로 구현됨
- 중간 값들이 모두 변수로 저장됨
- 순전파 과정이 명확함

**2. Backward Propagation (명시적):**

```python
# Output layer gradient
grad_z3 = (preds - y) / y.shape[0]
grad_w3 = np.dot(a2.T, grad_z3)
grad_b3 = np.sum(grad_z3, axis=0)

# Hidden layer 2 gradient
grad_a2 = np.dot(grad_z3, w3.T)
grad_z2 = a2 * (1 - a2) * grad_a2
grad_w2 = np.dot(a1.T, grad_z2)
grad_b2 = np.sum(grad_z2, axis=0)

# Hidden layer 1 gradient
grad_a1 = np.dot(grad_z2, w2.T)
grad_z1 = a1 * (1 - a1) * grad_a1
grad_w1 = np.dot(x.T, grad_z1)
grad_b1 = np.sum(grad_z1, axis=0)
```

**특징:**
- 역전파가 순서대로 명시적으로 구현됨
- 각 레이어의 그래디언트 계산이 보임
- 연쇄 법칙이 명확히 드러남

**3. Parameter Update (명시적):**

```python
w1 -= learning_rate * grad_w1
b1 -= learning_rate * grad_b1
w2 -= learning_rate * grad_w2
b2 -= learning_rate * grad_b2
w3 -= learning_rate * grad_w3
b3 -= learning_rate * grad_b3
```

**특징:**
- 단순한 SGD
- 각 파라미터가 개별적으로 업데이트됨
- 학습률이 하드코딩됨

### 9.1.5 Problems with Version 1

```python
def analyze_v1_problems():
    """Version 1의 문제점 분석"""
    
    print("=" * 70)
    print("Version 1 - Problems and Limitations")
    print("=" * 70)
    
    problems = {
        "1. Code Duplication": [
            "순전파 코드가 학습/평가에서 중복",
            "레이어 추가 시 여러 곳 수정 필요",
            "실수하기 쉬움"
        ],
        "2. Hard-coded Values": [
            "학습률, 배치 크기가 하드코딩",
            "네트워크 구조 변경 어려움",
            "하이퍼파라미터 실험 불편"
        ],
        "3. No Abstraction": [
            "레이어가 추상화되지 않음",
            "재사용 불가능",
            "다른 활성화 함수 사용 어려움"
        ],
        "4. Manual Gradient": [
            "모든 그래디언트를 수동으로 계산",
            "네트워크 변경 시 역전파도 변경 필요",
            "오류 발생 가능성 높음"
        ],
        "5. No State Management": [
            "파라미터가 개별 변수로 존재",
            "저장/로딩 구현 필요",
            "모델 관리 어려움"
        ],
        "6. Limited Functionality": [
            "데이터 로딩이 제한적",
            "옵티마이저 선택 불가",
            "학습률 스케줄링 없음"
        ]
    }
    
    for category, issues in problems.items():
        print(f"\n[{category}]")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n" + "=" * 70)

analyze_v1_problems()
```

```
======================================================================
Version 1 - Problems and Limitations
======================================================================

[1. Code Duplication]
  - 순전파 코드가 학습/평가에서 중복
  - 레이어 추가 시 여러 곳 수정 필요
  - 실수하기 쉬움

[2. Hard-coded Values]
  - 학습률, 배치 크기가 하드코딩
  - 네트워크 구조 변경 어려움
  - 하이퍼파라미터 실험 불편

[3. No Abstraction]
  - 레이어가 추상화되지 않음
  - 재사용 불가능
  - 다른 활성화 함수 사용 어려움

[4. Manual Gradient]
  - 모든 그래디언트를 수동으로 계산
  - 네트워크 변경 시 역전파도 변경 필요
  - 오류 발생 가능성 높음

[5. No State Management]
  - 파라미터가 개별 변수로 존재
  - 저장/로딩 구현 필요
  - 모델 관리 어려움

[6. Limited Functionality]
  - 데이터 로딩이 제한적
  - 옵티마이저 선택 불가
  - 학습률 스케줄링 없음
======================================================================
```

### 9.1.6 Code Metrics

```python
def analyze_v1_metrics():
    """Version 1 코드 메트릭 분석"""
    
    print("\n" + "=" * 70)
    print("Version 1 - Code Metrics")
    print("=" * 70)
    
    metrics = {
        "Lines of Code": {
            "Data loading": "~40 lines",
            "Math functions": "~35 lines",
            "Main script": "~90 lines",
            "Total": "~165 lines"
        },
        "Functions": {
            "Data functions": 3,
            "Math functions": 5,
            "Total": 8
        },
        "Variables (Parameters)": {
            "Weights": 3,
            "Biases": 3,
            "Gradients": 6,
            "Total": 12
        },
        "Flexibility": {
            "Network depth": "Fixed (3 layers)",
            "Activation": "Fixed (Sigmoid)",
            "Optimizer": "Fixed (SGD)",
            "Loss": "Fixed (Cross-Entropy)"
        }
    }
    
    for category, data in metrics.items():
        print(f"\n[{category}]")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key:<25} {value}")
        else:
            print(f"  {data}")
    
    print("=" * 70)

analyze_v1_metrics()
```

```
======================================================================
Version 1 - Code Metrics
======================================================================

[Lines of Code]
  Data loading              ~40 lines
  Math functions            ~35 lines
  Main script               ~90 lines
  Total                     ~165 lines

[Functions]
  Data functions            3
  Math functions            5
  Total                     8

[Variables (Parameters)]
  Weights                   3
  Biases                    3
  Gradients                 6
  Total                     12

[Flexibility]
  Network depth             Fixed (3 layers)
  Activation                Fixed (Sigmoid)
  Optimizer                 Fixed (SGD)
  Loss                      Fixed (Cross-Entropy)
======================================================================
```

### 9.1.7 When to Use Version 1

```python
def when_to_use_v1():
    """Version 1이 적합한 상황"""
    
    print("\n" + "=" * 70)
    print("When to Use Version 1")
    print("=" * 70)
    
    use_cases = {
        "✓ Good For": [
            "학습 목적 - 신경망 원리 이해",
            "교육용 코드 - 각 단계 명확히 표시",
            "디버깅 - 모든 값을 쉽게 확인",
            "작은 프로토타입 - 빠른 실험",
            "알고리즘 검증 - 수식 확인"
        ],
        "✗ Not Good For": [
            "프로덕션 코드 - 유지보수 어려움",
            "대규모 실험 - 코드 재사용 불가",
            "복잡한 모델 - 확장성 부족",
            "팀 프로젝트 - 구조화 부족",
            "다양한 시도 - 유연성 부족"
        ]
    }
    
    for category, cases in use_cases.items():
        print(f"\n{category}")
        for case in cases:
            print(f"  • {case}")
    
    print("\n" + "=" * 70)
    print("\n결론: Version 1은 학습과 이해에는 완벽하지만,")
    print("      실제 프로젝트에는 추가 개선이 필요합니다.")
    print("=" * 70)

when_to_use_v1()
```

```
======================================================================
When to Use Version 1
======================================================================

✓ Good For
  • 학습 목적 - 신경망 원리 이해
  • 교육용 코드 - 각 단계 명확히 표시
  • 디버깅 - 모든 값을 쉽게 확인
  • 작은 프로토타입 - 빠른 실험
  • 알고리즘 검증 - 수식 확인

✗ Not Good For
  • 프로덕션 코드 - 유지보수 어려움
  • 대규모 실험 - 코드 재사용 불가
  • 복잡한 모델 - 확장성 부족
  • 팀 프로젝트 - 구조화 부족
  • 다양한 시도 - 유연성 부족

======================================================================

결론: Version 1은 학습과 이해에는 완벽하지만,
      실제 프로젝트에는 추가 개선이 필요합니다.
======================================================================
```

### 9.1.8 Summary

| 항목 | Version 1 |
|------|----------|
| **코드 라인 수** | ~165 lines |
| **파일 수** | 1 |
| **클래스 수** | 0 |
| **함수 수** | 8 |
| **추상화 수준** | 매우 낮음 |
| **재사용성** | 낮음 |
| **확장성** | 낮음 |
| **가독성** | 높음 (명시적) |
| **유지보수성** | 낮음 |
| **학습 용이성** | 매우 높음 |

**핵심 특징:**

1. **명시성**: 모든 연산이 명확히 드러남
2. **단순성**: 추상화가 없어 이해하기 쉬움
3. **직관성**: 수식과 코드가 1:1 대응
4. **제한성**: 확장과 재사용이 어려움

**코드 구조:**

```
01_mnist_manual.py
├── Data Loading Functions
│   ├── load_mnist_images()
│   ├── load_mnist_labels()
│   └── get_mnist()
│
├── Math Functions
│   ├── one_hot()
│   ├── sigmoid()
│   ├── softmax()
│   ├── cross_entropy()
│   └── accuracy()
│
└── Main Script
    ├── Data Loading & Preprocessing
    ├── Network Initialization (w1,b1,w2,b2,w3,b3)
    ├── Training Loop
    │   ├── Mini-batch Generation
    │   ├── Forward Propagation (explicit)
    │   ├── Backward Propagation (explicit)
    │   └── Parameter Update (explicit)
    └── Evaluation
```

**다음 단계 (Version 2):**

Version 1의 문제를 해결하기 위해 다음을 개선합니다:
- DataLoader 추상화
- 미니배치 처리 모듈화
- 데이터 로딩 개선
