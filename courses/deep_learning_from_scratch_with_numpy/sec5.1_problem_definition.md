## 6.1 Problem Definition

이진 분류(Binary Classification)는 입력 데이터를 두 개의 클래스 중 하나로 분류하는 문제입니다. MNIST 데이터셋에서 숫자 0과 1만을 선택하여 이진 분류 문제로 단순화할 수 있습니다.

### 6.1.1 Task Characteristics

**입력과 출력:**
- 입력: $\mathbf{x} \in \mathbb{R}^{784}$ (28×28 픽셀을 평탄화한 벡터)
- 출력: $\hat{y} \in [0, 1]$ (클래스 1에 속할 확률)

**클래스 정의:**
- 클래스 0: 숫자 "0" 이미지
- 클래스 1: 숫자 "1" 이미지

**예시:**
- 입력 이미지: 손글씨 숫자 "1"
- 정답 레이블: $y = 1$
- 모델 출력: $\hat{y} = 0.92$ (92% 확률로 클래스 1 예측)

**목표:**

모델이 출력하는 확률 $\hat{y}$가 실제 정답 $y$와 최대한 가까워지도록 학습합니다.

### 6.1.2 Mathematical Formulation

**출력층 활성화 (Sigmoid):**

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

여기서:
- $z$: 출력층의 로짓(logit) 값
- $\sigma$: 시그모이드 함수
- $\hat{y}$: 클래스 1에 속할 확률

**시그모이드 함수의 성질:**

$$\sigma(z) \in (0, 1)$$

$$\sigma(-z) = 1 - \sigma(z)$$

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

**손실 함수 (Binary Cross-Entropy):**

$$L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**배치 데이터에 대한 평균:**

$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

여기서:
- $N$: 배치 크기
- $y_i \in \{0, 1\}$: 샘플 $i$의 정답 레이블
- $\hat{y}_i \in [0, 1]$: 샘플 $i$의 예측 확률

### 6.1.3 Network Architecture

MNIST 이진 분류를 위한 MLP 구조:

```
Input (784) → Hidden1 (256) → Hidden2 (128) → Output (1)
     ↓              ↓                ↓              ↓
   pixels      Sigmoid          Sigmoid        Sigmoid
```

**각 층의 상세:**

```python
import numpy as np

# 네트워크 파라미터
input_size = 28 * 28  # 784
hidden1_size = 256
hidden2_size = 128
output_size = 1       # Binary classification (단일 출력)

# 가중치 초기화 (He initialization)
w1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros(hidden1_size)

w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
b2 = np.zeros(hidden2_size)

w3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
b3 = np.zeros(output_size)

print("=" * 60)
print("MNIST Binary Classification Network")
print("=" * 60)
print(f"\nLayer 1: ({input_size:4d}, {hidden1_size:3d}) -> Sigmoid")
print(f"Layer 2: ({hidden1_size:4d}, {hidden2_size:3d}) -> Sigmoid")
print(f"Layer 3: ({hidden2_size:4d}, {output_size:3d}) -> Sigmoid")
print(f"\nTotal parameters: {w1.size + b1.size + w2.size + b2.size + w3.size + b3.size:,}")
```

```
============================================================
MNIST Binary Classification Network
============================================================

Layer 1: ( 784,  256) -> Sigmoid
Layer 2: ( 256,  128) -> Sigmoid
Layer 3: ( 128,    1) -> Sigmoid

Total parameters: 233,089
```

### 6.1.4 Prediction and Decision Rule

**예측 확률 계산:**

순전파를 통해 클래스 1에 속할 확률을 계산합니다:

```python
def predict_proba_binary(x, w1, b1, w2, b2, w3, b3):
    """
    이진 분류 확률 예측
    
    Parameters:
    -----------
    x : ndarray, shape (batch_size, 784)
        입력 이미지
    
    Returns:
    --------
    probs : ndarray, shape (batch_size, 1)
        클래스 1에 속할 확률
    """
    # Layer 1
    z1 = x @ w1 + b1
    a1 = sigmoid(z1)
    
    # Layer 2
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    
    # Layer 3 (Sigmoid output)
    z3 = a2 @ w3 + b3
    probs = sigmoid(z3)
    
    return probs


def sigmoid(x):
    """시그모이드 함수 (수치 안정성 고려)"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
```

**최종 예측 (Threshold):**

임계값(threshold)을 사용하여 클래스를 결정합니다. 일반적으로 0.5를 사용:

```python
def predict_binary(x, w1, b1, w2, b2, w3, b3, threshold=0.5):
    """
    이진 분류 예측
    
    Parameters:
    -----------
    threshold : float, default=0.5
        분류 임계값
    
    Returns:
    --------
    predictions : ndarray, shape (batch_size,)
        예측된 클래스 (0 또는 1)
    """
    probs = predict_proba_binary(x, w1, b1, w2, b2, w3, b3)
    predictions = (probs >= threshold).astype(np.int32).flatten()
    return predictions
```

**결정 규칙:**

$$\hat{c} = \begin{cases}
1 & \text{if } \hat{y} \geq 0.5 \\
0 & \text{if } \hat{y} < 0.5
\end{cases}$$

### 6.1.5 Example Prediction

```python
# 더미 데이터로 예측 테스트
np.random.seed(42)

# 작은 배치 생성
batch_size = 5
x_dummy = np.random.randn(batch_size, input_size)

# 확률 예측
probs = predict_proba_binary(x_dummy, w1, b1, w2, b2, w3, b3)

# 클래스 예측
predictions = predict_binary(x_dummy, w1, b1, w2, b2, w3, b3)

print("\n" + "=" * 60)
print("Binary Classification Prediction Example")
print("=" * 60)

for i in range(batch_size):
    print(f"\nSample {i+1}:")
    print(f"  P(class=1) = {probs[i, 0]:.4f}")
    print(f"  P(class=0) = {1 - probs[i, 0]:.4f}")
    print(f"  Predicted class: {predictions[i]}")
```

```
============================================================
Binary Classification Prediction Example
============================================================

Sample 1:
  P(class=1) = 0.4982
  P(class=0) = 0.5018
  Predicted class: 0

Sample 2:
  P(class=1) = 0.5021
  P(class=0) = 0.4979
  Predicted class: 1

Sample 3:
  P(class=1) = 0.4995
  P(class=0) = 0.5005
  Predicted class: 0

Sample 4:
  P(class=1) = 0.5008
  P(class=0) = 0.4992
  Predicted class: 1

Sample 5:
  P(class=1) = 0.4989
  P(class=0) = 0.5011
  Predicted class: 0
```

### 6.1.6 Performance Metrics

이진 분류에서 사용하는 주요 평가 지표:

**1. Accuracy (정확도):**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
def accuracy_binary(y_pred, y_true):
    """
    이진 분류 정확도
    
    Parameters:
    -----------
    y_pred : ndarray
        예측값 (0 또는 1)
    y_true : ndarray
        정답 (0 또는 1)
    
    Returns:
    --------
    acc : float
    """
    return np.mean(y_pred == y_true)
```

**2. Precision (정밀도):**

$$\text{Precision} = \frac{TP}{TP + FP}$$

**3. Recall (재현율):**

$$\text{Recall} = \frac{TP}{TP + FN}$$

**4. F1-Score:**

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

```python
def compute_metrics_binary(y_pred, y_true):
    """
    이진 분류 지표 계산
    
    Returns:
    --------
    metrics : dict
        accuracy, precision, recall, f1_score
    """
    # Confusion matrix 요소
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # 지표 계산
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
```

### 6.1.7 Threshold Analysis

임계값에 따른 성능 변화를 분석할 수 있습니다:

```python
def analyze_threshold_effect():
    """임계값 변화에 따른 영향 분석"""
    
    # 시뮬레이션 데이터
    np.random.seed(42)
    n_samples = 100
    
    # 가상의 예측 확률과 정답
    y_probs = np.random.rand(n_samples)
    y_true = (np.random.rand(n_samples) > 0.5).astype(np.int32)
    
    print("\n" + "=" * 70)
    print("Threshold Effect on Binary Classification")
    print("=" * 70)
    print(f"{'Threshold':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 70)
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_probs >= threshold).astype(np.int32)
        metrics = compute_metrics_binary(y_pred, y_true)
        
        print(f"{threshold:<15.1f} {metrics['accuracy']:<15.3f} "
              f"{metrics['precision']:<15.3f} {metrics['recall']:<15.3f}")
    
    print("=" * 70)
    print("\n관찰:")
    print("  - 낮은 threshold: Recall 증가, Precision 감소")
    print("  - 높은 threshold: Precision 증가, Recall 감소")
    print("  - 0.5가 일반적인 균형점")

analyze_threshold_effect()
```

```
======================================================================
Threshold Effect on Binary Classification
======================================================================
Threshold       Accuracy        Precision       Recall         
----------------------------------------------------------------------
0.3             0.500           0.526           0.741          
0.4             0.540           0.571           0.667          
0.5             0.570           0.625           0.556          
0.6             0.580           0.667           0.444          
0.7             0.550           0.714           0.370          
======================================================================

관찰:
  - 낮은 threshold: Recall 증가, Precision 감소
  - 높은 threshold: Precision 증가, Recall 감소
  - 0.5가 일반적인 균형점
```

### 6.1.8 Summary

| 구성 요소 | 설명 | 수식/값 |
|-----------|------|---------|
| **입력** | MNIST 이미지 (평탄화) | $\mathbf{x} \in \mathbb{R}^{784}$ |
| **출력** | 클래스 1 확률 | $\hat{y} \in (0, 1)$ |
| **활성화** | Sigmoid | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| **손실 함수** | Binary Cross-Entropy | $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| **예측** | Threshold | $\hat{c} = \mathbb{1}[\hat{y} \geq 0.5]$ |
| **평가 지표** | Accuracy, Precision, Recall, F1 | - |

**이진 분류의 특징:**

1. **출력 차원**: 단일 출력 뉴런 (클래스 1의 확률)
2. **확률 해석**: Sigmoid를 통해 (0, 1) 범위의 확률 출력
3. **상호 보완**: $P(class=0) = 1 - P(class=1)$
4. **레이블 형식**: 0 또는 1 (스칼라 값)
5. **결정 규칙**: 임계값 기반 (일반적으로 0.5)

**다중 클래스 분류와의 주요 차이점:**

| 항목 | 이진 분류 | 다중 클래스 분류 |
|------|----------|-----------------|
| 출력 뉴런 | 1개 | K개 (클래스 개수) |
| 활성화 함수 | Sigmoid | Softmax |
| 출력 범위 | $(0, 1)$ (단일 확률) | $[0, 1]^K$ (확률 분포) |
| 손실 함수 | Binary CE | Categorical CE |
| 레이블 형식 | 0 또는 1 | One-hot vector |
| 확률 합 | $\hat{y} + (1-\hat{y}) = 1$ | $\sum_k \hat{p}_k = 1$ |

**회귀와의 주요 차이점:**

| 항목 | 이진 분류 | 회귀 |
|------|----------|------|
| 출력 범위 | $(0, 1)$ (확률) | $(-\infty, +\infty)$ |
| 활성화 함수 | Sigmoid | None (Linear) |
| 손실 함수 | Binary CE | MSE |
| 정답 형식 | 이산형 (0 또는 1) | 연속형 |
| 평가 지표 | Accuracy, F1 | MAE, RMSE |
