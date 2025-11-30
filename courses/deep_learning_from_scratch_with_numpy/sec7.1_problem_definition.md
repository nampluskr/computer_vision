## 7.1 Problem Definition

다중 클래스 분류(Multiclass Classification)는 입력 데이터를 3개 이상의 클래스 중 하나로 분류하는 문제입니다. MNIST 손글씨 숫자 데이터셋의 경우, 0부터 9까지 10개의 클래스로 분류하는 전형적인 다중 클래스 분류 문제입니다.

### 7.1.1 Task Characteristics

**입력과 출력:**
- 입력: $\mathbf{x} \in \mathbb{R}^{784}$ (28×28 픽셀을 평탄화한 벡터)
- 출력: $\mathbf{y} \in \{0, 1\}^{10}$ (10개 클래스에 대한 one-hot 벡터)

**예시:**
- 입력 이미지: 손글씨 숫자 "7"
- 정답 레이블: $[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]$ (7번 인덱스가 1)
- 모델 출력: $[\hat{p}_0, \hat{p}_1, ..., \hat{p}_9]$ (각 클래스에 속할 확률)

**목표:**

모델이 출력하는 확률 분포 $\hat{\mathbf{p}}$가 실제 정답 분포 $\mathbf{y}$와 최대한 가까워지도록 학습합니다.

### 7.1.2 Mathematical Formulation

**출력층 활성화 (Softmax):**

$$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

여기서:
- $z_k$: 클래스 $k$에 대한 로짓(logit) 값
- $K$: 전체 클래스 개수 (MNIST의 경우 10)
- $\hat{p}_k$: 클래스 $k$에 속할 확률

**확률의 성질:**

$$\hat{p}_k \in [0, 1], \quad \sum_{k=1}^{K} \hat{p}_k = 1$$

**손실 함수 (Categorical Cross-Entropy):**

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})$$

여기서:
- $N$: 배치 크기
- $y_{ik}$: 샘플 $i$의 클래스 $k$에 대한 정답 (0 또는 1)
- $\hat{p}_{ik}$: 샘플 $i$가 클래스 $k$에 속할 예측 확률

**정답이 one-hot 인코딩된 경우 간소화:**

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log(\hat{p}_{i, y_i})$$

여기서 $y_i$는 샘플 $i$의 정답 클래스 인덱스입니다.

### 7.1.3 Network Architecture

MNIST 다중 클래스 분류를 위한 MLP 구조:

```
Input (784) → Hidden1 (256) → Hidden2 (128) → Output (10)
     ↓              ↓                ↓              ↓
   pixels      Sigmoid          Sigmoid        Softmax
```

**각 층의 상세:**

```python
import numpy as np

# 네트워크 파라미터
input_size = 28 * 28  # 784
hidden1_size = 256
hidden2_size = 128
output_size = 10      # 10 classes (0~9)

# 가중치 초기화 (He initialization)
w1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros(hidden1_size)

w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
b2 = np.zeros(hidden2_size)

w3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
b3 = np.zeros(output_size)

print("=" * 60)
print("MNIST Multiclass Classification Network")
print("=" * 60)
print(f"\nLayer 1: ({input_size:4d}, {hidden1_size:3d}) -> ReLU")
print(f"Layer 2: ({hidden1_size:4d}, {hidden2_size:3d}) -> ReLU")
print(f"Layer 3: ({hidden2_size:4d}, {output_size:3d}) -> Softmax")
print(f"\nTotal parameters: {w1.size + b1.size + w2.size + b2.size + w3.size + b3.size:,}")
```

```
============================================================
MNIST Multiclass Classification Network
============================================================

Layer 1: ( 784,  256) -> ReLU
Layer 2: ( 256,  128) -> ReLU
Layer 3: ( 128,   10) -> Softmax

Total parameters: 235,146
```

### 7.1.4 Prediction and Decision Rule

**예측 확률 계산:**

순전파를 통해 각 클래스에 대한 확률을 계산합니다:

```python
def predict_proba(x, w1, b1, w2, b2, w3, b3):
    """
    확률 예측
    
    Parameters:
    -----------
    x : ndarray, shape (batch_size, 784)
        입력 이미지
    
    Returns:
    --------
    probs : ndarray, shape (batch_size, 10)
        각 클래스에 속할 확률
    """
    # Layer 1
    z1 = x @ w1 + b1
    a1 = sigmoid(z1)
    
    # Layer 2
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    
    # Layer 3 (Softmax)
    z3 = a2 @ w3 + b3
    probs = softmax(z3)
    
    return probs


def sigmoid(x):
    """시그모이드 함수"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def softmax(x):
    """소프트맥스 함수"""
    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    else:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

**최종 예측 (argmax):**

가장 높은 확률을 가진 클래스를 선택합니다:

```python
def predict(x, w1, b1, w2, b2, w3, b3):
    """
    클래스 예측
    
    Returns:
    --------
    predictions : ndarray, shape (batch_size,)
        예측된 클래스 인덱스 (0~9)
    """
    probs = predict_proba(x, w1, b1, w2, b2, w3, b3)
    predictions = np.argmax(probs, axis=1)
    return predictions
```

### 7.1.5 Example Prediction

```python
# 더미 데이터로 예측 테스트
np.random.seed(42)

# 작은 배치 생성
batch_size = 5
x_dummy = np.random.randn(batch_size, input_size)

# 확률 예측
probs = predict_proba(x_dummy, w1, b1, w2, b2, w3, b3)

# 클래스 예측
predictions = predict(x_dummy, w1, b1, w2, b2, w3, b3)

print("\n" + "=" * 60)
print("Prediction Example")
print("=" * 60)

for i in range(batch_size):
    print(f"\nSample {i+1}:")
    print(f"  Probabilities: {probs[i]}")
    print(f"  Predicted class: {predictions[i]}")
    print(f"  Max probability: {probs[i].max():.4f}")
    print(f"  Probability sum: {probs[i].sum():.6f}")
```

```
============================================================
Prediction Example
============================================================

Sample 1:
  Probabilities: [0.0857 0.1024 0.0971 0.1105 0.099  0.1043 0.0971 0.1038 0.101  0.099 ]
  Predicted class: 3
  Max probability: 0.1105
  Probability sum: 1.000000

Sample 2:
  Probabilities: [0.0912 0.1067 0.0998 0.1089 0.0993 0.1032 0.0987 0.1045 0.1002 0.0875]
  Predicted class: 3
  Max probability: 0.1089
  Probability sum: 1.000000

Sample 3:
  Probabilities: [0.0889 0.1045 0.0984 0.1098 0.0991 0.1038 0.0979 0.1041 0.1006 0.0929]
  Predicted class: 3
  Max probability: 0.1098
  Probability sum: 1.000000

Sample 4:
  Probabilities: [0.0901 0.1056 0.0991 0.1093 0.0992 0.1035 0.0983 0.1043 0.1004 0.0902]
  Predicted class: 3
  Max probability: 0.1093
  Probability sum: 1.000000

Sample 5:
  Probabilities: [0.0895 0.1051 0.0987 0.1096 0.0991 0.1037 0.0981 0.1042 0.1005 0.0915]
  Predicted class: 3
  Max probability: 0.1096
  Probability sum: 1.000000
```

### 7.1.6 Performance Metrics

다중 클래스 분류에서 사용하는 주요 평가 지표:

**1. Accuracy (정확도):**

$$\text{Accuracy} = \frac{\text{올바르게 예측한 샘플 수}}{\text{전체 샘플 수}}$$

```python
def accuracy(y_pred, y_true):
    """
    정확도 계산
    
    Parameters:
    -----------
    y_pred : ndarray, shape (N, 10) or (N,)
        예측 (확률 또는 클래스 인덱스)
    y_true : ndarray, shape (N, 10) or (N,)
        정답 (one-hot 또는 클래스 인덱스)
    
    Returns:
    --------
    acc : float
        정확도 (0~1)
    """
    # 확률을 클래스로 변환
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    # one-hot을 클래스로 변환
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    return np.mean(y_pred == y_true)
```

**2. Per-Class Accuracy:**

각 클래스별 정확도를 계산하여 불균형 데이터셋에서 유용합니다:

```python
def per_class_accuracy(y_pred, y_true, num_classes=10):
    """클래스별 정확도"""
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    class_acc = []
    for c in range(num_classes):
        mask = (y_true == c)
        if mask.sum() == 0:
            class_acc.append(0.0)
        else:
            class_acc.append(np.mean(y_pred[mask] == c))
    
    return np.array(class_acc)
```

### 7.1.7 Summary

| 구성 요소 | 설명 | 수식/값 |
|-----------|------|---------|
| **입력** | MNIST 이미지 (평탄화) | $\mathbf{x} \in \mathbb{R}^{784}$ |
| **출력** | 10개 클래스 확률 | $\hat{\mathbf{p}} \in [0,1]^{10}, \sum_k \hat{p}_k = 1$ |
| **활성화** | Softmax | $\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$ |
| **손실 함수** | Categorical Cross-Entropy | $L = -\frac{1}{N}\sum_i \sum_k y_{ik} \log(\hat{p}_{ik})$ |
| **예측** | Argmax | $\hat{y} = \arg\max_k \hat{p}_k$ |
| **평가 지표** | Accuracy | $\frac{\text{correct}}{\text{total}}$ |

**다중 클래스 분류의 특징:**

1. **출력 차원**: 클래스 개수만큼의 출력 뉴런 (MNIST: 10개)
2. **확률 해석**: Softmax를 통해 각 클래스에 속할 확률을 출력
3. **정규화**: 모든 클래스 확률의 합은 1
4. **레이블 인코딩**: One-hot encoding 사용
5. **결정 규칙**: 가장 높은 확률을 가진 클래스 선택

**이진 분류와의 주요 차이점:**

| 항목 | 이진 분류 | 다중 클래스 분류 |
|------|----------|-----------------|
| 출력 뉴런 | 1개 | K개 (클래스 개수) |
| 활성화 함수 | Sigmoid | Softmax |
| 출력 범위 | $[0, 1]$ (단일 확률) | $[0, 1]^K$ (확률 분포) |
| 손실 함수 | Binary CE | Categorical CE |
| 레이블 형식 | 0 또는 1 | One-hot vector |
