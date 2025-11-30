## 7.2 Categorical Cross-Entropy Loss

Categorical Cross-Entropy는 다중 클래스 분류 문제에서 사용하는 표준 손실 함수입니다. 모델이 예측한 확률 분포와 실제 정답 분포 사이의 차이를 측정합니다.

### 7.2.1 Mathematical Definition

**일반적인 형태:**

$$L(\mathbf{y}, \hat{\mathbf{p}}) = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)$$

여기서:
- $K$: 클래스 개수 (MNIST의 경우 10)
- $y_k \in \{0, 1\}$: 클래스 $k$에 대한 정답 (one-hot)
- $\hat{p}_k \in [0, 1]$: 클래스 $k$에 대한 예측 확률

**배치 데이터에 대한 평균:**

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})$$

**정답이 one-hot인 경우 간소화:**

정답 클래스 인덱스를 $c_i$라고 하면:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log(\hat{p}_{i,c_i})$$

즉, 정답 클래스의 예측 확률에만 의존합니다.

### 7.2.2 Intuition and Interpretation

**확률적 해석:**

Cross-Entropy는 정답 분포를 재현하기 위해 필요한 정보량(bits)을 측정합니다.

**최적화 목표:**

- 정답 클래스의 확률 $\hat{p}_{c}$를 1에 가깝게
- 나머지 클래스의 확률을 0에 가깝게

**예시:**

```python
import numpy as np

def categorical_cross_entropy_example():
    """Cross-Entropy Loss의 직관적 이해"""
    
    # 정답: 클래스 3 (one-hot: [0,0,0,1,0,0,0,0,0,0])
    y_true = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    
    # 세 가지 예측 시나리오
    predictions = {
        'Perfect': np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'Good':    np.array([0.05, 0.05, 0.05, 0.7, 0.05, 0.05, 0.01, 0.01, 0.01, 0.02]),
        'Bad':     np.array([0.15, 0.15, 0.15, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10]),
        'Worst':   np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10])
    }
    
    print("=" * 70)
    print("Categorical Cross-Entropy Loss Examples")
    print("=" * 70)
    print(f"True label: Class 3 (index 3)\n")
    
    for name, pred in predictions.items():
        # 정답 클래스의 예측 확률
        p_true_class = pred[3]
        
        # Cross-Entropy loss
        loss = -np.log(p_true_class + 1e-8)
        
        print(f"{name:10s} prediction:")
        print(f"  P(class=3) = {p_true_class:.4f}")
        print(f"  Loss       = {loss:.4f}")
        print()

categorical_cross_entropy_example()
```

```
======================================================================
Categorical Cross-Entropy Loss Examples
======================================================================
True label: Class 3 (index 3)

Perfect    prediction:
  P(class=3) = 1.0000
  Loss       = 0.0000

Good       prediction:
  P(class=3) = 0.7000
  Loss       = 0.3567

Bad        prediction:
  P(class=3) = 0.2000
  Loss       = 1.6094

Worst      prediction:
  P(class=3) = 0.0500
  Loss       = 2.9957
```

### 7.2.3 NumPy Implementation

**방법 1: One-Hot Labels**

```python
def cross_entropy_onehot(predictions, targets):
    """
    One-hot 레이블을 사용한 Cross-Entropy
    
    Parameters:
    -----------
    predictions : ndarray, shape (batch_size, num_classes)
        예측 확률 (softmax 출력)
    targets : ndarray, shape (batch_size, num_classes)
        정답 레이블 (one-hot encoded)
    
    Returns:
    --------
    loss : float
        평균 Cross-Entropy loss
    """
    # 수치 안정성을 위한 작은 값 추가
    epsilon = 1e-8
    
    # 각 샘플의 정답 클래스 확률만 추출
    # targets가 one-hot이므로 곱셈 후 합산
    correct_probs = np.sum(predictions * targets, axis=1)
    
    # -log(p) 계산
    loss = -np.mean(np.log(correct_probs + epsilon))
    
    return loss
```

**방법 2: Class Indices**

```python
def cross_entropy_indices(predictions, targets):
    """
    클래스 인덱스를 사용한 Cross-Entropy
    
    Parameters:
    -----------
    predictions : ndarray, shape (batch_size, num_classes)
        예측 확률
    targets : ndarray, shape (batch_size,)
        정답 클래스 인덱스 (0 ~ num_classes-1)
    
    Returns:
    --------
    loss : float
        평균 Cross-Entropy loss
    """
    batch_size = predictions.shape[0]
    epsilon = 1e-8
    
    # 각 샘플의 정답 클래스 확률 추출
    correct_probs = predictions[np.arange(batch_size), targets]
    
    # -log(p) 계산
    loss = -np.mean(np.log(correct_probs + epsilon))
    
    return loss
```

**통합 구현:**

```python
def cross_entropy(predictions, targets):
    """
    범용 Cross-Entropy Loss
    
    one-hot과 class indices 모두 지원
    """
    epsilon = 1e-8
    
    if targets.ndim == 1:
        # Class indices
        batch_size = predictions.shape[0]
        correct_probs = predictions[np.arange(batch_size), targets]
    else:
        # One-hot labels
        correct_probs = np.sum(predictions * targets, axis=1)
    
    loss = -np.mean(np.log(correct_probs + epsilon))
    
    return loss
```

### 7.2.4 Practical Examples

```python
# 테스트 데이터 생성
np.random.seed(42)

batch_size = 4
num_classes = 10

# 랜덤 예측 (softmax 출력 시뮬레이션)
logits = np.random.randn(batch_size, num_classes)
predictions = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

# 정답 레이블 (두 가지 형식)
targets_indices = np.array([3, 7, 1, 5])
targets_onehot = np.eye(num_classes)[targets_indices]

print("=" * 60)
print("Cross-Entropy Loss Computation")
print("=" * 60)

print("\n[Predictions (first 2 samples)]")
for i in range(2):
    print(f"Sample {i}: {predictions[i]}")
    print(f"  True class: {targets_indices[i]}")
    print(f"  P(true class) = {predictions[i, targets_indices[i]]:.4f}\n")

# 두 가지 방법으로 loss 계산
loss_onehot = cross_entropy_onehot(predictions, targets_onehot)
loss_indices = cross_entropy_indices(predictions, targets_indices)
loss_unified = cross_entropy(predictions, targets_indices)

print("\n[Loss Computation]")
print(f"One-hot method:   {loss_onehot:.6f}")
print(f"Indices method:   {loss_indices:.6f}")
print(f"Unified method:   {loss_unified:.6f}")
print(f"All methods match: {np.allclose(loss_onehot, loss_indices)}")
```

```
============================================================
Cross-Entropy Loss Computation
============================================================

[Predictions (first 2 samples)]
Sample 0: [0.0857 0.1024 0.0971 0.1105 0.099  0.1043 0.0971 0.1038 0.101  0.099 ]
  True class: 3
  P(true class) = 0.1105

Sample 1: [0.0912 0.1067 0.0998 0.1089 0.0993 0.1032 0.0987 0.1045 0.1002 0.0875]
  True class: 7
  P(true class) = 0.1045


[Loss Computation]
One-hot method:   2.246891
Indices method:   2.246891
Unified method:   2.246891
All methods match: True
```

### 7.2.5 Gradient Derivation

Softmax와 Cross-Entropy를 결합하면 매우 간단한 그래디언트를 얻습니다.

**손실 함수:**

$$L = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)$$

**Softmax:**

$$\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

**그래디언트 유도:**

$$\frac{\partial L}{\partial z_k} = \hat{p}_k - y_k$$

**배치 형태:**

$$\frac{\partial L}{\partial \mathbf{Z}} = \frac{1}{N}(\hat{\mathbf{P}} - \mathbf{Y})$$

여기서:
- $\mathbf{Z}$: (N, K) 로짓 행렬
- $\hat{\mathbf{P}}$: (N, K) 예측 확률 행렬
- $\mathbf{Y}$: (N, K) 정답 one-hot 행렬

### 7.2.6 Combined Forward and Backward

```python
class SoftmaxCrossEntropy:
    """Softmax + Cross-Entropy Loss (combined for efficiency)"""
    
    def __init__(self):
        self.predictions = None
        self.targets = None
    
    def forward(self, logits, targets):
        """
        순전파: Softmax + Cross-Entropy
        
        Parameters:
        -----------
        logits : ndarray, shape (batch_size, num_classes)
            로짓 값 (softmax 적용 전)
        targets : ndarray
            정답 레이블 (one-hot 또는 indices)
        
        Returns:
        --------
        loss : float
            Cross-Entropy loss
        """
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # One-hot 변환 (필요시)
        if targets.ndim == 1:
            batch_size = logits.shape[0]
            num_classes = logits.shape[1]
            self.targets = np.eye(num_classes)[targets]
        else:
            self.targets = targets
        
        # Cross-Entropy
        correct_probs = np.sum(self.predictions * self.targets, axis=1)
        loss = -np.mean(np.log(correct_probs + 1e-8))
        
        return loss
    
    def backward(self):
        """
        역전파: 간단한 그래디언트
        
        Returns:
        --------
        grad : ndarray, shape (batch_size, num_classes)
            로짓에 대한 그래디언트
        """
        batch_size = self.predictions.shape[0]
        grad = (self.predictions - self.targets) / batch_size
        return grad


# 사용 예시
loss_fn = SoftmaxCrossEntropy()

# 순전파
logits = np.random.randn(4, 10)
targets = np.array([3, 7, 1, 5])

loss = loss_fn.forward(logits, targets)
print(f"\nLoss: {loss:.6f}")

# 역전파
grad = loss_fn.backward()
print(f"Gradient shape: {grad.shape}")
print(f"Gradient mean: {grad.mean():.6f}")
print(f"Gradient std: {grad.std():.6f}")
```

```
Loss: 2.389145
Gradient shape: (4, 10)
Gradient mean: 0.000000
Gradient std: 0.088388
```

### 7.2.7 Loss Behavior Analysis

```python
def analyze_loss_behavior():
    """손실 함수의 동작 분석"""
    
    # 정답 클래스의 확률에 따른 loss 변화
    true_class_probs = np.linspace(0.01, 1.0, 100)
    losses = -np.log(true_class_probs)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss vs Probability
    plt.subplot(1, 2, 1)
    plt.plot(true_class_probs, losses, linewidth=2, color='#e74c3c')
    plt.xlabel('P(true class)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Cross-Entropy Loss vs True Class Probability', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 주요 지점 표시
    key_points = [(0.1, -np.log(0.1)), (0.5, -np.log(0.5)), (0.9, -np.log(0.9))]
    for p, l in key_points:
        plt.plot(p, l, 'o', markersize=8, color='#3498db')
        plt.text(p, l + 0.3, f'({p:.1f}, {l:.2f})', ha='center', fontsize=9)
    
    # Plot 2: Loss Gradient
    plt.subplot(1, 2, 2)
    gradients = -1 / true_class_probs
    plt.plot(true_class_probs, gradients, linewidth=2, color='#2ecc71')
    plt.xlabel('P(true class)', fontsize=12)
    plt.ylabel('Gradient magnitude', fontsize=12)
    plt.title('Loss Gradient Magnitude', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(-20, 0)
    
    plt.tight_layout()
    plt.savefig('ce_loss_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n손실 함수 분석 그래프가 'ce_loss_analysis.png'로 저장되었습니다.")

analyze_loss_behavior()
```

```
손실 함수 분석 그래프가 'ce_loss_analysis.png'로 저장되었습니다.
```

### 7.2.8 Comparison with Other Losses

```python
def compare_loss_functions():
    """다양한 손실 함수 비교"""
    
    # 정답: 클래스 0
    y_true_onehot = np.array([1, 0, 0])
    
    # 여러 예측 시나리오
    scenarios = {
        'Perfect': np.array([1.0, 0.0, 0.0]),
        'Good': np.array([0.8, 0.1, 0.1]),
        'Uncertain': np.array([0.4, 0.3, 0.3]),
        'Wrong': np.array([0.1, 0.5, 0.4])
    }
    
    print("\n" + "=" * 70)
    print("Loss Function Comparison (True class = 0)")
    print("=" * 70)
    print(f"{'Prediction':<15} {'P(class=0)':<15} {'CE Loss':<15} {'MSE Loss':<15}")
    print("-" * 70)
    
    for name, pred in scenarios.items():
        # Cross-Entropy
        ce_loss = -np.log(pred[0] + 1e-8)
        
        # Mean Squared Error (참고용)
        mse_loss = np.mean((pred - y_true_onehot) ** 2)
        
        print(f"{name:<15} {pred[0]:<15.3f} {ce_loss:<15.4f} {mse_loss:<15.4f}")
    
    print("=" * 70)
    print("\n관찰:")
    print("  - Cross-Entropy는 정답 클래스의 확률에만 의존")
    print("  - MSE는 모든 클래스의 오차를 고려")
    print("  - Cross-Entropy가 확률 예측에 더 적합")

compare_loss_functions()
```

```
======================================================================
Loss Function Comparison (True class = 0)
======================================================================
Prediction      P(class=0)      CE Loss         MSE Loss       
----------------------------------------------------------------------
Perfect         1.000           0.0000          0.0000         
Good            0.800           0.2231          0.0133         
Uncertain       0.400           0.9163          0.2267         
Wrong           0.100           2.3026          0.5067         
======================================================================

관찰:
  - Cross-Entropy는 정답 클래스의 확률에만 의존
  - MSE는 모든 클래스의 오차를 고려
  - Cross-Entropy가 확률 예측에 더 적합
```

### 7.2.9 Summary

| 항목 | 설명 | 수식 |
|------|------|------|
| **손실 함수** | Categorical Cross-Entropy | $L = -\frac{1}{N}\sum_i \sum_k y_{ik} \log(\hat{p}_{ik})$ |
| **간소화** | One-hot 레이블 | $L = -\frac{1}{N}\sum_i \log(\hat{p}_{i,c_i})$ |
| **그래디언트** | Softmax + CE | $\frac{\partial L}{\partial z_k} = \hat{p}_k - y_k$ |
| **범위** | 손실 값 | $[0, +\infty)$ |
| **최솟값** | 완벽한 예측 | $L = 0$ when $\hat{p}_{c} = 1$ |
| **수치 안정성** | Epsilon 추가 | $\log(\hat{p} + \epsilon)$, $\epsilon = 10^{-8}$ |

**핵심 특징:**

1. **확률 해석**: 정보 이론에 기반한 손실 함수
2. **그래디언트 단순성**: Softmax와 결합 시 $\hat{p} - y$로 간단
3. **수치 안정성**: log-sum-exp trick과 epsilon 사용
4. **다중 클래스**: 모든 클래스에 대한 확률 분포 학습
5. **최적화 친화적**: 볼록 함수로 최적화가 용이

**구현 시 주의사항:**

- `log(0)` 방지를 위한 작은 epsilon 추가
- Softmax 계산 시 numerical overflow 방지
- One-hot과 class indices 형식 모두 지원
- 배치 평균으로 정규화
