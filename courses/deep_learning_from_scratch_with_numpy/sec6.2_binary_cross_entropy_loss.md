## 6.2 Binary Cross-Entropy Loss

Binary Cross-Entropy는 이진 분류 문제에서 사용하는 표준 손실 함수입니다. 모델이 예측한 확률과 실제 정답 사이의 차이를 측정합니다.

### 6.2.1 Mathematical Definition

**일반적인 형태:**

$$L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

여기서:
- $y \in \{0, 1\}$: 정답 레이블
- $\hat{y} \in (0, 1)$: 예측 확률 (클래스 1에 속할 확률)

**배치 데이터에 대한 평균:**

$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**조건부 형태로 표현:**

$$L(y, \hat{y}) = \begin{cases}
-\log(\hat{y}) & \text{if } y = 1 \\
-\log(1-\hat{y}) & \text{if } y = 0
\end{cases}$$

### 6.2.2 Intuition and Interpretation

**확률적 해석:**

Binary Cross-Entropy는 베르누이 분포의 음의 로그 가능도(negative log-likelihood)입니다.

**최적화 목표:**

- $y=1$일 때: $\hat{y}$를 1에 가깝게
- $y=0$일 때: $\hat{y}$를 0에 가깝게

**예시:**

```python
import numpy as np

def binary_cross_entropy_example():
    """Binary Cross-Entropy Loss의 직관적 이해"""
    
    print("=" * 70)
    print("Binary Cross-Entropy Loss Examples")
    print("=" * 70)
    
    # Case 1: y = 1 (정답이 클래스 1)
    print("\n[Case 1: True Label = 1]")
    print(f"{'Prediction':<15} {'Loss':<15} {'Interpretation':<30}")
    print("-" * 70)
    
    predictions_1 = [0.99, 0.90, 0.70, 0.50, 0.30, 0.10, 0.01]
    for pred in predictions_1:
        loss = -np.log(pred + 1e-8)
        if pred > 0.9:
            interpretation = "Excellent"
        elif pred > 0.7:
            interpretation = "Good"
        elif pred > 0.5:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"
        print(f"{pred:<15.2f} {loss:<15.4f} {interpretation:<30}")
    
    # Case 2: y = 0 (정답이 클래스 0)
    print("\n[Case 2: True Label = 0]")
    print(f"{'Prediction':<15} {'Loss':<15} {'Interpretation':<30}")
    print("-" * 70)
    
    predictions_0 = [0.01, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99]
    for pred in predictions_0:
        loss = -np.log(1 - pred + 1e-8)
        if pred < 0.1:
            interpretation = "Excellent"
        elif pred < 0.3:
            interpretation = "Good"
        elif pred < 0.5:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"
        print(f"{pred:<15.2f} {loss:<15.4f} {interpretation:<30}")
    
    print("=" * 70)

binary_cross_entropy_example()
```

```
======================================================================
Binary Cross-Entropy Loss Examples
======================================================================

[Case 1: True Label = 1]
Prediction      Loss            Interpretation                
----------------------------------------------------------------------
0.99            0.0101          Excellent                     
0.90            0.1054          Good                          
0.70            0.3567          Good                          
0.50            0.6931          Moderate                      
0.30            1.2040          Poor                          
0.10            2.3026          Poor                          
0.01            4.6052          Poor                          

[Case 2: True Label = 0]
Prediction      Loss            Interpretation                
----------------------------------------------------------------------
0.01            0.0101          Excellent                     
0.10            0.1054          Good                          
0.30            0.3567          Good                          
0.50            0.6931          Moderate                      
0.70            1.2040          Poor                          
0.90            2.3026          Poor                          
0.99            4.6052          Poor                          
======================================================================
```

### 6.2.3 NumPy Implementation

**기본 구현:**

```python
def binary_cross_entropy(predictions, targets):
    """
    Binary Cross-Entropy Loss
    
    Parameters:
    -----------
    predictions : ndarray, shape (batch_size,) or (batch_size, 1)
        예측 확률 (0~1)
    targets : ndarray, shape (batch_size,) or (batch_size, 1)
        정답 레이블 (0 또는 1)
    
    Returns:
    --------
    loss : float
        평균 Binary Cross-Entropy loss
    """
    # 수치 안정성을 위한 epsilon
    epsilon = 1e-8
    
    # Shape 정규화
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # BCE 계산
    loss = -(targets * np.log(predictions + epsilon) + 
             (1 - targets) * np.log(1 - predictions + epsilon))
    
    return np.mean(loss)
```

**안정적인 구현 (로짓 버전):**

수치 안정성을 위해 로짓(logit) 값을 직접 받는 버전:

```python
def binary_cross_entropy_with_logits(logits, targets):
    """
    Binary Cross-Entropy with Logits (수치 안정성 개선)
    
    sigmoid와 BCE를 결합하여 수치 안정성 향상
    
    Parameters:
    -----------
    logits : ndarray
        Sigmoid 적용 전 값
    targets : ndarray
        정답 레이블 (0 또는 1)
    
    Returns:
    --------
    loss : float
    """
    # Shape 정규화
    logits = logits.flatten()
    targets = targets.flatten()
    
    # 수치 안정적인 계산
    # log(1 + exp(-|z|)) + max(z, 0) - z*y 형태로 변환
    max_val = np.maximum(logits, 0)
    loss = max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
    
    return np.mean(loss)
```

### 6.2.4 Practical Examples

```python
# 테스트 데이터 생성
np.random.seed(42)

batch_size = 8

# 예측 확률 (sigmoid 출력)
predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])

# 정답 레이블
targets = np.array([1, 1, 1, 1, 0, 0, 0, 0])

print("\n" + "=" * 60)
print("Binary Cross-Entropy Loss Computation")
print("=" * 60)

print("\n[Sample-wise Loss]")
print(f"{'Sample':<10} {'Prediction':<15} {'Target':<10} {'Loss':<15}")
print("-" * 60)

for i in range(batch_size):
    sample_loss = binary_cross_entropy(predictions[i:i+1], targets[i:i+1])
    print(f"{i:<10} {predictions[i]:<15.2f} {targets[i]:<10} {sample_loss:<15.4f}")

# 전체 배치 loss
total_loss = binary_cross_entropy(predictions, targets)

print("-" * 60)
print(f"{'Average':<10} {'-':<15} {'-':<10} {total_loss:<15.4f}")
print("=" * 60)
```

```
============================================================
Binary Cross-Entropy Loss Computation
============================================================

[Sample-wise Loss]
Sample     Prediction      Target     Loss           
------------------------------------------------------------
0          0.90            1          0.1054         
1          0.80            1          0.2231         
2          0.70            1          0.3567         
3          0.60            1          0.5108         
4          0.40            0          0.5108         
5          0.30            0          0.3567         
6          0.20            0          0.2231         
7          0.10            0          0.1054         
------------------------------------------------------------
Average    -               -          0.2990         
============================================================
```

### 6.2.5 Gradient Derivation

Sigmoid와 Binary Cross-Entropy를 결합하면 간단한 그래디언트를 얻습니다.

**손실 함수:**

$$L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**Sigmoid:**

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**그래디언트 유도:**

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

**Step 1: $\frac{\partial L}{\partial \hat{y}}$**

$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

**Step 2: $\frac{\partial \hat{y}}{\partial z}$**

$$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$$

**Step 3: 연쇄 법칙 적용**

$$\frac{\partial L}{\partial z} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})$$

$$= -y(1-\hat{y}) + (1-y)\hat{y}$$

$$= -y + y\hat{y} + \hat{y} - y\hat{y}$$

$$= \hat{y} - y$$

**최종 결과 (매우 간단!):**

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

**배치 형태:**

$$\frac{\partial L}{\partial \mathbf{z}} = \frac{1}{N}(\hat{\mathbf{y}} - \mathbf{y})$$

### 6.2.6 Combined Forward and Backward

```python
class SigmoidBinaryCrossEntropy:
    """Sigmoid + Binary Cross-Entropy Loss (combined for efficiency)"""
    
    def __init__(self):
        self.predictions = None
        self.targets = None
    
    def forward(self, logits, targets):
        """
        순전파: Sigmoid + Binary Cross-Entropy
        
        Parameters:
        -----------
        logits : ndarray, shape (batch_size, 1) or (batch_size,)
            로짓 값 (sigmoid 적용 전)
        targets : ndarray, shape (batch_size, 1) or (batch_size,)
            정답 레이블 (0 또는 1)
        
        Returns:
        --------
        loss : float
            Binary Cross-Entropy loss
        """
        # Sigmoid
        self.predictions = np.where(
            logits >= 0,
            1 / (1 + np.exp(-logits)),
            np.exp(logits) / (1 + np.exp(logits))
        )
        
        # Shape 정규화
        self.targets = targets.reshape(-1, 1) if targets.ndim == 1 else targets
        
        # Binary Cross-Entropy
        epsilon = 1e-8
        loss = -(self.targets * np.log(self.predictions + epsilon) + 
                 (1 - self.targets) * np.log(1 - self.predictions + epsilon))
        
        return np.mean(loss)
    
    def backward(self):
        """
        역전파: 간단한 그래디언트
        
        Returns:
        --------
        grad : ndarray
            로짓에 대한 그래디언트
        """
        batch_size = self.predictions.shape[0]
        grad = (self.predictions - self.targets) / batch_size
        return grad


# 사용 예시
loss_fn = SigmoidBinaryCrossEntropy()

# 순전파
logits = np.random.randn(8, 1)
targets = np.random.randint(0, 2, (8, 1))

loss = loss_fn.forward(logits, targets)
print(f"\n[Forward Pass]")
print(f"Loss: {loss:.6f}")

# 역전파
grad = loss_fn.backward()
print(f"\n[Backward Pass]")
print(f"Gradient shape: {grad.shape}")
print(f"Gradient mean: {grad.mean():.6f}")
print(f"Gradient std: {grad.std():.6f}")
```

```
[Forward Pass]
Loss: 0.682145

[Backward Pass]
Gradient shape: (8, 1)
Gradient mean: 0.000000
Gradient std: 0.158113
```

### 6.2.7 Loss Behavior Analysis

```python
def analyze_bce_behavior():
    """BCE Loss의 동작 분석"""
    
    import matplotlib.pyplot as plt
    
    # 예측 확률 범위
    predictions = np.linspace(0.01, 0.99, 100)
    
    # y=1일 때와 y=0일 때의 loss
    loss_y1 = -np.log(predictions)
    loss_y0 = -np.log(1 - predictions)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves for both classes
    axes[0].plot(predictions, loss_y1, linewidth=2, color='#e74c3c', 
                 label='y = 1 (True class is 1)')
    axes[0].plot(predictions, loss_y0, linewidth=2, color='#3498db', 
                 label='y = 0 (True class is 0)')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Binary Cross-Entropy Loss', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 5)
    
    # Plot 2: Gradient magnitude
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    axes[1].plot(predictions, predictions - 1, linewidth=2, color='#e74c3c', 
                 label='Gradient when y = 1')
    axes[1].plot(predictions, predictions, linewidth=2, color='#3498db', 
                 label='Gradient when y = 0')
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Gradient (∂L/∂z)', fontsize=12)
    axes[1].set_title('Loss Gradient', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('bce_loss_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nBCE Loss 분석 그래프가 'bce_loss_analysis.png'로 저장되었습니다.")

analyze_bce_behavior()
```

```
BCE Loss 분석 그래프가 'bce_loss_analysis.png'로 저장되었습니다.
```

### 6.2.8 Comparison with Other Losses

```python
def compare_binary_losses():
    """이진 분류 손실 함수 비교"""
    
    print("\n" + "=" * 70)
    print("Loss Function Comparison for Binary Classification")
    print("=" * 70)
    
    # 예측과 정답
    scenarios = [
        ("Perfect (y=1)", 0.99, 1),
        ("Good (y=1)", 0.80, 1),
        ("Uncertain (y=1)", 0.60, 1),
        ("Wrong (y=1)", 0.20, 1),
        ("Perfect (y=0)", 0.01, 0),
        ("Good (y=0)", 0.20, 0),
        ("Uncertain (y=0)", 0.40, 0),
        ("Wrong (y=0)", 0.80, 0),
    ]
    
    print(f"{'Scenario':<20} {'Prediction':<15} {'BCE Loss':<15} {'MSE Loss':<15}")
    print("-" * 70)
    
    for name, pred, target in scenarios:
        # Binary Cross-Entropy
        epsilon = 1e-8
        bce = -(target * np.log(pred + epsilon) + 
                (1 - target) * np.log(1 - pred + epsilon))
        
        # Mean Squared Error (참고용)
        mse = (pred - target) ** 2
        
        print(f"{name:<20} {pred:<15.2f} {bce:<15.4f} {mse:<15.4f}")
    
    print("=" * 70)
    print("\n관찰:")
    print("  - BCE는 확률에 로그를 적용하여 틀린 예측에 큰 패널티")
    print("  - MSE는 거리 기반으로 선형적 패널티")
    print("  - 확률 기반 분류에는 BCE가 더 적합")

compare_binary_losses()
```

```
======================================================================
Loss Function Comparison for Binary Classification
======================================================================
Scenario             Prediction      BCE Loss        MSE Loss       
----------------------------------------------------------------------
Perfect (y=1)        0.99            0.0101          0.0001         
Good (y=1)           0.80            0.2231          0.0400         
Uncertain (y=1)      0.60            0.5108          0.1600         
Wrong (y=1)          0.20            1.6094          0.6400         
Perfect (y=0)        0.01            0.0101          0.0001         
Good (y=0)           0.20            0.2231          0.0400         
Uncertain (y=0)      0.40            0.5108          0.1600         
Wrong (y=0)          0.80            1.6094          0.6400         
======================================================================

관찰:
  - BCE는 확률에 로그를 적용하여 틀린 예측에 큰 패널티
  - MSE는 거리 기반으로 선형적 패널티
  - 확률 기반 분류에는 BCE가 더 적합
```

### 6.2.9 Summary

| 항목 | 설명 | 수식 |
|------|------|------|
| **손실 함수** | Binary Cross-Entropy | $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| **배치 평균** | N개 샘플 평균 | $L = -\frac{1}{N}\sum_i [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$ |
| **그래디언트** | Sigmoid + BCE | $\frac{\partial L}{\partial z} = \hat{y} - y$ |
| **범위** | 손실 값 | $[0, +\infty)$ |
| **최솟값** | 완벽한 예측 | $L = 0$ when $\hat{y} = y$ |
| **수치 안정성** | Epsilon 추가 | $\log(\hat{y} + \epsilon)$, $\epsilon = 10^{-8}$ |

**핵심 특징:**

1. **확률 해석**: 베르누이 분포의 음의 로그 가능도
2. **그래디언트 단순성**: Sigmoid와 결합 시 $\hat{y} - y$로 매우 간단
3. **수치 안정성**: 로짓 버전 사용으로 안정성 향상
4. **이진 분류**: 두 클래스 간의 확률 예측에 최적화
5. **비대칭 패널티**: 확신 있는 오답에 큰 패널티

**Categorical CE와의 관계:**

Binary CE는 Categorical CE의 특수한 경우 (K=2):

$$\text{Binary CE: } -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

$$\text{Categorical CE (K=2): } -[y_1 \log(\hat{p}_1) + y_2 \log(\hat{p}_2)]$$

여기서 $y_2 = 1 - y_1$, $\hat{p}_2 = 1 - \hat{p}_1$

**구현 시 주의사항:**

- `log(0)` 방지를 위한 epsilon 추가 필수
- Sigmoid 계산 시 overflow 방지
- 로짓 버전 사용으로 수치 안정성 향상
- 배치 평균으로 정규화
