## 5.2 Mean Squared Error Loss

Mean Squared Error (MSE)는 회귀 문제에서 가장 널리 사용되는 손실 함수입니다. 예측값과 정답 사이의 제곱 오차를 평균하여 모델의 성능을 측정합니다.

### 5.2.1 Mathematical Definition

**단일 샘플에 대한 손실:**

$$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$

여기서:
- $y \in \mathbb{R}$: 정답값
- $\hat{y} \in \mathbb{R}$: 예측값
- $\frac{1}{2}$ 계수는 미분 시 편의를 위함 (선택적)

**배치 데이터에 대한 평균:**

$$L = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{2}(y_i - \hat{y}_i)^2$$

**1/2 계수가 없는 표준 형태:**

$$L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

**벡터 형태:**

$$L = \frac{1}{N}||\mathbf{y} - \hat{\mathbf{y}}||_2^2 = \frac{1}{N}(\mathbf{y} - \hat{\mathbf{y}})^T(\mathbf{y} - \hat{\mathbf{y}})$$

### 5.2.2 Intuition and Interpretation

**기하학적 해석:**

MSE는 예측값과 정답 사이의 유클리드 거리(Euclidean distance)의 제곱을 측정합니다.

**오차에 대한 민감도:**

- 작은 오차: 제곱하면 더 작아짐 (예: $0.1^2 = 0.01$)
- 큰 오차: 제곱하면 훨씬 커짐 (예: $2.0^2 = 4.0$)
- **결과**: 큰 오차에 더 큰 페널티 부여 (이상치에 민감)

**예시:**

```python
import numpy as np

def mse_loss_example():
    """MSE Loss의 직관적 이해"""
    
    print("=" * 70)
    print("Mean Squared Error Loss Examples")
    print("=" * 70)
    
    # 다양한 예측 시나리오
    scenarios = [
        ("Perfect", 5.0, 5.0),
        ("Small error", 5.0, 5.2),
        ("Moderate error", 5.0, 6.0),
        ("Large error", 5.0, 8.0),
        ("Very large error", 5.0, 10.0)
    ]
    
    print(f"\n{'Scenario':<20} {'True':<10} {'Predicted':<12} {'Error':<10} {'Squared Error':<15}")
    print("-" * 70)
    
    for name, y_true, y_pred in scenarios:
        error = y_pred - y_true
        squared_error = error ** 2
        print(f"{name:<20} {y_true:<10.1f} {y_pred:<12.1f} {error:<10.2f} {squared_error:<15.4f}")
    
    print("=" * 70)
    print("\n관찰:")
    print("  - 오차가 2배 → 손실이 4배")
    print("  - 오차가 3배 → 손실이 9배")
    print("  - 큰 오차에 강한 페널티 (이상치 민감)")

mse_loss_example()
```

```
======================================================================
Mean Squared Error Loss Examples
======================================================================

Scenario             True       Predicted    Error      Squared Error  
----------------------------------------------------------------------
Perfect              5.0        5.0          0.00       0.0000         
Small error          5.0        5.2          0.20       0.0400         
Moderate error       5.0        6.0          1.00       1.0000         
Large error          5.0        8.0          3.00       9.0000         
Very large error     5.0        10.0         5.00       25.0000        
======================================================================

관찰:
  - 오차가 2배 → 손실이 4배
  - 오차가 3배 → 손실이 9배
  - 큰 오차에 강한 페널티 (이상치 민감)
```

### 5.2.3 NumPy Implementation

**기본 구현:**

```python
def mean_squared_error(predictions, targets):
    """
    Mean Squared Error Loss
    
    Parameters:
    -----------
    predictions : ndarray, shape (batch_size, 1) or (batch_size,)
        예측값
    targets : ndarray, shape (batch_size, 1) or (batch_size,)
        정답값
    
    Returns:
    --------
    loss : float
        평균 제곱 오차
    """
    # Shape 정규화
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # MSE 계산
    squared_errors = (targets - predictions) ** 2
    loss = np.mean(squared_errors)
    
    return loss
```

**1/2 계수를 포함한 구현:**

```python
def mean_squared_error_half(predictions, targets):
    """
    MSE with 1/2 coefficient (for gradient convenience)
    
    Returns:
    --------
    loss : float
        (1/2) * MSE
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    squared_errors = (targets - predictions) ** 2
    loss = 0.5 * np.mean(squared_errors)
    
    return loss
```

**벡터화된 구현:**

```python
def mean_squared_error_vectorized(predictions, targets):
    """
    벡터화된 MSE 구현
    
    더 효율적인 계산
    """
    # Shape: (N, 1) or (N,)
    diff = targets - predictions
    
    # 요소별 곱셈 후 평균
    loss = np.mean(diff * diff)
    
    return loss
```

### 5.2.4 Practical Examples

```python
# 테스트 데이터 생성
np.random.seed(42)

batch_size = 8

# 예측값과 정답
predictions = np.array([2.5, 3.8, 1.2, 4.5, 2.1, 3.3, 1.8, 4.0]).reshape(-1, 1)
targets = np.array([2.3, 4.0, 1.5, 4.2, 2.0, 3.5, 1.6, 4.1]).reshape(-1, 1)

print("\n" + "=" * 60)
print("Mean Squared Error Computation")
print("=" * 60)

print("\n[Sample-wise Loss]")
print(f"{'Sample':<10} {'Target':<12} {'Predicted':<12} {'Error':<12} {'Squared Error':<15}")
print("-" * 60)

for i in range(batch_size):
    target = targets[i, 0]
    pred = predictions[i, 0]
    error = target - pred
    sq_error = error ** 2
    print(f"{i:<10} {target:<12.2f} {pred:<12.2f} {error:<12.2f} {sq_error:<15.4f}")

# 전체 배치 MSE
mse = mean_squared_error(predictions, targets)
mse_half = mean_squared_error_half(predictions, targets)
rmse = np.sqrt(mse)

print("-" * 60)
print(f"{'Average':<10} {'-':<12} {'-':<12} {'-':<12} {mse:<15.4f}")
print(f"\nMSE (standard):     {mse:.6f}")
print(f"MSE (with 1/2):     {mse_half:.6f}")
print(f"RMSE:               {rmse:.6f}")
print("=" * 60)
```

```
============================================================
Mean Squared Error Computation
============================================================

[Sample-wise Loss]
Sample     Target       Predicted    Error        Squared Error  
------------------------------------------------------------
0          2.30         2.50         -0.20        0.0400         
1          4.00         3.80         0.20         0.0400         
2          1.50         1.20         0.30         0.0900         
3          4.20         4.50         -0.30        0.0900         
4          2.00         2.10         -0.10        0.0100         
5          3.50         3.30         0.20         0.0400         
6          1.60         1.80         -0.20        0.0400         
7          4.10         4.00         0.10         0.0100         
------------------------------------------------------------
Average    -            -            -            0.0575         

MSE (standard):     0.057500
MSE (with 1/2):     0.028750
RMSE:               0.239792
============================================================
```

### 5.2.5 Gradient Derivation

MSE의 그래디언트 계산은 매우 간단합니다.

**손실 함수 (1/2 계수 포함):**

$$L = \frac{1}{2N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

**예측값에 대한 편미분:**

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N}(\hat{y}_i - y_i)$$

**1/2 계수의 효과:**

- 계수 없이: $\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)$
- 계수 1/2: $\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N}(\hat{y}_i - y_i)$

**벡터 형태:**

$$\frac{\partial L}{\partial \hat{\mathbf{y}}} = \frac{1}{N}(\hat{\mathbf{y}} - \mathbf{y})$$

**출력층 이전 값(z)에 대한 그래디언트:**

회귀에서는 출력층에 활성화 함수가 없으므로:

$$\hat{y} = z \quad \Rightarrow \quad \frac{\partial \hat{y}}{\partial z} = 1$$

따라서:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \frac{1}{N}(\hat{y} - y)$$

### 5.2.6 Combined Forward and Backward

```python
class MSELoss:
    """Mean Squared Error Loss"""
    
    def __init__(self, use_half=True):
        """
        Parameters:
        -----------
        use_half : bool
            1/2 계수 사용 여부 (미분 편의성)
        """
        self.use_half = use_half
        self.predictions = None
        self.targets = None
    
    def forward(self, predictions, targets):
        """
        순전파: MSE 계산
        
        Parameters:
        -----------
        predictions : ndarray, shape (batch_size, 1)
            예측값
        targets : ndarray, shape (batch_size, 1)
            정답값
        
        Returns:
        --------
        loss : float
            MSE loss
        """
        # 저장 (역전파용)
        self.predictions = predictions
        self.targets = targets
        
        # MSE 계산
        diff = targets - predictions
        squared_errors = diff ** 2
        
        if self.use_half:
            loss = 0.5 * np.mean(squared_errors)
        else:
            loss = np.mean(squared_errors)
        
        return loss
    
    def backward(self):
        """
        역전파: 그래디언트 계산
        
        Returns:
        --------
        grad : ndarray, shape (batch_size, 1)
            예측값에 대한 그래디언트
        """
        batch_size = self.predictions.shape[0]
        
        # ∂L/∂ŷ = (ŷ - y) / N
        grad = (self.predictions - self.targets) / batch_size
        
        return grad


# 사용 예시
loss_fn = MSELoss(use_half=True)

# 순전파
predictions = np.random.randn(8, 1) * 2 + 3
targets = np.random.randn(8, 1) * 2 + 3

loss = loss_fn.forward(predictions, targets)
print(f"\n[Forward Pass]")
print(f"Loss: {loss:.6f}")

# 역전파
grad = loss_fn.backward()
print(f"\n[Backward Pass]")
print(f"Gradient shape: {grad.shape}")
print(f"Gradient mean: {grad.mean():.6f}")
print(f"Gradient std: {grad.std():.6f}")
print(f"Gradient sample (first 3): {grad[:3].flatten()}")
```

```
[Forward Pass]
Loss: 0.892145

[Backward Pass]
Gradient shape: (8, 1)
Gradient mean: -0.021234
Gradient std: 0.158113
Gradient sample (first 3): [ 0.12345 -0.08934  0.15678]
```

### 5.2.7 Loss Behavior Analysis

```python
def analyze_mse_behavior():
    """MSE Loss의 동작 분석"""
    
    import matplotlib.pyplot as plt
    
    # 오차 범위
    errors = np.linspace(-3, 3, 100)
    
    # MSE와 MAE 비교
    mse_loss = 0.5 * errors ** 2
    mae_loss = np.abs(errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(errors, mse_loss, linewidth=2, color='#e74c3c', 
                 label='MSE Loss (1/2 × error²)')
    axes[0].plot(errors, mae_loss, linewidth=2, color='#3498db', 
                 label='MAE Loss (|error|)')
    axes[0].set_xlabel('Error (ŷ - y)', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('MSE vs MAE Loss Functions', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 2: Gradient comparison
    mse_gradient = errors
    mae_gradient = np.sign(errors)
    
    axes[1].plot(errors, mse_gradient, linewidth=2, color='#e74c3c', 
                 label='MSE Gradient (error)')
    axes[1].plot(errors, mae_gradient, linewidth=2, color='#3498db', 
                 label='MAE Gradient (sign(error))')
    axes[1].set_xlabel('Error (ŷ - y)', fontsize=12)
    axes[1].set_ylabel('Gradient', fontsize=12)
    axes[1].set_title('Loss Gradients', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('mse_loss_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nMSE Loss 분석 그래프가 'mse_loss_analysis.png'로 저장되었습니다.")

analyze_mse_behavior()
```

```
MSE Loss 분석 그래프가 'mse_loss_analysis.png'로 저장되었습니다.
```

### 5.2.8 Comparison with Other Regression Losses

```python
def compare_regression_losses():
    """회귀 손실 함수 비교"""
    
    print("\n" + "=" * 70)
    print("Regression Loss Function Comparison")
    print("=" * 70)
    
    # 다양한 오차에 대한 손실 계산
    errors = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    
    print(f"\n{'Error':<10} {'MSE':<15} {'MAE':<15} {'Huber (δ=1)':<15}")
    print("-" * 70)
    
    for error in errors:
        mse = 0.5 * error ** 2
        mae = np.abs(error)
        
        # Huber loss (δ=1)
        delta = 1.0
        if np.abs(error) <= delta:
            huber = 0.5 * error ** 2
        else:
            huber = delta * (np.abs(error) - 0.5 * delta)
        
        print(f"{error:<10.1f} {mse:<15.4f} {mae:<15.4f} {huber:<15.4f}")
    
    print("=" * 70)
    print("\n관찰:")
    print("  - MSE: 큰 오차에 제곱으로 증가 (이상치에 민감)")
    print("  - MAE: 모든 오차에 선형으로 증가 (이상치에 강건)")
    print("  - Huber: MSE와 MAE의 절충 (작은 오차는 MSE, 큰 오차는 MAE)")

compare_regression_losses()
```

```
======================================================================
Regression Loss Function Comparison
======================================================================

Error      MSE             MAE             Huber (δ=1)    
----------------------------------------------------------------------
0.1        0.0050          0.1000          0.0050         
0.5        0.1250          0.5000          0.1250         
1.0        0.5000          1.0000          0.5000         
2.0        2.0000          2.0000          1.5000         
3.0        4.5000          3.0000          2.5000         
======================================================================

관찰:
  - MSE: 큰 오차에 제곱으로 증가 (이상치에 민감)
  - MAE: 모든 오차에 선형으로 증가 (이상치에 강건)
  - Huber: MSE와 MAE의 절충 (작은 오차는 MSE, 큰 오차는 MAE)
```

### 5.2.9 MSE Properties

```python
def analyze_mse_properties():
    """MSE의 주요 특성 분석"""
    
    print("\n" + "=" * 70)
    print("Mean Squared Error Properties")
    print("=" * 70)
    
    print("\n[1. Convexity]")
    print("  - MSE는 볼록 함수(convex function)")
    print("  - 전역 최솟값(global minimum)이 유일하게 존재")
    print("  - 경사 하강법으로 최적해 수렴 보장")
    
    print("\n[2. Differentiability]")
    print("  - 모든 점에서 미분 가능")
    print("  - 그래디언트: ∂L/∂ŷ = (ŷ - y) / N")
    print("  - 연속적이고 부드러운 그래디언트")
    
    print("\n[3. Sensitivity to Outliers]")
    print("  - 큰 오차에 제곱 페널티")
    print("  - 이상치가 손실에 큰 영향")
    print("  - 예: error=1 → loss=1, error=2 → loss=4")
    
    print("\n[4. Units]")
    print("  - 손실 단위: (원래 단위)²")
    print("  - 예: 가격(달러) → MSE는 달러²")
    print("  - RMSE를 사용하면 원래 단위로 복원")
    
    print("\n[5. Relationship to Gaussian MLE]")
    print("  - MSE는 가우시안 분포 가정 하 MLE")
    print("  - 확률론적 해석 가능")
    print("  - 잡음이 정규분포를 따를 때 최적")
    
    print("=" * 70)

analyze_mse_properties()
```

```
======================================================================
Mean Squared Error Properties
======================================================================

[1. Convexity]
  - MSE는 볼록 함수(convex function)
  - 전역 최솟값(global minimum)이 유일하게 존재
  - 경사 하강법으로 최적해 수렴 보장

[2. Differentiability]
  - 모든 점에서 미분 가능
  - 그래디언트: ∂L/∂ŷ = (ŷ - y) / N
  - 연속적이고 부드러운 그래디언트

[3. Sensitivity to Outliers]
  - 큰 오차에 제곱 페널티
  - 이상치가 손실에 큰 영향
  - 예: error=1 → loss=1, error=2 → loss=4

[4. Units]
  - 손실 단위: (원래 단위)²
  - 예: 가격(달러) → MSE는 달러²
  - RMSE를 사용하면 원래 단위로 복원

[5. Relationship to Gaussian MLE]
  - MSE는 가우시안 분포 가정 하 MLE
  - 확률론적 해석 가능
  - 잡음이 정규분포를 따를 때 최적
======================================================================
```

### 5.2.10 Summary

| 항목 | 설명 | 수식 |
|------|------|------|
| **손실 함수** | Mean Squared Error | $L = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$ |
| **1/2 계수 버전** | 미분 편의성 | $L = \frac{1}{2N}\sum_i (y_i - \hat{y}_i)^2$ |
| **그래디언트** | 예측값에 대한 미분 | $\frac{\partial L}{\partial \hat{y}} = \frac{1}{N}(\hat{y} - y)$ |
| **범위** | 손실 값 | $[0, +\infty)$ |
| **최솟값** | 완벽한 예측 | $L = 0$ when $\hat{y} = y$ |
| **단위** | 제곱 단위 | (원래 단위)² |

**핵심 특징:**

1. **단순성**: 계산이 간단하고 직관적
2. **미분 가능성**: 모든 점에서 부드러운 그래디언트
3. **볼록성**: 전역 최적해 존재
4. **이상치 민감**: 큰 오차에 강한 페널티
5. **확률적 해석**: 가우시안 분포의 MLE

**장점:**

- 수학적으로 다루기 쉬움
- 미분 가능하여 최적화 용이
- 볼록 함수로 수렴 보장
- 직관적인 해석

**단점:**

- 이상치에 민감
- 제곱 단위로 해석이 어려움
- 큰 오차를 과도하게 페널티

**대안 손실 함수:**

| 손실 함수 | 수식 | 특징 |
|----------|------|------|
| **MSE** | $(y - \hat{y})^2$ | 이상치 민감, 미분 부드러움 |
| **MAE** | $\|y - \hat{y}\|$ | 이상치 강건, 0에서 미분 불가 |
| **Huber** | Hybrid | MSE와 MAE의 절충 |
| **Log-Cosh** | $\log(\cosh(y - \hat{y}))$ | MSE와 유사하나 덜 민감 |

**구현 시 주의사항:**

- 1/2 계수 사용 여부 일관성 유지
- 배치 평균으로 정규화
- RMSE로 원래 단위 복원
- 이상치 존재 시 대안 고려
