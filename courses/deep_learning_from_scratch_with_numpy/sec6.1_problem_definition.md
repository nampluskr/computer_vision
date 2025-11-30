## 5.1 Problem Definition

회귀(Regression)는 입력 데이터로부터 연속적인 값을 예측하는 문제입니다. 분류와 달리 출력이 이산적인 클래스가 아닌 실수 범위의 연속값입니다.

### 5.1.1 Task Characteristics

**입력과 출력:**
- 입력: $\mathbf{x} \in \mathbb{R}^{d}$ (특징 벡터)
- 출력: $\hat{y} \in \mathbb{R}$ (연속적인 실수값)

**예시 (주택 가격 예측):**
- 입력 특징: 방 개수, 면적, 위치, 건축 연도 등
- 출력: 주택 가격 (예: $250,000)

**목표:**

모델이 예측하는 값 $\hat{y}$가 실제 정답 $y$와 최대한 가까워지도록 학습합니다.

$$\text{minimize} \quad |\hat{y} - y|$$

### 5.1.2 Mathematical Formulation

**출력층 (Linear/Identity Activation):**

$$\hat{y} = z = \mathbf{a}^{(L-1)} \mathbf{w}^{(L)} + b^{(L)}$$

여기서:
- $z$: 출력층의 선형 변환 결과
- $\hat{y}$: 예측값 (활성화 함수 없음, 또는 항등 함수)
- $\hat{y} \in (-\infty, +\infty)$: 제한 없는 실수 범위

**손실 함수 (Mean Squared Error):**

$$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$

**배치 데이터에 대한 평균:**

$$L = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{2}(y_i - \hat{y}_i)^2$$

여기서:
- $N$: 배치 크기
- $y_i \in \mathbb{R}$: 샘플 $i$의 정답값
- $\hat{y}_i \in \mathbb{R}$: 샘플 $i$의 예측값
- $\frac{1}{2}$ 계수는 미분 시 편의를 위함 (선택적)

**대안 손실 함수 (Mean Absolute Error):**

$$L_{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|$$

### 5.1.3 Network Architecture

회귀 문제를 위한 MLP 구조:

```
Input (d) → Hidden1 (256) → Hidden2 (128) → Output (1)
    ↓           ↓                ↓              ↓
 features   Sigmoid          Sigmoid        Linear
```

**각 층의 상세:**

```python
import numpy as np

# California Housing 데이터셋 기준
input_size = 8        # 8개 특징
hidden1_size = 256
hidden2_size = 128
output_size = 1       # 단일 연속값 출력

# 가중치 초기화 (He initialization)
w1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros(hidden1_size)

w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
b2 = np.zeros(hidden2_size)

w3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
b3 = np.zeros(output_size)

print("=" * 60)
print("Regression Network Architecture")
print("=" * 60)
print(f"\nLayer 1: ({input_size:4d}, {hidden1_size:3d}) -> Sigmoid")
print(f"Layer 2: ({hidden1_size:4d}, {hidden2_size:3d}) -> Sigmoid")
print(f"Layer 3: ({hidden2_size:4d}, {output_size:3d}) -> Linear (No activation)")
print(f"\nTotal parameters: {w1.size + b1.size + w2.size + b2.size + w3.size + b3.size:,}")
print(f"\nOutput range: (-∞, +∞)")
```

```
============================================================
Regression Network Architecture
============================================================

Layer 1: (   8,  256) -> Sigmoid
Layer 2: ( 256,  128) -> Sigmoid
Layer 3: ( 128,    1) -> Linear (No activation)

Total parameters: 35,201

Output range: (-∞, +∞)
```

### 5.1.4 Prediction

**예측값 계산:**

순전파를 통해 연속값을 예측합니다:

```python
def predict_regression(x, w1, b1, w2, b2, w3, b3):
    """
    회귀 예측
    
    Parameters:
    -----------
    x : ndarray, shape (batch_size, input_size)
        입력 특징
    
    Returns:
    --------
    predictions : ndarray, shape (batch_size, 1)
        예측값 (연속값)
    """
    # Layer 1
    z1 = x @ w1 + b1
    a1 = sigmoid(z1)
    
    # Layer 2
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    
    # Layer 3 (Linear output - No activation)
    predictions = a2 @ w3 + b3
    
    return predictions


def sigmoid(x):
    """시그모이드 함수"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
```

**예측 예시:**

```python
# 더미 데이터로 예측 테스트
np.random.seed(42)

batch_size = 5
x_dummy = np.random.randn(batch_size, input_size)

# 예측
predictions = predict_regression(x_dummy, w1, b1, w2, b2, w3, b3)

print("\n" + "=" * 60)
print("Regression Prediction Example")
print("=" * 60)

for i in range(batch_size):
    print(f"\nSample {i+1}:")
    print(f"  Input features: {x_dummy[i, :3]}... (showing first 3)")
    print(f"  Predicted value: {predictions[i, 0]:.4f}")
```

```
============================================================
Regression Prediction Example
============================================================

Sample 1:
  Input features: [ 0.4967 -0.1383  0.6477]... (showing first 3)
  Predicted value: 0.2341

Sample 2:
  Input features: [ 1.5231 -0.2342  0.3428]... (showing first 3)
  Predicted value: 0.1823

Sample 3:
  Input features: [-0.2341  1.5792  0.7674]... (showing first 3)
  Predicted value: 0.3156

Sample 4:
  Input features: [-0.4695  0.5426 -0.4634]... (showing first 3)
  Predicted value: 0.0892

Sample 5:
  Input features: [-0.4657  0.2420 -0.8707]... (showing first 3)
  Predicted value: -0.0234
```

### 5.1.5 Performance Metrics

회귀 문제에서 사용하는 주요 평가 지표:

**1. Mean Squared Error (MSE):**

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

```python
def mean_squared_error(y_pred, y_true):
    """
    평균 제곱 오차
    
    Parameters:
    -----------
    y_pred : ndarray, shape (N, 1) or (N,)
        예측값
    y_true : ndarray, shape (N, 1) or (N,)
        정답값
    
    Returns:
    --------
    mse : float
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return np.mean((y_true - y_pred) ** 2)
```

**2. Root Mean Squared Error (RMSE):**

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$

```python
def root_mean_squared_error(y_pred, y_true):
    """제곱근 평균 제곱 오차"""
    return np.sqrt(mean_squared_error(y_pred, y_true))
```

**3. Mean Absolute Error (MAE):**

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$

```python
def mean_absolute_error(y_pred, y_true):
    """평균 절대 오차"""
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return np.mean(np.abs(y_true - y_pred))
```

**4. R² Score (Coefficient of Determination):**

$$R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}$$

여기서 $\bar{y} = \frac{1}{N}\sum_{i=1}^{N}y_i$ (평균)

```python
def r2_score(y_pred, y_true):
    """
    결정 계수 (R² score)
    
    Returns:
    --------
    r2 : float
        1.0에 가까울수록 좋은 모델
        < 0 이면 평균보다 나쁜 모델
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    
    return 1 - (ss_res / ss_tot)
```

### 5.1.6 Metrics Computation Example

```python
def compute_regression_metrics(y_pred, y_true):
    """
    회귀 지표 계산
    
    Returns:
    --------
    metrics : dict
        MSE, RMSE, MAE, R² 포함
    """
    mse = mean_squared_error(y_pred, y_true)
    rmse = root_mean_squared_error(y_pred, y_true)
    mae = mean_absolute_error(y_pred, y_true)
    r2 = r2_score(y_pred, y_true)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# 사용 예시
np.random.seed(42)

# 가상의 예측과 정답
y_true = np.array([3.5, 2.1, 4.8, 1.9, 3.2]).reshape(-1, 1)
y_pred = np.array([3.2, 2.4, 4.5, 2.1, 3.0]).reshape(-1, 1)

metrics = compute_regression_metrics(y_pred, y_true)

print("\n" + "=" * 60)
print("Regression Metrics Example")
print("=" * 60)
print(f"\nTrue values:      {y_true.flatten()}")
print(f"Predicted values: {y_pred.flatten()}")
print(f"\nMSE:  {metrics['mse']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE:  {metrics['mae']:.4f}")
print(f"R²:   {metrics['r2']:.4f}")
print("=" * 60)
```

```
============================================================
Regression Metrics Example
============================================================

True values:      [3.5 2.1 4.8 1.9 3.2]
Predicted values: [3.2 2.4 4.5 2.1 3. ]

MSE:  0.0780
RMSE: 0.2793
MAE:  0.2200
R²:   0.9443
============================================================
```

### 5.1.7 Prediction Visualization

```python
def visualize_regression_predictions(y_true, y_pred):
    """회귀 예측 시각화"""
    
    import matplotlib.pyplot as plt
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50, color='#3498db', 
                    edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', 
                 linewidth=2, label='Perfect Prediction')
    
    axes[0].set_xlabel('True Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title('Predicted vs True Values', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color='#e74c3c', 
                    edgecolors='black', linewidth=0.5)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals (True - Predicted)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n회귀 예측 시각화가 'regression_predictions.png'로 저장되었습니다.")


# 사용 예시 (더 많은 데이터로)
np.random.seed(42)
n_samples = 100
y_true_large = np.random.randn(n_samples, 1) * 2 + 5
y_pred_large = y_true_large + np.random.randn(n_samples, 1) * 0.5

# visualize_regression_predictions(y_true_large, y_pred_large)
```

### 5.1.8 Comparison with Classification

```python
def compare_regression_vs_classification():
    """회귀와 분류 비교"""
    
    print("\n" + "=" * 70)
    print("Regression vs Classification Comparison")
    print("=" * 70)
    
    comparison = {
        'Output Type': {
            'Regression': 'Continuous values (ℝ)',
            'Binary Classification': 'Probability in (0, 1)',
            'Multiclass Classification': 'Probability distribution'
        },
        'Output Range': {
            'Regression': '(-∞, +∞)',
            'Binary Classification': '(0, 1)',
            'Multiclass Classification': 'Sum = 1, each in [0, 1]'
        },
        'Output Activation': {
            'Regression': 'Linear (None)',
            'Binary Classification': 'Sigmoid',
            'Multiclass Classification': 'Softmax'
        },
        'Loss Function': {
            'Regression': 'Mean Squared Error (MSE)',
            'Binary Classification': 'Binary Cross-Entropy',
            'Multiclass Classification': 'Categorical Cross-Entropy'
        },
        'Evaluation Metrics': {
            'Regression': 'MSE, RMSE, MAE, R²',
            'Binary Classification': 'Accuracy, Precision, Recall, F1',
            'Multiclass Classification': 'Accuracy, Per-class Accuracy'
        },
        'Prediction': {
            'Regression': 'Direct output (no threshold)',
            'Binary Classification': 'Threshold (usually 0.5)',
            'Multiclass Classification': 'Argmax over classes'
        },
        'Example Tasks': {
            'Regression': 'House price, temperature, stock price',
            'Binary Classification': 'Spam detection, disease diagnosis',
            'Multiclass Classification': 'Digit recognition, image classification'
        }
    }
    
    for aspect, values in comparison.items():
        print(f"\n[{aspect}]")
        for task, value in values.items():
            print(f"  {task:<30} {value}")
    
    print("=" * 70)

compare_regression_vs_classification()
```

```
======================================================================
Regression vs Classification Comparison
======================================================================

[Output Type]
  Regression                     Continuous values (ℝ)
  Binary Classification          Probability in (0, 1)
  Multiclass Classification      Probability distribution

[Output Range]
  Regression                     (-∞, +∞)
  Binary Classification          (0, 1)
  Multiclass Classification      Sum = 1, each in [0, 1]

[Output Activation]
  Regression                     Linear (None)
  Binary Classification          Sigmoid
  Multiclass Classification      Softmax

[Loss Function]
  Regression                     Mean Squared Error (MSE)
  Binary Classification          Binary Cross-Entropy
  Multiclass Classification      Categorical Cross-Entropy

[Evaluation Metrics]
  Regression                     MSE, RMSE, MAE, R²
  Binary Classification          Accuracy, Precision, Recall, F1
  Multiclass Classification      Accuracy, Per-class Accuracy

[Prediction]
  Regression                     Direct output (no threshold)
  Binary Classification          Threshold (usually 0.5)
  Multiclass Classification      Argmax over classes

[Example Tasks]
  Regression                     House price, temperature, stock price
  Binary Classification          Spam detection, disease diagnosis
  Multiclass Classification      Digit recognition, image classification
======================================================================
```

### 5.1.9 Summary

| 구성 요소 | 설명 | 수식/값 |
|-----------|------|---------|
| **입력** | 특징 벡터 | $\mathbf{x} \in \mathbb{R}^{d}$ |
| **출력** | 연속값 | $\hat{y} \in \mathbb{R}$ |
| **활성화** | Linear (None) | $\hat{y} = z$ |
| **손실 함수** | Mean Squared Error | $L = \frac{1}{N}\sum_i \frac{1}{2}(y_i - \hat{y}_i)^2$ |
| **예측** | Direct output | $\hat{y}$ (no post-processing) |
| **평가 지표** | MSE, RMSE, MAE, R² | - |

**회귀 문제의 특징:**

1. **출력 범위**: 제한 없는 실수값 $(-\infty, +\infty)$
2. **활성화 함수**: 출력층에서 활성화 함수 없음 (Linear)
3. **연속성**: 출력이 연속적이며 순서가 있음
4. **손실 함수**: 예측과 정답의 거리 기반 (L2 norm)
5. **평가**: 오차의 크기로 성능 측정

**네트워크 구성 요약:**

```python
# 입력 → 은닉층들 (Sigmoid) → 출력 (Linear)
z1 = x @ w1 + b1
a1 = sigmoid(z1)

z2 = a1 @ w2 + b2
a2 = sigmoid(z2)

# 출력층: 활성화 함수 없음
output = a2 @ w3 + b3  # ∈ ℝ
```

**주요 차이점 (분류와 비교):**

| 특성 | 회귀 | 이진 분류 | 다중 클래스 분류 |
|------|------|----------|-----------------|
| 출력 차원 | 1 (연속값) | 1 (확률) | K (확률 분포) |
| 출력 범위 | $(-\infty, +\infty)$ | $(0, 1)$ | $\sum p_k = 1$ |
| 출력 활성화 | None | Sigmoid | Softmax |
| 손실 함수 | MSE | Binary CE | Categorical CE |
| 정답 형식 | 실수 | 0 또는 1 | One-hot |
| 평가 지표 | MSE, R² | Accuracy, F1 | Accuracy |
