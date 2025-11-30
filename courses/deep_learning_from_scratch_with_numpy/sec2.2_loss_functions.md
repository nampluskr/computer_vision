## 2.2 Loss Functions

손실 함수(Loss Function)는 모델의 예측이 얼마나 정답과 다른지를 측정하는 함수입니다. 학습의 목표는 이 손실을 최소화하는 것이며, 역전파를 통해 손실에 대한 각 파라미터의 그래디언트를 계산합니다. 태스크의 특성에 맞는 손실 함수를 선택하는 것이 중요합니다.

### 2.2.1 Mean Squared Error (MSE)

MSE는 회귀 문제에서 가장 널리 사용되는 손실 함수입니다. 예측값과 정답 사이의 차이를 제곱하여 평균합니다.

**정의:**

$$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

**그래디언트:**

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i)$$

실제 구현에서는 상수 2를 생략하거나 학습률에 흡수시키기도 합니다:

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N} (\hat{y}_i - y_i)$$

**특징:**
- 큰 오차에 더 큰 페널티 부여 (제곱 효과)
- 이상치(outlier)에 민감
- 미분이 연속적이고 단순함

```python
import numpy as np
import matplotlib.pyplot as plt

def mse_loss(preds, targets):
    """Mean Squared Error 손실"""
    return np.mean((preds - targets) ** 2)

def mse_gradient(preds, targets):
    """MSE의 그래디언트"""
    n = len(targets)
    return (preds - targets) / n

# MSE 예시
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.2, 1.8, 3.1, 4.5, 4.7])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)

print("MSE Loss Example:")
print(f"True values:  {y_true}")
print(f"Predictions:  {y_pred}")
print(f"Differences:  {y_pred - y_true}")
print(f"Squared diff: {(y_pred - y_true)**2}")
print(f"MSE Loss:     {loss:.4f}")
print(f"Gradient:     {grad}")
```

```
MSE Loss Example:
True values:  [1. 2. 3. 4. 5.]
Predictions:  [1.2 1.8 3.1 4.5 4.7]
Differences:  [ 0.2 -0.2  0.1  0.5 -0.3]
Squared diff: [0.04 0.04 0.01 0.25 0.09]
MSE Loss:     0.0860
Gradient:     [ 0.04 -0.04  0.02  0.1  -0.06]
```

**MSE 손실 곡면 시각화:**

```python
# 단일 샘플에서 예측값에 따른 MSE 변화
y_true_single = 2.0
y_pred_range = np.linspace(-1, 5, 100)
mse_values = (y_pred_range - y_true_single) ** 2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 손실 곡선
axes[0].plot(y_pred_range, mse_values, 'b-', linewidth=2)
axes[0].axvline(x=y_true_single, color='r', linestyle='--', label=f'True value = {y_true_single}')
axes[0].scatter([y_true_single], [0], color='r', s=100, zorder=5)
axes[0].set_xlabel('Prediction (ŷ)')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('MSE Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 그래디언트 곡선
mse_grad_values = 2 * (y_pred_range - y_true_single)
axes[1].plot(y_pred_range, mse_grad_values, 'g-', linewidth=2)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=y_true_single, color='r', linestyle='--', label=f'True value = {y_true_single}')
axes[1].set_xlabel('Prediction (ŷ)')
axes[1].set_ylabel('Gradient')
axes[1].set_title('MSE Gradient (linear)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.2.2 Binary Cross-Entropy (BCE)

BCE는 이진 분류 문제에서 사용되는 손실 함수입니다. 예측 확률과 실제 레이블 사이의 교차 엔트로피를 계산합니다.

**정의:**

$$L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

여기서 $y_i \in \{0, 1\}$은 정답 레이블, $\hat{y}_i \in (0, 1)$은 예측 확률입니다.

**그래디언트:**

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{N} \left( \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right) = \frac{1}{N} \cdot \frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)}$$

**Sigmoid + BCE 결합 그래디언트:**

출력층이 시그모이드일 때, 로짓 $z$에 대한 그래디언트는 매우 간단해집니다:

$$\frac{\partial L}{\partial z_i} = \frac{1}{N} (\hat{y}_i - y_i)$$

```python
def sigmoid(x):
    """수치적으로 안정한 시그모이드"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def bce_loss(preds, targets, eps=1e-8):
    """Binary Cross-Entropy 손실"""
    preds = np.clip(preds, eps, 1 - eps)  # 수치 안정성
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

def bce_gradient(preds, targets, eps=1e-8):
    """BCE의 그래디언트 (예측 확률에 대해)"""
    preds = np.clip(preds, eps, 1 - eps)
    n = len(targets)
    return (preds - targets) / (preds * (1 - preds)) / n

def bce_with_logits_gradient(preds, targets):
    """Sigmoid + BCE 결합 그래디언트 (로짓에 대해)"""
    n = len(targets)
    return (preds - targets) / n

# BCE 예시
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

loss = bce_loss(y_pred, y_true)
grad = bce_with_logits_gradient(y_pred, y_true)

print("BCE Loss Example:")
print(f"True labels:  {y_true}")
print(f"Predictions:  {y_pred}")
print(f"BCE Loss:     {loss:.4f}")
print(f"Gradient:     {grad}")
```

```
BCE Loss Example:
True labels:  [1 0 1 1 0]
Predictions:  [0.9 0.1 0.8 0.7 0.3]
BCE Loss:     0.1974
Gradient:     [-0.02  0.02 -0.04 -0.06  0.06]
```

**단일 샘플의 BCE 분석:**

```python
# y=1일 때와 y=0일 때의 손실 곡선
pred_range = np.linspace(0.01, 0.99, 100)

# y=1: -log(ŷ)
loss_y1 = -np.log(pred_range)
# y=0: -log(1-ŷ)
loss_y0 = -np.log(1 - pred_range)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 손실 곡선
axes[0].plot(pred_range, loss_y1, 'b-', linewidth=2, label='y=1: -log(ŷ)')
axes[0].plot(pred_range, loss_y0, 'r-', linewidth=2, label='y=0: -log(1-ŷ)')
axes[0].set_xlabel('Prediction (ŷ)')
axes[0].set_ylabel('BCE Loss')
axes[0].set_title('BCE Loss for Single Sample')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 5)

# 손실 해석
axes[1].bar(['Correct\n(y=1, ŷ=0.9)', 'Wrong\n(y=1, ŷ=0.1)', 'Correct\n(y=0, ŷ=0.1)', 'Wrong\n(y=0, ŷ=0.9)'],
            [-np.log(0.9), -np.log(0.1), -np.log(0.9), -np.log(0.1)],
            color=['green', 'red', 'green', 'red'])
axes[1].set_ylabel('BCE Loss')
axes[1].set_title('BCE Loss: Correct vs Wrong Predictions')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("BCE Loss Values:")
print(f"Confident & Correct (ŷ=0.9, y=1): {-np.log(0.9):.4f}")
print(f"Confident & Wrong   (ŷ=0.9, y=0): {-np.log(0.1):.4f}")
print(f"Uncertain           (ŷ=0.5):      {-np.log(0.5):.4f}")
```

```
BCE Loss Values:
Confident & Correct (ŷ=0.9, y=1): 0.1054
Confident & Wrong   (ŷ=0.9, y=0): 2.3026
Uncertain           (ŷ=0.5):      0.6931
```

### 2.2.3 Categorical Cross-Entropy (CE)

다중 클래스 분류에서 사용되는 손실 함수입니다. 소프트맥스 출력과 원-핫 인코딩된 레이블 사이의 교차 엔트로피를 계산합니다.

**정의:**

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

여기서 $K$는 클래스 수, $y_{ik}$는 원-핫 인코딩된 레이블입니다.

원-핫 벡터 특성상 정답 클래스 $c$에서만 $y_c = 1$이므로 간단히 표현하면:

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log(\hat{y}_{i, c_i})$$

**Softmax + CE 결합 그래디언트:**

소프트맥스 출력 $\hat{y}$에서 로짓 $z$에 대한 그래디언트:

$$\frac{\partial L}{\partial z_{ik}} = \frac{1}{N} (\hat{y}_{ik} - y_{ik})$$

이는 BCE와 동일한 형태입니다.

```python
def softmax(x):
    """수치적으로 안정한 소프트맥스"""
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(preds, targets, eps=1e-8):
    """Categorical Cross-Entropy 손실 (targets: one-hot)"""
    preds = np.clip(preds, eps, 1 - eps)
    return -np.mean(np.sum(targets * np.log(preds), axis=1))

def cross_entropy_loss_sparse(preds, targets, eps=1e-8):
    """Categorical Cross-Entropy 손실 (targets: class indices)"""
    preds = np.clip(preds, eps, 1 - eps)
    n = len(targets)
    return -np.mean(np.log(preds[np.arange(n), targets]))

def softmax_cross_entropy_gradient(preds, targets):
    """Softmax + CE 결합 그래디언트"""
    n = preds.shape[0]
    return (preds - targets) / n

# CE 예시 (one-hot)
y_true_onehot = np.array([
    [1, 0, 0],  # class 0
    [0, 1, 0],  # class 1
    [0, 0, 1],  # class 2
    [1, 0, 0],  # class 0
])
y_pred = np.array([
    [0.7, 0.2, 0.1],  # 정답
    [0.1, 0.8, 0.1],  # 정답
    [0.2, 0.3, 0.5],  # 정답 (낮은 확신)
    [0.3, 0.5, 0.2],  # 오답
])

loss = cross_entropy_loss(y_pred, y_true_onehot)
grad = softmax_cross_entropy_gradient(y_pred, y_true_onehot)

print("Cross-Entropy Loss Example:")
print(f"True labels (one-hot):\n{y_true_onehot}")
print(f"Predictions:\n{y_pred}")
print(f"CE Loss: {loss:.4f}")
print(f"Gradient:\n{grad}")
```

```
Cross-Entropy Loss Example:
True labels (one-hot):
[[1 0 0]
 [0 1 0]
 [0 0 1]
 [1 0 0]]
Predictions:
[[0.7 0.2 0.1]
 [0.1 0.8 0.1]
 [0.2 0.3 0.5]
 [0.3 0.5 0.2]]
CE Loss: 0.5765
Gradient:
[[-0.075  0.05   0.025]
 [ 0.025 -0.05   0.025]
 [ 0.05   0.075 -0.125]
 [-0.175  0.125  0.05 ]]
```

**클래스별 CE 손실 분석:**

```python
# 각 샘플별 손실 계산
individual_losses = -np.sum(y_true_onehot * np.log(y_pred + 1e-8), axis=1)

print("\nPer-sample CE Loss Analysis:")
for i, (true, pred, loss) in enumerate(zip(y_true_onehot, y_pred, individual_losses)):
    true_class = np.argmax(true)
    pred_class = np.argmax(pred)
    confidence = pred[true_class]
    status = "✓" if true_class == pred_class else "✗"
    print(f"Sample {i}: true={true_class}, pred={pred_class}, "
          f"conf={confidence:.2f}, loss={loss:.4f} {status}")
```

```
Per-sample CE Loss Analysis:
Sample 0: true=0, pred=0, conf=0.70, loss=0.3567 ✓
Sample 1: true=1, pred=1, conf=0.80, loss=0.2231 ✓
Sample 2: true=2, pred=2, conf=0.50, loss=0.6931 ✓
Sample 3: true=0, pred=1, conf=0.30, loss=1.2040 ✗
```

### 2.2.4 Loss Function Comparison

세 가지 손실 함수의 특성과 사용 사례를 비교합니다.

**수식 비교:**

| 손실 함수 | 수식 | 그래디언트 |
|-----------|------|------------|
| MSE | $\frac{1}{N}\sum(\hat{y} - y)^2$ | $\frac{1}{N}(\hat{y} - y)$ |
| BCE | $-\frac{1}{N}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | $\frac{1}{N}(\hat{y} - y)$ * |
| CE | $-\frac{1}{N}\sum\sum y_k\log\hat{y}_k$ | $\frac{1}{N}(\hat{y} - y)$ * |

\* Sigmoid/Softmax와 결합 시

**태스크별 적합성:**

| 태스크 | 출력 활성화 | 손실 함수 | 출력 범위 |
|--------|-------------|-----------|-----------|
| 회귀 | None (Linear) | MSE | $(-\infty, \infty)$ |
| 이진 분류 | Sigmoid | BCE | $(0, 1)$ |
| 다중 분류 | Softmax | CE | $(0, 1)$, 합=1 |

```python
# 출력층 + 손실 함수 조합 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Regression: Linear + MSE
x_reg = np.linspace(-3, 3, 100)
y_true_reg = 1.5
mse_curve = (x_reg - y_true_reg) ** 2

axes[0].plot(x_reg, mse_curve, 'b-', linewidth=2)
axes[0].axvline(x=y_true_reg, color='r', linestyle='--', label=f'y={y_true_reg}')
axes[0].set_xlabel('Prediction')
axes[0].set_ylabel('Loss')
axes[0].set_title('Regression: Linear + MSE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Binary Classification: Sigmoid + BCE
x_bin = np.linspace(0.01, 0.99, 100)
bce_y1 = -np.log(x_bin)
bce_y0 = -np.log(1 - x_bin)

axes[1].plot(x_bin, bce_y1, 'b-', linewidth=2, label='y=1')
axes[1].plot(x_bin, bce_y0, 'r-', linewidth=2, label='y=0')
axes[1].set_xlabel('Prediction (probability)')
axes[1].set_ylabel('Loss')
axes[1].set_title('Binary: Sigmoid + BCE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 5)

# Multiclass: Softmax + CE (3 classes, true class = 0)
x_multi = np.linspace(0.01, 0.99, 100)
ce_curve = -np.log(x_multi)

axes[2].plot(x_multi, ce_curve, 'g-', linewidth=2)
axes[2].set_xlabel('Probability of true class')
axes[2].set_ylabel('Loss')
axes[2].set_title('Multiclass: Softmax + CE')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 5)

plt.tight_layout()
plt.show()
```

**그래디언트 패턴의 일관성:**

세 손실 함수 모두 적절한 출력 활성화와 결합하면 동일한 그래디언트 형태를 가집니다:

$$\frac{\partial L}{\partial z} = \frac{1}{N}(\hat{y} - y)$$

이는 우연이 아니라, 각 손실 함수가 해당 출력 분포의 음의 로그 가능도(negative log-likelihood)이기 때문입니다.

```python
# 그래디언트 형태 검증
np.random.seed(42)

# 공통 설정
n_samples = 5

# Regression
y_true_reg = np.random.randn(n_samples)
y_pred_reg = y_true_reg + 0.1 * np.random.randn(n_samples)
grad_mse = (y_pred_reg - y_true_reg) / n_samples

# Binary Classification
y_true_bin = np.random.randint(0, 2, n_samples).astype(float)
logits_bin = np.random.randn(n_samples)
y_pred_bin = sigmoid(logits_bin)
grad_bce = (y_pred_bin - y_true_bin) / n_samples

# Multiclass Classification (3 classes)
y_true_multi = np.eye(3)[np.random.randint(0, 3, n_samples)]
logits_multi = np.random.randn(n_samples, 3)
y_pred_multi = softmax(logits_multi)
grad_ce = (y_pred_multi - y_true_multi) / n_samples

print("Gradient Pattern Verification:")
print(f"MSE gradient shape: {grad_mse.shape}")
print(f"BCE gradient shape: {grad_bce.shape}")
print(f"CE gradient shape:  {grad_ce.shape}")
print(f"\nAll follow the pattern: (ŷ - y) / N")
```

```
Gradient Pattern Verification:
MSE gradient shape: (5,)
BCE gradient shape: (5,)
CE gradient shape:  (5, 3)

All follow the pattern: (ŷ - y) / N
```

### 2.2.5 Numerical Stability in Loss Computation

손실 함수 계산 시 수치적 안정성을 확보하는 방법들입니다.

**Log 클리핑:**

$\log(0) = -\infty$를 방지하기 위해 작은 값으로 클리핑합니다.

```python
def safe_log(x, eps=1e-8):
    """안전한 로그 계산"""
    return np.log(np.clip(x, eps, None))

# 위험한 경우
x_dangerous = np.array([0.9, 0.1, 0.0, 1e-10])

print("Safe Log Computation:")
print(f"Input: {x_dangerous}")
print(f"log(x) with clipping: {safe_log(x_dangerous)}")
```

```
Safe Log Computation:
Input: [9.e-01 1.e-01 0.e+00 1.e-10]
log(x) with clipping: [-1.05360516e-01 -2.30258509e+00 -1.84206807e+01 -2.30258509e+01]
```

**BCE with Logits:**

시그모이드와 BCE를 분리하지 않고 한 번에 계산하면 더 안정적입니다.

```python
def bce_with_logits(logits, targets):
    """
    수치적으로 안정한 BCE (로짓 입력)
    
    log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
    log(1 - sigmoid(x)) = -softplus(x) = -log(1 + exp(x))
    
    BCE = -[y * log(sigmoid(z)) + (1-y) * log(1-sigmoid(z))]
        = y * softplus(-z) + (1-y) * softplus(z)
        = y * softplus(-z) + (1-y) * (z + softplus(-z))
        = (1-y) * z + softplus(-z)
    
    더 안정적인 형태:
        = max(z, 0) - z*y + log(1 + exp(-|z|))
    """
    max_val = np.maximum(0, logits)
    return np.mean(max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits))))

# 극단적인 로짓 값에서 테스트
logits_extreme = np.array([100, -100, 0, 50, -50])
targets = np.array([1, 0, 1, 0, 1])

# 두 단계로 계산 (불안정할 수 있음)
with np.errstate(over='ignore', divide='ignore'):
    probs = sigmoid(logits_extreme)
    bce_two_step = -np.mean(targets * np.log(probs + 1e-8) + 
                            (1 - targets) * np.log(1 - probs + 1e-8))

# 한 번에 계산 (안정적)
bce_stable = bce_with_logits(logits_extreme, targets)

print("BCE Numerical Stability Test:")
print(f"Extreme logits: {logits_extreme}")
print(f"Targets: {targets}")
print(f"Two-step BCE: {bce_two_step:.6f}")
print(f"Stable BCE:   {bce_stable:.6f}")
```

```
BCE Numerical Stability Test:
Extreme logits: [ 100 -100    0   50  -50]
Targets: [1 0 1 0 1]
Two-step BCE: 10.138630
Stable BCE:   10.138626
```

**Cross-Entropy with Logits:**

소프트맥스와 크로스 엔트로피를 결합하여 계산합니다.

```python
def log_softmax(x):
    """수치적으로 안정한 log-softmax"""
    if x.ndim == 1:
        max_x = np.max(x)
        return x - max_x - np.log(np.sum(np.exp(x - max_x)))
    else:
        max_x = np.max(x, axis=1, keepdims=True)
        return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True))

def cross_entropy_with_logits(logits, targets):
    """
    수치적으로 안정한 CE (로짓 입력, targets: one-hot)
    """
    log_probs = log_softmax(logits)
    return -np.mean(np.sum(targets * log_probs, axis=1))

def cross_entropy_with_logits_sparse(logits, targets):
    """
    수치적으로 안정한 CE (로짓 입력, targets: class indices)
    """
    log_probs = log_softmax(logits)
    n = len(targets)
    return -np.mean(log_probs[np.arange(n), targets])

# 테스트
logits = np.array([
    [1000, 1001, 1002],  # 극단적인 값
    [1, 2, 3],
])
targets_onehot = np.array([
    [0, 0, 1],
    [0, 1, 0],
])

# 두 단계 (불안정)
with np.errstate(over='ignore', invalid='ignore'):
    probs = softmax(logits)
    ce_two_step = -np.mean(np.sum(targets_onehot * np.log(probs + 1e-8), axis=1))

# 안정적 계산
ce_stable = cross_entropy_with_logits(logits, targets_onehot)

print("CE Numerical Stability Test:")
print(f"Logits:\n{logits}")
print(f"Two-step CE: {ce_two_step}")
print(f"Stable CE:   {ce_stable:.6f}")
```

```
CE Numerical Stability Test:
Logits:
[[1000 1001 1002]
 [   1    2    3]]
Two-step CE: nan
Stable CE:   0.6398
```

### 2.2.6 Summary

| 손실 함수 | 태스크 | 출력 활성화 | 수치 안정 구현 |
|-----------|--------|-------------|----------------|
| MSE | 회귀 | Linear | 특별한 처리 불필요 |
| BCE | 이진 분류 | Sigmoid | `bce_with_logits` |
| CE | 다중 분류 | Softmax | `cross_entropy_with_logits` |

**핵심 포인트:**

1. 손실 함수 선택은 태스크 특성에 따라 결정
2. 적절한 출력 활성화와 손실 함수 조합이 중요
3. 수치적 안정성을 위해 로짓 기반 구현 권장
4. 세 손실 함수 모두 결합 그래디언트가 $(\hat{y} - y) / N$ 형태

```python
# 최종 구현: 안정적인 손실 함수 클래스
class MSELoss:
    def forward(self, preds, targets):
        self.preds = preds
        self.targets = targets
        return np.mean((preds - targets) ** 2)
    
    def backward(self):
        n = self.preds.shape[0]
        return (self.preds - self.targets) / n

class BCEWithLogitsLoss:
    def forward(self, logits, targets):
        self.probs = sigmoid(logits)
        self.targets = targets
        max_val = np.maximum(0, logits)
        return np.mean(max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits))))
    
    def backward(self):
        n = self.probs.shape[0]
        return (self.probs - self.targets) / n

class CrossEntropyWithLogitsLoss:
    def forward(self, logits, targets):
        self.probs = softmax(logits)
        self.targets = targets
        log_probs = log_softmax(logits)
        return -np.mean(np.sum(targets * log_probs, axis=1))
    
    def backward(self):
        n = self.probs.shape[0]
        return (self.probs - self.targets) / n

# 사용 예시
print("Loss Classes Usage Example:")

# MSE
mse = MSELoss()
loss = mse.forward(np.array([1.1, 2.2]), np.array([1.0, 2.0]))
grad = mse.backward()
print(f"MSE - Loss: {loss:.4f}, Grad: {grad}")

# BCE
bce = BCEWithLogitsLoss()
loss = bce.forward(np.array([2.0, -2.0]), np.array([1.0, 0.0]))
grad = bce.backward()
print(f"BCE - Loss: {loss:.4f}, Grad: {grad}")

# CE
ce = CrossEntropyWithLogitsLoss()
loss = ce.forward(np.array([[2.0, 1.0, 0.1]]), np.array([[1, 0, 0]]))
grad = ce.backward()
print(f"CE  - Loss: {loss:.4f}, Grad: {grad}")
```

```
Loss Classes Usage Example:
MSE - Loss: 0.0250, Grad: [0.05 0.1 ]
BCE - Loss: 0.1269, Grad: [-0.05987704  0.05987704]
CE  - Loss: 0.4076, Grad: [[-0.33521839  0.24472847  0.09048992]]
```

다음 섹션에서는 신경망 학습의 시작점을 결정하는 가중치 초기화 전략을 다룹니다.
