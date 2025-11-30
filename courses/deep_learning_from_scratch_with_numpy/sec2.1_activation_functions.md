## 2.1 Activation Functions

활성화 함수(Activation Function)는 신경망에 비선형성을 도입하는 핵심 요소입니다. 활성화 함수가 없다면 여러 층의 선형 변환은 결국 하나의 선형 변환과 동일해지므로, 복잡한 패턴을 학습할 수 없습니다.

### 2.1.1 Why Non-linearity?

선형 변환만으로 구성된 네트워크의 한계를 살펴봅니다.

두 개의 선형 레이어가 있다고 가정합니다:

$$z^{(1)} = xW^{(1)} + b^{(1)}$$
$$z^{(2)} = z^{(1)}W^{(2)} + b^{(2)}$$

이를 전개하면:

$$z^{(2)} = (xW^{(1)} + b^{(1)})W^{(2)} + b^{(2)} = xW^{(1)}W^{(2)} + b^{(1)}W^{(2)} + b^{(2)}$$

$W' = W^{(1)}W^{(2)}$, $b' = b^{(1)}W^{(2)} + b^{(2)}$로 치환하면:

$$z^{(2)} = xW' + b'$$

결국 하나의 선형 변환과 동일합니다. 따라서 층을 깊게 쌓아도 표현력이 증가하지 않습니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 선형 변환만으로는 XOR 문제를 풀 수 없음
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# 선형 분류기 시도
W = np.array([1, 1])
b = -0.5
preds_linear = (X @ W + b > 0).astype(int)

print("XOR Problem with Linear Classifier:")
print(f"Input:\n{X}")
print(f"True labels: {y_xor}")
print(f"Predictions: {preds_linear}")
print(f"Accuracy: {np.mean(preds_linear == y_xor):.2f}")

# 시각화
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
colors = ['blue' if label == 0 else 'red' for label in y_xor]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black')
x_line = np.linspace(-0.5, 1.5, 100)
y_line = -x_line + 0.5  # W·x + b = 0 → x1 + x2 = 0.5
plt.plot(x_line, y_line, 'g--', label='Linear boundary')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR: Linear boundary fails')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black')
# 비선형 결정 경계 (개념적)
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(0.5 + 0.7*np.cos(theta), 0.5 + 0.7*np.sin(theta), 'g--', label='Non-linear boundary')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR: Non-linear boundary needed')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.1.2 Sigmoid

시그모이드 함수는 입력을 $(0, 1)$ 범위로 압축합니다. 확률을 출력해야 하는 이진 분류의 출력층에서 사용됩니다.

**정의:**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**미분:**

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**특징:**
- 출력 범위: $(0, 1)$
- 중심: $\sigma(0) = 0.5$
- 포화 영역: $|x|$가 크면 그래디언트가 0에 가까워짐 (vanishing gradient)

```python
def sigmoid(x):
    """수치적으로 안정한 시그모이드"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    """시그모이드의 미분"""
    s = sigmoid(x)
    return s * (1 - s)

# 시각화
x = np.linspace(-8, 8, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 함수
axes[0].plot(x, sigmoid(x), 'b-', linewidth=2)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('σ(x)')
axes[0].set_title('Sigmoid Function')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.1, 1.1)

# 미분
axes[1].plot(x, sigmoid_derivative(x), 'r-', linewidth=2)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel("σ'(x)")
axes[1].set_title('Sigmoid Derivative (max = 0.25)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 최대 미분값
print(f"Maximum derivative at x=0: {sigmoid_derivative(0):.4f}")
```

```
Maximum derivative at x=0: 0.2500
```

### 2.1.3 Tanh

하이퍼볼릭 탄젠트(Tanh)는 시그모이드의 스케일 및 이동 버전으로, 출력이 $(-1, 1)$ 범위입니다.

**정의:**

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**미분:**

$$\tanh'(x) = 1 - \tanh^2(x)$$

**특징:**
- 출력 범위: $(-1, 1)$
- 중심: $\tanh(0) = 0$ (zero-centered)
- 시그모이드보다 그래디언트가 큼 (최대 1.0)

```python
def tanh(x):
    """Tanh 함수"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh의 미분"""
    return 1 - np.tanh(x) ** 2

# 시그모이드와 비교
x = np.linspace(-4, 4, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 함수 비교
axes[0].plot(x, sigmoid(x), 'b-', linewidth=2, label='Sigmoid')
axes[0].plot(x, tanh(x), 'r-', linewidth=2, label='Tanh')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Sigmoid vs Tanh')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 미분 비교
axes[1].plot(x, sigmoid_derivative(x), 'b-', linewidth=2, label="Sigmoid' (max=0.25)")
axes[1].plot(x, tanh_derivative(x), 'r-', linewidth=2, label="Tanh' (max=1.0)")
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel("f'(x)")
axes[1].set_title('Derivatives Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Sigmoid max derivative: {sigmoid_derivative(0):.4f}")
print(f"Tanh max derivative: {tanh_derivative(0):.4f}")
```

```
Sigmoid max derivative: 0.2500
Tanh max derivative: 1.0000
```

### 2.1.4 ReLU and Variants

ReLU(Rectified Linear Unit)는 현대 딥러닝에서 가장 널리 사용되는 활성화 함수입니다.

**ReLU 정의:**

$$\text{ReLU}(x) = \max(0, x)$$

**ReLU 미분:**

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**장점:**
- 계산이 매우 빠름
- 양수 영역에서 그래디언트가 1로 유지 (vanishing gradient 완화)
- 희소 활성화 (sparse activation)

**단점:**
- 음수 입력에서 그래디언트가 0 (dying ReLU 문제)

```python
def relu(x):
    """ReLU 함수"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU의 미분"""
    return (x > 0).astype(float)

# ReLU 시각화
x = np.linspace(-4, 4, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, relu(x), 'g-', linewidth=2)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('ReLU(x)')
axes[0].set_title('ReLU Function')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, relu_derivative(x), 'g-', linewidth=2)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel("ReLU'(x)")
axes[1].set_title('ReLU Derivative')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.1, 1.5)

plt.tight_layout()
plt.show()
```

**Leaky ReLU:**

Dying ReLU 문제를 해결하기 위해 음수 영역에 작은 기울기를 부여합니다.

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

일반적으로 $\alpha = 0.01$ 또는 $\alpha = 0.2$를 사용합니다.

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU 함수"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU의 미분"""
    return np.where(x > 0, 1, alpha)

# ReLU variants 비교
x = np.linspace(-4, 4, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, relu(x), 'g-', linewidth=2, label='ReLU')
axes[0].plot(x, leaky_relu(x, 0.1), 'b-', linewidth=2, label='Leaky ReLU (α=0.1)')
axes[0].plot(x, leaky_relu(x, 0.2), 'r-', linewidth=2, label='Leaky ReLU (α=0.2)')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('ReLU Variants')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, relu_derivative(x), 'g-', linewidth=2, label='ReLU')
axes[1].plot(x, leaky_relu_derivative(x, 0.1), 'b-', linewidth=2, label='Leaky ReLU (α=0.1)')
axes[1].plot(x, leaky_relu_derivative(x, 0.2), 'r-', linewidth=2, label='Leaky ReLU (α=0.2)')
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel("f'(x)")
axes[1].set_title('ReLU Variants Derivatives')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.1, 1.5)

plt.tight_layout()
plt.show()
```

### 2.1.5 Softmax

소프트맥스는 다중 클래스 분류의 출력층에서 사용됩니다. 로짓 벡터를 확률 분포로 변환합니다.

**정의:**

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**특징:**
- 모든 출력이 $(0, 1)$ 범위
- 출력의 합이 1 (확률 분포)
- 가장 큰 입력이 가장 높은 확률을 가짐

```python
def softmax(x):
    """수치적으로 안정한 소프트맥스"""
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

# 소프트맥스 예시
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print("Softmax Example:")
print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Sum: {np.sum(probs):.4f}")
print(f"Predicted class: {np.argmax(probs)}")

# 온도에 따른 소프트맥스 분포
def softmax_with_temperature(x, temperature=1.0):
    """온도 파라미터가 있는 소프트맥스"""
    return softmax(x / temperature)

logits = np.array([2.0, 1.0, 0.5])
temperatures = [0.5, 1.0, 2.0, 5.0]

print("\nSoftmax with Different Temperatures:")
print(f"Logits: {logits}")
for temp in temperatures:
    probs = softmax_with_temperature(logits, temp)
    print(f"T={temp}: {probs} (max={np.max(probs):.3f})")
```

```
Softmax Example:
Logits: [2.  1.  0.1]
Probabilities: [0.65900114 0.24243297 0.09856589]
Sum: 1.0000
Predicted class: 0

Softmax with Different Temperatures:
Logits: [2.  1.  0.5]
T=0.5: [0.84379473 0.1141952  0.04201007] (max=0.844)
T=1.0: [0.66524096 0.24472847 0.09003057] (max=0.665)
T=2.0: [0.50648039 0.30719589 0.18632372] (max=0.506)
T=5.0: [0.38961837 0.32898829 0.28139333] (max=0.390)
```

### 2.1.6 Numerical Stability

활성화 함수 구현 시 수치적 안정성을 고려해야 합니다. 특히 지수 함수는 오버플로우와 언더플로우에 취약합니다.

**시그모이드의 수치 안정성:**

```python
def sigmoid_naive(x):
    """단순 구현 (수치적으로 불안정)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_stable(x):
    """수치적으로 안정한 구현"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

# 큰 음수 값에서 테스트
x_large_negative = -1000

print("Numerical Stability Test for Sigmoid:")
print(f"x = {x_large_negative}")

# naive 버전은 경고 발생 가능
with np.errstate(over='ignore'):
    result_naive = sigmoid_naive(x_large_negative)
result_stable = sigmoid_stable(x_large_negative)

print(f"Naive implementation: {result_naive}")
print(f"Stable implementation: {result_stable}")
```

```
Numerical Stability Test for Sigmoid:
x = -1000
Naive implementation: 0.0
Stable implementation: 0.0
```

**소프트맥스의 수치 안정성:**

```python
def softmax_naive(x):
    """단순 구현 (수치적으로 불안정)"""
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def softmax_stable(x):
    """수치적으로 안정한 구현 (최댓값 빼기)"""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# 큰 값에서 테스트
x_large = np.array([1000, 1001, 1002])

print("Numerical Stability Test for Softmax:")
print(f"x = {x_large}")

with np.errstate(over='ignore', invalid='ignore'):
    result_naive = softmax_naive(x_large)
result_stable = softmax_stable(x_large)

print(f"Naive implementation: {result_naive}")
print(f"Stable implementation: {result_stable}")
```

```
Numerical Stability Test for Softmax:
x = [1000 1001 1002]
Naive implementation: [nan nan nan]
Stable implementation: [0.09003057 0.24472847 0.66524096]
```

**Log-Softmax:**

크로스 엔트로피 계산 시 $\log(\text{softmax}(x))$를 직접 계산하면 수치적으로 더 안정합니다.

```python
def log_softmax(x):
    """수치적으로 안정한 log-softmax"""
    if x.ndim == 1:
        max_x = np.max(x)
        return x - max_x - np.log(np.sum(np.exp(x - max_x)))
    else:
        max_x = np.max(x, axis=1, keepdims=True)
        return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True))

# 비교
x = np.array([1000, 1001, 1002])

# 방법 1: softmax 후 log (불안정)
with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
    result_two_step = np.log(softmax_naive(x))

# 방법 2: log-softmax 직접 계산 (안정)
result_direct = log_softmax(x)

print("Log-Softmax Comparison:")
print(f"Two-step (log(softmax)): {result_two_step}")
print(f"Direct (log_softmax):    {result_direct}")
```

```
Log-Softmax Comparison:
Two-step (log(softmax)): [nan nan nan]
Direct (log_softmax):    [-2.40760596 -1.40760596 -0.40760596]
```

### 2.1.7 Summary

| 활성화 함수 | 수식 | 범위 | 장점 | 단점 | 주 사용처 |
|-------------|------|------|------|------|-----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | 확률 해석 가능 | Vanishing gradient | 이진 분류 출력층 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1, 1)$ | Zero-centered | Vanishing gradient | RNN (과거) |
| ReLU | $\max(0, x)$ | $[0, \infty)$ | 빠른 계산, 희소성 | Dying ReLU | 은닉층 (기본) |
| Leaky ReLU | $\max(\alpha x, x)$ | $(-\infty, \infty)$ | Dying ReLU 해결 | 하이퍼파라미터 | 은닉층 |
| Softmax | $\frac{e^{z_i}}{\sum e^{z_j}}$ | $(0, 1)$, 합=1 | 확률 분포 출력 | 계산 비용 | 다중 분류 출력층 |

```python
# 모든 활성화 함수 비교 시각화
x = np.linspace(-4, 4, 200)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoid')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Tanh
axes[0, 1].plot(x, tanh(x), 'r-', linewidth=2)
axes[0, 1].set_title('Tanh')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# ReLU
axes[0, 2].plot(x, relu(x), 'g-', linewidth=2)
axes[0, 2].set_title('ReLU')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Leaky ReLU
axes[1, 0].plot(x, leaky_relu(x, 0.1), 'm-', linewidth=2)
axes[1, 0].set_title('Leaky ReLU (α=0.1)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# 미분 비교
axes[1, 1].plot(x, sigmoid_derivative(x), 'b-', linewidth=2, label='Sigmoid')
axes[1, 1].plot(x, tanh_derivative(x), 'r-', linewidth=2, label='Tanh')
axes[1, 1].plot(x, relu_derivative(x), 'g-', linewidth=2, label='ReLU')
axes[1, 1].set_title('Derivatives Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(-0.1, 1.5)

# Softmax 예시
z = np.linspace(-2, 2, 5)
axes[1, 2].bar(range(5), softmax(z))
axes[1, 2].set_title(f'Softmax of {z.round(1)}')
axes[1, 2].set_xlabel('Class')
axes[1, 2].set_ylabel('Probability')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

다음 섹션에서는 각 태스크에 적합한 손실 함수를 자세히 다룹니다.
