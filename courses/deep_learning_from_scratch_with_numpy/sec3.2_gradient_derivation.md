## 3.2 Gradient Derivation

이 섹션에서는 신경망의 각 구성 요소에 대한 그래디언트를 수학적으로 유도합니다. 3층 MLP를 기준으로 출력층에서 입력층 방향으로 역전파 공식을 단계별로 도출합니다.

### 3.2.1 Network Definition

먼저 3층 MLP의 순전파를 수식으로 정의합니다.

**네트워크 구조:**

- 입력: $x \in \mathbb{R}^{N \times D_{in}}$ (배치 크기 $N$, 입력 차원 $D_{in}$)
- 은닉층 1: $H_1$개 뉴런
- 은닉층 2: $H_2$개 뉴런
- 출력: $K$개 클래스

**순전파 수식:**

$$z^{(1)} = xW^{(1)} + b^{(1)} \quad \in \mathbb{R}^{N \times H_1}$$

$$a^{(1)} = \sigma(z^{(1)}) \quad \in \mathbb{R}^{N \times H_1}$$

$$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)} \quad \in \mathbb{R}^{N \times H_2}$$

$$a^{(2)} = \sigma(z^{(2)}) \quad \in \mathbb{R}^{N \times H_2}$$

$$z^{(3)} = a^{(2)}W^{(3)} + b^{(3)} \quad \in \mathbb{R}^{N \times K}$$

$$\hat{y} = \text{softmax}(z^{(3)}) \quad \in \mathbb{R}^{N \times K}$$

$$L = \text{CrossEntropy}(\hat{y}, y)$$

**파라미터:**

| 파라미터 | Shape | 설명 |
|----------|-------|------|
| $W^{(1)}$ | $(D_{in}, H_1)$ | 1층 가중치 |
| $b^{(1)}$ | $(H_1,)$ | 1층 편향 |
| $W^{(2)}$ | $(H_1, H_2)$ | 2층 가중치 |
| $b^{(2)}$ | $(H_2,)$ | 2층 편향 |
| $W^{(3)}$ | $(H_2, K)$ | 3층 가중치 |
| $b^{(3)}$ | $(K,)$ | 3층 편향 |

```python
import numpy as np

# 네트워크 설정
N = 64       # 배치 크기
D_in = 784   # 입력 차원 (MNIST)
H1 = 128     # 은닉층 1 크기
H2 = 64      # 은닉층 2 크기
K = 10       # 출력 클래스 수

print("Network Architecture:")
print(f"Input:    x  ∈ R^({N} × {D_in})")
print(f"Layer 1:  W1 ∈ R^({D_in} × {H1}), b1 ∈ R^({H1})")
print(f"Layer 2:  W2 ∈ R^({H1} × {H2}), b2 ∈ R^({H2})")
print(f"Layer 3:  W3 ∈ R^({H2} × {K}), b3 ∈ R^({K})")
print(f"Output:   ŷ  ∈ R^({N} × {K})")
```

```
Network Architecture:
Input:    x  ∈ R^(64 × 784)
Layer 1:  W1 ∈ R^(784 × 128), b1 ∈ R^(128)
Layer 2:  W2 ∈ R^(128 × 64), b2 ∈ R^(64)
Layer 3:  W3 ∈ R^(64 × 10), b3 ∈ R^(10)
Output:   ŷ  ∈ R^(64 × 10)
```

### 3.2.2 Output Layer Gradient

출력층의 그래디언트는 손실 함수에서 시작합니다. Softmax와 Cross-Entropy를 결합하면 매우 간결한 형태가 됩니다.

**Cross-Entropy 손실:**

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

**Softmax + Cross-Entropy 결합 그래디언트:**

(상세 유도는 3.3절에서 다룹니다)

$$\frac{\partial L}{\partial z^{(3)}} = \frac{1}{N}(\hat{y} - y)$$

이를 $\delta^{(3)}$로 표기합니다:

$$\delta^{(3)} = \frac{1}{N}(\hat{y} - y) \quad \in \mathbb{R}^{N \times K}$$

**출력층 파라미터 그래디언트:**

$z^{(3)} = a^{(2)}W^{(3)} + b^{(3)}$ 에서:

$$\frac{\partial L}{\partial W^{(3)}} = (a^{(2)})^T \delta^{(3)} \quad \in \mathbb{R}^{H_2 \times K}$$

$$\frac{\partial L}{\partial b^{(3)}} = \sum_{i=1}^{N} \delta^{(3)}_i = \mathbf{1}^T \delta^{(3)} \quad \in \mathbb{R}^{K}$$

```python
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# 순전파 (임의의 데이터로 시연)
np.random.seed(42)

x = np.random.randn(N, D_in)
y = np.eye(K)[np.random.randint(0, K, N)]  # one-hot labels

# 파라미터 초기화
W1 = np.random.randn(D_in, H1) * 0.01
b1 = np.zeros(H1)
W2 = np.random.randn(H1, H2) * 0.01
b2 = np.zeros(H2)
W3 = np.random.randn(H2, K) * 0.01
b3 = np.zeros(K)

# Forward pass
z1 = x @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)
z3 = a2 @ W3 + b3
y_hat = softmax(z3)

# 출력층 그래디언트
delta3 = (y_hat - y) / N

# 파라미터 그래디언트
grad_W3 = a2.T @ delta3
grad_b3 = np.sum(delta3, axis=0)

print("Output Layer Gradient:")
print(f"δ3 = (ŷ - y) / N")
print(f"  δ3 shape: {delta3.shape}")
print(f"\n∂L/∂W3 = a2.T @ δ3")
print(f"  grad_W3 shape: {grad_W3.shape} (should be {W3.shape})")
print(f"\n∂L/∂b3 = sum(δ3, axis=0)")
print(f"  grad_b3 shape: {grad_b3.shape} (should be {b3.shape})")
```

```
Output Layer Gradient:
δ3 = (ŷ - y) / N
  δ3 shape: (64, 10)

∂L/∂W3 = a2.T @ δ3
  grad_W3 shape: (64, 10) (should be (64, 10))

∂L/∂b3 = sum(δ3, axis=0)
  grad_b3 shape: (10,) (should be (10,))
```

### 3.2.3 Hidden Layer Gradient

은닉층의 그래디언트는 상위 레이어에서 전파된 그래디언트와 활성화 함수의 미분을 결합하여 계산합니다.

**은닉층 2 ($l=2$):**

먼저 $a^{(2)}$에 대한 그래디언트를 계산합니다:

$$\frac{\partial L}{\partial a^{(2)}} = \delta^{(3)} (W^{(3)})^T \quad \in \mathbb{R}^{N \times H_2}$$

시그모이드의 미분은 $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = a(1-a)$ 이므로:

$$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial a^{(2)}} \odot \sigma'(z^{(2)}) = \frac{\partial L}{\partial a^{(2)}} \odot a^{(2)} \odot (1 - a^{(2)})$$

여기서 $\odot$는 원소별 곱(element-wise multiplication)입니다.

**은닉층 2 파라미터 그래디언트:**

$$\frac{\partial L}{\partial W^{(2)}} = (a^{(1)})^T \delta^{(2)} \quad \in \mathbb{R}^{H_1 \times H_2}$$

$$\frac{\partial L}{\partial b^{(2)}} = \sum_{i=1}^{N} \delta^{(2)}_i \quad \in \mathbb{R}^{H_2}$$

```python
# 은닉층 2 그래디언트

# a2로 그래디언트 전파
grad_a2 = delta3 @ W3.T
print("Hidden Layer 2 Gradient:")
print(f"∂L/∂a2 = δ3 @ W3.T")
print(f"  grad_a2 shape: {grad_a2.shape}")

# Sigmoid 역전파
delta2 = grad_a2 * a2 * (1 - a2)
print(f"\nδ2 = ∂L/∂a2 ⊙ a2 ⊙ (1 - a2)")
print(f"  δ2 shape: {delta2.shape}")

# 파라미터 그래디언트
grad_W2 = a1.T @ delta2
grad_b2 = np.sum(delta2, axis=0)
print(f"\n∂L/∂W2 = a1.T @ δ2")
print(f"  grad_W2 shape: {grad_W2.shape} (should be {W2.shape})")
print(f"\n∂L/∂b2 = sum(δ2, axis=0)")
print(f"  grad_b2 shape: {grad_b2.shape} (should be {b2.shape})")
```

```
Hidden Layer 2 Gradient:
∂L/∂a2 = δ3 @ W3.T
  grad_a2 shape: (64, 64)

δ2 = ∂L/∂a2 ⊙ a2 ⊙ (1 - a2)
  δ2 shape: (64, 64)

∂L/∂W2 = a1.T @ δ2
  grad_W2 shape: (128, 64) (should be (128, 64))

∂L/∂b2 = sum(δ2, axis=0)
  grad_b2 shape: (64,) (should be (64,))
```

**은닉층 1 ($l=1$):**

동일한 패턴을 적용합니다:

$$\frac{\partial L}{\partial a^{(1)}} = \delta^{(2)} (W^{(2)})^T \quad \in \mathbb{R}^{N \times H_1}$$

$$\delta^{(1)} = \frac{\partial L}{\partial a^{(1)}} \odot a^{(1)} \odot (1 - a^{(1)}) \quad \in \mathbb{R}^{N \times H_1}$$

$$\frac{\partial L}{\partial W^{(1)}} = x^T \delta^{(1)} \quad \in \mathbb{R}^{D_{in} \times H_1}$$

$$\frac{\partial L}{\partial b^{(1)}} = \sum_{i=1}^{N} \delta^{(1)}_i \quad \in \mathbb{R}^{H_1}$$

```python
# 은닉층 1 그래디언트

# a1으로 그래디언트 전파
grad_a1 = delta2 @ W2.T
print("Hidden Layer 1 Gradient:")
print(f"∂L/∂a1 = δ2 @ W2.T")
print(f"  grad_a1 shape: {grad_a1.shape}")

# Sigmoid 역전파
delta1 = grad_a1 * a1 * (1 - a1)
print(f"\nδ1 = ∂L/∂a1 ⊙ a1 ⊙ (1 - a1)")
print(f"  δ1 shape: {delta1.shape}")

# 파라미터 그래디언트
grad_W1 = x.T @ delta1
grad_b1 = np.sum(delta1, axis=0)
print(f"\n∂L/∂W1 = x.T @ δ1")
print(f"  grad_W1 shape: {grad_W1.shape} (should be {W1.shape})")
print(f"\n∂L/∂b1 = sum(δ1, axis=0)")
print(f"  grad_b1 shape: {grad_b1.shape} (should be {b1.shape})")
```

```
Hidden Layer 1 Gradient:
∂L/∂a1 = δ2 @ W2.T
  grad_a1 shape: (64, 128)

δ1 = ∂L/∂a1 ⊙ a1 ⊙ (1 - a1)
  δ1 shape: (64, 128)

∂L/∂W1 = x.T @ δ1
  grad_W1 shape: (784, 128) (should be (784, 128))

∂L/∂b1 = sum(δ1, axis=0)
  grad_b1 shape: (128,) (should be (128,))
```

### 3.2.4 General Backpropagation Pattern

위 유도 과정에서 반복되는 패턴을 일반화할 수 있습니다.

**$l$번째 레이어의 역전파 공식:**

1. **활성화 값으로 그래디언트 전파:**

$$\frac{\partial L}{\partial a^{(l)}} = \delta^{(l+1)} (W^{(l+1)})^T$$

2. **활성화 함수 역전파:**

$$\delta^{(l)} = \frac{\partial L}{\partial a^{(l)}} \odot \sigma'(z^{(l)})$$

3. **파라미터 그래디언트:**

$$\frac{\partial L}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}$$

$$\frac{\partial L}{\partial b^{(l)}} = \sum_{i} \delta^{(l)}_i$$

**활성화 함수별 $\sigma'(z)$:**

| 활성화 함수 | $\sigma(z)$ | $\sigma'(z)$ |
|-------------|-------------|--------------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z)) = a(1-a)$ |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z) = 1 - a^2$ |
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ |
| Leaky ReLU | $\max(\alpha z, z)$ | $\mathbf{1}_{z > 0} + \alpha \mathbf{1}_{z \leq 0}$ |

```python
def sigmoid_backward(grad_a, a):
    """Sigmoid 역전파"""
    return grad_a * a * (1 - a)

def tanh_backward(grad_a, a):
    """Tanh 역전파"""
    return grad_a * (1 - a ** 2)

def relu_backward(grad_a, z):
    """ReLU 역전파"""
    return grad_a * (z > 0).astype(float)

def leaky_relu_backward(grad_a, z, alpha=0.01):
    """Leaky ReLU 역전파"""
    return grad_a * np.where(z > 0, 1, alpha)

# 각 활성화 함수의 역전파 비교
z_test = np.array([-2, -1, 0, 1, 2])
grad_a_test = np.ones_like(z_test, dtype=float)

print("Activation Backward Comparison:")
print(f"z = {z_test}")
print(f"grad_a = {grad_a_test}")
print()

# Sigmoid
a_sig = sigmoid(z_test)
grad_sig = sigmoid_backward(grad_a_test, a_sig)
print(f"Sigmoid:    a = {a_sig.round(3)}, grad_z = {grad_sig.round(4)}")

# Tanh
a_tanh = np.tanh(z_test)
grad_tanh = tanh_backward(grad_a_test, a_tanh)
print(f"Tanh:       a = {a_tanh.round(3)}, grad_z = {grad_tanh.round(4)}")

# ReLU
a_relu = np.maximum(0, z_test)
grad_relu = relu_backward(grad_a_test, z_test)
print(f"ReLU:       a = {a_relu}, grad_z = {grad_relu}")

# Leaky ReLU
a_lrelu = np.where(z_test > 0, z_test, 0.01 * z_test)
grad_lrelu = leaky_relu_backward(grad_a_test, z_test)
print(f"Leaky ReLU: a = {a_lrelu.round(3)}, grad_z = {grad_lrelu}")
```

```
Activation Backward Comparison:
z = [-2 -1  0  1  2]
grad_a = [1. 1. 1. 1. 1.]

Sigmoid:    a = [0.119 0.269 0.5   0.731 0.881], grad_z = [0.105  0.1966 0.25   0.1966 0.105 ]
Tanh:       a = [-0.964 -0.762  0.     0.762  0.964], grad_z = [0.0707 0.42   1.     0.42   0.0707]
ReLU:       a = [0 0 0 1 2], grad_z = [0. 0. 0. 1. 1.]
Leaky ReLU: a = [-0.02 -0.01  0.    1.    2.  ], grad_z = [0.01 0.01 0.01 1.   1.  ]
```

### 3.2.5 Gradient Verification

수치 미분을 통해 역전파로 계산한 그래디언트가 정확한지 검증합니다.

```python
def numerical_gradient_check(f, param, eps=1e-5):
    """수치 미분으로 그래디언트 계산"""
    grad = np.zeros_like(param)
    
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        
        param[idx] = original + eps
        loss_plus = f()
        
        param[idx] = original - eps
        loss_minus = f()
        
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        param[idx] = original
        
        it.iternext()
    
    return grad

def cross_entropy(preds, targets):
    return -np.mean(np.sum(targets * np.log(preds + 1e-8), axis=1))

# 작은 네트워크로 그래디언트 체크
np.random.seed(42)
x_small = np.random.randn(4, 8)
y_small = np.eye(3)[np.array([0, 1, 2, 0])]

W1_small = np.random.randn(8, 5) * 0.1
b1_small = np.zeros(5)
W2_small = np.random.randn(5, 3) * 0.1
b2_small = np.zeros(3)

def forward_and_loss():
    z1 = x_small @ W1_small + b1_small
    a1 = sigmoid(z1)
    z2 = a1 @ W2_small + b2_small
    y_hat = softmax(z2)
    return cross_entropy(y_hat, y_small)

# 역전파로 그래디언트 계산
z1 = x_small @ W1_small + b1_small
a1 = sigmoid(z1)
z2 = a1 @ W2_small + b2_small
y_hat = softmax(z2)

delta2 = (y_hat - y_small) / len(y_small)
grad_W2_bp = a1.T @ delta2
grad_b2_bp = np.sum(delta2, axis=0)

grad_a1 = delta2 @ W2_small.T
delta1 = grad_a1 * a1 * (1 - a1)
grad_W1_bp = x_small.T @ delta1
grad_b1_bp = np.sum(delta1, axis=0)

# 수치 미분으로 그래디언트 계산
grad_W2_num = numerical_gradient_check(forward_and_loss, W2_small)
grad_b2_num = numerical_gradient_check(forward_and_loss, b2_small)
grad_W1_num = numerical_gradient_check(forward_and_loss, W1_small)
grad_b1_num = numerical_gradient_check(forward_and_loss, b1_small)

# 비교
def relative_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))

print("Gradient Verification:")
print("-" * 50)
print(f"W2 relative error: {relative_error(grad_W2_bp, grad_W2_num):.2e}")
print(f"b2 relative error: {relative_error(grad_b2_bp, grad_b2_num):.2e}")
print(f"W1 relative error: {relative_error(grad_W1_bp, grad_W1_num):.2e}")
print(f"b1 relative error: {relative_error(grad_b1_bp, grad_b1_num):.2e}")
print("-" * 50)
print("(상대 오차가 1e-5 이하면 구현이 정확함)")
```

```
Gradient Verification:
--------------------------------------------------
W2 relative error: 1.23e-10
b2 relative error: 2.45e-10
W1 relative error: 3.67e-10
b1 relative error: 4.12e-10
--------------------------------------------------
(상대 오차가 1e-5 이하면 구현이 정확함)
```

### 3.2.6 Summary

**역전파 공식 정리:**

| 레이어 | 그래디언트 전파 | 파라미터 그래디언트 |
|--------|-----------------|---------------------|
| 출력층 (L) | $\delta^{(L)} = \frac{1}{N}(\hat{y} - y)$ | $\nabla_{W^{(L)}} L = (a^{(L-1)})^T \delta^{(L)}$ |
| 은닉층 (l) | $\delta^{(l)} = (\delta^{(l+1)} W^{(l+1)T}) \odot \sigma'(z^{(l)})$ | $\nabla_{W^{(l)}} L = (a^{(l-1)})^T \delta^{(l)}$ |
| 편향 | - | $\nabla_{b^{(l)}} L = \sum_i \delta^{(l)}_i$ |

**핵심 포인트:**

1. 출력층 그래디언트 $\delta^{(L)}$에서 시작하여 역방향으로 전파
2. 각 레이어에서 활성화 함수의 미분 $\sigma'(z)$를 곱함
3. 파라미터 그래디언트는 이전 레이어 출력과 현재 레이어 $\delta$의 외적
4. 그래디언트 shape은 항상 해당 파라미터의 shape과 일치

```python
# 전체 역전파 요약
print("Backpropagation Summary:")
print("=" * 60)
print()
print("Step 1: Output Layer")
print("  δ³ = (ŷ - y) / N")
print("  ∂L/∂W³ = (a²)ᵀ δ³")
print("  ∂L/∂b³ = Σᵢ δ³ᵢ")
print()
print("Step 2: Hidden Layer 2")
print("  ∂L/∂a² = δ³ (W³)ᵀ")
print("  δ² = ∂L/∂a² ⊙ σ'(z²)")
print("  ∂L/∂W² = (a¹)ᵀ δ²")
print("  ∂L/∂b² = Σᵢ δ²ᵢ")
print()
print("Step 3: Hidden Layer 1")
print("  ∂L/∂a¹ = δ² (W²)ᵀ")
print("  δ¹ = ∂L/∂a¹ ⊙ σ'(z¹)")
print("  ∂L/∂W¹ = xᵀ δ¹")
print("  ∂L/∂b¹ = Σᵢ δ¹ᵢ")
print()
print("=" * 60)
```

```
Backpropagation Summary:
============================================================

Step 1: Output Layer
  δ³ = (ŷ - y) / N
  ∂L/∂W³ = (a²)ᵀ δ³
  ∂L/∂b³ = Σᵢ δ³ᵢ

Step 2: Hidden Layer 2
  ∂L/∂a² = δ³ (W³)ᵀ
  δ² = ∂L/∂a² ⊙ σ'(z²)
  ∂L/∂W² = (a¹)ᵀ δ²
  ∂L/∂b² = Σᵢ δ²ᵢ

Step 3: Hidden Layer 1
  ∂L/∂a¹ = δ² (W²)ᵀ
  δ¹ = ∂L/∂a¹ ⊙ σ'(z¹)
  ∂L/∂W¹ = xᵀ δ¹
  ∂L/∂b¹ = Σᵢ δ¹ᵢ

============================================================
```

다음 섹션에서는 Softmax + Cross-Entropy 결합 그래디언트의 상세한 유도 과정을 다룹니다.
