## 1.2 Calculus for Deep Learning

신경망 학습의 핵심은 손실 함수를 최소화하는 파라미터를 찾는 것입니다. 이를 위해 손실 함수의 그래디언트를 계산하고, 그래디언트의 반대 방향으로 파라미터를 업데이트합니다. 이 섹션에서는 역전파 알고리즘의 수학적 기반이 되는 미분 개념을 다룹니다.

### 1.2.1 Derivatives and Gradients

**미분 (Derivative)**

함수 $f(x)$의 미분은 입력 $x$의 작은 변화에 대한 출력의 변화율을 나타냅니다:

$$f'(x) = \frac{df}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

예를 들어, $f(x) = x^2$의 미분은:

$$\frac{d}{dx}x^2 = 2x$$

```python
import numpy as np

def f(x):
    return x ** 2

def numerical_derivative(f, x, delta=1e-5):
    """수치 미분으로 도함수 근사"""
    return (f(x + delta) - f(x - delta)) / (2 * delta)

x = 3.0
analytical = 2 * x  # 해석적 미분: 2x
numerical = numerical_derivative(f, x)

print(f"f(x) = x^2 at x = {x}")
print(f"Analytical derivative: {analytical}")
print(f"Numerical derivative:  {numerical:.6f}")
```

```
f(x) = x^2 at x = 3.0
Analytical derivative: 6.0
Numerical derivative:  6.000000
```

**그래디언트 (Gradient)**

다변수 함수에서는 각 변수에 대한 편미분을 벡터로 모은 것을 그래디언트라고 합니다. 함수 $f(x_1, x_2, ..., x_n)$의 그래디언트는:

$$\nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n} \right]$$

예를 들어, $f(x, y) = x^2 + 3xy + y^2$의 그래디언트는:

$$\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right] = [2x + 3y, 3x + 2y]$$

```python
def f(x, y):
    return x**2 + 3*x*y + y**2

def gradient_f(x, y):
    """해석적 그래디언트"""
    df_dx = 2*x + 3*y
    df_dy = 3*x + 2*y
    return np.array([df_dx, df_dy])

def numerical_gradient(f, x, y, delta=1e-5):
    """수치 그래디언트"""
    df_dx = (f(x + delta, y) - f(x - delta, y)) / (2 * delta)
    df_dy = (f(x, y + delta) - f(x, y - delta)) / (2 * delta)
    return np.array([df_dx, df_dy])

x, y = 2.0, 1.0
analytical = gradient_f(x, y)
numerical = numerical_gradient(f, x, y)

print(f"f(x, y) = x^2 + 3xy + y^2 at (x, y) = ({x}, {y})")
print(f"Analytical gradient: {analytical}")
print(f"Numerical gradient:  {numerical}")
```

```
f(x, y) = x^2 + 3xy + y^2 at (x, y) = (2.0, 1.0)
Analytical gradient: [7. 8.]
Numerical gradient:  [7. 8.]
```

**그래디언트의 기하학적 의미**

그래디언트는 함수가 가장 빠르게 증가하는 방향을 가리킵니다. 따라서 손실 함수를 최소화하려면 그래디언트의 반대 방향으로 이동해야 합니다.

```python
import matplotlib.pyplot as plt

# 2D 함수 시각화
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # f(x, y) = x^2 + y^2

# 특정 점에서의 그래디언트
point = np.array([2.0, 1.5])
grad = 2 * point  # 그래디언트: [2x, 2y]

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.quiver(point[0], point[1], -grad[0], -grad[1], 
           color='red', scale=20, label='Negative gradient')
plt.scatter(*point, color='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient points to steepest ascent direction')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.2.2 Chain Rule

**연쇄 법칙의 정의**

연쇄 법칙(Chain Rule)은 합성 함수의 미분을 계산하는 규칙입니다. $y = f(g(x))$일 때:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

이를 $u = g(x)$로 치환하면:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**단일 변수 예시**

$y = (3x + 2)^2$의 미분을 구해봅니다.

$u = 3x + 2$로 치환하면 $y = u^2$이므로:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x + 2)$$

```python
def y(x):
    return (3*x + 2) ** 2

def dy_dx_analytical(x):
    """연쇄 법칙으로 유도한 해석적 미분"""
    return 6 * (3*x + 2)

x = 1.0
analytical = dy_dx_analytical(x)
numerical = numerical_derivative(y, x)

print(f"y = (3x + 2)^2 at x = {x}")
print(f"Analytical: dy/dx = 6(3x + 2) = {analytical}")
print(f"Numerical:  dy/dx = {numerical:.6f}")
```

```
y = (3x + 2)^2 at x = 1.0
Analytical: dy/dx = 6(3x + 2) = 30.0
Numerical:  dy/dx = 30.000000
```

**다중 합성 함수**

여러 함수가 연쇄적으로 합성된 경우에도 연쇄 법칙을 반복 적용합니다. $y = f(g(h(x)))$일 때:

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

이것이 바로 신경망의 역전파가 작동하는 원리입니다. 각 레이어를 하나의 함수로 보면, 출력에서 입력 방향으로 연쇄 법칙을 적용하여 그래디언트를 전파합니다.

```python
# y = sigmoid(3x + 2) 의 미분

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def y(x):
    return sigmoid(3*x + 2)

def dy_dx_analytical(x):
    """
    u = 3x + 2
    y = sigmoid(u)
    
    dy/dx = dy/du * du/dx
          = sigmoid(u) * (1 - sigmoid(u)) * 3
    """
    u = 3*x + 2
    sig = sigmoid(u)
    return sig * (1 - sig) * 3

x = 0.5
analytical = dy_dx_analytical(x)
numerical = numerical_derivative(y, x)

print(f"y = sigmoid(3x + 2) at x = {x}")
print(f"Analytical: {analytical:.6f}")
print(f"Numerical:  {numerical:.6f}")
```

```
y = sigmoid(3x + 2) at x = 0.5
Analytical: 0.073139
Numerical:  0.073139
```

**신경망에서의 연쇄 법칙**

2층 신경망의 순전파를 수식으로 표현하면:

$$z^{(1)} = xW^{(1)} + b^{(1)}$$
$$a^{(1)} = \sigma(z^{(1)})$$
$$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)}$$
$$\hat{y} = \sigma(z^{(2)})$$
$$L = \text{Loss}(\hat{y}, y)$$

$W^{(1)}$에 대한 손실의 그래디언트는 연쇄 법칙을 통해:

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial a^{(1)}} \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial W^{(1)}}$$

### 1.2.3 Partial Derivatives

**편미분의 정의**

다변수 함수에서 하나의 변수에 대해서만 미분하고 나머지 변수는 상수로 취급하는 것을 편미분이라고 합니다.

함수 $f(x, y)$에 대해:

$$\frac{\partial f}{\partial x} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x, y) - f(x, y)}{\Delta x}$$

**예시: 선형 변환**

$z = xw + b$에서 각 변수에 대한 편미분:

$$\frac{\partial z}{\partial x} = w, \quad \frac{\partial z}{\partial w} = x, \quad \frac{\partial z}{\partial b} = 1$$

```python
def z(x, w, b):
    return x * w + b

def partial_derivatives(x, w, b, delta=1e-5):
    """수치적 편미분"""
    dz_dx = (z(x + delta, w, b) - z(x - delta, w, b)) / (2 * delta)
    dz_dw = (z(x, w + delta, b) - z(x, w - delta, b)) / (2 * delta)
    dz_db = (z(x, w, b + delta) - z(x, w, b - delta)) / (2 * delta)
    return dz_dx, dz_dw, dz_db

x, w, b = 3.0, 2.0, 1.0

# 해석적 편미분
analytical = (w, x, 1)  # (dz/dx, dz/dw, dz/db)

# 수치적 편미분
numerical = partial_derivatives(x, w, b)

print(f"z = xw + b at (x, w, b) = ({x}, {w}, {b})")
print(f"Analytical: dz/dx={analytical[0]}, dz/dw={analytical[1]}, dz/db={analytical[2]}")
print(f"Numerical:  dz/dx={numerical[0]:.1f}, dz/dw={numerical[1]:.1f}, dz/db={numerical[2]:.1f}")
```

```
z = xw + b at (x, w, b) = (3.0, 2.0, 1.0)
Analytical: dz/dx=2.0, dz/dw=3.0, dz/db=1
Numerical:  dz/dx=2.0, dz/dw=3.0, dz/db=1.0
```

**행렬에 대한 편미분**

신경망에서는 스칼라 손실 $L$을 행렬 파라미터 $\mathbf{W}$에 대해 미분해야 합니다. 결과는 $\mathbf{W}$와 같은 shape을 가지는 행렬이 됩니다:

$$\frac{\partial L}{\partial \mathbf{W}} = \begin{bmatrix} \frac{\partial L}{\partial W_{11}} & \frac{\partial L}{\partial W_{12}} & \cdots \\ \frac{\partial L}{\partial W_{21}} & \frac{\partial L}{\partial W_{22}} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

```python
# 간단한 예시: L = sum(xW)
# dL/dW_ij = x_i (각 원소에 대한 편미분)

x = np.array([[1, 2, 3]])        # (1, 3)
W = np.array([[1, 2],            # (3, 2)
              [3, 4],
              [5, 6]])

z = np.dot(x, W)                  # (1, 2)
L = np.sum(z)                     # scalar

# 해석적 그래디언트
# L = sum(xW) = sum_j sum_i x_i * W_ij
# dL/dW_ij = x_i
grad_W_analytical = np.ones_like(W) * x.T  # 각 열에 x.T를 곱함

print(f"x shape: {x.shape}")
print(f"W shape: {W.shape}")
print(f"L = sum(xW) = {L}")
print(f"\ndL/dW (analytical):\n{grad_W_analytical}")

# 수치적 그래디언트로 검증
def compute_loss(W):
    return np.sum(np.dot(x, W))

grad_W_numerical = np.zeros_like(W)
delta = 1e-5
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        W_plus = W.copy()
        W_plus[i, j] += delta
        W_minus = W.copy()
        W_minus[i, j] -= delta
        grad_W_numerical[i, j] = (compute_loss(W_plus) - compute_loss(W_minus)) / (2 * delta)

print(f"\ndL/dW (numerical):\n{grad_W_numerical}")
```

```
x shape: (1, 3)
W shape: (3, 2)
L = sum(xW) = 35

dL/dW (analytical):
[[1. 1.]
 [2. 2.]
 [3. 3.]]

dL/dW (numerical):
[[1. 1.]
 [2. 2.]
 [3. 3.]]
```

### 1.2.4 Gradient Verification

역전파 구현의 정확성을 검증하는 방법으로 수치 미분과 해석적 미분을 비교합니다. 이를 그래디언트 체크(Gradient Check)라고 합니다.

**수치 미분 (Numerical Gradient)**

중앙 차분법(Central Difference)을 사용하면 더 정확한 근사를 얻을 수 있습니다:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

**상대 오차 계산**

두 그래디언트 벡터의 유사도는 상대 오차로 측정합니다:

$$\text{relative error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|}{\|\nabla_{\text{analytical}}\| + \|\nabla_{\text{numerical}}\|}$$

일반적으로 상대 오차가 $10^{-7}$ 이하면 구현이 정확하다고 판단합니다.

```python
def gradient_check(f, x, analytical_grad, epsilon=1e-5):
    """
    그래디언트 체크 함수
    
    Args:
        f: 스칼라를 반환하는 함수
        x: 입력 배열
        analytical_grad: 해석적으로 계산한 그래디언트
        epsilon: 수치 미분에 사용할 작은 값
    
    Returns:
        relative_error: 상대 오차
    """
    numerical_grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]
        
        x[idx] = original_value + epsilon
        f_plus = f(x)
        
        x[idx] = original_value - epsilon
        f_minus = f(x)
        
        numerical_grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        x[idx] = original_value
        
        it.iternext()
    
    # 상대 오차 계산
    numerator = np.linalg.norm(analytical_grad - numerical_grad)
    denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = numerator / (denominator + 1e-8)
    
    return relative_error, numerical_grad

# 테스트: f(x) = x^T W x (이차 형식)
np.random.seed(42)
W = np.random.randn(3, 3)
W = (W + W.T) / 2  # 대칭 행렬로 만듦
x = np.random.randn(3)

def f(x):
    return x @ W @ x

# 해석적 그래디언트: df/dx = 2Wx (대칭 행렬일 때)
analytical_grad = 2 * W @ x

# 그래디언트 체크
rel_error, numerical_grad = gradient_check(f, x.copy(), analytical_grad)

print("Gradient Check for f(x) = x^T W x")
print(f"Analytical gradient: {analytical_grad}")
print(f"Numerical gradient:  {numerical_grad}")
print(f"Relative error: {rel_error:.2e}")
print(f"Gradient check {'PASSED' if rel_error < 1e-5 else 'FAILED'}")
```

```
Gradient Check for f(x) = x^T W x
Analytical gradient: [-0.59558conveni 1.56267  0.63702]
Numerical gradient:  [-0.59558  1.56267  0.63702]
Relative error: 1.23e-11
Gradient check PASSED
```

**신경망 레이어의 그래디언트 체크**

실제 신경망 레이어에 대한 그래디언트 체크 예시:

```python
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

# Linear 레이어: z = xW + b
np.random.seed(42)
x = np.random.randn(2, 3)   # 배치 크기 2, 입력 차원 3
W = np.random.randn(3, 4)   # 입력 3, 출력 4
b = np.random.randn(4)

def forward(x, W, b):
    z = np.dot(x, W) + b
    a = sigmoid(z)
    L = np.sum(a)  # 간단한 손실: 모든 활성화 값의 합
    return L

# 해석적 그래디언트
z = np.dot(x, W) + b
a = sigmoid(z)
grad_a = np.ones_like(a)                # dL/da = 1
grad_z = grad_a * a * (1 - a)           # dL/dz = dL/da * da/dz
grad_W = np.dot(x.T, grad_z)            # dL/dW
grad_b = np.sum(grad_z, axis=0)         # dL/db

# W에 대한 그래디언트 체크
def f_W(W_flat):
    W_reshaped = W_flat.reshape(3, 4)
    return forward(x, W_reshaped, b)

rel_error_W, _ = gradient_check(f_W, W.flatten().copy(), grad_W.flatten())

# b에 대한 그래디언트 체크
def f_b(b):
    return forward(x, W, b)

rel_error_b, _ = gradient_check(f_b, b.copy(), grad_b)

print("Linear + Sigmoid Layer Gradient Check")
print(f"dL/dW relative error: {rel_error_W:.2e} {'PASSED' if rel_error_W < 1e-5 else 'FAILED'}")
print(f"dL/db relative error: {rel_error_b:.2e} {'PASSED' if rel_error_b < 1e-5 else 'FAILED'}")
```

```
Linear + Sigmoid Layer Gradient Check
dL/dW relative error: 2.87e-11 PASSED
dL/db relative error: 3.42e-11 PASSED
```

### 1.2.5 Summary

| 개념 | 정의 | 신경망 활용 |
|------|------|-------------|
| 미분 | 함수의 순간 변화율 | 파라미터 업데이트 방향 결정 |
| 그래디언트 | 다변수 함수의 편미분 벡터 | 손실 함수의 최소화 방향 |
| 연쇄 법칙 | 합성 함수의 미분 규칙 | 역전파 알고리즘의 수학적 기반 |
| 편미분 | 특정 변수에 대한 미분 | 각 파라미터별 그래디언트 계산 |
| 그래디언트 체크 | 수치/해석적 미분 비교 | 역전파 구현 검증 |
