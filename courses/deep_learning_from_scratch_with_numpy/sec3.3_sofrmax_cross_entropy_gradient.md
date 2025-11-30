## 3.3 Softmax + Cross-Entropy Gradient

이 섹션에서는 다중 클래스 분류에서 사용되는 Softmax와 Cross-Entropy의 결합 그래디언트를 상세히 유도합니다. 이 결합 그래디언트가 $\frac{1}{N}(\hat{y} - y)$라는 간결한 형태가 되는 이유를 수학적으로 증명합니다.

### 3.3.1 Softmax Function

**정의:**

Softmax 함수는 로짓(logit) 벡터 $z \in \mathbb{R}^K$를 확률 분포로 변환합니다:

$$\hat{y}_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**특성:**

- 모든 출력이 $(0, 1)$ 범위
- 출력의 합이 1: $\sum_{k=1}^{K} \hat{y}_k = 1$
- 단조 증가: $z_i > z_j \Rightarrow \hat{y}_i > \hat{y}_j$

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """수치적으로 안정한 Softmax"""
    if z.ndim == 1:
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z)
    else:
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

# Softmax 예시
z = np.array([2.0, 1.0, 0.1])
y_hat = softmax(z)

print("Softmax Example:")
print(f"Logits z = {z}")
print(f"Softmax(z) = {y_hat.round(4)}")
print(f"Sum = {y_hat.sum():.4f}")
print(f"Predicted class = {np.argmax(y_hat)}")
```

```
Softmax Example:
Logits z = [2.  1.  0.1]
Softmax(z) = [0.6590 0.2424 0.0986]
Sum = 1.0000
Predicted class = 0
```

### 3.3.2 Softmax Derivative

Softmax의 편미분 $\frac{\partial \hat{y}_k}{\partial z_i}$는 $i = k$인 경우와 $i \neq k$인 경우를 나누어 계산해야 합니다.

**Case 1: $i = k$ (같은 인덱스)**

$$\frac{\partial \hat{y}_k}{\partial z_k} = \frac{\partial}{\partial z_k} \left( \frac{e^{z_k}}{\sum_j e^{z_j}} \right)$$

몫의 미분법 $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$를 적용합니다:

- $f = e^{z_k}$, $f' = e^{z_k}$
- $g = \sum_j e^{z_j}$, $g' = e^{z_k}$

$$\frac{\partial \hat{y}_k}{\partial z_k} = \frac{e^{z_k} \cdot \sum_j e^{z_j} - e^{z_k} \cdot e^{z_k}}{(\sum_j e^{z_j})^2}$$

$$= \frac{e^{z_k}}{\sum_j e^{z_j}} - \frac{e^{z_k}}{\sum_j e^{z_j}} \cdot \frac{e^{z_k}}{\sum_j e^{z_j}}$$

$$= \hat{y}_k - \hat{y}_k^2 = \hat{y}_k(1 - \hat{y}_k)$$

**Case 2: $i \neq k$ (다른 인덱스)**

$$\frac{\partial \hat{y}_k}{\partial z_i} = \frac{\partial}{\partial z_i} \left( \frac{e^{z_k}}{\sum_j e^{z_j}} \right)$$

분자 $e^{z_k}$는 $z_i$ ($i \neq k$)에 대해 상수이므로:

- $f = e^{z_k}$, $f' = 0$
- $g = \sum_j e^{z_j}$, $g' = e^{z_i}$

$$\frac{\partial \hat{y}_k}{\partial z_i} = \frac{0 \cdot \sum_j e^{z_j} - e^{z_k} \cdot e^{z_i}}{(\sum_j e^{z_j})^2}$$

$$= -\frac{e^{z_k}}{\sum_j e^{z_j}} \cdot \frac{e^{z_i}}{\sum_j e^{z_j}} = -\hat{y}_k \hat{y}_i$$

**통합 표현:**

크로네커 델타 $\delta_{ki}$를 사용하면 ($i=k$일 때 1, 아니면 0):

$$\frac{\partial \hat{y}_k}{\partial z_i} = \hat{y}_k(\delta_{ki} - \hat{y}_i)$$

```python
def softmax_jacobian(z):
    """Softmax의 야코비안 행렬 계산"""
    y = softmax(z)
    K = len(z)
    jacobian = np.zeros((K, K))
    
    for k in range(K):
        for i in range(K):
            if k == i:
                jacobian[k, i] = y[k] * (1 - y[k])
            else:
                jacobian[k, i] = -y[k] * y[i]
    
    return jacobian

# 야코비안 계산
z = np.array([2.0, 1.0, 0.1])
y_hat = softmax(z)
J = softmax_jacobian(z)

print("Softmax Jacobian Matrix:")
print(f"z = {z}")
print(f"ŷ = {y_hat.round(4)}")
print(f"\n∂ŷ/∂z (Jacobian):")
print(J.round(4))
print(f"\nVerification: J[0,0] = ŷ₀(1-ŷ₀) = {y_hat[0]*(1-y_hat[0]):.4f}")
print(f"Verification: J[0,1] = -ŷ₀ŷ₁ = {-y_hat[0]*y_hat[1]:.4f}")
```

```
Softmax Jacobian Matrix:
z = [2.  1.  0.1]
ŷ = [0.6590 0.2424 0.0986]

∂ŷ/∂z (Jacobian):
[[ 0.2247 -0.1598 -0.0650]
 [-0.1598  0.1836 -0.0239]
 [-0.0650 -0.0239  0.0889]]

Verification: J[0,0] = ŷ₀(1-ŷ₀) = 0.2247
Verification: J[0,1] = -ŷ₀ŷ₁ = -0.1598
```

### 3.3.3 Cross-Entropy Derivative

**Cross-Entropy 손실 함수:**

원-핫 인코딩된 정답 레이블 $y$에 대해:

$$L = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)$$

원-핫 벡터의 특성상 정답 클래스 $c$에서만 $y_c = 1$이므로:

$$L = -\log(\hat{y}_c)$$

**Softmax 출력에 대한 미분:**

$$\frac{\partial L}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}$$

원-핫 벡터이므로 정답 클래스 $c$에서만:

$$\frac{\partial L}{\partial \hat{y}_c} = -\frac{1}{\hat{y}_c}$$

```python
def cross_entropy(y_hat, y):
    """Cross-Entropy 손실"""
    eps = 1e-8
    return -np.sum(y * np.log(y_hat + eps))

def cross_entropy_gradient_wrt_yhat(y_hat, y):
    """Cross-Entropy의 ŷ에 대한 그래디언트"""
    eps = 1e-8
    return -y / (y_hat + eps)

# Cross-Entropy 그래디언트 예시
y_hat = softmax(np.array([2.0, 1.0, 0.1]))
y = np.array([1, 0, 0])  # 정답: 클래스 0

loss = cross_entropy(y_hat, y)
grad_yhat = cross_entropy_gradient_wrt_yhat(y_hat, y)

print("Cross-Entropy Gradient w.r.t. ŷ:")
print(f"ŷ = {y_hat.round(4)}")
print(f"y = {y}")
print(f"L = -log(ŷ₀) = -log({y_hat[0]:.4f}) = {loss:.4f}")
print(f"\n∂L/∂ŷ = -y/ŷ = {grad_yhat.round(4)}")
```

```
Cross-Entropy Gradient w.r.t. ŷ:
ŷ = [0.6590 0.2424 0.0986]
y = [1 0 0]
L = -log(ŷ₀) = -log(0.6590) = 0.4170

∂L/∂ŷ = -y/ŷ = [-1.5175 -0.     -0.    ]
```

### 3.3.4 Combined Gradient Derivation

이제 Softmax + Cross-Entropy의 결합 그래디언트를 유도합니다.

**연쇄 법칙 적용:**

$$\frac{\partial L}{\partial z_i} = \sum_{k=1}^{K} \frac{\partial L}{\partial \hat{y}_k} \cdot \frac{\partial \hat{y}_k}{\partial z_i}$$

각 항을 대입합니다:

$$\frac{\partial L}{\partial z_i} = \sum_{k=1}^{K} \left( -\frac{y_k}{\hat{y}_k} \right) \cdot \hat{y}_k(\delta_{ki} - \hat{y}_i)$$

$$= \sum_{k=1}^{K} -y_k(\delta_{ki} - \hat{y}_i)$$

$$= -\sum_{k=1}^{K} y_k \delta_{ki} + \sum_{k=1}^{K} y_k \hat{y}_i$$

**첫 번째 항:**

$$-\sum_{k=1}^{K} y_k \delta_{ki} = -y_i$$

크로네커 델타의 성질에 의해 $k = i$인 항만 남습니다.

**두 번째 항:**

$$\sum_{k=1}^{K} y_k \hat{y}_i = \hat{y}_i \sum_{k=1}^{K} y_k = \hat{y}_i \cdot 1 = \hat{y}_i$$

원-핫 벡터의 합은 1입니다.

**최종 결과:**

$$\frac{\partial L}{\partial z_i} = -y_i + \hat{y}_i = \hat{y}_i - y_i$$

벡터 형태로:

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

배치 처리 시 평균을 취하면:

$$\frac{\partial L}{\partial z} = \frac{1}{N}(\hat{y} - y)$$

```python
def softmax_cross_entropy_gradient(z, y):
    """Softmax + Cross-Entropy 결합 그래디언트 (직접 계산)"""
    y_hat = softmax(z)
    return y_hat - y

# 두 가지 방법으로 그래디언트 계산 비교
z = np.array([2.0, 1.0, 0.1])
y = np.array([1, 0, 0])  # one-hot
y_hat = softmax(z)

# 방법 1: 연쇄 법칙으로 단계별 계산
grad_yhat = cross_entropy_gradient_wrt_yhat(y_hat, y)  # ∂L/∂ŷ
J = softmax_jacobian(z)  # ∂ŷ/∂z
grad_z_chain = J.T @ grad_yhat  # 연쇄 법칙

# 방법 2: 결합 공식 직접 적용
grad_z_direct = softmax_cross_entropy_gradient(z, y)

print("Gradient Comparison:")
print(f"z = {z}")
print(f"y = {y}")
print(f"ŷ = {y_hat.round(4)}")
print(f"\nMethod 1 (Chain Rule): ∂L/∂z = Jᵀ @ (∂L/∂ŷ)")
print(f"  = {grad_z_chain.round(6)}")
print(f"\nMethod 2 (Direct):     ∂L/∂z = ŷ - y")
print(f"  = {grad_z_direct.round(6)}")
print(f"\nDifference: {np.max(np.abs(grad_z_chain - grad_z_direct)):.2e}")
```

```
Gradient Comparison:
z = [2.  1.  0.1]
y = [1 0 0]
ŷ = [0.6590 0.2424 0.0986]

Method 1 (Chain Rule): ∂L/∂z = Jᵀ @ (∂L/∂ŷ)
  = [-0.340984  0.242429  0.098555]

Method 2 (Direct):     ∂L/∂z = ŷ - y
  = [-0.340984  0.242429  0.098555]

Difference: 5.55e-17
```

### 3.3.5 Intuitive Understanding

결합 그래디언트 $\hat{y} - y$의 직관적 의미를 살펴봅니다.

**그래디언트의 의미:**

- $\hat{y}_k - y_k > 0$: 클래스 $k$의 확률이 너무 높음 → $z_k$를 감소시켜야 함
- $\hat{y}_k - y_k < 0$: 클래스 $k$의 확률이 너무 낮음 → $z_k$를 증가시켜야 함
- $\hat{y}_k - y_k = 0$: 완벽한 예측 → 업데이트 불필요

**예시 분석:**

```python
def analyze_gradient(z, true_class):
    """그래디언트 분석"""
    y = np.zeros(len(z))
    y[true_class] = 1
    y_hat = softmax(z)
    grad = y_hat - y
    
    print(f"True class: {true_class}")
    print(f"Logits z:       {z}")
    print(f"Softmax ŷ:      {y_hat.round(4)}")
    print(f"One-hot y:      {y.astype(int)}")
    print(f"Gradient ŷ - y: {grad.round(4)}")
    print()
    
    for k in range(len(z)):
        if grad[k] > 0.01:
            print(f"  Class {k}: grad={grad[k]:+.4f} → decrease z[{k}] (predicted too high)")
        elif grad[k] < -0.01:
            print(f"  Class {k}: grad={grad[k]:+.4f} → increase z[{k}] (predicted too low)")
        else:
            print(f"  Class {k}: grad={grad[k]:+.4f} → nearly correct")
    
    return grad

print("=" * 60)
print("Case 1: Correct prediction with high confidence")
print("=" * 60)
analyze_gradient(np.array([3.0, 1.0, 0.5]), true_class=0)

print("\n" + "=" * 60)
print("Case 2: Wrong prediction")
print("=" * 60)
analyze_gradient(np.array([1.0, 3.0, 0.5]), true_class=0)

print("\n" + "=" * 60)
print("Case 3: Uncertain prediction")
print("=" * 60)
analyze_gradient(np.array([1.0, 1.0, 1.0]), true_class=0)
```

```
============================================================
Case 1: Correct prediction with high confidence
============================================================
True class: 0
Logits z:       [3.  1.  0.5]
Softmax ŷ:      [0.8360 0.1131 0.0687]
One-hot y:      [1 0 0]
Gradient ŷ - y: [-0.1640  0.1131  0.0687]

  Class 0: grad=-0.1640 → increase z[0] (predicted too low)
  Class 1: grad=+0.1131 → decrease z[1] (predicted too high)
  Class 2: grad=+0.0687 → decrease z[2] (predicted too high)

============================================================
Case 2: Wrong prediction
============================================================
True class: 0
Logits z:       [1.  3.  0.5]
Softmax ŷ:      [0.1131 0.8360 0.0687]
One-hot y:      [1 0 0]
Gradient ŷ - y: [-0.8869  0.8360  0.0687]

  Class 0: grad=-0.8869 → increase z[0] (predicted too low)
  Class 1: grad=+0.8360 → decrease z[1] (predicted too high)
  Class 2: grad=+0.0687 → decrease z[2] (predicted too high)

============================================================
Case 3: Uncertain prediction
============================================================
True class: 0
Logits z:       [1. 1. 1.]
Softmax ŷ:      [0.3333 0.3333 0.3333]
One-hot y:      [1 0 0]
Gradient ŷ - y: [-0.6667  0.3333  0.3333]

  Class 0: grad=-0.6667 → increase z[0] (predicted too low)
  Class 1: grad=+0.3333 → decrease z[1] (predicted too high)
  Class 2: grad=+0.3333 → decrease z[2] (predicted too high)
```

**그래디언트 크기 비교:**

```python
# 예측 신뢰도에 따른 그래디언트 크기
confidences = np.linspace(0.1, 0.99, 50)
grad_magnitudes = []

for conf in confidences:
    # 정답 클래스의 확률이 conf일 때
    y_hat = np.array([conf, (1-conf)/2, (1-conf)/2])
    y = np.array([1, 0, 0])
    grad = y_hat - y
    grad_magnitudes.append(np.linalg.norm(grad))

plt.figure(figsize=(10, 5))
plt.plot(confidences, grad_magnitudes, 'b-', linewidth=2)
plt.xlabel('Prediction Confidence for True Class')
plt.ylabel('Gradient Magnitude ||ŷ - y||')
plt.title('Gradient Magnitude vs Prediction Confidence')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.show()

print("Gradient magnitude at different confidence levels:")
for conf in [0.1, 0.5, 0.9, 0.99]:
    y_hat = np.array([conf, (1-conf)/2, (1-conf)/2])
    y = np.array([1, 0, 0])
    grad = y_hat - y
    print(f"  Confidence {conf:.2f}: ||grad|| = {np.linalg.norm(grad):.4f}")
```

```
Gradient magnitude at different confidence levels:
  Confidence 0.10: ||grad|| = 1.1402
  Confidence 0.50: ||grad|| = 0.7071
  Confidence 0.90: ||grad|| = 0.1414
  Confidence 0.99: ||grad|| = 0.0141
```

### 3.3.6 Numerical Stability

Softmax + Cross-Entropy 계산 시 수치적 안정성을 확보하는 방법입니다.

**문제점:**

1. $e^{z_k}$가 매우 큰 값이 되어 오버플로우 발생
2. $\log(\hat{y}_k)$가 매우 작은 값에서 언더플로우 발생

**해결책: Log-Sum-Exp Trick**

$$\log\left(\sum_j e^{z_j}\right) = \log\left(\sum_j e^{z_j - z_{max}}\right) + z_{max}$$

```python
def log_softmax_stable(z):
    """수치적으로 안정한 log-softmax"""
    if z.ndim == 1:
        z_max = np.max(z)
        log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
        return z - log_sum_exp
    else:
        z_max = np.max(z, axis=1, keepdims=True)
        log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max), axis=1, keepdims=True))
        return z - log_sum_exp

def cross_entropy_with_logits(z, y):
    """수치적으로 안정한 Softmax + Cross-Entropy"""
    log_probs = log_softmax_stable(z)
    return -np.sum(y * log_probs) / (z.shape[0] if z.ndim > 1 else 1)

# 수치 안정성 테스트
z_extreme = np.array([1000.0, 1001.0, 1002.0])
y = np.array([0, 0, 1])

# 불안정한 방법
def unstable_cross_entropy(z, y):
    with np.errstate(over='ignore', invalid='ignore'):
        y_hat = np.exp(z) / np.sum(np.exp(z))
        return -np.sum(y * np.log(y_hat + 1e-8))

# 안정한 방법
loss_unstable = unstable_cross_entropy(z_extreme, y)
loss_stable = cross_entropy_with_logits(z_extreme, y)

print("Numerical Stability Test:")
print(f"Extreme logits: z = {z_extreme}")
print(f"Unstable method: L = {loss_unstable}")
print(f"Stable method:   L = {loss_stable:.6f}")

# 그래디언트도 안정적으로 계산
y_hat_stable = softmax(z_extreme)  # 내부적으로 z - max(z) 사용
grad_stable = y_hat_stable - y

print(f"\nStable gradient: {grad_stable.round(6)}")
```

```
Numerical Stability Test:
Extreme logits: z = [1000. 1001. 1002.]
Unstable method: L = nan
Stable method:   L = 0.407606

Stable gradient: [0.090031 0.244728 -0.33476]
```

### 3.3.7 Summary

**Softmax + Cross-Entropy 그래디언트 유도 정리:**

| 단계 | 수식 |
|------|------|
| Softmax 정의 | $\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$ |
| Softmax 미분 | $\frac{\partial \hat{y}_k}{\partial z_i} = \hat{y}_k(\delta_{ki} - \hat{y}_i)$ |
| Cross-Entropy 정의 | $L = -\sum_k y_k \log(\hat{y}_k)$ |
| CE의 $\hat{y}$ 미분 | $\frac{\partial L}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}$ |
| 결합 그래디언트 | $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$ |

**핵심 포인트:**

1. Softmax 미분은 야코비안 행렬 형태로 표현됨
2. Cross-Entropy와 결합하면 $\hat{y} - y$로 단순화
3. 이 단순한 형태가 역전파 구현을 매우 효율적으로 만듦
4. 수치 안정성을 위해 log-sum-exp 트릭 사용

```python
# 최종 구현
class SoftmaxCrossEntropy:
    """Softmax + Cross-Entropy 손실 함수"""
    
    def forward(self, z, y):
        """
        Args:
            z: 로짓 (N, K)
            y: 원-핫 레이블 (N, K)
        Returns:
            loss: 스칼라
        """
        self.y_hat = softmax(z)
        self.y = y
        
        # 수치적으로 안정한 계산
        log_probs = log_softmax_stable(z)
        loss = -np.mean(np.sum(y * log_probs, axis=1))
        return loss
    
    def backward(self):
        """
        Returns:
            grad: z에 대한 그래디언트 (N, K)
        """
        N = self.y.shape[0]
        return (self.y_hat - self.y) / N

# 사용 예시
loss_fn = SoftmaxCrossEntropy()

z = np.array([[2.0, 1.0, 0.1],
              [0.5, 2.5, 1.0],
              [1.0, 1.0, 3.0]])
y = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

loss = loss_fn.forward(z, y)
grad = loss_fn.backward()

print("SoftmaxCrossEntropy Usage:")
print(f"Logits z:\n{z}")
print(f"Labels y:\n{y}")
print(f"\nLoss: {loss:.4f}")
print(f"\nGradient ∂L/∂z:\n{grad.round(4)}")
```

```
SoftmaxCrossEntropy Usage:
Logits z:
[[2.  1.  0.1]
 [0.5 2.5 1. ]
 [1.  1.  3. ]]
Labels y:
[[1 0 0]
 [0 1 0]
 [0 0 1]]

Loss: 0.3658

Gradient ∂L/∂z:
[[-0.1137  0.0808  0.0329]
 [ 0.0448 -0.0858  0.0410]
 [ 0.0422  0.0422 -0.0843]]
```

다음 섹션에서는 역전파에서 각 연산의 행렬 차원을 상세히 분석합니다.
