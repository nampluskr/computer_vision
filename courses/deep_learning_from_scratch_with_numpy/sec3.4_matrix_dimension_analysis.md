## 3.4 Matrix Dimension Analysis

역전파 구현에서 가장 흔한 오류는 행렬 차원 불일치입니다. 이 섹션에서는 순전파와 역전파의 모든 연산에서 행렬 차원이 어떻게 변화하는지 체계적으로 분석합니다.

### 3.4.1 Notation and Setup

분석에 사용할 네트워크 설정입니다.

**네트워크 구성:**

| 기호 | 의미 | 예시 값 |
|------|------|---------|
| $N$ | 배치 크기 | 64 |
| $D_{in}$ | 입력 차원 | 784 |
| $H_1$ | 은닉층 1 크기 | 256 |
| $H_2$ | 은닉층 2 크기 | 128 |
| $K$ | 출력 클래스 수 | 10 |

**표기 규칙:**

- 행렬/벡터: 볼드체 ($\mathbf{W}$, $\mathbf{x}$)
- Shape 표기: $(행, 열)$ 형태
- `@`: 행렬 곱 (np.dot)
- `*`: 원소별 곱 (element-wise)

```python
import numpy as np

# 네트워크 설정
N = 64       # 배치 크기
D_in = 784   # 입력 차원
H1 = 256     # 은닉층 1
H2 = 128     # 은닉층 2
K = 10       # 출력 클래스

print("Network Configuration:")
print(f"{'Symbol':<8} {'Meaning':<20} {'Value':<10}")
print("-" * 40)
print(f"{'N':<8} {'Batch size':<20} {N:<10}")
print(f"{'D_in':<8} {'Input dimension':<20} {D_in:<10}")
print(f"{'H1':<8} {'Hidden layer 1':<20} {H1:<10}")
print(f"{'H2':<8} {'Hidden layer 2':<20} {H2:<10}")
print(f"{'K':<8} {'Output classes':<20} {K:<10}")
```

```
Network Configuration:
Symbol   Meaning              Value     
----------------------------------------
N        Batch size           64        
D_in     Input dimension      784       
H1       Hidden layer 1       256       
H2       Hidden layer 2       128       
K        Output classes       10        
```

### 3.4.2 Parameter Dimensions

각 레이어의 파라미터 차원입니다.

**차원 결정 규칙:**

- 가중치 $W^{(l)}$: (이전 레이어 출력 차원, 현재 레이어 출력 차원)
- 편향 $b^{(l)}$: (현재 레이어 출력 차원,)

```python
# 파라미터 초기화 및 차원 확인
np.random.seed(42)

# Layer 1: D_in -> H1
W1 = np.random.randn(D_in, H1) * np.sqrt(2.0 / D_in)
b1 = np.zeros(H1)

# Layer 2: H1 -> H2
W2 = np.random.randn(H1, H2) * np.sqrt(2.0 / H1)
b2 = np.zeros(H2)

# Layer 3: H2 -> K
W3 = np.random.randn(H2, K) * np.sqrt(2.0 / H2)
b3 = np.zeros(K)

print("Parameter Dimensions:")
print("=" * 50)
print(f"{'Parameter':<12} {'Shape':<15} {'Size':<12} {'Description'}")
print("-" * 50)
print(f"{'W1':<12} {str(W1.shape):<15} {W1.size:<12} Input -> Hidden1")
print(f"{'b1':<12} {str(b1.shape):<15} {b1.size:<12} Hidden1 bias")
print(f"{'W2':<12} {str(W2.shape):<15} {W2.size:<12} Hidden1 -> Hidden2")
print(f"{'b2':<12} {str(b2.shape):<15} {b2.size:<12} Hidden2 bias")
print(f"{'W3':<12} {str(W3.shape):<15} {W3.size:<12} Hidden2 -> Output")
print(f"{'b3':<12} {str(b3.shape):<15} {b3.size:<12} Output bias")
print("-" * 50)
total_params = W1.size + b1.size + W2.size + b2.size + W3.size + b3.size
print(f"{'Total':<12} {'':<15} {total_params:<12}")
```

```
Parameter Dimensions:
==================================================
Parameter    Shape           Size         Description
--------------------------------------------------
W1           (784, 256)      200704       Input -> Hidden1
b1           (256,)          256          Hidden1 bias
W2           (256, 128)      32768        Hidden1 -> Hidden2
b2           (128,)          128          Hidden2 bias
W3           (128, 10)       1280         Hidden2 -> Output
b3           (10,)           10           Output bias
--------------------------------------------------
Total                        234146      
```

### 3.4.3 Forward Pass Dimensions

순전파의 각 연산에서 차원 변화를 추적합니다.

**순전파 수식과 차원:**

```
x       : (N, D_in)         = (64, 784)

Layer 1:
z1 = x @ W1 + b1
     (N, D_in) @ (D_in, H1) + (H1,)
     (64, 784) @ (784, 256) + (256,)
     = (64, 256) + (256,)    [broadcasting]
     = (64, 256)
a1 = sigmoid(z1)             = (64, 256)

Layer 2:
z2 = a1 @ W2 + b2
     (N, H1) @ (H1, H2) + (H2,)
     (64, 256) @ (256, 128) + (128,)
     = (64, 128) + (128,)    [broadcasting]
     = (64, 128)
a2 = sigmoid(z2)             = (64, 128)

Layer 3:
z3 = a2 @ W3 + b3
     (N, H2) @ (H2, K) + (K,)
     (64, 128) @ (128, 10) + (10,)
     = (64, 10) + (10,)      [broadcasting]
     = (64, 10)
ŷ = softmax(z3)              = (64, 10)
```

```python
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# 입력 데이터
x = np.random.randn(N, D_in)

print("Forward Pass Dimension Tracking:")
print("=" * 70)

# Layer 1
print("\n[Layer 1]")
print(f"  x:  {x.shape}")
print(f"  W1: {W1.shape}")
print(f"  x @ W1: {x.shape} @ {W1.shape} = {(x @ W1).shape}")
z1 = x @ W1 + b1
print(f"  z1 = x @ W1 + b1: {z1.shape}")
a1 = sigmoid(z1)
print(f"  a1 = sigmoid(z1): {a1.shape}")

# Layer 2
print("\n[Layer 2]")
print(f"  a1: {a1.shape}")
print(f"  W2: {W2.shape}")
print(f"  a1 @ W2: {a1.shape} @ {W2.shape} = {(a1 @ W2).shape}")
z2 = a1 @ W2 + b2
print(f"  z2 = a1 @ W2 + b2: {z2.shape}")
a2 = sigmoid(z2)
print(f"  a2 = sigmoid(z2): {a2.shape}")

# Layer 3
print("\n[Layer 3]")
print(f"  a2: {a2.shape}")
print(f"  W3: {W3.shape}")
print(f"  a2 @ W3: {a2.shape} @ {W3.shape} = {(a2 @ W3).shape}")
z3 = a2 @ W3 + b3
print(f"  z3 = a2 @ W3 + b3: {z3.shape}")
y_hat = softmax(z3)
print(f"  ŷ = softmax(z3): {y_hat.shape}")

print("\n" + "=" * 70)
```

```
Forward Pass Dimension Tracking:
======================================================================

[Layer 1]
  x:  (64, 784)
  W1: (784, 256)
  x @ W1: (64, 784) @ (784, 256) = (64, 256)
  z1 = x @ W1 + b1: (64, 256)
  a1 = sigmoid(z1): (64, 256)

[Layer 2]
  a1: (64, 256)
  W2: (256, 128)
  a1 @ W2: (64, 256) @ (256, 128) = (64, 128)
  z2 = a1 @ W2 + b2: (64, 128)
  a2 = sigmoid(z2): (64, 128)

[Layer 3]
  a2: (64, 128)
  W3: (128, 10)
  a2 @ W3: (64, 128) @ (128, 10) = (64, 10)
  z3 = a2 @ W3 + b3: (64, 10)
  ŷ = softmax(z3): (64, 10)

======================================================================
```

### 3.4.4 Backward Pass Dimensions

역전파의 각 연산에서 차원 변화를 추적합니다. 핵심 규칙은 **그래디언트의 shape은 해당 변수의 shape과 동일**해야 한다는 것입니다.

**역전파 수식과 차원:**

```
y       : (N, K)             = (64, 10)
ŷ       : (N, K)             = (64, 10)

Output Layer:
δ3 = (ŷ - y) / N
     (N, K) / scalar
     = (64, 10)

∂L/∂W3 = a2.T @ δ3
         (H2, N) @ (N, K)
         (128, 64) @ (64, 10)
         = (128, 10)          ✓ matches W3.shape

∂L/∂b3 = sum(δ3, axis=0)
         = (10,)              ✓ matches b3.shape

Hidden Layer 2:
∂L/∂a2 = δ3 @ W3.T
         (N, K) @ (K, H2)
         (64, 10) @ (10, 128)
         = (64, 128)          ✓ matches a2.shape

δ2 = ∂L/∂a2 * a2 * (1 - a2)
     (N, H2) * (N, H2) * (N, H2)
     = (64, 128)

∂L/∂W2 = a1.T @ δ2
         (H1, N) @ (N, H2)
         (256, 64) @ (64, 128)
         = (256, 128)         ✓ matches W2.shape

∂L/∂b2 = sum(δ2, axis=0)
         = (128,)             ✓ matches b2.shape

Hidden Layer 1:
∂L/∂a1 = δ2 @ W2.T
         (N, H2) @ (H2, H1)
         (64, 128) @ (128, 256)
         = (64, 256)          ✓ matches a1.shape

δ1 = ∂L/∂a1 * a1 * (1 - a1)
     (N, H1) * (N, H1) * (N, H1)
     = (64, 256)

∂L/∂W1 = x.T @ δ1
         (D_in, N) @ (N, H1)
         (784, 64) @ (64, 256)
         = (784, 256)         ✓ matches W1.shape

∂L/∂b1 = sum(δ1, axis=0)
         = (256,)             ✓ matches b1.shape
```

```python
# 정답 레이블 (one-hot)
y = np.eye(K)[np.random.randint(0, K, N)]

print("Backward Pass Dimension Tracking:")
print("=" * 70)

# Output Layer
print("\n[Output Layer]")
delta3 = (y_hat - y) / N
print(f"  ŷ: {y_hat.shape}, y: {y.shape}")
print(f"  δ3 = (ŷ - y) / N: {delta3.shape}")

grad_W3 = a2.T @ delta3
grad_b3 = np.sum(delta3, axis=0)
print(f"\n  ∂L/∂W3 = a2.T @ δ3")
print(f"    a2.T: {a2.T.shape}, δ3: {delta3.shape}")
print(f"    Result: {grad_W3.shape} ✓ matches W3: {W3.shape}")
print(f"\n  ∂L/∂b3 = sum(δ3, axis=0)")
print(f"    Result: {grad_b3.shape} ✓ matches b3: {b3.shape}")

# Hidden Layer 2
print("\n[Hidden Layer 2]")
grad_a2 = delta3 @ W3.T
print(f"  ∂L/∂a2 = δ3 @ W3.T")
print(f"    δ3: {delta3.shape}, W3.T: {W3.T.shape}")
print(f"    Result: {grad_a2.shape} ✓ matches a2: {a2.shape}")

delta2 = grad_a2 * a2 * (1 - a2)
print(f"\n  δ2 = ∂L/∂a2 * a2 * (1 - a2)")
print(f"    Element-wise: {grad_a2.shape} * {a2.shape} * {a2.shape}")
print(f"    Result: {delta2.shape}")

grad_W2 = a1.T @ delta2
grad_b2 = np.sum(delta2, axis=0)
print(f"\n  ∂L/∂W2 = a1.T @ δ2")
print(f"    a1.T: {a1.T.shape}, δ2: {delta2.shape}")
print(f"    Result: {grad_W2.shape} ✓ matches W2: {W2.shape}")
print(f"\n  ∂L/∂b2 = sum(δ2, axis=0)")
print(f"    Result: {grad_b2.shape} ✓ matches b2: {b2.shape}")

# Hidden Layer 1
print("\n[Hidden Layer 1]")
grad_a1 = delta2 @ W2.T
print(f"  ∂L/∂a1 = δ2 @ W2.T")
print(f"    δ2: {delta2.shape}, W2.T: {W2.T.shape}")
print(f"    Result: {grad_a1.shape} ✓ matches a1: {a1.shape}")

delta1 = grad_a1 * a1 * (1 - a1)
print(f"\n  δ1 = ∂L/∂a1 * a1 * (1 - a1)")
print(f"    Element-wise: {grad_a1.shape} * {a1.shape} * {a1.shape}")
print(f"    Result: {delta1.shape}")

grad_W1 = x.T @ delta1
grad_b1 = np.sum(delta1, axis=0)
print(f"\n  ∂L/∂W1 = x.T @ δ1")
print(f"    x.T: {x.T.shape}, δ1: {delta1.shape}")
print(f"    Result: {grad_W1.shape} ✓ matches W1: {W1.shape}")
print(f"\n  ∂L/∂b1 = sum(δ1, axis=0)")
print(f"    Result: {grad_b1.shape} ✓ matches b1: {b1.shape}")

print("\n" + "=" * 70)
```

```
Backward Pass Dimension Tracking:
======================================================================

[Output Layer]
  ŷ: (64, 10), y: (64, 10)
  δ3 = (ŷ - y) / N: (64, 10)

  ∂L/∂W3 = a2.T @ δ3
    a2.T: (128, 64), δ3: (64, 10)
    Result: (128, 10) ✓ matches W3: (128, 10)

  ∂L/∂b3 = sum(δ3, axis=0)
    Result: (10,) ✓ matches b3: (10,)

[Hidden Layer 2]
  ∂L/∂a2 = δ3 @ W3.T
    δ3: (64, 10), W3.T: (10, 128)
    Result: (64, 128) ✓ matches a2: (64, 128)

  δ2 = ∂L/∂a2 * a2 * (1 - a2)
    Element-wise: (64, 128) * (64, 128) * (64, 128)
    Result: (64, 128)

  ∂L/∂W2 = a1.T @ δ2
    a1.T: (256, 64), δ2: (64, 128)
    Result: (256, 128) ✓ matches W2: (256, 128)

  ∂L/∂b2 = sum(δ2, axis=0)
    Result: (128,) ✓ matches b2: (128,)

[Hidden Layer 1]
  ∂L/∂a1 = δ2 @ W2.T
    δ2: (64, 128), W2.T: (128, 256)
    Result: (64, 256) ✓ matches a1: (64, 256)

  δ1 = ∂L/∂a1 * a1 * (1 - a1)
    Element-wise: (64, 256) * (64, 256) * (64, 256)
    Result: (64, 256)

  ∂L/∂W1 = x.T @ δ1
    x.T: (784, 64), δ1: (64, 256)
    Result: (784, 256) ✓ matches W1: (784, 256)

  ∂L/∂b1 = sum(δ1, axis=0)
    Result: (256,) ✓ matches b1: (256,)

======================================================================
```

### 3.4.5 Dimension Rules

차원 분석에서 발견되는 일반적인 규칙들입니다.

**규칙 1: 그래디언트 Shape 일치**

$$\text{shape}(\nabla_W L) = \text{shape}(W)$$
$$\text{shape}(\nabla_b L) = \text{shape}(b)$$

**규칙 2: 가중치 그래디언트 공식**

$$\nabla_W L = (\text{input})^T @ \delta$$

차원 분석:
- input: $(N, D_{in})$
- δ: $(N, D_{out})$
- input.T @ δ: $(D_{in}, N) @ (N, D_{out}) = (D_{in}, D_{out})$ ✓

**규칙 3: 그래디언트 전파 공식**

$$\nabla_a L = \delta @ W^T$$

차원 분석:
- δ: $(N, D_{out})$
- W: $(D_{in}, D_{out})$
- δ @ W.T: $(N, D_{out}) @ (D_{out}, D_{in}) = (N, D_{in})$ ✓

**규칙 4: 편향 그래디언트**

$$\nabla_b L = \sum_{i=1}^{N} \delta_i = \text{sum}(\delta, \text{axis}=0)$$

배치 차원을 합산하여 제거합니다.

```python
def verify_dimension_rules():
    """차원 규칙 검증"""
    print("Dimension Rules Verification:")
    print("=" * 60)
    
    # 일반적인 레이어 설정
    N, D_in, D_out = 32, 100, 50
    
    # Forward: z = x @ W + b
    x = np.random.randn(N, D_in)
    W = np.random.randn(D_in, D_out)
    b = np.zeros(D_out)
    
    z = x @ W + b
    
    print("\n[Forward]")
    print(f"  x: {x.shape}")
    print(f"  W: {W.shape}")
    print(f"  b: {b.shape}")
    print(f"  z = x @ W + b: {z.shape}")
    
    # Backward
    delta = np.random.randn(N, D_out)  # 상위 레이어에서 전파된 그래디언트
    
    print("\n[Backward]")
    print(f"  δ (from upper layer): {delta.shape}")
    
    # Rule 1 & 2: Weight gradient
    grad_W = x.T @ delta
    print(f"\n  Rule 2: ∇W = x.T @ δ")
    print(f"    x.T: {x.T.shape} @ δ: {delta.shape} = {grad_W.shape}")
    print(f"    Matches W? {grad_W.shape == W.shape} ✓")
    
    # Rule 3: Gradient propagation
    grad_x = delta @ W.T
    print(f"\n  Rule 3: ∇x = δ @ W.T")
    print(f"    δ: {delta.shape} @ W.T: {W.T.shape} = {grad_x.shape}")
    print(f"    Matches x? {grad_x.shape == x.shape} ✓")
    
    # Rule 4: Bias gradient
    grad_b = np.sum(delta, axis=0)
    print(f"\n  Rule 4: ∇b = sum(δ, axis=0)")
    print(f"    sum({delta.shape}, axis=0) = {grad_b.shape}")
    print(f"    Matches b? {grad_b.shape == b.shape} ✓")
    
    print("\n" + "=" * 60)

verify_dimension_rules()
```

```
Dimension Rules Verification:
============================================================

[Forward]
  x: (32, 100)
  W: (100, 50)
  b: (50,)
  z = x @ W + b: (32, 50)

[Backward]
  δ (from upper layer): (32, 50)

  Rule 2: ∇W = x.T @ δ
    x.T: (100, 32) @ δ: (32, 50) = (100, 50)
    Matches W? True ✓

  Rule 3: ∇x = δ @ W.T
    δ: (32, 50) @ W.T: (50, 100) = (32, 100)
    Matches x? True ✓

  Rule 4: ∇b = sum(δ, axis=0)
    sum((32, 50), axis=0) = (50,)
    Matches b? True ✓

============================================================
```

### 3.4.6 Common Dimension Errors

자주 발생하는 차원 오류와 해결 방법입니다.

```python
def demonstrate_common_errors():
    """흔한 차원 오류 시연"""
    print("Common Dimension Errors:")
    print("=" * 60)
    
    N, D_in, D_out = 32, 100, 50
    x = np.random.randn(N, D_in)
    W = np.random.randn(D_in, D_out)
    delta = np.random.randn(N, D_out)
    
    # Error 1: 전치 누락
    print("\n[Error 1] Missing transpose in weight gradient")
    print(f"  Wrong:  x @ δ → {x.shape} @ {delta.shape}")
    try:
        wrong_grad = x @ delta
    except ValueError as e:
        print(f"  Error: {e}")
    print(f"  Correct: x.T @ δ → {x.T.shape} @ {delta.shape} = {(x.T @ delta).shape}")
    
    # Error 2: 전치 방향 오류
    print("\n[Error 2] Wrong transpose direction in gradient propagation")
    print(f"  Wrong:  δ.T @ W → {delta.T.shape} @ {W.shape}")
    try:
        wrong_prop = delta.T @ W
        print(f"  Result: {wrong_prop.shape} (should be {x.shape})")
    except ValueError as e:
        print(f"  Error: {e}")
    print(f"  Correct: δ @ W.T → {delta.shape} @ {W.T.shape} = {(delta @ W.T).shape}")
    
    # Error 3: 축 합산 방향 오류
    print("\n[Error 3] Wrong axis in bias gradient")
    wrong_axis = np.sum(delta, axis=1)
    correct_axis = np.sum(delta, axis=0)
    print(f"  Wrong:  sum(δ, axis=1) → {wrong_axis.shape}")
    print(f"  Correct: sum(δ, axis=0) → {correct_axis.shape}")
    
    # Error 4: 원소별 곱에서 브로드캐스팅 오류
    print("\n[Error 4] Broadcasting error in element-wise operations")
    a = np.random.randn(N, D_out)
    grad_a = np.random.randn(N, D_out)
    
    # Sigmoid backward: correct
    correct_sigmoid = grad_a * a * (1 - a)
    print(f"  Correct: grad_a * a * (1-a)")
    print(f"    {grad_a.shape} * {a.shape} * {a.shape} = {correct_sigmoid.shape}")
    
    print("\n" + "=" * 60)

demonstrate_common_errors()
```

```
Common Dimension Errors:
============================================================

[Error 1] Missing transpose in weight gradient
  Wrong:  x @ δ → (32, 100) @ (32, 50)
  Error: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 32 is different from 100)
  Correct: x.T @ δ → (100, 32) @ (32, 50) = (100, 50)

[Error 2] Wrong transpose direction in gradient propagation
  Wrong:  δ.T @ W → (50, 32) @ (100, 50)
  Error: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 100 is different from 32)
  Correct: δ @ W.T → (32, 50) @ (50, 100) = (32, 100)

[Error 3] Wrong axis in bias gradient
  Wrong:  sum(δ, axis=1) → (32,)
  Correct: sum(δ, axis=0) → (50,)

[Error 4] Broadcasting error in element-wise operations
  Correct: grad_a * a * (1-a)
    (32, 50) * (32, 50) * (32, 50) = (32, 50)

============================================================
```

### 3.4.7 Dimension Checklist

구현 시 확인해야 할 차원 체크리스트입니다.

```python
def dimension_checklist(layer_sizes):
    """레이어 구성에 따른 차원 체크리스트 생성"""
    print("Dimension Checklist:")
    print("=" * 70)
    
    n_layers = len(layer_sizes) - 1
    
    print(f"\nNetwork: {' -> '.join(map(str, layer_sizes))}")
    print(f"Batch size: N")
    
    print("\n[Parameters]")
    print(f"{'Layer':<8} {'Weight':<20} {'Bias':<15}")
    print("-" * 45)
    for i in range(n_layers):
        d_in, d_out = layer_sizes[i], layer_sizes[i+1]
        print(f"{i+1:<8} W{i+1}: ({d_in}, {d_out}){'':<8} b{i+1}: ({d_out},)")
    
    print("\n[Forward Pass]")
    print(f"{'Variable':<10} {'Shape':<20} {'Operation'}")
    print("-" * 60)
    print(f"{'x':<10} {'(N, ' + str(layer_sizes[0]) + ')':<20} Input")
    for i in range(n_layers):
        d_in, d_out = layer_sizes[i], layer_sizes[i+1]
        print(f"{'z' + str(i+1):<10} {'(N, ' + str(d_out) + ')':<20} a{i} @ W{i+1} + b{i+1}" if i > 0 
              else f"{'z1':<10} {'(N, ' + str(d_out) + ')':<20} x @ W1 + b1")
        act = 'softmax' if i == n_layers - 1 else 'sigmoid'
        print(f"{'a' + str(i+1):<10} {'(N, ' + str(d_out) + ')':<20} {act}(z{i+1})")
    
    print("\n[Backward Pass]")
    print(f"{'Variable':<10} {'Shape':<20} {'Operation'}")
    print("-" * 60)
    
    # Output layer
    d_out = layer_sizes[-1]
    print(f"{'δ' + str(n_layers):<10} {'(N, ' + str(d_out) + ')':<20} (ŷ - y) / N")
    
    for i in range(n_layers, 0, -1):
        d_in = layer_sizes[i-1]
        d_out = layer_sizes[i]
        
        print(f"{'∂L/∂W' + str(i):<10} {'(' + str(d_in) + ', ' + str(d_out) + ')':<20} a{i-1}.T @ δ{i}" if i > 1
              else f"{'∂L/∂W1':<10} {'(' + str(d_in) + ', ' + str(d_out) + ')':<20} x.T @ δ1")
        print(f"{'∂L/∂b' + str(i):<10} {'(' + str(d_out) + ',)':<20} sum(δ{i}, axis=0)")
        
        if i > 1:
            d_prev = layer_sizes[i-1]
            print(f"{'∂L/∂a' + str(i-1):<10} {'(N, ' + str(d_prev) + ')':<20} δ{i} @ W{i}.T")
            print(f"{'δ' + str(i-1):<10} {'(N, ' + str(d_prev) + ')':<20} ∂L/∂a{i-1} * σ'(z{i-1})")
    
    print("\n" + "=" * 70)

# 체크리스트 생성
dimension_checklist([784, 256, 128, 10])
```

```
Dimension Checklist:
======================================================================

Network: 784 -> 256 -> 128 -> 10
Batch size: N

[Parameters]
Layer    Weight               Bias           
---------------------------------------------
1        W1: (784, 256)       b1: (256,)
2        W2: (256, 128)       b2: (128,)
3        W3: (128, 10)        b3: (10,)

[Forward Pass]
Variable   Shape                Operation
------------------------------------------------------------
x          (N, 784)             Input
z1         (N, 256)             x @ W1 + b1
a1         (N, 256)             sigmoid(z1)
z2         (N, 128)             a1 @ W2 + b2
a2         (N, 128)             sigmoid(z2)
z3         (N, 10)              a2 @ W3 + b3
a3         (N, 10)              softmax(z3)

[Backward Pass]
Variable   Shape                Operation
------------------------------------------------------------
δ3         (N, 10)              (ŷ - y) / N
∂L/∂W3     (128, 10)            a2.T @ δ3
∂L/∂b3     (10,)                sum(δ3, axis=0)
∂L/∂a2     (N, 128)             δ3 @ W3.T
δ2         (N, 128)             ∂L/∂a2 * σ'(z2)
∂L/∂W2     (256, 128)           a1.T @ δ2
∂L/∂b2     (128,)               sum(δ2, axis=0)
∂L/∂a1     (N, 256)             δ2 @ W2.T
δ1         (N, 256)             ∂L/∂a1 * σ'(z1)
∂L/∂W1     (784, 256)           x.T @ δ1
∂L/∂b1     (256,)               sum(δ1, axis=0)

======================================================================
```

### 3.4.8 Summary

| 연산 | 수식 | 입력 Shape | 출력 Shape |
|------|------|------------|------------|
| Linear Forward | $z = xW + b$ | $(N, D_{in})$, $(D_{in}, D_{out})$ | $(N, D_{out})$ |
| Weight Gradient | $\nabla_W = x^T \delta$ | $(D_{in}, N)$, $(N, D_{out})$ | $(D_{in}, D_{out})$ |
| Bias Gradient | $\nabla_b = \sum \delta$ | $(N, D_{out})$ | $(D_{out},)$ |
| Gradient Propagation | $\nabla_x = \delta W^T$ | $(N, D_{out})$, $(D_{out}, D_{in})$ | $(N, D_{in})$ |
| Activation Backward | $\delta' = \nabla_a \odot \sigma'(z)$ | $(N, D)$, $(N, D)$ | $(N, D)$ |

**핵심 포인트:**

1. 그래디언트 shape은 항상 해당 파라미터의 shape과 일치
2. 가중치 그래디언트: (입력).T @ δ
3. 그래디언트 전파: δ @ (가중치).T
4. 편향 그래디언트: 배치 차원(axis=0)을 합산
5. 원소별 연산은 shape이 동일해야 함

다음 섹션에서는 계산 그래프를 시각적으로 표현하여 역전파의 흐름을 직관적으로 이해합니다.
