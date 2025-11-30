## 3.5 Computational Graph Visualization

계산 그래프(Computational Graph)는 신경망의 연산을 노드와 엣지로 표현한 것입니다. 순전파에서는 입력에서 출력 방향으로, 역전파에서는 출력에서 입력 방향으로 그래디언트가 흐릅니다. 이 섹션에서는 계산 그래프를 통해 역전파의 흐름을 시각적으로 이해합니다.

### 3.5.1 What is a Computational Graph

계산 그래프는 수학적 연산을 그래프 구조로 표현합니다.

**구성 요소:**

- **노드 (Node)**: 연산 또는 변수
- **엣지 (Edge)**: 데이터 흐름 방향

**간단한 예시:**

$f(x, y, z) = (x + y) \times z$를 계산 그래프로 표현하면:

```
     x ─────┐
             ├──► (+) ──► q ──┐
     y ─────┘                  ├──► (×) ──► f
                               │
     z ────────────────────────┘
```

여기서 $q = x + y$, $f = q \times z$ 입니다.

```python
import numpy as np

def simple_graph_example():
    """간단한 계산 그래프 예시"""
    # Forward pass
    x, y, z = 2.0, 3.0, 4.0
    
    q = x + y      # 덧셈 노드
    f = q * z      # 곱셈 노드
    
    print("Forward Pass: f(x, y, z) = (x + y) × z")
    print(f"  x = {x}, y = {y}, z = {z}")
    print(f"  q = x + y = {q}")
    print(f"  f = q × z = {f}")
    
    # Backward pass
    # ∂f/∂f = 1 (시작점)
    df_df = 1.0
    
    # 곱셈 노드의 역전파: f = q × z
    # ∂f/∂q = z, ∂f/∂z = q
    df_dq = z * df_df
    df_dz = q * df_df
    
    # 덧셈 노드의 역전파: q = x + y
    # ∂q/∂x = 1, ∂q/∂y = 1
    df_dx = 1 * df_dq
    df_dy = 1 * df_dq
    
    print("\nBackward Pass:")
    print(f"  ∂f/∂f = {df_df}")
    print(f"  ∂f/∂q = z = {df_dq}")
    print(f"  ∂f/∂z = q = {df_dz}")
    print(f"  ∂f/∂x = ∂f/∂q × ∂q/∂x = {df_dx}")
    print(f"  ∂f/∂y = ∂f/∂q × ∂q/∂y = {df_dy}")
    
    # 검증
    print("\nVerification (numerical gradient):")
    eps = 1e-5
    df_dx_num = ((x + eps + y) * z - (x - eps + y) * z) / (2 * eps)
    df_dy_num = ((x + y + eps) * z - (x + y - eps) * z) / (2 * eps)
    df_dz_num = ((x + y) * (z + eps) - (x + y) * (z - eps)) / (2 * eps)
    print(f"  ∂f/∂x (numerical) = {df_dx_num}")
    print(f"  ∂f/∂y (numerical) = {df_dy_num}")
    print(f"  ∂f/∂z (numerical) = {df_dz_num}")

simple_graph_example()
```

```
Forward Pass: f(x, y, z) = (x + y) × z
  x = 2.0, y = 3.0, z = 4.0
  q = x + y = 5.0
  f = q × z = 20.0

Backward Pass:
  ∂f/∂f = 1.0
  ∂f/∂q = z = 4.0
  ∂f/∂z = q = 5.0
  ∂f/∂x = ∂f/∂q × ∂q/∂x = 4.0
  ∂f/∂y = ∂f/∂q × ∂q/∂y = 4.0

Verification (numerical gradient):
  ∂f/∂x (numerical) = 4.000000000026205
  ∂f/∂y (numerical) = 4.000000000026205
  ∂f/∂z (numerical) = 5.000000000032756
```

### 3.5.2 Forward Graph of MLP

3층 MLP의 순전파 계산 그래프입니다.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FORWARD PASS                                        │
│                         Input ────────────────► Output                           │
└─────────────────────────────────────────────────────────────────────────────────┘

  x          W1         z1         a1         W2         z2         a2
(N,784)   (784,256)   (N,256)   (N,256)   (256,128)   (N,128)   (N,128)
   │          │          │          │          │          │          │
   │          │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼          ▼
   │          │          │          │          │          │          │
   └────┬─────┘          │          └────┬─────┘          │          │
        │                │               │                │          │
        ▼                │               ▼                │          │
   ┌─────────┐           │          ┌─────────┐           │          │
   │  MatMul │           │          │  MatMul │           │          │
   │  x @ W1 │           │          │ a1 @ W2 │           │          │
   └────┬────┘           │          └────┬────┘           │          │
        │                │               │                │          │
        ▼     b1         │               ▼     b2         │          │
   ┌─────────┐(256,)     │          ┌─────────┐(128,)     │          │
   │   Add   │◄──────────┤          │   Add   │◄──────────┤          │
   │  + b1   │           │          │  + b2   │           │          │
   └────┬────┘           │          └────┬────┘           │          │
        │                │               │                │          │
        ▼                │               ▼                │          │
   ┌─────────┐           │          ┌─────────┐           │          │
   │ Sigmoid │───────────┘          │ Sigmoid │───────────┘          │
   │  σ(z1)  │                      │  σ(z2)  │                      │
   └────┬────┘                      └────┬────┘                      │
        │                                │                           │
        ▼                                ▼                           │
       a1                               a2                           │
     (N,256)                          (N,128)                        │
                                         │                           │
                                         │         W3                │
                                         │      (128,10)             │
                                         │          │                │
                                         └────┬─────┘                │
                                              │                      │
                                              ▼                      │
                                         ┌─────────┐                 │
                                         │  MatMul │                 │
                                         │ a2 @ W3 │                 │
                                         └────┬────┘                 │
                                              │                      │
                                              ▼     b3               │
                                         ┌─────────┐(10,)            │
                                         │   Add   │◄────────────────┘
                                         │  + b3   │
                                         └────┬────┘
                                              │
                                              ▼
                                             z3
                                           (N,10)
                                              │
                                              ▼
                                         ┌─────────┐
                                         │ Softmax │
                                         └────┬────┘
                                              │
                                              ▼
                                              ŷ
                                           (N,10)
                                              │
                                              │       y
                                              │    (N,10)
                                              │       │
                                              └───┬───┘
                                                  │
                                                  ▼
                                             ┌─────────┐
                                             │   CE    │
                                             │  Loss   │
                                             └────┬────┘
                                                  │
                                                  ▼
                                                  L
                                              (scalar)
```

```python
def forward_graph_demo():
    """순전파 그래프 실행"""
    np.random.seed(42)
    
    # 설정
    N, D_in, H1, H2, K = 4, 784, 256, 128, 10
    
    # 파라미터
    W1 = np.random.randn(D_in, H1) * 0.01
    b1 = np.zeros(H1)
    W2 = np.random.randn(H1, H2) * 0.01
    b2 = np.zeros(H2)
    W3 = np.random.randn(H2, K) * 0.01
    b3 = np.zeros(K)
    
    # 입력
    x = np.random.randn(N, D_in)
    y = np.eye(K)[np.array([0, 1, 2, 3])]  # one-hot labels
    
    # Forward pass - 각 노드의 출력을 저장
    cache = {}
    
    print("Forward Pass Execution:")
    print("=" * 60)
    
    # Layer 1
    cache['x'] = x
    print(f"x:  {x.shape}")
    
    cache['z1'] = x @ W1 + b1
    print(f"z1 = x @ W1 + b1:  {cache['z1'].shape}")
    
    cache['a1'] = 1 / (1 + np.exp(-cache['z1']))  # sigmoid
    print(f"a1 = sigmoid(z1):  {cache['a1'].shape}")
    
    # Layer 2
    cache['z2'] = cache['a1'] @ W2 + b2
    print(f"z2 = a1 @ W2 + b2: {cache['z2'].shape}")
    
    cache['a2'] = 1 / (1 + np.exp(-cache['z2']))  # sigmoid
    print(f"a2 = sigmoid(z2):  {cache['a2'].shape}")
    
    # Layer 3
    cache['z3'] = cache['a2'] @ W3 + b3
    print(f"z3 = a2 @ W3 + b3: {cache['z3'].shape}")
    
    # Softmax
    e_z = np.exp(cache['z3'] - np.max(cache['z3'], axis=1, keepdims=True))
    cache['y_hat'] = e_z / np.sum(e_z, axis=1, keepdims=True)
    print(f"ŷ = softmax(z3):   {cache['y_hat'].shape}")
    
    # Loss
    loss = -np.mean(np.sum(y * np.log(cache['y_hat'] + 1e-8), axis=1))
    print(f"L = CE(ŷ, y):      scalar = {loss:.4f}")
    
    return cache, W1, b1, W2, b2, W3, b3, y

cache, W1, b1, W2, b2, W3, b3, y = forward_graph_demo()
```

```
Forward Pass Execution:
============================================================
x:  (4, 784)
z1 = x @ W1 + b1:  (4, 256)
a1 = sigmoid(z1):  (4, 256)
z2 = a1 @ W2 + b2: (4, 128)
a2 = sigmoid(z2):  (4, 128)
z3 = a2 @ W3 + b3: (4, 10)
ŷ = softmax(z3):   (4, 10)
L = CE(ŷ, y):      scalar = 2.3201
```

### 3.5.3 Backward Graph of MLP

역전파 계산 그래프입니다. 순전파의 역순으로 그래디언트가 흐릅니다.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKWARD PASS                                       │
│                         Output ────────────────► Input                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                                                  L
                                              (scalar)
                                                  │
                                                  │ ∂L/∂L = 1
                                                  ▼
                                         ┌───────────────┐
                                         │ Softmax + CE  │
                                         │   Backward    │
                                         └───────┬───────┘
                                                 │
                                                 │ δ3 = (ŷ - y) / N
                                                 ▼
                                                δ3
                                              (N,10)
                         ┌───────────────────────┼───────────────────────┐
                         │                       │                       │
                         ▼                       ▼                       ▼
                   ┌───────────┐           ┌───────────┐           ┌───────────┐
                   │ ∂L/∂W3    │           │ ∂L/∂b3    │           │ ∂L/∂a2    │
                   │= a2.T @ δ3│           │= Σδ3      │           │= δ3 @ W3.T│
                   └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
                         │                       │                       │
                         ▼                       ▼                       ▼
                    (128,10)                  (10,)                   (N,128)
                         │                       │                       │
                    [Update W3]            [Update b3]                   │
                                                                         ▼
                                                                  ┌───────────┐
                                                                  │ Sigmoid   │
                                                                  │ Backward  │
                                                                  │δ2=∂L/∂a2 │
                                                                  │ ⊙a2⊙(1-a2)│
                                                                  └─────┬─────┘
                                                                        │
                                                                        ▼
                                                                       δ2
                                                                     (N,128)
                         ┌───────────────────────┼───────────────────────┐
                         │                       │                       │
                         ▼                       ▼                       ▼
                   ┌───────────┐           ┌───────────┐           ┌───────────┐
                   │ ∂L/∂W2    │           │ ∂L/∂b2    │           │ ∂L/∂a1    │
                   │= a1.T @ δ2│           │= Σδ2      │           │= δ2 @ W2.T│
                   └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
                         │                       │                       │
                         ▼                       ▼                       ▼
                    (256,128)                 (128,)                  (N,256)
                         │                       │                       │
                    [Update W2]            [Update b2]                   │
                                                                         ▼
                                                                  ┌───────────┐
                                                                  │ Sigmoid   │
                                                                  │ Backward  │
                                                                  │δ1=∂L/∂a1 │
                                                                  │ ⊙a1⊙(1-a1)│
                                                                  └─────┬─────┘
                                                                        │
                                                                        ▼
                                                                       δ1
                                                                     (N,256)
                         ┌───────────────────────┴───────────────────────┐
                         │                                               │
                         ▼                                               ▼
                   ┌───────────┐                                   ┌───────────┐
                   │ ∂L/∂W1    │                                   │ ∂L/∂b1    │
                   │= x.T @ δ1 │                                   │= Σδ1      │
                   └─────┬─────┘                                   └─────┬─────┘
                         │                                               │
                         ▼                                               ▼
                    (784,256)                                         (256,)
                         │                                               │
                    [Update W1]                                    [Update b1]
```

```python
def backward_graph_demo(cache, W1, b1, W2, b2, W3, b3, y):
    """역전파 그래프 실행"""
    N = y.shape[0]
    grads = {}
    
    print("\nBackward Pass Execution:")
    print("=" * 60)
    
    # ∂L/∂L = 1 (암묵적)
    print("Starting from L (loss scalar)")
    print("∂L/∂L = 1")
    
    # Softmax + CE backward
    delta3 = (cache['y_hat'] - y) / N
    grads['delta3'] = delta3
    print(f"\nδ3 = (ŷ - y) / N: {delta3.shape}")
    
    # Layer 3 gradients
    grads['dW3'] = cache['a2'].T @ delta3
    grads['db3'] = np.sum(delta3, axis=0)
    grad_a2 = delta3 @ W3.T
    
    print(f"\n[Layer 3]")
    print(f"  ∂L/∂W3 = a2.T @ δ3: {grads['dW3'].shape}")
    print(f"  ∂L/∂b3 = sum(δ3):   {grads['db3'].shape}")
    print(f"  ∂L/∂a2 = δ3 @ W3.T: {grad_a2.shape}")
    
    # Sigmoid backward (Layer 2)
    delta2 = grad_a2 * cache['a2'] * (1 - cache['a2'])
    grads['delta2'] = delta2
    print(f"\n[Sigmoid Backward]")
    print(f"  δ2 = ∂L/∂a2 ⊙ a2 ⊙ (1-a2): {delta2.shape}")
    
    # Layer 2 gradients
    grads['dW2'] = cache['a1'].T @ delta2
    grads['db2'] = np.sum(delta2, axis=0)
    grad_a1 = delta2 @ W2.T
    
    print(f"\n[Layer 2]")
    print(f"  ∂L/∂W2 = a1.T @ δ2: {grads['dW2'].shape}")
    print(f"  ∂L/∂b2 = sum(δ2):   {grads['db2'].shape}")
    print(f"  ∂L/∂a1 = δ2 @ W2.T: {grad_a1.shape}")
    
    # Sigmoid backward (Layer 1)
    delta1 = grad_a1 * cache['a1'] * (1 - cache['a1'])
    grads['delta1'] = delta1
    print(f"\n[Sigmoid Backward]")
    print(f"  δ1 = ∂L/∂a1 ⊙ a1 ⊙ (1-a1): {delta1.shape}")
    
    # Layer 1 gradients
    grads['dW1'] = cache['x'].T @ delta1
    grads['db1'] = np.sum(delta1, axis=0)
    
    print(f"\n[Layer 1]")
    print(f"  ∂L/∂W1 = x.T @ δ1: {grads['dW1'].shape}")
    print(f"  ∂L/∂b1 = sum(δ1):  {grads['db1'].shape}")
    
    return grads

grads = backward_graph_demo(cache, W1, b1, W2, b2, W3, b3, y)
```

```
Backward Pass Execution:
============================================================
Starting from L (loss scalar)
∂L/∂L = 1

δ3 = (ŷ - y) / N: (4, 10)

[Layer 3]
  ∂L/∂W3 = a2.T @ δ3: (128, 10)
  ∂L/∂b3 = sum(δ3):   (10,)
  ∂L/∂a2 = δ3 @ W3.T: (4, 128)

[Sigmoid Backward]
  δ2 = ∂L/∂a2 ⊙ a2 ⊙ (1-a2): (4, 128)

[Layer 2]
  ∂L/∂W2 = a1.T @ δ2: (256, 128)
  ∂L/∂b2 = sum(δ2):   (128,)
  ∂L/∂a1 = δ2 @ W2.T: (4, 256)

[Sigmoid Backward]
  δ1 = ∂L/∂a1 ⊙ a1 ⊙ (1-a1): (4, 256)

[Layer 1]
  ∂L/∂W1 = x.T @ δ1: (784, 256)
  ∂L/∂b1 = sum(δ1):  (256,)
```

### 3.5.4 Single Layer Gradient Flow

단일 레이어(Linear + Activation)의 그래디언트 흐름을 상세히 살펴봅니다.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE LAYER: Linear + Activation                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                              FORWARD
   ─────────────────────────────────────────────────────────────────────►

        a_in                  z                   a_out
       (N,Din)             (N,Dout)             (N,Dout)
          │                   │                    │
          │    W (Din,Dout)   │                    │
          │        │          │                    │
          ▼        ▼          ▼                    ▼
     ┌─────────────────┐  ┌─────────┐         ┌─────────┐
     │     MatMul      │  │   Add   │         │  σ(z)   │
     │   a_in @ W      │─►│   + b   │────────►│Activation│
     └─────────────────┘  └─────────┘         └─────────┘
                               ▲
                               │
                           b (Dout,)


                              BACKWARD
   ◄─────────────────────────────────────────────────────────────────────

        ∂L/∂a_in              δ                ∂L/∂a_out
        (N,Din)            (N,Dout)            (N,Dout)
          ▲                   ▲                    │
          │                   │                    │
          │                   │                    ▼
     ┌─────────────────┐      │              ┌─────────┐
     │   δ @ W.T       │      │              │  σ'(z)  │
     │  (N,Dout)@      │◄─────┤◄─────────────│ ⊙ ∂L/∂a │
     │  (Dout,Din)     │      │              └─────────┘
     └─────────────────┘      │
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
     ┌─────────┐        ┌─────────┐         ┌─────────┐
     │ ∂L/∂W   │        │ ∂L/∂b   │         │Propagate│
     │a_in.T@δ │        │ Σ δ     │         │to prev  │
     └─────────┘        └─────────┘         └─────────┘
      (Din,Dout)         (Dout,)
```

```python
class LinearLayer:
    """단일 Linear 레이어의 순전파/역전파"""
    
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)
        self.cache = {}
        self.grads = {}
    
    def forward(self, a_in):
        """순전파: z = a_in @ W + b"""
        self.cache['a_in'] = a_in
        z = a_in @ self.W + self.b
        return z
    
    def backward(self, delta):
        """역전파: 그래디언트 계산 및 전파"""
        a_in = self.cache['a_in']
        
        # 파라미터 그래디언트
        self.grads['dW'] = a_in.T @ delta
        self.grads['db'] = np.sum(delta, axis=0)
        
        # 이전 레이어로 전파
        grad_a_in = delta @ self.W.T
        
        return grad_a_in


class SigmoidLayer:
    """Sigmoid 활성화 레이어"""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, z):
        """순전파: a = sigmoid(z)"""
        a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        self.cache['a'] = a
        return a
    
    def backward(self, grad_a):
        """역전파: δ = grad_a ⊙ a ⊙ (1-a)"""
        a = self.cache['a']
        delta = grad_a * a * (1 - a)
        return delta


# 단일 레이어 테스트
print("Single Layer Gradient Flow Demo:")
print("=" * 60)

np.random.seed(42)
N, D_in, D_out = 4, 10, 5

# 레이어 생성
linear = LinearLayer(D_in, D_out)
sigmoid = SigmoidLayer()

# 입력
a_in = np.random.randn(N, D_in)

# Forward
print("\n[Forward Pass]")
print(f"a_in: {a_in.shape}")

z = linear.forward(a_in)
print(f"z = a_in @ W + b: {z.shape}")

a_out = sigmoid.forward(z)
print(f"a_out = sigmoid(z): {a_out.shape}")

# Backward (가상의 상위 그래디언트)
grad_a_out = np.random.randn(N, D_out)

print("\n[Backward Pass]")
print(f"∂L/∂a_out (from upper layer): {grad_a_out.shape}")

delta = sigmoid.backward(grad_a_out)
print(f"δ = ∂L/∂a_out ⊙ a ⊙ (1-a): {delta.shape}")

grad_a_in = linear.backward(delta)
print(f"∂L/∂a_in = δ @ W.T: {grad_a_in.shape}")
print(f"∂L/∂W = a_in.T @ δ: {linear.grads['dW'].shape}")
print(f"∂L/∂b = sum(δ): {linear.grads['db'].shape}")

# Shape 검증
print("\n[Shape Verification]")
print(f"grad_a_in.shape == a_in.shape: {grad_a_in.shape == a_in.shape}")
print(f"dW.shape == W.shape: {linear.grads['dW'].shape == linear.W.shape}")
print(f"db.shape == b.shape: {linear.grads['db'].shape == linear.b.shape}")
```

```
Single Layer Gradient Flow Demo:
============================================================

[Forward Pass]
a_in: (4, 10)
z = a_in @ W + b: (4, 5)
a_out = sigmoid(z): (4, 5)

[Backward Pass]
∂L/∂a_out (from upper layer): (4, 5)
δ = ∂L/∂a_out ⊙ a ⊙ (1-a): (4, 5)
∂L/∂a_in = δ @ W.T: (4, 10)
∂L/∂W = a_in.T @ δ: (10, 5)
∂L/∂b = sum(δ): (5,)

[Shape Verification]
grad_a_in.shape == a_in.shape: True
dW.shape == W.shape: True
db.shape == b.shape: True
```

### 3.5.5 Gradient Flow Patterns

역전파에서 그래디언트가 흐르는 주요 패턴들입니다.

**패턴 1: 덧셈 노드 (Add)**

덧셈 노드는 그래디언트를 그대로 분배합니다.

```
Forward:  c = a + b
Backward: ∂L/∂a = ∂L/∂c × 1 = ∂L/∂c
          ∂L/∂b = ∂L/∂c × 1 = ∂L/∂c
```

**패턴 2: 곱셈 노드 (Multiply)**

곱셈 노드는 그래디언트를 교차하여 전달합니다.

```
Forward:  c = a × b
Backward: ∂L/∂a = ∂L/∂c × b
          ∂L/∂b = ∂L/∂c × a
```

**패턴 3: 행렬 곱 노드 (MatMul)**

행렬 곱은 전치를 사용하여 그래디언트를 전파합니다.

```
Forward:  C = A @ B
Backward: ∂L/∂A = ∂L/∂C @ B.T
          ∂L/∂B = A.T @ ∂L/∂C
```

**패턴 4: 분기 노드 (Fork)**

하나의 값이 여러 경로로 사용되면 그래디언트가 합산됩니다.

```
Forward:  a가 b와 c 모두에 사용됨
Backward: ∂L/∂a = ∂L/∂b + ∂L/∂c
```

```python
def gradient_flow_patterns():
    """그래디언트 흐름 패턴 시연"""
    
    print("Gradient Flow Patterns:")
    print("=" * 60)
    
    # Pattern 1: Add
    print("\n[Pattern 1: Addition]")
    print("Forward:  c = a + b")
    a, b = 3.0, 2.0
    c = a + b
    grad_c = 1.0  # ∂L/∂c
    grad_a = grad_c * 1  # ∂c/∂a = 1
    grad_b = grad_c * 1  # ∂c/∂b = 1
    print(f"Backward: ∂L/∂a = {grad_a}, ∂L/∂b = {grad_b}")
    print("→ Gradient is distributed equally")
    
    # Pattern 2: Multiply
    print("\n[Pattern 2: Multiplication]")
    print("Forward:  c = a × b")
    a, b = 3.0, 2.0
    c = a * b
    grad_c = 1.0
    grad_a = grad_c * b  # ∂c/∂a = b
    grad_b = grad_c * a  # ∂c/∂b = a
    print(f"Backward: ∂L/∂a = ∂L/∂c × b = {grad_a}")
    print(f"          ∂L/∂b = ∂L/∂c × a = {grad_b}")
    print("→ Gradient is scaled by the other input")
    
    # Pattern 3: MatMul
    print("\n[Pattern 3: Matrix Multiplication]")
    print("Forward:  C = A @ B")
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    C = A @ B
    grad_C = np.ones_like(C)
    grad_A = grad_C @ B.T
    grad_B = A.T @ grad_C
    print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}")
    print(f"Backward: ∂L/∂A = ∂L/∂C @ B.T → {grad_A.shape}")
    print(f"          ∂L/∂B = A.T @ ∂L/∂C → {grad_B.shape}")
    print("→ Transpose is used to match dimensions")
    
    # Pattern 4: Fork (분기)
    print("\n[Pattern 4: Fork (Branch)]")
    print("Forward:  a is used in both b and c")
    a = 3.0
    b = a * 2    # 경로 1
    c = a + 5    # 경로 2
    L = b + c    # 두 경로 합침
    
    # Backward
    grad_L = 1.0
    grad_b = grad_L  # ∂L/∂b = 1
    grad_c = grad_L  # ∂L/∂c = 1
    grad_a_from_b = grad_b * 2  # 경로 1에서 온 그래디언트
    grad_a_from_c = grad_c * 1  # 경로 2에서 온 그래디언트
    grad_a = grad_a_from_b + grad_a_from_c  # 합산
    
    print(f"a = {a}, b = a×2 = {b}, c = a+5 = {c}, L = b+c = {L}")
    print(f"Backward: ∂L/∂a = (∂L/∂b × ∂b/∂a) + (∂L/∂c × ∂c/∂a)")
    print(f"        = {grad_a_from_b} + {grad_a_from_c} = {grad_a}")
    print("→ Gradients from multiple paths are summed")
    
    print("\n" + "=" * 60)

gradient_flow_patterns()
```

```
Gradient Flow Patterns:
============================================================

[Pattern 1: Addition]
Forward:  c = a + b
Backward: ∂L/∂a = 1.0, ∂L/∂b = 1.0
→ Gradient is distributed equally

[Pattern 2: Multiplication]
Forward:  c = a × b
Backward: ∂L/∂a = ∂L/∂c × b = 2.0
          ∂L/∂b = ∂L/∂c × a = 3.0
→ Gradient is scaled by the other input

[Pattern 3: Matrix Multiplication]
Forward:  C = A @ B
A: (3, 4), B: (4, 5), C: (3, 5)
Backward: ∂L/∂A = ∂L/∂C @ B.T → (3, 4)
          ∂L/∂B = A.T @ ∂L/∂C → (4, 5)
→ Transpose is used to match dimensions

[Pattern 4: Fork (Branch)]
Forward:  a is used in both b and c
a = 3.0, b = a×2 = 6.0, c = a+5 = 8.0, L = b+c = 14.0
Backward: ∂L/∂a = (∂L/∂b × ∂b/∂a) + (∂L/∂c × ∂c/∂a)
        = 2.0 + 1.0 = 3.0
→ Gradients from multiple paths are summed

============================================================
```

### 3.5.6 Code-Graph Correspondence

코드와 계산 그래프의 대응 관계를 정리합니다.

```python
def full_forward_backward_with_comments():
    """코드와 계산 그래프의 대응"""
    np.random.seed(42)
    
    # 설정
    N, D, H, K = 4, 8, 5, 3
    
    # 파라미터
    W1 = np.random.randn(D, H) * 0.1
    b1 = np.zeros(H)
    W2 = np.random.randn(H, K) * 0.1
    b2 = np.zeros(K)
    
    # 입력
    x = np.random.randn(N, D)
    y = np.eye(K)[np.array([0, 1, 2, 0])]
    
    print("Code-Graph Correspondence:")
    print("=" * 70)
    
    # ==================== FORWARD ====================
    print("\n" + "─" * 70)
    print("FORWARD PASS")
    print("─" * 70)
    
    # [Node: MatMul] z1 = x @ W1
    z1_matmul = x @ W1
    print(f"[MatMul] z1_mm = x @ W1        | {x.shape} @ {W1.shape} = {z1_matmul.shape}")
    
    # [Node: Add] z1 = z1_matmul + b1
    z1 = z1_matmul + b1
    print(f"[Add]    z1 = z1_mm + b1       | {z1_matmul.shape} + {b1.shape} = {z1.shape}")
    
    # [Node: Sigmoid] a1 = sigmoid(z1)
    a1 = 1 / (1 + np.exp(-z1))
    print(f"[Sigmoid] a1 = σ(z1)           | {z1.shape} → {a1.shape}")
    
    # [Node: MatMul] z2 = a1 @ W2
    z2_matmul = a1 @ W2
    print(f"[MatMul] z2_mm = a1 @ W2       | {a1.shape} @ {W2.shape} = {z2_matmul.shape}")
    
    # [Node: Add] z2 = z2_matmul + b2
    z2 = z2_matmul + b2
    print(f"[Add]    z2 = z2_mm + b2       | {z2_matmul.shape} + {b2.shape} = {z2.shape}")
    
    # [Node: Softmax] y_hat = softmax(z2)
    e_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    y_hat = e_z / np.sum(e_z, axis=1, keepdims=True)
    print(f"[Softmax] ŷ = softmax(z2)      | {z2.shape} → {y_hat.shape}")
    
    # [Node: CE Loss] L = cross_entropy(y_hat, y)
    L = -np.mean(np.sum(y * np.log(y_hat + 1e-8), axis=1))
    print(f"[CE Loss] L = CE(ŷ, y)         | {y_hat.shape}, {y.shape} → scalar({L:.4f})")
    
    # ==================== BACKWARD ====================
    print("\n" + "─" * 70)
    print("BACKWARD PASS")
    print("─" * 70)
    
    # [Node: Softmax+CE Backward]
    delta2 = (y_hat - y) / N
    print(f"[Softmax+CE] δ2 = (ŷ-y)/N     | {y_hat.shape} - {y.shape} = {delta2.shape}")
    
    # [Node: Add Backward] gradient flows through unchanged
    # (분배 법칙: 둘 다 delta2를 받음)
    
    # [Node: MatMul Backward for W2]
    dW2 = a1.T @ delta2
    print(f"[MatMul←] dW2 = a1.T @ δ2      | {a1.T.shape} @ {delta2.shape} = {dW2.shape}")
    
    # [Node: Add Backward for b2]
    db2 = np.sum(delta2, axis=0)
    print(f"[Add←]    db2 = sum(δ2)        | sum({delta2.shape}) = {db2.shape}")
    
    # [Node: MatMul Backward - propagate to a1]
    da1 = delta2 @ W2.T
    print(f"[MatMul←] da1 = δ2 @ W2.T      | {delta2.shape} @ {W2.T.shape} = {da1.shape}")
    
    # [Node: Sigmoid Backward]
    delta1 = da1 * a1 * (1 - a1)
    print(f"[Sigmoid←] δ1 = da1⊙a1⊙(1-a1) | {da1.shape} ⊙ {a1.shape} = {delta1.shape}")
    
    # [Node: MatMul Backward for W1]
    dW1 = x.T @ delta1
    print(f"[MatMul←] dW1 = x.T @ δ1       | {x.T.shape} @ {delta1.shape} = {dW1.shape}")
    
    # [Node: Add Backward for b1]
    db1 = np.sum(delta1, axis=0)
    print(f"[Add←]    db1 = sum(δ1)        | sum({delta1.shape}) = {db1.shape}")
    
    print("\n" + "=" * 70)

full_forward_backward_with_comments()
```

```
Code-Graph Correspondence:
======================================================================

──────────────────────────────────────────────────────────────────────
FORWARD PASS
──────────────────────────────────────────────────────────────────────
[MatMul] z1_mm = x @ W1        | (4, 8) @ (8, 5) = (4, 5)
[Add]    z1 = z1_mm + b1       | (4, 5) + (5,) = (4, 5)
[Sigmoid] a1 = σ(z1)           | (4, 5) → (4, 5)
[MatMul] z2_mm = a1 @ W2       | (4, 5) @ (5, 3) = (4, 3)
[Add]    z2 = z2_mm + b2       | (4, 3) + (3,) = (4, 3)
[Softmax] ŷ = softmax(z2)      | (4, 3) → (4, 3)
[CE Loss] L = CE(ŷ, y)         | (4, 3), (4, 3) → scalar(1.1349)

──────────────────────────────────────────────────────────────────────
BACKWARD PASS
──────────────────────────────────────────────────────────────────────
[Softmax+CE] δ2 = (ŷ-y)/N     | (4, 3) - (4, 3) = (4, 3)
[MatMul←] dW2 = a1.T @ δ2      | (5, 4) @ (4, 3) = (5, 3)
[Add←]    db2 = sum(δ2)        | sum((4, 3)) = (3,)
[MatMul←] da1 = δ2 @ W2.T      | (4, 3) @ (3, 5) = (4, 5)
[Sigmoid←] δ1 = da1⊙a1⊙(1-a1) | (4, 5) ⊙ (4, 5) = (4, 5)
[MatMul←] dW1 = x.T @ δ1       | (8, 4) @ (4, 5) = (8, 5)
[Add←]    db1 = sum(δ1)        | sum((4, 5)) = (5,)

======================================================================
```

### 3.5.7 Summary

| 노드 타입 | Forward | Backward |
|-----------|---------|----------|
| Add | $c = a + b$ | $\nabla a = \nabla c$, $\nabla b = \nabla c$ |
| Multiply | $c = a \times b$ | $\nabla a = \nabla c \times b$, $\nabla b = \nabla c \times a$ |
| MatMul | $C = A @ B$ | $\nabla A = \nabla C @ B^T$, $\nabla B = A^T @ \nabla C$ |
| Sigmoid | $a = \sigma(z)$ | $\nabla z = \nabla a \odot a \odot (1-a)$ |
| Softmax+CE | $L = \text{CE}(\text{softmax}(z), y)$ | $\nabla z = (\hat{y} - y) / N$ |
| Sum | $c = \sum a$ | $\nabla a = \nabla c \cdot \mathbf{1}$ |
| Fork | $a$ 가 여러 곳에 사용 | $\nabla a = \sum_i \nabla a_i$ |

**핵심 포인트:**

1. 계산 그래프는 순전파의 연산을 노드와 엣지로 표현
2. 역전파는 그래프를 역순으로 순회하며 그래디언트 계산
3. 각 노드는 로컬 그래디언트와 상위 그래디언트를 곱함 (연쇄 법칙)
4. 분기점에서는 그래디언트가 합산됨
5. 중간값(캐시)은 순전파에서 저장하고 역전파에서 사용

다음 챕터에서는 MLP의 구조와 순전파/역전파 구현을 다룹니다.
