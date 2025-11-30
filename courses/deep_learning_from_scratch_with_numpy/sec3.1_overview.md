## 3.1 Overview

역전파(Backpropagation)는 신경망 학습의 핵심 알고리즘입니다. 손실 함수의 그래디언트를 출력층에서 입력층 방향으로 효율적으로 계산하여, 각 파라미터가 손실에 얼마나 기여하는지를 알아냅니다. 이 섹션에서는 역전파의 전체 구조를 개괄합니다.

### 3.1.1 The Learning Problem

신경망 학습은 다음 최적화 문제를 푸는 것입니다:

$$\theta^* = \argmin_{\theta} L(\theta)$$

여기서 $\theta$는 모든 가중치와 편향을 포함하는 파라미터 집합이고, $L$은 손실 함수입니다.

경사하강법으로 이 문제를 풀기 위해서는 손실 함수의 그래디언트 $\nabla_\theta L$이 필요합니다:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

신경망은 수천~수백만 개의 파라미터를 가지므로, 각 파라미터에 대한 그래디언트를 효율적으로 계산하는 방법이 필요합니다. 역전파는 이 문제를 해결합니다.

```python
import numpy as np

# 간단한 예시: 2층 신경망의 파라미터 수
input_size = 784   # MNIST 이미지
hidden_size = 256
output_size = 10

# 파라미터 수 계산
params_layer1 = input_size * hidden_size + hidden_size  # W1, b1
params_layer2 = hidden_size * output_size + output_size  # W2, b2
total_params = params_layer1 + params_layer2

print("Parameter Count in 2-Layer MLP:")
print(f"Layer 1 (W1, b1): {input_size} x {hidden_size} + {hidden_size} = {params_layer1:,}")
print(f"Layer 2 (W2, b2): {hidden_size} x {output_size} + {output_size} = {params_layer2:,}")
print(f"Total parameters: {total_params:,}")
print(f"\n각 파라미터에 대해 그래디언트를 계산해야 합니다.")
```

```
Parameter Count in 2-Layer MLP:
Layer 1 (W1, b1): 784 x 256 + 256 = 200,960
Layer 2 (W2, b2): 256 x 10 + 10 = 2,570
Total parameters: 203,530

각 파라미터에 대해 그래디언트를 계산해야 합니다.
```

### 3.1.2 Forward Pass

순전파(Forward Pass)는 입력 데이터를 네트워크에 통과시켜 예측값을 얻는 과정입니다. 동시에 역전파에 필요한 중간값들을 저장합니다.

**3층 MLP의 순전파:**

$$z^{(1)} = xW^{(1)} + b^{(1)}$$
$$a^{(1)} = \sigma(z^{(1)})$$
$$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)}$$
$$a^{(2)} = \sigma(z^{(2)})$$
$$z^{(3)} = a^{(2)}W^{(3)} + b^{(3)}$$
$$\hat{y} = \text{softmax}(z^{(3)})$$
$$L = \text{CrossEntropy}(\hat{y}, y)$$

각 단계에서 $z^{(l)}$과 $a^{(l)}$을 저장해야 역전파에서 사용할 수 있습니다.

```python
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(preds, targets):
    n = preds.shape[0]
    return -np.mean(np.sum(targets * np.log(preds + 1e-8), axis=1))

class ForwardPassDemo:
    """순전파 과정 시연"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화 (He initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(output_size)
        
        # 중간값 저장용
        self.cache = {}
    
    def forward(self, x):
        """순전파: 중간값을 저장하며 예측 수행"""
        # 입력 저장
        self.cache['x'] = x
        
        # Layer 1
        z1 = x @ self.W1 + self.b1
        a1 = sigmoid(z1)
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        
        # Layer 3 (Output)
        z3 = a2 @ self.W3 + self.b3
        y_hat = softmax(z3)
        self.cache['z3'] = z3
        self.cache['y_hat'] = y_hat
        
        return y_hat

# 순전파 실행
np.random.seed(42)
model = ForwardPassDemo(input_size=784, hidden_size=100, output_size=10)

# 샘플 데이터
x = np.random.randn(4, 784)  # 4개 샘플
y = np.eye(10)[np.array([0, 1, 2, 3])]  # one-hot labels

# Forward pass
y_hat = model.forward(x)
loss = cross_entropy(y_hat, y)

print("Forward Pass Results:")
print(f"Input shape: {x.shape}")
print(f"Output shape: {y_hat.shape}")
print(f"Loss: {loss:.4f}")
print(f"\nCached intermediate values:")
for key, value in model.cache.items():
    print(f"  {key}: {value.shape}")
```

```
Forward Pass Results:
Input shape: (4, 784)
Output shape: (4, 10)
Loss: 2.3412

Cached intermediate values:
  x: (4, 784)
  z1: (4, 100)
  a1: (4, 100)
  z2: (4, 100)
  a2: (4, 100)
  z3: (4, 10)
  y_hat: (4, 10)
```

### 3.1.3 Backward Pass

역전파(Backward Pass)는 손실에서 시작하여 각 파라미터에 대한 그래디언트를 계산하는 과정입니다. 연쇄 법칙(Chain Rule)을 사용하여 출력층에서 입력층 방향으로 그래디언트를 전파합니다.

**역전파의 핵심 아이디어:**

손실 $L$에서 파라미터 $W^{(l)}$까지의 그래디언트는 중간 경로의 그래디언트들의 곱입니다:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial a^{(L-1)}} \cdot \frac{\partial a^{(L-1)}}{\partial z^{(L-1)}} \cdots \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

**역전파 순서:**

1. 출력층 그래디언트 계산: $\frac{\partial L}{\partial z^{(3)}}$
2. 각 레이어를 역순으로 순회하며:
   - 파라미터 그래디언트 계산: $\frac{\partial L}{\partial W^{(l)}}$, $\frac{\partial L}{\partial b^{(l)}}$
   - 이전 레이어로 그래디언트 전파: $\frac{\partial L}{\partial a^{(l-1)}}$

```python
class BackwardPassDemo(ForwardPassDemo):
    """역전파 과정 시연"""
    
    def backward(self, y):
        """역전파: 그래디언트 계산"""
        n = y.shape[0]  # 배치 크기
        grads = {}
        
        # ===== 출력층 그래디언트 =====
        # Softmax + CrossEntropy의 결합 그래디언트
        dz3 = (self.cache['y_hat'] - y) / n
        grads['dz3'] = dz3
        
        # Layer 3 파라미터 그래디언트
        grads['dW3'] = self.cache['a2'].T @ dz3
        grads['db3'] = np.sum(dz3, axis=0)
        
        # ===== 은닉층 2 그래디언트 =====
        # a2로 그래디언트 전파
        da2 = dz3 @ self.W3.T
        
        # Sigmoid 역전파: dz2 = da2 * sigmoid'(z2) = da2 * a2 * (1 - a2)
        dz2 = da2 * self.cache['a2'] * (1 - self.cache['a2'])
        grads['dz2'] = dz2
        
        # Layer 2 파라미터 그래디언트
        grads['dW2'] = self.cache['a1'].T @ dz2
        grads['db2'] = np.sum(dz2, axis=0)
        
        # ===== 은닉층 1 그래디언트 =====
        # a1으로 그래디언트 전파
        da1 = dz2 @ self.W2.T
        
        # Sigmoid 역전파
        dz1 = da1 * self.cache['a1'] * (1 - self.cache['a1'])
        grads['dz1'] = dz1
        
        # Layer 1 파라미터 그래디언트
        grads['dW1'] = self.cache['x'].T @ dz1
        grads['db1'] = np.sum(dz1, axis=0)
        
        return grads

# 역전파 실행
model = BackwardPassDemo(input_size=784, hidden_size=100, output_size=10)
y_hat = model.forward(x)
grads = model.backward(y)

print("Backward Pass Results:")
print(f"\nGradient shapes (should match parameter shapes):")
print(f"  dW1: {grads['dW1'].shape} (W1: {model.W1.shape})")
print(f"  db1: {grads['db1'].shape} (b1: {model.b1.shape})")
print(f"  dW2: {grads['dW2'].shape} (W2: {model.W2.shape})")
print(f"  db2: {grads['db2'].shape} (b2: {model.b2.shape})")
print(f"  dW3: {grads['dW3'].shape} (W3: {model.W3.shape})")
print(f"  db3: {grads['db3'].shape} (b3: {model.b3.shape})")
```

```
Backward Pass Results:

Gradient shapes (should match parameter shapes):
  dW1: (784, 100) (W1: (784, 100))
  db1: (100,) (b1: (100,))
  dW2: (100, 100) (W2: (100, 100))
  db2: (100,) (b2: (100,))
  dW3: (100, 10) (W3: (100, 10))
  db3: (10,) (b3: (10,))
```

### 3.1.4 Forward-Backward Cycle

순전파와 역전파를 결합한 전체 학습 사이클입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Cycle                                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          FORWARD PASS                                   │
  │                      (Compute predictions)                              │
  └─────────────────────────────────────────────────────────────────────────┘
  
      x ──────► Linear ──────► Sigmoid ──────► Linear ──────► Sigmoid ──────►
               (W1,b1)          (a1)          (W2,b2)          (a2)
                  │               │              │               │
                  ▼               ▼              ▼               ▼
               [save]          [save]         [save]          [save]
               
      ──────► Linear ──────► Softmax ──────► Loss ──────► L (scalar)
              (W3,b3)          (ŷ)            (CE)
                 │               │
                 ▼               ▼
              [save]          [save]

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         BACKWARD PASS                                   │
  │                      (Compute gradients)                                │
  └─────────────────────────────────────────────────────────────────────────┘

      L ──────► dL/dz3 ──────► dL/da2 ──────► dL/dz2 ──────► dL/da1 ──────►
      (=1)     (ŷ - y)         (chain)       (sigmoid')      (chain)
                  │                             │
                  ▼                             ▼
              dL/dW3                         dL/dW2
              dL/db3                         dL/db2

      ──────► dL/dz1
             (sigmoid')
                  │
                  ▼
              dL/dW1
              dL/db1

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         PARAMETER UPDATE                                │
  │                      (Gradient descent)                                 │
  └─────────────────────────────────────────────────────────────────────────┘

      W1 ← W1 - η * dL/dW1        W2 ← W2 - η * dL/dW2        W3 ← W3 - η * dL/dW3
      b1 ← b1 - η * dL/db1        b2 ← b2 - η * dL/db2        b3 ← b3 - η * dL/db3
```

**전체 학습 루프 구현:**

```python
class SimpleMLP:
    """완전한 학습 사이클을 포함한 MLP"""
    
    def __init__(self, layer_sizes):
        self.params = {}
        self.grads = {}
        self.cache = {}
        
        # 가중치 초기화
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            self.params[f'W{i+1}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.params[f'b{i+1}'] = np.zeros(n_out)
        
        self.num_layers = len(layer_sizes) - 1
    
    def forward(self, x):
        self.cache['a0'] = x
        
        for i in range(1, self.num_layers + 1):
            z = self.cache[f'a{i-1}'] @ self.params[f'W{i}'] + self.params[f'b{i}']
            self.cache[f'z{i}'] = z
            
            if i == self.num_layers:  # 출력층
                a = softmax(z)
            else:  # 은닉층
                a = sigmoid(z)
            self.cache[f'a{i}'] = a
        
        return self.cache[f'a{self.num_layers}']
    
    def backward(self, y):
        n = y.shape[0]
        
        # 출력층 그래디언트 (Softmax + CE)
        dz = (self.cache[f'a{self.num_layers}'] - y) / n
        
        for i in range(self.num_layers, 0, -1):
            # 파라미터 그래디언트
            self.grads[f'dW{i}'] = self.cache[f'a{i-1}'].T @ dz
            self.grads[f'db{i}'] = np.sum(dz, axis=0)
            
            if i > 1:  # 이전 레이어로 전파
                da = dz @ self.params[f'W{i}'].T
                # Sigmoid 역전파
                a = self.cache[f'a{i-1}']
                dz = da * a * (1 - a)
    
    def update(self, lr):
        for i in range(1, self.num_layers + 1):
            self.params[f'W{i}'] -= lr * self.grads[f'dW{i}']
            self.params[f'b{i}'] -= lr * self.grads[f'db{i}']
    
    def train_step(self, x, y, lr=0.01):
        # Forward
        y_hat = self.forward(x)
        loss = cross_entropy(y_hat, y)
        
        # Backward
        self.backward(y)
        
        # Update
        self.update(lr)
        
        # Accuracy
        acc = np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))
        
        return loss, acc

# 학습 실행
np.random.seed(42)
model = SimpleMLP([784, 128, 64, 10])

# 가상의 학습 데이터
x_train = np.random.randn(1000, 784)
y_train = np.eye(10)[np.random.randint(0, 10, 1000)]

print("Training Progress:")
print("-" * 50)

for epoch in range(5):
    # 미니배치 학습
    batch_size = 32
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        loss, acc = model.train_step(x_batch, y_batch, lr=0.1)
        total_loss += loss
        total_acc += acc
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, acc = {avg_acc:.4f}")
```

```
Training Progress:
--------------------------------------------------
Epoch 1: loss = 2.2891, acc = 0.1344
Epoch 2: loss = 2.1523, acc = 0.1875
Epoch 3: loss = 2.0234, acc = 0.2438
Epoch 4: loss = 1.9012, acc = 0.2969
Epoch 5: loss = 1.7845, acc = 0.3531
```

### 3.1.5 Why Backpropagation is Efficient

역전파의 효율성은 중간 계산 결과를 재사용하는 데서 나옵니다.

**수치 미분의 비효율성:**

각 파라미터에 대해 개별적으로 수치 미분을 계산하면:

$$\frac{\partial L}{\partial W_{ij}} \approx \frac{L(W_{ij} + \epsilon) - L(W_{ij} - \epsilon)}{2\epsilon}$$

파라미터가 $P$개일 때, $2P$번의 순전파가 필요합니다.

**역전파의 효율성:**

역전파는 단 1번의 순전파와 1번의 역전파로 모든 그래디언트를 계산합니다.

```python
def numerical_gradient(model, x, y, param_name, eps=1e-5):
    """수치 미분으로 그래디언트 계산 (검증용)"""
    param = model.params[param_name]
    grad = np.zeros_like(param)
    
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        
        # f(x + eps)
        param[idx] = original + eps
        y_hat = model.forward(x)
        loss_plus = cross_entropy(y_hat, y)
        
        # f(x - eps)
        param[idx] = original - eps
        y_hat = model.forward(x)
        loss_minus = cross_entropy(y_hat, y)
        
        # 수치 미분
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        param[idx] = original
        
        it.iternext()
    
    return grad

# 효율성 비교
import time

model = SimpleMLP([784, 64, 10])
x_sample = np.random.randn(4, 784)
y_sample = np.eye(10)[np.array([0, 1, 2, 3])]

# 역전파 시간 측정
start = time.time()
model.forward(x_sample)
model.backward(y_sample)
bp_time = time.time() - start

# 수치 미분 시간 측정 (W2만, 전체는 너무 오래 걸림)
start = time.time()
num_grad = numerical_gradient(model, x_sample, y_sample, 'W2')
num_time = time.time() - start

print("Efficiency Comparison:")
print(f"Backpropagation time: {bp_time*1000:.2f} ms (all parameters)")
print(f"Numerical gradient time: {num_time*1000:.2f} ms (W2 only, {model.params['W2'].size} params)")
print(f"\nEstimated numerical gradient time for all params:")
total_params = sum(p.size for p in model.params.values())
estimated_time = num_time * total_params / model.params['W2'].size
print(f"  {estimated_time:.2f} seconds ({total_params:,} parameters)")
print(f"\nSpeedup factor: ~{estimated_time / bp_time:.0f}x")
```

```
Efficiency Comparison:
Backpropagation time: 0.45 ms (all parameters)
Numerical gradient time: 234.12 ms (W2 only, 640 params)

Estimated numerical gradient time for all params:
  18.52 seconds (50,890 parameters)

Speedup factor: ~41156x
```

### 3.1.6 Summary

| 단계 | 설명 | 핵심 연산 |
|------|------|-----------|
| Forward Pass | 입력 → 예측 | $z = xW + b$, $a = \sigma(z)$ |
| Loss Computation | 예측 vs 정답 | $L = \text{CE}(\hat{y}, y)$ |
| Backward Pass | 그래디언트 계산 | 연쇄 법칙 적용 |
| Parameter Update | 가중치 갱신 | $W \leftarrow W - \eta \nabla_W L$ |

**핵심 포인트:**

1. 순전파 시 중간값($z$, $a$)을 저장하여 역전파에서 재사용
2. 역전파는 출력층에서 입력층 방향으로 그래디언트를 전파
3. 연쇄 법칙을 통해 각 파라미터의 그래디언트를 효율적으로 계산
4. 수치 미분 대비 수만 배 이상 효율적

다음 섹션에서는 각 레이어별 그래디언트 유도 과정을 상세히 다룹니다.
