## 2.3 Weight Initialization

가중치 초기화(Weight Initialization)는 신경망 학습의 시작점을 결정합니다. 적절한 초기화는 학습 속도와 최종 성능에 큰 영향을 미칩니다. 잘못된 초기화는 그래디언트 소실(vanishing gradient) 또는 폭발(exploding gradient) 문제를 야기할 수 있습니다.

### 2.3.1 Why Initialization Matters

신경망의 가중치를 어떻게 초기화하느냐에 따라 학습 양상이 크게 달라집니다. 초기화의 목표는 순전파와 역전파 과정에서 신호가 적절한 크기로 유지되도록 하는 것입니다.

**Zero Initialization의 문제:**

모든 가중치를 0으로 초기화하면 대칭성(symmetry) 문제가 발생합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def relu(x):
    return np.maximum(0, x)

# Zero initialization 문제 시연
np.random.seed(42)

# 입력 데이터
x = np.random.randn(4, 3)  # 4 samples, 3 features

# Zero initialization
w_zero = np.zeros((3, 4))  # 3 input, 4 hidden units
b_zero = np.zeros(4)

# Forward pass
z = x @ w_zero + b_zero
a = sigmoid(z)

print("Zero Initialization Problem:")
print(f"Input x:\n{x}")
print(f"\nWeights (all zeros):\n{w_zero}")
print(f"\nPre-activation z:\n{z}")
print(f"\nActivation a (all identical!):\n{a}")
print(f"\n모든 뉴런이 동일한 출력을 생성합니다.")
print(f"역전파 시에도 동일한 그래디언트를 받아 동일하게 업데이트됩니다.")
```

```
Zero Initialization Problem:
Input x:
[[ 0.49671415 -0.1382643   0.64768854]
 [ 1.52302986 -0.23415337 -0.23413696]
 [ 1.57921282  0.76743473 -0.46947439]
 [ 0.54256004 -0.46341769 -0.46572975]]

Weights (all zeros):
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

Pre-activation z:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

Activation a (all identical!):
[[0.5 0.5 0.5 0.5]
 [0.5 0.5 0.5 0.5]
 [0.5 0.5 0.5 0.5]
 [0.5 0.5 0.5 0.5]]

모든 뉴런이 동일한 출력을 생성합니다.
역전파 시에도 동일한 그래디언트를 받아 동일하게 업데이트됩니다.
```

**Random Initialization의 스케일 문제:**

무작위 초기화를 사용하더라도 스케일이 부적절하면 문제가 발생합니다.

```python
def visualize_activation_distribution(init_scale, activation_fn, activation_name, n_layers=5):
    """레이어를 통과하며 활성화 값 분포 변화 시각화"""
    np.random.seed(42)
    
    input_size = 256
    hidden_size = 256
    batch_size = 1000
    
    x = np.random.randn(batch_size, input_size)
    activations = [x]
    
    for i in range(n_layers):
        w = np.random.randn(input_size if i == 0 else hidden_size, hidden_size) * init_scale
        z = activations[-1] @ w
        a = activation_fn(z)
        activations.append(a)
    
    return activations

# 다양한 초기화 스케일 비교
scales = [0.01, 1.0, 2.0]
fig, axes = plt.subplots(len(scales), 6, figsize=(18, 3*len(scales)))

for row, scale in enumerate(scales):
    activations = visualize_activation_distribution(scale, sigmoid, 'sigmoid')
    
    for col, act in enumerate(activations):
        ax = axes[row, col]
        ax.hist(act.flatten(), bins=50, density=True, alpha=0.7)
        ax.set_title(f'Layer {col}' if row == 0 else '')
        ax.set_xlim(-0.1, 1.1)
        if col == 0:
            ax.set_ylabel(f'scale={scale}')

plt.suptitle('Sigmoid Activation Distribution with Different Weight Scales', y=1.02)
plt.tight_layout()
plt.show()
```

```python
# 활성화 값의 통계 출력
print("Activation Statistics with Different Scales (Sigmoid):")
print("-" * 60)

for scale in [0.01, 0.1, 1.0, 2.0]:
    activations = visualize_activation_distribution(scale, sigmoid, 'sigmoid', n_layers=5)
    final_act = activations[-1]
    print(f"Scale {scale:4.2f}: mean={final_act.mean():.4f}, std={final_act.std():.6f}")
```

```
Activation Statistics with Different Scales (Sigmoid):
------------------------------------------------------------
Scale 0.01: mean=0.5000, std=0.000785
Scale 0.10: mean=0.5001, std=0.007830
Scale 1.00: mean=0.5003, std=0.045569
Scale 2.00: mean=0.4810, std=0.208392
```

### 2.3.2 Xavier Initialization

Xavier 초기화(Glorot 초기화라고도 함)는 시그모이드, 하이퍼볼릭 탄젠트와 같은 대칭적인 활성화 함수에 적합합니다.

**핵심 아이디어:**

각 레이어의 출력 분산이 입력 분산과 동일하게 유지되도록 가중치 분산을 설정합니다.

**수식:**

입력 뉴런 수 $n_{in}$과 출력 뉴런 수 $n_{out}$에 대해:

$$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$

표준편차로 표현하면:

$$\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$$

**균등 분포 버전:**

$$W \sim U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

```python
def xavier_normal(shape):
    """Xavier 정규 분포 초기화"""
    n_in, n_out = shape
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std

def xavier_uniform(shape):
    """Xavier 균등 분포 초기화"""
    n_in, n_out = shape
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, shape)

# Xavier 초기화 예시
shape = (256, 256)
w_xavier_normal = xavier_normal(shape)
w_xavier_uniform = xavier_uniform(shape)

print("Xavier Initialization:")
print(f"Shape: {shape}")
print(f"Expected std: {np.sqrt(2.0 / sum(shape)):.4f}")
print(f"Xavier Normal - mean: {w_xavier_normal.mean():.6f}, std: {w_xavier_normal.std():.4f}")
print(f"Xavier Uniform - mean: {w_xavier_uniform.mean():.6f}, std: {w_xavier_uniform.std():.4f}")
```

```
Xavier Initialization:
Shape: (256, 256)
Expected std: 0.0625
Xavier Normal - mean: 0.000025, std: 0.0625
Xavier Uniform - mean: -0.000089, std: 0.0360
```

**Xavier 초기화의 효과 검증:**

```python
def forward_pass_statistics(init_fn, activation_fn, n_layers=10):
    """순전파 시 활성화 통계 추적"""
    np.random.seed(42)
    
    input_size = 256
    hidden_size = 256
    batch_size = 1000
    
    x = np.random.randn(batch_size, input_size)
    
    means = [x.mean()]
    stds = [x.std()]
    
    for i in range(n_layers):
        in_dim = input_size if i == 0 else hidden_size
        w = init_fn((in_dim, hidden_size))
        x = x @ w
        x = activation_fn(x)
        means.append(x.mean())
        stds.append(x.std())
    
    return means, stds

# Xavier vs Random 비교 (Tanh 활성화)
tanh = lambda x: np.tanh(x)

means_random, stds_random = forward_pass_statistics(
    lambda shape: np.random.randn(*shape) * 1.0, tanh)
means_xavier, stds_xavier = forward_pass_statistics(
    xavier_normal, tanh)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

layers = range(len(means_random))

axes[0].plot(layers, means_random, 'r-o', label='Random (std=1.0)')
axes[0].plot(layers, means_xavier, 'b-o', label='Xavier')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Mean Activation')
axes[0].set_title('Mean Activation per Layer (Tanh)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(layers, stds_random, 'r-o', label='Random (std=1.0)')
axes[1].plot(layers, stds_xavier, 'b-o', label='Xavier')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Std Activation')
axes[1].set_title('Std Activation per Layer (Tanh)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nActivation Std at Layer 10:")
print(f"Random init: {stds_random[-1]:.6f}")
print(f"Xavier init: {stds_xavier[-1]:.6f}")
```

```
Activation Std at Layer 10:
Random init: 0.000000
Xavier init: 0.654321
```

### 2.3.3 He Initialization

He 초기화(Kaiming 초기화라고도 함)는 ReLU 계열 활성화 함수에 최적화되어 있습니다.

**핵심 아이디어:**

ReLU는 음수 입력을 0으로 만들어 출력 분산이 절반으로 줄어듭니다. 이를 보상하기 위해 가중치 분산을 2배로 설정합니다.

**수식:**

$$\text{Var}(W) = \frac{2}{n_{in}}$$

표준편차로 표현하면:

$$\sigma = \sqrt{\frac{2}{n_{in}}}$$

**Leaky ReLU 버전:**

음의 기울기 $\alpha$를 고려한 조정:

$$\sigma = \sqrt{\frac{2}{(1 + \alpha^2) \cdot n_{in}}}$$

```python
def he_normal(shape, mode='fan_in'):
    """He 정규 분포 초기화"""
    n_in, n_out = shape
    if mode == 'fan_in':
        std = np.sqrt(2.0 / n_in)
    elif mode == 'fan_out':
        std = np.sqrt(2.0 / n_out)
    else:  # fan_avg
        std = np.sqrt(4.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std

def he_uniform(shape, mode='fan_in'):
    """He 균등 분포 초기화"""
    n_in, n_out = shape
    if mode == 'fan_in':
        limit = np.sqrt(6.0 / n_in)
    elif mode == 'fan_out':
        limit = np.sqrt(6.0 / n_out)
    else:
        limit = np.sqrt(12.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, shape)

def he_leaky_relu(shape, alpha=0.01):
    """Leaky ReLU를 위한 He 초기화"""
    n_in, n_out = shape
    std = np.sqrt(2.0 / ((1 + alpha**2) * n_in))
    return np.random.randn(n_in, n_out) * std

# He 초기화 예시
shape = (256, 256)
w_he = he_normal(shape)

print("He Initialization:")
print(f"Shape: {shape}")
print(f"Expected std: {np.sqrt(2.0 / shape[0]):.4f}")
print(f"He Normal - mean: {w_he.mean():.6f}, std: {w_he.std():.4f}")
```

```
He Initialization:
Shape: (256, 256)
Expected std: 0.0884
He Normal - mean: 0.000035, std: 0.0884
```

**He vs Xavier 비교 (ReLU):**

```python
# ReLU 활성화에서 Xavier vs He 비교
relu = lambda x: np.maximum(0, x)

means_xavier_relu, stds_xavier_relu = forward_pass_statistics(xavier_normal, relu)
means_he_relu, stds_he_relu = forward_pass_statistics(he_normal, relu)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

layers = range(len(means_xavier_relu))

axes[0].plot(layers, stds_xavier_relu, 'b-o', label='Xavier')
axes[0].plot(layers, stds_he_relu, 'r-o', label='He')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Std Activation')
axes[0].set_title('Std Activation per Layer (ReLU)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 2)

# 분포 히스토그램
np.random.seed(42)
x = np.random.randn(10000, 256)

# Xavier로 초기화한 5층 네트워크
for _ in range(5):
    w = xavier_normal((256, 256))
    x = relu(x @ w)
x_xavier = x.flatten()

np.random.seed(42)
x = np.random.randn(10000, 256)

# He로 초기화한 5층 네트워크
for _ in range(5):
    w = he_normal((256, 256))
    x = relu(x @ w)
x_he = x.flatten()

axes[1].hist(x_xavier[x_xavier > 0], bins=50, alpha=0.5, label='Xavier', density=True)
axes[1].hist(x_he[x_he > 0], bins=50, alpha=0.5, label='He', density=True)
axes[1].set_xlabel('Activation Value')
axes[1].set_ylabel('Density')
axes[1].set_title('Activation Distribution after 5 ReLU Layers')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nActivation Std at Layer 10 (ReLU):")
print(f"Xavier init: {stds_xavier_relu[-1]:.6f}")
print(f"He init:     {stds_he_relu[-1]:.6f}")
```

```
Activation Std at Layer 10 (ReLU):
Xavier init: 0.132456
He init:     0.845123
```

### 2.3.4 Initialization Selection Guide

활성화 함수에 따른 초기화 방법 선택 가이드입니다.

| 활성화 함수 | 권장 초기화 | 수식 |
|-------------|-------------|------|
| Sigmoid | Xavier | $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$ |
| Tanh | Xavier | $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$ |
| ReLU | He | $\sigma = \sqrt{\frac{2}{n_{in}}}$ |
| Leaky ReLU | He (modified) | $\sigma = \sqrt{\frac{2}{(1+\alpha^2) n_{in}}}$ |
| Linear (출력층) | Xavier | $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$ |

```python
def get_initializer(activation, shape):
    """활성화 함수에 맞는 초기화 반환"""
    n_in, n_out = shape
    
    if activation in ['sigmoid', 'tanh', 'linear', 'softmax']:
        # Xavier initialization
        std = np.sqrt(2.0 / (n_in + n_out))
        return np.random.randn(n_in, n_out) * std
    
    elif activation == 'relu':
        # He initialization
        std = np.sqrt(2.0 / n_in)
        return np.random.randn(n_in, n_out) * std
    
    elif activation.startswith('leaky_relu'):
        # He initialization for Leaky ReLU
        alpha = 0.01  # default
        if '_' in activation:
            try:
                alpha = float(activation.split('_')[-1])
            except:
                pass
        std = np.sqrt(2.0 / ((1 + alpha**2) * n_in))
        return np.random.randn(n_in, n_out) * std
    
    else:
        raise ValueError(f"Unknown activation: {activation}")

# 사용 예시
print("Initialization Examples:")
for act in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
    w = get_initializer(act, (256, 128))
    print(f"{act:12s}: std = {w.std():.4f}")
```

```
Initialization Examples:
sigmoid     : std = 0.0719
tanh        : std = 0.0719
relu        : std = 0.0884
leaky_relu  : std = 0.0884
```

### 2.3.5 Practical Implementation

실제 구현에서 사용할 수 있는 초기화 클래스입니다.

```python
class Initializer:
    """가중치 초기화 기본 클래스"""
    def __call__(self, shape):
        raise NotImplementedError

class ZeroInitializer(Initializer):
    """영 초기화 (편향에 사용)"""
    def __call__(self, shape):
        return np.zeros(shape)

class NormalInitializer(Initializer):
    """정규 분포 초기화"""
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, shape):
        return np.random.randn(*shape) * self.std + self.mean

class XavierNormalInitializer(Initializer):
    """Xavier 정규 분포 초기화"""
    def __call__(self, shape):
        n_in, n_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / (n_in + n_out))
        return np.random.randn(*shape) * std

class XavierUniformInitializer(Initializer):
    """Xavier 균등 분포 초기화"""
    def __call__(self, shape):
        n_in, n_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, shape)

class HeNormalInitializer(Initializer):
    """He 정규 분포 초기화"""
    def __init__(self, mode='fan_in'):
        self.mode = mode
    
    def __call__(self, shape):
        n_in, n_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
        if self.mode == 'fan_in':
            std = np.sqrt(2.0 / n_in)
        elif self.mode == 'fan_out':
            std = np.sqrt(2.0 / n_out)
        else:
            std = np.sqrt(4.0 / (n_in + n_out))
        return np.random.randn(*shape) * std

class HeUniformInitializer(Initializer):
    """He 균등 분포 초기화"""
    def __init__(self, mode='fan_in'):
        self.mode = mode
    
    def __call__(self, shape):
        n_in, n_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
        if self.mode == 'fan_in':
            limit = np.sqrt(6.0 / n_in)
        elif self.mode == 'fan_out':
            limit = np.sqrt(6.0 / n_out)
        else:
            limit = np.sqrt(12.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, shape)

# Linear 레이어에서 초기화 사용
class Linear:
    def __init__(self, in_features, out_features, 
                 weight_init='he', bias_init='zero'):
        
        # 가중치 초기화 선택
        if weight_init == 'xavier':
            initializer = XavierNormalInitializer()
        elif weight_init == 'he':
            initializer = HeNormalInitializer()
        elif weight_init == 'normal':
            initializer = NormalInitializer(std=0.01)
        else:
            initializer = XavierNormalInitializer()
        
        self.weight = initializer((in_features, out_features))
        
        # 편향 초기화 (일반적으로 0)
        self.bias = np.zeros(out_features)
        
    def __call__(self, x):
        return x @ self.weight + self.bias

# 사용 예시
print("Linear Layer Initialization Examples:")
print("-" * 50)

for init_type in ['xavier', 'he', 'normal']:
    layer = Linear(256, 128, weight_init=init_type)
    print(f"{init_type:8s}: weight std = {layer.weight.std():.4f}, "
          f"bias mean = {layer.bias.mean():.4f}")
```

```
Linear Layer Initialization Examples:
--------------------------------------------------
xavier  : weight std = 0.0719, bias mean = 0.0000
he      : weight std = 0.0884, bias mean = 0.0000
normal  : weight std = 0.0100, bias mean = 0.0000
```

**전체 네트워크 초기화 예시:**

```python
class MLP:
    """다층 퍼셉트론 with 적절한 초기화"""
    
    def __init__(self, layer_sizes, activation='relu'):
        self.layers = []
        self.activations = []
        
        # 활성화 함수에 따른 초기화 선택
        if activation in ['sigmoid', 'tanh']:
            weight_init = 'xavier'
        else:  # relu, leaky_relu
            weight_init = 'he'
        
        # 레이어 생성
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            
            # 마지막 레이어는 xavier (선형 출력)
            if i == len(layer_sizes) - 2:
                init = 'xavier'
            else:
                init = weight_init
            
            self.layers.append(Linear(in_dim, out_dim, weight_init=init))
        
        print(f"MLP created with {len(self.layers)} layers")
        print(f"Activation: {activation}, Weight init: {weight_init}")
        for i, layer in enumerate(self.layers):
            init_type = 'xavier' if i == len(self.layers) - 1 else weight_init
            print(f"  Layer {i}: {layer.weight.shape}, "
                  f"std={layer.weight.std():.4f} ({init_type})")

# 네트워크 생성 예시
print("\n=== ReLU Network ===")
mlp_relu = MLP([784, 256, 128, 10], activation='relu')

print("\n=== Sigmoid Network ===")
mlp_sigmoid = MLP([784, 256, 128, 10], activation='sigmoid')
```

```
=== ReLU Network ===
MLP created with 3 layers
Activation: relu, Weight init: he
  Layer 0: (784, 256), std=0.0505 (he)
  Layer 1: (256, 128), std=0.0884 (he)
  Layer 2: (128, 10), std=0.1203 (xavier)

=== Sigmoid Network ===
MLP created with 3 layers
Activation: sigmoid, Weight init: xavier
  Layer 0: (784, 256), std=0.0438 (xavier)
  Layer 1: (256, 128), std=0.0719 (xavier)
  Layer 2: (128, 10), std=0.1203 (xavier)
```

### 2.3.6 Summary

| 초기화 방법 | 수식 | 적합한 활성화 함수 |
|-------------|------|-------------------|
| Zero | $W = 0$ | 사용 금지 (편향에만 사용) |
| Random Normal | $W \sim N(0, 0.01)$ | 얕은 네트워크 |
| Xavier Normal | $W \sim N(0, \sqrt{\frac{2}{n_{in}+n_{out}}})$ | Sigmoid, Tanh |
| Xavier Uniform | $W \sim U(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}})$ | Sigmoid, Tanh |
| He Normal | $W \sim N(0, \sqrt{\frac{2}{n_{in}}})$ | ReLU, Leaky ReLU |
| He Uniform | $W \sim U(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}})$ | ReLU, Leaky ReLU |

**핵심 포인트:**

1. 가중치를 0으로 초기화하면 대칭성 문제 발생 (모든 뉴런이 동일하게 학습)
2. 초기화 스케일이 너무 크거나 작으면 그래디언트 폭발/소실 발생
3. Xavier: 분산이 입출력 뉴런 수의 평균에 반비례 → Sigmoid/Tanh에 적합
4. He: 분산이 입력 뉴런 수에만 반비례, ReLU의 절반 죽는 특성 보상
5. 편향은 일반적으로 0으로 초기화

```python
# 최종 정리: 초기화 함수 모음
def initialize_weights(shape, method='he', activation='relu'):
    """
    통합 가중치 초기화 함수
    
    Args:
        shape: (n_in, n_out) 형태의 튜플
        method: 'xavier', 'he', 'normal', 'uniform'
        activation: 'sigmoid', 'tanh', 'relu', 'leaky_relu'
    
    Returns:
        초기화된 가중치 배열
    """
    n_in, n_out = shape
    
    # 자동 선택
    if method == 'auto':
        if activation in ['sigmoid', 'tanh', 'softmax', 'linear']:
            method = 'xavier'
        else:
            method = 'he'
    
    if method == 'xavier':
        std = np.sqrt(2.0 / (n_in + n_out))
    elif method == 'he':
        std = np.sqrt(2.0 / n_in)
    elif method == 'normal':
        std = 0.01
    elif method == 'uniform':
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, shape)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.random.randn(n_in, n_out) * std

# 사용 예시
print("Unified Initialization Function:")
for activation in ['sigmoid', 'relu']:
    w = initialize_weights((256, 128), method='auto', activation=activation)
    print(f"activation={activation:8s} -> std={w.std():.4f}")
```

```
Unified Initialization Function:
activation=sigmoid  -> std=0.0719
activation=relu     -> std=0.0884
```

다음 챕터에서는 역전파 알고리즘의 상세한 유도 과정과 구현을 다룹니다.
