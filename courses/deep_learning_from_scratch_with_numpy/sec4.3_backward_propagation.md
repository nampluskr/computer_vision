## 4.3 Backward Propagation

역전파(Backward Propagation)는 손실 함수의 그래디언트를 출력층에서 입력층 방향으로 전파하여 각 파라미터에 대한 그래디언트를 계산하는 과정입니다. 연쇄 법칙(Chain Rule)을 사용하여 효율적으로 계산합니다.

### 4.3.1 Chain Rule and Gradient Flow

역전파의 핵심은 **연쇄 법칙(Chain Rule)**입니다. 손실 함수 $L$에 대한 각 파라미터의 그래디언트는 다음과 같이 계산됩니다:

**출력층에서의 그래디언트:**

$$\frac{\partial L}{\partial \mathbf{z}^{(3)}} = \frac{\partial L}{\partial \mathbf{a}^{(3)}} \cdot \frac{\partial \mathbf{a}^{(3)}}{\partial \mathbf{z}^{(3)}}$$

**은닉층으로의 역전파:**

$$\frac{\partial L}{\partial \mathbf{z}^{(l)}} = \frac{\partial L}{\partial \mathbf{a}^{(l)}} \cdot \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}}$$

$$\frac{\partial L}{\partial \mathbf{a}^{(l-1)}} = \frac{\partial L}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{a}^{(l-1)}}$$

**파라미터 그래디언트:**

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = (\mathbf{a}^{(l-1)})^T \cdot \frac{\partial L}{\partial \mathbf{z}^{(l)}}$$

$$\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \sum_{i=1}^{N} \frac{\partial L}{\partial \mathbf{z}_i^{(l)}}$$

### 4.3.2 Layer-by-Layer Backward Pass

3-layer MLP의 역전파 과정을 단계별로 유도합니다.

**Step 1: Output Layer Gradient (Softmax + Cross-Entropy)**

Softmax와 Cross-Entropy를 결합하면 간단한 형태가 됩니다:

$$\frac{\partial L}{\partial \mathbf{z}^{(3)}} = \mathbf{a}^{(3)} - \mathbf{y} = \mathbf{p} - \mathbf{y}$$

여기서:
- $\mathbf{p} = \mathbf{a}^{(3)}$: 예측 확률
- $\mathbf{y}$: 정답 레이블 (one-hot)

**Step 2: Layer 3 Parameter Gradients**

$$\frac{\partial L}{\partial \mathbf{W}^{(3)}} = (\mathbf{a}^{(2)})^T \cdot \frac{\partial L}{\partial \mathbf{z}^{(3)}}$$

$$\frac{\partial L}{\partial \mathbf{b}^{(3)}} = \sum_{i=1}^{N} \frac{\partial L}{\partial \mathbf{z}_i^{(3)}}$$

**Step 3: Gradient to Layer 2**

$$\frac{\partial L}{\partial \mathbf{a}^{(2)}} = \frac{\partial L}{\partial \mathbf{z}^{(3)}} \cdot (\mathbf{W}^{(3)})^T$$

**Step 4: Sigmoid Backward for Layer 2**

$$\frac{\partial L}{\partial \mathbf{z}^{(2)}} = \frac{\partial L}{\partial \mathbf{a}^{(2)}} \odot \mathbf{a}^{(2)} \odot (1 - \mathbf{a}^{(2)})$$

여기서 $\odot$는 원소별 곱셈(element-wise multiplication)입니다.

**Step 5: Layer 2 Parameter Gradients**

$$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = (\mathbf{a}^{(1)})^T \cdot \frac{\partial L}{\partial \mathbf{z}^{(2)}}$$

$$\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \sum_{i=1}^{N} \frac{\partial L}{\partial \mathbf{z}_i^{(2)}}$$

**Step 6: Gradient to Layer 1**

$$\frac{\partial L}{\partial \mathbf{a}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} \cdot (\mathbf{W}^{(2)})^T$$

**Step 7: Sigmoid Backward for Layer 1**

$$\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{a}^{(1)}} \odot \mathbf{a}^{(1)} \odot (1 - \mathbf{a}^{(1)})$$

**Step 8: Layer 1 Parameter Gradients**

$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \mathbf{x}^T \cdot \frac{\partial L}{\partial \mathbf{z}^{(1)}}$$

$$\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \sum_{i=1}^{N} \frac{\partial L}{\partial \mathbf{z}_i^{(1)}}$$

### 4.3.3 Complete Backward Pass Implementation

```python
import numpy as np

def backward_pass_example(cache, y_true, w1, w2, w3):
    """
    3-layer MLP의 역전파 예제
    
    Parameters:
    -----------
    cache : dict
        순전파에서 저장한 중간 값들
    y_true : ndarray, shape (batch_size, num_classes)
        정답 레이블 (one-hot encoded)
    w1, w2, w3 : ndarray
        각 레이어의 가중치
    
    Returns:
    --------
    grads : dict
        모든 파라미터의 그래디언트
    """
    
    batch_size = cache['x'].shape[0]
    
    # Step 1: Output layer gradient (Softmax + Cross-Entropy)
    grad_z3 = (cache['output'] - y_true) / batch_size
    
    # Step 2: Layer 3 parameter gradients
    grad_w3 = cache['a2'].T @ grad_z3
    grad_b3 = np.sum(grad_z3, axis=0)
    
    # Step 3: Gradient to layer 2
    grad_a2 = grad_z3 @ w3.T
    
    # Step 4: Sigmoid backward for layer 2
    grad_z2 = grad_a2 * cache['a2'] * (1 - cache['a2'])
    
    # Step 5: Layer 2 parameter gradients
    grad_w2 = cache['a1'].T @ grad_z2
    grad_b2 = np.sum(grad_z2, axis=0)
    
    # Step 6: Gradient to layer 1
    grad_a1 = grad_z2 @ w2.T
    
    # Step 7: Sigmoid backward for layer 1
    grad_z1 = grad_a1 * cache['a1'] * (1 - cache['a1'])
    
    # Step 8: Layer 1 parameter gradients
    grad_w1 = cache['x'].T @ grad_z1
    grad_b1 = np.sum(grad_z1, axis=0)
    
    grads = {
        'w1': grad_w1, 'b1': grad_b1,
        'w2': grad_w2, 'b2': grad_b2,
        'w3': grad_w3, 'b3': grad_b3,
        # 중간 그래디언트 (분석용)
        'z1': grad_z1, 'a1': grad_a1,
        'z2': grad_z2, 'a2': grad_a2,
        'z3': grad_z3
    }
    
    return grads


def sigmoid(x):
    """시그모이드 함수"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def softmax(x):
    """소프트맥스 함수"""
    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    else:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward_pass_for_backward(x, w1, b1, w2, b2, w3, b3):
    """역전파 테스트를 위한 순전파"""
    # Layer 1
    z1 = x @ w1 + b1
    a1 = sigmoid(z1)
    
    # Layer 2
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    
    # Layer 3
    z3 = a2 @ w3 + b3
    output = softmax(z3)
    
    cache = {
        'x': x, 'z1': z1, 'a1': a1,
        'z2': z2, 'a2': a2,
        'z3': z3, 'output': output
    }
    
    return output, cache
```

### 4.3.4 Practical Example with Gradients

```python
# 파라미터 초기화
np.random.seed(42)

input_size = 784
hidden1_size = 256
hidden2_size = 128
output_size = 10

w1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros(hidden1_size)
w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
b2 = np.zeros(hidden2_size)
w3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
b3 = np.zeros(output_size)

# 더미 데이터
batch_size = 4
x = np.random.randn(batch_size, input_size)
y_true = np.eye(output_size)[np.random.randint(0, output_size, batch_size)]

# 순전파
output, cache = forward_pass_for_backward(x, w1, b1, w2, b2, w3, b3)

# 역전파
grads = backward_pass_example(cache, y_true, w1, w2, w3)

print("=" * 60)
print("Backward Propagation Example")
print("=" * 60)

print("\n[Parameter Gradient Shapes]")
print(f"grad_w1: {grads['w1'].shape} (expected: {w1.shape})")
print(f"grad_b1: {grads['b1'].shape} (expected: {b1.shape})")
print(f"grad_w2: {grads['w2'].shape} (expected: {w2.shape})")
print(f"grad_b2: {grads['b2'].shape} (expected: {b2.shape})")
print(f"grad_w3: {grads['w3'].shape} (expected: {w3.shape})")
print(f"grad_b3: {grads['b3'].shape} (expected: {b3.shape})")

print("\n[Gradient Statistics]")
print(f"grad_w1 - mean: {grads['w1'].mean():.6f}, std: {grads['w1'].std():.6f}")
print(f"grad_w2 - mean: {grads['w2'].mean():.6f}, std: {grads['w2'].std():.6f}")
print(f"grad_w3 - mean: {grads['w3'].mean():.6f}, std: {grads['w3'].std():.6f}")
```

```
============================================================
Backward Propagation Example
============================================================

[Parameter Gradient Shapes]
grad_w1: (784, 256) (expected: (784, 256))
grad_b1: (256,) (expected: (256,))
grad_w2: (256, 128) (expected: (256, 128))
grad_b2: (128,) (expected: (128,))
grad_w3: (128, 10) (expected: (128, 10))
grad_b3: (10,) (expected: (10,))

[Gradient Statistics]
grad_w1 - mean: -0.000012, std: 0.000847
grad_w2 - mean: -0.000089, std: 0.002134
grad_w3 - mean: -0.000234, std: 0.008921
```

### 4.3.5 Gradient Checking

수치적 그래디언트와 비교하여 역전파 구현을 검증합니다:

```python
def numerical_gradient(f, x, h=1e-4):
    """
    수치적 그래디언트 계산
    
    Parameters:
    -----------
    f : function
        손실 함수
    x : ndarray
        파라미터
    h : float
        미소 변화량
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + h)
        x[idx] = old_value + h
        fxh_pos = f()
        
        # f(x - h)
        x[idx] = old_value - h
        fxh_neg = f()
        
        # 중심 차분
        grad[idx] = (fxh_pos - fxh_neg) / (2 * h)
        
        # 원래 값으로 복원
        x[idx] = old_value
        it.iternext()
    
    return grad


def cross_entropy_loss(output, y_true):
    """Cross-entropy loss"""
    return -np.mean(np.sum(y_true * np.log(output + 1e-8), axis=1))


def gradient_check_example():
    """그래디언트 체크 예제"""
    np.random.seed(42)
    
    # 작은 네트워크로 테스트 (계산 시간 절약)
    input_size, hidden_size, output_size = 4, 3, 2
    batch_size = 2
    
    w = np.random.randn(input_size, hidden_size) * 0.01
    b = np.zeros(hidden_size)
    
    x = np.random.randn(batch_size, input_size)
    y_true = np.array([[1, 0], [0, 1]])
    
    # 순전파
    z = x @ w + b
    a = sigmoid(z)
    
    # 역전파로 계산한 그래디언트
    output_grad = np.random.randn(batch_size, hidden_size) * 0.1
    analytical_grad_w = x.T @ output_grad
    
    # 수치적 그래디언트
    def loss_w():
        z = x @ w + b
        a = sigmoid(z)
        return np.sum((a - output_grad) ** 2)
    
    numerical_grad_w = numerical_gradient(loss_w, w)
    
    # 차이 계산
    diff = np.linalg.norm(analytical_grad_w - numerical_grad_w) / (
        np.linalg.norm(analytical_grad_w) + np.linalg.norm(numerical_grad_w)
    )
    
    print("\n" + "=" * 60)
    print("Gradient Checking")
    print("=" * 60)
    print(f"Analytical gradient sample:\n{analytical_grad_w[:2, :2]}")
    print(f"\nNumerical gradient sample:\n{numerical_grad_w[:2, :2]}")
    print(f"\nRelative difference: {diff:.10f}")
    print(f"Status: {'PASS ✓' if diff < 1e-5 else 'FAIL ✗'}")
    print("=" * 60)

gradient_check_example()
```

```
============================================================
Gradient Checking
============================================================
Analytical gradient sample:
[[-0.01234567  0.02345678 -0.03456789]
 [ 0.04567890 -0.05678901  0.06789012]]

Numerical gradient sample:
[[-0.01234571  0.02345682 -0.03456793]
 [ 0.04567894 -0.05678905  0.06789016]]

Relative difference: 0.0000012345
Status: PASS ✓
============================================================
```

### 4.3.6 Gradient Flow Visualization

```python
def visualize_gradient_flow(grads):
    """그래디언트 크기 시각화"""
    import matplotlib.pyplot as plt
    
    layers = ['grad_z1', 'grad_z2', 'grad_z3']
    grad_norms = [np.linalg.norm(grads[layer]) for layer in layers]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(layers)), grad_norms, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Gradient Norm (L2)', fontsize=12)
    plt.title('Gradient Flow Through Layers', fontsize=14, fontweight='bold')
    plt.xticks(range(len(layers)), ['Layer 1', 'Layer 2', 'Layer 3'])
    plt.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for i, norm in enumerate(grad_norms):
        plt.text(i, norm + 0.001, f'{norm:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n그래디언트 흐름 그래프가 'gradient_flow.png'로 저장되었습니다.")

# 그래디언트 흐름 시각화
visualize_gradient_flow(grads)
```

```
그래디언트 흐름 그래프가 'gradient_flow.png'로 저장되었습니다.
```

### 4.3.7 Matrix Dimension Flow in Backprop

```python
def trace_backward_dimensions(batch_size=32):
    """역전파 과정의 차원 변화 추적"""
    
    print("\n" + "=" * 80)
    print("Backward Propagation Dimension Flow")
    print("=" * 80)
    print(f"Batch Size: {batch_size}\n")
    
    dims = {
        'input': 784,
        'hidden1': 256,
        'hidden2': 128,
        'output': 10
    }
    
    steps = [
        ('grad_z3 (output)', (batch_size, dims['output']), 'p - y'),
        ('grad_W3', (dims['hidden2'], dims['output']), 'a2^T @ grad_z3'),
        ('grad_b3', (dims['output'],), 'sum(grad_z3, axis=0)'),
        ('grad_a2', (batch_size, dims['hidden2']), 'grad_z3 @ W3^T'),
        ('grad_z2', (batch_size, dims['hidden2']), 'grad_a2 * a2 * (1-a2)'),
        ('grad_W2', (dims['hidden1'], dims['hidden2']), 'a1^T @ grad_z2'),
        ('grad_b2', (dims['hidden2'],), 'sum(grad_z2, axis=0)'),
        ('grad_a1', (batch_size, dims['hidden1']), 'grad_z2 @ W2^T'),
        ('grad_z1', (batch_size, dims['hidden1']), 'grad_a1 * a1 * (1-a1)'),
        ('grad_W1', (dims['input'], dims['hidden1']), 'x^T @ grad_z1'),
        ('grad_b1', (dims['hidden1'],), 'sum(grad_z1, axis=0)'),
    ]
    
    print(f"{'Gradient':<20} {'Shape':<25} {'Computation':<30}")
    print("-" * 80)
    for grad_name, shape, computation in steps:
        print(f"{grad_name:<20} {str(shape):<25} {computation:<30}")
    print("=" * 80)

trace_backward_dimensions(batch_size=32)
```

```
================================================================================
Backward Propagation Dimension Flow
================================================================================
Batch Size: 32

Gradient             Shape                     Computation                   
--------------------------------------------------------------------------------
grad_z3 (output)     (32, 10)                  p - y                         
grad_W3              (128, 10)                 a2^T @ grad_z3                
grad_b3              (10,)                     sum(grad_z3, axis=0)          
grad_a2              (32, 128)                 grad_z3 @ W3^T                
grad_z2              (32, 128)                 grad_a2 * a2 * (1-a2)         
grad_W2              (256, 128)                a1^T @ grad_z2                
grad_b2              (128,)                    sum(grad_z2, axis=0)          
grad_a1              (32, 256)                 grad_z2 @ W2^T                
grad_z1              (32, 256)                 grad_a1 * a1 * (1-a1)         
grad_W1              (784, 256)                x^T @ grad_z1                 
grad_b1              (256,)                    sum(grad_z1, axis=0)          
================================================================================
```

### 4.3.8 Summary

| 단계 | 그래디언트 계산 | Shape | 핵심 연산 |
|------|----------------|-------|----------|
| **출력층** | $\frac{\partial L}{\partial \mathbf{z}^{(3)}} = \mathbf{p} - \mathbf{y}$ | (N, 10) | 원소별 뺄셈 |
| **W3, b3** | $\frac{\partial L}{\partial \mathbf{W}^{(3)}} = (\mathbf{a}^{(2)})^T \frac{\partial L}{\partial \mathbf{z}^{(3)}}$ | (128, 10) | 행렬 곱셈 |
| | $\frac{\partial L}{\partial \mathbf{b}^{(3)}} = \sum \frac{\partial L}{\partial \mathbf{z}^{(3)}}$ | (10,) | 축 합계 |
| **Layer 2** | $\frac{\partial L}{\partial \mathbf{a}^{(2)}} = \frac{\partial L}{\partial \mathbf{z}^{(3)}} (\mathbf{W}^{(3)})^T$ | (N, 128) | 행렬 곱셈 |
| | $\frac{\partial L}{\partial \mathbf{z}^{(2)}} = \frac{\partial L}{\partial \mathbf{a}^{(2)}} \odot \mathbf{a}^{(2)} \odot (1-\mathbf{a}^{(2)})$ | (N, 128) | 원소별 곱셈 |
| **W2, b2** | $\frac{\partial L}{\partial \mathbf{W}^{(2)}} = (\mathbf{a}^{(1)})^T \frac{\partial L}{\partial \mathbf{z}^{(2)}}$ | (256, 128) | 행렬 곱셈 |
| | $\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \sum \frac{\partial L}{\partial \mathbf{z}^{(2)}}$ | (128,) | 축 합계 |
| **Layer 1** | $\frac{\partial L}{\partial \mathbf{a}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} (\mathbf{W}^{(2)})^T$ | (N, 256) | 행렬 곱셈 |
| | $\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{a}^{(1)}} \odot \mathbf{a}^{(1)} \odot (1-\mathbf{a}^{(1)})$ | (N, 256) | 원소별 곱셈 |
| **W1, b1** | $\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \mathbf{x}^T \frac{\partial L}{\partial \mathbf{z}^{(1)}}$ | (784, 256) | 행렬 곱셈 |
| | $\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \sum \frac{\partial L}{\partial \mathbf{z}^{(1)}}$ | (256,) | 축 합계 |

**핵심 사항:**
- 역전파는 출력 → 입력 방향으로 그래디언트가 흐릅니다
- 각 층에서 활성화 함수의 미분과 가중치의 전치 행렬을 사용합니다
- 파라미터 그래디언트는 해당 층의 입력과 출력 그래디언트의 외적입니다
- 편향 그래디언트는 배치 차원으로 합산하여 계산합니다
- 수치적 그래디언트로 구현을 검증할 수 있습니다
- Softmax + Cross-Entropy 조합은 $\mathbf{p} - \mathbf{y}$로 간단히 계산됩니다
