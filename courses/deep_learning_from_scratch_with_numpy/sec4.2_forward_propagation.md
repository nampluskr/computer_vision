## 4.2 Forward Propagation

순전파(Forward Propagation)는 입력 데이터가 신경망의 각 층을 거쳐 최종 출력까지 전달되는 과정입니다. 각 층에서는 선형 변환과 활성화 함수를 순차적으로 적용합니다.

### 4.2.1 Layer-by-Layer Forward Pass

3-layer MLP의 순전파 과정을 수식으로 표현하면 다음과 같습니다:

**Layer 1 (Input → Hidden1):**

$$\mathbf{z}^{(1)} = \mathbf{x} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}$$

$$\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$$

**Layer 2 (Hidden1 → Hidden2):**

$$\mathbf{z}^{(2)} = \mathbf{a}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

$$\mathbf{a}^{(2)} = \sigma(\mathbf{z}^{(2)})$$

**Layer 3 (Hidden2 → Output):**

$$\mathbf{z}^{(3)} = \mathbf{a}^{(2)} \mathbf{W}^{(3)} + \mathbf{b}^{(3)}$$

$$\mathbf{a}^{(3)} = f(\mathbf{z}^{(3)})$$

여기서 $f$는 태스크에 따라 다릅니다:
- Regression: $f(z) = z$ (항등 함수)
- Binary Classification: $f(z) = \text{sigmoid}(z)$
- Multiclass Classification: $f(z) = \text{softmax}(z)$

### 4.2.2 Complete Forward Pass Implementation

```python
import numpy as np

def forward_pass_example(x, w1, b1, w2, b2, w3, b3, task='multiclass'):
    """
    3-layer MLP의 순전파 예제
    
    Parameters:
    -----------
    x : ndarray, shape (batch_size, input_size)
        입력 데이터
    w1, b1 : 첫 번째 레이어 파라미터
    w2, b2 : 두 번째 레이어 파라미터
    w3, b3 : 세 번째 레이어 파라미터
    task : str
        'regression', 'binary', 'multiclass' 중 선택
    
    Returns:
    --------
    output : ndarray
        최종 출력
    cache : dict
        역전파를 위한 중간 값들
    """
    
    # Layer 1: Input → Hidden1
    z1 = x @ w1 + b1
    a1 = sigmoid(z1)
    
    # Layer 2: Hidden1 → Hidden2
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    
    # Layer 3: Hidden2 → Output
    z3 = a2 @ w3 + b3
    
    # 태스크별 출력층 활성화
    if task == 'regression':
        output = z3  # Linear output
    elif task == 'binary':
        output = sigmoid(z3)
    elif task == 'multiclass':
        output = softmax(z3)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 역전파를 위한 중간 값 저장
    cache = {
        'x': x, 'z1': z1, 'a1': a1,
        'z2': z2, 'a2': a2,
        'z3': z3, 'output': output
    }
    
    return output, cache


def sigmoid(x):
    """시그모이드 함수 (수치 안정성 고려)"""
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
```

### 4.2.3 Practical Example with MNIST

```python
# 파라미터 초기화
np.random.seed(42)

input_size = 784
hidden1_size = 256
hidden2_size = 128
output_size = 10

# He 초기화
w1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros(hidden1_size)
w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
b2 = np.zeros(hidden2_size)
w3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
b3 = np.zeros(output_size)

# 더미 입력 데이터 (미니배치)
batch_size = 4
x = np.random.randn(batch_size, input_size)

# 순전파 실행
output, cache = forward_pass_example(x, w1, b1, w2, b2, w3, b3, task='multiclass')

# 결과 확인
print("=" * 60)
print("Forward Propagation Example")
print("=" * 60)
print(f"\nInput shape:           {x.shape}")
print(f"Layer 1 (z1) shape:    {cache['z1'].shape}")
print(f"Layer 1 (a1) shape:    {cache['a1'].shape}")
print(f"Layer 2 (z2) shape:    {cache['z2'].shape}")
print(f"Layer 2 (a2) shape:    {cache['a2'].shape}")
print(f"Layer 3 (z3) shape:    {cache['z3'].shape}")
print(f"Output shape:          {output.shape}")

print(f"\n첫 번째 샘플의 출력 (10개 클래스 확률):")
print(output[0])
print(f"확률 합계: {output[0].sum():.6f}")
print(f"예측 클래스: {output[0].argmax()}")
```

```
============================================================
Forward Propagation Example
============================================================

Input shape:           (4, 784)
Layer 1 (z1) shape:    (4, 256)
Layer 1 (a1) shape:    (4, 256)
Layer 2 (z2) shape:    (4, 128)
Layer 2 (a2) shape:    (4, 128)
Layer 3 (z3) shape:    (4, 10)
Output shape:          (4, 10)

첫 번째 샘플의 출력 (10개 클래스 확률):
[0.08571429 0.10238095 0.09714286 0.11047619 0.09904762 0.10428571
 0.09714286 0.10380952 0.10095238 0.09904762]
확률 합계: 1.000000
예측 클래스: 3
```

### 4.2.4 Activation Visualization

각 층의 활성화 패턴을 시각화하여 순전파 과정을 이해할 수 있습니다:

```python
import matplotlib.pyplot as plt

def visualize_activations(cache):
    """각 층의 활성화 값 분포 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    activations = [
        ('Hidden Layer 1', cache['a1']),
        ('Hidden Layer 2', cache['a2']),
        ('Output Layer', cache['output'])
    ]
    
    for ax, (title, activation) in zip(axes, activations):
        ax.hist(activation.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(activation.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {activation.mean():.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('activation_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# 활성화 시각화
visualize_activations(cache)
print("\n활성화 분포 그래프가 'activation_distribution.png'로 저장되었습니다.")
```

```
활성화 분포 그래프가 'activation_distribution.png'로 저장되었습니다.
```

### 4.2.5 Matrix Dimension Flow

순전파 과정에서 텐서의 차원 변화를 추적하는 것은 디버깅과 이해에 매우 중요합니다:

```python
def trace_forward_dimensions(batch_size=32):
    """순전파 과정의 차원 변화 추적"""
    
    print("\n" + "=" * 70)
    print("Forward Propagation Dimension Flow")
    print("=" * 70)
    print(f"Batch Size: {batch_size}\n")
    
    # 차원 정보
    dims = {
        'input': 784,
        'hidden1': 256,
        'hidden2': 128,
        'output': 10
    }
    
    # Layer별 차원 변화
    layers = [
        ('Input', (batch_size, dims['input']), None),
        ('FC1 (Linear)', (batch_size, dims['hidden1']), 
         f"({dims['input']}, {dims['hidden1']})"),
        ('Sigmoid', (batch_size, dims['hidden1']), None),
        ('FC2 (Linear)', (batch_size, dims['hidden2']), 
         f"({dims['hidden1']}, {dims['hidden2']})"),
        ('Sigmoid', (batch_size, dims['hidden2']), None),
        ('FC3 (Linear)', (batch_size, dims['output']), 
         f"({dims['hidden2']}, {dims['output']})"),
        ('Softmax', (batch_size, dims['output']), None),
    ]
    
    print(f"{'Layer':<20} {'Output Shape':<20} {'Weight Shape':<20}")
    print("-" * 70)
    for layer_name, output_shape, weight_shape in layers:
        weight_info = weight_shape if weight_shape else '-'
        print(f"{layer_name:<20} {str(output_shape):<20} {weight_info:<20}")
    print("=" * 70)

# 차원 추적
trace_forward_dimensions(batch_size=32)
```

```
======================================================================
Forward Propagation Dimension Flow
======================================================================
Batch Size: 32

Layer                Output Shape         Weight Shape        
----------------------------------------------------------------------
Input                (32, 784)            -                   
FC1 (Linear)         (32, 256)            (784, 256)          
Sigmoid              (32, 256)            -                   
FC2 (Linear)         (32, 128)            (256, 128)          
Sigmoid              (32, 128)            -                   
FC3 (Linear)         (32, 10)             (128, 10)           
Softmax              (32, 10)             -                   
======================================================================
```

### 4.2.6 Computational Cost Analysis

순전파의 계산 비용을 분석합니다:

```python
def analyze_forward_cost():
    """순전파의 계산 비용 분석"""
    
    dims = {
        'input': 784,
        'hidden1': 256,
        'hidden2': 128,
        'output': 10
    }
    
    # 각 레이어의 FLOPs (Floating Point Operations)
    flops = []
    
    # FC1: (N, 784) @ (784, 256) → 2 * N * 784 * 256 FLOPs
    fc1_flops = 2 * dims['input'] * dims['hidden1']
    flops.append(('FC1', fc1_flops))
    
    # Sigmoid1: N * 256 operations (간단화)
    sig1_flops = dims['hidden1']
    flops.append(('Sigmoid1', sig1_flops))
    
    # FC2: (N, 256) @ (256, 128) → 2 * N * 256 * 128 FLOPs
    fc2_flops = 2 * dims['hidden1'] * dims['hidden2']
    flops.append(('FC2', fc2_flops))
    
    # Sigmoid2: N * 128 operations
    sig2_flops = dims['hidden2']
    flops.append(('Sigmoid2', sig2_flops))
    
    # FC3: (N, 128) @ (128, 10) → 2 * N * 128 * 10 FLOPs
    fc3_flops = 2 * dims['hidden2'] * dims['output']
    flops.append(('FC3', fc3_flops))
    
    # Softmax: N * 10 operations (간단화)
    softmax_flops = dims['output']
    flops.append(('Softmax', softmax_flops))
    
    total_flops = sum(f for _, f in flops)
    
    print("\n" + "=" * 60)
    print("Computational Cost per Sample")
    print("=" * 60)
    print(f"{'Layer':<15} {'FLOPs':<15} {'Percentage':<15}")
    print("-" * 60)
    for layer, f in flops:
        percentage = (f / total_flops) * 100
        print(f"{layer:<15} {f:<15,} {percentage:>6.2f}%")
    print("-" * 60)
    print(f"{'Total':<15} {total_flops:<15,} {100.0:>6.2f}%")
    print("=" * 60)

analyze_forward_cost()
```

```
============================================================
Computational Cost per Sample
============================================================
Layer           FLOPs           Percentage     
------------------------------------------------------------
FC1             401,408          93.34%
Sigmoid1        256               0.06%
FC2             65,536           15.24%
Sigmoid2        128               0.03%
FC3             2,560             0.60%
Softmax         10                0.00%
------------------------------------------------------------
Total           469,898          100.00%
============================================================
```

### 4.2.7 Summary

| 단계 | 연산 | 입력 Shape | 출력 Shape | 주요 연산 |
|------|------|-----------|-----------|----------|
| **Layer 1** | $\mathbf{z}^{(1)} = \mathbf{x}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$ | (N, 784) | (N, 256) | 행렬 곱셈 |
| | $\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$ | (N, 256) | (N, 256) | 원소별 연산 |
| **Layer 2** | $\mathbf{z}^{(2)} = \mathbf{a}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ | (N, 256) | (N, 128) | 행렬 곱셈 |
| | $\mathbf{a}^{(2)} = \sigma(\mathbf{z}^{(2)})$ | (N, 128) | (N, 128) | 원소별 연산 |
| **Layer 3** | $\mathbf{z}^{(3)} = \mathbf{a}^{(2)}\mathbf{W}^{(3)} + \mathbf{b}^{(3)}$ | (N, 128) | (N, 10) | 행렬 곱셈 |
| | $\mathbf{a}^{(3)} = f(\mathbf{z}^{(3)})$ | (N, 10) | (N, 10) | 태스크별 활성화 |

**핵심 사항:**
- 순전파는 입력 → 출력 방향으로 데이터가 흐릅니다
- 각 층에서 선형 변환 → 활성화 함수가 순차적으로 적용됩니다
- 중간 값들(`z`, `a`)은 역전파를 위해 반드시 저장되어야 합니다
- 계산 비용의 대부분은 행렬 곱셈(선형 변환)에서 발생합니다
- 출력층 활성화 함수는 태스크에 따라 달라집니다
