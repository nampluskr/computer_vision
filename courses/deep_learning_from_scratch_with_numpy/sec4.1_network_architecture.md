## 4.1 Network Architecture

MLP(Multi-Layer Perceptron)는 가장 기본적인 형태의 완전 연결 신경망(Fully Connected Neural Network)입니다. 입력층, 하나 이상의 은닉층, 출력층으로 구성되며, 각 층의 모든 뉴런이 다음 층의 모든 뉴런과 연결되어 있습니다.

### 4.1.1 Layer Structure

MLP의 각 레이어는 **선형 변환(Linear Transformation)**과 **활성화 함수(Activation Function)**로 구성됩니다.

**선형 변환:**

$$\mathbf{z}^{(l)} = \mathbf{a}^{(l-1)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)}$$

여기서:
- $\mathbf{a}^{(l-1)}$: 이전 층의 활성화 출력 (batch_size, in_features)
- $\mathbf{W}^{(l)}$: 가중치 행렬 (in_features, out_features)
- $\mathbf{b}^{(l)}$: 편향 벡터 (out_features)
- $\mathbf{z}^{(l)}$: 선형 변환 결과 (batch_size, out_features)

**활성화 함수:**

$$\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$$

여기서 $\sigma$는 비선형 활성화 함수입니다.

### 4.1.2 NumPy Implementation

```python
import numpy as np

class Linear:
    """선형 변환 레이어"""
    
    def __init__(self, in_features, out_features):
        # He 초기화 (은닉층용)
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        
        # 그래디언트 저장용
        self.grad_W = None
        self.grad_b = None
        
        # 역전파를 위한 입력 저장
        self.x = None
    
    def forward(self, x):
        """순전파: z = xW + b"""
        self.x = x  # 역전파를 위해 저장
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        """역전파: 그래디언트 계산"""
        # ∂L/∂W = x^T @ ∂L/∂z
        self.grad_W = self.x.T @ grad_output
        # ∂L/∂b = sum(∂L/∂z, axis=0)
        self.grad_b = np.sum(grad_output, axis=0)
        # ∂L/∂x = ∂L/∂z @ W^T
        grad_input = grad_output @ self.W.T
        return grad_input


class Sigmoid:
    """시그모이드 활성화 함수"""
    
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        """순전파: σ(x) = 1 / (1 + e^(-x))"""
        # 수치 안정성을 위한 구현
        self.out = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        return self.out
    
    def backward(self, grad_output):
        """역전파: ∂L/∂x = ∂L/∂a * σ(x) * (1 - σ(x))"""
        return grad_output * self.out * (1 - self.out)


class Softmax:
    """소프트맥스 활성화 함수"""
    
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        """순전파: softmax(x_i) = e^(x_i) / Σ e^(x_j)"""
        if x.ndim == 1:
            exp_x = np.exp(x - np.max(x))
            self.out = exp_x / np.sum(exp_x)
        else:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
```

### 4.1.3 3-Layer MLP Architecture

본 교재에서 사용할 표준 MLP 구조는 다음과 같습니다:

```
Input (784) → Hidden1 (256) → Hidden2 (128) → Output (task-dependent)
```

**구조 상세:**

```python
class MLP:
    """3-layer Multi-Layer Perceptron"""
    
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Layer 1: Input → Hidden1
        self.fc1 = Linear(input_size, hidden1_size)
        self.act1 = Sigmoid()
        
        # Layer 2: Hidden1 → Hidden2
        self.fc2 = Linear(hidden1_size, hidden2_size)
        self.act2 = Sigmoid()
        
        # Layer 3: Hidden2 → Output
        self.fc3 = Linear(hidden2_size, output_size)
        # 출력층 활성화는 태스크에 따라 달라짐
        self.act3 = None  # Regression: None, Binary: Sigmoid, Multiclass: Softmax
    
    def forward(self, x):
        """순전파 과정"""
        # Hidden Layer 1
        z1 = self.fc1.forward(x)
        a1 = self.act1.forward(z1)
        
        # Hidden Layer 2
        z2 = self.fc2.forward(a1)
        a2 = self.act2.forward(z2)
        
        # Output Layer
        z3 = self.fc3.forward(a2)
        if self.act3 is not None:
            out = self.act3.forward(z3)
        else:
            out = z3  # Linear output (Regression)
        
        return out
```

### 4.1.4 Example Usage

```python
# MNIST 분류를 위한 MLP 생성
input_size = 28 * 28  # MNIST 이미지 크기
hidden1_size = 256
hidden2_size = 128
output_size = 10  # 10개 클래스

model = MLP(input_size, hidden1_size, hidden2_size, output_size)

# 더미 데이터로 테스트
batch_size = 32
x_dummy = np.random.randn(batch_size, input_size)
output = model.forward(x_dummy)

print(f"Input shape:  {x_dummy.shape}")
print(f"Output shape: {output.shape}")
```

```
Input shape:  (32, 784)
Output shape: (32, 10)
```

### 4.1.5 Task-Specific Output Configuration

동일한 MLP 구조에서 출력층만 변경하여 다양한 태스크를 수행할 수 있습니다:

```python
# Regression (회귀)
model_regression = MLP(784, 256, 128, 1)
model_regression.act3 = None  # Linear output

# Binary Classification (이진 분류)
model_binary = MLP(784, 256, 128, 1)
model_binary.act3 = Sigmoid()  # Sigmoid output

# Multiclass Classification (다중 클래스 분류)
model_multiclass = MLP(784, 256, 128, 10)
model_multiclass.act3 = Softmax()  # Softmax output
```

### 4.1.6 Summary

| 구성 요소 | 역할 | 입력 Shape | 출력 Shape |
|-----------|------|-----------|-----------|
| **Input Layer** | 데이터 입력 | (N, 784) | (N, 784) |
| **fc1 + act1** | 첫 번째 은닉층 | (N, 784) | (N, 256) |
| **fc2 + act2** | 두 번째 은닉층 | (N, 256) | (N, 128) |
| **fc3** | 선형 변환 | (N, 128) | (N, output_size) |
| **act3** | 태스크별 활성화 | (N, output_size) | (N, output_size) |

**출력층 구성 (태스크별):**

| 태스크 | Output Size | 활성화 함수 | 출력 범위 |
|--------|-------------|-------------|-----------|
| **Regression** | 1 (연속값) | None (Linear) | $(-\infty, +\infty)$ |
| **Binary Classification** | 1 (확률) | Sigmoid | $(0, 1)$ |
| **Multiclass Classification** | K (클래스 개수) | Softmax | $\sum_i p_i = 1$ |

**공통 설정:**
- 은닉층 활성화: Sigmoid (Part 2), ReLU (Part 3에서 개선)
- 가중치 초기화: He initialization
- 편향 초기화: Zero initialization
