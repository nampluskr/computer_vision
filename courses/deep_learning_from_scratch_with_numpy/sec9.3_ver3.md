## 9.3 Version 3 - Module Abstraction

Version 3에서는 레이어와 네트워크를 모듈로 추상화하여 PyTorch의 `nn.Module`과 유사한 구조를 구현합니다. 이를 통해 코드의 재사용성과 확장성을 크게 향상시킵니다.

### 9.3.1 Overview

**Version 3의 개선 사항:**

- Module 베이스 클래스 도입
- 레이어를 독립적인 클래스로 캡슐화
- 순전파/역전파 메서드 분리
- 파라미터 자동 관리
- 네트워크를 모듈의 조합으로 구성

**Version 2 → Version 3 변화:**

```python
# Version 2: 명시적 순전파
z1 = np.dot(x, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)
z3 = np.dot(a2, w3) + b3
preds = softmax(z3)

# Version 3: 모듈 기반 순전파
model = MLP(input_size=784, hidden_size=100, output_size=10)
preds = model.forward(x)
```

### 9.3.2 Module Base Class

```python
import numpy as np

class Module:
    """
    모든 신경망 모듈의 베이스 클래스
    PyTorch의 nn.Module과 유사
    """
    
    def __init__(self):
        self._modules = {}
        self._parameters = {}
    
    def forward(self, x):
        """
        순전파 (서브클래스에서 구현)
        """
        raise NotImplementedError
    
    def backward(self, grad):
        """
        역전파 (서브클래스에서 구현)
        """
        raise NotImplementedError
    
    def parameters(self):
        """
        모든 파라미터 반환 (재귀적)
        
        Returns:
        --------
        params : list of dict
            {'name': str, 'weight': ndarray, 'grad': ndarray}
        """
        params = []
        
        # 자신의 파라미터
        for name, param in self._parameters.items():
            params.append({
                'name': name,
                'weight': param['weight'],
                'grad': param['grad']
            })
        
        # 서브모듈의 파라미터 (재귀적)
        for name, module in self._modules.items():
            sub_params = module.parameters()
            for param in sub_params:
                param['name'] = f"{name}.{param['name']}"
            params.extend(sub_params)
        
        return params
    
    def __call__(self, x):
        """모듈을 함수처럼 호출 가능하게"""
        return self.forward(x)
```

### 9.3.3 Layer Modules

**Linear Layer:**

```python
class Linear(Module):
    """
    선형 변환 레이어: y = xW + b
    """
    
    def __init__(self, in_features, out_features):
        """
        Parameters:
        -----------
        in_features : int
            입력 차원
        out_features : int
            출력 차원
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # He 초기화
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        
        # 그래디언트 초기화
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        
        # 파라미터 등록
        self._parameters['weight'] = {
            'weight': self.weight,
            'grad': self.weight_grad
        }
        self._parameters['bias'] = {
            'weight': self.bias,
            'grad': self.bias_grad
        }
        
        # 역전파를 위한 입력 저장
        self.x = None
    
    def forward(self, x):
        """
        순전파: y = xW + b
        
        Parameters:
        -----------
        x : ndarray, shape (batch_size, in_features)
        
        Returns:
        --------
        out : ndarray, shape (batch_size, out_features)
        """
        self.x = x
        out = x @ self.weight + self.bias
        return out
    
    def backward(self, grad_output):
        """
        역전파
        
        Parameters:
        -----------
        grad_output : ndarray, shape (batch_size, out_features)
            출력에 대한 그래디언트
        
        Returns:
        --------
        grad_input : ndarray, shape (batch_size, in_features)
            입력에 대한 그래디언트
        """
        # 파라미터 그래디언트 계산
        self.weight_grad = self.x.T @ grad_output
        self.bias_grad = np.sum(grad_output, axis=0)
        
        # 파라미터 딕셔너리 업데이트
        self._parameters['weight']['grad'] = self.weight_grad
        self._parameters['bias']['grad'] = self.bias_grad
        
        # 입력 그래디언트 계산
        grad_input = grad_output @ self.weight.T
        
        return grad_input
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"
```

**Activation Layers:**

```python
class Sigmoid(Module):
    """시그모이드 활성화 함수"""
    
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, x):
        """순전파: σ(x)"""
        self.out = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        return self.out
    
    def backward(self, grad_output):
        """역전파: grad * σ(x) * (1 - σ(x))"""
        grad_input = grad_output * self.out * (1 - self.out)
        return grad_input
    
    def __repr__(self):
        return "Sigmoid()"


class ReLU(Module):
    """ReLU 활성화 함수"""
    
    def __init__(self):
        super().__init__()
        self.mask = None
    
    def forward(self, x):
        """순전파: max(0, x)"""
        self.mask = (x > 0)
        return x * self.mask
    
    def backward(self, grad_output):
        """역전파: grad * (x > 0)"""
        return grad_output * self.mask
    
    def __repr__(self):
        return "ReLU()"


class Softmax(Module):
    """소프트맥스 활성화 함수"""
    
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, x):
        """순전파: softmax(x)"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
    
    def backward(self, grad_output):
        """
        역전파 (보통 손실 함수와 결합되어 사용)
        여기서는 단순 통과
        """
        return grad_output
    
    def __repr__(self):
        return "Softmax()"
```

### 9.3.4 Sequential Container

```python
class Sequential(Module):
    """
    레이어를 순차적으로 연결하는 컨테이너
    PyTorch의 nn.Sequential과 유사
    """
    
    def __init__(self, *layers):
        """
        Parameters:
        -----------
        *layers : Module
            순차적으로 연결할 레이어들
        """
        super().__init__()
        
        self.layers = layers
        
        # 서브모듈로 등록
        for i, layer in enumerate(layers):
            self._modules[f'layer_{i}'] = layer
    
    def forward(self, x):
        """
        순전파: 각 레이어를 순차적으로 통과
        
        Parameters:
        -----------
        x : ndarray
            입력
        
        Returns:
        --------
        out : ndarray
            최종 출력
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad):
        """
        역전파: 역순으로 각 레이어를 통과
        
        Parameters:
        -----------
        grad : ndarray
            출력에 대한 그래디언트
        
        Returns:
        --------
        grad_input : ndarray
            입력에 대한 그래디언트
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def __repr__(self):
        layer_str = '\n  '.join([f"({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"Sequential(\n  {layer_str}\n)"
```

### 9.3.5 MLP Model

```python
class MLP(Module):
    """
    Multi-Layer Perceptron
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Parameters:
        -----------
        input_size : int
            입력 차원
        hidden_size : int
            은닉층 차원
        output_size : int
            출력 차원
        """
        super().__init__()
        
        # Sequential을 사용한 네트워크 구성
        self.network = Sequential(
            Linear(input_size, hidden_size),
            Sigmoid(),
            Linear(hidden_size, hidden_size),
            Sigmoid(),
            Linear(hidden_size, output_size),
            Softmax()
        )
        
        # 서브모듈로 등록
        self._modules['network'] = self.network
    
    def forward(self, x):
        """순전파"""
        return self.network.forward(x)
    
    def backward(self, grad):
        """역전파"""
        return self.network.backward(grad)
    
    def __repr__(self):
        return f"MLP(\n  {self.network}\n)"


# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = MLP(input_size=784, hidden_size=100, output_size=10)
    
    print("=" * 70)
    print("Model Architecture")
    print("=" * 70)
    print(model)
    
    # 파라미터 확인
    params = model.parameters()
    print(f"\n>> Total parameters: {len(params)}")
    
    total_elements = 0
    for param in params:
        num_elements = param['weight'].size
        total_elements += num_elements
        print(f"  {param['name']:<30} shape: {param['weight'].shape}")
    
    print(f"\n>> Total elements: {total_elements:,}")
    
    # 순전파 테스트
    x = np.random.randn(32, 784)
    out = model.forward(x)
    print(f"\n>> Forward test:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    # 역전파 테스트
    grad_out = np.random.randn(32, 10)
    grad_in = model.backward(grad_out)
    print(f"\n>> Backward test:")
    print(f"  Grad output shape: {grad_out.shape}")
    print(f"  Grad input shape:  {grad_in.shape}")
```

```
======================================================================
Model Architecture
======================================================================
MLP(
  Sequential(
  (0): Linear(in_features=784, out_features=100)
  (1): Sigmoid()
  (2): Linear(in_features=100, out_features=100)
  (3): Sigmoid()
  (4): Linear(in_features=100, out_features=10)
  (5): Softmax()
)
)

>> Total parameters: 6
  network.layer_0.weight         shape: (784, 100)
  network.layer_0.bias           shape: (100,)
  network.layer_2.weight         shape: (100, 100)
  network.layer_2.bias           shape: (100,)
  network.layer_4.weight         shape: (100, 10)
  network.layer_4.bias           shape: (10,)

>> Total elements: 89,510

>> Forward test:
  Input shape:  (32, 784)
  Output shape: (32, 10)

>> Backward test:
  Grad output shape: (32, 10)
  Grad input shape:  (32, 784)
```

### 9.3.6 Complete Version 3 Code

```python
import os
import numpy as np
import gzip

#################################################################
## Module Classes
#################################################################

class Module:
    """베이스 모듈 클래스"""
    
    def __init__(self):
        self._modules = {}
        self._parameters = {}
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def parameters(self):
        params = []
        for name, param in self._parameters.items():
            params.append({
                'name': name,
                'weight': param['weight'],
                'grad': param['grad']
            })
        for name, module in self._modules.items():
            sub_params = module.parameters()
            for param in sub_params:
                param['name'] = f"{name}.{param['name']}"
            params.extend(sub_params)
        return params
    
    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """선형 레이어"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        
        self._parameters['weight'] = {'weight': self.weight, 'grad': self.weight_grad}
        self._parameters['bias'] = {'weight': self.bias, 'grad': self.bias_grad}
        
        self.x = None
    
    def forward(self, x):
        self.x = x
        return x @ self.weight + self.bias
    
    def backward(self, grad_output):
        self.weight_grad = self.x.T @ grad_output
        self.bias_grad = np.sum(grad_output, axis=0)
        self._parameters['weight']['grad'] = self.weight_grad
        self._parameters['bias']['grad'] = self.bias_grad
        return grad_output @ self.weight.T


class Sigmoid(Module):
    """시그모이드 활성화"""
    
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, x):
        self.out = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return self.out
    
    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


class Softmax(Module):
    """소프트맥스 활성화"""
    
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
    
    def backward(self, grad_output):
        return grad_output


class Sequential(Module):
    """순차 컨테이너"""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(layers):
            self._modules[f'layer_{i}'] = layer
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class MLP(Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = Sequential(
            Linear(input_size, hidden_size),
            Sigmoid(),
            Linear(hidden_size, hidden_size),
            Sigmoid(),
            Linear(hidden_size, output_size),
            Softmax()
        )
        self._modules['network'] = self.network
    
    def forward(self, x):
        return self.network.forward(x)
    
    def backward(self, grad):
        return self.network.backward(grad)


#################################################################
## DataLoader (from Version 2)
#################################################################

class DataLoader:
    """데이터 로더"""
    
    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(x)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.num_samples)
        else:
            self.indices = np.arange(self.num_samples)
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        self.current_idx = end_idx
        return self.x[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        return self.num_batches


#################################################################
## Helper Functions
#################################################################

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def get_mnist(data_dir, split="train"):
    if split == "train":
        images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    else:
        images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
    return images, labels


def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


def cross_entropy(preds, targets):
    if targets.ndim == 1:
        batch_size = preds.shape[0]
        probs = preds[np.arange(batch_size), targets]
    else:
        probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))


def accuracy(preds, targets):
    preds = preds.argmax(axis=1)
    if targets.ndim == 2:
        targets = targets.argmax(axis=1)
    return (preds == targets).mean()


#################################################################
## Training Script
#################################################################

if __name__ == "__main__":
    
    np.random.seed(42)
    
    # Data Loading
    data_dir = "/mnt/d/datasets/mnist"
    x_train, y_train = get_mnist(data_dir, split="train")
    x_test, y_test = get_mnist(data_dir, split="test")
    
    print("\n>> Loading data...")
    
    # Preprocessing
    def preprocess(images, labels):
        images = images.astype(np.float32).reshape(-1, 28*28) / 255
        labels = one_hot(labels, num_classes=10).astype(np.int64)
        return images, labels
    
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    
    # Create DataLoaders
    train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)
    
    # Create Model
    model = MLP(input_size=784, hidden_size=100, output_size=10)
    print(f"\n>> Model created:")
    print(f"Parameters: {len(model.parameters())}")
    
    # Training
    num_epochs = 10
    learning_rate = 0.01
    
    print("\n>> Training start ...")
    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        for x_batch, y_batch in train_loader:
            batch_size = x_batch.shape[0]
            train_samples += batch_size
            
            # Forward
            preds = model.forward(x_batch)
            loss = cross_entropy(preds, y_batch)
            acc = accuracy(preds, y_batch)
            
            # Backward
            grad = (preds - y_batch) / batch_size
            model.backward(grad)
            
            # Update
            for param in model.parameters():
                param['weight'] -= learning_rate * param['grad']
            
            train_loss += loss * batch_size
            train_acc += acc * batch_size
        
        print(f"[{epoch:3d}/{num_epochs}] "
              f"loss:{train_loss/train_samples:.3f} "
              f"acc:{train_acc/train_samples:.3f}")
    
    # Evaluation
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    for x_batch, y_batch in test_loader:
        batch_size = x_batch.shape[0]
        test_samples += batch_size
        
        preds = model.forward(x_batch)
        loss = cross_entropy(preds, y_batch)
        acc = accuracy(preds, y_batch)
        
        test_loss += loss * batch_size
        test_acc += acc * batch_size
    
    print(f"\n>> Evaluation: loss:{test_loss/test_samples:.3f} "
          f"acc:{test_acc/test_samples:.3f}")
```

### 9.3.7 Key Improvements

```python
def compare_v2_v3():
    """Version 2와 Version 3 비교"""
    
    print("\n" + "=" * 70)
    print("Version 2 vs Version 3 Comparison")
    print("=" * 70)
    
    comparison = {
        "Network Definition": {
            "Version 2": "개별 변수 (w1, b1, w2, b2, w3, b3)",
            "Version 3": "모델 클래스 (MLP)"
        },
        "Forward Pass": {
            "Version 2": "명시적 계산 (z1=x@w1+b1, a1=sigmoid(z1), ...)",
            "Version 3": "model.forward(x) 호출"
        },
        "Backward Pass": {
            "Version 2": "수동 그래디언트 계산 (grad_z3, grad_w3, ...)",
            "Version 3": "model.backward(grad) 호출"
        },
        "Parameter Management": {
            "Version 2": "수동으로 각 파라미터 업데이트",
            "Version 3": "model.parameters()로 자동 관리"
        },
        "Layer Addition": {
            "Version 2": "모든 곳에 코드 추가 필요",
            "Version 3": "Sequential에 레이어 추가만"
        },
        "Reusability": {
            "Version 2": "레이어 재사용 불가",
            "Version 3": "모듈 재사용 가능"
        },
        "Flexibility": {
            "Version 2": "구조 변경 어려움",
            "Version 3": "구조 변경 쉬움"
        }
    }
    
    for aspect, versions in comparison.items():
        print(f"\n[{aspect}]")
        for version, description in versions.items():
            print(f"  {version}: {description}")
    
    print("=" * 70)

compare_v2_v3()
```

```
======================================================================
Version 2 vs Version 3 Comparison
======================================================================

[Network Definition]
  Version 2: 개별 변수 (w1, b1, w2, b2, w3, b3)
  Version 3: 모델 클래스 (MLP)

[Forward Pass]
  Version 2: 명시적 계산 (z1=x@w1+b1, a1=sigmoid(z1), ...)
  Version 3: model.forward(x) 호출

[Backward Pass]
  Version 2: 수동 그래디언트 계산 (grad_z3, grad_w3, ...)
  Version 3: model.backward(grad) 호출

[Parameter Management]
  Version 2: 수동으로 각 파라미터 업데이트
  Version 3: model.parameters()로 자동 관리

[Layer Addition]
  Version 2: 모든 곳에 코드 추가 필요
  Version 3: Sequential에 레이어 추가만

[Reusability]
  Version 2: 레이어 재사용 불가
  Version 3: 모듈 재사용 가능

[Flexibility]
  Version 2: 구조 변경 어려움
  Version 3: 구조 변경 쉬움
======================================================================
```

### 9.3.8 Summary

| 항목 | Version 2 | Version 3 |
|------|----------|----------|
| **코드 라인 수** | ~185 lines | ~250 lines |
| **클래스 수** | 1 (DataLoader) | 7 (Module, Linear, Sigmoid, etc.) |
| **추상화 수준** | 낮음 | 높음 |
| **레이어 재사용** | 불가 | 가능 |
| **네트워크 정의** | 수동 | 선언적 (Sequential) |
| **파라미터 관리** | 수동 | 자동 (parameters()) |
| **확장성** | 낮음 | 높음 |
| **유지보수성** | 낮음 | 높음 |

**핵심 개선사항:**

1. **모듈화**: 레이어를 독립적인 클래스로 캡슐화
2. **재사용성**: 동일한 레이어를 여러 곳에서 사용 가능
3. **확장성**: 새로운 레이어 추가 용이
4. **파라미터 관리**: 자동으로 파라미터 수집 및 관리
5. **PyTorch 스타일**: nn.Module 패턴 적용

**다음 단계 (Version 4):**

옵티마이저를 분리하여 다양한 최적화 알고리즘을 쉽게 적용할 수 있도록 개선합니다.
