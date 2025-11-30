## 9.4 Version 4 - Optimizer

Version 4에서는 파라미터 업데이트 로직을 Optimizer 클래스로 분리하여 다양한 최적화 알고리즘을 쉽게 적용할 수 있도록 합니다. PyTorch의 `torch.optim`과 유사한 인터페이스를 구현합니다.

### 9.4.1 Overview

**Version 4의 개선 사항:**

- Optimizer 베이스 클래스 도입
- SGD, Momentum, Adam 등 다양한 옵티마이저 구현
- 파라미터 업데이트 로직 캡슐화
- 학습률 스케줄링 지원 준비

**Version 3 → Version 4 변화:**

```python
# Version 3: 수동 파라미터 업데이트
for param in model.parameters():
    param['weight'] -= learning_rate * param['grad']

# Version 4: Optimizer 사용
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.step()
```

### 9.4.2 Optimizer Base Class

```python
import numpy as np

class Optimizer:
    """
    모든 옵티마이저의 베이스 클래스
    PyTorch의 torch.optim.Optimizer와 유사
    """
    
    def __init__(self, parameters, lr=0.01):
        """
        Parameters:
        -----------
        parameters : list of dict
            model.parameters()로 얻은 파라미터 리스트
        lr : float
            학습률
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """
        파라미터 업데이트 (서브클래스에서 구현)
        """
        raise NotImplementedError
    
    def zero_grad(self):
        """
        모든 그래디언트를 0으로 초기화
        """
        for param in self.parameters:
            param['grad'].fill(0)
```

### 9.4.3 SGD Optimizer

```python
class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    
    weight = weight - lr * grad
    """
    
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        """
        Parameters:
        -----------
        parameters : list of dict
            파라미터 리스트
        lr : float
            학습률
        momentum : float
            모멘텀 계수 (0이면 일반 SGD)
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        
        # 모멘텀을 위한 velocity 초기화
        if momentum > 0:
            self.velocities = []
            for param in self.parameters:
                velocity = np.zeros_like(param['weight'])
                self.velocities.append(velocity)
    
    def step(self):
        """파라미터 업데이트"""
        if self.momentum == 0:
            # 일반 SGD
            for param in self.parameters:
                param['weight'] -= self.lr * param['grad']
        else:
            # Momentum SGD
            for i, param in enumerate(self.parameters):
                # v = momentum * v - lr * grad
                self.velocities[i] = (self.momentum * self.velocities[i] - 
                                     self.lr * param['grad'])
                # weight = weight + v
                param['weight'] += self.velocities[i]


# 사용 예시
if __name__ == "__main__":
    # 더미 파라미터 생성
    dummy_params = [
        {'name': 'weight', 'weight': np.random.randn(5, 3), 
         'grad': np.random.randn(5, 3)},
        {'name': 'bias', 'weight': np.random.randn(3), 
         'grad': np.random.randn(3)}
    ]
    
    print("=" * 70)
    print("SGD Optimizer Example")
    print("=" * 70)
    
    # SGD without momentum
    optimizer_sgd = SGD(dummy_params, lr=0.1, momentum=0.0)
    print("\n[Before update]")
    print(f"Weight[0,0]: {dummy_params[0]['weight'][0, 0]:.4f}")
    
    optimizer_sgd.step()
    print("\n[After SGD update]")
    print(f"Weight[0,0]: {dummy_params[0]['weight'][0, 0]:.4f}")
    
    # Reset
    dummy_params[0]['weight'] = np.random.randn(5, 3)
    dummy_params[0]['grad'] = np.random.randn(5, 3)
    
    # SGD with momentum
    optimizer_momentum = SGD(dummy_params, lr=0.1, momentum=0.9)
    print("\n[SGD with Momentum]")
    print(f"Before: {dummy_params[0]['weight'][0, 0]:.4f}")
    
    optimizer_momentum.step()
    print(f"After:  {dummy_params[0]['weight'][0, 0]:.4f}")
```

```
======================================================================
SGD Optimizer Example
======================================================================

[Before update]
Weight[0,0]: 0.4967

[After SGD update]
Weight[0,0]: 0.4828

[SGD with Momentum]
Before: -0.2342
After:  -0.2483
```

### 9.4.4 Adam Optimizer

```python
class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) Optimizer
    
    Reference:
    Kingma & Ba, 2014: "Adam: A Method for Stochastic Optimization"
    """
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Parameters:
        -----------
        parameters : list of dict
            파라미터 리스트
        lr : float
            학습률
        betas : tuple of float
            (beta1, beta2) - 모멘트 추정 지수 감쇠율
        eps : float
            수치 안정성을 위한 작은 값
        """
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # 1차 모멘트 (평균)와 2차 모멘트 (분산) 초기화
        self.m = []  # 1st moment vector
        self.v = []  # 2nd moment vector
        
        for param in self.parameters:
            self.m.append(np.zeros_like(param['weight']))
            self.v.append(np.zeros_like(param['weight']))
        
        # 타임스텝
        self.t = 0
    
    def step(self):
        """파라미터 업데이트"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            grad = param['grad']
            
            # 1차 및 2차 모멘트 업데이트
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 파라미터 업데이트
            param['weight'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# 사용 예시
if __name__ == "__main__":
    dummy_params = [
        {'name': 'weight', 'weight': np.random.randn(5, 3), 
         'grad': np.random.randn(5, 3)}
    ]
    
    print("\n" + "=" * 70)
    print("Adam Optimizer Example")
    print("=" * 70)
    
    optimizer = Adam(dummy_params, lr=0.001)
    
    print(f"\n[Initial weight]")
    print(f"Weight[0,0]: {dummy_params[0]['weight'][0, 0]:.6f}")
    
    # 여러 스텝 업데이트
    for step in range(1, 4):
        dummy_params[0]['grad'] = np.random.randn(5, 3)
        optimizer.step()
        print(f"[Step {step}] Weight[0,0]: {dummy_params[0]['weight'][0, 0]:.6f}")
```

```
======================================================================
Adam Optimizer Example
======================================================================

[Initial weight]
Weight[0,0]: 0.496714

[Step 1] Weight[0,0]: 0.495714
[Step 2] Weight[0,0]: 0.494721
[Step 3] Weight[0,0]: 0.493735
```

### 9.4.5 RMSprop Optimizer

```python
class RMSprop(Optimizer):
    """
    RMSprop Optimizer
    
    Root Mean Square Propagation
    """
    
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8):
        """
        Parameters:
        -----------
        parameters : list of dict
            파라미터 리스트
        lr : float
            학습률
        alpha : float
            이동 평균 지수 감쇠율
        eps : float
            수치 안정성을 위한 작은 값
        """
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        
        # 제곱 그래디언트의 이동 평균
        self.square_avg = []
        for param in self.parameters:
            self.square_avg.append(np.zeros_like(param['weight']))
    
    def step(self):
        """파라미터 업데이트"""
        for i, param in enumerate(self.parameters):
            grad = param['grad']
            
            # 제곱 그래디언트의 이동 평균 업데이트
            self.square_avg[i] = (self.alpha * self.square_avg[i] + 
                                  (1 - self.alpha) * (grad ** 2))
            
            # 파라미터 업데이트
            param['weight'] -= self.lr * grad / (np.sqrt(self.square_avg[i]) + self.eps)
```

### 9.4.6 Complete Version 4 Code

```python
import os
import numpy as np
import gzip

#################################################################
## Optimizer Classes
#################################################################

class Optimizer:
    """베이스 옵티마이저"""
    
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for param in self.parameters:
            param['grad'].fill(0)


class SGD(Optimizer):
    """SGD with optional momentum"""
    
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        
        if momentum > 0:
            self.velocities = []
            for param in self.parameters:
                self.velocities.append(np.zeros_like(param['weight']))
    
    def step(self):
        if self.momentum == 0:
            for param in self.parameters:
                param['weight'] -= self.lr * param['grad']
        else:
            for i, param in enumerate(self.parameters):
                self.velocities[i] = (self.momentum * self.velocities[i] - 
                                     self.lr * param['grad'])
                param['weight'] += self.velocities[i]


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.m = []
        self.v = []
        for param in self.parameters:
            self.m.append(np.zeros_like(param['weight']))
            self.v.append(np.zeros_like(param['weight']))
        
        self.t = 0
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            grad = param['grad']
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param['weight'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


#################################################################
## Module Classes (from Version 3)
#################################################################

class Module:
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
    def __init__(self, in_features, out_features):
        super().__init__()
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
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, x):
        self.out = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return self.out
    
    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


class Softmax(Module):
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
## Training Script with Optimizer
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
    print(f"\n>> Model created")
    
    # Create Optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    print(f">> Optimizer: Adam (lr=0.001)")
    
    # Training
    num_epochs = 10
    
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
            
            # Update with Optimizer
            optimizer.step()
            
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

### 9.4.7 Optimizer Comparison

```python
def compare_optimizers():
    """다양한 옵티마이저 비교"""
    
    print("\n" + "=" * 70)
    print("Optimizer Comparison")
    print("=" * 70)
    
    comparison = {
        "SGD": {
            "Update Rule": "w = w - lr * grad",
            "Memory": "O(n) - parameters only",
            "Hyperparameters": "lr, momentum (optional)",
            "Pros": "Simple, stable",
            "Cons": "Slow convergence, same lr for all params"
        },
        "SGD + Momentum": {
            "Update Rule": "v = momentum*v - lr*grad; w = w + v",
            "Memory": "O(2n) - parameters + velocities",
            "Hyperparameters": "lr, momentum",
            "Pros": "Faster convergence, escapes local minima",
            "Cons": "Additional hyperparameter"
        },
        "RMSprop": {
            "Update Rule": "w = w - lr * grad / sqrt(square_avg)",
            "Memory": "O(2n) - parameters + square_avg",
            "Hyperparameters": "lr, alpha",
            "Pros": "Adaptive lr per parameter",
            "Cons": "Can be unstable"
        },
        "Adam": {
            "Update Rule": "Combines momentum + RMSprop",
            "Memory": "O(3n) - parameters + m + v",
            "Hyperparameters": "lr, beta1, beta2",
            "Pros": "Fast, adaptive, generally works well",
            "Cons": "More memory, more hyperparameters"
        }
    }
    
    for opt_name, properties in comparison.items():
        print(f"\n[{opt_name}]")
        for key, value in properties.items():
            print(f"  {key:<20} {value}")
    
    print("=" * 70)

compare_optimizers()
```

```
======================================================================
Optimizer Comparison
======================================================================

[SGD]
  Update Rule          w = w - lr * grad
  Memory               O(n) - parameters only
  Hyperparameters      lr, momentum (optional)
  Pros                 Simple, stable
  Cons                 Slow convergence, same lr for all params

[SGD + Momentum]
  Update Rule          v = momentum*v - lr*grad; w = w + v
  Memory               O(2n) - parameters + velocities
  Hyperparameters      lr, momentum
  Pros                 Faster convergence, escapes local minima
  Cons                 Additional hyperparameter

[RMSprop]
  Update Rule          w = w - lr * grad / sqrt(square_avg)
  Memory               O(2n) - parameters + square_avg
  Hyperparameters      lr, alpha
  Pros                 Adaptive lr per parameter
  Cons                 Can be unstable

[Adam]
  Update Rule          Combines momentum + RMSprop
  Memory               O(3n) - parameters + m + v
  Hyperparameters      lr, beta1, beta2
  Pros                 Fast, adaptive, generally works well
  Cons                 More memory, more hyperparameters
======================================================================
```

### 9.4.8 Key Improvements

```python
def compare_v3_v4():
    """Version 3과 Version 4 비교"""
    
    print("\n" + "=" * 70)
    print("Version 3 vs Version 4 Comparison")
    print("=" * 70)
    
    comparison = {
        "Parameter Update": {
            "Version 3": "Manual loop: param['weight'] -= lr * param['grad']",
            "Version 4": "optimizer.step()"
        },
        "Optimization Algorithm": {
            "Version 3": "Only SGD (하드코딩)",
            "Version 4": "SGD, Momentum, RMSprop, Adam (선택 가능)"
        },
        "Learning Rate": {
            "Version 3": "Global variable",
            "Version 4": "Optimizer에 캡슐화"
        },
        "Flexibility": {
            "Version 3": "알고리즘 변경 어려움",
            "Version 4": "Optimizer만 교체"
        },
        "Code Reusability": {
            "Version 3": "업데이트 로직 재사용 불가",
            "Version 4": "Optimizer 재사용 가능"
        },
        "Advanced Features": {
            "Version 3": "구현 어려움",
            "Version 4": "쉽게 추가 가능 (lr scheduling, etc.)"
        }
    }
    
    for aspect, versions in comparison.items():
        print(f"\n[{aspect}]")
        for version, description in versions.items():
            print(f"  {version}: {description}")
    
    print("=" * 70)

compare_v3_v4()
```

```
======================================================================
Version 3 vs Version 4 Comparison
======================================================================

[Parameter Update]
  Version 3: Manual loop: param['weight'] -= lr * param['grad']
  Version 4: optimizer.step()

[Optimization Algorithm]
  Version 3: Only SGD (하드코딩)
  Version 4: SGD, Momentum, RMSprop, Adam (선택 가능)

[Learning Rate]
  Version 3: Global variable
  Version 4: Optimizer에 캡슐화

[Flexibility]
  Version 3: 알고리즘 변경 어려움
  Version 4: Optimizer만 교체

[Code Reusability]
  Version 3: 업데이트 로직 재사용 불가
  Version 4: Optimizer 재사용 가능

[Advanced Features]
  Version 3: 구현 어려움
  Version 4: 쉽게 추가 가능 (lr scheduling, etc.)
======================================================================
```

### 9.4.9 Summary

| 항목 | Version 3 | Version 4 |
|------|----------|----------|
| **코드 라인 수** | ~250 lines | ~320 lines |
| **클래스 수** | 7 | 10 (+ 3 Optimizers) |
| **옵티마이저** | SGD (하드코딩) | SGD, Adam, RMSprop |
| **업데이트 방식** | 수동 루프 | optimizer.step() |
| **확장성** | 중간 | 높음 |
| **유연성** | 중간 | 높음 |
| **PyTorch 유사도** | 중간 | 높음 |

**핵심 개선사항:**

1. **옵티마이저 추상화**: 파라미터 업데이트 로직 캡슐화
2. **다양한 알고리즘**: SGD, Momentum, RMSprop, Adam 지원
3. **간단한 인터페이스**: optimizer.step() 한 줄로 업데이트
4. **확장 용이**: 새로운 옵티마이저 추가 쉬움
5. **PyTorch 스타일**: torch.optim 패턴 적용

**Optimizer 인터페이스:**

```python
# 옵티마이저 생성
optimizer = Adam(model.parameters(), lr=0.001)

# 학습 루프
for x_batch, y_batch in train_loader:
    # Forward & Loss
    preds = model.forward(x_batch)
    loss = loss_fn(preds, y_batch)
    
    # Backward
    model.backward(grad)
    
    # Update
    optimizer.step()  # ← 간단!
```

**다음 단계 (Version 5):**

ReLU 활성화 함수와 Adam 옵티마이저를 기본으로 사용하여 성능을 개선합니다.
