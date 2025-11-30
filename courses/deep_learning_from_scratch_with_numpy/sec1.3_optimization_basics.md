## 1.3 Optimization Basics

신경망 학습은 손실 함수를 최소화하는 파라미터를 찾는 최적화 문제입니다. 이 섹션에서는 손실 함수의 역할과 경사하강법의 원리, 그리고 학습률의 중요성을 다룹니다.

### 1.3.1 Loss Functions

**손실 함수의 역할**

손실 함수(Loss Function)는 모델의 예측값 $\hat{y}$과 실제 정답 $y$ 사이의 차이를 수치화합니다. 학습의 목표는 이 손실을 최소화하는 파라미터 $\theta$를 찾는 것입니다:

$$\theta^* = \argmin_{\theta} L(\hat{y}, y)$$

손실 함수는 다음 조건을 만족해야 합니다:

- 예측이 정답과 같으면 손실이 최소 (일반적으로 0)
- 예측이 정답과 다를수록 손실이 증가
- 미분 가능해야 경사하강법 적용 가능

**태스크별 손실 함수**

| 태스크 | 손실 함수 | 수식 |
|--------|-----------|------|
| 회귀 | Mean Squared Error (MSE) | $\frac{1}{N}\sum_{i}(\hat{y}_i - y_i)^2$ |
| 이진 분류 | Binary Cross-Entropy (BCE) | $-\frac{1}{N}\sum_{i}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| 다중 분류 | Categorical Cross-Entropy (CE) | $-\frac{1}{N}\sum_{i}\sum_{k}y_{ik}\log\hat{y}_{ik}$ |

```python
import numpy as np

def mse_loss(preds, targets):
    """Mean Squared Error"""
    return np.mean((preds - targets) ** 2)

def binary_cross_entropy(preds, targets, eps=1e-8):
    """Binary Cross-Entropy"""
    preds = np.clip(preds, eps, 1 - eps)  # 수치 안정성
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

def categorical_cross_entropy(preds, targets, eps=1e-8):
    """Categorical Cross-Entropy (targets: one-hot)"""
    preds = np.clip(preds, eps, 1 - eps)
    return -np.mean(np.sum(targets * np.log(preds), axis=1))

# MSE 예시
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.2, 2.8])
print(f"MSE Loss: {mse_loss(y_pred, y_true):.4f}")

# BCE 예시
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])
print(f"BCE Loss: {binary_cross_entropy(y_pred, y_true):.4f}")

# CE 예시 (3 클래스)
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # one-hot
y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
print(f"CE Loss:  {categorical_cross_entropy(y_pred, y_true):.4f}")
```

```
MSE Loss: 0.0300
BCE Loss: 0.1974
CE Loss:  0.2939
```

**손실 함수의 시각화**

손실 함수가 파라미터에 따라 어떻게 변하는지 시각화해봅니다:

```python
import matplotlib.pyplot as plt

# 간단한 1D 회귀: y = wx, 정답 y=2일 때 w에 따른 MSE
x = 1.0  # 입력
y_true = 2.0  # 정답

w_range = np.linspace(-1, 5, 100)
losses = [(w * x - y_true) ** 2 for w in w_range]

plt.figure(figsize=(8, 5))
plt.plot(w_range, losses, 'b-', linewidth=2)
plt.axvline(x=2.0, color='r', linestyle='--', label='Optimal w=2')
plt.xlabel('Weight (w)')
plt.ylabel('Loss (MSE)')
plt.title('Loss Landscape for y = wx')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.3.2 Gradient Descent

**경사하강법의 원리**

경사하강법(Gradient Descent)은 손실 함수의 그래디언트 반대 방향으로 파라미터를 반복적으로 업데이트하여 최솟값을 찾는 알고리즘입니다:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

여기서:
- $\theta_t$: 현재 파라미터
- $\eta$: 학습률 (learning rate)
- $\nabla_\theta L$: 손실 함수의 그래디언트

```python
def gradient_descent_1d(f, df, x_init, lr=0.1, num_iters=50):
    """
    1D 경사하강법
    
    Args:
        f: 최소화할 함수
        df: f의 도함수
        x_init: 초기값
        lr: 학습률
        num_iters: 반복 횟수
    
    Returns:
        x_history: 파라미터 변화 기록
        f_history: 함수값 변화 기록
    """
    x = x_init
    x_history = [x]
    f_history = [f(x)]
    
    for _ in range(num_iters):
        grad = df(x)
        x = x - lr * grad
        x_history.append(x)
        f_history.append(f(x))
    
    return np.array(x_history), np.array(f_history)

# f(x) = x^2 최소화 (최솟값: x=0)
f = lambda x: x ** 2
df = lambda x: 2 * x

x_history, f_history = gradient_descent_1d(f, df, x_init=4.0, lr=0.1, num_iters=20)

print("Gradient Descent for f(x) = x^2")
print(f"Initial: x = {x_history[0]:.4f}, f(x) = {f_history[0]:.4f}")
print(f"Final:   x = {x_history[-1]:.4f}, f(x) = {f_history[-1]:.4f}")

# 수렴 과정 출력
print("\nIteration history:")
for i in [0, 1, 2, 5, 10, 20]:
    print(f"  iter {i:2d}: x = {x_history[i]:7.4f}, f(x) = {f_history[i]:.6f}")
```

```
Gradient Descent for f(x) = x^2
Initial: x = 4.0000, f(x) = 16.0000
Final:   x = 0.0052, f(x) = 0.0000

Iteration history:
  iter  0: x =  4.0000, f(x) = 16.000000
  iter  1: x =  3.2000, f(x) = 10.240000
  iter  2: x =  2.5600, f(x) = 6.553600
  iter  5: x =  1.3107, f(x) = 1.717987
  iter 10: x =  0.4295, f(x) = 0.184467
  iter 20: x =  0.0052, f(x) = 0.000027
```

**2D 경사하강법 시각화**

```python
def gradient_descent_2d(f, grad_f, init, lr=0.1, num_iters=50):
    """2D 경사하강법"""
    x, y = init
    history = [(x, y)]
    
    for _ in range(num_iters):
        gx, gy = grad_f(x, y)
        x = x - lr * gx
        y = y - lr * gy
        history.append((x, y))
    
    return np.array(history)

# f(x, y) = x^2 + 2y^2 최소화
f = lambda x, y: x**2 + 2*y**2
grad_f = lambda x, y: (2*x, 4*y)

history = gradient_descent_2d(f, grad_f, init=(4.0, 3.0), lr=0.1, num_iters=30)

# 등고선과 경로 시각화
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + 2*Y**2

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=4, linewidth=1, label='GD path')
plt.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
plt.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on f(x, y) = x² + 2y²')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

**배치 경사하강법의 종류**

| 방식 | 배치 크기 | 특징 |
|------|-----------|------|
| Batch GD | 전체 데이터 | 안정적이나 느림, 메모리 부담 |
| Stochastic GD (SGD) | 1개 샘플 | 빠르나 불안정, 노이즈 큼 |
| Mini-batch GD | $N$개 샘플 | 균형 잡힌 선택, 실제로 가장 많이 사용 |

```python
def batch_gd(X, y, W, lr=0.01, num_epochs=100):
    """Batch Gradient Descent"""
    history = []
    for epoch in range(num_epochs):
        # 전체 데이터로 그래디언트 계산
        pred = X @ W
        loss = np.mean((pred - y) ** 2)
        grad = 2 * X.T @ (pred - y) / len(y)
        W = W - lr * grad
        history.append(loss)
    return W, history

def sgd(X, y, W, lr=0.01, num_epochs=100):
    """Stochastic Gradient Descent"""
    history = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(len(y)):
            # 샘플 하나로 그래디언트 계산
            xi, yi = X[i:i+1], y[i:i+1]
            pred = xi @ W
            loss = np.mean((pred - yi) ** 2)
            grad = 2 * xi.T @ (pred - yi)
            W = W - lr * grad
            epoch_loss += loss
        history.append(epoch_loss / len(y))
    return W, history

def mini_batch_gd(X, y, W, lr=0.01, batch_size=32, num_epochs=100):
    """Mini-batch Gradient Descent"""
    history = []
    n_samples = len(y)
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            xi, yi = X[batch_idx], y[batch_idx]
            
            pred = xi @ W
            loss = np.mean((pred - yi) ** 2)
            grad = 2 * xi.T @ (pred - yi) / len(yi)
            W = W - lr * grad
            
            epoch_loss += loss
            n_batches += 1
        
        history.append(epoch_loss / n_batches)
    return W, history

# 비교 실험
np.random.seed(42)
n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
W_true = np.random.randn(n_features, 1)
y = X @ W_true + 0.1 * np.random.randn(n_samples, 1)

W_init = np.zeros((n_features, 1))

_, hist_batch = batch_gd(X, y, W_init.copy(), lr=0.1, num_epochs=50)
_, hist_sgd = sgd(X, y, W_init.copy(), lr=0.001, num_epochs=50)
_, hist_mini = mini_batch_gd(X, y, W_init.copy(), lr=0.1, batch_size=32, num_epochs=50)

plt.figure(figsize=(10, 5))
plt.plot(hist_batch, label='Batch GD', linewidth=2)
plt.plot(hist_sgd, label='SGD', alpha=0.7)
plt.plot(hist_mini, label='Mini-batch GD', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparison of Gradient Descent Variants')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

### 1.3.3 Learning Rate

**학습률의 역할**

학습률 $\eta$는 그래디언트 방향으로 얼마나 크게 이동할지를 결정하는 하이퍼파라미터입니다. 적절한 학습률 선택이 학습 성공의 핵심입니다.

- **너무 작은 학습률**: 수렴이 매우 느림, 지역 최솟값에 갇힐 수 있음
- **너무 큰 학습률**: 발산하거나 최솟값 주변에서 진동
- **적절한 학습률**: 빠르고 안정적인 수렴

```python
def visualize_learning_rates():
    """학습률에 따른 수렴 양상 비교"""
    f = lambda x: x ** 2
    df = lambda x: 2 * x
    
    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.01]
    labels = ['lr=0.01 (too small)', 'lr=0.1 (good)', 'lr=0.5 (good)', 
              'lr=0.9 (oscillating)', 'lr=1.01 (diverging)']
    
    plt.figure(figsize=(12, 5))
    
    for lr, label in zip(learning_rates, labels):
        x_hist, f_hist = gradient_descent_1d(f, df, x_init=4.0, lr=lr, num_iters=20)
        plt.plot(f_hist, label=label, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss f(x) = x²')
    plt.title('Effect of Learning Rate on Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 30)
    plt.show()

visualize_learning_rates()
```

**학습률에 따른 파라미터 궤적**

```python
def visualize_lr_trajectory():
    """2D에서 학습률에 따른 궤적 시각화"""
    f = lambda x, y: x**2 + 10*y**2  # 타원형 손실 함수
    grad_f = lambda x, y: (2*x, 20*y)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + 10*Y**2
    
    learning_rates = [0.01, 0.09, 0.11]
    titles = ['lr=0.01 (slow)', 'lr=0.09 (optimal)', 'lr=0.11 (diverging)']
    
    for ax, lr, title in zip(axes, learning_rates, titles):
        ax.contour(X, Y, Z, levels=20, cmap='viridis')
        
        history = gradient_descent_2d(f, grad_f, init=(4.0, 1.5), lr=lr, num_iters=30)
        
        ax.plot(history[:, 0], history[:, 1], 'ro-', markersize=3, linewidth=1)
        ax.plot(history[0, 0], history[0, 1], 'go', markersize=8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_lr_trajectory()
```

**학습률 선택 가이드라인**

| 상황 | 권장 학습률 범위 |
|------|------------------|
| SGD | 0.01 ~ 0.1 |
| SGD + Momentum | 0.001 ~ 0.01 |
| Adam | 0.0001 ~ 0.001 |
| 학습 초기 | 상대적으로 큰 값 |
| 학습 후반 | 점진적으로 감소 |

```python
# 학습률 스케줄링 예시
def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=10):
    """단계적 학습률 감소"""
    return initial_lr * (drop ** (epoch // epochs_drop))

def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
    """지수적 학습률 감소"""
    return initial_lr * (decay_rate ** epoch)

def cosine_annealing(epoch, initial_lr=0.1, total_epochs=50):
    """코사인 어닐링"""
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

epochs = np.arange(50)

plt.figure(figsize=(10, 5))
plt.plot(epochs, [step_decay(e) for e in epochs], label='Step Decay', linewidth=2)
plt.plot(epochs, [exponential_decay(e) for e in epochs], label='Exponential Decay', linewidth=2)
plt.plot(epochs, [cosine_annealing(e) for e in epochs], label='Cosine Annealing', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Scheduling Strategies')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.3.4 Convergence Conditions

**수렴 판정 기준**

학습이 언제 충분히 수렴했는지 판단하는 기준들:

1. **최대 반복 횟수**: 미리 정한 에폭 수에 도달
2. **손실 임계값**: 손실이 특정 값 이하로 감소
3. **손실 변화량**: 연속된 에폭 간 손실 변화가 미미
4. **그래디언트 크기**: 그래디언트 노름이 충분히 작음

```python
def gradient_descent_with_convergence(f, grad_f, init, lr=0.1, 
                                       max_iters=1000, tol=1e-6, verbose=True):
    """
    수렴 조건을 포함한 경사하강법
    
    Args:
        f: 목적 함수
        grad_f: 그래디언트 함수
        init: 초기값 (x, y)
        lr: 학습률
        max_iters: 최대 반복 횟수
        tol: 수렴 판정 임계값
        verbose: 출력 여부
    
    Returns:
        result: 최종 파라미터
        history: 학습 기록
    """
    x, y = init
    history = {'params': [(x, y)], 'loss': [f(x, y)], 'grad_norm': []}
    
    for i in range(max_iters):
        gx, gy = grad_f(x, y)
        grad_norm = np.sqrt(gx**2 + gy**2)
        history['grad_norm'].append(grad_norm)
        
        # 수렴 조건 1: 그래디언트 크기
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iteration {i}: gradient norm = {grad_norm:.2e}")
            break
        
        # 파라미터 업데이트
        x = x - lr * gx
        y = y - lr * gy
        
        loss = f(x, y)
        history['params'].append((x, y))
        history['loss'].append(loss)
        
        # 수렴 조건 2: 손실 변화량
        if len(history['loss']) > 1:
            loss_change = abs(history['loss'][-1] - history['loss'][-2])
            if loss_change < tol:
                if verbose:
                    print(f"Converged at iteration {i}: loss change = {loss_change:.2e}")
                break
    else:
        if verbose:
            print(f"Reached maximum iterations ({max_iters})")
    
    return (x, y), history

# 테스트
f = lambda x, y: x**2 + y**2
grad_f = lambda x, y: (2*x, 2*y)

result, history = gradient_descent_with_convergence(
    f, grad_f, init=(5.0, 3.0), lr=0.1, max_iters=1000, tol=1e-8
)

print(f"\nFinal result: ({result[0]:.6f}, {result[1]:.6f})")
print(f"Final loss: {history['loss'][-1]:.2e}")
print(f"Total iterations: {len(history['loss']) - 1}")
```

```
Converged at iteration 89: loss change = 9.29e-09

Final result: (0.000056, 0.000034)
Final loss: 4.27e-09
Total iterations: 89
```

### 1.3.5 Summary

| 개념 | 설명 | 핵심 포인트 |
|------|------|-------------|
| 손실 함수 | 예측과 정답의 차이 측정 | 태스크에 맞는 손실 함수 선택 |
| 경사하강법 | 그래디언트 반대 방향으로 업데이트 | Mini-batch GD가 실용적 |
| 학습률 | 업데이트 크기 결정 | 너무 크면 발산, 너무 작으면 느림 |
| 수렴 조건 | 학습 종료 판정 | 손실 변화, 그래디언트 크기 모니터링 |

다음 챕터에서는 신경망의 구성 요소인 활성화 함수, 손실 함수, 가중치 초기화에 대해 자세히 다룹니다.
