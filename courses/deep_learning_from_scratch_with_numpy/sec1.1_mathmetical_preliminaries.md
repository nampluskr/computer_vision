## 1.1 Linear Algebra Essentials

신경망의 연산은 본질적으로 선형대수 연산입니다. 입력 데이터, 가중치, 활성화 값 모두 벡터 또는 행렬로 표현되며, 순전파와 역전파 과정은 행렬 곱셈과 원소별 연산의 조합으로 이루어집니다. 이 섹션에서는 신경망 구현에 필요한 핵심 선형대수 개념을 다룹니다.

### 1.1.1 Scalars, Vectors, Matrices, and Tensors

딥러닝에서 다루는 데이터는 차원에 따라 다음과 같이 분류됩니다:

| 이름 | 차원 | 표기 | 예시 |
|------|------|------|------|
| 스칼라 (Scalar) | 0차원 | $x$ | 손실 값: $L = 0.523$ |
| 벡터 (Vector) | 1차원 | $\mathbf{x}$ | 편향: $\mathbf{b} = [0.1, 0.2, 0.3]$ |
| 행렬 (Matrix) | 2차원 | $\mathbf{X}$ | 가중치: $\mathbf{W} \in \mathbb{R}^{784 \times 100}$ |
| 텐서 (Tensor) | 3차원 이상 | $\mathcal{X}$ | 이미지 배치: $(N, H, W)$ |

NumPy에서는 모두 `ndarray`로 표현되며, `ndim` 속성으로 차원을 확인할 수 있습니다:

```python
import numpy as np

scalar = np.array(0.523)        # ndim = 0, shape = ()
vector = np.array([0.1, 0.2])   # ndim = 1, shape = (2,)
matrix = np.random.randn(3, 4)  # ndim = 2, shape = (3, 4)
tensor = np.random.randn(2,3,4) # ndim = 3, shape = (2, 3, 4)

print(f"scalar: ndim={scalar.ndim}, shape={scalar.shape}")
print(f"vector: ndim={vector.ndim}, shape={vector.shape}")
print(f"matrix: ndim={matrix.ndim}, shape={matrix.shape}")
print(f"tensor: ndim={tensor.ndim}, shape={tensor.shape}")
```

```
scalar: ndim=0, shape=()
vector: ndim=1, shape=(2,)
matrix: ndim=2, shape=(3, 4)
tensor: ndim=3, shape=(2, 3, 4)
```

### 1.1.2 Matrix Multiplication

신경망의 선형 변환 $z = xW + b$에서 핵심 연산은 행렬 곱셈입니다.

행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$와 $\mathbf{B} \in \mathbb{R}^{n \times p}$의 곱 $\mathbf{C} = \mathbf{A}\mathbf{B}$는 다음과 같이 정의됩니다:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

결과 행렬 $\mathbf{C}$의 shape은 $(m, p)$입니다. 이때 $\mathbf{A}$의 열 수와 $\mathbf{B}$의 행 수가 같아야 합니다 ($n$).

```python
A = np.random.randn(3, 4)  # (3, 4)
B = np.random.randn(4, 5)  # (4, 5)
C = np.dot(A, B)           # (3, 5)

print(f"A: {A.shape}")
print(f"B: {B.shape}")
print(f"C = A @ B: {C.shape}")
```

```
A: (3, 4)
B: (4, 5)
C = A @ B: (3, 5)
```

신경망에서의 예시를 살펴보면, 배치 크기 $N=64$, 입력 차원 $D_{in}=784$, 출력 차원 $D_{out}=100$일 때:

$$\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$$

```python
N, D_in, D_out = 64, 784, 100

x = np.random.randn(N, D_in)     # 입력: (64, 784)
W = np.random.randn(D_in, D_out) # 가중치: (784, 100)
b = np.zeros(D_out)              # 편향: (100,)

z = np.dot(x, W) + b             # 출력: (64, 100)
print(f"x: {x.shape}, W: {W.shape}, b: {b.shape} -> z: {z.shape}")
```

```
x: (64, 784), W: (784, 100), b: (100,) -> z: (64, 100)
```

### 1.1.3 Transpose

행렬의 전치는 행과 열을 교환하는 연산입니다. 행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$의 전치 $\mathbf{A}^T \in \mathbb{R}^{n \times m}$는 다음을 만족합니다:

$$(A^T)_{ij} = A_{ji}$$

역전파에서 그래디언트 계산 시 전치가 빈번하게 사용됩니다:

```python
A = np.random.randn(3, 4)
print(f"A: {A.shape}")
print(f"A.T: {A.T.shape}")
```

```
A: (3, 4)
A.T: (4, 3)
```

역전파에서의 활용 예시:

```python
# Forward: z = x @ W
# Backward: grad_W = x.T @ grad_z

x = np.random.randn(64, 784)      # (N, D_in)
grad_z = np.random.randn(64, 100) # (N, D_out)

grad_W = np.dot(x.T, grad_z)      # (D_in, N) @ (N, D_out) = (D_in, D_out)
print(f"x.T: {x.T.shape}, grad_z: {grad_z.shape} -> grad_W: {grad_W.shape}")
```

```
x.T: (784, 64), grad_z: (64, 100) -> grad_W: (784, 100)
```

### 1.1.4 Element-wise Operations

행렬 간 덧셈, 뺄셈, 곱셈 등을 원소 단위로 수행하는 연산입니다. 두 행렬의 shape이 동일해야 합니다.

$$(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} \cdot B_{ij}$$

여기서 $\odot$는 원소별 곱(Hadamard product)을 나타냅니다.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B (element-wise addition):")
print(A + B)

print("\nA * B (element-wise multiplication):")
print(A * B)
```

```
A + B (element-wise addition):
[[ 6  8]
 [10 12]]

A * B (element-wise multiplication):
[[ 5 12]
 [21 32]]
```

활성화 함수의 역전파에서 원소별 곱이 사용됩니다:

```python
# Sigmoid 역전파: grad_z = grad_a * a * (1 - a)
a = np.random.rand(64, 100)       # sigmoid 출력
grad_a = np.random.randn(64, 100) # 상위 레이어에서 전파된 그래디언트

grad_z = grad_a * a * (1 - a)     # 원소별 곱
print(f"grad_z shape: {grad_z.shape}")
```

```
grad_z shape: (64, 100)
```

### 1.1.5 Broadcasting

NumPy의 브로드캐스팅은 shape이 다른 배열 간의 연산을 가능하게 합니다. 신경망에서 편향 덧셈 $z = xW + b$가 대표적인 예입니다.

브로드캐스팅 규칙:

1. 두 배열의 차원 수가 다르면, 작은 쪽 배열의 shape 앞에 1을 추가
2. 각 차원에서 크기가 1인 축은 다른 배열의 해당 축 크기로 확장
3. 크기가 다르고 둘 다 1이 아니면 오류 발생

```python
# 편향 덧셈: (N, D) + (D,) -> (N, D)
z = np.random.randn(64, 100)  # (64, 100)
b = np.random.randn(100)      # (100,) -> 브로드캐스팅 시 (1, 100) -> (64, 100)

result = z + b
print(f"z: {z.shape}, b: {b.shape} -> result: {result.shape}")
```

```
z: (64, 100), b: (100,) -> result: (64, 100)
```

브로드캐스팅 과정을 시각화하면:

```
z:      (64, 100)
b:          (100,)  →  (1, 100)  →  (64, 100)
result: (64, 100)
```

다양한 브로드캐스팅 예시:

```python
# (3, 4) + (4,) -> (3, 4)
A = np.ones((3, 4))
b = np.array([1, 2, 3, 4])
print(f"(3,4) + (4,) = {(A + b).shape}")

# (3, 4) + (3, 1) -> (3, 4)
A = np.ones((3, 4))
c = np.array([[1], [2], [3]])
print(f"(3,4) + (3,1) = {(A + c).shape}")

# (3, 1, 4) + (1, 5, 4) -> (3, 5, 4)
A = np.ones((3, 1, 4))
B = np.ones((1, 5, 4))
print(f"(3,1,4) + (1,5,4) = {(A + B).shape}")
```

```
(3,4) + (4,) = (3, 4)
(3,4) + (3,1) = (3, 4)
(3,1,4) + (1,5,4) = (3, 5, 4)
```

### 1.1.6 Axis Operations

행렬의 특정 축을 따라 합계, 평균 등을 계산합니다. 역전파에서 편향 그래디언트 계산 시 사용됩니다.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"A shape: {A.shape}")
print(f"A:\n{A}")

# axis=0: 행 방향으로 합 (열 보존)
print(f"\nsum(axis=0): {np.sum(A, axis=0)}, shape: {np.sum(A, axis=0).shape}")

# axis=1: 열 방향으로 합 (행 보존)
print(f"sum(axis=1): {np.sum(A, axis=1)}, shape: {np.sum(A, axis=1).shape}")

# keepdims=True: 차원 유지
print(f"sum(axis=1, keepdims=True):\n{np.sum(A, axis=1, keepdims=True)}")
```

```
A shape: (2, 3)
A:
[[1 2 3]
 [4 5 6]]

sum(axis=0): [5 7 9], shape: (3,)
sum(axis=1): [ 6 15], shape: (2,)
sum(axis=1, keepdims=True):
[[ 6]
 [15]]
```

편향 그래디언트 계산 예시:

```python
# grad_b = sum(grad_z, axis=0)
# 배치 내 모든 샘플의 그래디언트를 합산

grad_z = np.random.randn(64, 100)  # (N, D_out)
grad_b = np.sum(grad_z, axis=0)    # (D_out,)

print(f"grad_z: {grad_z.shape} -> grad_b: {grad_b.shape}")
```

```
grad_z: (64, 100) -> grad_b: (100,)
```

### 1.1.7 Summary

| 연산 | 수식 | NumPy | 신경망 활용 |
|------|------|-------|-------------|
| 행렬 곱셈 | $\mathbf{C} = \mathbf{A}\mathbf{B}$ | `np.dot(A, B)` 또는 `A @ B` | 선형 변환 $z = xW$ |
| 전치 | $\mathbf{A}^T$ | `A.T` | 역전파 그래디언트 계산 |
| 원소별 곱 | $\mathbf{A} \odot \mathbf{B}$ | `A * B` | 활성화 함수 역전파 |
| 브로드캐스팅 | - | 자동 적용 | 편향 덧셈 $z + b$ |
| 축 합계 | $\sum_i A_{ij}$ | `np.sum(A, axis=0)` | 편향 그래디언트 |
