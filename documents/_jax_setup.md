# JAX 환경 구축 매뉴얼

**WSL2 + Anaconda + JAX + CUDA 환경 구축하기**

Google의 고성능 수치 계산 라이브러리 JAX 설치 가이드입니다.

---

## 목차
1. [JAX 소개](#1-jax-소개)
2. [환경 생성 및 설치](#2-환경-생성-및-설치)
3. [설치 확인 및 테스트](#3-설치-확인-및-테스트)
4. [JAX 기본 사용법](#4-jax-기본-사용법)
5. [고급 기능](#5-고급-기능)
6. [Flax와 Optax 설치](#6-flax와-optax-설치)
7. [문제 해결](#7-문제-해결)

---

## 1. JAX 소개

### JAX란?
- **Google Research**에서 개발한 고성능 수치 계산 라이브러리
- **NumPy API**와 거의 동일한 인터페이스
- **자동 미분(AutoGrad)** 기능 내장
- **JIT 컴파일(XLA)** 지원으로 매우 빠른 속도
- **자동 벡터화(vmap)** 및 병렬화(pmap)

### 주요 특징
- **NumPy 호환**: `import jax.numpy as jnp`로 NumPy 대체
- **함수형 프로그래밍**: Pure function 기반
- **Composable transformations**: `grad`, `jit`, `vmap`, `pmap`
- **GPU/TPU 지원**: 자동 가속

### 사용 사례
- 딥러닝 연구 (Flax, Haiku, Equinox)
- 과학 계산 및 시뮬레이션
- 강화학습 (RLax, Acme)
- 확률론적 프로그래밍 (NumPyro)

---

## 2. 환경 생성 및 설치

### 2-1. JAX 환경 생성

```bash
# 환경 비활성화
conda deactivate

# Python 3.10으로 환경 생성
conda create -n jax_env python=3.10 -y

# 환경 활성화
conda activate jax_env
```

### 2-2. JAX 설치 (CUDA 지원)

JAX는 **CUDA 버전에 따라** 다른 설치 방법을 사용합니다.

#### CUDA 버전 확인

```bash
nvidia-smi
# 출력에서 "CUDA Version: 12.x" 확인
```

#### CUDA 12.x용 JAX 설치 (권장)

```bash
# JAX with CUDA 12.x support
pip install --upgrade "jax[cuda12]"
```

#### CUDA 11.x용 JAX 설치

```bash
# JAX with CUDA 11.x support (레거시)
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### CPU 전용 설치 (테스트용)

```bash
# CPU only
pip install --upgrade jax
```

### 2-3. 필수 패키지 설치

```bash
# NumPy 및 SciPy
conda install -y numpy scipy

# 시각화
pip install matplotlib seaborn plotly

# 데이터 처리
conda install -y pandas

# 유틸리티
pip install tqdm rich

# Jupyter
conda install -y jupyter ipykernel
```

### 2-4. JAX 생태계 라이브러리

```bash
# Flax (신경망 라이브러리)
pip install flax

# Optax (최적화 라이브러리)
pip install optax

# Orbax (체크포인트 관리)
pip install orbax-checkpoint

# Chex (테스트 유틸리티)
pip install chex

# Equinox (함수형 신경망)
pip install equinox

# JAXtyping (타입 힌트)
pip install jaxtyping
```

---

## 3. 설치 확인 및 테스트

### 3-1. 기본 Import 테스트

```bash
python << EOF
import sys
print("="*60)
print("JAX Installation Check")
print("="*60)

# 1. JAX 버전
import jax
print(f"JAX version: {jax.__version__}")

# 2. 백엔드 확인
import jax.numpy as jnp
print(f"Default backend: {jax.default_backend()}")

# 3. 사용 가능한 디바이스
devices = jax.devices()
print(f"\nAvailable devices: {len(devices)}")
for i, device in enumerate(devices):
    print(f"  Device {i}: {device}")

# 4. GPU 정보
if jax.default_backend() == 'gpu':
    from jax.lib import xla_bridge
    print(f"\nGPU backend: {xla_bridge.get_backend().platform}")
    print(f"GPU count: {jax.device_count()}")

print("\n✓ JAX installation successful!")
print("="*60)
EOF
```

**예상 출력**:
```
============================================================
JAX Installation Check
============================================================
JAX version: 0.4.35
Default backend: gpu

Available devices: 1
  Device 0: cuda:0

GPU backend: CUDA
GPU count: 1

✓ JAX installation successful!
============================================================
```

### 3-2. GPU 연산 테스트

```bash
python << EOF
import jax
import jax.numpy as jnp
import time

print("\n=== GPU Computation Test ===\n")

# GPU에서 행렬 생성
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5000, 5000))
y = jax.random.normal(key, (5000, 5000))

# JIT 컴파일된 행렬 곱셈
@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

# 첫 실행 (컴파일 포함)
print("First run (with compilation)...")
start = time.time()
z = matmul(x, y).block_until_ready()
first_time = time.time() - start
print(f"Time: {first_time:.4f} seconds")

# 두 번째 실행 (컴파일 제외)
print("\nSecond run (cached)...")
start = time.time()
z = matmul(x, y).block_until_ready()
second_time = time.time() - start
print(f"Time: {second_time:.4f} seconds")

print(f"\nResult shape: {z.shape}")
print(f"Speedup: {first_time/second_time:.2f}x")
print("\n✓ GPU computation successful!")
EOF
```

### 3-3. NumPy vs JAX 속도 비교

```bash
python << EOF
import numpy as np
import jax.numpy as jnp
import jax
import time

print("\n=== NumPy vs JAX Speed Comparison ===\n")

size = 3000

# NumPy (CPU)
print("NumPy (CPU) computing...")
x_np = np.random.randn(size, size).astype(np.float32)
y_np = np.random.randn(size, size).astype(np.float32)
start = time.time()
z_np = np.matmul(x_np, y_np)
cpu_time = time.time() - start

# JAX (GPU) - JIT 컴파일
@jax.jit
def jax_matmul(a, b):
    return jnp.matmul(a, b)

print("JAX (GPU) computing...")
key = jax.random.PRNGKey(0)
x_jax = jax.random.normal(key, (size, size), dtype=jnp.float32)
y_jax = jax.random.normal(key, (size, size), dtype=jnp.float32)

# Warm-up
_ = jax_matmul(x_jax, y_jax).block_until_ready()

start = time.time()
z_jax = jax_matmul(x_jax, y_jax).block_until_ready()
gpu_time = time.time() - start

print(f"\n{'='*50}")
print(f"Matrix size: {size}x{size}")
print(f"CPU (NumPy) time: {cpu_time:.4f} seconds")
print(f"GPU (JAX) time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time/gpu_time:.2f}x faster")
print(f"{'='*50}")
EOF
```

---

## 4. JAX 기본 사용법

### 4-1. NumPy 스타일 배열 연산

```bash
python << 'EOF'
import jax.numpy as jnp
import jax

print("\n=== JAX Basic Operations ===\n")

# 배열 생성
x = jnp.array([1, 2, 3, 4, 5])
print(f"Array: {x}")
print(f"Type: {type(x)}")
print(f"Device: {x.device()}")

# 수학 연산
y = jnp.exp(x)
z = jnp.sqrt(x)
print(f"\nexp(x): {y}")
print(f"sqrt(x): {z}")

# 행렬 연산
A = jnp.array([[1, 2], [3, 4]])
B = jnp.array([[5, 6], [7, 8]])
C = jnp.dot(A, B)
print(f"\nMatrix multiplication:\n{C}")

# 난수 생성 (PRNG key 필요)
key = jax.random.PRNGKey(42)
random_array = jax.random.normal(key, (3, 3))
print(f"\nRandom array:\n{random_array}")
EOF
```

### 4-2. 자동 미분 (grad)

```bash
python << 'EOF'
import jax
import jax.numpy as jnp

print("\n=== Automatic Differentiation ===\n")

# 함수 정의
def f(x):
    return x**3 + 2*x**2 - 5*x + 3

# 미분 함수 생성
df = jax.grad(f)

# 계산
x = 2.0
y = f(x)
dy = df(x)

print(f"f({x}) = {y}")
print(f"f'({x}) = {dy}")

# 다변수 함수
def g(x, y):
    return x**2 + y**2

# 편미분
dg_dx = jax.grad(g, argnums=0)
dg_dy = jax.grad(g, argnums=1)

x, y = 3.0, 4.0
print(f"\ng({x}, {y}) = {g(x, y)}")
print(f"∂g/∂x = {dg_dx(x, y)}")
print(f"∂g/∂y = {dg_dy(x, y)}")
EOF
```

### 4-3. JIT 컴파일

```bash
python << 'EOF'
import jax
import jax.numpy as jnp
import time

print("\n=== JIT Compilation ===\n")

# 일반 함수
def slow_function(x):
    return jnp.sum(x ** 2) + jnp.sum(x ** 3)

# JIT 컴파일된 함수
@jax.jit
def fast_function(x):
    return jnp.sum(x ** 2) + jnp.sum(x ** 3)

# 테스트 데이터
x = jax.random.normal(jax.random.PRNGKey(0), (10000,))

# 속도 비교
print("Without JIT:")
start = time.time()
for _ in range(100):
    result = slow_function(x).block_until_ready()
no_jit_time = time.time() - start
print(f"Time: {no_jit_time:.4f} seconds")

print("\nWith JIT:")
# Warm-up
_ = fast_function(x).block_until_ready()
start = time.time()
for _ in range(100):
    result = fast_function(x).block_until_ready()
jit_time = time.time() - start
print(f"Time: {jit_time:.4f} seconds")

print(f"\nSpeedup: {no_jit_time/jit_time:.2f}x")
EOF
```

### 4-4. 벡터화 (vmap)

```bash
python << 'EOF'
import jax
import jax.numpy as jnp

print("\n=== Vectorization with vmap ===\n")

# 단일 입력 함수
def single_prediction(params, x):
    return jnp.dot(params, x)

# 배치 처리 (수동)
def batch_prediction_manual(params, X):
    return jnp.array([single_prediction(params, x) for x in X])

# 배치 처리 (vmap)
batch_prediction_vmap = jax.vmap(single_prediction, in_axes=(None, 0))

# 테스트
params = jnp.array([1.0, 2.0, 3.0])
X = jax.random.normal(jax.random.PRNGKey(0), (5, 3))

result_manual = batch_prediction_manual(params, X)
result_vmap = batch_prediction_vmap(params, X)

print(f"Manual result: {result_manual}")
print(f"vmap result: {result_vmap}")
print(f"Results match: {jnp.allclose(result_manual, result_vmap)}")
EOF
```

---

## 5. 고급 기능

### 5-1. 다중 GPU 병렬화 (pmap)

```bash
python << 'EOF'
import jax
import jax.numpy as jnp

print("\n=== Parallel Computation (pmap) ===\n")

# GPU 개수 확인
n_devices = jax.device_count()
print(f"Available devices: {n_devices}")

if n_devices > 1:
    # 병렬 함수 정의
    @jax.pmap
    def parallel_square(x):
        return x ** 2
    
    # 데이터를 여러 디바이스로 분할
    x = jnp.arange(n_devices * 4).reshape(n_devices, 4)
    print(f"Input: {x}")
    
    # 병렬 실행
    result = parallel_square(x)
    print(f"Result: {result}")
else:
    print("Only 1 GPU available, pmap example skipped")
    print("pmap is useful when you have multiple GPUs")
EOF
```

### 5-2. Pytree 사용

```bash
python << 'EOF'
import jax
import jax.numpy as jnp

print("\n=== PyTree Operations ===\n")

# Pytree: 중첩된 파이썬 구조
params = {
    'layer1': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
    'layer2': {'w': jnp.ones((4, 2)), 'b': jnp.zeros(2)},
}

# Pytree map
def scale_params(params, factor):
    return jax.tree_map(lambda x: x * factor, params)

scaled = scale_params(params, 2.0)
print("Original layer1 w:")
print(params['layer1']['w'])
print("\nScaled layer1 w:")
print(scaled['layer1']['w'])

# Pytree flatten/unflatten
leaves, treedef = jax.tree_flatten(params)
print(f"\nNumber of parameters: {len(leaves)}")
print(f"Total elements: {sum(x.size for x in leaves)}")
EOF
```

### 5-3. 커스텀 Gradient

```bash
python << 'EOF'
import jax
import jax.numpy as jnp

print("\n=== Custom Gradient ===\n")

# 커스텀 미분 정의
@jax.custom_vjp
def f(x):
    return jnp.sin(x)

def f_fwd(x):
    return f(x), x

def f_bwd(x, g):
    # 커스텀 gradient: cos(x) 대신 1.0 사용
    return (g * 1.0,)

f.defvjp(f_fwd, f_bwd)

# 테스트
x = 1.0
y = f(x)
dy = jax.grad(f)(x)

print(f"f({x}) = {y}")
print(f"Custom gradient: {dy}")
print(f"True gradient (cos): {jnp.cos(x)}")
EOF
```

---

## 6. Flax와 Optax 설치

### 6-1. Flax (신경망 라이브러리)

```bash
python << 'EOF'
from flax import linen as nn
import jax
import jax.numpy as jnp

print("\n=== Flax Neural Network ===\n")

# 간단한 MLP 정의
class MLP(nn.Module):
    features: tuple = (128, 64, 10)
    
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

# 모델 초기화
model = MLP()
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (32, 784))  # batch of 32

# 파라미터 초기화
params = model.init(key, x)

# Forward pass
output = model.apply(params, x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of parameters: {sum(x.size for x in jax.tree_leaves(params))}")
print("\n✓ Flax model created successfully!")
EOF
```

### 6-2. Optax (최적화 라이브러리)

```bash
python << 'EOF'
import jax
import jax.numpy as jnp
import optax

print("\n=== Optax Optimizer ===\n")

# 간단한 손실 함수
def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

# 최적화 설정
learning_rate = 0.01
optimizer = optax.adam(learning_rate)

# 파라미터 초기화
key = jax.random.PRNGKey(0)
params = jax.random.normal(key, (10,))
opt_state = optimizer.init(params)

# 더미 데이터
x = jax.random.normal(key, (100, 10))
y = jax.random.normal(key, (100,))

# 학습 스텝 함수
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 몇 번의 학습 스텝
print("Training...")
for i in range(5):
    params, opt_state, loss = train_step(params, opt_state, x, y)
    print(f"Step {i+1}, Loss: {loss:.6f}")

print("\n✓ Optax optimizer working!")
EOF
```

### 6-3. Flax + Optax 통합 예시

```bash
cat > ~/flax_training_example.py << 'EOF'
"""Flax + Optax 학습 예시"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """간단한 CNN 모델"""
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def create_train_state(rng, learning_rate):
    """학습 상태 생성"""
    model = SimpleCNN()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jax.jit
def train_step(state, batch):
    """학습 스텝"""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    """메인 함수"""
    print("\n=== Flax + Optax Training Example ===\n")
    
    # 초기화
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate=1e-3)
    
    # 더미 데이터
    key = jax.random.PRNGKey(0)
    dummy_batch = {
        'image': jax.random.normal(key, (32, 28, 28, 1)),
        'label': jax.random.randint(key, (32,), 0, 10)
    }
    
    # 학습
    print("Training for 10 steps...")
    for step in tqdm(range(10)):
        state, loss = train_step(state, dummy_batch)
        if step % 2 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
    
    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
EOF

# 실행
python ~/flax_training_example.py
```

---

## 7. 문제 해결

### 7-1. `No GPU/TPU found` 에러

**원인**: CUDA가 제대로 인식되지 않음

**해결**:
```bash
# JAX 재설치
pip uninstall jax jaxlib -y
pip install --upgrade "jax[cuda12]"

# CUDA 확인
nvidia-smi

# WSL 재시작
# PowerShell에서: wsl --shutdown
```

### 7-2. `XlaRuntimeError` 발생

**원인**: JIT 컴파일 실패

**해결**:
```python
# JIT 비활성화하여 테스트
jax.config.update('jax_disable_jit', True)

# 또는 디버그 모드
jax.config.update('jax_debug_nans', True)
```

### 7-3. 메모리 부족

**해결**:
```python
# GPU 메모리 사전 할당 비활성화
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# 메모리 사용량 제한
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # 70%만 사용

import jax
```

### 7-4. Float64 정밀도 활성화

**기본적으로 JAX는 Float32 사용**:
```python
import jax
jax.config.update("jax_enable_x64", True)

# 확인
import jax.numpy as jnp
x = jnp.array([1.0])
print(x.dtype)  # float64
```

### 7-5. CuDNN 관련 에러

**해결**:
```bash
conda activate jax_env
conda install -c conda-forge cudnn -y
```

---

## 8. JAX vs PyTorch/TensorFlow 비교

### 8-1. 주요 차이점

| 특징 | JAX | PyTorch | TensorFlow |
|------|-----|---------|-----------|
| 패러다임 | 함수형 | 객체지향 | 혼합 |
| 자동 미분 | grad | autograd | GradientTape |
| 가속 | JIT (XLA) | TorchScript | XLA |
| NumPy 호환 | 거의 완벽 | 유사 | 부분적 |
| 가변성 | Immutable | Mutable | Mutable |
| 학습 곡선 | 중간 | 쉬움 | 어려움 |

### 8-2. 코드 비교

**PyTorch**:
```python
import torch

x = torch.randn(100, 100, requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward()
grad = x.grad
```

**JAX**:
```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

x = jax.random.normal(jax.random.PRNGKey(0), (100, 100))
grad_fn = jax.grad(f)
grad = grad_fn(x)
```

---

## 9. 환경 관리

### 9-1. 단축 명령 추가

```bash
cat >> ~/.bashrc << 'EOF'

# JAX 환경 단축 명령
alias jx='conda activate jax_env'
EOF

source ~/.bashrc
```

**사용법**: `jx` → jax_env 활성화

### 9-2. 패키지 백업

```bash
conda activate jax_env
conda list --export > ~/jax_env_packages.txt
pip freeze > ~/jax_requirements.txt
```

### 9-3. JAX 설정 파일

```bash
cat > ~/.jaxrc << 'EOF'
# JAX 기본 설정
import os

# GPU 메모리 사전 할당 비활성화
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Float64 활성화
os.environ['JAX_ENABLE_X64'] = '1'

# 디버그 모드 (필요시)
# os.environ['JAX_DEBUG_NANS'] = '1'
EOF
```

Python에서 불러오기:
```python
import os
exec(open(os.path.expanduser('~/.jaxrc')).read())
import jax
```

---

## 10. 전체 설치 체크리스트

| 단계 | 내용 | 확인 |
|------|------|------|
| 1 | jax_env 환경 생성 | ☐ |
| 2 | JAX 설치 (CUDA 12.x) | ☐ |
| 3 | `import jax` 성공 | ☐ |
| 4 | GPU 인식 확인 | ☐ |
| 5 | GPU 연산 테스트 | ☐ |
| 6 | JIT 컴파일 테스트 | ☐ |
| 7 | 자동 미분 테스트 | ☐ |
| 8 | Flax 설치 | ☐ |
| 9 | Optax 설치 | ☐ |
| 10 | 통합 학습 예시 | ☐ |

---

## 11. 전체 환경 요약

이제 **5개의 독립된 딥러닝 환경**이 완성되었습니다!

| 환경 | Python | 주요 라이브러리 | 용도 |
|------|--------|----------------|------|
| pytorch_env | 3.10 | PyTorch 2.5.1 | 범용 딥러닝 |
| cupy_env | 3.10 | CuPy 13.6.0 | NumPy GPU 가속 |
| tensorflow_env | 3.10 | TensorFlow 2.18.0 | TensorFlow 딥러닝 |
| anomalib_env | 3.10 | Anomalib 1.1.x | 이상 감지 |
| jax_env | 3.10 | JAX 0.4.x + Flax | 함수형 딥러닝 |

### 빠른 환경 전환

```bash
pt   # PyTorch
cu   # CuPy
tf   # TensorFlow
al   # Anomalib
jx   # JAX
ca   # 비활성화
gpu  # GPU 모니터링
```

---

## 12. 참고 자료

- **JAX 공식 문서**: https://jax.readthedocs.io/
- **JAX GitHub**: https://github.com/google/jax
- **Flax 문서**: https://flax.readthedocs.io/
- **Optax 문서**: https://optax.readthedocs.io/
- **JAX Ecosystem**: https://github.com/n2cholas/awesome-jax

---

**이 매뉴얼을 따라하면 JAX 환경을 완벽하게 구축할 수 있습니다!** 🎉