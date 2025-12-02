# CuPy 환경 구축 매뉴얼

**WSL2 + Anaconda + CuPy + CUDA 환경 구축하기**

---

## 목차
1. [CuPy 환경 생성](#1-cupy-환경-생성)
2. [CUDA Toolkit 설치 방법 선택](#2-cuda-toolkit-설치-방법-선택)
3. [CuPy 설치 및 검증](#3-cupy-설치-및-검증)
4. [NumPy to CuPy 변환 가이드](#4-numpy-to-cupy-변환-가이드)
5. [문제 해결](#5-문제-해결)

---

## 1. CuPy 환경 생성

### 1-1. 기본 환경 생성

```bash
# 현재 환경 비활성화
conda deactivate

# Python 3.10으로 CuPy 환경 생성
conda create -n cupy_env python=3.10 -y

# 환경 활성화
conda activate cupy_env
```

---

## 2. CUDA Toolkit 설치 방법 선택

CuPy는 **CUDA Toolkit**이 반드시 필요합니다. 두 가지 설치 방법이 있습니다.

### 방법 비교

| 구분 | 방법 1: Conda 설치 (권장) | 방법 2: 시스템 설치 |
|------|--------------------------|-------------------|
| 설치 위치 | conda 환경 내부 | WSL2 시스템 전역 |
| 관리 용이성 | ⭐⭐⭐⭐⭐ 매우 쉬움 | ⭐⭐⭐ 보통 |
| 환경 독립성 | ⭐⭐⭐⭐⭐ 완전 독립 | ⭐⭐ 전역 공유 |
| 설치 시간 | 5-10분 | 20-30분 |
| 디스크 사용량 | 환경별 중복 | 한 번만 설치 |
| 권장 대상 | 대부분의 사용자 | 고급 사용자 |

**권장**: 처음 설치하거나 간편하게 사용하려면 **방법 1 (Conda 설치)** 선택

---

## 방법 1: Conda로 CUDA Toolkit 포함 설치 (권장)

### 2-1. CuPy + CUDA Toolkit 한번에 설치

```bash
conda activate cupy_env

# CuPy와 CUDA Toolkit 12.4를 함께 설치
conda install -c conda-forge cupy cuda-toolkit=12.4 -y
```

**설치 시간**: 약 5-10분 소요

### 2-2. 추가 패키지 설치

```bash
# 데이터 처리 및 시각화
conda install -y numpy scipy pandas matplotlib jupyter

# 유틸리티
pip install scikit-learn opencv-python tqdm
```

### 2-3. 설치 확인

```bash
python << EOF
import cupy as cp
print("="*50)
print("CuPy Installation Check (Conda Method)")
print("="*50)
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"CUDA Runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
print(f"Number of GPUs: {cp.cuda.runtime.getDeviceCount()}")

# GPU 속성 가져오기 (CuPy 13.x 방식)
device = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(device.id)
print(f"GPU Name: {props['name'].decode()}")
print(f"GPU Memory: {device.mem_info[1] / 1024**3:.2f} GB")
print(f"Compute Capability: {props['major']}.{props['minor']}")
print("="*50)
print("✓ CuPy installation SUCCESS!")
EOF
```

**✅ 이 방법으로 성공하면 [3. CuPy 설치 및 검증](#3-cupy-설치-및-검증)으로 건너뛰기**

---

## 방법 2: WSL2 시스템에 CUDA Toolkit 직접 설치

이 방법은 시스템 전역에 CUDA Toolkit을 설치하여 모든 환경에서 공유합니다.

### 2-1. 기존 CUDA 관련 repo 정리 (있는 경우)

```bash
# 기존 CUDA repo 제거
sudo rm -f /etc/apt/sources.list.d/cuda*.list
sudo rm -f /etc/apt/sources.list.d/nvidia*.list

# 패키지 목록 업데이트
sudo apt update
```

### 2-2. NVIDIA CUDA GPG 키 등록

```bash
# GPG 키 다운로드 및 등록
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-key.gpg
```

### 2-3. CUDA Repository 추가

```bash
# CUDA repo 추가
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-key.gpg] \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" \
| sudo tee /etc/apt/sources.list.d/cuda.list
```

### 2-4. 패키지 목록 업데이트

```bash
sudo apt update
```

### 2-5. CUDA Toolkit 12.4 설치

```bash
# CUDA Toolkit 설치 (약 3GB, 20-30분 소요)
sudo apt install -y cuda-toolkit-12-4
```

### 2-6. 환경 변수 설정

```bash
# CUDA 경로를 .bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# CUDA 12.4 Environment
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
EOF

# 적용
source ~/.bashrc
```

### 2-7. CUDA 설치 확인

```bash
# CUDA 버전 확인
nvcc --version
```

출력 예시:
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.4
```

```bash
# NVRTC 라이브러리 확인 (CuPy가 필요로 함)
ls /usr/local/cuda-12.4/lib64 | grep nvrtc
```

출력 예시:
```
libnvrtc.so
libnvrtc.so.12
libnvrtc.so.12.4.127
```

### 2-8. CuPy 설치

```bash
conda activate cupy_env

# CuPy 설치 (CUDA 12.x 용)
pip install cupy-cuda12x

# 추가 패키지
conda install -y numpy scipy pandas matplotlib jupyter
pip install scikit-learn opencv-python tqdm
```

### 2-9. 설치 확인

```bash
python << EOF
import cupy as cp
print("="*50)
print("CuPy Installation Check (System Method)")
print("="*50)
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"CUDA Runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
print(f"Number of GPUs: {cp.cuda.runtime.getDeviceCount()}")
print(f"GPU Name: {cp.cuda.Device(0).name.decode()}")
print(f"GPU Memory: {cp.cuda.Device(0).mem_info[1] / 1024**3:.2f} GB")
print("="*50)
EOF
```

---

## 3. CuPy 설치 및 검증

### 3-1. 기본 연산 테스트

```bash
conda activate cupy_env
python << EOF
import cupy as cp
import numpy as np

print("\n=== Basic Array Operations ===")
# CuPy 배열 생성
x = cp.array([1, 2, 3, 4, 5])
print(f"CuPy array: {x}")
print(f"Type: {type(x)}")

# 수학 연산
y = cp.sqrt(x)
print(f"Square root: {y}")

# NumPy로 변환
y_cpu = cp.asnumpy(y)
print(f"Converted to NumPy: {y_cpu}")
print(f"Type: {type(y_cpu)}")
EOF
```

### 3-2. GPU 행렬 연산 테스트

```bash
python << EOF
import cupy as cp
import time

print("\n=== GPU Matrix Multiplication Test ===")

# 큰 행렬 생성
size = 5000
x = cp.random.randn(size, size, dtype=cp.float32)
y = cp.random.randn(size, size, dtype=cp.float32)

# GPU 연산 시간 측정
cp.cuda.Stream.null.synchronize()  # GPU 동기화
start = time.time()
z = cp.matmul(x, y)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start

print(f"Matrix size: {size}x{size}")
print(f"GPU computation time: {gpu_time:.4f} seconds")

# 메모리 사용량 확인 (수정된 방법)
mempool = cp.get_default_memory_pool()
print(f"Memory used: {mempool.used_bytes() / 1024**2:.2f} MB")
print(f"Memory total: {mempool.total_bytes() / 1024**2:.2f} MB")
print("GPU computation: SUCCESS ✓")
EOF
```

### 3-3. NumPy vs CuPy 속도 비교

```bash
python << EOF
import cupy as cp
import numpy as np
import time

print("\n=== NumPy vs CuPy Speed Comparison ===")

size = 3000

# NumPy (CPU)
x_cpu = np.random.randn(size, size).astype(np.float32)
y_cpu = np.random.randn(size, size).astype(np.float32)
start = time.time()
z_cpu = np.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start

# CuPy (GPU)
x_gpu = cp.random.randn(size, size, dtype=cp.float32)
y_gpu = cp.random.randn(size, size, dtype=cp.float32)
cp.cuda.Stream.null.synchronize()
start = time.time()
z_gpu = cp.matmul(x_gpu, y_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start

print(f"Matrix size: {size}x{size}")
print(f"CPU (NumPy) time: {cpu_time:.4f} seconds")
print(f"GPU (CuPy) time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time/gpu_time:.2f}x faster")
EOF
```

**예상 출력**:
```
Matrix size: 3000x3000
CPU (NumPy) time: 0.8234 seconds
GPU (CuPy) time: 0.0156 seconds
Speedup: 52.78x faster
```

---

## 4. NumPy to CuPy 변환 가이드

### 4-1. Backend 파일 생성 (권장)

NumPy와 CuPy를 쉽게 전환할 수 있는 backend 모듈 생성:

```bash
cat > ~/backend.py << 'EOF'
"""
Backend module for seamless NumPy/CuPy switching
Usage:
    from backend import backend
    xp = backend.xp
"""

import numpy as _np

try:
    import cupy as _cp
    GPU_AVAILABLE = True
except ImportError:
    _cp = None
    GPU_AVAILABLE = False


class Backend:
    """Backend wrapper for NumPy/CuPy compatibility"""
    
    def __init__(self, use_gpu=True):
        """
        Args:
            use_gpu (bool): Use GPU if available
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = _cp if self.use_gpu else _np
        
    def asarray(self, x, dtype=None):
        """Convert array to appropriate backend"""
        if self.use_gpu:
            return self.xp.asarray(x, dtype=dtype)
        return _np.asarray(x, dtype=dtype)
    
    def to_cpu(self, x):
        """Move array to CPU (NumPy)"""
        if self.use_gpu and hasattr(self.xp, "asnumpy"):
            return self.xp.asnumpy(x)
        return x
    
    def to_gpu(self, x):
        """Move array to GPU (CuPy)"""
        if self.use_gpu:
            return self.xp.asarray(x)
        return x


# Global backend instance
backend = Backend(use_gpu=True)

# Convenience exports
xp = backend.xp
asarray = backend.asarray
to_cpu = backend.to_cpu
to_gpu = backend.to_gpu
EOF
```

### 4-2. Backend 사용 예시

```bash
python << 'EOF'
from backend import backend, xp

# 이제 xp를 사용하면 자동으로 GPU 사용
print(f"Using: {'CuPy (GPU)' if backend.use_gpu else 'NumPy (CPU)'}")

# NumPy 스타일 그대로 사용
x = xp.array([1, 2, 3, 4, 5])
y = xp.exp(x)
z = xp.sqrt(y)

print(f"Result: {z}")
print(f"Type: {type(z)}")

# CPU로 변환 (출력용)
z_cpu = backend.to_cpu(z)
print(f"CPU array: {z_cpu}")
EOF
```

### 4-3. NumPy 코드 변환 규칙

#### ✅ 자동 변환 가능 (xp로 교체만 하면 됨)

```python
# NumPy
import numpy as np
x = np.zeros((3, 3))
y = np.exp(x)
z = np.dot(x, y)

# CuPy (xp 사용)
from backend import xp
x = xp.zeros((3, 3))
y = xp.exp(x)
z = xp.dot(x, y)
```

#### ⚠️ 주의 필요 (수정 필요)

```python
# ❌ np.frombuffer는 CuPy에서 미지원
data = np.frombuffer(file.read(), dtype=np.uint8)

# ✅ NumPy로 읽고 CuPy로 변환
import numpy as np
from backend import xp
data = np.frombuffer(file.read(), dtype=np.uint8)
data = xp.asarray(data)  # GPU로 이동
```

```python
# ❌ NumPy random 혼용 금지
x_cpu = np.random.randn(100, 100)
x_gpu = xp.asarray(x_cpu)  # 느림 (CPU→GPU 전송)

# ✅ CuPy random 직접 사용
x_gpu = xp.random.randn(100, 100)  # 빠름 (GPU에서 바로 생성)
```

### 4-4. 변환 체크리스트

| NumPy 함수 | CuPy 호환 | 비고 |
|-----------|----------|------|
| `np.array()` | ✅ `xp.array()` | |
| `np.zeros()` | ✅ `xp.zeros()` | |
| `np.random.randn()` | ✅ `xp.random.randn()` | |
| `np.exp()`, `np.sqrt()` | ✅ `xp.exp()`, `xp.sqrt()` | |
| `np.dot()`, `np.matmul()` | ✅ `xp.dot()`, `xp.matmul()` | |
| `np.pad()` | ✅ `xp.pad()` | 일부 모드 차이 있음 |
| `np.frombuffer()` | ❌ | NumPy로 읽고 xp.asarray() |
| `np.save()` | ⚠️ | CPU로 변환 후 저장 권장 |

---

## 5. 문제 해결

### 5-1. `ImportError: libcuda.so.1: cannot open shared object file`

**원인**: NVIDIA 드라이버 미설치 또는 WSL2 GPU 패스스루 문제

**해결**:
```bash
# WSL2에서 GPU 확인
nvidia-smi

# 안되면 Windows에서 PowerShell 실행
wsl --shutdown
# WSL2 재시작
```

### 5-2. `CuPy is not correctly installed` 에러

**원인**: CUDA Toolkit 미설치 또는 경로 문제

**해결 (방법 1 사용한 경우)**:
```bash
conda activate cupy_env
conda install -c conda-forge cupy cuda-toolkit=12.4 -y --force-reinstall
```

**해결 (방법 2 사용한 경우)**:
```bash
# CUDA 경로 확인
echo $LD_LIBRARY_PATH

# 경로가 없으면 다시 설정
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# .bashrc에 영구 저장
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 5-3. `CUDA driver version is insufficient` 에러

**원인**: Windows NVIDIA 드라이버가 너무 오래됨

**해결**:
1. Windows에서 최신 NVIDIA 드라이버 설치
2. WSL2 재시작: `wsl --shutdown` (PowerShell)

### 5-4. CuPy 연산이 매우 느림

**원인**: Python for-loop에서 CuPy 배열 사용

**해결**:
```python
# ❌ 느림 - for-loop
result = xp.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):
        result[i, j] = xp.sin(i + j)  # 매우 느림!

# ✅ 빠름 - 벡터화
i, j = xp.meshgrid(xp.arange(1000), xp.arange(1000), indexing='ij')
result = xp.sin(i + j)  # 매우 빠름!
```

### 5-5. `CUDA out of memory` 에러

**원인**: GPU 메모리 부족

**해결**:
```python
import cupy as cp

# 메모리 해제
del x, y, z
cp.get_default_memory_pool().free_all_blocks()

# 메모리 사용량 확인
mempool = cp.get_default_memory_pool()
print(f"Used: {mempool.used_bytes() / 1024**2:.2f} MB")
print(f"Total: {mempool.total_bytes() / 1024**2:.2f} MB")
```

---

## 6. 환경 관리 명령어

### 6-1. 환경 전환

```bash
# CuPy 환경 활성화
conda activate cupy_env

# PyTorch 환경으로 전환
conda activate pytorch_env

# 환경 비활성화
conda deactivate
```

### 6-2. 단축 명령 설정 (선택사항)

```bash
# .bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# CuPy 환경 단축 명령
alias cu='conda activate cupy_env'
alias pt='conda activate pytorch_env'
alias ca='conda deactivate'
EOF

source ~/.bashrc
```

이제 다음처럼 사용:
- `cu` → cupy_env 활성화
- `pt` → pytorch_env 활성화
- `ca` → 환경 비활성화

### 6-3. 패키지 목록 백업

```bash
conda activate cupy_env

# conda 패키지 목록
conda list --export > cupy_env_packages.txt

# pip 패키지 목록
pip freeze > cupy_requirements.txt
```

---

## 7. 전체 설치 체크리스트

| 단계 | 내용 | 확인 |
|------|------|------|
| 1 | cupy_env 환경 생성 (Python 3.10) | ☐ |
| 2 | CUDA Toolkit 설치 (방법 1 or 2) | ☐ |
| 3 | CuPy 설치 | ☐ |
| 4 | `import cupy` 성공 | ☐ |
| 5 | CUDA 버전 확인 | ☐ |
| 6 | GPU 정보 출력 성공 | ☐ |
| 7 | 기본 배열 연산 테스트 | ☐ |
| 8 | 행렬 곱셈 테스트 | ☐ |
| 9 | NumPy vs CuPy 속도 비교 | ☐ |
| 10 | backend.py 파일 생성 | ☐ |

---

## 8. 다음 단계

CuPy 환경 구축이 완료되었습니다! 이제 다음을 시도해보세요:

1. **NumPy 딥러닝 코드를 CuPy로 변환**
   - MLP (Multi-Layer Perceptron)
   - CNN (Convolutional Neural Network)

2. **과학 계산 가속**
   - FFT (Fast Fourier Transform)
   - 선형대수 연산
   - 통계 계산

3. **이미지 처리 가속**
   - 필터링, 변환
   - 특징 추출

---

## 9. 참고 링크

- **CuPy 공식 문서**: https://docs.cupy.dev/
- **CuPy GitHub**: https://github.com/cupy/cupy
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **NumPy to CuPy 마이그레이션 가이드**: https://docs.cupy.dev/en/stable/user_guide/difference.html

---

**이 매뉴얼을 따라하면 CuPy 환경을 완벽하게 구축할 수 있습니다!**