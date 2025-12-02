아래는 **WSL2 + CUDA + CuPy 설치부터 NumPy 딥러닝 코드를 CuPy 기반 GPU 코드로 변환하는 전체 과정을 초보자도 그대로 따라 할 수 있도록 작성한 “완전 매뉴얼/튜토리얼”**입니다.
이번 과정에서 사용하신 시행착오를 모두 반영해, **실패 없이 재현 가능한 절차**로 구성했습니다.

---

# CuPy + WSL2 + CUDA + NumPy 딥러닝 코드 GPU 변환 튜토리얼

**(WSL2 Ubuntu + NVIDIA GPU 환경, CuPy로 NumPy 딥러닝 가속하기)**

---

# 1. WSL2 환경 확인 및 버전 체크

### 1) PowerShell에서 WSL 상태 확인

```powershell
wsl --status
```

### 2) Ubuntu 버전 확인 (WSL 내부)

```bash
lsb_release -a
```

필수 조건:

* Ubuntu 20.04 또는 22.04 권장
* NVIDIA 드라이버 최신 버전 (CUDA 12.x 지원)

---

# 2. NVIDIA GPU 드라이버 확인 (host Windows)

### Windows PowerShell

```powershell
nvidia-smi
```

정상 출력 예:

```
NVIDIA-SMI 552.xx
CUDA Version: 12.8
```

여기서 **CUDA Version 12.x**가 보이면 WS에게 CUDA 런타임이 자동 패스스루됨.

---

# 3. WSL2 Ubuntu 내부에서 CUDA Toolkit 설치 (매우 중요)

WSL2 환경은 GPU 패스스루 방식이므로 따로 CUDA 설치 필요 없다?
→ **CuPy는 local CUDA Toolkit의 NVRTC, cuBLAS 등을 반드시 필요로 함.**
→ **WSL 내부에도 CUDA Toolkit을 반드시 설치해야 함.**

### 1) CUDA repo 삭제(문제가 있었던 경우)

```bash
sudo rm /etc/apt/sources.list.d/cuda.list
sudo rm /etc/apt/sources.list.d/nvidia*  
sudo apt update
```

### 2) NVIDIA CUDA repo GPG 키 등록

```bash
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-key.gpg
```

### 3) CUDA repo 추가

```bash
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-key.gpg] \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" \
| sudo tee /etc/apt/sources.list.d/cuda.list
```

### 4) 업데이트

```bash
sudo apt update
```

### 5) CUDA Toolkit 설치 (권장: 12.4 또는 12.5)

```bash
sudo apt install -y cuda-toolkit-12-4
```

설치 폴더 확인:

```bash
ls /usr/local/cuda-12.4/lib64 | grep nvrtc
```

정상 예:

```
libnvrtc.so
libnvrtc.so.12
libnvrtc.so.12.4
```

---

# 4. CUDA 환경 설정

```bash
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

CUDA 버전 확인:

```bash
nvcc --version
```

---

# 5. CuPy 설치 (conda 사용)

### 1) 새로운 환경 생성

```bash
conda create -n cupy_env python=3.10 -y
conda activate cupy_env
```

### 2) CuPy 설치 (CUDA 12 전용)

```bash
pip install cupy-cuda12x
```

### 3) 추가 라이브러리

```bash
pip install tqdm matplotlib
```

---

# 6. CuPy 동작 테스트

```bash
python - <<EOF
import cupy as cp
print(cp.arange(5))
import cupy.cuda.runtime as rt
print("CUDA Runtime:", rt.runtimeGetVersion())
EOF
```

---

# 7. NumPy → CuPy 변환을 위한 백엔드(backend.py) 파일 작성

이 파일은 NumPy와 CuPy를 **완전 호환 API**로 사용하게 해줌.

---

## backend.py

```python
# backend.py
import numpy as _np

try:
    import cupy as _cp
    GPU_AVAILABLE = True
except ImportError:
    _cp = None
    GPU_AVAILABLE = False


class Backend:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = _cp if self.use_gpu else _np

    def asarray(self, x, dtype=None):
        if self.use_gpu:
            return self.xp.asarray(x, dtype=dtype)
        return _np.asarray(x, dtype=dtype)

    def to_cpu(self, x):
        if self.use_gpu and hasattr(self.xp, "asnumpy"):
            return self.xp.asnumpy(x)
        return x


backend = Backend(use_gpu=True)
```

---

# 8. 기존 NumPy 코드 변환 규칙 (매우 중요)

## 1) 모든 np → xp 로 변경

NumPy API를 CuPy API와 동일하게 사용하는 핵심 포인트:

```python
from backend import backend
xp = backend.xp
```

### 예)

```python
x = xp.zeros((3,3))
y = xp.exp(x)
z = xp.dot(x, y)
```

즉:

* numpy: np.exp, np.dot
* cupy: cp.exp, cp.dot
* ⇒ 백엔드 통일: xp.exp, xp.dot

---

## 2) np.frombuffer → **np.frombuffer 유지**

frombuffer는 **무조건 NumPy**로 읽고
→ 이후 xp.asarray로 GPU로 옮김 필요

예:

```python
import numpy as np

raw = np.frombuffer(f.read(), np.uint8)
data = xp.asarray(raw)
```

※ CuPy는 frombuffer를 지원하지 않음

---

## 3) xp.pad vs numpy.pad 차이 주의점

CuPy pad는 일부 모드(np.pad와 100% 동일하지 않음).

해결 방법:

* MNIST처럼 단순 패딩은 CuPy pad 사용 OK
* 변환 오류 시 반드시 dtype 또는 shape 확인

---

## 4) MaxPool2d 같은 loop 기반 구현은 CuPy에서 비효율적

CuPy에서는 python for-loop가 **매우 느리거나 멈춤처럼 보임**

해결책:

* 완전 벡터화된 MaxPool2d forward/backward 필요 (이미 적용함)

---

# 9. CNN 과 같은 합성곱 연산의 im2col 최적화

* im2col / col2im은 CuPy에서 빠르게 동작함
* 단, reshape/transpose 연산이 많으므로 batch size 과도하게 키우지 않기

---

# 10. CuPy 환경에서 반드시 기억해야 할 주의사항

### ✔ 1) xp.arange / xp.random 등은 반드시 xp로

np.random 혼용 금지 → 모든 난수 생성은 xp.random

### ✔ 2) backward에서 scatter는 CuPy가 더 엄격함

NumPy에서 되던 방식도 CuPy에서는 shape mismatch로 실패할 가능성 큼
→ 벡터화된 안전한 scatter 구현 필요 (이미 구현)

### ✔ 3) numpy array와 cupy array 혼용 금지

예:

```python
a = xp.zeros((3,3))
b = np.zeros((3,3))
```

b를 xp 연산에 입력하면 오류 발생

### ✔ 4) CPU ↔ GPU 이동 시 backend.to_cpu 사용

모델 평가/프린트용으로 GPU array를 CPU array로 변환해야 할 때 사용

---

# 11. MNIST MLP/CNN 학습 실행

```bash
conda activate cupy_env
python 10_mnist_cnn_sequential_cupy.py
```

MLP → 즉시 빠르게 동작
CNN → CuPy로 안전하게 벡터화한 MaxPool2d backward로 정상 속도 유지

---

# 12. 전체 과정 요약 (핵심 체크리스트)

| 단계 | 내용                            | 성공 여부 체크 |
| -- | ----------------------------- | -------- |
| 1  | Windows NVIDIA 드라이버 최신 설치     | ✔        |
| 2  | WSL2 Ubuntu 20/22 버전 확인       | ✔        |
| 3  | WSL 내부 CUDA Toolkit 12.x 설치   | ✔        |
| 4  | PATH/LD_LIBRARY 설정            | ✔        |
| 5  | cupy_env 생성 및 cupy-cuda12x 설치 | ✔        |
| 6  | CuPy 동작 테스트                   | ✔        |
| 7  | backend.py 작성                 | ✔        |
| 8  | 모든 NumPy 코드 → xp API로 변경      | ✔        |
| 9  | MaxPool2d backward 벡터화 적용     | ✔        |
| 10 | CNN/MLP 학습 정상 동작              | ✔        |

---

# 13. 다음에 CuPy/NumPy 코드 변환 시 더 빠르게 시작하는 방법

## 준비해야 할 파일

* `backend.py` (위 버전 그대로 복사)
* `maxpool2d.py` 벡터화 버전
* `conv2d.py` im2col 기반 버전
* 기본 유틸(softmax, cross_entropy 등) xp 버전

매 프로젝트 시작 시:

```python
from backend import backend
xp = backend.xp
```

이 한 줄로 CPU/GPU 전환도 가능.

---

# 14. 필요 시 추가 제공 가능

원하시면:

* **Conv2d backward을 PyTorch 수준으로 최적화**
* **RawKernel을 이용한 고성능 GPU Kernel 커스텀**
* **딥러닝 프레임워크 형태로 구조 리팩토링**
* **CuPy Automatic Mixed Precision 적용**
* **CuPy Memory Pool 및 pinned memory 최적화**

등도 모두 지원해드릴 수 있습니다.

---

이 문서만 있으면
**WSL2 + CUDA + CuPy 설치부터 NumPy GPU 변환까지 모든 과정을 100% 재현 가능**합니다.
