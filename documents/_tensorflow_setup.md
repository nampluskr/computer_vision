GTX 1080 Ti는 TensorFlow를 충분히 지원합니다! 현재 깨끗하게 재설치한 상태이므로, 시행착오 없이 TensorFlow 환경을 구축할 수 있습니다.

## TensorFlow + CUDA 환경 구축 전략

### 중요한 포인트
1. **시스템 CUDA Toolkit 설치 불필요** - conda가 모두 처리
2. **TensorFlow 2.15+ 버전 사용** - CUDA 12.x 지원
3. **독립된 conda 환경** - 다른 환경과 충돌 없음

---

## TensorFlow 환경 구축 매뉴얼

### 1. TensorFlow 환경 생성

```bash
# 환경 생성
conda create -n tensorflow_env python=3.10 -y

# 환경 활성화
conda activate tensorflow_env
```

### 2. TensorFlow 설치 (CUDA 포함)

**방법 1: pip 설치 (권장 - 최신 버전)**

```bash
# TensorFlow 최신 버전 설치 (CUDA 자동 포함)
pip install tensorflow[and-cuda]

# 또는 특정 버전
pip install tensorflow[and-cuda]==2.15.0
```

**방법 2: conda 설치**

```bash
# conda로 설치 (CUDA 의존성 자동 처리)
conda install -c conda-forge tensorflow-gpu -y
```

### 3. 추가 패키지 설치

```bash
# 데이터 처리 및 시각화
conda install -y numpy pandas matplotlib seaborn jupyter

# Computer Vision
pip install opencv-python pillow

# 유틸리티
pip install tqdm scikit-learn

# TensorFlow 추가 도구
pip install tensorboard keras
```

### 4. 설치 확인

```bash
python << EOF
import tensorflow as tf
print("="*50)
print("TensorFlow Installation Check")
print("="*50)
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# GPU 목록
gpus = tf.config.list_physical_devices('GPU')
print(f"\nNumber of GPUs: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu}")
    
# GPU 세부 정보
if gpus:
    gpu_details = tf.config.experimental.get_device_details(gpus[0])
    print(f"\nGPU Details:")
    for key, value in gpu_details.items():
        print(f"  {key}: {value}")

print("="*50)
print("✓ TensorFlow installation SUCCESS!")
EOF
```

### 5. GPU 연산 테스트

```bash
python << EOF
import tensorflow as tf
import time

print("\n=== GPU Computation Test ===")

# GPU에서 연산 실행
with tf.device('/GPU:0'):
    # 큰 행렬 생성
    a = tf.random.normal([5000, 5000])
    b = tf.random.normal([5000, 5000])
    
    # 연산 시간 측정
    start = time.time()
    c = tf.matmul(a, b)
    tf.keras.backend.clear_session()
    gpu_time = time.time() - start
    
    print(f"Matrix multiplication (5000x5000)")
    print(f"GPU computation time: {gpu_time:.4f} seconds")
    print(f"Result shape: {c.shape}")
    print("✓ GPU computation SUCCESS!")
EOF
```

### 6. 간단한 신경망 학습 테스트

```bash
python << EOF
import tensorflow as tf
from tensorflow import keras
import numpy as np

print("\n=== Neural Network Training Test ===")

# 간단한 데이터셋
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

# 모델 생성
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 학습
print("\nTraining on GPU...")
history = model.fit(
    x_train[:1000], y_train[:1000],
    batch_size=32,
    epochs=3,
    validation_split=0.2,
    verbose=1
)

print("\n✓ Training test SUCCESS!")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
EOF
```

---

## 문제 해결 가이드

### 문제 1: `Could not load dynamic library 'libcudnn.so'`

**원인**: cuDNN 라이브러리 미설치

**해결**:
```bash
conda activate tensorflow_env
conda install -c conda-forge cudnn -y
```

### 문제 2: GPU가 인식되지 않음

**확인**:
```bash
# WSL에서 GPU 확인
nvidia-smi

# TensorFlow가 GPU를 보는지 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**해결**: 출력이 비어있으면
```bash
# TensorFlow 재설치
pip uninstall tensorflow -y
pip install tensorflow[and-cuda]
```

### 문제 3: 메모리 부족 에러

**해결**:
```python
import tensorflow as tf

# GPU 메모리 동적 할당 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 또는 메모리 제한 설정
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]  # 8GB
)
```

---

## 3개 환경 비교 및 관리

### 환경 요약

| 환경 | 용도 | 주요 라이브러리 | CUDA 설치 방법 |
|------|------|----------------|---------------|
| pytorch_env | PyTorch 딥러닝 | PyTorch, torchvision, torchmetrics | pip (포함) |
| cupy_env | NumPy GPU 가속 | CuPy, NumPy, SciPy | conda (포함) |
| tensorflow_env | TensorFlow 딥러닝 | TensorFlow, Keras | pip (포함) |

### 환경 전환 명령어

```bash
# PyTorch 환경
conda activate pytorch_env

# CuPy 환경
conda activate cupy_env

# TensorFlow 환경
conda activate tensorflow_env

# 환경 비활성화
conda deactivate
```

### 단축 명령 설정

```bash
cat >> ~/.bashrc << 'EOF'

# 딥러닝 환경 단축 명령
alias pt='conda activate pytorch_env'
alias cu='conda activate cupy_env'
alias tf='conda activate tensorflow_env'
alias ca='conda deactivate'
alias gpu='nvidia-smi'
EOF

source ~/.bashrc
```

---

## 전체 설치 체크리스트

| 단계 | 내용 | 확인 |
|------|------|------|
| 1 | tensorflow_env 환경 생성 | ☐ |
| 2 | TensorFlow 설치 (CUDA 포함) | ☐ |
| 3 | `import tensorflow` 성공 | ☐ |
| 4 | GPU 인식 확인 | ☐ |
| 5 | GPU 연산 테스트 | ☐ |
| 6 | MNIST 학습 테스트 | ☐ |
| 7 | 추가 패키지 설치 | ☐ |

---

## 핵심 포인트

✅ **시스템 CUDA Toolkit 설치 불필요** - pip/conda가 자동 처리
✅ **환경 독립성** - 각 환경이 자체 CUDA 라이브러리 포함
✅ **충돌 없음** - pytorch_env, cupy_env, tensorflow_env 모두 공존 가능
✅ **GTX 1080 Ti 완벽 지원** - Compute Capability 6.1

---

TensorFlow의 warning 메시지들을 정리해드리겠습니다. 대부분은 정보성 메시지이므로 숨겨도 괜찮습니다.

## Warning 메시지 분석 및 제거

### 1. 각 Warning 설명

| Warning | 의미 | 숨겨도 되는가? |
|---------|------|--------------|
| `CPU instructions AVX2 FMA` | CPU 최적화 관련 정보 | ✅ 예 (성능에 큰 영향 없음) |
| `absl::InitializeLog()` | 로깅 시스템 초기화 정보 | ✅ 예 (정보성 메시지) |
| `input_shape` deprecation | Keras API 변경 경고 | ⚠️ 코드 수정 권장 |
| XLA/cuDNN 로드 메시지 | GPU 컴파일러 정보 | ✅ 예 (정상 작동 확인) |

---

## 방법 1: 환경 변수로 Warning 숨기기 (권장)

### 영구 설정 (.bashrc에 추가)

```bash
cat >> ~/.bashrc << 'EOF'

# TensorFlow Warning 제거
export TF_CPP_MIN_LOG_LEVEL=2  # 0=모두표시, 1=INFO숨김, 2=WARNING숨김, 3=ERROR만표시
export TF_ENABLE_ONEDNN_OPTS=0  # oneDNN 최적화 메시지 숨김
EOF

source ~/.bashrc
```

### 임시 설정 (현재 세션만)

```bash
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
```

---

## 방법 2: Python 코드 내에서 설정

### 스크립트 시작 부분에 추가

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # WARNING 이상만 표시
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# absl 로깅도 숨기기
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
```

### 완전히 깨끗한 출력을 원할 때

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR만 표시
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# 이제 TensorFlow 코드 실행
```

---

## 방법 3: 레벨별 제어 (추천)

### 개발 중 (정보 많이 보기)

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 모든 로그 표시
import tensorflow as tf
```

### 학습 중 (중요한 것만)

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # INFO 숨김
import tensorflow as tf
```

### 배포/데모 (깨끗한 출력)

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR만 표시
import tensorflow as tf
```

---

## Warning별 개별 제어

### 1. CPU 최적화 메시지 숨기기

```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```

### 2. Keras Deprecation Warning 해결

**현재 코드** (warning 발생):
```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # ⚠️
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
```

**수정된 코드** (warning 없음):
```python
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),  # ✅ Input 레이어 사용
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
```

### 3. XLA/cuDNN 메시지는 유용함 (첫 실행 시만 표시됨)

이 메시지들은 GPU가 정상 작동하는지 확인할 수 있어 유용합니다:
```
Loaded cuDNN version 91600
Created device /job:localhost/replica:0/task:0/device:GPU:0
```

첫 실행 후에는 자동으로 줄어듭니다.

---

## 테스트: Warning 제거 후

### 깨끗한 출력 예시

```bash
conda activate tensorflow_env

python << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("\n=== Clean TensorFlow Test ===")

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs: {len(gpus)}")

# 간단한 연산
with tf.device('/GPU:0'):
    a = tf.random.normal([3000, 3000])
    b = tf.random.normal([3000, 3000])
    c = tf.matmul(a, b)
    print(f"Matrix multiplication completed")
    print(f"Result shape: {c.shape}")

print("✓ Test completed without warnings!")
EOF
```

**출력 (깨끗함)**:
```
=== Clean TensorFlow Test ===
Number of GPUs: 1
Matrix multiplication completed
Result shape: (3000, 3000)
✓ Test completed without warnings!
```

---

## 추천 설정 템플릿

### 파일 생성: `~/tf_config.py`

```bash
cat > ~/tf_config.py << 'EOF'
"""
TensorFlow 기본 설정 - 모든 스크립트에서 import
Usage: from tf_config import *
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # WARNING 숨김
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # CPU 최적화 메시지 숨김

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# GPU 메모리 동적 할당
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

print("TensorFlow configured: warnings minimized, GPU memory growth enabled")
EOF
```

### 사용법

```python
from tf_config import *  # TensorFlow 자동 설정
# 이제 깨끗한 출력으로 코드 작성
```

---

## PyTorch와 CuPy도 깨끗하게

### PyTorch Warning 제거

```python
import warnings
warnings.filterwarnings('ignore')

import torch
# 깨끗한 출력
```

### CuPy Warning 제거

```python
import warnings
warnings.filterwarnings('ignore')

import cupy as cp
# 깨끗한 출력
```

---

## 최종 권장 설정

### ~/.bashrc에 추가 (모든 환경에 적용)

```bash
cat >> ~/.bashrc << 'EOF'

# TensorFlow Warning 최소화
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Python Warning 최소화 (선택사항)
# export PYTHONWARNINGS="ignore"
EOF

source ~/.bashrc
```

이제 TensorFlow를 실행하면 훨씬 깨끗한 출력을 볼 수 있습니다! 

필요할 때만 `TF_CPP_MIN_LOG_LEVEL=0`으로 설정하여 디버깅할 수 있습니다.