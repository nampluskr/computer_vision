아주 좋은 타이밍에 TensorFlow를 넣으시는 것 같습니다.
이미 WSL2 + CUDA 12.4 + PyTorch / CuPy 환경이 잡혀 있어서, TensorFlow 쪽은 비교적 “깨끗한 새 환경 하나 만들고, 공식 방식으로 설치”만 잘 따라가면 됩니다.

아래는 **WSL2 Ubuntu 22.04 + Anaconda + GPU TensorFlow용 `tensorflow_env` 환경을 만드는 전체 튜토리얼**입니다.
위에서 CuPy 하실 때처럼, 그대로 복붙해서 순서대로 실행하시면 됩니다.

---

## 0. 개념 정리 – TensorFlow GPU와 CUDA 12.4의 관계

* 현재 TensorFlow는 **`pip install "tensorflow[and-cuda]"`** 방식으로 설치하면,

  * TensorFlow + 호환되는 **CUDA Toolkit (12.3)** + **cuDNN (8.9.7)** 을 **패키지 안에 같이 설치**해 줍니다.([TensorFlow][1])
* 즉,

  * WSL 내부에 이미 설치된 `cuda-toolkit-12-4`는 있어도 되고 없어도 됩니다.
  * **TensorFlow용 환경 안에서는 별도로 `cudatoolkit`이나 `cudnn`을 conda로 설치하면 안 됩니다.**
    (충돌/라이브러리 중복 가능성이 있음)
* 이미 PyTorch용 `pytorch_env`, CuPy용 `cupy_env`와 **완전히 분리된 conda env**에서 진행하므로, 기존 환경에는 영향을 주지 않습니다.

---

## 1. 사전 점검 (이미 대부분 OK일 가능성이 높음)

### 1-1. WSL2 Ubuntu 버전 확인

```bash
lsb_release -a
# 또는
cat /etc/os-release
```

Ubuntu 22.04 (jammy) 인지 확인합니다. (이미 확인하셨으니 스킵 가능)

### 1-2. WSL2 내부에서 GPU 인식 확인

WSL Ubuntu 터미널에서:

```bash
nvidia-smi
```

* GPU 모델과 **CUDA Version 12.8 (Driver)** 가 보이면 OK입니다.
* 이 단계는 이미 CuPy / PyTorch에서 GPU가 잘 동작하고 있으니 통과된 상태라고 보시면 됩니다.

### 1-3. WSL2용 CUDA 라이브러리 경로 설정 확인 (있으면 좋음)

WSL2에서 GPU 라이브러리가 `/usr/lib/wsl/lib` 아래에 있습니다.
`~/.bashrc`에 다음과 같은 설정이 있는지 한 번 체크해 보세요 (없으면 추가):

```bash
grep LD_LIBRARY_PATH ~/.bashrc
```

없다면:

```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

이 설정은 TensorFlow뿐 아니라 CuPy/PyTorch에도 도움이 됩니다. ([NVIDIA Docs][2])

---

## 2. TensorFlow용 conda 환경 생성 (`tensorflow_env`)

Anaconda base 환경에서 진행합니다.

```bash
# base 활성화 (이미 되어 있으면 생략)
conda activate base

# Python 3.10 또는 3.11 권장 (TensorFlow는 3.9~3.12 지원) :contentReference[oaicite:2]{index=2}
conda create -n tensorflow_env python=3.10 -y

# 환경 활성화
conda activate tensorflow_env
```

확인:

```bash
python --version
# -> Python 3.10.x
```

---

## 3. TensorFlow GPU 설치 (pip + and-cuda)

TensorFlow 공식 문서 권장 방식 그대로 갑니다.([TensorFlow][1])

### 3-1. pip 업데이트

```bash
pip install --upgrade pip
```

### 3-2. TensorFlow + CUDA + cuDNN 설치

```bash
# GPU 버전
pip install "tensorflow[and-cuda]"
```

이 한 줄로 아래가 자동으로 설치됩니다.

* `tensorflow` (최신 안정 버전, 예: 2.16.x / 2.17.x 등)
* 해당 TF 버전에 맞는

  * `nvidia-cuda-runtime-cu12`
  * `nvidia-cudnn-cu12`
  * 기타 필요한 CUDA 관련 wheel들

**주의사항 (중요)**

* 이 환경 안에서는 아래와 같이 하시면 안 됩니다:

  * `conda install cudatoolkit=...`
  * `conda install cudnn=...`
* 이미 시스템에 설치된 `cuda-toolkit-12-4`와는 별개로,
  TensorFlow용 CUDA 라이브러리가 **가상환경 내부에 wheel 형태로 들어옵니다.**

---

## 4. GPU 인식 여부 확인

`tensorflow_env` 활성화 상태에서 아래 두 줄을 실행합니다.

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

* 정상 예시:

```text
2.16.1
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

* 만약 빈 리스트 `[]` 가 나오면,

  * 윗부분의 `LD_LIBRARY_PATH` 설정이 누락됐거나
  * WSL2 / 드라이버 / CUDA 경로 충돌이 있을 수 있습니다.

TensorFlow 공식 문서에서는, GPU를 못 찾을 때 아래와 같은 **심볼릭 링크 작업**을 권장합니다.([TensorFlow][1])

```bash
# 1) TensorFlow 패키지 디렉토리에서 NVIDIA lib를 링크
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

# 2) ptxas 링크 (필요시)
ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas

# 다시 GPU 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

대부분의 경우 `tensorflow[and-cuda]` 설치 + `LD_LIBRARY_PATH`만 맞으면 위 링크 작업은 필요 없습니다.

---

## 5. Jupyter / VS Code에서 `tensorflow_env` 사용하기

### 5-1. Jupyter 커널 등록

```bash
# tensorflow_env 안에서
pip install ipykernel

python -m ipykernel install --user --name tensorflow_env --display-name "Python (tensorflow_env)"
```

이제:

* VS Code → Jupyter Notebook → **커널 선택**에서 `Python (tensorflow_env)` 선택
* 또는 터미널에서 바로 노트북 실행:

```bash
jupyter notebook  # 또는 jupyter lab
```

노트북에서 아래 코드로 다시 한 번 GPU 확인:

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

---

## 6. 간단한 GPU 테스트 (Keras 예제)

`tensorflow_env` + Jupyter 노트북에서 다음 셀을 실행해 보시면,
GPU 메모리가 올라가는 것을 `nvidia-smi`로 확인하실 수 있습니다.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# 간단한 MNIST CNN 예제
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[..., None] / 255.0
x_test  = x_test[..., None] / 255.0

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss:", test_loss, "Test acc:", test_acc)
```

동시에 다른 터미널에서:

```bash
watch -n 1 nvidia-smi
```

를 실행해 GPU 메모리/연산이 실제로 사용되는지 확인하시면 됩니다.

---

## 7. TensorFlow GPU 환경에서 자주 발생하는 문제와 주의사항

### 7-1. GPU가 안 잡힐 때

1. `nvidia-smi` 먼저 확인 (WSL Ubuntu 내부에서)
2. `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` 설정 확인
3. `pip install tensorflow[and-cuda]` 를 **반드시 TensorFlow 전용 환경에서만** 수행했는지 확인
4. 필요시 TensorFlow 공식 문서의 **심볼릭 링크 작업** 수행([TensorFlow][1])

### 7-2. conda로 `cudatoolkit`, `cudnn` 설치 금지 (이 env 안에서는)

* `tensorflow_env` 안에선:

  * `conda install cudatoolkit`
  * `conda install cudnn`
    를 하지 마십시오.
* 이미 `tensorflow[and-cuda]`가 알맞은 CUDA/cuDNN wheel을 설치하며,
  여기서 다른 버전을 섞으면 라이브러리 충돌로 GPU 인식이 안 되거나, 런타임 에러가 날 수 있습니다.

### 7-3. CuPy / PyTorch와의 공존

* 현재 구조:

  * `pytorch_env` : PyTorch + CuPy (직접 CUDA Toolkit 12.4, CuPy와 연동)
  * `cupy_env`    : 수동으로 작성하신 NumPy→CuPy 딥러닝 코드용
  * `tensorflow_env` : TensorFlow + `tensorflow[and-cuda]` (자체 번들 CUDA)
* 서로 **각자 가상환경 안에서만** 라이브러리를 사용하므로,

  * 한 환경에서 문제가 생겨도 다른 환경에는 영향을 주지 않습니다.
* 단, **같은 터미널 세션 내에서 환경을 헷갈리지 않는 것**만 주의하시면 됩니다:

  * 프롬프트 앞에 `(tensorflow_env)`, `(pytorch_env)`, `(cupy_env)`가 어떤지 늘 확인

---

## 8. 메모리 관리 (옵션)

TensorFlow는 기본적으로 GPU 메모리를 크게 잡으려는 경향이 있습니다.
CuPy/PyTorch와 같이 사용하거나, 여러 실험을 반복할 때는 다음 설정을 같이 써주시면 좋습니다.

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

* 이 코드를 **TensorFlow 사용하는 노트북/스크립트의 맨 위**에 넣으면,

  * 필요할 때만 GPU 메모리를 조금씩 늘려 사용합니다.

---

## 9. 정리 – 앞으로 TensorFlow GPU 환경을 다시 만들 때 요약

나중에 환경을 새로 만들 때, 최소한으로 기억하면 되는 요약입니다.

1. WSL2 Ubuntu에서 `nvidia-smi` 확인
2. `conda create -n tensorflow_env python=3.10 -y`
3. `conda activate tensorflow_env`
4. `pip install --upgrade pip`
5. `pip install "tensorflow[and-cuda]"` ([TensorFlow][1])
6. `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
7. Jupyter 커널 등록: `python -m ipykernel install --user --name tensorflow_env --display-name "Python (tensorflow_env)"`

---

혹시 이 과정 중 어느 단계에서든 에러 메시지가 나오면,
에러 내용 전체를 복사해 주시면, 어떤 단계에서 꼬였는지 같이 디버깅해 드리겠습니다.

[1]: https://www.tensorflow.org/install/pip "Install TensorFlow with pip"
[2]: https://docs.nvidia.com/cuda/wsl-user-guide/index.html?utm_source=chatgpt.com "CUDA on WSL User Guide"
