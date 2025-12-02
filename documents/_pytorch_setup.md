# WSL2 + Anaconda + PyTorch 완전 설치 매뉴얼

**Windows 11 + WSL2 + NVIDIA GPU 환경에서 PyTorch 딥러닝 환경 구축하기**

---

## 목차
1. [Windows 11 설치 및 기본 설정](#1-windows-11-설치-및-기본-설정)
2. [NVIDIA GPU 드라이버 설치](#2-nvidia-gpu-드라이버-설치)
3. [WSL2 설치 및 설정](#3-wsl2-설치-및-설정)
4. [Anaconda 설치](#4-anaconda-설치)
5. [PyTorch 환경 구축](#5-pytorch-환경-구축)
6. [설치 검증](#6-설치-검증)
7. [문제 해결](#7-문제-해결)

---

## 1. Windows 11 설치 및 기본 설정

### 1-1. Windows 업데이트 확인
```
설정 → Windows Update → 업데이트 확인
```
최신 버전으로 업데이트 완료

### 1-2. 시스템 사양 확인
- Windows 11 버전: 21H2 이상
- NVIDIA GPU: GTX 1000 시리즈 이상 권장
- RAM: 16GB 이상 권장
- 저장공간: SSD 50GB 이상 여유 공간

---

## 2. NVIDIA GPU 드라이버 설치

### 2-1. 현재 드라이버 확인 (선택사항)
**PowerShell** 실행:
```powershell
nvidia-smi
```

### 2-2. 최신 드라이버 다운로드 및 설치

**방법 1: NVIDIA 공식 사이트 (권장)**
1. https://www.nvidia.com/Download/index.aspx 접속
2. GPU 모델 선택 (예: GeForce GTX 1080 Ti)
3. 운영체제: Windows 11 선택
4. **Game Ready Driver** 또는 **Studio Driver** 다운로드
5. 다운로드한 파일 실행 → 사용자 지정 설치 → 클린 설치 체크

**방법 2: GeForce Experience (간편)**
1. https://www.nvidia.com/en-us/geforce/geforce-experience/ 에서 다운로드
2. 설치 후 실행 → 드라이버 탭 → 업데이트 확인

### 2-3. 설치 확인
재부팅 후 **PowerShell**에서:
```powershell
nvidia-smi
```

정상 출력 예시:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.70                 Driver Version: 572.70         CUDA Version: 12.8     |
+-----------------------------------------------------------------------------------------+
| GPU  Name                           | Bus-Id          | GPU-Util  Memory-Usage         |
|   0  NVIDIA GeForce GTX 1080 Ti     | 00000000:01:00.0|    0%        752MiB / 11264MiB|
+-----------------------------------------------------------------------------------------+
```

**중요**: `CUDA Version: 12.x`가 표시되면 정상

---

## 3. WSL2 설치 및 설정

### 3-1. WSL 기능 활성화

**PowerShell을 관리자 권한으로 실행**:

```powershell
# WSL 설치 (Ubuntu 22.04 자동 설치)
wsl --install

# 재부팅 필요
```

**재부팅 후** 자동으로 Ubuntu 터미널이 실행되며 사용자 이름과 비밀번호 설정 요구

### 3-2. 기존 WSL이 있는 경우 업데이트

```powershell
# WSL 업데이트
wsl --update

# WSL 버전 확인
wsl --version

# 설치된 배포판 확인
wsl --list --verbose
```

### 3-3. Ubuntu 22.04가 아닌 경우

```powershell
# Ubuntu 22.04 설치
wsl --install -d Ubuntu-22.04

# 기본 배포판으로 설정
wsl --set-default Ubuntu-22.04
```

### 3-4. WSL2 시스템 업데이트

**WSL2 Ubuntu 터미널**에서:

```bash
# 패키지 목록 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 빌드 도구 설치
sudo apt install -y build-essential git wget curl vim
```

### 3-5. GPU 패스스루 확인

```bash
nvidia-smi
```

정상 출력되면 WSL2에서 GPU 사용 가능

---

## 4. Anaconda 설치

### 4-1. Anaconda 다운로드

**WSL2 Ubuntu 터미널**에서:

```bash
# 홈 디렉토리로 이동
cd ~

# Anaconda 최신 버전 다운로드
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

**최신 버전 확인**: https://repo.anaconda.com/archive/

### 4-2. Anaconda 설치

```bash
# 설치 실행
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

**설치 중 선택사항**:
1. **Enter** 키를 눌러 라이센스 읽기 (스페이스바로 빠르게 스크롤)
2. `yes` 입력 → 라이센스 동의
3. **Enter** 키 → 기본 설치 경로 사용 (`/home/사용자명/anaconda3`)
4. `yes` 입력 → **conda init 실행** (매우 중요!)

### 4-3. 설치 파일 삭제 및 적용

```bash
# 설치 파일 삭제
rm Anaconda3-2024.10-1-Linux-x86_64.sh

# 설정 적용
source ~/.bashrc

# conda 버전 확인
conda --version
```

출력 예: `conda 24.7.1`

### 4-4. Conda 기본 설정

```bash
# conda 업데이트
conda update -n base conda -y

# conda-forge 채널 추가
conda config --add channels conda-forge

# 채널 우선순위 확인
conda config --show channels
```

---

## 5. PyTorch 환경 구축

### 5-1. PyTorch 전용 환경 생성

```bash
# Python 3.10으로 환경 생성 (안정성 최고)
conda create -n pytorch_env python=3.10 -y

# 환경 활성화
conda activate pytorch_env

# 프롬프트가 (pytorch_env)로 변경됨 확인
```

### 5-2. NumPy 먼저 설치 (중요!)

```bash
# conda로 numpy 설치 (MKL 의존성 포함)
conda install numpy -y
```

**왜 먼저 설치?**
- PyTorch가 NumPy의 MKL 라이브러리를 필요로 함
- conda로 설치하면 MKL이 자동으로 포함됨
- 순서를 지키지 않으면 `libmkl_rt.so.2` 에러 발생 가능

### 5-3. PyTorch 설치

```bash
# pip로 PyTorch 설치 (CUDA 12.1 버전)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 버전 선택 가이드**:
- `nvidia-smi`에서 확인한 CUDA Version이 **12.x** → `cu121` 사용
- CUDA Version이 **11.x** → `cu118` 사용

```bash
# CUDA 11.8용 (참고)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5-4. TorchMetrics 설치

```bash
pip install torchmetrics
```

### 5-5. 추가 필수 패키지 설치

```bash
# 데이터 처리 및 시각화 (conda로 설치)
conda install -y pandas matplotlib seaborn jupyter

# Computer Vision 라이브러리
pip install opencv-python pillow albumentations timm

# 유틸리티
pip install tqdm scikit-learn

# 학습 모니터링 도구
pip install tensorboard wandb

# 추가 딥러닝 도구
pip install lightning einops
```

### 5-6. 패키지 목록 저장 (백업용)

```bash
# conda 환경 내 패키지 목록 저장
conda list --export > pytorch_env_packages.txt

# pip 패키지 목록 저장
pip freeze > requirements.txt
```

---

## 6. 설치 검증

### 6-1. PyTorch 기본 테스트

```bash
conda activate pytorch_env
python << EOF
import torch
print("="*50)
print("PyTorch Installation Check")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("="*50)
EOF
```

**정상 출력 예시**:
```
==================================================
PyTorch Installation Check
==================================================
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
cuDNN version: 90100
Number of GPUs: 1
GPU Name: NVIDIA GeForce GTX 1080 Ti
GPU Memory: 11.00 GB
==================================================
```

### 6-2. GPU 연산 테스트

```bash
python << EOF
import torch

# GPU에서 행렬 곱셈 테스트
print("GPU Computation Test...")
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f"Matrix multiplication on GPU: SUCCESS")
print(f"Result shape: {z.shape}")
print(f"Allocated GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
EOF
```

### 6-3. TorchMetrics 테스트

```bash
python << EOF
import torch
from torchmetrics import Accuracy

# 정확도 메트릭 테스트
acc = Accuracy(task="multiclass", num_classes=10).cuda()
preds = torch.randn(10, 10).cuda()
target = torch.randint(0, 10, (10,)).cuda()
result = acc(preds, target)
print(f"TorchMetrics test: SUCCESS")
print(f"Accuracy: {result.item():.4f}")
EOF
```

### 6-4. 간단한 신경망 학습 테스트

```bash
python << EOF
import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 모델 정의
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
).cuda()

# 더미 데이터
x = torch.randn(32, 10).cuda()
y = torch.randint(0, 10, (32,)).cuda()

# 학습
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nTraining test: SUCCESS")
EOF
```

---

## 7. 문제 해결

### 7-1. `libmkl_rt.so.2: cannot open shared object file` 에러

**원인**: NumPy의 MKL 라이브러리 누락

**해결**:
```bash
conda activate pytorch_env
conda install numpy -y
```

### 7-2. `undefined symbol: iJIT_NotifyEvent` 에러

**원인**: Intel MKL 라이브러리 충돌

**해결**:
```bash
conda activate pytorch_env

# 기존 패키지 제거
pip uninstall torch torchvision torchaudio -y
conda uninstall pytorch torchvision torchaudio mkl mkl-service --force -y

# numpy 재설치
conda install numpy -y

# PyTorch 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 7-3. `CUDA out of memory` 에러

**원인**: GPU 메모리 부족

**해결**:
```python
# 배치 사이즈 줄이기
batch_size = 32  # → 16 또는 8로 변경

# 사용하지 않는 변수 삭제
del x, y, z
torch.cuda.empty_cache()

# 혼합 정밀도 학습 사용
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### 7-4. WSL2에서 `nvidia-smi` 실행 안됨

**원인**: Windows NVIDIA 드라이버 미설치 또는 구버전

**해결**:
1. Windows에서 최신 NVIDIA 드라이버 재설치
2. WSL2 재시작: `wsl --shutdown` (PowerShell)
3. WSL2 다시 실행

### 7-5. conda 명령이 작동하지 않음

**원인**: conda init 미실행

**해결**:
```bash
source ~/anaconda3/bin/activate
conda init bash
source ~/.bashrc
```

---

## 8. 환경 관리 명령어 모음

### 8-1. 환경 활성화/비활성화
```bash
# 환경 활성화
conda activate pytorch_env

# 환경 비활성화
conda deactivate

# 기본(base) 환경으로
conda activate base
```

### 8-2. 환경 목록 및 정보
```bash
# 모든 환경 목록
conda env list

# 현재 환경의 패키지 목록
conda list

# 특정 환경의 패키지 목록
conda list -n pytorch_env
```

### 8-3. 환경 백업 및 복원
```bash
# 환경 내보내기
conda env export > pytorch_env.yml

# 환경 복원 (다른 시스템에서)
conda env create -f pytorch_env.yml
```

### 8-4. 환경 삭제
```bash
# 환경 삭제
conda env remove -n pytorch_env

# 확인
conda env list
```

---

## 9. 자주 사용하는 단축 명령 설정 (선택사항)

```bash
# .bashrc에 별칭 추가
cat >> ~/.bashrc << 'EOF'

# PyTorch 환경 단축 명령
alias pt='conda activate pytorch_env'
alias ptoff='conda deactivate'

# GPU 상태 확인
alias gpu='nvidia-smi'

# Jupyter Notebook 실행
alias jn='jupyter notebook --no-browser'
EOF

# 적용
source ~/.bashrc
```

이제 다음과 같이 사용 가능:
- `pt` → pytorch_env 활성화
- `ptoff` → 환경 비활성화
- `gpu` → GPU 상태 확인
- `jn` → Jupyter Notebook 실행

---

## 10. 전체 설치 체크리스트

| 단계 | 내용 | 확인 |
|------|------|------|
| 1 | Windows 11 최신 업데이트 | ☐ |
| 2 | NVIDIA 드라이버 설치 (572.x 이상) | ☐ |
| 3 | `nvidia-smi` 정상 작동 (Windows) | ☐ |
| 4 | WSL2 설치 및 Ubuntu 22.04 | ☐ |
| 5 | WSL2에서 `nvidia-smi` 정상 작동 | ☐ |
| 6 | WSL2 시스템 업데이트 | ☐ |
| 7 | Anaconda 설치 | ☐ |
| 8 | `conda --version` 확인 | ☐ |
| 9 | pytorch_env 환경 생성 | ☐ |
| 10 | NumPy 설치 (conda) | ☐ |
| 11 | PyTorch 설치 (pip, cu121) | ☐ |
| 12 | TorchMetrics 설치 | ☐ |
| 13 | PyTorch CUDA 사용 가능 확인 | ☐ |
| 14 | GPU 연산 테스트 성공 | ☐ |
| 15 | 간단한 학습 테스트 성공 | ☐ |

---

## 11. 다음 단계 학습 자료

설치가 완료되면 다음 단계로 진행하세요:

1. **PyTorch 공식 튜토리얼**
   - https://pytorch.org/tutorials/

2. **Computer Vision 학습**
   - MNIST 손글씨 분류
   - CIFAR-10 이미지 분류
   - Transfer Learning (ResNet, EfficientNet 등)

3. **TorchMetrics 활용**
   - https://torchmetrics.readthedocs.io/

4. **Weights & Biases 실험 관리**
   - https://wandb.ai/

---

## 12. 참고 링크

- **PyTorch 공식 사이트**: https://pytorch.org/
- **NVIDIA 드라이버 다운로드**: https://www.nvidia.com/Download/index.aspx
- **WSL2 공식 문서**: https://learn.microsoft.com/en-us/windows/wsl/
- **Anaconda 다운로드**: https://www.anaconda.com/download
- **TorchMetrics**: https://lightning.ai/docs/torchmetrics/

---

**이 매뉴얼을 따라하면 시행착오 없이 PyTorch 환경을 구축할 수 있습니다!**