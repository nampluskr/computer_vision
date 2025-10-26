# Models - 상세 모델 설명

## 목차

1. [개요](#1-개요)
2. [모델 카테고리](#2-모델-카테고리)
3. [Memory-Based Models](#3-memory-based-models)
4. [Normalizing Flow Models](#4-normalizing-flow-models)
5. [Knowledge Distillation Models](#5-knowledge-distillation-models)
6. [Reconstruction-Based Models](#6-reconstruction-based-models)
7. [Feature Adaptation Models](#7-feature-adaptation-models)
8. [Foundation Models](#8-foundation-models)
9. [모델 선택 가이드](#9-모델-선택-가이드)
10. [모델별 설정](#10-모델별-설정)

---

## 1. 개요

본 프레임워크는 34개의 고유 모델과 44개의 구성을 지원합니다. 각 모델은 서로 다른 접근 방식을 사용하여 이상을 감지하며, 데이터셋과 응용 분야에 따라 성능이 다릅니다.

### 1.1. 모델 분류 체계

| 카테고리 | 모델 수 | 특징 | 대표 모델 |
|---------|--------|------|----------|
| Memory-Based | 3 | 학습 불필요, 빠른 추론 | PaDiM, PatchCore |
| Normalizing Flow | 4 | 확률 기반, 높은 정확도 | FastFlow, CFlow |
| Knowledge Distillation | 4 | Teacher-Student 구조 | STFPM, EfficientAD |
| Reconstruction | 4 | 재구성 오차 기반 | Autoencoder, DRAEM |
| Feature Adaptation | 2 | 특징 적응 학습 | DFM, CFA |
| Foundation Models | 3 | 대규모 사전학습 모델 | Dinomaly, UniNet |

### 1.2. 성능 vs 속도 트레이드오프

```
High Performance
    │
    │  ● Dinomaly
    │  ● EfficientAD
    │  ● PatchCore
    │     ● FastFlow
    │        ● STFPM
    │           ● PaDiM
    │              ● Autoencoder
    │
    └─────────────────────────────> Fast Inference
```

---

## 2. 모델 카테고리

### 2.1. 카테고리별 특성

#### Memory-Based Models
- **원리**: 정상 샘플의 특징을 메모리에 저장하고 거리 기반 비교
- **장점**: 학습 불필요, 빠른 실행, 해석 가능
- **단점**: 메모리 사용량 높음, 대규모 데이터에 부적합
- **적용**: 소규모 데이터셋, 빠른 프로토타이핑

#### Normalizing Flow Models
- **원리**: 정규화 플로우를 통한 확률 밀도 추정
- **장점**: 높은 정확도, 이론적 근거 명확
- **단점**: 학습 시간 오래 걸림, 복잡한 구조
- **적용**: 고정밀 검사, 연구 목적

#### Knowledge Distillation Models
- **원리**: Teacher 네트워크가 Student 네트워크를 지도
- **장점**: 높은 정확도, 경량화 가능
- **단점**: Teacher 네트워크 필요
- **적용**: 산업 응용, 실시간 검사

#### Reconstruction-Based Models
- **원리**: 정상 패턴 재구성 후 오차 측정
- **장점**: 직관적, 구현 간단
- **단점**: 재구성 품질에 의존, 오버피팅 위험
- **적용**: 기본 솔루션, 교육 목적

#### Feature Adaptation Models
- **원리**: 도메인 특화 특징 학습
- **장점**: 전이 학습 효과적
- **단점**: 하이퍼파라미터 튜닝 필요
- **적용**: 다양한 도메인 적응

#### Foundation Models
- **원리**: 대규모 사전학습 모델 활용
- **장점**: 최신 성능, 범용성
- **단점**: 리소스 요구량 높음
- **적용**: 최신 연구, 고성능 요구 응용

---

## 3. Memory-Based Models

### 3.1. PaDiM (2020)

#### 개요
Patch Distribution Modeling을 통한 이상 감지 방법입니다.

**논문**: [A Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/abs/2011.08785)

#### 동작 원리

1. **특징 추출**: 사전학습된 CNN의 다중 레이어에서 특징 추출
2. **차원 축소**: PCA를 통한 특징 차원 축소
3. **분포 모델링**: 각 패치 위치에서 다변량 가우시안 분포 추정
4. **이상 점수**: Mahalanobis 거리 계산

#### 모델 구성

```python
ModelRegistry.register("padim", "models.model_padim.PadimTrainer",
    dict(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"]
    ),
    dict(
        num_epochs=1,
        batch_size=4,
        normalize=True,
        img_size=256
    )
)
```

#### 특징

- **학습 시간**: 매우 빠름 (1 epoch)
- **추론 속도**: 중간
- **메모리**: 높음
- **정확도**: 중상

#### 사용 예시

```python
train("mvtec", "bottle", "padim", num_epochs=1)
```

#### 장단점

**장점:**
- 학습 불필요 (단일 epoch)
- 구현 간단
- 해석 가능한 이상 맵

**단점:**
- 메모리 사용량 높음
- 대규모 이미지에 부적합

---

### 3.2. PatchCore (2022)

#### 개요
Coreset 샘플링을 통한 효율적인 메모리 기반 이상 감지입니다.

**논문**: [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)

#### 동작 원리

1. **특징 추출**: Wide ResNet-50-2의 중간 레이어 사용
2. **Coreset 샘플링**: K-Center Greedy 알고리즘으로 대표 특징 선택
3. **이상 점수**: K-NN 기반 거리 계산

#### 모델 구성

```python
ModelRegistry.register("patchcore", "models.model_patchcore.PatchcoreTrainer",
    dict(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"]
    ),
    dict(
        num_epochs=1,
        batch_size=8,
        normalize=True,
        img_size=256
    )
)
```

#### 특징

- **학습 시간**: 빠름 (1 epoch)
- **추론 속도**: 빠름
- **메모리**: 중간 (Coreset 샘플링으로 감소)
- **정확도**: 매우 높음

#### 사용 예시

```python
train("mvtec", "bottle", "patchcore", num_epochs=1)
```

#### 장단점

**장점:**
- 높은 정확도
- 효율적인 메모리 사용
- 빠른 추론

**단점:**
- Coreset 샘플링 시간 필요
- 하이퍼파라미터 민감

---

### 3.3. DFKDE (2022)

#### 개요
Deep Feature Kernel Density Estimation을 통한 이상 감지입니다.

**논문**: Deep Feature Kernel Density Estimation (Anomalib)

#### 모델 구성

```python
ModelRegistry.register("dfkde", "models.model_dfkde.DFKDETrainer",
    dict(
        backbone="resnet50",
        layers=["layer4"],
        pre_trained=True
    ),
    dict(
        num_epochs=1,
        batch_size=8,
        normalize=True,
        img_size=256
    )
)
```

---

## 4. Normalizing Flow Models

### 4.1. CFlow (2021)

#### 개요
Conditional Normalizing Flows를 이용한 실시간 이상 감지입니다.

**논문**: [Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows](https://arxiv.org/abs/2107.12571)

#### 동작 원리

1. **특징 추출**: ResNet에서 다중 스케일 특징 추출
2. **조건부 플로우**: 위치 인코딩을 조건으로 하는 플로우 학습
3. **확률 계산**: Log-likelihood 기반 이상 점수

#### 모델 구성

```python
# CFlow with ResNet-18
ModelRegistry.register("cflow-resnet18", "models.model_cflow.CflowTrainer",
    dict(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"]
    ),
    dict(
        num_epochs=250,
        batch_size=8,
        normalize=True,
        img_size=256
    )
)

# CFlow with ResNet-50
ModelRegistry.register("cflow-resnet50", "models.model_cflow.CflowTrainer",
    dict(
        backbone="resnet50",
        layers=["layer1", "layer2", "layer3"]
    ),
    dict(
        num_epochs=250,
        batch_size=8,
        normalize=True,
        img_size=256
    )
)
```

#### 특징

- **학습 시간**: 오래 걸림 (250 epochs)
- **추론 속도**: 빠름
- **메모리**: 중간
- **정확도**: 높음

---

### 4.2. FastFlow (2021)

#### 개요
2D Normalizing Flows를 통한 빠른 이상 감지입니다.

**논문**: [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/abs/2111.07677)

#### 동작 원리

1. **특징 추출**: 사전학습된 백본 (ResNet, ViT)
2. **2D 플로우**: 2차원 정규화 플로우 학습
3. **빠른 추론**: 단순한 구조로 빠른 속도

#### 모델 구성

```python
# FastFlow with ResNet-50
ModelRegistry.register("fastflow-resnet50", "models.model_fastflow.FastflowTrainer",
    dict(
        backbone="resnet50",
        flow_steps=8
    ),
    dict(
        num_epochs=500,
        batch_size=8,
        normalize=True,
        img_size=256
    )
)

# FastFlow with CaiT
ModelRegistry.register("fastflow-cait", "models.model_fastflow.FastflowTrainer",
    dict(
        backbone="cait_m48_448",
        flow_steps=8
    ),
    dict(
        num_epochs=500,
        batch_size=2,
        normalize=True,
        img_size=448
    )
)

# FastFlow with DeiT
ModelRegistry.register("fastflow-deit", "models.model_fastflow.FastflowTrainer",
    dict(
        backbone="deit_base_distilled_patch16_384",
        flow_steps=8
    ),
    dict(
        num_epochs=500,
        batch_size=2,
        normalize=True,
        img_size=384
    )
)
```

#### 특징

- **학습 시간**: 매우 오래 걸림 (500 epochs)
- **추론 속도**: 매우 빠름
- **메모리**: 중간
- **정확도**: 매우 높음

#### Backbone 선택

- **resnet50**: 균형잡힌 성능
- **cait_m48_448**: 최고 정확도, 느린 학습
- **deit_base_distilled_patch16_384**: 중간 성능

---

### 4.3. CS-Flow (2021)

#### 개요
Cross-Scale Flows를 통한 다중 스케일 이상 감지입니다.

**논문**: [Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection](https://arxiv.org/abs/2110.02855)

#### 모델 구성

```python
ModelRegistry.register("csflow", "models.model_csflow.CsflowTrainer",
    dict(
        backbone="efficientnet_b5",
        n_coupling_blocks=4
    ),
    dict(
        num_epochs=500,
        batch_size=2,
        normalize=True,
        img_size=256
    )
)
```

---

### 4.4. U-Flow (2022)

#### 개요
U-shaped Normalizing Flow with Unsupervised Threshold입니다.

**논문**: [A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold](https://arxiv.org/abs/2211.12353)

#### 모델 구성

```python
# U-Flow with ResNet-50
ModelRegistry.register("uflow-resnet50", "models.model_uflow.UflowTrainer",
    dict(
        backbone="resnet50",
        flow_steps=4
    ),
    dict(
        num_epochs=500,
        batch_size=8,
        normalize=True,
        img_size=256
    )
)

# U-Flow with Multi-CaiT
ModelRegistry.register("uflow-mcait", "models.model_uflow.UflowTrainer",
    dict(
        backbone="mcait",
        flow_steps=4
    ),
    dict(
        num_epochs=500,
        batch_size=4,
        normalize=True,
        img_size=448
    )
)
```

---

## 5. Knowledge Distillation Models

### 5.1. STFPM (2021)

#### 개요
Student-Teacher Feature Pyramid Matching을 통한 이상 감지입니다.

**논문**: [Student-Teacher Feature Pyramid Matching for Anomaly Detection](https://arxiv.org/abs/2103.04257)

#### 동작 원리

1. **Teacher 네트워크**: 사전학습된 ResNet (frozen)
2. **Student 네트워크**: 초기화된 ResNet (학습)
3. **Feature Matching**: 다중 레이어에서 특징 매칭
4. **이상 점수**: Teacher와 Student의 특징 차이

#### 모델 구성

```python
ModelRegistry.register("stfpm", "models.model_stfpm.STFPMTrainer",
    dict(
        backbone="resnet50",
        layers=["layer1", "layer2", "layer3"]
    ),
    dict(
        num_epochs=50,
        batch_size=16,
        normalize=True,
        img_size=256
    )
)
```

#### 특징

- **학습 시간**: 중간 (50 epochs)
- **추론 속도**: 빠름
- **메모리**: 중간
- **정확도**: 높음

#### 사용 예시

```python
train("mvtec", "bottle", "stfpm", num_epochs=50)
```

#### 장단점

**장점:**
- 안정적인 학습
- 높은 정확도
- 빠른 추론

**단점:**
- Teacher-Student 구조 필요
- 메모리 사용량 높음

---

### 5.2. FRE (2023)

#### 개요
Feature Reconstruction Error를 이용한 빠른 이상 감지입니다.

**논문**: [A Fast Method For Anomaly Detection And Segmentation](https://papers.bmvc2023.org/0614.pdf)

#### 모델 구성

```python
ModelRegistry.register("fre", "models.model_fre.FRETrainer",
    dict(
        backbone="resnet50",
        layer="layer3"
    ),
    dict(
        num_epochs=50,
        batch_size=16,
        normalize=True,
        img_size=256
    )
)
```

---

### 5.3. Reverse Distillation (2022)

#### 개요
역방향 지식 증류를 통한 이상 감지입니다.

**논문**: [Anomaly Detection via Reverse Distillation from One-Class Embedding](https://arxiv.org/abs/2201.10703)

#### 모델 구성

```python
ModelRegistry.register("reverse-distillation", "models.model_reverse_distillation.ReverseDistillationTrainer",
    dict(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"]
    ),
    dict(
        num_epochs=50,
        batch_size=8,
        normalize=True,
        img_size=256
    )
)
```

---

### 5.4. EfficientAD (2024)

#### 개요
밀리초 수준의 지연시간을 가진 정확한 이상 감지입니다.

**논문**: [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535)

#### 동작 원리

1. **Teacher 네트워크**: PDN (Patch Description Network)
2. **Student 네트워크**: 경량 네트워크
3. **AutoEncoder**: 추가 재구성 경로
4. **이상 점수**: Teacher-Student + Reconstruction 결합

#### 모델 구성

```python
# EfficientAD Small
ModelRegistry.register("efficientad-small", "models.model_efficientad.EfficientAdTrainer",
    dict(model_size="small"),
    dict(
        num_epochs=20,
        batch_size=1,
        normalize=False,
        img_size=256
    )
)

# EfficientAD Medium
ModelRegistry.register("efficientad-medium", "models.model_efficientad.EfficientAdTrainer",
    dict(model_size="medium"),
    dict(
        num_epochs=20,
        batch_size=1,
        normalize=False,
        img_size=256
    )
)
```

#### 특징

- **학습 시간**: 짧음 (20 epochs)
- **추론 속도**: 매우 빠름 (밀리초 수준)
- **메모리**: 낮음
- **정확도**: 매우 높음

#### 주의사항

- `normalize=False` 필수
- `batch_size=1` 권장
- Imagenette2 데이터셋 필요 (Teacher 사전학습)

---

## 6. Reconstruction-Based Models

### 6.1. Autoencoder (Baseline)

#### 개요
Vanilla Autoencoder를 이용한 기본 재구성 기반 이상 감지입니다.

#### 동작 원리

1. **인코더**: 입력 이미지를 저차원 잠재 공간으로 압축
2. **디코더**: 잠재 벡터를 원본 이미지로 재구성
3. **이상 점수**: 재구성 오차 (MSE + SSIM)

#### 모델 구성

```python
ModelRegistry.register("autoencoder", "models.model_autoencoder.AutoencoderTrainer",
    dict(latent_dim=128),
    dict(
        num_epochs=50,
        batch_size=16,
        normalize=False,
        img_size=256
    )
)
```

#### 특징

- **학습 시간**: 빠름
- **추론 속도**: 매우 빠름
- **메모리**: 낮음
- **정확도**: 중간

#### 사용 예시

```python
train("mvtec", "bottle", "autoencoder", num_epochs=50, normalize=False)
```

#### 장단점

**장점:**
- 구현 간단
- 학습 빠름
- 해석 용이

**단점:**
- 정확도 제한적
- 복잡한 이상 감지 어려움
- Identity mapping 위험

---

### 6.2. GANomaly (2018)

#### 개요
GAN 기반 준지도 이상 감지입니다.

**논문**: [GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training](https://arxiv.org/abs/1805.06725)

#### 모델 구성

```python
ModelRegistry.register("ganomaly", "models.model_ganomaly.GanomalyTrainer",
    dict(
        input_size=(256, 256),
        n_features=64,
        latent_vec_size=256,
        gamma=0.01
    ),
    dict(
        num_epochs=20,
        batch_size=8,
        normalize=False,
        img_size=256
    )
)
```

---

### 6.3. DRAEM (2021)

#### 개요
Discriminatively trained Reconstruction Embedding입니다.

**논문**: [DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection](https://arxiv.org/abs/2108.07610)

#### 동작 원리

1. **이상 시뮬레이션**: Perlin noise로 인공 이상 생성
2. **재구성 네트워크**: 이상을 제거하여 정상 이미지 재구성
3. **판별 네트워크**: 이상 영역 분할
4. **결합 손실**: 재구성 + 분할 손실

#### 모델 구성

```python
ModelRegistry.register("draem", "models.model_draem.DraemTrainer",
    dict(sspcab=True),
    dict(
        num_epochs=10,
        batch_size=8,
        normalize=False,
        img_size=256
    )
)
```

#### 특징

- **학습 시간**: 짧음 (10 epochs)
- **추론 속도**: 빠름
- **메모리**: 중간
- **정확도**: 높음

#### 주의사항

- DTD 데이터셋 필요 (텍스처 소스)
- `normalize=False` 필수

---

### 6.4. DSR (2022)

#### 개요
Dual Subspace Re-Projection Network입니다.

**논문**: [A Dual Subspace Re-Projection Network for Surface Anomaly Detection](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31)

#### 모델 구성

```python
ModelRegistry.register("dsr", "models.model_dsr.DsrTrainer",
    dict(
        latent_anomaly_strength=0.2,
        embedding_dim=128,
        num_embeddings=4096
    ),
    dict(
        num_epochs=50,
        batch_size=8,
        normalize=False,
        img_size=256
    )
)
```

---

## 7. Feature Adaptation Models

### 7.1. DFM (2019)

#### 개요
Deep Feature Modeling for Anomaly Detection입니다.

**논문**: [Deep Feature Modeling](https://arxiv.org/abs/1909.11786)

#### 모델 구성

```python
ModelRegistry.register("dfm", "models.model_dfm.DFMTrainer",
    dict(
        backbone="resnet50",
        layer="layer3",
        score_type="fre"
    ),
    dict(
        num_epochs=1,
        batch_size=16,
        normalize=True,
        img_size=256
    )
)
```

---

### 7.2. CFA (2022)

#### 개요
Coupled-hypersphere Feature Adaptation입니다.

**논문**: [Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization](https://arxiv.org/abs/2206.04325)

#### 모델 구성

```python
ModelRegistry.register("cfa", "models.model_cfa.CfaTrainer",
    dict(backbone="wide_resnet50_2"),
    dict(
        num_epochs=20,
        batch_size=16,
        normalize=True,
        img_size=256
    )
)
```

---

## 8. Foundation Models

### 8.1. Dinomaly (2025)

#### 개요
DINOv2 기반의 최신 다중 클래스 이상 감지입니다.

**논문**: [Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection](https://arxiv.org/abs/2405.14325)

#### 동작 원리

1. **DINOv2 인코더**: 강력한 사전학습 모델 (frozen)
2. **경량 디코더**: ViT 기반 디코더 (학습)
3. **특징 융합**: 레이어 그룹 기반 융합
4. **코사인 유사도**: 인코더-디코더 특징 비교

#### 모델 구성

```python
# Dinomaly Small (224x224)
ModelRegistry.register("dinomaly-small-224", "models.model_dinomaly.DinomalyTrainer",
    dict(
        encoder_name="dinov2_vit_small_14",
        bottleneck_dropout=0.2,
        decoder_depth=8
    ),
    dict(
        num_epochs=15,
        batch_size=32,
        normalize=True,
        img_size=224
    )
)

# Dinomaly Base (224x224)
ModelRegistry.register("dinomaly-base-224", "models.model_dinomaly.DinomalyTrainer",
    dict(
        encoder_name="dinov2_vit_base_14",
        bottleneck_dropout=0.2,
        decoder_depth=8
    ),
    dict(
        num_epochs=15,
        batch_size=16,
        normalize=True,
        img_size=224
    )
)

# Dinomaly Large (224x224)
ModelRegistry.register("dinomaly-large-224", "models.model_dinomaly.DinomalyTrainer",
    dict(
        encoder_name="dinov2_vit_large_14",
        bottleneck_dropout=0.2,
        decoder_depth=8
    ),
    dict(
        num_epochs=15,
        batch_size=8,
        normalize=True,
        img_size=224
    )
)
```

#### 다양한 이미지 크기

Dinomaly는 224, 392, 448 크기를 지원합니다:

```python
# 392x392
"dinomaly-small-392", "dinomaly-base-392", "dinomaly-large-392"

# 448x448
"dinomaly-small-448", "dinomaly-base-448", "dinomaly-large-448"
```

#### 특징

- **학습 시간**: 짧음 (15 epochs at 224)
- **추론 속도**: 중간
- **메모리**: 높음 (Large 모델)
- **정확도**: 최고 수준

#### 모델 크기 선택

- **Small**: 빠른 학습, 적은 메모리
- **Base**: 균형잡힌 성능 (권장)
- **Large**: 최고 정확도, 많은 리소스 필요

---

### 8.2. SuperSimpleNet (2024)

#### 개요
Fast and Reliable Surface Defect Detection입니다.

**논문**: [SuperSimpleNet: Unifying Unsupervised and Supervised Learning](https://arxiv.org/abs/2408.03143)

#### 모델 구성

```python
# Unsupervised
ModelRegistry.register("supersimplenet", "models.model_supersimplenet.SupersimplenetTrainer",
    dict(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        supervised=False
    ),
    dict(
        num_epochs=50,
        batch_size=16,
        normalize=True,
        img_size=256
    )
)

# Supervised
ModelRegistry.register("supersimplenet-supervised", "models.model_supersimplenet.SupersimplenetTrainer",
    dict(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        supervised=True
    ),
    dict(
        num_epochs=50,
        batch_size=16,
        normalize=True,
        img_size=256
    )
)
```

---

### 8.3. UniNet (2025)

#### 개요
Unified Contrastive Learning Framework입니다.

**논문**: [UniNet](https://pangdatangtt.github.io/#poster)

#### 모델 구성

```python
ModelRegistry.register("uninet", "models.model_uninet.UniNetTrainer",
    dict(
        student_backbone="wide_resnet50_2",
        teacher_backbone="wide_resnet50_2",
        temperature=0.4
    ),
    dict(
        num_epochs=20,
        batch_size=4,
        normalize=True,
        img_size=256
    )
)
```

---

## 9. 모델 선택 가이드

### 9.1. 용도별 추천 모델

#### 빠른 프로토타이핑

```python
# 1순위: PaDiM (학습 불필요)
train("mvtec", "bottle", "padim", num_epochs=1)

# 2순위: PatchCore (학습 불필요)
train("mvtec", "bottle", "patchcore", num_epochs=1)
```

#### 최고 정확도

```python
# 1순위: Dinomaly Base
train("mvtec", "bottle", "dinomaly-base-224", num_epochs=15)

# 2순위: PatchCore
train("mvtec", "bottle", "patchcore", num_epochs=1)

# 3순위: EfficientAD
train("mvtec", "bottle", "efficientad-medium", num_epochs=20)
```

#### 실시간 응용

```python
# 1순위: EfficientAD
train("mvtec", "bottle", "efficientad-small", num_epochs=20)

# 2순위: PatchCore
train("mvtec", "bottle", "patchcore", num_epochs=1)
```

#### 교육/연구

```python
# 1순위: Autoencoder (기본 개념)
train("mvtec", "bottle", "autoencoder", num_epochs=50, normalize=False)

# 2순위: STFPM (지식 증류)
train("mvtec", "bottle", "stfpm", num_epochs=50)

# 3순위: FastFlow (정규화 플로우)
train("mvtec", "bottle", "fastflow-resnet50", num_epochs=500)
```

### 9.2. 데이터셋 크기별 추천

#### 소규모 (< 100 images)

```python
# Memory-based models
train("mvtec", "bottle", "padim", num_epochs=1)
train("mvtec", "bottle", "patchcore", num_epochs=1)
```

#### 중규모 (100-500 images)

```python
# Knowledge distillation
train("mvtec", "bottle", "stfpm", num_epochs=50)
train("mvtec", "bottle", "efficientad-small", num_epochs=20)
```

#### 대규모 (> 500 images)

```python
# Foundation models
train("mvtec", "bottle", "dinomaly-base-224", num_epochs=15)
train("mvtec", "bottle", "uninet", num_epochs=20)
```

### 9.3. 리소스별 추천

#### 제한적인 GPU (< 10GB)

```python
# Small models with reduced batch size
train("mvtec", "bottle", "padim", num_epochs=1, batch_size=4)
train("mvtec", "bottle", "autoencoder", num_epochs=50, batch_size=8)
train("mvtec", "bottle", "dinomaly-small-224", num_epochs=15, batch_size=16)
```

#### 충분한 GPU (> 20GB)

```python
# Large models
train("mvtec", "bottle", "dinomaly-large-448", num_epochs=10, batch_size=4)
train("mvtec", "bottle", "fastflow-cait", num_epochs=500, batch_size=2)
```

---

## 10. 모델별 설정

### 10.1. 공통 설정

모든 모델은 다음 파라미터를 지원합니다:

```python
train(
    dataset_type="mvtec",
    category="bottle",
    model_type="stfpm",
    num_epochs=50,        # Override default epochs
    batch_size=16,        # Override default batch size
    img_size=256,         # Override default image size
    normalize=True        # Override normalization
)
```

### 10.2. 모델별 기본값

| 모델 | epochs | batch_size | img_size | normalize |
|------|--------|------------|----------|-----------|
| padim | 1 | 4 | 256 | True |
| patchcore | 1 | 8 | 256 | True |
| stfpm | 50 | 16 | 256 | True |
| efficientad-small | 20 | 1 | 256 | False |
| dinomaly-base-224 | 15 | 16 | 224 | True |
| autoencoder | 50 | 16 | 256 | False |
| fastflow-resnet50 | 500 | 8 | 256 | True |
| draem | 10 | 8 | 256 | False |

### 10.3. 정규화 설정

**normalize=True를 사용하는 모델:**
- Memory-based: PaDiM, PatchCore, DFKDE
- Knowledge Distillation: STFPM, FRE, Reverse Distillation
- Normalizing Flow: CFlow, FastFlow, CSFlow, U-Flow
- Feature Adaptation: DFM, CFA
- Foundation: Dinomaly, SuperSimpleNet, UniNet

**normalize=False를 사용하는 모델:**
- Reconstruction: Autoencoder, GANomaly, DRAEM, DSR
- Knowledge Distillation: EfficientAD

### 10.4. 메모리 최적화

GPU 메모리 부족 시:

```python
# Reduce batch size
train("mvtec", "bottle", "stfpm", batch_size=8)

# Reduce image size
train("mvtec", "bottle", "stfpm", img_size=224)

# Use smaller model
train("mvtec", "bottle", "dinomaly-small-224")  # Instead of large
```

---

**다음 문서:** [Datasets](04-datasets.md) - 데이터셋 준비 가이드