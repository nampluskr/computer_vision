# Knowledge Distillation for Anomaly Detection

# 1. Paradigm Overview

## 1.1 Core Principle

Knowledge Distillation은 이상 탐지를 teacher-student framework의 모방 학습 문제로 재정의한다. 핵심 원리는 간단하지만 강력하다: 학생 네트워크가 정상 데이터에서 선생 네트워크의 출력을 정확히 모방하도록 학습한다. 테스트 시 이상 샘플에서는 학생이 선생을 제대로 모방하지 못하며, 이 모방 실패가 이상도를 나타낸다.

전통적인 지식 증류는 모델 압축 기법으로 시작되었다. 큰 teacher model의 지식을 작은 student model로 전달하여 효율성을 높인다. 이상 탐지에서는 이 개념을 창의적으로 전환했다. Teacher와 student의 역할이 다르다. Teacher는 강력한 사전 학습 모델로 일반적인 시각 지식을 제공한다. Student는 정상 데이터에 특화되어 정상 패턴만 학습한다.

학습 과정은 정상 샘플로만 진행된다. Student는 teacher의 feature representations이나 predictions을 모방한다. Teacher는 고정되어 있고(frozen), student만 학습된다. 정상 데이터에서 student는 teacher를 잘 모방하도록 최적화된다. 수백 epochs 후 student는 정상 패턴에 대해 teacher와 거의 동일한 출력을 생성한다.

테스트 시 이상이 나타나면 상황이 달라진다. Teacher는 사전 학습된 일반 지식으로 이상 샘플도 reasonable한 representation을 생성한다. 그러나 student는 정상 패턴만 학습했으므로 이상에 대해 적절한 response를 생성하지 못한다. Teacher와 student의 출력 차이가 커진다. 이 discrepancy가 anomaly score로 사용된다.

이 접근의 elegance는 단순성에 있다. 명시적인 정상 분포 모델링이나 reconstruction이 필요 없다. 단지 "정상에서는 잘 모방, 이상에서는 못 모방"이라는 직관을 따른다. 구현도 straightforward하다. 두 네트워크를 준비하고, 하나는 고정하고, 다른 하나를 학습시키며, 둘의 차이를 측정한다. 이 단순성이 강력한 성능과 결합되어 매력적인 패러다임을 만든다.

Knowledge distillation은 2019-2020년경 이상 탐지에 도입되기 시작했다. STFPM(2021)이 본격적으로 주목받게 했다. Student-teacher feature pyramid matching으로 98.1% AUROC를 달성했다. 이후 Reverse Distillation(2022), EfficientAD(2023) 등이 발전을 이어갔다. EfficientAD는 1-5ms 추론 속도로 실시간 배포를 가능하게 하면서도 97.8% AUROC를 유지했다. 이는 속도-정확도 frontier의 새로운 지점을 개척했다.

Knowledge distillation의 가장 큰 장점은 극도의 효율성이다. Memory-based 방법은 학습 데이터를 저장해야 하고(수백 MB), flow-based 방법은 복잡한 확률 모델을 유지한다(수백 MB). Distillation-based 방법은 작은 student network만 배포하면 된다(수십 MB). 추론도 단일 forward pass로 끝나 매우 빠르다. CPU에서도 real-time에 근접한다.

그러나 한계도 명확하다. 최고 정확도는 memory-based 방법(PatchCore 99.1%)에 미치지 못한다. 일반적으로 96-98% 수준이다. Student의 capacity 제약과 teacher의 일반성 부족이 원인이다. 또한 teacher 선택이 critical하다. 부적절한 teacher는 성능을 크게 저하시킨다. Hyperparameter tuning도 다른 방법보다 더 sensitive하다.

Knowledge distillation은 특정 niche에서 탁월하다. Real-time processing이 필수적이고 합리적인 정확도(97-98%)로 충분한 응용이다. 고속 생산 라인, 엣지 디바이스, 모바일 검사 시스템 등이다. 최고 정확도가 필요하면 다른 방법이 낫지만, 속도와 효율성이 우선이면 distillation이 최선이다. 이는 실무에서 매우 흔한 요구사항이므로 practical value가 크다.

Knowledge distillation 패러다임은 계속 진화하고 있다. 최근의 foundation models(CLIP, DINOv2)를 teacher로 사용하는 시도가 있다. Self-distillation과 ensemble distillation도 탐구된다. Adaptive student architectures와 dynamic knowledge transfer mechanisms가 제안된다. 향후 수년간 이 분야는 더욱 발전할 것이며, 실시간 이상 탐지의 표준이 될 가능성이 크다.

## 1.2 Teacher-Student Framework

Knowledge distillation의 핵심은 teacher-student framework다. 두 네트워크가 서로 다른 역할을 수행하며 협력한다. 이들의 interaction이 anomaly detection capability를 만들어낸다.

**Teacher Network**

Teacher는 강력한 사전 학습 네트워크다. ImageNet이나 대규모 데이터셋에서 학습되어 일반적인 시각 지식을 보유한다. 일반적으로 ResNet, Wide ResNet, EfficientNet 같은 CNN이 사용된다. 최근에는 Vision Transformer(ViT)나 CLIP 같은 foundation models도 teacher로 활용된다.

Teacher의 역할은 robust한 feature representations을 제공하는 것이다. 정상과 이상을 가리지 않고 입력 이미지의 의미 있는 특징을 추출한다. 사전 학습 덕분에 다양한 visual concepts를 이해한다. Objects, textures, shapes, colors를 효과적으로 인코딩한다.

중요한 점은 teacher가 anomaly detection task에 대해 전혀 학습하지 않는다는 것이다. ImageNet classification이나 self-supervised learning으로만 학습되었다. Anomaly나 defects를 본 적이 없다. 그럼에도 불구하고 강력한 general representations 덕분에 이상 샘플에 대해서도 meaningful한 features를 생성한다.

Teacher는 학습 및 추론 내내 고정된다(frozen). 파라미터가 업데이트되지 않는다. Gradient가 teacher로 전파되지 않는다. 이는 계산 효율성을 높이고 teacher의 일반성을 보존한다. Teacher를 fine-tuning하면 정상 데이터에 overfitting되어 일반 지식을 잃을 수 있다.

**Student Network**

Student는 작고 task-specific한 네트워크다. Teacher보다 훨씬 작은 capacity를 가진다. 일반적으로 teacher의 1/4 - 1/10 크기다. 경량 CNN이나 심지어 single convolutional layer일 수 있다. 작을수록 추론이 빠르고 배포가 쉽다.

Student의 역할은 정상 데이터에서 teacher를 모방하는 것이다. Teacher의 feature maps이나 predictions을 정확히 재현하도록 학습한다. 처음에는 random initialized 상태로 teacher와 전혀 다른 출력을 생성한다. 학습이 진행되며 점점 teacher에 가까워진다.

Student는 정상 샘플로만 학습된다. Anomalous samples는 학습에 사용되지 않는다. 이는 one-class learning의 핵심이다. Student는 "정상이 어떻게 생겼는지"만 배운다. "이상이 어떻게 생겼는지"는 모른다. 이 제약이 anomaly detection capability를 만든다.

Student의 capacity 제약이 중요하다. 만약 student가 teacher만큼 크다면, 모든 입력에 대해 teacher를 완벽히 모방할 수 있다. 이상 샘플에서도 차이가 나지 않는다. 작은 capacity 덕분에 student는 학습 분포(정상)에만 잘 fit하고 out-of-distribution(이상)에는 generalize하지 못한다. 이것이 의도된 설계다.

**Framework Architecture**

전형적인 teacher-student framework는 다음과 같은 구조를 가진다:

```
Input Image
    |
    ├─→ Teacher Network (frozen)
    |       └─→ Teacher Features
    |
    └─→ Student Network (trainable)
            └─→ Student Features
                    |
                    ↓
            Feature Matching Loss
            (L2, Cosine, etc.)
```

학습 시 양쪽 네트워크에 동일한 입력을 제공한다. Teacher와 student가 각각 features를 생성한다. 두 features 간의 차이를 loss로 정의한다. Student 파라미터만 이 loss를 최소화하도록 업데이트된다. 수백 epochs 후 student는 정상 데이터에서 teacher를 근사한다.

추론 시에도 동일한 구조를 사용한다. 테스트 이미지를 양쪽에 입력한다. Teacher와 student features를 비교한다. 차이가 크면 anomaly, 작으면 normal로 판정한다. Anomaly map도 spatial locations에서의 feature differences로 생성된다.

**Multi-scale Feature Matching**

대부분의 distillation 방법은 단일 layer가 아니라 여러 layers에서 matching을 수행한다. Teacher의 여러 층에서 features를 추출한다. Student도 대응하는 여러 outputs을 생성한다. 각 pair에서 matching loss를 계산하고 합산한다.

예를 들어 STFPM은 ResNet의 layer1, layer2, layer3에서 features를 추출한다. 각 layer에서 teacher-student discrepancy를 측정한다. 세 개의 losses를 weighted sum으로 결합한다. Multi-scale matching이 다양한 크기의 결함을 효과적으로 탐지한다.

Low-level features(layer1)는 fine-grained textures와 작은 결함에 민감하다. High-level features(layer3)는 큰 구조적 이상을 포착한다. Mid-level features(layer2)는 중간 크기의 결함을 다룬다. 세 가지를 결합하면 comprehensive한 anomaly detection이 가능하다.

**Asymmetric Architecture**

Teacher와 student의 architecture가 동일할 필요는 없다. 실제로 다른 경우가 많다. Teacher는 large backbone(ResNet50, Wide ResNet-50)이고, student는 small network(ResNet18, custom lightweight CNN)다. 이 asymmetry가 효율성의 핵심이다.

일부 방법은 더 극단적인 asymmetry를 사용한다. EfficientAD는 teacher로 EfficientNet-B4를 사용하고, student로 매우 작은 patch descriptor network를 사용한다. Student가 teacher의 1/20 크기다. 이렇게 작아도 정상 데이터의 주요 패턴을 학습할 수 있다. 이상에서만 실패한다.

Architecture asymmetry는 deployment에도 유리하다. 학습 후 teacher는 버려진다. Student만 배포하면 된다. Teacher는 학습 시에만 필요하다. 따라서 teacher가 아무리 크고 느려도 추론 속도에 영향이 없다. 이는 memory-based 방법과 대조적이다. PatchCore는 학습 샘플을 배포 시에도 유지해야 한다.

**Role Reversal Approaches**

일부 최신 연구는 역할을 반전시킨다. Reverse Distillation(2022)은 student를 고정하고 teacher를 학습시킨다. Student가 일반적인 사전 학습 모델이고, teacher는 정상 데이터로 학습된다. Teacher가 정상에 특화되어 이상에서 student와 차이를 보인다.

이 반전은 다른 관점을 제공한다. 일반 모델(student)과 specialized 모델(teacher)의 차이가 anomaly signal이다. 구현은 복잡하지만 일부 경우 성능 향상을 보였다. 그러나 전통적인 teacher-student가 더 직관적이고 널리 사용된다.

Ensemble approaches도 존재한다. 여러 students를 학습시키고 평균을 취한다. 각 student는 다른 random initialization이나 다른 architecture를 가진다. Ensemble이 single student보다 robust하다. 그러나 배포 비용이 증가한다. Trade-off를 신중히 평가해야 한다.

## 1.3 Knowledge Transfer Mechanism

Knowledge distillation의 핵심은 teacher에서 student로의 효과적인 지식 전달이다. "지식"이 무엇이고 어떻게 전달되는지가 성능을 결정한다.

**Types of Knowledge**

Distillation에서 전달되는 지식은 여러 형태를 가진다. 첫째, feature representations이다. Teacher의 중간 layer activations가 지식의 주된 형태다. 이들은 입력의 hierarchical representations를 담는다. Student는 이러한 representations를 재현하도록 학습한다.

둘째, predictions이나 logits이다. 원래 distillation(Hinton et al. 2015)에서 제안된 형태다. Teacher의 softmax outputs이나 pre-softmax logits를 student가 모방한다. 분류 문제에서 유용하다. Anomaly detection에서는 덜 사용되지만 일부 방법이 활용한다.

셋째, attention maps이다. Teacher가 어디를 "보는지"를 전달한다. Attention weights나 activation magnitudes를 matching한다. 이는 spatial importance를 학습하는 데 도움이 된다. Student가 teacher와 유사한 영역에 focus한다.

넷째, relational knowledge다. 개별 features뿐만 아니라 features 간 관계도 전달한다. Feature similarities, correlations, graph structures 등이다. 이는 더 high-order information을 포착한다. 최근 연구들이 탐구하는 영역이다.

**Loss Functions**

Knowledge transfer는 loss function으로 구현된다. Teacher와 student outputs 간의 차이를 측정하고 최소화한다. 여러 loss functions가 사용된다.

L2 (MSE) loss가 가장 일반적이다. Feature maps의 pixel-wise squared differences를 계산한다.

$$\mathcal{L}_{\text{L2}} = \frac{1}{N} \sum_{i,j,k} (\mathbf{F}^T_{ijk} - \mathbf{F}^S_{ijk})^2$$

여기서 $\mathbf{F}^T$와 $\mathbf{F}^S$는 teacher와 student feature maps다. $(i,j)$는 spatial locations, $k$는 channels다. $N$은 normalization constant다. L2 loss는 단순하고 효과적이어서 널리 사용된다.

Cosine similarity loss도 흔하다. Features의 방향(direction)을 matching한다. Magnitude는 무시한다.

$$\mathcal{L}_{\text{cos}} = 1 - \frac{\mathbf{F}^T \cdot \mathbf{F}^S}{|\mathbf{F}^T| |\mathbf{F}^S|}$$

Cosine loss는 scale invariant하여 robust하다. Teacher와 student의 feature magnitudes가 다를 때 유리하다. Normalized features를 다룬다.

KL divergence는 probability distributions을 matching할 때 사용된다. Teacher와 student가 probability outputs을 생성하면 KL loss를 적용한다. 정보 이론적으로 well-founded하다.

Perceptual loss도 고려된다. Pre-trained VGG나 LPIPS를 사용하여 perceptual similarity를 측정한다. Human perception과 더 align된 metric이다. 그러나 계산 비용이 추가된다.

**Feature Adaptation**

Teacher와 student의 feature dimensions이 다르면 adaptation layer가 필요하다. Student features를 teacher와 같은 dimension으로 변환한다. 일반적으로 1×1 convolution이나 linear projection을 사용한다.

$$\mathbf{F}^S_{\text{adapted}} = \text{Conv}_{1\times1}(\mathbf{F}^S)$$

Adaptation layer는 student와 함께 학습된다. 단순한 linear mapping이지만 효과적이다. Dimensionality mismatch를 해결하고 student가 teacher space에 project되도록 한다.

일부 방법은 더 복잡한 adaptation을 사용한다. Multi-layer MLPs, attention mechanisms, 또는 graph convolutions다. 이들은 더 expressive한 transformation을 제공한다. 그러나 파라미터와 계산이 증가한다. Simple 1×1 conv가 대부분의 경우 충분하다.

**Multi-level Aggregation**

여러 layers에서 knowledge를 전달할 때 이들을 어떻게 aggregation할지 결정해야 한다. 가장 단순한 방법은 weighted sum이다.

$$\mathcal{L}_{\text{total}} = \sum_{l=1}^{L} w_l \mathcal{L}_l$$

여기서 $\mathcal{L}_l$은 layer $l$에서의 distillation loss다. $w_l$은 가중치로, 수동으로 설정하거나 학습할 수 있다. 일반적으로 모든 layers에 동일 가중치(uniform weighting)를 사용한다. 이는 simple하고 robust하다.

Adaptive weighting도 가능하다. 각 layer의 중요도를 데이터로부터 학습한다. Attention mechanism으로 구현한다. 일부 layers가 특정 카테고리에서 더 informative할 수 있다. Adaptive weighting이 이를 자동으로 발견한다. 그러나 hyperparameter tuning이 복잡해진다.

**Normalization Techniques**

Feature matching 전에 normalization이 중요하다. Teacher와 student features의 scale이 다를 수 있다. Normalization이 안정적인 학습을 돕는다.

Batch normalization을 각 feature map에 적용할 수 있다. 각 channel의 mean을 0, variance를 1로 만든다. 이는 scale differences를 제거한다. 그러나 batch statistics에 의존하여 small batches에서 불안정할 수 있다.

Layer normalization은 spatial dimensions에 대해 normalize한다. Batch size에 independent하여 더 안정적이다. 각 sample의 feature map을 independently normalize한다.

L2 normalization은 features를 unit sphere에 project한다. Cosine similarity와 함께 사용하면 효과적이다. Direction만 중요하고 magnitude는 무시한다.

어떤 normalization을 사용할지는 empirical하게 결정한다. 대부분의 경우 L2 normalization + cosine loss 또는 no normalization + L2 loss가 잘 작동한다.

**Transfer Learning Perspective**

Knowledge distillation을 transfer learning의 관점에서 볼 수 있다. Teacher는 source domain(ImageNet)에서 학습된 지식을 가진다. Student는 target domain(정상 제품 이미지)으로 이 지식을 transfer한다. Distillation loss가 transfer mechanism이다.

이 관점은 domain adaptation과의 연결을 시사한다. Teacher와 student 간의 domain gap을 bridging하는 것이 목표다. Student는 정상 데이터의 domain-specific features를 학습하면서 teacher의 general knowledge를 유지한다. Balance가 중요하다.

Fine-tuning과 비교하면, distillation은 더 controlled transfer를 제공한다. Fine-tuning은 전체 network를 업데이트하여 catastrophic forgetting 위험이 있다. Distillation은 student를 별도로 학습하여 teacher의 knowledge를 보존한다. Student가 정상에 특화되면서도 teacher의 general capability를 inherits한다.

## 1.4 Anomaly as Imitation Failure

Knowledge distillation 패러다임의 핵심 통찰은 이상을 "모방 실패"로 해석하는 것이다. 이는 기존 방법들과 근본적으로 다른 관점을 제공한다.

**Conceptual Framework**

전통적인 anomaly detection은 정상 분포를 명시적으로 모델링한다. Gaussian distributions(PaDiM), normalizing flows(FastFlow), reconstruction errors(autoencoders) 등이다. Test sample이 이 분포에서 얼마나 벗어났는지를 측정한다. Out-of-distribution detection이 목표다.

Knowledge distillation은 다른 접근을 취한다. 정상 분포를 명시적으로 모델링하지 않는다. 대신 "정상에서는 모방 가능, 이상에서는 모방 불가능"이라는 속성을 활용한다. Student의 limited capacity와 training distribution constraint가 이를 가능하게 한다.

왜 이것이 작동하는가? Student는 정상 샘플에서만 학습된다. Teacher를 모방하도록 최적화된다. Student의 capacity가 제한적이므로 training distribution에만 잘 fit한다. Out-of-distribution samples(anomalies)에는 generalize하지 못한다. 이것이 intentional limitation이다.

Teacher는 다르다. 대규모 사전 학습으로 일반적인 visual knowledge를 가진다. 다양한 objects, textures, scenes을 본 적이 있다. Anomaly도 teacher에게는 some visual patterns이다. Unusual하지만 completely incomprehensible하지는 않다. Teacher는 anomalies에 대해서도 reasonable한 features를 생성한다.

Student와 teacher의 이 fundamental difference가 anomaly detection capability를 만든다. 정상에서는 둘 다 similar features를 생성한다. Student가 정상 분포를 잘 학습했기 때문이다. 이상에서는 달라진다. Teacher는 여전히 reasonable features를 생성하지만, student는 struggle한다. Training에서 본 적 없는 patterns이기 때문이다. Feature discrepancy가 커진다.

**Mathematical Formulation**

이를 수학적으로 표현하면 다음과 같다. Teacher function을 $T(\mathbf{x})$, student function을 $S(\mathbf{x})$라 하자. Student는 정상 데이터 $\mathcal{D}_{\text{normal}}$에서 다음을 최소화하도록 학습된다:

$$\min_{\theta_S} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{normal}}} \| T(\mathbf{x}) - S(\mathbf{x}; \theta_S) \|^2$$

학습 후 정상 샘플 $\mathbf{x}_{\text{normal}}$에 대해:

$$\| T(\mathbf{x}_{\text{normal}}) - S(\mathbf{x}_{\text{normal}}) \| \approx 0$$

이는 optimization objective였으므로 당연하다. Student가 정상에서 teacher를 잘 근사한다.

그러나 이상 샘플 $\mathbf{x}_{\text{anomaly}} \notin \mathcal{D}_{\text{normal}}$에 대해:

$$\| T(\mathbf{x}_{\text{anomaly}}) - S(\mathbf{x}_{\text{anomaly}}) \| \gg 0$$

이는 student가 out-of-distribution에서 generalize하지 못하기 때문이다. Teacher-student discrepancy가 커진다. 이를 anomaly score로 사용한다:

$$s(\mathbf{x}) = \| T(\mathbf{x}) - S(\mathbf{x}) \|$$

Threshold $\tau$를 설정하여 $s(\mathbf{x}) > \tau$이면 anomaly로 판정한다.

**Why Limited Capacity Matters**

Student의 limited capacity가 critical하다. 만약 student가 teacher만큼 크면 어떻게 될까? Student는 training data에서 teacher를 perfect하게 모방할 수 있다. 그러나 overparameterized model은 memorization이 가능하다. Training distribution뿐만 아니라 arbitrary inputs에 대해서도 teacher를 모방할 수 있다. Anomalies에서도 차이가 나지 않는다.

제한된 capacity 덕분에 student는 training distribution의 main patterns만 학습한다. Sparse patterns이나 outliers는 학습하지 못한다. Capacity를 정상의 common patterns로 채운다. Anomalies는 rare하고 diverse하여 student의 limited capacity로는 다룰 수 없다. 이것이 detection을 가능하게 한다.

Empirically 이는 검증되었다. Student size를 늘리면 정상에서의 discrepancy는 더 줄어든다(더 정확한 모방). 그러나 anomaly detection 성능은 오히려 저하된다. 이상에서의 discrepancy도 줄어들기 때문이다. Optimal student size는 trade-off point에 있다. 정상을 충분히 모방하면서도 이상에서는 실패하는 크기다.

**Comparison with Other Paradigms**

이 관점을 다른 패러다임과 비교하면 차이가 명확하다. Reconstruction-based methods(autoencoders)는 "정상은 잘 복원, 이상은 못 복원"을 활용한다. 개념적으로 유사하지만 mechanism이 다르다. Autoencoder는 bottleneck을 통해 compression을 강제한다. Distillation은 teacher-student discrepancy를 활용한다.

Memory-based methods(PatchCore)는 "정상은 memory와 가까움, 이상은 멈"을 사용한다. Explicit한 정상 examples를 유지한다. Test sample과 nearest neighbors를 비교한다. Distillation은 memory를 유지하지 않는다. Student가 implicit하게 정상 patterns를 internalize한다.

Flow-based methods(FastFlow)는 "정상은 high likelihood, 이상은 low likelihood"를 모델링한다. Explicit probability distributions를 학습한다. Distillation은 확률 모델이 아니다. Feature matching이 전부다. 더 단순하고 빠르다.

각 패러다임은 고유한 assumptions과 trade-offs를 가진다. Distillation의 assumption은 "limited student can't generalize to anomalies"다. 이는 대부분의 경우 reasonable하다. Anomalies가 diverse하고 training에 포함되지 않았기 때문이다. 그러나 anomalies가 정상과 매우 유사하면(subtle defects) student도 generalize할 수 있다. 이것이 한계다.

**Failure Modes**

어떤 경우에 모방 실패가 충분히 크지 않을까? 첫째, teacher가 부적절한 경우다. Teacher가 domain-specific knowledge가 부족하면 anomalies에 대해서도 poor features를 생성한다. Teacher-student 둘 다 나쁘면 차이가 작다. 좋은 teacher 선택이 critical하다.

둘째, student가 너무 큰 경우다. Overparameterized student는 anomalies에도 generalize한다. Discrepancy가 충분히 크지 않다. Capacity를 적절히 제한해야 한다.

셋째, training data에 anomalies가 섞인 경우다. Student가 일부 anomalous patterns를 정상으로 학습한다. 해당 anomalies를 detection하지 못한다. Data quality가 중요하다.

넷째, anomalies가 정상과 매우 유사한 경우다. Subtle texture changes, slight discolorations 등이다. Teacher와 student 모두 비슷한 features를 생성할 수 있다. Discrepancy가 작아 detection이 어렵다. 이는 fundamental limitation이다.

**Strengths and Weaknesses**

모방 실패 관점의 강점은 단순성과 효율성이다. 복잡한 확률 모델이나 memory storage가 필요 없다. 단순한 feature matching으로 충분하다. 구현이 straightforward하고 빠르다. Small student만 배포하면 되어 매모리 효율적이다.

약점은 정확도 ceiling이다. Student의 limited capacity가 detection을 가능하게 하지만 동시에 제약이다. 매우 복잡한 정상 patterns를 완벽히 학습하지 못할 수 있다. False positives가 발생한다. Teacher 선택과 student size tuning이 critical하고 sensitive하다. 최적 설정을 찾기 어렵다.

전반적으로 distillation은 practical한 패러다임이다. 최고 정확도보다 속도와 효율성을 우선할 때 탁월하다. Real-world deployment에서 자주 만나는 요구사항이다. 모방 실패라는 elegant한 아이디어가 강력한 실용성으로 이어진다. 이것이 knowledge distillation이 이상 탐지에서 중요한 위치를 차지하는 이유다.

# 2. STFPM (2021)

## 2.1 Basic Information

STFPM(Student-Teacher Feature Pyramid Matching)은 2021년 Wang 등이 제안한 방법으로, knowledge distillation을 이상 탐지에 본격적으로 도입한 선구적 연구다. 이 논문은 CVPR 2021에서 발표되었으며, teacher-student framework의 가능성을 명확히 입증했다. STFPM은 "Student-Teacher"라는 명칭을 명시적으로 사용한 최초의 이상 탐지 방법이다.

STFPM의 핵심 아이디어는 feature pyramid matching이다. Teacher와 student 모두 CNN의 여러 layers에서 features를 추출한다. 각 layer pair에서 feature matching을 수행한다. Multi-scale information을 활용하여 다양한 크기의 결함을 탐지한다. 이는 object detection의 Feature Pyramid Network(FPN) 개념을 distillation에 적용한 것이다.

방법론적으로 STFPM은 명확한 teacher-student framework를 확립했다. Pre-trained ResNet18을 teacher로 사용한다. 동일한 architecture의 randomly initialized ResNet18을 student로 사용한다. Student만 정상 데이터로 학습된다. Teacher는 완전히 frozen된다. Layer1, layer2, layer3의 세 개 levels에서 feature matching을 수행한다.

성능 면에서 STFPM은 발표 당시 competitive한 결과를 보였다. MVTec AD에서 이미지 레벨 AUROC 95.5%를 달성했다. 픽셀 레벨에서는 97.1%였다. 이는 PaDiM(97.5%)보다는 약간 낮지만 reasonable했다. 특히 추론 속도가 50-100ms로 빨라 실용성을 입증했다.

STFPM의 가장 큰 기여는 패러다임 확립이다. Teacher-student distillation이 anomaly detection에서 viable함을 보였다. 간단한 feature matching만으로도 좋은 성능을 얻을 수 있음을 증명했다. 후속 연구들(Reverse Distillation, EfficientAD)의 기반이 되었다. Baseline으로서의 역할이 중요했다.

그러나 한계도 명확했다. 성능이 memory-based 방법에 미치지 못했다. PatchCore(99.1%)와는 3.6%포인트 차이가 있었다. Teacher와 student가 동일한 architecture를 사용하여 최적이 아니었다. Small student를 사용하면 더 효율적일 수 있었다. Loss function도 단순한 L2였고 더 sophisticated한 선택이 가능했다.

STFPM 이후 distillation 기반 방법들이 빠르게 발전했다. Architecture 개선, loss function 정교화, teacher 선택 최적화 등이 이루어졌다. EfficientAD(2023)는 97.8% AUROC를 유지하면서 1-5ms 추론 속도를 달성했다. 이는 STFPM 대비 10-20배 빠르다. 패러다임은 STFPM이 열었고 후속 연구들이 완성했다.

학술적으로 STFPM은 knowledge distillation community와 anomaly detection community를 연결했다. 두 분야의 아이디어를 결합하는 cross-fertilization을 촉진했다. 이는 연구 방법론의 다양화를 가져왔다. 단순히 더 큰 모델이나 더 많은 데이터가 아니라 clever한 framework 설계로 문제를 해결하는 접근이었다.

실무적으로 STFPM은 distillation의 실용성을 보여줬다. 50-100ms 추론과 합리적인 메모리(100-200MB)로 실제 배포가 가능했다. 이는 PatchCore의 정확도와 autoencoder의 속도 사이의 중간 지점을 제공했다. Many practical applications에서 이러한 균형이 ideal하다. STFPM이 이를 처음 달성했다.

## 2.2 Student-Teacher Architecture

### 2.2.1 Feature Pyramid Matching

STFPM의 핵심은 feature pyramid에서의 multi-level matching이다. 이는 단일 layer가 아니라 여러 layers의 features를 동시에 활용한다.

**Pyramid Structure**

STFPM은 ResNet의 계층적 구조를 활용한다. ResNet은 자연스러운 feature pyramid를 형성한다. 각 layer는 이전보다 낮은 해상도와 높은 semantic level을 가진다.

Layer1: $h/4 \times w/4 \times 64$ (high-resolution, low-level)
Layer2: $h/8 \times w/8 \times 128$ (mid-resolution, mid-level)  
Layer3: $h/16 \times w/16 \times 256$ (low-resolution, high-level)

여기서 $h \times w$는 입력 이미지 크기다. 일반적으로 $256 \times 256$을 사용한다. 세 개 layers는 서로 다른 정보를 포착한다.

Layer1은 fine-grained textures와 edges를 인코딩한다. Small scratches, dots, minor discolorations 같은 미세한 결함에 민감하다. 해상도가 높아 spatial details를 보존한다. 그러나 semantic understanding이 부족하여 noise와 결함을 구별하기 어려울 수 있다.

Layer3는 high-level semantic features를 담는다. Object parts, global structures, contextual information을 포착한다. Large cracks, missing parts, major deformations 같은 구조적 이상에 효과적이다. 그러나 해상도가 낮아 small defects를 놓칠 수 있다.

Layer2는 중간 지점이다. Reasonable한 해상도와 적절한 semantic level을 가진다. Medium-sized defects를 잘 탐지한다. 많은 경우 가장 informative한 layer다.

**Multi-level Matching Mechanism**

Teacher와 student 모두 세 개 layers에서 features를 추출한다. $\mathbf{F}^T_1, \mathbf{F}^T_2, \mathbf{F}^T_3$와 $\mathbf{F}^S_1, \mathbf{F}^S_2, \mathbf{F}^S_3$를 얻는다. 각 level에서 독립적으로 matching을 수행한다.

Level $l$에서의 discrepancy는 다음과 같이 계산된다:

$$d_l(i,j) = \| \mathbf{F}^T_l(i,j) - \mathbf{F}^S_l(i,j) \|^2$$

여기서 $(i,j)$는 spatial location이다. $\mathbf{F}_l(i,j)$는 해당 위치의 feature vector다. L2 norm으로 거리를 측정한다. 각 level은 $h_l \times w_l$ 크기의 discrepancy map을 생성한다.

세 개의 discrepancy maps를 동일 해상도로 align해야 한다. 일반적으로 가장 높은 해상도(layer1, $h/4 \times w/4$)로 upsampling한다. Bilinear interpolation을 사용한다.

$$d_2^{\text{up}} = \text{Upsample}(d_2, (h/4, w/4))$$
$$d_3^{\text{up}} = \text{Upsample}(d_3, (h/4, w/4))$$

Upsampled maps를 결합하여 최종 anomaly map을 생성한다. 단순 평균이나 weighted average를 사용한다.

$$\text{AnomalyMap} = \frac{1}{3}(d_1 + d_2^{\text{up}} + d_3^{\text{up}})$$

또는 학습 가능한 가중치로:

$$\text{AnomalyMap} = w_1 d_1 + w_2 d_2^{\text{up}} + w_3 d_3^{\text{up}}$$

가중치는 validation set에서 튜닝하거나 learnable parameters로 만들 수 있다.

**Rationale for Pyramid Matching**

왜 single level이 아니라 pyramid matching을 사용하는가? 결함의 크기가 다양하기 때문이다. Small defects는 high-resolution features(layer1)에서 잘 보인다. Large defects는 low-resolution features(layer3)에서 명확하다. Single level만 사용하면 특정 크기의 결함을 놓칠 수 있다.

Empirically 각 level의 기여도를 분석한 연구가 있다. Layer1 only: 93.5% AUROC. Layer2 only: 94.8% AUROC. Layer3 only: 93.2% AUROC. Three levels combined: 95.5% AUROC. Multi-level이 1-2%포인트 향상을 제공한다. 이는 통계적으로 유의미하다.

카테고리별로 어떤 level이 중요한지 다르다. Texture 카테고리(carpet, grid)는 layer1이 가장 informative하다. Fine-grained patterns이 중요하기 때문이다. Object 카테고리(bottle, cable)는 layer2-3가 더 중요하다. Structural information이 주된 signal이다.

Pyramid matching은 이러한 카테고리별 차이를 자동으로 처리한다. 모든 levels를 포함하므로 어떤 카테고리든 적절한 level이 기여한다. Manual tuning 없이 robust한 성능을 제공한다.

**Computational Considerations**

Multi-level matching의 대가는 계산 비용이다. Three levels를 처리하고 각각에서 discrepancy를 계산해야 한다. Single level보다 약 3배 느리다. 그러나 절대 시간은 여전히 reasonable하다(50-100ms).

메모리도 증가한다. 세 개의 feature maps를 동시에 유지해야 한다. 특히 high-resolution layer1이 메모리를 많이 사용한다. Batch inference 시 제약이 될 수 있다. Small batch size를 사용하거나 gradient checkpointing을 적용한다.

Optimization으로 속도를 향상시킬 수 있다. Intermediate feature maps를 efficient하게 cache한다. Redundant computations를 제거한다. GPU에서 parallel processing을 maximize한다. TensorRT 같은 inference engine을 사용한다.

실무에서는 trade-off를 평가해야 한다. 최고 정확도가 필요하면 three levels를 사용한다. 속도가 critical하면 single level(layer2)을 선택한다. 1-2%포인트 정확도 감소로 2-3배 속도 향상을 얻는다. 대부분의 응용에서 reasonable한 trade-off다.

### 2.2.2 Multi-scale Knowledge Transfer

Feature pyramid matching은 knowledge transfer를 multi-scale로 수행한다. 각 scale에서 서로 다른 유형의 지식이 전달된다.

**Scale-specific Knowledge**

Low-level features(layer1)는 texture knowledge를 transfer한다. Teacher는 다양한 textures를 ImageNet에서 학습했다. Fabrics, surfaces, patterns의 appearances를 인코딩한다. Student는 이러한 texture understanding을 정상 제품에 특화하여 학습한다.

정상 제품의 표면 texture가 어떻게 생겼는지를 배운다. Smooth한지, rough한지, patterned인지를 구별한다. 결함은 texture의 disruption으로 나타난다. Scratches는 smoothness를 깨뜨리고, contamination은 pattern을 방해한다. Layer1에서의 discrepancy가 이를 포착한다.

Mid-level features(layer2)는 pattern knowledge를 transfer한다. Repeated structures, local shapes, part configurations를 다룬다. Teacher는 일반적인 visual patterns를 안다. Student는 정상 제품의 specific patterns를 학습한다.

제품의 특정 부분들이 어떻게 배열되는지를 배운다. 나사의 위치, 라벨의 방향, 부품의 alignment 등이다. 결함은 pattern의 irregularity로 나타난다. Misaligned parts, missing components, wrong configurations 등이다. Layer2에서의 discrepancy가 이를 탐지한다.

High-level features(layer3)는 structural knowledge를 transfer한다. Global shapes, object categories, semantic contexts를 인코딩한다. Teacher는 objects의 overall structure를 이해한다. Student는 정상 제품의 expected structure를 학습한다.

제품의 전체적인 형태와 구조를 배운다. Bottle의 실루엣, cable의 curvature, screw의 geometry 등이다. 결함은 structure의 violation으로 나타난다. Deformations, breaks, large missing parts 등이다. Layer3에서의 discrepancy가 이를 감지한다.

**Complementary Information**

Multi-scale transfer의 장점은 complementary information이다. 각 scale이 다른 aspects를 담당한다. Small details에서 global structure까지 spectrum을 커버한다. 하나의 scale이 놓친 것을 다른 scale이 포착한다.

예를 들어 small scratch를 생각해보자. Layer1은 texture disruption을 강하게 감지한다. High discrepancy를 보인다. Layer3는 전체 구조가 intact하므로 low discrepancy다. Combined signal은 "local texture anomaly, global structure normal"을 나타낸다. 이는 정확히 small scratch의 특성이다.

Large crack의 경우 반대다. Layer1은 crack의 fine details를 보지만 context가 부족하다. Layer3는 structural damage를 명확히 감지한다. High discrepancy를 보인다. Combined signal은 "major structural anomaly"를 나타낸다. 이는 large crack과 일치한다.

Medium defects는 layer2가 주로 담당한다. Pattern disruptions, part misalignments 등이다. Layer1과 layer3의 기여는 moderate하다. Multi-scale의 adaptive nature가 드러난다. 결함 유형에 따라 다른 scales가 주도적이다.

**Transfer Dynamics**

학습 과정에서 각 scale의 transfer 속도가 다르다. Low-level features가 먼저 converge한다. Texture matching이 relatively simple하기 때문이다. Layer1의 loss가 빠르게 감소한다. 수십 epochs 안에 plateau에 도달한다.

High-level features는 느리게 converge한다. Semantic understanding의 transfer가 더 어렵다. Layer3의 loss가 천천히 감소한다. 수백 epochs가 필요할 수 있다. 충분한 training time이 중요하다.

이러한 dynamics는 curriculum learning과 유사하다. Easy tasks(texture matching)를 먼저 학습하고 hard tasks(structure matching)를 나중에 학습한다. Multi-scale framework가 자연스럽게 이를 구현한다. 명시적인 curriculum design 없이도 효과를 얻는다.

Learning rate scheduling이 이를 도울 수 있다. 초기에는 높은 learning rate로 빠른 convergence를 촉진한다. 후반에는 낮은 learning rate로 fine-tuning한다. Cosine annealing이 적합하다. Multi-scale transfer를 smooth하게 만든다.

**Scale-adaptive Weighting**

고정된 가중치 대신 adaptive weighting을 고려할 수 있다. 각 scale의 중요도를 데이터에서 학습한다. Attention mechanism을 사용한다.

$$\alpha_l = \text{Softmax}(\text{MLP}(\text{GlobalPool}(\mathbf{F}^T_l, \mathbf{F}^S_l)))$$

여기서 $\alpha_l$은 scale $l$의 가중치다. MLP는 small network로 features의 global statistics를 보고 가중치를 결정한다. 학습 가능하므로 최적 weighting을 찾는다.

Adaptive weighting은 카테고리별 차이를 자동으로 처리한다. Texture 카테고리에서는 layer1에 높은 가중치를 할당한다. Object 카테고리에서는 layer2-3에 집중한다. Manual tuning이 불필요하다.

그러나 복잡성이 증가한다. Hyperparameter tuning이 더 어려워진다. Overfitting 위험도 있다. 소규모 데이터셋에서는 simple uniform weighting이 더 robust할 수 있다. Empirically 성능 향상이 미미한 경우가 많다(0.5%포인트 미만).

### 2.2.3 Loss Functions

STFPM의 loss function은 단순하지만 효과적이다. Multi-scale feature matching을 L2 distance로 구현한다.

**Primary Loss: L2 Distance**

각 pyramid level에서 L2 loss를 계산한다. Teacher와 student features의 pixel-wise squared differences다.

$$\mathcal{L}_l = \frac{1}{h_l \times w_l \times c_l} \sum_{i,j,k} (\mathbf{F}^T_{l,ijk} - \mathbf{F}^S_{l,ijk})^2$$

여기서 $h_l \times w_l$은 spatial dimensions, $c_l$은 channels다. $(i,j,k)$는 spatial location과 channel index다. Normalization constant는 feature map의 total elements다.

전체 loss는 세 levels의 weighted sum이다.

$$\mathcal{L}_{\text{total}} = \sum_{l=1}^{3} w_l \mathcal{L}_l$$

STFPM은 uniform weighting을 사용한다: $w_1 = w_2 = w_3 = 1$. 이는 간단하고 robust하다. 각 level이 동등하게 기여한다.

L2 loss의 장점은 단순성과 안정성이다. 계산이 straightforward하고 gradient가 well-behaved하다. 모든 spatial locations를 equally treat한다. Outliers에 sensitive할 수 있지만 대부분의 경우 문제없다.

**Alternative Loss: Cosine Similarity**

일부 구현은 cosine similarity loss를 사용한다. Features의 direction을 matching한다. Magnitude는 무시한다.

$$\mathcal{L}_{\text{cos}} = 1 - \frac{\mathbf{F}^T \cdot \mathbf{F}^S}{\|\mathbf{F}^T\| \|\mathbf{F}^S\|}$$

Cosine loss는 scale-invariant하다. Teacher와 student의 feature magnitudes가 크게 다를 때 유리하다. Normalized features를 implicitly 다룬다.

실험적으로 L2와 cosine loss의 성능 차이는 작다. 대부분의 카테고리에서 0.5%포인트 이내다. L2가 약간 더 나은 경향이 있다. Default로 L2를 권장한다. Cosine은 specific issues(scale mismatch)가 있을 때 고려한다.

**Perceptual Loss Consideration**

Perceptual loss를 추가할 수도 있다. Pre-trained VGG network로 perceptual similarity를 측정한다. Human perception과 더 align된다.

$$\mathcal{L}_{\text{percept}} = \sum_{l} \| \phi_l(\mathbf{F}^T) - \phi_l(\mathbf{F}^S) \|^2$$

여기서 $\phi_l$은 VGG의 layer $l$ features다. Multiple VGG layers에서 distance를 계산하고 합산한다.

그러나 perceptual loss는 computational overhead를 추가한다. 별도의 VGG forward pass가 필요하다. 메모리와 시간이 증가한다. STFPM은 이를 사용하지 않는다. Simple L2가 충분하다.

**Regularization Terms**

명시적인 regularization terms는 포함되지 않는다. Weight decay가 optimizer에 추가될 수 있다. 일반적으로 $10^{-4}$ - $10^{-5}$를 사용한다. Student의 overfitting을 방지한다.

Dropout이나 other explicit regularizers는 사용하지 않는다. Student의 limited capacity가 implicit regularization을 제공한다. 추가 regularization이 불필요한 경우가 많다.

Early stopping이 주된 regularization이다. Validation loss를 모니터링하고 개선이 멈추면 학습을 중단한다. Overfitting을 효과적으로 방지한다.

**Loss Normalization**

Feature dimensions가 level마다 다르므로 normalization이 중요하다. Layer1은 64 channels, layer2는 128, layer3는 256이다. Raw losses의 scale이 다르다. Larger channels가 더 큰 loss를 생성한다.

Normalization constant로 나눠 이를 해결한다. Total elements($h_l \times w_l \times c_l$)로 나누면 per-element average loss가 된다. Levels 간 비교 가능하다.

Alternative로 per-channel normalization을 고려할 수 있다. 각 channel을 independently normalize한다. Channel-wise mean과 variance를 0과 1로 만든다. 그러나 이는 복잡성을 증가시킨다. Simple element-wise normalization이 대부분의 경우 충분하다.

## 2.3 Technical Details

STFPM의 구현 세부사항은 재현성과 성능에 critical하다.

**Network Architecture**

Teacher와 student 모두 ResNet18을 사용한다. ImageNet pre-trained weights를 teacher에 로드한다. Student는 random initialization으로 시작한다. Identical architecture이지만 서로 다른 initialization과 training status를 가진다.

ResNet18은 relatively small하고 fast하다. 파라미터가 11.7M이고 FLOPs가 1.8G다. 추론이 빠르다(10-20ms for feature extraction). 더 큰 ResNet50이나 Wide ResNet을 사용하면 성능이 약간 향상되지만 속도가 느려진다.

Batch normalization layers를 처리하는 방식이 중요하다. Teacher는 eval mode로 고정한다. Batch norm이 running statistics를 사용한다. Student는 train mode로 batch norm을 업데이트한다. 이는 standard practice다.

Final classification layer(FC layer)는 제거한다. Feature extraction만 필요하고 classification은 하지 않는다. Layer1, layer2, layer3의 convolutional features만 사용한다.

**Training Configuration**

Adam optimizer를 learning rate $10^{-3}$으로 사용한다. Batch size는 32다. Epochs는 100이 default다. 일부 difficult 카테고리에서는 200 epochs까지 연장한다.

Data augmentation은 minimal하다. Random horizontal flip과 random crop만 사용한다. Images를 256×256으로 resize한다. ImageNet normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])을 적용한다.

Learning rate scheduling으로 cosine annealing을 사용한다. 학습 후반부에 learning rate를 점진적으로 줄인다. Final learning rate는 $10^{-5}$다. 이는 convergence를 돕는다.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs,
    eta_min=1e-5
)
```

Early stopping을 validation loss로 수행한다. Patience는 20 epochs다. 20 epochs 동안 개선이 없으면 학습을 중단한다. Best model checkpoint를 저장하고 복원한다.

**Inference Pipeline**

추론은 다음 단계로 진행된다:

1. **Preprocessing**: Input image를 256×256으로 resize하고 normalize한다.

2. **Feature Extraction**: Teacher와 student에 동시에 입력한다. Layer1, layer2, layer3에서 features를 추출한다.

3. **Discrepancy Computation**: 각 level에서 L2 distance를 계산한다. Spatial map of discrepancies를 얻는다.

4. **Upsampling**: Layer2와 layer3 discrepancy maps를 layer1 해상도로 upsample한다. Bilinear interpolation을 사용한다.

5. **Aggregation**: 세 maps를 average하여 final anomaly map을 생성한다.

6. **Scoring**: Anomaly map의 maximum value를 image-level anomaly score로 사용한다.

```python
def inference(image, teacher, student):
    x = preprocess(image)
    
    # Feature extraction
    with torch.no_grad():
        feats_t = teacher(x)  # {layer1, layer2, layer3}
        feats_s = student(x)
    
    # Discrepancy computation
    discrep_maps = []
    for feat_t, feat_s in zip(feats_t, feats_s):
        d = (feat_t - feat_s) ** 2
        d = d.mean(dim=1)  # Average over channels
        discrep_maps.append(d)
    
    # Upsampling
    target_size = discrep_maps[0].shape[-2:]
    discrep_maps[1] = F.interpolate(discrep_maps[1], target_size)
    discrep_maps[2] = F.interpolate(discrep_maps[2], target_size)
    
    # Aggregation
    anomaly_map = sum(discrep_maps) / len(discrep_maps)
    anomaly_map = F.interpolate(anomaly_map, size=256)
    
    # Scoring
    image_score = anomaly_map.max()
    
    return anomaly_map, image_score
```

**Threshold Selection**

Threshold는 validation set의 ROC analysis로 선택한다. 정상과 이상 샘플의 anomaly scores 분포를 분석한다. 목표 재현율(예: 95%)에 대응하는 threshold를 찾는다.

Alternatively F1 score를 최대화하는 threshold를 선택할 수 있다. Precision과 recall의 harmonic mean을 최적화한다. 이는 balanced metric이다.

Per-category threshold가 권장된다. 각 카테고리마다 다른 정상 점수 분포를 가진다. Global threshold는 suboptimal하다. Category-specific tuning이 1-2%포인트 향상을 제공한다.

**Memory and Speed Optimization**

메모리 사용량은 주로 feature maps에서 나온다. Three levels를 동시에 유지해야 한다. Batch size를 제한하거나 gradient checkpointing을 사용한다.

Mixed precision(FP16) training과 inference가 효과적이다. 메모리를 절반으로 줄이고 속도도 향상시킨다. 정확도 저하는 negligible하다(0.1%포인트 미만).

TensorRT로 모델을 optimize하면 추론이 2-3배 빠르다. Kernel fusion과 layer optimization을 자동으로 수행한다. 50ms에서 20-30ms로 줄어든다.

Model pruning이나 quantization도 고려할 수 있다. Student network를 INT8로 quantize하면 크기가 1/4이 된다. 정확도 저하는 0.5-1%포인트다. Deployment constraints가 엄격하면 valuable한 trade-off다.

## 2.4 Performance Analysis

STFPM의 성능을 다양한 측면에서 분석한다.

**Benchmark Results**

MVTec AD에서 STFPM은 다음 성능을 달성했다:
- Image-level AUROC: 95.5%
- Pixel-level AUROC: 97.1%

발표 당시(2021) 이는 competitive했다. PaDiM(97.5%)보다는 낮지만 reasonable gap이었다. PatchCore(99.1%)와는 큰 차이가 있었지만 distillation 방법의 첫 시도로서 promising했다.

**Category-wise Performance**

Texture 카테고리에서 STFPM은 강력했다. Carpet(97.8%), grid(98.2%), leather(99.1%), tile(97.5%), wood(98.4%)로 대부분 97% 이상이었다. Texture patterns의 distillation이 효과적이었다.

Object 카테고리에서는 약간 낮았다. Bottle(96.3%), cable(94.1%), capsule(93.8%), hazelnut(92.5%)로 90% 중후반이었다. Complex 3D structures가 더 challenging했다. Memory-based 방법들과의 gap이 여기서 컸다.

가장 어려운 카테고리는 screw(90.8%)와 metal_nut(91.2%)였다. 작은 parts와 높은 정상 변동이 difficulty를 높였다. 모든 방법이 어려워하는 카테고리였다.

**Ablation Studies**

Multi-scale의 효과를 분석했다:
- Layer1 only: 93.2% AUROC
- Layer2 only: 94.8% AUROC  
- Layer3 only: 92.5% AUROC
- All three levels: 95.5% AUROC

Multi-scale이 1-2%포인트 향상을 제공했다. 각 level이 complementary information을 기여했다. Layer2가 single level로는 가장 좋았다.

Teacher network 크기의 영향:
- ResNet18 teacher: 95.5% AUROC
- ResNet34 teacher: 96.1% AUROC
- ResNet50 teacher: 96.8% AUROC
- Wide ResNet-50 teacher: 97.2% AUROC

Larger teacher가 better performance를 제공했다. 더 강력한 representations가 학습에 도움이 되었다. 그러나 returns가 diminishing했다. ResNet18이 속도-정확도 balance로 reasonable했다.

Student architecture 실험:
- Same as teacher(ResNet18): 95.5% AUROC
- Smaller student(ResNet10): 94.2% AUROC
- Larger student(ResNet34): 95.3% AUROC

Smaller student가 1.3%포인트 낮았지만 2배 빠르다. Larger student는 향상이 미미했다(0.2%포인트). Same architecture가 balance가 좋았다.

**Comparison with Other Methods**

STFPM vs PaDiM: PaDiM이 2%포인트 우위(97.5% vs 95.5%)였다. Gaussian modeling이 feature matching보다 약간 효과적이었다. 그러나 STFPM이 더 빠르다(50-100ms vs 100-150ms).

STFPM vs PatchCore: PatchCore가 3.6%포인트 우위(99.1% vs 95.5%)였다. Coreset과 k-NN이 매우 강력했다. 그러나 PatchCore는 메모리를 많이 사용한다(수백 MB). STFPM은 더 compact하다(100-200MB).

STFPM vs FastFlow: FastFlow가 3%포인트 우위(98.5% vs 95.5%)였다. Normalizing flow의 probabilistic modeling이 더 정확했다. 그러나 FastFlow도 더 느리다(20-50ms vs 50-100ms). STFPM은 중간 지점이었다.

STFPM vs Autoencoder: STFPM이 명확히 우수했다(95.5% vs 90-92%). Reconstruction-based approach의 한계를 넘어섰다. Distillation이 더 효과적인 framework임을 입증했다.

**Inference Speed Analysis**

Component별 시간 분해:
- Teacher forward: 10-15ms
- Student forward: 10-15ms  
- Discrepancy computation: 5-10ms
- Upsampling and aggregation: 5-10ms
- Total: 30-50ms (single image), 50-100ms (with overhead)

Batch processing으로 throughput을 높일 수 있다. Batch size 32로 초당 20-30 images를 처리한다. 대부분의 생산 라인 속도를 충족한다.

GPU별 성능:
- RTX 3090: 30-50ms
- RTX 2080: 50-80ms
- Jetson Xavier: 100-150ms
- CPU(i9): 500-800ms

GPU에서 real-time에 가깝다. Edge device에서는 느리지만 사용 가능하다. CPU는 실용적이지 않다.

**Pixel-level Localization**

Anomaly maps의 품질을 평가했다. Pixel-level AUROC 97.1%로 좋았다. 결함 위치를 reasonably 정확히 특정했다. 그러나 boundaries가 약간 blurry했다. Upsampling 과정에서 detail 손실이 있었다.

False positives는 주로 high-variance regions에서 발생했다. 조명 변화, 반사, normal variations가 오탐되었다. 이는 student가 모든 정상 변동을 학습하지 못했기 때문이다.

False negatives는 subtle defects에서 발생했다. 작거나 contrast가 낮은 결함이 놓쳤다. Teacher와 student 모두 감지하지 못하면 discrepancy가 작다. 이는 fundamental limitation이다.

## 2.5 Baseline Establishment

STFPM의 가장 큰 기여는 knowledge distillation baseline을 확립한 것이다. 후속 연구들이 이를 기반으로 개선했다.

**Methodological Contributions**

STFPM은 teacher-student framework를 anomaly detection에 명확히 정의했다. Teacher: pre-trained, frozen. Student: trainable, specialized to normal. Feature matching as training objective. Discrepancy as anomaly score. 이 simple recipe가 효과적임을 보였다.

Multi-scale feature pyramid matching을 도입했다. 단일 layer가 아니라 hierarchical features를 활용했다. 이는 후속 연구들의 standard가 되었다. 대부분의 distillation 방법이 multi-scale을 사용한다.

Simplicity를 강조했다. Complex losses나 elaborate architectures 없이 basic L2 matching으로 좋은 성능을 얻었다. 이는 over-engineering을 피하고 essentials에 집중하게 했다. 후속 연구들도 이 철학을 따랐다.

**Limitations Identified**

STFPM은 자신의 한계도 명확히 했다. 이것이 후속 연구의 방향을 제시했다. 첫째, student와 teacher가 same architecture였다. 이는 efficiency 측면에서 suboptimal이었다. Smaller student를 사용하면 더 빠를 것이다.

둘째, feature matching이 단순한 L2였다. 더 sophisticated한 distance metrics나 perceptual losses를 고려할 수 있다. Relational knowledge transfer도 탐구할 가치가 있다.

셋째, teacher 선택이 고정적이었다. ResNet18만 시도했다. 다른 architectures나 pre-training methods가 더 나을 수 있다. Foundation models(CLIP, DINOv2)의 potential을 탐구할 필요가 있다.

넷째, training efficiency가 충분하지 않았다. 100 epochs가 필요했고 학습 시간이 수 시간이었다. Faster convergence나 fewer epochs를 목표로 할 수 있다.

**Subsequent Improvements**

Reverse Distillation(2022)은 역할을 반전시켰다. Student를 고정하고 teacher를 학습시켰다. 이는 다른 관점을 제공했고 일부 경우 더 나았다. STFPM이 열어놓은 exploration space였다.

EfficientAD(2023)는 efficiency를 극대화했다. 매우 작은 student와 efficient teacher를 사용했다. 1-5ms 추론을 달성하면서 97.8% AUROC를 유지했다. STFPM이 지적한 efficiency limitation을 해결했다.

UniAD(2022)는 unified framework를 제안했다. Query-based transformer로 distillation을 구현했다. Multiple teacher layers를 efficiently aggregate했다. STFPM의 multi-scale 개념을 확장했다.

APRIL-GAN(2023)은 generative adversarial training을 추가했다. Distillation과 generation을 결합했다. Teacher-student-generator의 three-way framework였다. STFPM의 basic distillation을 넘어섰다.

**Impact on Research Community**

STFPM 이후 distillation-based 방법들이 급증했다. 2021년 이전에는 소수였지만 이후 major category가 되었다. CVPR, ICCV, ECCV 등 top conferences에서 다수의 distillation 논문이 발표되었다. Paradigm shift를 촉발했다.

학술적으로 두 fields를 연결했다. Knowledge distillation(model compression)과 anomaly detection(one-class learning)이 독립적이었다. STFPM이 bridge를 놓았다. Cross-pollination이 활발해졌다.

실무적으로 practical alternative를 제공했다. Memory-based 방법(high accuracy, slow)과 reconstruction-based(fast, low accuracy) 사이였다. 합리적인 accuracy와 속도를 제공했다. 많은 industrial applications에 적합했다.

Benchmarking standard를 설정했다. 후속 연구들이 STFPM과 비교하는 것이 convention이 되었다. Performance improvements를 STFPM baseline 대비로 측정했다. Reproducible results와 code 공개가 이를 촉진했다.

**Current Status**

현재 STFPM은 historical baseline으로 간주된다. 최신 방법들(EfficientAD, UniAD)이 성능과 효율성에서 앞서 있다. 그러나 STFPM의 conceptual framework는 여전히 유효하다. Teacher-student distillation의 core principles는 변하지 않았다.

교육적 가치가 크다. Distillation-based anomaly detection을 배우는 starting point다. Simple하고 이해하기 쉬운 방법이다. 복잡한 최신 방법을 이해하기 전에 STFPM을 먼저 공부하는 것이 권장된다.

실무에서도 여전히 사용된다. 최고 성능이 필요 없고 안정성과 simplicity가 중요한 경우다. Proven method로서의 신뢰가 있다. 새로운 방법의 위험을 감수하고 싶지 않을 때 STFPM은 safe choice다.

STFPM의 legacy는 명확하다. Knowledge distillation을 anomaly detection의 mainstream으로 만들었다. Baseline을 확립하고 research direction을 제시했다. 후속 발전의 foundation을 놓았다. 이것이 landmark paper로서의 가치다. 최고 성능을 달성하지 않았지만 field를 변화시켰다. 이것이 진정한 contribution이다.

# 3. FRE (2023)

## 3.1 Basic Information

FRE(Feature Reconstruction Error)는 2023년 초반에 제안된 knowledge distillation 기반 방법으로, feature reconstruction을 통한 anomaly detection을 시도했다. 이 연구는 arxiv에만 공개되었고 peer-reviewed conference나 journal에는 publish되지 않았다. FRE는 distillation과 reconstruction을 결합하려는 시도였지만 학술적으로나 실무적으로 큰 영향을 미치지 못했다.

FRE의 핵심 아이디어는 student가 teacher features를 직접 모방하는 대신 reconstruction을 통해 학습하는 것이다. Student는 compressed representation을 거쳐 teacher features를 재구성한다. Autoencoder와 유사하지만 reconstruction target이 input image가 아니라 teacher features다. 이는 "feature space에서의 reconstruction"이라는 새로운 관점을 제안했다.

방법론적으로 FRE는 encoder-decoder student network를 사용한다. Encoder가 input image를 compressed features로 변환한다. Decoder가 이를 다시 teacher features로 reconstruction한다. Bottleneck이 compression을 강제하여 student가 essential patterns만 학습하도록 한다. Reconstruction error가 anomaly score가 된다.

성능 면에서 FRE는 기대에 미치지 못했다. MVTec AD에서 이미지 레벨 AUROC 약 96.5%를 달성했다. 이는 STFPM(95.5%)보다는 약간 높지만 FastFlow(98.5%)나 PatchCore(99.1%)에는 크게 못 미쳤다. 추론 속도도 80-120ms로 STFPM(50-100ms)보다 느렸다. 효율성 목표도 달성하지 못했다.

FRE의 주된 문제는 incremental improvement에 그쳤다는 것이다. STFPM 대비 1%포인트 향상은 통계적으로 유의미하지만 practical impact가 제한적이다. Revolutionary innovation이 아니라 minor variation이었다. Feature reconstruction이라는 개념은 흥미로웠지만 충분한 이점을 제공하지 못했다.

또한 FRE는 complexity를 증가시켰다. Encoder-decoder architecture가 STFPM의 단순한 feature matching보다 복잡하다. Hyperparameter tuning이 더 어렵고 학습이 덜 안정적이었다. Bottleneck size, decoder depth, reconstruction loss weights 등 많은 설계 결정이 필요했다. 이러한 복잡성이 미미한 성능 향상을 정당화하지 못했다.

학술적으로 FRE는 peer review를 통과하지 못했다. 주요 conferences나 journals에 reject되었을 가능성이 크다. Novelty 부족, insufficient improvement, unclear motivation 등이 이유였을 것이다. Arxiv-only publication은 이를 시사한다. Research community가 FRE의 기여를 인정하지 않았다.

실무적으로도 채택되지 않았다. STFPM이나 FastFlow 같은 established methods를 대체할 이유가 없었다. 성능 향상이 작고 복잡성은 높았다. 구현 난이도와 디버깅 어려움이 추가 장벽이었다. Industrial practitioners는 proven methods를 선호한다. FRE는 이 기준을 충족하지 못했다.

FRE의 실패는 중요한 교훈을 제공한다. Incremental improvements는 충분하지 않다. Significant novelty나 substantial performance gain이 필요하다. Complexity는 명확한 benefits로 정당화되어야 한다. Simple과 effective가 complex보다 낫다. 이러한 lessons는 future research에 valuable하다.

## 3.2 Feature Reconstruction Approach

FRE의 핵심 메커니즘은 teacher features의 reconstruction이다. 이는 전통적인 image reconstruction과는 다른 접근이다.

**Conceptual Framework**

전통적인 autoencoder는 input image를 reconstruction한다. Encoder가 image를 compressed representation(latent code)로 변환한다. Decoder가 latent code에서 image를 재구성한다. Bottleneck이 compression을 강제한다. 정상 이미지는 잘 재구성되고 이상은 못 재구성된다는 가정이다.

FRE는 이를 feature space로 옮긴다. Input은 여전히 image지만 reconstruction target은 teacher features다. Student encoder가 image를 compressed features로 변환한다. Student decoder가 이를 teacher features로 재구성한다. Teacher features는 target ground truth로 작동한다.

왜 이것이 의미 있는가? Teacher features는 high-level semantic information을 담는다. Raw images보다 structured되어 있다. Reconstruction이 더 쉬울 수 있다. 또한 teacher features는 pre-trained knowledge를 반영한다. Student가 이 knowledge를 internalize하도록 강제한다.

수학적으로 다음과 같이 표현된다. Teacher function $T(\mathbf{x})$가 features $\mathbf{F}^T$를 생성한다. Student는 encoder $E_S$와 decoder $D_S$로 구성된다. Reconstruction process는:

$$\mathbf{z} = E_S(\mathbf{x})$$
$$\hat{\mathbf{F}}^T = D_S(\mathbf{z})$$

여기서 $\mathbf{z}$는 compressed representation이고 $\hat{\mathbf{F}}^T$는 reconstructed teacher features다. Reconstruction loss는:

$$\mathcal{L}_{\text{recon}} = \| \mathbf{F}^T - \hat{\mathbf{F}}^T \|^2$$

Student는 정상 데이터에서 이를 최소화하도록 학습된다. 학습 후 정상 샘플은 낮은 reconstruction error를, 이상 샘플은 높은 error를 보인다는 가정이다.

**Architecture Design**

FRE의 student network는 encoder-decoder 구조다. Encoder는 여러 convolutional layers로 구성된다. 점진적으로 spatial dimensions를 줄이고 channels를 늘린다. 최종 bottleneck은 compressed representation이다.

```
Encoder:
Input (256×256×3)
  → Conv+ReLU (128×128×64)
  → Conv+ReLU (64×64×128)
  → Conv+ReLU (32×32×256)
  → Bottleneck (16×16×512)
```

Decoder는 symmetric하게 설계된다. Transposed convolutions나 upsampling으로 spatial dimensions를 복원한다. Channels를 줄여 teacher features와 match시킨다.

```
Decoder:
Bottleneck (16×16×512)
  → TransConv+ReLU (32×32×256)
  → TransConv+ReLU (64×64×128)
  → TransConv+ReLU (128×128×64)
  → Output (matching teacher feature dimensions)
```

Bottleneck size는 critical hyperparameter다. 너무 크면 모든 정보를 preserve하여 compression이 없다. 너무 작으면 정상도 제대로 reconstruct하지 못한다. FRE는 teacher feature dimensions의 1/4 - 1/2를 사용한다.

Skip connections를 추가할 수도 있다. U-Net style로 encoder와 decoder를 연결한다. Low-level details를 preserve하는 데 도움이 된다. 그러나 이는 bottleneck constraint를 약화시킨다. Trade-off가 있다.

**Multi-scale Reconstruction**

STFPM처럼 FRE도 multi-scale을 시도했다. Teacher의 여러 layers(layer1, layer2, layer3)에서 features를 추출한다. Student decoder는 multiple outputs을 생성하여 각각을 reconstruct한다.

각 scale에서 reconstruction loss를 계산하고 합산한다.

$$\mathcal{L}_{\text{total}} = \sum_{l=1}^{3} w_l \| \mathbf{F}^T_l - \hat{\mathbf{F}}^T_l \|^2$$

이는 구조를 더 복잡하게 만든다. Decoder가 multiple branches를 가져야 한다. 각 branch는 다른 해상도의 features를 생성한다. 계산 비용이 증가한다.

실험적으로 multi-scale reconstruction의 이득은 제한적이었다. Single scale(layer2)이 93.5% AUROC, three scales가 96.5%로 3%포인트 향상이었다. 그러나 complexity와 computation 증가를 고려하면 cost-benefit이 불리했다.

**Comparison with Direct Matching**

Feature reconstruction과 direct feature matching(STFPM)의 차이는 무엇인가? Direct matching은 student가 teacher features를 직접 생성하도록 한다. Reconstruction은 bottleneck을 거쳐 생성하도록 한다.

이론적으로 reconstruction이 더 강한 constraint다. Bottleneck이 compression을 강제하여 essential information만 학습한다. Direct matching은 제약이 약하다. Student가 모든 details를 모방할 수 있다.

그러나 실무적으로 차이가 미미했다. FRE(96.5%)와 STFPM(95.5%)의 1%포인트 차이는 크지 않다. Student의 limited capacity가 이미 implicit constraint를 제공한다. Explicit bottleneck의 추가 이득이 작다.

또한 reconstruction은 학습을 어렵게 만든다. Encoder-decoder training이 direct matching보다 불안정하다. Gradient vanishing이나 mode collapse 위험이 있다. Careful initialization과 tuning이 필요하다.

**Theoretical Motivation Revisited**

FRE의 motivation은 "feature space reconstruction이 image space보다 쉽다"였다. Teacher features가 더 structured되고 semantic하므로 reconstruction target으로 적합하다는 주장이었다. 이는 reasonable한 가설이다.

그러나 empirical evidence가 약했다. Feature reconstruction의 우위를 명확히 입증하지 못했다. Image reconstruction(standard autoencoder)과 비교했을 때 marginal advantage만 보였다. 일부 카테고리에서는 오히려 image reconstruction이 더 나았다.

근본적 문제는 teacher features도 충분히 complex하다는 것이다. ResNet layer3 features는 256 channels × 32 × 32 = 262K dimensions다. 이는 원본 image(3 × 256 × 256 = 196K)보다 많다. "Simpler reconstruction target"이라는 가정이 성립하지 않았다.

또한 teacher features의 quality가 critical하다. Poor teacher는 poor reconstruction target이다. Student가 잘못된 target을 학습한다. Teacher selection이 FRE의 성능을 크게 좌우했다. 이는 additional complexity다.

## 3.3 Lightweight Architecture

FRE는 efficiency를 목표 중 하나로 삼았다. Lightweight student architecture로 빠른 추론을 달성하려 했다. 그러나 이 목표도 완전히 성공하지 못했다.

**Network Size Reduction**

FRE의 student는 teacher보다 훨씬 작다. Teacher가 ResNet18(11.7M parameters)이면, student는 약 3-5M parameters다. 1/3 - 1/4 크기다. 이론적으로 추론이 빠르고 메모리가 적어야 한다.

구체적인 설계는 다음과 같다. Encoder는 5-6 convolutional layers로 얕다. 각 layer는 small kernels(3×3)과 moderate channels(64-256)를 사용한다. Decoder도 symmetric하게 얕다. Total parameters가 3-5M 범위에 있다.

Lightweight design의 장점은 명확하다. Training이 빠르다. Small network가 convergence가 빠르다. Deployment가 쉽다. 적은 메모리로 여러 카테고리를 동시에 load할 수 있다. Edge devices에서도 실행 가능하다.

그러나 단점도 있다. Limited capacity가 성능을 제약한다. 복잡한 정상 패턴을 충분히 학습하지 못할 수 있다. Underfitting 위험이 있다. Optimal capacity를 찾기 어렵다.

**Efficiency Bottlenecks**

FRE의 추론 시간을 분석하면 bottlenecks가 드러난다.

- Teacher forward: 15-20ms (고정, 변경 불가)
- Student encoder: 10-15ms
- Student decoder: 15-25ms
- Reconstruction error computation: 5-10ms
- Total: 45-70ms (best case), 80-120ms (typical)

Decoder가 가장 큰 병목이다. Upsampling과 transposed convolutions이 expensive하다. Encoder보다 오래 걸린다. 이는 예상 밖이었다. Lightweight 설계에도 불구하고 decoder가 dominant cost다.

Teacher forward pass도 줄일 수 없다. Pre-trained model이므로 고정되어 있다. 이것만으로 15-20ms를 차지한다. Student를 아무리 경량화해도 이 부분은 남는다.

결과적으로 FRE는 STFPM(50-100ms)보다 느렸다. Lightweight 목표가 실패했다. Decoder overhead가 예상보다 컸다. Direct feature matching(STFPM)이 오히려 더 efficient했다.

**Optimization Attempts**

FRE 저자들은 여러 최적화를 시도했다. Decoder를 더 shallow하게 만들었다. 3-4 layers로 줄였다. 그러나 reconstruction quality가 저하되었다. Performance가 1-2%포인트 떨어졌다.

Depthwise separable convolutions을 사용했다. MobileNet style로 convolutions을 factorize했다. Parameters와 FLOPs가 감소했다. 그러나 추론 속도 향상은 미미했다(10-15% 정도). GPU에서 memory bandwidth가 병목이었다.

Knowledge distillation을 decoder에도 적용했다. Large decoder를 teacher로 사용하고 small decoder를 student로 학습시켰다. 이는 meta-distillation이다. Complexity가 크게 증가했고 이득은 불분명했다.

Quantization(INT8)을 시도했다. Model size가 1/4로 줄었다. 그러나 reconstruction quality 저하가 컸다. AUROC가 1.5-2%포인트 떨어졌다. Feature reconstruction이 quantization에 민감했다.

**Comparison with EfficientAD**

같은 시기에 제안된 EfficientAD와 비교하면 FRE의 한계가 명확하다. EfficientAD는 1-5ms 추론을 달성했다. FRE는 80-120ms로 10배 이상 느렸다. 같은 "efficient" 목표를 가졌지만 결과가 극명히 달랐다.

EfficientAD의 성공 요인은 무엇이었나? Extremely small student network(1M parameters 미만). Patch-based processing으로 computational cost 감소. Clever teacher feature selection. Optimized inference pipeline. 이 모든 것이 결합되어 극적인 속도 향상을 이뤘다.

FRE는 이러한 radical optimization을 하지 않았다. Conventional encoder-decoder design을 따랐다. Incremental improvements만 시도했다. Decoder overhead를 근본적으로 해결하지 못했다. Revolutionary approach가 아니라 evolutionary였다.

**Practical Implications**

FRE의 efficiency 실패는 중요한 시사점을 준다. Lightweight architecture만으로는 충분하지 않다. Overall pipeline optimization이 필요하다. Bottlenecks를 정확히 식별하고 targeted improvements를 해야 한다.

또한 efficiency와 accuracy의 trade-off를 신중히 다뤄야 한다. Aggressive simplification은 performance 저하를 초래한다. Optimal point를 찾기 어렵다. Extensive experimentation이 필요하다.

Incremental optimization의 한계도 명확하다. 10-20% 속도 향상은 practical impact가 제한적이다. 2-5배 향상이 meaningful하다. 이는 architectural innovation이나 algorithmic breakthroughs를 요구한다. FRE는 이를 달성하지 못했다.

## 3.4 Speed Optimization Attempt

FRE는 speed optimization을 명시적 목표로 삼았다. 여러 techniques을 시도했지만 대부분 실패했다.

**Computational Graph Optimization**

FRE는 computational graph를 analyze하여 redundant operations을 제거하려 했다. Multiple forward passes(teacher + student)에서 shared computations를 identify했다. 예를 들어 initial layers는 공유 가능할 수 있다.

그러나 실제로 sharing이 제한적이었다. Teacher는 frozen이고 student는 trainable이므로 완전히 분리되어야 한다. Batch normalization statistics가 다르다. Shared layers는 training dynamics를 복잡하게 만든다.

결국 minimal sharing만 구현했다. Preprocessing pipeline을 공유했다(resize, normalize). 이는 trivial optimization이다. Core computation은 여전히 independent했다. 속도 향상은 5% 미만이었다.

**Memory Access Optimization**

GPU에서 memory bandwidth가 종종 병목이다. FRE는 memory access patterns를 optimize하려 했다. Feature maps를 cache-friendly하게 layout했다. Tiled processing으로 locality를 향상시켰다.

이론적으로 타당했지만 practical impact는 작았다. Modern GPUs는 sophisticated memory hierarchies를 가진다. Manual optimization의 여지가 제한적이다. Framework(PyTorch, TensorFlow)가 이미 많은 최적화를 한다.

실험 결과 memory optimization은 5-10% 속도 향상을 제공했다. Significant하지만 transformative하지 않다. 80ms에서 72ms로 줄었다. 여전히 target(30-50ms)에 훨씬 못 미쳤다.

**Mixed Precision Training**

FP16 mixed precision으로 training과 inference를 accelerate하려 했다. Modern GPUs(Volta 이후)는 FP16 operations이 2-3배 빠르다. 메모리도 절반으로 줄어든다.

Training에서 mixed precision은 잘 작동했다. Loss scaling으로 numerical stability를 유지했다. Training speed가 1.5-2배 향상되었다. 메모리 절약으로 larger batch sizes를 사용할 수 있었다.

Inference에서는 문제가 있었다. Feature reconstruction이 precision에 sensitive했다. FP16으로 inference하면 reconstruction quality가 저하되었다. AUROC가 0.5-1%포인트 떨어졌다. 이는 acceptable하지 않았다.

결국 training은 FP16, inference는 FP32를 사용했다. Inference speed 향상은 없었다. Mixed precision의 이득을 fully leverage하지 못했다.

**Model Pruning**

Unimportant weights를 pruning하여 model size를 줄이려 했다. Magnitude-based pruning으로 small weights를 제거했다. 30-50% sparsity를 목표로 했다.

Pruning 후 fine-tuning으로 accuracy를 recover했다. 30% sparsity에서 AUROC 저하가 0.3%포인트로 minimal했다. Model size가 크게 줄었다. 이는 성공적이었다.

그러나 sparse models의 inference acceleration은 제한적이었다. Standard frameworks(PyTorch)는 sparse operations을 efficiently support하지 않는다. Specialized libraries(TensorRT, ONNX Runtime)가 필요했다. Even then, speedup이 기대에 못 미쳤다(10-20%).

Structured pruning(entire filters 제거)을 시도했지만 accuracy 저하가 컸다. 1-2%포인트 떨어졌다. Trade-off가 불리했다. Pruning은 deployment size 감소에는 유용했지만 speed optimization에는 실패했다.

**Ensemble Reduction**

일부 implementations는 ensemble of students를 사용했다. Multiple students를 학습하고 평균을 취해 robustness를 높인다. 그러나 이는 inference를 N배(ensemble size) 느리게 만든다.

FRE는 ensemble을 single student로 distill하려 했다. Ensemble의 knowledge를 하나의 compact model로 transfer한다. 이는 second-order distillation이다. Complexity가 매우 높았다.

결과는 mixed였다. Single distilled student가 ensemble 성능의 80-90%를 달성했다. 그러나 여전히 single STFPM student보다 크게 낫지 않았다. Additional complexity가 정당화되지 않았다.

또한 ensemble training과 distillation의 computational cost가 컸다. Total training time이 5-10배 증가했다. Deployment 시 속도는 빨라졌지만 development cycle이 느려졌다. Practical workflow에서 불리했다.

**Failed Optimizations: Lessons**

FRE의 optimization attempts는 대부분 실패했다. 이는 중요한 lessons를 제공한다. 첫째, low-hanging fruits는 이미 대부분 picked되었다. Standard frameworks가 많은 basic optimizations를 제공한다. Manual micro-optimization의 여지가 작다.

둘째, architectural bottlenecks는 algorithm-level changes를 요구한다. Decoder overhead는 implementation tricks로 해결되지 않는다. Fundamental redesign이 필요했다. FRE는 이를 하지 않았다.

셋째, optimization은 holistic해야 한다. Individual components를 optimize하는 것으로는 부족하다. End-to-end pipeline의 coherent design이 필요하다. FRE는 piecemeal approach를 취했다.

넷째, measurement와 profiling이 critical하다. Assumed bottlenecks가 actual bottlenecks와 다를 수 있다. Careful profiling으로 정확히 identify해야 한다. FRE는 이를 충분히 하지 않았을 수 있다.

## 3.5 Why FRE Failed

FRE의 실패를 체계적으로 분석하면 중요한 insights를 얻을 수 있다. 이는 future research에 valuable하다.

### 3.5.1 Insufficient Improvement

FRE의 가장 큰 문제는 insufficient improvement였다. 기존 방법 대비 개선이 미미했다.

**Marginal Performance Gain**

FRE는 STFPM 대비 1%포인트 향상을 달성했다(96.5% vs 95.5%). 이는 통계적으로 유의미할 수 있지만 practical significance가 제한적이다. 1%포인트 차이는 many applications에서 거의 구별되지 않는다.

예를 들어 1000개 제품 중 정상 950개, 이상 50개가 있다고 하자. 95.5% AUROC로 약 47-48개를 탐지한다. 96.5%로 48-49개를 탐지한다. 1-2개 차이다. Operational impact가 미미하다.

또한 1%포인트는 experimental noise 범위 내일 수 있다. Different random seeds, data splits, hyperparameters로 0.5-1%포인트 변동이 일반적이다. FRE의 향상이 robust한지 불확실했다. Extensive ablations나 statistical tests가 부족했다.

Comparison with state-of-the-art도 불리했다. PatchCore(99.1%)와는 2.6%포인트 차이가 있었다. FastFlow(98.5%)와도 2%포인트 차이였다. FRE는 top-tier에 진입하지 못했다. Competitive position이 약했다.

**No Speed Advantage**

FRE는 efficiency를 목표로 삼았지만 달성하지 못했다. 80-120ms 추론은 STFPM(50-100ms)보다 느렸다. FastFlow(20-50ms)와는 비교도 안 되었다. Speed advantage가 없었다.

실무에서 speed는 critical factor다. Real-time processing이 필요한 응용이 많다. 80ms는 초당 12 프레임에 불과하다. 많은 생산 라인이 초당 30-60 프레임을 요구한다. FRE는 이를 충족하지 못했다.

또한 latency가 user experience에 영향을 미친다. Interactive inspection systems에서 100ms delay는 noticeable하다. 50ms 이하가 desirable하다. FRE는 이 threshold를 넘었다.

**Complexity Increase**

FRE는 complexity를 증가시켰다. Encoder-decoder architecture가 STFPM의 direct matching보다 복잡하다. Hyperparameters가 많다: bottleneck size, decoder depth, reconstruction weights, loss balancing 등.

Complexity는 practical deployment에 장벽이다. 구현이 어렵다. Debugging이 까다롭다. Maintenance burden이 크다. Industrial practitioners는 simple solutions를 선호한다. Complexity는 명확한 benefits로 정당화되어야 한다.

FRE의 complexity가 제공하는 benefits는 무엇인가? 1%포인트 accuracy 향상? 이는 충분하지 않다. Speed나 memory efficiency? 오히려 더 나빴다. Theoretical elegance? 실무에서는 secondary다. Complexity-benefit balance가 불리했다.

**Unclear Value Proposition**

FRE의 value proposition이 불명확했다. "언제 FRE를 사용해야 하는가?"에 대한 답이 없었다. STFPM, FastFlow, PatchCore 등 established methods가 있는 상황에서 FRE의 niche가 무엇인가?

Highest accuracy가 필요하면 PatchCore를 선택한다. Fastest speed가 필요하면 EfficientAD를 선택한다. Balanced performance가 필요하면 FastFlow를 선택한다. Simple baseline이 필요하면 STFPM을 선택한다. FRE는 어디에 fit하는가? 명확하지 않았다.

Marketing perspective에서 이는 치명적이다. Research community든 industry든 clear value proposition이 필요하다. "Why should I use this instead of alternatives?"에 compelling answer가 있어야 한다. FRE는 이를 제공하지 못했다.

### 3.5.2 Incremental vs Revolutionary

FRE는 incremental innovation이었다. Revolutionary breakthrough가 아니었다. 이것이 근본적인 문제였다.

**Incremental Nature**

FRE는 기존 ideas의 combination이었다. Distillation(STFPM에서), reconstruction(autoencoder에서), feature space learning(many methods에서). 각각은 known concepts였다. FRE는 이들을 combine했을 뿐이다.

Combination 자체가 나쁜 것은 아니다. 많은 successful research가 smart combinations다. 그러나 combination이 synergy를 만들어야 한다. 1+1이 3이 되어야 한다. FRE는 1+1=1.5 정도였다. Synergy가 충분하지 않았다.

Incremental research도 가치가 있다. Science는 gradual progress다. Small steps가 누적되어 큰 발전을 이룬다. 그러나 incremental work는 higher standards를 충족해야 한다. Thorough analysis, extensive experiments, clear insights가 필요하다. FRE는 이것도 부족했다.

**Lack of Novel Insights**

FRE는 novel insights를 제공하지 못했다. "왜 feature reconstruction이 effective한가?"에 대한 deep understanding이 없었다. Empirical results만 보고했다. Theoretical analysis나 intuitive explanations가 부족했다.

좋은 research는 "what works"뿐만 아니라 "why it works"를 설명한다. Mechanisms을 이해하고 principles을 추출한다. FRE는 "feature reconstruction으로 96.5% AUROC 달성"이 전부였다. Why가 missing이었다.

Ablation studies도 superficial했다. Component X를 제거하면 성능이 Y% 떨어진다는 수준이었다. 각 component의 role이나 interaction에 대한 deep dive가 없었다. Design choices의 rationale이 unclear했다.

Failure analysis도 부족했다. 어떤 경우에 FRE가 실패하는가? 왜 실패하는가? 어떻게 개선할 수 있는가? 이러한 질문들에 대한 답이 없었다. Limitations를 솔직히 논의하지 않았다.

**Revolutionary Alternatives**

같은 시기의 revolutionary works와 비교하면 차이가 명확하다. EfficientAD는 1-5ms 추론으로 10-20배 speedup을 달성했다. 이는 transformative였다. Real-time deployment의 문을 열었다. Paradigm shift였다.

WinCLIP은 foundation models(CLIP)를 anomaly detection에 도입했다. Zero-shot capability를 제공했다. Training 없이 new categories를 처리했다. 이는 game changer였다. Few-shot learning의 새로운 가능성을 열었다.

이러한 revolutionary works는 common characteristics를 가진다. Bold ideas를 제시한다. Significant performance jumps를 달성한다(5-10%포인트 이상). New capabilities를 enable한다. Clear value propositions를 가진다. FRE는 이 중 어느 것도 충족하지 못했다.

**Missed Opportunities**

FRE는 potentially interesting directions를 충분히 explore하지 않았다. Feature space reconstruction은 실제로 promising할 수 있다. 그러나 shallow exploration에 그쳤다.

예를 들어 hierarchical reconstruction을 시도할 수 있었다. Coarse-to-fine으로 multiple resolutions에서 reconstruct한다. Progressive refinement로 더 정확한 reconstruction을 달성한다. 이는 탐구되지 않았다.

또는 adversarial reconstruction을 고려할 수 있었다. GAN-based approach로 더 realistic한 reconstructions를 생성한다. Discriminator가 reconstruction quality를 enforce한다. 이것도 시도되지 않았다.

Self-supervised learning의 recent advances를 leverage할 수도 있었다. Contrastive learning이나 masked autoencoding을 feature reconstruction에 integrate한다. 이는 potentially powerful combination이다. Missed opportunity였다.

### 3.5.3 Lessons Learned

FRE의 실패에서 배울 수 있는 lessons는 무엇인가?

**Lesson 1: Incremental is Not Enough**

Incremental improvements만으로는 insufficient하다. Top conferences나 journals는 significant novelty를 요구한다. Industry는 clear advantages를 원한다. 1-2%포인트 향상은 insufficient하다.

Revolutionary ideas를 추구해야 한다. 물론 모든 research가 revolutionary일 수는 없다. 그러나 bold attempts를 해야 한다. Safe하고 incremental한 work는 impact가 제한적이다.

만약 incremental approach를 취한다면 extremely thorough해야 한다. Comprehensive analysis, extensive experiments, deep insights로 compensate해야 한다. Incremental work도 valuable할 수 있지만 higher bar가 있다.

**Lesson 2: Complexity Needs Justification**

Complexity는 명확한 benefits로 정당화되어야 한다. Simple solution이 잘 작동하면 complex solution이 필요 없다. Occam's razor를 존중해야 한다.

FRE의 encoder-decoder가 STFPM의 direct matching보다 complex하다. 이 complexity가 meaningful advantages를 제공했는가? No. 그렇다면 simple solution(STFPM)이 선호된다.

새로운 complexity를 introduce할 때 항상 물어야 한다: "이것이 정말 필요한가? Simpler alternatives는 없는가? Benefits가 costs를 justify하는가?" FRE는 이 questions에 satisfactory answers를 제공하지 못했다.

**Lesson 3: Clear Value Proposition**

Research는 clear value proposition이 필요하다. "왜 이 work를 care해야 하는가?"에 compelling answer가 있어야 한다. Academic이든 industrial이든 마찬가지다.

Value proposition은 여러 형태일 수 있다. Highest accuracy. Fastest speed. Lowest memory. Best interpretability. Novel capabilities. Theoretical insights. 무엇이든 something이 exceptional해야 한다.

FRE는 어느 dimension에서도 exceptional하지 않았다. Jack of all trades, master of none이었다. 이는 adoption의 큰 장벽이다. Clear positioning이 필요했다.

**Lesson 4: Execution Matters**

Good idea도 poor execution으로 실패할 수 있다. FRE의 core idea(feature reconstruction)는 reasonable했다. 그러나 execution이 부족했다.

Thorough experimentation이 필요했다. Multiple datasets, architectures, settings를 test해야 했다. Statistical rigor를 갖춰야 했다. Multiple runs, confidence intervals, significance tests가 필요했다.

Clear presentation도 critical하다. Motivations를 명확히 설명해야 한다. Results를 convincingly present해야 한다. Limitations를 honestly discuss해야 한다. FRE의 paper writing이 이 standards를 충족하지 못했을 가능성이 크다.

**Lesson 5: Timing and Context**

Research는 timing과 context도 중요하다. FRE가 2019년에 나왔다면 다르게 받아들여졌을 수 있다. 그러나 2023년에는 이미 많은 strong baselines가 있었다. Competition이 fierce했다.

Field의 maturity level을 고려해야 한다. Early stage에서는 exploratory work가 valuable하다. Mature stage에서는 significant advances가 필요하다. Anomaly detection은 2023년에 이미 relatively mature했다. FRE의 incremental contribution이 insufficient했다.

또한 community의 interests와 align해야 한다. 2023년에 community는 foundation models, zero-shot learning, extreme efficiency 등에 관심이 있었다. FRE의 conventional approach가 out of sync였다.

**Positive Takeaways**

실패에서도 positive takeaways가 있다. FRE는 feature reconstruction이라는 idea를 explore했다. 비록 성공하지 못했지만 direction 자체는 valid하다. Future work가 better execution으로 succeed할 수 있다.

또한 FRE는 what not to do를 보여줬다. 이것도 valuable lesson이다. 다른 researchers가 similar pitfalls를 피할 수 있다. Negative results도 community에 기여한다.

FRE의 experiments는 일부 useful insights를 제공했다. Bottleneck size의 영향, multi-scale reconstruction의 효과 등이다. 비록 major contribution은 아니지만 incremental knowledge를 추가했다.

**Conclusion on FRE**

FRE는 unsuccessful research의 case study다. Reasonable idea, decent execution, but insufficient impact. Incremental improvement, added complexity, no clear advantage. 결과적으로 community에 adopt되지 않았다.

그러나 FRE는 valuable lessons를 제공한다. Bold over safe. Simple over complex. Clear value proposition. Thorough execution. Proper timing. 이러한 principles는 successful research의 필수 요소다.

Future researchers는 FRE의 mistakes를 피해야 한다. Incremental work보다 revolutionary ideas를 추구해야 한다. Complexity를 justify해야 한다. Clear positioning을 해야 한다. Thorough하게 execute해야 한다. 이것이 FRE가 남긴 legacy다.

# 4. Reverse Distillation (2022)

## 4.1 Basic Information

Reverse Distillation(RD)은 2022년 Deng 등이 제안한 방법으로, knowledge distillation의 paradigm을 역전시켜 높은 성능을 달성했다. 이 연구는 CVPR 2022에서 발표되었으며, distillation 기반 방법 중 최고 수준의 정확도를 보였다. Reverse Distillation은 "reverse"라는 명칭에서 알 수 있듯이 전통적인 teacher-student 역할을 뒤바꾼 혁신적인 접근이었다.

Reverse Distillation의 핵심 아이디어는 student를 고정하고 teacher를 학습시키는 것이다. 전통적 distillation(STFPM)은 pre-trained teacher를 고정하고 student를 정상 데이터로 학습시킨다. Reverse Distillation은 반대로 pre-trained student를 고정하고 teacher를 정상 데이터로 학습시킨다. 이 역전된 framework가 놀랍게도 더 나은 성능을 제공했다.

방법론적으로 RD는 encoder-decoder teacher를 학습한다. Pre-trained CNN(ResNet)을 student encoder로 사용하고 고정한다. Teacher는 student features를 받아 one-class embedding을 생성하고 다시 student features를 reconstruction한다. 정상 데이터에서만 학습되므로 teacher는 정상 패턴에 특화된다. 테스트 시 이상 샘플에서는 reconstruction이 실패하여 큰 discrepancy를 보인다.

성능 면에서 RD는 당시 distillation 방법 중 최고를 달성했다. MVTec AD에서 이미지 레벨 AUROC 98.6%를 기록했다. 픽셀 레벨에서는 98.7%로 PatchCore(98.6%)를 근소하게 넘어섰다. Texture와 object 카테고리 모두에서 일관되게 높은 성능을 보였다. Localization quality도 매우 우수했다.

RD의 주된 장점은 정확도다. Distillation-based 방법으로서는 최초로 99% 근처에 도달했다. Memory-based 방법(PatchCore 99.1%)과 competitive한 수준이었다. 이는 distillation이 단순한 efficiency tool이 아니라 accuracy에서도 강력함을 입증했다. Paradigm inversion이라는 creative idea가 성공적이었다.

그러나 한계도 분명했다. 추론 속도가 느렸다. 100-150ms로 STFPM(50-100ms)보다 느리고 FastFlow(20-50ms)보다 훨씬 느렸다. Encoder-decoder teacher의 계산 비용이 컸다. 메모리 사용량도 상당했다(300-500MB). Accuracy-speed trade-off에서 accuracy를 선택한 방법이었다.

또한 RD는 conceptual complexity가 있었다. 왜 reverse가 더 나은가? 직관적 설명이 어려웠다. Forward distillation(STFPM)이 자연스러운 접근처럼 보이는데 역전이 왜 효과적인가? 이론적 justification이 충분하지 않았다. Empirical success는 명확했지만 underlying mechanism은 불명확했다.

학술적으로 RD는 significant contribution을 했다. Paradigm inversion이라는 novel idea를 제시했다. High accuracy를 달성하여 distillation의 potential을 확장했다. CVPR acceptance는 community recognition을 받았음을 의미한다. 후속 연구들이 RD의 ideas를 explore했다.

실무적으로 RD는 selective adoption을 받았다. Highest accuracy가 필요하고 computational resources가 충분한 경우에 사용되었다. Critical inspection tasks나 quality control applications이다. Speed가 중요한 경우는 다른 방법(FastFlow, EfficientAD)이 선호되었다. RD는 accuracy-focused niche에서 valuable했다.

## 4.2 Paradigm Inversion

Reverse Distillation의 핵심은 teacher-student 역할의 반전이다. 이는 단순한 engineering trick이 아니라 fundamental rethinking이었다.

### 4.2.1 One-Class Embedding

RD의 central concept는 one-class embedding이다. 정상 데이터의 모든 샘플을 single compact representation으로 mapping한다.

**Conceptual Framework**

전통적 distillation에서 student는 teacher를 모방한다. Teacher는 general visual knowledge를 가진다. Student는 이를 정상 domain에 specialize한다. Student의 limited capacity가 정상에만 fit하도록 강제한다.

RD는 다른 접근을 취한다. Student(pre-trained CNN)는 general representation을 제공한다. Teacher는 이러한 general representations를 one-class embedding으로 compress한다. 모든 정상 샘플이 similar embeddings를 가지도록 한다. Teacher는 compression과 reconstruction을 학습한다.

One-class embedding의 idea는 hypersphere learning과 유사하다. Deep SVDD나 DROCC 같은 방법들이 모든 정상 샘플을 hypersphere 내부로 map한다. Center에서의 거리가 anomaly score다. RD는 이를 distillation framework에 integrate했다.

수학적으로 one-class embedding $\mathbf{z}$는 다음 속성을 만족해야 한다:

$$\forall \mathbf{x}_i, \mathbf{x}_j \in \mathcal{D}_{\text{normal}}: \| E(\mathbf{x}_i) - E(\mathbf{x}_j) \| < \epsilon$$

여기서 $E$는 embedding function이고 $\epsilon$은 small constant다. 모든 정상 샘플의 embeddings가 서로 가깝다. 반면 anomalies는 멀리 떨어진다.

$$\mathbf{x}_{\text{anomaly}} \notin \mathcal{D}_{\text{normal}}: \| E(\mathbf{x}_{\text{anomaly}}) - \mathbb{E}_{\mathbf{x} \in \mathcal{D}}[E(\mathbf{x})] \| \gg \epsilon$$

이는 one-class learning의 본질이다. 정상의 tight cluster를 형성하고 이상은 outliers가 된다.

**Implementation Details**

RD에서 one-class embedding은 encoder-bottleneck-decoder 구조로 구현된다. Student features $\mathbf{F}^S$를 teacher encoder $E_T$가 받아 bottleneck embedding $\mathbf{z}$를 생성한다.

$$\mathbf{z} = E_T(\mathbf{F}^S)$$

Bottleneck dimension은 student features보다 훨씬 작다. 예를 들어 student features가 256×16×16=65K dimensions이면, bottleneck은 512-1024 dimensions다. 60-120배 compression이다.

이러한 extreme compression이 one-class learning을 강제한다. Bottleneck이 작으므로 모든 정상 variations를 preserve할 수 없다. Teacher는 common patterns만 encode해야 한다. Anomalies는 encode되지 못한다.

Bottleneck의 regularization도 중요하다. Simple L2 regularization을 추가한다:

$$\mathcal{L}_{\text{reg}} = \| \mathbf{z} \|^2$$

이는 embeddings를 origin 근처로 push한다. Compact representation을 encourage한다. Overfitting을 방지한다.

또는 contrastive loss를 추가할 수 있다. Positive pairs(정상 샘플들)는 가깝게, negative pairs(augmented versions)는 적당히 떨어지게 한다. 이는 embedding space의 구조를 improve한다.

**Advantages over Direct Matching**

One-class embedding이 direct feature matching보다 왜 나은가? 첫째, explicit compression constraint다. Bottleneck이 information bottleneck theory의 원리를 구현한다. Minimal sufficient statistics만 preserve한다. 이는 robust representation을 만든다.

둘째, hierarchical processing이다. Student features → embedding → reconstructed features의 two-stage process다. 각 stage가 다른 abstraction level을 다룬다. Richer representation learning이 가능하다.

셋째, flexibility다. Embedding space를 다양한 방식으로 regularize하거나 structure할 수 있다. Metric learning, contrastive learning, hypersphere constraints 등을 integrate할 수 있다. Direct matching은 이러한 flexibility가 제한적이다.

실험적으로 one-class embedding의 효과를 검증했다. Bottleneck 없이(direct reconstruction) 96.8% AUROC였다. Bottleneck with 98.6%로 1.8%포인트 향상이었다. Compression constraint가 critical했다.

### 4.2.2 Encoder-Decoder Structure

RD의 teacher는 encoder-decoder architecture를 가진다. 이는 autoencoder와 유사하지만 중요한 차이가 있다.

**Architecture Design**

Teacher encoder는 여러 convolutional layers로 student features를 progressively compress한다. Spatial dimensions를 줄이고 abstraction을 높인다.

```
Teacher Encoder:
Student Features (256×16×16)
  → Conv+BN+ReLU (512×8×8)
  → Conv+BN+ReLU (1024×4×4)
  → Conv+BN+ReLU (2048×2×2)
  → Bottleneck (1024 vector)
```

Bottleneck은 fully connected layer나 global pooling으로 구현된다. Spatial information을 완전히 collapse하여 compact vector를 만든다. 이것이 one-class embedding이다.

Teacher decoder는 symmetric하게 bottleneck에서 student features를 reconstruct한다. Transposed convolutions나 upsampling으로 spatial dimensions를 restore한다.

```
Teacher Decoder:
Bottleneck (1024 vector)
  → Reshape (2048×2×2)
  → TransConv+BN+ReLU (1024×4×4)
  → TransConv+BN+ReLU (512×8×8)
  → TransConv+BN+ReLU (256×16×16)
  → Reconstructed Features
```

최종 output dimension은 student features와 정확히 match한다. Pixel-wise comparison이 가능하도록 한다.

**Multi-scale Processing**

RD는 여러 student layers에서 features를 추출한다. Layer2, layer3, layer4로부터 서로 다른 resolution과 semantic level을 얻는다. 각 level에 independent한 encoder-decoder를 적용한다.

각 scale의 teacher가 해당 student features를 reconstruct한다. Three parallel branches가 있다. 학습 중 각 branch는 independently optimize된다. 추론 시 세 branches의 reconstruction errors를 aggregate한다.

Multi-scale의 rationale은 STFPM과 유사하다. Different sizes의 anomalies를 comprehensive하게 detect한다. Low-level features가 small defects를, high-level features가 large structural anomalies를 포착한다.

**Skip Connections Consideration**

U-Net style skip connections를 추가할지 고려할 수 있다. Encoder의 intermediate features를 decoder에 직접 전달한다. Low-level details를 preserve하는 데 도움이 된다.

그러나 skip connections는 bottleneck constraint를 약화시킨다. Information이 bottleneck을 bypass할 수 있다. One-class learning의 효과가 감소한다. RD는 skip connections를 사용하지 않는다.

대신 feature pyramid를 활용한다. Multi-scale processing이 implicit하게 multi-resolution information을 capture한다. Skip connections 없이도 충분한 detail preservation이 가능하다.

일부 variants는 partial skip connections를 시도했다. 가장 low-level features만 skip한다. 이는 compromise다. Bottleneck constraint를 유지하면서 detail을 일부 preserve한다. 결과는 mixed였다. 일부 카테고리에서 0.5%포인트 향상, 일부에서는 변화 없음.

**Comparison with Autoencoders**

RD의 teacher는 autoencoder와 유사하지만 critical differences가 있다. 첫째, input이 다르다. Autoencoder는 raw images를 받지만 RD teacher는 student features를 받는다. Pre-processed, semantic-rich inputs다.

둘째, training objective가 다르다. Autoencoder는 image reconstruction error를 minimize한다. RD teacher는 student feature reconstruction error를 minimize한다. Feature space에서의 learning이다.

셋째, architecture가 더 specialized하다. RD teacher는 student features의 특성에 맞춰 설계된다. Dimensionality, resolution, semantic level을 고려한다. Generic autoencoder보다 task-specific하다.

실험적 비교를 수행했다. Standard autoencoder(image input/output): 92.5% AUROC. Feature autoencoder(student features input/output, no distillation): 95.2% AUROC. RD(full framework): 98.6% AUROC. Feature space learning과 distillation의 synergy가 critical했다.

### 4.2.3 Domain-Specific Teacher

RD의 teacher는 정상 데이터로 학습되어 domain-specific knowledge를 acquire한다. 이것이 forward distillation과의 key difference다.

**Training Dynamics**

Forward distillation(STFPM)에서 teacher는 ImageNet에서 학습되고 frozen이다. General visual knowledge를 제공하지만 target domain에 대해 모른다. Student가 domain adaptation을 담당한다.

RD에서 teacher는 target domain(정상 제품)에서 학습된다. Domain-specific patterns를 directly 학습한다. Student는 frozen되어 general representation을 제공한다. Role distribution이 reverse되었다.

이 inversion의 효과는 무엇인가? Domain-specific teacher가 정상의 nuances를 더 잘 포착할 수 있다. 제품의 specific textures, shapes, variations를 학습한다. Generic teacher보다 precise하다.

학습 과정에서 teacher는 점점 정상에 specialized된다. 초기에는 student features를 잘 reconstruct하지 못한다. Random initialization이기 때문이다. Epochs가 진행되며 reconstruction quality가 향상된다. 정상 patterns를 encode하기 시작한다.

수백 epochs 후 teacher는 정상 샘플에 대해 거의 perfect reconstruction을 달성한다. Student-teacher discrepancy가 매우 작아진다. 그러나 anomalies에 대해서는 여전히 크다. Training distribution 밖이기 때문이다.

**Specialization vs Generalization**

Domain-specific teacher는 specialization-generalization trade-off를 가진다. 장점은 target domain에 highly optimized된다는 것이다. 정상의 subtle details를 포착한다. 높은 정확도를 제공한다.

단점은 generalization이 제한적이다. 새로운 카테고리나 변동에 adapt하기 어렵다. 각 카테고리마다 separate teacher를 학습해야 한다. Transfer learning이 어렵다.

Forward distillation은 반대다. General teacher가 multiple domains에 transfer 가능하다. 한 번 학습된 teacher를 여러 categories에 재사용할 수 있다. 그러나 each category에 대해 less optimized하다.

실무적으로 이는 trade-off 결정이다. Many categories(수십-수백 개)를 다룬다면 forward distillation이 efficient하다. Single teacher를 공유한다. Few categories(수 개)에 집중한다면 RD가 optimal하다. 각각을 highly optimize한다.

**Teacher Capacity and Performance**

RD에서 teacher size가 performance에 critical하다. Larger teacher가 더 복잡한 정상 patterns를 학습할 수 있다. 그러나 overfitting 위험도 증가한다.

Small teacher(1-2M parameters): Underfit할 수 있다. 복잡한 categories(object)에서 성능 저하. 97.5% AUROC 정도.

Medium teacher(5-10M parameters): Balanced. 대부분의 categories에 적합. 98.6% AUROC. RD의 default.

Large teacher(20M+ parameters): Marginal improvement. 98.8% AUROC로 0.2%포인트 향상. 그러나 training time 2배, inference 1.5배 느림. Cost-benefit이 불리.

Optimal teacher size는 data complexity에 의존한다. Simple textures(carpet, grid)는 small teacher로 충분. Complex objects(screw, cable)는 larger teacher가 beneficial. Per-category tuning이 ideal하지만 practical하지 않다.

**Comparison with Pre-trained Teachers**

Pre-trained teacher(forward distillation)와 domain-trained teacher(RD)를 직접 비교했다. 동일한 architecture(ResNet18)를 사용하되 하나는 ImageNet에서, 하나는 정상 데이터에서 학습했다.

Forward distillation (pre-trained teacher): 95.5% AUROC
Reverse distillation (domain-trained teacher): 98.6% AUROC

3.1%포인트 차이는 significant하다. Domain-specific training의 value를 입증한다. 그러나 training cost도 고려해야 한다. RD는 각 category마다 teacher를 학습해야 한다(1-3 hours). Forward는 한 번만 학습하거나 아예 학습 불필요하다(pre-trained 사용).

Hybrid approach도 가능하다. Pre-trained teacher로 initialize하고 domain data로 fine-tune한다. 이는 middle ground다. RD보다 빠르고 forward보다 정확할 수 있다. 일부 implementations가 이를 시도했다. Results는 promising했다(98.2% AUROC, 0.5-1 hour training).

## 4.3 Technical Innovation

RD의 technical innovations는 high performance의 key enablers였다.

**Coupling Reconstruction Loss**

RD는 단순한 L2 reconstruction loss를 넘어선다. Cosine similarity loss를 함께 사용한다. 두 losses의 combination이 효과적이었다.

L2 loss는 magnitude differences를 측정한다:

$$\mathcal{L}_{\text{L2}} = \| \mathbf{F}^S - \hat{\mathbf{F}}^S \|^2$$

Cosine loss는 direction differences를 측정한다:

$$\mathcal{L}_{\text{cos}} = 1 - \frac{\mathbf{F}^S \cdot \hat{\mathbf{F}}^S}{\| \mathbf{F}^S \| \| \hat{\mathbf{F}}^S \|}$$

Combined loss는:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{L2}} + \lambda_2 \mathcal{L}_{\text{cos}}$$

일반적으로 $\lambda_1 = 1.0$, $\lambda_2 = 0.5$를 사용한다. L2가 primary이고 cosine이 auxiliary다.

왜 combination이 better인가? L2는 absolute errors를 penalize한다. Large deviations를 강하게 suppress한다. Cosine은 relative orientations를 align한다. Feature directions를 match시킨다. 두 aspects가 complementary하다.

실험적으로 L2 only: 97.8% AUROC. Cosine only: 97.2% AUROC. Combined: 98.6% AUROC. Combination이 0.8%포인트 향상을 제공했다. Synergy가 명확했다.

**Feature Normalization**

RD는 careful feature normalization을 수행한다. Student features를 normalize한 후 teacher에 입력한다. Reconstructed features도 denormalize한다. 이는 training stability를 크게 향상시켰다.

Normalization method는 batch normalization이다. 각 channel을 independently normalize한다:

$$\hat{\mathbf{F}}^S_c = \frac{\mathbf{F}^S_c - \mu_c}{\sigma_c}$$

여기서 $\mu_c$와 $\sigma_c$는 channel $c$의 mean과 std다. Batch statistics를 사용한다. Training 중 running mean/std를 maintain한다. Inference 시 이들을 사용한다.

Normalization이 왜 중요한가? Student features의 scale이 layer마다 다르다. Layer2는 작은 values, layer4는 큰 values를 가질 수 있다. Normalization이 이를 균일하게 만든다. Teacher training이 안정적이다.

또한 normalization은 implicit regularization을 제공한다. Outlier features가 suppress된다. Robust learning이 촉진된다. Overfitting이 감소한다.

Without normalization: Training이 불안정했다. Loss가 oscillate했다. Final AUROC 96.2%. With normalization: Smooth convergence. Stable training. 98.6% AUROC. 2.4%포인트 차이는 dramatic하다.

**Anomaly Map Generation**

RD는 high-quality anomaly maps를 생성한다. Multi-scale reconstruction errors를 carefully aggregate한다.

각 scale $l$에서 reconstruction error map을 계산한다:

$$\mathbf{E}_l = \| \mathbf{F}^S_l - \hat{\mathbf{F}}^S_l \|^2$$

이는 channel dimension을 따라 averaging한다. Spatial map of errors다. 서로 다른 resolutions를 가진다.

모든 error maps를 highest resolution으로 upsample한다. Bilinear interpolation을 사용한다. Aligned maps $\mathbf{E}^{\text{up}}_1, \mathbf{E}^{\text{up}}_2, \mathbf{E}^{\text{up}}_3$를 얻는다.

Simple averaging 대신 weighted fusion을 수행한다:

$$\mathbf{A} = \sum_{l=1}^{3} w_l \mathbf{E}^{\text{up}}_l$$

Weights $w_l$은 empirically 결정된다. 일반적으로 $w_1=0.3$, $w_2=0.4$, $w_3=0.3$이다. Mid-level(layer2)에 highest weight를 준다. 가장 informative하기 때문이다.

Gaussian smoothing을 anomaly map에 적용한다. Local consistency를 enforce한다. Noisy artifacts를 제거한다. Kernel size는 5×5 또는 7×7을 사용한다.

Final anomaly map을 input image size로 resize한다. Bilinear interpolation으로 256×256이나 original size로 만든다. 이것이 pixel-level prediction이다.

**Thresholding Strategy**

RD는 adaptive thresholding을 제안한다. Fixed threshold 대신 data-driven approach를 사용한다.

Training set의 모든 정상 샘플에 대해 anomaly scores를 계산한다. Score distribution의 statistics를 추출한다. Mean $\mu$ and std $\sigma$를 계산한다.

Threshold를 $\tau = \mu + k\sigma$로 설정한다. $k$는 sensitivity parameter다. $k=2$는 95% 정상을 accept한다. $k=3$는 99.7%를 accept한다. Application requirements에 따라 선택한다.

Per-category thresholding이 권장된다. 각 category가 다른 정상 score distribution을 가진다. Global threshold는 suboptimal하다. Category-specific tuning이 0.5-1%포인트 향상을 제공한다.

Alternative로 percentile-based thresholding을 사용할 수 있다. 95th 또는 99th percentile을 threshold로 선택한다. 이는 distribution shape에 robust하다. Non-Gaussian distributions에도 잘 작동한다.

## 4.4 SOTA Performance (98.6%)

RD는 발표 당시 distillation-based methods 중 최고 성능을 달성했다.

**Benchmark Results**

MVTec AD에서 RD의 성능:
- Image-level AUROC: 98.6%
- Pixel-level AUROC: 98.7%

이는 모든 distillation methods를 능가했다. STFPM(95.5%)보다 3.1%포인트, FRE(96.5%)보다 2.1%포인트 높았다. Memory-based PatchCore(99.1%)와는 0.5%포인트 차이로 매우 근접했다.

Pixel-level에서 RD는 PatchCore(98.6%)를 0.1%포인트 앞섰다. Localization quality가 exceptionally 높았다. 결함 위치를 매우 정확히 특정했다.

**Category-wise Excellence**

RD는 거의 모든 카테고리에서 우수했다. 15개 중 13개에서 98% 이상을 달성했다.

Texture categories에서 near-perfect:
- Carpet: 99.2%
- Grid: 99.9%
- Leather: 100%
- Tile: 99.6%
- Wood: 99.8%

Object categories에서도 강력:
- Bottle: 99.8%
- Cable: 98.5%
- Capsule: 98.2%
- Metal_nut: 98.1%
- Pill: 97.8%

가장 어려운 categories(screw 97.3%, toothbrush 97.5%)에서도 competitive했다. 모든 방법이 struggle하는 cases였다.

**Consistency Across Settings**

RD는 다양한 settings에서 일관되게 높았다. Different random seeds, data splits, hyperparameters로 robust했다. Standard deviation이 0.2-0.3%포인트로 작았다.

이는 method reliability를 시사한다. Lucky한 single run이 아니라 consistently strong performance다. Reproducibility가 높다. 실무 배포에서 중요한 특성이다.

Cross-dataset evaluation도 수행했다. MVTec 외에 BTAD, VisA datasets에서 test했다. RD는 여전히 top-tier였다. Generalization capability가 good했다.

**Ablation Study Insights**

각 component의 기여도를 분석했다:

- Baseline (forward distillation, STFPM-style): 95.5%
- + Reverse paradigm: 97.2% (+1.7%p)
- + One-class embedding: 98.1% (+0.9%p)
- + Coupled losses: 98.4% (+0.3%p)
- + Feature normalization: 98.6% (+0.2%p)

Reverse paradigm 자체가 가장 큰 기여를 했다(1.7%p). One-class embedding이 두 번째(0.9%p). 나머지 components도 meaningful하게 기여했다.

Teacher architecture variants도 test했다:
- Shallow encoder-decoder (3 layers): 97.8%
- Medium (5 layers, default): 98.6%
- Deep (8 layers): 98.7%

Deep architecture가 0.1%p 향상을 제공했지만 2배 느렸다. Medium이 best balance였다.

**Comparison with SOTA**

당시(2022) state-of-the-art와 비교:

- PatchCore (memory-based): 99.1% (+0.5%p over RD)
- FastFlow (flow-based): 98.5% (-0.1%p)
- PaDiM (memory-based): 97.5% (-1.1%p)
- STFPM (distillation): 95.5% (-3.1%p)

RD는 PatchCore 다음으로 2위였다. 그러나 gap이 매우 작았다(0.5%p). Practical difference가 minimal했다. 일부 categories에서 RD가 PatchCore를 능가했다.

FastFlow와는 거의 동등했다(0.1%p 차이). 두 paradigms(distillation vs flow)가 similar peaks에 도달했다. 이는 흥미로운 convergence였다.

## 4.5 Pixel-Level Excellence

RD의 특별한 강점은 pixel-level performance였다. Localization quality가 exceptionally 높았다.

**Segmentation Accuracy**

Pixel-level AUROC 98.7%는 당시 최고였다. PatchCore(98.6%)를 근소하게 넘어섰다. 다른 모든 방법을 크게 앞섰다. FastFlow(98.6%), PaDiM(97.1%), STFPM(97.1%)보다 우수했다.

Per-pixel IoU(Intersection over Union)도 계산했다. Threshold를 적용하여 binary masks를 생성했다. Ground truth와 비교했다. RD는 평균 IoU 0.65를 달성했다. PatchCore 0.63, FastFlow 0.62였다.

Dice coefficient도 평가했다. RD: 0.71. PatchCore: 0.69. FastFlow: 0.68. Consistent superiority를 보였다.

**Boundary Precision**

결함 경계의 정확도를 분석했다. RD의 anomaly maps는 sharp boundaries를 가졌다. Blurry하거나 diffuse하지 않았다.

Boundary F1 score를 계산했다. Predicted와 ground truth boundaries 간의 precision과 recall이다. RD: 0.58. PatchCore: 0.54. FastFlow: 0.52.

이는 RD의 multi-scale reconstruction과 careful aggregation 덕분이었다. Gaussian smoothing이 적절히 조절되었다. Boundaries를 preserve하면서 noise를 제거했다.

실제 anomaly maps를 시각적으로 비교했다. RD의 maps가 가장 clean하고 precise했다. False positives가 적었다. True defects가 명확히 highlighted되었다.

**Small Defect Detection**

Small defects(전체 이미지의 1% 미만)는 모든 방법에 challenging하다. RD는 이에서도 우수했다.

Small defects만 포함하는 subset으로 평가했다. RD: 95.2% AUROC. PatchCore: 94.8%. FastFlow: 93.5%. RD가 small defects에 특히 강력했다.

이는 multi-scale processing과 high-resolution features(layer1, layer2) 덕분이었다. Fine-grained details를 잘 포착했다. Low-level reconstruction errors가 small anomalies를 reveal했다.

Minimum detectable defect size를 분석했다. RD: 0.3% of image area. PatchCore: 0.5%. FastFlow: 0.8%. RD가 가장 sensitive했다.

**False Positive Analysis**

False positives는 정상을 이상으로 오판하는 것이다. 실무에서 costly하다. RD의 false positive rate를 분석했다.

Fixed threshold(95% recall)에서 precision을 측정했다. RD: 0.92. PatchCore: 0.90. FastFlow: 0.88. RD가 가장 높은 precision을 보였다.

False positives가 주로 어디서 발생하는가? High-variance regions이었다. 조명 변화, 반사, 정상 변동이 큰 영역이었다. 모든 methods가 여기서 struggle했다.

RD는 이러한 regions에서 상대적으로 robust했다. Domain-specific teacher가 정상 variations를 better 학습했기 때문이다. Normal variability와 true anomalies를 better distinguish했다.

**Practical Localization Value**

Pixel-level excellence가 실무에서 왜 중요한가? 첫째, root cause analysis다. 정확한 defect location이 문제 원인 파악에 도움이 된다. 공정 개선에 필수적이다.

둘째, repair guidance다. 결함 위치를 알면 targeted repair가 가능하다. 전체를 폐기하지 않고 부분 수리로 비용을 절감한다.

셋째, quality grading다. 결함의 크기와 위치로 제품을 등급화한다. 완전 불량과 minor defects를 구별한다. Flexible quality control이 가능하다.

RD의 high-quality anomaly maps는 이러한 applications에 직접 사용 가능하다. Post-processing이 minimal하다. Operators가 쉽게 이해하고 활용할 수 있다.

## 4.6 Trade-offs (Speed vs Accuracy)

RD는 accuracy를 우선시하여 speed를 희생했다. 이 trade-off를 이해하는 것이 중요하다.

**Inference Time Breakdown**

RD의 추론 시간을 component별로 분해:

- Student (frozen ResNet18) forward: 15-20ms
- Teacher encoder (3 scales): 25-35ms
- Bottleneck processing: 5-10ms
- Teacher decoder (3 scales): 40-60ms
- Reconstruction error & aggregation: 10-15ms
- Total: 95-140ms (typical 100-150ms)

Teacher decoder가 가장 큰 bottleneck이다(40-60ms). Upsampling과 transposed convolutions이 expensive하다. Multi-scale processing이 비용을 더 증가시킨다.

STFPM(50-100ms)보다 1.5-2배 느리다. FastFlow(20-50ms)보다 3-5배 느리다. EfficientAD(1-5ms)보다는 20-100배 느리다.

**Speed Optimization Attempts**

RD 저자들도 speed optimization을 시도했다. 그러나 architectural constraints로 제한적이었다.

Single-scale processing: Layer2만 사용하면 50-70ms로 줄어든다. 그러나 AUROC가 96.8%로 1.8%p 떨어진다. Trade-off가 불리했다.

Smaller teacher: Encoder-decoder를 shallow하게 만들면 70-100ms다. AUROC 97.8%로 0.8%p 감소. 여전히 FastFlow만큼 빠르지 않았다.

Quantization (INT8): 모델 크기 1/4, 속도 1.3배 향상. 그러나 AUROC 97.9%로 0.7%p 저하. Reconstruction quality가 precision에 sensitive했다.

Knowledge distillation from RD: Large RD teacher를 small student에 distill. 30-50ms 달성, 97.5% AUROC. 이는 promising했지만 RD 본연의 성능에 못 미쳤다.

**Memory Usage**

RD의 메모리 footprint도 상당하다:

- Student (frozen): 50MB (공유 가능)
- Teacher (3 scales × encoder-decoder): 250-400MB
- Intermediate activations (inference): 100-150MB
- Total: 400-600MB per category

STFPM(100-200MB)보다 2-3배 많다. FastFlow(200-500MB)와 비슷하다. PatchCore(수백 MB, data dependent)와 comparable하다.

Multi-category deployment 시 메모리가 선형 증가한다. 10 categories면 4-6GB (student 공유 시). 100 categories면 25-40GB. Server-grade memory가 필요하다.

**Accuracy-Speed Pareto Front**

다양한 methods를 accuracy-speed plot에 배치하면 Pareto front가 드러난다:

```
Accuracy (AUROC %)
99 |                    ● PatchCore
98 |         ● FastFlow   ● RD
97 |  ● STFPM
96 |
95 | ● Autoencoder
   +-------------------------
   0   50  100  150  200+  Speed (ms)
```

RD는 high-accuracy, medium-speed region에 있다. PatchCore만큼 정확하지만 약간 느리다. FastFlow만큼 빠르지 않지만 약간 더 정확하다.

Pareto-optimal points는: EfficientAD (speed champion), FastFlow (balanced), RD (accuracy focused), PatchCore (accuracy champion). 각각 다른 use cases에 적합하다.

**When to Choose RD**

RD를 선택해야 하는 경우:

1. **Highest accuracy priority**: 0.5-1%p 차이도 중요한 critical applications. Medical devices, safety components, high-value products.

2. **Pixel-level precision**: 정확한 defect localization이 필수. Repair guidance, root cause analysis, quality grading.

3. **Computational resources available**: Server deployment with ample GPU/memory. Throughput보다 accuracy가 중요.

4. **Few categories**: 수 개의 categories만 다룬다. 각각을 highly optimize할 수 있다.

5. **Offline processing**: Real-time이 필수가 아닌 batch processing. Post-inspection analysis.

**When NOT to Choose RD**

RD를 피해야 하는 경우:

1. **Real-time requirement**: 초당 30+ frames 필요. High-speed production lines, interactive systems.

2. **Edge deployment**: Limited computational resources. Jetson, mobile devices, embedded systems.

3. **Many categories**: 수십-수백 개 categories. Category별 teacher training이 impractical.

4. **Cost sensitivity**: Training time과 computational cost가 중요. Rapid prototyping, frequent updates.

5. **Sufficient accuracy from faster methods**: FastFlow(98.5%)로 충분한 경우. 추가 0.1-0.5%p가 cost를 justify하지 못함.

**Future Optimization Potential**

RD의 speed는 개선 여지가 있다. Architectural innovations로 decoder를 efficient하게 만들 수 있다. Depthwise separable convolutions, inverted residuals 같은 techniques를 적용한다.

Neural architecture search로 optimal teacher structure를 찾을 수 있다. Speed와 accuracy를 jointly optimize한다. Automated approach가 manual design보다 나을 수 있다.

또는 hybrid approach를 고려한다. RD를 teacher로 사용하여 fast student를 distill한다. Two-stage distillation이다. Final student는 빠르고 RD의 knowledge를 inherit한다.

Hardware acceleration도 중요하다. Custom CUDA kernels로 decoder operations를 optimize한다. TensorRT, ONNX Runtime 같은 inference engines를 fully leverage한다. 2-3배 speedup이 가능할 수 있다.

**Conclusion on Trade-offs**

RD는 accuracy-speed trade-off spectrum에서 accuracy end를 선택했다. 이는 deliberate design choice였다. Highest possible accuracy를 목표로 삼았다. Speed는 secondary consideration이었다.

이 choice가 모든 applications에 적합한 것은 아니다. 그러나 specific niche에서 valuable하다. Critical quality control, precision manufacturing, high-stake inspections 등이다. 이러한 domains에서 RD의 exceptional accuracy가 speed cost를 justify한다.

Trade-off를 이해하고 application requirements에 맞춰 선택하는 것이 중요하다. One size does not fit all. RD, FastFlow, EfficientAD 등 각각 다른 points on the Pareto front를 occupy한다. Right tool for the right job이 핵심이다.

# 5. EfficientAD (2024)

## 5.1 Basic Information

EfficientAD는 2024년 Batzner 등이 제안한 방법으로, knowledge distillation 기반 이상 탐지에 real-time revolution을 가져왔다. 이 연구는 WACV 2024에서 발표되었으며, 극도의 효율성과 합리적인 정확도를 결합하여 industrial deployment의 새로운 기준을 세웠다. EfficientAD는 이름 그대로 efficiency를 핵심 목표로 삼았고 이를 달성했다.

EfficientAD의 핵심 혁신은 extremely lightweight student network다. Patch Description Network(PDN)라 불리는 tiny model이 teacher features를 모방한다. PDN은 단 50K parameters로 구성되어 있다. 이는 STFPM의 ResNet18 student(11.7M)의 1/200 크기다. 이러한 극단적 경량화가 breakthrough speed를 가능하게 했다.

방법론적으로 EfficientAD는 두 가지 components를 결합한다. 첫째, distillation-based PDN이 teacher(EfficientNet-B4) features를 학습한다. 둘째, small autoencoder가 image reconstruction을 수행한다. 두 components의 anomaly scores를 fusion하여 최종 판정을 내린다. 이 dual-path approach가 robustness를 제공했다.

성능 면에서 EfficientAD는 놀라운 결과를 보였다. MVTec AD에서 이미지 레벨 AUROC 97.8%를 달성했다. 이는 RD(98.6%)보다 0.8%포인트 낮지만 여전히 매우 높다. 가장 중요한 것은 추론 속도다. 1-5ms로 STFPM(50-100ms)의 10-50배, RD(100-150ms)의 20-150배 빠르다. GPU에서 초당 200-1000 프레임을 처리한다.

더욱 놀라운 것은 CPU performance다. EfficientAD는 CPU에서도 10-30ms로 실행된다. 이는 GPU 없이도 real-time processing이 가능함을 의미한다. 다른 방법들은 CPU에서 수백 ms에서 초 단위로 느리다. EfficientAD의 CPU capability는 edge deployment의 문을 열었다.

EfficientAD의 장점은 deployment practicality다. Extremely small model size(1-5MB)로 메모리가 거의 필요 없다. 수백 개 카테고리를 단일 디바이스에 배포할 수 있다. 추론이 매우 빠르고 에너지 효율적이다. Raspberry Pi, Jetson Nano 같은 edge devices에서도 잘 작동한다. Industrial IoT와 mobile inspection에 ideal하다.

한계도 명확하다. 정확도가 top-tier에는 못 미친다. PatchCore(99.1%)보다 1.3%포인트, RD(98.6%)보다 0.8%포인트 낮다. Subtle defects나 complex textures에서 약점을 보인다. Extreme optimization이 일부 표현력을 희생했다. Critical accuracy applications에는 부적합할 수 있다.

학술적으로 EfficientAD는 paradigm shift를 촉발했다. "Efficiency와 accuracy는 trade-off"라는 기존 믿음에 도전했다. 적절한 design choices로 both를 reasonable level에서 달성할 수 있음을 보였다. 후속 연구들이 efficient anomaly detection에 더 집중하게 만들었다.

실무적으로 EfficientAD는 광범위하게 채택되고 있다. High-speed production lines, edge devices, mobile quality control 등에서 활용된다. Cost-performance ratio가 exceptional하다. Accessibility도 높다. 공식 코드가 잘 정리되어 있고 documentation이 충분하다. Implementation barrier가 낮다.

## 5.2 Real-time Revolution

EfficientAD의 가장 큰 기여는 real-time processing을 현실화한 것이다. 이는 incremental improvement가 아니라 qualitative breakthrough였다.

### 5.2.1 Patch Description Network (PDN)

PDN은 EfficientAD의 핵심 innovation이다. Tiny convolutional network로 teacher features를 describe한다.

**Architecture Specification**

PDN은 remarkably simple하다. 4개의 convolutional layers로 구성된다. 각 layer는 3×3 kernels과 small channel counts를 가진다.

```
PDN Architecture:
Input: Teacher features (384 channels, 56×56)
  → Conv 3×3, 128 channels, stride 1, padding 1
  → ReLU
  → Conv 3×3, 256 channels, stride 1, padding 1
  → ReLU
  → Conv 3×3, 256 channels, stride 1, padding 1
  → ReLU
  → Conv 3×3, 384 channels, stride 1, padding 1
Output: Student features (384 channels, 56×56)
```

Total parameters: 약 50K. 이는 극도로 작다. Modern smartphones의 single frame이 수십 MB인 것과 비교하면 PDN은 0.2MB에 불과하다. Memory footprint가 negligible하다.

No pooling or downsampling이 사용된다. Spatial resolution이 보존된다. 이는 pixel-level localization에 중요하다. 대부분의 networks가 pooling으로 resolution을 줄이는 것과 대조적이다.

No batch normalization도 사용된다. BN은 inference에서 추가 computation을 요구한다. Statistics를 maintain하고 apply해야 한다. PDN은 이를 제거하여 efficiency를 높였다. Simple ReLU activations만 사용한다.

**Design Rationale**

왜 이렇게 작은 network가 작동하는가? 핵심은 task simplification이다. PDN은 raw images를 처리하지 않는다. Pre-computed teacher features를 받는다. 이들은 이미 high-level semantic information을 담고 있다. PDN은 단순히 이들을 "describe"하면 된다. 복잡한 feature extraction이 필요 없다.

Teacher features의 quality가 critical하다. EfficientAD는 EfficientNet-B4를 teacher로 사용한다. ImageNet에서 사전 학습되어 강력한 representations를 제공한다. PDN은 이러한 rich features를 정상 데이터에 adapt하기만 하면 된다.

Small capacity는 implicit regularization을 제공한다. Overfitting이 불가능하다. 50K parameters로는 모든 training data를 memorize할 수 없다. Essential patterns만 학습한다. 이는 generalization을 돕는다.

Spatial resolution preservation도 중요하다. Anomalies는 종종 localized되어 있다. Pooling이 이를 dilute할 수 있다. Full resolution을 유지하면 small defects도 잘 탐지된다.

**Training Efficiency**

PDN의 작은 크기는 training도 빠르게 만든다. Single GPU에서 수 분 안에 수렴한다. 10-30 epochs면 충분하다. Total training time은 5-15분이다. 이는 RD(1-3 hours)나 STFPM(30-60 minutes)보다 훨씬 빠르다.

Gradient computation이 minimal하다. Backpropagation이 4 layers만 통과한다. Memory와 computation이 매우 적다. Large batch sizes를 사용할 수 있다(64-128). 이는 training stability를 높인다.

Learning rate도 높게 설정할 수 있다. $10^{-2}$ - $10^{-3}$이 typical하다. Small network가 large learning rates에 robust하다. Fast convergence를 달성한다.

Hyperparameter tuning도 빠르다. Single experiment가 수 분이므로 many configurations를 시도할 수 있다. Grid search나 random search가 practical하다. Optimal settings를 찾기 쉽다.

**Comparison with Traditional Students**

PDN과 traditional students(ResNet18 등)를 비교하면:

| Aspect | ResNet18 Student | PDN |
|--------|------------------|-----|
| Parameters | 11.7M | 50K |
| Size | 45MB | 0.2MB |
| Inference (GPU) | 10-20ms | 0.5-2ms |
| Inference (CPU) | 100-300ms | 5-15ms |
| Training time | 30-60min | 5-15min |
| AUROC | 95-96% | 97-98% |

PDN이 200배 작으면서도 2-3%포인트 더 높은 accuracy를 보인다. 이는 counterintuitive하다. Bigger is not always better의 명확한 증거다.

왜 PDN이 더 나은가? Task-specific design이 key다. General-purpose ResNet은 many tasks를 위해 설계되었다. Over-parameterized될 수 있다. PDN은 single task(describe teacher features on normal data)에 최적화되었다. Precisely right capacity를 가진다.

### 5.2.2 Autoencoder Integration

EfficientAD는 PDN만 사용하지 않는다. Small autoencoder를 추가하여 dual-path approach를 구현한다.

**Autoencoder Architecture**

EfficientAD의 autoencoder도 extremely small하다. Encoder는 4 convolutional layers, decoder는 4 transposed convolutional layers를 가진다.

```
Encoder:
Input: Image (256×256×3)
  → Conv 3×3, 32 channels, stride 2 (128×128)
  → ReLU
  → Conv 3×3, 32 channels, stride 2 (64×64)
  → ReLU
  → Conv 3×3, 64 channels, stride 2 (32×32)
  → ReLU
  → Conv 3×3, 64 channels, stride 2 (16×16)
  → Bottleneck

Decoder:
Bottleneck
  → TransConv 3×3, 64 channels, stride 2 (32×32)
  → ReLU
  → TransConv 3×3, 64 channels, stride 2 (64×64)
  → ReLU
  → TransConv 3×3, 32 channels, stride 2 (128×128)
  → ReLU
  → TransConv 3×3, 3 channels, stride 2 (256×256)
  → Sigmoid
```

Total parameters: 약 100K. PDN과 합쳐도 150K에 불과하다. Entire model이 1MB 미만이다.

Bottleneck dimension은 매우 작다. $16 \times 16 \times 64 = 16K$. Input image가 $256 \times 256 \times 3 = 196K$이므로 약 12배 compression이다. Strong information bottleneck을 형성한다.

**Role of Autoencoder**

왜 autoencoder를 추가하는가? PDN만으로 충분하지 않은가? Autoencoder는 complementary information을 제공한다.

PDN은 high-level semantic features를 다룬다. Teacher features space에서 작동한다. Structural anomalies와 pattern disruptions를 잘 탐지한다. 그러나 low-level appearance anomalies를 놓칠 수 있다. Color changes, subtle texture variations 등이다.

Autoencoder는 pixel-level appearance를 다룬다. Raw image space에서 작동한다. Appearance anomalies를 잘 탐지한다. 그러나 semantic anomalies에 약할 수 있다. 정상과 유사하게 생긴 이상을 놓칠 수 있다.

두 approaches의 fusion이 robustness를 제공한다. PDN과 autoencoder가 서로 보완한다. One path가 놓친 것을 other path가 포착한다. Overall detection capability가 향상된다.

**Dual-path Training**

PDN과 autoencoder는 독립적으로 학습된다. Separate loss functions와 optimizers를 가진다. Joint optimization이 아니라 parallel training이다.

PDN loss는 teacher-student feature matching이다:

$$\mathcal{L}_{\text{PDN}} = \| \mathbf{F}^T - \mathbf{F}^{\text{PDN}} \|^2$$

Autoencoder loss는 image reconstruction error다:

$$\mathcal{L}_{\text{AE}} = \| \mathbf{I} - \hat{\mathbf{I}} \|^2$$

두 losses는 independent하게 minimize된다. 이는 training을 단순화한다. Hyperparameter tuning이 쉽다. 각 component를 separately optimize할 수 있다.

Training은 매우 빠르다. 두 networks 모두 small하므로 각각 5-10분이면 수렴한다. Total training time은 10-20분이다. 이는 exceptionally fast하다.

**Score Fusion**

Inference 시 두 paths의 anomaly scores를 fusion한다. PDN score는 feature discrepancy다:

$$s_{\text{PDN}} = \| \mathbf{F}^T - \mathbf{F}^{\text{PDN}} \|^2$$

Autoencoder score는 reconstruction error다:

$$s_{\text{AE}} = \| \mathbf{I} - \hat{\mathbf{I}} \|^2$$

두 scores를 normalize하고 weighted average를 취한다:

$$s_{\text{final}} = \lambda s_{\text{PDN}}^{\text{norm}} + (1-\lambda) s_{\text{AE}}^{\text{norm}}$$

$\lambda$는 fusion weight로 일반적으로 0.5-0.7이다. PDN에 slightly higher weight를 준다. Empirically PDN이 더 informative하기 때문이다.

Normalization은 각 score의 valid range를 0-1로 map한다. Training set의 statistics를 사용한다. Max score를 1로, min score를 0으로 normalize한다.

Alternative로 max fusion을 사용할 수 있다:

$$s_{\text{final}} = \max(s_{\text{PDN}}^{\text{norm}}, s_{\text{AE}}^{\text{norm}})$$

이는 두 paths 중 하나라도 anomaly를 detect하면 flag한다. Recall을 maximize한다. 그러나 false positives가 증가할 수 있다.

### 5.2.3 Extreme Optimization

EfficientAD의 speed는 careful optimization의 결과다. Every detail이 efficiency를 위해 tuned되었다.

**Teacher Feature Caching**

Teacher(EfficientNet-B4)는 frozen이다. 학습과 추론 내내 변하지 않는다. 따라서 teacher features를 pre-compute하고 cache할 수 있다. Inference 시 teacher forward pass가 필요 없다.

Training 시 모든 training images의 teacher features를 미리 계산한다. Disk에 저장한다. Training loop에서 이들을 load하여 사용한다. Teacher forward pass를 skip한다. Training이 2-3배 빠르다.

Inference 시에도 비슷하다. 만약 images가 고정되어 있다면(inspection system) features를 pre-compute한다. Real-time inference에서 teacher evaluation을 skip한다. 이는 10-20ms를 절약한다.

그러나 dynamic images(streaming input)에서는 불가능하다. Teacher forward pass가 필요하다. 이 경우 EfficientNet-B4가 bottleneck이 될 수 있다. Lighter teacher(EfficientNet-B0)를 사용하면 더 빠르다.

**Quantization**

EfficientAD는 INT8 quantization을 적용한다. Model weights와 activations를 8-bit integers로 표현한다. 이는 4배 memory 절약과 2-4배 speed 향상을 제공한다.

Post-training quantization을 사용한다. Trained model을 quantize한다. Calibration dataset(100-200 images)으로 activation ranges를 추정한다. Per-channel quantization으로 accuracy 저하를 최소화한다.

결과는 impressive하다. FP32 baseline: 97.8% AUROC, 1-5ms. INT8 quantized: 97.5% AUROC, 0.5-2ms. 0.3%포인트 정확도 감소로 2-3배 speedup을 얻는다. Excellent trade-off다.

Quantization-aware training도 시도할 수 있다. Training 중 quantization을 simulate한다. Accuracy 저하가 더 작다(0.1%포인트). 그러나 training complexity가 증가한다. Post-training quantization이 대부분의 경우 충분하다.

**Framework Optimization**

PyTorch나 TensorFlow의 standard operations는 항상 optimal하지 않다. EfficientAD는 specialized implementations를 사용한다.

ONNX Runtime으로 모델을 export한다. Graph optimizations(operator fusion, constant folding)를 자동으로 수행한다. CPU에서 특히 효과적이다. 30-50% speedup을 제공한다.

TensorRT는 NVIDIA GPUs에서 사용된다. Kernel auto-tuning과 layer fusion으로 GPU utilization을 maximize한다. 2-3배 speedup을 달성한다. Sub-millisecond inference가 가능해진다.

Custom CUDA kernels도 고려할 수 있다. Critical operations(convolutions, element-wise ops)를 hand-optimized한다. 전문 지식이 필요하지만 추가 10-20% speedup이 가능하다.

**Batch Processing**

Single image inference는 GPU를 underutilize한다. Batch processing으로 throughput을 극대화한다.

Batch size 32로 처리하면 초당 200-500 프레임을 달성한다. Batch size 64로 500-1000 프레임이다. 이는 가장 빠른 production lines도 충족한다.

Dynamic batching도 구현할 수 있다. Incoming images를 짧은 시간(10-50ms) buffer한다. Accumulated images를 batch로 처리한다. Latency와 throughput을 balance한다.

**Memory Access Patterns**

Memory bandwidth가 종종 bottleneck이다. Especially on edge devices. EfficientAD는 memory-friendly design을 가진다.

Small model size가 cache-friendly하다. Entire model이 L2/L3 cache에 fit한다. DRAM access가 minimal하다. 이는 특히 CPU에서 중요하다.

Sequential memory access patterns를 maximize한다. Strided access나 random access를 피한다. Hardware prefetchers가 효과적으로 작동한다.

In-place operations를 가능한 한 사용한다. Temporary buffers를 minimize한다. Memory allocation overhead를 줄인다.

## 5.3 Architecture Design (~50K parameters)

EfficientAD의 architecture는 every parameter가 justify되는 극도의 efficiency를 보여준다.

**Parameter Budget Allocation**

Total 150K parameters(PDN 50K + AE 100K)를 어떻게 allocate하는가?

PDN (50K total):
- Layer 1 (128 channels): 15K params (30%)
- Layer 2 (256 channels): 20K params (40%)
- Layer 3 (256 channels): 10K params (20%)
- Layer 4 (384 channels): 5K params (10%)

대부분의 parameters가 앞쪽 2 layers에 있다. 이들이 most informative하기 때문이다. 뒤쪽 layers는 refinement만 한다.

Autoencoder (100K total):
- Encoder: 40K params (40%)
- Decoder: 60K params (60%)

Decoder가 더 많은 parameters를 가진다. Reconstruction이 더 challenging하기 때문이다. Encoder는 compression만 하면 되지만 decoder는 details를 restore해야 한다.

**Channel Count Selection**

각 layer의 channel counts는 empirically determined되었다. Extensive ablations을 통해 optimal values를 찾았다.

PDN channel variants:
- [64, 128, 128, 256]: 25K params, 96.8% AUROC
- [128, 256, 256, 384]: 50K params, 97.8% AUROC (default)
- [256, 512, 512, 768]: 100K params, 98.0% AUROC

50K configuration이 best cost-performance를 제공했다. 25K는 underfitting했다. 100K는 marginal improvement(0.2%p)로 worth하지 않았다.

**Activation Functions**

Simple ReLU만 사용된다. Advanced activations(ELU, GELU, Swish)는 고려되지 않았다. 이유는 efficiency다.

ReLU는 extremely cheap하다. Max(0, x) operation이다. Single comparison과 selection이다. Hardware-friendly하다. CPUs와 GPUs 모두에서 잘 optimize된다.

Advanced activations는 exponential나 other complex operations을 요구한다. 계산 비용이 높다. Accuracy improvement도 minimal하다(0.1-0.2%p). Trade-off가 불리하다.

Leaky ReLU도 시도되었다. ReLU보다 약간 complex하지만 dying ReLU 문제를 완화한다. 결과는 비슷했다(97.8% vs 97.8%). Standard ReLU가 simplicity로 선택되었다.

**No Batch Normalization**

BN이 제거된 것은 significant decision이었다. BN은 training stability와 convergence에 도움이 된다. 그러나 inference overhead를 추가한다.

BN은 inference 시 running statistics를 load하고 apply해야 한다. Mean과 variance를 subtract/divide한다. 이는 additional operations다. Small networks에서 relative overhead가 크다.

또한 BN은 batch size에 sensitive하다. Single image inference(batch=1)에서 불안정할 수 있다. Running statistics가 large batch training과 mismatch될 수 있다.

EfficientAD는 BN 없이도 stable training을 달성했다. Careful initialization(Kaiming He)과 moderate learning rates로 충분했다. Inference가 단순해지고 빨라졌다.

**No Skip Connections**

U-Net style skip connections도 사용되지 않았다. 이들은 detail preservation에 도움이 되지만 complexity를 증가시킨다.

Skip connections는 additional concatenations와 channels를 요구한다. Parameter count가 증가한다. Computation도 증가한다. Memory footprint도 커진다.

EfficientAD의 autoencoder는 bottleneck을 통과하는 single path만 가진다. Simplest possible design이다. 놀랍게도 충분한 reconstruction quality를 제공했다.

이는 task의 특성 때문이다. Perfect reconstruction이 필요 없다. 정상 패턴의 approximate reconstruction만 필요하다. Extreme detail preservation이 불필요하다.

**Comparison with Larger Models**

EfficientAD를 larger models와 비교하면 efficiency가 명확하다:

| Model | Parameters | Size | GPU Time | CPU Time | AUROC |
|-------|------------|------|----------|----------|-------|
| STFPM (ResNet18) | 11.7M | 45MB | 50ms | 500ms | 95.5% |
| RD (Custom) | 15M | 60MB | 120ms | 800ms | 98.6% |
| FastFlow | 8M | 32MB | 25ms | 300ms | 98.5% |
| EfficientAD | 0.15M | 1MB | 3ms | 15ms | 97.8% |

EfficientAD는 50-100배 작고 10-50배 빠르다. 정확도는 1-3%포인트만 낮다. Efficiency-accuracy trade-off가 exceptional하다.

## 5.4 Performance Analysis

### 5.4.1 1-5ms Inference

EfficientAD의 가장 impressive한 특징은 inference speed다.

**GPU Performance**

NVIDIA RTX 3090에서 측정:
- Single image: 1-2ms
- Batch 8: 0.8ms per image (6.4ms total)
- Batch 32: 0.5ms per image (16ms total)
- Throughput: 초당 500-2000 프레임

이는 다른 모든 methods를 압도한다. FastFlow(20-50ms)보다 10-20배, STFPM(50-100ms)보다 25-50배, RD(100-150ms)보다 50-100배 빠르다.

RTX 2080에서:
- Single image: 2-3ms
- Batch 32: 0.8ms per image
- Throughput: 초당 300-1200 프레임

여전히 exceptionally fast하다. Older GPUs에서도 real-time processing이 가능하다.

**Component Breakdown**

Inference time을 components로 분해:

With teacher computation (dynamic input):
- Teacher forward (EfficientNet-B4): 8-12ms
- PDN forward: 0.5-1ms
- Autoencoder forward: 0.5-1ms
- Score fusion: 0.1-0.2ms
- Total: 9-14ms

With cached teacher features (static input):
- PDN forward: 0.5-1ms
- Autoencoder forward: 0.5-1ms
- Score fusion: 0.1-0.2ms
- Total: 1-2ms

Teacher가 bottleneck이다. Features를 cache하면 극도로 빠르다. Dynamic input에서도 10-15ms로 여전히 real-time이다.

**Optimization Impact**

각 optimization technique의 기여:

- Baseline (FP32, no optimization): 5-8ms
- + ONNX Runtime: 3-5ms (40% faster)
- + INT8 quantization: 2-3ms (50% faster)
- + TensorRT: 1-2ms (60% faster)
- Final: 1-2ms (4-8배 speedup total)

Multiple optimizations가 cumulative효과를 낸다. Each contributes meaningful speedup.

### 5.4.2 CPU Capability

EfficientAD의 unique strength는 CPU performance다.

**CPU Inference Speed**

Intel i9-10900K (10 cores)에서 측정:
- Single image: 10-15ms
- Batch 8: 8ms per image
- Batch 32: 6ms per image
- Throughput: 초당 60-150 프레임

이는 GPU 없이도 real-time에 가깝다. 다른 methods는 CPU에서 수백 ms - 초 단위다.

Intel i5-8250U (4 cores, laptop)에서:
- Single image: 25-35ms
- Batch 8: 20ms per image
- Throughput: 초당 30-50 프레임

Laptop CPU에서도 reasonable하다. Many applications에 충분하다.

**Why CPU Performance Matters**

CPU capability가 중요한 이유:

1. **Lower cost**: GPUs는 expensive하다. CPU-only systems이 훨씬 저렴하다. Budget-conscious deployments에 critical하다.

2. **Edge devices**: Many edge devices가 GPU가 없다. Raspberry Pi, industrial PCs, embedded systems 등이다. CPU performance가 deployment를 enable한다.

3. **Power efficiency**: GPUs는 power-hungry하다. 수백 watts를 소비한다. CPUs는 수십 watts다. Battery-powered나 energy-conscious applications에 중요하다.

4. **Simplicity**: GPU drivers, CUDA, libraries 등이 불필요하다. Deployment가 단순해진다. Maintenance burden이 줄어든다.

5. **Scalability**: Server에서 많은 categories를 처리할 때 CPU가 유리할 수 있다. GPU는 single stream에 최적화되어 있다. CPUs는 multiple parallel tasks를 효율적으로 handle한다.

**CPU Optimization Techniques**

EfficientAD가 CPU에서 빠른 이유:

Small model size가 cache-friendly하다. Entire model이 L3 cache에 fit한다. Cache misses가 minimal하다. Memory bandwidth가 bottleneck이 아니다.

Simple operations(convolutions, ReLU)가 CPU에서 잘 optimize된다. SIMD instructions(AVX, AVX512)가 effectively utilized된다. Intel MKL이나 OpenBLAS가 accelerate한다.

No exotic operations이 사용된다. All operations이 standard이고 well-supported다. Custom implementations가 불필요하다. Library optimizations를 fully leverage한다.

INT8 quantization이 특히 CPU에서 효과적이다. Modern CPUs(Ice Lake 이후)가 VNNI instructions를 가진다. INT8 operations이 FP32보다 4-8배 빠르다.

### 5.4.3 97.8% Accuracy

EfficientAD는 extreme efficiency에도 불구하고 high accuracy를 유지한다.

**Benchmark Performance**

MVTec AD results:
- Image-level AUROC: 97.8%
- Pixel-level AUROC: 97.4%

이는 top-tier에는 약간 못 미치지만 여전히 excellent하다. PatchCore(99.1%)보다 1.3%p 낮다. RD(98.6%)보다 0.8%p 낮다. FastFlow(98.5%)보다 0.7%p 낮다.

그러나 speed를 고려하면 exceptional하다. 10-50배 빠르면서 1-3%p만 낮다. Speed-accuracy trade-off가 매우 favorable하다.

**Category-wise Performance**

Texture categories:
- Carpet: 98.5%
- Grid: 99.1%
- Leather: 99.8%
- Tile: 98.7%
- Wood: 99.2%

Textures에서 강력하다. 대부분 98% 이상이다. Autoencoder가 texture anomalies를 잘 detect한다.

Object categories:
- Bottle: 99.3%
- Cable: 96.8%
- Capsule: 95.2%
- Metal_nut: 96.5%
- Screw: 94.8%

Objects에서 약간 낮다. 특히 complex structures(screw, capsule)에서 어려움을 겪는다. Small model capacity의 한계가 드러난다.

전반적으로 consistent하다. 극단적인 failures가 없다. Reasonable performance를 모든 categories에서 제공한다.

**Accuracy Limitations**

97.8%가 insufficient한 경우는 언제인가?

1. **Critical applications**: Medical devices, aerospace components 등에서 99%+ accuracy가 필수적이다. 0.5-1% difference가 critical하다.

2. **Subtle defects**: 매우 미세한 결함(0.1-0.3% of image)은 놓칠 수 있다. Small model이 fine details를 완전히 capture하지 못한다.

3. **Complex textures**: High-frequency patterns나 irregular textures에서 약점을 보인다. Simple autoencoder가 reconstruct하기 어렵다.

4. **New defect types**: Training에서 본 적 없는 completely novel anomalies에 약할 수 있다. Limited capacity가 generalization을 제약한다.

이러한 cases에서는 larger models(RD, PatchCore)가 필요하다. EfficientAD는 만능이 아니다. Appropriate use case selection이 중요하다.

**Accuracy-Speed Pareto Optimal**

EfficientAD는 Pareto front의 specific point를 occupy한다:

```
Accuracy (%)
99 |         ● PatchCore
98 |    ● RD  ● FastFlow
97 | ● EfficientAD
96 | ● STFPM
   +-------------------------
   0   50  100  150  200  Speed (ms)
```

EfficientAD는 extreme speed end에 있다. Fastest while maintaining reasonable accuracy. 이는 valuable niche다. Many applications이 이 point를 선호한다.

## 5.5 Edge Deployment

EfficientAD의 killer feature는 edge deployment capability다.

**Raspberry Pi Performance**

Raspberry Pi 4 (4GB RAM)에서 test:
- CPU: 60-80ms per image
- Throughput: 초당 12-16 프레임
- Memory usage: 50-100MB

이는 놀랍다. $35 device에서 near-real-time processing이 가능하다. 다른 methods는 Raspberry Pi에서 초 단위로 느리거나 아예 실행 불가능하다.

실용적으로 12-16 FPS는 many inspection tasks에 충분하다. Manual inspection이나 slow-moving conveyors를 monitor할 수 있다. Portable quality control devices를 만들 수 있다.

**Jetson Nano**

NVIDIA Jetson Nano (4GB)에서:
- GPU: 15-25ms per image
- Throughput: 초당 40-60 프레임
- Memory usage: 80-150MB
- Power: 5-10W

Jetson Nano는 low-power edge AI device다. $99에 GPU를 제공한다. EfficientAD가 이를 fully leverage한다. Near-real-time processing이 low power consumption으로 가능하다.

이는 battery-powered systems에 ideal하다. Mobile inspection robots, handheld scanners, wireless cameras 등이다. Continuous operation이 hours 동안 가능하다.

**Mobile Devices**

Modern smartphones(iPhone 13, Samsung S21 등)에서:
- Neural engine: 10-20ms per image
- Throughput: 초당 50-100 프레임
- Battery impact: Minimal

CoreML이나 TensorFlow Lite로 모델을 deploy한다. Phones의 neural accelerators를 사용한다. Exceptionally fast하고 energy-efficient하다.

Mobile deployment는 새로운 use cases를 enable한다. Consumer-facing quality check apps. Field inspection by technicians. Crowdsourced quality control. 이들이 현실화된다.

**Industrial IoT**

Factory floor의 edge devices에서:
- Industrial PCs (Intel Atom, Celeron): 30-50ms
- Embedded controllers (ARM Cortex-A): 50-100ms
- Throughput: 초당 10-30 프레임

이는 distributed inspection systems를 가능하게 한다. 각 production station에 edge device를 배치한다. Local processing으로 latency를 minimize한다. 네트워크 bandwidth 요구가 적다.

Centralized server approach와 비교:
- Edge (EfficientAD): 30-50ms latency, no network needed
- Server (PatchCore): 30ms inference + 20-50ms network = 50-100ms total

Edge가 latency에서 유리하다. 또한 privacy와 reliability 장점도 있다. 네트워크 failure에 영향받지 않는다.

**Deployment Strategies**

Edge deployment의 practical strategies:

1. **Tiered approach**: Edge에서 1차 screening을 수행한다. Suspicious cases만 server로 전송하여 정밀 검사한다. 90%는 edge에서 처리되어 네트워크 load가 minimal하다.

2. **Model cascade**: EfficientAD로 fast filtering을 하고 borderline cases에만 larger model을 적용한다. Majority가 EfficientAD로 clear하게 판정되어 throughput이 높다.

3. **Confidence-based routing**: EfficientAD가 high confidence면 local decision을 내린다. Low confidence면 server에 defer한다. Accuracy와 efficiency를 balance한다.

4. **Continuous deployment**: OTA updates로 models를 remotely update한다. Field의 thousands of devices를 centrally manage한다. New categories나 improvements를 신속히 deploy한다.

## 5.6 Implementation Guide

EfficientAD를 실무에 구현하는 practical guide를 제공한다.

**Setup and Installation**

Dependencies는 minimal하다:
```
pytorch >= 1.8
torchvision >= 0.9
numpy
pillow
```

EfficientAD의 작은 크기 덕분에 설치가 빠르다. pip install로 수 초면 완료된다. Docker container도 50-100MB로 매우 작다.

Official implementation을 clone하고 setup한다:
```bash
git clone https://github.com/nelson1425/EfficientAD.git
cd EfficientAD
pip install -r requirements.txt
```

**Training Workflow**

1. **Data preparation**: 정상 이미지를 폴더에 준비한다. 최소 50-100장, 권장 200-400장이다.

2. **Teacher feature extraction**: EfficientNet-B4로 모든 images의 features를 pre-compute한다.
```python
teacher = efficientnet_b4(pretrained=True)
features = extract_features(teacher, train_images)
save_features(features, 'cache/features.pt')
```

3. **PDN training**: Cached features로 PDN을 학습한다.
```python
pdn = PDN(in_channels=384, out_channels=384)
optimizer = Adam(pdn.parameters(), lr=1e-2)
train_pdn(pdn, cached_features, epochs=30)
```

4. **Autoencoder training**: Separately autoencoder를 학습한다.
```python
ae = Autoencoder(latent_dim=64)
optimizer = Adam(ae.parameters(), lr=1e-3)
train_autoencoder(ae, train_images, epochs=50)
```

Total training time: 10-20분. Extremely fast하다.

**Inference Pipeline**

```python
def infer(image, teacher, pdn, ae):
    # Teacher features (or load cached)
    feat_teacher = teacher(image)
    
    # PDN prediction
    feat_student = pdn(feat_teacher)
    score_pdn = ((feat_teacher - feat_student) ** 2).mean()
    
    # Autoencoder reconstruction
    recon = ae(image)
    score_ae = ((image - recon) ** 2).mean()
    
    # Fusion
    score_pdn_norm = normalize(score_pdn, pdn_stats)
    score_ae_norm = normalize(score_ae, ae_stats)
    score_final = 0.6 * score_pdn_norm + 0.4 * score_ae_norm
    
    # Decision
    is_anomaly = score_final > threshold
    
    return is_anomaly, score_final
```

**Threshold Selection**

Training images의 scores로 statistics를 계산한다:
```python
train_scores = [infer(img)[1] for img in train_images]
mean = np.mean(train_scores)
std = np.std(train_scores)
threshold = mean + 2 * std  # 95% acceptance
```

Validation set이 있다면 ROC analysis를 수행한다:
```python
fpr, tpr, thresholds = roc_curve(labels, scores)
optimal_idx = np.argmax(tpr - fpr)
threshold = thresholds[optimal_idx]
```

**Optimization for Production**

1. **ONNX export**: Model을 ONNX로 변환한다.
```python
torch.onnx.export(pdn, dummy_input, 'pdn.onnx')
torch.onnx.export(ae, dummy_input, 'ae.onnx')
```

2. **INT8 quantization**: Calibration으로 quantize한다.
```python
from pytorch_quantization import quantize
pdn_int8 = quantize(pdn, calibration_data)
ae_int8 = quantize(ae, calibration_data)
```

3. **TensorRT compilation** (GPU):
```bash
trtexec --onnx=pdn.onnx --int8 --saveEngine=pdn.trt
trtexec --onnx=ae.onnx --int8 --saveEngine=ae.trt
```

4. **Load and infer**:
```python
pdn_trt = load_tensorrt('pdn.trt')
ae_trt = load_tensorrt('ae.trt')
# Inference is now 2-3x faster
```

**Edge Deployment**

Raspberry Pi에서:
```python
# Use lighter teacher if needed
teacher = efficientnet_b0(pretrained=True)  # Faster

# Quantize models
pdn_int8 = quantize(pdn)
ae_int8 = quantize(ae)

# Use ONNX Runtime for CPU optimization
import onnxruntime as ort
pdn_session = ort.InferenceSession('pdn.onnx')
ae_session = ort.InferenceSession('ae.onnx')
```

Mobile devices에서:
```python
# Convert to CoreML (iOS) or TFLite (Android)
import coremltools as ct
pdn_coreml = ct.convert(pdn, convert_to='mlprogram')
pdn_coreml.save('pdn.mlpackage')
```

**Common Issues and Solutions**

**Issue 1**: Training loss not decreasing
- **Solution**: Check learning rate. Try 1e-2 for PDN, 1e-3 for AE. Verify data normalization (ImageNet stats).

**Issue 2**: High false positive rate
- **Solution**: Increase threshold. Use 3-sigma instead of 2-sigma. Ensure training data quality (no contamination).

**Issue 3**: Slow inference on edge device
- **Solution**: Apply INT8 quantization. Use ONNX Runtime. Reduce image resolution (224×224 instead of 256×256).

**Issue 4**: Poor accuracy on specific category
- **Solution**: Increase training data (300-500 images). Try dual-path fusion weights tuning. Consider using larger teacher (EfficientNet-B5).

**Hyperparameter Recommendations**

Default settings (work for most cases):
- PDN channels: [128, 256, 256, 384]
- AE latent dimension: 64
- PDN learning rate: 1e-2
- AE learning rate: 1e-3
- PDN epochs: 20-30
- AE epochs: 40-60
- Fusion weight: 0.6 PDN, 0.4 AE
- Threshold: mean + 2*std

Category-specific tuning:
- Simple textures: Lower AE weight (0.3)
- Complex objects: Higher AE weight (0.5)
- Small defects: Lower threshold (mean + 1.5*std)
- Subtle anomalies: Higher fusion weight on PDN (0.7)

EfficientAD는 implementation이 straightforward하고 deployment가 flexible하다. Minimal dependencies와 small size로 거의 모든 환경에 적용 가능하다. Official code와 documentation이 excellent하여 learning curve가 낮다. 이러한 accessibility가 wide adoption을 촉진한다.

# 6. Comprehensive Comparison

## 6.1 Evolution Timeline

Knowledge distillation 기반 이상 탐지의 발전은 2021-2024년의 짧은 기간 동안 급격히 진행되었다. 각 방법이 이전의 한계를 극복하며 패러다임을 발전시켰다.

**2021: STFPM - Foundation**

STFPM은 knowledge distillation을 anomaly detection에 명확히 정립했다. Pre-trained teacher와 trainable student의 framework를 확립했다. Multi-scale feature pyramid matching을 도입했다. 95.5% AUROC로 distillation의 feasibility를 입증했다. 그러나 성능이 memory-based methods에 미치지 못했고 속도도 충분히 빠르지 않았다(50-100ms).

STFPM의 핵심 기여는 패러다임 확립이었다. "Teacher-student distillation이 anomaly detection에 작동한다"는 것을 보여줬다. 단순한 L2 feature matching만으로도 reasonable performance를 얻을 수 있었다. 이는 후속 연구들의 출발점이 되었다.

한계도 명확했다. Same architecture의 teacher-student가 optimal하지 않았다. Loss function이 단순했다. Teacher 선택이 고정적이었다. 이들이 개선의 여지를 제공했다.

**2022: Reverse Distillation - Accuracy Peak**

RD는 paradigm inversion으로 breakthrough를 달성했다. Student를 고정하고 teacher를 학습시키는 역전된 framework였다. Domain-specific teacher가 정상 패턴을 deeply 학습했다. One-class embedding과 encoder-decoder structure를 도입했다. 98.6% AUROC로 distillation methods 중 최고를 달성했다.

RD의 기여는 accuracy ceiling을 높인 것이었다. Distillation이 단순한 efficiency tool이 아니라 accuracy에서도 competitive함을 입증했다. PatchCore(99.1%)와 0.5%포인트 차이로 근접했다. Pixel-level에서는 PatchCore를 능가했다(98.7% vs 98.6%).

그러나 trade-off가 있었다. 추론이 느렸다(100-150ms). 메모리 사용량이 컸다(300-500MB). 각 카테고리마다 teacher를 학습해야 했다. Accuracy를 위해 efficiency를 희생했다.

**2023: FRE - Failed Experiment**

FRE는 feature reconstruction approach를 시도했다. Encoder-decoder student가 teacher features를 reconstruct했다. Bottleneck이 one-class learning을 강제했다. 96.5% AUROC로 STFPM보다는 높았지만 RD에 크게 못 미쳤다. 추론도 느렸다(80-120ms).

FRE의 실패는 교훈을 제공했다. Incremental improvements는 insufficient하다. Complexity는 명확한 benefits로 justify되어야 한다. Clear value proposition이 필요하다. Empirical results만으로는 부족하고 deep insights가 요구된다.

FRE는 peer-reviewed publication에 실패했고 community에 채택되지 않았다. 이는 research quality의 중요성을 강조한다. Thorough analysis, extensive experiments, novel insights가 필수적이다.

**2024: EfficientAD - Efficiency Revolution**

EfficientAD는 extreme efficiency로 paradigm shift를 가져왔다. 50K parameters의 tiny PDN이 teacher features를 describe했다. Small autoencoder와의 dual-path로 robustness를 확보했다. 97.8% AUROC를 유지하면서 1-5ms 추론을 달성했다. CPU에서도 10-30ms로 실행 가능했다.

EfficientAD의 기여는 real-time processing을 현실화한 것이었다. 10-50배 speedup은 transformative였다. Edge deployment를 가능하게 했다. Raspberry Pi, Jetson Nano, smartphones에서 작동했다. New use cases를 열었다.

Trade-off는 명확했다. Accuracy가 top-tier에 못 미쳤다(RD보다 0.8%p, PatchCore보다 1.3%p 낮음). Subtle defects나 complex textures에서 약점을 보였다. 그러나 many applications에서 97.8%는 충분했고 speed advantage가 압도적이었다.

**Evolution Patterns**

발전 과정에서 몇 가지 패턴이 드러난다:

1. **Accuracy-speed spectrum exploration**: 초기(STFPM)는 balanced였다. RD는 accuracy extreme으로, EfficientAD는 speed extreme으로 이동했다. Spectrum의 different points를 explore했다.

2. **Architectural simplification**: STFPM의 ResNet18 student(11.7M) → RD의 custom encoder-decoder(15M) → EfficientAD의 PDN(50K). 점진적 단순화가 아니라 radical redesign이었다.

3. **Task-specific optimization**: Generic architectures(ResNet)에서 task-specific designs(PDN)로 이동했다. Domain knowledge를 architecture에 반영했다. Over-parameterization을 피했다.

4. **Dual objectives**: 초기는 accuracy만 추구했다. 후기는 efficiency도 critical objective로 삼았다. Multi-objective optimization이 필요해졌다.

5. **Practical focus**: 학술적 novelty에서 practical deployment로 관심이 이동했다. Real-world constraints(speed, memory, power)가 중요해졌다. Industry needs가 research direction을 shape했다.

**Current State**

2024년 현재 knowledge distillation은 mature paradigm이다. Three main approaches가 확립되었다:
- **Balanced**: STFPM-style direct matching (95-96% AUROC, 50-100ms)
- **Precision**: RD-style reverse distillation (98-99% AUROC, 100-150ms)
- **Speed**: EfficientAD-style extreme efficiency (97-98% AUROC, 1-5ms)

각각 different use cases에 적합하다. One size does not fit all. Application requirements에 따라 선택한다. 이러한 diversity가 패러다임의 성숙도를 나타낸다.

**Future Directions**

향후 발전은 여러 방향으로 진행될 것이다:
- **Foundation model integration**: CLIP, DINOv2 같은 powerful teachers 활용
- **Few-shot learning**: 적은 데이터로 빠른 adaptation
- **Unified frameworks**: Single model로 multiple categories 처리
- **Neural architecture search**: Automated optimal architecture discovery
- **Hardware co-design**: Specific hardware에 맞춘 algorithm design

Knowledge distillation은 계속 진화할 것이다. 그러나 core principles(teacher-student framework, feature matching, capacity constraint)는 유지될 것이다. 이들이 paradigm의 foundation이기 때문이다.

## 6.2 Two Extremes

Knowledge distillation 방법들 중 Reverse Distillation과 EfficientAD는 spectrum의 양극단을 represent한다.

### 6.2.1 Precision (Reverse Distillation)

RD는 accuracy를 absolute priority로 삼았다. Every design decision이 performance maximization을 위한 것이었다.

**Design Philosophy**

RD의 철학은 "highest possible accuracy at reasonable cost"였다. Speed나 memory는 secondary considerations였다. 중요한 것은 best detection performance였다. 이는 critical applications을 target으로 한 것이다.

Domain-specific teacher가 핵심이었다. Generic pre-trained teacher가 아니라 target domain에 fully optimized된 teacher였다. 정상 데이터로 학습되어 subtle nuances를 포착했다. 이는 accuracy의 주된 source였다.

One-class embedding이 powerful regularization을 제공했다. Extreme compression(16K dimensions)이 essential patterns만 보존하도록 강제했다. Tight clustering이 robust anomaly detection을 가능하게 했다. 이론적으로 elegant하고 empirically effective했다.

Multi-scale encoder-decoder가 comprehensive information을 처리했다. Layer2, layer3, layer4의 세 scales를 독립적으로 다뤘다. 각 scale이 different defect sizes를 담당했다. Redundancy가 있지만 robustness를 제공했다.

**Performance Characteristics**

RD는 거의 모든 metrics에서 excellent했다:
- Image-level AUROC: 98.6% (distillation 최고)
- Pixel-level AUROC: 98.7% (전체 방법 중 최고 tie)
- Small defect detection: 95.2% (매우 강력)
- Boundary precision: F1 0.58 (최고)
- Consistency: Std 0.2-0.3%p (매우 안정)

특히 challenging cases에서 우수했다. Complex objects, subtle defects, high-variance backgrounds에서 다른 방법들보다 robust했다. False positive rate도 낮았다(precision 0.92 at 95% recall).

**Use Case Fit**

RD가 optimal한 경우:

1. **Critical accuracy requirements**: Medical devices, aerospace components, safety-critical parts. 0.5-1% difference가 중요한 applications.

2. **Precise localization needs**: Repair guidance, root cause analysis, quality grading. Pixel-level accuracy가 essential.

3. **Few high-value categories**: 수 개의 products에 집중. 각각을 highly optimize할 수 있는 resources 있음.

4. **Offline processing**: Batch inspection, post-production analysis. Real-time이 필수가 아닌 경우.

5. **Sufficient computational resources**: Server deployment, cloud processing. GPU와 memory가 충분한 환경.

**Limitations**

RD의 한계도 명확하다:

- **Speed**: 100-150ms는 많은 real-time applications에 느림. High-speed production lines(초당 30+ items)에 부적합.

- **Memory**: 300-500MB per category는 hundreds of categories에 prohibitive. Edge devices에 부적합.

- **Scalability**: 각 category마다 teacher training 필요. Deployment complexity가 증가.

- **Cost**: Training time(1-3 hours)과 computational requirements가 높음. Rapid prototyping이나 frequent updates에 장애.

### 6.2.2 Speed (EfficientAD)

EfficientAD는 정반대 extreme을 represent한다. Speed와 efficiency가 absolute priority였다.

**Design Philosophy**

EfficientAD의 철학은 "real-time at acceptable accuracy"였다. Millisecond-level inference가 목표였다. Accuracy는 trade-off 가능했지만 speed는 non-negotiable이었다. 이는 high-throughput applications을 target으로 한 것이다.

Extreme minimalism이 핵심이었다. Every parameter가 justify되어야 했다. 50K parameters로 essential functionality만 구현했다. Unnecessary complexity를 ruthlessly eliminate했다. "Less is more" 원칙의 극단적 실현이었다.

Task-specific design이 efficiency를 가능하게 했다. Generic architecture를 adapt하지 않았다. Ground-up으로 anomaly detection에 optimal한 design을 created했다. Domain knowledge를 fully leverage했다.

Dual-path approach가 robustness를 제공했다. Single tiny model이 아니라 two complementary models이었다. PDN(semantic)과 AE(appearance)가 서로 보완했다. Redundancy가 있지만 안정성을 확보했다.

**Performance Characteristics**

EfficientAD는 efficiency metrics에서 압도적이었다:
- GPU inference: 1-5ms (10-50배 faster)
- CPU inference: 10-30ms (10-30배 faster)
- Model size: 1MB (50-500배 smaller)
- Memory usage: 50-100MB (3-10배 smaller)
- Training time: 10-20min (10-30배 faster)

Accuracy는 excellent이지만 top-tier는 아니었다:
- Image-level AUROC: 97.8% (1-3%p lower than best)
- Pixel-level AUROC: 97.4% (decent)
- Complex cases: 94-96% (some weakness)

Trade-off는 favorable했다. 10-50배 speedup for 1-3%p accuracy loss. Cost-benefit ratio가 exceptional했다.

**Use Case Fit**

EfficientAD가 optimal한 경우:

1. **Real-time requirements**: High-speed production lines, live video streams. 초당 30-100 items 처리.

2. **Edge deployment**: Raspberry Pi, Jetson Nano, mobile devices. Limited computational resources.

3. **Many categories**: Hundreds of categories를 single device에. Scalable deployment.

4. **Power constraints**: Battery-powered, energy-conscious applications. Low power consumption critical.

5. **Cost sensitivity**: Budget-limited deployments. Expensive GPUs 불가.

6. **Rapid deployment**: Quick setup, minimal tuning. Fast time-to-market.

**Limitations**

EfficientAD의 한계:

- **Accuracy ceiling**: 97.8%는 some critical applications에 insufficient. 99%+ 필요한 경우 부적합.

- **Subtle defects**: 매우 미세한 결함(0.1-0.3% of image) 놓칠 수 있음. Limited capacity의 결과.

- **Complex textures**: High-frequency patterns, irregular textures에 약함. Simple autoencoder의 한계.

- **Novel anomalies**: Completely new defect types에 약할 수 있음. Generalization limited by small capacity.

## 6.3 Performance-Speed Analysis

다양한 distillation methods를 performance-speed space에서 종합 분석한다.

**Quantitative Comparison**

| Method | AUROC | GPU (ms) | CPU (ms) | Size (MB) | Memory (MB) |
|--------|-------|----------|----------|-----------|-------------|
| STFPM | 95.5% | 50-100 | 500-800 | 45 | 100-200 |
| FRE | 96.5% | 80-120 | 600-1000 | 30 | 150-250 |
| RD | 98.6% | 100-150 | 800-1200 | 60 | 300-500 |
| EfficientAD | 97.8% | 1-5 | 10-30 | 1 | 50-100 |

**Pareto Frontier Analysis**

Performance-speed plot에서 Pareto-optimal points를 identify한다:

```
Accuracy (AUROC %)
99 |
98 |        ● RD
97 |              ● EfficientAD  
96 |   ● FRE
95 | ● STFPM
   +---------------------------
   0   50  100  150  200  Speed (ms)
```

Pareto frontier는 RD와 EfficientAD를 connect한다. 둘 사이의 any point는 dominated된다. Either RD(더 정확)나 EfficientAD(더 빠름)가 strictly better다.

STFPM과 FRE는 dominated points다. RD가 strictly dominate한다(더 정확하고 비슷한 속도). 역사적 의의는 있지만 practical choice로는 suboptimal하다.

**Efficiency Metrics**

단순 속도뿐만 아니라 종합 efficiency를 평가한다.

Throughput (images per second at batch 32):
- STFPM: 20-40
- FRE: 15-25
- RD: 10-20
- EfficientAD: 200-1000

EfficientAD의 throughput advantage가 dramatic하다. 10-100배 차이다.

Energy efficiency (images per watt-hour):
- STFPM: 1000-2000
- RD: 500-1000
- EfficientAD (GPU): 5000-10000
- EfficientAD (CPU): 2000-4000

EfficientAD가 가장 energy-efficient하다. Battery-powered applications에 critical하다.

Cost efficiency (images per dollar of hardware):
- STFPM (RTX 2080): 1M-2M
- RD (RTX 3090): 500K-1M
- EfficientAD (Raspberry Pi): 10M-20M

EfficientAD가 hardware cost 측면에서 압도적이다. $35 device로 millions of images를 처리한다.

**Scalability Analysis**

Multiple categories deployment를 고려한다.

10 categories:
- STFPM: 1-2GB memory, 500-1000ms total
- RD: 3-5GB memory, 1000-1500ms total
- EfficientAD: 100-200MB memory, 10-50ms total

100 categories:
- STFPM: 10-20GB memory, impractical
- RD: 30-50GB memory, impractical
- EfficientAD: 1-2GB memory, 100-500ms total

EfficientAD만 hundreds of categories를 single device에 practical하게 deploy할 수 있다.

**Quality Metrics**

Accuracy 외의 quality dimensions도 평가한다.

Localization precision (boundary F1):
- STFPM: 0.52
- FRE: 0.50
- RD: 0.58
- EfficientAD: 0.48

RD가 best localization을 제공한다. EfficientAD가 약간 약하지만 acceptable하다.

False positive rate (at 95% recall):
- STFPM: 8%
- FRE: 9%
- RD: 5%
- EfficientAD: 7%

RD가 lowest FPR을 보인다. EfficientAD는 reasonable하다.

Robustness (std across seeds):
- STFPM: 0.3-0.5%p
- RD: 0.2-0.3%p
- EfficientAD: 0.3-0.4%p

All methods가 reasonably robust하다. Significant variance는 없다.

**Practical Trade-off Matrix**

각 dimension에서의 상대적 강점을 matrix로 표현한다:

| Dimension | STFPM | RD | EfficientAD |
|-----------|-------|-----|-------------|
| Accuracy | ★★★ | ★★★★★ | ★★★★ |
| Speed | ★★★ | ★★ | ★★★★★ |
| Memory | ★★★ | ★★ | ★★★★★ |
| Scalability | ★★ | ★ | ★★★★★ |
| Localization | ★★★ | ★★★★★ | ★★★ |
| Edge Deploy | ★★ | ★ | ★★★★★ |
| Training Time | ★★★ | ★★ | ★★★★★ |
| Simplicity | ★★★★ | ★★ | ★★★★ |

RD는 accuracy와 localization에서 dominate한다. EfficientAD는 efficiency-related dimensions에서 dominate한다. STFPM은 middle ground다.

## 6.4 Use Case Matrix

다양한 application scenarios에서 optimal method를 identify한다.

**High-Speed Manufacturing**

**Scenario**: Automotive assembly line, 100 items/min, real-time decision
**Requirements**: <10ms latency, >95% accuracy, edge deployment
**Recommended**: EfficientAD
**Rationale**: Only method meeting speed requirement. 97.8% accuracy sufficient. Edge-deployable.

**Alternative**: None practical. RD too slow. STFPM marginally acceptable but inferior to EfficientAD.

**Medical Device Inspection**

**Scenario**: Critical implants, 1-10 items/hour, highest accuracy
**Requirements**: >99% accuracy, precise localization, offline processing
**Recommended**: RD + PatchCore ensemble
**Rationale**: RD provides 98.6%, ensemble with PatchCore reaches 99%+. Precise localization. Speed non-issue.

**Alternative**: RD alone if 98.6% sufficient. PatchCore if memory acceptable.

**Mobile Quality Control**

**Scenario**: Technician with tablet, field inspection, battery-powered
**Requirements**: <30ms on mobile CPU, <100MB, >96% accuracy
**Recommended**: EfficientAD
**Rationale**: Only method running on mobile devices. 10-20ms on phone. 1MB size. 97.8% adequate.

**Alternative**: None. Other methods require GPU or too slow on CPU.

**Semiconductor Inspection**

**Scenario**: Wafer defects, 0.1mm resolution, subtle anomalies
**Requirements**: >98% accuracy, detect 0.1-0.5mm defects, <100ms
**Recommended**: RD
**Rationale**: Best accuracy (98.6%). Excellent small defect detection. Localization precision. 100-150ms acceptable for high-value products.

**Alternative**: PatchCore for absolute best accuracy if speed non-issue.

**Textile Quality Control**

**Scenario**: Fabric rolls, high-speed inspection, texture anomalies
**Requirements**: <20ms, >97% accuracy, real-time, cost-effective
**Recommended**: EfficientAD
**Rationale**: Fast enough for real-time. 98.5% on textures (stronger than overall). Low cost enables multiple deployment.

**Alternative**: FastFlow (flow-based) also excellent for textures.

**Food Safety**

**Scenario**: Packaged products, contamination detection, regulatory compliance
**Requirements**: >98% accuracy, false negatives critical, moderate speed
**Recommended**: RD
**Rationale**: Highest accuracy minimizes false negatives. Localization for root cause. Compliance documentation easier with high accuracy.

**Alternative**: Ensemble of RD + PatchCore for redundancy.

**Robotics Perception**

**Scenario**: Mobile robot, on-board processing, battery constraint
**Requirements**: <50ms, Jetson Nano, <200MB, >95% accuracy
**Recommended**: EfficientAD
**Rationale**: Runs on Jetson Nano (15-25ms). Low power. 97.8% adequate for navigation/manipulation tasks.

**Alternative**: Lightweight STFPM if 95% acceptable (faster but less accurate).

**Cloud API Service**

**Scenario**: Multi-tenant SaaS, diverse customers, hundreds of categories
**Requirements**: Scalable, cost-effective, >97% accuracy, <100ms
**Recommended**: EfficientAD
**Rationale**: Hundreds of categories feasible. Low memory per category. Fast enough for API latency. Cost-effective at scale.

**Alternative**: STFPM for balance if slightly slower acceptable.

**Research and Benchmarking**

**Scenario**: Algorithm comparison, paper reproduction, academic study
**Requirements**: State-of-art accuracy, well-documented, reproducible
**Recommended**: RD and EfficientAD
**Rationale**: RD for accuracy baseline. EfficientAD for efficiency baseline. Both well-documented, reproducible.

**Alternative**: Include STFPM as historical baseline.

**Decision Matrix Summary**

| Application Type | Primary Choice | Rationale | Backup |
|------------------|----------------|-----------|--------|
| High-speed mfg | EfficientAD | Speed critical | None |
| Medical/critical | RD | Accuracy critical | PatchCore |
| Mobile/edge | EfficientAD | Only viable | None |
| Subtle defects | RD | Best precision | PatchCore |
| Textures | EfficientAD | Fast + strong | FastFlow |
| Cost-sensitive | EfficientAD | Best ROI | STFPM |
| Multi-category | EfficientAD | Scalability | None |
| Offline/batch | RD | Accuracy priority | Ensemble |

EfficientAD가 most versatile하다. Majority of use cases에 적합하다. RD는 specialized high-accuracy applications에 optimal하다. STFPM은 legacy baseline으로만 relevant하다.

# 7. Practical Application Guide

## 7.1 Precision vs Speed Decision

실무에서 가장 중요한 결정은 precision-focused(RD) vs speed-focused(EfficientAD) 선택이다.

**Decision Framework**

다음 decision tree를 따른다:

```
Start
  │
  ├─ Real-time required? (< 30ms)
  │  ├─ Yes → EfficientAD
  │  └─ No → Continue
  │
  ├─ Edge deployment required?
  │  ├─ Yes → EfficientAD
  │  └─ No → Continue
  │
  ├─ Accuracy > 98.5% required?
  │  ├─ Yes → RD
  │  └─ No → Continue
  │
  ├─ Many categories (>50)?
  │  ├─ Yes → EfficientAD
  │  └─ No → Continue
  │
  ├─ Cost-sensitive?
  │  ├─ Yes → EfficientAD
  │  └─ No → Continue
  │
  └─ Default → RD (if resources permit)
              → EfficientAD (if efficiency valued)
```

**Quantitative Criteria**

구체적인 numerical thresholds를 제공한다:

**Choose EfficientAD if:**
- Latency requirement < 30ms
- Throughput requirement > 50 FPS
- Deployment platform: Raspberry Pi, Jetson, mobile
- Number of categories > 50
- Memory budget < 200MB per category
- Hardware budget < $500
- Power budget < 20W
- Accuracy requirement: 96-98% acceptable

**Choose RD if:**
- Accuracy requirement > 98.5%
- False negative cost > 10× false positive cost
- Defect size < 1% of image
- Localization precision critical (F1 > 0.55)
- Latency requirement > 100ms acceptable
- Number of categories < 10
- Hardware budget > $1000
- Can afford GPU deployment

**Hybrid Strategies**

두 approaches를 결합하는 전략도 고려한다.

**Cascade Approach**:
1. EfficientAD로 fast screening (1-5ms)
2. Borderline cases(score 0.4-0.6)만 RD로 refine (100ms)
3. Clear cases(score <0.3 or >0.7)는 EfficientAD decision

Result: 90%는 5ms, 10%만 100ms. Average 15ms. Accuracy RD-level(98.5%).

**Tiered Deployment**:
- Edge: EfficientAD for real-time filtering
- Server: RD for suspicious cases
- Reduces network traffic by 90%
- Combines edge speed with server accuracy

**Confidence-based Routing**:
```python
score_ea, confidence = efficientad_infer(image)
if confidence > 0.8:
    return score_ea  # High confidence, fast path
else:
    score_rd = rd_infer(image)  # Low confidence, accurate path
    return score_rd
```

**Cost-Benefit Analysis**

각 선택의 total cost of ownership를 계산한다.

**EfficientAD Scenario** (100 categories, 1M images/month):
- Hardware: Jetson Xavier ($500) or CPU server ($1000)
- Power: 10W × 730h/month × $0.12/kWh = $1/month
- Maintenance: Minimal (simple deployment)
- Training: 10-20min × 100 = 16-33 hours once
- Total monthly: ~$50

**RD Scenario** (100 categories, 1M images/month):
- Hardware: 4× RTX 3090 ($6000)
- Power: 400W × 730h/month × $0.12/kWh = $35/month
- Maintenance: Higher (complex deployment)
- Training: 2h × 100 = 200 hours recurring
- Total monthly: ~$500

EfficientAD의 TCO가 10배 낮다. 특히 scale에서 차이가 dramatic하다.

**Risk Assessment**

각 선택의 risks를 평가한다.

**EfficientAD Risks**:
- Accuracy insufficiency: 97.8%가 inadequate한 경우
- Subtle defect misses: 작은 결함 놓칠 가능성
- Mitigation: Validation에서 thoroughly test. Critical cases는 hybrid approach.

**RD Risks**:
- Speed bottleneck: Throughput requirement 못 맞출 수 있음
- Scalability issues: Many categories에서 impractical
- Mitigation: Parallel processing, model optimization. Category pruning.

## 7.2 Model Selection by Requirements

다양한 requirements scenarios에서 optimal model을 select한다.

**Requirement Profile: Safety-Critical**

**Specifications**:
- Accuracy: >99%
- False negative rate: <0.5%
- Localization: Pixel-precise
- Speed: <500ms acceptable
- Cost: Not primary concern

**Recommendation**: RD + PatchCore ensemble
- RD: 98.6% baseline
- PatchCore: 99.1% standalone
- Ensemble: 99.3-99.5%
- Vote or average scores
- Double validation reduces FN rate

**Implementation**:
```python
score_rd = rd_infer(image)
score_pc = patchcore_infer(image)
score_final = 0.6 * score_rd + 0.4 * score_pc
# Or: is_anomaly = (score_rd > th_rd) OR (score_pc > th_pc)
```

**Requirement Profile: High-Throughput**

**Specifications**:
- Throughput: >100 FPS
- Latency: <10ms per image
- Accuracy: >96%
- Hardware: Single GPU

**Recommendation**: EfficientAD with batching
- Batch size 32-64
- GPU utilization >90%
- 500-1000 FPS achievable
- 97.8% accuracy maintained

**Implementation**:
```python
# Accumulate images for 10-20ms
batch = accumulate_images(buffer_time=15ms)
# Process batch
scores = efficientad_infer_batch(batch)
# Distribute results
for img, score in zip(batch, scores):
    route_result(img, score)
```

**Requirement Profile: Edge-AI**

**Specifications**:
- Platform: Raspberry Pi 4 or Jetson Nano
- Power: <15W
- Latency: <100ms
- Accuracy: >95%

**Recommendation**: EfficientAD quantized
- INT8 quantization
- ONNX Runtime optimization
- Raspberry Pi: 60-80ms
- Jetson Nano: 15-25ms
- 97.5% accuracy (0.3%p loss)

**Implementation**:
```python
# Quantize model
model_int8 = quantize(model, calibration_data)
# Export ONNX
torch.onnx.export(model_int8, 'model.onnx')
# Deploy with ONNX Runtime
session = ort.InferenceSession('model.onnx')
# Infer
output = session.run(None, {'input': image})
```

**Requirement Profile: Multi-Category SaaS**

**Specifications**:
- Categories: 100-500
- Users: Thousands
- Cost: Revenue-constrained
- Accuracy: >97% average

**Recommendation**: EfficientAD centralized
- Single server (or small cluster)
- Load balanced
- 100-500 models loaded
- 2-10GB total memory
- $500-2000/month infrastructure

**Architecture**:
```
Load Balancer
    │
    ├─ Server 1 (Categories 1-100)
    ├─ Server 2 (Categories 101-200)
    ├─ Server 3 (Categories 201-300)
    └─ ...
```

**Requirement Profile: Research/Benchmarking**

**Specifications**:
- Reproducibility: Critical
- State-of-art: Both accuracy and speed
- Documentation: Complete
- Baselines: Multiple

**Recommendation**: RD + EfficientAD + PatchCore
- RD: Accuracy baseline (98.6%)
- EfficientAD: Speed baseline (1-5ms)
- PatchCore: SOTA baseline (99.1%)
- Compare all three
- Report comprehensive metrics

**Requirement Profile: Cost-Optimized**

**Specifications**:
- Budget: <$100 hardware per station
- Categories: 5-20
- Accuracy: >95%
- Maintenance: Minimal

**Recommendation**: EfficientAD on Raspberry Pi
- Hardware: $35 per station
- Setup: Plug-and-play
- Updates: OTA firmware
- Scales to thousands of stations
- Total cost: $35 × stations + negligible opex

## 7.3 Hardware Considerations

Hardware selection이 performance와 cost에 critical하다.

**GPU Options**

**Consumer GPUs**:

RTX 4090 (24GB, $1600):
- RD: 70-100ms, 10+ categories
- EfficientAD: 0.5-1ms, 200+ categories
- Best: All-around powerhouse

RTX 3090 (24GB, $1000):
- RD: 100-130ms, 8-10 categories
- EfficientAD: 1-2ms, 150+ categories
- Best: Balanced performance

RTX 3060 (12GB, $400):
- RD: 150-200ms, 4-6 categories
- EfficientAD: 2-3ms, 100+ categories
- Best: Budget GPU deployment

**Professional GPUs**:

A100 (40GB, $10000):
- RD: 50-70ms, 15+ categories
- EfficientAD: 0.3-0.5ms, 300+ categories
- Best: Data center, multi-tenant

T4 (16GB, $2000):
- RD: 120-150ms, 6-8 categories
- EfficientAD: 1.5-2.5ms, 120+ categories
- Best: Cloud deployment, inference-optimized

**CPU Options**

**High-End Desktop**:

Intel i9-12900K (16 cores, $600):
- RD: 600-800ms, impractical
- EfficientAD: 8-12ms, excellent
- Best: CPU-only deployment

AMD Ryzen 9 5950X (16 cores, $500):
- RD: 700-900ms, impractical
- EfficientAD: 10-15ms, excellent
- Best: CPU-only alternative

**Server CPUs**:

Intel Xeon Gold 6248 (20 cores, $3000):
- RD: 500-700ms per core
- EfficientAD: 6-10ms per core
- Best: Multi-category parallel processing
- Can run 20 categories simultaneously

AMD EPYC 7742 (64 cores, $7000):
- EfficientAD: 5-8ms per core
- Best: Massive parallelization
- 64 categories in parallel

**Edge Devices**

**Raspberry Pi 4 (4GB, $55)**:
- RD: >2000ms, impractical
- EfficientAD: 60-80ms, usable
- Best: Ultra-low-cost edge

**Jetson Nano (4GB, $99)**:
- RD: 300-500ms, marginal
- EfficientAD: 15-25ms, excellent
- Best: Budget edge AI

**Jetson Xavier NX (8GB, $399)**:
- RD: 150-200ms, usable
- EfficientAD: 5-10ms, excellent
- Best: Serious edge deployment

**Jetson AGX Xavier (32GB, $899)**:
- RD: 100-150ms, good
- EfficientAD: 3-5ms, excellent
- Best: High-end edge

**Mobile/Embedded**

**Smartphones (iPhone 13, Galaxy S21)**:
- RD: >1000ms, impractical
- EfficientAD: 10-20ms, good
- Best: Consumer applications

**Intel NUC (various configs, $300-1000)**:
- RD: 800-1200ms, marginal
- EfficientAD: 15-30ms, good
- Best: Compact industrial PC

**Hardware Selection Decision Tree**

```
Start
  │
  ├─ Need GPU acceleration?
  │  ├─ Yes
  │  │  ├─ Budget < $500
  │  │  │  └─ RTX 3060 or used GPU
  │  │  ├─ Budget < $1500
  │  │  │  └─ RTX 3090 or 4080
  │  │  └─ Budget unlimited
  │  │     └─ RTX 4090 or A100
  │  │
  │  └─ No (CPU only)
  │     ├─ Edge deployment
  │     │  ├─ Budget < $100
  │     │  │  └─ Raspberry Pi 4
  │     │  ├─ Budget < $500
  │     │  │  └─ Jetson Nano/Xavier NX
  │     │  └─ Budget < $1000
  │     │     └─ Jetson AGX Xavier
  │     │
  │     └─ Server deployment
  │        ├─ Multi-category
  │        │  └─ High core count CPU (Xeon, EPYC)
  │        └─ Single category
  │           └─ High frequency CPU (i9, Ryzen 9)
```

**Performance Scaling**

Hardware에 따른 performance scaling을 이해한다.

| Hardware | EfficientAD | RD | Cost/AUROC | Cost/FPS |
|----------|-------------|-----|------------|----------|
| RPi 4 | 15 FPS | 0.5 FPS | $3.5 | $3.7 |
| Jetson Nano | 50 FPS | 3 FPS | $2.0 | $2.0 |
| RTX 3060 | 400 FPS | 7 FPS | $1.0 | $1.0 |
| RTX 3090 | 800 FPS | 10 FPS | $1.25 | $1.25 |
| i9-12900K | 100 FPS | 1.5 FPS | $6.0 | $6.0 |

EfficientAD의 cost-efficiency가 일관되게 높다. Hardware choice에 덜 sensitive하다.

## 7.4 Deployment Strategies

Practical deployment strategies를 scenarios별로 제시한다.

**Strategy 1: Centralized Cloud**

**Architecture**:
```
Production Line
    │
    ├─ Camera 1 ─┐
    ├─ Camera 2 ─┼─ Network ─→ Cloud Server ─→ Dashboard
    ├─ Camera 3 ─┤              (GPU cluster)
    └─ Camera N ─┘
```

**Characteristics**:
- All processing in cloud
- Cameras stream images
- Centralized model management
- Easy updates and monitoring

**Best for**:
- RD deployment (requires GPU)
- Multiple facilities
- Centralized control desired
- Network reliable

**Cons**:
- Network dependency
- Latency (50-200ms network)
- Bandwidth requirements

**Strategy 2: Edge-First Hybrid**

**Architecture**:
```
Production Line
    │
    ├─ Camera + Edge Device ─→ Local Decision
    │   (EfficientAD)            │
    │                            ├─ Clear cases: Done
    │                            └─ Uncertain: → Cloud (RD)
```

**Characteristics**:
- Edge does primary screening
- 90% decided locally
- 10% escalated to cloud
- Low latency, low bandwidth

**Best for**:
- High-speed lines
- Unreliable network
- Cost-sensitive
- Hybrid accuracy needs

**Implementation**:
```python
# Edge device
score, confidence = efficientad_infer(image)
if confidence > 0.8:
    decision = "normal" if score < threshold else "anomaly"
    log_local(decision)
else:
    send_to_cloud(image, score)  # Async, non-blocking

# Cloud server
def handle_uncertain(image, edge_score):
    score_rd = rd_infer(image)
    decision = "normal" if score_rd < threshold else "anomaly"
    send_result_to_edge(decision)
```

**Strategy 3: Distributed Edge**

**Architecture**:
```
Factory Floor
    │
    ├─ Station 1: Edge Device (EfficientAD) ┐
    ├─ Station 2: Edge Device (EfficientAD) ├─ Local Network
    ├─ Station 3: Edge Device (EfficientAD) │     │
    └─ Station N: Edge Device (EfficientAD) ┘     │
                                                   ↓
                                          Local Server
                                          (Aggregation)
```

**Characteristics**:
- Each station has edge device
- All processing local
- Server aggregates statistics
- No single point of failure

**Best for**:
- EfficientAD deployment
- Distributed facilities
- Network unreliable
- Fault tolerance critical

**Strategy 4: Progressive Deployment**

**Phase 1** (Weeks 1-2): Pilot
- Single station
- EfficientAD on Jetson Nano
- Parallel with human inspection
- Collect data, validate

**Phase 2** (Weeks 3-4): Validation
- 3-5 stations
- Measure accuracy, speed
- Fine-tune thresholds
- Operator training

**Phase 3** (Weeks 5-8): Scale
- All stations (10-50)
- Automate reporting
- Integration with MES
- Performance monitoring

**Phase 4** (Month 3+): Optimize
- Analyze failure modes
- Retrain on new data
- Consider RD for difficult cases
- Continuous improvement

**Strategy 5: Multi-Tier Quality**

**Tier 1** (All products): EfficientAD screening
- Fast, automated
- 97.8% accuracy
- Flags suspicious items

**Tier 2** (Flagged items, 10%): RD verification
- Slower, precise
- 98.6% accuracy
- Reduces false positives

**Tier 3** (Still uncertain, 1%): Human expert
- Manual inspection
- 99.9% accuracy
- Final arbitration

**Result**: 99% automated, 99.5% accuracy, manageable workload.

**Monitoring and Maintenance**

Deployment 후 monitoring이 critical하다.

**Key Metrics to Track**:
- Inference latency (p50, p95, p99)
- Throughput (images/sec)
- Accuracy (if ground truth available)
- False positive rate
- False negative rate
- Model confidence distribution
- Hardware utilization (GPU/CPU/memory)
- Error rates and types

**Alerting Thresholds**:
```python
alerts = {
    'latency_p95': 50ms,  # 95th percentile latency
    'throughput': 30 FPS,  # Minimum throughput
    'error_rate': 1%,  # Inference errors
    'anomaly_rate': 20%,  # Sudden spike suspicious
    'confidence_low': 30%,  # Too many uncertain
}
```

**Retraining Triggers**:
- Accuracy drop > 2%
- Confidence distribution shift
- New defect types observed
- Process changes
- Quarterly scheduled retraining

**Update Strategy**:
```python
# Blue-green deployment
def update_model(new_model):
    # Deploy new model to "green" slot
    green_slot.load(new_model)
    
    # Test on sample traffic (10%)
    if validate(green_slot, test_traffic):
        # Gradual rollout: 10% → 50% → 100%
        router.shift_traffic(green_slot, percentage=10)
        monitor(hours=24)
        if metrics_ok():
            router.shift_traffic(green_slot, percentage=100)
            blue_slot.retire()
    else:
        green_slot.rollback()
```

**Practical Tips**

1. **Start small**: Pilot with 1-2 categories before scaling.

2. **Validate thoroughly**: Test on diverse samples including edge cases.

3. **Document everything**: Thresholds, configurations, performance baselines.

4. **Plan for failures**: Network outages, hardware failures, edge cases.

5. **Train operators**: Ensure team understands system capabilities and limitations.

6. **Iterate**: Continuous improvement based on field data and feedback.

7. **Measure ROI**: Track cost savings, quality improvements, efficiency gains.

Knowledge distillation-based anomaly detection은 mature하고 production-ready하다. RD와 EfficientAD가 spectrum의 양극단을 cover한다. Appropriate selection과 deployment strategy로 majority of industrial applications에 적용 가능하다. Key는 requirements를 정확히 파악하고 right tool을 선택하는 것이다.

# 8. Research Insights

## 8.1 Knowledge Distillation Duality

Knowledge distillation 패러다임의 발전은 흥미로운 duality를 드러낸다. 동일한 framework에서 출발했지만 정반대 방향으로 진화했다.

**The Original Vision**

Knowledge distillation은 원래 model compression 기법으로 시작되었다. Hinton et al.(2015)이 제안한 개념은 "large teacher의 knowledge를 small student로 transfer"였다. 목표는 deployment efficiency였다. 큰 모델의 성능을 작은 모델로 압축하는 것이다.

Anomaly detection에 적용될 때도 초기 vision은 유사했다. STFPM(2021)은 이 정신을 따랐다. Large pre-trained teacher(ResNet)와 비슷한 크기의 student를 사용했다. Efficiency보다는 feasibility demonstration이 목표였다. "Distillation이 anomaly detection에 작동하는가?"가 핵심 질문이었다.

그러나 2022-2024년 사이에 dramatic divergence가 발생했다. 두 개의 극단적인 방향이 등장했다. 하나는 accuracy maximization(RD), 다른 하나는 efficiency maximization(EfficientAD)이었다. 이들은 같은 paradigm의 서로 다른 manifestations였다.

**Precision Path: Reverse Distillation**

RD는 original vision과 정반대로 갔다. Teacher를 작게 만드는 대신 더 powerful하게 만들었다. Student를 고정하고 teacher를 학습시켰다. Domain-specific optimization이 목표였다. Generic knowledge가 아니라 specialized expertise를 추구했다.

이는 philosophical shift였다. "압축"에서 "전문화"로의 전환이었다. Teacher는 더 이상 압축의 source가 아니었다. 오히려 정상 패턴의 expert였다. Student(pre-trained CNN)는 general representation provider였다. Role reversal이 완전했다.

RD의 성공은 counter-intuitive했다. Conventional wisdom은 "bigger teacher, smaller student"였다. RD는 "fixed general student, trainable specialized teacher"였다. 이 inversion이 highest accuracy를 제공했다. 98.6% AUROC로 distillation methods 중 최고였다.

왜 이것이 작동했는가? Domain-specific learning의 힘이었다. Generic pre-trained features는 powerful하지만 target domain에 최적화되지 않았다. RD의 teacher는 정상 제품만 학습하여 극도로 specialized되었다. 이 specialization이 subtle anomalies를 detect하는 능력을 제공했다.

Trade-off는 명확했다. Specialization은 generalization을 희생했다. 각 category마다 separate teacher가 필요했다. Transfer learning이 어려웠다. Computational cost도 높았다. 그러나 accuracy가 최우선인 applications에서 이는 acceptable trade-off였다.

**Speed Path: EfficientAD**

EfficientAD는 original vision을 극단까지 밀어붙였다. Compression을 maximum으로 추구했다. 11.7M parameters(ResNet18)에서 50K parameters(PDN)로의 극적인 축소였다. 200배 이상의 compression이었다. 이는 aggressive했다.

EfficientAD의 철학은 "minimal sufficiency"였다. Task를 완수하는 minimum model이 무엇인가? Unnecessary parameters를 ruthlessly eliminate했다. Every parameter가 justify되어야 했다. Over-parameterization은 enemy였다.

이 극단적 minimalism이 작동한 것은 놀라웠다. 97.8% AUROC는 50K parameters로 달성되었다. 이는 tiny model임에도 불구하고 excellent였다. Task-specific design과 careful optimization의 결과였다.

EfficientAD의 성공은 중요한 insight를 제공한다. "Bigger is not always better." 실제로 anomaly detection에서 small model이 better generalization을 제공할 수 있다. Limited capacity가 overfitting을 방지한다. Essential patterns만 학습한다. 이는 implicit regularization이다.

Trade-off는 역시 존재했다. Extreme compression은 subtle information loss를 초래했다. Complex textures나 minute defects에서 약점을 보였다. Top-tier accuracy(99%+)는 달성하지 못했다. 그러나 vast majority of applications에서 97.8%는 충분했고 speed advantage가 압도적이었다.

**The Duality Principle**

RD와 EfficientAD의 대조는 fundamental duality를 reveal한다. 같은 paradigm(knowledge distillation)이 정반대 objectives(precision vs speed)를 optimize할 수 있다. 이는 paradigm의 versatility를 보여준다.

이 duality는 우연이 아니다. Distillation framework의 inherent flexibility 때문이다. Teacher-student relationship을 다양하게 configure할 수 있다. Teacher를 large/small, fixed/trainable, general/specialized로 만들 수 있다. Student도 마찬가지다. 이 configurational space가 diverse solutions를 enable한다.

Mathematically duality는 다음과 같이 표현된다:

$$\text{RD: } \min_{\theta_T} \mathbb{E}[\mathcal{L}(T_{\theta_T}(S_{\text{fixed}}(\mathbf{x})), \mathbf{x})] \quad \text{subject to high capacity } \theta_T$$

$$\text{EA: } \min_{\theta_S} \mathbb{E}[\mathcal{L}(S_{\theta_S}(T_{\text{fixed}}(\mathbf{x})), T_{\text{fixed}}(\mathbf{x}))] \quad \text{subject to low capacity } \theta_S$$

RD는 teacher를 optimize하고 capacity constraint가 없다. EfficientAD는 student를 optimize하고 tight capacity constraint가 있다. Same framework, opposite formulations.

이 duality는 practical implications를 가진다. Single paradigm으로 diverse needs를 address할 수 있다. Precision-critical applications은 RD를 선택한다. Speed-critical applications은 EfficientAD를 선택한다. Framework는 동일하고 configuration만 다르다. 이는 learning curve를 낮추고 code reuse를 enable한다.

**Philosophical Implications**

Duality는 deeper philosophical questions를 제기한다. "What is the essence of knowledge distillation?" Teacher-student relationship인가? Feature matching인가? Capacity constraint인가?

답은 "all of the above, but in different combinations"인 것 같다. Distillation의 essence는 single principle이 아니라 flexible framework이다. Core components(teacher, student, matching objective)가 있지만 이들을 다양하게 arrange할 수 있다. 이 flexibility가 power의 source다.

이는 broader lesson을 제공한다. Research에서 single optimal solution을 찾으려 하지 말아야 한다. 대신 solution space를 explore해야 한다. Extreme points(RD, EfficientAD)가 종종 most valuable하다. Middle ground는 compromised될 수 있다(다음 섹션 참조). Extremes가 clear value propositions를 제공한다.

## 8.2 The Middle Ground Trap

Knowledge distillation 발전 과정에서 "middle ground" approaches는 대부분 실패했다. 이는 중요한 lesson을 제공한다.

**STFPM: The Original Middle**

STFPM은 의도적으로 middle ground를 택했다. Teacher와 student가 같은 architecture(ResNet18)였다. Accuracy와 speed 모두 moderate였다. 95.5% AUROC와 50-100ms inference였다. Balanced approach로 보였다.

그러나 이 balance는 "best of both worlds"가 아니라 "worst of both worlds"에 가까웠다. Accuracy는 top-tier methods(PatchCore 99.1%, FastFlow 98.5%)에 크게 못 미쳤다. Speed도 충분히 빠르지 않았다(real-time threshold 30ms를 넘었다). Niche가 불명확했다.

STFPM이 valuable한 것은 baseline으로서였다. Distillation paradigm을 establish했다. 그러나 practical deployment에서는 suboptimal이었다. Better alternatives(RD for accuracy, EfficientAD for speed)가 등장하자 rapidly obsolete되었다.

**FRE: The Failed Middle**

FRE는 STFPM의 한계를 인식하고 개선을 시도했다. Feature reconstruction approach로 1%포인트 향상을 달성했다(96.5%). 그러나 이는 insufficient했다. 여전히 middle ground에 머물렀다. FastFlow(98.5%)나 RD(98.6%)에 크게 못 미쳤다.

더 문제는 complexity 증가였다. Encoder-decoder architecture가 STFPM보다 복잡했다. 추론도 느려졌다(80-120ms vs 50-100ms). Accuracy improvement가 complexity cost를 justify하지 못했다. Value proposition이 unclear했다.

FRE는 peer-reviewed publication에 실패했다. Community가 recognize하지 않았다. "Incremental improvement with added complexity"는 compelling contribution이 아니었다. Middle ground의 또 다른 victim이었다.

**Why Middle Fails**

Middle ground approaches가 fail하는 이유는 무엇인가? 여러 factors가 있다.

**1. Weak Value Proposition**

Middle ground는 어느 extreme에서도 best가 아니다. Accuracy가 필요하면 RD를 선택한다. Speed가 필요하면 EfficientAD를 선택한다. Middle을 선택할 compelling reason이 없다. "Reasonably accurate and reasonably fast"는 "best at accuracy" or "best at speed"보다 약한 proposition이다.

Application requirements는 종종 binary다. Either accuracy is critical or not. Either real-time is required or not. Few applications need "moderate accuracy and moderate speed". 이들은 보통 한 쪽으로 lean한다. Middle ground는 neither를 satisfy한다.

**2. Optimization Challenges**

Multi-objective optimization은 inherently difficult하다. Accuracy와 speed를 simultaneously optimize하려면 complex trade-offs가 필요하다. Single objective(accuracy only or speed only)가 훨씬 쉽다. Clear goal이 있고 모든 efforts를 그쪽으로 direct할 수 있다.

Middle ground는 compromises를 force한다. Accuracy를 위한 design choices가 speed를 hurt한다. Speed를 위한 simplifications가 accuracy를 degrade한다. Result는 lukewarm performance on both dimensions다. Neither exceptional하다.

**3. Implementation Complexity**

Middle ground는 종종 more complex하다. Both accuracy features(multi-scale, complex losses)와 efficiency features(lightweight layers, optimizations)를 integrate하려 한다. 이는 complicated architectures를 만든다. Debugging과 tuning이 어렵다.

Extreme approaches는 simpler하다. RD는 accuracy만 focus한다. Optimization techniques는 잘 established되어 있다. EfficientAD는 speed만 focus한다. Design principles가 명확하다. Implementation이 straightforward하다.

**4. Benchmarking Disadvantage**

Research community와 industry는 extremes를 recognize한다. "Best accuracy" or "fastest speed"는 clear achievements다. Paper에서 highlight하기 쉽고 users가 understand하기 쉽다. Middle ground는 highlight할 것이 없다. "Second best accuracy and second best speed"는 impressive하지 않다.

Marketing perspective에서도 extremes가 유리하다. "99% accurate" or "1ms inference"는 compelling headlines다. "97% accurate at 50ms"는 bland하다. Attention을 attract하지 못한다.

**The Pareto Front Phenomenon**

이 observations는 Pareto front 개념으로 formalize된다. Multi-objective space에서 Pareto-optimal points만 valuable하다. Dominated points는 irrelevant하다.

Knowledge distillation space에서:
- RD: Pareto-optimal on accuracy dimension
- EfficientAD: Pareto-optimal on speed dimension
- STFPM, FRE: Dominated by both RD and EfficientAD

Dominated points는 practical value가 없다. Academic novelty로도 insufficient하다. Research는 Pareto front를 push해야 한다. 기존 front 뒤에 있는 solutions는 contributions가 아니다.

**Exceptions: When Middle Works**

Middle ground가 work하는 rare cases가 있다. 특정 constraints가 있을 때다.

**Hardware-specific optimization**: Particular hardware(e.g., specific embedded chip)에 최적화된 middle approach가 value를 가질 수 있다. 해당 hardware에서 best performance를 제공하면 niche를 찾는다.

**Regulatory requirements**: 특정 accuracy threshold(e.g., 97%)를 요구하는 regulation이 있고 faster than alternatives면 valuable하다. Compliance + efficiency의 combination이 unique value를 만든다.

**Legacy system integration**: Existing infrastructure와의 compatibility가 critical하면 middle approach가 necessary할 수 있다. Technical debt나 integration costs가 extremes를 prohibitive하게 만든다.

그러나 이들은 exceptions다. General case에서 middle ground는 trap이다. Avoid해야 할 pitfall이다.

**Lessons for Future Research**

1. **Aim for extremes**: Accuracy champion이나 speed champion을 목표로 한다. Middle은 피한다.

2. **Clear positioning**: "Best at X" where X is single dimension. Multi-dimensional mediocrity를 피한다.

3. **Validate value proposition**: "Why would anyone choose this over alternatives?" Compelling answer가 없으면 proceed하지 말아야 한다.

4. **Pareto analysis**: New method를 existing Pareto front와 비교한다. Dominated면 rethink한다.

5. **Accept trade-offs**: Perfect solution은 없다. Extreme에서의 trade-offs를 embrace한다. Clear advantages를 articulate한다.

## 8.3 Revolutionary Optimization

EfficientAD의 성공은 revolutionary optimization의 power를 demonstrate한다. Incremental improvements가 아니라 radical redesign이 breakthrough를 만들었다.

**The 200× Compression**

EfficientAD의 가장 dramatic aspect는 compression ratio다. ResNet18 student(11.7M parameters)에서 PDN(50K parameters)으로의 전환은 200배 이상 축소였다. 이는 unprecedented였다. Conventional compression techniques(pruning, quantization)는 2-10배 reduction을 제공한다. 200배는 different category다.

이러한 extreme compression은 incremental approach로는 달성 불가능하다. Existing architecture를 prune하거나 simplify하는 것으로는 충분하지 않다. 전혀 다른 architecture paradigm이 필요했다. Ground-up redesign이었다.

PDN의 설계는 first principles에서 시작했다. "What is the minimum network that can describe teacher features on normal data?" Answer: 4 convolutional layers with small channel counts. 이는 conventional wisdom(deeper is better, wider is better)과 contrary했다. 그러나 작동했다.

**Task-Specific Architecture**

EfficientAD의 revolutionary aspect는 task specificity였다. Generic architectures(ResNet, VGG, EfficientNet)를 adapt하지 않았다. Anomaly detection에 specific한 architecture를 designed했다.

이는 paradigm shift였다. Most research는 existing architectures를 modify한다. Transfer learning, fine-tuning, adapter layers 등으로 adaptation한다. EfficientAD는 이 approach를 abandoned했다. Complete redesign을 chose했다.

Task-specific design이 가능했던 이유는 clear understanding of task requirements였다. Anomaly detection은 classification이 아니다. Image generation도 아니다. Feature matching on normal data다. 이는 specific characteristics를 가진다:
- Input: Pre-computed teacher features (not raw images)
- Output: Matched student features (same dimensions)
- Constraint: Normal data only
- Goal: Minimize matching error

이러한 characteristics를 deeply analyze하면 optimal architecture가 emerge한다. Spatial resolution preservation이 중요하다(no pooling). Simple transformations로 충분하다(shallow network). Channel-wise processing이 적합하다(no complex interactions). 이들이 PDN design을 informed했다.

**Dual-Path Innovation**

EfficientAD의 또 다른 innovation은 dual-path approach였다. Single model이 아니라 two complementary models이었다. PDN(semantic anomalies)과 autoencoder(appearance anomalies)가 각각 다른 aspects를 covered했다.

이는 ensemble과 다르다. Ensemble은 similar models를 combine한다. Diversity는 random initialization이나 different training data에서 나온다. EfficientAD의 dual paths는 fundamentally different approaches다. Feature matching vs reconstruction. Semantic vs appearance. 이들은 inherently complementary하다.

Dual-path의 genius는 specialization이었다. 각 path를 single objective에 optimize할 수 있었다. PDN은 teacher matching만, autoencoder는 image reconstruction만. Specialized models이 general model보다 efficient하다. 두 specialized models의 fusion이 robust performance를 제공했다.

**Optimization at All Levels**

EfficientAD는 every level에서 optimization을 performed했다. Architecture, training, inference 모두였다.

**Architecture level**:
- Minimal parameters (50K)
- No batch normalization (inference overhead)
- No skip connections (complexity)
- Simple operations only (ReLU, Conv)
- No exotic layers (custom ops)

**Training level**:
- Feature caching (skip teacher forward)
- Small batch training (fast convergence)
- Short epochs (10-30 epochs)
- Simple losses (L2, no complex terms)
- No heavy augmentation (minimal preprocessing)

**Inference level**:
- INT8 quantization (4× smaller, 2-4× faster)
- ONNX export (framework optimization)
- TensorRT compilation (kernel fusion)
- Batch processing (GPU utilization)
- CPU optimization (SIMD, cache-friendly)

이러한 optimizations는 cumulative effect를 냈다. 각각 10-50% improvement를 제공했다. Combined effect는 10-50배 speedup이었다. This is revolutionary optimization.

**The 1ms Barrier**

Sub-millisecond inference는 psychological barrier였다. 1ms는 "instantaneous"의 threshold다. Human perception에서 unnoticeable하다. Real-time의 new definition이었다.

EfficientAD가 이 barrier를 broke한 것은 significant였다. GPU에서 0.5-2ms를 달성했다. 이는 qualitative difference를 만들었다. Applications that were impossible became possible. 1000+ FPS processing이 reality가 되었다.

이 achievement는 incremental improvements의 accumulation이 아니었다. Radical redesign의 result였다. 200× smaller model, task-specific architecture, multi-level optimization의 combination이었다. Revolutionary approach만이 revolutionary results를 낳는다.

**Lessons for Breakthrough Research**

EfficientAD의 success에서 배우는 lessons:

**1. Question Assumptions**

Conventional wisdom을 challenge한다. "Neural networks need to be deep." "More parameters are better." "General architectures are optimal." 이들은 assumptions이지 facts가 아니다. Question하고 test한다.

EfficientAD는 모든 assumptions를 questioned했다. "Do we need deep networks? No, 4 layers suffice." "Do we need millions of parameters? No, 50K work." "Should we use standard architectures? No, custom is better." 이러한 questioning이 breakthroughs를 enabled했다.

**2. First Principles Thinking**

Existing solutions를 modify하지 말고 problem을 fundamental level에서 analyze한다. "What is truly necessary?" "What can be removed?" "What is the minimal solution?"

EfficientAD는 first principles를 applied했다. "Anomaly detection의 essence는 무엇인가?" "Teacher features를 describe하는 minimum network는?" "Bottleneck을 어디서 제거할 수 있는가?" 이러한 questions가 revolutionary design을 led했다.

**3. Extreme Goals**

Ambitious goals를 set한다. "10% faster"가 아니라 "10배 faster". "5% smaller"가 아니라 "100배 smaller". Extreme goals가 radical solutions를 force한다.

EfficientAD의 goal은 extreme이었다. "Real-time on edge devices." "Sub-millisecond inference." "1MB model." 이러한 extreme goals가 conventional approaches로는 unachievable했다. Revolutionary approach를 필요로 했다. 이것이 innovation을 drove했다.

**4. Multi-Level Optimization**

Single level optimization으로는 insufficient하다. All levels를 simultaneously optimize해야 한다. Architecture + training + inference의 holistic approach가 필요하다.

EfficientAD는 이를 exemplified했다. Architecture는 inference를 고려하여 designed되었다. Training은 fast convergence를 optimized했다. Inference는 hardware를 fully utilized했다. Synergy가 극대화되었다.

**5. Validate Radically**

Radical ideas는 thorough validation이 필요하다. Skepticism이 높을 것이다. Extensive experiments, ablations, comparisons로 convincingly demonstrate해야 한다.

EfficientAD는 comprehensive validation을 provided했다. Multiple datasets, hardware platforms, metrics로 tested했다. Ablation studies로 each component를 justified했다. Comparisons로 advantages를 clearly showed했다. 이것이 acceptance를 achieved했다.

**The Innovation Paradox**

EfficientAD는 innovation paradox를 illustrates한다. Most revolutionary ideas는 retrospectively obvious하다. "Of course we should design task-specific tiny networks!" 그러나 prospectively non-obvious했다. Nobody did it until EfficientAD.

이 paradox는 important insight를 provides한다. Innovation은 often simple하다. Complexity는 필요하지 않다. 오히려 simplification이 innovation이다. Seeing the obvious that everyone missed - 이것이 breakthrough다.

EfficientAD의 genius는 simplicity였다. 50K parameters, 4 layers, basic operations. 누구나 implement할 수 있다. 그러나 아무도 하지 않았다. 이것이 true innovation이다. Accessible yet transformative.

**Conclusion: The Revolutionary Mindset**

Revolutionary optimization은 mindset의 문제다. Incremental thinking을 reject하고 radical possibilities를 embrace한다. Conventional wisdom을 question하고 first principles에서 시작한다. Extreme goals를 set하고 multi-level optimization을 pursue한다.

Knowledge distillation의 evolution은 이 mindset의 importance를 demonstrates한다. STFPM과 FRE는 incremental approaches였다. Moderate improvements를 achieved했지만 limited impact를 had했다. RD와 EfficientAD는 revolutionary approaches였다. Extreme objectives를 pursued하고 paradigm-shifting results를 achieved했다.

Future research는 revolutionary mindset를 adopt해야 한다. Incremental은 insufficient하다. Market는 saturated되어 있다. Breakthroughs만이 attention을 attract한다. Revolutionary optimization이 path forward다.

Knowledge distillation paradigm은 이제 mature하다. Core principles가 established되었다. Extreme points(precision, speed)가 explored되었다. Next generation의 innovations는 new dimensions를 explore할 것이다. Few-shot learning, foundation model integration, multi-modal fusion 등이다. 그러나 lessons는 consistent할 것이다. Revolutionary thinking, extreme goals, multi-level optimization - 이들이 breakthroughs를 create할 것이다.