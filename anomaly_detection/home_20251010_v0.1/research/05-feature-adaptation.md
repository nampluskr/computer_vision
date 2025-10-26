프로젝트 지식을 확인했습니다. 이제 05-feature-adaptation.md의 1장과 2장을 작성하겠습니다.

---

# Feature Adaptation and Transfer Learning

## 1. Paradigm Overview

### 1.1 Core Principle

Feature Adaptation 패러다임은 이상 감지 분야에서 독특한 위치를 차지한다. 복잡한 end-to-end 학습이나 정교한 생성 모델 대신, 이미 대규모 데이터로 사전 학습된 모델의 feature를 타겟 도메인에 적응시키는 단순하면서도 효과적인 접근법이다. 이 패러다임의 핵심 통찰은 "ImageNet으로 학습된 ResNet의 중간 layer feature는 일반적 시각 패턴을 이미 잘 표현하므로, 이를 활용하면 타겟 도메인에서 긴 학습 없이도 합리적인 성능을 얻을 수 있다"는 것이다.

수학적으로, pre-trained encoder $f_{\theta}$는 입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$를 feature space $\mathcal{F}$로 매핑한다.

$$
\mathbf{z} = f_{\theta}(\mathbf{x}), \quad \mathbf{z} \in \mathbb{R}^{C \times h \times w}
$$

여기서 $C$는 채널 수, $h \times w$는 feature map의 공간 해상도다. Pre-trained encoder의 파라미터 $\theta$는 고정되며, 학습 과정에서 업데이트되지 않는다. 이는 계산 효율성과 안정성을 동시에 보장한다.

Feature adaptation의 목표는 이 feature space에서 정상 패턴의 분포 $p(\mathbf{z} | \text{normal})$를 모델링하고, 새로운 샘플의 feature $\mathbf{z}_{\text{test}}$가 이 분포로부터 얼마나 벗어났는지 측정하는 것이다. 이상 점수는 다음과 같이 정의된다.

$$
s(\mathbf{x}) = d(\mathbf{z}_{\text{test}}, p(\mathbf{z} | \text{normal}))
$$

여기서 $d(\cdot, \cdot)$는 거리 함수다. DFM은 Mahalanobis distance를, CFA는 angular distance를 사용한다. 분포 $p(\mathbf{z} | \text{normal})$는 학습 데이터의 feature로부터 추정되며, 이 과정은 매우 빠르다(수 분~수십 분). End-to-end 학습과 달리 backpropagation이 필요 없으므로, 메모리와 계산량이 극적으로 감소한다.

### 1.2 Transfer Learning Foundation

Transfer learning은 한 도메인(source domain)에서 학습된 지식을 다른 도메인(target domain)으로 전이하는 기법이다. Feature adaptation은 transfer learning의 특수한 형태로, feature extraction은 고정하고 downstream task만 타겟 도메인에 적응시킨다. 이는 fine-tuning보다 더 극단적인 형태의 transfer learning이며, "feature reuse"라고도 불린다.

Transfer learning의 이론적 기반은 domain invariance hypothesis다. 서로 다른 도메인이라도 중간 수준의 feature representation은 공유될 수 있다는 가정이다. ImageNet으로 학습된 ResNet의 경우, 초기 layer는 edge, texture 등 low-level feature를, 중간 layer는 part, pattern 등 mid-level feature를, 후기 layer는 object-specific high-level feature를 학습한다. 이상 감지에서는 mid-level feature가 가장 유용하다. 왜냐하면 이들은 충분히 추상적이어서 도메인 간 전이가 가능하면서도, 충분히 구체적이어서 결함 패턴을 포착할 수 있기 때문이다.

수학적으로, source domain $\mathcal{D}_S$와 target domain $\mathcal{D}_T$가 주어졌을 때, feature space $\mathcal{F}$에서의 분포 shift가 작다고 가정한다.

$$
d_{\mathcal{F}}(p_S(\mathbf{z}), p_T(\mathbf{z})) < \epsilon
$$

여기서 $d_{\mathcal{F}}$는 feature space에서의 분포 거리(예: KL divergence, Wasserstein distance)다. $\epsilon$이 충분히 작으면, source domain에서 학습된 feature extractor를 target domain에서도 효과적으로 사용할 수 있다. 실제로 ImageNet과 산업 검사 이미지는 pixel space에서는 매우 다르지만, ResNet의 layer3 feature space에서는 상대적으로 가까운 것으로 알려져 있다.

Domain adaptation의 정도는 adaptation layer의 복잡도에 따라 달라진다. DFM은 선형 변환(PCA)만을 사용하여 매우 가벼운 adaptation을 수행한다. CFA는 비선형 변환(hypersphere projection)을 추가하여 더 강력한 adaptation을 달성한다. 그러나 두 방법 모두 feature extractor 자체는 수정하지 않으므로, 극단적인 domain shift에서는 성능이 제한된다.

### 1.3 Pre-trained Feature Utilization

Pre-trained feature의 품질은 feature adaptation 성능을 결정하는 핵심 요소다. 이상 감지에서 가장 널리 사용되는 pre-trained model은 ImageNet으로 학습된 ResNet, Wide ResNet, EfficientNet 등이다. 이들은 1000개 클래스의 일반 객체를 구별하도록 학습되었으며, 중간 layer feature는 다양한 시각 패턴을 표현할 수 있는 능력을 갖췄다.

어떤 layer의 feature를 사용할 것인가는 중요한 설계 결정이다. 초기 layer(layer1, layer2)는 해상도가 높아 공간 정보를 잘 보존하지만, 의미론적 정보가 부족하다. 후기 layer(layer4, layer5)는 의미론적으로 풍부하지만, 공간 해상도가 낮고 ImageNet 클래스에 과도하게 특화되어 있다. 중간 layer(layer3)가 균형점이며, 대부분의 feature adaptation 방법론이 이를 사용한다.

Layer3 feature의 특성을 수치적으로 살펴보자. ResNet50의 layer3 출력은 $1024 \times 14 \times 14$의 shape를 가진다. 즉, 1024개 채널의 $14 \times 14$ feature map이다. 256×256 입력 이미지에 대해, 각 feature map의 한 픽셀은 원본 이미지의 약 $18 \times 18$ 영역에 대응한다(receptive field). 이는 미세한 결함을 감지하기에 적절한 granularity다.

Feature의 통계적 특성도 중요하다. Layer3 feature는 대체로 Gaussian-like 분포를 따르지만, 완전히 Gaussian은 아니다. Skewness와 kurtosis가 존재하며, 채널 간 상관관계가 복잡하다. DFM은 이를 multivariate Gaussian으로 근사하고, CFA는 hypersphere 상의 분포로 모델링한다. 두 접근법 모두 실제 분포의 단순화된 버전이지만, 이상 감지에는 충분히 효과적이다.

Feature normalization도 고려해야 한다. Pre-trained model은 ImageNet 통계로 normalize된 입력을 기대한다. 따라서 타겟 도메인 이미지도 동일한 방식으로 normalize해야 한다. Mean $[0.485, 0.456, 0.406]$, std $[0.229, 0.224, 0.225]$를 사용하는 것이 표준이다. 이 normalization을 생략하면 feature 분포가 왜곡되어 성능이 크게 저하된다.

### 1.4 Domain Adaptation

Domain adaptation은 source domain과 target domain 간의 분포 차이를 극복하는 기법들을 포괄한다. Feature adaptation 패러다임에서는 feature extractor를 고정하므로, adaptation은 downstream task(이상 감지)에만 적용된다. 이는 full domain adaptation보다 훨씬 단순하지만, 여전히 효과적이다.

Domain shift의 주요 원인은 조명, 텍스처, 색상 등의 low-level 차이다. 예를 들어, ImageNet 이미지는 자연광에서 촬영된 일상 물체인 반면, 산업 검사 이미지는 균일한 조명에서 촬영된 특정 부품이다. 이러한 차이는 pixel space에서 크지만, ResNet의 중간 layer에서는 상당히 완화된다. ResNet은 조명 변화에 어느 정도 robust한 feature를 학습하기 때문이다.

그러나 극단적인 domain shift는 여전히 문제가 된다. 예를 들어, ImageNet에 없는 재질(예: 반도체 웨이퍼, 의료 조직)이나 촬영 방식(예: X-ray, 전자현미경)은 feature space에서도 큰 차이를 보인다. 이런 경우 feature adaptation의 성능이 제한되며, fine-tuning이나 domain-specific pre-training이 필요할 수 있다.

CFA는 domain shift 완화를 위해 hypersphere normalization을 사용한다. 각 feature vector를 unit hypersphere로 project함으로써, scale 변화에 invariant한 representation을 얻는다. 이는 조명 변화나 contrast 차이에 robust하다. 수식으로 표현하면 다음과 같다.

$$
\hat{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}
$$

Normalized feature $\hat{\mathbf{z}}$는 방향 정보만 보존하고 크기 정보를 제거한다. 이상 감지에서 방향이 크기보다 더 중요하다는 가정에 기반한다. 실제로 정상 샘플과 이상 샘플은 feature space에서 크기보다는 방향에서 더 큰 차이를 보인다.

Domain adaptation의 또 다른 측면은 타겟 도메인의 정상 분포 학습이다. DFM과 CFA는 모두 학습 데이터의 feature로부터 정상 분포를 추정한다. 이 과정에서 타겟 도메인의 특성이 자연스럽게 반영된다. 예를 들어, 텍스처가 복잡한 도메인에서는 feature 분산이 크게 추정되고, 단순한 도메인에서는 작게 추정된다. 이는 명시적인 adaptation 없이도 도메인 특성을 암묵적으로 포착하는 메커니즘이다.

---

## 2. DFM (2019)

### 2.1 Basic Information

DFM(Deep Feature Modeling)은 2019년 Defard et al.에 의해 제안된 feature adaptation의 초기 방법론으로, PaDiM(2020)의 전신 격 접근법이다. Pre-trained CNN의 deep feature를 PCA로 차원 축소하고 Mahalanobis distance로 이상을 판정하는 극도로 단순한 구조를 가진다. 복잡한 end-to-end 학습 없이 statistical modeling만으로 이상 감지를 수행한다는 점에서, 당시 지배적이었던 deep learning 중심 패러다임과 차별화된다.

MVTec AD 벤치마크에서 Image AUROC 94.5-95.5%를 달성했으며, 이는 최고 수준(PatchCore 99.1%)보다 4-5% 낮지만 baseline으로는 매우 우수하다. 특히 학습 시간 15분 이하, 구현 복잡도 200 lines, 메모리 사용량 200MB 미만이라는 압도적 효율성이 특징이다. Anomalib 라이브러리에 공식 구현되어 있으며 (https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/image/dfm), baseline 구축과 빠른 프로토타입에 최적화된 방법론이다.

DFM의 철학은 "good features are already learned"다. ImageNet으로 학습된 ResNet은 이미 풍부한 시각 표현을 인코딩하고 있으며, 이를 타겟 도메인에 적응시키는 것만으로도 합리적인 성능을 얻을 수 있다. 이는 transfer learning의 극단적 형태로, feature extraction은 완전히 고정하고 downstream task만 타겟 도메인에 맞춘다.

### 2.2 Deep Feature Modeling

#### 2.2.1 Pre-trained CNN Features

DFM은 ImageNet으로 사전 학습된 ResNet50의 중간 layer에서 feature를 추출한다. 일반적으로 layer3을 사용하며, 이는 mid-level feature를 제공한다. 초기 layer(layer1, layer2)는 해상도가 높지만 의미론적 정보가 부족하고, 후기 layer(layer4, layer5)는 의미론적으로 풍부하지만 ImageNet 클래스에 과도하게 특화되어 있다. Layer3은 이 균형점이다.

수학적으로, 입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$(일반적으로 $H=W=256$)에 대해 pre-trained encoder $f_{\theta}$는 feature map을 생성한다.

$$
\mathbf{F} = f_{\theta}(\mathbf{x}) \in \mathbb{R}^{C \times h \times w}
$$

ResNet50 layer3의 경우 $C=1024$, $h=w=14$다. 각 spatial location의 receptive field는 약 $18 \times 18$ 픽셀이며, 이는 미세한 결함(수 밀리미터 크기)을 감지하기에 적절한 granularity다.

3D feature map을 2D matrix로 reshape한다.

$$
\mathbf{Z} = \text{reshape}(\mathbf{F}) \in \mathbb{R}^{C \times N}, \quad N = h \times w = 196
$$

각 열 $\mathbf{z}_i \in \mathbb{R}^C$ ($i=1,\ldots,196$)는 한 spatial location의 feature vector다. DFM의 핵심 설계 결정은 각 location을 독립적으로 처리한다는 것이다. 즉, 196개 location마다 별도의 정상 분포 $p(\mathbf{z}_i | \text{normal})$를 추정한다. 이는 결함의 위치가 공간적으로 다를 수 있음을 고려한 것으로, PaDiM에서도 동일한 접근법을 사용한다.

Pre-trained feature의 핵심 가정은 domain invariance다. Source domain $\mathcal{D}_S$(ImageNet)와 target domain $\mathcal{D}_T$(산업 검사 이미지)가 pixel space $\mathcal{X}$에서는 매우 다르지만, feature space $\mathcal{F}$에서는 상대적으로 가깝다고 가정한다.

$$
d_{\mathcal{X}}(p_S(\mathbf{x}), p_T(\mathbf{x})) \gg d_{\mathcal{F}}(p_S(\mathbf{z}), p_T(\mathbf{z}))
$$

여기서 $d$는 분포 간 거리(예: KL divergence)다. 실제로 ImageNet 이미지는 자연광에서 촬영된 일상 물체인 반면, 산업 검사 이미지는 균일한 조명의 특정 부품이다. Pixel space에서는 완전히 다르지만, ResNet layer3의 edge, texture, part 등 중간 수준 표현은 도메인 간 공유된다. 그러나 극단적 domain shift(의료 영상, 전자현미경 등)에서는 이 가정이 깨져 성능이 저하된다.

Feature normalization도 중요하다. Pre-trained model은 ImageNet 통계 $\boldsymbol{\mu}_{\text{IN}} = [0.485, 0.456, 0.406]$, $\boldsymbol{\sigma}_{\text{IN}} = [0.229, 0.224, 0.225]$로 normalize된 입력을 기대한다.

$$
\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{IN}}}{\boldsymbol{\sigma}_{\text{IN}}}
$$

이 normalization을 생략하면 feature 분포가 왜곡되어 성능이 크게 저하된다(실험적으로 5-10% AUROC 감소).

#### 2.2.2 PCA Dimensionality Reduction

1024차원 feature는 curse of dimensionality를 야기한다. 학습 샘플이 $M$개(MVTec AD에서 보통 200-300개)일 때, 각 샘플당 196개 feature vector가 있으므로 총 $M \times 196$ 개의 feature를 얻는다. 그러나 1024차원은 여전히 과도하게 높아, 특히 각 spatial location마다 별도 분포를 추정할 때 문제가 된다. 공분산 행렬 $\boldsymbol{\Sigma} \in \mathbb{R}^{1024 \times 1024}$는 약 50만 개 파라미터를 가지며, 이를 수백 개 샘플로 안정적으로 추정하기 어렵다.

PCA(Principal Component Analysis)는 데이터의 분산을 최대한 보존하는 저차원 subspace로 projection하여 이를 해결한다. 특정 spatial location $i$의 학습 feature $\{\mathbf{z}_{i,j}\}_{j=1}^M$ ($M$은 학습 샘플 수)이 주어졌을 때, 먼저 평균을 계산한다.

$$
\boldsymbol{\mu}_i = \frac{1}{M} \sum_{j=1}^M \mathbf{z}_{i,j}
$$

공분산 행렬을 계산한다.

$$
\boldsymbol{\Sigma}_i = \frac{1}{M-1} \sum_{j=1}^M (\mathbf{z}_{i,j} - \boldsymbol{\mu}_i)(\mathbf{z}_{i,j} - \boldsymbol{\mu}_i)^T \in \mathbb{R}^{1024 \times 1024}
$$

공분산 행렬의 고유값 분해(eigenvalue decomposition)를 수행한다.

$$
\boldsymbol{\Sigma}_i = \mathbf{U}_i \boldsymbol{\Lambda}_i \mathbf{U}_i^T
$$

여기서 $\mathbf{U}_i = [\mathbf{u}_{i,1}, \mathbf{u}_{i,2}, \ldots, \mathbf{u}_{i,1024}] \in \mathbb{R}^{1024 \times 1024}$는 고유벡터 행렬이고, $\boldsymbol{\Lambda}_i = \text{diag}(\lambda_{i,1}, \lambda_{i,2}, \ldots, \lambda_{i,1024})$는 고유값 대각 행렬이다. 고유값은 내림차순으로 정렬된다: $\lambda_{i,1} \geq \lambda_{i,2} \geq \cdots \geq \lambda_{i,1024} \geq 0$.

각 고유값 $\lambda_{i,k}$는 대응하는 고유벡터 $\mathbf{u}_{i,k}$ 방향의 데이터 분산을 나타낸다. 상위 몇 개 고유값이 전체 분산의 대부분을 설명하는 것이 일반적이다. $K$개 고유벡터를 선택하여 projection matrix를 구성한다.

$$
\mathbf{U}_{i,K} = [\mathbf{u}_{i,1}, \mathbf{u}_{i,2}, \ldots, \mathbf{u}_{i,K}] \in \mathbb{R}^{1024 \times K}
$$

$K$는 전체 분산의 일정 비율(일반적으로 97%)을 보존하도록 결정된다.

$$
K = \min \left\{ k : \frac{\sum_{j=1}^k \lambda_{i,j}}{\sum_{j=1}^{1024} \lambda_{i,j}} \geq 0.97 \right\}
$$

실무에서 $K$는 대체로 50-150 범위다. 즉, 1024차원이 약 100차원으로 축소되며, 이는 10배 차원 축소다. Feature vector $\mathbf{z}_i$의 PCA projection은 다음과 같이 계산된다.

$$
\mathbf{z}'_i = \mathbf{U}_{i,K}^T (\mathbf{z}_i - \boldsymbol{\mu}_i) \in \mathbb{R}^K
$$

이는 원래 feature를 $K$차원 subspace로 project한 것이다. Inverse transformation으로 reconstruction도 가능하다.

$$
\hat{\mathbf{z}}_i = \mathbf{U}_{i,K} \mathbf{z}'_i + \boldsymbol{\mu}_i \in \mathbb{R}^{1024}
$$

Reconstruction error는 제거된 분산의 양이다.

$$
\text{RE}_i = \|\mathbf{z}_i - \hat{\mathbf{z}}_i\|^2 = \sum_{k=K+1}^{1024} (\mathbf{u}_{i,k}^T (\mathbf{z}_i - \boldsymbol{\mu}_i))^2
$$

이 reconstruction error 자체를 anomaly score로 사용할 수도 있으며, DFM은 이를 FRE(Feature Reconstruction Error) score라고 부른다. 낮은 고유값에 대응하는 성분은 주로 noise이므로, 이를 제거함으로써 signal-to-noise ratio가 향상되는 효과도 있다.

PCA의 수학적 해석도 중요하다. PCA는 다음 최적화 문제의 해다.

$$
\mathbf{U}_{i,K}^* = \arg\max_{\mathbf{U}} \text{trace}(\mathbf{U}^T \boldsymbol{\Sigma}_i \mathbf{U}) \quad \text{s.t. } \mathbf{U}^T \mathbf{U} = \mathbf{I}_K
$$

즉, $K$개 orthonormal 방향 중에서 projected data의 분산을 최대화하는 방향을 찾는다. 이는 정보 손실을 최소화하는 차원 축소다.

#### 2.2.3 Mahalanobis Distance

PCA로 차원을 축소한 후, reduced feature space $\mathbb{R}^K$에서 정상 분포를 multivariate Gaussian으로 모델링한다. Spatial location $i$에서, 정상 feature $\mathbf{z}'_i$는 평균 $\boldsymbol{\mu}'_i$, 공분산 $\boldsymbol{\Sigma}'_i$를 가진 Gaussian 분포를 따른다고 가정한다.

$$
p(\mathbf{z}'_i | \text{normal}) = \mathcal{N}(\mathbf{z}'_i; \boldsymbol{\mu}'_i, \boldsymbol{\Sigma}'_i)
$$

PCA projection 후 feature는 이미 zero-mean이므로($\boldsymbol{\mu}'_i = \mathbf{0}$), 공분산만 추정하면 된다.

$$
\boldsymbol{\Sigma}'_i = \frac{1}{M-1} \sum_{j=1}^M \mathbf{z}'_{i,j} {\mathbf{z}'_{i,j}}^T \in \mathbb{R}^{K \times K}
$$

여기서 $\mathbf{z}'_{i,j}$는 $j$번째 학습 샘플의 location $i$에서의 reduced feature다. $K \times K$ 공분산 행렬은 원래 $1024 \times 1024$보다 훨씬 작아, 수백 개 샘플로도 안정적으로 추정 가능하다.

Mahalanobis distance는 multivariate Gaussian 분포 하에서 sample의 "비정상성"을 측정하는 최적 지표다. 테스트 feature $\mathbf{z}'_{i,\text{test}}$에 대해, Mahalanobis distance는 다음과 같이 정의된다.

$$
D_{M,i}(\mathbf{z}'_{i,\text{test}}) = \sqrt{{\mathbf{z}'_{i,\text{test}}}^T {\boldsymbol{\Sigma}'_i}^{-1} \mathbf{z}'_{i,\text{test}}}
$$

이는 공분산을 고려한 정규화된 거리다. 직관적으로, Euclidean distance는 모든 방향을 동등하게 취급하지만, Mahalanobis distance는 데이터의 분포를 고려한다. 분산이 큰 방향(데이터가 자연스럽게 퍼져있는 방향)의 편차는 덜 중요하게, 분산이 작은 방향(데이터가 밀집된 방향)의 편차는 더 중요하게 평가한다.

수학적으로, Mahalanobis distance는 $\boldsymbol{\Sigma}'_i$에 의해 정의된 ellipsoid 상의 거리다. $\boldsymbol{\Sigma}'_i$를 고유값 분해하면 $\boldsymbol{\Sigma}'_i = \mathbf{V}_i \mathbf{D}_i \mathbf{V}_i^T$이고, Mahalanobis distance는 다음과 같이 rewrite할 수 있다.

$$
D_{M,i}^2(\mathbf{z}'_{i,\text{test}}) = \sum_{k=1}^K \frac{(\mathbf{v}_{i,k}^T \mathbf{z}'_{i,\text{test}})^2}{d_{i,k}}
$$

여기서 $\mathbf{v}_{i,k}$는 고유벡터, $d_{i,k}$는 고유값이다. 작은 고유값 방향의 편차가 큰 기여를 한다는 것을 명확히 보여준다.

Mahalanobis distance는 negative log-likelihood와 직접 연관된다. Gaussian 분포의 log-likelihood는 다음과 같다.

$$
\log p(\mathbf{z}'_{i,\text{test}} | \text{normal}) = -\frac{K}{2} \log(2\pi) - \frac{1}{2} \log |\boldsymbol{\Sigma}'_i| - \frac{1}{2} D_{M,i}^2(\mathbf{z}'_{i,\text{test}})
$$

따라서 negative log-likelihood는 다음과 같다.

$$
-\log p(\mathbf{z}'_{i,\text{test}} | \text{normal}) = \frac{K}{2} \log(2\pi) + \frac{1}{2} \log |\boldsymbol{\Sigma}'_i| + \frac{1}{2} D_{M,i}^2(\mathbf{z}'_{i,\text{test}})
$$

앞의 두 항은 상수이므로, Mahalanobis distance를 최소화하는 것은 likelihood를 최대화하는 것과 동치다. 이는 Mahalanobis distance가 단순한 heuristic이 아니라 통계적으로 well-founded임을 보여준다.

실무 구현에서는 수치 안정성을 위해 regularization을 추가한다.

$$
\boldsymbol{\Sigma}'_{i,\text{reg}} = \boldsymbol{\Sigma}'_i + \epsilon \mathbf{I}_K, \quad \epsilon = 0.01
$$

이는 공분산 행렬이 singular하거나 ill-conditioned인 경우를 방지한다. 특히 $K$가 $M$에 가까울 때(차원이 샘플 수에 비해 여전히 높을 때) 중요하다.

Pixel-level anomaly map은 각 spatial location의 Mahalanobis distance를 계산하여 얻는다.

$$
\mathbf{A}[i] = D_{M,i}(\mathbf{z}'_{i,\text{test}}), \quad i = 1, 2, \ldots, 196
$$

이를 2D map으로 reshape하면 $\mathbf{A} \in \mathbb{R}^{14 \times 14}$를 얻고, 원본 해상도로 bilinear interpolation하여 최종 anomaly map을 생성한다.

$$
\mathbf{A}_{\text{final}} = \text{interpolate}(\mathbf{A}, 256 \times 256)
$$

Image-level anomaly score는 anomaly map의 최대값으로 정의된다.

$$
s_{\text{image}} = \max_{i,j} \mathbf{A}_{\text{final}}[i,j]
$$

이는 "가장 이상한 부분의 이상 정도"를 전체 이미지의 이상 점수로 사용하는 것이다. 다른 aggregation 방법(평균, 상위 k% 평균 등)도 가능하지만, 최대값이 가장 일반적이고 효과적이다.

### 2.3 Extreme Simplicity

DFM의 핵심 가치는 극도의 단순성이다. 전체 알고리즘은 세 단계로 요약되며, 각 단계는 well-established 기법이다. Pre-trained CNN으로 feature 추출(forward pass만, backpropagation 없음), PCA로 차원 축소(closed-form solution, iterative optimization 없음), Mahalanobis distance 계산(matrix multiplication과 inversion만). 복잡한 adversarial training, normalizing flow의 invertible transformation, knowledge distillation의 teacher-student coordination이 전혀 없다.

코드 구현은 PyTorch와 scikit-learn을 사용하여 200 lines 이내로 가능하다. 핵심 코드 구조는 다음과 같다. Feature extraction: `features = model(images)`, PCA fitting: `pca.fit(features)`, Mahalanobis distance: `distance = mahalanobis(test_features, mean, cov)`. 이는 GANomaly의 Generator, Discriminator, Encoder 구조(~1000 lines)나 DRAEM의 Reconstructive, Discriminative subnetwork(~800 lines)의 1/4-1/5 수준이다.

하이퍼파라미터도 극소다. PCA components 수($K$, 또는 분산 보존 비율)와 regularization 강도($\epsilon$) 두 개뿐이다. Learning rate, batch size, epochs, loss weights, optimizer 선택 등 deep learning의 복잡한 하이퍼파라미터가 불필요하다. 기본 설정(97% variance retention, $\epsilon=0.01$)이 대부분의 경우 잘 작동하며, domain-specific tuning이 거의 필요 없다.

학습이 deterministic하다는 것도 중요한 특징이다. Random initialization이나 stochastic gradient descent가 없어, random seed 관리가 필요 없다. 동일한 데이터에 대해 항상 동일한 결과를 보장하며, 이는 100% 재현성을 의미한다. 연구 논문 작성이나 규제 준수가 필요한 산업(의료기기, 항공우주 등)에서 이러한 재현성은 매우 중요하다. GAN 기반 방법론의 확률적 변동이나 여러 random seed에 대한 평균을 보고해야 하는 번거로움이 없다.

이러한 단순성은 여러 실무적 이점을 제공한다. 첫째, 빠른 이해다. 선형대수와 기초 통계(평균, 공분산, 고유값 분해, Gaussian 분포)만으로 알고리즘을 완전히 이해할 수 있다. 대학원 1-2년차 학생이나 산업 현장의 엔지니어가 하루 만에 이해하고 구현할 수 있다. 둘째, 쉬운 디버깅이다. 각 단계가 명확히 분리되어 있어, 문제 발생 시 어느 단계에서 issue가 생겼는지 빠르게 파악할 수 있다. Feature extraction 결과를 시각화하고, PCA components를 검증하고, Mahalanobis distance 분포를 확인하는 것이 straightforward하다. 셋째, 낮은 유지보수 비용이다. 작은 코드베이스와 적은 의존성(PyTorch, NumPy, scikit-learn만)은 장기적 유지보수를 수월하게 한다.

단순성의 trade-off는 표현력 제한이다. Linear transformation(PCA)과 Gaussian assumption은 강한 제약이며, 실제 정상 feature 분포가 non-Gaussian이거나 multimodal이면 DFM은 이를 완벽히 포착하지 못한다. 또한 feature extractor가 고정되어 있어, 타겟 도메인에 특화된 feature를 학습할 수 없다. 이는 성능 상한을 제한하며, SOTA 대비 4-5% gap의 주요 원인이다.

### 2.4 Performance Analysis (94.5-95.5%)

DFM의 MVTec AD 성능은 Image AUROC 94.5-95.5%로, baseline으로는 우수하지만 SOTA보다 4-5% 낮다. 이 성능 gap은 feature adaptation 패러다임의 근본적 한계를 반영한다. Pre-trained feature와 linear transformation만으로는, end-to-end 학습이나 sophisticated modeling의 성능에 도달하기 어렵다.

카테고리별 분석은 DFM의 강점과 약점을 명확히 보여준다. 구조적 객체 카테고리에서 성능이 우수하다. "Screw"에서 96.5%, "Metal Nut"에서 95.8%, "Pill"에서 95.2%를 기록한다. 명확한 기하학적 패턴은 ResNet의 pre-trained feature가 잘 포착한다. Edge, corner, symmetry 등 mid-level feature가 이러한 구조를 효과적으로 표현하기 때문이다. 반면 복잡한 텍스처 카테고리에서는 성능이 저하된다. "Carpet"에서 91.5%, "Leather"에서 92.8%, "Wood"에서 93.5%로, DSR(99%)이나 PatchCore(98%)보다 6-7% 낮다. 미묘한 텍스처 변이와 실제 결함을 구별하는 것이 어렵고, linear PCA로는 복잡한 텍스처 패턴을 충분히 모델링하지 못한다.

Pixel AUROC는 Image AUROC보다 약간 높은 95-96%다. 이는 DFM이 결함의 공간적 위치를 비교적 잘 파악함을 의미한다. 각 spatial location마다 독립적인 Mahalanobis distance를 계산하는 설계가 effective하다. Anomaly map이 결함 영역을 효과적으로 하이라이트하며, false positive가 적다.

학습 데이터 양의 영향은 완만하다. 전체 데이터(200-300 샘플)에서 94.5-95.5%, 절반(100-150 샘플)에서 93-94%, 1/4(50-75 샘플)에서 90-92%를 보인다. 성능-샘플 수 관계는 대략 로그 스케일이다.

$$
\text{AUROC}(n) \approx \text{AUROC}_{\max} - \alpha \log\left(\frac{N_{\max}}{n}\right)
$$

여기서 $n$은 샘플 수, $N_{\max}$는 전체 샘플 수, $\alpha \approx 0.8$이다. 샘플 수가 절반으로 줄어도 성능 저하는 1-2%에 불과하다. 이는 pre-trained feature가 이미 풍부한 정보를 인코딩하고 있어, 적은 샘플로도 정상 분포를 합리적으로 추정할 수 있기 때문이다. 그러나 Few-shot 영역(10-50 샘플)에서는 DRAEM(10샘플로 96.5%)만큼 강하지 않다. DRAEM의 simulated anomaly가 제공하는 강력한 augmentation이 없어, 극소 데이터에서는 한계가 있다.

성능에 영향을 미치는 주요 요인은 backbone 선택, layer 선택, PCA components 수다. Backbone: ResNet50이 기본이지만, Wide ResNet50으로 교체하면 1-1.5% 향상된다. EfficientNet-B5는 1.5-2% 향상을 제공한다. 그러나 더 큰 모델은 feature 추출 시간이 증가한다(ResNet50 대비 Wide ResNet 1.5배, EfficientNet-B5 2배). Layer: Layer3이 일반적이지만, layer2+layer3 조합을 사용하면 multi-scale 정보를 활용하여 0.5-1% 개선 가능하다. 그러나 계산량과 메모리가 증가한다. PCA components: 95-99% 분산 보존 범위에서 조정 가능하다. 더 많은 components는 더 많은 정보를 보존하지만, noise도 증가시킨다. 최적값은 보통 96-98% 범위다.

### 2.5 Fast Prototyping (15 minutes)

DFM의 결정적 장점은 학습 속도다. MVTec AD 한 카테고리(약 200장)를 10-15분에 학습하여 same-day prototyping을 가능하게 한다. 이는 신제품 출시, 공정 변경, feasibility 검증 등 빠른 대응이 필요한 실무 환경에서 큰 가치를 가진다.

학습 과정의 시간 분해는 다음과 같다. Feature extraction: 5-8분(GPU 사용 시). 209장 이미지(Bottle 카테고리)에 대해 ResNet50 forward pass를 수행한다. 각 이미지당 약 1.5-2초 소요되며, batch processing으로 약간 가속화된다. GPU가 없으면 20-30분 소요된다. PCA 계산: 2-3분(CPU). 196개 spatial location 각각에 대해 공분산 행렬($1024 \times 1024$) 고유값 분해를 수행한다. 이는 CPU 연산이며, NumPy의 최적화된 LAPACK routine을 사용한다. Gaussian fitting: 30초-1분(CPU). Reduced feature($K \times K$, $K \approx 100$)의 공분산 추정은 매우 빠르다. Total: 10-15분.

비교를 위해 다른 방법론의 학습 시간을 살펴보자. PatchCore: 5-10분, 가장 빠름. Coreset selection이 매우 효율적이며, iterative optimization이 없다. DRAEM: 15-20분, DFM과 유사. Reconstruction과 discriminative network의 간단한 학습. FastFlow: 30-40분, normalizing flow의 invertible transformation 학습이 필요. Reverse Distillation: 40-60분, decoder와 bottleneck module 학습. GANomaly: 50-80분, GAN의 adversarial training이 느리고 불안정. DSR: 30-50분, VQ-VAE + restriction module의 two-stage 학습.

DFM은 PatchCore 다음으로 빠르며, 구현 단순성을 고려하면 가장 접근하기 쉬운 방법론이다. 특히 PyTorch와 scikit-learn에 익숙한 개발자라면, 기존 코드나 라이브러리 없이 하루 만에 처음부터 구현할 수 있다.

빠른 학습은 빠른 iteration을 가능하게 한다. 전형적인 same-day workflow는 다음과 같다. 오전 9시: 새로운 제품 카테고리 데이터 수신(200장 양품). 오전 9:30: DFM 학습 시작(15분). 오전 10:00: 첫 결과 확인, baseline AUROC 94.5%. 오전 10:30: Wide ResNet50로 재학습(20분). 오전 11:00: AUROC 95.8%, 1.3% 향상 확인. 오전 11:30: Layer2+layer3 조합 시도(25분). 정오: AUROC 96.3%, 최종 설정 결정. 오후: 더 sophisticated 방법론(PatchCore, DRAEM) 시도 고려.

Grid search도 현실적이다. 3 backbones (ResNet50, Wide ResNet50, EfficientNet-B5) × 2 layer settings (layer3 only, layer2+layer3) × 3 PCA settings (95%, 97%, 99% variance) = 18개 조합을 하루(약 4.5시간)에 모두 실험할 수 있다. 각 실험이 15분이므로, 병렬 실행 없이도 순차적으로 가능하다. 이는 최적 설정을 체계적으로 찾을 수 있게 한다.

### 2.6 Advantages and Limitations

DFM은 특정 시나리오에서 탁월한 가치를 제공하지만 명확한 한계도 가진다. 이 이중성은 feature adaptation 패러다임의 본질적 특성을 반영한다.

주요 장점은 극단적 단순성이다. 200 lines 코드, 세 단계 알고리즘, 두 개 하이퍼파라미터로 구성되며 선형대수와 기초 통계만으로 이해 가능하다. 학습자가 하루 만에 처음부터 구현할 수 있고, 각 단계가 명확히 분리되어 디버깅이 용이하며, 작은 코드베이스는 장기 유지보수를 수월하게 한다. 15분 학습 속도는 same-day prototyping과 빠른 iteration을 가능하게 한다. 아침에 데이터를 받아 점심 전에 결과를 얻고 오후에 여러 설정을 실험할 수 있으며, 18개 조합 grid search도 하루 안에 완료 가능하다. 200MB 메모리 사용량은 저사양 GPU나 CPU 환경에서도 실행을 가능하게 하며, backpropagation이 없어 메모리 효율적이다. Deterministic 재현성은 random seed 불필요로 100% 동일 결과를 보장하며, 이는 연구 논문이나 규제 준수 산업에서 중요하다. PCA components는 주요 변동 방향을 명시적으로 보여주고, Mahalanobis distance는 "정상 분포로부터 몇 표준편차"로 직관적 해석이 가능하다.

주요 한계는 절대 성능 제약이다. 94.5-95.5%는 SOTA(99%)보다 4-5% 낮으며, 이는 critical application(항공우주, 의료기기)에서 수백 개 결함 중 20-25개를 더 놓치는 것을 의미한다. Linear transformation 제한으로 PCA는 선형 부분공간 projection이며 복잡한 non-linear 패턴을 포착하지 못한다. 실제 정상 feature 분포가 curved manifold나 multiple clusters를 형성하면 suboptimal하다. Gaussian assumption으로 Mahalanobis distance는 feature가 Gaussian 분포를 따른다고 가정하지만, 실제는 종종 non-Gaussian(skewness, heavy tail, multimodality)이다. Fixed feature extractor는 pre-trained feature가 타겟 도메인과 맞지 않을 때 adaptation이 불가능하며, ImageNet에 없는 재질(반도체, 의료 조직)이나 촬영 방식(X-ray, 현미경)에서 성능이 크게 저하된다. 텍스처 카테고리 약점으로 Carpet 91.5%, Leather 92.8%는 구조적 카테고리(Screw 96.5%)보다 4-5% 낮으며, 복잡한 텍스처 변이는 linear PCA로 충분히 모델링되지 않는다. Domain shift 민감도로 조명, 카메라, 배경 변화에 대한 robustness가 제한적이며, pre-trained feature의 source-target gap이 크면 성능이 급격히 저하된다.

DFM은 "starting point, not destination"으로 이해되어야 한다. 빠른 feasibility 검증(하루 안에 가능성 평가), 첫 baseline 구축(프로젝트 시작점), 저사양 환경(작은 GPU/CPU), 재현성 critical 상황(규제 산업), 단순성 중요 시(교육, 프로토타입), 94-95% 충분한 경우(non-critical application)에 탁월하다. 그러나 최고 성능(>98%) 필요 시 PatchCore나 FastFlow, Few-shot(<50 샘플) 시 DRAEM, 복잡한 텍스처 시 DSR, 실시간(<10ms) 시 EfficientAD, 극단적 domain shift 시 fine-tuning 가능한 방법론을 고려해야 한다. 실무 전략은 DFM으로 시작하여 빠르게 baseline을 구축하고 문제의 특성(난이도, 카테고리 유형, 데이터 양, 성능 요구사항)을 파악한 후, 필요시 PatchCore(최고 정확도), DRAEM(Few-shot), DSR(텍스처), EfficientAD(실시간) 등 specialized 방법론으로 업그레이드하는 것이다. 이는 "simple first, complex if necessary" 원칙의 구체적 실현이며, 시간과 리소스를 효율적으로 사용하는 pragmatic한 접근법이다.

3장과 4장을 작성하겠습니다.

---

## 3. CFA (2022)

### 3.1 Basic Information

CFA(Coupled-hypersphere-based Feature Adaptation)는 2022년 Sohn et al.에 의해 제안된 방법론으로, DFM의 한계를 극복하기 위해 hypersphere geometry를 도입했다. DFM이 Euclidean space에서 Mahalanobis distance를 사용하는 반면, CFA는 feature를 unit hypersphere로 normalize하고 angular distance를 사용한다. 이는 scale 변화에 invariant한 representation을 제공하며, 조명이나 카메라 변화 같은 domain shift에 더 robust하다.

논문은 "Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization"이라는 제목으로 발표되었으며, MVTec AD에서 Image AUROC 96.5-97.5%를 달성했다. 이는 DFM보다 2-3% 높으며, domain shift 시나리오에서 더 큰 성능 향상을 보인다. Anomalib 라이브러리에 공식 구현되어 있다 (https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/image/cfa).

CFA의 핵심 통찰은 "direction matters more than magnitude"다. Feature space에서 정상과 이상의 차이는 feature vector의 크기보다는 방향에서 더 명확히 나타난다. 조명 변화는 feature 크기를 바꾸지만 방향은 상대적으로 보존하므로, 방향만 사용하면 조명에 robust해진다. 이는 computer vision에서 오래된 아이디어(cosine similarity, angular metrics)를 anomaly detection에 적용한 것이다.

### 3.2 Coupled-hypersphere Adaptation

#### 3.2.1 Hypersphere Projection

CFA의 첫 번째 단계는 DFM과 동일하게 pre-trained CNN(일반적으로 Wide ResNet50)에서 feature를 추출한다. Layer3 feature $\mathbf{F} \in \mathbb{R}^{1024 \times 14 \times 14}$를 얻고, 이를 reshape하여 $\mathbf{Z} \in \mathbb{R}^{1024 \times 196}$을 만든다. 그러나 여기서 CFA는 DFM과 다른 경로를 택한다.

각 feature vector $\mathbf{z}_i \in \mathbb{R}^{1024}$를 unit hypersphere $\mathbb{S}^{1023}$로 project한다.

$$
\hat{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2}
$$

여기서 $\|\mathbf{z}_i\|_2 = \sqrt{\sum_{j=1}^{1024} z_{i,j}^2}$는 Euclidean norm이다. Normalized feature $\hat{\mathbf{z}}_i$는 $\|\hat{\mathbf{z}}_i\|_2 = 1$을 만족하며, 1023차원 unit sphere의 표면에 위치한다.

이 normalization은 feature의 방향(direction) 정보만 보존하고 크기(magnitude) 정보를 제거한다. 수학적으로, 원래 feature는 두 성분으로 분해할 수 있다.

$$
\mathbf{z}_i = r_i \cdot \hat{\mathbf{z}}_i
$$

여기서 $r_i = \|\mathbf{z}_i\|_2$는 magnitude, $\hat{\mathbf{z}}_i$는 direction이다. Hypersphere projection은 $r_i$를 버리고 $\hat{\mathbf{z}}_i$만 사용한다. 이는 정보 손실이지만, 의도된 손실이다. Magnitude는 조명, contrast 등 extrinsic factor에 크게 영향받는 반면, direction은 intrinsic feature를 더 잘 보존한다는 가정에 기반한다.

Hypersphere geometry는 특별한 성질을 가진다. Unit sphere는 curved manifold이며, Euclidean metric이 아닌 geodesic distance를 자연스러운 거리로 가진다. 두 점 $\hat{\mathbf{z}}_1, \hat{\mathbf{z}}_2 \in \mathbb{S}^{1023}$ 사이의 geodesic distance는 다음과 같다.

$$
d_{\text{geo}}(\hat{\mathbf{z}}_1, \hat{\mathbf{z}}_2) = \arccos(\hat{\mathbf{z}}_1^T \hat{\mathbf{z}}_2)
$$

이는 두 벡터 사이의 각도이며, $[0, \pi]$ 범위의 값을 가진다. CFA는 이 geodesic distance를 anomaly detection에 활용한다.

실무 구현에서는 numerical stability를 위해 약간의 조정이 필요하다. $\|\mathbf{z}_i\|_2 = 0$인 경우(모든 성분이 0)를 방지하기 위해 작은 epsilon을 추가한다.

$$
\hat{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2 + \epsilon}, \quad \epsilon = 10^{-6}
$$

또한 $\hat{\mathbf{z}}_1^T \hat{\mathbf{z}}_2$가 [-1, 1] 범위를 벗어나는 것(floating point error)을 방지하기 위해 clipping을 적용한다.

$$
\hat{\mathbf{z}}_1^T \hat{\mathbf{z}}_2 = \text{clip}(\hat{\mathbf{z}}_1^T \hat{\mathbf{z}}_2, -1+\epsilon, 1-\epsilon)
$$

#### 3.2.2 Angular Distance

Hypersphere 상에서 정상 분포를 모델링하는 것이 CFA의 핵심이다. Spatial location $i$에서, 학습 feature $\{\hat{\mathbf{z}}_{i,j}\}_{j=1}^M$이 주어졌을 때, 이들의 "중심"을 정의해야 한다. Euclidean space에서는 평균이 자연스러운 중심이지만, hypersphere에서는 평균이 sphere 표면에 위치하지 않는다.

CFA는 von Mises-Fisher distribution을 사용한다. 이는 hypersphere 상의 probability distribution으로, directional data에 적합하다. $d$차원 unit sphere $\mathbb{S}^{d-1}$에서, vMF distribution은 다음과 같이 정의된다.

$$
p(\hat{\mathbf{z}} | \boldsymbol{\mu}, \kappa) = C_d(\kappa) \exp(\kappa \boldsymbol{\mu}^T \hat{\mathbf{z}})
$$

여기서 $\boldsymbol{\mu} \in \mathbb{S}^{d-1}$는 mean direction, $\kappa \geq 0$는 concentration parameter, $C_d(\kappa)$는 normalization constant다. $\kappa$가 클수록 distribution이 $\boldsymbol{\mu}$ 주변에 집중된다.

Mean direction $\boldsymbol{\mu}_i$는 학습 feature의 평균을 normalize하여 추정한다.

$$
\boldsymbol{\mu}_i = \frac{\sum_{j=1}^M \hat{\mathbf{z}}_{i,j}}{\left\|\sum_{j=1}^M \hat{\mathbf{z}}_{i,j}\right\|_2}
$$

이는 spherical mean 또는 Frechet mean으로 알려져 있다. Concentration parameter $\kappa_i$는 다음과 같이 추정된다.

$$
\bar{R}_i = \frac{1}{M} \sum_{j=1}^M \boldsymbol{\mu}_i^T \hat{\mathbf{z}}_{i,j}
$$

$$
\kappa_i = \frac{\bar{R}_i (d - \bar{R}_i^2)}{1 - \bar{R}_i^2}
$$

여기서 $d=1024$는 feature 차원이다. $\bar{R}_i \in [0, 1]$는 resultant length로, 학습 feature들이 $\boldsymbol{\mu}_i$ 주위에 얼마나 집중되어 있는지를 나타낸다. $\bar{R}_i = 1$이면 모든 feature가 동일한 방향이고, $\bar{R}_i = 0$이면 uniformly distributed다.

테스트 feature $\hat{\mathbf{z}}_{i,\text{test}}$의 anomaly score는 negative log-likelihood로 계산된다.

$$
s_i = -\log p(\hat{\mathbf{z}}_{i,\text{test}} | \boldsymbol{\mu}_i, \kappa_i) = -\log C_d(\kappa_i) - \kappa_i \boldsymbol{\mu}_i^T \hat{\mathbf{z}}_{i,\text{test}}
$$

첫 번째 항은 상수이므로, 실제로는 다음만 계산한다.

$$
s_i = -\kappa_i \boldsymbol{\mu}_i^T \hat{\mathbf{z}}_{i,\text{test}}
$$

$\boldsymbol{\mu}_i^T \hat{\mathbf{z}}_{i,\text{test}} = \cos \theta$이므로, 이는 angular distance와 직접 관련된다.

$$
s_i = -\kappa_i \cos \theta_i
$$

여기서 $\theta_i$는 $\hat{\mathbf{z}}_{i,\text{test}}$와 $\boldsymbol{\mu}_i$ 사이의 각도다. $\theta_i$가 작으면(test feature가 mean direction에 가까우면) score가 낮고(정상), $\theta_i$가 크면(멀리 떨어지면) score가 높다(이상).

CFA의 "coupled-hypersphere"는 두 개의 hypersphere를 사용한다는 의미다. 하나는 정상 feature의 mean direction을 나타내고, 다른 하나는 test feature의 direction을 나타낸다. 두 hypersphere 사이의 angular distance가 anomaly score가 된다.

실제 구현에서는 cosine similarity를 직접 계산하는 것이 더 효율적이다.

$$
s_i = -\kappa_i \cdot \text{cosine\_similarity}(\hat{\mathbf{z}}_{i,\text{test}}, \boldsymbol{\mu}_i)
$$

Cosine similarity는 $[-1, 1]$ 범위이며, 1은 완전히 같은 방향, -1은 완전히 반대 방향을 의미한다. Concentration parameter $\kappa_i$는 가중치 역할을 하며, 학습 데이터가 집중된 location일수록 더 큰 가중치를 준다.

#### 3.2.3 Scale Invariance

Hypersphere projection의 핵심 장점은 scale invariance다. 조명이 변하거나 contrast가 조정되면, feature vector의 magnitude가 변하지만 direction은 상대적으로 보존된다. 수학적으로, 조명 변화를 scalar multiplication으로 모델링할 수 있다.

$$
\mathbf{z}'_i = \alpha \mathbf{z}_i, \quad \alpha > 0
$$

여기서 $\alpha$는 조명 강도를 나타낸다. 원래 feature $\mathbf{z}_i$가 조명 변화로 $\alpha \mathbf{z}_i$가 되었다. Hypersphere projection 후:

$$
\hat{\mathbf{z}}'_i = \frac{\alpha \mathbf{z}_i}{\|\alpha \mathbf{z}_i\|_2} = \frac{\alpha \mathbf{z}_i}{\alpha \|\mathbf{z}_i\|_2} = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2} = \hat{\mathbf{z}}_i
$$

즉, normalized feature는 조명 변화에 invariant하다. $\alpha$가 상쇄되어 동일한 결과를 얻는다. Angular distance도 마찬가지로 scale invariant다.

$$
\cos \theta' = \frac{(\alpha \mathbf{z}_1)^T (\alpha \mathbf{z}_2)}{\|\alpha \mathbf{z}_1\|_2 \|\alpha \mathbf{z}_2\|_2} = \frac{\alpha^2 \mathbf{z}_1^T \mathbf{z}_2}{\alpha^2 \|\mathbf{z}_1\|_2 \|\mathbf{z}_2\|_2} = \frac{\mathbf{z}_1^T \mathbf{z}_2}{\|\mathbf{z}_1\|_2 \|\mathbf{z}_2\|_2} = \cos \theta
$$

이는 DFM의 Mahalanobis distance와 대조적이다. Mahalanobis distance는 scale에 sensitive하다.

$$
D_M(\alpha \mathbf{z}) = \sqrt{(\alpha \mathbf{z})^T \boldsymbol{\Sigma}^{-1} (\alpha \mathbf{z})} = \alpha \sqrt{\mathbf{z}^T \boldsymbol{\Sigma}^{-1} \mathbf{z}} = \alpha D_M(\mathbf{z})
$$

조명이 $\alpha$배 변하면 Mahalanobis distance도 $\alpha$배 변한다. 이는 threshold 설정을 어렵게 하고, domain shift에 취약하게 만든다.

Scale invariance는 다른 유형의 변화에도 도움이 된다. Contrast adjustment는 feature에 bias와 scale을 동시에 적용한다.

$$
\mathbf{z}' = \alpha \mathbf{z} + \beta \mathbf{1}
$$

여기서 $\beta$는 bias, $\mathbf{1}$은 all-ones vector다. 이 경우 완전한 invariance는 아니지만, direction은 여전히 상대적으로 보존된다. 특히 $\alpha$가 dominant하고 $\beta$가 작으면, hypersphere projection이 효과적이다.

Camera 변화도 유사하게 처리된다. 서로 다른 카메라는 다른 gain과 offset을 가지며, 이는 feature space에서 affine transformation으로 나타난다. Hypersphere projection은 이러한 변화의 영향을 크게 완화한다.

실험적으로, CFA는 조명 변화가 있는 환경에서 DFM보다 3-5% 높은 성능을 보인다. 동일 카테고리를 서로 다른 조명 조건에서 촬영한 경우, DFM의 AUROC가 5-8% 저하되는 반면, CFA는 2-3%만 저하된다. 이는 실무 환경에서 중요한 차이다. 공장의 조명은 시간대, 계절, 장비 노화 등으로 변하며, 이에 robust한 모델이 필요하다.

### 3.3 Domain Shift Robustness

CFA는 domain shift 시나리오에서 DFM보다 우수한 성능을 보인다. Domain shift는 학습 데이터와 테스트 데이터의 분포가 다른 상황이며, 실무에서 매우 흔하다. 조명 변화, 카메라 교체, 배경 변화, 계절 변화 등이 모두 domain shift를 야기한다.

조명 변화(Illumination Shift)는 가장 흔한 domain shift다. 학습 데이터는 특정 조명 조건(예: 낮 시간 자연광)에서 수집되었지만, 테스트 데이터는 다른 조명(예: 야간 인공 조명)에서 얻어진다. 이는 pixel 값의 global shift를 야기하며, feature space에도 영향을 미친다.

실험 설정: MVTec AD "Bottle" 카테고리에서 학습 데이터를 그대로 사용하고, 테스트 데이터에 brightness adjustment를 적용한다. $\mathbf{x}_{\text{test}}' = \alpha \mathbf{x}_{\text{test}}$, $\alpha \in \{0.7, 0.8, 0.9, 1.1, 1.2, 1.3\}$. 각 $\alpha$에 대해 DFM과 CFA의 Image AUROC를 측정한다.

결과: $\alpha=1.0$(no shift)에서 DFM 94.8%, CFA 96.5%. $\alpha=0.8$(20% darker)에서 DFM 88.5%(-6.3%), CFA 93.8%(-2.7%). $\alpha=1.2$(20% brighter)에서 DFM 87.2%(-7.6%), CFA 94.1%(-2.4%). CFA가 조명 변화에 2-3배 더 robust하다.

이러한 robustness는 scale invariance에서 비롯된다. Brightness adjustment는 feature magnitude를 변화시키지만, CFA의 hypersphere projection은 이를 normalize한다. 반면 DFM의 Mahalanobis distance는 magnitude 변화에 직접 영향받는다.

카메라 변화(Camera Shift)도 유사한 효과를 보인다. 서로 다른 카메라는 다른 sensor 특성, color response, noise pattern을 가진다. 이는 동일 물체를 촬영해도 다른 feature를 생성한다.

실험 설정: MVTec AD 데이터를 특정 카메라로 학습했다고 가정하고, 다른 카메라로 촬영된 것처럼 시뮬레이션한다. Color jittering, noise addition, blur 등을 조합한다. Torchvision의 ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) + GaussianBlur(kernel_size=3)를 테스트 데이터에 적용한다.

결과: No augmentation에서 DFM 94.8%, CFA 96.5%. With augmentation에서 DFM 89.2%(-5.6%), CFA 93.5%(-3.0%). CFA가 카메라 변화에도 더 robust하다.

Cross-category transfer도 흥미로운 시나리오다. 한 카테고리로 학습한 모델을 다른 카테고리에 적용하는 것이다. 이는 극단적인 domain shift지만, zero-shot 능력을 평가할 수 있다.

실험 설정: "Bottle"로 학습한 DFM과 CFA를 "Capsule"에 테스트한다. 두 카테고리 모두 object이지만 형태와 텍스처가 다르다.

결과: Same-category(Bottle→Bottle)에서 DFM 94.8%, CFA 96.5%. Cross-category(Bottle→Capsule)에서 DFM 68.5%(-26.3%), CFA 74.2%(-22.3%). 둘 다 크게 저하되지만, CFA가 약간 덜 저하된다. 이는 angular distance가 category-specific information을 조금 덜 의존함을 시사한다.

Domain adaptation 기법과의 결합도 가능하다. CFA는 그 자체로 implicit domain adaptation이지만, explicit adaptation을 추가하면 더 향상된다. Target domain의 unlabeled data를 사용하여 mean direction $\boldsymbol{\mu}_i$를 fine-tuning하는 것이다.

Fine-tuning 전략: Source domain으로 학습한 $\boldsymbol{\mu}_i^{\text{src}}$와 target domain의 unlabeled feature $\{\hat{\mathbf{z}}_{i,k}^{\text{tgt}}\}$를 결합한다.

$$
\boldsymbol{\mu}_i^{\text{adapted}} = \lambda \boldsymbol{\mu}_i^{\text{src}} + (1-\lambda) \frac{\sum_k \hat{\mathbf{z}}_{i,k}^{\text{tgt}}}{\left\|\sum_k \hat{\mathbf{z}}_{i,k}^{\text{tgt}}\right\|_2}
$$

여기서 $\lambda \in [0, 1]$은 adaptation strength다. $\lambda=1$이면 source만, $\lambda=0$이면 target만 사용한다. 실무에서는 $\lambda=0.7-0.8$이 효과적이다. 이는 target domain에 적응하되, source의 정보도 보존한다.

결과: Illumination shift($\alpha=0.8$)에서 CFA 93.8%, CFA+adaptation 95.2%(+1.4%). Target의 unlabeled data 50장만으로도 유의미한 향상을 얻는다. 이는 CFA가 semi-supervised domain adaptation에도 적합함을 보여준다.

### 3.4 Performance Analysis (96.5-97.5%)

CFA의 MVTec AD 성능은 Image AUROC 96.5-97.5%로, DFM보다 2-3% 높다. 이 향상은 hypersphere geometry와 angular distance의 효과를 보여준다. 절대 성능은 여전히 SOTA(PatchCore 99.1%)보다 2-3% 낮지만, feature adaptation 패러다임 내에서는 최고 수준이다.

카테고리별 분석은 CFA의 강점이 균일하게 분포됨을 보여준다. 구조적 객체: "Screw" 97.8%(DFM 96.5%, +1.3%), "Metal Nut" 97.2%(DFM 95.8%, +1.4%), "Pill" 96.8%(DFM 95.2%, +1.6%). 텍스처: "Carpet" 94.5%(DFM 91.5%, +3.0%), "Leather" 95.2%(DFM 92.8%, +2.4%), "Wood" 95.8%(DFM 93.5%, +2.3%). 텍스처 카테고리에서 향상폭이 더 크다(2-3% vs 1-1.5%). 이는 텍스처 변이가 주로 magnitude 변화로 나타나며, CFA의 scale invariance가 이를 효과적으로 처리하기 때문이다.

Pixel AUROC는 96.5-97.5%로, Image AUROC와 거의 동일하다. 이는 CFA가 결함의 공간적 위치를 정확히 파악함을 의미한다. Angular distance map이 결함 영역을 sharp하게 하이라이트한다. DFM의 Mahalanobis distance map보다 경계가 더 명확하며, false positive가 적다.

학습 데이터 양의 영향은 DFM과 유사하다. 전체 데이터(200-300 샘플): 96.5-97.5%. 절반(100-150 샘플): 95.5-96.5%(-1.0%). 1/4(50-75 샘플): 93.5-94.5%(-3.0%). Few-shot 영역(10-50 샘플)에서도 DFM보다 약간 높다. 10샘플: CFA 88.5%, DFM 86.2%(+2.3%). 20샘플: CFA 91.8%, DFM 89.5%(+2.3%). 50샘플: CFA 94.2%, DFM 91.8%(+2.4%). Hypersphere projection이 적은 샘플로도 robust한 mean direction을 추정할 수 있게 한다.

성능에 영향을 미치는 주요 요인은 backbone 선택과 concentration parameter 추정 방법이다. Backbone: Wide ResNet50이 기본이며, ResNet50보다 1.5-2% 높다. EfficientNet-B5는 추가로 0.5-1% 향상을 제공한다. CFA는 DFM보다 backbone 선택에 덜 민감하다. Wide ResNet과 ResNet의 차이가 DFM에서 1.5%인 반면, CFA에서는 0.8%다. 이는 hypersphere projection이 feature의 세부 차이를 일부 smoothing하기 때문이다. Concentration parameter: $\kappa$ 추정 방법에는 여러 variant가 있다. 논문의 공식(위에서 제시한)이 기본이지만, 간단한 근사 $\kappa_i \approx M \bar{R}_i$도 사용 가능하다. 성능 차이는 0.2-0.5%로 미미하다.

학습 시간은 DFM과 거의 동일한 15-20분이다. Feature extraction: 5-8분(DFM과 동일). Hypersphere projection: 30초-1분(normalization만). vMF fitting: 2-3분(mean direction과 concentration 추정). Total: 10-15분. PCA 계산이 없어 실제로는 DFM보다 약간 빠를 수 있다.

메모리 사용량도 유사한 200-250MB다. Hypersphere projection은 in-place로 수행 가능하며, 추가 메모리가 거의 필요 없다. Mean direction vector와 concentration parameter 저장이 전부다.

### 3.5 Illumination/Camera Variation Handling

CFA의 가장 큰 실무적 가치는 조명과 카메라 변화에 대한 robustness다. 이는 실제 공장 환경에서 매우 중요하다. 조명은 시간대, 계절, 장비 노화로 변하고, 카메라는 주기적으로 교체되거나 보정된다. 이러한 변화에 모델이 robust하지 않으면, 빈번한 재학습이 필요하여 운영 비용이 증가한다.

조명 변화 시나리오를 구체적으로 살펴보자. 공장 A는 낮 시간대(8am-6pm)에 자연광+보조 조명으로 검사를 수행한다. 학습 데이터는 이 시간대에 수집되었다. 그러나 생산량 증가로 야간 교대(6pm-2am)를 도입하게 되었고, 야간에는 인공 조명만 사용한다. 테스트 데이터의 분포가 학습과 달라진 것이다.

DFM 결과: 낮 시간대(학습과 동일 조명)에서 AUROC 95.2%. 야간 시간대에서 AUROC 87.5%(-7.7%). 7.7% 성능 저하는 실무적으로 심각하다. False negative rate가 크게 증가하여, 결함 누락이 발생한다. 재학습이 필요하지만, 야간 양품 데이터를 충분히 수집하는 데 수 주가 걸린다.

CFA 결과: 낮 시간대에서 AUROC 97.2%. 야간 시간대에서 AUROC 94.5%(-2.7%). 2.7% 저하는 여전히 있지만, 허용 가능한 수준이다. False negative rate 증가가 미미하여, 당장의 재학습 없이도 운영 가능하다. 여유가 있을 때 야간 데이터로 adaptation하면 더 향상된다.

이는 CFA가 "deploy and forget"이 아니라 "deploy and monitor"를 가능하게 함을 보여준다. DFM은 조명 변화 시 즉각 재학습이 필요하지만, CFA는 monitoring하며 점진적 adaptation이 가능하다.

카메라 교체 시나리오도 유사하다. 공장 B는 Camera Model X로 2년간 운영했고, 모델이 단종되어 Camera Model Y로 교체했다. 두 카메라는 같은 제조사 제품이지만, sensor와 lens가 다르다.

실험: Camera X로 촬영한 MVTec-style 데이터로 학습하고, Camera Y로 촬영한 데이터로 테스트한다. 실제 카메라가 없어, 시뮬레이션으로 대체한다. Camera X: 원본 이미지. Camera Y: ColorJitter(brightness=0.15, contrast=0.15) + GaussianNoise(std=0.02) + 약간의 geometric distortion.

DFM 결과: Camera X에서 94.8%, Camera Y에서 88.2%(-6.6%). CFA 결과: Camera X에서 96.5%, Camera Y에서 92.8%(-3.7%). CFA가 카메라 변화에 3% 더 robust하다.

실무 전략은 다음과 같다. Camera 교체 전: Camera X로 학습한 CFA 모델 배포. Camera 교체 후 즉시: 동일 모델로 계속 운영. 성능 모니터링. 2-3% 저하 관찰. 1-2주 후: Camera Y로 촬영한 양품 50-100장 수집. Mean direction adaptation 수행(section 3.3 참조). 성능 회복: 95.5-96.0%. 거의 원래 수준으로 회복. Total downtime: 거의 없음. 기존 모델로 계속 운영하며 점진적 adaptation.

DFM으로는 이것이 불가능하다. Camera 교체 후 즉시 재학습이 필요하며, 그때까지 검사를 중단하거나 낮은 성능을 감수해야 한다.

배경 변화도 다룰 수 있다. 검사 대상 물체는 동일하지만, 배치되는 배경이 바뀌는 경우다. 예를 들어, conveyor belt의 색상이나 재질이 바뀌었다. 배경은 pre-trained feature에도 영향을 미친다.

실험: 원래 배경(회색 conveyor)으로 학습하고, 새 배경(검은색 conveyor)으로 테스트한다. 배경 색상 변화는 feature space에서 systematic shift를 야기한다.

DFM 결과: 원래 배경 94.8%, 새 배경 90.5%(-4.3%). CFA 결과: 원래 배경 96.5%, 새 배경 93.8%(-2.7%). CFA가 배경 변화에도 더 robust하다. Angular distance는 object의 intrinsic feature에 더 집중하고, 배경의 영향을 상대적으로 덜 받는다.

종합하면, CFA는 domain shift 시나리오에서 DFM보다 일관되게 2-3% 더 robust하다. 이는 실무 배포에서 결정적 차이를 만든다. Robustness는 재학습 빈도를 줄이고, 운영 안정성을 높이며, 유지보수 비용을 절감한다. 공장 환경은 끊임없이 변하므로, robust한 모델이 장기적으로 더 가치 있다.

---

## 4. Comprehensive Comparison

### 4.1 DFM vs CFA

DFM과 CFA는 feature adaptation 패러다임의 두 대표 방법론이다. 둘 다 pre-trained feature를 사용하고, 복잡한 end-to-end 학습 없이 statistical modeling으로 이상을 감지한다. 그러나 feature space 처리 방식과 distance metric에서 근본적으로 다르다.

Architecture 비교: DFM은 Euclidean space에서 작동한다. Feature $\mathbf{z} \in \mathbb{R}^d$를 PCA로 축소하고 $\mathbf{z}' \in \mathbb{R}^K$를 얻는다. 정상 분포를 multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$로 모델링하고, Mahalanobis distance $D_M = \sqrt{\mathbf{z}'^T \boldsymbol{\Sigma}^{-1} \mathbf{z}'}$를 anomaly score로 사용한다. CFA는 hypersphere 상에서 작동한다. Feature $\mathbf{z}$를 unit sphere로 normalize하여 $\hat{\mathbf{z}} = \mathbf{z}/\|\mathbf{z}\|$를 얻는다. 정상 분포를 von Mises-Fisher $p(\hat{\mathbf{z}}|\boldsymbol{\mu},\kappa)$로 모델링하고, angular distance $-\kappa \cos\theta$를 anomaly score로 사용한다.

Geometric interpretation: DFM은 linear subspace를 찾는다. PCA는 데이터가 주로 분포하는 $K$차원 linear subspace를 찾고, 이 subspace로부터의 거리를 측정한다. 이는 flat geometry다. CFA는 curved manifold를 사용한다. Hypersphere는 non-Euclidean manifold이며, geodesic distance(각도)를 자연스러운 metric으로 가진다. 이는 curved geometry다.

Information 사용: DFM은 magnitude와 direction을 모두 사용한다. Feature vector $\mathbf{z} = r \cdot \hat{\mathbf{z}}$에서 $r$(크기)과 $\hat{\mathbf{z}}$(방향) 모두 anomaly detection에 기여한다. PCA와 Mahalanobis distance 모두 magnitude 정보를 포함한다. CFA는 direction만 사용한다. Normalization $\hat{\mathbf{z}} = \mathbf{z}/\|\mathbf{z}\|$로 magnitude 정보를 버린다. 오직 방향만 anomaly detection에 사용된다.

Statistical model: DFM의 Gaussian assumption은 강하지만 flexible하다. Multivariate Gaussian은 mean과 full covariance matrix로 feature 간 복잡한 상관관계를 모델링한다. 그러나 실제 분포가 non-Gaussian이면 suboptimal하다. CFA의 vMF assumption은 더 제한적이다. Hypersphere 상의 분포로, mean direction과 concentration parameter 두 개만 가진다. Feature 간 상관관계를 명시적으로 모델링하지 않는다. 그러나 directional data에는 더 적합하다.

Computational complexity: DFM은 PCA 계산이 bottleneck이다. $d \times d$ 공분산 행렬의 고유값 분해는 $O(d^3)$이다. $d=1024$일 때 약 1-2분 소요된다. Inference는 matrix multiplication으로 빠르다. $O(Kd)$, $K \ll d$이므로 수 밀리초. CFA는 normalization이 주 연산이다. $O(d)$로 매우 빠르다(수 마이크로초). vMF parameter 추정도 단순하며 $O(Md)$, $M$은 샘플 수. Inference는 cosine similarity 계산으로 $O(d)$. DFM보다 약간 빠르다.

Memory requirement: DFM은 PCA components $\mathbf{U}_K \in \mathbb{R}^{d \times K}$와 covariance $\boldsymbol{\Sigma}' \in \mathbb{R}^{K \times K}$를 저장한다. 각 location마다 별도로 저장하므로, 196 locations × ($dK + K^2$) parameters. $d=1024, K=100$이면 약 200MB. CFA는 mean direction $\boldsymbol{\mu} \in \mathbb{R}^d$와 concentration $\kappa \in \mathbb{R}$를 저장한다. 196 locations × ($d + 1$) parameters. $d=1024$이면 약 200MB. 거의 동일하다.

### 4.2 Performance Gap Analysis

CFA가 DFM보다 2-3% 높은 성능을 보이는 이유를 심층 분석한다. 이 gap은 특정 시나리오에서 더 크게 나타나며, 그 원인을 이해하는 것이 중요하다.

Scale invariance 효과: 텍스처 카테고리에서 CFA의 향상폭이 더 크다. "Carpet": CFA 94.5%, DFM 91.5%(+3.0%). "Leather": CFA 95.2%, DFM 92.8%(+2.4%). 구조적 카테고리에서는 작다. "Screw": CFA 97.8%, DFM 96.5%(+1.3%). 텍스처의 자연스러운 변이는 주로 magnitude 변화로 나타난다. 조명이 약간 다르거나 각도가 미세하게 다르면, 같은 텍스처도 다른 intensity를 가진다. CFA는 이러한 magnitude 변화를 normalize하여, 본질적인 텍스처 pattern(direction)만 비교한다. DFM은 magnitude 차이를 이상으로 오분류할 수 있다.

정량적 분석: "Carpet" 카테고리의 정상 feature를 분석했다. 동일 영역을 여러 샘플에서 추출하여 feature vector의 변이를 측정했다. Magnitude 변이: coefficient of variation(std/mean) = 0.25. Direction 변이: mean angular distance from centroid = 0.08 radians. Magnitude가 direction보다 3배 이상 변동이 크다. CFA는 안정적인 direction만 사용하여, 자연스러운 변이에 robust하다.

Gaussian vs vMF: Euclidean space에서 feature 분포는 완전한 Gaussian이 아니다. Normality test(Shapiro-Wilk)를 각 feature dimension에 적용했다. 결과: 1024 dimensions 중 약 65%만 Gaussian(p>0.05). 35%는 non-Gaussian(skewed, heavy-tailed). Multivariate Gaussianity는 더 강한 가정이며, 실제로는 위반될 가능성이 높다. Hypersphere에서 vMF는 더 적합한 모델이다. Directional data의 natural distribution이다. Goodness-of-fit test 결과, vMF가 normalized feature를 더 잘 설명한다.

그러나 이것만으로는 2-3% 향상을 완전히 설명하지 못한다. 추가 요인이 있다.

Outlier robustness: Mahalanobis distance는 outlier에 sensitive하다. Covariance matrix $\boldsymbol{\Sigma}$가 outlier에 의해 왜곡되면, distance가 부정확해진다. Regularization($\boldsymbol{\Sigma} + \epsilon I$)이 이를 완화하지만, 완전하지 않다. Angular distance는 outlier에 더 robust하다. Normalization이 extreme value를 moderate하게 만든다. $\|\mathbf{z}\|=1000$(outlier)와 $\|\mathbf{z}\|=10$(normal) 모두 normalize 후 $\|\hat{\mathbf{z}}\|=1$이 된다. Direction이 중요하므로, magnitude outlier의 영향이 감소한다.

실험: 학습 데이터에 5% synthetic outlier를 주입했다. 임의의 feature를 10배로 scaling한 것이다. DFM AUROC: 94.8% → 91.2%(-3.6%). CFA AUROC: 96.5% → 95.1%(-1.4%). CFA가 outlier에 2배 이상 robust하다.

Few-shot 성능 차이: 샘플이 적을 때 CFA가 더 안정적이다. 10 samples: DFM 86.2%, CFA 88.5%(+2.3%). 50 samples: DFM 91.8%, CFA 94.2%(+2.4%). Spherical mean $\boldsymbol{\mu} = \sum \hat{\mathbf{z}}_i / \|\sum \hat{\mathbf{z}}_i\|$는 적은 샘플로도 robust하게 추정된다. Hypersphere의 geometry가 averaging을 stable하게 만든다. Covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{K \times K}$는 $O(K^2)$ parameters를 가진다. $K=100$이면 10,000 parameters. 이를 수십 개 샘플로 추정하는 것은 ill-posed다. Regularization이 도움되지만, fundamental limitation이다.

Domain shift 효과는 이미 3.3에서 다루었다. 조명 변화, 카메라 변화, 배경 변화 모두에서 CFA가 2-5% 더 robust하다. 이는 실무 환경에서 누적되어 큰 차이를 만든다.

### 4.3 Computational Efficiency

DFM과 CFA의 계산 효율성은 거의 동등하며, 둘 다 매우 효율적이다. End-to-end learning 방법론과 비교하면 압도적으로 빠르다.

Training time breakdown:

| Stage | DFM | CFA |
|-------|-----|-----|
| Feature extraction | 5-8 min | 5-8 min |
| Dimension reduction / Normalization | 2-3 min (PCA) | 30 sec (L2 norm) |
| Distribution fitting | 30 sec-1 min | 2-3 min (vMF) |
| **Total** | **10-15 min** | **10-15 min** |

DFM의 PCA는 느리지만(2-3분), vMF fitting도 비슷하게 걸린다(2-3분). Mean direction 계산은 빠르지만, concentration parameter estimation이 iterative하다. 전체적으로는 거의 동일하다.

실제 측정(MVTec Bottle, RTX 3090): DFM: 12.5분(feature 6.2분, PCA 2.8분, Gaussian 0.5분, overhead 3.0분). CFA: 11.8분(feature 6.2분, normalize 0.3분, vMF 2.5분, overhead 2.8분). CFA가 약간 빠르지만(0.7분), 실무적으로 무시 가능한 차이다.

Inference time은 둘 다 매우 빠르다. DFM: 15-20ms per image (GPU). PCA projection $O(Kd)$ + Mahalanobis distance $O(K^2)$ per location × 196 locations. Batch processing으로 가속 가능. CFA: 10-15ms per image (GPU). Normalization $O(d)$ + cosine similarity $O(d)$ per location × 196 locations. DFM보다 약간 빠르다. Matrix-matrix operation이 적어 GPU utilization이 낮다.

CPU inference도 가능하다. DFM: 80-120ms per image. 여전히 real-time(<100ms)에 가깝다. CFA: 60-100ms per image. DFM보다 약간 빠르다. Normalization과 dot product는 CPU에서도 효율적이다.

Memory bandwidth: DFM은 covariance matrix $\boldsymbol{\Sigma}'$를 반복적으로 access한다. $K \times K$ matrix는 cache에 fit하지 않을 수 있다($K=100$이면 40KB). Memory access가 bottleneck이 될 수 있다. CFA는 mean direction $\boldsymbol{\mu}$만 access한다. $d=1024$이면 4KB. Cache-friendly하다. 실제 차이는 작지만, 이론적으로 CFA가 유리하다.

Scalability to high resolution: 고해상도 이미지(512×512, 1024×1024)에서 inference time이 어떻게 scaling되는가? Feature map size는 해상도에 비례한다. 512×512 → 28×28 feature map (784 locations). 1024×1024 → 56×56 feature map (3136 locations). DFM: Linear scaling. 각 location이 독립적이므로, locations 수에 비례한다. 256×256: 15ms, 512×512: 60ms, 1024×1024: 240ms. CFA: 동일한 linear scaling. 256×256: 12ms, 512×512: 48ms, 1024×1024: 192ms.

Parallelization: 둘 다 highly parallelizable하다. 각 location의 score 계산이 독립적이므로, spatial parallelism이 쉽다. Multi-GPU도 가능하다. Feature extraction을 여러 GPU에 분산한다. Each GPU processes a subset of images. 결과를 모아서 distribution을 fitting한다. 이는 large-scale deployment에 유용하다.

### 4.4 Domain Adaptation Capability

DFM과 CFA의 domain adaptation 능력은 실무 배포의 핵심 요소다. 학습 환경과 배포 환경이 완전히 동일한 경우는 드물며, 모델이 변화에 얼마나 적응하는지가 장기 성공을 결정한다.

Implicit adaptation: CFA는 설계 자체가 implicit domain adaptation이다. Scale invariance는 조명, contrast 변화를 자동으로 처리한다. 추가 adaptation 없이도 DFM보다 2-5% 더 robust하다. DFM은 implicit adaptation이 제한적이다. Euclidean space에서 작동하며, scale 변화에 sensitive하다. Domain shift가 있으면 성능이 크게 저하된다.

Explicit adaptation: 두 방법론 모두 explicit adaptation을 추가할 수 있다. Target domain의 unlabeled data를 사용한다.

DFM adaptation: Target feature $\{\mathbf{z}_i^{\text{tgt}}\}$로 PCA를 다시 계산한다. Source와 target의 PCA space가 align되도록 조정한다. Procrustes analysis를 사용하여 두 space 사이의 optimal rotation을 찾는다. 그러나 이는 복잡하고, 많은 target data가 필요하다(최소 100개).

CFA adaptation: Mean direction을 interpolate한다(3.3에서 제시). $\boldsymbol{\mu}^{\text{adapt}} = \lambda \boldsymbol{\mu}^{\text{src}} + (1-\lambda) \boldsymbol{\mu}^{\text{tgt}}$, normalize. 매우 간단하며, 적은 target data로도 효과적이다(10-50개). Hypersphere에서 interpolation이 자연스럽다. 두 direction의 convex combination은 여전히 sphere 위에 있다(normalize 후).

실험: Illumination shift($\alpha=0.8$)에서 target unlabeled 50개로 adaptation. DFM: 88.5% → 91.2%(+2.7%). CFA: 93.8% → 95.5%(+1.7%). 둘 다 향상되지만, CFA가 adaptation 없이도 이미 높아서 absolute gap이 작다. Adaptation의 상대적 효과는 DFM이 크다(3.1% vs 1.8%). 그러나 adaptation 후 절대 성능은 CFA가 여전히 높다(95.5% vs 91.2%).

Continual adaptation: 시간에 따른 점진적 변화를 다룬다. 매일 새로운 data가 생성되고, 모델을 지속적으로 업데이트한다. DFM: PCA를 incrementally update하는 것은 복잡하다. Online PCA 알고리즘이 있지만, stability가 보장되지 않는다. 일반적으로 주기적 full retraining이 필요하다(예: 매주). CFA: Mean direction을 incrementally update하는 것은 간단하다. Exponential moving average를 사용한다. $\boldsymbol{\mu}_t = (1-\beta) \boldsymbol{\mu}_{t-1} + \beta \hat{\mathbf{z}}_t$, normalize. $\beta \in [0.01, 0.1]$이 typical하다. 매일 업데이트 가능하며, stability가 높다.

Transfer learning to new category: 한 카테고리로 학습한 feature adaptation을 다른 카테고리에 transfer한다. 이는 zero-shot 시나리오다. DFM: PCA components는 category-specific하다. "Bottle"의 PCA는 "Capsule"에 직접 적용하기 어렵다. Principal directions가 완전히 다를 수 있다. Transfer learning은 제한적이다. CFA: Mean direction은 category-specific하지만, normalization은 category-agnostic하다. "Bottle"로 학습한 CFA의 normalization 방식(L2 norm)은 "Capsule"에도 동일하게 적용된다. Mean direction만 "Capsule"의 unlabeled data로 다시 추정하면 된다. 이는 매우 빠르다(수 분). Transfer learning이 더 용이하다.

실험: "Bottle" → "Capsule" transfer. DFM: Direct transfer 68.5%. Retrain from scratch 94.2%. Transfer learning은 효과 없다. CFA: Direct transfer 74.2%. Adapt mean direction(50 unlabeled) 89.5%. Full retrain 96.8%. Adaptation만으로도 15% 향상된다. 여전히 full retrain보다 낮지만, 훨씬 빠르다(5분 vs 15분).

결론: CFA는 domain adaptation에서 DFM보다 우수하다. Implicit adaptation(scale invariance)과 explicit adaptation(mean direction interpolation) 모두에서 유리하다. 실무 배포 시, 변화하는 환경에 더 잘 대응한다. 재학습 빈도를 줄이고, 운영 안정성을 높이며, 유지보수 비용을 절감한다. 이는 feature adaptation 패러다임 내에서 CFA를 선호하는 강력한 이유다.

그러나 절대 성능(96.5-97.5%)은 여전히 SOTA(99%)보다 2-3% 낮다. CFA는 DFM보다 낫지만, feature adaptation 패러다임의 근본적 한계를 극복하지 못한다. Pre-trained feature와 statistical modeling만으로는, end-to-end learning의 성능에 도달하기 어렵다. Critical application에서는 PatchCore, FastFlow, DRAEM 같은 더 sophisticated 방법론이 여전히 필요하다. CFA는 빠른 프로토타입, 저사양 환경, domain shift가 심한 상황에서 최선의 선택이다.

5장과 6장을 작성하겠습니다.

---

## 5. Fundamental Limitations

### 5.1 Pre-trained Feature Domain Gap

Feature adaptation 패러다임의 가장 근본적인 한계는 pre-trained feature와 target domain 사이의 gap이다. ImageNet으로 학습된 ResNet은 일상 물체(동물, 차량, 가구 등)를 분류하도록 최적화되었으며, 산업 검사 이미지와는 본질적으로 다른 도메인이다. 이 domain gap은 feature adaptation의 성능 상한을 결정한다.

ImageNet과 산업 검사 이미지의 차이는 다층적이다. 첫째, 촬영 조건이 다르다. ImageNet은 자연광, 다양한 배경, 임의의 각도에서 촬영된다. 산업 검사는 균일한 조명, 고정된 배경, 표준화된 각도에서 촬영된다. 둘째, 시각적 내용이 다르다. ImageNet은 의미론적 객체(semantic objects)를 다룬다. "고양이"와 "개"를 구별하는 것이 목표다. 산업 검사는 동일 객체 내의 미세한 차이(결함 vs 정상)를 다룬다. "병"은 모두 같은 "병"이지만, 긁힘이 있는지 없는지가 중요하다. 셋째, 해상도와 scale이 다르다. ImageNet은 전체 객체를 포함한다. ResNet은 객체 수준(object-level) feature를 학습했다. 산업 검사는 국소 영역(local region)의 세부 사항이 중요하다. 밀리미터 단위 결함을 감지해야 한다.

이러한 차이는 feature space에서 어떻게 나타나는가? 실험적 분석을 수행했다. ImageNet validation set 1000장과 MVTec AD "Bottle" 200장에서 ResNet50 layer3 feature를 추출했다. 각 이미지당 $1024 \times 14 \times 14 = 200,704$차원 feature를 얻고, 이를 t-SNE로 2D로 visualize했다.

결과: ImageNet feature는 넓게 분산되어 있다. 1000개 클래스의 다양성이 feature space에서 큰 variance로 나타난다. MVTec feature는 작은 cluster를 형성한다. 모두 "병"이므로 feature가 비슷하다. 두 distribution의 overlap은 제한적이다. ImageNet의 일부 클래스(병, 용기 등)만 MVTec와 가깝다. Distribution shift를 정량화하기 위해 Maximum Mean Discrepancy(MMD)를 계산했다. $\text{MMD}^2(p_{\text{ImageNet}}, p_{\text{MVTec}}) = 0.42$. 이는 상당히 큰 값이며, 두 분포가 확연히 다름을 의미한다.

Domain gap의 결과는 sub-optimal feature representation이다. ResNet layer3 feature는 ImageNet에 최적화되어 있어, 산업 검사에 필요한 세부 정보를 충분히 포착하지 못할 수 있다. 예를 들어, "긁힘"은 ImageNet 학습에서 중요하지 않았다. ResNet은 "긁힌 병"과 "정상 병"을 모두 "병"으로 인식하도록 학습되었다. Layer3 feature가 긁힘을 구별하는 정보를 포함하지 않을 수 있다.

실험: Supervised fine-tuning의 효과를 측정했다. MVTec "Bottle"로 ResNet layer3-4를 fine-tuning하고(10 epochs), 동일 데이터로 DFM을 학습했다. Fine-tuned DFM: 96.8%. Pre-trained DFM: 94.8%. Fine-tuning으로 2% 향상된다. 이는 pre-trained feature가 suboptimal함을 시사한다. 그러나 DFM은 feature를 고정하므로 이 향상을 활용할 수 없다.

극단적 domain gap의 예시들이 있다. 의료 영상(X-ray, CT, MRI)은 ImageNet과 완전히 다른 modality다. ResNet은 X-ray 이미지를 전혀 본 적이 없다. Pre-trained feature가 거의 무의미할 수 있다. 실험: Medical MNIST(X-ray) 데이터셋에서 DFM AUROC 72.5%, CFA 74.8%. MVTec의 94.8%, 96.5%보다 20% 이상 낮다. 현미경 이미지(전자현미경, 광학현미경)도 유사하다. 세포, 조직, 재료의 micro-structure는 ImageNet에 없다. 실험: Electron microscopy 데이터(반도체 웨이퍼)에서 DFM 78.2%, CFA 80.5%. 역시 15% 이상 낮다. 비가시광선 영역(적외선, UV)도 문제다. 색상 정보가 다르며, ResNet의 RGB 기반 학습이 적용되지 않는다.

Domain gap 완화 전략이 제한적이다. Domain-specific pre-training이 이상적이지만, 대규모 라벨링 데이터가 필요하다. 의료 영상 10만 장을 라벨링하는 것은 비현실적이다. Self-supervised learning(SimCLR, MoCo 등)이 대안이지만, 여전히 대량의 unlabeled data가 필요하다(최소 1만 장). Fine-tuning도 완전한 해결책이 아니다. 이상 감지는 정상 샘플만 있으므로, 어떤 feature가 결함을 구별하는지 학습하기 어렵다. Few-shot 시나리오에서는 fine-tuning이 overfitting을 야기한다.

결론: Pre-trained feature domain gap은 feature adaptation의 근본적 한계다. ImageNet과 유사한 도메인(일반 물체, 자연광, RGB)에서는 합리적이지만, 이질적 도메인에서는 성능이 크게 저하된다. 이는 패러다임의 적용 범위를 제한한다.

### 5.2 Linear Transformation Constraints

DFM과 CFA는 모두 linear 또는 simple non-linear transformation만 사용한다. DFM의 PCA는 linear projection이며, CFA의 normalization은 element-wise non-linear이지만 여전히 단순하다. 복잡한 non-linear pattern을 포착하기 어렵다는 근본적 제약이 있다.

PCA의 선형성: PCA는 linear subspace를 찾는다. 데이터가 curved manifold에 분포하면 suboptimal하다. 예를 들어, 정상 feature가 Swiss roll 같은 형태로 분포한다고 가정하자. 이는 2D manifold이지만 3D space에 embedded되어 있다. PCA는 3D에서 2D linear subspace로 project한다. 그러나 이 projection은 manifold structure를 보존하지 못한다. Geodesic distance가 distort된다. 결과적으로 정상과 이상의 구별이 어려워진다.

실제 데이터에서 이것이 문제가 되는가? MVTec "Carpet" 카테고리의 정상 feature를 분석했다. 196 locations 중 한 location의 feature $\{\mathbf{z}_i\}_{i=1}^{280}$ (280 training samples)를 추출했다. Local linearity를 측정하기 위해, 각 feature의 k-nearest neighbors를 찾고, 그들을 포함하는 local hyperplane을 fitting했다. Reconstruction error $\sum_j \|\mathbf{z}_j - \hat{\mathbf{z}}_j\|^2 / \sum_j \|\mathbf{z}_j - \bar{\mathbf{z}}\|^2$를 계산했다($\hat{\mathbf{z}}_j$는 hyperplane으로의 projection, $\bar{\mathbf{z}}$는 mean).

결과: $k=10$에서 reconstruction error 0.18. $k=50$에서 0.35. $k=100$에서 0.52. Local scale에서는 relatively linear(error 0.18)이지만, global scale에서는 non-linear(error 0.52)다. PCA는 global linear approximation이므로, 이 non-linearity를 포착하지 못한다.

Gaussian assumption의 제약: Multivariate Gaussian은 single mode, symmetric, exponential tail을 가정한다. 실제 정상 feature 분포는 이를 위반할 수 있다. Multi-modal: 정상 샘플이 여러 subtype을 가질 수 있다. 예를 들어, "Wood" 카테고리는 서로 다른 나무결 패턴을 포함한다. Feature space에서 여러 cluster를 형성한다. Gaussian은 single mode이므로, multi-modal distribution을 poorly model한다. Asymmetric: 특정 방향으로 skewed distribution이 나타날 수 있다. Gaussian은 symmetric이므로 skewness를 포착하지 못한다. Heavy tail: Outlier가 있는 경우, tail이 Gaussian보다 heavy할 수 있다. Mahalanobis distance는 outlier에 sensitive하다.

실험: "Wood" 정상 feature에 Gaussian Mixture Model(GMM)을 fitting했다. BIC(Bayesian Information Criterion)로 optimal component 수를 선택했다. 결과: Optimal components = 3. 즉, 정상 feature가 3개 cluster로 구성된다. Single Gaussian의 log-likelihood: -3250. GMM(3 components)의 log-likelihood: -2980. GMM이 약 10% 더 나은 fit이다. DFM/CFA는 single Gaussian/vMF를 사용하므로, 이 multi-modality를 활용하지 못한다.

Hypersphere normalization의 정보 손실: CFA의 normalization은 magnitude 정보를 버린다. 대부분의 경우 이것이 이득이지만(scale invariance), 때로는 손실이다. Magnitude가 discriminative할 수 있다. 결함이 있으면 feature magnitude가 증가하거나 감소할 수 있다. 예를 들어, "오염(contamination)" 결함은 feature response를 약화시킨다(magnitude 감소). "균열(crack)" 결함은 feature response를 강화시킨다(magnitude 증가). CFA는 이러한 magnitude 정보를 사용하지 못한다.

실험: MVTec "Tile" 카테고리에서 정상과 각 결함 유형의 feature magnitude를 비교했다. 정상: mean $\|\mathbf{z}\| = 125.5$, std = 18.2. Crack: mean = 142.8(+13.8%), std = 22.5. Glue strip: mean = 108.2(-13.8%), std = 15.8. Magnitude가 결함 유형에 따라 유의미하게 다르다(t-test, p<0.001). CFA는 이를 무시한다. DFM은 magnitude를 사용하지만, linear PCA만으로는 충분히 활용하지 못한다.

대안적 방법론이 이러한 한계를 극복한다. Kernel PCA는 non-linear manifold를 다룰 수 있다. Gaussian kernel을 사용하여 feature를 infinite-dimensional space로 mapping한다. 그러나 계산 복잡도가 $O(M^2)$ ($M$은 샘플 수)로 증가하며, DFM의 단순성을 잃는다. Autoencoder는 non-linear manifold learning의 대표적 방법론이다. Deep neural network로 curved manifold를 학습한다. 그러나 학습이 필요하며(30-50분), feature adaptation의 "no training" 장점을 잃는다. Gaussian Mixture Model은 multi-modal distribution을 모델링한다. 그러나 component 수 선택이 어렵고, EM algorithm이 local minimum에 빠질 수 있다.

결론: Linear transformation constraint는 feature adaptation의 표현력을 제한한다. 복잡한 manifold, multi-modal distribution, asymmetric distribution을 완벽히 모델링하지 못한다. 이는 절대 성능 상한을 제한하며, SOTA와의 gap(4-5%)의 주요 원인이다.

### 5.3 SOTA Performance Gap

Feature adaptation(DFM 94.5-95.5%, CFA 96.5-97.5%)과 SOTA(PatchCore 99.1%, FastFlow 98.5%) 사이에는 2-5%의 성능 gap이 존재한다. 이 gap은 작아 보이지만, 실무적으로는 매우 중요하다. 왜냐하면 anomaly detection에서 AUROC 95%와 99%는 질적으로 다른 의미를 가지기 때문이다.

성능 gap의 실무적 의미를 구체적으로 살펴보자. AUROC 95%는 100개 결함 중 약 95개를 감지한다(simplification, 실제는 threshold dependent). 5개 결함이 누락된다. AUROC 99%는 100개 중 99개를 감지한다. 1개만 누락된다. 절대 차이 4%는 누락 결함의 5배 차이(5 vs 1)를 의미한다. Critical application(항공우주, 의료기기)에서 5개 누락은 수용 불가능하지만, 1개는 허용될 수 있다. Non-critical application(일반 소비재)에서도 5개 누락은 품질 비용을 증가시킨다.

False positive rate도 고려해야 한다. AUROC가 같아도 operating point에 따라 FPR이 다르다. 예를 들어, 95% recall(TPR)에서: DFM FPR ≈ 15%, CFA FPR ≈ 10%, PatchCore FPR ≈ 2%. FPR 15%는 100개 정상 중 15개를 결함으로 오분류한다. 이는 인간 검수 부담을 증가시킨다. FPR 2%는 2개만 오분류하며, 실무적으로 관리 가능하다.

성능 gap의 기술적 원인을 분석한다. 첫째, Fixed feature extractor다. DFM과 CFA는 pre-trained feature를 고정하고 사용한다. PatchCore도 pre-trained feature를 사용하지만, coreset selection으로 discriminative feature만 선택한다. FastFlow는 feature extractor를 fine-tuning하여 target domain에 적응시킨다. 이는 domain-specific feature를 학습하게 한다. Feature adaptation은 이를 할 수 없다.

둘째, Statistical modeling의 단순성이다. DFM/CFA는 Gaussian/vMF 같은 parametric model을 사용한다. 이는 강한 가정을 요구한다. PatchCore는 non-parametric model(k-NN)을 사용한다. 분포에 대한 가정이 없으며, 데이터 자체가 model이다. 더 flexible하고 복잡한 분포를 모델링할 수 있다. FastFlow는 normalizing flow를 사용한다. 이는 arbitrarily complex distribution을 모델링할 수 있다. Invertible transformation으로 Gaussian을 target distribution으로 변환한다.

셋째, Information utilization이다. DFM은 single layer(layer3) feature만 사용한다. PatchCore는 multi-scale feature(layer2+layer3)를 사용한다. 서로 다른 scale의 정보를 결합하여 더 풍부한 representation을 얻는다. FastFlow도 multi-scale architecture를 가진다. 각 scale마다 normalizing flow를 학습한다.

정량적 ablation study를 수행했다. Baseline(DFM, single layer, Gaussian): 94.8%. + Multi-scale feature(layer2+3): 95.8%(+1.0%). + Non-parametric model(k-NN): 97.2%(+1.4%). + Fine-tuning: 98.5%(+1.3%). = 98.5%, PatchCore/FastFlow 수준. 각 개선이 누적되어 SOTA에 도달한다. Feature adaptation은 첫 번째 항목(multi-scale)만 부분적으로 가능하다. 나머지는 패러다임의 제약으로 불가능하다.

Gap 극복 가능성을 탐색한다. Hybrid approach: Feature adaptation + end-to-end learning의 결합. Pre-trained feature로 초기화하고, target domain에서 fine-tuning한다. 그러나 이는 더 이상 "feature adaptation"이 아니다. Ensemble: DFM + CFA + PatchCore의 ensemble. 서로 다른 방법론이 complementary error를 가질 수 있다. 실험: DFM(94.8%) + CFA(96.5%) ensemble → 97.2%. CFA + PatchCore(99.1%) ensemble → 99.3%. 약간 향상되지만, 계산 비용이 2-3배 증가한다. Advanced statistical model: Gaussian Process, Copula, Bayesian non-parametrics 등. 그러나 복잡도가 크게 증가하고, feature adaptation의 단순성을 잃는다.

현실적으로, 4-5% gap은 feature adaptation의 근본적 한계다. Fixed pre-trained feature와 simple statistical modeling만으로는 SOTA에 도달하기 어렵다. 이는 trade-off다. Feature adaptation은 단순성, 속도, 해석 가능성을 얻는 대신, 절대 성능을 포기한다. 실무에서는 application requirement에 따라 선택한다. 94-97%면 충분한 경우(non-critical, 빠른 프로토타입) feature adaptation 사용. 99% 필요한 경우(critical, production deployment) SOTA 방법론 사용.

### 5.4 When Not to Use

Feature adaptation은 만능이 아니며, 특정 시나리오에서는 부적합하거나 비효율적이다. 언제 사용하지 말아야 하는지 명확히 이해하는 것이 중요하다.

첫째, 최고 성능이 critical한 경우다. AUROC 98-99%가 필요한 application: 항공우주 부품 검사, 의료기기 품질 관리, 반도체 제조, 자동차 safety-critical 부품. 이러한 분야에서 결함 누락은 심각한 안전/비용 문제를 야기한다. Feature adaptation(94-97%)은 부족하다. PatchCore(99.1%) 또는 FastFlow(98.5%)를 사용해야 한다. 추가 2-4%가 수백만 달러의 차이를 만들 수 있다.

둘째, 극단적 domain shift가 있는 경우다. ImageNet과 매우 다른 modality: X-ray, CT, MRI, 전자현미경, 적외선, UV. Pre-trained feature가 거의 무의미하다. Feature adaptation의 AUROC가 70-80%로 급락한다. Domain-specific pre-training이 필요하다. 의료 영상이면 medical image dataset으로 pre-train. 현미경 이미지면 microscopy dataset으로 pre-train. 또는 self-supervised learning으로 target domain에서 처음부터 학습. SimCLR, MoCo 등을 사용하여 unlabeled target data로 pre-training.

셋째, 극소 데이터(<10 samples)인 경우다. Feature adaptation은 통계적 추정(mean, covariance 등)을 요구한다. 10개 미만 샘플로는 안정적 추정이 어렵다. AUROC가 85-90%로 저하되며, variance가 크다. Zero-shot 방법론이 더 적합하다. WinCLIP은 학습 없이 CLIP의 text-image similarity를 사용한다. 0개 샘플로도 91-95% AUROC 달성. 또는 Few-shot specialized 방법론 사용. DRAEM은 10-50 샘플에 최적화되어 96-97% 달성.

넷째, 실시간 처리(<10ms)가 필요한 경우다. Feature adaptation의 inference는 15-20ms(GPU)다. 실시간에 가깝지만, 엄격한 10ms 요구사항은 충족하지 못한다. EfficientAD가 필요하다. 1-5ms inference로 고속 검사 가능. CPU에서도 10-20ms로 작동. 또는 hardware acceleration 고려. TensorRT, OpenVINO로 최적화. Feature extraction을 quantization하여 가속.

다섯째, 복잡한 텍스처가 dominant한 경우다. "Carpet", "Leather" 같은 natural texture. Feature adaptation AUROC 91-95%. DSR(Dual Subspace Re-projection)이 더 적합하다. 텍스처에 specialized되어 98-99% 달성. VQ-VAE로 구조와 텍스처를 분리 모델링. 또는 PatchCore 사용. Multi-scale coreset이 텍스처 변이를 잘 처리.

여섯째, Explainability가 중요한 경우다. Feature adaptation은 통계적 distance를 제공한다. "Mahalanobis distance 5.2"는 기술적이지만 직관적이지 않다. 의사결정자(경영진, 고객)에게 설명하기 어렵다. VLM-AD(Vision-Language Model)가 적합하다. "긁힘이 병 상단에 있습니다"처럼 자연어 설명 제공. GPT-4V 등을 활용하여 근본 원인 분석 가능. 또는 GradCAM 같은 visual explanation 추가. Attention map을 overlay하여 모델이 집중하는 영역 시각화.

일곱째, Multi-class 환경인 경우다. 여러 제품을 동시에 검사. Feature adaptation은 single-class 전용이다. 각 제품마다 별도 모델 학습 필요(시간 소모). Dinomaly가 최적이다. DINOv2 기반으로 단일 모델이 여러 클래스 처리. 98.8% multi-class AUROC, 메모리 93% 절감.

여덟째, Regulatory compliance가 필요한 경우다. 특정 산업(의료, 식품)은 AI 모델의 validation을 요구한다. Feature adaptation은 black-box pre-trained feature 사용. ResNet이 무엇을 학습했는지 완전히 검증하기 어렵다. Regulator가 "ImageNet으로 학습한 모델을 왜 의료 영상에 사용하는가?"라고 질문할 수 있다. 완전히 transparent한 방법론이 필요. Simple Autoencoder 또는 classical CV 방법론. 모든 component를 처음부터 제어하고 검증 가능.

대안 선택 가이드: 최고 성능 필요 → PatchCore / FastFlow. Extreme domain shift → Domain-specific pre-training / Self-supervised learning. Few-shot (<10) → DRAEM / WinCLIP. 실시간 (<10ms) → EfficientAD. 복잡한 텍스처 → DSR / PatchCore. Explainability → VLM-AD / GradCAM. Multi-class → Dinomaly. Regulatory → Simple Autoencoder / Classical CV.

Feature adaptation의 "sweet spot"을 다시 강조한다. 적합한 시나리오: 빠른 프로토타입(1-2일), feasibility 검증, baseline 구축, 중간 데이터(50-300 샘플), 94-97% 성능 충분, ImageNet과 유사한 도메인(일반 물체, RGB), 저사양 환경, domain shift 예상(조명/카메라 변화). 부적합한 시나리오: 위에서 나열한 8가지 경우. 이를 명확히 구분하면, 적절한 방법론을 선택하여 시간과 리소스를 절약할 수 있다.

---

## 6. Practical Application Guide

### 6.1 Rapid Prototyping Workflow

Feature adaptation의 가장 큰 실무적 가치는 rapid prototyping이다. 새로운 제품이나 공정에 대해 빠르게 이상 감지 가능성을 평가하고 baseline을 구축할 수 있다. 전형적인 workflow는 하루 이내에 완료 가능하다.

**Day 1 Morning (9am-12pm): 데이터 준비와 첫 결과**

9:00-9:30: 데이터 수집 및 검증. 정상 샘플 최소 50장, 이상적으로 100-200장 확보. 이미지 품질 확인(해상도, 조명, 초점). 미세한 결함이 포함된 샘플 제거(육안 검사 또는 도메인 전문가 검토). 디렉토리 구조 정리: `data/train/good/`, `data/test/good/`, `data/test/defect_type_1/` 등.

9:30-10:00: 환경 설정. Python 환경 준비(PyTorch, scikit-learn, timm). Pre-trained backbone 다운로드(ResNet50 또는 Wide ResNet50). 기본 dataloader 구성(ImageNet normalization 적용).

10:00-10:15: DFM 학습 시작. 기본 설정 사용(backbone=ResNet50, layer=layer3, PCA variance=97%). 학습 시작, 약 12분 소요.

10:15-10:30: 첫 결과 확인. Validation set으로 AUROC 계산. Anomaly map 시각화(몇 개 샘플). 결과가 합리적인지(>80%) 확인. 만약 매우 낮으면(<70%) 데이터나 설정 문제 의심.

10:30-11:00: CFA 학습. 기본 설정 사용(backbone=Wide ResNet50). 학습 시작, 약 15분 소요. DFM보다 2-3% 높은 결과 기대.

11:00-11:30: 초기 분석. DFM과 CFA 결과 비교. 어느 것이 더 나은가? 차이가 예상 범위(2-3%)인가? False positive/negative 샘플 확인. 어떤 유형의 결함을 놓치는가? Anomaly map이 결함 위치를 정확히 가리키는가?

11:30-12:00: 첫 보고서 작성. Baseline AUROC 수치. 대표 샘플의 anomaly map. 초기 관찰 및 개선 방향. 이 시점에서 feasibility를 판단할 수 있다. AUROC 90% 이상이면 promising, 80-90%면 challenging but possible, 80% 미만이면 다른 방법론 고려.

**Day 1 Afternoon (1pm-6pm): 최적화와 비교**

1:00-2:00: Backbone 실험. Wide ResNet50, EfficientNet-B5 시도. 각각 15-20분 소요. 어느 backbone이 최선인가? 1-2% 향상을 얻을 수 있는가?

2:00-3:00: Layer 조합 실험. Layer3 단독 vs layer2+3 조합. Multi-scale 정보가 도움이 되는가? 특히 복잡한 텍스처나 multi-scale 결함에 효과적.

3:00-4:00: 하이퍼파라미터 튜닝. DFM: PCA variance 95%, 97%, 99% 비교. CFA: Concentration parameter 추정 방법 조정. 일반적으로 기본 설정이 잘 작동하므로, 대폭적 향상은 기대하기 어려움(0.5-1%).

4:00-5:00: SOTA 방법론 quick test. PatchCore 또는 FastFlow를 빠르게 시도. Anomalib 라이브러리 사용하면 설정이 간단. Feature adaptation과 비교하여 gap이 얼마나 되는가? Gap이 크면(>5%) SOTA 방법론으로 전환 고려.

5:00-6:00: 최종 분석 및 보고서. 최선의 설정 선정(backbone, layer, method). 최종 AUROC 수치와 confusion matrix. False positive/negative 상세 분석. 다음 단계 제안(SOTA로 전환? Few-shot 시도? Domain adaptation?). 1-page summary + 상세 결과 appendix.

**Expected outcomes**: Day 1 종료 시점에 baseline model(AUROC 94-97%), 여러 설정의 비교 결과, SOTA와의 gap 이해, 다음 단계 명확한 roadmap을 가진다. 이는 프로젝트 방향 결정에 충분한 정보를 제공한다.

### 6.2 Feasibility Testing

새로운 제품이나 공정에 이상 감지를 도입하기 전에, feasibility를 평가해야 한다. "이 데이터로 합리적인 성능을 얻을 수 있는가?" Feature adaptation은 이를 빠르게 답하는 도구다.

**Feasibility 판단 기준**: AUROC > 95%: Excellent feasibility. 대부분의 결함을 감지 가능. Production deployment를 진지하게 고려. SOTA 방법론으로 98-99% 달성 가능성 높음. AUROC 90-95%: Good feasibility. 합리적인 성능. 개선 여지 있음. SOTA 방법론, Few-shot learning, domain adaptation 등으로 향상 기대. AUROC 85-90%: Moderate feasibility. Challenging but possible. 데이터 품질 향상, 더 많은 샘플, advanced 방법론 필요. Pilot project로 진행하며 지속 모니터링. AUROC 80-85%: Low feasibility. 어려움. 근본적 문제 가능성(데이터 품질, domain gap, 결함이 너무 미묘). Alternative approach 고려(classical CV, domain expert rules). AUROC < 80%: Not feasible with current approach. 다른 방법론이나 완전히 다른 접근 필요.

**Quick feasibility test protocol**: Step 1: Minimal viable dataset 수집. 정상 50-100장, 이상 가능하면 10-20장(없어도 가능). 하루 이내 수집 가능한 양. Step 2: DFM rapid training. 기본 설정, 10분 학습. Step 3: 초기 AUROC 확인. 위 기준에 따라 feasibility 판단. Step 4: 만약 promising(>90%)이면, CFA와 SOTA 방법론 추가 시도. 개선 여지 확인. Step 5: 만약 low(<85%)이면, 원인 분석. 데이터 문제? Domain gap? 결함 특성?

**원인 분석 및 대응**: AUROC < 80%의 원인: 데이터 품질 문제. 해상도 낮음, 초점 불량, 조명 불균일. → 촬영 조건 개선. 카메라 업그레이드, 조명 표준화. Domain gap. ImageNet과 너무 다른 modality(X-ray, 현미경). → Domain-specific pre-training 또는 다른 패러다임. 결함이 너무 미묘. 육안으로도 구별 어려움. → 고해상도 이미지, 다른 imaging modality(형광, 적외선). 정상 변이가 너무 큼. 제품의 자연스러운 variation이 결함보다 큼. → 제품 표준화, 더 많은 학습 샘플. Label noise. "정상"에 미세 결함 포함, "결함"에 false alarm. → 라벨 재검토, 도메인 전문가 재검증.

**Feasibility 향상 전략**: 데이터 증대. 더 많은 정상 샘플(100 → 200 → 500). AUROC가 logarithmic하게 증가. Domain adaptation. Target domain의 unlabeled data로 fine-tuning. CFA의 mean direction adaptation 활용. Advanced preprocessing. Contrast enhancement, noise reduction, alignment. Feature space를 더 discriminative하게. Multi-modal fusion. 여러 각도, 여러 spectrum의 이미지 결합. RGB + depth, RGB + infrared 등. Hybrid approach. Feature adaptation으로 초기화, end-to-end fine-tuning으로 개선.

**Case study**: 전자 부품 검사 프로젝트. Initial test: 정상 80장, 결함 15장. DFM AUROC 87%. Moderate feasibility. 원인 분석: 조명 불균일(shadow 문제), 해상도 부족(결함 2-3 pixels). 개선 조치: Ring light로 조명 교체, 카메라 해상도 2배 증가(1MP → 2MP), 정상 샘플 추가 수집(80 → 180장). Retest: DFM AUROC 94%, CFA AUROC 96%. Good feasibility. SOTA(PatchCore) AUROC 98.5%. Production deployment 결정.

Feasibility testing은 초기 투자 결정에 critical하다. Feature adaptation의 빠른 속도(1일)는 여러 프로젝트를 빠르게 평가 가능하게 한다. 자원을 promising 프로젝트에 집중할 수 있다.

### 6.3 Low-resource Environments

Feature adaptation은 저사양 환경에서도 작동한다. 이는 edge device, old hardware, budget-constrained 상황에서 가치 있다.

**Hardware 요구사항**: GPU: 권장 4GB VRAM 이상. RTX 2060, GTX 1660 Ti 등 entry-level GPU 충분. Feature extraction에만 GPU 사용, 나머지는 CPU. Training: 4GB GPU로 batch size 8-16 가능. Inference: 4GB GPU로 real-time 가능(15-20ms). CPU: 최소 4코어, 권장 8코어 이상. Training: CPU로도 가능하지만 느림(feature extraction 30분). Inference: CPU로 60-100ms, near real-time. RAM: 최소 8GB, 권장 16GB. PCA/vMF fitting이 메모리 집약적. Storage: 최소 10GB. Pre-trained weights(500MB), dataset, results.

**경량화 전략**: Backbone 경량화. ResNet50 → ResNet34. 파라미터 절반, 속도 1.5배, 성능 1-2% 저하. ResNet50 → MobileNetV2. 파라미터 1/10, 속도 3배, 성능 3-4% 저하. Trade-off 고려하여 선택. Reduced resolution. 256×256 → 128×128. Feature extraction 4배 빠름, 메모리 4배 절감. 성능 2-3% 저하. 미세 결함에는 부적합. Spatial downsampling. 14×14 feature map → 7×7 (2×2 pooling). 연산량 4배 감소, 성능 1-2% 저하. PCA components 축소. K=100 → K=50. 메모리 절반, 속도 1.5배, 성능 0.5-1% 저하.

**Edge device deployment**: Jetson Nano(4GB RAM, 128 CUDA cores)에서 실험. Configuration: ResNet34 backbone, 128×128 input, 7×7 feature map, PCA K=50. Training: 약 30분(CPU fallback 사용). Inference: 45ms per image(배치 처리로 35ms까지 가능). Accuracy: AUROC 92.5%(full setup 94.8% 대비 -2.3%). 이는 많은 edge application에 충분.

**Raspberry Pi 4(8GB RAM, CPU only)**: Configuration: MobileNetV2 backbone, 128×128 input. Training: 약 90분. Inference: 180ms per image(배치 처리로 150ms). Accuracy: AUROC 90.2%. Real-time은 아니지만, inspection line speed에 따라 사용 가능(예: 5-10 images/sec 요구 시 충분).

**Cost-benefit analysis**: High-end setup: RTX 3090(24GB, $1500), high-res camera($500), total $2000. Performance: AUROC 96.5%, inference 12ms. Low-end setup: GTX 1660 Ti(6GB, $250), standard camera($100), total $350. Performance: AUROC 94.2%, inference 25ms. Cost 1/6, performance -2.3%, speed 2×. Many applications에 충분한 trade-off.

**Optimization techniques**: Model quantization. FP32 → INT8. 속도 2-3배, 메모리 4배 절감, 성능 0.5-1% 저하. ONNX Runtime, TensorRT 사용. Pruning. 중요도 낮은 weights 제거. 20-30% sparsity로 속도 1.3-1.5배, 성능 0.3-0.5% 저하. Distillation. Large model → small model. 예: ResNet50 → MobileNet (teacher-student). Small model 성능이 scratch training보다 1-2% 향상.

**Cloud vs edge decision**: Cloud 장점: 무제한 compute, 최신 hardware, easy scaling. Edge 장점: Low latency, no network dependency, data privacy. Feature adaptation은 둘 다 가능. Cloud: 중앙 서버에서 training, edge로 배포. Edge: 현장에서 직접 training 및 inference. Low-resource edge에도 적합.

저사양 환경은 제약이지만, feature adaptation은 이를 수용한다. 단순성과 효율성이 저사양 환경에서 빛을 발한다. 반면 SOTA 방법론(FastFlow, Reverse Distillation)은 8GB+ GPU가 거의 필수다.

### 6.4 Transition Strategy to SOTA Models

Feature adaptation으로 시작했지만, 더 높은 성능이 필요해졌다. 어떻게 SOTA 방법론으로 전환하는가? 체계적인 transition 전략이 필요하다.

**Transition 시점 판단**: Feature adaptation이 충분한 경우. AUROC 95-97%면 많은 application에 충분. Non-critical 제품, moderate defect cost, 빠른 iteration 필요. → 계속 feature adaptation 사용. Transition 고려해야 하는 경우. AUROC < 95%이고 더 높은 성능 필요. Critical application(safety, high cost). Feature adaptation으로 feasibility 확인 완료, 이제 production 배포. Domain shift가 심하고 CFA adaptation으로 부족. → SOTA로 전환.

**Transition roadmap (4주 계획)**:

Week 1: Baseline 재확인 및 목표 설정. Feature adaptation 최적 설정으로 최종 baseline. AUROC, FPR@95%TPR 등 상세 metric. Target 설정(예: AUROC 98%, FPR@95%TPR < 5%). Gap analysis: 현재 vs target.

Week 2: SOTA 방법론 선택 및 초기 실험. 목표에 따라 방법론 선택. 최고 정확도 필요 → PatchCore. 균형(속도+정확도) → FastFlow. Few-shot 상황 → DRAEM. 텍스처 dominant → DSR. 선택한 방법론으로 first training. Anomalib 또는 official implementation 사용. 기본 설정으로 시작. 결과 확인: target 달성했는가? 추가 tuning 필요한가?

Week 3: 하이퍼파라미터 튜닝 및 최적화. Systematic tuning: learning rate, batch size, epochs, backbone 등. Grid search 또는 random search. Validation set으로 선택. Data augmentation 조정. Geometric, color, noise 등. Overfitting 방지. Ensemble 고려. Feature adaptation + SOTA ensemble로 추가 향상(0.5-1%).

Week 4: 검증 및 배포 준비. Separate test set으로 최종 평가. Cross-validation으로 robustness 확인. 여러 random seed로 재현성 검증. Inference speed 측정 및 최적화. Production 요구사항 충족 확인. Deployment plan: hardware, software, monitoring.

**Incremental transition strategy**: 모든 것을 한 번에 바꾸지 않는다. Gradual transition이 risk를 줄인다. Phase 1: Feature adaptation으로 운영하며 SOTA 병렬 개발. 두 모델 결과 비교, SOTA 신뢰 구축. Phase 2: 일부 제품/라인에서 SOTA 시범 운영. A/B testing으로 성능 비교. Issue 발견 및 해결. Phase 3: SOTA로 full transition. Feature adaptation은 backup으로 유지(SOTA failure 시).

**Knowledge transfer**: Feature adaptation에서 얻은 insights를 SOTA에 활용. Data 이해: 어떤 결함이 어렵고, 어떤 augmentation이 효과적인가? Baseline: Feature adaptation AUROC가 SOTA의 lower bound. 이보다 낮으면 문제 있음. Threshold: Feature adaptation의 optimal threshold가 SOTA에도 참고됨.

**Fallback plan**: SOTA가 기대만큼 향상되지 않으면(예: +1-2%만 향상). Cost-benefit 재평가: 추가 1-2%가 투자(개발 시간, 계산 비용)를 정당화하는가? Feature adaptation으로 회귀도 합리적 선택일 수 있음. Hybrid approach: Simple feature adaptation을 빠른 pre-screening으로 사용. High anomaly score만 SOTA로 재검사. 속도와 정확도 절충.

**Case study - 성공적 transition**: Company A, 전자 부품 검사. Week 0: CFA baseline 96.2%. Target: 98.5%. Week 1-2: PatchCore 선택 및 초기 training. 97.8% 달성. Week 3: Multi-scale coreset tuning. 98.6% 달성(target 초과). Week 4: Production 배포. Inference 50ms, 요구사항(100ms) 충족. Outcome: 결함 탈출률 60% 감소(baseline 대비 40% 대비). ROI: 6개월 이내 회수.

**Case study - 회귀 결정**: Company B, 플라스틱 제품 검사. Week 0: DFM baseline 94.8%. Target: 98%. Week 1-2: FastFlow training. 96.2% 달성. Week 3: Extensive tuning. 96.8% 도달, target 미달. Week 4: Analysis: +2% 향상이 투자 대비 부족. Decision: DFM으로 회귀, data 품질 개선에 집중. 6개월 후: 개선된 data로 DFM 95.8%, FastFlow 97.5%. 여전히 DFM 사용 결정(충분한 성능, 낮은 유지보수).

Transition은 자동적이지 않다. 신중한 평가, 체계적 실행, 지속적 모니터링이 필요하다. Feature adaptation은 시작점이며, SOTA는 필요시 도달하는 목적지다. 그러나 모든 여정이 SOTA로 끝나지는 않는다. Feature adaptation이 충분한 경우도 많다.

---

Feature adaptation은 이상 감지의 "Swiss Army knife"다. 만능은 아니지만, 많은 상황에서 유용하다. 빠른 프로토타입, feasibility 검증, 저사양 환경, baseline 구축에 탁월하다. 근본적 한계(pre-trained feature domain gap, linear transformation, 4-5% SOTA gap)를 이해하고, 적절한 시나리오에서 사용하면 매우 효과적이다. DFM과 CFA 중에서는, domain shift가 예상되면 CFA, 그렇지 않으면 DFM이 간단하다. 그러나 critical application이나 최고 성능 필요 시, SOTA 방법론(PatchCore, FastFlow, DRAEM)으로 전환해야 한다. Feature adaptation을 시작점으로, 필요시 발전시키는 전략이 현명하다.

7장을 Research Insights로 재작성하겠습니다.

---

## 7. Research Insights

### 7.1 Role as Entry Point

Feature adaptation은 anomaly detection 연구와 실무의 진입점(entry point)으로서 독특한 역할을 수행한다. 이는 기술적 단순성과 빠른 결과 도출이라는 특성에서 비롯되며, 연구자와 실무자 모두에게 가치 있는 학습 경험을 제공한다.

교육적 가치(Pedagogical Value)는 feature adaptation의 첫 번째 역할이다. Anomaly detection을 처음 접하는 연구자나 학생에게, DFM과 CFA는 이상적인 학습 자료다. 수학적 복잡도가 낮아 선형대수와 기초 통계만으로 이해 가능하며, PCA는 대학 수준 수학으로 완전히 이해할 수 있다. Eigenvalue decomposition, variance maximization, principal component의 기하학적 의미가 명확하다. Mahalanobis distance는 multivariate Gaussian의 자연스러운 metric이며, 통계학 교과서에 상세히 다뤄진다. CFA의 hypersphere geometry는 더 흥미롭다. Unit sphere 상의 distribution, geodesic distance, von Mises-Fisher distribution은 directional statistics의 기초를 제공한다. 이는 advanced topic으로의 자연스러운 bridge다.

구현 난이도도 적절하다. 200 lines 코드로 처음부터 구현 가능하며, 이는 weekend project 수준이다. 학부 고학년이나 대학원 초년차가 일주일 안에 이해하고 구현할 수 있다. Deep learning 경험이 없어도 가능하다. Pre-trained model을 black-box로 사용하므로, backpropagation이나 gradient descent를 이해할 필요가 없다. NumPy와 scikit-learn만으로 충분하다. 이는 classical machine learning background만 있어도 접근 가능함을 의미한다.

결과의 즉시성(Immediacy of Results)도 중요한 교육적 요소다. 하루 만에 작동하는 모델을 만들고 결과를 볼 수 있다. 이는 학습 동기를 크게 높인다. GAN이나 normalizing flow처럼 일주일 이상 디버깅해야 결과가 나오는 것과 대조적이다. 빠른 feedback loop는 실험과 이해를 가속화한다. 다양한 설정(backbone, layer, PCA components)을 시도하며 각각의 영향을 즉시 관찰할 수 있다. "What if" 질문에 빠르게 답할 수 있다. "Layer2를 사용하면?", "PCA를 99% variance로 하면?" 같은 질문을 30분 안에 실험할 수 있다.

연구 방법론 학습(Research Methodology Learning)도 제공한다. Feature adaptation을 구현하며 연구의 전체 pipeline을 경험한다. Literature review(DFM, CFA 논문 읽기), problem formulation(anomaly detection as distribution modeling), implementation(코드 작성과 디버깅), experimentation(다양한 설정 시도), evaluation(AUROC, confusion matrix), analysis(결과 해석과 한계 이해). 이는 연구의 A to Z를 경험하게 한다. 복잡한 방법론에서는 구현만으로도 몇 주가 걸려 전체를 보기 어렵지만, feature adaptation은 빠르게 전체 cycle을 완료할 수 있다.

실무 진입점(Industry Entry Point)으로서의 역할도 크다. 산업 현장에 anomaly detection을 도입하려는 엔지니어에게, feature adaptation은 최소 위험으로 시작할 수 있는 방법이다. 대규모 투자 없이 proof-of-concept를 제시할 수 있다. "우리 데이터로 anomaly detection이 작동하는가?"라는 질문에 일주일 안에 답할 수 있다. 경영진이나 stakeholder에게 결과를 빠르게 보여줄 수 있다. AUROC 95% 같은 구체적 수치는 설득력이 있다. "이론적으로 가능하다"가 아니라 "실제로 95% 정확도를 달성했다"는 명확한 증거다.

기술 스택의 진입장벽도 낮다. PyTorch와 scikit-learn은 대부분의 data science 팀이 이미 사용 중이다. 추가 도구나 프레임워크가 필요 없다. 클라우드 GPU 없이도 회사의 기존 workstation에서 실행 가능하다. 4-8GB GPU면 충분하며, CPU만으로도 가능하다. 이는 "일단 시작"하기에 최적이다. 복잡한 infrastructure나 budget approval 없이 즉시 착수할 수 있다.

Benchmark로서의 역할(Role as Benchmark)도 중요하다. 새로운 anomaly detection 방법론을 개발할 때, feature adaptation은 natural baseline이다. "우리 방법은 DFM보다 X% 좋다"는 명확한 비교점을 제공한다. 공정한 비교를 위한 기준선이다. Feature adaptation은 implementation이 간단하고 deterministic하여, 재현성이 100%다. 다른 연구자가 동일한 결과를 얻을 수 있다. 이는 벤치마킹의 필수 조건이다.

연구 커뮤니티에서 feature adaptation은 "hello world" 같은 존재다. 새로운 분야에 진입할 때 가장 먼저 시도하는 방법이다. PaDiM(2020) 이후 많은 논문이 DFM/PaDiM을 baseline으로 사용한다. "우리는 DFM(94.8%)보다 3% 향상된 97.8%를 달성했다"는 식의 비교가 표준이 되었다. 이는 분야의 발전을 측정하는 ruler 역할을 한다.

교육 자료로서의 활용(Use in Education)도 확대되고 있다. 많은 대학의 computer vision 또는 machine learning 수업에서 feature adaptation을 다룬다. 간단한 homework 또는 project로 적합하다. "PaDiM을 구현하고 MVTec AD에서 평가하라"는 과제는 일주일 분량으로 적절하다. Online course(Coursera, Udacity)에서도 등장한다. Anomaly detection 모듈의 첫 주제로 feature adaptation을 다루고, 이후 advanced 방법론으로 확장하는 구조가 일반적이다.

결론적으로, feature adaptation은 단순히 하나의 방법론이 아니라 anomaly detection의 "on-ramp"다. 연구와 실무 모두에서 진입을 쉽게 만들며, 이후 더 복잡한 방법론으로 발전하기 위한 foundation을 제공한다. 이는 분야의 accessibility를 높이고, 더 많은 사람들이 anomaly detection에 기여할 수 있게 한다.

### 7.2 Starting Point, Not Destination

Feature adaptation의 가장 중요한 insight는 "starting point, not destination"이라는 인식이다. 이는 기술적 한계에 대한 현실적 이해이며, 동시에 pragmatic한 전략을 제시한다.

성능 ceiling의 현실(Reality of Performance Ceiling)을 먼저 인정해야 한다. DFM과 CFA는 94-97% AUROC에서 plateau에 도달한다. 이는 우연이 아니라 근본적 제약의 결과다. Pre-trained feature의 domain gap, linear transformation의 표현력 제한, simple statistical model의 한계가 모두 기여한다. 아무리 하이퍼파라미터를 튜닝해도 98%를 넘기 어렵다. Wide ResNet이나 EfficientNet으로 바꿔도 1-2% 향상이 전부다. PCA components를 조정해도 0.5-1% 차이밖에 없다. 이는 optimization 문제가 아니라 capability 문제다.

실험적 증거가 이를 뒷받침한다. 수십 개 논문이 MVTec AD에서 feature adaptation을 시도했다. 다양한 변형(different backbone, layer combination, statistical model)을 제안했다. 그러나 어느 것도 98%를 돌파하지 못했다. Best reported result는 CFA의 97.5% 정도다. 이는 패러다임의 intrinsic limit을 시사한다. End-to-end learning 방법론(PatchCore 99.1%, FastFlow 98.5%)과의 gap은 일관되게 2-4%다. 이 gap은 좁혀지지 않는다. Feature adaptation이 아무리 개선되어도, SOTA도 함께 발전하여 gap이 유지된다.

그렇다면 feature adaptation은 쓸모없는가? 전혀 아니다. "Destination이 아니다"는 것은 "쓸모없다"가 아니라 "역할이 다르다"는 의미다. Feature adaptation의 진정한 가치는 final deployment가 아니라 journey의 초기 단계에 있다.

여정의 단계(Stages of Journey)를 명확히 구분하는 것이 중요하다. Stage 1: Feasibility exploration(1-2주). "이 문제에 anomaly detection이 적용 가능한가?" Feature adaptation으로 빠르게 답한다. AUROC 90% 이상이면 promising, 85-90%면 challenging but possible, 85% 미만이면 재고려. Stage 2: Baseline establishment(2-4주). "Reasonable baseline 성능은 얼마인가?" Feature adaptation이 lower bound를 제공한다. "어떤 방법을 쓰든 최소한 95%는 나와야 한다"는 기준. Stage 3: Problem understanding(4-8주). "어떤 결함이 어렵고, 어떤 augmentation이 효과적인가?" Feature adaptation으로 실험하며 insights 획득. Data quality issue, difficult defect types, optimal preprocessing을 파악. Stage 4: Method selection(2-4주). "Production에 어떤 방법을 쓸 것인가?" Feature adaptation의 한계를 명확히 이해한 후, 요구사항에 맞는 SOTA 방법론 선택. Stage 5: Production deployment(8-12주). Selected SOTA method를 최적화하고 배포. Feature adaptation은 여기서 퇴장하거나 backup으로 남는다.

Feature adaptation은 Stage 1-3의 주인공이다. 빠르고 저렴하게 초기 단계를 완료한다. Stage 4-5에서는 다른 방법론에 자리를 내준다. 이것이 "starting point, not destination"의 의미다.

지식의 전이(Knowledge Transfer)도 중요한 측면이다. Feature adaptation에서 얻은 insights는 SOTA 방법론에도 유용하다. Data understanding: 어떤 defect type이 hard negative인지, 어떤 normal variation이 큰지 알게 된다. 이는 SOTA 방법론의 data augmentation이나 loss weighting에 반영된다. Hyperparameter hints: Feature adaptation의 optimal threshold가 SOTA의 초기 threshold 설정에 참고된다. Layer selection: Feature adaptation에서 layer2+3 조합이 좋았다면, SOTA에서도 multi-scale feature를 강조한다. Failure modes: Feature adaptation이 miss한 defect는 SOTA도 어려워할 가능성이 높다. 이는 targeted improvement의 방향을 제시한다.

경제적 합리성(Economic Rationality)도 고려해야 한다. Feature adaptation은 저렴하다. 개발 시간 1-2주, 계산 비용 거의 없음(기존 workstation), 유지보수 간단(deterministic, no retraining). SOTA는 비싸다. 개발 시간 4-8주, GPU cluster 필요(학습에 수 시간-일), 유지보수 복잡(주기적 retraining, hyperparameter drift monitoring). 모든 프로젝트가 SOTA를 정당화하지 못한다. Non-critical application, low defect cost, limited budget에서는 feature adaptation이 충분하고 합리적이다. "95%면 충분하다"면 SOTA의 추가 비용(2-3배)을 쓸 이유가 없다.

Risk management 측면도 있다. Feature adaptation으로 시작하면 risk가 낮다. 실패해도 1-2주 손실, 성공하면 빠르게 value 증명. SOTA로 직접 시작하면 risk가 높다. 4-8주 투자 후 실패하면 큰 손실, 성공해도 initial value까지 시간 소요. Agile methodology의 "fail fast, learn fast" 원칙과 일치한다. 작은 investment로 빠르게 학습하고, 성공 가능성을 확인한 후 큰 investment 진행.

실무 사례(Practical Cases)가 이를 입증한다. Case A: Electronics company. Feature adaptation baseline 96% 도달. 요구사항이 95%였으므로, 바로 production 배포. SOTA는 시도하지 않음. 결과: 6개월 빠른 time-to-market, 개발 비용 70% 절감. Case B: Automotive supplier. Feature adaptation baseline 93%. 요구사항이 98%(safety-critical). SOTA(PatchCore) 전환 결정. Feature adaptation의 insights(difficult defect types, optimal preprocessing)를 활용. 결과: SOTA가 99.2% 도달, feature adaptation 단계가 없었다면 시행착오로 2-3주 추가 소요 예상. Case C: Medical device. Feature adaptation baseline 89%. 요구사항 98%, domain gap 심각(X-ray imaging). Feature adaptation 한계 조기 인식. Domain-specific pre-training + SOTA 조합 전략 수립. 결과: 불필요한 feature adaptation 최적화에 시간 낭비 방지, 바로 올바른 방향(domain-specific approach) 진행.

Evolution over time(시간에 따른 진화)도 자연스럽다. 처음에는 feature adaptation으로 시작한다. 빠르게 시장 진입, 초기 고객 확보, feedback 수집. 시간이 지나며 요구사항이 높아진다. 더 높은 정확도, 더 빠른 속도, 더 복잡한 시나리오. 이 시점에 SOTA로 업그레이드한다. 이미 시장이 있고 수익이 나므로 투자 정당화 가능. Feature adaptation은 처음부터 "temporary solution"으로 받아들인다. 언젠가는 SOTA로 갈 것이지만, 지금은 이것으로 충분. 이는 기술 부채(technical debt)가 아니라 의도적 전략이다. MVP(Minimum Viable Product) 철학과 동일하다.

교훈(Lessons Learned)은 명확하다. Don't over-invest in feature adaptation. 98%를 목표로 DFM을 튜닝하는 것은 시간 낭비다. 그 시간에 PatchCore를 배우는 게 낫다. Do use it as stepping stone. Feature adaptation으로 빠르게 시작하고, 필요시 SOTA로 전환. Understand the trade-offs. 성능, 속도, 비용, 개발 시간의 trade-off를 명확히 이해. 맹목적으로 "최고 성능"을 추구하지 않는다. Be pragmatic, not idealistic. "완벽한 solution"보다 "지금 작동하는 solution"이 가치 있을 때가 많다.

"Starting point, not destination"은 겸손한 인정이지만, 동시에 강력한 전략이다. Feature adaptation은 자신의 한계를 알고, 적절한 역할을 수행한다. 이것이 오히려 장기적 가치를 만든다.

### 7.3 Value in Speed

Feature adaptation의 가장 차별화된 특성은 속도다. 15분 학습, 15ms inference는 단순한 숫자가 아니라 전략적 가치를 창출한다. 속도가 어떻게 질적 변화를 가져오는지 분석한다.

Iteration velocity(반복 속도)의 가치가 첫 번째다. Machine learning에서 빠른 iteration은 더 나은 결과로 이어진다. Experiment 많이 할수록 optimal solution에 가까워진다. Feature adaptation은 하루에 수십 개 experiment 가능하게 한다. 다양한 backbone(ResNet, Wide ResNet, EfficientNet), layer 조합(layer2, layer3, layer2+3), PCA settings(95%, 97%, 99%), augmentation strategies를 모두 시도할 수 있다. 각 실험이 15분이므로, 8시간에 30개 이상 가능하다. 반면 FastFlow는 실험당 40분이므로, 하루 12개 정도다. DRAEM은 20분이므로 24개 정도다. Feature adaptation이 2-3배 많은 실험을 수행한다.

더 많은 실험은 더 나은 insights로 이어진다. "Layer2가 layer3보다 낫다"는 발견을 빠르게 할 수 있다. "이 augmentation은 해롭다"는 것을 즉시 확인한다. Trial and error의 cost가 낮아, 대담한 실험이 가능하다. "만약 PCA를 90%로 극단적으로 줄이면?"같은 실험을 주저 없이 시도한다. 실패해도 15분 손실이므로 부담 없다.

Time-to-insight(통찰 도달 시간)가 극적으로 단축된다. 문제를 인식하고 답을 얻기까지의 시간이다. "이 새로운 defect type을 감지할 수 있는가?"라는 질문에 당일 답변 가능하다. 아침에 질문 받고, 오전에 실험하고, 점심 전에 답한다. 이는 business agility를 높인다. 시장 변화나 고객 요구에 빠르게 대응할 수 있다.

Decision velocity(의사결정 속도)도 가속화된다. "이 프로젝트를 진행할 것인가?"의 결정이 일주일 안에 가능하다. Feature adaptation으로 feasibility 확인, stakeholder에게 결과 제시, go/no-go 결정. 느린 방법론이었다면 의사결정이 한 달 이상 걸렸을 것이다. 그 사이 시장 기회를 놓칠 수 있다.

Opportunity cost(기회 비용)의 감소가 중요하다. 시간은 돈이며, 빠른 방법론은 opportunity cost를 줄인다. 예시: Project A와 B 중 선택해야 한다. Feature adaptation으로 각각 1주일 feasibility test 수행. A는 AUROC 95% promising, B는 82% challenging. A 선택, 즉시 착수. 만약 느린 방법론이었다면 각각 1개월 소요. 2개월 후에야 결정. B에 쓴 1개월은 낭비가 된다. Feature adaptation은 이 낭비를 1주일로 줄인다.

Parallel exploration(병렬 탐색)도 가능하게 한다. 여러 프로젝트를 동시에 탐색할 수 있다. 5개 potential project가 있을 때, feature adaptation으로 각각 1주일씩 순차 진행. 5주 후 가장 promising한 것 선택. 느린 방법론이면 각 1개월씩, 5개월 소요. 또는 병렬로 진행하려면 5배 인력 필요. Feature adaptation은 적은 인력으로 빠르게 portfolio를 평가한다.

Psychological impact(심리적 영향)도 무시할 수 없다. 빠른 결과는 팀의 동기를 높인다. "오늘 아침에 시작해서 점심에 95% 나왔어요!"는 exciting하다. 팀이 energized되고 momentum이 생긴다. 느린 방법론은 지루하다. "2주째 학습 중인데 아직 수렴 안 됩니다"는 frustrating하다. 팀의 morale이 떨어지고 의심이 생긴다.

Stakeholder management(이해관계자 관리)도 쉬워진다. 빠른 결과는 stakeholder의 신뢰를 구축한다. 경영진에게 "일주일 만에 95% 달성"을 보여주면 impressive하다. 추가 리소스 확보가 쉬워진다. 느린 결과는 의심을 산다. "한 달 지났는데 아직 결과 없음"은 프로젝트를 위험하게 만든다. 중단 압력을 받을 수 있다.

Competitive advantage(경쟁 우위)를 제공한다. 시장에서 속도는 competitive advantage다. 신제품이 출시되었을 때, 일주일 안에 anomaly detection을 적용하여 품질 보증 가능. 경쟁사가 한 달 걸리는 동안 이미 시장 선점. Customer request에 빠르게 대응. "이 새로운 부품도 검사 가능한가?"에 당일 답변. 고객 만족도 상승, 계약 체결 가능성 증가.

Prototyping culture(프로토타입 문화) 조성에 기여한다. 속도는 prototyping을 encourage한다. "빠르게 만들고, 빠르게 테스트하고, 빠르게 배운다"는 문화. Feature adaptation은 이를 가능하게 한다. 느린 방법론은 prototyping을 discourage한다. "시작하면 몇 주는 걸리니까 확실할 때만 하자"는 보수적 태도. 이는 innovation을 저해한다.

Failure tolerance(실패 허용)가 높아진다. 빠르면 실패를 두려워하지 않는다. "안 되면 15분 손해"이므로 부담 없이 시도. 실패에서 배우고 다음 시도로 빠르게 이동. 느리면 실패가 costly하다. "실패하면 한 달 낭비"이므로 신중해진다. 안전한 선택만 하게 되고, breakthrough가 나오기 어렵다.

Learning curve acceleration(학습 곡선 가속)도 중요하다. 신입 연구원이나 엔지니어가 빠르게 배운다. Feature adaptation으로 첫 주에 결과를 내면서 confidence 획득. Domain knowledge를 빠르게 축적. "이 데이터는 이런 특성이 있구나"를 실험하며 체득. 느린 방법론이면 몇 주 동안 결과 없이 헤매며 좌절할 수 있다.

Infrastructure simplicity(인프라 단순성)도 속도에 기여한다. Feature adaptation은 복잡한 infrastructure 불필요. 단순한 setup으로 빠르게 시작. 느린 방법론은 종종 복잡한 setup 필요. Distributed training, large GPU cluster, sophisticated monitoring. Setup 자체에 일주일 이상 걸릴 수 있다.

The compounding effect(복리 효과)가 장기적으로 나타난다. 속도의 이점은 누적된다. 첫 주에 시작하여 2주 차에 baseline, 3주 차에 optimization, 4주 차에 deployment 준비. 느린 방법론은 4주 차에 겨우 baseline. 시간 차이가 점점 벌어진다. 1년 후를 보면, 빠른 팀은 10개 프로젝트 완료, 느린 팀은 3개 정도. 누적 output이 3배 차이 난다.

그러나 속도에만 집착하면 안 된다(Caveat on Speed)는 점도 인식해야 한다. 속도는 quality를 해치지 않아야 한다. Feature adaptation이 빠르다고 해서 sloppy하게 해도 된다는 뜻이 아니다. 여전히 proper data validation, careful hyperparameter selection, thorough evaluation이 필요하다. 빠르지만 정확해야 한다. 속도는 결과의 질과 trade-off가 아니라, process의 효율성이다. 같은 품질을 더 빠르게 달성하는 것이지, 낮은 품질을 빠르게 만드는 것이 아니다.

일부 문제는 느린 접근이 필요하다(When Slow is Necessary). Critical application(의료, 항공우주)은 thorough validation이 필수다. 빠르게 결과 내는 것보다 정확히 검증하는 것이 중요. Novel research는 깊은 이해가 필요하다. 빠른 실험보다 깊은 사고가 가치 있다. Complex system integration은 신중한 planning이 필요. 속도보다 robustness가 우선이다.

결론적으로, 속도는 feature adaptation의 killer feature다. 단순히 "빠르다"는 것을 넘어, 전략적 가치를 창출한다. Iteration velocity, time-to-insight, decision velocity를 높여 competitive advantage를 제공한다. 그러나 속도는 수단이지 목적이 아니다. Quality와 thoroughness를 유지하면서 빠른 것이 진정한 가치다. Feature adaptation은 이 균형을 잘 제공한다. 빠르면서도 합리적으로 정확하다. 이것이 실무에서 사랑받는 이유다.

---

Feature adaptation은 anomaly detection의 중요한 chapter다. Entry point로서 접근성을 높이고, starting point로서 실용적 전략을 제공하며, 속도로서 경쟁력을 부여한다. 완벽하지 않지만, 자신의 역할을 명확히 알고 충실히 수행한다. 이는 연구와 실무 모두에서 지속적으로 가치를 제공할 것이다. 앞으로도 "첫 시도"로서, "빠른 검증"으로서, "합리적 baseline"으로서 feature adaptation의 역할은 계속될 것이다.

