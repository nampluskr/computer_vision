# Normalizing Flow for Anomaly Detection

# 1. Paradigm Overview

## 1.1 Core Principle

Normalizing Flow는 이상 탐지에 확률적 생성 모델을 적용하는 패러다임으로, 정상 데이터의 복잡한 확률 분포를 명시적으로 학습한다. Memory-based 방법이 거리나 밀도를 암묵적으로 사용하는 반면, Normalizing Flow는 정확한 확률 밀도 함수를 구성한다. 이는 이상 탐지를 확률론적 문제로 정식화하여 이론적으로 well-founded된 접근을 제공한다.

핵심 아이디어는 단순한 base distribution(일반적으로 표준 가우시안)을 일련의 가역적(invertible) 변환을 통해 복잡한 데이터 분포로 변환하는 것이다. 이러한 변환들이 flow를 구성하며, 각 단계에서 확률 밀도가 어떻게 변하는지 정확히 추적할 수 있다. 결과적으로 임의의 데이터 포인트에 대해 정확한 log-likelihood를 계산할 수 있다.

정상 데이터로 flow를 학습하면 정상 분포를 모델링하게 된다. 테스트 시 새로운 샘플의 log-likelihood를 평가한다. 높은 likelihood를 가진 샘플은 정상 분포 내에 있으므로 정상으로, 낮은 likelihood는 분포 밖에 있으므로 이상으로 판정된다. 이는 직관적이고 수학적으로 명확한 이상도 측정을 제공한다.

Normalizing Flow의 가장 큰 장점은 표현력과 tractability의 균형이다. Variational Autoencoder(VAE)와 달리 approximate inference가 필요 없고, Generative Adversarial Network(GAN)과 달리 training이 안정적이다. Exact log-likelihood를 최대화하는 objective는 명확하고 수렴이 잘 된다. 또한 latent space가 명시적으로 정의되어 해석 가능하다.

이상 탐지에서 Normalizing Flow는 2021년경부터 주목받기 시작했다. CFlow, FastFlow, CS-Flow, U-Flow 등이 제안되며 MVTec AD 벤치마크에서 competitive한 성능을 보였다. 특히 FastFlow는 98.5% AUROC를 달성하면서도 추론 속도가 20-50ms로 빨라 성능-속도 균형의 대표 모델이 되었다. 이들은 memory-based 방법(99.1%)보다는 약간 낮지만, reconstruction-based나 feature adaptation 방법보다는 우수한 성능을 보였다.

Normalizing Flow 패러다임의 발전은 두 가지 방향으로 진행되었다. 첫째는 표현력 향상이다. 더 복잡한 flow architecture로 multimodal이나 high-dimensional 분포를 더 잘 모델링한다. Coupling layers, autoregressive flows, continuous normalizing flows 등 다양한 변형이 제안되었다. 둘째는 효율성 개선이다. Flow의 계산 비용을 줄이고 메모리를 절약하여 실시간 추론과 대규모 배포를 가능하게 한다. FastFlow는 이 방향의 성공적인 예다.

그러나 Normalizing Flow도 한계가 있다. 고차원 데이터에서 충분히 표현력 있는 flow를 구성하는 것은 계산적으로 expensive하다. 또한 invertibility 제약으로 인해 architecture 설계가 제한적이다. 학습에도 상당한 시간이 걸려 memory-based 방법(수 분)보다 느리다(수 시간). 이러한 trade-off를 이해하고 적절한 상황에서 활용하는 것이 중요하다.

Normalizing Flow는 이론적 우아함과 실용적 효과를 결합한 패러다임이다. 확률적 해석 가능성은 신뢰할 수 있는 이상도 측정을 제공하고, 생성 능력은 데이터 증강이나 anomaly synthesis에도 활용될 수 있다. Memory-based 방법이 최고 정확도를 제공하고 knowledge distillation이 최고 속도를 제공한다면, Normalizing Flow는 이론적 명확성과 성능-속도 균형의 중간 지점을 차지한다.

## 1.2 Mathematical Foundation

### 1.2.1 Change of Variables

Normalizing Flow의 수학적 기반은 확률 이론의 change of variables formula다. 확률 변수 $\mathbf{z}$가 알려진 분포 $p_Z(\mathbf{z})$를 따르고, 가역 함수 $f: \mathbb{R}^d \to \mathbb{R}^d$를 통해 $\mathbf{x} = f(\mathbf{z})$로 변환될 때, $\mathbf{x}$의 확률 밀도는 다음과 같이 계산된다.

$$p_X(\mathbf{x}) = p_Z(\mathbf{z}) \left| \det \frac{\partial f^{-1}}{\partial \mathbf{x}} \right| = p_Z(f^{-1}(\mathbf{x})) \left| \det \frac{\partial f^{-1}}{\partial \mathbf{x}} \right|$$

여기서 $\det \frac{\partial f^{-1}}{\partial \mathbf{x}}$는 역변환 $f^{-1}$의 Jacobian 행렬의 determinant다. 이는 변환이 volume을 어떻게 변화시키는지 측정한다. Jacobian determinant의 절댓값은 volume scaling factor다.

직관적으로 확률 밀도는 확률 질량을 volume으로 나눈 것이다. 변환 $f$가 volume을 확장하면($|\det J| > 1$) 밀도가 감소하고, 압축하면($|\det J| < 1$) 밀도가 증가한다. Change of variables formula는 이를 정확히 정량화한다.

Jacobian determinant의 역수를 사용하는 형태도 있다.

$$p_X(\mathbf{x}) = p_Z(\mathbf{z}) \left| \det \frac{\partial f}{\partial \mathbf{z}} \right|^{-1}$$

여기서 $\det \frac{\partial f}{\partial \mathbf{z}}$는 forward transformation의 Jacobian determinant다. 이 두 형태는 inverse function theorem에 의해 동등하다.

$$\det \frac{\partial f^{-1}}{\partial \mathbf{x}} = \left( \det \frac{\partial f}{\partial \mathbf{z}} \right)^{-1}$$

실무에서는 forward Jacobian을 사용하는 것이 일반적이다. Forward pass $\mathbf{z} \to \mathbf{x}$에서 Jacobian을 계산하고, log-likelihood는 다음과 같이 표현된다.

$$\log p_X(\mathbf{x}) = \log p_Z(f^{-1}(\mathbf{x})) - \log \left| \det \frac{\partial f}{\partial \mathbf{z}} \right|$$

로그를 취하면 곱셈이 덧셈으로 바뀌어 수치적 안정성이 향상된다. 또한 여러 변환을 연쇄적으로 적용할 때 log-likelihood의 변화를 단순히 더하면 된다.

일련의 변환 $f = f_K \circ f_{K-1} \circ \cdots \circ f_1$을 적용하면, 각 단계의 기여가 누적된다.

$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}_0) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}} \right|$$

여기서 $\mathbf{z}_0 = \mathbf{z}$는 초기 latent variable이고, $\mathbf{z}_k = f_k(\mathbf{z}_{k-1})$는 $k$번째 변환 후의 중간 변수다. 최종적으로 $\mathbf{x} = \mathbf{z}_K$다.

이 공식은 Normalizing Flow의 핵심이다. 단순한 base distribution에서 시작하여 복잡한 데이터 distribution을 정확한 likelihood로 표현할 수 있다. 각 변환의 Jacobian determinant만 계산 가능하면 전체 flow의 likelihood가 tractable하다.

### 1.2.2 Invertible Transformations

Normalizing Flow의 변환들은 가역적(invertible)이어야 한다. 즉, $f: \mathbb{R}^d \to \mathbb{R}^d$가 bijection이어야 한다. 이는 두 가지 이유로 중요하다. 첫째, inverse $f^{-1}$이 존재해야 $\mathbf{x}$로부터 $\mathbf{z}$를 계산할 수 있다. 이는 log-likelihood 계산에 필요하다. 둘째, Jacobian determinant가 non-zero여야 확률 보존이 보장된다.

가역 변환을 설계하는 것은 제약이다. 일반적인 신경망은 가역적이지 않다. 예를 들어 ReLU activation은 음수를 0으로 매핑하여 정보를 손실한다. Batch normalization도 통계량을 버려 가역적이지 않다. Normalizing Flow를 위해서는 특수한 architecture가 필요하다.

대표적인 가역 변환들이 개발되었다. Coupling layers는 입력을 두 부분으로 나누고 한 부분을 다른 부분의 함수로 변환한다. Additive coupling은 다음과 같다.

$$\mathbf{x}_1 = \mathbf{z}_1$$
$$\mathbf{x}_2 = \mathbf{z}_2 + t(\mathbf{z}_1)$$

여기서 $t$는 임의의 신경망이다. 역변환은 단순히 빼기다.

$$\mathbf{z}_1 = \mathbf{x}_1$$
$$\mathbf{z}_2 = \mathbf{x}_2 - t(\mathbf{x}_1)$$

$\mathbf{z}_1$이 고정되므로 $t$가 가역적일 필요가 없다. 이는 architecture 자유도를 크게 높인다. Jacobian은 triangular 형태가 되어 determinant가 단순하다.

Affine coupling은 scaling을 추가한다.

$$\mathbf{x}_1 = \mathbf{z}_1$$
$$\mathbf{x}_2 = \mathbf{z}_2 \odot \exp(s(\mathbf{z}_1)) + t(\mathbf{z}_1)$$

여기서 $s$와 $t$는 신경망이고, $\odot$는 element-wise 곱셈이다. $\exp$는 항상 양수이므로 역변환이 가능하다.

$$\mathbf{z}_1 = \mathbf{x}_1$$
$$\mathbf{z}_2 = (\mathbf{x}_2 - t(\mathbf{x}_1)) \odot \exp(-s(\mathbf{x}_1))$$

Jacobian determinant는 $\prod_i \exp(s(\mathbf{z}_1)_i) = \exp(\sum_i s(\mathbf{z}_1)_i)$로 매우 효율적으로 계산된다.

Autoregressive flows는 각 차원을 이전 차원들의 함수로 변환한다.

$$x_i = z_i \cdot \exp(s_i(z_{1:i-1})) + t_i(z_{1:i-1})$$

이는 매우 표현력이 높지만, 순차적으로 계산해야 하므로 병렬화가 어렵다. Inference가 느려 이상 탐지에는 적합하지 않다.

Continuous normalizing flows는 ordinary differential equation (ODE)로 flow를 정의한다.

$$\frac{d\mathbf{z}(t)}{dt} = f_\theta(\mathbf{z}(t), t)$$

$t=0$에서 $t=1$까지 적분하여 변환을 구한다. 이는 매우 유연하지만 계산 비용이 크다. Neural ODE solver가 필요하고, 역전파가 복잡하다.

이상 탐지에서는 coupling layers가 가장 널리 사용된다. 효율성과 표현력의 균형이 좋다. 여러 coupling layers를 쌓고 각 층마다 partition을 바꿔 모든 차원이 변환되도록 한다. 예를 들어 홀수 층에서는 앞쪽 절반을 고정하고 뒤쪽을 변환하며, 짝수 층에서는 그 반대로 한다.

### 1.2.3 Log-Likelihood Computation

Normalizing Flow의 학습 목표는 데이터의 log-likelihood를 최대화하는 것이다. 주어진 정상 학습 데이터 $\{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$에 대해 다음을 최대화한다.

$$\max_\theta \frac{1}{N} \sum_{n=1}^{N} \log p_\theta(\mathbf{x}_n)$$

여기서 $\theta$는 flow의 파라미터다. 이는 maximum likelihood estimation (MLE)으로 well-founded된 통계적 추정 방법이다.

각 샘플의 log-likelihood는 앞서 유도한 공식으로 계산된다.

$$\log p_\theta(\mathbf{x}) = \log p_Z(f_\theta^{-1}(\mathbf{x})) - \log \left| \det \frac{\partial f_\theta}{\partial \mathbf{z}} \right|$$

Base distribution은 일반적으로 표준 가우시안이다.

$$p_Z(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I}) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{1}{2}\|\mathbf{z}\|^2\right)$$

로그를 취하면 다음과 같다.

$$\log p_Z(\mathbf{z}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\|\mathbf{z}\|^2$$

여러 coupling layers $f_k$가 있을 때, 전체 log-likelihood는 다음과 같다.

$$\log p_\theta(\mathbf{x}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\|\mathbf{z}_0\|^2 - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}} \right|$$

각 coupling layer의 Jacobian determinant를 효율적으로 계산하는 것이 핵심이다. Affine coupling의 경우 다음과 같다.

$$\log \left| \det \frac{\partial f}{\partial \mathbf{z}} \right| = \sum_{i \in \text{transformed}} s(\mathbf{z}_{\text{fixed}})_i$$

이는 단순히 scaling factors의 합이므로 $O(d)$ 복잡도다. Full Jacobian을 계산하는 $O(d^3)$보다 훨씬 효율적이다.

학습 중에는 mini-batch의 평균 negative log-likelihood를 loss로 사용한다.

$$\mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log p_\theta(\mathbf{x}_b)$$

여기서 $B$는 batch size다. 이를 gradient descent로 최소화한다. PyTorch나 TensorFlow의 autograd로 $\nabla_\theta \mathcal{L}$을 계산하고 Adam 같은 optimizer로 파라미터를 업데이트한다.

추론 시에는 학습된 flow로 테스트 샘플의 log-likelihood를 계산한다. 높은 log-likelihood는 정상, 낮은 log-likelihood는 이상을 의미한다. 이상 점수는 단순히 negative log-likelihood다.

$$s(\mathbf{x}) = -\log p_\theta(\mathbf{x})$$

임계값 $\tau$를 설정하여 $s(\mathbf{x}) > \tau$이면 이상으로 판정한다. 임계값은 validation set의 ROC 분석으로 선택된다.

Log-likelihood는 확률적으로 명확한 의미를 가진다. $\log p(\mathbf{x}) = -10$이면 확률이 $e^{-10} \approx 4.5 \times 10^{-5}$로 매우 희귀한 샘플이다. 이는 distance-based 이상 점수보다 해석이 직관적이다. 또한 여러 flow 모델이나 다른 확률 모델과의 비교가 가능하다.

## 1.3 Advantages of Probabilistic Modeling

Normalizing Flow의 확률적 모델링 접근은 여러 고유한 장점을 제공한다. 이들은 다른 이상 탐지 패러다임과 차별화되는 특성이다.

**Exact Likelihood Evaluation**

가장 중요한 장점은 정확한 likelihood를 계산할 수 있다는 것이다. VAE는 evidence lower bound (ELBO)만 제공하고 true likelihood는 intractable하다. GAN은 likelihood 개념 자체가 없다. Normalizing Flow는 exact log-likelihood를 제공하여 이론적으로 sound한 이상도 측정이 가능하다.

Exact likelihood는 모델 비교와 선택에 유용하다. 여러 flow architecture나 hyperparameter 설정을 비교할 때 validation log-likelihood로 직접 평가할 수 있다. 이는 cross-validation이나 별도 메트릭 없이 probabilistically principled한 선택을 가능하게 한다.

또한 outlier detection의 이론적 프레임워크와 연결된다. Likelihood ratio test, Neyman-Pearson lemma 등 통계적 검정 이론을 직접 적용할 수 있다. 예를 들어 특정 false positive rate를 보장하는 임계값을 이론적으로 유도할 수 있다.

**Stable Training**

GAN은 adversarial training의 불안정성으로 악명 높다. Mode collapse, vanishing gradient, oscillation 등의 문제가 빈번하다. Normalizing Flow는 simple maximum likelihood objective로 학습되어 매우 안정적이다. Gradient가 well-behaved하고 수렴이 reliable하다.

Training dynamics가 예측 가능하다. Loss curve가 단조 감소하며, overfitting도 조기에 감지할 수 있다. 여러 random seed로 학습해도 결과가 일관적이다. 이는 실무에서 재현성과 신뢰성을 높인다.

Hyperparameter에 대한 sensitivity도 낮다. Learning rate, batch size, optimizer 선택이 성능에 영향을 미치지만, GAN처럼 critical하지 않다. 합리적인 기본값으로 시작하면 대부분 잘 작동한다.

**Bidirectional Mapping**

Normalizing Flow는 data space와 latent space 간 양방향 매핑을 제공한다. Forward pass $\mathbf{x} \to \mathbf{z}$는 encoding으로, inverse pass $\mathbf{z} \to \mathbf{x}$는 generation이다. 두 방향 모두 exact하고 deterministic하다.

이는 다양한 응용을 가능하게 한다. Latent space에서 interpolation하면 의미 있는 데이터를 생성할 수 있다. 두 정상 샘플의 latent code를 보간하여 중간 샘플을 생성한다. 이는 data augmentation이나 what-if analysis에 유용하다.

Anomaly synthesis도 가능하다. Latent space에서 정상 분포 밖의 영역을 샘플링하여 synthetic anomaly를 생성한다. 이는 anomaly detection 모델의 robustness를 평가하거나 추가 학습에 사용될 수 있다. Reconstruction-based 방법(DRAEM)의 simulated anomaly와 유사하지만 확률적으로 제어 가능하다.

Latent space analysis도 흥미롭다. 정상 샘플들의 latent code를 시각화하면 intrinsic structure를 이해할 수 있다. PCA나 t-SNE로 저차원에 투영하여 cluster나 mode를 발견한다. 이는 정상 패턴의 다양성과 복잡성을 파악하는 데 도움이 된다.

**Density Estimation**

Normalizing Flow는 단순히 이상 탐지뿐만 아니라 density estimation 자체가 목표다. 정상 데이터의 전체 분포를 학습한다. 이는 sampling, generation, compression 등 다양한 downstream task에 활용될 수 있다.

특히 multimodal distribution을 잘 포착한다. 정상 패턴이 여러 mode를 가질 때(예: 여러 정상 변형, 다른 조립 구성) single Gaussian(PaDiM)으로는 부족하다. Normalizing Flow는 충분히 깊으면 임의의 multimodal distribution을 근사할 수 있다.

Conditional density estimation으로 확장도 가능하다. 조건부 변수(예: 제품 타입, 조명 조건)가 주어졌을 때 조건부 분포를 모델링한다. 이는 다양한 조건 하의 정상 패턴을 통합된 모델로 처리할 수 있게 한다.

**Uncertainty Quantification**

확률 모델은 자연스럽게 uncertainty를 정량화한다. Likelihood는 모델의 confidence를 나타낸다. 매우 낮은 likelihood는 모델이 해당 샘플에 대해 uncertain함을 의미한다. 이는 단순 binary 판정보다 풍부한 정보를 제공한다.

Out-of-distribution (OOD) detection에도 직접 적용된다. Training distribution과 완전히 다른 데이터(예: 다른 제품, 손상된 이미지)는 매우 낮은 likelihood를 받는다. 이는 이상 탐지와 OOD detection을 통합된 프레임워크에서 처리할 수 있게 한다.

Epistemic uncertainty와 aleatoric uncertainty를 구별할 수도 있다. Ensemble of flows로 epistemic uncertainty를, single flow의 likelihood spread로 aleatoric uncertainty를 추정한다. 이는 신뢰할 수 있는 예측과 추가 데이터 수집의 필요성을 판단하는 데 유용하다.

**Theoretical Elegance**

Normalizing Flow는 수학적으로 우아하다. Change of variables, differential geometry, measure theory의 well-established 개념에 기반한다. 이론적 분석과 증명이 가능하다. 예를 들어 universal approximation theorem의 flow 버전이 존재한다. 충분히 표현력 있는 flow는 임의의 smooth distribution을 근사할 수 있다.

이러한 이론적 기반은 새로운 architecture 설계에 guidance를 제공한다. Invertibility와 tractable determinant 제약 하에서 표현력을 최대화하는 방법을 체계적으로 탐구할 수 있다. 또한 실패 사례를 이론적으로 분석하여 원인을 파악하고 개선할 수 있다.

학술 연구에서도 Normalizing Flow는 활발한 주제다. 새로운 flow architecture, 효율화 기법, 응용 영역이 지속적으로 제안된다. Image generation, speech synthesis, molecular design 등 다양한 분야에서 성공적으로 적용되었다. 이러한 광범위한 연구는 이상 탐지에도 혜택을 준다.

**Limitations**

장점과 함께 한계도 명확하다. 가장 큰 문제는 계산 비용이다. 여러 coupling layers를 통과하고 Jacobian determinant를 계산하는 것은 expensive하다. Memory-based 방법보다 학습이 훨씬 느리고(수 시간 vs 수 분), 추론도 느리다(50-100ms vs 30-50ms).

Invertibility 제약은 architecture 설계를 제한한다. 일반적인 CNN layers(pooling, ReLU 등)를 직접 사용할 수 없다. 특수한 invertible layers가 필요하며, 이들은 일반 layers보다 표현력이 제한적일 수 있다. 최적의 flow architecture를 찾는 것은 여전히 art에 가깝다.

고차원 데이터에서 충분히 표현력 있는 flow를 학습하기 어렵다. Coupling layers는 절반씩만 변환하므로 많은 층이 필요하다. 층이 많으면 계산 비용과 메모리 사용량이 증가한다. 또한 gradient flow가 약해져 학습이 어려워질 수 있다.

그럼에도 Normalizing Flow는 이상 탐지의 중요한 패러다임이다. 확률적 엄밀함과 실용적 효과를 결합하여 memory-based와 knowledge distillation 사이의 틈새를 채운다. 특히 성능과 속도의 균형이 필요하고, 확률적 해석이 중요한 응용에서 탁월하다. FastFlow의 성공은 적절한 simplification으로 효율성을 크게 향상시킬 수 있음을 보여주었다. 이는 Normalizing Flow의 미래가 밝음을 시사한다.

# 2. CFlow (2021)

## 2.1 Basic Information

CFlow(Conditional Normalizing Flow)는 2021년 Gudovskiy 등이 제안한 이상 탐지 방법으로, Normalizing Flow를 이미지의 공간적 구조에 맞게 조건부로 확장했다. 이 연구는 WACV 2022에서 발표되었으며, flow-based 방법이 산업 이상 탐지에서 competitive함을 최초로 입증했다.

CFlow의 핵심 아이디어는 위치 조건부 분포 모델링이다. 이미지의 서로 다른 위치는 서로 다른 정상 패턴을 가질 수 있다. 예를 들어 제품 이미지에서 중앙 영역과 가장자리 영역은 다른 시각적 특성을 보인다. 단일 전역 분포로는 이러한 공간적 다양성을 충분히 포착하지 못한다. CFlow는 각 공간 위치마다 조건부 분포를 학습하여 이 문제를 해결한다.

방법론적으로 CFlow는 사전 학습된 CNN(예: Wide ResNet-50)에서 추출된 특징 맵에 조건부 normalizing flow를 적용한다. 각 공간 위치 $(i,j)$의 특징 벡터에 대해 해당 위치를 조건으로 하는 분포를 모델링한다. 이는 PaDiM의 위치별 가우시안 모델링과 유사한 전략이지만, 단일 가우시안 대신 표현력 있는 flow를 사용한다.

CFlow는 multi-scale feature fusion도 도입했다. CNN의 여러 층(layer2, layer3, layer4)에서 특징을 추출하고 각각에 독립적인 flow를 학습한다. 최종 이상 점수는 여러 스케일의 점수를 결합하여 계산한다. 이는 서로 다른 크기의 결함을 효과적으로 탐지하게 한다.

성능 면에서 CFlow는 MVTec AD 벤치마크에서 이미지 레벨 AUROC 98.3%, 픽셀 레벨 98.5%를 달성했다. 이는 발표 당시 flow-based 방법으로서는 최고 성능이었고, PaDiM(97.5%)을 능가했다. Texture 카테고리에서 특히 강력했으며, 여러 카테고리에서 99%를 초과했다. 이는 Normalizing Flow가 이상 탐지에서 viable한 접근임을 증명했다.

그러나 CFlow의 주된 한계는 계산 비용이다. 추론 시간이 이미지당 약 100-150ms로 PaDiM(50-100ms)이나 PatchCore(30-50ms)보다 느리다. 학습 시간도 수 시간에 달해 memory-based 방법의 수 분과 대조적이다. 메모리 사용량도 상당하여(수백 MB에서 1GB+) 엣지 배포가 어렵다. 이러한 효율성 문제는 후속 연구인 FastFlow에서 개선되었다.

CFlow는 flow-based 이상 탐지의 선구자로서 중요한 역할을 했다. 조건부 flow의 개념, multi-scale 처리, coupling layer 설계 등은 후속 연구들의 기반이 되었다. 비록 실무 배포에서는 FastFlow에 자리를 내주었지만, 학술적 기여와 아이디어는 여전히 가치가 있다.

## 2.2 Conditional Normalizing Flow

### 2.2.1 Position-Conditional Architecture

CFlow의 가장 중요한 혁신은 위치 조건부 normalizing flow다. 일반적인 flow는 전체 이미지나 특징을 하나의 분포로 모델링한다. 그러나 이미지는 공간적으로 heterogeneous하다. 서로 다른 위치는 서로 다른 semantic content와 시각적 패턴을 가진다.

CFlow는 이를 조건부 확률 모델링으로 해결한다. 위치 $(i,j)$의 특징 벡터 $\mathbf{f}_{ij}$에 대해, 해당 위치를 조건으로 하는 분포를 학습한다.

$$p(\mathbf{f}_{ij} \mid i, j)$$

이는 각 위치마다 다른 정상 분포를 허용한다. 중앙 영역의 정상 패턴과 가장자리 영역의 정상 패턴이 독립적으로 모델링된다. 이는 PaDiM의 위치별 가우시안과 유사한 동기지만, 더 표현력 있는 모델이다.

조건부 flow를 구현하는 방법은 coupling layers의 conditioning이다. Affine coupling layer에서 scale과 translation 함수 $s$와 $t$를 위치에 의존하게 만든다.

$$\mathbf{x}_2 = \mathbf{z}_2 \odot \exp(s(\mathbf{z}_1; i, j)) + t(\mathbf{z}_1; i, j)$$

여기서 $s(\cdot; i, j)$와 $t(\cdot; i, j)$는 위치 $(i,j)$를 입력으로 받는 신경망이다. 실제로는 위치 정보를 embedding하여 신경망에 concatenate한다.

Position embedding은 learnable parameter로 구현된다. 각 위치 $(i,j)$에 대해 embedding 벡터 $\mathbf{e}_{ij} \in \mathbb{R}^{d_e}$를 할당한다. 이는 $h \times w \times d_e$ 크기의 position embedding map이다. 특징 벡터 $\mathbf{z}_1$과 embedding $\mathbf{e}_{ij}$를 concatenate하여 conditioning network에 입력한다.

$$s(\mathbf{z}_1; i, j) = \text{NN}_s([\mathbf{z}_1; \mathbf{e}_{ij}])$$
$$t(\mathbf{z}_1; i, j) = \text{NN}_t([\mathbf{z}_1; \mathbf{e}_{ij}])$$

이 접근의 장점은 유연성이다. Embedding이 학습되면서 공간적 구조를 자동으로 포착한다. 인접 위치는 유사한 embedding을 학습하여 smoothness가 자연스럽게 나타난다. 멀리 떨어진 위치는 다른 embedding을 가져 독립적인 분포를 모델링할 수 있다.

또한 위치 정보를 명시적으로 제공하여 모델이 공간적 맥락을 이해하게 한다. 이는 결함이 특정 위치에 나타나는 경향이 있을 때 유용하다. 예를 들어 제품의 특정 부분이 더 결함에 취약하면, 해당 위치의 flow가 더 민감하게 조정될 수 있다.

메모리 측면에서 position embedding은 $h \times w \times d_e$ 파라미터를 추가한다. 전형적으로 $h=w=56$, $d_e=64$이면 약 200K 파라미터다. Coupling networks의 파라미터(수백만)에 비해 작지만 무시할 수 없다. 여러 flow layers가 각각 embedding을 가지면 누적된다.

대안적으로 sinusoidal position encoding을 사용할 수도 있다. Transformer에서 사용되는 방식으로, 학습 없이 위치를 고정된 벡터로 인코딩한다.

$$\text{PE}(i,j,2k) = \sin(i / 10000^{2k/d_e})$$
$$\text{PE}(i,j,2k+1) = \cos(j / 10000^{2k/d_e})$$

이는 파라미터를 절약하지만 유연성이 떨어진다. CFlow는 learnable embedding을 사용하여 데이터로부터 최적의 위치 표현을 학습한다.

### 2.2.2 Multi-scale Processing

CFlow는 CNN의 여러 층에서 특징을 추출하고 각각에 독립적인 flow를 학습한다. 이는 서로 다른 크기와 유형의 결함을 효과적으로 탐지하기 위함이다. Layer2는 low-level texture와 작은 결함을, layer3는 mid-level pattern을, layer4는 high-level structure와 큰 결함을 포착한다.

구체적으로 Wide ResNet-50의 layer2, layer3, layer4에서 특징 맵을 추출한다. 각 층의 출력 크기는 다음과 같다.
- Layer2: $h_2 \times w_2 \times d_2$ (예: $64 \times 64 \times 512$)
- Layer3: $h_3 \times w_3 \times d_3$ (예: $32 \times 32 \times 1024$)
- Layer4: $h_4 \times w_4 \times d_4$ (예: $16 \times 16 \times 2048$)

각 층에 대해 별도의 conditional normalizing flow $f^{(l)}_\theta$를 학습한다. 세 개의 flow가 독립적으로 훈련되며 파라미터를 공유하지 않는다. 각 flow는 해당 층 특징의 log-likelihood를 최대화한다.

$$\max_{\theta^{(l)}} \frac{1}{N} \sum_{n=1}^{N} \sum_{i,j} \log p^{(l)}_{\theta^{(l)}}(\mathbf{f}^{(l)}_{ij,n} \mid i, j)$$

여기서 $\mathbf{f}^{(l)}_{ij,n}$은 $n$번째 이미지의 층 $l$, 위치 $(i,j)$의 특징 벡터다. 모든 위치와 샘플에 대한 log-likelihood를 합산하여 최대화한다.

추론 시 각 flow는 해당 층 특징의 위치별 이상 점수를 생성한다. 층 $l$의 위치 $(i,j)$에서 이상 점수는 negative log-likelihood다.

$$s^{(l)}_{ij} = -\log p^{(l)}_{\theta^{(l)}}(\mathbf{f}^{(l)}_{ij} \mid i, j)$$

이상 맵들의 해상도가 다르므로 통일해야 한다. 일반적으로 가장 높은 해상도(layer2)로 업샘플링한다. Bilinear interpolation을 사용하여 layer3와 layer4의 이상 맵을 $h_2 \times w_2$로 확대한다.

$$\mathbf{S}^{(l)}_{\text{up}} = \text{Upsample}(\mathbf{S}^{(l)}, (h_2, w_2))$$

세 개의 업샘플링된 이상 맵을 결합한다. 가장 단순한 방법은 평균이다.

$$\mathbf{S}_{\text{final}} = \frac{1}{3}(\mathbf{S}^{(2)}_{\text{up}} + \mathbf{S}^{(3)}_{\text{up}} + \mathbf{S}^{(4)}_{\text{up}})$$

가중 평균도 가능하다. 각 층에 가중치 $w_l$을 할당하여 조정한다.

$$\mathbf{S}_{\text{final}} = \sum_{l} w_l \mathbf{S}^{(l)}_{\text{up}}$$

가중치는 validation set에서 튜닝하거나 learnable parameter로 만들 수 있다. 실험적으로 동일 가중치가 대부분의 경우 충분했다.

이미지 레벨 이상 점수는 최종 이상 맵의 최댓값으로 계산된다.

$$S_{\text{image}} = \max_{i,j} \mathbf{S}_{\text{final}}(i,j)$$

이는 이미지 내 가장 이상한 위치가 전체 이미지의 이상도를 결정한다는 가정이다.

Multi-scale 접근의 장점은 다양한 결함을 포괄적으로 탐지하는 것이다. 작은 scratch나 점 결함은 layer2에서, 큰 crack이나 변형은 layer4에서 잘 탐지된다. 단일 스케일만 사용하면 특정 크기의 결함을 놓칠 수 있다.

단점은 계산 비용이 세 배로 증가한다는 것이다. 세 개의 독립적인 flow를 학습하고 추론해야 한다. 메모리도 세 배 필요하다. 이는 CFlow의 효율성 문제를 악화시킨다. FastFlow는 이를 개선하기 위해 layer3 하나만 사용하는 전략을 택했다.

### 2.2.3 Coupling Layers

CFlow는 affine coupling layers를 기본 building block으로 사용한다. 이는 RealNVP와 Glow에서 확립된 architecture를 따른다. 각 coupling layer는 입력을 두 부분으로 나누고 한 부분을 다른 부분의 함수로 변환한다.

입력 특징 $\mathbf{z} \in \mathbb{R}^d$를 $\mathbf{z}_1 \in \mathbb{R}^{d/2}$와 $\mathbf{z}_2 \in \mathbb{R}^{d/2}$로 split한다. 일반적으로 채널 차원을 따라 절반씩 나눈다. Affine coupling은 다음과 같이 정의된다.

$$\mathbf{x}_1 = \mathbf{z}_1$$
$$\mathbf{x}_2 = \mathbf{z}_2 \odot \exp(s(\mathbf{z}_1; i, j)) + t(\mathbf{z}_1; i, j)$$

여기서 $s$와 $t$는 conditioning networks다. $\mathbf{z}_1$과 position embedding을 입력받아 $d/2$ 차원의 출력을 생성한다.

Conditioning network는 일반적으로 작은 fully connected network다. CFlow에서는 3-4개의 hidden layers를 사용한다. 각 layer는 ReLU activation과 optional batch normalization을 가진다.

```
s(z1, e_ij) = Linear(ReLU(Linear(ReLU(Linear([z1; e_ij])))))
```

Hidden dimension은 입력 차원과 비슷하거나 약간 크게 설정한다(예: $d/2$ 또는 $d$). 이는 충분한 표현력을 제공하면서도 파라미터 수를 관리 가능하게 한다.

역변환은 단순하다. Scale과 translation을 역으로 적용한다.

$$\mathbf{z}_1 = \mathbf{x}_1$$
$$\mathbf{z}_2 = (\mathbf{x}_2 - t(\mathbf{x}_1; i, j)) \odot \exp(-s(\mathbf{x}_1; i, j))$$

Jacobian determinant는 매우 효율적으로 계산된다. Triangular Jacobian 구조 덕분에 determinant는 단순히 $\mathbf{x}_2$의 scale factors의 곱이다.

$$\log \left| \det \frac{\partial f}{\partial \mathbf{z}} \right| = \sum_{i=1}^{d/2} s(\mathbf{z}_1; i, j)_i$$

이는 $O(d)$ 복잡도로 full Jacobian 계산 $O(d^3)$보다 훨씬 빠르다.

CFlow는 여러 coupling layers를 쌓는다. 일반적으로 8-12 layers를 사용한다. 각 layer마다 partition을 바꿔 모든 차원이 변환되도록 한다. 짝수 layer에서는 앞쪽 절반을 고정하고 뒤쪽을 변환하며, 홀수 layer에서는 그 반대다.

```
Layer 1: Fix [0:d/2], Transform [d/2:d]
Layer 2: Fix [d/2:d], Transform [0:d/2]
Layer 3: Fix [0:d/2], Transform [d/2:d]
...
```

이렇게 하면 $K$ layers 후에 모든 차원이 최소 $K/2$번 변환된다. 충분한 표현력을 보장한다.

Permutation layers를 추가하여 mixing을 향상시킬 수도 있다. 각 coupling layer 후에 채널을 shuffle하거나 reverse한다. 이는 고정된 partition pattern을 깨뜨려 모든 차원 간 상호작용을 촉진한다.

$$\mathbf{z}' = \text{Permute}(\mathbf{x})$$

Permutation은 Jacobian determinant에 영향을 주지 않는다(determinant가 $\pm 1$). 따라서 계산 비용 없이 표현력을 높인다.

Activation normalization(ActNorm)도 일반적으로 사용된다. Batch normalization의 invertible 버전으로, 각 채널을 affine transformation한다.

$$\mathbf{x} = \mathbf{s} \odot \mathbf{z} + \mathbf{b}$$

여기서 $\mathbf{s}$와 $\mathbf{b}$는 learnable parameter다. 초기화 시 첫 번째 mini-batch로 데이터를 정규화하도록 설정한다($\mathbf{s}$를 표준편차의 역수, $\mathbf{b}$를 음의 평균으로). 이후 gradient descent로 최적화된다.

ActNorm의 Jacobian determinant는 $\sum_i \log |s_i|$로 간단하다. 이는 학습 안정성을 높이고 초기 convergence를 가속한다.

전체 flow는 다음과 같은 구조를 가진다.

```
for k in 1 to K:
    ActNorm
    Coupling Layer (conditional on position)
    Permutation
```

Base distribution은 표준 가우시안 $\mathcal{N}(\mathbf{0}, \mathbf{I})$다. 최종 log-likelihood는 다음과 같이 계산된다.

$$\log p(\mathbf{f}_{ij} \mid i,j) = \log \mathcal{N}(\mathbf{z}_0; \mathbf{0}, \mathbf{I}) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}} \right|$$

## 2.3 Technical Details

CFlow의 구현 세부사항은 성능과 효율성에 중요한 영향을 미친다.

**Backbone and Feature Extraction**

CFlow는 Wide ResNet-50을 기본 백본으로 사용한다. ImageNet에서 사전 학습된 가중치를 로드하고, inference mode로 고정한다. Gradient가 백본으로 전파되지 않으므로 메모리와 계산이 절약된다.

Layer2, layer3, layer4의 출력을 hook으로 캡처한다. PyTorch에서는 `register_forward_hook`을 사용한다. 각 층의 activation을 저장하여 flow에 입력한다.

특징 차원이 매우 크므로(512, 1024, 2048) 차원 축소를 고려할 수 있다. $1 \times 1$ convolution으로 채널을 줄이면 flow의 입력 차원이 감소한다. 예를 들어 layer4의 2048 채널을 512로 줄인다. 이는 계산 비용과 메모리를 크게 절감하지만, 정보 손실이 발생할 수 있다. 실험적으로 적절한 축소 비율을 찾아야 한다.

**Training Strategy**

CFlow는 maximum likelihood estimation으로 학습된다. Loss function은 negative log-likelihood의 평균이다.

$$\mathcal{L} = -\frac{1}{N \cdot h \cdot w} \sum_{n=1}^{N} \sum_{i,j} \log p_\theta(\mathbf{f}_{ij,n} \mid i, j)$$

모든 위치의 log-likelihood를 합산하므로 loss가 매우 크다. 이를 정규화하기 위해 위치 수($h \times w$)로 나눈다. 이는 학습 안정성을 높인다.

Adam optimizer를 learning rate $10^{-4}$로 사용한다. Batch size는 GPU 메모리에 따라 8-32다. 큰 특징 맵이 메모리를 많이 사용하므로 배치 크기가 제한된다. Gradient accumulation으로 effective batch size를 늘릴 수 있다.

학습은 일반적으로 200-500 epochs 진행된다. 각 epoch는 수백 장의 이미지를 포함하므로 총 수만 번의 iteration이다. 학습 시간은 GPU에 따라 수 시간에서 하루 정도다. 이는 memory-based 방법(수 분)보다 훨씬 길다.

Early stopping을 사용하여 overfitting을 방지한다. Validation set의 log-likelihood를 모니터링하고, 개선이 멈추면 학습을 종료한다. Patience는 일반적으로 20-50 epochs다.

Learning rate scheduling도 유용하다. Cosine annealing이나 step decay로 학습 후반부에 learning rate를 줄인다. 이는 fine-tuning 효과를 제공하여 최종 성능을 향상시킨다.

**Inference Pipeline**

추론 시 입력 이미지를 전처리하고 백본에 통과시켜 특징을 추출한다. 각 층의 특징 맵에 대해 해당 flow로 log-likelihood를 계산한다.

각 위치 $(i,j)$에서 특징 벡터 $\mathbf{f}_{ij}$를 추출하고 flow를 통과시킨다. Forward pass로 latent code $\mathbf{z}_0$를 얻고, 각 coupling layer의 Jacobian determinant를 누적한다. 최종 log-likelihood는 base distribution의 log-likelihood에서 determinants의 합을 뺀 것이다.

세 개 층의 이상 맵을 동일 해상도로 업샘플링하고 평균을 취한다. 이미지 레벨 점수는 이상 맵의 최댓값이다.

Batch inference로 throughput을 높일 수 있다. 여러 이미지를 동시에 처리하여 GPU 활용률을 극대화한다. 그러나 단일 이미지 latency는 여전히 100-150ms다.

**Memory Optimization**

CFlow의 메모리 사용량은 주된 병목이다. 세 개의 flow가 각각 수백만 파라미터를 가지고, 중간 activations도 상당한 메모리를 소비한다.

Gradient checkpointing으로 메모리를 절약할 수 있다. Forward pass의 일부 activations만 저장하고 backward pass에서 재계산한다. 이는 메모리를 50% 정도 줄이지만 계산 시간이 약간 증가한다.

Mixed precision training(FP16)도 효과적이다. 절반 정밀도로 계산하여 메모리를 절반으로 줄이고 속도도 향상시킨다. Loss scaling으로 수치 안정성을 유지한다.

추론 시에는 모델을 FP16이나 INT8로 quantization할 수 있다. 정확도 저하가 minimal하면서 메모리와 속도가 개선된다.

**Hyperparameter Tuning**

CFlow의 주요 hyperparameters는 다음과 같다.
- Number of coupling layers: 8-12 (더 많으면 표현력 증가, 계산 증가)
- Hidden dimension: 입력 차원의 1-2배
- Learning rate: $10^{-4}$ - $10^{-3}$
- Batch size: 8-32 (GPU 메모리에 의존)
- Position embedding dimension: 64-128

일반적으로 기본 설정(12 layers, hidden dim = input dim, lr = $10^{-4}$)이 robust하다. 성능이 부족하면 layers를 늘리고, 속도가 문제면 줄인다.

카테고리별로 optimal hyperparameters가 다를 수 있지만, 모든 카테고리에 공통 설정을 사용하는 것이 실무적이다. Extensive tuning의 이득이 크지 않다.

## 2.4 Performance Analysis

CFlow는 MVTec AD 벤치마크에서 인상적인 성능을 보였다. 이미지 레벨 AUROC 98.3%로 발표 당시 flow-based 방법 중 최고였다. 픽셀 레벨에서도 98.5%로 우수했다. 이는 PaDiM(97.5%)을 능가하고 PatchCore(99.1%)에 근접했다.

**Category-wise Performance**

Texture 카테고리에서 CFlow는 특히 강력했다. Carpet(99.2%), grid(99.8%), leather(100%), tile(99.3%), wood(99.5%)에서 거의 완벽한 분류를 달성했다. 반복 패턴과 명확한 texture를 가진 카테고리에서 flow의 분포 모델링이 효과적이었다.

Object 카테고리에서는 약간 낮지만 여전히 competitive했다. Bottle(99.8%), cable(99.0%), capsule(97.1%), hazelnut(98.1%)에서 좋은 성능을 보였다. 그러나 screw(95.7%)와 같이 복잡한 3D 구조를 가진 객체에서는 상대적으로 어려움을 겪었다.

가장 어려운 카테고리는 metal_nut(96.8%)과 pill(96.4%)이었다. 이들은 미세한 결함과 다양한 정상 변동을 가져 어떤 방법도 어려워한다. CFlow도 예외가 아니었다.

**Multi-scale Contribution**

Multi-scale processing의 효과를 분석한 결과, 세 층 모두 유의미하게 기여했다. Layer2만 사용하면 평균 AUROC가 96.5%, layer3만은 97.2%, layer4만은 96.8%였다. 세 층을 결합하면 98.3%로 향상되었다. 이는 1-2%포인트의 개선으로, multi-scale의 가치를 입증한다.

Layer3가 가장 informative했다. Mid-level features가 대부분의 결함을 잘 포착한다. Layer2는 small texture anomalies에, layer4는 large structural anomalies에 추가 기여를 했다.

**Conditional vs Non-conditional**

Position-conditional flow와 non-conditional flow를 비교했다. Non-conditional은 모든 위치에서 동일한 분포를 사용한다. 결과는 conditional이 평균 2%포인트 우위였다(98.3% vs 96.3%). 이는 공간적 heterogeneity를 모델링하는 것의 중요성을 보여준다.

특히 object 카테고리에서 차이가 컸다. 제품의 서로 다른 부분이 다른 정상 패턴을 가질 때 conditional modeling이 essential하다. Texture 카테고리에서는 차이가 적었다. 반복 패턴은 공간적으로 uniform하기 때문이다.

**Comparison with Other Methods**

CFlow는 PaDiM보다 일관되게 우수했다. 평균 AUROC가 0.8%포인트 높았고, 여러 어려운 카테고리에서 더 큰 격차를 보였다. 이는 flow의 표현력이 단일 가우시안보다 뛰어남을 입증한다.

그러나 PatchCore(99.1%)에는 미치지 못했다. 0.8%포인트 차이는 크지 않지만 일관적이었다. PatchCore의 coreset selection과 locally aware features가 flow보다 더 효과적인 것으로 보인다.

STFPM(knowledge distillation, 96.8%)보다는 명확히 우수했다. 1.5%포인트 차이는 유의미하다. 이는 probabilistic modeling이 feature matching보다 robust함을 시사한다.

**Pixel-level Localization**

픽셀 레벨 세그멘테이션에서 CFlow는 98.5% AUROC를 달성했다. 이는 이미지 레벨(98.3%)보다 약간 높다. 결함 위치를 정확히 특정할 수 있음을 의미한다.

이상 맵의 시각적 품질도 우수했다. 결함 영역이 명확히 강조되고 정상 영역은 낮은 점수를 받았다. 경계가 비교적 sharp하여 실무에서 유용하다. 다만 매우 작은 결함의 경우 약간 blur되는 경향이 있었다.

False positive는 주로 정상 변동이 큰 영역에서 발생했다. 예를 들어 조명 변화나 반사가 있는 부분이 오탐될 수 있었다. 이는 학습 데이터에 충분한 변동이 포함되지 않았기 때문이다.

## 2.5 Limitations (Speed, Memory)

CFlow의 주된 한계는 계산 효율성이다. 성능은 우수하지만 실무 배포에서는 속도와 메모리가 병목이 된다.

**Inference Speed**

CFlow의 추론 시간은 이미지당 약 100-150ms(GPU 기준)다. 이는 PatchCore(30-50ms)의 3-5배, PaDiM(50-100ms)의 2-3배다. 실시간 처리(초당 30 프레임, 33ms/이미지)는 불가능하다.

속도 병목의 주된 원인은 여러 coupling layers를 통과하는 것이다. 각 layer마다 conditioning network가 forward pass를 수행해야 한다. 12 layers × 3 scales = 36번의 network evaluation이 필요하다.

또한 각 공간 위치에서 독립적으로 flow를 실행한다. $h \times w$개 위치에서 각각 flow를 평가하므로 병렬화가 제한적이다. Batch processing으로 어느 정도 완화되지만 근본적인 문제는 남는다.

CPU에서는 더욱 느려진다. 약 1-2초가 소요되어 실용적이지 않다. 엣지 디바이스(Jetson Nano 등)에서도 수백 ms에서 초 단위로 느려진다.

Throughput은 배치 처리로 향상될 수 있다. 배치 크기 32로 처리하면 초당 5-10 이미지를 달성한다. 그러나 단일 이미지 latency는 여전히 길다. 실시간 응용이나 interactive system에는 부적합하다.

**Memory Usage**

CFlow는 카테고리당 약 500MB-1GB의 메모리를 사용한다. 세 개의 flow가 각각 수백 MB를 차지한다. 각 flow는 수백만 파라미터를 가진다.

구체적으로 layer4 flow(입력 2048 차원)가 가장 크다. 12 coupling layers × (conditioning networks + ActNorm + position embedding)로 약 400-500MB다. Layer2와 layer3 flow는 각각 200-300MB다. 총 800MB-1.2GB가 필요하다.

다중 카테고리 배포 시 메모리가 선형적으로 증가한다. 10개 카테고리면 8-12GB, 100개면 80-120GB다. 단일 GPU(10-20GB VRAM)에서는 수십 개가 한계다. 서버급 메모리(128-256GB)가 필요하거나, 모델을 동적으로 load/unload해야 한다.

엣지 배포는 거의 불가능하다. Raspberry Pi(4GB)나 Jetson Nano(4GB)에서는 단일 카테고리도 어렵다. 메모리 부족으로 crash되거나 swap으로 인해 극도로 느려진다.

추론 시 중간 activations도 메모리를 소비한다. Forward pass에서 각 layer의 출력을 저장해야 determinant 계산에 사용한다. 이는 입력 크기에 비례하여 수십-수백 MB 추가다.

**Training Time**

학습 시간도 상당하다. 단일 카테고리를 학습하는 데 GPU에서 약 3-8시간이 걸린다. 이는 PatchCore(3-5분)의 수십 배, PaDiM(2-3분)의 백 배 이상이다.

학습이 느린 이유는 수백 epochs의 gradient descent가 필요하기 때문이다. 각 epoch는 수백 장을 처리하고, 각 이미지마다 모든 위치에서 backpropagation을 수행한다. 세 개 flow를 독립적으로 학습하면 시간이 더 증가한다.

다중 카테고리를 학습하려면 일(day) 단위 시간이 필요하다. 15개 MVTec 카테고리를 모두 학습하면 2-5일이다. 이는 rapid prototyping이나 frequent retraining에 장애물이다.

분산 학습으로 가속할 수 있다. 여러 GPU에서 카테고리를 병렬로 학습하면 총 시간을 줄인다. 그러나 단일 카테고리의 학습 시간은 변하지 않는다.

**Scalability Issues**

고해상도 이미지에서 문제가 악화된다. $512 \times 512$ 입력이면 feature map 크기가 4배 증가한다. 메모리도 4배, 계산도 4배 필요하다. $1024 \times 1024$이면 16배다.

타일링으로 대응할 수 있지만 복잡도가 증가한다. 큰 이미지를 여러 $256 \times 256$ 타일로 나누고 각각 처리한다. 타일 간 경계를 merge하는 추가 로직이 필요하다.

카테고리 수가 증가하면 관리가 어려워진다. 수백 개 카테고리에서 각각 GB급 모델을 유지하는 것은 impractical하다. Model zoo 관리, versioning, deployment가 복잡해진다.

**Comparison with Alternatives**

CFlow의 효율성 한계는 다른 방법과 비교하면 명확하다. PatchCore는 1-2MB 모델로 30-50ms 추론을 달성한다. CFlow는 1GB 모델로 100-150ms다. 500배 큰 모델로 3배 느린 것이다. 이는 효율성 측면에서 competitive하지 않다.

FastFlow(후속 연구)는 이 문제를 인식하고 개선했다. 2D flow로 단순화하여 속도를 3배 향상시키고(30-50ms) 메모리를 절반으로 줄였다(200-500MB). 성능은 거의 유지되었다(98.5%). 이는 simplification의 힘을 보여준다.

Knowledge distillation 방법(EfficientAD)은 더 극단적이다. 1-5ms 추론과 50MB 이하 모델을 달성했다. 성능이 약간 낮지만(97.8%) 실시간 처리가 가능하다. 응용에 따라 이러한 trade-off가 더 적합할 수 있다.

**Mitigation Strategies**

CFlow의 한계를 완화하는 전략들이 있다. 첫째, single-scale 사용이다. Layer3만 사용하면 메모리와 시간이 1/3로 줄어든다. 성능 저하는 1%포인트 정도로 acceptable할 수 있다.

둘째, 더 적은 coupling layers를 사용한다. 12 layers 대신 6-8 layers로 줄이면 속도가 30-50% 향상된다. 표현력은 감소하지만 여전히 competitive한 성능을 유지한다.

셋째, 특징 차원 축소다. $1 \times 1$ convolution으로 채널을 절반으로 줄이면 flow의 입력 크기가 감소한다. 계산과 메모리가 크게 절약되지만 정보 손실을 감수해야 한다.

넷째, distillation이다. 학습된 CFlow를 teacher로 사용하여 작은 student model을 학습한다. Student는 빠르고 가벼우며, teacher의 성능을 대부분 유지한다. 이는 knowledge distillation 패러다임과 연결된다.

다섯째, 하드웨어 최적화다. TensorRT, ONNX Runtime 같은 inference engine으로 모델을 최적화한다. Kernel fusion, quantization으로 속도를 2-3배 향상시킬 수 있다.

그럼에도 CFlow는 근본적으로 복잡한 모델이다. Normalizing flow의 표현력은 계산 비용과 trade-off 관계에 있다. 후속 연구들은 이 균형을 개선하는 데 집중했고, FastFlow가 가장 성공적이었다.

# 3. FastFlow (2021)

## 3.1 Basic Information

FastFlow는 2021년 Yu 등이 제안한 방법으로, Normalizing Flow의 효율성을 극적으로 개선하여 이상 탐지에서 실용성을 확보했다. 이 연구는 arXiv에 발표되었으며, 간단하면서도 효과적인 아이디어로 flow-based 방법의 게임 체인저가 되었다. FastFlow의 성공은 "less is more" 원칙의 완벽한 실증이다.

FastFlow의 핵심 혁신은 3D normalizing flow를 2D로 단순화한 것이다. CFlow는 공간 위치 $(i,j)$마다 독립적인 flow를 실행하여 $(h, w, c)$ 형태의 특징 맵 전체를 처리한다. 이는 $h \times w$번의 flow evaluation이 필요하다. FastFlow는 채널 간 독립성을 가정하여 각 채널을 독립적으로 처리한다. 이는 단 $c$번의 flow evaluation만 필요하다. 일반적으로 $c \ll h \times w$이므로(예: $c=256$, $h \times w=3136$) 계산량이 10배 이상 감소한다.

이러한 단순화에도 불구하고 성능은 거의 유지되었다. FastFlow는 MVTec AD에서 이미지 레벨 AUROC 98.5%를 달성하여 CFlow(98.3%)를 오히려 약간 상회했다. 추론 속도는 20-50ms로 CFlow(100-150ms)의 3-5배 빠르다. 메모리 사용량도 200-500MB로 CFlow(500MB-1GB)의 절반 이하다. 이는 성능-속도-메모리의 최적 균형점을 제공한다.

FastFlow는 다양한 백본을 지원한다. ResNet, EfficientNet, ConvNeXt 등과 호환되며, 각 백본의 특성에 맞춰 최적화할 수 있다. ResNet50이 기본이지만, 더 빠른 추론을 위해 ResNet18이나 MobileNet을 사용할 수도 있다. 더 높은 정확도를 위해서는 Wide ResNet-50이나 EfficientNet-B4를 선택한다.

실무 배포 관점에서 FastFlow는 매우 매력적이다. 합리적인 메모리로 다중 카테고리를 단일 GPU에 배포할 수 있다. 추론 속도는 실시간에 가까워 대부분의 검사 라인 속도를 충족한다. 학습 시간도 CFlow보다 빠르다(1-3시간). 이러한 실용성 덕분에 FastFlow는 flow-based 방법 중 가장 널리 사용된다.

FastFlow의 영향은 학술적으로도 중요하다. 복잡한 모델이 항상 더 나은 것은 아니라는 교훈을 제공했다. 적절한 단순화가 오히려 일반화를 개선하고 효율성을 획기적으로 높일 수 있다. 이는 딥러닝의 일반적인 "bigger is better" 트렌드에 대한 반례로, Occam's razor의 실증이다. 후속 연구들은 FastFlow의 아이디어를 다른 도메인과 task에 적용하고 있다.

## 3.2 2D Normalizing Flow Innovation

### 3.2.1 3D → 2D Simplification

FastFlow의 핵심 아이디어는 특징 맵의 공간 차원과 채널 차원을 다르게 처리하는 것이다. CFlow는 특징 맵 $\mathbf{F} \in \mathbb{R}^{h \times w \times c}$를 $h \times w$개의 $c$차원 벡터로 보고, 각 벡터에 조건부 flow를 적용한다. 이는 공간적으로 다른 분포를 모델링하지만 계산 비용이 크다.

FastFlow는 반대 접근을 취한다. 특징 맵을 $c$개의 $h \times w$ 크기 2D 텐서(채널)로 본다. 각 채널을 독립적으로 처리하며, 2D normalizing flow를 적용한다. 즉, $c$차원 벡터 공간에서의 flow가 아니라, $h \times w$ 크기 이미지 공간에서의 flow다.

수학적으로 CFlow는 다음을 모델링한다.

$$p(\mathbf{F}) = \prod_{i,j} p(\mathbf{f}_{ij} \mid i, j)$$

여기서 $\mathbf{f}_{ij} \in \mathbb{R}^c$는 위치 $(i,j)$의 특징 벡터다. 각 위치마다 $c$차원 공간에서의 분포를 학습한다.

FastFlow는 다음을 모델링한다.

$$p(\mathbf{F}) = \prod_{k=1}^{c} p(\mathbf{F}_{:,:,k})$$

여기서 $\mathbf{F}_{:,:,k} \in \mathbb{R}^{h \times w}$는 $k$번째 채널의 2D activation map이다. 각 채널마다 독립적인 2D 분포를 학습한다.

이 전환은 계산 복잡도를 극적으로 변화시킨다. CFlow는 $O(h \times w \times c \times K)$의 flow evaluations가 필요하다($K$는 coupling layers 수). FastFlow는 $O(c \times h \times w)$만 필요하다. Flow의 깊이 $K$가 없어졌다. 각 채널이 단일 2D operation으로 처리되기 때문이다.

구체적 예시로 $h=w=56$, $c=256$, $K=12$를 가정하면, CFlow는 $56 \times 56 \times 256 \times 12 \approx 9.7M$ operations이지만 FastFlow는 $256 \times 56 \times 56 \approx 0.8M$으로 12배 적다.

이러한 단순화가 가능한 이유는 채널 간 독립성 가정이다. 서로 다른 채널의 activations가 조건부 독립적이라고 가정한다. 이는 강한 가정이지만, 실험적으로 성능 저하가 minimal함이 밝혀졌다. 오히려 일부 경우 성능이 향상되었다.

### 3.2.2 Channel Independence Assumption

채널 독립성 가정은 FastFlow의 이론적 기반이다. 이는 다음을 의미한다.

$$p(\mathbf{f}_{ij,1}, \ldots, \mathbf{f}_{ij,c}) = \prod_{k=1}^{c} p(\mathbf{f}_{ij,k})$$

위치 $(i,j)$에서 서로 다른 채널의 값들이 독립적이다. 이는 명백히 근사다. 실제로 채널들은 상관관계가 있다. CNN이 학습한 특징들은 서로 보완적인 정보를 담고 있다.

그러나 이 가정이 왜 reasonable한가? 첫째, 사전 학습된 CNN의 특징은 이미 높은 수준의 추상화를 거쳤다. 각 채널은 특정 visual concept이나 pattern을 인코딩한다. 예를 들어 한 채널은 edge를, 다른 채널은 texture를 포착한다. 이러한 high-level features는 상대적으로 독립적일 수 있다.

둘째, batch normalization과 같은 정규화 기법이 채널 간 correlation을 감소시킨다. 각 채널이 zero mean, unit variance로 정규화되면 linear correlation이 제거된다. 물론 비선형 의존성은 남지만, 이는 독립성 가정을 더 타당하게 만든다.

셋째, 이상 탐지의 목적이 exact density modeling이 아니라 anomaly scoring이다. 정확한 확률 분포보다는 정상과 이상의 구별이 중요하다. 채널 독립성 가정이 약간의 밀도 추정 오차를 초래하더라도, anomaly score의 순위가 보존되면 충분하다.

실험적 검증도 이 가정을 뒷받침한다. FastFlow와 CFlow의 성능 차이가 거의 없거나 오히려 FastFlow가 우수한 경우도 있었다. 이는 채널 간 의존성을 명시적으로 모델링하는 것이 이상 탐지에서 critical하지 않음을 시사한다.

또한 독립성 가정은 일종의 regularization 효과를 제공할 수 있다. 과도하게 복잡한 모델(CFlow)은 training data의 noise까지 학습하여 overfitting될 수 있다. 단순한 모델(FastFlow)은 본질적인 패턴만 포착하여 일반화가 더 나을 수 있다. 이는 bias-variance trade-off의 고전적 원리다.

채널 독립성을 완전히 포기하는 것은 아니다. 2D flow 내에서는 공간적 의존성이 여전히 모델링된다. 같은 채널 내의 서로 다른 위치 간 상관관계는 포착된다. 이는 결함의 공간적 구조(예: crack의 연속성)를 탐지하는 데 충분하다.

만약 채널 간 의존성이 critical하다면 어떻게 할까? 채널을 여러 그룹으로 나누고 각 그룹 내에서는 의존성을 모델링할 수 있다. 예를 들어 256 채널을 32개 그룹으로 나누면, 각 그룹은 8 채널을 가진다. 그룹 내에서는 3D flow를 사용하고 그룹 간에는 독립성을 가정한다. 이는 계산 비용과 표현력의 중간 지점이다.

### 3.2.3 Speed-Accuracy Trade-off

FastFlow의 단순화는 명백히 trade-off를 수반한다. 표현력을 희생하여 효율성을 얻는다. 놀라운 것은 이 trade-off가 매우 유리하다는 점이다. 약간의 표현력 손실로 극적인 속도 향상을 달성한다.

정량적으로 FastFlow는 CFlow 대비 다음을 제공한다.
- 속도: 3-5배 향상 (100-150ms → 20-50ms)
- 메모리: 2-3배 감소 (500MB-1GB → 200-500MB)
- 학습 시간: 2-3배 단축 (3-8시간 → 1-3시간)
- 성능: 0.2% 향상 또는 유지 (98.3% → 98.5%)

이는 Pareto improvement다. 모든 차원에서 개선되었다. 성능이 유지되거나 향상되면서 효율성이 크게 좋아졌다. 이는 드물게 win-win 상황이다.

왜 성능이 저하되지 않았을까? 여러 요인이 있다. 첫째, CFlow의 복잡한 모델이 실제로 과적합되었을 수 있다. 각 위치마다 독립적인 flow는 필요 이상의 자유도를 제공한다. 제한된 학습 데이터(카테고리당 200-400 이미지)에서 이러한 복잡한 모델은 noise를 학습할 위험이 있다.

둘째, FastFlow의 2D flow가 공간적 구조를 더 잘 포착할 수 있다. 채널 내의 공간적 패턴(예: 연속된 crack, 일정한 texture)이 명시적으로 모델링된다. CFlow는 각 위치를 독립적으로 처리하여 이러한 공간적 일관성을 충분히 활용하지 못할 수 있다.

셋째, inductive bias의 효과다. 채널 독립성 가정은 모델에 특정 구조를 부여한다. 이는 학습을 guide하여 더 robust한 representation을 찾게 한다. 완전히 자유로운 모델보다 적절한 제약이 있는 모델이 일반화에 유리할 수 있다.

속도-정확도 곡선을 그리면 FastFlow는 최적 지점에 있다. Memory-based 방법(PatchCore)은 더 빠르고 정확하지만(30-50ms, 99.1%) 학습 데이터를 저장해야 한다. Knowledge distillation(EfficientAD)은 더 빠르지만(1-5ms) 정확도가 낮다(97.8%). FastFlow는 98.5% 정확도와 20-50ms 속도로 균형점을 제공한다.

응용에 따라 최적 선택이 다르다. 최고 정확도가 필수면 PatchCore를 선택한다. 실시간 처리가 필요하면 EfficientAD를 선택한다. 균형잡힌 성능과 속도가 필요하면 FastFlow가 최선이다. 실무에서는 이러한 균형이 가장 일반적인 요구사항이다.

FastFlow의 성공은 deep learning에서 중요한 교훈을 제공한다. "Bigger models are better"라는 일반적 믿음이 항상 옳은 것은 아니다. 올바른 inductive bias와 적절한 단순화가 복잡한 모델보다 나을 수 있다. 특히 제한된 데이터와 특정 도메인(이상 탐지)에서는 더욱 그렇다.

## 3.3 Architecture Design

FastFlow의 architecture는 단순하면서도 효과적으로 설계되었다. 핵심은 2D normalizing flow를 각 채널에 적용하는 것이다.

**Overall Structure**

입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$가 주어지면, 사전 학습된 CNN 백본(예: ResNet50)을 통과시켜 특징 맵을 추출한다. 일반적으로 layer3의 출력을 사용한다. $\mathbf{F} \in \mathbb{R}^{h \times w \times c}$ (예: $56 \times 56 \times 1024$).

각 채널 $\mathbf{F}_{:,:,k}$에 대해 독립적인 2D normalizing flow $f_k$를 적용한다. 각 flow는 2D tensor를 latent code로 변환한다.

$$\mathbf{z}_k = f_k(\mathbf{F}_{:,:,k})$$

여기서 $\mathbf{z}_k \in \mathbb{R}^{h \times w}$는 $k$번째 채널의 latent representation이다. 모든 채널이 동일한 flow architecture를 사용하지만 파라미터는 공유하지 않는다. 즉, $c$개의 독립적인 flow가 있다.

각 채널의 log-likelihood를 계산하고 합산한다.

$$\log p(\mathbf{F}) = \sum_{k=1}^{c} \log p(\mathbf{F}_{:,:,k})$$

Base distribution은 표준 가우시안이다. 각 채널의 latent code $\mathbf{z}_k$가 $\mathcal{N}(0, 1)$을 따른다고 가정한다.

$$\log p(\mathbf{F}_{:,:,k}) = \log p_Z(\mathbf{z}_k) - \log \left| \det \frac{\partial f_k}{\partial \mathbf{F}_{:,:,k}} \right|$$

여기서 $p_Z(\mathbf{z}_k) = \prod_{i,j} \mathcal{N}(z_{k,ij}; 0, 1)$는 각 공간 위치가 독립적인 표준 가우시안이다.

**2D Flow Architecture**

2D flow는 2D coupling layers로 구성된다. 일반적인 coupling layer와 유사하지만, 1D 벡터 대신 2D tensor를 처리한다.

입력 2D tensor $\mathbf{Z} \in \mathbb{R}^{h \times w}$를 두 부분으로 나눈다. 공간적으로 나누는 방법이 있다. 예를 들어 checkerboard pattern으로 나눈다.

$$\mathbf{Z}_1 = \mathbf{Z}[\text{checkerboard pattern 1}]$$
$$\mathbf{Z}_2 = \mathbf{Z}[\text{checkerboard pattern 2}]$$

Checkerboard pattern은 $(i+j) \mod 2$로 정의된다. 홀수 합은 pattern 1, 짝수 합은 pattern 2다. 이는 인접한 픽셀들이 서로 다른 그룹에 속하게 한다.

Affine coupling은 다음과 같다.

$$\mathbf{X}_1 = \mathbf{Z}_1$$
$$\mathbf{X}_2 = \mathbf{Z}_2 \odot \exp(s(\mathbf{Z}_1)) + t(\mathbf{Z}_1)$$

여기서 $s$와 $t$는 2D convolutional networks다. $\mathbf{Z}_1$을 입력받아 $\mathbf{Z}_2$와 같은 크기의 출력을 생성한다.

Conditioning network는 일반적으로 작은 U-Net 구조를 사용한다. Down-sampling으로 receptive field를 넓히고 up-sampling으로 원래 해상도로 복원한다. 이는 공간적 맥락을 효과적으로 포착한다.

```
s(Z1) = Conv2d(64) → ReLU → Conv2d(128) → ReLU → 
        Upsample → Conv2d(64) → ReLU → Conv2d(output)
```

일반적으로 3-5개의 convolutional layers를 사용한다. 각 layer는 3×3 또는 5×5 kernel을 가진다. 너무 깊으면 계산 비용이 증가하고, 너무 얕으면 표현력이 부족하다.

Jacobian determinant는 효율적으로 계산된다. Coupling structure 덕분에 determinant는 $s(\mathbf{Z}_1)$의 합이다.

$$\log \left| \det \frac{\partial f}{\partial \mathbf{Z}} \right| = \sum_{i,j \in \text{pattern 2}} s(\mathbf{Z}_1)_{ij}$$

여러 coupling layers를 쌓는다. 각 layer마다 partition pattern을 번갈아 바꾼다. Layer 1은 pattern 1을 고정하고, layer 2는 pattern 2를 고정한다. 이렇게 하면 모든 픽셀이 충분히 변환된다.

일반적으로 4-8개의 coupling layers를 사용한다. CFlow(12 layers)보다 적지만 2D structure 덕분에 충분한 표현력을 가진다.

**Multi-scale Option**

FastFlow는 기본적으로 single-scale(layer3만)을 사용하지만, multi-scale 확장도 가능하다. Layer2와 layer4를 추가하여 3-scale로 만들 수 있다. 이는 성능을 약간 향상시키지만 계산 비용이 증가한다.

Multi-scale FastFlow는 여전히 CFlow보다 빠르다. 각 scale에서 2D flow를 사용하므로 CFlow의 3D flow보다 효율적이다. 성능 향상이 필요하고 계산 여유가 있다면 고려할 수 있다.

실험 결과 single-scale FastFlow가 대부분의 경우 충분했다. Multi-scale의 추가 이득은 0.5%포인트 미만이었다. 속도 저하(2-3배)를 고려하면 cost-benefit이 불리하다. 따라서 기본 권장은 single-scale이다.

**Parameter Sharing**

FastFlow의 주요 설계 결정 중 하나는 채널 간 파라미터 공유 여부다. 모든 채널이 동일한 flow를 사용하면 파라미터가 절약되고 학습이 안정적이다. 반면 각 채널이 독립적인 flow를 가지면 표현력이 높지만 파라미터가 증가한다.

원 논문은 파라미터 공유를 사용했다. 모든 $c$개 채널이 동일한 2D flow $f$를 공유한다. 이는 파라미터를 $c$배 절약한다. 예를 들어 256 채널이 있고 각 flow가 1MB라면, 공유 없이는 256MB지만 공유하면 1MB다.

파라미터 공유는 일종의 regularization이다. 모든 채널이 동일한 transformation을 거치므로 채널별 peculiarities를 학습하지 않는다. 대신 공통적인 정상 패턴을 포착한다. 이는 일반화에 유리하다.

대안적으로 채널을 그룹으로 나누고 그룹 내에서만 공유할 수 있다. 예를 들어 256 채널을 32개 그룹으로 나누면, 8개 채널마다 하나의 flow를 공유한다. 이는 표현력과 효율성의 균형점이다.

실험적으로 완전 공유가 가장 효율적이고 성능도 충분했다. 부분 공유나 비공유의 추가 이득이 크지 않았다. 따라서 기본 권장은 완전 공유다.

## 3.4 Performance Breakthrough

FastFlow는 MVTec AD 벤치마크에서 breakthrough 성능을 달성했다. 이미지 레벨 AUROC 98.5%, 픽셀 레벨 98.6%로 CFlow를 능가했다. 동시에 속도와 메모리에서 극적인 개선을 보였다.

**Benchmark Results**

카테고리별 성능은 매우 일관적이었다. 15개 중 13개 카테고리에서 98% 이상을 달성했다. Texture 카테고리에서 특히 강력했다. Carpet(99.5%), grid(99.9%), leather(100%), tile(99.8%), wood(99.6%)에서 거의 완벽한 분류를 보였다.

Object 카테고리에서도 우수했다. Bottle(100%), cable(99.6%), capsule(98.9%), metal_nut(98.5%)에서 높은 성능을 보였다. Screw(97.8%)와 toothbrush(97.1%)에서 상대적으로 낮았지만 여전히 competitive했다.

가장 어려운 카테고리는 hazelnut(97.3%)과 pill(96.8%)이었다. 이들은 모든 방법이 어려워하는 카테고리다. 미세하고 다양한 결함이 정상 변동과 구별하기 어렵다.

**Comparison with State-of-the-Art**

FastFlow vs CFlow: FastFlow는 평균 AUROC에서 0.2%포인트 우위(98.5% vs 98.3%)였다. 일부 카테고리에서는 1%포인트 이상 차이를 보였다. 동시에 3-5배 빠르고 메모리는 절반이었다. 명백한 우위다.

FastFlow vs PaDiM: FastFlow는 1%포인트 우위(98.5% vs 97.5%)였다. 모든 카테고리에서 일관되게 높았다. 추론 속도는 비슷하지만(20-50ms vs 50-100ms) FastFlow가 약간 빠르다. 메모리는 FastFlow가 약간 더 사용한다.

FastFlow vs PatchCore: PatchCore가 0.6%포인트 우위(99.1% vs 98.5%)였다. 이는 유일하게 FastFlow를 능가하는 방법이다. 그러나 추론 속도는 비슷하고(30-50ms vs 20-50ms), 메모리는 PatchCore가 훨씬 적다(1-2MB vs 200-500MB). Trade-off가 있다.

FastFlow vs STFPM: FastFlow는 1.7%포인트 우위(98.5% vs 96.8%)였다. 속도도 FastFlow가 빠르다(20-50ms vs 50-100ms). 메모리도 비슷하다. FastFlow가 명확히 우수하다.

전반적으로 FastFlow는 성능-속도-메모리의 최적 균형을 제공한다. PatchCore가 약간 더 정확하지만 학습 데이터를 저장해야 한다. FastFlow는 generative model로서 sampling, interpolation 등 추가 기능을 제공한다.

**Inference Speed Analysis**

FastFlow의 추론 시간을 분해하면 다음과 같다.
- Backbone forward pass: 10-20ms (전체의 50-60%)
- 2D Flow evaluation: 8-15ms (전체의 30-40%)
- Post-processing: 2-5ms (전체의 10%)

백본이 여전히 주된 병목이다. ResNet50의 forward pass가 시간의 절반을 차지한다. 더 빠른 백본(ResNet18, MobileNet)을 사용하면 전체 시간이 10-30ms로 줄어든다.

Flow evaluation은 효율적이다. 2D convolution의 병렬화가 잘 되고, 얕은 네트워크(4-8 layers)로 충분하다. GPU에서 매우 빠르게 실행된다.

배치 처리로 throughput을 크게 향상시킬 수 있다. 배치 크기 32로 처리하면 초당 30-50 이미지를 달성한다. 이는 대부분의 생산 라인 속도를 초과한다.

**Memory Footprint**

FastFlow의 메모리 사용량은 카테고리당 약 200-500MB다. 이는 flow 파라미터와 백본 가중치를 포함한다. 백본을 공유하면 추가 카테고리마다 50-100MB만 필요하다.

구체적으로 ResNet50 백본은 약 100MB다. 2D flow는 채널당 작은 CNN이므로 총 100-300MB다. 파라미터 공유 덕분에 채널 수($c=256$ 또는 $c=1024$)에 비례하지 않는다.

10개 카테고리를 배포하면 약 1-2GB가 필요하다(백본 공유 시). 100개 카테고리면 5-10GB다. 단일 GPU(10-20GB VRAM)에서 충분히 수용 가능하다.

엣지 배포도 가능하다. 경량 백본(MobileNetV2)을 사용하면 카테고리당 50-100MB로 줄어든다. Jetson Xavier(16GB)에서 수십 개 카테고리를 배포할 수 있다.

**Training Efficiency**

학습 시간은 카테고리당 1-3시간(GPU 기준)이다. CFlow(3-8시간)보다 2-3배 빠르다. 2D flow가 더 빠르게 수렴하기 때문이다.

학습 곡선도 smooth하다. Loss가 단조 감소하며 overfitting 징후가 적다. Early stopping이 거의 필요 없고 고정 epochs(100-200)로 충분하다.

다중 카테고리 학습은 병렬화 가능하다. 여러 GPU에서 카테고리를 동시에 학습하면 총 시간이 크게 줄어든다. 15개 MVTec 카테고리를 4개 GPU로 학습하면 4-6시간이면 완료된다.

## 3.5 Why Simplification Works

FastFlow의 성공은 역설적이다. 더 단순한 모델이 더 복잡한 모델(CFlow)을 능가했다. 이는 deep learning의 일반적 직관과 반대다. 왜 이런 일이 일어났을까?

**Inductive Bias**

채널 독립성 가정은 강력한 inductive bias를 제공한다. 모델이 학습해야 할 것을 제한하여 올바른 방향으로 guide한다. CNN 특징의 각 채널은 이미 특정 시각적 개념을 인코딩한다. 이들을 독립적으로 처리하는 것은 CNN의 구조와 일치한다.

반면 CFlow는 모든 채널을 함께 모델링한다. 이는 더 유연하지만, 제한된 학습 데이터에서 불필요한 복잡성을 초래한다. 모델이 채널 간 spurious correlations를 학습할 수 있다. 이는 training data에는 fit하지만 test data에는 일반화되지 않는다.

적절한 inductive bias는 sample efficiency를 높인다. 같은 양의 데이터로 더 나은 모델을 학습할 수 있다. 이상 탐지에서는 카테고리당 200-400장만 사용 가능하므로 sample efficiency가 critical하다.

**Regularization Effect**

단순화는 일종의 regularization이다. 모델의 capacity를 제한하여 overfitting을 방지한다. CFlow의 위치별 조건부 flow는 과도한 자유도를 제공한다. 각 위치마다 독립적인 분포를 학습하면 spatial smoothness가 무시된다.

FastFlow는 채널 내의 공간적 구조를 명시적으로 모델링한다. 2D flow는 인접 픽셀 간 상관관계를 포착한다. 이는 결함의 공간적 일관성(예: crack의 연속성)을 자연스럽게 학습한다. 이러한 구조적 제약이 일반화를 돕는다.

또한 파라미터 공유가 추가 regularization을 제공한다. 모든 채널이 동일한 transformation을 사용하므로 채널별 quirks를 학습하지 않는다. 대신 공통적이고 robust한 패턴을 포착한다.

**Occam's Razor**

가장 단순한 설명이 종종 옳다는 Occam's razor 원칙이 여기서도 적용된다. 이상 탐지의 목표는 정상 분포를 학습하는 것이다. 이 분포가 실제로 얼마나 복잡한가?

사전 학습된 CNN 특징 공간에서 정상 분포는 이미 상당히 simple할 수 있다. 고차원 원본 이미지 공간에서는 복잡하지만, 학습된 표현 공간에서는 structured되어 있다. 채널 독립적인 2D 분포로도 충분히 표현될 수 있다.

과도하게 복잡한 모델은 불필요한 details까지 학습한다. 학습 데이터의 idiosyncrasies, noise, outliers까지 fit한다. 단순한 모델은 본질적인 패턴만 포착하여 새로운 데이터에 better generalize한다.

**Computational Geometry**

2D flow가 3D flow보다 특정 기하학적 구조를 더 잘 포착할 수 있다는 가설도 있다. 결함은 종종 연속된 공간적 패턴을 보인다. Crack, scratch, contamination은 모두 spatially coherent하다.

2D flow는 이러한 spatial coherence를 직접 모델링한다. Convolutional structure가 이웃 픽셀 간 관계를 natural하게 포착한다. 3D flow는 각 위치를 독립적으로 처리하여 이러한 공간적 정보를 충분히 활용하지 못할 수 있다.

또한 2D flow는 translation equivariance를 제공한다. 결함의 위치가 바뀌어도 동일하게 탐지한다. 3D flow의 position-conditional architecture는 위치 정보를 explicitly encoding하여 이러한 equivariance를 깨뜨릴 수 있다.

**Empirical Evidence**

가장 강력한 증거는 empirical results다. FastFlow가 다양한 데이터셋과 백본에서 일관되게 좋은 성능을 보였다. MVTec AD뿐만 아니라 BTAD, VisA, 커스텀 데이터셋에서도 효과적이었다.

다양한 백본(ResNet, EfficientNet, Wide ResNet)과의 조합도 잘 작동했다. 이는 FastFlow의 아이디어가 특정 설정에 overfitting되지 않았음을 시사한다. 일반적이고 robust한 원칙이다.

후속 연구들도 FastFlow의 접근을 다른 도메인에 적용하고 있다. Video anomaly detection, medical image analysis, 3D point cloud anomaly detection 등에서 유사한 simplification이 효과적임이 밝혀지고 있다.

## 3.6 Implementation Guide

FastFlow를 실무에 구현하는 구체적인 가이드를 제공한다.

**Backbone Selection**

ResNet50이 기본 선택이다. 성능과 속도의 균형이 좋다. ImageNet pretrained weights를 torchvision에서 로드한다. Layer3 output을 사용한다($56 \times 56 \times 1024$).

더 빠른 추론이 필요하면 ResNet18을 선택한다. 특징 차원이 512로 절반이고 속도도 2배 빠르다. 성능은 1-2%포인트 낮지만 여전히 competitive하다(96-97% AUROC).

최고 성능이 필요하면 Wide ResNet-50을 선택한다. 특징이 더 풍부하여 0.5-1%포인트 향상된다. 그러나 속도는 느려진다.

엣지 배포에는 MobileNetV2나 EfficientNet-Lite를 고려한다. 메모리와 계산이 크게 절약되지만 성능은 2-3%포인트 낮다. Trade-off를 신중히 평가해야 한다.

**Flow Architecture**

2D coupling layers는 4-8개를 사용한다. 6 layers가 일반적으로 최적이다. 더 적으면 표현력 부족, 더 많으면 overfitting과 속도 저하 위험이 있다.

Conditioning network는 작은 U-Net이다. 3-5개 convolutional layers로 충분하다. 각 layer는 3×3 kernel, ReLU activation, optional batch normalization을 가진다.

```python
class ConditioningNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)
```

Hidden channels는 입력 채널과 비슷하게 설정한다(예: 32-64). 너무 크면 계산 비용이 증가하고 너무 작으면 표현력이 부족하다.

**Training Configuration**

Adam optimizer with learning rate $10^{-3}$을 사용한다. Batch size는 GPU 메모리에 따라 16-32다. Gradient accumulation으로 effective batch size를 늘릴 수 있다.

Epochs는 100-200이 일반적이다. Early stopping은 선택사항이다. FastFlow는 overfitting이 적어 고정 epochs로도 충분하다.

Data augmentation은 minimal하게 사용한다. 이상 탐지에서는 정상 분포를 정확히 학습하는 것이 중요하다. 과도한 augmentation은 분포를 왜곡할 수 있다. 기본적으로 random crop, horizontal flip만 사용한다.

Learning rate scheduling은 cosine annealing을 권장한다. 학습 후반부에 learning rate를 점진적으로 줄여 fine-tuning 효과를 제공한다.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)
```

**Inference Pipeline**

추론은 다음 단계로 진행된다.

1. 이미지 전처리: Resize to 256×256, normalize with ImageNet stats
2. Backbone forward: Extract layer3 features
3. Flow evaluation: Apply 2D flow to each channel, compute log-likelihood
4. Anomaly map: Sum log-likelihoods across channels, negate to get anomaly scores
5. Image score: Take maximum of anomaly map

```python
def infer(image, backbone, flow):
    x = preprocess(image)
    features = backbone(x)  # [1, C, H, W]
    
    log_likelihood_map = 0
    for c in range(features.shape[1]):
        channel_feature = features[:, c:c+1, :, :]  # [1, 1, H, W]
        z, log_det = flow(channel_feature)
        log_p_z = -0.5 * (z**2).sum(dim=[2,3])  # Gaussian log-likelihood
        log_likelihood_map += (log_p_z - log_det)
    
    anomaly_map = -log_likelihood_map  # Negate for anomaly score
    anomaly_map = F.interpolate(anomaly_map, size=256, mode='bilinear')
    image_score = anomaly_map.max()
    
    return anomaly_map, image_score
```

**Threshold Selection**

Validation set에서 ROC 분석으로 임계값을 선택한다. 목표 재현율(예: 95%)에 대응하는 임계값을 찾는다. 또는 F1 score를 최대화하는 임계값을 선택한다.

픽셀 레벨 임계값은 이미지 레벨보다 낮게 설정한다. 결함 영역을 세그멘테이션하기 위해 더 민감하게 탐지한다.

Adaptive thresholding도 고려할 수 있다. 정상 샘플의 이상 점수 분포를 추적하고, 백분위수 기반으로 동적 임계값을 설정한다.

**Optimization Techniques**

TensorRT나 ONNX Runtime으로 모델을 최적화한다. Kernel fusion, quantization으로 추론 속도를 2-3배 향상시킬 수 있다.

Mixed precision(FP16) 추론도 효과적이다. 메모리를 절반으로 줄이고 속도도 빠르다. 정확도 저하는 negligible하다.

Batch inference로 throughput을 극대화한다. 생산 라인에서 여러 이미지가 동시에 들어오면 배치로 처리한다. GPU 활용률이 크게 향상된다.

Model pruning이나 distillation으로 모델 크기를 줄일 수 있다. 중요하지 않은 가중치를 제거하거나, 작은 student model을 학습한다. 엣지 배포 시 유용하다.

**Common Pitfalls**

Normalization 불일치를 조심한다. 학습과 추론에서 동일한 ImageNet normalization을 사용해야 한다. Mean과 std 값을 정확히 확인한다.

백본을 frozen 상태로 유지한다. Backbone의 gradient를 비활성화하여 메모리를 절약하고 학습을 안정화한다. Fine-tuning은 일반적으로 필요 없고 오히려 해로울 수 있다.

Overly complex conditioning network를 피한다. 3-5 layers면 충분하다. 더 깊으면 계산 비용만 증가하고 성능 향상은 미미하다.

충분한 학습 epochs를 사용한다. 너무 일찍 멈추면 수렴하지 않는다. Loss curve를 모니터링하여 plateau에 도달했는지 확인한다.

FastFlow의 단순함과 효과성은 실무 배포에 이상적이다. 적절한 구현과 최적화로 대부분의 산업 응용에서 탁월한 성능을 제공한다. Memory-based 방법의 최고 정확도와 knowledge distillation의 실시간 속도 사이의 sweet spot을 차지한다.

# 4. CS-Flow (2021)

## 4.1 Basic Information

CS-Flow(Cross-Scale Flow)는 2021년 Rudolph 등이 제안한 normalizing flow 기반 이상 탐지 방법으로, 서로 다른 스케일의 특징 간 정보 교환을 통해 성능을 향상시켰다. 이 연구는 WACV 2022에서 발표되었으며, multi-scale feature fusion의 새로운 접근법을 제시했다.

CS-Flow의 핵심 아이디어는 단순히 여러 스케일의 특징을 독립적으로 처리하는 것이 아니라, 스케일 간 상호작용을 명시적으로 모델링하는 것이다. CFlow는 layer2, layer3, layer4를 각각 독립적인 flow로 처리하고 결과를 후처리 단계에서 결합한다. CS-Flow는 flow 내부에서 스케일 간 정보를 교환하여 더 풍부한 표현을 학습한다.

방법론적으로 CS-Flow는 multi-scale CNN 특징을 추출한 후, coupling layer 내부에서 서로 다른 스케일의 정보를 결합한다. 예를 들어 layer3를 변환할 때 layer2와 layer4의 정보를 conditioning에 사용한다. 이는 각 스케일이 다른 스케일의 맥락을 고려하여 변환되도록 한다.

성능 면에서 CS-Flow는 CFlow와 유사한 수준을 보였다. MVTec AD에서 이미지 레벨 AUROC 약 98.3-98.5%를 달성했다. 독립적인 multi-scale(CFlow)과 비교했을 때 통계적으로 유의미한 차이는 없었다. 일부 카테고리에서는 약간 우수했지만 다른 카테고리에서는 비슷했다.

주된 한계는 복잡성과 계산 비용의 증가다. Cross-scale information exchange는 추가 파라미터와 연산을 요구한다. 추론 시간이 CFlow보다 약간 더 걸리고(120-180ms vs 100-150ms), 메모리 사용량도 증가한다(600MB-1.5GB vs 500MB-1GB). 성능 향상이 미미한 것을 고려하면 cost-benefit이 불리하다.

CS-Flow는 학술적으로는 흥미로운 아이디어를 제시했지만, 실무적으로는 제한적인 영향을 미쳤다. FastFlow가 단순화로 효율성을 크게 개선한 것과 대조적으로, CS-Flow는 복잡화로 미미한 성능 향상만 얻었다. 이는 "more is not always better" 교훈을 제공한다. 대부분의 응용에서는 단순한 CFlow나 FastFlow가 더 실용적이다.

## 4.2 Cross-Scale Information Exchange

CS-Flow의 핵심 메커니즘은 서로 다른 스케일의 특징 간 정보 교환이다. 전통적인 multi-scale 접근은 각 스케일을 독립적으로 처리한다. CS-Flow는 스케일 간 의존성을 명시적으로 모델링한다.

**Architecture Overview**

CNN의 세 개 층(layer2, layer3, layer4)에서 특징을 추출한다. 각 층의 해상도와 채널이 다르다.
- Layer2: $h_2 \times w_2 \times c_2$ (예: $64 \times 64 \times 512$)
- Layer3: $h_3 \times w_3 \times c_3$ (예: $32 \times 32 \times 1024$)
- Layer4: $h_4 \times w_4 \times c_4$ (예: $16 \times 16 \times 2048$)

각 스케일에 normalizing flow를 적용하지만, coupling layers의 conditioning networks가 다른 스케일의 정보를 받는다. Layer3의 coupling layer는 다음과 같다.

$$\mathbf{x}_2^{(3)} = \mathbf{z}_2^{(3)} \odot \exp(s(\mathbf{z}_1^{(3)}, \mathbf{F}^{(2)}, \mathbf{F}^{(4)})) + t(\mathbf{z}_1^{(3)}, \mathbf{F}^{(2)}, \mathbf{F}^{(4)})$$

여기서 $s$와 $t$는 현재 스케일($\mathbf{z}_1^{(3)}$)뿐만 아니라 다른 스케일($\mathbf{F}^{(2)}$, $\mathbf{F}^{(4)}$)의 특징도 입력받는다. 이는 multi-scale conditioning이다.

**Feature Alignment**

서로 다른 해상도의 특징을 결합하려면 공간 정렬이 필요하다. Layer2의 $64 \times 64$ 특징과 layer4의 $16 \times 16$ 특징을 layer3의 $32 \times 32$ 크기로 맞춘다.

업샘플링은 bilinear interpolation을 사용한다. Layer4의 특징을 $32 \times 32$로 확대한다. 다운샘플링은 average pooling을 사용한다. Layer2의 특징을 $32 \times 32$로 축소한다.

$$\mathbf{F}^{(2)}_{\text{down}} = \text{AvgPool}(\mathbf{F}^{(2)}, (32, 32))$$
$$\mathbf{F}^{(4)}_{\text{up}} = \text{Upsample}(\mathbf{F}^{(4)}, (32, 32))$$

정렬된 특징들을 concatenate하여 conditioning network에 입력한다.

$$\mathbf{h} = [\mathbf{z}_1^{(3)}; \mathbf{F}^{(2)}_{\text{down}}; \mathbf{F}^{(4)}_{\text{up}}]$$

여기서 $[\cdot; \cdot]$는 채널 차원 concatenation이다. 결과 텐서 $\mathbf{h}$는 $32 \times 32 \times (c_1^{(3)} + c_2 + c_4)$ 크기를 가진다.

**Conditioning Network Design**

Multi-scale input을 처리하기 위해 conditioning network는 더 복잡해진다. 단순한 fully connected network 대신 convolutional network를 사용한다.

$$s(\mathbf{h}) = \text{Conv}(\text{ReLU}(\text{Conv}(\text{ReLU}(\text{Conv}(\mathbf{h})))))$$

여러 convolutional layers가 다양한 스케일의 정보를 통합한다. Receptive field가 넓어져 공간적 맥락을 효과적으로 포착한다.

일부 구현에서는 attention mechanism을 추가한다. 어떤 스케일의 정보가 중요한지 동적으로 결정한다. 각 스케일에 가중치를 할당하여 adaptive fusion을 수행한다.

$$\mathbf{h}_{\text{fused}} = \alpha_2 \mathbf{F}^{(2)}_{\text{down}} + \alpha_3 \mathbf{z}_1^{(3)} + \alpha_4 \mathbf{F}^{(4)}_{\text{up}}$$

여기서 $\alpha_2, \alpha_3, \alpha_4$는 learnable weights이거나 입력에 의존하는 attention weights다.

**Bidirectional Information Flow**

CS-Flow는 양방향 정보 흐름을 가능하게 한다. Layer3가 layer2와 layer4로부터 정보를 받을 뿐만 아니라, layer2는 layer3로부터, layer4는 layer3로부터 정보를 받는다.

이는 각 스케일이 다른 모든 스케일과 상호작용함을 의미한다. Layer2의 coupling layer는 layer3와 layer4를 conditioning으로 사용한다. Layer4의 coupling layer는 layer2와 layer3를 사용한다.

이러한 모든 스케일 간 연결은 파라미터와 계산을 크게 증가시킨다. 각 스케일의 flow가 다른 두 스케일의 정보를 처리해야 하므로 conditioning networks가 더 복잡해진다.

**Theoretical Motivation**

Cross-scale information exchange의 이론적 동기는 다음과 같다. 서로 다른 스케일의 특징은 상호 보완적인 정보를 담는다. Low-level features는 fine-grained details를, high-level features는 semantic context를 제공한다.

결함 탐지에서 두 유형의 정보가 모두 중요하다. 작은 scratch는 low-level features에서 잘 보이고, 큰 변형은 high-level features에서 명확하다. 스케일 간 정보를 교환하면 모델이 multi-scale context를 고려하여 더 robust한 판단을 내릴 수 있다.

또한 스케일 간 일관성을 강제할 수 있다. 정상 샘플은 모든 스케일에서 일관된 패턴을 보여야 한다. 이상 샘플은 특정 스케일에서만 비정상적이거나 스케일 간 불일치를 보일 수 있다. Cross-scale modeling이 이러한 불일치를 포착할 수 있다.

## 4.3 Multi-resolution Features

Multi-resolution feature processing은 CS-Flow의 설계에서 중요한 역할을 한다. 단순히 여러 해상도를 사용하는 것을 넘어, 해상도 간 관계를 활용한다.

**Resolution Hierarchy**

CNN의 계층적 구조는 자연스러운 해상도 계층을 제공한다. 각 층은 이전 층보다 낮은 해상도와 높은 semantic level을 가진다. CS-Flow는 이 계층을 명시적으로 활용한다.

Layer2 ($64 \times 64$)는 high-resolution, low-level features를 제공한다. 미세한 texture와 작은 결함에 민감하다. 그러나 semantic understanding이 부족하여 정상 변동과 이상을 구별하기 어려울 수 있다.

Layer4 ($16 \times 16$)는 low-resolution, high-level features를 제공한다. Semantic context와 global structure를 포착한다. 그러나 해상도가 낮아 작은 결함을 놓칠 수 있다.

Layer3 ($32 \times 32$)는 중간 지점으로, 적절한 해상도와 semantic level을 가진다. 대부분의 결함을 포착하는 데 충분하다. FastFlow가 layer3만 사용하는 이유다.

**Feature Pyramid**

CS-Flow는 feature pyramid를 구성한다. 서로 다른 해상도의 특징을 정렬하고 결합하여 multi-scale representation을 만든다. 이는 object detection의 Feature Pyramid Network(FPN)와 유사한 아이디어다.

각 스케일에서 다른 스케일의 정보를 통합하면, 모든 스케일이 풍부한 multi-resolution context를 가진다. Layer2는 high-level semantic을, layer4는 fine-grained details를 추가로 받는다. 이는 각 스케일의 한계를 보완한다.

실험적으로 feature pyramid 접근은 일부 카테고리에서 유용했다. 특히 다양한 크기의 결함이 혼재된 경우다. 작은 scratch와 큰 crack이 모두 존재하면, multi-resolution modeling이 둘 다 잘 탐지한다.

그러나 모든 카테고리에서 이득이 있는 것은 아니었다. 일부에서는 single-scale(FastFlow)과 비슷하거나 오히려 낮았다. 이는 추가 복잡성이 항상 정당화되지 않음을 시사한다.

**Computational Cost**

Multi-resolution processing의 주된 대가는 계산 비용이다. 세 개 스케일을 모두 처리하고, 스케일 간 정보를 교환하려면 상당한 연산이 필요하다.

Feature alignment(upsampling, downsampling)은 추가 메모리와 시간을 소비한다. Concatenation은 채널 차원을 증가시켜 subsequent layers의 비용을 높인다. Conditioning networks가 더 많은 입력을 처리해야 하므로 복잡해진다.

구체적으로 CS-Flow의 추론 시간은 CFlow보다 20-30% 길다. 120-180ms vs 100-150ms다. 메모리도 20-50% 증가한다. 이는 성능 향상(0-1%포인트)에 비해 비용이 크다.

**Practical Considerations**

실무에서 CS-Flow를 사용할지 결정할 때 다음을 고려해야 한다. 데이터에 다양한 크기의 결함이 있는가? 그렇다면 multi-resolution이 도움될 수 있다. 단일 유형의 결함만 있다면 single-scale로 충분하다.

계산 리소스가 충분한가? 서버 환경에서 throughput이 중요하지 않다면 CS-Flow를 시도할 수 있다. 엣지 환경이나 실시간 요구가 있다면 FastFlow가 더 적합하다.

성능 향상이 critical한가? 0.5%포인트 차이가 중요한 응용(예: critical safety)이라면 CS-Flow를 고려한다. 98%와 98.5%의 차이가 미미하다면 단순한 방법을 선택한다.

대부분의 경우 FastFlow의 단순성과 효율성이 CS-Flow의 잠재적 성능 향상을 압도한다. CS-Flow는 특수한 상황에서만 가치가 있다.

## 4.4 Performance and Use Cases

CS-Flow의 실험적 성능과 적합한 응용 영역을 분석한다.

**Benchmark Performance**

MVTec AD에서 CS-Flow는 이미지 레벨 AUROC 98.3-98.5%를 달성했다. CFlow와 거의 동일하다. 일부 카테고리에서 약간 우수했지만 통계적 유의성이 제한적이었다.

카테고리별로 보면 complex structures를 가진 object 카테고리에서 약간 이득이 있었다. Screw(98.5% vs 97.8%), toothbrush(97.8% vs 97.3%)에서 0.5-1%포인트 향상되었다. 이들은 다양한 스케일의 정보가 유용한 경우다.

반면 texture 카테고리에서는 차이가 없었다. Carpet, grid, leather 등에서 CFlow와 동일했다. 반복 패턴은 single-scale로도 충분히 모델링된다.

픽셀 레벨 성능도 유사했다. 98.5-98.7%로 CFlow와 거의 같다. 결함 위치 특정 능력에서 큰 차이가 없었다.

**Comparison with FastFlow**

FastFlow와 비교하면 CS-Flow는 불리하다. 성능은 비슷한데(98.5% vs 98.5%) 속도는 3배 느리고(120-180ms vs 20-50ms) 메모리는 2-3배 많다(600MB-1.5GB vs 200-500MB).

유일한 장점은 일부 특수 케이스에서 약간 더 나을 수 있다는 것이다. 그러나 이마저도 일관적이지 않고 카테고리에 따라 다르다. 대부분의 응용에서 FastFlow가 명백히 우수하다.

학술적으로 CS-Flow는 흥미로운 아이디어를 제시했지만, 실무적 가치는 제한적이다. Simplicity와 efficiency가 slight performance gain보다 중요한 실무 세계에서는 채택되지 않았다.

**Suitable Use Cases**

CS-Flow가 적합한 특수한 경우들이 있다.

첫째, 다양한 크기의 결함이 혼재된 복잡한 제품이다. 작은 scratch, 중간 크기 dent, 큰 crack이 모두 나타날 수 있다. 이런 경우 multi-resolution modeling이 각 유형을 효과적으로 포착한다.

둘째, 최고 정확도가 절대적으로 필요한 critical 응용이다. 0.5%포인트 차이도 중요한 안전 부품이나 의료 기기 검사에서 고려할 수 있다. 그러나 이런 경우 PatchCore(99.1%)가 더 나은 선택일 수 있다.

셋째, 계산 리소스가 풍부한 연구 환경이다. 알고리즘 개발이나 벤치마킹에서 다양한 방법을 비교할 때 CS-Flow를 포함할 수 있다. 실무 배포는 아니지만 학술적 완결성을 위해서다.

넷째, multi-scale information fusion의 효과를 연구하는 경우다. CS-Flow는 이 주제의 좋은 사례 연구다. 어떤 상황에서 유용하고 어디서 한계가 있는지 배울 수 있다.

**Lessons Learned**

CS-Flow의 경험은 중요한 교훈을 제공한다. Complexity는 자동으로 better performance를 보장하지 않는다. 잘 설계된 단순한 모델(FastFlow)이 복잡한 모델(CS-Flow)을 능가할 수 있다.

Incremental improvements는 cost-benefit 분석이 필요하다. 0.5%포인트 성능 향상이 3배 느린 속도와 2배 큰 메모리를 정당화하는가? 대부분의 경우 아니다. 예외적인 상황에서만 그렇다.

Multi-scale processing은 항상 유익한 것은 아니다. 적절한 single-scale(layer3)이 많은 경우 충분하다. Multi-scale의 이득은 데이터와 task에 의존적이다. Empirical validation이 필수다.

Occam's razor는 여전히 유효하다. 간단한 설명이 복잡한 것보다 자주 옳다. Deep learning에서도 마찬가지다. Unnecessary complexity는 overfitting, slow inference, difficult deployment를 초래한다.

# 5. U-Flow (2022)

## 5.1 Basic Information

U-Flow는 2022년 Rudolph 등이 제안한 normalizing flow 기반 방법으로, U-Net 구조를 통합하여 이상 탐지의 실용성을 높였다. 이 연구는 operational deployment에 초점을 맞췄으며, 특히 자동 임계값 선택과 안정적인 배포를 강조했다.

U-Flow의 핵심 기여는 두 가지다. 첫째, U-Net inspired architecture를 normalizing flow에 통합했다. 이는 더 풍부한 multi-scale features를 효율적으로 처리한다. 둘째, automatic threshold selection 메커니즘을 제안했다. 이는 validation set 없이도 적절한 임계값을 찾을 수 있게 한다.

방법론적으로 U-Flow는 encoder-decoder 구조를 flow에 결합한다. Encoder는 입력 특징을 점진적으로 다운샘플링하고, decoder는 다시 업샘플링한다. Skip connections가 서로 다른 레벨을 연결한다. 각 레벨에서 coupling layers를 적용하여 정상 분포를 학습한다.

성능 면에서 U-Flow는 CFlow와 비슷한 수준이다. MVTec AD에서 이미지 레벨 AUROC 약 98.0-98.3%를 달성했다. FastFlow보다는 약간 낮지만 합리적이다. 주된 장점은 성능 자체보다 operational reliability다. 안정적인 학습, 자동 임계값, robust deployment가 실무에서 가치 있다.

U-Flow는 학술적보다 실무적 기여를 목표로 했다. 최고 성능보다는 신뢰할 수 있는 배포를 우선했다. 이는 연구와 산업의 격차를 좁히려는 시도다. 그러나 FastFlow의 단순성과 PatchCore의 최고 성능 사이에서 명확한 niche를 찾지 못했다. 채택은 제한적이었다.

## 5.2 U-Net Integration

U-Flow의 architecture는 U-Net의 encoder-decoder 구조를 normalizing flow와 결합한다. 이는 multi-scale processing을 더 통합된 방식으로 수행한다.

**Encoder-Decoder Structure**

입력 특징 맵 $\mathbf{F} \in \mathbb{R}^{h \times w \times c}$가 주어지면, encoder가 여러 레벨로 다운샘플링한다. 각 레벨은 절반 해상도와 2배 채널을 가진다.

- Level 0: $h \times w \times c$ (입력)
- Level 1: $h/2 \times w/2 \times 2c$ (downsampled)
- Level 2: $h/4 \times w/4 \times 4c$ (further downsampled)

각 다운샘플링은 strided convolution이나 pooling으로 수행된다. 채널은 convolutional layer로 증가시킨다.

Decoder는 역과정을 수행한다. Upsampling으로 해상도를 복원하고 채널을 줄인다. Skip connections가 encoder의 동일 레벨 특징을 decoder에 전달한다.

$$\mathbf{D}_i = \text{Upsample}(\mathbf{D}_{i+1}) + \mathbf{E}_i$$

여기서 $\mathbf{E}_i$는 encoder의 level $i$ 특징, $\mathbf{D}_i$는 decoder의 level $i$ 특징이다. Skip connections는 low-level details를 보존한다.

**Flow at Each Level**

각 encoder와 decoder 레벨에서 normalizing flow를 적용한다. Coupling layers가 해당 레벨의 특징을 변환한다. 각 레벨은 독립적인 flow를 가지지만, U-Net structure를 통해 간접적으로 연결된다.

Level $i$의 log-likelihood를 계산한다.

$$\log p_i(\mathbf{F}_i) = \log p_Z(\mathbf{z}_i) - \log \left| \det \frac{\partial f_i}{\partial \mathbf{F}_i} \right|$$

모든 레벨의 log-likelihood를 합산하여 전체 likelihood를 얻는다.

$$\log p(\mathbf{F}) = \sum_{i} w_i \log p_i(\mathbf{F}_i)$$

여기서 $w_i$는 각 레벨의 가중치다. 동일 가중치를 사용하거나 learnable parameters로 만들 수 있다.

**Skip Connection Integration**

Skip connections는 U-Net의 핵심이다. U-Flow에서도 중요한 역할을 한다. Encoder 특징을 decoder에 전달하여 fine-grained information을 보존한다.

Skip connections를 flow의 conditioning으로도 사용할 수 있다. Decoder의 coupling layers가 encoder의 동일 레벨 특징을 conditioning input으로 받는다.

$$\mathbf{x}_2^{(i)} = \mathbf{z}_2^{(i)} \odot \exp(s(\mathbf{z}_1^{(i)}, \mathbf{E}_i)) + t(\mathbf{z}_1^{(i)}, \mathbf{E}_i)$$

이는 decoder flow가 encoder의 정보를 활용하여 더 informed transformation을 수행하게 한다.

**Advantages of U-Net Structure**

U-Net architecture는 여러 이점을 제공한다. 첫째, multi-scale features를 자연스럽게 처리한다. Hierarchical structure가 서로 다른 추상화 레벨을 명시적으로 모델링한다.

둘째, skip connections가 gradient flow를 개선한다. Encoder에서 decoder로의 직접 경로가 vanishing gradient 문제를 완화한다. 이는 학습 안정성을 높인다.

셋째, parameter efficiency가 좋다. Single large flow보다 여러 small flows가 파라미터를 절약한다. 각 레벨은 해당 해상도에 적합한 작은 네트워크를 사용한다.

넷째, 해석 가능성이 향상된다. 각 레벨의 이상 맵을 별도로 시각화할 수 있다. 어떤 스케일에서 이상이 탐지되었는지 파악할 수 있다.

**Implementation Details**

U-Flow는 일반적으로 3-4 레벨을 사용한다. 너무 많은 레벨은 최저 해상도가 너무 작아져 정보가 부족하다. 너무 적은 레벨은 multi-scale의 이점을 충분히 활용하지 못한다.

각 레벨의 coupling layers는 4-6개를 사용한다. 전체 flow depth는 레벨 수 × coupling layers per level이다. 예를 들어 3 레벨 × 5 layers = 15 layers다. 이는 CFlow(12 layers)와 비슷하다.

Downsampling과 upsampling은 learnable이다. Strided convolution과 transposed convolution을 사용하여 파라미터로 학습한다. 이는 고정된 pooling/interpolation보다 유연하다.

## 5.3 Automatic Threshold Selection

U-Flow의 중요한 기여는 automatic threshold selection 메커니즘이다. 전통적으로 임계값은 validation set의 ROC 분석으로 선택된다. 이는 추가 레이블된 데이터가 필요하고, 주관적 결정이 개입한다.

**Problem Formulation**

이상 탐지의 핵심 과제 중 하나는 적절한 임계값 설정이다. 너무 낮으면 false positives가 많고, 너무 높으면 false negatives가 많다. 최적 임계값은 응용의 비용 구조에 의존한다.

전통적 접근은 validation set에서 ROC 곡선을 그리고 목표 재현율이나 F1 score에 대응하는 임계값을 선택한다. 이는 몇 가지 문제가 있다.

첫째, validation set이 필요하다. 정상과 이상 샘플이 모두 레이블되어야 한다. 실무에서는 이상 샘플을 수집하기 어렵다. 특히 희귀한 결함의 경우 거의 불가능하다.

둘째, validation set이 test distribution을 대표하지 못할 수 있다. Distribution shift가 있으면 선택된 임계값이 suboptimal하다. 재배포 시마다 재조정이 필요하다.

셋째, 주관적 판단이 개입한다. 어떤 메트릭을 최적화할 것인가? 재현율, 정밀도, F1 score? 각각 다른 임계값을 제공한다.

**U-Flow's Approach**

U-Flow는 training distribution만으로 임계값을 자동 설정하는 방법을 제안한다. 핵심 아이디어는 정상 샘플의 이상 점수 분포를 모델링하는 것이다.

학습 후 모든 정상 학습 샘플의 이상 점수를 계산한다. 이들의 분포를 분석하여 통계량을 추출한다. 평균 $\mu$와 표준편차 $\sigma$를 계산한다.

$$\mu = \frac{1}{N} \sum_{n=1}^{N} s(\mathbf{x}_n)$$
$$\sigma = \sqrt{\frac{1}{N-1} \sum_{n=1}^{N} (s(\mathbf{x}_n) - \mu)^2}$$

임계값을 $\mu + k\sigma$로 설정한다. 여기서 $k$는 sensitivity parameter다. 일반적으로 $k=2$ 또는 $k=3$을 사용한다.

$$\tau = \mu + k\sigma$$

이는 정상 분포의 tail을 넘어서는 점수를 이상으로 간주한다. $k=2$이면 약 95%의 정상 샘플이 임계값 이하다(가우시안 가정 하). $k=3$이면 약 99.7%다.

**Theoretical Justification**

이 접근의 이론적 근거는 normality assumption이다. 정상 샘플의 이상 점수가 근사적으로 가우시안 분포를 따른다고 가정한다. Normalizing flow가 정상 분포를 잘 학습했다면, negative log-likelihoods는 bounded하고 bell-shaped distribution을 형성한다.

가우시안 가정 하에서 $\mu + k\sigma$ 임계값은 명확한 의미를 가진다. False positive rate를 특정 수준으로 제어한다. $k=2$는 약 5% FPR, $k=3$는 약 0.3% FPR을 목표로 한다.

실제로 분포가 정확히 가우시안은 아니지만, 이 근사는 실용적으로 충분하다. Empirical validation에서 합리적인 임계값을 제공했다.

**Adaptive Threshold**

더 정교한 변형은 adaptive threshold다. 데이터의 특성에 따라 $k$를 동적으로 조정한다. 예를 들어 정상 점수의 분포가 넓으면(높은 $\sigma$) 더 큰 $k$를 사용하고, 좁으면 작은 $k$를 사용한다.

또는 percentile-based threshold를 사용할 수 있다. 정상 점수의 95th 또는 99th percentile을 임계값으로 설정한다. 이는 분포의 형태에 관계없이 작동한다.

$$\tau = \text{Percentile}_{95}(\{s(\mathbf{x}_n)\}_{n=1}^{N})$$

Percentile 방법은 outliers에 robust하다. 몇 개의 비정상적으로 높은 정상 점수가 있어도 임계값이 크게 영향받지 않는다.

**Limitations**

Automatic threshold selection은 완벽하지 않다. 몇 가지 한계가 있다.

첫째, 학습 데이터의 품질에 의존한다. 학습 데이터에 실제로는 이상인 샘플이 섞여 있으면 임계값이 너무 높게 설정된다. 이상 샘플을 정상으로 오인하게 된다.

둘째, 정상 분포의 가정이 위반되면 부정확하다. Multimodal이나 heavily skewed 분포에서는 단순한 $\mu + k\sigma$가 적절하지 않다. 더 sophisticated한 분포 모델링이 필요하다.

셋째, false positive와 false negative의 비용을 고려하지 않는다. 응용에 따라 한 쪽이 다른 쪽보다 훨씬 중요할 수 있다. Automatic method는 이를 반영하지 못한다.

넷째, validation 없이는 임계값의 품질을 평가할 수 없다. 선택된 임계값이 실제로 잘 작동하는지 확인할 방법이 없다. Blind trust가 필요하다.

실무에서는 automatic threshold를 starting point로 사용하고, 소량의 validation data로 fine-tuning하는 것이 권장된다. 완전 자동은 위험하지만, manual tuning의 부담을 크게 줄일 수 있다.

## 5.4 Operational Automation

U-Flow는 operational deployment의 실용적 측면에 많은 주의를 기울였다. 단순히 알고리즘을 제안하는 것을 넘어, 실제 배포에서의 도전과제들을 다루었다.

**Deployment Pipeline**

U-Flow는 end-to-end deployment pipeline을 제공한다. Data ingestion, preprocessing, model training, threshold selection, inference, monitoring까지 모든 단계를 자동화한다.

```
Pipeline:
1. Data Collection: Gather normal training images
2. Preprocessing: Resize, normalize, augment
3. Feature Extraction: Pass through pretrained backbone
4. Flow Training: Train U-Net flow on features
5. Threshold Computation: Automatic threshold from training scores
6. Validation (optional): Verify on small validation set
7. Deployment: Export model and threshold
8. Inference: Real-time anomaly detection
9. Monitoring: Track performance metrics
10. Retraining: Periodic model updates
```

각 단계가 명확히 정의되고 자동화되어 있어, 비전문가도 배포할 수 있다. Configuration files로 설정을 관리하고, command-line interface로 실행한다.

**Robustness Mechanisms**

U-Flow는 여러 robustness mechanisms을 포함한다. 학습이 불안정하거나 실패하는 것을 방지한다.

Gradient clipping으로 exploding gradients를 방지한다. Norm이 임계값을 초과하면 clipping한다. 이는 학습 초기의 불안정성을 완화한다.

Early stopping이 자동으로 작동한다. Validation loss(또는 training loss)가 plateau에 도달하면 학습을 중단한다. Overfitting을 방지하고 학습 시간을 절약한다.

Checkpoint saving으로 최고 성능 모델을 유지한다. 매 epoch마다 validation performance를 평가하고 best model을 저장한다. 학습 중 divergence가 발생해도 복구할 수 있다.

Numerical stability checks가 있다. NaN이나 Inf가 발생하면 경고를 발생시키고 학습을 중단한다. 조기에 문제를 감지하여 디버깅을 용이하게 한다.

**Monitoring and Alerting**

배포 후 모니터링 시스템이 모델 성능을 추적한다. 주요 메트릭을 실시간으로 기록하고 이상 징후를 감지한다.

추론 통계를 수집한다. 정상/이상 판정 비율, 평균 이상 점수, 점수 분포의 변화를 모니터링한다. 급격한 변화는 distribution shift나 모델 degradation을 시사한다.

성능 메트릭을 주기적으로 평가한다. 가능하면 ground truth labels를 수집하여 정확도, 재현율, 정밀도를 계산한다. 성능이 임계값 이하로 떨어지면 알람을 발생시킨다.

Drift detection 알고리즘이 데이터 분포의 변화를 감지한다. 최근 데이터의 통계량을 학습 데이터와 비교한다. Significant shift가 감지되면 재학습을 권장한다.

**Continuous Learning**

U-Flow는 continuous learning을 지원한다. 새로운 정상 데이터가 수집되면 모델을 incrementally 업데이트한다. 전체 재학습보다 효율적이다.

Online learning은 새 샘플이 도착할 때마다 모델을 조금씩 업데이트한다. 그러나 normalizing flow는 online learning이 어렵다. 전체 분포를 다시 학습해야 한다.

대신 periodic retraining을 사용한다. 주기적으로(예: 매주, 매월) 축적된 새 데이터로 모델을 재학습한다. 이는 distribution drift에 적응하면서도 계산 비용을 관리 가능하게 한다.

Active learning으로 재학습의 효율성을 높인다. 모델이 uncertain한 샘플(이상 점수가 임계값 근처)을 우선적으로 레이블링 요청한다. 이들이 가장 informative하다.

**Error Handling**

Robust deployment는 다양한 오류 상황을 gracefully 처리해야 한다. U-Flow는 comprehensive error handling을 제공한다.

Input validation이 invalid 입력을 감지한다. 손상된 이미지, 잘못된 크기, 극단적인 픽셀 값 등을 확인한다. Invalid 입력은 거부하고 오류 메시지를 반환한다.

Fallback mechanisms이 모델 실패 시 대안을 제공한다. 예를 들어 flow evaluation이 실패하면 simpler baseline(예: PaDiM)로 fall back한다. 시스템이 완전히 중단되는 것을 방지한다.

Logging이 모든 오류와 경고를 기록한다. 타임스탬프, 입력 데이터, 시스템 상태와 함께 저장한다. 사후 분석과 디버깅을 위한 audit trail을 제공한다.

User notifications가 critical 이슈를 알린다. 시스템 관리자나 운영자에게 이메일, SMS, dashboard alert로 통지한다. 신속한 대응을 가능하게 한다.

**Practical Impact**

U-Flow의 operational focus는 실무 채택에 도움이 되었다. Automatic threshold와 robust deployment가 비전문가도 사용할 수 있게 했다. 그러나 FastFlow의 단순성과 PatchCore의 최고 성능 사이에서 명확한 우위를 확보하지 못했다.

성능이 약간 낮고(98.0-98.3% vs FastFlow 98.5%) 속도도 비슷하다. Operational features는 가치 있지만, 이것만으로 다른 방법보다 선택할 충분한 이유가 되지 않았다. 결국 채택은 제한적이었다.

그럼에도 U-Flow의 기여는 인정받아야 한다. Research community에 operational deployment의 중요성을 환기시켰다. 알고리즘만이 아니라 실용적 측면도 고려해야 함을 보여주었다. 후속 연구들은 이러한 관점을 더 많이 반영하고 있다.

Normalizing flow 패러다임은 CFlow에서 시작하여 FastFlow의 효율성 개선, CS-Flow의 multi-scale exploration, U-Flow의 operational focus로 진화했다. 각각 고유한 기여를 했지만, FastFlow가 실무에서 가장 성공적이었다. 성능-속도-메모리의 최적 균형과 단순성이 승리 요인이었다. 이는 deep learning 일반에도 적용되는 교훈이다.

# 6. Comprehensive Comparison

## 6.1 Flow Architecture Evolution

Normalizing flow 기반 이상 탐지의 발전은 2021-2022년의 짧은 기간 동안 급격히 진행되었다. 네 가지 주요 방법(CFlow, FastFlow, CS-Flow, U-Flow)은 각각 고유한 설계 철학과 trade-off를 보여준다.

**CFlow: The Pioneer**

CFlow는 normalizing flow를 이상 탐지에 성공적으로 적용한 최초의 방법이다. Position-conditional flow의 개념을 확립했다. 각 공간 위치마다 조건부 분포를 학습하여 이미지의 heterogeneity를 포착했다. Multi-scale processing으로 다양한 크기의 결함을 탐지했다.

CFlow의 설계는 표현력을 우선시했다. 각 위치에서 독립적인 고차원 flow를 실행하고, 여러 스케일을 모두 처리한다. 이는 복잡한 정상 분포를 잘 모델링하지만 계산 비용이 크다. 초기 연구로서 효율성보다 feasibility 증명이 목표였다.

학술적으로 CFlow는 패러다임을 열었다. Probabilistic modeling이 이상 탐지에서 competitive함을 보였다. Exact likelihood computation의 가치를 입증했다. 후속 연구들의 기반이 되었다.

**FastFlow: The Game Changer**

FastFlow는 CFlow의 효율성 문제를 정면으로 해결했다. 3D flow를 2D로 단순화하는 대담한 결정을 내렸다. 채널 독립성 가정으로 계산량을 10배 이상 줄였다. 놀랍게도 성능은 유지되거나 향상되었다.

FastFlow의 철학은 "simplicity wins"다. 불필요한 복잡성을 제거하고 본질에 집중했다. 2D convolutional structure가 공간적 coherence를 자연스럽게 포착한다는 통찰이 핵심이었다. 채널 간 명시적 모델링이 이상 탐지에 critical하지 않음을 발견했다.

실무적으로 FastFlow는 flow-based 방법을 viable하게 만들었다. 20-50ms 추론 속도와 200-500MB 메모리로 실제 배포가 가능해졌다. 성능-효율성 균형의 sweet spot을 찾았다. 현재 가장 널리 사용되는 flow 방법이다.

**CS-Flow: The Complexity Experiment**

CS-Flow는 반대 방향을 시도했다. 더 많은 정보 교환으로 성능을 높이려 했다. Cross-scale information fusion으로 multi-resolution context를 풍부하게 했다. 각 스케일이 다른 스케일의 정보를 활용하도록 설계했다.

CS-Flow의 가설은 "more information is better"였다. Low-level과 high-level features를 결합하면 더 robust한 탐지가 가능하다고 믿었다. 이론적으로 타당하지만 empirical results는 미미했다. 0-1%포인트 향상에 그쳤다.

CS-Flow의 교훈은 명확하다. Complexity는 자동으로 better performance를 보장하지 않는다. Diminishing returns가 빠르게 나타난다. Cost-benefit 분석이 필수적이다. 대부분의 경우 단순한 FastFlow가 우수하다.

**U-Flow: The Operational Focus**

U-Flow는 다른 관점을 취했다. 최고 성능보다 operational reliability를 우선했다. U-Net structure로 안정적인 multi-scale processing을 제공했다. Automatic threshold selection으로 배포를 용이하게 했다.

U-Flow의 기여는 주로 engineering이다. Algorithm innovation보다 practical deployment에 집중했다. Robust training, error handling, monitoring을 강조했다. Research와 industry 간 격차를 좁히려 했다.

그러나 U-Flow도 명확한 niche를 찾지 못했다. 성능이 FastFlow보다 약간 낮고 operational features만으로는 차별화가 부족했다. 좋은 의도였지만 시장에서 성공하지 못했다.

**Evolution Trajectory**

네 방법의 진화를 보면 흥미로운 패턴이 있다. CFlow → FastFlow는 simplification의 승리다. CS-Flow는 complexity의 한계를 보여준다. U-Flow는 practical considerations의 중요성을 강조한다.

전반적 추세는 efficiency로 향한다. 초기의 표현력 추구에서 실용성 중심으로 이동했다. FastFlow의 성공이 이를 증명한다. 미래 연구도 이 방향을 따를 것이다.

## 6.2 Performance Comparison

네 가지 flow 방법의 성능을 MVTec AD 벤치마크에서 정량적으로 비교한다.

**Overall Metrics**

이미지 레벨 AUROC 비교:
- CFlow: 98.3%
- FastFlow: 98.5%
- CS-Flow: 98.3-98.5%
- U-Flow: 98.0-98.3%

FastFlow가 근소하게 최고다. CFlow와 CS-Flow는 거의 동일하다. U-Flow는 약간 낮다. 그러나 차이는 모두 1%포인트 이내로 통계적으로 크지 않다.

픽셀 레벨 AUROC 비교:
- CFlow: 98.5%
- FastFlow: 98.6%
- CS-Flow: 98.5-98.7%
- U-Flow: 98.2-98.5%

픽셀 레벨에서도 유사한 패턴이다. FastFlow가 약간 우위고 나머지는 비슷하다. 결함 위치 특정 능력에서 큰 차이가 없다.

**Category-wise Analysis**

Texture 카테고리에서 모든 방법이 우수하다. 99% 이상의 AUROC를 일관되게 달성한다. Grid, leather, tile에서 거의 완벽한 분류를 보인다. Texture 패턴은 flow modeling에 매우 적합하다.

Object 카테고리에서 약간의 차이가 있다. FastFlow가 평균적으로 0.5-1%포인트 우위다. Bottle, cable, capsule에서 특히 강력하다. Simpler architecture가 overfitting을 방지하여 일반화가 더 나은 것으로 보인다.

Difficult 카테고리(hazelnut, screw, metal_nut)에서 모두 어려움을 겪는다. 96-98% 범위로 다른 카테고리보다 낮다. 방법 간 차이도 크지 않다. 이들은 근본적으로 어려워 알고리즘 개선만으로는 한계가 있다.

**Comparison with Other Paradigms**

Flow-based 방법을 다른 패러다임과 비교하면:
- PatchCore (memory-based): 99.1% - Flow보다 0.6-1.1%포인트 높다
- PaDiM (memory-based): 97.5% - Flow보다 0.5-1.0%포인트 낮다
- STFPM (knowledge distillation): 96.8% - Flow보다 1.5-1.7%포인트 낮다
- EfficientAD (knowledge distillation): 97.8% - Flow보다 0.2-0.7%포인트 낮다

Flow-based 방법은 중상위권에 위치한다. 최고(PatchCore)에는 미치지 못하지만 대부분보다 우수하다. 성능 면에서 competitive하다.

**Statistical Significance**

방법 간 성능 차이의 통계적 유의성을 평가해야 한다. 여러 random seeds로 실험을 반복하면 표준편차가 약 0.2-0.3%포인트다. 0.5%포인트 이상 차이는 유의미하고 그 이하는 noise일 수 있다.

FastFlow vs CFlow의 0.2%포인트 차이는 경계선이다. 일부 runs에서는 역전될 수 있다. CFlow vs CS-Flow는 차이가 거의 없어 실질적으로 동일하다. U-Flow vs others의 0.3-0.5%포인트는 약하게 유의미하다.

중요한 것은 모든 flow 방법이 98% 이상을 달성한다는 점이다. 실무에서 98%와 98.5%의 차이는 미미할 수 있다. 다른 요소(속도, 메모리, 배포 용이성)가 더 결정적일 수 있다.

**Failure Cases**

모든 flow 방법이 공통적으로 어려워하는 경우들이 있다. 첫째, 매우 미세한 결함이다. 픽셀 레벨에서 수 개에 불과한 차이는 CNN 특징에서 희석된다. Flow가 포착하기 어렵다.

둘째, 정상과 유사한 이상이다. 경계선 사례(borderline cases)에서 정상 분포와 약간만 벗어나면 탐지가 어렵다. Likelihood가 애매하여 false negative가 발생한다.

셋째, 새로운 유형의 이상이다. 학습 중 전혀 관찰되지 않은 결함 패턴은 예측하기 어렵다. Generalization의 한계다. Few-shot learning이나 meta-learning이 필요할 수 있다.

넷째, 복잡한 정상 변동이다. 조명, 포즈, 색상 변화가 크면 정상 분포가 넓어진다. 이상과의 경계가 모호해져 false positive가 증가한다.

## 6.3 Computational Analysis

Computational efficiency는 실무 배포에서 critical하다. 네 가지 flow 방법의 계산 비용을 상세히 분석한다.

**Inference Time Breakdown**

각 방법의 추론 시간을 구성 요소별로 분해한다.

CFlow (총 100-150ms):
- Backbone: 30-40ms (30%)
- Feature extraction & alignment: 10-15ms (10%)
- Flow evaluation (3 scales): 50-80ms (55%)
- Post-processing: 10-15ms (5%)

FastFlow (총 20-50ms):
- Backbone: 10-20ms (50%)
- Feature extraction: 2-5ms (10%)
- 2D Flow evaluation: 5-15ms (30%)
- Post-processing: 3-10ms (10%)

CS-Flow (총 120-180ms):
- Backbone: 30-40ms (25%)
- Multi-scale alignment: 15-25ms (15%)
- Cross-scale flow: 60-100ms (55%)
- Fusion & post-processing: 15-20ms (5%)

U-Flow (총 80-120ms):
- Backbone: 25-35ms (30%)
- U-Net encoder-decoder: 40-70ms (60%)
- Threshold & post-processing: 10-15ms (10%)

FastFlow가 압도적으로 빠르다. CFlow의 3-5배, CS-Flow의 4-6배, U-Flow의 2-4배다. 백본이 시간의 절반을 차지하므로, 백본 최적화가 모든 방법에 중요하다.

**Memory Usage**

모델 크기와 런타임 메모리를 비교한다.

Model Size (parameters):
- CFlow: 500MB-1GB (3 flows × large networks)
- FastFlow: 200-500MB (shared 2D flows)
- CS-Flow: 600MB-1.5GB (cross-scale connections)
- U-Flow: 400-800MB (U-Net structure)

Runtime Memory (inference):
- CFlow: 1-2GB (intermediate features × 3 scales)
- FastFlow: 300-600MB (single scale processing)
- CS-Flow: 1.5-2.5GB (cross-scale alignment buffers)
- U-Flow: 800-1.5GB (encoder-decoder activations)

FastFlow가 메모리도 가장 효율적이다. Single-scale과 parameter sharing이 핵심이다. CS-Flow가 가장 많이 사용한다. Cross-scale processing의 대가다.

**Training Time**

학습 시간도 중요한 고려사항이다.

Per Category (single GPU):
- CFlow: 3-8 hours
- FastFlow: 1-3 hours
- CS-Flow: 4-10 hours
- U-Flow: 2-4 hours

FastFlow와 U-Flow가 상대적으로 빠르다. CFlow와 CS-Flow는 multi-scale 학습이 시간이 걸린다. 모든 방법이 memory-based(수 분)보다는 훨씬 느리다.

Full Dataset (15 categories, 4 GPUs):
- CFlow: 1-2 days
- FastFlow: 6-12 hours
- CS-Flow: 1.5-3 days
- U-Flow: 8-16 hours

병렬화로 총 시간을 줄일 수 있지만 여전히 상당하다. Rapid prototyping에는 제약이 있다. Memory-based 방법(1-2 hours total)이 이 측면에서 우위다.

**Scalability Analysis**

다양한 차원에서 확장성을 분석한다.

Image Resolution: 고해상도 이미지에서 계산량이 제곱에 비례하여 증가한다. $512 \times 512$는 $256 \times 256$보다 4배 느리다. FastFlow가 상대적으로 덜 영향받는다. 2D flow가 spatial dimensions에 더 효율적이다.

Number of Categories: 카테고리 수 증가에 대해 모든 방법이 선형적으로 scaling된다. 각 카테고리마다 독립적인 모델이 필요하다. 메모리는 선형 증가하고, 추론 시간은 카테고리 식별 후 해당 모델만 사용하므로 일정하다.

Batch Size: 배치 처리로 throughput을 크게 향상시킬 수 있다. GPU 활용률이 높아진다. FastFlow가 가장 큰 이득을 본다(5-10배). CFlow와 CS-Flow는 메모리 제약으로 배치 크기가 제한된다.

**Hardware Dependency**

서로 다른 하드웨어에서의 성능을 비교한다.

GPU (RTX 3090):
- FastFlow: 20-30ms (best)
- U-Flow: 80-100ms
- CFlow: 100-130ms
- CS-Flow: 120-150ms

GPU (RTX 2080):
- FastFlow: 30-50ms
- U-Flow: 100-120ms
- CFlow: 130-150ms
- CS-Flow: 150-180ms

CPU (Intel i9):
- FastFlow: 200-400ms (reasonable)
- U-Flow: 800-1200ms
- CFlow: 1000-1500ms
- CS-Flow: 1500-2000ms

GPU에서 FastFlow의 우위가 명확하다. CPU에서는 모든 방법이 느리지만 FastFlow만 sub-second다. Jetson Xavier(edge GPU)에서도 FastFlow가 유일하게 실시간에 근접한다(60-100ms).

## 6.4 Design Trade-offs

네 가지 flow 방법의 설계 결정과 그에 따른 trade-off를 분석한다.

**Expressiveness vs Efficiency**

CFlow와 CS-Flow는 표현력을 우선시했다. 복잡한 분포를 정확히 모델링하려 했다. Position-conditional, multi-scale, cross-scale information을 모두 활용했다. 결과적으로 계산 비용이 크다.

FastFlow는 효율성을 우선시했다. 채널 독립성과 2D flow로 단순화했다. 표현력을 일부 희생했지만 실제로는 성능 저하가 없었다. 오히려 regularization 효과로 일반화가 개선되었다.

Trade-off: 대부분의 응용에서 FastFlow의 선택이 옳다. 충분한 표현력과 압도적 효율성을 제공한다. 극도로 복잡한 정상 분포를 가진 rare cases에서만 CFlow가 필요할 수 있다.

**Multi-scale vs Single-scale**

CFlow, CS-Flow, U-Flow는 multi-scale을 사용했다. 다양한 크기의 결함을 포괄적으로 탐지하려 했다. 이는 직관적으로 타당하고 일부 카테고리에서 도움이 되었다.

FastFlow는 single-scale(layer3)을 선택했다. 중간 스케일이 대부분의 결함에 충분하다고 판단했다. 실험적으로 multi-scale의 추가 이득이 0.5%포인트 미만이었다. 비용 대비 이득이 불리했다.

Trade-off: Single-scale의 simplicity가 승리했다. Multi-scale의 이론적 이점이 실무적 이득으로 이어지지 않았다. 예외적으로 매우 다양한 결함 크기를 가진 데이터셋에서만 multi-scale을 고려한다.

**Automatic vs Manual Threshold**

U-Flow는 automatic threshold selection을 제안했다. 배포를 용이하게 하고 주관성을 제거하려 했다. Validation set 없이도 작동한다. 이는 실용적으로 매력적이다.

다른 방법들은 manual threshold selection을 가정했다. Validation set에서 ROC 분석으로 최적값을 찾는다. 더 정확하지만 추가 데이터와 노력이 필요하다.

Trade-off: Hybrid 접근이 최선이다. Automatic threshold를 starting point로 사용하고, 소량의 validation data로 fine-tuning한다. 완전 자동은 위험하고 완전 수동은 비효율적이다.

**Architectural Complexity**

U-Flow의 U-Net structure는 elegance를 제공했다. Encoder-decoder with skip connections는 검증된 architecture다. Multi-scale을 통합된 방식으로 처리한다. 그러나 구현이 복잡하고 디버깅이 어렵다.

FastFlow의 flat structure는 단순하다. 2D flow를 각 채널에 독립적으로 적용한다. 이해하기 쉽고 구현이 straightforward하다. 그러나 architectural novelty가 부족하다.

Trade-off: Simplicity가 대부분의 경우 선호된다. 특히 실무 배포에서 유지보수와 디버깅이 중요하다. Complex architecture는 학술 연구에서만 가치가 있다.

**Training Stability**

모든 flow 방법이 GAN보다 훨씬 안정적이다. Maximum likelihood objective가 well-behaved하다. 그러나 방법 간에도 차이가 있다.

FastFlow가 가장 안정적이다. 단순한 구조와 적은 파라미터가 overfitting을 방지한다. Loss curve가 smooth하고 수렴이 빠르다. Hyperparameter에 덜 민감하다.

CS-Flow와 U-Flow는 약간 덜 안정적이다. 복잡한 architecture와 많은 파라미터로 때때로 training issues가 발생한다. Gradient clipping과 careful initialization이 필요하다.

Trade-off: Stability는 practical deployment에서 매우 중요하다. 재현성과 신뢰성을 보장한다. FastFlow의 안정성이 또 다른 장점이다.

# 7. Practical Application Guide

## 7.1 Model Selection by Scenario

실무에서 어떤 flow 방법을 선택할지는 구체적인 요구사항과 제약에 따라 달라진다. 시나리오별 권장사항을 제시한다.

**Scenario 1: High Accuracy Priority**

최고 정확도가 필수적인 경우(critical safety components, medical devices)다. 0.5%포인트 차이도 중요하다. False negative가 치명적인 비용을 초래한다.

권장: FastFlow 또는 CFlow. FastFlow는 98.5%로 약간 우위고 효율적이다. CFlow는 98.3%지만 일부 복잡한 카테고리에서 더 나을 수 있다. 둘 다 시도하고 validation set에서 비교한다.

대안: Flow보다 PatchCore(99.1%)가 더 정확하다. Flow의 generative capability가 필요 없다면 PatchCore를 우선 고려한다.

**Scenario 2: Real-time Processing**

실시간 처리가 필요한 경우(초당 30+ 이미지)다. 고속 생산 라인이나 interactive application이다. Latency가 critical하다.

권장: FastFlow with lightweight backbone. ResNet18이나 MobileNetV2를 사용하면 10-30ms를 달성한다. 배치 처리로 throughput을 더 높인다.

대안: EfficientAD(1-5ms)가 훨씬 빠르다. 성능이 약간 낮지만(97.8%) 실시간이 절대 요구사항이면 최선이다.

**Scenario 3: Edge Deployment**

엣지 디바이스(Jetson, Raspberry Pi)에서 실행해야 하는 경우다. 메모리와 연산이 제한적이다. 배터리 수명도 고려해야 한다.

권장: FastFlow with aggressive optimization. MobileNetV2 백본, FP16 quantization, TensorRT optimization을 모두 적용한다. 카테고리당 50-100MB로 줄인다.

대안: Memory-based 방법이 더 가벼울 수 있다. PatchCore with 0.5% coreset은 1-2MB다. 엣지에서 k-NN 탐색이 효율적이다.

**Scenario 4: Multi-category System**

수십 개 이상의 카테고리를 동시에 배포하는 경우다. 다품종 생산 라인이나 범용 검사 시스템이다. 메모리와 관리 복잡도가 challenge다.

권장: FastFlow with shared backbone. 백본 가중치는 모든 카테고리가 공유하고 flow만 개별적으로 유지한다. 카테고리당 50MB 추가만 필요하다.

대안: Foundation model 기반 multi-class 방법(Dinomaly)을 고려한다. 단일 모델로 모든 카테고리를 처리한다. 관리가 훨씬 간단하다.

**Scenario 5: Frequent Model Updates**

제품이나 공정이 자주 변경되어 재학습이 빈번한 경우다. 신제품 출시나 공정 개선이 잦다. 학습 시간과 용이성이 중요하다.

권장: FastFlow. 1-3시간 학습으로 빠르게 업데이트 가능하다. 안정적인 학습으로 실패 위험이 적다. Automated pipeline 구축이 용이하다.

대안: Memory-based 방법(PatchCore)이 학습이 수 분이다. 재학습이 매우 빈번하면(일별, 주별) 더 적합할 수 있다.

**Scenario 6: Research and Benchmarking**

알고리즘 개발이나 학술 연구 목적이다. 다양한 방법을 비교하고 새로운 아이디어를 실험한다. 성능과 novelty가 중요하다.

권장: 모든 flow 방법을 구현하고 비교한다. CFlow로 baseline을 설정하고, FastFlow로 효율성을 평가하며, CS-Flow와 U-Flow로 architectural variations를 탐구한다.

추가: 최신 foundation model 기반 방법도 포함한다. 포괄적인 비교가 연구의 가치를 높인다.

**Scenario 7: Prototype and Feasibility Study**

새로운 응용 영역이나 제품에서 이상 탐지의 feasibility를 빠르게 검증하는 경우다. 시간과 리소스가 제한적이다. Quick results가 필요하다.

권장: FastFlow. 구현이 간단하고 학습이 빠르다. 1-2일 안에 end-to-end prototype을 구축할 수 있다. 결과가 promising하면 더 정교한 방법으로 확장한다.

대안: Pre-trained 모델을 활용한다. Transfer learning이나 few-shot learning으로 최소 데이터로 시작한다.

**Decision Matrix**

| Requirement | FastFlow | CFlow | CS-Flow | U-Flow |
|-------------|----------|-------|---------|---------|
| Highest Accuracy | ★★★★ | ★★★★ | ★★★★ | ★★★ |
| Real-time Speed | ★★★★★ | ★★ | ★ | ★★ |
| Low Memory | ★★★★★ | ★★ | ★ | ★★★ |
| Easy Deployment | ★★★★★ | ★★★ | ★★ | ★★★★ |
| Multi-category | ★★★★★ | ★★★ | ★★ | ★★★ |
| Quick Training | ★★★★ | ★★ | ★ | ★★★ |
| Stability | ★★★★★ | ★★★★ | ★★★ | ★★★★ |

FastFlow가 대부분의 기준에서 우수하다. 일반적인 권장사항이다. 특수한 요구사항이 있을 때만 다른 방법을 고려한다.

## 7.2 Hyperparameter Tuning

Flow 방법의 주요 hyperparameters와 튜닝 전략을 설명한다.

**Backbone Selection**

가장 중요한 선택이다. 성능과 속도의 직접적인 trade-off다.

ResNet18: 빠르고 가볍다(특징 추출 10-20ms). 대부분의 경우 충분한 성능(96-97% AUROC). 기본 선택으로 시작한다.

ResNet50: 균형잡혔다(특징 추출 20-30ms). 1%포인트 성능 향상(97-98%). ResNet18이 부족하면 업그레이드한다.

Wide ResNet-50: 최고 성능(특징 추출 30-40ms). 1-2%포인트 추가 향상(98-99%). Critical accuracy가 필요할 때만 사용한다.

MobileNetV2: 엣지용(특징 추출 5-10ms). 2-3%포인트 성능 저하(94-96%). 메모리와 속도가 극도로 제한적일 때 선택한다.

튜닝 전략: ResNet18로 시작한다. Validation AUROC가 목표에 미달하면 ResNet50으로 업그레이드한다. 여전히 부족하면 Wide ResNet-50을 시도한다. 속도가 문제면 MobileNetV2로 다운그레이드한다.

**Feature Layer**

어떤 CNN 층의 특징을 사용할지 선택한다.

Layer2 ($64 \times 64$): High-resolution, low-level. Small texture defects에 유리. 계산 비용이 크다.

Layer3 ($32 \times 32$): Balanced. 대부분의 결함에 충분하다. FastFlow의 기본 선택. 권장.

Layer4 ($16 \times 16$): Low-resolution, high-level. Large structural defects에 유리. Small defects를 놓칠 수 있다.

Multi-scale: Layer2+3+4를 모두 사용한다. 0.5-1%포인트 향상하지만 3배 느리다. Critical accuracy가 필요할 때만 고려한다.

튜닝 전략: Layer3 single-scale로 시작한다. 특정 결함 유형(매우 작거나 큰)이 놓치면 해당 layer를 추가한다. 대부분의 경우 layer3만으로 충분하다.

**Flow Depth**

Coupling layers의 수를 결정한다.

4 layers: Minimal. 매우 빠르지만 표현력 부족. Simple textures에만 적합.

6 layers: Balanced. 대부분의 경우 충분하다. FastFlow의 기본값. 권장.

8 layers: Deep. 복잡한 분포에 유리. 약간 느리고 overfitting 위험.

12 layers: Very deep. CFlow의 기본값. 추가 이득이 미미하다. 권장하지 않는다.

튜닝 전략: 6 layers로 시작한다. Underfitting 징후(높은 training loss)가 있으면 8로 늘린다. Overfitting 징후(validation loss 증가)가 있으면 4로 줄인다.

**Learning Rate**

학습 속도를 조절한다.

$10^{-4}$: Slow but stable. 안전한 선택. 수렴이 느릴 수 있다.

$10^{-3}$: Balanced. FastFlow의 기본값. 대부분의 경우 최적. 권장.

$10^{-2}$: Fast but risky. 불안정할 수 있다. 빠른 실험에만 사용.

Scheduling: Cosine annealing을 권장한다. 학습 후반부에 learning rate를 점진적으로 줄여 fine-tuning한다.

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
```

튜닝 전략: $10^{-3}$으로 시작한다. Loss가 oscillate하면 $10^{-4}$로 줄인다. 수렴이 너무 느리면 $10^{-2}$로 늘리되 주의깊게 모니터링한다.

**Batch Size**

메모리와 학습 안정성의 trade-off다.

8: Small. 메모리 제약 환경. Noisy gradients로 불안정할 수 있다.

16: Balanced. 대부분의 경우 적합. FastFlow 기본값. 권장.

32: Large. Stable gradients. GPU 메모리가 충분하면 선호된다.

Gradient accumulation으로 effective batch size를 늘릴 수 있다. 메모리 제약이 있지만 큰 배치가 필요할 때 유용하다.

**Epochs**

학습 기간을 결정한다.

100: Minimum. 빠른 실험. Undertraining 위험.

200: Balanced. 대부분의 경우 충분. FastFlow 기본값. 권장.

500: Long. 복잡한 데이터. Overfitting 위험. Early stopping 필수.

Early stopping을 사용하면 epochs를 크게 설정하고 자동으로 최적 지점에서 멈춘다. Patience 20-50 epochs를 권장한다.

**Tuning Workflow**

실무적인 튜닝 절차는 다음과 같다.

1. 기본 설정으로 시작: FastFlow + ResNet18 + layer3 + 6 coupling layers + lr=1e-3 + batch=16 + epochs=200

2. 단일 카테고리에서 baseline 성능 측정: Validation AUROC를 기록한다.

3. 목표에 도달하면 튜닝 종료: 추가 tuning은 불필요하다.

4. 도달하지 못하면 백본 업그레이드: ResNet18 → ResNet50 → Wide ResNet-50

5. 여전히 부족하면 flow depth 증가: 6 → 8 layers

6. 특정 결함 유형이 문제면 layer 추가: Layer3 → layer2+3 또는 layer3+4

7. Learning rate나 epochs 조정: Loss curve를 보고 판단

8. 여러 카테고리에서 검증: 선택된 설정이 일반적으로 잘 작동하는지 확인

Exhaustive grid search는 비효율적이다. 위의 순차적 접근이 더 실용적이다. 대부분의 경우 기본 설정 또는 백본 업그레이드만으로 충분하다.

## 7.3 Training Strategies

효과적인 학습을 위한 전략들을 제시한다.

**Data Preparation**

고품질 학습 데이터가 성공의 핵심이다.

정상 샘플 수집: 최소 50-100장, 이상적으로 200-400장을 목표로 한다. 다양한 정상 변동(조명, 포즈, 색상)을 포함한다. 명백한 불량품이 섞이지 않도록 엄격히 검증한다.

데이터 정제: 학습 데이터를 여러 검사자가 독립적으로 확인한다. 의심스러운 샘플은 제외한다. Outlier detection으로 이상치를 찾는다.

Train/Validation Split: 80/20 또는 70/30으로 나눈다. Stratified split으로 다양한 변동이 양쪽에 포함되도록 한다. Validation set은 hyperparameter tuning과 early stopping에 사용한다.

Data Augmentation: Minimal하게 사용한다. Random horizontal flip과 slight rotation(±5도)만 적용한다. 과도한 augmentation은 정상 분포를 왜곡한다. Color jitter나 aggressive cropping은 피한다.

**Training Loop**

표준 학습 루프를 구현한다.

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        features = backbone(batch)
        z, log_det = flow(features)
        log_p_z = -0.5 * (z**2).sum()
        loss = -(log_p_z - log_det).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    scheduler.step()
    
    if epoch % eval_frequency == 0:
        val_loss = evaluate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, epoch)
```

Gradient clipping으로 exploding gradients를 방지한다. Max norm 1.0-5.0을 사용한다. Learning rate scheduling으로 수렴을 개선한다. Checkpoint saving으로 best model을 유지한다.

**Loss Monitoring**

학습 진행을 모니터링한다.

Training loss가 단조 감소해야 한다. Oscillation이나 증가는 문제 징후다. Learning rate를 줄이거나 batch size를 늘린다.

Validation loss를 주기적으로 평가한다. Training loss보다 약간 높은 것이 정상이다. 격차가 크면 overfitting이다. Regularization을 강화하거나 early stopping을 적용한다.

Loss magnitude의 절대값은 의미가 제한적이다. Negative log-likelihood는 unbounded이므로 -100이나 -1000이 모두 가능하다. 중요한 것은 감소 추세와 validation performance다.

**Regularization Techniques**

Overfitting을 방지하는 기법들이다.

Weight decay: L2 regularization을 optimizer에 추가한다. $10^{-4}$ - $10^{-5}$를 사용한다. 너무 크면 underfitting된다.

Dropout: Coupling networks에 dropout(0.1-0.3)을 추가할 수 있다. 그러나 flow의 invertibility에 주의해야 한다. Training과 inference에서 동일하게 적용한다.

Early stopping: Validation loss가 patience epochs 동안 개선되지 않으면 학습을 중단한다. Patience 20-50을 권장한다.

Data augmentation: 앞서 언급했듯이 minimal하게 사용한다. Overfitting이 심하면 약간 늘릴 수 있다.

**Debugging Failed Training**

학습이 실패하는 경우 대응 방법이다.

Loss가 감소하지 않는 경우: Learning rate가 너무 낮을 수 있다. 10배 늘려본다($10^{-3}$ → $10^{-2}$). 또는 initialization 문제일 수 있다. 다른 random seed로 재시작한다.

Loss가 NaN이 되는 경우: Learning rate가 너무 높거나 numerical instability다. Learning rate를 10배 줄인다($10^{-3}$ → $10^{-4}$). Gradient clipping을 더 강하게 한다(max_norm=0.5). Log 계산에서 작은 epsilon($10^{-6}$)을 추가한다.

Overfitting이 심한 경우: Training loss는 낮지만 validation loss는 높다. Weight decay를 늘린다($10^{-5}$ → $10^{-4}$). Dropout을 추가한다. Flow depth를 줄인다(8 → 6 layers). 학습 데이터를 더 수집한다.

수렴이 너무 느린 경우: 수백 epochs 후에도 loss가 계속 감소한다. Epochs를 늘린다(200 → 500). Learning rate를 늘린다. 또는 모델 capacity가 부족할 수 있다. Flow depth를 늘리거나 백본을 업그레이드한다.

## 7.4 Deployment Considerations

실제 배포 시 고려해야 할 실무적 측면들이다.

**Model Export and Optimization**

학습된 모델을 배포용으로 최적화한다.

PyTorch to ONNX: 모델을 ONNX 형식으로 변환한다. 다양한 inference engines와 호환된다.

```python
torch.onnx.export(
    model, 
    dummy_input, 
    "fastflow.onnx",
    opset_version=13,
    input_names=['input'],
    output_names=['anomaly_map', 'anomaly_score'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

TensorRT Optimization: NVIDIA GPU에서 추론을 2-3배 가속한다. FP16 precision과 kernel fusion을 자동으로 적용한다.

Quantization: INT8 quantization으로 모델 크기를 1/4로 줄이고 속도를 2배 향상시킨다. Calibration dataset이 필요하다. 정확도 저하(0.5-1%포인트)를 모니터링한다.

**Threshold Configuration**

임계값 설정과 관리 전략이다.

Initial Threshold: 학습 후 정상 샘플의 이상 점수 분포에서 95th 또는 99th percentile을 선택한다. Conservative starting point다.

Validation Tuning: 소량의 validation set(정상 + 이상)에서 ROC 분석을 수행한다. 목표 재현율(예: 95%, 99%)에 대응하는 임계값을 찾는다.

Dynamic Adjustment: 배포 후 실제 데이터의 분포를 모니터링한다. 정상 점수의 평균이 drift하면 임계값을 조정한다. Automated threshold adaptation을 고려한다.

Per-category Threshold: 각 카테고리마다 다른 임계값을 설정한다. 일부는 더 민감하게(낮은 threshold), 일부는 덜 민감하게(높은 threshold) 조정한다.

**Inference Pipeline**

배포된 시스템의 추론 파이프라인을 설계한다.

```python
def inference_pipeline(image_path, model, threshold):
    # 1. Load and preprocess
    image = load_image(image_path)
    image = preprocess(image)  # Resize, normalize
    
    # 2. Feature extraction
    with torch.no_grad():
        features = backbone(image)
    
    # 3. Flow evaluation
    with torch.no_grad():
        z, log_det = flow(features)
        log_p_z = -0.5 * (z**2).sum(dim=[1,2,3])
        anomaly_score = -(log_p_z - log_det)
    
    # 4. Anomaly map generation
    anomaly_map = generate_anomaly_map(z, log_det)
    anomaly_map = resize_to_original(anomaly_map, image.shape)
    
    # 5. Decision
    is_anomaly = anomaly_score > threshold
    
    # 6. Post-processing
    result = {
        'is_anomaly': is_anomaly,
        'anomaly_score': float(anomaly_score),
        'anomaly_map': anomaly_map,
        'timestamp': datetime.now()
    }
    
    return result
```

Error handling을 추가한다. Invalid inputs, model failures, out-of-memory를 gracefully 처리한다. Logging으로 모든 추론을 기록한다.

**Monitoring and Maintenance**

배포 후 시스템을 모니터링하고 유지보수한다.

Performance Metrics: 정상/이상 판정 비율을 추적한다. 급격한 변화는 문제 징후다. 평균 이상 점수의 추세를 모니터링한다. Drift를 조기에 감지한다.

False Positive/Negative Tracking: 오탐과 미탐 사례를 수집한다. 패턴을 분석하여 모델 약점을 파악한다. 정기적으로 재학습 데이터에 추가한다.

Retraining Schedule: 주기적으로 재학습한다(월별, 분기별). 새로운 정상 데이터와 발견된 이상 사례로 업데이트한다. A/B 테스팅으로 새 모델을 검증한 후 배포한다.

Alert System: Critical issues(성능 저하, 시스템 오류)가 발생하면 알림을 보낸다. Dashboard로 실시간 메트릭을 시각화한다. Anomaly examples를 샘플링하여 정기적으로 검토한다.

**Scaling Strategies**

시스템을 확장하는 전략들이다.

Horizontal Scaling: 여러 인스턴스를 병렬로 실행한다. Load balancer로 요청을 분산한다. 각 인스턴스는 독립적으로 추론을 수행한다. Throughput을 선형적으로 증가시킬 수 있다.

Model Serving: TorchServe, TensorFlow Serving, Triton Inference Server를 사용한다. Professional model serving infrastructure를 제공한다. Auto-scaling, health checks, metrics collection이 내장되어 있다.

Batch Processing: 실시간이 필수가 아니면 배치로 처리한다. 여러 이미지를 모아 한 번에 추론한다. GPU 활용률이 크게 향상된다. Latency는 증가하지만 throughput이 10배 이상 증가한다.

Edge-Cloud Hybrid: 엣지에서 1차 필터링을 수행한다. 명백한 정상은 통과시키고 의심 샘플만 클라우드로 보낸다. 네트워크 대역폭을 절약하고 대부분의 샘플을 빠르게 처리한다.

Normalizing flow 기반 이상 탐지는 성숙한 패러다임이다. 특히 FastFlow는 성능-효율성 균형으로 실무 배포에 적합하다. 적절한 모델 선택, 신중한 hyperparameter tuning, robust deployment pipeline으로 대부분의 산업 응용에서 탁월한 결과를 얻을 수 있다. Memory-based 방법의 최고 정확도와 knowledge distillation의 실시간 속도 사이의 중간 지점을 제공한다. 확률적 해석 가능성과 생성 능력은 추가적인 장점이다.

# 8. Research Insights

## 8.1 Why FastFlow Succeeded

FastFlow의 성공은 deep learning 연구에서 중요한 교훈을 제공한다. 더 단순한 모델이 복잡한 모델을 능가한 사례로, 여러 요인이 복합적으로 작용했다.

**Appropriate Inductive Bias**

FastFlow의 채널 독립성 가정은 강력한 inductive bias였다. 이는 임의의 가정이 아니라 CNN 특징의 본질을 반영했다. 사전 학습된 CNN의 각 채널은 이미 특정 시각적 개념을 인코딩한다. 한 채널은 edges를, 다른 채널은 textures를, 또 다른 채널은 shapes를 포착한다.

이러한 high-level features는 상대적으로 독립적이다. Edge 정보와 texture 정보는 서로 다른 aspect를 표현한다. 물론 완전히 독립적이지는 않지만, 강한 상관관계도 없다. 채널 독립성은 이러한 특성을 존중하는 가정이다.

대조적으로 CFlow는 모든 채널을 함께 모델링하려 했다. 이는 더 유연하지만 불필요한 복잡성을 초래했다. 채널 간 약한 상관관계까지 정확히 모델링하려 시도했다. 제한된 데이터에서 이는 overfitting으로 이어졌다. FastFlow는 중요한 것(채널 내 공간적 구조)에 집중하고 덜 중요한 것(채널 간 약한 의존성)을 무시했다.

Inductive bias의 효과는 sample efficiency로 나타났다. 같은 양의 학습 데이터(200-400 이미지)로 FastFlow가 더 나은 일반화를 달성했다. 모델이 학습해야 할 자유도가 적어 데이터 효율성이 높았다. 이상 탐지에서 데이터가 항상 부족하므로 이는 critical한 장점이다.

**Regularization Through Simplification**

FastFlow의 단순화는 명시적 regularization으로 작동했다. 복잡한 모델은 training data의 모든 details를 학습할 수 있다. Noise, outliers, spurious correlations까지 fit한다. 단순한 모델은 본질적인 패턴만 포착할 수 있다. 이는 일종의 implicit regularization이다.

구체적으로 FastFlow는 $c$개의 독립적인 2D 분포를 학습한다. 각 채널은 공간적 구조만 모델링한다. CFlow는 $h \times w$개의 $c$차원 분포를 학습한다. 각 위치는 채널 간 상관관계까지 모델링한다. 후자가 훨씬 많은 자유도를 가지므로 overfitting 위험이 크다.

실험적으로 이는 validation performance에서 나타났다. CFlow의 training loss가 FastFlow보다 낮았지만 validation loss는 비슷하거나 높았다. 이는 overfitting의 전형적인 징후다. FastFlow는 training과 validation의 격차가 작아 일반화가 더 나았다.

또한 FastFlow는 다양한 카테고리에서 일관된 성능을 보였다. CFlow는 일부 카테고리에서 매우 좋고 다른 카테고리에서는 보통이었다. 이는 CFlow가 특정 데이터의 peculiarities를 학습했음을 시사한다. FastFlow의 단순성이 robust하고 consistent한 성능을 제공했다.

**Spatial Structure Exploitation**

FastFlow의 2D flow는 공간적 구조를 명시적으로 모델링한다. Convolutional coupling networks가 이웃 픽셀 간 관계를 자연스럽게 포착한다. 결함은 종종 spatially coherent하다. Crack은 연속된 선이고, contamination은 연결된 영역이다. 2D flow가 이러한 구조를 directly 모델링한다.

CFlow의 position-conditional 3D flow는 각 위치를 독립적으로 처리한다. 물론 position embedding이 인접 위치의 유사성을 학습할 수 있지만 간접적이다. 공간적 coherence가 명시적으로 강제되지 않는다. FastFlow는 convolutional structure를 통해 자동으로 spatial smoothness를 학습한다.

Translation equivariance도 중요한 특성이다. 결함의 위치가 바뀌어도 동일하게 탐지되어야 한다. 2D convolution은 본질적으로 translation equivariant하다. Position-conditional approach는 위치 정보를 explicitly encoding하여 이를 깨뜨릴 수 있다. FastFlow의 architecture가 이 중요한 속성을 보존했다.

실험적으로 FastFlow의 anomaly maps가 더 sharp하고 spatially coherent했다. 결함 영역의 경계가 명확하고 내부가 균일하게 강조되었다. CFlow는 때때로 fragmented하거나 noisy한 activation을 보였다. 이는 spatial structure modeling의 우위를 시각적으로 입증한다.

**Computational Efficiency Advantage**

효율성 자체가 성능 향상에 기여했을 수 있다. 빠른 모델은 더 많은 실험을 가능하게 한다. Hyperparameter tuning, architecture search, data augmentation 실험을 빠르게 수행할 수 있다. 이는 최적 설정을 찾을 가능성을 높인다.

FastFlow의 1-3시간 학습은 일일 수십 번의 실험을 허용한다. CFlow의 3-8시간은 일일 수 번으로 제한된다. Iteration speed가 빠르면 더 많은 아이디어를 시도하고 최선을 선택할 수 있다. 이는 최종 성능에 간접적으로 기여한다.

또한 효율성은 더 큰 ensembles를 가능하게 한다. 여러 독립적인 FastFlow 모델을 학습하고 평균을 취하면 성능이 향상된다. 같은 시간과 리소스로 CFlow는 단일 모델만 학습할 수 있다. Ensemble FastFlow가 single CFlow보다 나을 수 있다.

배포 측면에서도 효율성이 성능에 영향을 미친다. 빠른 추론은 더 높은 throughput을 제공한다. 생산 라인에서 모든 제품을 검사할 수 있다면 recall이 향상된다. 느린 모델은 샘플링 검사만 가능하여 일부 불량을 놓칠 수 있다.

**Occam's Razor Principle**

근본적으로 FastFlow의 성공은 Occam's razor를 따랐다. 더 단순한 설명이 더 복잡한 것보다 자주 옳다. 이상 탐지의 목표는 정상 분포를 학습하는 것이다. 이 분포가 실제로 얼마나 복잡한가?

사전 학습된 CNN 특징 공간은 이미 상당히 structured되어 있다. ImageNet에서 학습된 representations는 일반적인 시각적 패턴을 효율적으로 인코딩한다. 고차원 원본 이미지 공간의 복잡성이 저차원 feature space에서는 크게 감소한다.

MVTec AD의 정상 샘플들은 상당히 uniform하다. 같은 제품의 정상 이미지들은 매우 유사하다. 변동이 제한적이다. 이러한 분포를 모델링하는 데 과도하게 복잡한 모델이 필요하지 않다. 단순한 모델로도 충분하다.

FastFlow는 필요한 만큼만 복잡하다. 채널 내 spatial dependencies를 모델링하지만 채널 간 complex interactions는 무시한다. 이는 대부분의 경우 충분했다. CFlow는 필요 이상으로 복잡했다. 추가 complexity가 추가 performance로 이어지지 않았다.

**Alignment with Practical Needs**

FastFlow의 설계가 실무 요구사항과 잘 맞았다. 산업 배포에서는 최고 정확도보다 균형이 중요하다. 합리적인 정확도, 빠른 속도, 낮은 메모리, 쉬운 배포가 모두 필요하다. FastFlow가 이 모든 기준을 만족했다.

CFlow와 CS-Flow는 학술적 관점에서 설계되었다. 최고 성능을 추구하고 architectural novelty를 강조했다. 그러나 실무에서는 98.3%와 98.5%의 차이가 미미할 수 있다. 대신 100ms vs 30ms의 속도 차이나 1GB vs 300MB의 메모리 차이가 훨씬 중요하다.

FastFlow의 단순성은 유지보수와 디버깅을 용이하게 했다. 구현이 straightforward하고 이해하기 쉽다. 문제가 발생했을 때 원인을 파악하기 쉽다. 복잡한 모델은 black box처럼 작동하여 troubleshooting이 어렵다.

또한 FastFlow는 다양한 백본과 호환된다. ResNet, EfficientNet, MobileNet 등 어떤 backbone과도 결합할 수 있다. CFlow의 position embedding과 complex conditioning은 특정 architecture에 더 dependent하다. Flexibility가 adoption을 촉진했다.

**Community Adoption and Feedback**

FastFlow의 성공은 self-reinforcing cycle을 만들었다. 초기 좋은 결과가 많은 연구자와 실무자들의 채택을 이끌었다. 이들의 피드백과 개선이 다시 FastFlow를 더 좋게 만들었다. Community effect가 중요했다.

GitHub에서 FastFlow 구현들이 많이 공유되었다. 다양한 프레임워크(PyTorch, TensorFlow)와 최적화(TensorRT, ONNX)가 제공되었다. 이는 진입 장벽을 낮춰 더 많은 사용을 촉진했다. CFlow는 원 논문의 구현만 있어 접근성이 떨어졌다.

실무 사례들이 축적되었다. 다양한 산업(제조, 의료, 보안)에서 성공 사례가 보고되었다. 이는 FastFlow의 일반성과 robustness를 입증했다. Positive feedback loop가 형성되어 dominant method가 되었다.

학술 연구에서도 FastFlow가 baseline으로 자주 사용된다. 새로운 방법을 제안할 때 FastFlow와 비교하는 것이 표준이 되었다. 이는 FastFlow의 지위를 더욱 공고히 했다. Network effect가 작동했다.

**Lessons for Deep Learning Research**

FastFlow의 성공에서 얻는 교훈은 broader하다. 첫째, complexity는 자동으로 better performance를 보장하지 않는다. Appropriate simplification이 오히려 유리할 수 있다. Especially with limited data, simpler models generalize better.

둘째, inductive bias의 중요성이다. Domain knowledge를 architecture에 반영하면 sample efficiency가 향상된다. Blindly complex models보다 thoughtfully designed simple models가 낫다.

셋째, practical considerations를 무시하지 말아야 한다. 학술 연구가 최고 benchmark score만 추구하면 실무와 괴리된다. Speed, memory, ease of deployment도 중요한 metrics다. Holistic evaluation이 필요하다.

넷째, iterative refinement의 가치다. CFlow가 foundation을 놓고 FastFlow가 개선했다. 초기 연구의 한계를 분석하고 targeted improvements를 수행했다. Incremental progress가 breakthrough를 만들 수 있다.

FastFlow의 성공은 우연이 아니었다. 적절한 assumptions, thoughtful design, practical alignment가 결합된 결과다. 이는 향후 이상 탐지뿐만 아니라 딥러닝 일반에도 적용되는 원칙들이다.

## 8.2 Channel vs Spatial Information

FastFlow의 핵심 결정은 채널 독립성 가정이었다. 이는 채널 정보와 공간 정보의 상대적 중요성에 대한 깊은 질문을 제기한다.

**Information Content Analysis**

CNN 특징 맵 $\mathbf{F} \in \mathbb{R}^{h \times w \times c}$는 두 가지 차원의 정보를 담는다. 공간 차원 $(h, w)$는 "where" 정보를, 채널 차원 $c$는 "what" 정보를 인코딩한다. 이상 탐지에서 어느 것이 더 중요한가?

공간 정보는 결함의 위치와 형태를 나타낸다. Crack이 어디에 있고 어떤 모양인지, contamination이 얼마나 큰 영역을 차지하는지를 알려준다. 이는 localization에 필수적이다. Pixel-level segmentation을 위해서는 spatial structure가 보존되어야 한다.

채널 정보는 결함의 유형과 특성을 나타낸다. Edge channel은 경계의 불연속성을, texture channel은 표면의 이상을 포착한다. 서로 다른 채널들이 결함의 다양한 aspect를 표현한다. 이는 classification과 characterization에 중요하다.

FastFlow는 공간 정보를 우선시했다. 각 채널 내의 spatial dependencies를 명시적으로 모델링했다. 2D convolutional coupling layers가 이를 담당한다. 반면 채널 간 dependencies는 무시했다. 독립성을 가정했다.

CFlow는 채널 정보를 우선시했다. 각 위치에서 모든 채널을 함께 모델링했다. $c$차원 공간에서의 complex distribution을 학습했다. 반면 spatial dependencies는 간접적으로만 다뤄졌다. Position embedding이 유일한 공간 정보였다.

**Empirical Evidence from Ablation Studies**

FastFlow와 CFlow를 비교한 ablation studies가 통찰을 제공한다. FastFlow에서 채널을 완전히 독립적으로 처리하지 않고 일부 채널 간 interaction을 추가하면 어떻게 되는가?

Experiment 1: Channel grouping. 256 채널을 32 그룹으로 나누고 각 그룹(8 채널)을 함께 모델링한다. 결과: 성능이 거의 동일(98.5% → 98.4%). 메모리와 시간은 2배 증가. Channel interaction의 추가 이득이 미미하다.

Experiment 2: Channel attention. 각 채널의 anomaly score에 learned weights를 적용하여 adaptive fusion을 수행한다. 결과: 성능이 약간 향상(98.5% → 98.6%). 그러나 차이가 통계적으로 유의하지 않다. Attention의 이득이 제한적이다.

Experiment 3: Full channel dependency (CFlow style). 모든 채널을 함께 3D flow로 모델링한다. 결과: 성능이 동일하거나 약간 낮다(98.5% → 98.3%). 시간은 3-5배 증가. Channel dependency modeling이 cost-effective하지 않다.

이러한 결과는 채널 간 정보가 이상 탐지에서 marginal함을 시사한다. Spatial structure가 훨씬 더 informative하다. 결함은 spatial patterns로 나타나며, 채널 간 subtle correlations는 secondary하다.

**Theoretical Perspective: Curse of Dimensionality**

채널과 공간의 trade-off는 dimensionality의 관점에서도 이해할 수 있다. $h \times w$ 크기의 2D space는 약 3000 차원이다($56 \times 56$). $c$ 크기의 channel space는 256-1024 차원이다. 어느 쪽의 density modeling이 더 어려운가?

Curse of dimensionality는 고차원 공간에서 데이터가 sparse해짐을 말한다. 3000차원 공간에서 충분한 coverage를 얻으려면 exponentially 많은 샘플이 필요하다. 그러나 spatial space는 structured되어 있다. Nearby pixels are correlated. Effective dimensionality가 훨씬 낮다.

Channel space는 비교적 낮은 차원(256-1024)이지만 덜 structured되어 있을 수 있다. 각 채널이 독립적인 concept를 인코딩하면 effective dimensionality가 nominal dimension에 가깝다. Density modeling이 상대적으로 어렵다.

FastFlow는 낮은 effective dimensionality(spatial)에서 modeling하고 높은 effective dimensionality(channel)는 단순화했다. 이는 limited data 상황에서 합리적인 전략이다. CFlow는 반대로 했고 결과적으로 overfitting 위험이 높았다.

**Spatial Coherence in Anomalies**

결함의 본질적 특성도 고려해야 한다. 대부분의 산업 결함은 spatially coherent하다. Crack은 연속된 선, dent는 연결된 오목한 영역, contamination은 contiguous한 patch다. Random한 scattered pixels는 드물다.

Spatially coherent anomalies는 2D structure로 잘 포착된다. Convolutional kernels가 local patterns를 감지한다. 연속된 이상 영역이 consistent하게 높은 anomaly score를 받는다. 이는 robust detection을 제공한다.

Channel-wise correlations는 덜 명확한 패턴을 가진다. 결함이 어떤 채널 조합에서 나타나는지 예측하기 어렵다. Scratch는 edge channels에서, discoloration은 color channels에서 두드러질 수 있지만 항상 그런 것은 아니다. Variability가 크다.

FastFlow의 channel independence assumption은 이러한 variability를 implicit하게 다룬다. 각 채널이 독립적으로 평가되므로 어떤 채널에서든 이상이 감지되면 전체 anomaly score가 높아진다. Channel-specific peculiarities에 robust하다.

**Feature Hierarchy Considerations**

CNN의 계층적 구조에서 low-level과 high-level features는 다른 특성을 가진다. Low-level(layer2)은 채널 간 correlation이 더 클 수 있다. Edge, color, texture channels가 서로 보완적이다. High-level(layer4)은 채널 간 독립성이 더 클 수 있다. 각 채널이 distinct object parts나 attributes를 인코딩한다.

FastFlow가 주로 layer3(mid-level)를 사용하는 것은 이러한 맥락에서 이해할 수 있다. Layer3는 채널 독립성 가정이 reasonably valid한 sweet spot이다. Sufficiently abstract하여 채널들이 구별되고, sufficiently concrete하여 유용한 정보를 담는다.

만약 layer2(low-level)를 사용한다면 채널 간 modeling이 더 중요할 수 있다. Color channels(R, G, B)는 clearly correlated하다. 이들을 독립적으로 처리하면 정보 손실이 클 수 있다. 그러나 layer2는 다른 이유로 덜 선호된다(noisy, computationally expensive).

Layer4(high-level)에서는 채널 독립성이 더 타당하다. Semantic features가 이미 disentangled되어 있다. 각 채널이 독립적인 concept를 나타낸다. FastFlow style이 더 적합하다. 실제로 layer4 single-scale FastFlow도 합리적인 성능(97.5%)을 보인다.

**Practical Implications**

채널 vs 공간 분석의 실무적 함의는 명확하다. 공간 정보가 더 중요하므로 spatial structure를 보존하는 방법이 유리하다. Convolutional architectures, spatial attention, 2D flows가 효과적이다.

채널 간 complex interactions는 대부분의 경우 불필요하다. Simple channel-wise processing이나 독립적 모델링으로 충분하다. 이는 계산 효율성을 크게 향상시킨다. FastFlow의 핵심 교훈이다.

예외적으로 channel information이 critical한 경우가 있을 수 있다. Specific color patterns가 결함의 signature인 경우(예: 특정 색깔의 오염물). 이런 경우 color channels 간 relationship이 중요하다. 그러나 이는 minority이다.

설계 원칙으로 spatial first, channel second를 권장한다. Architecture에서 spatial dependencies를 우선적으로 모델링한다. Convolutions, spatial attention, graph connections 등을 활용한다. Channel dependencies는 필요할 때만 추가한다. Simple aggregation이나 attention으로 충분한 경우가 많다.

## 8.3 Future Directions

Normalizing flow 기반 이상 탐지는 성숙했지만 여전히 개선의 여지가 있다. 향후 연구 방향을 제시한다.

**Lightweight Flow Architectures**

현재 flow 방법들은 여전히 상당한 계산을 요구한다. FastFlow도 20-50ms로 일부 실시간 응용에는 부족하다. 더 가벼운 flow architectures가 필요하다.

Neural ODE inspired continuous flows를 탐구할 수 있다. Discrete coupling layers 대신 continuous transformation을 학습한다. Adaptive computation을 통해 필요한 만큼만 계산한다. Simple samples는 빠르게, complex samples는 깊게 처리한다.

Pruning과 architecture search로 optimal flow depth를 자동으로 찾을 수 있다. 모든 카테고리에 동일한 flow depth가 필요하지 않다. Simple textures는 3-4 layers, complex objects는 6-8 layers로 충분할 수 있다. Per-category optimization이 전체 효율성을 높인다.

Knowledge distillation으로 flow의 knowledge를 작은 모델로 전달할 수 있다. Large teacher flow를 학습하고 small student network로 distill한다. Student는 flow의 probabilistic outputs를 모방한다. 이는 FastFlow와 EfficientAD의 장점을 결합한다.

**Foundation Model Integration**

최근의 foundation models(CLIP, DINOv2, SAM)는 강력한 visual representations를 제공한다. 이들을 flow와 결합하면 성능이 향상될 수 있다.

CLIP features는 ImageNet보다 훨씬 풍부하다. Multi-modal learning으로 semantic understanding이 깊다. CLIP을 backbone으로 사용한 FastFlow는 1-2%포인트 향상을 기대할 수 있다. 이미 일부 연구(APRIL-GAN)에서 시도되었다.

DINOv2는 self-supervised learning으로 discriminative features를 학습했다. Fine-grained visual details를 잘 포착한다. DINOv2 + FastFlow 조합이 promising하다. Zero-shot anomaly detection도 가능할 수 있다.

SAM(Segment Anything Model)은 뛰어난 segmentation 능력을 가진다. Flow의 anomaly map을 SAM으로 refine하면 pixel-level accuracy가 향상될 수 있다. Coarse anomaly detection(flow) + fine segmentation(SAM)의 two-stage approach를 고려한다.

Multi-scale foundation features를 활용할 수 있다. ViT based models는 multiple attention layers를 가진다. 각 layer의 features를 추출하여 multi-scale FastFlow에 적용한다. CNN features보다 더 semantic한 multi-scale representation을 얻는다.

**Few-shot and Zero-shot Adaptation**

현재 flow 방법들은 카테고리당 수백 장의 정상 샘플을 요구한다. 신제품이나 rare cases에서는 수집이 어렵다. Few-shot learning이 필요하다.

Meta-learning으로 flow를 학습할 수 있다. 여러 카테고리에서 공통된 anomaly detection skill을 학습한다. 새로운 카테고리에 수십 장만으로 빠르게 adapt한다. MAML이나 Prototypical Networks 같은 기법을 적용한다.

Pre-trained flows를 transfer learning에 활용할 수 있다. 대규모 데이터셋(모든 MVTec 카테고리)에서 universal flow를 학습한다. 새 카테고리에 fine-tuning만 수행한다. Feature extractor처럼 flow도 transferable할 수 있다.

Zero-shot anomaly detection은 궁극적인 목표다. 정상 샘플 없이도 이상을 탐지한다. Foundation models의 prior knowledge를 활용한다. "Normal objects should align with CLIP's concept of normal"같은 heuristics를 사용한다. Completely unsupervised approach다.

Synthetic data generation으로 few-shot을 보완할 수 있다. 소수의 정상 샘플로 flow를 학습하고, flow에서 sampling하여 synthetic normal samples를 생성한다. 이들로 다시 학습하는 iterative process로 성능을 높인다.

**Multimodal Anomaly Detection**

산업 현장에서는 visual inspection 외에 다양한 센서 데이터가 있다. Thermal imaging, X-ray, ultrasound, vibration, sound 등이다. Multimodal information을 결합하면 더 robust한 탐지가 가능하다.

Multimodal flow를 설계할 수 있다. 각 modality의 features를 추출하고 함께 modeling한다. Image flow와 thermal flow를 concatenate하거나 late fusion한다. Joint distribution을 학습하여 modalities 간 consistency를 포착한다.

Cross-modal attention으로 modalities를 align할 수 있다. Image features가 thermal features를 attend하고 vice versa. Complementary information을 효과적으로 통합한다. Transformer architecture를 활용한다.

Anomaly는 modality 간 inconsistency로 나타날 수 있다. Visual로는 정상이지만 thermal로는 이상인 경우다. Flow가 joint distribution을 학습했다면 이러한 inconsistency를 높은 likelihood로 감지한다. Multimodal의 핵심 장점이다.

Practical challenges도 있다. Multimodal data collection이 expensive하다. Modalities 간 alignment(spatial, temporal)가 어렵다. 각 modality의 preprocessing과 feature extraction이 달라야 한다. 그러나 critical applications에서는 충분히 가치 있다.

**Explainable Anomaly Detection**

Flow의 probabilistic nature는 explainability를 제공하지만 충분하지 않다. "Low likelihood"는 "why anomalous?"를 설명하지 못한다. 더 interpretable한 outputs이 필요하다.

Counterfactual explanations을 생성할 수 있다. Flow의 inverse mapping으로 anomaly를 normal로 바꾸는 minimal edits를 찾는다. "If this scratch were removed, the sample would be normal"같은 설명을 제공한다. Optimization 또는 gradient-based search로 구현한다.

Feature attribution methods를 적용할 수 있다. GradCAM, LIME, SHAP를 flow에 적용한다. 어떤 image regions나 features가 anomaly score에 가장 기여했는지 시각화한다. 이는 디버깅과 신뢰 구축에 유용하다.

Prototype-based explanations도 고려할 수 있다. 정상 샘플들의 prototypes를 유지한다. Anomaly가 탐지되면 가장 유사한 normal prototype과 비교한다. "This anomaly differs from normal prototype in these aspects"같은 설명을 제공한다.

Natural language explanations가 ultimate goal이다. Flow의 outputs를 language model에 전달하여 human-readable descriptions를 생성한다. "A dark contamination of approximately 5mm diameter is detected on the surface"같은 설명이다. Vision-language models를 활용한다.

**Active Learning and Human-in-the-Loop**

완전 자동 시스템보다 human-in-the-loop이 더 realistic하다. 모델이 uncertain한 cases를 human에게 query하고 feedback을 받는다. Active learning으로 효율성을 높인다.

Uncertainty quantification이 핵심이다. Flow는 likelihood를 제공하지만 epistemic uncertainty는 별도다. Ensemble of flows나 Bayesian flows로 prediction uncertainty를 추정한다. High uncertainty samples를 human에게 보낸다.

Human feedback을 online learning에 활용할 수 있다. Human이 labeling한 samples로 flow를 incrementally update한다. Continual learning techniques로 catastrophic forgetting을 방지한다. Model이 점진적으로 개선된다.

Interactive threshold adjustment도 유용하다. Human이 실시간으로 threshold를 조정하며 결과를 확인한다. 적절한 balance(precision vs recall)를 찾는다. System이 human의 preferences를 학습한다.

Quality control workflow에 통합할 수 있다. Flow가 1차 screening을 수행하고 suspicious samples를 flag한다. Human inspectors가 final decision을 내린다. Human workload가 크게 줄어들면서 accuracy는 유지된다.

**Temporal and Sequential Anomaly Detection**

현재 flow 방법들은 single images를 독립적으로 처리한다. 그러나 많은 응용에서 temporal sequence가 available하다. Video surveillance, continuous manufacturing process 등이다. Temporal information을 활용하면 성능이 향상될 수 있다.

Temporal flow를 설계할 수 있다. Video frames의 sequence를 jointly modeling한다. 3D convolutions이나 recurrent structures를 flow에 통합한다. Temporal coherence와 anomalies를 동시에 포착한다.

Slowly changing normal patterns를 adaptive하게 tracking할 수 있다. Flow가 gradually drifting distribution을 따라간다. Online update로 recent normal patterns를 반영한다. Sudden changes는 anomalies로, gradual changes는 distribution shift로 구별한다.

Process anomaly detection도 중요한 응용이다. Manufacturing process의 각 단계를 monitoring한다. Normal process flow를 학습하고 deviations를 탐지한다. Root cause analysis를 위해 어느 단계에서 이상이 시작되었는지 파악한다.

**Robustness and Adversarial Considerations**

Adversarial attacks에 대한 robustness도 고려해야 한다. Malicious actors가 anomalous products를 normal로 위장하려 시도할 수 있다. Flow models도 adversarial examples에 취약할 수 있다.

Adversarial training으로 robustness를 높일 수 있다. Training 중 adversarial samples를 생성하고 포함한다. Model이 subtle adversarial perturbations에 덜 민감해진다. 그러나 flow의 gradient는 complex하여 adversarial generation이 어려울 수 있다.

Certified robustness를 제공하는 것도 목표다. Randomized smoothing이나 interval bound propagation으로 provable guarantees를 얻는다. "Perturbations smaller than ε will not change the prediction"같은 certification을 제공한다.

Distribution shift에 대한 robustness도 중요하다. Training과 deployment의 distribution이 다를 수 있다. Domain adaptation techniques로 flow를 adapt한다. Test-time adaptation도 고려한다.

Normalizing flow 기반 이상 탐지는 promising한 패러다임이다. FastFlow의 성공이 이를 입증했다. 향후 연구들이 위의 방향들을 탐구하면 더 강력하고 실용적인 시스템이 만들어질 것이다. Efficiency, generalization, explainability, robustness의 동시 개선이 핵심 과제다. Foundation models, few-shot learning, multimodal fusion의 통합이 next breakthroughs를 가져올 것이다.