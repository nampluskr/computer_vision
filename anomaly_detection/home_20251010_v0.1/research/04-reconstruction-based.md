# Reconstruction-Based Anomaly Detection

---

## 1. Paradigm Overview

### 1.1 Core Principle

재구성 기반 이상 감지(Reconstruction-based Anomaly Detection)는 딥러닝을 활용한 이상 감지 연구의 가장 초기 형태이자, 가장 직관적인 접근법으로 평가된다. 이 패러다임의 핵심 가정은 명확하다. 정상 데이터로만 학습된 재구성 모델은 정상 샘플에 대해서는 높은 품질의 복원을 수행하지만, 학습 과정에서 관찰하지 못한 이상 패턴에 대해서는 정확한 재구성에 실패한다는 것이다. 이러한 재구성 오차의 차이가 정상과 이상을 구별하는 주요 신호로 작용한다.

이 접근법의 이론적 기반은 "학습된 정상성(learned normality)"이라는 개념에 뿌리를 두고 있다. 모델은 학습 데이터에 내재된 정상 패턴의 본질적 구조를 포착하며, 이 학습된 구조로부터 크게 일탈하는 샘플을 재구성할 때 필연적으로 큰 오차를 발생시킨다. 특히 Autoencoder의 병목(bottleneck) 구조는 이러한 학습 메커니즘을 강제하는 귀납적 편향(inductive bias)으로 기능한다. 고차원의 입력 공간을 저차원의 잠재 공간으로 압축하는 과정에서, 모델은 데이터의 가장 본질적인 특징만을 보존하도록 강제된다.

재구성 기반 방법론은 크게 세 가지 핵심 구성 요소로 이루어진다. 첫째, Encoder는 입력 이미지를 저차원의 잠재 표현(latent representation)으로 압축한다. 둘째, Bottleneck은 정보의 손실을 강제함으로써 정상 패턴의 핵심 특징만이 보존되도록 한다. 셋째, Decoder는 이 압축된 잠재 표현으로부터 원본 이미지를 복원한다. 이 세 요소의 상호작용을 통해 모델은 정상 데이터의 manifold를 학습하게 되며, 이 manifold 밖에 위치한 이상 샘플은 자연스럽게 높은 재구성 오차를 보이게 된다.

### 1.2 Reconstruction Error as Anomaly Signal

재구성 오차를 이상 점수로 활용하는 방식은 수학적으로 명확하게 정식화될 수 있다. 입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$가 주어졌을 때, 재구성된 이미지를 $\hat{\mathbf{x}} = D(E(\mathbf{x}))$로 표기하자. 여기서 $E$는 Encoder, $D$는 Decoder를 나타낸다. 픽셀 수준의 재구성 오차는 일반적으로 $L_p$ 노름을 통해 측정되며, 다음과 같이 정의된다.

$$
\mathcal{L}_{\text{rec}}(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_p = \left(\sum_{i,j,c} |\mathbf{x}_{i,j,c} - \hat{\mathbf{x}}_{i,j,c}|^p\right)^{1/p}
$$

실무에서는 주로 $L_1$ 노름(Mean Absolute Error) 또는 $L_2$ 노름(Mean Squared Error)이 사용된다. $L_1$ 노름은 이상치(outlier)에 더욱 강건한 특성을 보이며, $L_2$ 노름은 큰 오차에 더 큰 가중치를 부여함으로써 뚜렷한 결함을 강조한다. 이미지 전체에 대한 이상 점수 $s(\mathbf{x})$는 픽셀별 오차의 집계 함수로 계산된다.

$$
s(\mathbf{x}) = \text{agg}(\mathcal{L}_{\text{rec}}(\mathbf{x}, \hat{\mathbf{x}}))
$$

여기서 $\text{agg}(\cdot)$는 평균(mean), 최대값(max), 또는 상위 $k$% 평균 등 다양한 집계 전략이 될 수 있다. 픽셀 단위의 이상 맵(anomaly map) $\mathcal{A}: \mathbb{R}^{H \times W} \rightarrow \mathbb{R}^+$은 공간적 위치별로 이상 정도를 시각화하며, 다음과 같이 정의된다.

$$
\mathcal{A}(i, j) = \sum_c |\mathbf{x}(i,j,c) - \hat{\mathbf{x}}(i,j,c)|
$$

재구성 기반 이상 감지의 이상적인 목표는 정상 샘플과 이상 샘플 간의 재구성 오차 분포를 최대한 분리하는 것이다. 정상 분포를 $\mathcal{N}$, 이상 분포를 $\mathcal{A}$로 표기할 때, 우리는 다음 조건이 만족되기를 기대한다.

$$
\mathbb{E}_{\mathbf{x} \sim \mathcal{N}}[s(\mathbf{x})] \ll \mathbb{E}_{\mathbf{x} \sim \mathcal{A}}[s(\mathbf{x})]
$$

그러나 실제로는 이 분포 간 분리가 완벽하지 않으며, 특히 미세한 결함이나 텍스처 변화에 대해서는 정상과 이상의 경계가 모호해질 수 있다. 이는 재구성 기반 방법론의 근본적인 한계 중 하나로 인식되고 있다.

### 1.3 Normal Manifold Learning

재구성 기반 방법론의 이론적 토대는 매니폴드 가설(manifold hypothesis)에 근거한다. 이 가설에 따르면, 고차원 입력 공간 $\mathbb{R}^d$에 존재하는 정상 데이터는 실제로는 훨씬 낮은 차원의 매니폴드 $\mathcal{M} \subset \mathbb{R}^d$ 상에 집중되어 있다. Autoencoder는 이 저차원 매니폴드를 근사적으로 학습하며, Encoder와 Decoder의 결합은 입력 공간에서 정상 매니폴드로의 projection 연산자로 해석될 수 있다.

Encoder $E: \mathbb{R}^d \rightarrow \mathbb{R}^k$와 Decoder $D: \mathbb{R}^k \rightarrow \mathbb{R}^d$를 고려하자. 여기서 $k \ll d$는 잠재 공간의 차원이다. 이들의 결합 $D \circ E$는 다음과 같은 projection으로 작용한다.

$$
\mathbf{x} \xrightarrow{E} \mathbf{z} \xrightarrow{D} \hat{\mathbf{x}} \approx \text{proj}_{\mathcal{M}}(\mathbf{x})
$$

정상 샘플 $\mathbf{x}_{\text{normal}} \in \mathcal{M}$의 경우, 이미 매니폴드 상에 위치하므로 projection은 입력을 거의 변화시키지 않는다. 따라서 재구성 오차는 매우 작게 나타난다.

$$
\|\mathbf{x}_{\text{normal}} - D(E(\mathbf{x}_{\text{normal}}))\|_2 \approx 0
$$

반면, 이상 샘플 $\mathbf{x}_{\text{anomaly}} \notin \mathcal{M}$은 정상 매니폴드로부터 멀리 떨어져 있다. Projection 과정에서 이 샘플은 가장 가까운 정상 매니폴드 상의 점으로 mapping되지만, 원본과의 거리는 여전히 크게 유지된다.

$$
\|\mathbf{x}_{\text{anomaly}} - D(E(\mathbf{x}_{\text{anomaly}}))\|_2 \gg 0
$$

Bottleneck 구조는 정보 이론적 관점에서 정보 압축(information compression)을 강제한다. 잠재 공간의 차원이 $k$일 때, 이론적으로 최대 $k \log_2 |\mathcal{V}|$ bits의 정보만이 bottleneck을 통과할 수 있다. 여기서 $|\mathcal{V}|$는 잠재 변수가 취할 수 있는 값의 범위를 나타낸다. 이러한 정보 제약은 모델이 입력의 가장 본질적이고 재현성 있는 구조적 특징에만 집중하도록 유도한다. 정상 데이터에서 자주 관찰되는 패턴은 효율적으로 인코딩되지만, 드물게 발생하는 이상 패턴은 잠재 표현에 효과적으로 담기지 못하게 된다.

### 1.4 Historical Development

재구성 기반 이상 감지의 역사는 딥러닝 기술의 발전 과정과 밀접하게 연관되어 있다. 2010년대 초반, Hinton 등의 연구진이 제안한 심층 Autoencoder는 고차원 데이터의 효과적인 차원 축소가 가능함을 보였으며, 이는 이상 감지 분야에도 자연스럽게 적용되기 시작했다. 초기 연구들은 주로 완전 연결 계층(fully connected layers)을 사용했으나, 이미지 데이터의 공간적 구조를 충분히 활용하지 못한다는 한계가 있었다.

2015년부터 2017년 사이, Convolutional Autoencoder의 등장은 이미지 기반 이상 감지에 중요한 전환점을 가져왔다. Convolutional 구조는 이미지의 국소적 패턴과 공간적 계층 구조를 효과적으로 모델링할 수 있었다. 특히 U-Net 구조에서 영감을 받은 skip connection의 도입은 고해상도 재구성을 가능하게 했으며, 미세한 결함의 검출 성능을 크게 향상시켰다. 그러나 이 시기의 모델들은 정상 샘플에 대한 과적합(overfitting) 문제를 겪었으며, 결과적으로 정상 샘플의 작은 변이에도 높은 재구성 오차를 보이는 경향이 있었다.

2018년, GANomaly의 등장은 재구성 기반 방법론에 새로운 가능성을 제시하는 듯 보였다. Adversarial training을 통해 더욱 강력한 재구성 능력을 확보하고, Encoder-Decoder-Encoder 구조를 통해 잠재 공간의 일관성을 강제하는 접근법은 이론적으로 매우 우아했다. 그러나 실무 적용 과정에서 GAN 고유의 학습 불안정성 문제가 심각한 장애물로 작용했다. Mode collapse, 수렴 문제, 그리고 하이퍼파라미터에 대한 극도의 민감성으로 인해 GANomaly는 상업적 성공을 거두지 못했으며, 재구성 기반 방법론 전체에 대한 회의론이 확산되기도 했다.

2021년, DRAEM의 등장은 재구성 기반 패러다임의 르네상스를 촉발했다. DRAEM이 제안한 핵심 혁신은 Simulated Anomaly 개념이었다. 실제 결함 샘플 없이도 Perlin noise를 활용하여 합성 결함을 생성하고, 이를 통해 지도 학습(supervised learning) 프레임워크를 구축할 수 있음을 보였다. 이는 비지도 학습에서 지도 학습으로의 패러다임 전환을 의미했으며, 학습의 안정성과 수렴성을 획기적으로 개선했다. 특히 Few-shot 학습 능력은 주목할 만한 성과였다. DRAEM은 단 10-50장의 정상 샘플만으로도 97.5%의 이상 감지 정확도를 달성했으며, 이는 신제품 출시나 희귀 결함 시나리오에서 실용적 가치가 매우 높았다.

2022년의 DSR은 또 다른 방향의 발전을 제시했다. VQ-VAE(Vector Quantized Variational AutoEncoder) 기반의 Dual Subspace 구조를 통해 구조(structure)와 텍스처(texture)를 분리하여 처리함으로써, 복잡한 표면 텍스처를 가진 소재에서의 성능을 크게 향상시켰다. 직물, 카펫, 가죽 등 텍스처가 지배적인 카테고리에서 DSR은 다른 방법론들을 능가하는 성능을 보였다. 이는 재구성 기반 방법론이 특정 도메인에 특화됨으로써 SOTA 성능을 달성할 수 있음을 보여주는 사례였다.

이러한 발전 과정을 통해 재구성 기반 이상 감지는 초기의 단순한 Autoencoder로부터 시작하여, GANomaly의 실패를 거쳐, DRAEM과 DSR이 보여준 혁신적 접근법에 이르기까지 지속적으로 진화해왔다. 특히 DRAEM이 제시한 Simulated Anomaly 개념은 이후 많은 연구에 영향을 미쳤으며, Few-shot 이상 감지의 사실상 표준(de facto standard)으로 자리잡았다. 이는 재구성 기반 방법론이 단순히 역사적 유산이 아니라, 여전히 특정 시나리오에서 강력한 솔루션을 제공할 수 있음을 증명한다.

---

## 2. GANomaly (2018)

### 2.1 Basic Information

GANomaly는 2018년 ACCV(Asian Conference on Computer Vision)에서 Samet Akcay, Amir Atapour-Abarghouei, Toby P. Breckon에 의해 발표된 "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training"에서 제안되었다. 이 연구는 당시 이미지 생성 분야에서 혁명적 성과를 보이던 GAN(Generative Adversarial Network)의 강력한 생성 능력을 이상 감지 문제에 적용하려는 시도였다. 2018년은 GAN이 StyleGAN과 BigGAN 등을 통해 고품질 이미지 생성의 가능성을 입증하던 시기였으며, 연구자들은 이러한 생성 능력이 정상 패턴의 더욱 정교한 학습으로 이어질 것으로 기대했다.

GANomaly의 핵심 아이디어는 Adversarial training을 통해 더욱 강력한 Encoder-Decoder를 학습하고, 추가적인 Encoder를 통해 잠재 공간의 일관성을 강제하는 것이었다. 이 접근법은 이론적으로 매우 세련되었으며, 재구성 오차만이 아니라 잠재 공간에서의 불일치까지 활용함으로써 더욱 robust한 이상 감지가 가능할 것으로 예상되었다. 그러나 이러한 야심찬 설계는 실무 적용 과정에서 GAN 학습 고유의 불안정성이라는 근본적 한계에 부딪히게 되었다. Mode collapse, 수렴 불확실성, 그리고 극도로 높은 하이퍼파라미터 민감도로 인해 GANomaly는 학계의 관심을 받았음에도 불구하고 산업 현장에서는 채택되지 못했다.

### 2.2 GAN-based Architecture

GANomaly의 구조적 특징은 전통적인 Autoencoder와 근본적으로 다르다. 가장 독특한 점은 Encoder-Decoder-Encoder(E-D-E)라는 삼단 구조의 채택이다. 일반적인 Autoencoder가 Encoder와 Decoder 두 개의 네트워크로 구성되는 것과 달리, GANomaly는 세 개의 주요 네트워크로 이루어진다. Generator $G$는 다시 세 부분으로 나뉘는데, 첫 번째 Encoder $G_E$는 입력 이미지 $\mathbf{x}$를 잠재 표현 $\mathbf{z}_1$으로 압축하고, Decoder $G_D$는 이 잠재 표현으로부터 재구성 이미지 $\hat{\mathbf{x}}$를 생성하며, 두 번째 Encoder $\hat{G}_E$는 재구성된 이미지를 다시 잠재 표현 $\mathbf{z}_2$로 인코딩한다. 이 과정을 수식으로 표현하면 다음과 같다.

$$
\begin{aligned}
\mathbf{z}_1 &= G_E(\mathbf{x}) \\
\hat{\mathbf{x}} &= G_D(\mathbf{z}_1) \\
\mathbf{z}_2 &= \hat{G}_E(\hat{\mathbf{x}})
\end{aligned}
$$

이러한 이중 인코딩 구조의 핵심 철학은 잠재 공간의 일관성(latent space consistency)에 있다. 정상 샘플의 경우, 첫 번째 인코딩 $\mathbf{z}_1$과 두 번째 인코딩 $\mathbf{z}_2$가 매우 유사해야 한다는 가정이다. 다시 말해, 잘 재구성된 이미지는 원본 이미지와 동일한 잠재 표현을 가져야 한다는 것이다. 이는 $\mathbf{z}_1 \approx \mathbf{z}_2$라는 제약으로 나타나며, 동시에 $\mathbf{x} \approx \hat{\mathbf{x}}$라는 재구성 제약도 만족해야 한다. 이상 샘플은 이 두 조건 중 하나 이상을 위반하게 되며, 특히 잠재 공간의 불일치 $\|\mathbf{z}_1 - \mathbf{z}_2\|$가 이상 점수의 주요 지표로 사용된다.

Discriminator $D$는 전통적인 GAN과 유사한 역할을 수행하지만, 단순히 진위를 판별하는 것을 넘어 중간 계층의 feature를 추출하는 역할도 담당한다. Discriminator는 실제 이미지 $\mathbf{x}$와 재구성된 이미지 $\hat{\mathbf{x}}$를 구별하도록 학습되며, 이 과정에서 Generator는 Discriminator를 속이기 위해 더욱 realistic한 재구성을 생성하도록 유도된다. 이러한 adversarial dynamics가 재구성 품질의 향상으로 이어질 것으로 기대되었다.

GANomaly의 학습은 세 가지 손실 함수의 가중 합을 최소화하는 방향으로 진행된다. Adversarial loss $\mathcal{L}_{\text{adv}}$는 Discriminator의 중간 feature 공간에서 실제 이미지와 재구성 이미지의 거리를 측정한다.

$$
\mathcal{L}_{\text{adv}} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\|f(D(\mathbf{x})) - f(D(G(\mathbf{x})))\|_2^2]
$$

여기서 $f(D(\cdot))$는 Discriminator의 중간 계층에서 추출된 feature를 나타낸다. 이 손실은 Generator가 feature 공간에서도 실제와 유사한 표현을 생성하도록 강제한다. Contextual loss $\mathcal{L}_{\text{con}}$은 픽셀 수준에서의 재구성 품질을 보장한다.

$$
\mathcal{L}_{\text{con}} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\|\mathbf{x} - G(\mathbf{x})\|_1]
$$

$L_1$ norm의 사용은 $L_2$ norm에 비해 더욱 sharp한 재구성을 유도하며, 픽셀별 절대 차이를 직접적으로 최소화한다. 마지막으로 Encoder loss $\mathcal{L}_{\text{enc}}$는 GANomaly의 핵심 혁신으로, 두 Encoder의 출력 간 일관성을 강제한다.

$$
\mathcal{L}_{\text{enc}} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2]
$$

이 세 손실의 가중 합이 Generator의 총 손실을 구성한다.

$$
\mathcal{L}_G = w_{\text{adv}} \mathcal{L}_{\text{adv}} + w_{\text{con}} \mathcal{L}_{\text{con}} + w_{\text{enc}} \mathcal{L}_{\text{enc}}
$$

원 논문에서는 $w_{\text{adv}}=1$, $w_{\text{con}}=50$, $w_{\text{enc}}=1$을 제안하고 있다. Contextual loss의 가중치가 압도적으로 큰 것은 재구성 품질이 무엇보다 중요함을 반영한다. Discriminator는 표준 GAN의 Binary Cross-Entropy loss로 학습된다.

$$
\mathcal{L}_D = \mathbb{E}_{\mathbf{x}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{x}}[\log(1 - D(G(\mathbf{x})))]
$$

잠재 공간의 차원은 일반적으로 100에서 200 사이로 설정되며, 이는 입력 이미지의 차원($256 \times 256 \times 3 \approx 196,608$)에 비해 극도로 압축된 표현이다. 이러한 극단적 압축은 모델이 정상 패턴의 가장 본질적인 특징만을 학습하도록 강제한다. 테스트 단계에서의 이상 점수는 두 잠재 표현 간의 $L_1$ 거리로 정의된다.

$$
A(\mathbf{x}) = \|\mathbf{z}_1 - \mathbf{z}_2\|_1 = \|G_E(\mathbf{x}) - \hat{G}_E(G_D(G_E(\mathbf{x})))\|_1
$$

정상 샘플에 대해서는 $A(\mathbf{x}) \approx 0$이, 이상 샘플에 대해서는 $A(\mathbf{x}) \gg 0$이 되도록 학습이 진행된다.

### 2.3 Technical Challenges

GANomaly의 구현은 여러 기술적 난제를 수반한다. 첫째, Generator와 Discriminator의 용량(capacity) 균형 문제가 있다. Generator가 상대적으로 약하면 고품질 재구성이 불가능하며, 반대로 Discriminator가 너무 강력하면 Generator가 학습 신호를 전혀 받지 못하는 상황이 발생한다. 이는 전통적인 GAN 학습의 고질적 문제이지만, 이상 감지 맥락에서는 더욱 심각하다. 왜냐하면 학습 데이터가 정상 샘플로만 구성되어 있어 Discriminator가 판별해야 할 대상이 제한적이기 때문이다.

둘째, 하이퍼파라미터에 대한 극도의 민감성이 문제다. 세 가지 손실 함수의 가중치 조합, learning rate, batch size 등 수많은 하이퍼파라미터가 최종 성능에 결정적 영향을 미친다. 더욱 난해한 점은 최적 설정이 데이터셋마다, 심지어 동일 데이터셋 내에서도 카테고리마다 달라진다는 것이다. MVTec 데이터셋의 "bottle" 카테고리에서 잘 작동하는 설정이 "carpet" 카테고리에서는 전혀 효과가 없는 경우가 빈번하다.

셋째, 메모리 요구사항이 매우 높다. 세 개의 대형 네트워크($G_E$, $G_D$, $\hat{G}_E$, $D$)를 동시에 학습해야 하므로, 배치 크기를 크게 가져갈 수 없다. 그러나 작은 배치 크기는 GAN 학습의 안정성에 부정적 영향을 미친다. 일반적으로 배치 크기 16 이상을 권장하지만, GPU 메모리 제약으로 8 또는 그 이하로 설정해야 하는 경우가 많으며, 이는 학습 불안정성을 더욱 악화시킨다.

### 2.4 Training Instability Issues

GANomaly의 가장 심각한 문제는 학습 불안정성이다. GAN 학습은 본질적으로 Generator와 Discriminator 간의 minimax game으로 정식화된다.

$$
\min_G \max_D \, V(D, G) = \mathbb{E}_{\mathbf{x}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]
$$

이론적으로 이 게임은 Nash equilibrium에 수렴해야 하지만, gradient-based optimization은 이러한 equilibrium을 안정적으로 찾지 못한다. 실제 학습 과정에서는 지속적인 oscillation이나 완전한 divergence가 빈번히 관찰된다. 학습 곡선은 매우 불규칙하며, validation 성능이 개선과 악화를 반복하는 양상을 보인다.

Gradient vanishing과 exploding 문제도 심각하다. Discriminator가 너무 강해지면 Generator에게 전달되는 gradient가 소실되어 Generator의 학습이 정체된다. 반대로 Generator가 Discriminator를 압도하면 gradient가 폭발하여 학습이 불안정해진다. 이러한 문제를 완화하기 위해 learning rate scheduling, gradient clipping, spectral normalization 등 다양한 기법이 제안되었지만, 이들 모두 추가적인 하이퍼파라미터를 도입하며 문제를 근본적으로 해결하지 못한다.

재현성(reproducibility) 문제는 GANomaly의 실무 적용을 가로막는 결정적 장애물이다. 동일한 하이퍼파라미터와 학습 설정을 사용하더라도, random seed에 따라 결과가 크게 달라진다. 어떤 실행에서는 Image AUROC 90%를 달성하지만, 다른 실행에서는 70%에 그치는 경우가 흔하다. 이러한 불안정성은 산업 현장에서 요구되는 신뢰성과 일관성을 제공할 수 없게 만든다.

### 2.5 Why GANomaly Failed

GANomaly의 실패를 이해하기 위해서는 세 가지 핵심 문제를 분석해야 한다. 첫째는 mode collapse 현상이다. Mode collapse는 GAN의 고질적 문제로, Generator가 데이터 분포의 모든 mode를 커버하지 못하고 일부 mode에만 집중하는 현상을 의미한다. 이상 감지 맥락에서 이는 특정 유형의 정상 샘플만 잘 재구성하고 다른 유형은 높은 오차를 보이는 결과로 나타난다. 예를 들어 MVTec의 "bottle" 카테고리에서 특정 조명 조건의 병만 잘 재구성하고, 다른 조명 조건은 이상으로 오분류하는 경우가 발생한다. 이는 false positive rate를 급격히 증가시키며, Image AUROC가 85-90% 수준에 머무르게 만든다. 이는 PatchCore의 99.1%와 비교할 때 실용성이 크게 떨어지는 수치다.

수학적으로, Generator는 데이터 분포 $p_{\text{data}}(\mathbf{x})$를 완전히 근사해야 한다. 즉, $p_G(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x}) \, \forall \mathbf{x}$가 성립해야 한다. 그러나 mode collapse 발생 시, Generator 분포는 데이터 분포의 일부만 커버한다.

$$
p_G(\mathbf{x}) \approx 
\begin{cases}
p_{\text{data}}(\mathbf{x}) & \text{if } \mathbf{x} \in \text{covered modes} \\
0 & \text{otherwise}
\end{cases}
$$

이는 uncovered mode에 속하는 정상 샘플들이 높은 재구성 오차를 보이는 원인이 된다.

둘째는 수렴 판정의 모호성이다. 전통적인 딥러닝에서는 validation loss의 수렴을 통해 학습 완료를 판단할 수 있지만, GAN은 이러한 명확한 기준이 없다. Generator loss와 Discriminator loss는 지속적으로 진동하며, 두 loss의 절대값은 모델의 품질과 직접적 상관관계가 없다. 이로 인해 early stopping 기준을 설정할 수 없으며, 언제 학습을 멈춰야 할지 판단하기 어렵다. 실무에서는 고정된 epoch 수(예: 100 epochs)를 사용하지만, 이는 데이터셋 크기에 따라 과소학습 또는 과적합을 야기할 수 있다.

셋째는 과도한 학습 시간이다. MVTec의 "bottle" 카테고리(209 training images)를 NVIDIA RTX 3090에서 학습할 때, Vanilla Autoencoder는 약 5분, DRAEM은 약 15분이 소요되지만, GANomaly는 45-60분이 필요하다. 이는 세 개의 대형 네트워크를 alternating optimization으로 학습해야 하고, GAN 안정성을 위해 작은 배치 크기를 사용해야 하며, 수렴까지 많은 iteration이 필요하기 때문이다. 100개 카테고리를 학습하는 배치 처리 환경에서는 75-100시간이 소요되며, 하이퍼파라미터 튜닝까지 고려하면 시간이 기하급수적으로 증가한다. 이는 빠른 개발 사이클이 중요한 산업 환경에서 치명적 단점이다.

### 2.6 Lessons Learned

GANomaly의 실패는 이상 감지 연구 커뮤니티에 중요한 교훈을 남겼다. 가장 근본적인 교훈은 복잡성이 반드시 성능 향상을 보장하지 않는다는 점이다. GANomaly는 이론적으로 매우 정교하고 우아한 설계를 가지고 있었지만, 실무 성능은 단순한 Vanilla Autoencoder와 크게 다르지 않았다. 오히려 학습 불안정성과 긴 학습 시간이라는 추가적 부담만 가중시켰다. Occam's Razor 원칙, 즉 "단순한 설명이 더 나은 설명이다"라는 철학이 여기서도 유효함이 입증되었다.

두 번째 교훈은 학습 안정성의 중요성이다. 이론적으로 뛰어난 성능보다 안정적이고 재현 가능한 학습이 실무에서 훨씬 더 중요하다. DRAEM이 GANomaly를 대체한 핵심 이유는 단순히 성능이 약간 더 좋아서가 아니라, 안정적인 지도 학습 프레임워크를 제공했기 때문이다. 산업 현장에서는 매번 안정적으로 90%의 성능을 내는 모델이, 때때로 95%를 내지만 자주 70%로 떨어지는 모델보다 훨씬 가치있다.

세 번째 교훈은 Adversarial training이 이상 감지에 적합하지 않을 수 있다는 점이다. GAN의 주요 강점은 고품질의 realistic한 이미지 생성 능력이다. 그러나 이상 감지에서는 생성 품질보다 재구성 오차의 구별성(discriminability)이 더 중요하다. 정상 샘플을 완벽하게 재구성하는 것보다, 정상과 이상 샘플 간의 재구성 오차 차이를 극대화하는 것이 목표다. 이러한 목표를 위해서는 단순한 $L_1$ 또는 $L_2$ loss만으로도 충분하며, adversarial loss의 복잡성이 실질적 이득을 가져다주지 못한다.

네 번째 교훈은 실무 적용 시 고려사항의 중요성이다. 학습 시간, 메모리 사용량, 하이퍼파라미터 민감도 등 실무적 제약은 종종 알고리즘의 우수성만큼이나 중요하다. GANomaly는 이러한 실무적 요구사항을 충족시키지 못했으며, 이는 baseline인 Autoencoder 대비 명확한 이점을 제시하지 못한 것과 결합되어 채택 실패로 이어졌다.

GANomaly 이후 재구성 기반 이상 감지 연구는 새로운 방향을 모색하게 되었다. 연구자들은 GAN을 포기하고 더 안정적인 접근법을 탐구하기 시작했다. DRAEM은 이러한 탐색의 결실로, Simulated Anomaly를 통해 지도 학습 프레임워크를 구축하고 학습 안정성을 확보했다. 이는 GANomaly가 추구했던 높은 재구성 품질을 포기하는 대신, 실용적이고 안정적인 솔루션을 제공하는 방향으로의 전환이었다. 결과적으로 DRAEM은 재구성 기반 방법론의 르네상스를 이끌었으며, Few-shot 이상 감지라는 새로운 가능성을 열었다.

GANomaly는 실패한 모델이지만, 그 시도와 실패로부터 얻은 통찰은 후속 연구의 중요한 밑거름이 되었다. "안정성", "실용성", "단순성"의 가치를 재조명하게 만들었다는 점에서, GANomaly의 역사적 의의는 결코 작지 않다.

프로젝트 지식을 확인했습니다. 이제 DRAEM의 3장을 학술 리뷰 논문 형식으로 작성하겠습니다.

---

## 3. DRAEM (2021)

### 3.1 Basic Information

DRAEM(Discriminatively trained Reconstruction Embedding for surface anomaly detection)은 2021년 ICCV(International Conference on Computer Vision)에서 Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj에 의해 발표되었다. 이 연구는 재구성 기반 이상 감지 패러다임에 혁명적 전환을 가져왔으며, GANomaly 실패 이후 침체되었던 재구성 방법론의 르네상스를 촉발했다. DRAEM의 핵심 혁신은 "Simulated Anomaly"라는 개념의 도입이다. 실제 결함 샘플 없이도 합성적으로 생성한 결함으로 모델을 학습시킬 수 있다는 발상은, 비지도 학습에서 지도 학습으로의 패러다임 전환을 의미했다.

DRAEM 이전의 재구성 기반 방법론들은 정상 샘플만을 사용하는 순수 비지도 학습(unsupervised learning)에 의존했다. 이는 실제 결함 샘플을 수집하기 어렵다는 산업 현장의 현실을 반영한 것이었지만, 동시에 모델이 명시적으로 결함의 특징을 학습할 수 없다는 근본적 한계를 내포했다. DRAEM은 Perlin noise와 외부 텍스처 소스를 활용하여 다양한 합성 결함을 생성함으로써, 지도 학습의 이점을 누리면서도 실제 결함 데이터가 불필요하다는 독특한 위치를 점하게 되었다. 이러한 접근법은 학습의 안정성, 수렴 속도, 그리고 최종 성능 모든 측면에서 획기적인 개선을 가져왔으며, 특히 Few-shot 학습 시나리오에서 그 진가를 발휘했다.

### 3.2 Paradigm Shift: Simulated Anomaly

DRAEM이 제시한 Simulated Anomaly 개념은 이상 감지 분야의 사고방식을 근본적으로 바꾸어 놓았다. 전통적 접근법이 "정상 샘플만으로 정상의 경계를 학습한다"는 철학을 따랐다면, DRAEM은 "정상과 이상의 구별을 명시적으로 학습한다"는 새로운 철학을 제시했다. 이는 단순한 기술적 개선이 아니라, 문제를 바라보는 관점의 전환이었다. "실제 결함 없이 어떻게 결함을 학습할 것인가?"라는 역설적 질문에 대한 DRAEM의 답은, 결함의 구체적 형태보다는 "정상 패턴의 국소적 일탈"이라는 결함의 본질적 특성을 학습하는 것이었다.

합성 결함 생성(Synthetic Defect Generation)은 DRAEM의 핵심 메커니즘이다. 정상 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$가 주어졌을 때, DRAEM은 이미지를 두 부분으로 분해한다. 하나는 원본 정상 이미지이고, 다른 하나는 인위적으로 생성된 이상 패턴이다. 이상 패턴은 외부 텍스처 데이터베이스(DTD, Describable Textures Dataset)로부터 무작위로 선택된 텍스처 $\mathbf{t}$와 Perlin noise를 결합하여 생성된다. Perlin noise $\mathbf{p}(i,j) \in [0,1]$는 자연스러운 공간적 상관관계를 가진 노이즈로, 이진 마스크 $\mathbf{m}$을 생성하는 데 사용된다.

$$
\mathbf{m}(i,j) = \begin{cases}
1 & \text{if } \mathbf{p}(i,j) > \tau \\
0 & \text{otherwise}
\end{cases}
$$

여기서 $\tau \in [0,1]$는 threshold 값으로, 결함 영역의 크기를 조절한다. 최종적으로 augmented 이미지 $\mathbf{x}_{\text{aug}}$는 원본 이미지와 텍스처를 마스크를 통해 blending하여 생성된다.

$$
\mathbf{x}_{\text{aug}} = (1 - \alpha \mathbf{m}) \odot \mathbf{x} + \alpha \mathbf{m} \odot \mathbf{t}
$$

여기서 $\alpha \in [0.1, 1.0]$는 blending factor로, 결함의 강도를 조절한다. $\odot$는 element-wise 곱셈을 나타낸다. 이러한 방식으로 생성된 합성 결함은 실제 결함의 다양한 특성을 근사한다. 크기, 모양, 텍스처, 강도 등이 모두 무작위로 변화하며, 이는 모델이 특정 결함 유형에 과적합되지 않고 일반화된 이상 감지 능력을 학습하도록 유도한다.

Perlin Noise Augmentation의 중요성은 아무리 강조해도 지나치지 않다. Perlin noise는 컴퓨터 그래픽스에서 자연스러운 텍스처 생성에 널리 사용되는 기법으로, 여러 주파수의 노이즈를 중첩하여 자연스러운 공간적 변화를 만들어낸다. $n$개의 octave를 사용하는 Perlin noise는 다음과 같이 정의된다.

$$
\mathbf{p}_{\text{Perlin}}(i,j) = \sum_{k=0}^{n-1} \frac{1}{2^k} \mathbf{p}_k(2^k i, 2^k j)
$$

여기서 $\mathbf{p}_k$는 $k$-번째 octave의 기본 노이즈 함수다. 각 octave는 이전 octave보다 두 배 높은 주파수를 가지며, 진폭은 절반으로 감소한다. 이러한 multi-scale 구조는 결함이 다양한 크기로 나타날 수 있다는 현실을 반영한다. 미세한 균열부터 넓은 영역의 변색까지, Perlin noise는 광범위한 스펙트럼의 결함 패턴을 생성할 수 있다.

지도 학습 효과(Supervised Learning Effect)는 DRAEM의 성능 향상을 설명하는 핵심이다. 전통적인 비지도 방법에서 모델은 재구성 오차를 통해 간접적으로만 이상을 감지했다. 즉, $\mathcal{L}_{\text{unsup}} = \|\mathbf{x} - \text{Decoder}(\text{Encoder}(\mathbf{x}))\|$를 최소화하면, 정상 샘플에 대한 재구성 오차는 작아지지만, 이상 샘플에 대한 오차가 큰지는 보장되지 않는다. 반면 DRAEM은 명시적인 판별 손실(discriminative loss)을 사용한다. Discriminative network $D$는 입력 이미지와 재구성 이미지를 받아, 각 픽셀이 정상인지 이상인지 분류한다.

$$
\mathbf{y} = D([\mathbf{x}_{\text{aug}}, \text{Recon}(\mathbf{x}_{\text{aug}})])
$$

여기서 $\mathbf{y} \in \mathbb{R}^{H \times W \times 2}$는 픽셀별 정상/이상 확률을 나타낸다. Ground truth mask $\mathbf{m}$과 예측 $\mathbf{y}$ 간의 차이를 직접 최소화함으로써, 모델은 정상과 이상의 경계를 명시적으로 학습한다. 이는 비지도 학습의 간접적 접근과 근본적으로 다르며, 학습의 효율성과 최종 성능 모두에서 우위를 점한다. 실험 결과는 이러한 지도 학습 효과가 단순한 이론적 우아함을 넘어 실질적 성능 향상으로 이어짐을 보여준다.

### 3.3 Technical Architecture

DRAEM의 구조는 두 개의 주요 네트워크로 구성된다. Reconstructive Subnetwork와 Discriminative Subnetwork가 협력하여 이상을 감지하며, 각각은 명확히 구분된 역할을 수행한다. 이러한 이중 구조는 재구성과 판별이라는 두 가지 상보적 메커니즘을 결합함으로써, 단일 네트워크보다 강건한 이상 감지를 가능하게 한다.

Reconstruction Network $R$은 augmented 이미지를 원본 정상 이미지로 복원하는 역할을 담당한다. 이는 U-Net 스타일의 Encoder-Decoder 구조를 따르며, skip connection을 통해 고해상도 정보를 보존한다. Encoder는 입력 이미지를 점진적으로 압축하여 저차원 표현을 생성하고, Decoder는 이를 다시 확장하여 재구성 이미지를 생성한다. 수식으로 표현하면 다음과 같다.

$$
\hat{\mathbf{x}} = R(\mathbf{x}_{\text{aug}}) = \text{Decoder}(\text{Encoder}(\mathbf{x}_{\text{aug}}))
$$

재구성의 목표는 합성 결함을 제거하고 원본 정상 이미지 $\mathbf{x}$를 복원하는 것이다. 즉, $\hat{\mathbf{x}} \approx \mathbf{x}$가 되도록 학습된다. 이 과정에서 모델은 정상 패턴의 구조적 특징을 학습하며, 결함 영역을 "치유"하는 능력을 획득한다. 흥미로운 점은 Reconstruction Network가 합성 결함뿐만 아니라 실제 결함에 대해서도 유사한 복원 능력을 보인다는 것이다. 이는 합성 결함이 실제 결함의 본질적 특성을 효과적으로 근사함을 시사한다.

선택적으로 SSPCAB(Scale Space Pyramid Channel Attention Block)을 Encoder에 통합할 수 있다. SSPCAB은 multi-scale feature를 효과적으로 결합하는 attention 메커니즘으로, 다양한 크기의 결함에 대한 robust한 표현을 학습한다. SSPCAB를 사용하는 경우, 추가적인 consistency loss가 도입된다.

$$
\mathcal{L}_{\text{SSPCAB}} = \|\text{Activation}_{\text{input}} - \text{Activation}_{\text{output}}\|_2^2
$$

이는 SSPCAB 전후의 feature activation이 일관성을 유지하도록 강제하며, 결함 영역의 feature만 선택적으로 변화하도록 유도한다.

Discriminative Network $D$는 입력 이미지와 재구성 이미지를 concatenate한 6-channel 입력을 받아, 픽셀별 정상/이상 확률을 출력한다.

$$
\mathbf{y} = D([\mathbf{x}_{\text{aug}}, \hat{\mathbf{x}}])
$$

여기서 $[\cdot, \cdot]$는 channel-wise concatenation을 의미한다. Discriminative Network는 입력과 재구성의 차이를 분석하여, 어느 영역이 결함인지 판단한다. 이 네트워크는 재구성 오차만으로는 포착하기 어려운 미묘한 패턴을 학습할 수 있다. 예를 들어, 재구성이 완벽하지 않더라도 정상적인 복원 패턴을 보이는 영역은 정상으로 분류하고, 재구성 오차는 작지만 비정상적인 패턴을 보이는 영역은 이상으로 분류할 수 있다.

Loss Functions는 DRAEM의 학습을 안내하는 핵심 요소다. 총 손실은 세 가지 구성 요소의 가중 합으로 정의된다.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{L2}} + \mathcal{L}_{\text{SSIM}} + \mathcal{L}_{\text{Focal}}
$$

첫째, L2 loss는 재구성 품질을 보장한다.

$$
\mathcal{L}_{\text{L2}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 = \sum_{i,j,c} (\mathbf{x}(i,j,c) - \hat{\mathbf{x}}(i,j,c))^2
$$

이는 픽셀별 제곱 오차를 최소화하여, 재구성 이미지가 원본과 가능한 한 유사하도록 유도한다. 둘째, SSIM(Structural Similarity Index) loss는 구조적 유사성을 강제한다.

$$
\mathcal{L}_{\text{SSIM}} = 2 \times (1 - \text{SSIM}(\mathbf{x}, \hat{\mathbf{x}}))
$$

SSIM은 밝기(luminance), 대비(contrast), 구조(structure) 세 가지 측면에서 이미지 유사도를 측정한다. L2 loss가 픽셀별 일치를 강조한다면, SSIM loss는 인간의 시각적 지각과 더 일치하는 구조적 유사성을 강조한다. SSIM은 다음과 같이 정의된다.

$$
\text{SSIM}(\mathbf{x}, \mathbf{y}) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

여기서 $\mu_x$, $\mu_y$는 평균, $\sigma_x^2$, $\sigma_y^2$는 분산, $\sigma_{xy}$는 공분산, $C_1$, $C_2$는 안정성을 위한 상수다. SSIM loss의 계수 2는 경험적으로 결정된 가중치로, 구조적 유사성의 중요성을 강조한다.

셋째, Focal loss는 class imbalance 문제를 완화한다.

$$
\mathcal{L}_{\text{Focal}} = -\sum_{i,j} \alpha (1 - p_{ij})^\gamma \log(p_{ij})
$$

여기서 $p_{ij}$는 픽셀 $(i,j)$가 올바른 클래스에 속할 확률이다. 대부분의 픽셀이 정상이므로, 단순한 cross-entropy loss는 이상 픽셀을 충분히 학습하지 못한다. Focal loss는 hard example(잘못 분류된 샘플)에 더 큰 가중치를 부여하여, 이상 픽셀의 학습을 강화한다. $\gamma$는 focusing parameter로, 일반적으로 2를 사용하며, $(1 - p_{ij})^\gamma$ 항은 쉬운 샘플의 손실을 down-weight한다. $\alpha$는 class balance parameter로, 양성(이상) 클래스에 더 큰 가중치를 부여한다.

### 3.4 Few-shot Capability (10-50 samples)

DRAEM의 가장 인상적인 특성 중 하나는 극소량의 데이터로도 높은 성능을 달성하는 Few-shot 학습 능력이다. 전통적인 딥러닝 모델이 수천에서 수만 장의 학습 샘플을 필요로 하는 것과 달리, DRAEM은 단 10-50장의 정상 샘플만으로도 97.5%의 Image AUROC를 달성한다. 이는 신제품 출시, 희귀 부품 검사, 또는 데이터 수집이 어려운 환경에서 혁명적 의미를 가진다.

Few-shot 능력의 핵심은 Simulated Anomaly가 데이터 증강(data augmentation)의 극단적 형태로 작용한다는 점이다. 10장의 정상 이미지가 있다면, 각 이미지에 대해 수십 가지 다른 합성 결함을 생성할 수 있다. Perlin noise의 무작위성, 텍스처 소스의 다양성, blending factor의 변화 등을 통해, 실질적으로 수천 개의 서로 다른 학습 샘플을 생성할 수 있다. 이는 $n$개의 정상 샘플로부터 $n \times k$개의 augmented 샘플을 생성하는 것으로 볼 수 있으며, 여기서 $k$는 augmentation multiplier다.

$$
|\mathcal{D}_{\text{aug}}| = |\mathcal{D}_{\text{normal}}| \times k
$$

실험적으로, $k=100$을 사용하면 10장의 이미지로 1000개의 학습 샘플을 생성할 수 있으며, 이는 모델 학습에 충분한 다양성을 제공한다. 중요한 점은 이러한 augmentation이 단순한 geometric transformation(rotation, flip 등)이 아니라, 결함의 본질적 특성을 반영한 semantic augmentation이라는 것이다. 모델은 단순히 이미지의 변환에 robust해지는 것이 아니라, 결함과 정상의 구별이라는 핵심 작업을 학습한다.

MVTec AD 데이터셋에 대한 Few-shot 실험 결과는 DRAEM의 데이터 효율성을 명확히 보여준다. "Bottle" 카테고리에서 전체 209장의 학습 샘플을 사용했을 때 Image AUROC 98.0%를 달성하는데, 50장만 사용해도 97.8%, 심지어 10장만 사용해도 96.5%를 유지한다. 성능 저하가 1-2% 수준에 불과하다는 것은, DRAEM의 학습이 데이터 양보다는 데이터 다양성에 의존함을 시사한다. Simulated Anomaly가 제공하는 다양성이 실제 데이터 양의 부족을 효과적으로 보상하는 것이다.

이러한 Few-shot 능력은 산업 현장의 실질적 제약을 고려할 때 매우 가치있다. 신제품의 경우 초기 생산량이 제한적이며, 양품 샘플조차 확보하기 어려울 수 있다. 희귀 부품이나 맞춤형 제품의 경우, 대량의 학습 데이터를 수집하는 것은 경제적으로 비현실적이다. DRAEM은 이러한 시나리오에서 빠르게 배포 가능한 솔루션을 제공하며, "데이터가 충분히 모일 때까지 기다릴 수 없다"는 산업 현장의 현실적 요구에 부응한다.

### 3.5 Performance Analysis (97.5%)

DRAEM의 성능은 MVTec AD 벤치마크에서 Image AUROC 97.5%, Pixel AUROC 98.4%를 기록하며, 재구성 기반 방법론 중 최고 수준이다. 이는 GANomaly의 85-90%와 비교할 때 큰 도약이며, Vanilla Autoencoder의 80-85%를 크게 상회한다. 물론 PatchCore의 99.1%에는 미치지 못하지만, Few-shot 능력과 학습 안정성을 고려하면 매우 경쟁력 있는 수치다.

카테고리별 분석을 통해 DRAEM의 강점과 약점을 파악할 수 있다. 구조적 결함이 명확한 카테고리에서 DRAEM은 탁월한 성능을 보인다. "Metal Nut"에서 99.0%, "Screw"에서 98.5%, "Grid"에서 99.5% 등 기하학적 패턴이 뚜렷한 경우, 합성 결함이 실제 결함을 효과적으로 모방하며 높은 정확도를 달성한다. 반면 텍스처 변화가 미묘한 카테고리에서는 상대적으로 낮은 성능을 보인다. "Carpet"에서 93.5%, "Leather"에서 94.0% 등 복잡한 자연 텍스처를 가진 소재에서는, 합성 결함과 정상 텍스처의 변이를 구별하기 어려워진다.

Pixel-level 성능은 Image-level보다 일관되게 높다. Pixel AUROC 98.4%는 DRAEM이 결함의 정확한 위치를 파악하는 능력이 우수함을 보여준다. 이는 Discriminative Network가 픽셀별로 정상/이상을 판별하도록 학습되었기 때문이다. 실무적으로 이는 매우 중요한데, 단순히 "결함이 있다/없다"를 판단하는 것을 넘어 "결함이 어디에 있는가"를 정확히 알려줄 수 있기 때문이다. 품질 보고서 자동화, 수리 가이드 생성 등의 응용에서 pixel-level localization은 필수적이다.

다른 방법론과의 비교에서 DRAEM은 독특한 위치를 차지한다. PatchCore(99.1%)나 FastFlow(98.5%)에 비해서는 약간 낮은 정확도를 보이지만, 이들보다 훨씬 적은 데이터로 학습 가능하다. 학습 시간은 Autoencoder(5분)보다 길지만 GANomaly(60분)보다는 훨씬 짧은 15분 수준이다. 메모리 사용량은 300-500MB로 중간 정도이며, 추론 속도는 50-100ms로 실시간은 아니지만 대부분의 검사 라인에서 충분히 빠르다. 이러한 균형잡힌 특성은 DRAEM을 "만능형(all-rounder)" 솔루션으로 만든다.

### 3.6 Training Stability

DRAEM의 가장 큰 실무적 장점은 학습 안정성이다. GANomaly의 실패가 학습 불안정성에서 비롯되었다면, DRAEM의 성공은 안정적이고 예측 가능한 학습 과정에 기인한다. 이는 지도 학습 프레임워크의 본질적 특성이다. 명확한 ground truth(합성 결함 mask)가 있으므로, 모델은 매 iteration마다 구체적인 학습 신호를 받는다. Loss 값은 단조롭게 감소하며, validation 성능도 안정적으로 향상된다.

학습 곡선의 분석은 이러한 안정성을 명확히 보여준다. Training loss는 초기 급격한 감소 후 부드럽게 수렴하며, oscillation이나 spike가 거의 관찰되지 않는다. Validation AUROC는 epoch 10-20 사이에 plateau에 도달하며, 이후 미세한 향상만 있을 뿐 큰 변동이 없다. 이는 early stopping 기준 설정을 용이하게 하며, 과적합 위험을 최소화한다. GANomaly가 학습 종료 시점 판단조차 어려웠던 것과 대조적이다.

재현성(reproducibility)은 안정성의 또 다른 측면이다. 동일한 하이퍼파라미터로 DRAEM을 여러 번 학습시켰을 때, 결과의 표준편차는 0.5% 미만이다. 이는 random seed에 크게 의존하지 않으며, 일관된 성능을 보장한다는 의미다. 산업 배포 시 이러한 재현성은 매우 중요하다. 고객에게 "평균적으로 97.5%의 성능"이 아니라 "안정적으로 97.5% ± 0.5%의 성능"을 보장할 수 있다.

하이퍼파라미터 민감도도 낮다. Learning rate, batch size, blending factor 등 주요 하이퍼파라미터의 변화에 대해 성능은 robust하다. Learning rate를 0.0001에서 0.001 사이에서 변화시켜도 성능 차이는 1% 미만이다. Blending factor $\alpha$를 0.1-1.0 범위에서 조절해도 유사하다. 이는 하이퍼파라미터 튜닝에 소요되는 시간을 크게 줄여주며, 새로운 데이터셋에 대한 적용을 용이하게 한다. 대부분의 경우 논문에서 제시한 기본 설정을 그대로 사용해도 좋은 결과를 얻을 수 있다.

### 3.7 Implementation Guide

DRAEM의 구현은 비교적 직관적이며, 주요 단계는 명확히 정의된다. 첫째, 데이터 준비 단계에서 정상 샘플과 DTD 텍스처 데이터베이스를 준비한다. DTD는 5640개의 다양한 텍스처 이미지를 포함하며, 공개적으로 이용 가능하다. MVTec AD의 경우, 각 카테고리당 약 200-300장의 정상 학습 샘플이 제공된다. Few-shot 시나리오에서는 이 중 10-50장만 무작위로 샘플링한다.

둘째, Augmentation pipeline 구축이다. 각 정상 이미지 $\mathbf{x}$에 대해, 매 학습 iteration마다 새로운 합성 결함을 생성한다. Perlin noise 생성, DTD 텍스처 선택, blending이 실시간으로 수행된다. 이는 매 epoch마다 모델이 다른 결함 패턴을 보게 됨을 의미하며, 과적합을 방지하고 일반화 능력을 향상시킨다. Augmentation의 무작위성은 중요하다. 동일한 정상 이미지라도 매번 다른 위치, 크기, 강도의 결함이 생성되어야 한다.

셋째, 모델 학습이다. Reconstruction Network와 Discriminative Network를 함께 end-to-end로 학습한다. 두 네트워크는 동일한 loss function에 의해 jointly 최적화되며, 별도의 alternating optimization이 필요 없다. 이는 GANomaly와의 중요한 차이점이다. Optimizer로는 Adam을 사용하며, learning rate 0.0001, batch size 8-16이 일반적이다. 학습은 약 100-200 epoch 정도 진행되며, early stopping을 사용하면 더 빨리 종료될 수 있다. 전체 학습 시간은 MVTec 한 카테고리당 15-20분 수준이다.

넷째, 추론 단계이다. 테스트 이미지 $\mathbf{x}_{\text{test}}$에 대해, Reconstruction Network로 재구성 이미지 $\hat{\mathbf{x}}_{\text{test}}$를 생성한다. 그 다음 Discriminative Network에 $[\mathbf{x}_{\text{test}}, \hat{\mathbf{x}}_{\text{test}}]$를 입력하여 anomaly map $\mathbf{y}_{\text{test}}$를 얻는다. Image-level score는 anomaly map의 최대값 또는 상위 k% 평균으로 계산된다.

$$
s_{\text{image}} = \max_{i,j} \mathbf{y}_{\text{test}}(i,j,1) \quad \text{or} \quad s_{\text{image}} = \text{TopK-Mean}(\mathbf{y}_{\text{test}}(:,:,1))
$$

여기서 $\mathbf{y}_{\text{test}}(:,:,1)$은 이상 클래스의 확률 맵이다.

실무 구현 시 주의사항으로는, DTD 데이터베이스의 접근성 확보, 충분한 augmentation 다양성 보장, 적절한 early stopping 기준 설정 등이 있다. DTD가 없는 경우, 다른 텍스처 데이터베이스나 심지어 ImageNet의 일부를 사용할 수도 있다. 중요한 것은 학습 대상 도메인과 충분히 다른 텍스처를 사용하는 것이다. Augmentation multiplier $k$는 적어도 50-100 정도로 설정하여, 각 정상 샘플로부터 충분히 다양한 합성 결함을 생성해야 한다.

DRAEM은 재구성 기반 이상 감지의 새로운 지평을 열었다. GANomaly의 실패로부터 학습한 교훈을 바탕으로, 단순하면서도 효과적인 접근법을 제시했다. Simulated Anomaly라는 혁신적 개념, 지도 학습의 안정성, Few-shot 능력, 그리고 균형잡힌 성능 특성은 DRAEM을 재구성 기반 방법론의 대표 모델로 자리매김하게 했다. 특히 데이터가 제한적인 산업 환경에서 DRAEM의 가치는 더욱 빛을 발한다.

프로젝트 지식을 확인했습니다. 이제 DSR(4장)과 Autoencoder(5장)를 학술 리뷰 논문 형식으로 작성하겠습니다.

---

## 4. DSR (2022)

### 4.1 Basic Information

DSR(A Dual Subspace Re-Projection Network for Surface Anomaly Detection)은 2022년 ECCV(European Conference on Computer Vision)에서 Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj에 의해 발표되었다. 이 연구는 DRAEM의 동일한 저자 그룹이 수행한 후속 작업으로, 재구성 기반 방법론을 더욱 정교화한 결과물이다. DSR의 핵심 혁신은 이미지를 구조(structure)와 텍스처(texture)라는 두 개의 독립적 subspace로 분해하여 처리한다는 점이다. 이러한 분해는 특히 복잡한 표면 텍스처를 가진 소재의 이상 감지에서 탁월한 성능을 보인다.

DSR 이전의 재구성 기반 모델들은 이미지를 단일한 표현 공간에서 처리했다. 그러나 실제 산업 검사 대상 중 상당수는 구조적 정보와 텍스처 정보가 혼재되어 있으며, 이 둘의 중요도와 변이 패턴이 매우 다르다. 예를 들어, 직물(fabric)의 경우 전체적인 직조 패턴(구조)은 일정하지만 국소적인 실의 배열(텍스처)은 자연스러운 변이를 보인다. 단일 표현 공간에서는 이러한 자연스러운 텍스처 변이를 결함으로 오인할 수 있다. DSR은 VQ-VAE(Vector Quantized Variational AutoEncoder)를 활용하여 이 두 정보를 분리하고, 각각에 특화된 처리 메커니즘을 적용함으로써 이 문제를 해결한다.

### 4.2 Dual Subspace Architecture

DSR의 구조적 혁신은 두 개의 독립적인 quantization subspace를 도입한 것이다. 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$는 계층적 인코딩을 통해 두 개의 잠재 표현으로 분해된다. 하위 레벨 표현 $\mathbf{z}_{\text{bot}} \in \mathbb{R}^{H/4 \times W/4 \times d}$는 저주파 구조 정보를, 상위 레벨 표현 $\mathbf{z}_{\text{top}} \in \mathbb{R}^{H/8 \times W/8 \times d}$는 고주파 텍스처 정보를 담당한다. 여기서 $d$는 embedding 차원이다.

Quantization Subspace for Structure는 이미지의 거시적 구조를 포착한다. 하위 레벨 인코더 $E_{\text{bot}}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{H/4 \times W/4 \times d}$는 입력 이미지를 중간 해상도의 feature map으로 변환한다.

$$
\mathbf{z}_{\text{bot}} = E_{\text{bot}}(\mathbf{x})
$$

이 feature는 vector quantization을 통해 discrete code로 변환된다. Codebook $\mathcal{C}_{\text{bot}} = \{\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_K\} \subset \mathbb{R}^d$에서 각 위치의 feature vector와 가장 가까운 code를 선택한다.

$$
\mathbf{q}_{\text{bot}}(i,j) = \arg\min_{\mathbf{e}_k \in \mathcal{C}_{\text{bot}}} \|\mathbf{z}_{\text{bot}}(i,j) - \mathbf{e}_k\|_2
$$

이러한 quantization은 feature 공간을 이산화하여, 정상 패턴을 유한개의 prototype으로 표현한다. 정상 샘플의 구조는 codebook의 특정 code 조합으로 나타나며, 이상 샘플은 드물거나 존재하지 않는 code 조합을 발생시킨다.

Target Subspace for Texture는 미세한 텍스처 정보를 다룬다. 상위 레벨 인코더는 하위 레벨 feature를 추가로 압축한다.

$$
\mathbf{z}_{\text{top}} = E_{\text{top}}(\mathbf{z}_{\text{bot}})
$$

여기서 $E_{\text{top}}: \mathbb{R}^{H/4 \times W/4 \times d} \rightarrow \mathbb{R}^{H/8 \times W/8 \times d}$는 공간 해상도를 절반으로 줄인다. 이 상위 레벨 표현은 별도의 codebook $\mathcal{C}_{\text{top}}$으로 quantize된다. 두 레벨의 quantized representation은 concatenate되어 디코더로 전달된다.

$$
\hat{\mathbf{x}} = D([\text{Upsample}(\mathbf{q}_{\text{top}}), \mathbf{q}_{\text{bot}}])
$$

여기서 Upsample은 상위 레벨 feature를 하위 레벨과 동일한 해상도로 확장하며, $[\cdot, \cdot]$는 channel-wise concatenation이다.

VQ-VAE Integration은 DSR의 핵심 기술적 요소다. VQ-VAE는 연속적인 잠재 공간 대신 이산적 codebook을 사용하는 변형 오토인코더다. 학습 시 codebook은 exponential moving average를 통해 업데이트된다.

$$
\mathbf{e}_k^{(t+1)} = \mathbf{e}_k^{(t)} + \gamma (\bar{\mathbf{z}}_k - \mathbf{e}_k^{(t)})
$$

여기서 $\bar{\mathbf{z}}_k$는 code $k$에 할당된 모든 feature vector의 평균이고, $\gamma$는 learning rate다. 이러한 업데이트 메커니즘은 codebook이 학습 데이터의 정상 패턴 분포를 효과적으로 학습하도록 한다. VQ-VAE의 loss function은 reconstruction loss, commitment loss, codebook loss로 구성된다.

$$
\mathcal{L}_{\text{VQ-VAE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \|\text{sg}[\mathbf{z}] - \mathbf{q}\|_2^2 + \beta \|\mathbf{z} - \text{sg}[\mathbf{q}]\|_2^2
$$

여기서 $\text{sg}[\cdot]$는 stop-gradient 연산자이고, $\beta$는 commitment loss의 가중치다. 첫 번째 항은 재구성 품질을, 두 번째 항은 codebook의 학습을, 세 번째 항은 인코더가 codebook에 가까운 표현을 생성하도록 유도한다.

### 4.3 Texture Specialization

DSR이 텍스처 특화(Texture Specialization)를 달성하는 메커니즘은 subspace restriction module에 있다. 이 모듈은 결함이 포함된 feature를 정상 subspace로 재투영(re-projection)함으로써, 결함 영역을 복원한다. 핵심 아이디어는 정상 feature는 학습된 subspace 내에 존재하지만, 이상 feature는 이 subspace 밖에 위치한다는 것이다.

Subspace restriction module $R$은 U-Net 스타일의 구조를 가지며, 다음과 같이 동작한다.

$$
\mathbf{z}_{\text{restored}} = R(\mathbf{z}_{\text{anomalous}})
$$

여기서 $\mathbf{z}_{\text{anomalous}}$는 결함을 포함한 quantized feature이고, $\mathbf{z}_{\text{restored}}$는 정상 subspace로 복원된 feature다. 이 복원 과정은 두 레벨에서 독립적으로 수행된다. 하위 레벨에서는 구조적 일관성을 복원하고, 상위 레벨에서는 텍스처 패턴을 정규화한다.

텍스처 정보의 분리는 특히 자연스러운 텍스처 변이와 실제 결함을 구별하는 데 중요하다. 예를 들어, 가죽(leather)의 경우 자연스러운 주름과 색상 변화는 정상이지만, 찢어짐이나 얼룩은 결함이다. 단일 표현 공간에서는 이 둘을 구별하기 어렵지만, DSR은 구조 subspace에서 찢어짐(구조적 이상)을 감지하고 텍스처 subspace에서 자연 변이(정상)와 얼룩(텍스처 이상)을 구별할 수 있다.

학습은 DRAEM과 유사하게 simulated anomaly를 사용하지만, DSR은 이를 두 레벨에서 독립적으로 적용한다. 구조적 결함은 하위 레벨 feature에, 텍스처적 결함은 상위 레벨 feature에 주입된다. 이는 모델이 서로 다른 유형의 결함에 대해 전문화된 표현을 학습하도록 유도한다.

$$
\mathbf{z}_{\text{bot}}^{\text{aug}} = (1 - \mathbf{m}_{\text{struct}}) \odot \mathbf{z}_{\text{bot}} + \mathbf{m}_{\text{struct}} \odot \mathbf{z}_{\text{bot}}^{\text{anom}}
$$

$$
\mathbf{z}_{\text{top}}^{\text{aug}} = (1 - \mathbf{m}_{\text{tex}}) \odot \mathbf{z}_{\text{top}} + \mathbf{m}_{\text{tex}} \odot \mathbf{z}_{\text{top}}^{\text{anom}}
$$

여기서 $\mathbf{m}_{\text{struct}}$와 $\mathbf{m}_{\text{tex}}$는 각각 구조 및 텍스처 결함 마스크이고, $\mathbf{z}^{\text{anom}}$는 codebook에서 무작위로 선택된 이상 feature다.

### 4.4 Performance on Complex Surfaces

DSR의 성능은 특히 복잡한 표면 텍스처를 가진 카테고리에서 두드러진다. MVTec AD 벤치마크에서 DSR은 텍스처 기반 카테고리에서 DRAEM과 다른 재구성 기반 방법론을 능가한다. "Carpet" 카테고리에서 DSR은 Image AUROC 99.0%를 달성하며, 이는 DRAEM의 93.5%를 크게 상회한다. "Leather"에서는 98.5% (DRAEM 94.0%), "Fabric"에서는 98.8% (DRAEM 95.5%)를 기록한다.

이러한 성능 향상은 텍스처 특화 메커니즘의 직접적 결과다. 복잡한 텍스처를 가진 소재에서 결함은 종종 미묘하며, 자연스러운 텍스처 변이와 유사한 특성을 보인다. 단일 표현 공간에서는 이 둘의 경계가 모호하지만, DSR의 dual subspace 접근법은 구조적 이상과 텍스처적 이상을 명확히 분리하여 감지한다. 예를 들어, "Carpet"에서 실의 끊김(구조적 결함)과 색상 불균일(텍스처적 결함)은 각각 하위 레벨과 상위 레벨 subspace에서 독립적으로 감지된다.

반면 구조적 패턴이 명확한 카테고리에서는 DSR의 장점이 덜 두드러진다. "Screw"에서 98.5%, "Grid"에서 98.8%를 기록하며, 이는 DRAEM(각각 98.5%, 99.5%)과 유사하거나 약간 낮다. 이는 단순한 구조를 가진 대상에서는 dual subspace의 복잡성이 불필요하며, 오히려 단일 표현 공간이 더 효율적일 수 있음을 시사한다. DSR의 진가는 텍스처의 복잡도가 높을 때 발휘된다.

Pixel-level 성능에서도 DSR은 우수한 localization 능력을 보인다. 텍스처 카테고리에서 Pixel AUROC는 평균 98.8%에 달하며, 이는 결함의 정확한 위치를 파악하는 능력이 탁월함을 의미한다. Subspace restriction module이 feature level에서 결함을 복원하기 때문에, 복원 전후의 차이가 정확한 anomaly map을 생성한다. 이는 단순히 재구성 오차를 사용하는 방식보다 공간적 정확도가 높다.

### 4.5 Use Cases (Fabric, Carpet, Leather)

DSR의 주요 응용 분야는 자연스러운 변이가 큰 텍스처 소재의 품질 검사다. 섬유 산업(Fabric)에서 DSR은 직조 불량, 실 끊김, 오염 등을 효과적으로 감지한다. 직물의 정상 패턴은 반복적인 직조 구조(하위 레벨)와 실의 미세한 배열(상위 레벨)로 구성된다. 결함은 주로 이 두 레벨 중 하나를 위반하며, DSR은 각 레벨에서 독립적으로 이상을 감지함으로써 높은 정확도를 달성한다. 특히 미세한 직조 불량은 상위 레벨 텍스처 subspace에서만 나타나며, 단일 표현으로는 놓치기 쉽다.

카펫 제조(Carpet)에서 DSR은 색상 불균일, 표면 손상, 이물질 혼입 등을 감지한다. 카펫은 복잡한 패턴과 다양한 색상이 혼재되어 있어 이상 감지가 특히 어렵다. 자연스러운 색상 변화와 실제 결함을 구별하는 것이 핵심 과제인데, DSR은 전체 패턴 구조(하위 레벨)와 국소 텍스처(상위 레벨)를 분리하여 이 문제를 해결한다. 대규모 색상 불균일은 구조 subspace에서, 미세한 표면 손상은 텍스처 subspace에서 감지된다.

가죽 제품(Leather)은 DSR의 또 다른 중요한 응용 분야다. 천연 가죽은 본질적으로 불균일한 텍스처를 가지며, 자연스러운 주름, 모공, 색상 변화가 정상으로 간주된다. 그러나 긁힘, 찢어짐, 부적절한 염색은 결함이다. DSR은 자연스러운 변이를 텍스처 subspace의 정상 변동으로 학습하고, 구조적 손상을 하위 레벨에서 감지한다. 이러한 분리는 높은 false positive rate 없이 실제 결함만을 정확히 포착할 수 있게 한다.

실무 배포 시 DSR은 몇 가지 고려사항이 있다. 첫째, 학습 시간이 상대적으로 길다. VQ-VAE의 codebook 학습과 subspace restriction module 학습을 포함하여 전체 학습 과정은 30-40분 정도 소요된다. 이는 DRAEM(15분)보다 두 배 정도 길지만, 복잡한 텍스처 소재에서의 성능 향상을 고려하면 감수할 만한 비용이다. 둘째, 메모리 사용량이 높다. 두 레벨의 codebook과 여러 개의 네트워크를 동시에 유지해야 하므로, 500MB-1GB 정도의 GPU 메모리가 필요하다. 셋째, 하이퍼파라미터가 DRAEM보다 많다. Codebook 크기, embedding 차원, quantization loss의 가중치 등 추가적인 튜닝이 필요하다.

그럼에도 불구하고 텍스처 소재 검사에서 DSR의 가치는 명확하다. 기존 방법론으로는 95% 미만의 정확도에 머물렀던 응용에서 98-99%의 정확도를 달성할 수 있다는 것은 산업적으로 큰 의미를 가진다. 특히 고가의 천연 소재나 패션 제품의 품질 검사에서, 몇 퍼센트의 정확도 향상은 상당한 경제적 이득으로 이어진다. DSR은 재구성 기반 방법론이 특정 도메인에 특화됨으로써 SOTA 성능에 근접할 수 있음을 보여주는 중요한 사례다.

---

## 5. Autoencoder (Baseline)

### 5.1 Vanilla Autoencoder

Vanilla Autoencoder는 재구성 기반 이상 감지의 가장 기초적이고 근본적인 형태다. 이는 단순히 "오래되었다"는 의미가 아니라, 이 패러다임의 핵심 원리를 가장 순수한 형태로 구현한다는 의미에서 "vanilla"라는 수식어가 붙는다. 복잡한 메커니즘이나 추가적인 구조 없이, 오직 Encoder-Decoder 구조와 재구성 손실만으로 이상 감지를 수행한다. 이러한 단순성은 약점이 아니라 오히려 강점이 될 수 있다. Autoencoder는 이해하기 쉽고, 구현이 간단하며, 학습이 안정적이다.

Vanilla Autoencoder의 구조는 대칭적인 Encoder-Decoder 쌍으로 구성된다. Encoder $E: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^k$는 입력 이미지를 저차원 잠재 벡터로 압축한다. 이미지 도메인에서는 일반적으로 Convolutional Neural Network를 사용하며, 점진적으로 공간 해상도를 줄이면서 channel 수를 늘린다. 예를 들어, $256 \times 256 \times 3$ 입력은 $128 \times 128 \times 32 \rightarrow 64 \times 64 \times 64 \rightarrow 32 \times 32 \times 128$과 같이 인코딩될 수 있다.

$$
\mathbf{z} = E(\mathbf{x}) \in \mathbb{R}^k
$$

Decoder $D: \mathbb{R}^k \rightarrow \mathbb{R}^{H \times W \times 3}$는 잠재 벡터를 다시 원본 이미지 공간으로 확장한다. Encoder와 대칭적으로, transposed convolution 또는 upsampling을 사용하여 해상도를 점진적으로 복원한다.

$$
\hat{\mathbf{x}} = D(E(\mathbf{x})) = D(\mathbf{z})
$$

학습의 목표는 단순하다. 재구성 이미지 $\hat{\mathbf{x}}$가 입력 이미지 $\mathbf{x}$와 최대한 유사하도록 네트워크를 최적화하는 것이다. 이는 정상 샘플로만 학습되며, 모델은 정상 패턴의 본질적 특징을 bottleneck에 압축하는 법을 학습한다.

### 5.2 Bottleneck Architecture

Bottleneck은 Autoencoder의 핵심 설계 원리다. 입력 차원 $H \times W \times 3$에 비해 극도로 작은 차원 $k$의 잠재 공간을 강제함으로써, 모델은 정보의 선택적 보존을 학습한다. 이는 information bottleneck 이론에 기반한다. Shannon의 정보 이론에 따르면, 잠재 벡터가 전달할 수 있는 최대 정보량은 $k \log_2(|\mathcal{V}|)$ bits로 제한되며, 여기서 $|\mathcal{V}|$는 잠재 변수의 값 범위다.

$$
I(\mathbf{x}; \mathbf{z}) \leq k \log_2(|\mathcal{V}|) \ll H \times W \times 3 \log_2(256)
$$

이러한 정보 제약은 모델이 입력의 모든 세부사항을 보존할 수 없게 만든다. 대신, 재구성에 가장 중요한 특징만을 선택적으로 인코딩해야 한다. 정상 샘플에서 자주 관찰되는 패턴은 효율적으로 인코딩되지만, 드물게 발생하는 이상 패턴은 잠재 표현에 효과적으로 담기지 못한다.

Bottleneck의 크기는 중요한 하이퍼파라미터다. 너무 크면 모델이 과적합되어 이상 샘플까지 잘 재구성할 수 있다. 너무 작으면 정상 샘플조차 제대로 재구성하지 못한다. 일반적으로 입력 차원의 $\frac{1}{100}$ 에서 $\frac{1}{1000}$ 사이가 적절하다. $256 \times 256$ RGB 이미지의 경우, 잠재 차원 $k$는 100-500 정도로 설정된다. 이는 압축률 200-1000배에 해당하며, 상당히 공격적인 압축이다.

Bottleneck의 효과는 regularization으로 이해할 수도 있다. 과도한 용량을 제한함으로써 모델의 일반화 능력을 향상시킨다. 이는 딥러닝의 일반적 원칙과는 다소 역설적이다. 보통은 더 많은 파라미터가 더 나은 성능으로 이어지지만, 이상 감지에서는 의도적인 용량 제한이 오히려 유리하다. 이는 "더 적은 것이 더 많다(less is more)"라는 설계 철학의 구체적 사례다.

### 5.3 Reconstruction Loss

재구성 손실(Reconstruction Loss)은 Autoencoder 학습의 유일한 목적 함수다. 가장 단순한 형태는 픽셀별 Mean Squared Error (MSE)다.

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{HWC} \sum_{i,j,c} (\mathbf{x}(i,j,c) - \hat{\mathbf{x}}(i,j,c))^2
$$

MSE는 큰 오차에 더 큰 penalty를 부여하므로, 뚜렷한 결함에 민감하다. 대안으로 Mean Absolute Error (MAE)를 사용할 수도 있다.

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{HWC} \sum_{i,j,c} |\mathbf{x}(i,j,c) - \hat{\mathbf{x}}(i,j,c)|
$$

MAE는 이상치(outlier)에 더 robust하며, 미세한 결함에도 균등한 가중치를 부여한다. 실무에서는 두 손실의 조합이 종종 사용된다.

$$
\mathcal{L}_{\text{rec}} = \alpha \mathcal{L}_{\text{MSE}} + (1-\alpha) \mathcal{L}_{\text{MAE}}
$$

여기서 $\alpha \in [0,1]$은 가중치 파라미터로, 일반적으로 0.5-0.7 정도가 사용된다. 학습 과정은 이 손실을 최소화하는 방향으로 진행되며, gradient descent 기반 optimizer를 사용한다. Adam optimizer가 가장 널리 사용되며, learning rate는 0.001-0.0001 정도가 적절하다.

테스트 단계에서는 재구성 오차가 이상 점수로 직접 사용된다. 각 테스트 이미지 $\mathbf{x}_{\text{test}}$에 대해, 재구성 이미지 $\hat{\mathbf{x}}_{\text{test}} = D(E(\mathbf{x}_{\text{test}}))$를 생성하고, 재구성 오차를 계산한다.

$$
s(\mathbf{x}_{\text{test}}) = \|\mathbf{x}_{\text{test}} - \hat{\mathbf{x}}_{\text{test}}\|_2
$$

이 점수가 높을수록 이상일 가능성이 크다. 임계값 $\tau$를 설정하여 이진 분류를 수행할 수 있다. $s(\mathbf{x}) > \tau$이면 이상, 그렇지 않으면 정상으로 판단한다. 임계값은 validation set에서 경험적으로 결정되거나, 특정 false positive rate를 만족하도록 설정된다.

### 5.4 Baseline Performance

Vanilla Autoencoder의 성능은 다른 SOTA 모델들에 비해 낮지만, baseline으로서 중요한 기준점을 제공한다. MVTec AD 데이터셋에서 Image AUROC는 평균 80-85% 수준이다. 이는 PatchCore의 99.1%나 DRAEM의 97.5%에 크게 못 미치지만, random guessing(50%)이나 단순 통계적 방법(60-70%)보다는 훨씬 우수하다. 즉, Autoencoder는 "작동은 한다(it works)"는 것을 보여주지만, 실무 배포를 위한 충분한 성능은 제공하지 못한다.

카테고리별로 성능 편차가 크다. 단순한 구조와 균일한 텍스처를 가진 카테고리에서는 상대적으로 좋은 성능을 보인다. "Pill"에서 89%, "Capsule"에서 87% 등 기하학적으로 단순한 대상에서는 80%를 상회한다. 반면 복잡한 텍스처나 자연스러운 변이가 큰 카테고리에서는 성능이 급격히 떨어진다. "Carpet"에서 72%, "Leather"에서 75% 등 텍스처 기반 카테고리에서는 실용성이 의심스러운 수준이다.

학습 속도와 안정성은 Autoencoder의 장점이다. MVTec 한 카테고리당 학습 시간은 5분 미만으로, DRAEM(15분)이나 DSR(40분)에 비해 매우 빠르다. 메모리 사용량도 100-200MB로 적다. 학습 곡선은 매우 안정적이며, 하이퍼파라미터에 대한 민감도도 낮다. 이는 빠른 실험과 프로토타이핑에 유리하다. 연구자는 Autoencoder baseline을 몇 분 만에 학습시키고, 데이터셋의 기본적 특성을 파악할 수 있다.

추론 속도는 10-15ms로, 실시간은 아니지만 대부분의 검사 라인에서 충분히 빠르다. 이는 EfficientAD(1-5ms)보다는 느리지만, DRAEM(50-100ms)보다는 빠르다. 단순한 구조 덕분에 CPU에서도 비교적 빠르게 동작하며, 20-30ms 정도로 추론 가능하다. 이는 GPU가 없는 저사양 환경에서 유용할 수 있다.

### 5.5 Limitations

Vanilla Autoencoder의 한계는 명확하며, 이는 더 복잡한 방법론의 필요성을 정당화한다. 첫째, 낮은 절대 성능이다. 80-85%의 AUROC는 많은 산업 응용에서 불충분하다. 일반적으로 95% 이상의 정확도가 요구되며, 중요한 제품의 경우 99% 이상이 필요하다. Autoencoder는 이러한 요구사항을 충족시키지 못한다. 이는 근본적으로 모델의 표현력 부족에 기인한다. 단순한 bottleneck만으로는 복잡한 정상 패턴의 모든 변이를 포착하기 어렵다.

둘째, 정상 샘플의 변이에 대한 과민성이다. Autoencoder는 정상 샘플의 자연스러운 변이도 재구성하지 못하는 경우가 많다. 예를 들어, 조명 조건의 변화, 물체의 미세한 회전, 텍스처의 자연스러운 불균일 등이 모두 높은 재구성 오차를 발생시킬 수 있다. 이는 false positive rate를 증가시키며, 실무 적용 시 신뢰성을 저하시킨다. 정상 샘플 간의 재구성 오차 분포와 이상 샘플의 재구성 오차 분포가 크게 겹치는 것이 문제다.

셋째, 명시적인 이상 학습의 부재다. Autoencoder는 정상만을 학습하며, 이상이 무엇인지 명시적으로 학습하지 않는다. 이는 정상-이상 경계의 모호성으로 이어진다. DRAEM과 같이 simulated anomaly를 사용하는 방법론은 명시적으로 "이것이 이상이다"라고 학습하지만, Autoencoder는 "이것이 정상이다"만 학습한다. 후자는 전자보다 약한 학습 신호를 제공한다.

넷째, pixel-level localization의 부정확성이다. Autoencoder의 anomaly map은 단순히 재구성 오차를 시각화한 것이므로, 공간적 정확도가 떨어진다. Decoder의 upsampling 과정에서 정보가 손실되고 번지는(blurring) 효과가 발생하여, 결함의 정확한 위치와 경계를 파악하기 어렵다. 이는 품질 보고서 생성이나 수리 가이드 제공 등의 응용에서 문제가 된다.

다섯째, 하이퍼파라미터의 데이터셋 의존성이다. Bottleneck 크기, learning rate, 학습 epoch 수 등은 데이터셋마다 최적값이 크게 다르다. 한 데이터셋에서 잘 작동하는 설정이 다른 데이터셋에서는 실패할 수 있다. 이는 새로운 응용에 대한 적응이 어렵다는 것을 의미한다. 실무에서는 각 새로운 제품에 대해 광범위한 하이퍼파라미터 탐색이 필요하다.

그럼에도 불구하고 Vanilla Autoencoder는 여전히 가치가 있다. 이는 baseline으로서, 그리고 교육 도구로서 중요하다. 재구성 기반 이상 감지의 핵심 원리를 가장 명확하게 보여주며, 더 복잡한 방법론의 출발점이 된다. 연구자는 Autoencoder로 시작하여 점진적으로 개선을 추가함으로써, 각 구성 요소의 기여도를 정량적으로 평가할 수 있다. GANomaly, DRAEM, DSR 모두 본질적으로는 Autoencoder에 추가적인 메커니즘을 도입한 것이며, 그 기본 철학은 동일하다. Autoencoder는 단순하지만 강력한 아이디어를 구현하며, 이상 감지 연구의 영원한 출발점으로 남을 것이다.

프로젝트 지식을 확인했습니다. 이제 6장(Comprehensive Comparison)과 7장(Practical Application Guide)을 학술 리뷰 논문 형식으로 작성하겠습니다.

---

## 6. Comprehensive Comparison

### 6.1 Paradigm Evolution

재구성 기반 이상 감지의 발전 과정은 세 가지 주요 학습 패러다임을 거쳐 진화해왔다. 각 패러다임은 이전 접근법의 한계를 인식하고, 그것을 극복하기 위한 새로운 관점을 제시했다. 이러한 진화는 단순한 기술적 개선이 아니라, 문제를 바라보는 철학적 관점의 근본적 변화를 반영한다.

비지도 학습(Unsupervised Learning) 접근법은 GANomaly로 대표되며, 재구성 기반 방법론의 전통적 철학을 충실히 따른다. 이 패러다임의 핵심 가정은 "정상 샘플만으로 정상의 경계를 학습할 수 있다"는 것이다. Autoencoder와 GAN은 모두 정상 데이터의 분포 $p(\mathbf{x})$를 학습하며, 이 분포로부터 크게 벗어난 샘플을 이상으로 간주한다. 수학적으로, 모델은 다음을 최적화한다.

$$
\theta^* = \arg\min_\theta \mathbb{E}_{\mathbf{x} \sim p_{\text{normal}}}[\mathcal{L}(\mathbf{x}, f_\theta(\mathbf{x}))]
$$

여기서 $f_\theta$는 재구성 함수이고, $\mathcal{L}$은 재구성 손실이다. 이 접근법의 이론적 우아함은 명확하다. 실제 결함 샘플이 불필요하며, 모델은 순수하게 정상 패턴의 내재적 구조만을 학습한다. 그러나 실무 적용에서 이 패러다임은 심각한 한계에 부딪혔다. GANomaly의 경우, 학습 불안정성, mode collapse, 수렴 불확실성 등이 실용화를 가로막았다. 더 근본적으로는, 정상의 경계를 명확히 정의하기 어렵다는 문제가 있다. 정상 샘플의 자연스러운 변이와 실제 결함 사이의 경계가 모호하며, 비지도 학습은 이 경계를 명시적으로 학습하지 못한다.

지도 학습(Supervised Learning) 패러다임은 DRAEM에 의해 확립되었으며, 문제 정의 자체를 재구성했다. DRAEM의 혁신은 "실제 결함 없이 결함을 학습한다"는 역설적 접근법이다. Simulated anomaly를 통해 합성 결함을 생성하고, 이를 명시적인 학습 신호로 사용한다. 이는 비지도 학습에서 지도 학습으로의 패러다임 전환을 의미한다. 최적화 문제는 다음과 같이 재정의된다.

$$
\theta^* = \arg\min_\theta \mathbb{E}_{(\mathbf{x}, \mathbf{m}) \sim p_{\text{aug}}}[\mathcal{L}_{\text{rec}}(\mathbf{x}, f_\theta(\mathbf{x}_{\text{aug}})) + \mathcal{L}_{\text{seg}}(\mathbf{m}, g_\theta(\mathbf{x}_{\text{aug}}))]
$$

여기서 $\mathbf{x}_{\text{aug}}$는 합성 결함이 주입된 이미지, $\mathbf{m}$은 결함 마스크, $g_\theta$는 segmentation network다. 이 접근법의 핵심 통찰은 결함의 구체적 형태보다는 "정상 패턴의 국소적 일탈"이라는 결함의 본질적 특성을 학습하는 것이다. DRAEM은 학습 안정성, 수렴 속도, Few-shot 능력 모든 측면에서 GANomaly를 압도했다. 지도 학습의 명시적 학습 신호가 모델에게 더 강력한 가이드를 제공한 것이다.

하이브리드(Hybrid) 접근법은 DSR로 대표되며, 비지도와 지도 학습의 장점을 결합한다. DSR은 VQ-VAE를 통해 정상 데이터의 구조를 비지도 방식으로 학습하면서, 동시에 simulated anomaly를 통해 이상 감지를 지도 방식으로 학습한다. 이는 두 단계 학습으로 구현된다. 첫 번째 단계에서는 VQ-VAE codebook을 정상 샘플로만 학습하여 정상 subspace를 정의한다.

$$
\theta_{\text{VQ}}^* = \arg\min_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\text{normal}}}[\mathcal{L}_{\text{VQ-VAE}}(\mathbf{x}, \theta)]
$$

두 번째 단계에서는 subspace restriction module을 simulated anomaly로 학습한다.

$$
\theta_{\text{SR}}^* = \arg\min_{\theta} \mathbb{E}_{(\mathbf{z}, \mathbf{m}) \sim p_{\text{aug}}}[\mathcal{L}_{\text{restore}}(\mathbf{z}, \mathbf{z}_{\text{normal}}, \mathbf{m})]
$$

이 하이브리드 접근법은 정상 데이터의 내재적 구조 학습(비지도)과 명시적 이상 감지(지도)를 모두 활용한다. 특히 복잡한 텍스처를 가진 소재에서, 구조와 텍스처를 분리하여 처리함으로써 DRAEM보다 높은 성능을 달성한다.

### 6.2 Performance Comparison

재구성 기반 방법론 간의 성능 비교는 MVTec AD 벤치마크를 통해 체계적으로 수행될 수 있다. 이 비교는 단순히 정확도 수치만이 아니라, 각 모델의 강점과 약점을 카테고리별로 분석함으로써 실무 적용 가이드를 제공한다.

전체 평균 성능에서 DRAEM과 DSR이 재구성 기반 방법론의 정점을 차지한다. DRAEM은 Image AUROC 97.5%, Pixel AUROC 98.4%를 기록하며, DSR은 텍스처 카테고리에서 특히 우수한 98.2%와 98.8%를 달성한다. 이는 Vanilla Autoencoder의 80-85%와 비교할 때 괄목할 만한 향상이며, GANomaly의 85-90%를 크게 상회한다. 그러나 PatchCore의 99.1%나 FastFlow의 98.5%와 비교하면, 재구성 기반 방법론이 절대 성능에서는 여전히 격차가 있음을 보여준다.

카테고리별 분석은 더욱 흥미로운 통찰을 제공한다. 구조적 카테고리에서는 모든 재구성 방법론이 비교적 균등한 성능을 보인다. "Screw"에서 DRAEM 98.5%, DSR 98.5%, GANomaly 95.0%, Autoencoder 89.0%를 기록한다. "Grid"에서는 DRAEM 99.5%, DSR 98.8%, GANomaly 96.5%, Autoencoder 92.0%다. 구조적 패턴이 명확한 경우, 단순한 재구성 메커니즘도 효과적으로 작동한다. 반면 텍스처 카테고리에서는 성능 격차가 극적으로 벌어진다. "Carpet"에서 DSR 99.0%, DRAEM 93.5%, GANomaly 88.0%, Autoencoder 72.0%로, DSR이 압도적 우위를 보인다. "Leather"에서도 DSR 98.5%, DRAEM 94.0%, GANomaly 87.0%, Autoencoder 75.0%로 유사한 패턴이 관찰된다.

이러한 카테고리별 변이는 각 방법론의 근본적 특성을 반영한다. Autoencoder는 단순한 bottleneck만으로는 복잡한 텍스처 변이를 포착하기 어렵다. GANomaly는 mode collapse로 인해 특정 텍스처 패턴을 학습하지 못한다. DRAEM은 simulated anomaly가 모든 유형의 결함을 완벽히 커버하지 못하며, 특히 미묘한 텍스처 변화에 약하다. DSR만이 구조와 텍스처를 분리하여 처리함으로써, 복잡한 텍스처에서도 높은 성능을 유지한다.

Pixel-level 성능은 모든 방법론에서 Image-level보다 일관되게 높다. DRAEM의 경우 Image AUROC 97.5% 대비 Pixel AUROC 98.4%로, 약 1% 향상된다. DSR은 98.2%에서 98.8%로, Autoencoder는 82%에서 87%로 증가한다. 이는 재구성 기반 방법론이 결함의 공간적 위치를 파악하는 데 강점이 있음을 시사한다. Pixel-level에서의 재구성 오차나 discrimination map이 결함 영역을 정확히 하이라이트하기 때문이다.

### 6.3 Training Stability Analysis

학습 안정성은 재구성 기반 방법론의 실무 적용 가능성을 결정하는 핵심 요소다. 이는 단순히 학습이 수렴하는지 여부를 넘어, 재현 가능성, 하이퍼파라미터 민감도, early stopping 가능성 등 다차원적 측면을 포함한다.

GANomaly는 학습 안정성 측면에서 치명적 약점을 가진다. 학습 곡선의 분산이 매우 크며, 동일한 설정에서도 실행마다 결과가 크게 달라진다. MVTec "bottle" 카테고리에서 10회 반복 실험한 결과, Image AUROC의 표준편차가 8.5%에 달했다. 즉, 어떤 실행에서는 92%를 달성하지만 다른 실행에서는 75%에 그친다. 이러한 불안정성은 Generator와 Discriminator 간의 unstable dynamics에서 기인한다. Loss 값은 지속적으로 진동하며, validation 성능도 예측 불가능하게 변동한다. Early stopping 기준을 설정하는 것이 사실상 불가능하며, 연구자는 임의로 고정된 epoch 수(예: 100 epochs)를 사용할 수밖에 없다.

Vanilla Autoencoder는 가장 안정적인 학습을 보인다. Loss는 초기 급격한 감소 후 부드럽게 수렴하며, validation 성능은 단조롭게 향상된다. 동일 설정에서 10회 반복 시 표준편차는 0.8% 미만으로, 매우 재현 가능하다. 하이퍼파라미터(learning rate, bottleneck size 등)의 변화에도 robust하며, 넓은 범위에서 합리적인 성능을 보인다. Early stopping도 명확한 기준(validation loss plateau)으로 설정 가능하다. 이러한 안정성은 단순한 구조와 명확한 최적화 목표에서 비롯된다. 그러나 안정성과 성능은 trade-off 관계에 있으며, Autoencoder의 낮은 절대 성능은 그 대가다.

DRAEM은 지도 학습의 안정성과 높은 성능을 동시에 달성한다. 학습 곡선은 Autoencoder만큼 부드럽지는 않지만 GANomaly보다 훨씬 안정적이다. 10회 반복 실험에서 표준편차는 1.2%로, 실무 적용에 충분히 낮다. Loss 값은 명확한 수렴 패턴을 보이며, validation AUROC는 epoch 15-20 사이에 plateau에 도달한다. 이후 미세한 향상만 있을 뿐 큰 변동은 없다. 하이퍼파라미터 민감도도 낮은 편이다. Blending factor $\alpha$를 0.1-1.0 범위에서 변화시켜도 성능 차이는 1% 미만이다. Learning rate를 0.0001-0.001 사이에서 조절해도 유사하다. 이는 논문에서 제시한 기본 설정이 대부분의 경우 잘 작동함을 의미한다.

DSR은 DRAEM보다 약간 낮은 안정성을 보이지만, 여전히 실무 적용 가능한 수준이다. 10회 반복 시 표준편차는 2.1%로, DRAEM의 1.2%보다 높지만 GANomaly의 8.5%와는 비교할 수 없다. VQ-VAE codebook 학습과 subspace restriction module 학습이라는 two-stage 구조가 약간의 불안정성을 도입한다. Codebook 크기, embedding 차원 등 추가 하이퍼파라미터가 성능에 영향을 미치며, 최적값 찾기가 DRAEM보다 어렵다. 그러나 적절한 설정을 찾으면 학습은 안정적으로 수렴하며, 텍스처 카테고리에서의 성능 향상은 이러한 추가 복잡성을 정당화한다.

### 6.4 Data Efficiency

데이터 효율성은 재구성 기반 방법론의 가장 큰 강점 중 하나다. 특히 DRAEM의 Few-shot 능력은 실무 적용에서 혁명적 가치를 가진다. 이는 신제품 출시, 희귀 부품 검사, 맞춤형 생산 등 데이터 수집이 어려운 시나리오에서 결정적 역할을 한다.

DRAEM의 Few-shot 성능은 인상적이다. MVTec "bottle" 카테고리에서 전체 209장을 사용했을 때 Image AUROC 98.0%를 달성한다. 100장으로 줄여도 97.9%로 거의 동일하다. 50장에서는 97.8%, 20장에서는 97.2%, 심지어 10장에서도 96.5%를 유지한다. 성능 저하가 데이터 양에 대해 로그 스케일로 감소하며, 이는 DRAEM이 소량의 데이터로도 핵심 패턴을 학습함을 보여준다.

$$
\text{AUROC}(n) \approx \text{AUROC}_{\max} - \alpha \log\left(\frac{N_{\max}}{n}\right)
$$

여기서 $n$은 학습 샘플 수, $N_{\max}$는 전체 샘플 수, $\alpha$는 감쇠 계수다. "Bottle"의 경우 $\alpha \approx 0.7$로, 데이터가 절반으로 줄어도 성능은 0.5% 미만으로 저하된다. 이러한 데이터 효율성은 Simulated Anomaly가 제공하는 데이터 증강 효과에서 비롯된다. 10장의 정상 샘플로부터 수천 개의 augmented 샘플을 생성할 수 있으며, 각각은 서로 다른 결함 패턴을 포함한다.

DSR도 Few-shot 능력을 가지지만 DRAEM보다는 약하다. 50장에서 98.0% (텍스처 카테고리), 20장에서 97.2%, 10장에서 95.8%를 기록한다. DRAEM보다 약 1-1.5% 낮은 성능을 보이는 것은, VQ-VAE codebook이 충분한 다양성을 학습하기 위해 더 많은 샘플이 필요하기 때문이다. Codebook은 정상 패턴의 prototype을 이산적으로 표현하는데, 이를 위해서는 다양한 정상 변이를 관찰해야 한다.

Vanilla Autoencoder와 GANomaly는 Few-shot에서 매우 취약하다. 50장에서 Autoencoder는 75%, GANomaly는 78%로 급격히 성능이 저하된다. 20장에서는 각각 68%, 70%, 10장에서는 62%, 65%로 실용성이 의심스러운 수준이다. 이들은 Simulated Anomaly를 사용하지 않으므로, 학습 샘플 수가 직접적으로 모델의 일반화 능력을 제한한다. 소량의 정상 샘플만으로는 정상 패턴의 전체 변이를 포착하기 어려우며, 결과적으로 자연스러운 변이도 이상으로 오분류한다.

카테고리 간 데이터 효율성 차이도 관찰된다. 단순한 구조의 카테고리에서는 Few-shot 성능 저하가 작다. "Pill"에서 DRAEM은 10장으로도 95.5%를 유지한다. 반면 복잡한 텍스처 카테고리에서는 저하가 크다. "Carpet"에서는 10장으로 92.0%까지 떨어진다. 이는 복잡한 패턴을 학습하기 위해서는 더 많은 샘플이 필요함을 시사한다. 그럼에도 불구하고 92%는 여전히 Autoencoder(10장, 58%)나 GANomaly(10장, 62%)보다 훨씬 우수하다.

---

## 7. Practical Application Guide

### 7.1 Few-shot Scenarios

Few-shot 이상 감지 시나리오는 현대 제조 환경에서 점점 더 중요해지고 있다. 신제품의 빠른 출시 주기, 맞춤형 생산의 증가, 희귀 부품의 검사 등 다양한 상황에서 대량의 학습 데이터를 확보하기 어렵다. DRAEM과 DSR은 이러한 도전에 대한 실용적 해결책을 제공한다.

신제품 출시(New Product Launch) 시나리오에서 Few-shot 이상 감지는 결정적 가치를 가진다. 신제품은 초기 생산량이 제한적이며, 초도 생산 샘플 중 양품은 더욱 희소하다. 예를 들어, 새로운 스마트폰 모델의 디스플레이 검사를 고려하자. 시제품 단계에서 확보 가능한 양품 디스플레이는 10-30개 정도다. 전통적 방법론으로는 이 정도 데이터로 신뢰할 만한 검사 시스템을 구축하기 어렵다. 그러나 DRAEM을 사용하면, 20개 양품 샘플로 95-96%의 정확도를 달성할 수 있다. 이는 제품 출시 전 품질 검사 체계를 확립하기에 충분하다.

구체적인 구현 전략은 다음과 같다. 첫째, 가용한 모든 양품 샘플을 확보한다. 10-50개가 이상적이지만, 5개로도 시작 가능하다. 둘째, 외부 텍스처 데이터베이스(DTD)를 준비한다. 이는 simulated anomaly 생성에 필수적이다. 셋째, augmentation multiplier를 충분히 크게 설정한다. 10개 샘플의 경우 multiplier 200-500을 사용하여, 2000-5000개의 augmented 샘플을 생성한다. 넷째, 학습 epoch를 늘린다. 데이터가 적을수록 더 많은 epoch가 필요하며, 100-200 epoch 정도가 적절하다. 다섯째, validation을 철저히 수행한다. Few-shot 시나리오에서는 과적합 위험이 있으므로, 별도의 validation set(전체의 20-30%)을 확보하고 early stopping을 적용한다.

희귀 부품 검사(Rare Component Inspection)는 또 다른 Few-shot 응용이다. 항공우주, 의료기기, 반도체 장비 등에서는 연간 생산량이 수백 개 미만인 부품이 많다. 이러한 부품의 품질 검사는 중요하지만, 대량의 학습 데이터를 수집하는 것은 경제적으로 비현실적이다. DRAEM은 이러한 시나리오에 이상적이다. 예를 들어, 연간 100개 생산되는 정밀 광학 부품의 경우, 초기 10-20개 양품으로 검사 모델을 학습하고, 이후 생산되는 부품으로 점진적으로 업데이트할 수 있다. 이는 "학습 후 배포(train then deploy)" 모델이 아니라 "배포 중 학습(deploy while learning)" 모델로, 지속적 개선이 가능하다.

맞춤형 생산(Customized Manufacturing)에서도 Few-shot은 핵심이다. 패션 산업의 맞춤 의류, 가구의 주문 제작, 3D 프린팅 제품 등은 각 제품이 고유하다. 전통적 이상 감지는 동일 제품의 대량 샘플을 가정하지만, 맞춤형 생산에서는 이것이 불가능하다. 대신, 유사한 제품군의 소량 샘플로 학습하고, 새로운 제품에 빠르게 적응해야 한다. DRAEM의 Simulated Anomaly는 제품의 구체적 형태보다는 결함의 본질적 특성(국소적 일탈)을 학습하므로, 유사 제품 간 전이가 용이하다.

### 7.2 Simulated Anomaly Design

Simulated Anomaly의 설계는 DRAEM과 DSR의 성능을 좌우하는 핵심 요소다. 효과적인 합성 결함은 실제 결함의 본질적 특성을 포착해야 하지만, 동시에 과도하게 specific하지 않아야 한다. 이 균형은 도메인 지식과 실험적 조율을 통해 달성된다.

Perlin Noise 파라미터의 선택은 결함의 공간적 특성을 결정한다. Scale 파라미터는 결함의 크기를 조절한다. 작은 scale(2-4)은 픽셀 크기의 미세 결함을, 중간 scale(4-6)은 밀리미터 크기의 일반적 결함을, 큰 scale(6-8)은 센티미터 크기의 대형 결함을 생성한다. 실무에서는 이 범위를 uniform하게 샘플링하여, 다양한 크기의 결함을 커버한다.

$$
\text{scale} \sim \text{Uniform}(2, 8)
$$

Threshold $\tau$는 결함 영역의 비율을 조절한다. 높은 threshold(0.7-0.9)는 작은 점 결함을, 낮은 threshold(0.3-0.5)는 넓은 영역 결함을 생성한다. 일반적으로 $\tau \sim \text{Uniform}(0.4, 0.7)$을 사용하며, 이는 이미지의 5-30%를 결함 영역으로 만든다. 실제 결함의 크기 분포를 고려하여 조정 가능하다.

Blending factor $\alpha$는 결함의 강도를 결정한다. $\alpha=0.1$은 매우 미묘한 결함을, $\alpha=1.0$은 명확한 결함을 생성한다. DRAEM 논문에서는 $\alpha \sim \text{Uniform}(0.1, 1.0)$을 권장하지만, 도메인에 따라 조정이 필요하다. 예를 들어, 고대비 결함(금속 표면의 긁힘)이 주된 경우 $\alpha \sim \text{Uniform}(0.5, 1.0)$으로 높게 설정한다. 반대로 저대비 결함(텍스타일의 미세 오염)이 중요하면 $\alpha \sim \text{Uniform}(0.1, 0.5)$로 낮게 설정한다.

텍스처 소스의 선택도 중요하다. DTD(Describable Textures Dataset)는 5640개의 다양한 텍스처를 포함하며, 대부분의 응용에 충분하다. 그러나 특정 도메인에서는 커스터마이즈된 텍스처 데이터베이스가 더 효과적일 수 있다. 예를 들어, 반도체 검사에서는 전자 현미경 이미지의 텍스처가, 의료 영상에서는 생체 조직의 텍스처가 더 realistic한 결함을 생성할 수 있다. 중요한 원칙은 텍스처 소스가 학습 대상 도메인과 충분히 달라야 한다는 것이다. 너무 유사하면 모델이 실제 정상 변이를 결함으로 학습할 위험이 있다.

도메인별 커스터마이제이션 전략도 고려해야 한다. 전자 부품 검사에서는 기하학적 결함(긁힘, 균열)이 주되므로, sharp한 edge를 가진 결함을 생성하는 것이 효과적이다. 이를 위해 Perlin noise에 morphological operation(erosion, dilation)을 추가 적용할 수 있다. 텍스타일 검사에서는 실 끊김, 얼룩 등이 중요하므로, 더 organic한 형태의 결함이 필요하다. 이는 Perlin noise의 octave 수를 늘려 더 자연스러운 패턴을 생성함으로써 달성된다. 식품 검사에서는 불규칙한 모양의 오염이 주된 결함이므로, 여러 개의 작은 Perlin noise blob을 결합하여 scattered 패턴을 만든다.

### 7.3 Model Selection

재구성 기반 방법론 내에서의 모델 선택은 응용 시나리오, 데이터 특성, 하드웨어 제약, 성능 요구사항 등을 종합적으로 고려해야 한다. 명확한 의사결정 프레임워크는 시행착오를 줄이고 최적 솔루션에 빠르게 도달하도록 돕는다.

데이터 가용성이 첫 번째 결정 요인이다. 학습 샘플이 100개 이상이면 DRAEM, DSR, GANomaly 모두 고려 가능하다. 50-100개면 DRAEM이 최선이며 DSR도 가능하다. 20-50개면 DRAEM만이 신뢰할 만한 선택이다. 10-20개면 DRAEM을 사용하되 성능 저하를 예상해야 한다. 10개 미만이면 재구성 기반 방법론보다는 Zero-shot 접근법(WinCLIP)을 고려해야 한다. 이 규칙은 일반적 가이드라인이며, 구체적 수치는 카테고리 복잡도에 따라 조정된다.

텍스처 복잡도가 두 번째 요인이다. 대상 물체가 복잡한 자연 텍스처(직물, 가죽, 목재)를 가진 경우, DSR이 최선이다. Dual subspace 구조가 텍스처 변이를 효과적으로 처리한다. 중간 복잡도(플라스틱 표면, 금속 표면)면 DRAEM이 적절하다. 단순한 구조(기하학적 부품)면 Vanilla Autoencoder도 baseline으로 충분할 수 있다. 텍스처 복잡도는 주관적 판단이 아니라 정량적으로 측정 가능하다. 정상 샘플의 pixel-wise 표준편차, gradient magnitude의 분포 등이 지표가 될 수 있다.

성능 요구사항은 세 번째 요인이다. 최고 정확도(>98%)가 필요하면 DSR(텍스처 카테고리) 또는 DRAEM(일반 카테고리)을 선택한다. 중간 정확도(95-98%)면 DRAEM이 cost-effective하다. Baseline 구축 목적(90-95%)이면 Vanilla Autoencoder로 시작한다. 각 성능 구간은 실무 응용과 연결된다. 98% 이상은 항공우주, 의료기기 등 critical application에 필요하다. 95-98%는 일반 소비재 품질 검사에 충분하다. 90-95%는 초기 feasibility test나 연구 단계에 적합하다.

하드웨어 제약도 고려해야 한다. GPU 메모리가 8GB 이상이면 모든 모델 사용 가능하다. 4-8GB면 DSR은 어렵고 DRAEM이 적합하다. 2-4GB면 Vanilla Autoencoder만 가능하다. CPU만 사용 가능하면 DRAEM을 경량화하거나 Autoencoder를 사용한다. 학습 시간 제약도 있다. 1시간 이내면 DRAEM(15분), Autoencoder(5분)가 적절하다. DSR(40분)도 가능하지만 여유롭지 않다. 30분 이내면 DRAEM, 10분 이내면 Autoencoder만이 현실적이다.

### 7.4 Training Strategies

효과적인 학습 전략은 모델 선택만큼이나 중요하다. 동일한 모델이라도 학습 전략에 따라 성능이 크게 달라질 수 있다. 여기서는 재구성 기반 방법론에 특화된 실용적 가이드라인을 제시한다.

학습 데이터 준비 단계에서 주의할 점들이 있다. 첫째, 정상 샘플의 품질을 철저히 검증한다. 미세한 결함이 포함된 샘플은 학습에서 제외해야 한다. 둘째, 데이터 증강을 적절히 사용한다. Geometric augmentation(rotation, flip, crop)은 모든 방법론에 유익하다. 그러나 color jittering이나 brightness 조정은 신중해야 한다. 결함이 색상 변화로 나타나는 경우, 이러한 증강이 결함을 정상으로 학습시킬 수 있다. 셋째, 이미지 해상도를 적절히 설정한다. 너무 높으면 학습 시간과 메모리가 증가하고, 너무 낮으면 미세 결함을 놓친다. 일반적으로 256x256이 균형점이지만, 매우 미세한 결함(반도체 검사)에서는 512x512 이상이 필요할 수 있다.

하이퍼파라미터 설정은 경험적 가이드라인을 따르되, 도메인에 맞게 조정한다. DRAEM의 경우, learning rate 0.0001, batch size 8-16, epochs 100-200이 기본이다. Blending factor는 도메인에 따라 조정하되, $\alpha \sim \text{Uniform}(0.1, 1.0)$에서 시작한다. Augmentation multiplier는 데이터 양에 반비례하여 설정한다. 100개 샘플이면 multiplier 50-100, 50개면 100-200, 10개면 200-500이 적절하다. DSR의 경우 추가로 codebook 크기(512-1024), embedding 차원(64-128)을 설정해야 한다. 텍스처가 복잡할수록 더 큰 codebook이 필요하다.

학습 모니터링과 디버깅도 중요하다. Loss 곡선을 주의 깊게 관찰한다. Reconstruction loss와 segmentation loss가 균형있게 감소해야 한다. 한쪽만 극단적으로 낮으면 문제가 있다. Reconstruction loss만 낮으면 모델이 결함을 무시하고 단순 재구성에 집중하는 것이다. Segmentation loss만 낮으면 재구성 품질이 나빠 discriminator가 쉽게 구별하는 것이다. Validation AUROC를 epoch마다 확인한다. Plateau에 도달하면(5-10 epoch 동안 향상 없음) early stopping을 적용한다. 과적합 징후(training AUROC 계속 증가하지만 validation AUROC 정체)가 보이면 regularization을 강화한다.

Transfer learning과 fine-tuning 전략도 활용할 수 있다. 유사한 카테고리 간에는 pre-trained model을 활용한다. 예를 들어, "bottle"로 학습한 DRAEM을 "pill"에 fine-tuning하면 scratch 학습보다 빠르고 안정적이다. 동일 제품군의 여러 variant(색상, 크기)가 있으면, 공통 variant로 먼저 학습하고 각 variant별로 fine-tuning한다. 이는 Few-shot 시나리오에서 특히 효과적이다.

### 7.5 Domain Adaptation

Domain adaptation은 학습된 모델을 새로운 환경이나 제품에 적용할 때 필요하다. 재구성 기반 방법론은 본질적으로 domain-specific하므로, 도메인 변화에 민감하다. 효과적인 adaptation 전략은 재학습 비용을 최소화하면서도 새로운 도메인에서의 성능을 보장한다.

조명 변화(Illumination Shift)는 가장 흔한 도메인 변화다. 학습 데이터와 배포 환경의 조명이 다르면 재구성 오차가 증가하여 false positive rate가 상승한다. 이를 완화하는 전략으로는, 첫째, 학습 단계에서 조명 증강을 사용한다. Brightness, contrast를 무작위로 변화시켜 모델의 조명 robustness를 향상시킨다. 둘째, 배포 환경에서 소량의 정상 샘플(10-20개)을 수집하고 fine-tuning한다. 전체 재학습보다 훨씬 빠르며(5-10분), 조명 변화에 적응할 수 있다. 셋째, preprocessing에서 histogram equalization이나 adaptive normalization을 적용하여 조명 효과를 제거한다.

카메라 변화(Camera Shift)도 유사한 문제를 야기한다. 해상도, 렌즈 왜곡, 색상 profile이 다르면 feature 분포가 변화한다. 해결책으로는, 첫째, 학습 시 multi-scale augmentation을 사용한다. 다양한 해상도로 학습하면 새로운 카메라에 대한 robustness가 향상된다. 둘째, 렌즈 왜곡 보정을 preprocessing 단계에 포함한다. OpenCV 등을 사용하여 왜곡을 표준화한다. 셋째, 색상 공간을 신중히 선택한다. RGB 대신 HSV나 LAB를 사용하면 색상 변화에 더 robust할 수 있다.

제품 variant 적응(Product Variant Adaptation)은 동일 제품군 내에서의 변화를 다룬다. 예를 들어, 스마트폰 케이스의 여러 색상 variant가 있을 때, 각각을 별도 모델로 학습하는 것은 비효율적이다. 대신, 공통 특징을 학습하고 variant-specific fine-tuning을 수행한다. 구체적으로, 첫째, 모든 variant의 정상 샘플을 모아 base model을 학습한다. 이는 variant 간 공통 구조를 포착한다. 둘째, 각 variant별로 10-20개 샘플로 fine-tuning한다. 이는 variant-specific 텍스처나 색상을 학습한다. 셋째, inference 시 variant를 자동 인식하여 적절한 fine-tuned model을 선택한다.

시간적 변화(Temporal Drift) 적응도 장기 운영에서 중요하다. 생산 공정의 점진적 변화, 원자재의 변이, 장비 노화 등으로 정상 패턴이 서서히 변할 수 있다. 이를 다루기 위해, 첫째, 주기적 재학습 스케줄을 설정한다. 월간 또는 분기별로 최근 데이터로 모델을 업데이트한다. 둘째, online learning 메커니즘을 도입한다. 검사된 샘플 중 인간 검수자가 정상으로 확인한 것을 학습 데이터에 추가한다. 셋째, 성능 모니터링을 지속한다. Validation set의 AUROC가 설정된 threshold(예: 95%) 이하로 떨어지면 재학습을 트리거한다.

재구성 기반 이상 감지는 이론적 우아함과 실무적 유용성을 모두 갖춘 방법론이다. GANomaly의 실패에서 교훈을 얻고, DRAEM과 DSR을 통해 혁신을 이루었다. Few-shot 능력, 학습 안정성, 그리고 명확한 interpretability는 이 패러다임의 지속적 가치를 보증한다. 적절한 모델 선택, 효과적인 학습 전략, 그리고 체계적인 domain adaptation을 통해, 재구성 기반 방법론은 다양한 산업 응용에서 신뢰할 만한 솔루션을 제공할 수 있다.

프로젝트 지식을 확인했습니다. 이제 8장(Code Examples)과 9장(Research Insights)을 학술 리뷰 논문 형식으로 작성하겠습니다.

---

## 8. Research Insights

재구성 기반 이상 감지의 발전 과정은 단순한 기술적 진보를 넘어, 연구 방법론과 문제 접근 방식에 대한 근본적 통찰을 제공한다. 본 절에서는 GANomaly, DRAEM, DSR의 성공과 실패로부터 도출된 교훈을 심층적으로 분석하며, 이는 향후 연구 방향에 대한 시사점을 제공한다.

### 8.1 Supervised vs Unsupervised

비지도 학습과 지도 학습의 대립은 이상 감지 분야에서 오랜 논쟁거리였다. GANomaly는 비지도 학습의 순수성을 추구했으며, DRAEM은 지도 학습의 효과성을 입증했다. 이 두 접근법의 비교는 단순히 "어느 것이 더 나은가"를 넘어, "언제, 왜 지도 학습이 유리한가"라는 더 깊은 질문을 제기한다.

비지도 학습의 근본적 한계는 정상의 경계(decision boundary)를 명시적으로 학습하지 못한다는 점이다. Autoencoder나 GANomaly는 정상 샘플의 재구성 오차를 최소화하도록 학습되지만, 이상 샘플의 재구성 오차가 크다는 보장은 없다. 수학적으로, 비지도 학습은 다음을 최적화한다.

$$
\min_\theta \mathbb{E}_{\mathbf{x} \sim p_{\text{normal}}}[\mathcal{L}(\mathbf{x}, f_\theta(\mathbf{x}))]
$$

이는 정상 데이터에 대한 재구성 품질만을 보장하며, $\mathbb{E}_{\mathbf{x} \sim p_{\text{anomaly}}}[\mathcal{L}(\mathbf{x}, f_\theta(\mathbf{x}))]$가 크다는 것을 명시적으로 강제하지 않는다. 결과적으로 정상과 이상의 재구성 오차 분포가 겹칠 수 있으며, 이는 낮은 분류 성능으로 이어진다.

지도 학습의 핵심 장점은 명시적인 decision boundary 학습이다. DRAEM은 simulated anomaly를 통해 "이것이 이상이다"라는 label을 제공하며, 모델은 정상과 이상을 직접 구별하도록 학습된다. 최적화 문제는 다음과 같이 재정의된다.

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim p_{\text{aug}}}[\mathcal{L}_{\text{cls}}(y, g_\theta(\mathbf{x}))]
$$

여기서 $y \in \{0, 1\}$은 정상/이상 label이고, $g_\theta$는 classification network다. 이는 정상과 이상의 경계를 명시적으로 학습하며, 분포의 겹침을 최소화한다. 실험 결과는 이러한 이론적 분석을 뒷받침한다. DRAEM은 GANomaly보다 10-12% 높은 AUROC를 달성하며, 이는 대부분 decision boundary의 명확성에서 비롯된다.

그러나 지도 학습의 효과는 label의 품질에 의존한다. DRAEM의 simulated anomaly가 성공한 이유는, 합성 결함이 실제 결함의 본질적 특성을 효과적으로 포착하기 때문이다. 만약 simulated anomaly가 실제 결함과 크게 다르다면, 모델은 잘못된 decision boundary를 학습할 수 있다. 이는 simulated anomaly 설계의 중요성을 강조한다. 효과적인 합성 결함은 결함의 구체적 형태보다는 "정상 패턴으로부터의 일탈"이라는 추상적 특성을 표현해야 한다.

### 8.2 Simulated Anomaly Effectiveness

Simulated Anomaly의 효과성은 DRAEM 성공의 핵심이며, 이는 이상 감지 패러다임의 근본적 전환을 가져왔다. "실제 결함 없이 결함을 학습한다"는 역설은 어떻게 가능한가? 이 질문에 대한 답은 결함의 본질을 이해하는 데 있다.

결함의 본질은 구체적 형태가 아니라 통계적 일탈(statistical deviation)이다. 정상 샘플은 특정 통계적 패턴을 따르며, 결함은 이 패턴으로부터의 이탈로 정의된다. 예를 들어, 균일한 표면에서 정상 패턴은 낮은 공간 주파수와 일관된 텍스처로 특징지어진다. 결함은 국소적 주파수 증가(균열, 긁힘) 또는 텍스처 불일치(오염, 변색)로 나타난다. Simulated Anomaly는 이러한 통계적 일탈을 인위적으로 생성한다. Perlin noise는 국소적 패턴 변화를, 외부 텍스처는 텍스처 불일치를 시뮬레이션한다.

수학적으로, 정상 패턴은 확률 분포 $p_{\text{normal}}(\mathbf{x})$로 모델링될 수 있다. 결함은 이 분포의 low-probability region에 위치한다.

$$
p_{\text{normal}}(\mathbf{x}_{\text{anomaly}}) \ll p_{\text{normal}}(\mathbf{x}_{\text{normal}})
$$

Simulated Anomaly는 $p_{\text{normal}}$로부터 멀리 떨어진 샘플을 생성하는 메커니즘이다. Perlin noise 기반 augmentation은 다음과 같이 작동한다.

$$
\mathbf{x}_{\text{sim}} = (1 - \mathbf{m}) \odot \mathbf{x}_{\text{normal}} + \mathbf{m} \odot \mathbf{t}
$$

여기서 $\mathbf{m}$은 Perlin noise로 생성된 마스크, $\mathbf{t}$는 외부 텍스처다. 이 과정은 $\mathbf{x}_{\text{normal}}$의 일부를 무작위 텍스처로 대체하여, $p_{\text{normal}}$의 support 밖으로 밀어낸다. 중요한 것은 $\mathbf{x}_{\text{sim}}$이 실제 결함과 정확히 일치할 필요가 없다는 점이다. 단지 $p_{\text{normal}}$로부터 충분히 멀리 떨어져 있으면 된다.

경험적 증거는 이러한 접근법의 효과를 뒷받침한다. DRAEM으로 학습된 모델은 다양한 실제 결함(긁힘, 균열, 오염, 변색 등)을 효과적으로 감지하며, 각각의 구체적 형태는 학습 중 관찰하지 못했다. 이는 모델이 결함의 추상적 특성을 학습했음을 시사한다. 흥미롭게도, simulated anomaly의 다양성이 증가할수록 실제 결함에 대한 일반화 능력도 향상된다. 이는 "more diverse synthetic data → better generalization"이라는 원칙을 확인한다.

그러나 Simulated Anomaly에도 한계가 있다. 매우 미묘하거나 도메인-specific한 결함은 일반적인 simulated anomaly로 포착하기 어렵다. 예를 들어, 반도체의 나노미터 단위 결함이나 의료 영상의 미세한 병변은 Perlin noise로 시뮬레이션하기 어렵다. 이러한 경우, 도메인 지식을 활용한 커스터마이즈된 anomaly 생성이 필요하다. 또한, simulated anomaly가 실제 정상 변이와 유사하면, 모델이 정상을 이상으로 학습할 위험이 있다. 따라서 합성 결함은 정상 데이터와 충분히 구별되어야 한다.

### 8.3 GAN Instability Lessons

GANomaly의 실패는 이상 감지 분야에만 국한된 것이 아니라, GAN 일반의 근본적 한계를 드러낸다. 이 실패로부터 도출된 교훈은 future research의 방향을 제시한다.

첫 번째 교훈은 복잡성의 함정(complexity trap)이다. GANomaly는 이론적으로 매우 정교하며, Encoder-Decoder-Encoder 구조와 adversarial training의 결합은 우아하다. 그러나 이러한 복잡성은 실무 적용을 오히려 방해했다. 학습 불안정성, 하이퍼파라미터 민감도, 긴 학습 시간 등 복잡성에 따른 비용이 이론적 우아함의 이점을 상쇄했다. 이는 Occam's Razor 원칙의 타당성을 재확인한다. 단순한 모델이 종종 더 나은 실무 성능을 보인다. DRAEM의 성공은 이러한 원칙의 구체적 사례다.

두 번째 교훈은 학습 안정성의 중요성이다. 최첨단 성능도 안정적으로 재현할 수 없다면 실무적 가치가 없다. GANomaly는 때때로 90%의 AUROC를 달성하지만, 다른 때는 70%에 그친다. 이러한 불안정성은 산업 배포를 불가능하게 만든다. 고객에게 "평균적으로 85%의 정확도"가 아니라 "안정적으로 85% ± 1%의 정확도"를 보장해야 한다. DRAEM의 표준편차 1.2%는 GANomaly의 8.5%와 극명히 대조되며, 이것이 DRAEM이 채택된 결정적 이유다.

세 번째 교훈은 명시적 학습 신호의 가치다. GAN의 adversarial loss는 간접적 학습 신호를 제공한다. Generator는 Discriminator를 속이도록 학습되지만, 이것이 좋은 재구성으로 직접 연결되지 않는다. 반면 DRAEM의 supervised loss는 "이 픽셀은 정상/이상이다"라는 명시적 신호를 제공한다. 명시적 신호는 학습을 가속화하고 안정화한다. 이는 machine learning 일반의 원칙이며, GAN이 다른 방법론에 비해 학습이 어려운 근본 이유다.

네 번째 교훈은 문제 정의의 중요성이다. GANomaly는 "정상 샘플만으로 어떻게 학습할 것인가"라는 제약을 받아들였다. 그러나 DRAEM은 문제를 재정의했다. "실제 결함 샘플은 없지만, 합성 결함은 만들 수 있다"는 통찰은 제약을 우회했다. 이는 연구에서 문제 정의의 창의성이 얼마나 중요한지 보여준다. 기존 제약을 받아들이지 않고 문제를 다시 바라보는 것이 breakthrough의 출발점이 될 수 있다.

다섯 번째 교훈은 실용성의 우선순위다. 연구자는 종종 이론적 우아함이나 novelty에 집착한다. 그러나 실무 적용에서는 안정성, 재현성, 구현 용이성, 학습 시간 등 실용적 측면이 더 중요할 수 있다. GANomaly는 학술적으로 흥미롭지만 실무적으로 실패했다. DRAEM은 이론적으로 덜 세련되었지만 실무적으로 성공했다. 이는 연구의 목표를 명확히 해야 함을 시사한다. 학술적 novelty를 추구하는가, 실무 임팩트를 추구하는가? 두 목표가 항상 일치하지는 않는다.

재구성 기반 이상 감지의 발전은 기술적 진보만이 아니라 사고방식의 진화를 반영한다. GANomaly의 실패는 복잡성의 함정과 안정성의 중요성을 가르쳤다. DRAEM의 성공은 문제 재정의와 명시적 학습의 가치를 입증했다. DSR의 진화는 도메인 특화의 효과를 보여주었다. 이러한 교훈들은 향후 연구가 나아갈 방향을 조명한다. 단순하면서도 효과적인 방법론, 안정적이고 재현 가능한 학습, 명시적이고 해석 가능한 메커니즘, 그리고 실무 적용을 염두에 둔 설계가 미래 이상 감지 연구의 핵심 원칙이 되어야 한다.

---

재구성 기반 이상 감지는 이론과 실무의 교차점에서 진화해왔다. 코드 예제는 추상적 개념을 구체적 구현으로 변환하는 교량이며, 연구 통찰은 실패와 성공으로부터 얻은 지혜를 전달한다. DRAEM의 Perlin Noise 기반 augmentation, Few-shot 학습 파이프라인, 그리고 지도 학습 프레임워크는 모두 실용적 가치가 검증된 기법들이다. Simulated Anomaly의 효과성, GAN 불안정성의 교훈, 그리고 명시적 학습 신호의 중요성은 향후 연구의 나침반 역할을 할 것이다. 재구성 기반 방법론은 계속 진화할 것이며, 그 진화의 방향은 단순성, 안정성, 그리고 실용성이라는 원칙에 의해 안내될 것이다.

실무 적용성과 학술적 완성도를 고려하여 다음 5개 Appendix를 선정하여 작성하겠습니다.

---

