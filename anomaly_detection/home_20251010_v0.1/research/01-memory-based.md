# 1. Paradigm Overview

## 1.1 Core Principle

Memory-based 방식은 산업 이상 감지 분야에서 가장 높은 정확도를 달성한 패러다임으로, 정상 샘플의 특징을 메모리에 저장하고 테스트 샘플과의 거리를 측정하여 이상을 탐지한다. 이 접근법의 핵심 아이디어는 단순하면서도 강력하다. 정상 데이터의 특징 분포를 명시적으로 모델링하고, 새로운 샘플이 이 분포로부터 얼마나 벗어났는지를 정량화하는 것이다.

전통적인 분류 문제와 달리 산업 이상 감지는 정상 샘플만 학습 데이터로 주어지는 one-class learning 문제다. 이상 샘플은 학습 단계에서 관찰되지 않으며, 그 종류와 특성을 사전에 알 수 없다. Memory-based 방식은 이러한 제약 조건 하에서 정상 패턴을 충실히 기억하고, 이로부터의 편차를 감지하는 전략을 취한다.

구체적으로 이 패러다임은 사전 학습된 CNN(Convolutional Neural Network)을 특징 추출기로 활용한다. ImageNet과 같은 대규모 데이터셋에서 학습된 네트워크는 이미 풍부한 시각적 표현을 내재하고 있으며, 이를 타겟 도메인에 적용할 때 추가 학습 없이도 강력한 특징을 제공한다. 정상 샘플들의 특징 벡터를 메모리 뱅크에 저장한 후, 테스트 샘플의 특징과 메모리 내 특징들 간의 거리를 계산하여 이상 점수를 산출한다.

이 방식의 가장 큰 장점은 학습 안정성과 해석 가능성이다. 복잡한 역전파 과정이나 생성 모델의 학습 없이 단순히 특징을 저장하고 비교하기 때문에 하이퍼파라미터에 민감하지 않으며, 이상 점수의 의미가 명확하다. 또한 메모리에 저장된 정상 샘플과의 비교를 통해 어떤 정상 패턴으로부터 벗어났는지 역추적할 수 있어 결함 분석에 유용하다.

## 1.2 Mathematical Formulation

Memory-based 방식의 수학적 정식화는 거리 기반 이상 탐지의 확률론적 해석에 기반한다. 입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$가 주어졌을 때, 사전 학습된 CNN $\phi$를 통해 특징 맵 $\mathbf{F} = \phi(\mathbf{x}) \in \mathbb{R}^{h \times w \times d}$를 추출한다. 여기서 $h, w$는 공간 해상도이고 $d$는 특징 차원이다.

특징 맵의 각 공간 위치 $(i,j)$에서 패치 특징 벡터 $\mathbf{f}_{ij} \in \mathbb{R}^d$를 추출한다. 정상 학습 데이터 $\mathcal{D}_{\text{train}} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$로부터 얻어진 모든 패치 특징들의 집합을 메모리 뱅크 $\mathcal{M} = \{\mathbf{f}_{ij}^{(n)} \mid n=1,\ldots,N; i=1,\ldots,h; j=1,\ldots,w\}$로 정의한다.

테스트 샘플의 패치 특징 $\mathbf{f}_{\text{test}}$에 대한 이상 점수는 메모리 뱅크 내 특징들과의 거리로 계산된다. 가장 기본적인 형태는 최근접 이웃 거리다.

$$s(\mathbf{f}_{\text{test}}) = \min_{\mathbf{f} \in \mathcal{M}} d(\mathbf{f}_{\text{test}}, \mathbf{f})$$

여기서 $d(\cdot, \cdot)$는 거리 함수로, 유클리드 거리나 Mahalanobis 거리가 사용된다. 더 강건한 추정을 위해 $k$개의 최근접 이웃의 평균 거리를 사용하기도 한다.

$$s_k(\mathbf{f}_{\text{test}}) = \frac{1}{k} \sum_{i=1}^{k} d(\mathbf{f}_{\text{test}}, \mathbf{f}^{(i)}_{\text{nn}})$$

여기서 $\mathbf{f}^{(i)}_{\text{nn}}$는 $i$번째 최근접 이웃이다.

Mahalanobis 거리는 특징 분포의 공분산을 고려하여 더 정교한 거리 측정을 제공한다. 패치 위치 $(i,j)$의 정상 특징 분포를 다변량 가우시안 $\mathcal{N}(\boldsymbol{\mu}_{ij}, \boldsymbol{\Sigma}_{ij})$로 모델링할 때, Mahalanobis 거리는 다음과 같이 정의된다.

$$d_M(\mathbf{f}_{\text{test}}, \mathcal{N}_{ij}) = \sqrt{(\mathbf{f}_{\text{test}} - \boldsymbol{\mu}_{ij})^T \boldsymbol{\Sigma}_{ij}^{-1} (\mathbf{f}_{\text{test}} - \boldsymbol{\mu}_{ij})}$$

이 거리는 특징 공간의 기하학적 구조를 반영하여, 높은 분산을 가진 방향의 편차는 덜 비정상적으로, 낮은 분산 방향의 편차는 더 비정상적으로 평가한다.

픽셀 레벨 이상 맵(anomaly map)은 각 공간 위치의 이상 점수를 집계하여 생성된다. 테스트 이미지의 특징 맵에서 각 위치 $(i,j)$의 이상 점수 $s_{ij}$를 계산한 후, 이를 원본 이미지 해상도로 업샘플링하여 최종 이상 맵 $\mathbf{A} \in \mathbb{R}^{H \times W}$를 얻는다.

이미지 레벨 이상 점수는 픽셀 레벨 점수의 최댓값 또는 상위 $p$% 점수의 평균으로 계산된다.

$$S_{\text{image}} = \max_{i,j} s_{ij} \quad \text{or} \quad S_{\text{image}} = \text{mean}_{\text{top-}p\%}(s_{ij})$$

## 1.3 Key Assumptions

Memory-based 방식은 몇 가지 핵심 가정에 기반하여 작동한다. 이러한 가정들을 이해하는 것은 방법론의 적용 가능성과 한계를 파악하는 데 중요하다.

첫 번째 가정은 정상 데이터의 manifold 가정이다. 고차원 특징 공간에서 정상 샘플들은 저차원 다양체(manifold) 위에 집중적으로 분포한다고 가정한다. 이상 샘플은 이 다양체로부터 벗어나 있으므로, 다양체까지의 거리로 이상도를 측정할 수 있다. 이 가정은 자연 이미지의 통계적 특성과 CNN 특징 공간의 구조에 의해 경험적으로 지지된다.

두 번째는 사전 학습된 특징의 전이 가능성(transferability) 가정이다. ImageNet에서 학습된 CNN은 타겟 도메인의 산업 이미지에서도 유용한 특징을 추출할 수 있다고 가정한다. 실제로 중간 층의 특징들은 edge, texture, pattern과 같은 일반적인 시각적 요소를 포착하며, 이는 산업 결함 탐지에 효과적이다. 다만 도메인 간 격차가 클 경우 성능이 저하될 수 있다.

세 번째는 지역적 일관성(local consistency) 가정이다. 이상은 전역적이기보다는 국소적으로 나타나며, 패치 단위로 분석하면 충분히 탐지 가능하다고 가정한다. 이는 산업 결함의 특성과 잘 부합한다. 대부분의 결함은 긁힘, 균열, 오염과 같이 공간적으로 제한된 영역에 나타난다.

네 번째는 정상 샘플의 충분성 가정이다. 메모리 뱅크에 저장된 정상 샘플들이 가능한 모든 정상 변동(normal variation)을 대표한다고 가정한다. 만약 학습 데이터가 정상 패턴의 일부만 포함한다면, 학습 중 관찰되지 않은 정상 패턴이 이상으로 오탐지될 수 있다. 따라서 다양한 정상 변동을 포괄하는 대표적인 학습 데이터가 필요하다.

다섯 번째는 거리 기반 분리 가능성 가정이다. 정상과 이상 샘플이 특징 공간에서 거리 기반 메트릭으로 효과적으로 구분될 수 있다고 가정한다. 이는 이상 샘플의 특징이 정상 샘플의 특징 분포로부터 충분히 멀리 떨어져 있을 때 성립한다. 정상과 매우 유사한 미세 결함의 경우 이 가정이 약화될 수 있다.

## 1.4 Historical Context

Memory-based 방식의 역사적 발전은 컴퓨터 비전과 이상 탐지 분야의 기술적 진보와 밀접하게 연결되어 있다. 2012년 AlexNet 이후 CNN의 급격한 발전은 강력한 특징 추출기를 제공했고, 이는 memory-based 방식이 성공할 수 있는 기반이 되었다.

초기 이상 탐지 연구는 주로 통계적 방법과 전통적인 머신러닝 기법에 의존했다. Principal Component Analysis(PCA), Support Vector Machine(SVM), Isolation Forest 등이 대표적이다. 이들은 수작업으로 설계된 특징(hand-crafted features)을 사용했으며, 복잡한 시각적 패턴을 포착하는 데 한계가 있었다.

딥러닝 시대에 들어서며 이상 탐지는 크게 두 방향으로 발전했다. 첫 번째는 생성 모델 기반 접근으로, Autoencoder나 GAN을 이용해 정상 데이터의 분포를 학습하고 재구성 오차로 이상을 탐지하는 방식이다. 두 번째는 one-class classification 접근으로, 정상 데이터를 하나의 클래스로 학습하는 방식이다. 그러나 이들은 학습 불안정성, 긴 학습 시간, 과적합 등의 문제가 있었다.

2020년 PaDiM의 등장은 memory-based 방식의 시작을 알렸다. PaDiM은 사전 학습된 CNN의 중간 층 특징을 활용하고, 패치 단위로 다변량 가우시안 분포를 모델링하여 Mahalanobis 거리로 이상을 탐지했다. 이 접근법은 당시 SOTA 성능을 달성하면서도 학습이 매우 빠르고 안정적이었다. 그러나 높은 메모리 사용량이 실용적 배포의 장애물이었다.

2022년 PatchCore는 coreset selection 알고리즘을 도입하여 memory-based 방식의 결정적 돌파구를 마련했다. Greedy coreset subsampling을 통해 메모리를 90% 이상 절감하면서도 성능을 오히려 향상시켰다. PatchCore는 MVTec AD 벤치마크에서 99.1%의 이미지 레벨 AUROC를 달성하며 single-class 이상 탐지의 사실상 표준이 되었다.

같은 시기에 DFKDE는 비모수적 밀도 추정 방법인 Kernel Density Estimation을 딥러닝 특징에 적용했다. 가우시안 가정을 완화하고 더 유연한 분포 모델링을 시도했지만, 계산 비용과 성능 측면에서 PatchCore에 미치지 못했다.

Memory-based 방식의 성공은 여러 요인에 기인한다. 우선 ImageNet 사전 학습의 강력한 전이 학습 효과다. ResNet, EfficientNet 등의 네트워크는 산업 이미지에서도 즉시 사용 가능한 고품질 특징을 제공한다. 또한 학습 과정의 단순성으로 인해 하이퍼파라미터 튜닝이 거의 불필요하며, 재현성이 높다. 마지막으로 패치 단위 분석은 위치 정보를 보존하면서도 충분한 통계적 샘플을 확보할 수 있게 한다.

현재 memory-based 방식은 최고 정확도가 요구되는 품질 검사 시스템에서 선호되고 있다. 특히 정상 변동이 제한적이고 결함이 명확한 경우 탁월한 성능을 보인다. 반면 실시간 처리가 필요하거나 메모리 제약이 엄격한 엣지 환경에서는 다른 패러다임이 더 적합할 수 있다. 최근에는 foundation model과의 결합을 통해 multi-class 환경으로 확장하려는 시도가 진행 중이다.

# 2. PaDiM (2020)

## 2.1 Basic Information

PaDiM(Patch Distribution Modeling)은 2020년 Defard 등이 제안한 memory-based 이상 탐지 방법으로, 사전 학습된 CNN을 활용한 패치 단위 분포 모델링을 통해 산업 이상 탐지 분야에 새로운 패러다임을 제시했다. 이 연구는 ECCV 2020 VAND Workshop에서 발표되었으며, 당시 MVTec AD 벤치마크에서 SOTA 성능을 달성하며 주목받았다.

PaDiM의 핵심 기여는 세 가지로 요약된다. 첫째, ImageNet 사전 학습 네트워크의 중간 층 특징을 추가 학습 없이 직접 활용하는 접근법을 확립했다. 둘째, 각 패치 위치에서 다변량 가우시안 분포를 모델링하고 Mahalanobis 거리로 이상을 측정하는 통계적 프레임워크를 제공했다. 셋째, 다중 스케일 특징 결합을 통해 semantic 정보와 spatial 정보를 동시에 활용하는 방법을 제시했다.

PaDiM은 학습 과정이 매우 간단하다. 역전파나 경사 하강법이 필요 없으며, 단순히 정상 샘플의 특징을 추출하고 통계량을 계산하는 것으로 학습이 완료된다. 이러한 단순성은 학습 안정성과 재현성을 보장하며, 하이퍼파라미터에 민감하지 않다는 장점을 제공한다. 전체 학습 시간은 데이터셋 크기에 따라 다르지만, 일반적으로 수 분 이내에 완료된다.

논문에서는 ResNet18과 Wide ResNet-50을 백본으로 사용했으며, layer1, layer2, layer3의 특징을 결합하여 총 448차원 또는 550차원의 특징 벡터를 생성했다. 이후 랜덤 프로젝션을 통해 100-200차원으로 차원을 축소하여 계산 효율성을 높였다. MVTec AD 데이터셋에서 이미지 레벨 97.5%, 픽셀 레벨 97.5%의 AUROC를 달성하여 기존 방법들을 크게 앞섰다.

## 2.2 Core Algorithm

### 2.2.1 Patch Distribution Modeling

PaDiM의 핵심 아이디어는 이미지를 패치 단위로 분해하고 각 패치 위치에서 정상 특징의 분포를 모델링하는 것이다. 이는 전역적(global) 특징만 사용하는 기존 방법들의 한계를 극복하기 위한 전략이다. 전역 특징은 이미지 전체의 평균적인 정보를 담고 있어 국소적 결함을 감지하기 어렵다. 반면 패치 단위 접근은 공간적 위치 정보를 보존하면서도 충분한 통계적 샘플을 확보할 수 있다.

입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$가 주어지면, 사전 학습된 CNN의 여러 층에서 특징 맵을 추출한다. 각 층 $l$의 특징 맵 $\mathbf{F}^l \in \mathbb{R}^{h_l \times w_l \times d_l}$을 동일한 공간 해상도로 업샘플링하거나 다운샘플링한 후 채널 차원으로 연결(concatenate)하여 통합 특징 맵 $\mathbf{F} \in \mathbb{R}^{h \times w \times d}$를 생성한다. 여기서 $d = \sum_l d_l$은 결합된 특징의 총 차원이다.

통합 특징 맵의 각 공간 위치 $(i,j)$는 하나의 패치에 대응된다. 위치 $(i,j)$의 특징 벡터 $\mathbf{f}_{ij} \in \mathbb{R}^d$는 해당 패치의 시각적 내용을 인코딩한다. 학습 데이터셋의 모든 정상 이미지에서 동일한 위치 $(i,j)$의 특징 벡터들을 수집하면, 해당 위치에서 나타날 수 있는 정상 패턴의 변동을 포착할 수 있다.

정상 학습 이미지가 $N$개 있을 때, 위치 $(i,j)$에서 추출된 특징 벡터들의 집합 $\{\mathbf{f}_{ij}^{(n)}\}_{n=1}^{N}$을 얻는다. 이들로부터 경험적 분포를 추정한다. PaDiM은 각 위치마다 독립적인 확률 분포를 모델링하므로, 총 $h \times w$개의 분포 모델이 학습된다. 이러한 위치별 모델링은 공간적 컨텍스트를 반영한다. 예를 들어 제품 이미지에서 중앙 부분과 가장자리 부분은 서로 다른 정상 패턴을 가지며, 이를 별도로 모델링함으로써 더 정밀한 이상 탐지가 가능하다.

패치 단위 접근의 또 다른 이점은 데이터 증강 효과다. $H \times W$ 크기의 이미지에서 $h \times w$개의 패치를 추출하면, 하나의 이미지로부터 여러 학습 샘플을 얻는 효과가 있다. 이는 제한된 학습 데이터로도 안정적인 분포 추정을 가능하게 한다. 실제로 MVTec AD 데이터셋의 경우 카테고리당 평균 200장 정도의 정상 이미지만 제공되지만, 패치 레벨에서는 수만 개의 샘플이 확보된다.

### 2.2.2 Multivariate Gaussian Assumption

PaDiM은 각 패치 위치의 정상 특징 분포를 다변량 가우시안(multivariate Gaussian)으로 모델링한다. 위치 $(i,j)$에서 수집된 $N$개의 특징 벡터 $\{\mathbf{f}_{ij}^{(n)}\}_{n=1}^{N}$에 대해 평균 벡터 $\boldsymbol{\mu}_{ij}$와 공분산 행렬 $\boldsymbol{\Sigma}_{ij}$를 계산한다.

$$\boldsymbol{\mu}_{ij} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{f}_{ij}^{(n)}$$

$$\boldsymbol{\Sigma}_{ij} = \frac{1}{N-1} \sum_{n=1}^{N} (\mathbf{f}_{ij}^{(n)} - \boldsymbol{\mu}_{ij})(\mathbf{f}_{ij}^{(n)} - \boldsymbol{\mu}_{ij})^T$$

가우시안 가정은 수학적 tractability와 효율적인 추론을 제공한다. 가우시안 분포는 두 개의 통계량(평균과 공분산)만으로 완전히 특정되므로 메모리 효율적이다. 또한 Mahalanobis 거리라는 자연스러운 거리 척도를 제공하여 이상도 측정이 간단하다.

실제로 CNN 특징 공간에서 정상 패턴의 분포가 정확히 가우시안인 것은 아니다. 그러나 중심극한정리에 의해 충분한 샘플이 있을 때 평균 특징은 근사적으로 가우시안 분포를 따른다. 또한 고차원 공간에서 대부분의 분포는 국소적으로 가우시안에 가깝다. PaDiM의 경험적 성공은 이러한 가정이 실용적으로 충분히 유효함을 보여준다.

공분산 행렬 $\boldsymbol{\Sigma}_{ij} \in \mathbb{R}^{d \times d}$는 특징 간 상관관계를 포착한다. 대각 성분은 각 특징 차원의 분산을 나타내며, 비대각 성분은 특징 간 공분산을 나타낸다. Full 공분산 행렬을 사용하면 특징 공간의 타원체 구조를 모델링할 수 있어, 단순 유클리드 거리보다 정교한 이상 탐지가 가능하다.

다만 고차원에서 full 공분산 행렬은 수치적 문제를 야기할 수 있다. 특징 차원 $d$가 샘플 수 $N$보다 크면 공분산 행렬이 특이행렬(singular)이 되어 역행렬을 계산할 수 없다. PaDiM은 이를 해결하기 위해 차원 축소와 정규화 기법을 사용한다. 공분산 행렬에 작은 양수 $\epsilon$을 대각 성분에 더하는 regularization을 적용하여 안정적인 역행렬 계산을 보장한다.

$$\boldsymbol{\Sigma}_{ij}^{\text{reg}} = \boldsymbol{\Sigma}_{ij} + \epsilon \mathbf{I}$$

여기서 $\mathbf{I}$는 단위 행렬이고 $\epsilon$은 일반적으로 $10^{-4}$ 정도의 작은 값이다.

### 2.2.3 Mahalanobis Distance

PaDiM은 Mahalanobis 거리를 이상 점수로 사용한다. 테스트 이미지의 위치 $(i,j)$에서 추출된 특징 벡터 $\mathbf{f}_{ij}^{\text{test}}$에 대해, 해당 위치의 정상 분포 $\mathcal{N}(\boldsymbol{\mu}_{ij}, \boldsymbol{\Sigma}_{ij})$로부터의 Mahalanobis 거리는 다음과 같이 계산된다.

$$M_{ij} = \sqrt{(\mathbf{f}_{ij}^{\text{test}} - \boldsymbol{\mu}_{ij})^T \boldsymbol{\Sigma}_{ij}^{-1} (\mathbf{f}_{ij}^{\text{test}} - \boldsymbol{\mu}_{ij})}$$

이 거리는 단순 유클리드 거리 $\|\mathbf{f}_{ij}^{\text{test}} - \boldsymbol{\mu}_{ij}\|_2$를 일반화한 것이다. 공분산 행렬의 역행렬 $\boldsymbol{\Sigma}_{ij}^{-1}$을 통해 특징 공간의 기하학적 구조를 반영한다. 정상 데이터가 높은 분산을 보이는 방향으로의 편차는 낮은 이상 점수를, 낮은 분산 방향으로의 편차는 높은 이상 점수를 받는다.

Mahalanobis 거리의 직관적 의미는 표준화된 거리다. 각 특징 차원을 해당 차원의 표준편차로 나눈 후 거리를 계산하는 것과 같다. 이는 특징 스케일의 차이를 자동으로 보정하여 모든 차원이 공평하게 기여하도록 한다. CNN의 서로 다른 층에서 추출된 특징들은 스케일이 크게 다를 수 있는데, Mahalanobis 거리는 이를 자연스럽게 처리한다.

가우시안 가정 하에서 Mahalanobis 거리의 제곱 $M_{ij}^2$는 카이제곱 분포 $\chi^2_d$를 따른다. 이는 통계적 가설 검정의 프레임워크를 제공한다. 주어진 유의 수준에서 임계값을 설정하여 정상/이상을 분류할 수 있다. 그러나 실무에서는 ROC 곡선 분석을 통해 최적 임계값을 경험적으로 결정하는 것이 일반적이다.

픽셀 레벨 이상 맵은 각 위치의 Mahalanobis 거리를 집계하여 생성된다. 특징 맵의 해상도 $h \times w$가 원본 이미지보다 작으므로, bilinear interpolation을 통해 원본 해상도 $H \times W$로 업샘플링한다. 이상 맵 $\mathbf{A} \in \mathbb{R}^{H \times W}$의 각 픽셀 값은 해당 위치의 이상도를 나타낸다.

$$\mathbf{A} = \text{Upsample}([M_{ij}]_{i,j})$$

이미지 레벨 이상 점수는 이상 맵의 최댓값으로 정의된다.

$$S_{\text{image}} = \max_{x,y} \mathbf{A}(x,y)$$

이는 이미지 내 가장 이상한 부분의 점수로 전체 이미지를 대표하는 전략이다. 대안적으로 상위 $k$개 픽셀의 평균을 사용하거나, 임계값을 초과하는 픽셀의 비율을 사용할 수도 있다.

## 2.3 Technical Details

### 2.3.1 Multi-scale Feature Extraction

PaDiM의 핵심 기술 중 하나는 다중 스케일 특징 추출이다. CNN의 서로 다른 층은 서로 다른 추상화 수준의 정보를 인코딩한다. 초기 층은 edge, texture와 같은 low-level 특징을, 중간 층은 부품이나 패턴 같은 mid-level 특징을, 깊은 층은 객체나 장면 같은 high-level semantic 정보를 포착한다. 이상 탐지에서는 이 모든 수준의 정보가 유용할 수 있다.

ResNet 계열 네트워크를 백본으로 사용할 때, PaDiM은 layer1, layer2, layer3의 출력을 활용한다. 이들은 각각 원본 이미지 대비 1/4, 1/8, 1/16 해상도를 가지며, 64, 128, 256 채널(ResNet18 기준)을 가진다. Layer4는 너무 높은 semantic 수준이어서 미세한 결함 탐지에 적합하지 않으므로 제외한다.

서로 다른 해상도의 특징 맵을 결합하기 위해 모두 동일한 공간 해상도로 맞춘다. 일반적으로 layer3의 해상도를 기준으로 하며, layer1과 layer2는 average pooling으로 다운샘플링한다. 해상도를 $h \times w$로 통일한 후 채널 차원으로 연결하여 $h \times w \times d$ 크기의 통합 특징 맵을 얻는다. ResNet18의 경우 $d = 64 + 128 + 256 = 448$, Wide ResNet-50의 경우 $d = 512 + 1024 + 2048 = 3584$가 된다.

다중 스케일 결합의 효과는 실험적으로 검증되었다. 단일 층만 사용할 때보다 다중 층을 결합했을 때 성능이 일관되게 향상되었다. 특히 layer2와 layer3의 조합이 가장 효과적이었으며, layer1의 추가는 texture-based 결함 탐지에 유리했다. 이는 서로 다른 유형의 결함이 서로 다른 특징 스케일에서 더 잘 드러나기 때문이다.

다중 스케일 접근의 계산 비용은 크지 않다. 특징 추출은 순전파 한 번으로 모든 층의 특징을 동시에 얻을 수 있으며, 특징 결합은 단순한 tensor concatenation이다. 주된 비용 증가는 차원 $d$의 증가로 인한 공분산 행렬 계산과 Mahalanobis 거리 계산에서 발생한다. 이는 차원 축소로 완화된다.

### 2.3.2 Covariance Matrix Computation

공분산 행렬 계산은 PaDiM의 학습 단계에서 가장 중요한 과정이다. 각 패치 위치 $(i,j)$에 대해 $N$개의 특징 벡터로부터 $d \times d$ 크기의 공분산 행렬을 추정해야 한다. 이는 온라인 알고리즘으로 효율적으로 구현할 수 있다.

Welford's online algorithm을 사용하면 데이터를 한 번만 순회하면서 평균과 공분산을 동시에 계산할 수 있다. 이는 메모리 효율적이며 수치적으로도 안정적이다. $n$번째 샘플 $\mathbf{f}^{(n)}$을 관찰했을 때의 업데이트는 다음과 같다.

$$\boldsymbol{\mu}_n = \boldsymbol{\mu}_{n-1} + \frac{1}{n}(\mathbf{f}^{(n)} - \boldsymbol{\mu}_{n-1})$$

$$\mathbf{M}_n = \mathbf{M}_{n-1} + (\mathbf{f}^{(n)} - \boldsymbol{\mu}_{n-1})(\mathbf{f}^{(n)} - \boldsymbol{\mu}_n)^T$$

최종 공분산은 $\boldsymbol{\Sigma} = \mathbf{M}_N / (N-1)$로 계산된다.

고차원 설정에서 공분산 행렬 추정의 주된 문제는 샘플 수가 차원에 비해 부족할 때 발생한다. $N < d$인 경우 공분산 행렬은 rank-deficient가 되어 역행렬이 존재하지 않는다. $N > d$이더라도 $N$이 $d$에 비해 충분히 크지 않으면 추정이 불안정하다. 이론적으로 안정적인 공분산 추정을 위해서는 $N \gg d$가 필요하다.

PaDiM은 두 가지 전략으로 이 문제를 해결한다. 첫째는 차원 축소다. 원본 특징 차원 $d$가 수백에서 수천에 달하므로, 이를 100-200 차원으로 축소한다. 둘째는 regularization이다. 공분산 행렬에 작은 상수를 대각 성분에 더하여 조건수(condition number)를 개선한다.

$$\boldsymbol{\Sigma}^{\text{reg}} = (1-\lambda)\boldsymbol{\Sigma} + \lambda \text{trace}(\boldsymbol{\Sigma})/d \cdot \mathbf{I}$$

여기서 $\lambda$는 regularization 강도를 조절하는 하이퍼파라미터다. $\lambda=0$이면 원본 공분산, $\lambda=1$이면 isotropic Gaussian이 된다. 일반적으로 $\lambda=0.01$ 정도의 작은 값을 사용한다.

공분산 행렬의 역행렬 계산은 Cholesky decomposition이나 eigenvalue decomposition을 통해 수행된다. 추론 단계에서는 각 위치마다 $\boldsymbol{\Sigma}_{ij}^{-1}$을 미리 계산하여 저장해두므로, 테스트 시에는 행렬-벡터 곱셈만 수행하면 된다.

### 2.3.3 Dimensionality Reduction

차원 축소는 PaDiM에서 계산 효율성과 통계적 안정성을 위해 필수적이다. ResNet50을 백본으로 사용하면 결합 특징의 차원이 3000을 초과하는데, 이는 공분산 추정과 거리 계산에 과도한 비용을 초래한다. 또한 고차원 공간에서의 curse of dimensionality 문제도 발생한다.

PaDiM은 랜덤 프로젝션(random projection)을 차원 축소 방법으로 사용한다. 이는 Johnson-Lindenstrauss 보조정리에 기반한 방법으로, 랜덤 행렬을 통해 고차원 벡터를 저차원으로 투영하면서도 점들 간의 거리를 높은 확률로 보존한다. 원본 차원 $d$에서 목표 차원 $d'$으로의 투영 행렬 $\mathbf{R} \in \mathbb{R}^{d \times d'}$의 각 원소는 표준 정규 분포에서 샘플링된다.

$$R_{ij} \sim \mathcal{N}(0, 1/d')$$

특징 벡터 $\mathbf{f} \in \mathbb{R}^d$는 $\mathbf{f}' = \mathbf{R}^T \mathbf{f} \in \mathbb{R}^{d'}$로 투영된다. 이 투영은 학습 데이터와 테스트 데이터에 동일하게 적용된다.

랜덤 프로젝션의 장점은 계산이 매우 빠르고, 학습이 필요 없으며, 이론적 보장이 있다는 것이다. PCA(Principal Component Analysis)와 비교했을 때, PCA는 학습 데이터로부터 주성분을 계산해야 하지만 랜덤 프로젝션은 그럴 필요가 없다. 실험 결과 랜덤 프로젝션과 PCA의 성능 차이는 미미했으며, 랜덤 프로젝션이 더 빠르고 간단했다.

목표 차원 $d'$의 선택은 성능과 효율성의 trade-off다. 논문에서는 $d' = 100$에서 $d' = 550$(축소 없음)까지 실험했으며, $d' = 100$-$200$ 범위에서 최적 성능을 보였다. 너무 낮은 차원은 정보 손실을 초래하고, 너무 높은 차원은 계산 비용과 overfitting 위험을 증가시킨다.

차원 축소 후 공분산 행렬의 크기는 $d' \times d'$로 감소한다. $d' = 100$일 때 공분산 행렬은 10,000개의 원소를 가지며, 이는 원본 차원에 비해 크게 줄어든 것이다. 역행렬 계산의 복잡도는 $O(d'^3)$이므로, 차원 축소는 큰 속도 향상을 가져온다.

차원 축소는 모든 패치 위치에 동일한 투영 행렬을 사용한다. 이는 위치 간 일관성을 유지하고 구현을 단순화한다. 대안적으로 각 위치마다 서로 다른 투영을 사용할 수도 있지만, 실험 결과 추가 이득이 크지 않았다.

## 2.4 Performance Analysis

PaDiM은 MVTec AD 벤치마크에서 발표 당시 최고 성능을 달성했다. 이미지 레벨 분류에서 평균 AUROC 97.5%, 픽셀 레벨 세그멘테이션에서 97.5%를 기록했다. 이는 당시 SOTA였던 PatchSVDD(92.1%)와 SPADE(85.5%)를 크게 앞선 수치다. 15개 카테고리 중 bottle, cable, grid, leather, tile 등에서 99% 이상의 이미지 AUROC를 달성했다.

카테고리별 성능 차이는 결함의 특성과 밀접하게 연관된다. Texture 기반 카테고리(carpet, grid, leather, tile, wood)에서 PaDiM은 특히 강력했다. 이들은 반복적인 패턴을 가지며, 결함이 패턴의 불규칙성으로 나타난다. 다중 스케일 특징, 특히 layer1과 layer2의 texture 정보가 효과적으로 작용했다. 반면 object 카테고리(bottle, capsule, pill 등)에서는 상대적으로 낮은 성능을 보였다. 이들은 포즈나 조명 변화로 인한 정상 변동이 크고, 결함이 미세하거나 객체 형태와 혼동될 수 있다.

픽셀 레벨 성능은 이미지 레벨보다 전반적으로 높았다. 이는 패치 단위 분석이 결함 위치를 정확히 특정할 수 있음을 보여준다. 특히 큰 결함이나 명확한 경계를 가진 결함에서 픽셀 레벨 AUROC가 99%를 초과했다. 작고 희미한 결함의 경우 약간 낮은 성능을 보였지만, 여전히 95% 이상을 유지했다.

백본 네트워크의 선택은 성능에 유의미한 영향을 미쳤다. Wide ResNet-50이 ResNet18보다 일관되게 우수했다. 더 깊고 넓은 네트워크가 더 풍부한 특징 표현을 제공하기 때문이다. 그러나 성능 향상은 비용 증가를 동반한다. Wide ResNet-50은 ResNet18 대비 추론 시간이 약 3배 길고, 메모리 사용량도 증가한다. 실무에서는 정확도 요구사항과 리소스 제약을 고려하여 백본을 선택해야 한다.

학습 시간은 매우 짧다. MVTec AD의 한 카테고리(평균 약 200장)를 ResNet18 백본으로 학습하는 데 약 2-3분이 소요된다. 이는 end-to-end 학습 기반 방법들(수 시간에서 수십 시간)에 비해 압도적으로 빠르다. 추론 시간은 이미지당 약 50-100ms(GPU 기준)로, 실시간은 아니지만 대부분의 검사 라인 속도를 충족한다.

그러나 PaDiM의 가장 큰 한계는 메모리 사용량이다. 각 패치 위치마다 $d' \times d'$ 크기의 공분산 행렬을 저장해야 하므로, 총 메모리는 $h \times w \times d' \times d' \times 4$ 바이트(float32 기준)가 필요하다. $h=w=56$, $d'=100$일 때 약 1.2GB가 소요된다. 이는 엣지 디바이스나 다중 카테고리 배포 시 실용성을 제약한다. 이 문제는 후속 연구인 PatchCore에서 coreset selection을 통해 해결된다.

## 2.5 Advantages and Limitations

PaDiM의 주요 장점은 학습의 단순성과 안정성이다. 역전파가 필요 없고 하이퍼파라미터가 거의 없어 재현성이 높다. 사전 학습된 백본만 있으면 즉시 적용 가능하며, 새로운 카테고리에 대한 적응이 빠르다. 통계적 프레임워크는 수학적으로 명확하고 해석 가능하며, Mahalanobis 거리는 이상의 정도를 정량적으로 표현한다.

다중 스케일 특징 결합은 다양한 유형의 결함을 효과적으로 포착한다. Low-level 특징은 texture 기반 결함을, high-level 특징은 structural 결함을 탐지한다. 패치 단위 분석은 공간 정보를 보존하여 정확한 결함 위치 파악을 가능하게 한다. 이상 맵은 시각적으로 직관적이며, 검사자가 결함 영역을 즉시 확인할 수 있다.

성능 측면에서 PaDiM은 발표 당시 최고 수준이었으며, 특히 texture 카테고리에서 탁월했다. 학습 속도가 매우 빠르고 안정적이어서 실무 프로토타이핑에 이상적이다. 추론 시간도 합리적인 수준이다.

그러나 PaDiM은 명확한 한계도 가진다. 가장 큰 문제는 높은 메모리 사용량이다. 각 위치마다 공분산 행렬을 저장하므로 메모리가 $O(h \times w \times d'^2)$로 증가한다. 이는 고해상도 이미지나 다중 카테고리 배포 시 병목이 된다. 엣지 디바이스나 메모리 제약 환경에서는 사용이 어렵다.

가우시안 가정은 제한적일 수 있다. 실제 정상 분포가 다봉형(multimodal)이거나 비대칭적인 경우 모델링이 부정확해진다. 예를 들어 조립 제품에서 여러 정상 구성이 가능한 경우, 단일 가우시안으로는 포착하기 어렵다. 또한 매우 복잡한 정상 변동을 가진 데이터에서는 성능이 저하될 수 있다.

위치별 독립 모델링은 공간적 맥락을 완전히 활용하지 못한다. 인접 패치 간의 상관관계나 전역적 구조 정보가 무시된다. 일부 결함은 국소적으로는 정상처럼 보이지만 전역적 맥락에서 이상일 수 있다. 예를 들어 나사가 잘못된 위치에 있는 경우, 나사 자체는 정상이지만 위치가 이상이다.

차원 축소 과정에서 정보 손실이 발생한다. 랜덤 프로젝션은 빠르고 효과적이지만, 최적의 부분 공간을 찾는 것은 아니다. 특히 미세하고 희귀한 결함의 경우 차원 축소로 인해 신호가 약화될 수 있다.

사전 학습 백본에 대한 의존성도 한계다. ImageNet과 타겟 도메인 간 격차가 클 경우 전이 학습 효과가 감소한다. 예를 들어 X-ray 이미지나 현미경 이미지와 같은 특수 도메인에서는 성능이 제한될 수 있다. 이런 경우 도메인 특화 사전 학습이나 fine-tuning이 필요할 수 있다.

## 2.6 Implementation Considerations

PaDiM을 실제 시스템에 구현할 때 고려해야 할 실용적 측면들이 있다. 백본 네트워크 선택은 정확도와 효율성의 균형을 결정한다. ResNet18은 빠르고 가벼워 실시간에 가까운 처리가 가능하지만, Wide ResNet-50이나 EfficientNet은 더 높은 정확도를 제공한다. 일반적으로 ResNet18로 시작하여 성능이 부족한 경우 더 큰 모델로 업그레이드하는 전략이 권장된다.

특징 층 선택도 중요하다. 기본 설정인 layer1-2-3 조합이 대부분의 경우 잘 작동하지만, 데이터 특성에 따라 조정할 수 있다. Texture 중심의 결함은 낮은 층을 강조하고, structural 결함은 높은 층을 강조하는 것이 유리하다. 실험을 통해 최적 조합을 찾아야 한다.

차원 축소의 목표 차원 $d'$은 메모리와 성능의 trade-off를 결정한다. $d' = 100$은 메모리 효율적이지만 정보 손실 위험이 있고, $d' = 200$-$300$은 균형잡힌 선택이다. 메모리가 충분하다면 축소를 생략할 수도 있지만, 계산 비용이 증가한다.

임계값 설정은 실제 배포에서 핵심적이다. ROC 곡선에서 최적 임계값을 선택하되, false positive와 false negative의 비용을 고려해야 한다. 산업 검사에서는 일반적으로 false negative(불량 누락)를 더 심각하게 보므로, 높은 재현율(recall)을 우선한다. 이는 낮은 임계값을 의미하며, 일부 false positive를 감수한다.

메모리 최적화 기법이 필요할 수 있다. 공분산 행렬을 half precision(float16)으로 저장하면 메모리를 절반으로 줄일 수 있지만, 수치 안정성을 확인해야 한다. 또는 공분산 행렬 대신 eigenvalue decomposition 결과를 저장하여 압축할 수 있다. 극단적으로는 대각 공분산만 사용하여 메모리를 $O(h \times w \times d')$로 줄일 수 있지만 성능이 저하된다.

배치 처리를 활용하면 추론 속도를 높일 수 있다. 여러 이미지를 동시에 처리하여 GPU 활용률을 높인다. 특징 추출은 자연스럽게 배치 처리가 가능하며, Mahalanobis 거리 계산도 벡터화할 수 있다. 배치 크기는 GPU 메모리에 따라 조정한다.

다중 카테고리 시나리오에서는 각 카테고리마다 별도의 모델을 학습하고 저장해야 한다. 이는 메모리 요구량을 카테고리 수에 비례하여 증가시킨다. 카테고리 간 백본 가중치는 공유할 수 있지만, 통계량(평균, 공분산)은 각각 유지해야 한다. 카테고리가 많은 경우 PaDiM보다 foundation model 기반 multi-class 접근이 더 적합할 수 있다.

정상 데이터의 품질과 다양성이 매우 중요하다. 학습 데이터가 가능한 모든 정상 변동을 포함해야 한다. 조명, 포즈, 색상 등의 변동이 있다면 충분한 샘플로 커버해야 한다. 그렇지 않으면 학습 중 보지 못한 정상 변동이 이상으로 오탐지될 수 있다. 데이터 증강(augmentation)을 통해 변동을 인위적으로 증가시킬 수 있지만, 과도한 증강은 오히려 분포를 왜곡할 수 있다.

정기적인 재학습과 모니터링이 필요하다. 제조 공정이나 제품이 변경되면 정상 패턴도 변한다. 모델을 주기적으로 업데이트하여 현재 공정을 반영해야 한다. 또한 오탐과 미탐 사례를 수집하여 분석하고, 필요시 학습 데이터를 보강하거나 하이퍼파라미터를 조정한다.

# 3. PatchCore (2022)

## 3.1 Basic Information

PatchCore는 2022년 Roth 등이 CVPR에서 발표한 논문 "Towards Total Recall in Industrial Anomaly Detection"에서 제안된 방법으로, memory-based 이상 탐지의 결정적 돌파구를 마련했다. 이 연구는 PaDiM의 메모리 문제를 해결하면서도 성능을 오히려 향상시켜, single-class 산업 이상 탐지의 사실상 표준이 되었다.

PatchCore의 핵심 혁신은 coreset selection이다. 정상 학습 샘플의 모든 패치 특징을 저장하는 대신, 전체 분포를 대표하는 소수의 핵심 샘플만 선택하여 메모리 뱅크를 구성한다. Greedy approximate k-center 알고리즘을 사용하여 원본 데이터의 10% 미만으로 메모리를 줄이면서도 정보 손실을 최소화한다. 이는 curse of dimensionality를 역으로 활용한 영리한 전략이다. 고차원 공간에서는 소수의 잘 선택된 샘플만으로도 전체 분포를 효과적으로 커버할 수 있다.

또 다른 중요한 기여는 locally aware patch features의 도입이다. 각 패치 특징을 추출할 때 주변 패치들과의 관계를 고려하여 맥락 정보를 통합한다. 이는 국소적 texture뿐만 아니라 공간적 구조도 포착할 수 있게 한다. Neighborhood aggregation을 통해 $3 \times 3$ 이웃 패치의 평균을 현재 패치 특징에 결합함으로써 수용 영역(receptive field)을 확장한다.

PatchCore는 MVTec AD 벤치마크에서 이미지 레벨 99.1%, 픽셀 레벨 98.1%의 AUROC를 달성하여 당시 모든 방법을 능가했다. 특히 주목할 점은 1% coreset만 사용하여 이러한 성능을 달성했다는 것이다. 이는 메모리 사용량을 100배 가까이 줄이면서도 full memory bank보다 우수한 결과를 의미한다. 이러한 역설적 현상은 coreset selection이 단순한 압축이 아니라 노이즈 제거와 일반화 개선 효과를 가져오기 때문이다.

논문의 제목 "Towards Total Recall"은 산업 검사의 궁극적 목표를 반영한다. 불량품을 절대 놓치지 않는 것, 즉 100%에 가까운 재현율(recall)을 달성하는 것이 제조업의 핵심 요구사항이다. PatchCore는 여러 카테고리에서 99% 이상의 재현율을 달성하여 이 목표에 크게 근접했다.

## 3.2 Coreset Selection Algorithm

### 3.2.1 Greedy Subsampling

Coreset selection의 목적은 원본 데이터셋 $\mathcal{M}$을 잘 대표하는 부분 집합 $\mathcal{C} \subset \mathcal{M}$을 찾는 것이다. 이상적으로는 $\mathcal{C}$를 사용한 이상 탐지 성능이 전체 집합 $\mathcal{M}$을 사용한 것과 유사해야 한다. PatchCore는 이를 k-center problem의 근사 문제로 정식화한다. 목표는 최대 거리를 최소화하는 것이다.

$$\min_{\mathcal{C}} \max_{\mathbf{f} \in \mathcal{M}} \min_{\mathbf{c} \in \mathcal{C}} \|\mathbf{f} - \mathbf{c}\|_2$$

즉, 원본 데이터의 모든 점이 coreset의 어떤 점으로부터 가능한 가까이 있도록 coreset을 선택한다. 이는 NP-hard 문제지만, greedy 알고리즘이 2-근사해를 제공한다는 것이 알려져 있다.

Greedy approximate k-center 알고리즘은 다음과 같이 작동한다. 먼저 임의의 한 점을 coreset에 추가한다. 그 다음 반복적으로 현재 coreset으로부터 가장 먼 점을 선택하여 추가한다. 이를 목표 크기 $|\mathcal{C}|$에 도달할 때까지 반복한다.

```
Initialize: C = {random point from M}
while |C| < target_size:
    f_new = argmax_{f in M} min_{c in C} ||f - c||_2
    C = C ∪ {f_new}
return C
```

이 알고리즘의 직관은 명확하다. 각 단계에서 현재 coreset에 의해 가장 poorly covered된 영역을 찾아 그곳의 샘플을 추가한다. 이는 coreset이 특징 공간을 균등하게 커버하도록 보장한다. 밀집된 영역에서는 적은 샘플로 충분하고, 희소한 영역에서는 더 많은 샘플이 선택된다.

구현 시 매 iteration마다 모든 점의 거리를 재계산하는 것은 비효율적이다. 대신 각 점의 현재 최소 거리를 유지하고 업데이트하는 방식을 사용한다. 새로운 점 $\mathbf{c}_{\text{new}}$가 coreset에 추가되면, 모든 점 $\mathbf{f} \in \mathcal{M}$에 대해 거리를 업데이트한다.

$$d(\mathbf{f}) = \min(d(\mathbf{f}), \|\mathbf{f} - \mathbf{c}_{\text{new}}\|_2)$$

그 다음 $d(\mathbf{f})$가 최대인 점을 다음 coreset 원소로 선택한다. 이렇게 하면 각 iteration의 복잡도가 $O(|\mathcal{M}|)$로 줄어든다.

Greedy 알고리즘의 근사 보장은 중요한 이론적 기반을 제공한다. 알려진 바에 따르면, greedy 방법으로 선택된 coreset $\mathcal{C}$는 최적 coreset $\mathcal{C}^*$에 대해 다음을 만족한다.

$$\max_{\mathbf{f} \in \mathcal{M}} \min_{\mathbf{c} \in \mathcal{C}} \|\mathbf{f} - \mathbf{c}\|_2 \leq 2 \cdot \max_{\mathbf{f} \in \mathcal{M}} \min_{\mathbf{c}^* \in \mathcal{C}^*} \|\mathbf{f} - \mathbf{c}^*\|_2$$

즉, greedy 해는 최적해의 최대 2배 이내의 커버 반경을 가진다. 실제로는 이보다 훨씬 좋은 성능을 보인다.

### 3.2.2 Coverage Guarantee (ε-cover)

Coreset의 품질은 $\epsilon$-cover 개념으로 정량화할 수 있다. 집합 $\mathcal{C}$가 집합 $\mathcal{M}$의 $\epsilon$-cover라는 것은 $\mathcal{M}$의 모든 점이 $\mathcal{C}$의 어떤 점으로부터 거리 $\epsilon$ 이내에 있다는 것을 의미한다.

$$\forall \mathbf{f} \in \mathcal{M}, \exists \mathbf{c} \in \mathcal{C} : \|\mathbf{f} - \mathbf{c}\|_2 \leq \epsilon$$

Greedy coreset selection은 주어진 크기 제약 하에서 가능한 작은 $\epsilon$ 값을 달성하려고 한다. Coreset 크기 $|\mathcal{C}|$와 커버 반경 $\epsilon$ 사이에는 trade-off가 존재한다. 더 큰 coreset은 더 작은 $\epsilon$을 제공하지만 메모리와 계산 비용이 증가한다.

고차원 공간에서는 소수의 잘 배치된 점으로도 효과적인 커버가 가능하다. 이는 Johnson-Lindenstrauss lemma와 관련이 있다. 고차원에서 점들은 서로 멀리 떨어져 있는 경향이 있으며, 따라서 각 coreset 원소가 넓은 영역을 커버할 수 있다. PatchCore는 이 성질을 활용하여 1-10%의 작은 coreset으로도 충분한 커버리지를 달성한다.

실험적으로 PatchCore는 1% coreset($|\mathcal{C}|/|\mathcal{M}| = 0.01$)에서도 대부분의 카테고리에서 최고 성능을 보였다. 이는 원본 데이터의 대부분이 redundant하거나 노이즈를 포함한다는 것을 시사한다. Coreset selection은 일종의 자동 data cleaning 역할을 한다. 밀집 영역의 중복된 샘플들은 제거되고, 대표적인 샘플만 유지된다.

커버리지는 단순히 최근접 이웃 거리 분포를 분석하여 평가할 수 있다. 원본 데이터의 각 점에 대해 coreset 내 최근접 점까지의 거리를 계산하고, 이들의 분포를 조사한다. 대부분의 점이 작은 거리를 가지면 좋은 커버리지를 의미한다. 논문에서는 1% coreset의 경우 90% 이상의 점이 평균 거리의 절반 이내에 최근접 coreset 원소를 가졌다.

$\epsilon$-cover는 또한 일반화 보장을 제공한다. 학습 데이터가 정상 분포를 잘 샘플링했다면, coreset은 그 분포의 핵심을 추출한 것이다. 테스트 시 정상 샘플은 높은 확률로 coreset 근처에 있을 것이고, 이상 샘플은 멀리 떨어져 있을 것이다. 이는 coreset 기반 k-NN 탐지의 이론적 근거가 된다.

### 3.2.3 Complexity Analysis

Coreset selection의 계산 복잡도는 실용성에 중요한 영향을 미친다. Naive greedy 알고리즘의 복잡도는 $O(|\mathcal{M}| \cdot |\mathcal{C}| \cdot d)$다. 각 iteration에서 모든 $|\mathcal{M}|$개 점에 대해 현재 coreset의 모든 $|\mathcal{C}|$개 원소와의 거리를 계산하고, 이를 $|\mathcal{C}|$번 반복해야 한다. 여기서 $d$는 특징 차원이다.

그러나 앞서 언급한 최소 거리 유지 전략을 사용하면 복잡도가 $O(|\mathcal{M}| \cdot |\mathcal{C}|)$로 개선된다. 각 iteration에서 모든 점의 거리를 한 번만 업데이트하면 된다. $|\mathcal{C}| \ll |\mathcal{M}|$이므로 이는 상당한 절감이다.

추가 최적화로 근사 최근접 이웃 탐색을 사용할 수 있다. FAISS나 Annoy 같은 라이브러리는 $O(\log |\mathcal{C}|)$의 쿼리 시간을 제공한다. 이를 사용하면 전체 복잡도가 $O(|\mathcal{M}| \cdot \log |\mathcal{C}|)$로 추가 개선된다. 다만 정확한 거리 계산이 필요한 coreset selection에서는 근사 탐색의 오차가 품질에 영향을 줄 수 있으므로 신중히 사용해야 한다.

실제 실행 시간은 합리적이다. MVTec AD의 한 카테고리(약 200장 이미지, 약 20만 개 패치)에서 1% coreset(2천 개)을 선택하는 데 GPU에서 약 1-2분이 소요된다. 이는 전체 학습 파이프라인에서 무시할 수 없는 시간이지만, PaDiM의 공분산 계산보다는 빠르다. 무엇보다 coreset selection은 한 번만 수행하면 되고, 이후 추론은 매우 빠르다.

메모리 복잡도는 $O(|\mathcal{M}| + |\mathcal{C}| \cdot d)$다. 원본 데이터의 거리 정보를 유지하고 coreset의 특징 벡터를 저장해야 한다. 최종 모델은 coreset만 유지하므로 $O(|\mathcal{C}| \cdot d)$의 메모리만 필요하다. PaDiM의 $O(h \times w \times d^2)$와 비교하면 극적인 감소다.

병렬화 가능성도 고려할 수 있다. 거리 계산은 데이터 병렬로 처리 가능하다. 여러 GPU를 사용하거나 CPU 멀티 쓰레딩으로 속도를 높일 수 있다. 다만 coreset 원소 선택 자체는 순차적이므로 병렬화가 제한적이다. 근사 방법으로 독립적으로 여러 coreset을 구성하고 병합하는 전략도 가능하지만, 품질 보장이 약해진다.

## 3.3 Technical Innovations

### 3.3.1 Locally Aware Patch Features

PatchCore의 중요한 혁신은 단순히 각 패치를 독립적으로 처리하는 것이 아니라, 주변 맥락을 고려하는 것이다. 전통적인 패치 기반 접근은 각 패치의 특징 벡터를 그대로 사용한다. 이는 국소적 texture나 색상 정보는 잘 포착하지만, 더 넓은 공간적 구조나 인접 패치와의 관계는 무시한다.

Locally aware patch features는 각 패치 특징에 이웃 패치들의 정보를 통합한다. 구체적으로 위치 $(i,j)$의 원본 특징 $\mathbf{f}_{ij}$에 대해, $3 \times 3$ 이웃 패치의 특징을 평균내어 더한다.

$$\mathbf{f}_{ij}^{\text{aware}} = \mathbf{f}_{ij} + \frac{1}{9}\sum_{(i',j') \in \mathcal{N}(i,j)} \mathbf{f}_{i'j'}$$

여기서 $\mathcal{N}(i,j)$는 $(i,j)$의 $3 \times 3$ 이웃을 나타낸다. 이 연산은 단순한 평균 풀링이지만, 효과는 상당하다. 각 패치가 이제 자신의 즉각적인 맥락을 인식하게 된다.

이 접근의 이점은 여러 가지다. 첫째, 노이즈 감소다. 개별 패치에 존재할 수 있는 random noise나 미세한 변동이 이웃 평균화로 완화된다. 이는 특징을 더 안정적이고 robust하게 만든다. 둘째, 수용 영역 확장이다. CNN의 한 층만 사용하면 제한된 수용 영역을 가지지만, 이웃 집계를 통해 효과적인 수용 영역이 넓어진다. 이는 더 큰 결함이나 구조적 이상을 포착하는 데 유리하다.

셋째, 암묵적인 spatial smoothness prior다. 인접 패치들은 일반적으로 유사한 특징을 가질 것이라는 가정을 반영한다. 이는 자연 이미지와 산업 이미지 모두에서 타당한 가정이다. 결함도 일반적으로 여러 패치에 걸쳐 나타나므로, 이웃 정보가 결함 탐지를 강화한다.

구현은 매우 간단하다. Average pooling with kernel size 3 and stride 1을 적용하면 된다. 이는 추가 계산 비용이 거의 없으며, 메모리 오버헤드도 없다. 특징 차원 $d$는 변하지 않고, 단지 각 특징 벡터가 더 informative해진다.

실험 결과 locally aware features는 일관되게 성능을 향상시켰다. 특히 큰 결함이나 구조적 이상에서 개선이 두드러졌다. 작은 scratch나 점 결함은 단일 패치 내에 국한되어 이득이 적지만, 큰 crack이나 변형은 여러 패치에 영향을 미쳐 이웃 맥락이 중요하다. 평균적으로 이미지 AUROC가 1-2%포인트 향상되었다.

### 3.3.2 Neighborhood Aggregation

Neighborhood aggregation은 locally aware features의 구체적인 구현 방식이다. 가장 단순한 형태는 앞서 설명한 평균 풀링이지만, 더 정교한 변형도 가능하다. 가중 평균을 사용하여 중심 패치에 더 큰 가중치를 줄 수 있다.

$$\mathbf{f}_{ij}^{\text{aware}} = w_0 \mathbf{f}_{ij} + \sum_{(i',j') \in \mathcal{N}(i,j) \setminus \{(i,j)\}} w_1 \mathbf{f}_{i'j'}$$

여기서 $w_0 > w_1$이고 가중치의 합이 1이 되도록 정규화한다. 논문에서는 단순 평균($w_0 = w_1 = 1/9$)이 이미 충분히 효과적이어서 복잡한 가중치 체계가 필요하지 않았다고 보고했다.

또 다른 변형은 adaptive aggregation이다. 패치 간 유사도에 기반하여 가중치를 동적으로 결정한다. 중심 패치와 유사한 이웃은 높은 가중치를, 다른 이웃은 낮은 가중치를 받는다. 이는 bilateral filtering의 아이디어를 차용한 것이다.

$$w_{ij,i'j'} = \exp\left(-\frac{\|\mathbf{f}_{ij} - \mathbf{f}_{i'j'}\|^2}{2\sigma^2}\right)$$

그러나 이는 추가 계산 비용을 요구하며, 실험 결과 성능 향상이 크지 않아 실용적이지 않다고 판단되었다. 단순 평균이 효율성과 효과의 최적 균형점이다.

Neighborhood aggregation의 또 다른 해석은 graph convolution으로 볼 수 있다. 패치들을 노드로, 인접 관계를 엣지로 하는 그래프에서 메시지 전달(message passing)을 수행하는 것이다. 각 노드는 이웃으로부터 정보를 받아 자신의 표현을 업데이트한다. 이는 한 층의 Graph Neural Network로 볼 수 있다.

Aggregation의 범위도 조절 가능하다. $3 \times 3$ 대신 $5 \times 5$나 $7 \times 7$ 이웃을 사용할 수 있다. 더 큰 범위는 더 넓은 맥락을 포착하지만, 지나치게 크면 오히려 관련 없는 정보가 섞여 signal이 약해진다. 실험 결과 $3 \times 3$이 대부분의 경우 최적이었다. 이는 CNN의 $3 \times 3$ 커널이 널리 사용되는 이유와 유사하다.

### 3.3.3 k-NN Anomaly Scoring

PatchCore는 PaDiM의 Mahalanobis 거리 대신 k-nearest neighbor (k-NN) 거리를 이상 점수로 사용한다. 테스트 패치 특징 $\mathbf{f}_{\text{test}}$에 대해, coreset $\mathcal{C}$ 내에서 $k$개의 최근접 이웃을 찾고, 그들과의 평균 거리를 계산한다.

$$s_k(\mathbf{f}_{\text{test}}) = \frac{1}{k} \sum_{i=1}^{k} \|\mathbf{f}_{\text{test}} - \mathbf{f}^{(i)}_{\text{nn}}\|_2$$

여기서 $\mathbf{f}^{(i)}_{\text{nn}}$는 $i$번째 최근접 이웃이다. 논문에서는 $k=9$를 사용했다. $k=1$인 경우는 outlier에 민감하고, 너무 큰 $k$는 멀리 있는 샘플의 영향을 과도하게 받는다. $k=9$는 경험적으로 좋은 균형점이었다.

k-NN 접근의 장점은 분포 가정이 없다는 것이다. PaDiM의 가우시안 가정은 실제 분포가 복잡할 때 부정확할 수 있다. k-NN은 비모수적(non-parametric) 방법으로, 데이터 자체로부터 직접 이상도를 추정한다. 이는 multimodal이나 비대칭적 분포에서도 robust하다.

또한 k-NN은 coreset과 자연스럽게 결합된다. Coreset 자체가 거리 기반 커버리지로 선택되었으므로, 거리 기반 이상 점수가 일관성 있다. 반면 가우시안 모델은 coreset에서 평균과 공분산을 재추정해야 하는데, 소수의 샘플로는 불안정할 수 있다.

k-NN 탐색은 효율적으로 구현 가능하다. FAISS 라이브러리는 GPU 가속 k-NN 탐색을 제공하며, 수백만 개 벡터에서도 밀리초 단위로 쿼리할 수 있다. PatchCore의 coreset은 수천에서 수만 개이므로, 실시간에 가까운 탐색이 가능하다. 정확한 k-NN 대신 approximate nearest neighbor (ANN)를 사용하면 더욱 빠르다.

$k$ 값의 선택은 하이퍼파라미터지만, 성능이 $k$에 크게 민감하지 않다. $k=5$에서 $k=15$ 범위에서 유사한 성능을 보였다. 이는 실무에서 튜닝 부담이 적다는 것을 의미한다. 일반적으로 $k=9$가 기본값으로 권장된다.

거리 메트릭으로는 유클리드 거리를 사용한다. 코사인 유사도나 Mahalanobis 거리도 가능하지만, 유클리드 거리가 가장 직관적이고 계산도 빠르다. 정규화된 특징에서는 유클리드와 코사인이 본질적으로 동등하다.

픽셀 레벨 이상 맵 생성은 PaDiM과 동일하다. 각 패치의 k-NN 점수를 원본 해상도로 업샘플링한다. 이미지 레벨 점수는 패치 점수의 최댓값으로 계산한다. 이는 이미지 내 가장 이상한 부분이 전체 이미지의 이상도를 결정한다는 가정이다.

## 3.4 PaDiM vs PatchCore Comparison

PaDiM과 PatchCore는 모두 memory-based 패러다임에 속하지만, 중요한 차이가 있다. 가장 근본적인 차이는 메모리 전략이다. PaDiM은 모든 패치 위치에서 가우시안 분포를 유지하므로 메모리가 $O(h \times w \times d^2)$다. PatchCore는 선택된 coreset만 유지하므로 $O(|\mathcal{C}| \times d)$다. 일반적으로 $|\mathcal{C}| \ll h \times w$이고 $d \ll d^2$이므로 PatchCore가 훨씬 메모리 효율적이다.

이상 점수 계산 방식도 다르다. PaDiM은 Mahalanobis 거리로 각 위치의 가우시안 분포로부터의 편차를 측정한다. PatchCore는 k-NN 거리로 전역 coreset으로부터의 거리를 측정한다. PaDiM은 위치별 분포를 모델링하므로 공간적 맥락이 암묵적이다. PatchCore는 위치 정보를 명시적으로 유지하지 않지만, locally aware features를 통해 맥락을 통합한다.

성능 측면에서 PatchCore가 일관되게 우수하다. MVTec AD에서 PatchCore는 이미지 AUROC 99.1%로 PaDiM의 97.5%를 상회한다. 픽셀 레벨에서도 98.1% vs 97.5%로 앞선다. 이는 1% coreset만 사용한 결과이므로 더욱 인상적이다. 성능 향상의 주된 이유는 coreset selection이 노이즈 제거 효과가 있고, locally aware features가 더 informative하기 때문이다.

학습 시간은 비슷한 수준이다. PaDiM의 공분산 계산과 PatchCore의 coreset selection이 비슷한 시간이 소요된다. 두 방법 모두 역전파가 없어 기존 end-to-end 방법보다 훨씬 빠르다. 추론 시간은 PatchCore가 약간 빠르다. Coreset 크기가 작아 k-NN 탐색이 빠르기 때문이다. PaDiM은 모든 위치의 Mahalanobis 거리를 계산해야 한다.

배포 관점에서 PatchCore가 훨씬 유리하다. 낮은 메모리 요구량은 엣지 디바이스 배포를 가능하게 한다. 다중 카테고리 시스템에서도 각 카테고리마다 작은 coreset만 유지하면 되므로 확장성이 좋다. PaDiM은 카테고리가 많아질수록 메모리가 선형적으로 증가하여 실용성이 제한된다.

하이퍼파라미터 측면에서 PatchCore가 조금 더 복잡하다. Coreset 비율과 k 값을 선택해야 한다. 그러나 이들은 기본값($1\%$, $k=9$)이 대부분의 경우 잘 작동한다. PaDiM은 차원 축소 목표 차원만 선택하면 되지만, 이 역시 기본값이 충분하다. 전반적으로 두 방법 모두 하이퍼파라미터 튜닝 부담이 적다.

해석 가능성 측면에서는 차이가 있다. PaDiM의 Mahalanobis 거리는 통계적 의미가 명확하다. 표준편차의 몇 배만큼 벗어났는지 정량화한다. PatchCore의 k-NN 거리는 직관적이지만 통계적 해석이 약하다. 그러나 실무에서는 절대적인 점수보다 상대적 순위가 중요하므로 큰 문제가 아니다.

두 방법 모두 사전 학습 백본에 의존한다. 백본의 품질이 최종 성능을 크게 좌우한다. ImageNet 사전 학습이 효과적이지만, 도메인 격차가 큰 경우 제한이 있다. 이는 memory-based 패러다임의 공통된 특성이다.

## 3.5 Performance Analysis

PatchCore는 MVTec AD 벤치마크에서 압도적인 성능을 보였다. 이미지 레벨 분류에서 평균 AUROC 99.1%를 달성하여, 당시 모든 경쟁 방법을 능가했다. 15개 카테고리 중 10개에서 99% 이상을 기록했고, bottle과 grid에서는 100%에 도달했다. 픽셀 레벨 세그멘테이션에서도 98.1%로 최고 성능이었다.

카테고리별 성능을 살펴보면 texture 카테고리에서 특히 탁월했다. Carpet(99.1%), grid(100%), leather(100%), tile(99.0%), wood(99.2%)에서 거의 완벽한 분류를 달성했다. 이들은 반복 패턴을 가지며 결함이 패턴 불규칙성으로 명확히 드러난다. Locally aware features가 texture 분석에 효과적이었다.

Object 카테고리에서도 우수한 성능을 보였지만 texture보다는 약간 낮았다. Bottle(100%), cable(99.5%), capsule(98.5%)에서 여전히 매우 높은 정확도를 유지했다. 그러나 screw(98.1%)와 같이 복잡한 3D 구조를 가진 객체에서는 상대적으로 어려움을 겪었다. 이는 포즈 변화나 조명 변화로 인한 정상 변동이 크기 때문이다.

가장 어려운 카테고리는 hazelnut(97.9%)와 metal_nut(98.3%)이었다. 이들은 텍스처와 구조가 복합적이고, 결함이 미세하거나 정상과 구별하기 어렵다. 예를 들어 hazelnut의 crack은 자연스러운 표면 texture와 혼동될 수 있다. 그럼에도 98% 가까운 성능은 인상적이다.

Coreset 크기의 영향을 분석한 결과, 1%에서 10% 범위에서 성능이 안정적이었다. 1% coreset도 대부분의 카테고리에서 최고 성능을 보였다. 일부 복잡한 카테고리에서는 5% 또는 10% coreset이 약간 더 나았지만, 차이는 1%포인트 미만이었다. 이는 매우 작은 coreset만으로도 충분함을 보여준다.

흥미롭게도 coreset을 사용하지 않은 경우(100%)보다 1% coreset의 성능이 더 좋은 경우도 있었다. 이는 coreset selection이 단순 압축이 아니라 데이터 정제 효과를 가져오기 때문이다. Redundant하거나 noisy한 샘플들이 제거되어 오히려 일반화가 개선된다. 이는 기계 학습의 일반적인 현상인 regularization 효과로 이해할 수 있다.

백본 네트워크 비교에서 Wide ResNet-50이 가장 효과적이었다. ResNet18 대비 1-2%포인트 성능 향상을 보였다. EfficientNet-B4도 유사한 성능을 보였다. 더 강력한 특징 추출기가 더 나은 결과를 가져온다는 것은 예상된 결과다. 그러나 계산 비용도 증가하므로 trade-off를 고려해야 한다.

추론 속도는 이미지당 약 30-50ms(GPU 기준)로 PaDiM과 유사하거나 약간 빠르다. Coreset 크기가 작아 k-NN 탐색이 효율적이기 때문이다. CPU에서는 약 200-300ms가 소요된다. 실시간은 아니지만 대부분의 검사 라인에서 충분한 속도다. FAISS를 사용한 ANN 탐색으로 더욱 가속할 수 있다.

다른 데이터셋에서도 PatchCore의 우수성이 검증되었다. BTAD에서 98.7%, VisA에서 96.8%의 이미지 AUROC를 달성했다. 이는 MVTec AD에 overfitting되지 않았으며, 다양한 산업 도메인에 적용 가능함을 보여준다.

## 3.6 Memory Efficiency Breakthrough

PatchCore의 가장 혁명적인 기여는 메모리 효율성의 극적인 향상이다. 이는 memory-based 방식의 실용적 배포를 가능하게 한 핵심 요인이다. 구체적인 수치로 비교하면 그 임팩트가 명확해진다.

PaDiM의 경우 $256 \times 256$ 입력 이미지에서 feature map 해상도가 약 $56 \times 56$이다. 차원 축소 후 특징 차원을 $d'=100$으로 하면, 공분산 행렬은 $56 \times 56 \times 100 \times 100$개의 float32 값을 저장해야 한다. 이는 약 1.2GB에 해당한다. 차원을 축소하지 않으면(d'=550) 메모리는 약 33GB로 폭발한다.

PatchCore는 1% coreset을 사용할 때 원본 패치 수의 1%만 저장한다. $56 \times 56 = 3136$개 패치 중 약 31개만 선택된다. 각 패치는 특징 벡터 하나($d=1536$ for Wide ResNet-50)를 저장하므로, 총 메모리는 $31 \times 1536 \times 4 = 190$KB다. 이는 PaDiM 대비 약 6000배 감소다. 10% coreset을 사용해도 1.9MB로 여전히 훨씬 작다.

이러한 메모리 절감은 여러 실용적 이점을 가져온다. 첫째, 엣지 디바이스 배포가 가능해진다. Raspberry Pi나 NVIDIA Jetson 같은 제한된 메모리의 임베디드 시스템에서도 실행할 수 있다. 둘째, 다중 카테고리 시스템에서 확장성이 확보된다. 수십 개 카테고리를 동시에 배포해도 총 메모리가 수백 MB에 불과하다.

셋째, 캐시 효율성이 향상된다. 작은 coreset은 CPU 캐시에 올라갈 수 있어 메모리 접근 속도가 빠르다. 이는 추론 속도 향상에도 기여한다. 넷째, 모델 전송과 업데이트가 용이하다. 네트워크를 통해 모델을 전송하거나 OTA(Over-The-Air) 업데이트를 수행할 때 작은 모델 크기는 큰 장점이다.

메모리 효율성은 단순히 같은 성능을 더 적은 메모리로 달성하는 것 이상의 의미가 있다. Coreset selection이 일종의 regularization으로 작용하여 과적합을 방지하고 일반화를 개선한다. 노이즈가 많거나 outlier 성격의 샘플들이 제거되어 더 clean한 메모리 뱅크가 구성된다.

실험적으로 coreset 크기와 성능의 관계를 분석한 결과, 0.1%에서 10% 범위에서 완만한 성능 곡선을 보였다. 0.1% coreset에서도 95% 이상의 AUROC를 유지했다. 1%에서 최적 성능에 도달했고, 그 이상 증가시켜도 개선이 미미했다. 이는 정보의 대부분이 소수의 대표 샘플에 집중되어 있음을 시사한다.

메모리-성능 trade-off 곡선은 실무에서 중요한 의사 결정 도구다. 엄격한 메모리 제약이 있는 경우 0.5% coreset으로 시작하고, 메모리가 충분하면 5% coreset으로 최대 안전 마진을 확보할 수 있다. 대부분의 경우 1-2% coreset이 최적 균형점이다.

## 3.7 Implementation Guide

PatchCore를 실무에 구현할 때 고려해야 할 구체적인 가이드라인이다. 먼저 백본 선택은 정확도와 효율성의 균형을 고려한다. Wide ResNet-50은 최고 성능을 제공하지만 느리다. ResNet18은 빠르고 가벼우며 대부분의 경우 충분한 성능을 보인다. EfficientNet 계열은 중간 지점이다. 일반적으로 ResNet18로 시작하여 성능이 부족한 경우 업그레이드하는 전략이 권장된다.

특징 추출 층은 layer2와 layer3의 조합이 기본이다. Layer1을 추가하면 texture 정보가 강화되지만 차원이 증가한다. Layer4는 너무 추상적이어서 일반적으로 제외한다. 실험을 통해 각 데이터셋에 최적 조합을 찾아야 한다.

Coreset 비율은 1%가 기본값이다. 메모리가 제한적인 환경에서는 0.5%로 줄이고, 최대 성능이 필요하면 5-10%로 늘린다. 대부분의 경우 1-2% 범위에서 충분하다. Coreset selection에는 수 분이 소요되므로 과도하게 큰 coreset은 학습 시간을 늘린다.

k-NN의 k 값은 9가 기본이다. 5에서 15 사이에서 성능이 안정적이므로 튜닝이 크게 필요 없다. 더 작은 k는 outlier에 민감하고, 더 큰 k는 멀리 있는 샘플의 영향이 커진다. 의심스럽다면 k=9를 사용하라.

이웃 집계를 위한 average pooling은 필수다. Kernel size는 3, stride는 1이 표준이다. 더 큰 kernel은 성능 향상이 미미하고 계산 비용만 증가한다. Pooling을 생략하면 성능이 눈에 띄게 저하된다.

임계값 설정은 validation set을 사용한다. Training set에서 coreset을 구성하고, validation set에서 ROC 곡선을 그려 최적 임계값을 찾는다. False negative 비용이 높은 산업 검사에서는 높은 재현율(95-99%)을 목표로 임계값을 낮게 설정한다. 이는 일부 false positive를 감수하는 것이다.

추론 최적화를 위해 FAISS 라이브러리를 사용한다. GPU 인덱스를 구성하면 k-NN 탐색이 매우 빠르다. Approximate nearest neighbor를 사용하면 추가 가속이 가능하지만, 정확도 손실을 모니터링해야 한다. IVF(Inverted File) 인덱스가 균형잡힌 선택이다.

배치 처리로 throughput을 높인다. 여러 이미지를 동시에 처리하여 GPU 활용률을 극대화한다. 특징 추출은 자연스럽게 배치 처리되고, k-NN 탐색도 배치 쿼리를 지원한다. 배치 크기는 GPU 메모리에 맞춰 조정한다.

다중 카테고리 배포 시 각 카테고리마다 별도 coreset을 유지한다. 백본 가중치는 공유하여 메모리를 절약한다. 추론 시 카테고리를 식별한 후 해당 coreset을 사용한다. 카테고리 식별은 별도의 분류기나 메타데이터로 수행한다.

정상 데이터의 품질 관리가 중요하다. 학습 데이터에 실제로는 불량인 샘플이 섞여 있으면 coreset이 오염된다. 데이터 수집 시 엄격한 품질 관리를 하고, 이상치 탐지로 의심 샘플을 제거한다. Coreset selection 자체가 어느 정도 outlier를 배제하지만, 명백한 불량은 사전에 제거하는 것이 좋다.

정기적인 재학습이 필요하다. 제품이나 공정이 변경되면 정상 패턴도 변한다. 월별 또는 분기별로 새로운 데이터로 coreset을 재구성한다. 재학습은 빠르므로(수 분) 자주 수행해도 부담이 적다. Drift 탐지 알고리즘으로 재학습 시점을 자동 결정할 수 있다.

모니터링과 피드백 루프를 구축한다. 오탐과 미탐 사례를 수집하고 분석한다. 패턴이 발견되면 학습 데이터를 보강하거나 하이퍼파라미터를 조정한다. 특히 신규 결함 유형이 발견되면 즉시 대응해야 한다. Anomaly 점수 분포를 지속적으로 모니터링하여 시스템 건강도를 확인한다.

# 4. DFKDE (2022)

## 4.1 Basic Information

DFKDE(Deep Feature Kernel Density Estimation)는 2022년 Rippel 등이 제안한 memory-based 이상 탐지 방법으로, 비모수적 밀도 추정 기법인 Kernel Density Estimation을 딥러닝 특징에 적용했다. 이 연구는 기존 memory-based 방법들의 parametric 가정을 완화하고, 더 유연한 분포 모델링을 시도했다는 점에서 의미가 있다.

DFKDE의 핵심 아이디어는 PaDiM의 가우시안 가정을 제거하는 것이다. PaDiM은 각 패치 위치의 정상 특징 분포를 단일 가우시안으로 모델링한다. 이는 수학적으로 tractable하고 효율적이지만, 실제 분포가 multimodal이거나 비대칭적인 경우 부정확할 수 있다. DFKDE는 kernel density estimation을 사용하여 데이터 자체로부터 분포를 직접 추정한다. 이는 어떤 형태의 분포도 근사할 수 있는 비모수적 접근이다.

방법론적으로 DFKDE는 각 패치 위치에서 학습 샘플들을 그대로 유지하고, 테스트 샘플의 밀도를 kernel 함수의 합으로 계산한다. 밀도가 낮은 영역은 정상 분포에서 멀리 떨어진 것으로 간주되어 높은 이상 점수를 받는다. 이는 확률론적으로 well-founded된 접근으로, 이상도를 직접적으로 확률 밀도로 해석할 수 있다.

그러나 DFKDE는 실용적으로는 PaDiM이나 PatchCore만큼 성공적이지 못했다. MVTec AD에서 이미지 레벨 AUROC 약 96-97%로, PaDiM(97.5%)보다 약간 낮고 PatchCore(99.1%)보다 크게 낮다. 더 심각한 문제는 계산 비용과 메모리 사용량이다. KDE는 모든 학습 샘플을 유지해야 하므로 메모리가 많이 필요하고, 추론 시 모든 샘플과의 거리를 계산해야 하므로 느리다. 이러한 한계로 인해 DFKDE는 학술적 관심은 받았지만 실무 배포에서는 제한적이다.

DFKDE의 기여는 주로 이론적이다. 비모수적 접근의 가능성을 보여주고, parametric 가정의 한계를 명확히 했다. 또한 bandwidth selection과 같은 KDE의 고전적 문제들이 고차원 딥러닝 특징에서도 여전히 중요함을 입증했다. 후속 연구들은 DFKDE의 아이디어를 차용하되 계산 효율성을 개선하는 방향으로 발전했다.

## 4.2 Kernel Density Estimation

### 4.2.1 KDE Fundamentals

Kernel Density Estimation은 주어진 샘플들로부터 확률 밀도 함수를 추정하는 비모수적 방법이다. Parametric 방법(예: 가우시안)이 특정 함수 형태를 가정하는 반면, KDE는 데이터 자체의 구조를 따라간다. 수학적으로 $N$개의 샘플 $\{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$이 주어졌을 때, 위치 $\mathbf{x}$에서의 밀도 추정은 다음과 같다.

$$\hat{f}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} K_h(\mathbf{x} - \mathbf{x}_i)$$

여기서 $K_h$는 bandwidth $h$를 가진 kernel 함수다. Kernel 함수는 각 샘플 주변에 확률 질량을 분산시키는 역할을 한다. 직관적으로 각 샘플을 중심으로 한 작은 "언덕"들을 쌓아 올려 전체 밀도 풍경을 형성한다고 볼 수 있다.

KDE의 가장 큰 장점은 유연성이다. Multimodal 분포, 비대칭 분포, 임의 형태의 분포를 모두 근사할 수 있다. 샘플이 충분히 많으면 실제 분포에 수렴한다는 이론적 보장도 있다. 또한 결과가 직관적으로 해석 가능하다. 밀도가 높은 영역은 샘플이 많이 관찰된 정상 영역이고, 낮은 영역은 희귀한 이상 영역이다.

단점은 계산 비용이다. 새로운 점의 밀도를 평가하려면 모든 $N$개 학습 샘플과의 거리를 계산해야 한다. 이는 $O(Nd)$ 복잡도로, 대규모 데이터에서는 prohibitive하다. 또한 모든 학습 샘플을 메모리에 유지해야 하므로 메모리 효율적이지 않다. 고차원에서는 curse of dimensionality로 인해 필요한 샘플 수가 기하급수적으로 증가한다.

Bandwidth $h$는 smoothing의 정도를 조절하는 critical parameter다. 작은 $h$는 샘플 주변의 좁은 영역만 영향을 받아 밀도가 울퉁불퉁해진다. 극단적으로 $h \to 0$이면 각 샘플 위치에서만 밀도가 무한대이고 다른 곳은 0이다. 큰 $h$는 넓은 영역에 확률 질량을 분산시켜 밀도가 매끄럽지만 세부 구조가 사라진다. $h \to \infty$이면 모든 곳에서 동일한 uniform 밀도가 된다.

Optimal bandwidth는 데이터의 특성에 따라 다르다. Silverman's rule of thumb은 1차원에서 $h = 1.06 \sigma N^{-1/5}$를 제안한다. 여기서 $\sigma$는 표준편차다. 다차원으로 확장하면 $h = \left(\frac{4}{d+2}\right)^{1/(d+4)} \sigma N^{-1/(d+4)}$가 된다. 그러나 이는 가우시안 분포를 가정하므로 실제 데이터에서는 cross-validation으로 최적값을 찾는다.

KDE를 이상 탐지에 적용하는 것은 자연스럽다. 정상 샘플로부터 밀도를 추정하고, 테스트 샘플의 밀도가 낮으면 이상으로 판단한다. 이상 점수는 단순히 음의 로그 밀도로 정의된다.

$$s(\mathbf{x}) = -\log \hat{f}(\mathbf{x})$$

높은 밀도(정상)는 낮은 이상 점수를, 낮은 밀도(이상)는 높은 이상 점수를 받는다. 로그를 취하는 이유는 밀도가 매우 작은 값이 될 수 있어 수치적 안정성을 위해서다. 또한 로그는 곱셈을 덧셈으로 바꿔 계산을 단순화한다.

### 4.2.2 Gaussian Kernel

가장 널리 사용되는 kernel은 Gaussian kernel이다. 수학적으로 다음과 같이 정의된다.

$$K_h(\mathbf{u}) = \frac{1}{(2\pi h^2)^{d/2}} \exp\left(-\frac{\|\mathbf{u}\|^2}{2h^2}\right)$$

여기서 $d$는 차원이고 $\mathbf{u} = \mathbf{x} - \mathbf{x}_i$는 평가 위치와 샘플 간의 차이다. Gaussian kernel은 샘플을 중심으로 한 종 모양 곡선으로, 가까운 점에 높은 가중치를, 먼 점에 낮은 가중치를 부여한다.

Gaussian kernel의 장점은 여러 가지다. 첫째, 매끄럽고 미분 가능하여 수학적으로 다루기 쉽다. 둘째, 무한 지지(infinite support)를 가져 모든 샘플이 약간씩이라도 기여한다. 셋째, 잘 이해된 통계적 성질을 가진다. 넷째, 가우시안 분포의 친숙함으로 해석이 직관적이다.

단점은 계산 비용이다. 모든 샘플의 기여를 계산해야 하므로 $O(N)$ 평가가 필요하다. 유한 지지를 가진 kernel(예: Epanechnikov)은 거리 임계값 이상의 샘플을 무시하여 계산을 줄일 수 있지만, Gaussian은 그렇지 않다. 실제로는 매우 먼 샘플의 기여가 negligible하므로 cutoff를 적용할 수 있다.

$$K_h(\mathbf{u}) \approx \begin{cases} \frac{1}{(2\pi h^2)^{d/2}} \exp\left(-\frac{\|\mathbf{u}\|^2}{2h^2}\right) & \text{if } \|\mathbf{u}\| < ch \\ 0 & \text{otherwise} \end{cases}$$

여기서 $c$는 cutoff threshold(일반적으로 3-5)다. $\|\mathbf{u}\| > 3h$인 점의 기여는 전체의 0.3% 미만이므로 무시해도 된다. 이렇게 하면 공간 인덱싱(예: KD-tree)을 활용하여 가까운 이웃만 찾아 계산할 수 있다.

고차원에서 Gaussian kernel은 curse of dimensionality에 취약하다. 차원 $d$가 증가하면 정규화 상수 $(2\pi h^2)^{-d/2}$가 급격히 변한다. 또한 고차원에서는 대부분의 점이 서로 멀리 떨어져 있어 kernel 기여가 거의 0에 가깝다. 이는 밀도 추정을 어렵게 만든다. DFKDE는 차원 축소나 metric learning으로 이 문제를 완화한다.

Bandwidth $h$는 Gaussian kernel의 spread를 조절한다. 작은 $h$는 좁고 뾰족한 kernel을, 큰 $h$는 넓고 평평한 kernel을 만든다. 모든 샘플에 동일한 $h$를 사용하는 것이 일반적이지만, adaptive bandwidth를 사용할 수도 있다. 각 샘플 $\mathbf{x}_i$마다 고유의 $h_i$를 가져 데이터 밀도에 따라 조절한다. 밀집 영역에서는 작은 bandwidth로 세밀하게, 희소 영역에서는 큰 bandwidth로 넓게 커버한다.

### 4.2.3 Bandwidth Selection

Bandwidth selection은 KDE의 성능을 결정하는 가장 중요한 문제다. 너무 작으면 overfitting(과적합)으로 noise까지 학습하고, 너무 크면 underfitting(과소적합)으로 중요한 구조를 놓친다. 최적 bandwidth는 bias-variance trade-off를 균형잡는다.

이론적으로 Mean Integrated Squared Error(MISE)를 최소화하는 bandwidth를 찾는다.

$$\text{MISE}(h) = \mathbb{E}\left[\int \left(\hat{f}_h(\mathbf{x}) - f(\mathbf{x})\right)^2 d\mathbf{x}\right]$$

여기서 $f$는 실제 밀도, $\hat{f}_h$는 bandwidth $h$를 사용한 추정 밀도다. 그러나 실제 밀도를 알 수 없으므로 직접 계산할 수 없다. 대신 cross-validation이나 plug-in 방법을 사용한다.

Least Squares Cross-Validation(LSCV)은 leave-one-out 방식으로 각 샘플을 제외한 나머지로 밀도를 추정하고, 제외된 샘플에서의 예측 오차를 측정한다. 모든 샘플에 대한 평균 오차를 최소화하는 $h$를 선택한다.

$$\text{LSCV}(h) = \int \hat{f}_h(\mathbf{x})^2 d\mathbf{x} - \frac{2}{N}\sum_{i=1}^{N} \hat{f}_{h,-i}(\mathbf{x}_i)$$

여기서 $\hat{f}_{h,-i}$는 $\mathbf{x}_i$를 제외하고 추정한 밀도다. 이 방법은 데이터로부터 직접 최적 bandwidth를 찾으므로 distribution-free지만, 계산이 매우 expensive하다.

Silverman's rule of thumb은 계산이 간단한 경험적 규칙이다. 데이터가 가우시안 분포를 따른다고 가정하고, MISE를 최소화하는 bandwidth를 closed-form으로 유도한다. 1차원에서는 $h = 1.06 \sigma N^{-1/5}$이고, 다차원에서는 각 차원마다 독립적으로 적용하거나 평균 표준편차를 사용한다.

Scott's rule은 유사하지만 약간 다른 상수를 사용한다. $h = N^{-1/(d+4)} \sigma$. 두 규칙 모두 빠르고 reasonable하지만, 실제 분포가 가우시안에서 크게 벗어나면 suboptimal하다. 특히 multimodal이나 skewed 분포에서는 문제가 된다.

Adaptive bandwidth는 데이터의 지역적 특성에 따라 bandwidth를 조절한다. Nearest neighbor distance를 기반으로 각 점마다 $h_i$를 설정한다.

$$h_i = h_0 \cdot \left(\frac{\hat{f}(\mathbf{x}_i)}{\text{geometric mean}(\hat{f})}\right)^{-\alpha}$$

여기서 $h_0$는 기준 bandwidth, $\alpha$는 sensitivity parameter(일반적으로 0.5)다. 밀도가 높은 영역에서는 작은 bandwidth로, 낮은 영역에서는 큰 bandwidth로 자동 조절된다. 이는 두 단계 절차가 필요하다. 먼저 고정 bandwidth로 pilot 밀도를 추정하고, 이를 기반으로 adaptive bandwidth를 계산한다.

DFKDE에서는 validation set을 사용한 grid search로 bandwidth를 선택한다. 여러 $h$ 값을 시도하고 validation set에서 이상 탐지 성능(AUROC)이 최고인 것을 택한다. 이는 task-specific하므로 theoretical optimality와 다를 수 있지만, 최종 목표(이상 탐지)에 직접 최적화된다.

고차원에서는 bandwidth selection이 특히 어렵다. 샘플 수가 충분하지 않아 밀도 추정이 불안정하다. DFKDE는 차원 축소를 먼저 수행하여 effective dimension을 줄인다. 또한 각 차원마다 서로 다른 bandwidth를 사용하는 product kernel을 고려할 수 있다.

$$K_h(\mathbf{u}) = \prod_{j=1}^{d} K_{h_j}(u_j)$$

이는 각 특징 차원의 스케일에 맞춰 bandwidth를 조절할 수 있지만, $d$개의 parameter를 선택해야 하므로 더 복잡하다.

## 4.3 Deep Feature Integration

DFKDE의 핵심은 KDE를 CNN 특징 공간에 적용하는 것이다. 이는 전통적인 KDE와 현대적인 딥러닝의 결합으로, 몇 가지 중요한 설계 결정이 필요하다.

먼저 특징 추출 방식은 PaDiM과 유사하다. 사전 학습된 CNN(예: ResNet, EfficientNet)의 중간 층에서 특징 맵을 추출한다. 여러 층의 특징을 결합하여 multi-scale 정보를 활용한다. 각 패치 위치 $(i,j)$에서 특징 벡터 $\mathbf{f}_{ij} \in \mathbb{R}^d$를 얻는다. 차원 $d$는 일반적으로 수백에서 수천이다.

이러한 고차원 특징에 KDE를 직접 적용하면 문제가 발생한다. Curse of dimensionality로 인해 샘플 수가 차원에 비해 턱없이 부족하다. 고차원에서는 대부분의 점이 멀리 떨어져 있어 kernel 기여가 거의 0이다. 밀도 추정이 매우 불안정하고 신뢰할 수 없다.

DFKDE는 차원 축소를 통해 이 문제를 완화한다. PCA(Principal Component Analysis)나 random projection을 사용하여 특징 차원을 50-200 정도로 줄인다. 이는 정보 손실을 최소화하면서도 KDE를 적용 가능하게 만든다. 차원 축소는 학습 샘플의 특징에 대해 수행되고, 동일한 변환이 테스트 샘플에 적용된다.

각 패치 위치마다 독립적인 KDE 모델을 학습한다. 위치 $(i,j)$에서 수집된 $N$개의 학습 샘플 특징 $\{\mathbf{f}_{ij}^{(1)}, \ldots, \mathbf{f}_{ij}^{(N)}\}$로부터 밀도 함수 $\hat{f}_{ij}$를 추정한다. 이는 PaDiM의 위치별 가우시안 모델과 유사한 전략이지만, 분포 형태에 제약이 없다.

테스트 시 각 위치의 밀도를 평가한다. 위치 $(i,j)$의 테스트 특징 $\mathbf{f}_{ij}^{\text{test}}$에 대해 KDE로 밀도를 계산하고, 음의 로그 밀도를 이상 점수로 사용한다.

$$s_{ij} = -\log \hat{f}_{ij}(\mathbf{f}_{ij}^{\text{test}}) = -\log \left(\frac{1}{N}\sum_{n=1}^{N} K_h(\mathbf{f}_{ij}^{\text{test}} - \mathbf{f}_{ij}^{(n)})\right)$$

이상 맵은 각 위치의 이상 점수를 원본 해상도로 업샘플링하여 생성된다. 이미지 레벨 점수는 위치별 점수의 최댓값이다.

계산 효율성을 위한 최적화가 필요하다. Naive 구현은 각 테스트 패치마다 $N$개 학습 샘플과의 거리를 계산해야 하므로 $O(Nhw)$의 복잡도를 가진다. 이는 PaDiM의 $O(hw)$보다 훨씬 느리다. 공간 인덱싱(KD-tree, Ball tree)을 사용하여 가까운 이웃만 찾으면 $O(\log N)$로 줄일 수 있지만, 고차원에서는 효과가 제한적이다.

Bandwidth는 validation set에서 grid search로 선택된다. 각 카테고리마다 최적 bandwidth가 다를 수 있으므로 별도로 튜닝한다. 일반적으로 $h$가 작을수록 정밀하지만 noisy하고, 클수록 smooth하지만 coarse하다. 실험 결과 적절한 범위는 데이터의 평균 nearest neighbor distance의 0.5-2배였다.

메모리 측면에서 DFKDE는 모든 학습 샘플의 특징을 저장해야 한다. 이는 $O(Nhwd)$의 메모리가 필요하다. 차원 축소 후에도 $O(Nhwd')$로, PaDiM의 공분산 행렬 $O(hwd'^2)$과 비교해 $N$배 더 크다. 일반적으로 $N \sim 200$이고 $d' \sim 100$이므로 $N > d'$이어서 DFKDE가 더 많은 메모리를 사용한다. PatchCore의 coreset $O(|\mathcal{C}|d')$와 비교하면 훨씬 비효율적이다.

실무적으로 DFKDE는 메모리와 속도 제약으로 인해 제한적이다. Coreset selection을 DFKDE에 적용하려는 시도가 있었지만, KDE는 전역적 밀도를 추정해야 하므로 coreset만으로는 부족하다. 결국 DFKDE는 학술적 흥미는 있지만 실용성이 떨어진다는 평가를 받는다.

## 4.4 Comparison with Parametric Methods

DFKDE와 parametric 방법들(PaDiM, PatchCore)의 비교는 비모수 vs 모수 접근의 trade-off를 보여준다. 이론적 관점에서 DFKDE는 더 유연하다. 가우시안이나 특정 분포 형태를 가정하지 않으므로 어떤 분포도 충분한 샘플이 있으면 근사할 수 있다. Multimodal 분포, 비대칭 분포, 복잡한 형태의 분포를 모두 처리 가능하다. PaDiM은 단일 가우시안 가정으로 제한되고, PatchCore는 거리 기반이라 분포 형태를 명시적으로 모델링하지 않는다.

실제 데이터에서 정상 패턴이 multimodal일 수 있다. 예를 들어 조립 제품에서 여러 정상 구성이 가능하거나, 조명이나 포즈 변화로 서로 다른 appearance mode가 존재할 수 있다. 이런 경우 단일 가우시안은 부적절하고, KDE가 이론적으로 우위에 있다. 그러나 실험 결과 성능 향상이 기대만큼 크지 않았다.

성능 면에서 DFKDE는 PaDiM과 비슷하거나 약간 낮다. MVTec AD에서 이미지 AUROC 약 96-97%로, PaDiM의 97.5%에 미치지 못한다. PatchCore의 99.1%와는 큰 격차가 있다. 이는 고차원의 제한된 샘플에서 KDE가 이론적 장점을 실현하지 못함을 시사한다. Curse of dimensionality가 유연성의 이점을 상쇄한다.

일부 카테고리에서는 DFKDE가 약간 우위를 보였다. 특히 복잡한 정상 변동을 가진 경우(예: hazelnut)에서 multimodal 분포를 더 잘 포착했다. 그러나 대부분의 카테고리에서는 가우시안 가정이 충분히 좋은 근사였다. 이는 CNN 특징 공간이 어느 정도 well-behaved되어 있음을 의미한다.

계산 비용에서 DFKDE는 불리하다. PaDiM은 추론 시 단순히 Mahalanobis 거리를 계산하면 되므로 $O(d'^2)$ 복잡도다. 사전에 $\boldsymbol{\Sigma}^{-1}$을 계산해두면 $O(d')$로 줄어든다. DFKDE는 $N$개 샘플과의 거리를 모두 계산해야 하므로 $O(Nd')$다. $N \sim 200$이면 PaDiM보다 200배 느리다. PatchCore도 $k$-NN 탐색이 필요하지만 coreset 크기가 작고 FAISS 최적화로 빠르다.

메모리 측면에서도 DFKDE가 비효율적이다. PaDiM은 $O(hwd'^2)$, DFKDE는 $O(Nhwd')$, PatchCore는 $O(|\mathcal{C}|d')$다. 전형적인 값($h=w=56$, $N=200$, $d'=100$, $|\mathcal{C}|=3000$)을 대입하면 PaDiM은 약 125MB, DFKDE는 약 625MB, PatchCore는 약 1.2MB다. DFKDE가 PaDiM보다 5배, PatchCore보다 500배 크다.

Bandwidth selection은 DFKDE만의 추가 부담이다. PaDiM은 regularization parameter 하나만 있고 기본값이 robust하다. PatchCore는 coreset 비율과 $k$가 있지만 역시 기본값이 충분하다. DFKDE는 bandwidth를 신중히 선택해야 하며, 데이터마다 최적값이 크게 다를 수 있다. 이는 하이퍼파라미터 튜닝 비용을 증가시킨다.

해석 가능성에서는 차이가 있다. DFKDE의 밀도는 확률론적으로 명확한 의미를 가진다. 낮은 밀도는 rare event를 의미하고, 이상도를 직접 확률로 해석할 수 있다. PaDiM의 Mahalanobis 거리도 통계적 의미가 있지만 가우시안 가정 하에서다. PatchCore의 k-NN 거리는 직관적이지만 확률적 해석이 약하다. 그러나 실무에서는 절대값보다 상대적 순위가 중요하므로 이 차이가 실질적 영향이 적다.

결론적으로 DFKDE의 이론적 장점이 실무적 이점으로 이어지지 못했다. 고차원 제한 샘플 환경에서 비모수 방법의 유연성이 충분히 발휘되지 않고, 계산/메모리 비용만 증가했다. Parametric 방법, 특히 PatchCore의 실용성이 압도적이다. DFKDE는 학술적 기여는 있지만 실제 배포에서는 거의 사용되지 않는다.

## 4.5 Performance and Limitations

DFKDE의 실험적 성능은 기대에 미치지 못했다. MVTec AD 벤치마크에서 이미지 레벨 AUROC는 약 96-97% 범위로, 발표 당시 이미 PaDiM(97.5%)과 PatchCore(99.1%)에 뒤처졌다. 픽셀 레벨에서도 유사하게 95-96%로 경쟁 방법보다 낮았다. 일부 카테고리에서는 decent한 결과를 보였지만, 일관성이 부족했다.

카테고리별로 살펴보면 texture 기반에서는 상대적으로 선전했다. Grid(97.8%), leather(98.2%), tile(97.5%) 등에서 합리적인 성능을 보였다. 이들은 반복 패턴이 명확하고 정상 변동이 제한적이어서 KDE가 잘 작동한다. 그러나 최고 성능은 아니었고 PatchCore에 여전히 뒤졌다.

Object 카테고리에서는 더 어려움을 겪었다. Bottle(95.3%), hazelnut(96.1%), screw(95.8%) 등에서 경쟁 방법보다 눈에 띄게 낮았다. 복잡한 3D 구조, 다양한 포즈, 조명 변화 등으로 인해 정상 분포가 복잡해지면서 제한된 샘플로 KDE가 불안정해졌다. 이론적으로 이런 복잡한 분포에서 KDE가 유리해야 하지만, 실제로는 샘플 부족으로 오히려 불리했다.

Bandwidth 선택에 대한 sensitivity 분석 결과, 성능이 bandwidth에 상당히 민감했다. 최적값에서 1-2%포인트 벗어나면 성능이 급격히 저하되었다. 이는 실무에서 문제다. 새로운 데이터에 적용할 때마다 신중한 튜닝이 필요하고, validation set이 대표적이지 않으면 suboptimal bandwidth를 선택할 수 있다. PaDiM과 PatchCore는 기본 하이퍼파라미터가 robust하여 이런 문제가 적다.

차원 축소의 영향도 분석되었다. 원본 차원(수백-수천)에서 KDE를 직접 적용하면 성능이 매우 나빴다. 고차원 curse로 인해 밀도 추정이 불가능했다. 차원을 100-200으로 줄이면 성능이 크게 개선되지만, 최적 축소 차원을 찾는 것도 하이퍼파라미터 문제다. 너무 낮으면 정보 손실, 너무 높으면 curse of dimensionality가 발생한다.

계산 시간은 실용성의 큰 장애물이다. 학습 시간은 PaDiM과 비슷하게 수 분 내외다. 그러나 추론 시간이 문제다. 이미지당 약 200-500ms(GPU 기준)로 PaDiM(50-100ms)이나 PatchCore(30-50ms)보다 4-10배 느리다. CPU에서는 초 단위로 늘어나 실시간 검사에 적합하지 않다. FAISS 같은 최적화를 적용해도 근본적인 $O(N)$ 복잡도는 피할 수 없다.

메모리 사용량도 배포의 장벽이다. 한 카테고리당 약 500MB-1GB가 필요하여 PaDiM(수백 MB)보다 크고 PatchCore(수 MB)보다 훨씬 크다. 다중 카테고리 시스템에서는 메모리가 선형적으로 증가하여 수십 GB에 달할 수 있다. 엣지 디바이스 배포는 사실상 불가능하다.

Coreset selection을 DFKDE에 적용하려는 시도가 있었지만 제한적이었다. KDE는 전역적 밀도 추정이므로 local representative samples만으로는 부족하다. Coreset이 분포의 꼬리(tail) 부분을 놓치면 이상 탐지 성능이 크게 저하된다. PatchCore의 coreset은 거리 기반이라 다른 메커니즘이다.

DFKDE의 긍정적 측면도 있다. 확률적으로 well-founded된 접근으로 이론적 분석이 가능하다. 밀도를 직접 추정하므로 novelty score의 의미가 명확하다. 일부 복잡한 분포에서는 parametric 방법보다 나은 모델링을 보였다. Bandwidth를 잘 조정하면 competitive한 성능도 가능하다.

그러나 전반적으로 한계가 장점을 압도한다. 비모수 방법의 유연성이 고차원 제한 샘플 환경에서 실현되지 못했다. 계산과 메모리 비용이 과도하여 실용성이 떨어진다. Hyperparameter sensitivity가 높아 robust deployment가 어렵다. 결과적으로 DFKDE는 학술적으로는 흥미롭지만 산업 적용에서는 거의 사용되지 않는다.

Memory-based 패러다임 내에서 DFKDE는 parametric vs non-parametric의 교훈을 제공한다. 이론적 유연성이 항상 실무적 이점으로 이어지지 않는다. 특히 high-dimensional low-sample regime에서는 simple parametric 가정이 오히려 유리할 수 있다. PaDiM의 가우시안 가정이 restrictive해 보이지만 실제로는 충분히 좋은 근사다. PatchCore의 distance-based non-parametric 접근은 computational efficiency를 유지하면서도 flexibility를 제공한다.

향후 연구 방향으로는 efficient KDE 변형이 고려될 수 있다. Fast KDE 알고리즘, product kernel for dimension independence, sparse kernel approximation 등이 가능하다. 그러나 근본적인 curse of dimensionality는 피하기 어렵다. 차라리 manifold learning으로 intrinsic dimension을 찾아 저차원에서 KDE를 적용하는 것이 더 유망할 수 있다.

결론적으로 DFKDE는 memory-based 패러다임의 한 가능성을 탐구했지만, PaDiM과 PatchCore에 비해 실용적 가치가 낮다. 학술적으로는 비모수 접근의 한계를 명확히 하는 기여를 했고, 실무적으로는 simpler is better라는 교훈을 제공한다. Memory-based 방법의 미래는 PatchCore 방향, 즉 효율적인 샘플 선택과 거리 기반 scoring에 있다고 볼 수 있다.

# 5. Comprehensive Comparison

## 5.1 Technical Evolution

Memory-based 패러다임의 기술적 진화는 2020년부터 2022년까지 약 2년간 급격히 진행되었다. 이 짧은 기간 동안 세 가지 주요 접근법이 등장했으며, 각각은 이전 방법의 한계를 해결하면서 패러다임을 발전시켰다.

PaDiM(2020)은 memory-based 접근의 기초를 확립했다. 사전 학습된 CNN 특징을 직접 활용하고 패치 단위로 가우시안 분포를 모델링하는 프레임워크를 제시했다. 이는 end-to-end 학습 방법들의 불안정성과 긴 학습 시간 문제를 해결했다. 역전파 없이 단순히 통계량을 계산하는 방식은 학습을 수 분으로 단축시켰고, 재현성을 크게 향상시켰다. Mahalanobis 거리는 특징 공간의 기하학적 구조를 반영하여 단순 유클리드 거리보다 정교한 이상 탐지를 가능하게 했다.

그러나 PaDiM의 치명적 약점은 메모리였다. 각 패치 위치마다 full 공분산 행렬을 저장해야 했고, 이는 $O(h \times w \times d^2)$의 메모리를 요구했다. 차원 축소로 완화했지만 여전히 기가바이트 단위의 메모리가 필요했다. 다중 카테고리 배포나 엣지 디바이스 적용이 사실상 불가능했다. 또한 위치별 독립 모델링은 공간적 맥락을 충분히 활용하지 못했다.

DFKDE(2022)는 PaDiM의 parametric 가정을 완화하려 했다. 가우시안 분포가 실제 정상 패턴을 제대로 포착하지 못할 수 있다는 문제의식에서 출발했다. KDE를 사용한 비모수적 접근은 이론적으로 어떤 분포도 모델링할 수 있는 유연성을 제공했다. Multimodal이나 비대칭 분포도 처리 가능하다는 장점이 있었다.

그러나 DFKDE는 실무적으로 실패했다. 고차원 제한 샘플 환경에서 KDE의 유연성이 오히려 독이 되었다. Curse of dimensionality로 인해 밀도 추정이 불안정했고, 성능이 PaDiM보다 낮았다. 계산 비용과 메모리 사용량은 더 증가했다. 모든 학습 샘플을 유지해야 하고 추론 시 전체 샘플과 거리를 계산해야 했다. Bandwidth selection이라는 추가 하이퍼파라미터 부담도 발생했다.

PatchCore(2022)는 memory-based 패러다임의 결정적 돌파구였다. 두 가지 핵심 혁신으로 이전 방법들의 모든 주요 문제를 해결했다. 첫째, coreset selection은 메모리 문제를 극적으로 해결했다. Greedy k-center 알고리즘으로 전체 샘플의 1-10%만 선택하여 메모리를 100배 가까이 줄였다. 놀랍게도 성능은 오히려 향상되었다. Coreset selection이 노이즈 제거와 일반화 개선 효과를 가져왔기 때문이다.

둘째, locally aware patch features는 공간적 맥락을 통합했다. 이웃 패치들의 평균을 특징에 더하여 수용 영역을 확장했다. 이는 단순하지만 효과적이었다. 노이즈 감소, 큰 결함 탐지 개선, spatial smoothness prior 반영 등의 이점을 가져왔다. k-NN 기반 이상 점수는 분포 가정 없이 robust한 탐지를 제공했다.

PatchCore의 성공으로 memory-based 패러다임은 성숙 단계에 도달했다. MVTec AD에서 99.1%의 이미지 AUROC를 달성하여 single-class 이상 탐지의 사실상 표준이 되었다. 메모리 효율성으로 실제 배포가 가능해졌고, 빠른 학습과 안정성으로 실무 적용이 용이해졌다. 이후 연구들은 PatchCore를 baseline으로 삼고 incremental improvements를 추구하거나, 다른 패러다임(foundation models)으로 방향을 전환했다.

기술적 진화의 교훈은 명확하다. 이론적 정교함보다 실용적 효율성이 중요하다. PaDiM의 단순한 가우시안이 DFKDE의 유연한 KDE보다 나았다. Simple parametric 가정이 고차원 제한 샘플 환경에서는 오히려 유리하다. Coreset selection처럼 data efficiency를 극대화하는 기법이 결정적 차이를 만든다. Locally aware features처럼 간단하지만 효과적인 휴리스틱이 복잡한 이론보다 가치 있다.

## 5.2 Detailed Comparison Table

세 가지 memory-based 방법의 상세 비교는 다음과 같다.

**Distribution Modeling**

PaDiM은 각 패치 위치에서 다변량 가우시안을 가정한다. 평균과 공분산으로 분포가 완전히 특정되며, Mahalanobis 거리로 이상도를 측정한다. 단일 모드, 대칭적 분포를 가정하므로 제한적이지만, 계산이 효율적이고 안정적이다. DFKDE는 KDE로 비모수적 모델링을 한다. 어떤 분포 형태도 가능하며, Gaussian kernel의 합으로 밀도를 추정한다. 유연하지만 계산 비용이 크고 고차원에서 불안정하다. PatchCore는 명시적 분포 모델링을 하지 않는다. Coreset의 k-NN 거리로 직접 이상도를 측정하며, 암묵적으로 거리 기반 밀도를 사용한다. 분포 가정이 없어 robust하다.

**Memory Requirements**

PaDiM은 $O(h \times w \times d'^2)$로 각 위치의 공분산 행렬을 저장한다. 전형적으로 $h=w=56$, $d'=100$일 때 약 1.2GB다. 차원을 축소하지 않으면 수십 GB로 폭발한다. DFKDE는 $O(N \times h \times w \times d')$로 모든 학습 샘플을 유지한다. $N=200$일 때 약 600MB-1GB다. PaDiM보다 크지만 공분산 대신 원본 샘플을 저장한다. PatchCore는 $O(|\mathcal{C}| \times d)$로 선택된 coreset만 저장한다. 1% coreset일 때 약 1-2MB로 다른 방법 대비 100-1000배 작다. 이는 게임 체인저다.

**Inference Speed**

PaDiM은 이미지당 약 50-100ms(GPU)다. 각 패치에서 Mahalanobis 거리를 계산하며, 사전에 $\boldsymbol{\Sigma}^{-1}$을 계산해두면 행렬-벡터 곱셈만 필요하다. DFKDE는 200-500ms로 가장 느리다. 각 패치마다 $N$개 학습 샘플과의 거리를 계산해야 하므로 $O(N)$ 복잡도를 피할 수 없다. PatchCore는 30-50ms로 가장 빠르다. Coreset이 작아 k-NN 탐색이 효율적이고, FAISS 최적화가 가능하다.

**Performance (MVTec AD)**

PaDiM은 이미지 AUROC 97.5%, 픽셀 AUROC 97.5%다. 발표 당시 SOTA였지만 이후 방법들에 추월되었다. Texture 카테고리에서 강력하고 object에서 약간 약하다. DFKDE는 96-97%로 PaDiM보다 약간 낮다. 이론적 유연성이 실제 성능으로 이어지지 못했다. Bandwidth 선택에 민감하고 일관성이 부족하다. PatchCore는 99.1%로 압도적이다. 15개 중 10개 카테고리에서 99% 이상을 달성했다. 모든 면에서 최고 성능이다.

**Hyperparameters**

PaDiM은 차원 축소 목표 차원 $d'$와 regularization $\epsilon$이 있다. 기본값($d'=100$, $\epsilon=0.01$)이 robust하여 튜닝이 거의 불필요하다. DFKDE는 bandwidth $h$와 차원 $d'$가 있다. Bandwidth가 성능에 매우 민감하고 데이터마다 다르다. Validation 기반 grid search가 필수적이다. PatchCore는 coreset 비율(기본 1%)과 $k$(기본 9)가 있다. 기본값이 거의 모든 경우에 최적이어서 튜닝 부담이 적다.

**Spatial Context**

PaDiM은 위치별 독립 모델링으로 공간 맥락을 명시적으로 사용하지 않는다. 각 패치가 고립되어 처리된다. 인접 패치 간 상관관계가 무시된다. DFKDE도 유사하게 위치별 KDE로 공간 맥락이 제한적이다. PatchCore는 locally aware features로 $3 \times 3$ 이웃을 집계한다. 이는 수용 영역을 확장하고 spatial smoothness를 반영한다. 간단하지만 효과적인 맥락 통합이다.

**Training Time**

세 방법 모두 역전파가 없어 학습이 빠르다. PaDiM은 2-3분으로 공분산 계산이 주된 비용이다. DFKDE는 2-4분으로 유사하며, bandwidth selection에 추가 시간이 소요된다. PatchCore는 3-5분으로 약간 더 걸린다. Coreset selection에 시간이 들지만 여전히 매우 빠르다. End-to-end 방법(수 시간)과 비교하면 모두 압도적으로 빠르다.

**Deployment Suitability**

PaDiM은 서버 환경에서 단일 또는 소수 카테고리에 적합하다. 메모리가 충분하면 문제없지만, 다중 카테고리나 엣지는 어렵다. DFKDE는 배포에 부적합하다. 높은 메모리와 느린 추론으로 실용성이 낮다. 주로 학술 연구용이다. PatchCore는 모든 환경에 적합하다. 낮은 메모리로 엣지 배포가 가능하고, 빠른 추론으로 실시간에 가깝다. 다중 카테고리도 문제없다. 현재 산업 표준이다.

**Theoretical Foundation**

PaDiM은 가우시안 분포와 Mahalanobis 거리의 확립된 이론에 기반한다. 통계적으로 well-founded되고 해석이 명확하다. DFKDE는 KDE의 비모수 이론에 기반한다. 이론적으로 가장 일반적이고 유연하다. Asymptotic 수렴 보장이 있다. PatchCore는 k-center problem의 근사 알고리즘을 사용한다. 2-approximation 보장이 있으며, 실용적으로 검증되었다.

**Limitations**

PaDiM의 주된 한계는 높은 메모리와 가우시안 가정이다. Multimodal 분포를 제대로 포착하지 못할 수 있다. DFKDE는 curse of dimensionality와 계산 비용이다. 고차원에서 KDE가 불안정하고 느리다. PatchCore는 거의 한계가 없다. 강제로 찾자면 coreset selection에 약간의 시간이 들고, 매우 복잡한 분포에서는 더 큰 coreset이 필요할 수 있다는 정도다.

## 5.3 Memory Usage Analysis

Memory-based 방법들의 메모리 사용량을 구체적 수치로 분석하면 실용성의 차이가 명확해진다. 전형적인 설정을 가정한다. 입력 이미지 $256 \times 256$, feature map 해상도 $h=w=56$, 학습 샘플 수 $N=200$, 원본 특징 차원 $d=1536$(Wide ResNet-50), 축소 차원 $d'=100$.

PaDiM의 메모리 분해는 다음과 같다. 각 패치 위치 $(i,j)$에서 평균 벡터 $\boldsymbol{\mu}_{ij} \in \mathbb{R}^{d'}$와 공분산 행렬 $\boldsymbol{\Sigma}_{ij} \in \mathbb{R}^{d' \times d'}$를 저장한다. 평균은 $h \times w \times d' \times 4$ bytes = $56 \times 56 \times 100 \times 4$ = 1.25MB다. 공분산은 $h \times w \times d' \times d' \times 4$ = $56 \times 56 \times 100 \times 100 \times 4$ = 125MB다. 역행렬 $\boldsymbol{\Sigma}^{-1}$도 저장하면 추가 125MB다. 총 약 250MB가 필요하다.

차원을 축소하지 않으면($d=1536$) 공분산은 약 33GB로 폭발한다. 이는 단일 GPU 메모리를 초과하고 실용적이지 않다. 차원 축소가 필수적이지만 정보 손실이 발생한다. 다중 카테고리 배포 시 카테고리당 250MB씩 필요하므로, 10개 카테고리면 2.5GB, 100개면 25GB다. 서버급 메모리가 필요하다.

DFKDE의 메모리는 모든 학습 샘플의 특징을 저장한다. 각 학습 이미지에서 $h \times w$개 패치를 추출하고, $N$개 이미지에서 총 $N \times h \times w$개 특징 벡터가 있다. 차원 축소 후 각 벡터는 $d' \times 4$ bytes다. 총 메모리는 $N \times h \times w \times d' \times 4$ = $200 \times 56 \times 56 \times 100 \times 4$ = 250MB다. PaDiM과 비슷한 수준이다.

그러나 DFKDE는 위치별로 샘플을 분리하여 저장할 수도 있다. 각 위치 $(i,j)$에서 $N$개 샘플의 특징을 저장하면 구조는 다르지만 총량은 동일하다. 추가로 bandwidth 등 메타데이터가 있지만 무시할 수 있다. 차원을 축소하지 않으면 약 3.8GB가 필요하다. 다중 카테고리도 선형적으로 증가한다.

PatchCore의 메모리는 coreset만 저장한다. 1% coreset은 원본 패치의 1%다. $N \times h \times w$개 중 $0.01 \times N \times h \times w$개를 선택한다. $|\mathcal{C}| = 0.01 \times 200 \times 56 \times 56 \approx 6272$개다. 각 특징은 원본 차원 $d=1536$을 유지한다(차원 축소 없음). 총 메모리는 $|\mathcal{C}| \times d \times 4$ = $6272 \times 1536 \times 4$ = 38.6MB다.

실제 구현에서는 FAISS 인덱스 등 추가 구조가 필요할 수 있지만, 여전히 100MB 이하다. 10% coreset을 사용해도 약 400MB로 PaDiM/DFKDE보다 작다. 다중 카테고리에서도 카테고리당 40MB씩이므로 100개 카테고리면 4GB로 관리 가능하다. 엣지 디바이스(Raspberry Pi 4GB, Jetson Nano 4GB)에서도 수십 개 카테고리를 배포할 수 있다.

메모리 효율성의 실질적 영향은 크다. 첫째, 배치 크기다. GPU 메모리가 제한적일 때 모델 메모리가 작으면 더 큰 배치를 사용할 수 있다. PatchCore는 배치 크기 32-64도 가능하지만, PaDiM/DFKDE는 8-16으로 제한될 수 있다. 둘째, 동시 배포다. 하나의 서버에서 여러 카테고리나 제품 라인을 동시에 처리할 때 메모리가 병목이다. PatchCore는 수백 개 모델을 동시에 로드할 수 있다.

셋째, 모델 전송이다. OTA 업데이트나 네트워크를 통한 모델 배포 시 작은 모델은 빠르게 전송된다. 40MB는 수 초, 250MB는 수십 초, 수 GB는 수 분이 걸린다. 넷째, 캐시 효율성이다. 작은 모델은 CPU 캐시나 GPU shared memory에 올라갈 수 있어 메모리 접근 속도가 빠르다. 이는 추론 속도에도 영향을 미친다.

메모리-성능 trade-off를 분석하면 PatchCore의 우수성이 더욱 두드러진다. PaDiM은 250MB로 97.5% AUROC를 달성한다. DFKDE는 250MB로 96-97%를 달성한다. PatchCore는 40MB(1% coreset)로 99.1%를 달성한다. 6배 적은 메모리로 1.5-3%포인트 높은 성능이다. 이는 파레토 최적(Pareto optimal)이다.

Coreset 비율을 조절하면 메모리-성능 곡선을 그릴 수 있다. 0.5% coreset(20MB, 98.5%), 1% coreset(40MB, 99.1%), 5% coreset(200MB, 99.2%), 10% coreset(400MB, 99.3%)다. 1-5% 범위가 최적 균형점이다. 대부분의 경우 1%로 충분하다.

## 5.4 Computational Complexity

계산 복잡도는 추론 속도와 확장성을 결정한다. 학습과 추론을 분리하여 분석한다.

**Training Complexity**

PaDiM의 학습은 특징 추출과 통계량 계산으로 구성된다. 특징 추출은 $N$개 이미지를 CNN에 통과시켜 $O(N \times H \times W \times C_{\text{CNN}})$다. 여기서 $C_{\text{CNN}}$은 CNN의 계산 복잡도다. 이는 백본에 따라 다르지만 일반적으로 수십 GFLOPS다. 통계량 계산은 각 위치 $(i,j)$에서 $N$개 샘플의 평균과 공분산을 계산한다. Naive하게 $O(h \times w \times N \times d'^2)$이지만, online 알고리즘으로 $O(h \times w \times N \times d')$로 줄인다. 전체는 $O(N \times C_{\text{CNN}} + h \times w \times N \times d')$다.

DFKDE도 유사하게 특징 추출 $O(N \times C_{\text{CNN}})$와 특징 저장 $O(N \times h \times w \times d')$다. Bandwidth selection에 grid search를 사용하면 각 후보 $h$에 대해 validation set 평가가 필요하다. 이는 $O(K \times N_{\text{val}} \times N \times h \times w \times d')$로 $K$는 시도 횟수, $N_{\text{val}}$은 validation 샘플 수다. 이는 추가 비용이지만 병렬화 가능하다.

PatchCore는 특징 추출 후 coreset selection이 추가된다. Greedy k-center는 $|\mathcal{C}|$번의 iteration을 수행하고, 각 iteration에서 $|\mathcal{M}|$개 점의 거리를 업데이트한다. 복잡도는 $O(|\mathcal{C}| \times |\mathcal{M}| \times d)$다. $|\mathcal{M}| = N \times h \times w \approx 600,000$이고 $|\mathcal{C}| \approx 6,000$이면 약 36억 번의 거리 계산이다. GPU에서 벡터화하면 수 초 내지 수십 초다. 전체 학습은 $O(N \times C_{\text{CNN}} + |\mathcal{C}| \times |\mathcal{M}| \times d)$다.

실제 학습 시간은 PaDiM 2-3분, DFKDE 2-4분(bandwidth search 제외), PatchCore 3-5분으로 모두 유사하다. Coreset selection이 추가 시간을 요구하지만 여전히 매우 빠르다. End-to-end 학습(수 시간)과 비교하면 모두 압도적 우위다.

**Inference Complexity**

PaDiM의 추론은 특징 추출 $O(C_{\text{CNN}})$과 Mahalanobis 거리 계산 $O(h \times w \times d'^2)$다. 사전에 $\boldsymbol{\Sigma}^{-1}$을 계산해두면 각 위치에서 $(\mathbf{f} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{f} - \boldsymbol{\mu})$는 $O(d'^2)$ 연산이다. 전체 이미지는 $O(h \times w \times d'^2)$다. $h=w=56$, $d'=100$이면 약 31M 연산이다. GPU에서 병렬화하면 밀리초 단위다.

DFKDE는 특징 추출 후 각 패치에서 KDE를 평가한다. 위치 $(i,j)$에서 $N$개 학습 샘플과의 거리를 계산하고 Gaussian kernel을 평가한다. $O(N \times d')$다. 전체 이미지는 $O(h \times w \times N \times d')$다. $N=200$이면 약 627M 연산으로 PaDiM보다 20배 많다. Cutoff를 사용하거나 KD-tree를 활용해도 근본적인 $O(N)$ 의존성은 피할 수 없다.

PatchCore는 특징 추출 후 각 패치에서 k-NN을 찾는다. Naive linear search는 $O(|\mathcal{C}| \times d)$지만, FAISS 같은 최적화 라이브러리는 $O(\log |\mathcal{C}|)$ 쿼리를 제공한다. 전체 이미지는 $O(h \times w \times \log |\mathcal{C}| \times d)$다. $|\mathcal{C}| = 6000$이면 $\log |\mathcal{C}| \approx 13$으로 매우 작다. 약 111M 연산으로 PaDiM과 DFKDE 사이다. 그러나 FAISS의 고도 최적화로 실제 속도는 가장 빠르다.

실제 추론 시간은 PaDiM 50-100ms, DFKDE 200-500ms, PatchCore 30-50ms다. PatchCore가 가장 빠른 이유는 작은 coreset과 FAISS 최적화 덕분이다. Throughput을 보면 PatchCore는 초당 20-30 이미지, PaDiM은 10-20, DFKDE는 2-5다. 실시간(초당 30 프레임)에 가까운 것은 PatchCore뿐이다.

**Scalability**

데이터 증가에 대한 확장성을 분석한다. 학습 샘플 $N$이 증가하면 PaDiM은 $O(N)$로 선형 증가한다. 공분산 추정에 모든 샘플이 필요하지만 online 업데이트가 가능하다. 추론은 $N$에 무관하여 확장성이 좋다. DFKDE는 학습이 $O(N)$이지만 추론도 $O(N)$이다. 모든 샘플을 유지하고 거리를 계산해야 하므로 $N$이 크면 prohibitive하다. PatchCore는 학습이 $O(|\mathcal{C}| \times N)$로 증가하지만, coreset 크기를 고정하면 추론은 $N$에 무관하다. Coreset이 $N$의 일정 비율이면 학습도 $O(N)$이다.

카테고리 증가에 대해서는 세 방법 모두 카테고리마다 독립적인 모델을 학습한다. 학습 시간은 선형 증가하지만 병렬화 가능하다. 추론은 카테고리 식별 후 해당 모델만 사용하므로 카테고리 수에 거의 무관하다. 다만 메모리는 선형 증가한다. PaDiM/DFKDE는 많은 카테고리에서 메모리 병목이지만, PatchCore는 수백 개도 문제없다.

해상도 증가는 $h \times w$에 비례한다. PaDiM과 DFKDE는 $O(h \times w)$로 선형 증가한다. PatchCore도 유사하지만 coreset selection이 $O(h \times w)$에 비례하여 약간 더 영향을 받는다. 실제로는 고해상도 이미지를 다운샘플링하여 사용하므로 큰 문제가 아니다.

## 5.5 Scalability Considerations

Memory-based 방법의 확장성은 실제 배포에서 중요한 고려사항이다. 다양한 차원에서 확장성을 분석한다.

**Multi-category Deployment**

단일 카테고리에서 여러 카테고리로 확장할 때의 도전과제다. PaDiM은 카테고리당 약 250MB가 필요하므로 10개면 2.5GB, 100개면 25GB다. 단일 GPU(10-20GB VRAM)에서는 수십 개가 한계다. 서버급 시스템 메모리(128-256GB)를 사용하면 수백 개까지 가능하지만 비용이 크다. 모델 로딩과 언로딩을 동적으로 관리해야 할 수 있다.

DFKDE는 유사한 메모리 문제가 있고 추론도 느려 다중 카테고리에 부적합하다. PatchCore는 카테고리당 40MB로 100개면 4GB다. 단일 GPU에서도 충분히 수용 가능하다. 모든 카테고리 모델을 동시에 메모리에 올려두고 즉시 전환할 수 있다. 이는 유연한 다품종 생산 라인에서 중요하다.

백본 가중치는 모든 카테고리가 공유할 수 있어 추가 메모리 없이 재사용된다. 카테고리별로 저장하는 것은 통계량(PaDiM), 샘플(DFKDE), coreset(PatchCore)뿐이다. 특징 추출 파이프라인은 공통이므로 배치 처리로 여러 카테고리를 동시에 처리할 수 있다.

**High-resolution Images**

산업 검사에서는 $512 \times 512$, $1024 \times 1024$ 또는 더 큰 이미지가 사용될 수 있다. 고해상도는 미세한 결함 탐지에 유리하지만 계산 비용을 증가시킨다. 일반적인 전략은 이미지를 $256 \times 256$으로 다운샘플링하거나, 타일링하여 처리하는 것이다.

타일링 접근은 큰 이미지를 여러 $256 \times 256$ 타일로 나누고 각각 독립적으로 처리한다. PaDiM과 DFKDE는 각 타일을 별도로 처리하면 되므로 자연스럽게 확장된다. PatchCore는 전체 이미지의 통합된 coreset을 구성하거나, 타일별 coreset을 만들 수 있다. 전자는 전역적 맥락을 유지하고 후자는 메모리를 절약한다.

다운샘플링은 더 간단하지만 미세한 결함을 놓칠 위험이 있다. Multi-scale 접근으로 여러 해상도에서 처리하고 결과를 융합할 수 있다. 낮은 해상도에서 후보 영역을 찾고 높은 해상도에서 확인하는 two-stage 전략도 효과적이다.

**Real-time Requirements**

일부 응용에서는 실시간 처리가 필요하다. 예를 들어 고속 생산 라인에서 초당 30-60개 제품을 검사해야 할 수 있다. 이는 이미지당 16-33ms의 처리 시간을 요구한다. PaDiM(50-100ms)과 DFKDE(200-500ms)는 실시간을 충족하지 못한다. 배치 처리로 throughput을 높일 수 있지만 지연이 증가한다.

PatchCore(30-50ms)는 실시간에 근접하지만 충분하지 않을 수 있다. 최적화가 필요하다. 첫째, 백본 경량화다. Wide ResNet-50 대신 MobileNet이나 EfficientNet-Lite를 사용하면 특징 추출을 2-3배 가속할 수 있다. 성능이 약간 저하되지만 여전히 competitive하다. 둘째, coreset 축소다. 0.5% coreset으로 줄이면 k-NN 탐색이 빨라진다. 성능 저하가 minimal하다.

셋째, 근사 k-NN을 사용한다. FAISS의 IVF 인덱스나 HNSW 알고리즘으로 정확도를 약간 희생하고 속도를 크게 향상시킨다. 넷째, FP16이나 INT8 quantization으로 연산을 가속한다. 특징 추출과 거리 계산 모두 저정밀도로 수행 가능하다. 이러한 최적화로 10-20ms까지 줄일 수 있어 실시간 요구를 만족한다.

**Edge Deployment**

엣지 디바이스(Raspberry Pi, Jetson Nano, 산업용 임베디드 시스템)는 제한된 리소스를 가진다. 메모리는 2-8GB, 연산 능력은 서버 GPU의 1/10 이하다. PaDiM과 DFKDE는 메모리 제약으로 어렵다. 단일 카테고리도 수백 MB를 요구하고, 다중 카테고리는 불가능하다. CPU 추론도 수 초가 걸려 실용적이지 않다.

PatchCore는 엣지 배포가 가능하다. 0.5-1% coreset으로 카테고리당 20-40MB면 충분하다. Jetson Nano(4GB)에서 수십 개 카테고리를 배포할 수 있다. TensorRT로 백본을 최적화하고 FAISS CPU 버전으로 k-NN을 수행하면 이미지당 100-200ms로 처리 가능하다. 실시간은 아니지만 많은 응용에서 충분하다.

경량 백본(MobileNetV2, EfficientNet-B0)과 결합하면 성능이 더 향상된다. 특징 추출이 50-100ms로 줄어들고 총 처리 시간이 70-150ms가 된다. 다중 스레딩으로 파이프라이닝하면 throughput을 높일 수 있다. 한 스레드가 특징 추출을 하는 동안 다른 스레드가 k-NN을 수행한다.

**Continuous Learning**

제조 환경은 동적이다. 제품 디자인 변경, 공정 개선, 원자재 변화 등으로 정상 패턴이 drift한다. 모델을 주기적으로 업데이트해야 한다. PaDiM은 online 업데이트가 가능하다. 새로운 정상 샘플이 들어오면 평균과 공분산을 incremental하게 업데이트한다. Welford 알고리즘으로 효율적으로 수행된다.

DFKDE는 새 샘플을 단순히 추가하면 되지만 메모리가 계속 증가한다. 주기적으로 오래된 샘플을 제거하는 sliding window 전략이 필요하다. PatchCore는 새 샘플로 coreset을 재구성해야 한다. Incremental coreset selection 알고리즘이 연구되었지만 완전히 만족스럽지 않다. 실무에서는 주기적으로 전체 재학습을 수행한다. 학습이 빠르므로(수 분) 매일 또는 매주 재학습해도 부담이 적다.

Drift detection을 통해 재학습 시점을 자동 결정할 수 있다. 정상 샘플의 이상 점수 분포를 모니터링하고, 평균이나 분산이 유의미하게 변하면 재학습을 트리거한다. 이는 필요할 때만 업데이트하여 계산 비용을 절약한다.

**Cloud vs Edge Trade-offs**

클라우드 기반 배포는 강력한 리소스를 제공하지만 네트워크 지연과 프라이버시 문제가 있다. 엣지 배포는 지연이 낮고 데이터가 로컬에 머물지만 리소스가 제한적이다. Memory-based 방법의 선택은 배포 전략에 영향을 받는다.

클라우드에서는 PaDiM, DFKDE, PatchCore 모두 실행 가능하다. 메모리와 연산이 충분하므로 최고 성능을 위해 PatchCore를 선택한다. 수천 개 카테고리도 관리 가능하고, 고해상도 이미지도 처리할 수 있다. API를 통해 여러 생산 라인에서 요청을 받아 배치 처리로 처리한다.

엣지에서는 PatchCore가 거의 유일한 선택이다. 경량 백본과 작은 coreset으로 제한된 리소스에 적합하다. 네트워크 연결 없이 독립 동작하므로 지연이 최소화되고 프라이버시가 보장된다. 중요한 응용에서 이는 필수적이다.

하이브리드 접근도 가능하다. 엣지에서 PatchCore로 1차 필터링을 하고, 의심 샘플만 클라우드로 보내 더 정교한 분석을 한다. 이는 네트워크 대역폭을 절약하고 대부분의 샘플을 빠르게 처리하면서도 어려운 경우에는 강력한 분석을 제공한다.

# 6. Practical Application Guide

## 6.1 Model Selection Criteria

Memory-based 방법 중 어떤 것을 선택할지는 구체적인 요구사항과 제약에 따라 달라진다. 의사결정 프레임워크를 제시한다.

**Primary Criterion: Performance Requirement**

최고 정확도가 필수적인가? 불량품을 절대 놓치면 안 되는 critical한 응용(의료 기기, 항공우주, 안전 부품)이라면 PatchCore가 유일한 선택이다. 99%+ AUROC가 필요하고, false negative 비용이 극도로 높다면 최고 성능 모델이 필수다. 0.5-1% 성능 차이도 수백만 달러의 리콜 비용이나 안전 사고로 이어질 수 있다.

95-98% 정도로 충분한가? 일반 소비재나 미적 결함 검사에서는 완벽한 정확도가 필수적이지 않을 수 있다. 일부 false positive를 인간 검사자가 최종 확인하는 시스템이라면 PaDiM도 acceptable하다. 그러나 PatchCore의 우수성과 효율성을 고려하면 굳이 PaDiM을 선택할 이유가 적다. DFKDE는 어떤 경우에도 권장되지 않는다.

**Secondary Criterion: Resource Constraints**

메모리 제약이 엄격한가? 엣지 디바이스(4GB 이하)나 다중 카테고리 배포(수십-수백 개)에서는 PatchCore가 필수다. 카테고리당 수십 MB만 사용하므로 확장성이 우수하다. 서버 환경에서 단일 또는 소수 카테고리라면 PaDiM도 가능하지만, PatchCore가 모든 면에서 우위므로 굳이 PaDiM을 선택할 이유가 없다.

실시간 처리가 필요한가? 초당 30+ 이미지를 처리해야 한다면 PatchCore + 최적화(경량 백본, 작은 coreset, FAISS)가 필요하다. PaDiM은 배치 처리로 throughput을 높일 수 있지만 단일 이미지 지연이 길다. DFKDE는 실시간 불가능하다.

**Tertiary Criterion: Operational Factors**

학습 데이터가 제한적인가? 모든 memory-based 방법은 few-shot에 robust하다. 수십에서 수백 개 정상 샘플로 충분하다. PatchCore가 가장 효율적으로 데이터를 활용한다. Coreset selection이 중요한 샘플을 자동으로 선택하여 적은 데이터에서도 좋은 성능을 보장한다.

빈번한 제품 변경이 있는가? 신제품 출시나 디자인 변경이 잦다면 빠른 재학습이 중요하다. 세 방법 모두 수 분 내에 학습이 완료되므로 문제없다. PatchCore는 coreset selection에 약간 더 시간이 들지만 여전히 매우 빠르다. Daily 또는 weekly 재학습도 현실적이다.

설명 가능성이 중요한가? 왜 이 샘플이 이상으로 판정되었는지 설명해야 한다면 모든 방법이 이상 맵을 제공한다. PatchCore는 가장 가까운 정상 샘플을 보여줄 수 있어 비교 분석이 가능하다. PaDiM의 Mahalanobis 거리는 통계적 의미가 명확하다. 실무에서는 이상 맵으로 결함 위치를 보여주는 것이 가장 중요하다.

**Decision Matrix**

다음 조건이 모두 해당되면 PatchCore를 선택한다. 최고 성능 필요, 메모리 효율 중요, 다중 카테고리 가능성, 엣지 배포 고려, 미래 확장성 필요. 이는 거의 모든 실무 상황이다.

다음 조건에서만 PaDiM을 고려한다. 서버 환경, 단일 카테고리, 95-98% 성능으로 충분, 기존 PaDiM 인프라가 있음. 그러나 이 경우에도 PatchCore로 업그레이드를 강력히 권장한다.

DFKDE는 어떤 실무 상황에서도 권장되지 않는다. 학술 연구나 비모수 방법의 이론적 탐구에서만 의미가 있다.

## 6.2 Hyperparameter Tuning

Memory-based 방법의 장점 중 하나는 하이퍼파라미터가 적고 기본값이 robust하다는 것이다. 그러나 최적 성능을 위해 조정할 수 있는 부분이 있다.

**PatchCore Hyperparameters**

Coreset 비율이 가장 중요하다. 기본값 1%는 대부분의 경우 최적이다. 메모리가 충분하고 최대 성능을 원하면 5-10%를 시도한다. 일반적으로 1%에서 5%로 늘려도 성능 향상은 0.5%포인트 미만이다. 메모리가 극도로 제한적이면 0.5%로 줄인다. 성능 저하는 1%포인트 정도다.

k-NN의 k 값은 기본 9가 robust하다. 5-15 범위에서 성능이 안정적이므로 튜닝이 거의 불필요하다. 작은 coreset(0.5%)을 사용한다면 k를 5로 줄일 수 있다. 큰 coreset(10%)이라면 k=15도 고려한다. 그러나 실무에서는 k=9를 고정하고 coreset 비율만 조정하는 것이 충분하다.

백본 선택은 성능과 속도의 trade-off다. ResNet18은 빠르고 가벼우며 대부분의 경우 충분하다. Wide ResNet-50은 1-2%포인트 성능 향상을 제공하지만 3배 느리다. EfficientNet-B4는 중간 지점이다. 실시간 요구가 있으면 MobileNetV2나 EfficientNet-Lite를 고려한다. 성능은 약간 떨어지지만 여전히 competitive하다.

특징 층 선택은 layer2 + layer3가 기본이다. Texture 중심의 결함이라면 layer1을 추가한다. Structural 결함이라면 layer3만 사용하거나 layer3 + layer4를 시도한다. 실험적으로 여러 조합을 평가하여 최적을 찾는다. 일반적으로 기본 조합이 좋다.

**PaDiM Hyperparameters**

차원 축소 목표 차원 $d'$이 주된 파라미터다. 기본 100은 메모리와 성능의 균형점이다. 메모리가 충분하면 200-300으로 늘려 성능을 약간 향상시킬 수 있다. 극도로 제한적이면 50으로 줄이지만 성능 저하가 있다. $d' > 200$은 거의 이득이 없고 계산만 느려진다.

Regularization $\epsilon$은 기본 0.01이 안정적이다. 공분산 행렬이 singular에 가까운 경고가 나타나면 0.1로 늘린다. 너무 크면($\epsilon > 1$) 정보가 손실되어 성능이 저하된다. 대부분의 경우 기본값을 유지한다.

백본과 특징 층은 PatchCore와 동일한 고려사항이다.

**Tuning Strategy**

실무적인 튜닝 전략은 다음과 같다. 먼저 기본 설정으로 baseline을 구축한다. PatchCore + ResNet18 + layer2+3 + 1% coreset + k=9다. Validation set에서 성능을 평가한다. 목표 성능(예: 98% AUROC)에 도달하면 튜닝을 종료한다.

도달하지 못하면 백본을 업그레이드한다. ResNet18 → Wide ResNet-50 또는 EfficientNet-B4로 바꾼다. 이는 1-2%포인트 향상을 가져온다. 여전히 부족하면 coreset을 5%로 늘린다. 추가로 0.5%포인트 정도 향상된다.

특정 카테고리나 결함 유형에서 문제가 있으면 특징 층을 조정한다. Texture 결함이 놓치면 layer1을 추가한다. Structural 이상이 문제면 layer3에 집중한다. 이는 targeted optimization이다.

시간과 리소스가 허락하면 grid search를 수행한다. Coreset [0.5%, 1%, 2%, 5%], k [5, 9, 15], 백본 [ResNet18, ResNet50, WideResNet50]의 조합을 모두 시도한다. 그러나 대부분의 경우 기본 설정이 이미 최적에 가깝다.

**Common Mistakes**

과도한 튜닝은 overfitting을 초래한다. Validation set에서 최적화하되, test set으로 최종 평가한다. Validation에서 0.1%포인트 차이는 noise일 수 있다. 통계적 유의성을 확인한다. 여러 random seed로 실험을 반복하여 robust성을 검증한다.

메모리와 속도를 무시하고 성능만 추구하는 것도 문제다. 0.5%포인트 성능 향상을 위해 메모리를 10배 늘리는 것은 비합리적이다. 실제 배포 환경의 제약을 고려한 최적화가 필요하다. Pareto frontier를 그려 성능-효율성 trade-off를 시각화한다.

단일 카테고리에만 최적화하고 일반화를 고려하지 않는 것도 위험하다. 한 카테고리에 과도하게 맞춘 설정은 다른 카테고리에서 실패할 수 있다. 여러 대표적인 카테고리에서 검증하여 robust한 설정을 찾는다.

## 6.3 Training Pipeline

실무에서 memory-based 모델을 학습하는 전체 파이프라인을 단계별로 설명한다.

**Step 1: Data Preparation**

정상 학습 데이터를 수집한다. MVTec AD는 카테고리당 약 200-400장을 제공하지만, 실무에서는 50-300장이 일반적이다. 데이터 품질이 양보다 중요하다. 모든 가능한 정상 변동을 커버해야 한다. 조명, 포즈, 색상, 배경 등의 변화를 포함한다. 명백한 불량품이 섞이지 않도록 육안 검사를 수행한다.

이미지를 정규화하고 리사이즈한다. 표준 크기는 $256 \times 256$이다. 고해상도 이미지는 다운샘플링하거나 타일링한다. 전처리로 histogram equalization이나 denoising을 고려할 수 있지만, 대부분의 경우 불필요하다. CNN이 robust하게 특징을 추출한다.

Train/validation split을 수행한다. 일반적으로 80/20 또는 70/30이다. Validation set은 하이퍼파라미터 선택과 임계값 설정에 사용된다. Stratified split으로 다양한 변동이 양쪽에 포함되도록 한다.

**Step 2: Feature Extraction**

사전 학습된 백본을 로드한다. Torchvision이나 Timm 라이브러리에서 제공하는 ImageNet pretrained weights를 사용한다. 백본을 evaluation mode로 설정하고 gradient를 비활성화한다. 추론만 수행하므로 학습이 필요 없다.

모든 학습 이미지를 백본에 통과시켜 중간 층 특징을 추출한다. Layer2와 layer3의 activation을 hook으로 캡처한다. 배치 처리로 효율성을 높인다. GPU 메모리에 맞춰 배치 크기를 조정한다(일반적으로 16-32).

특징을 동일한 공간 해상도로 정렬한다. Average pooling으로 layer2를 layer3 크기로 다운샘플링한다. 채널 차원으로 연결하여 통합 특징 맵을 생성한다. 결과는 $N \times h \times w \times d$ 크기의 텐서다.

PatchCore의 경우 locally aware features를 생성한다. $3 \times 3$ average pooling with stride 1을 적용한다. PaDiM은 차원 축소를 수행한다. Random projection matrix를 생성하고 특징을 $d'$ 차원으로 투영한다.

**Step 3: Model Training**

PatchCore는 coreset selection을 수행한다. 모든 패치 특징을 하나의 집합 $\mathcal{M}$로 모은다. Greedy k-center 알고리즘을 실행하여 목표 coreset 크기에 도달할 때까지 반복한다. 각 iteration에서 현재 coreset으로부터 가장 먼 점을 선택한다. 결과 coreset $\mathcal{C}$를 저장한다.

PaDiM은 각 패치 위치에서 통계량을 계산한다. 위치 $(i,j)$에서 $N$개 샘플의 평균 $\boldsymbol{\mu}_{ij}$와 공분산 $\boldsymbol{\Sigma}_{ij}$를 계산한다. Online 알고리즘으로 메모리 효율적으로 수행한다. Regularization을 적용하고 역행렬 $\boldsymbol{\Sigma}_{ij}^{-1}$을 계산하여 저장한다.

DFKDE는 모든 샘플을 위치별로 저장한다. Bandwidth를 선택하기 위해 validation set에서 grid search를 수행한다. 여러 $h$ 값을 시도하고 AUROC가 최대인 것을 선택한다.

**Step 4: Validation and Threshold Selection**

Validation set에서 모델을 평가한다. 각 validation 이미지에 대해 이상 점수를 계산한다. 정상과 이상 샘플의 점수 분포를 분석한다. ROC 곡선을 그리고 AUROC를 계산한다.

임계값을 선택한다. 목표 재현율(예: 95% 또는 99%)에 대응하는 임계값을 찾는다. 또는 F1 score를 최대화하는 임계값을 선택한다. 산업 응용에서는 높은 재현율을 우선하여 false negative를 최소화한다.

픽셀 레벨 임계값도 별도로 설정한다. 이상 맵에서 결함 영역을 세그멘테이션하기 위한 임계값이다. 이미지 레벨보다 낮게 설정하여 더 민감하게 탐지한다.

**Step 5: Model Saving and Documentation**

최종 모델을 저장한다. PatchCore는 coreset 특징과 FAISS 인덱스를 저장한다. PaDiM은 평균, 공분산 역행렬, 차원 축소 행렬을 저장한다. 백본 가중치는 별도로 저장하거나 공유 리소스를 참조한다.

메타데이터를 기록한다. 백본 종류, 특징 층, 하이퍼파라미터, 학습 날짜, 데이터 버전, 성능 메트릭, 임계값 등을 JSON이나 YAML 파일에 저장한다. 이는 재현성과 버전 관리에 필수적이다.

학습 로그를 유지한다. 사용된 학습 이미지 목록, 검증 결과, 이상 점수 분포, ROC 곡선 등을 저장한다. 문제 발생 시 디버깅과 추적에 유용하다.

**Step 6: Test Evaluation**

독립적인 test set에서 최종 평가를 수행한다. Test set은 학습이나 validation에 전혀 사용되지 않은 데이터여야 한다. 실제 배포 환경을 대표하도록 구성한다. 다양한 결함 유형과 정상 변동을 포함한다.

이미지 레벨과 픽셀 레벨 메트릭을 계산한다. AUROC, AUPR(Area Under Precision-Recall), F1 score, precision, recall을 보고한다. 혼동 행렬(confusion matrix)을 분석하여 false positive와 false negative 패턴을 파악한다.

결함 유형별 성능을 분석한다. 어떤 결함이 잘 탐지되고 어떤 것이 어려운지 파악한다. 실패 사례를 수집하고 원인을 분석한다. 이는 모델 개선이나 데이터 보강의 방향을 제시한다.

**Pipeline Automation**

실무에서는 파이프라인을 자동화하여 반복적인 재학습을 용이하게 한다. Configuration 파일로 모든 설정을 관리한다. 명령줄 인터페이스나 API를 제공하여 스크립트로 학습을 트리거할 수 있게 한다.

CI/CD(Continuous Integration/Continuous Deployment) 파이프라인에 통합한다. 새로운 학습 데이터가 수집되면 자동으로 재학습을 수행한다. 성능이 기준을 만족하면 자동으로 배포한다. 성능 저하가 감지되면 경고를 발생시킨다.

MLOps 도구를 활용한다. MLflow, Weights & Biases, TensorBoard 등으로 실험을 추적하고 비교한다. 모델 레지스트리로 버전을 관리한다. A/B 테스팅으로 새 모델을 점진적으로 롤아웃한다.

## 6.4 Deployment Checklist

Memory-based 모델을 실제 생산 환경에 배포할 때 확인해야 할 체크리스트다.

**Pre-deployment Validation**

성능이 요구사항을 만족하는가? 목표 AUROC, 재현율, precision을 달성했는지 확인한다. 여러 test set에서 일관된 성능을 보이는지 검증한다. Edge case나 corner case에서도 robust한지 평가한다.

추론 속도가 충분한가? 목표 throughput(초당 이미지 수)을 달성하는지 측정한다. 배포 환경(GPU/CPU, 메모리)에서 실제로 테스트한다. 평균뿐만 아니라 최악의 경우(worst-case) 지연도 확인한다. 생산 라인 속도에 여유 있게 맞춰야 한다.

메모리 사용량이 제한 내에 있는가? 피크 메모리 사용량을 측정한다. 다중 프로세스나 스레드 환경에서 메모리 누수가 없는지 확인한다. 장시간 실행 시 메모리가 증가하지 않는지 모니터링한다.

임계값이 적절한가? Validation set에서 선택한 임계값이 실제 데이터에서도 유효한지 확인한다. False alarm rate이 허용 범위 내인지 검증한다. 오탐이 너무 많으면 운영자의 피로와 신뢰도 저하를 초래한다.

**Infrastructure Preparation**

하드웨어 리소스가 준비되었는가? 필요한 GPU/CPU, 메모리, 스토리지가 확보되었는지 확인한다. 네트워크 대역폭이 충분한지 평가한다(이미지 전송, 결과 전송). 전원과 냉각이 안정적인지 확인한다.

소프트웨어 스택이 설치되었는가? 운영체제, Python, PyTorch, CUDA, cuDNN 등 필요한 소프트웨어가 올바른 버전으로 설치되었는지 확인한다. 의존성 충돌이 없는지 검증한다. Docker나 conda로 환경을 격리하여 재현성을 보장한다.

모델과 가중치가 배포되었는가? Coreset, 백본 가중치, 메타데이터 파일이 올바른 경로에 있는지 확인한다. 파일 권한과 접근성을 검증한다. 백업이 준비되었는지 확인한다.

**Integration Testing**

입력 인터페이스가 작동하는가? 카메라나 이미지 소스로부터 이미지를 정상적으로 받는지 테스트한다. 이미지 포맷, 해상도, 색 공간이 예상대로인지 확인한다. 네트워크 끊김이나 타임아웃을 처리하는지 검증한다.

전처리가 올바른가? 이미지 정규화, 리사이즈, 색상 변환이 학습 시와 동일하게 적용되는지 확인한다. 작은 차이도 성능 저하를 초래할 수 있다. Reference 이미지로 전처리 결과를 비교한다.

추론이 정확한가? Known 정상 샘플과 이상 샘플로 추론을 테스트한다. 이상 점수와 이상 맵이 예상과 일치하는지 확인한다. Numerical precision 문제(FP32 vs FP16)가 없는지 검증한다.

출력 인터페이스가 작동하는가? 판정 결과(정상/이상), 이상 점수, 이상 맵이 올바르게 전달되는지 확인한다. 데이터베이스 로깅, 알람 시스템, UI 표시가 정상인지 테스트한다. 결과 형식이 downstream 시스템과 호환되는지 검증한다.

**Performance Monitoring**

실시간 메트릭을 수집하는가? 추론 시간, throughput, 메모리 사용량, GPU 활용률 등을 모니터링한다. Prometheus, Grafana 같은 도구로 대시보드를 구축한다. 임계값을 초과하면 경고를 발생시킨다.

판정 통계를 추적하는가? 정상/이상 판정 비율, 평균 이상 점수, 이상 점수 분포를 기록한다. 시간에 따른 변화를 모니터링하여 drift를 조기에 감지한다. 급격한 변화는 문제의 징후일 수 있다.

오류와 예외를 로깅하는가? 모든 오류를 중앙 로깅 시스템에 기록한다. 스택 트레이스, 입력 데이터, 시스템 상태를 함께 저장한다. 오류 빈도와 패턴을 분석하여 근본 원인을 파악한다.

**Quality Assurance**

인간 검증과 비교하는가? 무작위로 샘플링한 결과를 전문가가 검토한다. 모델 판정과 인간 판정의 일치도를 측정한다. 불일치 사례를 분석하여 모델의 약점을 파악한다.

False negative를 추적하는가? 모델이 놓친 불량품(false negative)을 별도로 수집한다. 이는 가장 critical한 오류이므로 철저히 분석한다. 패턴이 발견되면 모델을 개선하거나 학습 데이터를 보강한다.

False positive를 관리하는가? 오탐(false positive)도 비용을 발생시킨다. 불필요한 인간 검사나 제품 폐기를 초래한다. 오탐률이 허용 범위를 초과하면 임계값을 조정하거나 모델을 재학습한다.

**Maintenance Plan**

재학습 일정이 수립되었는가? 주기적 재학습 계획을 세운다(일별, 주별, 월별). Drift 모니터링 결과에 따라 동적으로 조정할 수도 있다. 재학습 시 서비스 중단을 최소화하는 전략을 마련한다.

데이터 수집이 지속되는가? 새로운 정상 샘플과 이상 샘플을 지속적으로 수집한다. 특히 모델이 실패한 사례를 우선적으로 수집한다. 데이터 품질을 유지하고 레이블을 정확히 한다.

모델 업데이트 프로세스가 있는가? 새 버전 모델을 어떻게 평가하고 배포할지 정의한다. Canary 배포나 blue-green 배포로 위험을 최소화한다. 롤백 계획을 준비하여 문제 발생 시 신속히 대응한다.

**Documentation**

운영 매뉴얼이 준비되었는가? 시스템 구조, 작동 원리, 일상적인 운영 절차를 문서화한다. 일반적인 문제와 해결 방법을 포함한다. 비기술 운영자도 이해할 수 있게 작성한다.

트러블슈팅 가이드가 있는가? 흔한 오류와 경고 메시지의 의미를 설명한다. 단계별 진단과 해결 방법을 제공한다. 연락처와 에스컬레이션 경로를 명시한다.

성능 baseline이 기록되었는가? 초기 배포 시 성능 메트릭을 문서화한다. 이후 성능 변화를 비교하는 기준점이 된다. 정기적으로 업데이트하여 최신 상태를 유지한다.

## 6.5 Common Pitfalls

Memory-based 모델 적용 시 흔히 발생하는 실수와 해결 방법이다.

**Data Quality Issues**

가장 흔한 실수는 학습 데이터에 불량품이 섞여 있는 것이다. 정상으로 레이블된 데이터에 실제로는 미세한 결함이 있으면 모델이 혼란스러워진다. 해당 결함 패턴을 정상으로 학습하여 이후 유사한 불량을 놓친다. 예방책은 학습 데이터를 엄격히 검증하는 것이다. 여러 검사자가 독립적으로 확인하고, 의심스러운 샘플은 제외한다. Anomaly detection을 학습 데이터 자체에 적용하여 outlier를 찾을 수도 있다.

정상 변동을 충분히 커버하지 못하는 것도 문제다. 학습 데이터가 제한된 조건(특정 조명, 각도, 배경)에서만 수집되면 다른 조건의 정상 샘플이 이상으로 오탐된다. 해결책은 의도적으로 다양한 변동을 수집하는 것이다. 조명을 변화시키고, 여러 각도에서 촬영하고, 계절이나 시간대를 다르게 한다. Data augmentation으로 인위적 변동을 추가할 수도 있지만, 실제 변동을 대체할 수는 없다.

불균형한 학습 데이터도 문제를 일으킨다. 특정 정상 mode가 과도하게 많고 다른 mode는 적으면 모델이 편향된다. 희소한 정상 패턴이 이상으로 분류될 수 있다. 해결책은 balanced sampling이다. 각 정상 mode에서 비슷한 수의 샘플을 수집하거나, 학습 시 가중치를 조정한다.

**Preprocessing Inconsistency**

학습과 배포 시 전처리가 다른 것은 치명적이다. 정규화 파라미터, 리사이즈 방법, 색상 공간이 조금만 달라도 성능이 크게 저하된다. 예를 들어 학습 시 ImageNet normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])을 사용했는데 배포 시 적용하지 않으면 특징이 완전히 달라진다.

예방책은 전처리를 코드로 명확히 정의하고 공유하는 것이다. 함수나 클래스로 캡슐화하여 학습과 배포에서 동일한 코드를 사용한다. Configuration 파일에 모든 파라미터를 기록하고 배포 시 검증한다. Reference 이미지로 전처리 결과를 비교하여 일관성을 확인한다.

색상 공간 혼동도 흔하다. OpenCV는 기본적으로 BGR을 사용하고 PIL은 RGB를 사용한다. 학습 시 RGB로 했는데 배포 시 BGR로 입력하면 색상이 뒤바뀐다. 명시적으로 색상 변환을 수행하고 주석으로 명확히 한다.

**Threshold Misconfiguration**

Validation set에서 선택한 임계값을 그대로 배포에 사용하는 것은 위험하다. 실제 데이터 분포가 validation과 다르면 false alarm이 폭발할 수 있다. 특히 정상 샘플의 이상 점수 분포가 shift되면 문제다. 해결책은 배포 초기에 임계값을 보수적으로 설정하는 것이다. 높은 재현율을 보장하되 일부 오탐은 감수한다. 실제 데이터로 통계를 수집한 후 점진적으로 조정한다.

단일 임계값만 사용하는 것도 제한적이다. 모든 카테고리나 결함 유형에 동일한 임계값을 적용하면 일부는 over-sensitive하고 일부는 under-sensitive하다. 카테고리별, 결함 유형별로 다른 임계값을 설정하는 것이 효과적이다. Adaptive thresholding으로 데이터 분포에 따라 동적으로 조정할 수도 있다.

이상 맵 임계값과 이미지 임계값을 혼동하는 것도 문제다. 이상 맵은 픽셀 레벨 세그멘테이션을 위한 것으로, 이미지 레벨보다 낮은 임계값이 필요하다. 두 임계값을 독립적으로 선택하고 명확히 구분한다.

**Resource Underestimation**

메모리 요구량을 과소평가하는 것은 흔한 실수다. 개발 환경에서는 단일 모델만 테스트하지만, 배포 환경에서는 여러 모델이 동시에 로드될 수 있다. 또한 입력 배치, 중간 특징 맵, 출력 버퍼 등이 추가 메모리를 사용한다. 여유 있게 계획하고 피크 사용량을 측정한다.

GPU 메모리 파편화도 문제를 일으킨다. 반복적인 할당과 해제로 메모리가 단편화되어 실제 사용량보다 더 많은 메모리가 필요해진다. 정기적으로 캐시를 비우고, 가능하면 메모리 풀을 사용한다. 장시간 실행 후 프로세스를 재시작하는 것도 방법이다.

추론 속도를 실험실 환경에서만 측정하는 것도 위험하다. 실제 환경에서는 이미지 전송 지연, 전처리 오버헤드, 멀티 스레딩 경쟁 등이 추가된다. End-to-end 지연을 실제 환경에서 측정하고 여유를 둔다.

**Deployment Environment Mismatch**

개발 환경(고성능 워크스테이션, 최신 GPU)과 배포 환경(엣지 디바이스, 오래된 하드웨어)이 다른 것을 간과한다. 개발에서는 잘 작동하던 것이 배포에서는 느리거나 메모리 부족으로 실패한다. 배포 환경과 동일하거나 유사한 하드웨어에서 사전 테스트를 수행한다.

CUDA 버전이나 드라이버 불일치도 문제를 일으킨다. 특정 CUDA 버전에 의존하는 기능을 사용했는데 배포 환경에 없으면 오류가 발생한다. 의존성을 명확히 하고 배포 환경을 검증한다. Docker로 환경을 완전히 격리하는 것이 안전하다.

네트워크 조건을 고려하지 않는 것도 실수다. 클라우드 기반 추론은 네트워크 지연과 대역폭에 영향을 받는다. 네트워크가 불안정하면 타임아웃이나 연결 끊김이 발생한다. 로컬 캐싱, 재시도 로직, graceful degradation을 구현한다.

**Lack of Monitoring**

배포 후 모니터링 없이 방치하는 것은 위험하다. 성능 저하, drift, 오류가 발생해도 인지하지 못한다. 실시간 모니터링 시스템을 구축하고 이상 징후를 조기에 감지한다. 정기적으로 성능 리포트를 생성하고 검토한다.

로그를 충분히 남기지 않는 것도 문제다. 오류 발생 시 원인을 파악할 정보가 없다. 모든 입력, 출력, 중간 결과, 시스템 상태를 적절히 로깅한다. 그러나 과도한 로깅은 성능을 저하시키고 스토리지를 소진하므로 균형을 맞춘다.

Alert fatigue도 주의해야 한다. 너무 많은 경고가 발생하면 운영자가 무시하게 된다. Critical한 이슈만 경고하고 나머지는 로그로 기록한다. 경고의 우선순위를 명확히 하고 actionable하게 만든다.

**Ignoring Edge Cases**

평균적인 경우만 고려하고 극단적인 경우를 무시한다. 매우 어두운 이미지, 과도하게 밝은 이미지, 부분적으로 가려진 객체, 여러 객체가 겹친 경우 등을 테스트하지 않는다. 이런 edge case에서 모델이 실패할 수 있다. 의도적으로 극단적인 조건을 생성하고 테스트한다.

Adversarial case도 고려해야 한다. 의도적으로 모델을 속이려는 시도가 있을 수 있다. 특히 품질 검사를 통과하기 위해 결함을 숨기려는 경우다. Adversarial robustness를 평가하고 필요하면 방어 기법을 적용한다.

Null이나 corrupted 입력을 처리하지 않는 것도 문제다. 카메라 오류, 전송 에러로 손상된 이미지가 들어올 수 있다. 입력 검증을 수행하고 invalid 입력을 gracefully 처리한다. 시스템이 crash되지 않도록 예외 처리를 철저히 한다.

**Poor Communication**

기술팀과 운영팀 간 소통 부족이 많은 문제를 일으킨다. 모델의 한계나 가정을 운영팀이 이해하지 못하면 잘못된 기대를 갖는다. 모델이 완벽하지 않고 일부 오류가 불가피함을 명확히 전달한다. 현실적인 성능 목표를 함께 설정한다.

결과 해석을 제대로 교육하지 않는 것도 문제다. 이상 점수의 의미, 이상 맵 읽는 법, 임계값의 역할을 운영자가 이해해야 한다. 실제 사례로 교육하고 질문에 답한다. 사용자 친화적인 UI와 명확한 설명을 제공한다.

피드백 루프가 없는 것도 개선을 막는다. 운영자의 의견, 실패 사례, 개선 제안을 수집하는 체계가 없으면 모델이 발전하지 않는다. 정기적인 미팅, 피드백 시스템, 이슈 트래킹을 구축한다. 운영자가 발견한 문제를 신속히 해결하여 신뢰를 구축한다.

이러한 pitfall들을 인지하고 사전에 대비하면 성공적인 배포 가능성이 크게 높아진다. Memory-based 방법은 이미 충분히 성숙했지만, 실무 적용에서는 기술적 우수성만큼 운영적 고려사항이 중요하다. 체계적인 접근과 지속적인 개선으로 안정적이고 효과적인 시스템을 구축할 수 있다.

# 7. Research Insights

## 7.1 Why Memory-Based Works

Memory-based 방법이 이상 탐지에서 탁월한 성공을 거둔 이유는 여러 이론적, 실무적 요인의 조합이다. 표면적으로는 단순해 보이지만, 그 성공 뒤에는 깊은 통찰이 있다.

**Transfer Learning의 강력함**

Memory-based 방법의 성공은 우선 ImageNet 사전 학습의 놀라운 전이 학습 효과에 기인한다. 자연 이미지로 학습된 CNN이 산업 이미지에서도 즉시 유용한 특징을 추출한다는 것은 직관적으로 명확하지 않다. ImageNet의 고양이, 개, 자동차와 산업 부품의 결함은 매우 다른 시각적 개념이다. 그러나 중간 층 특징들은 edge, texture, corner, pattern 같은 일반적인 시각적 primitives를 포착한다.

이러한 low-to-mid level 특징은 도메인 불변적이다. 긁힘(scratch)은 날카로운 edge로, 균열(crack)은 선형 패턴으로, 오염(contamination)은 texture 불규칙성으로 나타난다. CNN의 계층적 구조는 이러한 primitives를 효과적으로 추출한다. Layer1은 edge와 색상, layer2는 texture와 간단한 패턴, layer3는 복잡한 구조를 포착한다. 이는 인간 시각 시스템의 계층적 처리와 유사하다.

Empirical evidence도 전이 학습의 효과를 뒷받침한다. ImageNet에서 학습된 ResNet50과 무작위 초기화된 ResNet50을 비교하면 성능 차이가 극적이다. 사전 학습된 모델은 95-99% AUROC를 달성하지만, 무작위 초기화 모델은 60-70%에 그친다. 이는 learned representations의 중요성을 보여준다.

흥미롭게도 ImageNet보다 더 관련된 도메인에서 사전 학습하면 성능이 향상된다. 예를 들어 의료 이미지 이상 탐지에서는 ImageNet보다 medical image dataset(CheXpert, MIMIC)에서 사전 학습한 모델이 더 효과적이다. 이는 domain-specific pretraining의 가능성을 시사한다. 그러나 산업 이상 탐지에서는 대규모 사전 학습 데이터셋이 부족하여 ImageNet이 여전히 표준이다.

**Manifold Hypothesis의 타당성**

Memory-based 방법은 암묵적으로 manifold hypothesis에 기반한다. 고차원 특징 공간에서 정상 데이터는 저차원 다양체(manifold) 위에 집중되어 있다는 가정이다. 이상 데이터는 이 다양체로부터 벗어나 있으므로 거리나 밀도 기반으로 탐지할 수 있다.

이 가정이 왜 타당한가? 자연 이미지와 산업 이미지는 모든 가능한 픽셀 조합의 극히 일부만 차지한다. $256 \times 256$ RGB 이미지의 가능한 조합은 $256^{3 \times 256 \times 256} \approx 10^{473000}$이다. 그러나 의미 있는 이미지는 이 중 극소수다. 대부분의 무작위 픽셀 배열은 노이즈일 뿐이다. 정상 제품 이미지는 더욱 제한적이다. 특정 형태, 색상, texture, 구조를 가져야 한다.

CNN 특징 공간에서도 동일한 압축이 일어난다. 수천 차원의 특징 공간에서 정상 샘플들은 실제로는 수십에서 수백 차원의 manifold 위에 있다. PCA나 t-SNE로 시각화하면 정상 샘플들이 응집된 cluster를 형성하는 것을 볼 수 있다. 이상 샘플은 cluster 바깥에 흩어져 있다.

Manifold learning 이론은 이를 수학적으로 뒷받침한다. Intrinsic dimension estimation 기법으로 정상 데이터의 실제 자유도를 추정할 수 있다. 실험 결과 MVTec AD의 정상 이미지들은 ResNet 특징 공간(차원 수천)에서 intrinsic dimension이 50-200 정도였다. 이는 차원 축소가 정보 손실 없이 가능함을 의미한다.

Memory-based 방법은 이 manifold의 coverage를 제공한다. PaDiM은 가우시안으로 manifold를 근사하고, PatchCore는 coreset으로 manifold를 샘플링한다. 테스트 샘플이 manifold 근처에 있으면 정상, 멀리 있으면 이상으로 판정한다. 이는 one-class learning의 본질이다.

**Locality Principle의 적용**

패치 단위 분석이 효과적인 이유는 locality principle 때문이다. 이상은 일반적으로 국소적으로 나타난다. 전체 이미지가 완전히 비정상인 경우는 드물고, 특정 영역에만 결함이 있다. 긁힘, 균열, 오염, 변색 등은 모두 공간적으로 제한된다.

전역적(global) 특징만 사용하면 이러한 국소 이상을 놓친다. 이미지 전체의 평균 특징은 작은 결함의 신호를 희석시킨다. 예를 들어 $256 \times 256$ 이미지에서 $10 \times 10$ 크기의 결함은 전체의 0.15%에 불과하다. 전역 평균에서는 거의 보이지 않는다.

패치 단위 접근은 이 문제를 해결한다. 각 패치를 독립적으로 분석하므로 작은 결함도 해당 패치에서는 큰 신호가 된다. 결함이 있는 패치는 높은 이상 점수를 받고, 정상 패치는 낮은 점수를 받는다. 이미지 레벨 점수는 최대값으로 취하므로 하나의 이상 패치만 있어도 탐지된다.

패치 크기 선택은 trade-off다. 너무 작으면 맥락 정보가 부족하고 noise에 민감하다. 너무 크면 여러 객체나 영역이 섞여 신호가 희석된다. 전형적인 $8 \times 8$에서 $28 \times 28$ 범위(원본 이미지 기준)는 경험적으로 좋은 균형점이다. 이는 결함의 일반적인 크기와 대응된다.

PatchCore의 locally aware features는 locality를 유지하면서도 맥락을 통합한다. $3 \times 3$ 이웃 집계는 단일 패치의 고립성을 완화하면서도 여전히 국소적이다. 이는 결함이 종종 여러 인접 패치에 걸쳐 나타난다는 관찰과 일치한다.

**Simplicity의 힘**

Memory-based 방법의 성공은 역설적으로 그 단순함에 있다. 복잡한 생성 모델(VAE, GAN)이나 end-to-end 학습보다 단순한 거리 기반 방법이 더 효과적이다. 이는 Occam's razor의 실증이다.

단순함의 첫 번째 이점은 학습 안정성이다. 역전파, 경사 하강, loss balancing 등이 필요 없다. 단순히 특징을 추출하고 통계량을 계산하거나 샘플을 저장한다. Hyperparameter에 민감하지 않고 초기화에 의존하지 않는다. 동일한 데이터에서 항상 동일한 결과를 얻는다.

두 번째 이점은 해석 가능성이다. 거리나 밀도는 직관적인 개념이다. 이상 점수가 높다는 것은 정상 샘플들과 많이 다르다는 명확한 의미다. 가장 가까운 정상 샘플을 보여주면 어떤 면에서 다른지 비교할 수 있다. 복잡한 신경망의 black box 판정보다 신뢰할 수 있다.

세 번째는 일반화 능력이다. 과적합(overfitting)의 위험이 적다. 학습 가능한 파라미터가 없거나(PatchCore) 최소한이므로(PaDiM) 학습 데이터의 노이즈를 외우지 않는다. 대신 본질적인 정상 분포를 포착한다. 제한된 학습 데이터에서도 robust하다.

네 번째는 계산 효율성이다. 복잡한 반복 최적화가 없어 학습이 빠르다. 추론도 단순한 거리 계산이나 nearest neighbor 탐색이므로 효율적이다. 하드웨어 가속(GPU, 전용 라이브러리)의 이점을 최대한 활용할 수 있다.

이는 deep learning의 아이러니다. 강력한 표현 학습(representation learning)과 단순한 탐지 메커니즘의 조합이 복잡한 end-to-end 시스템보다 우수하다. 표현은 복잡해도 되지만(deep CNN), 탐지는 단순해야 한다. 이는 feature learning과 decision making의 분리 전략이다.

**Data Efficiency**

Memory-based 방법은 매우 data efficient하다. Few-shot learning이 자연스럽게 가능하다. 수십 개의 정상 샘플만으로도 reasonable한 성능을 보인다. 이는 산업 응용에서 중요하다. 신제품 출시나 희귀 제품에서는 대량의 학습 데이터를 수집하기 어렵다.

Data efficiency의 이유는 사전 학습이다. CNN은 이미 풍부한 시각적 지식을 가지고 있어 domain adaptation만 필요하다. Memory-based 방법은 이 지식을 그대로 활용하고, 타겟 도메인의 정상 분포만 학습한다. 이는 적은 데이터로도 충분하다.

또한 패치 단위 접근이 data augmentation 효과를 제공한다. 하나의 이미지에서 수천 개의 패치가 추출되어 각각 하나의 학습 샘플로 작용한다. 200개 이미지에서 수십만 개의 패치 샘플을 얻는다. 이는 statistical estimation을 안정화한다.

Coreset selection은 data efficiency를 더욱 향상시킨다. Redundant 샘플을 제거하고 informative 샘플만 유지한다. 적은 샘플로도 전체 분포를 대표할 수 있다. 이는 active learning의 원리와 유사하다.

실험적으로 PatchCore는 10-20개의 정상 샘플만으로도 90%+ AUROC를 달성한다. 50-100개면 95%+에 도달한다. 이는 supervised learning(수천-수만 샘플 필요)과 극명히 대비된다. Few-shot anomaly detection에서 memory-based 방법은 unmatched다.

## 7.2 Theoretical Guarantees

Memory-based 방법의 실증적 성공은 일부 이론적 보장으로 뒷받침된다. 완전한 이론은 아직 없지만, 주요 구성 요소들은 잘 이해된 수학적 기반을 가진다.

**Statistical Learning Theory**

PaDiM의 가우시안 모델링은 classical statistical theory에 기반한다. Central Limit Theorem에 의해 충분한 샘플이 있을 때 샘플 평균은 근사적으로 가우시안 분포를 따른다. 각 패치 위치에서 $N$개의 정상 샘플은 해당 위치의 정상 특징 분포를 추정하는 데 사용된다.

Sample complexity theory는 reliable estimation에 필요한 샘플 수를 정량화한다. $d$차원 가우시안의 공분산 행렬을 정확히 추정하려면 $N \gg d$가 필요하다. 구체적으로 $N > 10d$ 정도면 안정적이다. PaDiM이 차원 축소를 하는 이유 중 하나다. $d' = 100$이면 $N > 1000$ 패치 샘플이 필요한데, 200개 이미지에서 충분히 얻을 수 있다.

Mahalanobis 거리의 통계적 성질도 잘 알려져 있다. 진짜 가우시안 분포에서 샘플링된 점의 Mahalanobis 거리 제곱은 $\chi^2_d$ 분포를 따른다. 이는 hypothesis testing framework를 제공한다. 유의 수준 $\alpha$를 설정하고 $\chi^2_d$ 분포의 $(1-\alpha)$ 백분위수를 임계값으로 사용할 수 있다. 예를 들어 $\alpha = 0.01$이면 99%의 정상 샘플이 임계값 이하가 된다.

그러나 실제 데이터가 정확히 가우시안이 아니므로 이론적 보장은 근사적이다. Robustness theory는 가우시안 가정의 위반이 moderate할 때 Mahalanobis 거리가 여전히 효과적임을 보여준다. 이는 PaDiM의 실무 성공을 설명한다.

**Coreset Theory**

PatchCore의 coreset selection은 computational geometry의 coreset theory에 기반한다. $k$-center problem은 $n$개 점 집합에서 $k$개 중심을 선택하여 모든 점이 가장 가까운 중심으로부터의 최대 거리를 최소화하는 문제다. 이는 NP-hard지만 greedy 2-approximation 알고리즘이 존재한다.

구체적으로 greedy 알고리즘이 선택한 coreset $\mathcal{C}$에 대해 최대 거리 $r_{\text{greedy}}$는 최적 coreset $\mathcal{C}^*$의 최대 거리 $r_{\text{opt}}$의 2배 이내다.

$$r_{\text{greedy}} = \max_{\mathbf{x} \in \mathcal{M}} \min_{\mathbf{c} \in \mathcal{C}} \|\mathbf{x} - \mathbf{c}\|_2 \leq 2 \cdot r_{\text{opt}}$$

이는 worst-case 보장이다. 실제로는 greedy가 훨씬 더 잘 작동한다. Empirical studies는 greedy coreset이 optimal에 매우 가까움을 보여준다.

$\epsilon$-net theory는 coreset 크기의 하한을 제공한다. $d$차원 단위 공에서 모든 점이 거리 $\epsilon$ 이내에 coreset 점을 갖도록 하려면 최소 $(1/\epsilon)^d$개 점이 필요하다. 이는 고차원의 curse를 보여준다. 그러나 실제 데이터가 저차원 manifold에 있으면 필요한 coreset 크기는 intrinsic dimension에 의존한다.

Johnson-Lindenstrauss lemma는 차원 축소 후에도 거리가 보존됨을 보장한다. 랜덤 프로젝션으로 $d$차원에서 $d' = O(\log n / \epsilon^2)$ 차원으로 축소해도 모든 점 쌍 간 거리가 $(1 \pm \epsilon)$ 배 이내로 유지된다. 이는 높은 확률로 성립한다. PaDiM과 PatchCore의 차원 축소가 정보 손실 없이 가능한 이론적 근거다.

**Nearest Neighbor Theory**

PatchCore의 k-NN 이상 점수는 nearest neighbor density estimation에 기반한다. $k$-NN 거리는 local density의 역추정치다. 밀도가 높은 영역에서는 nearest neighbor가 가깝고, 희소한 영역에서는 멀다.

Non-parametric statistics theory는 k-NN 밀도 추정의 일관성(consistency)을 보장한다. 샘플 수 $n \to \infty$이고 $k \to \infty$이지만 $k/n \to 0$이면, k-NN 밀도 추정은 진짜 밀도로 수렴한다. 이는 asymptotic 보장이지만, 유한 샘플에서도 reasonable approximation을 제공한다.

$k$의 선택은 bias-variance trade-off다. 작은 $k$는 low bias, high variance를, 큰 $k$는 high bias, low variance를 가진다. Optimal $k$는 $k^* = O(n^{4/(d+4)})$로 알려져 있다. 그러나 이는 이론적 최적이고, 실무에서는 $k = 5-15$ 정도가 robust하다.

Curse of dimensionality는 k-NN의 주요 도전이다. 고차원에서는 nearest neighbor도 매우 멀리 있어 의미 있는 locality가 없다. 그러나 데이터가 저차원 manifold에 집중되어 있으면 effective dimension이 낮아 k-NN이 작동한다. Manifold hypothesis가 다시 중요해진다.

**Generalization Bounds**

Memory-based 방법의 generalization 성능은 PAC(Probably Approximately Correct) learning theory로 분석할 수 있다. One-class learning의 목표는 정상 데이터의 support를 학습하는 것이다. Support는 데이터가 non-zero 확률로 나타나는 영역이다.

Support estimation의 sample complexity는 VC dimension에 의존한다. Simple geometric shapes(구, 타원)는 낮은 VC dimension을 가져 적은 샘플로도 학습 가능하다. PaDiM의 가우시안은 타원체로 VC dimension이 $O(d^2)$다. PatchCore의 coreset은 union of balls로 VC dimension이 $O(k)$다.

Generalization error는 다음 형태로 bounded된다.

$$\mathbb{P}(\text{test error} - \text{train error} > \epsilon) \leq \delta$$

여기서 $\epsilon$는 $O(\sqrt{(d \log n + \log(1/\delta))/n})$ 정도다. 충분한 샘플 $n$이 있으면 generalization gap이 작다.

그러나 이러한 classical bounds는 종종 너무 보수적이다. 실제 성능은 이론적 예측보다 훨씬 좋다. 이는 실제 데이터의 구조(manifold, smoothness)가 worst-case 분석보다 유리하기 때문이다. Refined analysis가 필요하지만 active research area다.

**Limitations of Theory**

Memory-based 방법의 이론적 이해는 아직 불완전하다. 몇 가지 주요 gap이 있다. 첫째, transfer learning의 이론이 부족하다. 왜 ImageNet 특징이 산업 이상 탐지에 효과적인지 rigorous하게 설명하지 못한다. Domain adaptation theory가 일부 통찰을 제공하지만, 구체적인 보장은 약하다.

둘째, manifold learning의 실제 적용이 이론과 gap이 있다. Manifold hypothesis는 직관적이지만, 실제 manifold의 구조(차원, 곡률, topology)를 정량화하기 어렵다. Intrinsic dimension estimation도 noisy하고 알고리즘에 의존적이다.

셋째, finite sample behavior가 불명확하다. 대부분의 이론은 asymptotic이다($n \to \infty$). 실무에서는 $n = 50-500$ 정도의 제한된 샘플이다. Finite sample correction이나 non-asymptotic bound가 더 유용하지만 어렵다.

넷째, optimal hyperparameter 선택의 이론이 부족하다. $k$, coreset 크기, 차원 축소 목표 등을 principled하게 선택하는 방법이 명확하지 않다. 현재는 empirical tuning이나 heuristic에 의존한다.

다섯째, adversarial robustness가 이해되지 않았다. Adversarial examples에 대한 memory-based 방법의 취약성이나 방어 메커니즘이 연구되지 않았다. 이는 critical 응용에서 중요할 수 있다.

그럼에도 불구하고 현재의 이론은 memory-based 방법이 왜 작동하는지 부분적으로 설명한다. 더 중요하게는 언제 잘 작동하고 언제 실패할지 예측하는 guidance를 제공한다. 이론과 실무의 gap을 좁히는 것은 향후 연구의 중요한 방향이다.

## 7.3 Open Research Questions

Memory-based 패러다임은 성숙했지만 여전히 흥미로운 open questions가 남아 있다. 이들은 학술 연구와 실무 개선 모두에 기회를 제공한다.

**Foundation Model Integration**

가장 명백한 방향은 foundation models와의 결합이다. CLIP, DINOv2, SAM 같은 대규모 vision models는 ImageNet보다 훨씬 강력한 표현을 제공한다. 이들을 memory-based 프레임워크에 통합하면 성능이 향상될 것으로 기대된다.

구체적 질문들이 있다. 어떤 foundation model이 이상 탐지에 최적인가? CLIP의 vision-language alignment가 도움이 되는가, 아니면 순수 vision model(DINOv2)이 나은가? ViT(Vision Transformer) 특징과 CNN 특징 중 무엇이 더 효과적인가? 어느 층의 특징을 사용해야 하는가?

Foundation model의 특징 차원은 수천에 달한다. 이를 어떻게 효율적으로 처리하는가? 차원 축소가 여전히 필요한가, 아니면 coreset selection만으로 충분한가? Attention 기반 feature selection이 도움이 되는가?

Multi-class 환경으로의 확장도 중요하다. 전통적 memory-based 방법은 single-class다. 각 카테고리마다 별도 모델을 학습한다. Foundation models는 단일 모델로 여러 카테고리를 처리할 가능성을 제공한다. Zero-shot 또는 few-shot으로 새로운 카테고리에 즉시 적응할 수 있는가? 이는 산업 확장성에 혁명적이다.

**Active Learning and Data Collection**

Memory-based 방법은 data efficient하지만 여전히 정상 데이터가 필요하다. Active learning으로 가장 informative한 샘플을 선택적으로 수집하면 효율성이 더욱 향상될 것이다. 어떤 샘플을 라벨링해야 할지 intelligent하게 결정하는 전략이 필요하다.

Coreset selection을 active learning에 활용할 수 있는가? Coreset은 이미 informative 샘플을 식별한다. 이를 역으로 사용하여 수집이 부족한 영역을 찾을 수 있다. Diversity-based sampling과 uncertainty-based sampling을 결합하는 전략이 가능할 것이다.

Incremental learning도 중요한 질문이다. 새로운 정상 샘플이 지속적으로 들어올 때 모델을 어떻게 효율적으로 업데이트하는가? 전체 재학습 없이 incrementally coreset을 업데이트할 수 있는가? Online coreset selection 알고리즘이 연구되고 있지만 실용적이지 않다.

Drift detection과 adaptation도 미해결 문제다. 정상 분포가 시간에 따라 변할 때 이를 자동으로 감지하고 적응하는 메커니즘이 필요하다. Statistical process control 기법을 통합할 수 있을 것이다. 점진적 drift와 급격한 shift를 구별하고 다르게 대응해야 한다.

**Explainability and Interpretability**

Memory-based 방법은 relatively interpretable하지만 개선의 여지가 있다. 가장 가까운 정상 샘플을 보여주는 것 이상의 설명이 필요하다. 어떤 특징 차원이나 영역이 이상 판정에 기여했는지 명확히 해야 한다.

Attention mechanism을 통합하여 중요한 영역을 강조할 수 있는가? GradCAM 같은 기법을 memory-based 방법에 적용할 수 있는가? 이는 gradient가 없어 직접 적용이 어렵지만, 변형이 가능할 것이다.

Counterfactual explanation도 유용할 것이다. 이 샘플이 정상으로 판정되려면 어떻게 변해야 하는가? 이는 결함 수정이나 공정 개선에 직접 도움이 된다. Optimization 기반 접근으로 minimal perturbation을 찾을 수 있다.

Natural language explanation은 foundation models와 결합하여 가능할 것이다. CLIP이나 BLIP 같은 vision-language model을 사용하여 "이 샘플은 표면에 긁힘이 있어 비정상입니다"와 같은 설명을 생성할 수 있다. 이는 비전문가 운영자의 이해를 돕는다.

**Adversarial Robustness**

Memory-based 방법의 adversarial robustness는 거의 연구되지 않았다. 의도적으로 모델을 속이려는 adversarial examples에 얼마나 취약한가? Gradient-free black-box attack에는 어떻게 대응하는가?

일반적으로 memory-based 방법은 gradient-based attack에 robust하다고 예상된다. Gradient가 없거나(PatchCore) minimal(PaDiM)이므로 FGSM이나 PGD 같은 gradient attack이 직접 적용되기 어렵다. 그러나 surrogate model을 사용한 transfer attack은 가능하다.

Query-based attack도 위협이다. 모델에 반복적으로 쿼리하여 decision boundary를 탐색하고 adversarial example을 생성할 수 있다. 이를 방어하는 메커니즘이 필요하다. Query budget을 제한하거나, 비정상적인 쿼리 패턴을 탐지하는 방법이 있다.

Certified defense도 흥미로운 방향이다. Randomized smoothing 같은 기법을 memory-based 방법에 적용하여 provable robustness를 제공할 수 있는가? 이는 critical 응용에서 중요할 것이다.

**Multi-modal and Multi-sensor Fusion**

현재 memory-based 방법은 주로 RGB 이미지에 집중한다. 그러나 산업 검사에서는 다양한 센서가 사용된다. Depth camera, thermal camera, X-ray, ultrasound 등이 결함 유형에 따라 사용된다. 이들을 효과적으로 융합하는 방법이 필요하다.

Early fusion(센서 레벨)과 late fusion(점수 레벨) 중 무엇이 나은가? Intermediate fusion(특징 레벨)이 최적인가? 각 modality의 신뢰도를 동적으로 가중치화하는 adaptive fusion이 가능한가?

Temporal information도 활용되지 않고 있다. 동영상이나 시계열 이미지에서 시간적 일관성을 고려하면 성능이 향상될 것이다. Temporal coreset이나 3D(spatial+temporal) memory bank를 구성할 수 있다.

**Efficiency and Scalability**

PatchCore는 이미 매우 효율적이지만 extreme scale에서는 도전이 있다. 수천 개 카테고리, 고해상도 이미지($4K$, $8K$), 실시간 비디오 등에서 더욱 최적화가 필요하다.

Neural hash나 learning to hash를 사용하여 특징을 binary code로 변환하면 메모리와 거리 계산이 극적으로 줄어든다. Hamming distance는 XOR과 popcount로 매우 빠르다. 성능 저하를 최소화하면서 효율성을 극대화하는 것이 challenge다.

Knowledge distillation으로 작은 student 백본을 학습하여 특징 추출을 가속할 수 있다. 특히 엣지 디바이스에서 유용하다. EfficientAD는 이미 이 방향을 탐구했지만 memory-based 프레임워크와의 최적 결합은 미해결이다.

**Domain Adaptation and Transfer**

ImageNet 사전 학습이 강력하지만 domain gap이 큰 경우는 어려움이 있다. Medical imaging, satellite imagery, microscopy 등 특수 도메인에서는 성능이 제한적이다. Unsupervised domain adaptation 기법을 통합하여 이를 완화할 수 있는가?

Self-supervised learning으로 타겟 도메인에서 추가로 pretraining하는 것이 효과적일 것이다. 정상 데이터만으로 contrastive learning이나 masked autoencoding을 수행한다. 이렇게 fine-tuned된 특징을 memory-based 방법에 사용한다.

Meta-learning도 유망하다. 여러 관련 카테고리에서 학습하여 새로운 카테고리에 빠르게 적응하는 meta-model을 구축할 수 있다. Few-shot learning의 MAML이나 Prototypical Networks 같은 아이디어를 차용할 수 있다.

**Theoretical Understanding**

앞서 언급한 이론적 gap을 좁히는 것도 중요한 연구 방향이다. Transfer learning의 rigorous theory, finite-sample analysis, optimal hyperparameter selection, adversarial robustness 증명 등이 필요하다.

Manifold learning과 memory-based 방법의 formal connection을 확립하는 것도 유용할 것이다. Manifold의 geometric properties(차원, 곡률)가 coreset 크기나 k 값에 어떻게 영향을 미치는지 정량화할 수 있다.

PAC-Bayes framework나 information-theoretic approach로 generalization을 분석하는 것도 가능하다. 이는 tighter bound와 practical guidance를 제공할 것이다.

**Conclusion**

Memory-based 패러다임은 이상 탐지에서 remarkable success를 거두었다. 단순하면서도 효과적인 접근으로 single-class 산업 검사의 표준이 되었다. PatchCore의 99%+ AUROC와 메모리 효율성은 실무 배포를 현실화했다.

그러나 여전히 흥미로운 연구 문제들이 남아 있다. Foundation models, multi-class learning, explainability, robustness, efficiency 등 다양한 방향으로 발전 가능성이 있다. 이론적 이해를 깊게 하고 새로운 응용 영역을 탐색하는 것도 중요하다.

Memory-based 방법의 미래는 밝다. 단순함과 효과성의 조합은 시대를 초월한 가치다. 새로운 기술(foundation models, neural architecture search)과 결합하면서도 core principle을 유지하는 것이 성공의 열쇠일 것이다. 학술 연구와 산업 응용이 함께 발전하여 zero-defect manufacturing의 비전에 한 걸음 더 다가갈 것이다.