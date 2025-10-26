# Vision Anomaly Detection: Overview and Paradigm Evolution

**Document Version**: 1.0  
**Last Updated**: 2025.10.12
**Total Models**: 21  
**Paradigms**: 6

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [Challenges in Industrial Anomaly Detection](#11-challenges-in-industrial-anomaly-detection)
   - 1.2 [Deep Learning Revolution](#12-deep-learning-revolution)
   - 1.3 [Document Scope and Organization](#13-document-scope-and-organization)

2. [Six Major Paradigms](#2-six-major-paradigms)
   - 2.1 [Paradigm Classification Framework](#21-paradigm-classification-framework)
   - 2.2 [Evolution Timeline (2018-2025)](#22-evolution-timeline-2018-2025)
   - 2.3 [Paradigm Comparison Matrix](#23-paradigm-comparison-matrix)

3. [Key Technical Transitions](#3-key-technical-transitions)
   - 3.1 [Memory Efficiency Breakthrough](#31-memory-efficiency-breakthrough-padim--patchcore)
   - 3.2 [Speed Optimization](#32-speed-optimization-cflow--fastflow)
   - 3.3 [Paradigm Inversion](#33-paradigm-inversion-stfpm--reverse-distillation)
   - 3.4 [Learning Stability](#34-learning-stability-ganomaly--draem)
   - 3.5 [Multi-class Revolution](#35-multi-class-revolution-traditional--foundation-models)

4. [Performance Landscape](#4-performance-landscape)
   - 4.1 [MVTec AD Benchmark Overview](#41-mvtec-ad-benchmark-overview)
   - 4.2 [Top-5 Models by Accuracy](#42-top-5-models-by-accuracy)
   - 4.3 [Speed-Accuracy-Memory Trade-offs](#43-speed-accuracy-memory-trade-offs)

5. [Future Directions](#5-future-directions)
   - 5.1 [Short-term Outlook (2025-2026)](#51-short-term-outlook-2025-2026)
   - 5.2 [Mid-term Outlook (2026-2028)](#52-mid-term-outlook-2026-2028)
   - 5.3 [Long-term Vision (2028-2030)](#53-long-term-vision-2028-2030)
   - 5.4 [Zero-Defect Manufacturing](#54-zero-defect-manufacturing)

6. [Reading Guide](#6-reading-guide)
   - 6.1 [For Beginners](#61-for-beginners)
   - 6.2 [For Researchers](#62-for-researchers)
   - 6.3 [For Practitioners](#63-for-practitioners)

[References](#references)

---

## 1. Introduction

### 1.1 Challenges in Industrial Anomaly Detection

산업 현장에서 제품의 품질 검사는 생산성과 직결되는 핵심 과제이다. 전통적으로 숙련된 검사자가 육안으로 수행하던 품질 검사는 인력 부족, 일관성 문제, 그리고 미세 결함 탐지의 한계로 인해 자동화의 필요성이 대두되었다. 딥러닝 기반 이상 탐지(Anomaly Detection)는 이러한 문제를 해결할 수 있는 강력한 대안으로 주목받고 있다.

그러나 산업 환경의 이상 탐지는 일반적인 분류(Classification) 문제와 근본적으로 다른 도전 과제를 안고 있다. 첫째, 정상 샘플은 풍부하지만 이상 샘플은 극히 드물다는 점이다. 생산 라인에서 불량률이 1% 미만인 경우가 대부분이며, 특히 신제품의 경우 이상 샘플을 전혀 확보하지 못하는 경우도 흔하다. 이는 supervised learning 방법론을 직접 적용할 수 없음을 의미한다. 둘째, 이상의 종류가 무한히 다양하다는 점이다. 스크래치, 찌그러짐, 오염, 색상 불균일 등 예측 불가능한 형태로 나타나며, 학습 단계에서 모든 이상 유형을 다룰 수 없다. 이는 open-set recognition 문제로 귀결된다. 셋째, 실시간 처리와 높은 정확도를 동시에 요구한다는 점이다. 고속 생산 라인에서는 밀리초 단위의 추론 시간이 필요하지만, 동시에 99% 이상의 높은 탐지 정확도를 유지해야 한다.

이러한 제약 조건은 기존의 컴퓨터 비전 기술로는 충족시키기 어려웠다. 전통적인 통계적 방법(PCA, One-class SVM 등)은 고차원 이미지 데이터에서 복잡한 패턴을 포착하지 못했고, 초기 딥러닝 방법(Autoencoder 등)은 정상 패턴의 미묘한 변형까지 재구성하여 이상을 놓치는 문제가 있었다. 따라서 산업 이상 탐지는 오랜 기간 "풀리지 않는 문제"로 남아있었다.

### 1.2 Deep Learning Revolution

2018년부터 2025년까지 지난 7년간, 딥러닝 기반 이상 탐지 분야는 급격한 발전을 이루었다. 이 기간 동안 20개 이상의 주요 모델이 제안되었으며, 각 모델은 독특한 접근법으로 위에서 언급한 도전 과제들을 해결하고자 시도했다. 본 문서는 이러한 모델들을 기술적 원리와 방법론에 따라 6개의 주요 패러다임으로 분류하였다.

첫 번째 패러다임은 Memory-Based 또는 Feature Matching 방식이다. 이 접근법은 정상 샘플들의 특징 벡터를 메모리에 저장하고, 테스트 시점에 입력 샘플과의 거리를 계산하여 이상을 탐지한다. 가장 직관적인 방법이면서도, PatchCore가 달성한 99.1%의 정확도는 현재까지 single-class 환경에서 최고 수준을 유지하고 있다. 이 패러다임의 성공은 "복잡한 모델링 없이도 직접적인 비교만으로 강력한 성능을 달성할 수 있다"는 중요한 통찰을 제공한다.

두 번째 패러다임은 Normalizing Flow 방식이다. 생성 모델의 일종인 normalizing flow는 가역적인 변환을 통해 복잡한 데이터 분포를 단순한 분포로 매핑하고, 확률 밀도를 계산하여 이상을 탐지한다. FastFlow가 대표적이며, 98.5%의 정확도와 20-50ms의 빠른 속도로 실무에서 널리 채택되었다. 특히 3D flow를 2D flow로 단순화하여 속도는 3배 빨라지고 성능은 오히려 향상된 사례는, "문제의 본질을 이해하고 불필요한 복잡도를 제거하는 것"의 중요성을 보여준다.

세 번째 패러다임은 Knowledge Distillation 방식이다. Pre-trained teacher 네트워크의 지식을 student가 정상 데이터에서만 모방하도록 학습하며, 이상 샘플에서는 모방에 실패한다는 원리를 활용한다. 이 패러다임은 두 가지 극단적인 방향으로 발전했다. Reverse Distillation은 패러다임을 역전시켜 98.6%의 높은 정확도를 달성했고, EfficientAD는 극한의 최적화로 1-5ms의 실시간 처리를 현실화했다. 이러한 양극단의 존재는 동일한 기본 원리에서도 다양한 최적화 방향이 가능함을 입증한다.

네 번째 패러다임은 Reconstruction-Based 방식이다. Autoencoder나 GAN으로 정상 샘플 재구성을 학습하고, 재구성 오류로 이상을 탐지한다. 초기 GANomaly는 GAN의 학습 불안정성으로 실패했으나, DRAEM은 simulated anomaly를 도입하여 패러다임을 전환했다. 정상 이미지에 인위적 결함을 추가하고 이를 제거하도록 학습함으로써, 안정적인 학습과 97.5%의 성능을 달성했다. 특히 10-50장의 소량 데이터만으로 학습 가능한 few-shot 능력은 신제품이나 희귀 결함 시나리오에서 큰 가치가 있다.

다섯 번째 패러다임은 Feature Adaptation 방식이다. ImageNet 등 대규모 데이터셋으로 사전 학습된 모델의 특징을 산업 이미지 도메인에 적응시켜 활용한다. DFM은 PCA와 Mahalanobis distance만으로 15분 만에 94-95%의 baseline을 구축할 수 있어, 빠른 프로토타이핑에 유용하다. 그러나 94.5-97.5% 수준의 성능은 SOTA 대비 1.6-4.6%p 낮아, 본격적인 실무 배포에는 한계가 있다. 이 패러다임은 "시작점이지 종착점은 아니다"라는 명확한 역할을 가진다.

여섯 번째 패러다임은 Foundation Model 기반 방식이다. CLIP, DINOv2, GPT-4V 등 수억~수십억 개 샘플로 사전 학습된 범용 모델을 활용한다. 이는 이상 탐지 패러다임을 근본적으로 전환하고 있다. Dinomaly는 단일 모델로 multi-class 이상 탐지를 수행하여 98.8%를 달성했고, 15개 제품을 검사할 때 메모리를 93% 절감했다. WinCLIP은 zero-shot 이상 탐지를 가능하게 하여, 학습 데이터 없이도 즉시 배포할 수 있다. VLM-AD는 자연어로 결함을 설명하여 explainable AI를 실현했다. Foundation model의 등장은 "모든 제품을 하나의 모델로", "학습 데이터 없이 즉시", "왜 불량인지 설명 가능하게"라는 세 가지 차원에서 패러다임을 전환하고 있다.

### 1.3 Document Scope and Organization

본 문서는 vision anomaly detection 분야의 전체 지형도를 제공하는 것을 목표로 한다. 6개 패러다임의 핵심 원리, 대표 모델, 주요 기술적 전환점, 그리고 미래 연구 방향을 개괄적으로 다룬다. 각 패러다임의 상세한 수학적 정식화, 알고리즘 세부사항, 그리고 구현 가이드는 개별 패러다임 문서에서 제공된다.

본 문서의 구성은 다음과 같다. 2장에서는 6개 패러다임의 분류 체계와 시간순 발전 과정을 제시한다. 3장에서는 5개의 주요 기술적 전환점을 분석하며, 각 전환점이 어떻게 패러다임의 발전을 이끌었는지 설명한다. 4장에서는 MVTec AD 벤치마크를 기준으로 성능 현황을 정리하고, 정확도-속도-메모리의 trade-off를 논한다. 5장에서는 단기(2025-2026), 중기(2026-2028), 장기(2028-2030) 연구 방향과 산업 전망을 제시한다. 마지막으로 6장에서는 독자 유형별 추천 읽기 경로를 안내한다.

---

## 2. Six Major Paradigms

### 2.1 Paradigm Classification Framework

본 연구에서는 20개 이상의 이상 탐지 모델을 기술적 원리와 방법론에 따라 6개의 주요 패러다임으로 분류하였다. 분류 기준은 다음과 같다. 첫째, 이상 탐지의 핵심 메커니즘이 무엇인가. 둘째, 정상 데이터를 어떻게 모델링하는가. 셋째, 이상 점수를 어떻게 계산하는가. 이러한 기준에 따라 각 패러다임은 명확히 구분되는 특성을 가진다.

Memory-Based 방식은 정상 샘플의 특징 벡터를 명시적으로 저장하고, 테스트 샘플과의 거리를 직접 계산한다. 수학적으로는 거리 함수 $d(f_{test}, \mathcal{M}_{normal})$로 표현되며, Mahalanobis distance나 Euclidean distance를 사용한다. PaDiM, PatchCore, DFKDE가 이 패러다임에 속한다. PatchCore는 coreset selection 알고리즘으로 메모리 효율과 정확도를 동시에 달성하여, 현재까지 single-class 환경에서 99.1%의 최고 성능을 유지하고 있다.

Normalizing Flow 방식은 가역적인 변환을 통해 정상 데이터의 확률 분포를 명시적으로 모델링한다. Change of variables 공식 $\log p(x) = \log p(f(x)) + \log|\det \frac{\partial f}{\partial x}|$을 사용하여 log-likelihood를 계산하고, 이를 이상 점수로 사용한다. CFlow, FastFlow, CS-Flow, U-Flow가 이에 해당한다. FastFlow는 3D flow를 2D flow로 단순화하여 속도와 성능을 동시에 개선한 대표적 사례이다.

Knowledge Distillation 방식은 teacher-student 프레임워크를 활용한다. Pre-trained teacher의 지식을 student가 정상 데이터에서만 학습하며, 이상 점수는 $\|f_T(x) - f_S(x)\|$로 계산된다. STFPM, FRE, Reverse Distillation, EfficientAD가 포함된다. 이 패러다임은 양극단으로 발전했는데, Reverse Distillation은 정확도를 극대화하여 98.6%를 달성했고, EfficientAD는 속도를 극대화하여 1-5ms의 실시간 처리를 가능하게 했다.

Reconstruction-Based 방식은 autoencoder나 GAN으로 정상 샘플 재구성을 학습하고, 재구성 오류 $\|x - \hat{x}\|$를 이상 점수로 사용한다. GANomaly, DRAEM, DSR이 이에 속한다. GANomaly는 GAN의 학습 불안정성으로 실패했으나, DRAEM은 simulated anomaly를 사용하여 supervised learning의 안정성을 확보했다. DRAEM의 혁신은 10-50장의 소량 데이터로도 97.5%의 성능을 달성할 수 있는 few-shot 능력에 있다.

Feature Adaptation 방식은 ImageNet 등의 pre-trained 특징을 타겟 도메인에 적응시킨다. 수식으로는 $f_{adapted} = \mathcal{A}(f_{pretrained}, \mathcal{D}_{target})$로 표현된다. DFM과 CFA가 대표적이다. DFM은 PCA와 Mahalanobis distance만 사용하여 15분 만에 94-95%의 baseline을 구축할 수 있다. 그러나 성능이 SOTA 대비 1.6-4.6%p 낮아, 빠른 feasibility 검증용으로 적합하지만 본격 배포에는 제한이 있다.

Foundation Model 방식은 CLIP, DINOv2, GPT-4V 등 대규모 사전 학습 모델을 활용한다. 수억~수십억 개 샘플로 학습된 범용 표현을 이상 탐지에 전이시킨다. WinCLIP, Dinomaly, VLM-AD, SuperSimpleNet, UniNet이 포함된다. Dinomaly는 단일 모델로 multi-class 이상 탐지를 수행하여 98.8%를 달성했으며, 15개 제품 검사 시 메모리를 93% 절감한다. 이는 패러다임의 근본적 전환을 의미한다.

각 패러다임은 서로 다른 강점과 약점을 가지며, 특정 응용 시나리오에 최적화되어 있다. Memory-based는 최고 정확도를, normalizing flow는 확률적 해석을, knowledge distillation은 속도 최적화를, reconstruction은 few-shot 능력을, feature adaptation은 빠른 프로토타이핑을, foundation model은 multi-class와 zero-shot을 제공한다. 따라서 실무에서는 요구사항에 따라 적절한 패러다임을 선택해야 한다.

### 2.2 Evolution Timeline (2018-2025)

이상 탐지 분야의 발전은 다섯 개의 주요 시기로 구분할 수 있다. 각 시기는 독특한 기술적 특성과 도전 과제를 가지며, 다음 시기로의 전환점을 제공한다.

첫 번째 시기는 2018-2019년의 태동기이다. 이 시기는 딥러닝 기반 이상 탐지의 가능성을 탐색하는 초기 단계였다. GANomaly(2018)는 GAN을 이용한 선구적 시도였으나, mode collapse와 oscillation 등 GAN 특유의 학습 불안정성으로 인해 6-10시간의 긴 학습 시간이 필요했고, 수렴이 보장되지 않았다. 성능도 93-95%로 낮아 실무 적용에 실패했다. 이는 "GAN의 이론적 우아함이 항상 실용성으로 이어지지는 않는다"는 교훈을 남겼다. DFM(2019)은 PCA라는 가장 단순한 방법으로 94.5-95.5%를 달성했다. 5-15분의 학습 시간과 간단한 구현은 매력적이었지만, 선형 변환의 한계로 인해 복잡한 비선형 관계를 포착하지 못했다. 이 시기의 모델들은 "이상 탐지가 가능하다"는 것을 보여주었지만, 실무 요구사항인 99% 이상의 정확도와 안정적 학습을 충족하지 못했다.

두 번째 시기는 2020-2021년의 성장기이다. 이 시기에는 주요 패러다임이 확립되었고, 성능이 급격히 향상되었다. PaDiM(2020)은 memory-based 방식의 기초를 다졌다. 각 패치 위치에서 정상 패턴의 다변량 가우시안 분포를 모델링하고 Mahalanobis distance를 계산하여 96.5%의 성능을 달성했다. 하이퍼파라미터에 덜 민감하고 구현이 간단하다는 장점이 있었으나, 모든 패치 위치에서 공분산 행렬을 저장해야 하므로 메모리 사용량이 2-5GB에 달하는 치명적인 단점이 있었다. 2021년은 기술적 다양화의 해였다. Normalizing flow 진영에서는 CFlow가 position-conditional flow로 98.2%를 달성했고, FastFlow는 3D flow를 2D flow로 단순화하여 속도는 3배 향상시키고 성능은 98.5%로 오히려 높였다. Knowledge distillation에서는 STFPM이 teacher-student 패러다임을 확립하여 96.8%를 보였다. Reconstruction에서는 DRAEM이 simulated anomaly를 도입하여 패러다임을 전환하고 97.5%를 달성했다. 이 시기는 "다양한 접근법이 모두 95% 이상의 성능을 달성할 수 있다"는 것을 보여주었고, 문제는 "어떤 접근법이 실무에 가장 적합한가?"로 이동했다.

세 번째 시기는 2022년의 성숙기이다. 이 해에는 현재까지 이어지는 SOTA 모델들이 등장했다. PatchCore는 greedy coreset selection 알고리즘을 도입하여 PaDiM의 메모리 문제를 해결했다. 전체 분포를 대표하는 소수의 핵심 패치만 선택함으로써 메모리를 90% 감소시켰고(2-5GB → 100-500MB), 동시에 성능은 오히려 향상되었다(96.5% → 99.1%). 이는 간단한 아이디어가 메모리 효율과 정확도를 동시에 달성할 수 있음을 보여준 대표적 사례이다. Reverse Distillation은 전통적인 knowledge distillation을 역전시켰다. Teacher를 단순화하여 one-class embedding을 생성하고, student가 이를 역으로 재구성하도록 학습함으로써 98.6%를 달성했고, pixel AUROC 98.5%는 최고 수준의 localization 성능이다. U-Flow는 자동 임계값 설정으로 운영 자동화를, CFA는 hypersphere embedding으로 domain shift 대응을, DSR은 dual subspace로 텍스처 특화를 이루었다. 2022년은 "99%의 벽"을 넘은 해로 기록된다.

네 번째 시기는 2023년의 과도기이다. 이 시기는 속도 최적화에 대한 관심이 높아졌으나, 충분한 성과를 거두지 못한 시기였다. FRE(2023)는 STFPM의 2배 속도 향상을 목표로 경량화된 backbone과 간소화된 구조를 사용했다. 추론 속도는 20-40ms에서 10-30ms로 개선되었으나, 성능은 96.8%에서 95-96%로 저하되었다. 문제는 2배 향상이 실무에서 결정적 차이를 만들지 못했다는 것이다. 20ms든 10ms든 모두 "실시간은 아니다". 이후 EfficientAD가 1-5ms로 20-200배 향상을 달성하면서 FRE는 가치를 잃었다. FRE의 교훈은 명확하다. 점진적 개선은 충분하지 않으며, 혁명적 발전이 필요하다. 한편 WinCLIP(2023)은 OpenAI의 CLIP 모델을 활용하여 zero-shot 이상 탐지의 가능성을 열었다. 91-95%의 정확도는 낮지만, 학습 데이터 없이 텍스트 프롬프트만으로 즉시 배포할 수 있다는 점은 특정 시나리오에서 혁명적이다.

다섯 번째 시기는 2024-2025년의 foundation model 시대이다. 이 시기는 대규모 사전 학습 모델이 이상 탐지 패러다임을 근본적으로 전환하는 시기이다. EfficientAD(2024)는 patch description network(PDN)라는 경량 네트워크(약 50K 파라미터)와 autoencoder를 결합하여 1-5ms의 실시간 처리를 현실화했다. 이는 단순히 빠른 수준이 아니라, 초당 200-1000 프레임 처리가 가능하여 고속 생산 라인의 전수 검사를 가능하게 했다. VLM-AD(2024)는 GPT-4V 등 vision-language model을 활용하여 자연어로 결함을 설명한다. 단순히 이상 점수만 제공하는 것이 아니라, 결함 유형, 위치, 크기, 심각도, 가능한 원인, 개선 권장사항까지 제시한다. 정확도는 96-97%로 높지 않지만, explainable AI의 가치는 막대하다. Dinomaly(2025)는 DINOv2 foundation model로 단일 모델로 multi-class 이상 탐지를 수행한다. 98.8%의 multi-class 성능과 99.2%의 single-class 성능을 보이며, 15개 제품 검사 시 메모리를 7.5GB에서 500MB로 93% 절감한다. UniNet(2025)은 contrastive learning으로 98.3%를 달성하며 강건한 decision boundary를 학습한다. Foundation model 시대는 이상 탐지를 "single-class에서 multi-class로", "학습 필요에서 zero-shot으로", "수치만 제공에서 자연어 설명으로" 전환하고 있다.

### 2.3 Paradigm Comparison Matrix

6개 패러다임을 정량적으로 비교하면 다음과 같다. Memory-based 방식은 정확도 측면에서 절대 우위를 가진다. PatchCore의 99.1%는 현재까지 single-class 환경에서 최고 기록이다. 추론 속도는 50-100ms로 중간 수준이며, coreset selection 덕분에 메모리는 100-500MB로 실용적이다. 학습 복잡도는 중간 수준이고, 해석 가능성은 매우 높다. 거리 기반 측정이므로 왜 이상으로 판단했는지 명확히 설명할 수 있다. 적용 분야는 최고 정확도가 필요한 정밀 검사이다. 상세 분석은 [Memory-Based Methods](01-memory-based.md) 문서에서 제공된다.

Normalizing flow 방식은 균형잡힌 성능을 보인다. FastFlow는 98.5%의 정확도와 20-50ms의 속도로 실무에서 가장 많이 선택되는 모델 중 하나이다. 메모리는 500MB-1GB로 다소 높지만 허용 가능한 수준이다. 학습 복잡도는 높은 편으로, flow network 설계와 학습이 필요하다. 해석 가능성은 중간 수준으로, log-likelihood는 확률적 해석이 가능하지만 직관적이지는 않다. 적용 분야는 속도와 정확도가 모두 중요한 일반적인 검사이다. 상세 분석은 [Normalizing Flow](02-normalizing-flow.md) 문서에서 제공된다.

Knowledge distillation 방식은 양극단의 특성을 보인다. Reverse Distillation은 98.6%의 높은 정확도를 달성하지만 100-200ms의 느린 속도를 가진다. 반면 EfficientAD는 97.8%의 정확도를 약간 희생하고 1-5ms의 극한 속도를 달성한다. 메모리는 양쪽 모두 효율적이며, 학습 복잡도는 중간 수준이다. 해석 가능성은 낮은데, teacher와 student의 차이가 무엇을 의미하는지 직관적으로 설명하기 어렵다. 적용 분야는 명확히 구분되는데, Reverse Distillation은 정밀 검사용이고 EfficientAD는 실시간 라인용이다. 상세 분석은 [Knowledge Distillation](03-knowledge-distillation.md) 문서에서 제공된다.

Reconstruction-based 방식은 few-shot 능력이 독보적이다. DRAEM은 10-50장의 소량 데이터로 97.5%를 달성한다. 추론 속도는 50-100ms로 중간이고, 메모리는 300-500MB로 효율적이다. 학습 복잡도는 높은데, simulated anomaly 생성과 discriminative network 학습이 필요하다. 해석 가능성은 높은 편으로, 재구성된 이미지와 원본의 차이를 시각적으로 확인할 수 있다. 적용 분야는 신제품 출시 초기나 희귀 결함처럼 데이터 수집이 어려운 환경이다. 상세 분석은 [Reconstruction-Based](04-reconstruction-based.md) 문서에서 제공된다.

Feature adaptation 방식은 극단적인 간단함과 빠른 개발이 장점이다. DFM은 15분 만에 94.5-95.5%의 baseline을 구축할 수 있다. 추론 속도는 10-20ms로 빠르고, 메모리는 50-100MB로 극소이다. 학습 복잡도는 매우 낮은데, PCA와 거리 계산만 필요하다. 그러나 성능이 SOTA 대비 1.6-4.6%p 낮다는 것이 치명적인 약점이다. 해석 가능성은 중간 수준이다. 적용 분야는 빠른 feasibility 검증이나 저사양 환경이며, 본격 배포 전에 다른 모델로 전환해야 한다. 상세 분석은 [Feature Adaptation](05-feature-adaptation.md) 문서에서 제공된다.

Foundation model 방식은 multi-class, zero-shot, explainable이라는 세 가지 혁명을 가져왔다. Dinomaly는 단일 모델로 98.8%의 multi-class 성능을 보이며, 메모리를 93% 절감한다. 추론 속도는 80-120ms로 중간이고, 모델 크기는 300MB-1.5GB로 다소 크다. WinCLIP은 zero-shot으로 91-95%를 달성하며, 학습 데이터가 전혀 필요 없다. VLM-AD는 96-97%의 정확도로 자연어 설명을 제공한다. 학습 복잡도는 낮은데, foundation model은 이미 학습되어 있고 fine-tuning만 필요하다. 해석 가능성은 VLM-AD의 경우 매우 높다. 적용 분야는 다양한데, Dinomaly는 multi-class 환경, WinCLIP은 신제품 즉시 검사, VLM-AD는 품질 보고서 자동화에 적합하다. 상세 분석은 [Foundation Models](06-foundation-models.md) 문서에서 제공된다.

종합하면, 각 패러다임은 서로 다른 강점과 trade-off를 가진다. Memory-based는 최고 정확도를, normalizing flow는 균형을, knowledge distillation은 극한 최적화를, reconstruction은 few-shot을, feature adaptation은 빠른 시작을, foundation model은 범용성을 제공한다. 실무에서는 요구사항에 따라 적절한 패러다임을 선택해야 하며, 상세한 선택 가이드는 [Comprehensive Comparison](07-comparison.md) 문서에서 제공된다.

---

## 3. Key Technical Transitions

### 3.1 Memory Efficiency Breakthrough (PaDiM → PatchCore)

PaDiM에서 PatchCore로의 전환은 memory-based 패러다임을 실무에서 사용 가능하게 만든 결정적 혁신이었다. PaDiM(2020)은 각 패치 위치에서 정상 패턴의 다변량 가우시안 분포를 모델링하는 우아한 접근법이었다. 수학적으로는 각 위치 $(i,j)$에서 $p(x_{i,j}) = \mathcal{N}(\mu_{i,j}, \Sigma_{i,j})$로 표현되며, Mahalanobis distance $M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$를 이상 점수로 사용했다. 96.5%의 정확도는 당시로서는 매우 높은 수준이었고, 하이퍼파라미터에 덜 민감하며 구현이 간단하다는 장점이 있었다.

그러나 치명적인 문제가 있었다. 224×224 이미지에서 28×28 feature map을 사용하면 784개의 위치가 생기고, 각 위치마다 $d \times d$ 크기의 공분산 행렬을 저장해야 한다. 특징 차원 $d=550$일 경우 각 공분산 행렬이 302,500개의 파라미터를 가지므로, 총 메모리가 2-5GB에 달했다. 이는 GPU 메모리 제약이 있는 환경에서 심각한 장벽이었고, 여러 모델을 동시에 운영하거나 엣지 디바이스에 배포하는 것을 불가능하게 만들었다.

PatchCore(2022)는 이 문제를 coreset selection이라는 간단하지만 강력한 아이디어로 해결했다. 핵심 통찰은 "모든 학습 패치를 저장할 필요 없이, 전체 분포를 충분히 대표하는 소수의 핵심 패치만 있으면 된다"는 것이었다. Greedy 알고리즘을 사용하여 각 단계에서 기존 coreset과 가장 먼 샘플을 선택한다. 수학적으로는 $\mathcal{C} = \arg\min_{|\mathcal{C}|=M} \max_{x \in \mathcal{X}} \min_{c \in \mathcal{C}} \|x - c\|_2$로 표현되며, 이는 전체 특징 공간을 작은 coreset으로 효과적으로 커버함을 보장한다(ε-cover 이론).

결과는 놀라웠다. 메모리 사용량이 90% 이상 감소했다(2-5GB → 100-500MB). 더 놀라운 것은 성능이 오히려 향상되었다는 점이다(96.5% → 99.1%). 왜 데이터를 줄였는데 성능이 올랐을까. 첫째, coreset이 노이즈와 outlier를 제거하는 효과가 있다. 둘째, locally aware patch features를 사용하여 인접 픽셀의 context 정보를 포함시켰다. 셋째, Mahalanobis distance 대신 단순한 Euclidean distance를 사용하여 계산 복잡도를 $O(d^2)$에서 $O(d)$로 줄였다. 넷째, 과적합이 줄어들어 일반화 성능이 향상되었다.

PatchCore의 성공은 "간단한 아이디어가 복잡한 모델을 이긴다"는 중요한 교훈을 준다. 이론적 우아함보다 실용적 효과가 더 중요하며, 문제의 본질을 이해하면 간단한 해결책을 찾을 수 있다. 이 전환으로 memory-based 방식은 실무에서 널리 채택되었고, 현재까지 single-class 환경에서 첫 번째로 고려되는 방법론이 되었다.

### 3.2 Speed Optimization (CFlow → FastFlow)

CFlow에서 FastFlow로의 전환은 "단순화가 때로는 성능과 속도를 동시에 향상시킬 수 있다"는 반직관적 통찰을 보여준 사례이다. CFlow(2021)는 position-conditional normalizing flow를 사용하여 각 공간 위치에서 독립적인 분포를 학습했다. Multi-scale에서 여러 flow network를 사용하여 다양한 크기의 이상을 탐지했고, 98.2%의 높은 정확도를 달성했다. 확률론적 해석이 가능하고 pixel-level localization이 우수하다는 장점이 있었다.

그러나 치명적인 단점이 있었다. 3D tensor $(C \times H \times W)$에 flow를 직접 적용하면서 계산 복잡도가 매우 높아졌다. 추론 시간이 100-150ms로 느려서 실시간 처리가 불가능했고, flow network 학습에 2-3시간이 소요되어 개발 주기가 길었다. 이는 고속 생산 라인이나 실시간 품질 모니터링에 적용하기 어렵게 만들었다.

FastFlow(2021)는 이 문제를 과감한 단순화로 해결했다. 핵심 아이디어는 "채널 간 상관관계가 이상 탐지에서 공간 구조만큼 중요하지 않다"는 것이었다. 3D flow 대신 2D flow를 사용하여, 채널 차원을 분리하고 각 공간 위치 $(H \times W)$에서만 flow를 적용했다. 이는 계산 복잡도를 크게 줄였다.

결과는 놀라웠다. 추론 속도가 100-150ms에서 20-50ms로 2-3배 향상되었다. 학습 시간도 2-3시간에서 30-60분으로 3-4배 단축되었다. 더 놀라운 것은 성능이 오히려 향상되었다는 점이다(98.2% → 98.5%). 왜 채널 간 상관관계를 무시했는데 성능이 올랐을까. 첫째, 이상 탐지에서는 공간 구조(어디에 무엇이 있는가)가 채널 간 관계(색상 간 상관)보다 더 중요하다. 둘째, 간단한 모델이 과적합을 방지하여 일반화 성능을 높였다. 셋째, 학습이 빠르므로 더 많은 실험을 수행하여 최적 하이퍼파라미터를 찾을 수 있었다.

FastFlow의 성공은 "문제의 본질을 이해하고 불필요한 복잡도를 제거하는 것"의 중요성을 보여준다. 모든 정보를 사용하는 것이 항상 최선은 아니다. 오히려 핵심 정보에 집중하고 불필요한 부분을 제거하는 것이 더 나은 결과를 가져올 수 있다. 이 통찰은 딥러닝 모델 설계에서 일반적으로 적용 가능한 원칙이다.

### 3.3 Paradigm Inversion (STFPM → Reverse Distillation)

STFPM에서 Reverse Distillation으로의 전환은 "관점의 전환이 혁신을 낳는다"는 것을 보여준 대표적 사례이다. STFPM(2021)은 knowledge distillation 패러다임을 이상 탐지에 도입했다. Pre-trained teacher 네트워크(예: ResNet)와 student 네트워크를 사용하여, student가 정상 데이터에서 teacher의 특징을 모방하도록 학습한다. 정상 샘플에서는 $f_T \approx f_S$ (모방 성공)이고, 이상 샘플에서는 $f_T \neq f_S$ (모방 실패)가 되므로, $\|f_T(x) - f_S(x)\|$를 이상 점수로 사용한다. 96.8%의 성능과 20-40ms의 빠른 추론 속도로 많은 후속 연구의 baseline이 되었다.

그러나 STFPM은 근본적인 한계가 있었다. Teacher가 ImageNet의 일반적인 특징을 학습했기 때문에 산업 이미지의 특수한 패턴에 최적화되지 않았다. 예를 들어 ImageNet은 고양이, 자동차, 건물 등 고수준 semantic 정보에 특화되어 있지만, 산업 이상 탐지는 스크래치, 얼룩, 색상 불균일 등 저수준 texture 정보가 더 중요하다. 이러한 domain gap으로 인해 96.8%의 정확도는 당시 SOTA(PatchCore 99.1%)에 비해 낮았다.

Reverse Distillation(2022)은 이 문제를 패러다임 역전으로 해결했다. 전통적인 방향인 Teacher(복잡) → Student(단순)가 아니라, Teacher(단순) ← Student(복잡) 구조를 사용한다. 구체적으로, teacher가 정상 데이터의 압축된 one-class embedding을 생성하고, student가 이를 역으로 재구성하도록 학습한다. 이 접근법이 효과적인 이유는 무엇일까. STFPM의 teacher는 ImageNet의 일반 특징을 학습했지만, Reverse Distillation의 teacher는 타겟 도메인의 정상 패턴만 압축한다. 따라서 도메인 특화된 표현을 학습할 수 있다.

수학적으로는 다음과 같이 표현된다. Teacher network $T$는 정상 이미지 $x$를 저차원 embedding $z = T(x)$로 압축한다. Student network $S$는 encoder $E$와 decoder $D$로 구성되며, $\hat{x} = D(E(x))$로 이미지를 재구성한다. 학습 목표는 재구성된 이미지의 특징 $T(\hat{x})$가 원본 이미지의 특징 $T(x)$와 같도록 하는 것이다. Loss function은 $\mathcal{L} = \|T(x) - T(\hat{x})\|$이다. 테스트 시점에는 이상 샘플이 정상 manifold를 벗어나므로 재구성이 실패하고, $T(x)$와 $T(\hat{x})$의 차이가 커진다.

결과는 놀라웠다. Image AUROC 98.6%, Pixel AUROC 98.5%로 SOTA급 성능을 달성했다. STFPM 대비 1.8%p 향상은 패러다임 역전의 효과를 입증했다. 특히 Pixel AUROC 98.5%는 현재까지도 최고 수준의 localization 성능이다. 이는 정밀한 결함 위치 파악이 중요한 반도체나 의료 기기 검사에서 큰 가치가 있다.

그러나 복잡한 encoder-decoder 구조로 인해 추론 시간이 100-200ms로 느려졌다는 단점이 있다. 이는 정밀도와 속도의 trade-off를 보여주며, 응용 분야에 따라 적절한 선택이 필요함을 시사한다. Reverse Distillation의 성공은 "동일한 원리에서도 관점을 바꾸면 새로운 혁신이 가능하다"는 교훈을 준다.

### 3.4 Learning Stability (GANomaly → DRAEM)

GANomaly에서 DRAEM으로의 전환은 "실용적 supervised가 이론적 unsupervised보다 나을 수 있다"는 것을 보여준 사례이다. GANomaly(2018)는 GAN을 활용한 선구적 시도였다. Encoder-Decoder-Encoder(E-D-E) 구조를 사용하여, 입력 이미지를 두 번 인코딩하고 두 latent code의 차이를 이상 점수로 사용했다. 이론적으로는 우아한 접근법이었다.

그러나 GAN 특유의 학습 불안정성이 치명적인 문제였다. Mode collapse로 인해 generator가 다양한 정상 패턴을 생성하지 못했고, oscillation으로 인해 수렴이 보장되지 않았다. 6-10시간의 긴 학습 시간이 필요했고, 여러 번 재시도해야 했다. 성능도 93-95%로 낮았다. 4개의 네트워크(E-D-E + Discriminator)를 관리해야 하는 복잡도도 문제였다. 학습 과정에서 generator와 discriminator의 균형을 맞추기 위한 섬세한 하이퍼파라미터 튜닝이 필요했다.

DRAEM(2021)은 reconstruction 패러다임을 근본적으로 혁신했다. 핵심 아이디어는 simulated anomaly를 사용하여 supervised learning 효과를 얻는 것이었다. 정상 이미지에 인위적 결함을 추가하는 augmentation을 수행한다. 수학적으로는 $x_{aug} = (1 - m) \odot x_{normal} + m \odot t_{source}$로 표현되며, 여기서 $m$은 binary mask, $t_{source}$는 텍스처 소스이다. Perlin noise를 사용하여 자연스러운 결함 패턴을 생성하고, 이를 정상 이미지에 합성한다.

이는 패러다임 전환이었다. GANomaly는 정상 데이터만 사용하는 unsupervised learning이었고, 이상 샘플을 본 적이 없었다. DRAEM은 정상 + simulated anomaly를 사용하는 supervised learning으로, 이상 패턴을 명시적으로 학습한다. Reconstruction network는 augmented image에서 정상 이미지를 복원하도록 학습하고, discriminative network는 어디가 이상인지 segmentation map을 생성한다. SSIM loss와 focal loss를 결합하여 구조적 유사성과 pixel-wise classification을 동시에 학습한다.

결과는 놀라웠다. 97.5%의 높은 정확도를 달성했다(GANomaly 대비 +2.5~4.5%p). 학습이 안정적이고 2-4시간만에 수렴했다. 무엇보다 혁신적인 것은 10-50장의 정상 샘플만으로 학습 가능한 few-shot 능력이다. 신제품 출시 초기에 충분한 데이터를 확보하지 못한 상황에서도 97.5%를 달성할 수 있다는 것은 실무에서 큰 가치가 있다.

DRAEM의 성공 비결은 무엇일까. Simulated anomaly가 실제 결함과 유사한 패턴을 만들어내고, 이를 제거하는 과정에서 정상 manifold를 학습한다. GAN의 불안정성 없이도 명확한 학습 신호를 제공한다. Supervised learning이므로 gradient가 안정적이고 수렴이 빠르다. 이는 "이론적 우아함보다 실용적 효과"의 중요성을 보여준다.

### 3.5 Multi-class Revolution (Traditional → Foundation Models)

전통적 방법에서 foundation model로의 전환은 이상 탐지 패러다임의 근본적 변화를 의미한다. 전통적 방법(PatchCore, FastFlow 등)은 타겟 도메인 데이터로만 학습하며, 수백 장의 학습 데이터가 필요하고, single-class 모델을 클래스당 하나씩 구축해야 했다. 예를 들어 15개 제품을 검사하려면 15개의 독립적인 모델이 필요하고, 각 모델마다 데이터 수집, 학습, 검증, 배포 과정을 반복해야 했다. 총 메모리는 7.5GB에 달하고, 관리 복잡도가 매우 높았다.

Foundation model 방식은 이러한 제약을 돌파한다. CLIP, DINOv2, GPT-4V 등 수억~수십억 개 샘플로 사전 학습된 범용 모델을 활용한다. 이들 모델은 대규모 데이터에서 학습한 강력한 표현을 가지며, 다양한 downstream task에 전이 가능하다. 이상 탐지에서는 이러한 범용 표현을 활용하여 세 가지 혁명을 가져왔다.

첫 번째는 multi-class 혁명이다. Dinomaly(2025)는 DINOv2 foundation model을 활용하여 단일 모델로 multi-class 이상 탐지를 수행한다. DINOv2는 self-supervised learning으로 학습되어 ImageNet 라벨 없이도 semantic과 low-level 정보를 모두 포착한다. Dinomaly는 class-conditional memory bank를 사용하여 각 클래스의 대표 특징만 저장한다. 15개 제품을 검사할 때, 전통적 방법은 15개 모델(총 7.5GB)이 필요하지만, Dinomaly는 단일 모델(500MB)로 처리한다. 메모리 93% 절감은 단순히 용량 문제가 아니라, 저렴한 하드웨어(GPU 메모리 8GB → 2GB)로 배포 가능함을 의미한다. 성능도 98.8%(multi-class)로 높으며, single-class로 사용하면 99.2%로 PatchCore(99.1%)를 초과한다. 배포 시간도 15시간에서 3시간으로 80% 단축된다.

두 번째는 zero-shot 혁명이다. WinCLIP(2023)은 OpenAI의 CLIP 모델을 활용하여 텍스트 프롬프트만으로 이상 탐지를 수행한다. CLIP은 4억 개의 이미지-텍스트 쌍으로 학습되어, 이미지와 텍스트를 동일한 embedding space에 매핑한다. WinCLIP은 이를 활용하여 $\text{Score} = \text{sim}(I, \text{"defective"}) - \text{sim}(I, \text{"normal"})$로 이상을 탐지한다. 전통적 방법은 데이터 수집(2-4주), 학습(1-2시간), 검증(1주)이 필요하지만, WinCLIP은 프롬프트 작성(10분)만으로 즉시 배포할 수 있다. 91-95%의 정확도는 낮지만, 신제품 출시 즉시 검사나 다품종 소량 생산 환경에서 혁명적이다. 프롬프트만 바꾸면 새로운 제품에 즉시 적용할 수 있다.

세 번째는 explainable AI 혁명이다. VLM-AD(2024)는 GPT-4V 등 vision-language model을 활용하여 자연어로 결함을 설명한다. 전통적 모델은 "이상 점수 0.87"만 제공하지만, VLM-AD는 결함 유형(scratch), 위치(upper left corner), 크기(5mm), 심각도(moderate), 가능한 원인(handling damage), 개선 권장사항(inspect handling process)까지 제시한다. 이는 품질 엔지니어의 근본 원인 분석, 생산 관리자의 공정 개선, 감사 담당의 근거 문서 작성을 자동화한다. 96-97%의 정확도는 높지 않지만, 규제가 엄격한 산업(의료, 항공)에서 "왜 불량으로 판정했는가"를 설명할 수 있다는 점이 중요하다.

Foundation model의 성공은 "대규모 사전 학습의 힘"을 보여준다. 수억~수십억 개 샘플에서 학습한 범용 표현은 소량의 타겟 데이터로는 학습하기 어려운 복잡한 패턴을 포착한다. 이는 2025년 이후 이상 탐지의 주류가 될 것으로 전망된다. 2026-2027년에는 산업 특화 foundation model(Manufacturing CLIP, Industrial DINOv2)이 등장하여 zero-shot 정확도가 98% 이상으로 향상될 것으로 예상된다.

---

## 4. Performance Landscape

### 4.1 MVTec AD Benchmark Overview

MVTec AD(MVTec Anomaly Detection)는 산업 이상 탐지 분야의 표준 벤치마크로 자리잡았다. 2019년 독일 MVTec Software GmbH에서 공개한 이 데이터셋은 실제 산업 환경을 반영하도록 설계되었으며, 15개 카테고리로 구성되어 있다. 이는 텍스처(Texture) 5개 카테고리와 객체(Object) 10개 카테고리로 나뉜다.

텍스처 카테고리는 Carpet, Grid, Leather, Tile, Wood로 구성된다. 이들은 반복적인 패턴을 가진 표면으로, 색상 불균일, 패턴 변형, 오염 등의 결함이 주로 나타난다. 텍스처 카테고리는 공간적 변형(spatial variation)에 강건해야 하며, 미묘한 색상이나 질감 변화를 탐지해야 한다는 특징이 있다. 예를 들어 Carpet은 색상 얼룩이나 실 끊김을, Grid는 격자 패턴의 왜곡을, Leather는 주름이나 긁힘을 탐지해야 한다.

객체 카테고리는 Bottle, Cable, Capsule, Hazelnut, Metal Nut, Pill, Screw, Toothbrush, Transistor, Zipper로 구성된다. 이들은 명확한 형태를 가진 제품으로, 긁힘, 변형, 파손, 누락 등의 결함이 나타난다. 객체 카테고리는 정렬(alignment)과 형태 변화(shape deformation)에 민감하며, 3D 구조를 이해해야 한다는 특징이 있다. 예를 들어 Bottle은 병의 변형이나 오염을, Cable은 굽힘이나 손상을, Screw는 나사산 손상이나 머리 변형을 탐지해야 한다.

데이터셋 구성을 살펴보면, 총 5,354장의 고해상도 이미지(700×700에서 1024×1024)로 구성되며, 학습 데이터는 정상 샘플만 포함하고(3,629장), 테스트 데이터는 정상과 이상 샘플을 모두 포함한다(1,725장). 각 이상 샘플에는 pixel-level ground truth mask가 제공되어 localization 성능을 정량적으로 평가할 수 있다. 이는 실제 산업 환경과 유사하게 설계된 것으로, 정상 샘플은 충분하지만 이상 샘플은 제한적인 상황을 반영한다.

평가 지표는 두 가지 수준에서 측정된다. Image-level에서는 전체 이미지가 정상인지 이상인지 판단하며, AUROC(Area Under the Receiver Operating Characteristic curve)로 평가한다. Pixel-level에서는 어느 픽셀이 이상인지 localization하며, 역시 AUROC로 평가한다. 높은 image AUROC는 불량품 검출 능력을, 높은 pixel AUROC는 결함 위치 파악 능력을 나타낸다. 실무에서는 두 지표가 모두 중요한데, image AUROC는 불량품 선별에, pixel AUROC는 결함 원인 분석과 공정 개선에 사용된다.

MVTec AD 벤치마크의 중요성은 세 가지로 요약된다. 첫째, 표준화된 비교를 가능하게 한다. 동일한 데이터셋과 평가 지표로 다양한 모델을 공정하게 비교할 수 있다. 둘째, 실제 산업 환경을 반영한다. 고해상도 이미지, 다양한 결함 유형, 정상 데이터만 학습 가능한 조건 등이 실무와 유사하다. 셋째, 연구 발전을 가속화한다. 공개 데이터셋이므로 전세계 연구자들이 접근할 수 있고, 빠른 벤치마킹과 검증이 가능하다.

그러나 MVTec AD 벤치마크의 한계도 인지해야 한다. 벤치마크 환경은 통제된 조명, 고품질 이미지, 명확한 결함을 가지지만, 실무 환경은 조명 변화, 다양한 이미지 품질, 모호한 경계 사례를 포함한다. 따라서 벤치마크 성능이 실제 라인에서 3-5%p 하락할 수 있다. 또한 벤치마크는 15개 카테고리만 포함하지만, 실제 산업은 수백 개의 제품과 결함 유형을 다룬다. 그럼에도 불구하고 MVTec AD는 현재까지 가장 널리 사용되는 벤치마크로, 이상 탐지 연구의 표준으로 자리잡았다.

### 4.2 Top-5 Models by Accuracy

MVTec AD 벤치마크에서 최고 성능을 보이는 상위 5개 모델은 각기 다른 패러다임을 대표하며, 서로 다른 강점을 가진다.

첫 번째는 PatchCore로 99.1%의 image AUROC를 달성하여 현재까지 single-class 환경에서 최고 기록을 보유하고 있다. Pixel AUROC도 98.2%로 매우 높다. Coreset selection 알고리즘으로 메모리를 100-500MB로 효율화했으며, 추론 속도는 50-100ms이다. PatchCore의 강점은 절대적인 정확도와 안정성에 있다. 모든 15개 카테고리에서 일관되게 높은 성능을 보이며, 하이퍼파라미터에 덜 민감하다. 특히 Bottle(100%), Zipper(99.8%) 등 객체 카테고리에서 거의 완벽한 성능을 보인다. 약점은 single-class 제한으로, 클래스당 별도 모델이 필요하며, 확장 시 메모리와 관리 복잡도가 선형 증가한다는 점이다. 적용 분야는 최고 정확도가 필수인 반도체 웨이퍼 검사, 의료 기기, 항공 부품 등이다.

두 번째는 Dinomaly로 multi-class에서 98.8%, single-class에서 99.2%의 AUROC를 보인다. Pixel AUROC는 97.5%이다. 추론 속도는 80-120ms이고, 메모리는 300-500MB이다(전체 클래스 통합). Dinomaly의 혁신은 단일 모델로 모든 클래스를 처리한다는 점이다. 15개 제품 검사 시 전통적 방법은 15개 모델(7.5GB)이 필요하지만, Dinomaly는 1개 모델(500MB)로 처리하여 메모리를 93% 절감한다. DINOv2의 강력한 self-supervised 특징 덕분에 single-class로 사용하면 PatchCore를 초과하는 99.2%를 달성한다. 배포 시간도 15시간에서 3시간으로 80% 단축된다. 약점은 모델 크기가 다소 크고(1.5-2GB), 학습 시 모든 클래스 데이터가 필요하다는 점이다. 적용 분야는 여러 제품을 동시에 검사하는 multi-class 환경, 메모리 제약이 있는 환경, 모델 관리를 간소화하고 싶은 경우이다.

세 번째는 Reverse Distillation으로 98.6%의 image AUROC를 달성한다. Pixel AUROC는 98.5%로 현재까지 최고 수준이다. 추론 속도는 100-200ms로 느리고, 메모리는 500MB-1GB이다. Reverse Distillation의 강점은 pixel-level localization 성능이다. 98.5%의 pixel AUROC는 결함의 정확한 위치를 파악하는 능력이 뛰어남을 의미한다. 패러다임 역전(teacher 단순화, student 복잡화)으로 타겟 도메인 특화 표현을 학습하여, ImageNet의 일반적 특징에 의존하는 STFPM보다 1.8%p 향상되었다. 특히 Metal Nut(99.5%), Pill(99.2%) 등 미세한 결함 탐지가 중요한 카테고리에서 우수하다. 약점은 복잡한 encoder-decoder 구조로 인한 느린 속도와 높은 학습 복잡도이다. 적용 분야는 결함의 정확한 위치가 중요한 반도체, 의료 영상, 정밀 기계 부품 검사 등이다.

네 번째는 FastFlow로 98.5%의 image AUROC를 보인다. Pixel AUROC는 97.8%이고, 추론 속도는 20-50ms로 빠르다. 메모리는 500MB-1GB이다. FastFlow의 강점은 속도와 정확도의 균형이다. 98.5%의 높은 정확도를 유지하면서 20-50ms의 빠른 속도를 달성하여, 실무에서 가장 많이 선택되는 모델 중 하나이다. 2D normalizing flow로 단순화하여 3D flow 대비 속도는 3배 빨라지고 성능은 오히려 향상되었다. 확률적 해석이 가능하여 log-likelihood 기반 의사결정을 할 수 있다. Carpet(99.2%), Leather(99.1%) 등 텍스처 카테고리에서 특히 강하다. 약점은 flow network 설계와 하이퍼파라미터 튜닝이 다소 복잡하다는 점이다. 적용 분야는 속도와 정확도가 모두 중요한 일반적인 품질 검사, 준실시간 모니터링 등이다.

다섯 번째는 UniNet으로 98.3%의 image AUROC를 달성한다. Pixel AUROC는 97.0%이고, 추론 속도는 50-80ms이다. UniNet의 강점은 contrastive learning 기반의 강건한 decision boundary이다. 정상과 이상을 embedding space에서 명확히 분리하여, 경계 사례에서도 안정적인 성능을 보인다. 2025년 최신 모델로 향후 발전 가능성이 높다. 약점은 상대적으로 낮은 성능과 제한된 검증 기간이다. 적용 분야는 정상과 이상의 경계가 모호한 환경, 강건성이 중요한 경우 등이다.

상위 5개 모델의 특징을 종합하면, PatchCore는 정확도, Dinomaly는 multi-class 효율성, Reverse Distillation은 localization, FastFlow는 균형, UniNet은 강건성에서 각각 강점을 가진다. 실무에서는 요구사항에 따라 적절한 모델을 선택해야 하며, 단순히 AUROC 수치만으로 판단하지 말고 추론 속도, 메모리, 관리 복잡도, 적용 환경 등을 종합적으로 고려해야 한다.

### 4.3 Speed-Accuracy-Memory Trade-offs

이상 탐지 모델 선택에서 가장 중요한 것은 정확도(Accuracy), 추론 속도(Speed), 메모리 사용량(Memory)의 trade-off를 이해하는 것이다. 이 세 가지 요소는 삼각 관계를 형성하며, 모두를 동시에 최고로 만족하는 모델은 존재하지 않는다. 이는 계산 복잡도 이론과 하드웨어 제약에서 비롯되는 근본적인 한계이다.

정확도와 속도의 trade-off를 먼저 살펴보자. 일반적으로 높은 정확도를 달성하려면 복잡한 모델이 필요하고, 이는 더 많은 계산을 요구한다. PatchCore는 99.1%의 정확도를 위해 k-NN search와 patch matching을 수행하여 50-100ms가 소요된다. Reverse Distillation은 98.6%를 위해 encoder-decoder 재구성을 수행하여 100-200ms가 걸린다. 반면 EfficientAD는 경량 PDN(50K 파라미터)과 간단한 autoencoder로 97.8%를 달성하면서 1-5ms만 소요한다. 1.3%p의 정확도 차이로 20-200배의 속도 향상을 얻는다. 이는 정밀 검사 vs 실시간 처리라는 명확한 선택을 요구한다.

정확도와 메모리의 trade-off도 중요하다. 높은 정확도는 보통 더 많은 정보를 저장하거나 더 큰 모델을 필요로 한다. PaDiM은 96.5%를 위해 모든 패치 위치의 공분산 행렬을 저장하여 2-5GB를 사용한다. PatchCore는 99.1%를 유지하면서 coreset selection으로 100-500MB로 줄였다. EfficientAD는 97.8%로 정확도를 약간 희생하고 PDN만 저장하여 200MB 미만을 사용한다. Dinomaly는 multi-class 98.8%를 위해 300-500MB를 사용하지만, 15개 클래스를 단일 모델로 처리하므로 전통적 방법(7.5GB) 대비 93% 절감한다.

속도와 메모리의 trade-off는 덜 명확하다. 빠른 모델이 항상 메모리를 적게 사용하지는 않는다. EfficientAD는 1-5ms의 빠른 속도와 200MB 미만의 작은 메모리를 모두 달성했다. 이는 극한 최적화의 결과이다. 반면 FastFlow는 20-50ms의 중간 속도를 가지지만 500MB-1GB의 큰 메모리를 사용한다. Flow network가 메모리를 많이 차지하기 때문이다. 일반적으로 속도와 메모리는 독립적이지만, 극한 최적화를 추구하면 둘 다 개선할 수 있다.

현실적인 선택지는 다음과 같다. 첫째, 정확도와 메모리를 우선하고 속도를 희생하는 경우이다. PatchCore(99.1%, 100-500MB, 50-100ms)가 대표적이다. 반도체, 의료, 항공 등 최고 정확도가 필수인 분야에 적합하다. 50-100ms는 느리지만 해당 분야에서는 허용 가능하다. 둘째, 속도와 메모리를 우선하고 정확도를 약간 희생하는 경우이다. EfficientAD(97.8%, <200MB, 1-5ms)가 해당한다. 고속 생산 라인, 엣지 디바이스, CPU 환경에 적합하다. 97.8%도 실용적으로 충분히 높은 정확도이다. 셋째, 정확도와 속도를 우선하고 메모리를 희생하는 경우이다. FastFlow(98.5%, 20-50ms, 500MB-1GB)가 대표적이다. GPU 서버에서 실행하며 메모리 제약이 크지 않은 환경에 적합하다. 넷째, 세 가지의 균형점을 찾는 경우이다. Dinomaly(98.8%, 80-120ms, 300-500MB)가 해당한다. 특히 multi-class 환경에서 메모리를 93% 절감하므로 실질적으로 가장 효율적이다.

불가능한 조합도 명확하다. 99%+ 정확도, 10ms 미만 속도, 100MB 미만 메모리를 동시에 달성하는 모델은 현재 존재하지 않으며, 이론적으로도 어렵다. 높은 정확도는 복잡한 계산(느린 속도) 또는 많은 정보 저장(큰 메모리)을 필요로 하기 때문이다. 따라서 실무에서는 요구사항에 따라 적절한 trade-off를 선택해야 한다.

의사결정 가이드는 다음과 같다. 불량품 유출 비용이 매우 높다면(반도체, 의료) 정확도를 최우선하고 PatchCore나 Dinomaly를 선택한다. 고속 생산 라인에서 전수 검사가 필요하다면 속도를 최우선하고 EfficientAD를 선택한다. 여러 제품을 동시에 검사한다면 Dinomaly의 multi-class 효율성이 결정적이다. 일반적인 경우에는 FastFlow의 균형잡힌 성능이 적합하다. 중요한 것은 벤치마크 수치만 보지 말고, 실제 운영 환경의 제약 조건(하드웨어, 비용, 인력)을 종합적으로 고려하는 것이다.

---

## 5. Future Directions

### 5.1 Short-term Outlook (2025-2026)

향후 1-2년은 현재 기술의 성숙화와 산업 보편화가 진행될 것으로 전망된다.

첫째, multi-class 모델의 표준화가 이루어질 것이다. Dinomaly의 성공으로 단일 모델로 여러 제품을 검사하는 것이 산업 표준이 될 것이다. 현재는 초기 채택 단계이지만, 2026년에는 대부분의 신규 프로젝트가 multi-class 모델을 우선 고려할 것이다. 메모리 80-90% 절감과 관리 간소화의 경제적 가치가 명확히 입증될 것이다. 새로운 foundation model도 등장하여 multi-class 정확도가 99% 이상으로 향상될 것으로 예상된다. 이는 "한 번에 모든 문제를 해결"한다는 성배에 더 가까워지는 것을 의미한다.

둘째, 실시간 처리의 확산이다. EfficientAD의 1-5ms 성능으로 실시간 처리가 현실화되면서, 고속 생산 라인(초당 200+ 제품)에서 전수 검사가 보편화될 것이다. 엣지 디바이스(Jetson, Raspberry Pi)에서도 이상 탐지가 가능해져, 모바일 품질 검사나 드론 기반 검사 등 새로운 응용 분야가 열릴 것이다. 기술적으로는 더 경량화된 모델(100MB 미만), 양자화 INT4 지원, NPU 최적화 등이 진행될 것이다. 2026년에는 1ms 미만의 초고속 모델도 등장할 것으로 예상된다.

셋째, zero-shot 성능 향상이다. Foundation model의 발전으로 zero-shot 정확도가 95% 이상으로 향상될 것이다. 현재 WinCLIP은 91-95%, VLM-AD는 96-97% 수준이지만, 개선된 CLIP이나 산업 특화 foundation model이 등장하면 97-99%도 가능할 것이다. 이는 신제품 출시 즉시 검사, 다품종 소량 생산 등의 시나리오에서 학습 데이터 수집 없이도 높은 정확도를 달성할 수 있음을 의미한다. 프롬프트 엔지니어링 기법도 발전하여 zero-shot 성능을 더욱 끌어올릴 것이다.

### 5.2 Mid-term Outlook (2026-2028)

향후 2-4년은 근본적인 기술 혁신과 산업 구조 변화가 예상된다.

첫째, domain-specific foundation model의 등장이다. Manufacturing CLIP은 수천만 장의 산업 이미지로 학습되어 텍스처, 재질, 결함을 이해할 것이다. ImageNet의 일반적 특징이 아니라 산업 도메인에 특화된 표현을 학습하여 zero-shot 정확도를 98% 이상으로 끌어올릴 것이다. Industrial DINOv2는 반도체, 전자, 기계 특화 모델로, multi-class 정확도를 99% 이상으로 향상시킬 것이다. 이러한 모델은 API 서비스($0.001-0.01/img), on-premise 라이선스, industry consortium 공동 개발 등 다양한 비즈니스 모델로 제공될 것이다.

둘째, explainable AI의 필수화이다. 규제 강화로 설명 가능성이 필수 요구사항이 될 것이다. EU AI Act는 고위험 AI에 대한 설명 의무를 부과하고, FDA는 의료 기기 AI의 투명성을 요구하며, 자동차 산업은 ISO 26262 AI 안전 표준을 도입할 것이다. 기술적으로는 VLM 기반 설명이 표준화되고, attention map 시각화가 개선되며, 근본 원인 자동 분석이 가능해질 것이다. 산업 적용 측면에서는 모든 검사에 설명이 첨부되고, 품질 보고서가 자동 생성되며, 감사 추적(audit trail)이 자동화될 것이다.

셋째, multi-modal fusion의 확산이다. 이미지만이 아니라 온도, 진동, 음향, 3D 스캔 등 다양한 센서 데이터를 통합하여 이상을 탐지할 것이다. 이는 표면에 나타나지 않는 숨겨진 결함(내부 균열, 전기적 이상 등)을 탐지할 수 있게 하여 정확도를 99.5% 이상으로 향상시킬 것이다. 예측 정비(predictive maintenance)와도 연계되어, 결함 발생 전에 미리 감지하고 대응할 수 있게 될 것이다.

### 5.3 Long-term Vision (2028-2030)

향후 4-6년은 이상 탐지의 근본적인 패러다임 전환이 예상된다.

첫째, continual learning의 실현이다. 현재 모델은 고정되어 있어 새로운 패턴을 학습하지 못하고 주기적으로 재학습해야 한다. 미래에는 실시간으로 학습하면서도 catastrophic forgetting(이전 지식 망각)을 해결한 모델이 등장할 것이다. 무중단 업데이트가 가능해져, 계절적 변화에 자동으로 적응하고, 새로운 결함 유형을 즉시 학습하며, 공정 변경을 자동으로 반영할 것이다. 이는 모델을 살아있는 시스템(living system)으로 만들어, 지속적으로 진화하고 개선되도록 할 것이다.

둘째, self-supervised learning의 대규모 활용이다. 현재는 정상 샘플에 레이블이 필요하지만, 미래에는 생산 라인의 모든 이미지를 레이블 없이 자동으로 학습할 것이다. 수백만 장의 데이터에서 self-supervised로 학습하여 더 강력한 표현을 획득하고, 희귀 패턴도 자동으로 포착할 것이다. 데이터 수집 비용이 제로가 되어, 데이터가 많을수록 성능이 자동으로 향상되는 선순환 구조가 만들어질 것이다.

셋째, edge AI와 federated learning의 확산이다. 각 공장에서 로컬로 학습하고, 모델만 중앙 서버로 전송하여 데이터 프라이버시를 보호할 것이다. Jetson, TPU 등 엣지 디바이스에서 학습까지 가능해져, 클라우드에 의존하지 않고 지연을 최소화할 것이다. 네트워크 비용을 절감하고, 빠른 응답을 제공하며, 데이터 주권(data sovereignty) 문제를 해결할 것이다.

### 5.4 Zero-Defect Manufacturing

궁극적 목표는 zero-defect manufacturing, 즉 불량 제로 생산이다. 이는 단순히 불량품을 검출하는 수준을 넘어, 불량이 발생하지 않도록 하는 것을 의미한다.

기술 로드맵은 세 단계로 구분된다. 첫 단계(2025-2026)는 탐지 완성이다. 99.5% 이상의 정확도를 달성하고, 실시간 전수 검사를 가능하게 하며, multi-class 모델을 표준화한다. 이 단계에서는 "모든 불량을 찾는다"가 목표이다. 두 번째 단계(2026-2028)는 예측으로의 전환이다. 결함이 발생하기 전에 미리 예측하고, 공정을 자동으로 조정하며, closed-loop 품질 시스템을 구축한다. 이 단계에서는 "불량이 발생하기 전에 막는다"가 목표이다. 세 번째 단계(2028-2030)는 zero-defect 달성이다. AI 기반으로 전체 생산 과정을 최적화하고, self-healing 생산 시스템을 구축하며, 완전 자율 품질 관리를 실현한다. 이 단계에서는 "불량이 애초에 발생하지 않는다"가 목표이다.

비즈니스 임팩트는 막대할 것이다. 불량률이 현재 1%에서 0.01%로 감소하여 생산 손실이 99% 줄어든다. 검사 비용이 현재의 10%로 감소하여 운영 효율이 크게 향상된다. 생산성이 2배 향상되어 같은 시간에 더 많은 제품을 생산할 수 있다. 이는 제조업의 패러다임을 근본적으로 전환하여, AI 기반의 완전 자율 공장(lights-out factory)을 현실화할 것이다.

---

## 6. Reading Guide

### 6.1 For Beginners

이상 탐지를 처음 접하는 독자를 위한 추천 경로이다.

첫 번째 단계는 전체 지형도 파악이다. 본 문서(00-overview.md)를 정독하여 6개 패러다임의 전체 구조와 발전 과정을 이해한다. 각 패러다임의 핵심 원리, 대표 모델, 장단점을 파악한다. 시간이 부족하다면 2장(Six Major Paradigms)과 3장(Key Technical Transitions)에 집중한다.

두 번째 단계는 의사결정 프레임워크 학습이다. [Comprehensive Comparison](07-comparison.md) 문서의 6.1-6.4절(시나리오별 최적 모델 선택, 하드웨어 환경별 선택, 개발 단계별 로드맵, 의사결정 플로우차트)을 읽는다. 본인의 프로젝트가 어떤 시나리오에 해당하는지 파악하고, 적합한 모델을 선택하는 기준을 이해한다.

세 번째 단계는 빠른 시작이다. [Feature Adaptation](05-feature-adaptation.md) 문서의 DFM 섹션을 읽고, 15분 만에 94-95%의 baseline을 구축하는 방법을 학습한다. PixelVision [Getting Started](../docs/01-getting-started.md)와 [Training Guide](../docs/05-training.md)를 참조하여 실제로 DFM을 실행해본다. 이를 통해 이상 탐지가 본인의 데이터에서 작동하는지 빠르게 검증한다.

네 번째 단계는 본격 학습이다. Feasibility가 확인되면 [Memory-Based Methods](01-memory-based.md) 문서를 읽고 PatchCore를 학습한다. 99.1%의 최고 정확도를 달성하는 방법을 이해한다. 필요에 따라 [Normalizing Flow](02-normalizing-flow.md)의 FastFlow나 [Knowledge Distillation](03-knowledge-distillation.md)의 EfficientAD도 학습한다.

학습 목표는 다음과 같다. 6개 패러다임의 차이를 설명할 수 있다. 본인의 프로젝트에 적합한 모델을 선택할 수 있다. DFM으로 빠른 prototype을 구축할 수 있다. PatchCore나 FastFlow로 SOTA급 모델을 학습할 수 있다. 이 과정은 2-4주 정도 소요되며, 이후에는 실무 프로젝트를 시작할 수 있는 충분한 지식을 갖추게 된다.

### 6.2 For Researchers

이상 탐지 연구자를 위한 추천 경로이다.

첫 번째 단계는 연구 동향 파악이다. 본 문서(00-overview.md)의 2.2절(Evolution Timeline)과 3장(Key Technical Transitions)을 정독하여 2018-2025년의 발전 과정을 이해한다. 각 시기의 주요 breakthrough와 실패 사례를 분석한다. FRE의 실패에서 "점진적 개선의 한계"를, FastFlow의 성공에서 "단순화의 힘"을 배운다.

두 번째 단계는 기술적 심층 분석이다. 관심 있는 패러다임의 상세 문서를 정독한다. [Memory-Based Methods](01-memory-based.md)에서는 Coreset selection 알고리즘과 ε-cover 이론을 학습한다. [Normalizing Flow](02-normalizing-flow.md)에서는 change of variables 공식과 flow 수학을 이해한다. [Knowledge Distillation](03-knowledge-distillation.md)에서는 Reverse Distillation의 패러다임 역전과 EfficientAD의 극한 최적화를 분석한다. [Foundation Models](06-foundation-models.md)에서는 최신 트렌드인 multi-class, zero-shot, explainable AI를 연구한다.

세 번째 단계는 벤치마크 분석이다. [Comprehensive Comparison](07-comparison.md) 문서의 4장(성능 비교 및 벤치마크 분석)을 정독하여 MVTec AD에서의 성능을 정량적으로 비교한다. 각 모델의 카테고리별 강점과 약점을 파악한다. 벤치마크의 한계(domain gap, false positive 비용 등)도 이해한다.

네 번째 단계는 미래 연구 방향 탐색이다. 본 문서의 5장(Future Directions)과 각 패러다임 문서의 "Open Research Questions" 섹션을 읽는다. Few-shot에서 one-shot으로, 3D 이상 탐지, uncertainty estimation, causal inference 등 미해결 문제를 파악한다. 본인의 연구 아이디어가 어떤 gap을 채울 수 있는지 고민한다.

학습 목표는 다음과 같다. 각 패러다임의 수학적 기반을 이해한다. 주요 기술적 전환점과 그 이유를 설명할 수 있다. SOTA 모델의 한계와 개선 방향을 제시할 수 있다. 새로운 연구 아이디어를 도출할 수 있다. 이 과정을 통해 이상 탐지 분야의 전문가로 성장할 수 있다.

### 6.3 For Practitioners

실무 적용을 목표로 하는 엔지니어나 관리자를 위한 추천 경로이다.

첫 번째 단계는 의사결정 가이드이다. [Comprehensive Comparison](07-comparison.md) 문서를 최우선으로 읽는다. 6.1-6.4절의 시나리오별 선택, 하드웨어별 선택, 개발 로드맵, 의사결정 플로우차트가 가장 중요하다. 본인의 요구사항(정확도? 속도? 비용?)을 명확히 하고, 플로우차트를 따라 적합한 모델을 선택한다.

두 번째 단계는 옵션 이해이다. 본 문서(00-overview.md)의 2.3절(Paradigm Comparison Matrix)을 읽고, 선택한 모델이 속한 패러다임의 특성을 이해한다. 해당 패러다임 문서의 "Practical Application Guide" 섹션을 읽고, 구체적인 적용 방법을 학습한다.

세 번째 단계는 실제 구현이다. PixelVision [Training Guide](../docs/05-training.md)와 [Inference Guide](../docs/06-inference.md)를 참조하여 선택한 모델을 실제로 학습하고 배포한다. [Models Reference](../docs/03-models.md)에서 하이퍼파라미터 설정을 확인한다. 벤치마크 데이터로 먼저 검증한 후, 실제 데이터로 확장한다.

네 번째 단계는 ROI 분석이다. [Comprehensive Comparison](07-comparison.md)의 6.5절(비용-효과 분석)을 읽고, 초기 개발 비용, 운영 비용, 기대 효과를 계산한다. 불량품 검출 가치, 인력 절감, 장기 효과를 정량화하여 경영진을 설득한다.

학습 목표는 다음과 같다. 빠른 모델 선택(1일 이내). 실무 배포 가능(2-4주 이내). ROI 분석으로 의사결정 지원. 지속적인 성능 모니터링과 개선. 이 과정을 통해 이상 탐지를 성공적으로 산업 현장에 적용할 수 있다.

---

## References

### Research Documents

본 overview에서 다룬 각 패러다임의 상세 분석은 다음 문서에서 제공된다.

- [01-memory-based.md](01-memory-based.md) - Memory-Based and Feature Matching Methods
- [02-normalizing-flow.md](02-normalizing-flow.md) - Normalizing Flow Approaches
- [03-knowledge-distillation.md](03-knowledge-distillation.md) - Knowledge Distillation Methods
- [04-reconstruction-based.md](04-reconstruction-based.md) - Reconstruction-Based Approaches
- [05-feature-adaptation.md](05-feature-adaptation.md) - Feature Adaptation and Transfer Learning
- [06-foundation-models.md](06-foundation-models.md) - Foundation Model-Based Methods
- [07-comparison.md](07-comparison.md) - Comprehensive Comparison and Application Guide

### User Documentation

PixelVision 프레임워크의 사용 가이드는 다음 문서에서 제공된다.

- [Getting Started](../docs/01-getting-started.md) - Installation and Configuration
- [Architecture](../docs/02-architecture.md) - Framework Architecture
- [Models Reference](../docs/03-models.md) - Detailed Model Documentation
- [Datasets Guide](../docs/04-datasets.md) - Dataset Preparation
- [Training Guide](../docs/05-training.md) - Training Configuration and Tips
- [Inference Guide](../docs/06-inference.md) - Deployment and Inference

### Key Papers by Paradigm

각 패러다임의 대표 논문은 다음과 같다.

**Memory-Based:**
- PaDiM: Defard et al., "PaDiM: Patch Distribution Modeling Framework for Anomaly Detection and Localization", ICPR 2020
- PatchCore: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022

**Normalizing Flow:**
- CFlow: Gudovskiy et al., "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows", WACV 2022
- FastFlow: Yu et al., "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows", 2021

**Knowledge Distillation:**
- STFPM: Wang et al., "Student-Teacher Feature Pyramid Matching for Anomaly Detection", BMVC 2021
- Reverse Distillation: Deng and Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding", CVPR 2022
- EfficientAD: Batzner et al., "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies", WACV 2024

**Reconstruction:**
- GANomaly: Akcay et al., "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training", ACCV 2018
- DRAEM: Zavrtanik et al., "DRAEM: Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection", ICCV 2021

**Feature Adaptation:**
- DFM: Defard et al., "Deep Feature Kernel Density Estimation", 2019
- CFA: Lee et al., "CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization", 2022

**Foundation Models:**
- WinCLIP: Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation", CVPR 2023
- Dinomaly: Zhang et al., "Dinomaly: Multi-class Anomaly Detection via Self-Supervised Learning", 2025

### External Resources

- [Anomalib GitHub](https://github.com/openvinotoolkit/anomalib) - Original library repository
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) - Benchmark dataset
- [Papers with Code - Anomaly Detection](https://paperswithcode.com/task/anomaly-detection) - Latest papers and benchmarks

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Total Models Analyzed**: 21  
**Paradigms Covered**: 6  
**Maintainer**: PixelVision Research Team
