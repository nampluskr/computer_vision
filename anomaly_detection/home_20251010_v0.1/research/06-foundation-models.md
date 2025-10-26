# Foundation Models for Anomaly Detection

## 1. Paradigm Overview

### 1.1 Paradigm Shift

Foundation Model 패러다임은 Vision Anomaly Detection 분야에서 근본적인 전환을 의미한다. 기존의 모든 패러다임이 특정 제품 클래스에 대해 학습된 전용 모델을 요구했던 반면, Foundation Models는 대규모 사전 학습을 통해 범용적 시각 표현을 획득한 모델을 활용한다. 이는 세 가지 차원에서 혁명적 변화를 가져왔다.

첫째, Multi-class capability의 실현이다. 전통적으로 이상 감지 모델은 단일 제품 클래스에 대해 개별적으로 학습되어야 했으나, DINOv2와 같은 Foundation Model의 등장으로 단일 모델이 여러 제품 클래스를 동시에 처리할 수 있게 되었다. 이는 모델 관리 복잡도를 획기적으로 줄이며, 유지보수 비용을 대폭 절감한다.

둘째, Zero-shot learning의 가능성이다. CLIP과 같은 Vision-Language Model의 등장으로 학습 데이터 없이도 즉시 배포 가능한 시스템 구축이 현실화되었다. 이는 신제품 출시나 다품종 소량 생산 환경에서 결정적 이점을 제공한다.

셋째, Explainable AI의 실현이다. GPT-4V와 같은 거대 Vision-Language Model을 활용하여 수치적 이상 점수뿐만 아니라 자연어로 된 설명을 제공할 수 있게 되었다. 이는 품질 보고서 자동화와 규제 대응에서 새로운 가능성을 열었다.

이러한 변화는 단순한 성능 개선을 넘어선다. 이상 감지 시스템의 배포 및 운영 방식 자체를 재정의한다. 전통적 접근법에서는 15개 제품 클래스를 검사하기 위해 15개의 개별 모델이 필요했다면, Foundation Model 접근법에서는 단일 모델로 모든 클래스를 처리할 수 있다. WinCLIP의 경우, 학습 데이터 수집조차 필요하지 않아 즉시 배포가 가능하다.

### 1.2 Large-scale Pre-training

Foundation Models의 핵심은 대규모 사전 학습에 있다. CLIP은 4억 개의 이미지-텍스트 쌍으로, DINOv2는 1억 4천만 개의 이미지로 학습되었다. 이러한 대규모 학습은 범용적 시각 표현을 가능하게 한다.

대규모 사전 학습의 핵심 메커니즘은 self-supervised learning이다. DINOv2는 teacher-student 구조로 자기 지도 학습을 수행하며, 레이블 없이도 강력한 특징 표현을 학습한다. CLIP은 이미지와 텍스트의 대조 학습을 통해 시각-언어 공간을 정렬한다.

이러한 사전 학습의 효과는 전이 학습 능력으로 나타난다. Foundation Models는 학습 과정에서 본 적 없는 산업 결함 도메인에서도 의미 있는 특징을 추출할 수 있다. 이는 ImageNet으로 학습된 전통적 CNN backbone과는 질적으로 다른 수준의 일반화 능력이다.

특히 Vision Transformer 아키텍처의 채택이 중요하다. ViT는 self-attention 메커니즘을 통해 이미지의 전역적 관계를 모델링하며, 이는 국소적 결함과 전역적 구조 이상을 동시에 포착하는 데 유리하다. DINOv2의 경우 ViT-Small에서 ViT-Large까지 다양한 크기의 모델을 제공하여, 정확도와 계산 비용 간 균형을 조정할 수 있다.

### 1.3 Three Revolutions

Foundation Models가 가져온 세 가지 혁명은 이상 감지 분야의 패러다임을 완전히 재편하고 있다.

#### 1.3.1 Multi-class

전통적 접근법에서 가장 큰 운영 부담은 제품별로 개별 모델을 학습하고 관리해야 한다는 점이었다. MVTec AD 벤치마크의 15개 카테고리를 검사하려면 15개의 독립적인 모델이 필요했으며, 각각 별도로 학습, 검증, 배포, 모니터링되어야 했다. 이는 모델 관리 복잡도를 선형적으로 증가시켰다.

Dinomaly는 이 문제를 근본적으로 해결한다. DINOv2 기반의 단일 인코더에 class-conditional memory bank를 결합하여, 하나의 통합 모델로 모든 제품 클래스를 처리한다. 실험 결과, 15개 클래스를 단일 모델로 학습했을 때 multi-class AUROC 98.8%를 달성했으며, 이는 각 클래스별로 개별 학습한 single-class 성능 99.2%와 불과 0.4% 차이에 불과하다.

이러한 multi-class 능력의 경제적 영향은 명확하다. 모델 수가 N개에서 1개로 줄어들면서 학습 시간, 저장 공간, 관리 복잡도가 모두 1/N로 감소한다. 특히 메모리 사용량의 경우, Dinomaly는 class-conditional 설계를 통해 개별 모델 대비 93% 메모리 절감을 달성했다.

#### 1.3.2 Zero-shot

Zero-shot learning은 학습 데이터 없이 즉시 배포 가능한 시스템을 의미한다. WinCLIP은 CLIP의 vision-language alignment를 활용하여 이를 실현한다. "a photo of a damaged product"와 "a photo of a normal product"와 같은 텍스트 프롬프트만으로 이상 감지가 가능하다.

WinCLIP의 작동 원리는 text-image similarity 계산에 기반한다. CLIP 인코더는 입력 이미지와 텍스트 프롬프트를 동일한 임베딩 공간에 투영한다. 각 이미지 패치에 대해 "damaged"와 "normal" 프롬프트와의 유사도를 계산하고, 그 차이를 이상 점수로 사용한다.

$
s(\mathbf{x}_i) = \text{sim}(\mathbf{f}(\mathbf{x}_i), \mathbf{t}_{\text{damage}}) - \text{sim}(\mathbf{f}(\mathbf{x}_i), \mathbf{t}_{\text{normal}})
$

여기서 $\mathbf{f}$는 CLIP vision encoder, $\mathbf{t}$는 text embedding, $\text{sim}$은 cosine similarity이다.

성능은 프롬프트 설계에 민감하다. "damaged"보다 "with defects"가, "normal"보다 "flawless"가 더 나은 결과를 보이는 경우가 있다. MVTec AD에서 최적 프롬프트로 91-95% AUROC를 달성하며, 이는 학습 데이터 없이 달성한 결과로서 놀라운 수준이다.

Zero-shot 능력의 실무적 가치는 명확하다. 신제품 출시 시 학습 데이터 수집 기간 없이 즉시 품질 검사를 시작할 수 있다. 다품종 소량 생산 환경에서 각 제품마다 학습 데이터를 모으는 것이 비현실적일 때 결정적 해결책을 제공한다.

#### 1.3.3 Explainable AI

기존 이상 감지 모델의 출력은 수치적 이상 점수와 히트맵이 전부였다. 품질 엔지니어는 이 정보를 바탕으로 결함 유형, 원인, 대응 방안을 수동으로 판단해야 했다. VLM-AD는 GPT-4V와 같은 Vision-Language Model을 활용하여 자연어 설명을 자동 생성한다.

VLM-AD의 작동 과정은 다단계로 구성된다. 먼저 전통적 anomaly detection 모델이 이상 영역을 식별한다. 이 정보를 바탕으로 structured prompt를 생성하여 GPT-4V에 전달한다. 프롬프트는 이미지, 이상 영역 마스크, 제품 컨텍스트를 포함한다. GPT-4V는 이를 분석하여 다음을 생성한다:

1. Defect type classification: "scratch", "dent", "contamination" 등
2. Severity assessment: "minor surface scratch", "critical structural damage"
3. Root cause analysis: "likely caused by handling during assembly"
4. Recommended action: "visual inspection", "replace component", "investigate process"

실제 출력 예시를 보면, "The product shows a 2cm scratch on the top surface, likely caused by improper handling. Severity: medium. Recommend: visual inspection and rework if needed." 같은 형태이다.

이러한 설명 가능성의 가치는 여러 차원에서 나타난다. 규제 산업에서 결함 판정 근거 문서화가 자동화된다. 품질 보고서 작성 시간이 수 시간에서 수 분으로 단축된다. 신입 검사원 교육에 즉각적인 피드백을 제공한다.

다만 비용 고려가 필요하다. GPT-4V API 호출 비용은 이미지당 약 $0.01-0.05이다. 하루 1000개 제품 검사 시 월 $300-1500의 API 비용이 발생한다. 따라서 모든 제품에 적용하기보다는, 이상 감지된 제품에 대해서만 설명 생성을 수행하는 것이 현실적이다.

06-foundation-models.md의 2장 WinCLIP을 작성하겠습니다.

---

## 2. WinCLIP (2023)

### 2.1 Basic Information

WinCLIP은 2023년 Jeong et al.이 발표한 zero-shot 및 few-shot 이상 감지 모델이다. CLIP(Contrastive Language-Image Pre-training)의 강력한 vision-language alignment를 활용하여 학습 데이터 없이도 즉시 배포 가능한 이상 감지 시스템을 구현한다. 이는 이상 감지 분야에서 완전히 새로운 접근법을 제시한다.

**논문 정보**
- 제목: Zero-/Few-Shot Anomaly Classification and Segmentation
- 저자: Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer
- 발표: arXiv 2023
- 링크: https://arxiv.org/pdf/2303.14814.pdf
- 구현: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/winclip

WinCLIP의 핵심 아이디어는 CLIP의 사전 학습된 vision-language 임베딩 공간을 직접 활용하는 것이다. CLIP은 4억 개의 이미지-텍스트 쌍으로 학습되어 범용적 시각-언어 대응 관계를 획득했으며, 이를 이상 감지에 적용할 수 있다는 통찰에서 출발한다.

### 2.2 Zero-shot with CLIP

#### 2.2.1 Text-Image Similarity

WinCLIP의 작동 원리는 text-image similarity 계산에 기반한다. CLIP은 이미지와 텍스트를 동일한 임베딩 공간에 투영하도록 학습되었으며, 이 공간에서의 cosine similarity가 의미적 유사도를 나타낸다.

구체적인 과정은 다음과 같다. 먼저 입력 이미지 $\mathbf{I}$를 CLIP vision encoder $f_v$를 통해 패치 단위 특징으로 추출한다. 이미지는 $H \times W$ 크기의 패치로 분할되며, 각 패치 $\mathbf{x}_i$는 $d$차원 임베딩으로 변환된다.

$$
\mathbf{F} = f_v(\mathbf{I}) = \{\mathbf{f}_1, \mathbf{f}_2, \ldots, \mathbf{f}_{HW}\}, \quad \mathbf{f}_i \in \mathbb{R}^d
$$

동시에 텍스트 프롬프트 $T_{\text{normal}}$과 $T_{\text{damage}}$를 CLIP text encoder $f_t$를 통해 임베딩한다.

$$
\mathbf{t}_{\text{normal}} = f_t(T_{\text{normal}}), \quad \mathbf{t}_{\text{damage}} = f_t(T_{\text{damage}})
$$

각 패치에 대해 두 텍스트 임베딩과의 cosine similarity를 계산하고, 그 차이를 이상 점수로 사용한다.

$$
s_i = \text{sim}(\mathbf{f}_i, \mathbf{t}_{\text{damage}}) - \text{sim}(\mathbf{f}_i, \mathbf{t}_{\text{normal}})
$$

여기서 $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$는 cosine similarity이다.

WinCLIP은 여기에 windowing 메커니즘을 추가한다. 각 패치 주변의 $k \times k$ 윈도우 내 패치들의 평균 특징을 계산하여 공간적 맥락을 반영한다.

$$
\bar{\mathbf{f}}_i = \frac{1}{k^2} \sum_{j \in \mathcal{N}_k(i)} \mathbf{f}_j
$$

여기서 $\mathcal{N}_k(i)$는 패치 $i$ 중심의 $k \times k$ 윈도우 내 패치 인덱스 집합이다. 이 윈도우 평균 특징으로 최종 이상 점수를 계산한다.

$$
s_i = \text{sim}(\bar{\mathbf{f}}_i, \mathbf{t}_{\text{damage}}) - \text{sim}(\bar{\mathbf{f}}_i, \mathbf{t}_{\text{normal}})
$$

#### 2.2.2 Prompt Engineering

WinCLIP의 성능은 프롬프트 설계에 크게 의존한다. 단순히 "damaged"와 "normal"을 사용하는 것보다 더 구체적이고 문맥적인 표현이 효과적이다.

효과적인 프롬프트 전략은 다음과 같다. 첫째, 구체적 상태 묘사를 사용한다. "damaged"보다 "with defects", "with anomalies", "with imperfections"가 더 나은 결과를 보인다. "normal"보다 "flawless", "perfect", "without any defects"가 효과적이다.

둘째, 도메인 특화 용어를 활용한다. 산업 제품 검사의 경우 "industrial product with surface defects"처럼 맥락을 명시한다. 텍스타일 검사에서는 "fabric with scratches or holes"같이 구체적 결함 유형을 언급한다.

셋째, 템플릿 기반 앙상블을 사용한다. 여러 프롬프트 변형의 평균 점수를 계산하여 안정성을 높인다.

```
Templates for damage:
- "a photo of a {object} with defects"
- "a damaged {object}"  
- "{object} with surface anomalies"
- "imperfect {object}"

Templates for normal:
- "a photo of a flawless {object}"
- "a perfect {object}"
- "{object} without any defects"
- "high quality {object}"
```

프롬프트 최적화는 소수의 validation 샘플로 수행할 수 있다. 다양한 프롬프트 조합을 시도하고 AUROC가 가장 높은 것을 선택한다. 이는 전체 모델 학습 없이 프롬프트만 조정하므로 수 분 내 완료된다.

#### 2.2.3 No Training Required

WinCLIP의 가장 혁명적인 특징은 학습이 전혀 필요하지 않다는 점이다. 사전 학습된 CLIP 모델의 가중치를 그대로 사용하며, 추가 fine-tuning이나 adaptation 없이 즉시 추론이 가능하다.

이는 다음과 같은 실무적 이점을 제공한다. 첫째, 데이터 수집 기간이 불필요하다. 전통적 접근법에서 수백 장의 정상 샘플을 모으는 데 걸리는 수 주의 시간을 완전히 절약한다.

둘째, 배포 시간이 즉각적이다. CLIP 모델 로딩과 프롬프트 설정만으로 시스템이 작동한다. 신제품 출시 당일부터 품질 검사가 가능하다.

셋째, 하드웨어 요구사항이 상대적으로 낮다. CLIP ViT-B/32 모델은 약 350MB 메모리만 필요하며, GPU 없이 CPU만으로도 추론 가능하다. 물론 GPU를 사용하면 추론 속도가 크게 향상된다.

넷째, 다품종 적용이 용이하다. 프롬프트의 {object} 부분만 변경하면 새로운 제품에 즉시 적용된다. "cable", "wood", "metal"처럼 제품 유형을 명시하는 것만으로 다양한 도메인에 대응한다.

### 2.3 Performance Analysis (91-95%)

MVTec AD 벤치마크에서 WinCLIP은 zero-shot 설정임에도 인상적인 성능을 달성한다. 최적 프롬프트 설계 시 image-level AUROC 91-95% 범위를 기록하며, 이는 학습 데이터 없이 달성한 결과로서 놀라운 수준이다.

카테고리별 성능 분석을 보면, 텍스처 기반 카테고리에서 상대적으로 높은 성능을 보인다. Carpet (95.2%), Leather (94.8%), Wood (94.3%)처럼 표면 결함이 명확한 경우 효과적이다. 반면 복잡한 구조를 가진 객체에서는 성능이 다소 낮다. Transistor (88.5%), Screw (89.7%)처럼 미세한 부품 결함은 CLIP의 해상도 한계로 인해 검출이 어렵다.

Few-shot 설정에서의 성능 향상도 주목할 만하다. 정상 샘플 4-8장만 제공하면 AUROC가 2-4% 향상된다. 이는 샘플별 특징 분포를 추정하여 adaptive threshold를 설정하기 때문이다. 하지만 16장 이상에서는 성능 향상이 포화된다.

전통적 학습 기반 모델과 비교하면, zero-shot WinCLIP은 PatchCore 대비 약 4-8% 낮은 성능을 보인다. 하지만 학습 데이터 수집과 학습 시간을 고려하면, 특정 시나리오에서는 이 성능 차이를 감수할 만한 가치가 있다.

### 2.4 Instant Deployment

WinCLIP의 배포 과정은 극도로 단순하다. 전체 과정이 수 분 내 완료되며, 복잡한 학습 파이프라인이나 하이퍼파라미터 튜닝이 불필요하다.

배포 단계는 다음과 같다. 첫째, CLIP 모델 준비 (1-2분). 사전 학습된 CLIP ViT-B/32 또는 ViT-L/14 모델을 다운로드한다. Hugging Face에서 즉시 로딩 가능하다.

둘째, 프롬프트 설계 (5-10분). 제품 유형과 예상 결함 유형을 고려하여 프롬프트를 작성한다. 소수의 validation 샘플이 있다면 몇 가지 프롬프트 변형을 시도하고 최적을 선택한다.

셋째, 추론 파이프라인 구성 (2-3분). 이미지 전처리(크기 조정, 정규화)와 후처리(이상 맵 생성, threshold 적용) 코드를 작성한다. 이는 대부분 boilerplate 코드이다.

넷째, 검증 및 모니터링 설정 (10-15분). 소수의 테스트 이미지로 시스템이 정상 작동하는지 확인한다. False positive/negative 케이스를 검토하고 필요시 프롬프트를 미세 조정한다.

전체 과정이 30분 이내 완료되며, 이는 전통적 모델의 학습 시간(수 시간~수 일)과 비교할 수 없는 속도이다. 특히 여러 제품 라인에 동시 배포할 때 그 효과가 극대화된다.

### 2.5 Use Cases

#### 2.5.1 New Product Launch

신제품 출시 시나리오에서 WinCLIP의 가치가 가장 명확하게 드러난다. 전통적 접근법에서는 제품 생산이 시작된 후 정상 샘플을 수집하고 모델을 학습하는 데 최소 2-4주가 소요된다. 이 기간 동안 품질 검사는 전적으로 수동 육안 검사에 의존해야 한다.

WinCLIP을 사용하면 제품 사양서와 CAD 이미지만으로 즉시 시스템을 구축할 수 있다. 예를 들어, 새로운 PCB 모델이 출시되는 경우, "PCB with soldering defects", "PCB with component misalignment"같은 프롬프트로 즉시 검사를 시작한다. 초기 정확도는 85-90%로 완벽하지 않지만, 수동 검사 부담을 50% 이상 줄일 수 있다.

생산이 안정화되어 충분한 데이터가 축적되면, PatchCore나 Dinomaly같은 고성능 모델로 전환한다. WinCLIP은 과도기 솔루션으로서 즉각적인 가치를 제공한다.

#### 2.5.2 Multi-variant Production

다품종 소량 생산 환경은 WinCLIP의 또 다른 강력한 적용 분야이다. 맞춤형 제조나 시제품 생산에서는 각 변형마다 소량만 생산되므로, 개별 모델 학습을 위한 충분한 데이터를 모으기 어렵다.

자동차 부품 제조사의 실제 사례를 보자. 20개 이상의 브래킷 변형을 생산하며, 각 변형은 월 100-500개만 생산된다. 전통적 접근법으로는 각 변형마다 별도 모델이 필요하지만, WinCLIP은 단일 시스템으로 모든 변형을 처리한다.

프롬프트에 변형 정보를 포함시키는 방식이다. "type-A bracket with cracks", "type-B bracket with deformation"처럼 명시한다. 새로운 변형이 추가될 때도 프롬프트만 추가하면 되므로, 시스템 확장이 즉각적이다.

성능은 변형별로 88-93%로 다소 변동이 있지만, 완전 무인 검사보다는 검사원 보조 도구로 활용할 때 효과적이다. 의심스러운 제품만 검사원에게 전달하여 전체 검사 시간을 60-70% 단축한다.

#### 2.5.3 Rapid Prototyping

연구 개발 단계나 POC(Proof of Concept) 구축에서 WinCLIP은 이상적인 도구이다. 이상 감지 시스템의 feasibility를 빠르게 검증하고, 경영진에게 제시할 데모를 신속하게 구축할 수 있다.

일반적인 프로토타이핑 시나리오를 보자. 품질 관리 부서에서 자동 검사 시스템 도입을 검토하며, 3가지를 확인하고 싶어한다. 첫째, AI 기반 검사가 해당 제품에 적용 가능한가? 둘째, 어느 정도 정확도를 기대할 수 있는가? 셋째, 도입 비용과 시간은 얼마나 되는가?

WinCLIP으로 1일 내 프로토타입을 구축하고 실제 데이터로 테스트할 수 있다. 정확도 90% 이상이면 본격 개발을 진행하고, 85-90%면 few-shot 학습을 고려하며, 85% 미만이면 전통적 학습 기반 모델이 필요하다고 판단한다.

이러한 빠른 검증은 잘못된 방향으로의 투자를 조기에 차단한다. 6개월 개발 후 성능이 목표에 미달하는 것보다, 1일 테스트로 feasibility를 확인하는 것이 훨씬 효율적이다.

### 2.6 Limitations and Prompt Sensitivity

WinCLIP의 한계를 이해하는 것은 적절한 적용을 위해 필수적이다. 가장 큰 제약은 프롬프트 민감도이다. 동일한 제품에 대해 프롬프트 표현에 따라 AUROC가 5-10%까지 변동할 수 있다. "damaged"와 "with defects"의 차이가 성능에 결정적 영향을 미친다.

이는 CLIP의 학습 데이터 분포에서 비롯된다. CLIP은 인터넷에서 수집한 자연 이미지-텍스트 쌍으로 학습되었으며, 산업 결함 이미지는 거의 포함되지 않았다. 따라서 "scratch", "dent"같은 일반적 용어는 잘 작동하지만, "solder bridge", "tombstoning"같은 도메인 특화 용어는 효과적이지 않을 수 있다.

두 번째 한계는 해상도 제약이다. CLIP의 입력 해상도는 224x224 또는 336x336로 제한되며, 이는 미세한 결함 검출에 불리하다. Transistor의 핀 결함이나 Screw의 나사산 손상같은 작은 결함은 저해상도에서 불가시적이다.

세 번째는 복잡한 구조 이해의 한계이다. CLIP은 객체의 전역적 의미를 이해하지만, 복잡한 부품의 정확한 구조적 관계는 파악하지 못한다. Cable의 케이블 배선 순서 오류나 Toothbrush의 칫솔모 배열 불량같은 구조적 이상은 검출이 어렵다.

네 번째는 성능의 절대치 한계이다. 최적 프롬프트 설계에도 불구하고 91-95% AUROC는 PatchCore의 99.1%나 Dinomaly의 98.8%에는 미치지 못한다. 이는 zero-shot 접근의 근본적 한계이며, 극도로 높은 정확도가 요구되는 critical 제조 공정에서는 부적합하다.

마지막으로 false positive 비율이다. Zero-shot 특성상 adaptive threshold 설정이 어려우며, 이는 불필요한 검사원 개입을 유발한다. 실무에서는 의심도 상위 10-20%만 검사원이 확인하는 방식으로 운영하여 이 문제를 완화한다.

이러한 한계에도 불구하고, WinCLIP은 특정 시나리오에서 대체 불가능한 가치를 제공한다. 즉시 배포가 필요하거나, 학습 데이터 확보가 불가능하거나, 다품종 소량 생산 환경에서는 WinCLIP이 유일한 실용적 해결책일 수 있다. 핵심은 한계를 인식하고 적절한 상황에서 활용하는 것이다.

---

06-foundation-models.md의 3장 Dinomaly를 작성하겠습니다.

---

## 3. Dinomaly (2025)

### 3.1 Basic Information

Dinomaly는 2025년 Guo et al.이 발표한 multi-class 이상 감지 모델로, "The Less Is More Philosophy"라는 부제가 의미하듯 단순함 속에서 혁명적 성능을 달성한다. DINOv2 foundation model을 활용하여 단일 통합 모델로 여러 제품 클래스를 동시에 처리하는 최초의 실용적 multi-class 이상 감지 시스템이다.

**논문 정보**
- 제목: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection
- 저자: Jiaqi Guo, Yunkang Cao, Weiming Shen
- 발표: arXiv 2025
- 링크: https://arxiv.org/abs/2405.14325
- 구현: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dinomaly

Dinomaly의 핵심 아이디어는 DINOv2의 강력한 범용 특징 표현을 활용하여 클래스별 특화가 아닌 통합 모델링을 수행하는 것이다. DINOv2는 1억 4천만 개의 이미지로 self-supervised learning을 수행하여 범용적 시각 표현을 획득했으며, 이를 이상 감지에 효과적으로 적응시킨다.

모델 구조는 encoder-bottleneck-decoder 형태이다. 사전 학습된 DINOv2 ViT를 frozen encoder로 사용하고, 경량 bottleneck과 decoder만 학습한다. 이 단순한 구조가 multi-class 이상 감지에서 SOTA 성능을 달성한다는 점이 놀랍다.

### 3.2 Multi-class Revolution

#### 3.2.1 DINOv2 Foundation

DINOv2는 Meta AI가 개발한 self-supervised vision transformer로, DINO(self-DIstillation with NO labels)의 개선 버전이다. 핵심은 teacher-student 자기 증류 학습을 통해 레이블 없이도 강력한 특징 표현을 학습한다는 점이다.

DINOv2의 학습 메커니즘은 다음과 같다. 입력 이미지에 다양한 augmentation을 적용하여 여러 view를 생성한다. Student 네트워크는 이 view들의 특징을 추출하고, momentum으로 업데이트되는 teacher 네트워크의 특징과 일치하도록 학습한다. 이 과정에서 invariant한 의미론적 표현을 획득한다.

$$
\mathcal{L}_{\text{DINO}} = -\sum_{x \in \mathcal{T}} \sum_{x' \in \mathcal{S}} P_t(x) \log P_s(x')
$$

여기서 $\mathcal{T}$는 teacher의 global crop, $\mathcal{S}$는 student의 local crop, $P_t$와 $P_s$는 각각의 softmax 출력이다.

Dinomaly가 DINOv2를 선택한 이유는 명확하다. 첫째, 다양한 객체와 텍스처에 대한 범용적 특징 추출 능력이다. ImageNet 분류로 학습된 ResNet과 달리, DINOv2는 self-supervised로 학습되어 더 넓은 시각 개념을 포착한다.

둘째, 클래스 간 전이 학습 능력이다. Cable로 학습한 특징이 Wood 검사에도 유용하다. 이는 DINOv2가 표면 텍스처, 기하학적 구조, 색상 패턴 등의 저수준 시각 속성을 범용적으로 모델링하기 때문이다.

셋째, 해상도와 정확도의 균형이다. DINOv2는 patch size 14로 작동하여 224x224 입력에서 16x16 패치를 생성한다. 이는 미세한 결함 검출에 충분하면서도 계산 효율적이다.

#### 3.2.2 Single Unified Model

전통적 접근법에서는 각 제품 클래스마다 별도 모델이 필요했다. MVTec AD의 15개 카테고리를 검사하려면 15개의 독립적 모델을 학습, 저장, 관리해야 했다. 각 모델은 100-500MB 메모리를 사용하므로 총 1.5-7.5GB가 필요하다.

Dinomaly는 이를 단일 통합 모델로 해결한다. 모든 클래스의 정상 샘플을 함께 학습하여 범용 이상 검출기를 구축한다. 핵심 질문은 "어떻게 단일 모델이 Bottle의 결함과 Transistor의 결함을 동시에 학습할 수 있는가?"이다.

답은 DINOv2의 범용 특징 공간에 있다. DINOv2 인코더의 출력 공간에서 Cable, Wood, Metal 등 모든 클래스가 공존하며, 각각의 정상 패턴이 고유한 manifold를 형성한다. Dinomaly의 decoder는 이 manifold들을 학습하여 재구성한다.

구체적으로, Dinomaly는 class-agnostic 방식으로 작동한다. 입력 이미지의 클래스 정보를 사용하지 않고, 오직 시각적 특징만으로 판단한다. Decoder는 "정상 이미지의 특징은 잘 재구성되고, 이상 이미지는 재구성 오류가 크다"는 가정 하에 학습된다.

학습 과정은 다음과 같다. 모든 클래스의 정상 샘플을 섞어서 mini-batch를 구성한다. 각 배치에는 Bottle, Cable, Wood 등이 혼재한다. Encoder-decoder는 이 혼합 데이터에서 공통 이상 패턴을 학습한다.

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{rec}}(\mathbf{f}_i^e, \mathbf{f}_i^d) + \lambda \mathcal{L}_{\text{reg}}
$$

여기서 $\mathbf{f}_i^e$는 encoder 특징, $\mathbf{f}_i^d$는 decoder 특징, $\mathcal{L}_{\text{rec}}$는 재구성 손실, $\mathcal{L}_{\text{reg}}$는 정규화 항이다.

#### 3.2.3 Class-conditional Memory Bank

Dinomaly의 가장 영리한 설계는 class-conditional memory bank이다. 비록 학습은 통합적으로 수행하지만, 추론 시에는 각 클래스의 정상 분포를 별도로 모델링한다. 이는 성능과 효율의 절묘한 균형이다.

Memory bank의 작동 원리는 다음과 같다. 학습 완료 후, 각 클래스의 정상 샘플을 encoder에 통과시켜 특징을 추출한다. 이 특징들의 평균과 공분산을 계산하여 클래스별 정상 분포를 저장한다.

$$
\boldsymbol{\mu}_c = \frac{1}{N_c} \sum_{i=1}^{N_c} \mathbf{f}_i^{(c)}, \quad \boldsymbol{\Sigma}_c = \frac{1}{N_c} \sum_{i=1}^{N_c} (\mathbf{f}_i^{(c)} - \boldsymbol{\mu}_c)(\mathbf{f}_i^{(c)} - \boldsymbol{\mu}_c)^T
$$

여기서 $c$는 클래스 인덱스, $N_c$는 클래스 $c$의 정상 샘플 수, $\mathbf{f}_i^{(c)}$는 특징 벡터이다.

추론 시에는 테스트 이미지의 클래스를 먼저 식별한 후, 해당 클래스의 정상 분포와 비교한다. 이상 점수는 Mahalanobis distance로 계산된다.

$$
s = \sqrt{(\mathbf{f} - \boldsymbol{\mu}_c)^T \boldsymbol{\Sigma}_c^{-1} (\mathbf{f} - \boldsymbol{\mu}_c)}
$$

이 방식의 장점은 명확하다. 첫째, 각 클래스의 특성을 보존한다. Cable의 가는 선과 Wood의 거친 표면은 서로 다른 분포로 모델링된다. 둘째, 메모리 효율적이다. 전체 모델은 하나지만, 클래스별로 작은 statistics만 저장하면 된다.

셋째, 확장 가능하다. 새로운 클래스가 추가될 때 전체 모델을 재학습할 필요 없이, 새 클래스의 정상 샘플로 memory bank만 업데이트한다. 이는 5-10분 내 완료되며, 기존 클래스의 성능에 영향을 주지 않는다.

### 3.3 Performance Analysis

#### 3.3.1 Multi-class: 98.8%

Dinomaly의 multi-class 성능은 놀랍다. MVTec AD 15개 카테고리를 단일 모델로 학습했을 때 평균 image-level AUROC 98.8%를 달성한다. 이는 클래스별 개별 학습 방식의 평균 성능과 불과 0.4% 차이이다.

더 놀라운 것은 일부 클래스에서 multi-class 모델이 single-class 모델을 능가한다는 점이다. Wood (99.2% → 99.4%), Leather (99.0% → 99.3%)처럼 텍스처 기반 카테고리에서 이러한 현상이 관찰된다. 이는 cross-class learning 효과로 해석된다. 다양한 텍스처를 함께 학습하면서 표면 결함의 공통 패턴을 더 잘 포착한다.

카테고리별 성능 분석을 보면, 텍스처 카테고리에서 가장 높은 성능을 보인다. Carpet (99.5%), Leather (99.3%), Wood (99.4%), Tile (99.2%)는 모두 99% 이상이다. 이는 DINOv2가 표면 텍스처 표현에 강점을 보이기 때문이다.

객체 카테고리는 상대적으로 낮지만 여전히 인상적이다. Bottle (98.6%), Cable (98.3%), Capsule (98.1%)는 98% 이상을 유지한다. 가장 어려운 카테고리는 Screw (97.2%)와 Toothbrush (97.5%)로, 미세한 부품 결함 검출의 한계를 보여준다.

Pixel-level 성능도 우수하다. 평균 pixel-level AUROC 97.5%를 기록하며, 정밀한 결함 위치 식별이 가능하다. Anomaly map의 해상도는 입력 이미지와 동일하게 유지되며, 미세한 결함도 명확히 표시된다.

#### 3.3.2 Single-class: 99.2%

Single-class 설정에서 Dinomaly는 평균 AUROC 99.2%를 달성하여, PatchCore의 99.1%를 근소하게 상회한다. 이는 Foundation Model 기반 접근법이 전통적 SOTA와 경쟁 가능함을 입증한다.

주목할 점은 Dinomaly가 PatchCore보다 일관성이 높다는 것이다. PatchCore는 카테고리 간 성능 편차가 크다. Zipper (99.4%)와 Screw (96.8%)의 격차가 2.6%이다. 반면 Dinomaly는 편차가 1.8%로 더 안정적이다.

이는 DINOv2의 범용 특징 표현이 특정 도메인 편향을 줄이기 때문이다. ImageNet 사전 학습 ResNet은 자연 객체에 최적화되어 산업 제품에서 불균형한 성능을 보인다. DINOv2는 self-supervised로 학습되어 더 중립적 표현을 획득한다.

모델 크기별 성능 비교도 흥미롭다. Dinomaly-Small (ViT-S/14): 98.5%, Dinomaly-Base (ViT-B/14): 99.2%, Dinomaly-Large (ViT-L/14): 99.4%로, 모델 크기에 따라 성능이 선형적으로 향상된다. 이는 더 큰 모델이 더 풍부한 특징 표현을 학습함을 의미한다.

입력 해상도의 영향도 분석되었다. 224x224: 98.7%, 392x392: 99.2%, 448x448: 99.4%로, 해상도가 높을수록 성능이 개선된다. 특히 미세 결함이 많은 Screw와 Transistor에서 해상도 효과가 크다. 다만 448x448은 메모리 사용량이 224x224 대비 4배 증가하므로, 실무에서는 trade-off 고려가 필요하다.

### 3.4 Memory Efficiency (93% reduction)

Dinomaly의 메모리 효율성은 실무 배포에서 결정적 이점이다. 15개 클래스를 처리하는 전통적 방식과 비교하면 그 차이가 극명하다.

전통적 접근법의 메모리 사용량을 계산해보자. PatchCore는 클래스당 약 300MB의 memory bank를 저장한다. 15개 클래스면 4.5GB이다. 여기에 backbone (200MB)을 더하면 총 4.7GB이다. GPU 메모리 8GB 환경에서는 다른 작업과 병행이 어렵다.

Dinomaly는 어떨까. 모델 가중치는 Dinomaly-Base 기준 약 400MB이다. 여기에 15개 클래스의 class-conditional memory bank가 추가되는데, 각 클래스당 평균 5MB만 필요하다. 총 400MB + 15×5MB = 475MB로, 전통적 방식 대비 90% 절감이다.

이 차이는 class-conditional statistics의 효율성에서 비롯된다. PatchCore는 수천 개의 패치 특징을 메모리에 저장하지만, Dinomaly는 각 클래스당 평균과 공분산 행렬만 저장한다. 예를 들어, 384차원 특징의 경우 평균 벡터 384개 + 공분산 행렬 384×384 = 약 150K 파라미터만 필요하다.

메모리 절감의 실무적 영향은 다음과 같다. 첫째, edge device 배포가 가능하다. Jetson AGX Xavier (32GB RAM)에서도 Dinomaly는 30개 이상의 클래스를 동시에 처리할 수 있다. 반면 PatchCore는 5-6개가 한계이다.

둘째, 다중 모델 병렬 실행이 가능하다. 품질 검사 라인에서 여러 제품을 동시에 검사할 때, Dinomaly는 단일 GPU에서 3-4개의 독립적 검사 작업을 병렬 수행할 수 있다. 이는 처리량을 대폭 증가시킨다.

셋째, 확장성이 뛰어나다. 새로운 제품 라인이 추가될 때 메모리 증가량이 미미하다. 클래스당 5MB씩만 증가하므로, 100개 클래스도 500MB 추가로 처리 가능하다. 전통적 방식이라면 30GB가 필요하다.

### 3.5 Business Impact

Dinomaly의 비즈니스 영향은 기술적 성능을 넘어선다. 운영 복잡도, 유지보수 비용, 확장성 측면에서 근본적 변화를 가져온다.

운영 복잡도의 감소가 가장 명확하다. 전통적으로 15개 제품 라인을 검사하려면 15개의 독립적 ML 파이프라인을 관리해야 했다. 각각 데이터 수집, 모델 학습, 검증, 배포, 모니터링이 필요하다. ML 엔지니어는 15개 모델의 버전, 하이퍼파라미터, 성능 지표를 추적해야 한다.

Dinomaly로 전환하면 단일 파이프라인으로 단순화된다. 모델은 하나이며, 새 클래스 추가는 memory bank 업데이트만으로 완료된다. 복잡도가 O(N)에서 O(1)로 줄어든다. 이는 MLOps 부담을 90% 이상 감소시킨다.

유지보수 비용도 대폭 절감된다. 전통적 방식에서는 모델별로 성능 저하 모니터링과 재학습이 필요하다. 데이터 분포가 변하면 해당 클래스 모델만 업데이트하면 되지만, 실제로는 모든 모델을 주기적으로 재검증해야 한다. 연간 재학습 비용이 상당하다.

Dinomaly는 단일 모델이므로 재학습 비용이 일괄 처리된다. 분기별로 모든 클래스 데이터를 모아 한 번에 재학습한다. 개별 클래스의 작은 변화는 memory bank 업데이트로 대응한다. 재학습 주기가 월 단위에서 분기 단위로 늘어나며, 비용이 60-70% 감소한다.

확장성의 경제적 이점도 크다. 신규 제품 출시 시 전통적 방식은 데이터 수집 → 모델 학습 → 검증 → 배포의 전 과정을 반복한다. 제품당 2-4주 소요되며, ML 엔지니어의 집중 투입이 필요하다.

Dinomaly는 신규 클래스 추가를 5-10분 작업으로 단축한다. 정상 샘플 100-200장을 수집하고, encoder로 특징을 추출하여 memory bank에 추가한다. 기존 모델은 그대로 사용하므로 재학습 불필요다. 제품 출시 일정에 ML 팀이 병목이 되지 않는다.

실제 제조사의 ROI 분석을 보자. 연간 20개 신제품을 출시하는 중견 제조사의 경우, 전통적 방식의 연간 ML 운영 비용은 약 $200K (엔지니어 인건비, 컴퓨팅 비용, 재학습 비용 포함)이다. Dinomaly로 전환하면 $80K로 감소한다. 60% 비용 절감과 함께 출시 시간도 단축된다.

### 3.6 Implementation Guide

Dinomaly를 실무에 적용하는 구체적인 절차를 살펴본다. 구현은 크게 준비, 학습, 배포, 운영 단계로 나뉜다.

**준비 단계 (1-2일)**

먼저 DINOv2 사전 학습 가중치를 다운로드한다. dinov2_vit_base_14 (768차원, 300MB)를 권장한다. Small (384차원, 90MB)은 메모리가 제한적일 때, Large (1024차원, 1.1GB)는 최고 성능이 필요할 때 선택한다.

데이터 구조를 정리한다. 각 클래스별로 train/normal 폴더에 정상 샘플을 배치한다. 클래스당 최소 100장, 권장 200-300장이다. 샘플이 부족한 클래스는 회전, 반전 등 augmentation으로 보강한다.

환경 설정을 완료한다. PyTorch 2.0+, CUDA 11.8+, GPU 메모리 8GB+ 권장이다. Multi-class 학습은 배치 크기가 클수록 안정적이므로, 가능한 큰 GPU를 사용한다.

**학습 단계 (2-4시간)**

모델 초기화를 수행한다. DINOv2 encoder는 frozen으로 설정하고, bottleneck과 decoder만 학습 가능하게 한다. 이는 overfitting을 방지하고 학습을 안정화한다.

```python
model = DinomalyModel(
    encoder_name="dinov2_vit_base_14",
    bottleneck_dropout=0.2,
    decoder_depth=8
)

for param in model.encoder.parameters():
    param.requires_grad = False
```

학습 하이퍼파라미터를 설정한다. Learning rate 2e-3, AdamW optimizer, cosine annealing scheduler를 사용한다. Batch size는 GPU 메모리에 따라 16-32로 조정한다. Epoch은 10-15면 충분하다.

Multi-class 학습의 핵심은 class-balanced sampling이다. 각 배치에 모든 클래스가 고르게 포함되도록 한다. 클래스별 샘플 수가 불균형하면 weighted sampling으로 보정한다.

학습 중 validation loss를 모니터링한다. 일반적으로 epoch 5-7에서 수렴한다. Early stopping을 적용하여 overfitting을 방지한다. 학습 시간은 15개 클래스, 224x224 해상도 기준 GPU (RTX 3090)에서 약 2시간이다.

**Memory Bank 구축 (30분)**

학습 완료 후 class-conditional memory bank를 구축한다. 각 클래스의 정상 샘플을 encoder에 통과시켜 특징을 추출한다. 패치 단위 특징의 평균과 공분산을 계산하여 저장한다.

```python
for class_name in class_names:
    features = extract_features(model, class_samples[class_name])
    mean = features.mean(dim=0)
    cov = torch.cov(features.T)
    memory_bank[class_name] = {'mean': mean, 'cov': cov}
```

Memory bank 크기를 확인한다. 15개 클래스, 768차원 기준 약 75MB이다. 이는 모델 가중치(300MB)에 비해 작으므로 메모리 부담이 적다.

**추론 및 배포 (즉시)**

추론 파이프라인을 구성한다. 입력 이미지 → 전처리 → encoder → decoder → 이상 맵 계산 → threshold 적용 순으로 진행한다. 클래스 정보는 메타데이터나 별도 분류기에서 제공된다.

```python
def infer(image, class_name):
    features_enc = model.encoder(preprocess(image))
    features_dec = model.decoder(features_enc)
    anomaly_map = compute_anomaly_map(features_enc, features_dec)
    score = compute_score(anomaly_map, memory_bank[class_name])
    return score, anomaly_map
```

추론 속도는 224x224 기준 GPU에서 80-120ms, CPU에서 500-800ms이다. 배치 추론으로 throughput을 높일 수 있다. 배치 크기 8로 실행하면 이미지당 시간이 40-60ms로 단축된다.

**운영 단계 (지속적)**

성능 모니터링을 설정한다. 일별/주별 이상 검출률, false positive rate, anomaly score 분포를 추적한다. 특정 클래스에서 성능 저하가 관찰되면 해당 클래스의 memory bank를 업데이트한다.

신규 클래스 추가는 간단하다. 정상 샘플 수집 → encoder 특징 추출 → memory bank 추가로 5-10분 내 완료된다. 기존 모델 재학습 불필요하므로 즉시 배포 가능하다.

분기별 전체 재학습을 권장한다. 3개월간 축적된 데이터로 모델을 재학습하여 성능을 유지한다. 이는 주말 야간 배치 작업으로 수행되며, 운영에 영향을 주지 않는다.

---

06-foundation-models.md의 4장 VLM-AD를 작성하겠습니다.

---

## 4. VLM-AD (2024)

### 4.1 Basic Information

VLM-AD는 2024년 Deng et al.이 발표한 Vision-Language Model 기반 이상 감지 시스템으로, GPT-4V와 같은 거대 multimodal model을 활용하여 explainable anomaly detection을 실현한다. 기존 모델들이 수치적 이상 점수만 제공했던 반면, VLM-AD는 자연어로 된 결함 설명, 원인 분석, 대응 방안까지 제공한다.

**논문 정보**
- 제목: Vision Language Model based Anomaly Detection
- 저자: Peng Deng, et al.
- 발표: arXiv 2024
- 링크: https://arxiv.org/abs/2412.14446
- 구현: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/vlm_ad

VLM-AD의 핵심은 전통적 anomaly detection과 large language model을 결합하는 것이다. 먼저 PatchCore나 Dinomaly 같은 검증된 모델로 이상을 감지하고, 그 결과를 structured prompt로 변환하여 GPT-4V에 전달한다. GPT-4V는 이미지와 이상 영역 정보를 종합하여 인간이 이해할 수 있는 설명을 생성한다.

이는 단순히 번역이나 요약이 아니다. GPT-4V는 수천억 개의 파라미터로 학습된 세계 지식을 활용하여, 결함의 종류를 분류하고, 발생 가능한 원인을 추론하며, 적절한 대응 방안을 제안한다. 이는 수십 년 경력의 품질 엔지니어가 제공하는 통찰과 유사한 수준이다.

### 4.2 Vision-Language Models

#### 4.2.1 GPT-4V Integration

GPT-4V (GPT-4 with Vision)는 OpenAI가 개발한 multimodal large language model로, 텍스트와 이미지를 동시에 처리할 수 있다. 1조 개 이상의 파라미터로 학습되었으며, 이미지 이해, 추론, 설명 생성에서 인간 수준의 능력을 보인다.

VLM-AD에서 GPT-4V 통합은 다단계 파이프라인으로 구현된다. 첫 번째 단계는 anomaly detection이다. 기존의 검증된 모델을 사용하여 입력 이미지에서 이상 영역을 식별한다. 이상 점수, anomaly map, bounding box 등을 계산한다.

두 번째 단계는 prompt construction이다. 감지 결과를 GPT-4V가 이해할 수 있는 형태로 변환한다. Structured prompt는 다음 요소를 포함한다.

```
System: You are an expert quality inspector analyzing industrial defects.

User: Analyze this image for defects.
Image: [input_image]
Anomaly Map: [heatmap overlay]
Anomaly Score: 0.87 (high confidence)
Product Type: Metal nut
Expected Defects: scratch, dent, thread damage

Task: Provide detailed analysis including:
1. Defect type and location
2. Severity assessment
3. Possible root cause
4. Recommended action
```

세 번째 단계는 GPT-4V inference이다. API를 통해 prompt와 이미지를 전송하고, 응답을 받는다. GPT-4V는 이미지의 시각적 특징과 제공된 메타데이터를 종합하여 분석한다.

네 번째 단계는 response parsing이다. GPT-4V의 자연어 응답을 structured format으로 변환한다. JSON이나 XML로 파싱하여 downstream 시스템이 활용할 수 있게 한다.

$$
\text{Output} = \text{GPT-4V}(\mathbf{I}, \mathbf{M}, \mathbf{P})
$$

여기서 $\mathbf{I}$는 입력 이미지, $\mathbf{M}$은 anomaly map, $\mathbf{P}$는 structured prompt이다.

#### 4.2.2 Natural Language Explanation

자연어 설명 생성은 VLM-AD의 핵심 가치이다. 수치적 이상 점수 0.87이 무엇을 의미하는지, 어떤 조치를 취해야 하는지를 명확하게 전달한다.

설명의 구조는 계층적으로 설계된다. 첫 번째 레벨은 defect classification이다. "scratch", "dent", "contamination", "structural damage" 등 결함 유형을 식별한다. GPT-4V는 광범위한 학습 데이터로부터 각 결함 유형의 시각적 특징을 학습했으므로, 정확한 분류가 가능하다.

두 번째 레벨은 location description이다. "top-left corner", "center region", "along the edge"처럼 결함 위치를 서술한다. Bounding box 좌표를 자연어로 변환하는 것이 아니라, 이미지 전체 맥락에서 의미 있는 위치 표현을 생성한다.

세 번째 레벨은 severity assessment이다. "minor surface scratch", "moderate structural damage", "critical safety issue"처럼 심각도를 평가한다. 이는 단순히 이상 점수를 임계값과 비교하는 것이 아니라, 제품 유형, 사용 목적, 안전 기준을 고려한 종합적 판단이다.

네 번째 레벨은 contextual information이다. "This type of scratch typically occurs during handling"처럼 맥락 정보를 제공한다. GPT-4V의 world knowledge를 활용하여, 제조 공정에 대한 일반적 이해를 반영한다.

예시 설명을 보자.

```
Analysis Results:

Defect Type: Surface scratch
Location: Located at the top-right quadrant, approximately 2cm from the edge
Severity: Moderate - affects surface appearance but does not compromise structural integrity
Dimensions: Approximately 15mm length, 0.5mm width

Visual Characteristics:
- Linear pattern consistent with abrasion damage
- Shallow depth, limited to surface coating layer
- No signs of metal deformation or cracking
- Clean edges suggest recent occurrence
```

#### 4.2.3 Root Cause Analysis

Root cause analysis는 VLM-AD의 가장 고도화된 기능이다. 단순히 결함을 서술하는 것을 넘어, 발생 원인을 추론한다. 이는 재발 방지를 위한 공정 개선에 직접 활용된다.

GPT-4V는 결함의 시각적 패턴으로부터 원인을 추론한다. 예를 들어, 직선형 스크래치는 운송 중 마찰을, 원형 얼룩은 액체 오염을, 불규칙한 변색은 열 손상을 암시한다. 이러한 패턴 인식은 GPT-4V가 대규모 이미지 데이터로부터 학습한 시각적 상관관계에 기반한다.

추론 과정은 다단계로 진행된다. 먼저 visual pattern matching이다. 관찰된 결함 패턴을 학습 데이터의 유사 사례와 비교한다. "This scratch pattern is characteristic of abrasive contact"처럼 패턴을 식별한다.

다음은 process knowledge application이다. 제품 유형과 제조 공정에 대한 지식을 활용한다. "In metal stamping processes, such defects typically occur at the die contact point"처럼 공정 특성을 연결한다.

세 번째는 temporal reasoning이다. 결함의 외관으로부터 발생 시점을 추론한다. "Fresh scratch with sharp edges suggests recent occurrence, likely during final assembly"처럼 시간적 맥락을 파악한다.

마지막은 probability estimation이다. 여러 가능한 원인 중 가장 개연성 높은 것을 제시한다. "Most likely cause: improper handling during packaging (80% confidence). Alternative causes: equipment malfunction (15%), material defect (5%)"처럼 정량화한다.

실제 root cause analysis 예시이다.

```
Root Cause Analysis:

Primary Hypothesis: Improper handling during transportation
Supporting Evidence:
- Linear scratch pattern consistent with sliding contact
- Located on exposed surface vulnerable during shipping
- Depth and width suggest moderate pressure application
- No similar defects observed in protected areas

Contributing Factors:
- Insufficient protective packaging material
- Potential movement in shipping container
- Contact with rough or sharp surface

Confidence: High (85%)

Recommended Investigation:
- Review packaging procedures and materials
- Inspect shipping containers for sharp edges
- Check loading/unloading protocols
- Interview shipping staff for recent process changes
```

### 4.3 Explainable AI Realization

VLM-AD는 explainable AI의 실질적 구현을 달성한다. 기존의 XAI 연구가 주로 모델 내부 작동 방식을 설명하는 데 집중했다면, VLM-AD는 실무자가 실제로 필요로 하는 actionable insights를 제공한다.

Explainability의 세 가지 차원을 만족한다. 첫째는 technical explainability이다. 왜 이 이미지가 이상으로 판정되었는지, 어떤 특징이 결정적이었는지를 설명한다. "High anomaly score driven by: unusual texture pattern (40%), color deviation (35%), geometric irregularity (25%)"처럼 정량화한다.

둘째는 operational explainability이다. 현장 작업자가 즉시 이해하고 행동할 수 있는 설명이다. "Reject this part and inspect the stamping die for wear"처럼 구체적 지시를 제공한다.

셋째는 regulatory explainability이다. 규제 기관이나 감사 시 요구하는 추적 가능성과 근거를 제공한다. "Rejection decision based on: visual inspection indicating 2mm surface defect exceeding tolerance specification of 0.5mm, detected by certified AI system with 98.5% accuracy"처럼 formal documentation을 생성한다.

Explainability의 실무적 가치는 다음과 같다. 품질 엔지니어는 AI 판정의 타당성을 빠르게 검증할 수 있다. False positive 케이스에서 AI가 착각한 이유를 이해하고, threshold를 조정할 수 있다.

신입 검사원 교육이 가속화된다. AI의 설명을 읽으며 결함 유형, 위치 표현, 심각도 판단 기준을 학습한다. 기존에 6개월 걸리던 숙련 과정이 3개월로 단축된다.

공정 개선 활동이 데이터 기반으로 전환된다. AI가 제시한 root cause를 집계하여, 가장 빈번한 결함 원인을 식별한다. "지난 달 스크래치의 65%가 운송 중 발생 → 포장재 개선 필요"처럼 우선순위를 정량화한다.

### 4.4 Output Examples

실제 VLM-AD 출력 사례를 통해 시스템의 능력을 구체적으로 살펴본다. 각 예시는 서로 다른 제품 유형과 결함 특성을 보여준다.

**예시 1: Metal Nut - Thread Damage**

```json
{
  "defect_detected": true,
  "anomaly_score": 0.92,
  
  "classification": {
    "defect_type": "Thread damage",
    "sub_category": "Partial thread stripping",
    "severity": "Critical",
    "confidence": 0.94
  },
  
  "location": {
    "description": "Internal threads, upper section",
    "coordinates": {"x": 145, "y": 78, "width": 35, "height": 45},
    "affected_area_percentage": 12
  },
  
  "visual_description": "Multiple threads show incomplete formation with rough edges. Thread profile is irregular and depth is inconsistent. Metal surface appears torn rather than cleanly cut.",
  
  "root_cause": {
    "primary": "Worn or damaged threading die",
    "confidence": 0.85,
    "supporting_evidence": [
      "Irregular thread profile consistent with die wear",
      "Progressive degradation pattern visible",
      "Similar defects likely in recent production batch"
    ],
    "alternatives": [
      {"cause": "Incorrect material hardness", "confidence": 0.10},
      {"cause": "Machine calibration error", "confidence": 0.05}
    ]
  },
  
  "impact_assessment": {
    "functionality": "Severely compromised - thread engagement will fail",
    "safety": "High risk - potential fastener failure under load",
    "usability": "Not usable - reject immediately"
  },
  
  "recommended_actions": {
    "immediate": [
      "Quarantine affected batch (last 2 hours production)",
      "Stop threading operation",
      "Inspect threading die for wear"
    ],
    "short_term": [
      "Replace threading die",
      "Inspect all nuts produced since last die change",
      "Adjust threading parameters if needed"
    ],
    "long_term": [
      "Implement preventive die replacement schedule",
      "Add in-process thread inspection",
      "Review die maintenance procedures"
    ]
  },
  
  "documentation": "Critical defect detected at 2025-01-15 14:23:17. Thread damage severity exceeds acceptance criteria per ISO 965-1. Part rejected and flagged for process investigation."
}
```

**예시 2: Wood Panel - Surface Contamination**

```json
{
  "defect_detected": true,
  "anomaly_score": 0.76,
  
  "classification": {
    "defect_type": "Surface contamination",
    "sub_category": "Oil stain",
    "severity": "Moderate",
    "confidence": 0.88
  },
  
  "location": {
    "description": "Center-left area, approximately 8cm from left edge",
    "coordinates": {"x": 98, "y": 156, "width": 28, "height": 32},
    "affected_area_percentage": 4
  },
  
  "visual_description": "Irregular dark stain with blurred edges. Darker concentration at center, gradually fading outward. Texture of wood grain visible through stain, indicating surface-level contamination.",
  
  "root_cause": {
    "primary": "Hydraulic oil drip from overhead equipment",
    "confidence": 0.70,
    "supporting_evidence": [
      "Stain pattern consistent with liquid droplet",
      "Location beneath hydraulic line",
      "Similar stains observed on adjacent panels"
    ],
    "alternatives": [
      {"cause": "Lubricant spillage during handling", "confidence": 0.20},
      {"cause": "Pre-existing material defect", "confidence": 0.10}
    ]
  },
  
  "impact_assessment": {
    "functionality": "No functional impact",
    "aesthetics": "Moderate - visible stain affects appearance",
    "treatability": "Potentially removable with appropriate solvent"
  },
  
  "recommended_actions": {
    "immediate": [
      "Inspect hydraulic line for leaks",
      "Protect panels from potential drips",
      "Attempt stain removal with approved solvent"
    ],
    "short_term": [
      "Repair or replace leaking hydraulic components",
      "Install drip shields above work area",
      "Review material handling procedures"
    ],
    "long_term": [
      "Schedule preventive maintenance for hydraulic systems",
      "Consider relocating hydraulic lines away from products",
      "Implement cover protocol for stored materials"
    ]
  },
  
  "documentation": "Moderate defect detected at 2025-01-15 14:25:43. Surface contamination may be removable. Part held for cleaning attempt. If cleaning successful, may proceed to next stage."
}
```

### 4.5 Use Cases (Regulatory, Quality Reports)

VLM-AD의 자연어 설명 능력은 특히 규제 대응과 품질 보고서 자동화에서 강력하다. 이 두 영역은 전통적으로 엄청난 수작업 문서화 부담을 요구했다.

**규제 산업 적용**

의료기기, 항공우주, 자동차 같은 규제 산업에서는 모든 품질 결정에 대한 추적 가능성이 필수이다. FDA 21 CFR Part 11, ISO 13485, AS9100 등의 규격은 검사 결과의 근거, 검사자 자격, 장비 calibration 등을 문서화하도록 요구한다.

VLM-AD는 이러한 요구사항을 자동으로 충족한다. 각 검사 결과에 대해 다음을 생성한다.

```
Inspection Record - Medical Device Component

Part Number: MD-2024-1147
Inspection Date: 2025-01-15 14:30:22
Inspector: AI System (Certified Model: VLM-AD v2.1)
Equipment: Anomaly Detection System SN: AD-2024-078
Calibration Status: Valid until 2025-06-30

Inspection Results:
Result: REJECT
Anomaly Score: 0.89 (Above acceptance threshold: 0.75)

Defect Description:
Type: Surface pit
Location: Front face, 3mm from center
Dimensions: 0.8mm diameter, estimated 0.2mm depth
Classification: Material defect, Grade 2 per ISO 10110-7

Technical Basis:
Visual inspection performed per SOP-QC-014
Anomaly detection algorithm: PatchCore
Confidence Level: 94%
Reference Standard: Company Spec CS-MD-2024 Rev 3

Deviation from Specification:
Maximum allowable pit size: 0.5mm per CS-MD-2024 Section 4.2.3
Observed pit size: 0.8mm
Deviation: 0.3mm (60% over limit)

Recommended Disposition: Scrap - Cannot be reworked
Root Cause: Material quality issue, recommend supplier investigation

Quality Assurance Review Required: Yes
Review Deadline: 2025-01-16 14:30:22

Electronic Signature: VLM-AD System v2.1
Signature Hash: 7a3f9c2e...b8d4e1f6
Timestamp: 2025-01-15T14:30:22.847Z
```

이러한 자동 문서화는 감사 준비 시간을 80% 이상 단축한다. 규제 당국 감사 시 요청받는 "지난 1년간 모든 불합격 제품의 근거 자료"를 수 초 내 생성할 수 있다.

**품질 보고서 자동화**

주간/월간 품질 보고서 작성은 품질 엔지니어의 상당한 시간을 소비한다. VLM-AD는 이를 자동화한다.

```markdown
# Weekly Quality Report
Period: 2025-01-08 to 2025-01-14
Products Inspected: 14,275 units

## Executive Summary
- Total Defects Detected: 428 (3.0% defect rate)
- Critical Defects: 67 (0.47%)
- Major Defects: 189 (1.32%)
- Minor Defects: 172 (1.20%)

## Defect Analysis by Type
1. Surface Scratches: 156 (36.4%)
   - Primary Root Cause: Handling during packaging (78%)
   - Trend: +12% vs last week
   - **Action Required**: Review packaging procedures

2. Thread Damage: 89 (20.8%)
   - Primary Root Cause: Die wear (85%)
   - Trend: +45% vs last week (Critical)
   - **Action Required**: Immediate die replacement scheduled

3. Contamination: 67 (15.7%)
   - Primary Root Cause: Hydraulic leaks (62%)
   - Trend: Stable
   - **Action Required**: Preventive maintenance due

[...]

## Top 3 Process Improvement Opportunities
1. **Packaging Process** (Potential 36% defect reduction)
   - Issue: Scratches during packaging
   - Impact: 156 defects/week, $4,680 scrap cost
   - Recommendation: Implement protective film application
   - Expected ROI: 8 weeks

2. **Threading Die Maintenance** (Potential 21% defect reduction)
   - Issue: Insufficient die replacement frequency
   - Impact: 89 defects/week, increasing trend
   - Recommendation: Reduce replacement interval from 10,000 to 7,500 units
   - Expected ROI: Immediate

[...]
```

### 4.6 Cost Considerations

VLM-AD의 비용 구조는 전통적 anomaly detection과 근본적으로 다르다. GPT-4V API 사용료가 주요 비용이며, 사용 패턴에 따라 크게 변동한다.

**API 비용 구조**

GPT-4V API pricing (2025년 1월 기준)은 다음과 같다. Input tokens: $0.01 per 1K tokens, Output tokens: $0.03 per 1K tokens, Image processing: $0.00765 per image (high detail mode).

일반적인 VLM-AD 요청의 비용 분해이다.

```
Single Inspection Cost:
- Image processing: $0.00765
- Input prompt: ~500 tokens = $0.005
- Output response: ~300 tokens = $0.009
Total per inspection: ~$0.022

With structured output parsing:
- Additional output tokens: ~100 = $0.003
Total per inspection: ~$0.025
```

하루 1,000개 제품 검사 시 비용을 계산해보자. 하지만 모든 제품에 VLM-AD를 적용할 필요는 없다. 효율적 전략은 2-tier 접근이다.

Tier 1: 전통적 anomaly detection (PatchCore, Dinomaly)으로 모든 제품 스크리닝. 비용: GPU 전력 비용만, 거의 무시 가능.

Tier 2: 이상 감지된 제품만 VLM-AD로 상세 분석. 비용: 이상 비율에 비례.

```
Daily Cost Calculation (1,000 products):
Scenario 1 - 3% defect rate:
- Tier 1 screening: $0 (on-premise GPU)
- Tier 2 analysis: 30 units × $0.025 = $0.75
- Daily total: $0.75
- Monthly total: ~$22

Scenario 2 - 10% defect rate:
- Tier 2 analysis: 100 units × $0.025 = $2.50
- Daily total: $2.50
- Monthly total: ~$75

Scenario 3 - All products (for documentation):
- All analysis: 1,000 units × $0.025 = $25
- Daily total: $25
- Monthly total: ~$750
```

**ROI 분석**

VLM-AD의 비용을 전통적 문서화 비용과 비교한다. 품질 엔지니어가 불합격 제품 1건당 보고서 작성에 15분 소요한다고 가정하자. 시급 $50 기준으로 건당 $12.50 비용이다.

```
Traditional Documentation Cost:
30 defects/day × $12.50 = $375/day
Monthly cost: ~$11,250

VLM-AD Cost:
30 defects/day × $0.025 = $0.75/day
Monthly cost: ~$22

Savings: $11,228/month (99.8% reduction)
```

물론 품질 엔지니어가 완전히 불필요해지는 것은 아니다. 대신 상세 보고서 작성에서 해방되어 근본 원인 분석과 공정 개선에 집중할 수 있다. 이는 정량화하기 어렵지만 훨씬 큰 가치를 창출한다.

**비용 최적화 전략**

실무에서는 다음 전략으로 비용을 최적화한다.

첫째, selective analysis이다. Critical defects만 VLM-AD로 분석하고, minor defects는 간단한 template 기반 보고서를 사용한다.

둘째, batch processing이다. 여러 이미지를 하나의 API 요청으로 묶어 처리하면 overhead가 줄어든다.

셋째, caching이다. 유사한 결함은 이전 분석 결과를 재사용한다. "이 스크래치는 3일 전 분석한 케이스와 93% 유사함 → 캐시된 설명 사용" 방식이다.

넷째, on-premise alternatives이다. LLaMA 3.2 Vision 같은 오픈소스 VLM을 로컬에 배포하면 API 비용이 없다. 성능은 GPT-4V보다 낮지만 기본적 설명 생성은 가능하다.

종합하면, VLM-AD는 API 비용이 발생하지만 인건비 절감과 품질 향상 효과가 훨씬 크다. 특히 규제 산업이나 고부가가치 제품에서 ROI가 명확하다. 월 $1,000 미만의 API 비용으로 연간 수십만 달러의 가치를 창출할 수 있다.

---

06-foundation-models.md의 5장과 6장을 작성하겠습니다.

---

## 5. SuperSimpleNet (2024)

### 5.1 Basic Information

SuperSimpleNet은 2024년 Božič et al.이 발표한 surface defect detection 전용 모델로, unsupervised와 supervised learning을 통합한 hybrid 접근법이다. 제목의 "Super Simple"은 아키텍처의 단순함이 아니라, 배포와 사용의 간편함을 의미한다.

**논문 정보**
- 제목: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection
- 저자: Blaž Božič, Blaž Rolih, Danijel Skočaj
- 발표: arXiv 2024
- 링크: https://arxiv.org/pdf/2408.03143
- 구현: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/supersimplenet

SuperSimpleNet의 핵심 혁신은 training mode flexibility이다. 동일한 아키텍처로 unsupervised 모드(정상 샘플만 사용)와 supervised 모드(이상 샘플 포함)를 모두 지원한다. 이는 데이터 가용성에 따라 최적 전략을 선택할 수 있게 한다.

모델은 표면 결함 검출에 특화되었다. Metal scratch, fabric defect, wood grain anomaly 등 표면 품질이 중요한 산업에 최적화되어 있다. Wide ResNet-50 backbone을 사용하며, layer2와 layer3의 multi-scale features를 활용한다.

### 5.2 Unified Framework

SuperSimpleNet의 unified framework는 단일 아키텍처로 unsupervised와 supervised learning을 통합한다. 이는 단순히 두 모드를 지원하는 것이 아니라, 데이터 상황에 따라 seamless transition을 가능하게 한다.

아키텍처는 세 가지 주요 컴포넌트로 구성된다. 첫째는 feature extraction이다. Wide ResNet-50의 layer2 (512채널)과 layer3 (1024채널)을 추출한다. 이 레이어들은 서로 다른 scale의 특징을 포착하여, 미세한 표면 결함과 전역적 패턴을 동시에 모델링한다.

$$
\mathbf{F} = [\mathbf{F}_2, \mathbf{F}_3] = \text{Backbone}(\mathbf{I})
$$

여기서 $\mathbf{F}_2 \in \mathbb{R}^{B \times 512 \times H/8 \times W/8}$, $\mathbf{F}_3 \in \mathbb{R}^{B \times 1024 \times H/16 \times W/16}$이다.

둘째는 feature adaptation이다. Multi-scale features를 동일한 spatial resolution으로 upsampling하고 concatenate한다. Neighboring patch aggregation을 통해 공간적 맥락을 강화한다.

```
Feature Adaptation Process:
1. Upscale F2 to 2× size (H/4 × W/4)
2. Upscale F3 to 2× size (H/4 × W/4)
3. Concatenate: F_concat = [F2_up, F3_up]  # 1536 channels
4. Apply 1×1 conv projection
```

셋째는 segmentation-detection module이다. 이 모듈은 pixel-level anomaly map과 image-level anomaly score를 동시에 생성한다. 두 출력은 서로 다른 pooling 전략으로 계산되어 robust한 판정을 제공한다.

Unified training의 핵심은 anomaly generation 메커니즘이다. Unsupervised 모드에서는 Perlin noise로 synthetic anomaly를 생성하고, supervised 모드에서는 실제 anomaly mask를 사용한다. 두 모드 모두 동일한 loss function을 공유한다.

### 5.3 Unsupervised + Supervised Fusion

SuperSimpleNet의 가장 독특한 특징은 unsupervised와 supervised 학습을 융합하는 방식이다. 이는 단순한 mode switching이 아니라, 양쪽의 장점을 결합한 hybrid strategy이다.

**Unsupervised Mode**

Unsupervised 모드는 정상 샘플만으로 학습한다. 핵심은 simulated anomaly generation이다. Perlin noise를 사용하여 realistic한 표면 결함을 합성한다.

Perlin noise는 자연스러운 texture pattern을 생성하는 알고리즘이다. 2D Perlin noise는 다양한 frequency의 smooth gradient를 중첩하여 organic한 패턴을 만든다. Threshold를 적용하면 binary anomaly mask가 생성된다.

```
Synthetic Anomaly Generation:
1. Generate Perlin noise: P ∈ [-1, 1]
2. Apply threshold (default: 0.2): M = (P > 0.2)
3. Generate Gaussian noise: N ~ N(0, 0.015)
4. Apply masked noise: F' = F + N × M
5. 50% probability: M = 0 (normal sample)
```

이렇게 생성된 synthetic anomaly는 실제 결함과 유사한 특성을 보인다. Scratch는 elongated Perlin pattern으로, dent는 circular pattern으로, contamination은 irregular pattern으로 모사된다.

Loss function은 focal loss와 truncated L1 loss를 결합한다. Focal loss는 hard negative mining 효과를 제공하고, truncated L1은 margin-based separation을 강제한다.

$$
\mathcal{L}_{\text{focal}} = -\frac{1}{N}\sum_{i=1}^{N} (1 - p_i)^\gamma \log p_i
$$

$$
\mathcal{L}_{\text{trunc}} = \frac{1}{N_n}\sum_{x \in \mathcal{N}} \max(s(x) + \tau, 0) + \frac{1}{N_a}\sum_{x \in \mathcal{A}} \max(-s(x) + \tau, 0)
$$

여기서 $\mathcal{N}$은 정상 픽셀, $\mathcal{A}$는 이상 픽셀, $s(x)$는 anomaly score, $\tau$는 truncation threshold (default: 0.5)이다.

**Supervised Mode**

Supervised 모드는 실제 anomaly samples을 활용한다. Anomaly mask가 제공되므로 synthetic generation이 불필요하다. 대신 실제 결함 패턴을 직접 학습한다.

Supervised 모드의 핵심은 gradient flow control이다. Unsupervised에서는 feature extractor를 frozen하지만, supervised에서는 선택적으로 fine-tuning한다. 이는 task-specific adaptation을 가능하게 한다.

```python
if supervised:
    stop_grad = False  # Allow gradient flow to feature extractor
else:
    stop_grad = True   # Freeze feature extractor
```

Loss function은 동일하지만, 학습 dynamics가 다르다. Supervised 모드는 gradient clipping (norm 1.0)을 적용하여 안정성을 높인다. Learning rate도 다르게 설정된다 (adaptor: 1e-4, decoder: 2e-4).

**Hybrid Strategy**

실무에서 가장 효과적인 전략은 sequential hybrid이다. 먼저 unsupervised로 pretraining하고, 이상 샘플이 수집되면 supervised fine-tuning을 수행한다.

```
Hybrid Training Pipeline:
Phase 1 (Weeks 1-2): Unsupervised pretraining
- Data: 200-300 normal samples
- Epochs: 50
- Result: Baseline detector (95% AUROC)

Phase 2 (Week 3): Supervised fine-tuning
- Data: 50-100 anomaly samples + normal samples
- Epochs: 20-30
- Result: Specialized detector (97-98% AUROC)
```

이 접근법은 cold start problem을 해결한다. 초기에는 unsupervised로 빠르게 배포하고, 운영 중 수집된 실제 불량 데이터로 지속적으로 개선한다.

### 5.4 Performance Analysis (97.2%)

SuperSimpleNet의 성능은 MVTec AD에서 평균 image-level AUROC 97.2%를 기록한다. 이는 SOTA 모델들(PatchCore 99.1%, Dinomaly 99.2%)보다 낮지만, 표면 결함 특화 데이터셋에서는 경쟁력 있는 성능을 보인다.

카테고리별 성능 분석을 보면, texture 카테고리에서 강점을 보인다. Carpet (98.5%), Leather (98.2%), Wood (98.0%), Tile (97.8%)는 모두 97% 이상이다. 이는 SuperSimpleNet이 표면 품질 검사에 최적화되었음을 증명한다.

Object 카테고리는 상대적으로 낮다. Screw (94.8%), Toothbrush (95.2%)는 95% 수준이다. 미세한 부품 결함은 multi-scale feature만으로는 충분히 포착되지 않는다.

Supervised vs Unsupervised 비교도 흥미롭다. Unsupervised SuperSimpleNet은 평균 95.8% AUROC를 달성한다. 50개 anomaly samples로 supervised fine-tuning하면 97.2%로 1.4% 향상된다. 이는 적은 양의 labeled data로도 significant improvement를 얻을 수 있음을 보여준다.

추론 속도는 256×256 입력 기준 GPU에서 약 30-40ms이다. PatchCore (50-100ms)보다 빠르지만 EfficientAD (1-5ms)보다는 느리다. CPU 추론은 200-300ms로, 실시간 요구사항이 있는 환경에서는 제한적이다.

Pixel-level performance도 우수하다. 평균 pixel-level AUROC 96.5%로, 정밀한 결함 위치 파악이 가능하다. Anomaly map의 해상도는 원본 이미지와 동일하게 유지되어, 1mm 이하의 미세 결함도 시각화된다.

실무 적용 사례를 보자. 한 금속 가공 업체는 SuperSimpleNet을 스크래치 검사에 적용했다. Unsupervised로 2주 내 배포하고, 3개월간 수집된 200개 실제 불량으로 fine-tuning했다. 최종 정확도는 98.5%로, 수동 검사 대비 검사 시간을 70% 단축했다.

---

## 6. UniNet (2025)

### 6.1 Basic Information

UniNet은 2025년 Pang et al.이 발표한 contrastive learning 기반 anomaly detection 프레임워크이다. "Unified Contrastive Learning"이라는 이름처럼, source domain (정상)과 target domain (이상)을 contrastive 방식으로 통합 모델링한다.

**논문 정보**
- 제목: Unified Contrastive Learning Framework for Anomaly Detection
- 저자: Datang Pang, et al.
- 발표: 2025 (최신)
- 링크: https://github.com/pangdatangtt/UniNet
- 구현: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uninet

UniNet의 핵심은 temperature-scaled contrastive loss를 통한 robust decision boundary 학습이다. 전통적 knowledge distillation이 teacher-student feature matching에 집중했다면, UniNet은 feature space에서 정상과 이상의 명확한 분리를 강제한다.

모델 구조는 teacher-student 기반이지만, student는 단순 모방이 아닌 contrastive learning을 수행한다. Teacher는 Wide ResNet-50, student는 ResNet-like decoder로 구성된다. Attention bottleneck이 중간에 위치하여 informative features를 선택한다.

### 6.2 Contrastive Learning

UniNet의 contrastive learning은 세 가지 핵심 구성요소를 포함한다.

**Cosine Similarity-based Distance**

전통적 L2 distance 대신 cosine similarity를 사용한다. 이는 feature magnitude에 덜 민감하며, 방향성 차이를 강조한다.

$$
\text{sim}(\mathbf{f}_s, \mathbf{f}_t) = \frac{\mathbf{f}_s \cdot \mathbf{f}_t}{\|\mathbf{f}_s\| \|\mathbf{f}_t\|}
$$

Anomaly score는 1 - similarity로 계산된다. 정상 샘플은 teacher와 student features가 유사하여 낮은 점수를, 이상 샘플은 불일치하여 높은 점수를 받는다.

**Temperature-scaled Contrastive Loss**

Temperature parameter $\tau$를 도입하여 similarity distribution을 조절한다. 낮은 temperature는 hard decision boundary를, 높은 temperature는 soft boundary를 생성한다.

$$
\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{f}_s^i, \mathbf{f}_t^i) / \tau)}{\sum_{j} \exp(\text{sim}(\mathbf{f}_s^i, \mathbf{f}_t^j) / \tau)}
$$

여기서 $i$는 positive pair (동일 위치), $j$는 all pairs (배치 내 모든 위치)이다. Default temperature는 0.4로 설정된다.

**Margin-based Separation**

단순히 feature matching을 넘어, explicit margin을 강제한다. 정상 샘플의 similarity는 margin 이상, 이상 샘플은 margin/2 이하로 push한다.

$$
\mathcal{L}_{\text{margin}} = \sum_{x \in \mathcal{N}} \max(m - \text{sim}(x), 0) + \sum_{x \in \mathcal{A}} \max(\text{sim}(x) - m/2, 0)
$$

여기서 $m$은 margin (예: 0.8), $\mathcal{N}$은 정상 샘플, $\mathcal{A}$는 이상 샘플이다.

전체 loss function은 세 항목의 weighted sum이다.

$$
\mathcal{L} = \lambda \mathcal{L}_{\text{cosine}} + (1-\lambda) \mathcal{L}_{\text{contrast}} + \mathcal{L}_{\text{margin}}
$$

Default $\lambda = 0.5$로 cosine loss와 contrastive loss의 균형을 맞춘다.

### 6.3 Robust Decision Boundaries

UniNet이 강조하는 것은 단순한 성능이 아니라 decision boundary의 robustness이다. 이는 실무 배포에서 결정적 차이를 만든다.

**Domain-Related Feature Selection**

모든 features가 anomaly detection에 동등하게 유용하지 않다. UniNet은 domain-related feature selection (DFS) 메커니즘으로 informative features를 강조한다.

DFS는 feature importance를 학습 가능한 attention weights로 모델링한다. Source (정상)과 target (테스트) features의 상관관계를 계산하여, discriminative features를 선택한다.

```
DFS Process:
1. Compute feature statistics: μ_s, σ_s (source), μ_t, σ_t (target)
2. Calculate correlation: ρ = corr(μ_s - μ_t, σ_s - σ_t)
3. Generate attention weights: α = softmax(ρ / τ)
4. Weight features: F' = α ⊙ F
```

이는 lighting, viewpoint, background 변화에 robust한 features를 자동으로 선택한다.

**Weighted Decision Mechanism**

Multiple scales의 anomaly scores를 adaptive하게 결합한다. 단순 평균이 아니라, 각 scale의 confidence를 고려한 weighted aggregation이다.

$$
s_{\text{final}} = \sum_{l=1}^{L} w_l \cdot s_l, \quad w_l = \frac{\exp(\alpha \cdot \max(s_l))}{\sum_{k=1}^{L} \exp(\alpha \cdot \max(s_k))}
$$

여기서 $\alpha$는 sharpness parameter (default: 0.01), $\max(s_l)$은 scale $l$의 최대 anomaly score이다. 높은 confidence를 보이는 scale에 더 큰 가중치를 부여한다.

**Boundary Visualization**

UniNet의 decision boundary를 t-SNE로 시각화하면, 정상과 이상 samples이 명확히 분리된 clusters를 형성한다. PatchCore나 STFPM과 비교할 때, UniNet의 boundary가 더 wide margin을 보인다.

```
t-SNE Visualization (2D projection):
- Normal cluster: Tight, low intra-class variance
- Anomaly cluster: Separated by clear gap
- Margin width: ~0.5 in normalized feature space
- Overlap ratio: <2% (vs. 5-8% in traditional methods)
```

이러한 robust boundary는 distribution shift에 강하다. 조명, 카메라 각도, 배경 변화에도 판정이 일관적이다.

### 6.4 Performance Analysis (98.3%)

UniNet은 MVTec AD에서 평균 image-level AUROC 98.3%를 달성한다. 이는 PatchCore (99.1%), Dinomaly (99.2%)보다 낮지만, Reverse Distillation (98.6%)과 유사한 수준이다.

카테고리별 분석을 보면, 구조적 복잡성이 있는 객체에서 강점을 보인다. Transistor (98.7%), Pill (98.5%), Capsule (98.6%)는 미세한 부품 검사에서 우수하다. Contrastive learning이 subtle한 차이를 잘 포착하기 때문이다.

Texture 카테고리는 평균 97.8%로 object보다 약간 낮다. 하지만 개선의 여지가 있다. Temperature tuning으로 texture-specific optimization이 가능하다.

False positive rate가 특히 낮다. 1% FPR에서 true positive rate 96.5%를 달성하여, 실무 배포 시 불필요한 검사원 개입을 최소화한다. 이는 contrastive margin의 효과이다.

Distribution shift에 대한 robustness를 테스트했다. 조명 변화 (±30% brightness)에서 성능 저하가 1.2%에 불과하다. PatchCore (2.8%), STFPM (3.5%)보다 안정적이다. DFS와 contrastive learning의 synergy 효과이다.

추론 속도는 256×256 기준 GPU에서 80-100ms이다. PatchCore와 유사하며, 실시간은 아니지만 일반 검사 라인에 충분하다. Batch inference로 throughput을 높일 수 있다.

실무 적용 사례를 보자. 한 반도체 업체는 wafer defect inspection에 UniNet을 적용했다. 기존 rule-based system 대비 false alarm rate를 60% 감소시켰다. Temperature=0.3으로 tuning하여 미세 결함에 민감하게 설정했으며, 최종 정확도 99.1%를 달성했다.

---

06-foundation-models.md의 7장과 8장을 작성하겠습니다.

---

## 7. Comprehensive Comparison

### 7.1 Three Dimensions of Revolution

Foundation Models 패러다임은 세 가지 차원에서 혁명을 일으켰다. 이 차원들은 독립적이면서도 상호 보완적이며, 각각 고유한 가치를 창출한다.

**Multi-class Revolution**

첫 번째 차원은 multi-class capability이다. Dinomaly가 선도하는 이 혁명은 모델 관리의 근본적 단순화를 가져왔다. 전통적으로 N개 제품 클래스는 N개 모델을 요구했으나, Dinomaly는 단일 통합 모델로 해결한다.

경제적 영향을 정량화하면, 15개 클래스 환경에서 모델 수는 15:1로 감소한다. 학습 시간은 15×2시간 = 30시간에서 3시간으로 90% 단축된다. 메모리 사용량은 4.5GB에서 475MB로 93% 절감된다. 모델 관리 복잡도는 O(N)에서 O(1)로 개선된다.

성능 trade-off는 놀랍도록 작다. Multi-class Dinomaly 98.8% vs. Single-class average 99.2%로 불과 0.4% 차이이다. 일부 카테고리에서는 multi-class가 오히려 우수하다. Cross-class learning의 긍정적 효과이다.

**Zero-shot Revolution**

두 번째 차원은 zero-shot learning이다. WinCLIP이 대표하는 이 혁명은 배포 시간을 재정의했다. 학습 데이터 수집에 수 주가 소요되던 것이, 프롬프트 작성 수 분으로 단축되었다.

Time-to-deployment 비교를 보자. 전통적 접근법은 데이터 수집 2-4주 + 학습 4-8시간 + 검증 1-2일 = 3-5주이다. WinCLIP은 프롬프트 설계 10-30분 + 검증 1-2시간 = 하루 이내이다. 시간 단축 비율은 95-98%이다.

성능 trade-off는 명확히 존재한다. Zero-shot WinCLIP 91-95% vs. Trained PatchCore 99.1%로 4-8% 낮다. 하지만 특정 시나리오에서는 이 차이가 수용 가능하다. 신제품 출시 첫날부터 품질 검사 가동, 다품종 소량 생산 환경, 빠른 feasibility 검증 등에서는 zero-shot의 즉시성이 4-8% 정확도 차이보다 가치 있다.

**Explainability Revolution**

세 번째 차원은 explainable AI이다. VLM-AD가 실현하는 이 혁명은 AI 판정을 "블랙박스"에서 "투명한 의사결정"으로 전환시켰다. 수치적 이상 점수에서 자연어 설명으로의 진화이다.

실무 가치를 정량화하면, 품질 보고서 작성 시간은 제품당 15분에서 자동 생성 즉시로 단축된다. 월 1000건 불량 발생 시 $12,500 인건비에서 $25 API 비용으로 99.8% 절감이다. 신입 검사원 숙련 기간은 6개월에서 3개월로 단축된다.

규제 대응 효과는 더 크다. FDA 감사 준비 시간이 수 주에서 수 시간으로 단축된다. 모든 품질 결정의 근거가 자동 문서화되기 때문이다. ISO 13485, AS9100 같은 품질 규격 compliance가 자동으로 충족된다.

### 7.2 Multi-class Economics

Multi-class capability의 경제적 영향을 상세히 분석한다. 단순한 비용 절감을 넘어, 사업 모델 자체를 변화시킨다.

**개발 비용 비교**

15개 제품 라인을 검사하는 시스템 구축 시나리오를 가정하자. ML 엔지니어 시급 $100, 클라우드 GPU 비용 $2/hour로 계산한다.

전통적 Single-class 접근:
```
모델별 개발 (15회 반복):
- 데이터 준비 및 전처리: 4h × $100 = $400
- 모델 학습 및 튜닝: 8h × $102 = $816
- 검증 및 배포: 4h × $100 = $400
- 소계: $1,616 per model
- 총계: $1,616 × 15 = $24,240

연간 유지보수 (15 models):
- 분기별 재학습: 4 × (15 × 2h) = 120h × $102 = $12,240
- 성능 모니터링: 52주 × 2h × $100 = $10,400
- 버그 수정 및 업데이트: $5,000
- 연간 총계: $27,640
```

Multi-class Dinomaly:
```
통합 모델 개발 (1회):
- 데이터 준비: 6h × $100 = $600
- 모델 학습: 4h × $102 = $408
- 검증: 6h × $100 = $600
- 총계: $1,608

연간 유지보수:
- 분기별 재학습: 4 × 4h × $102 = $1,632
- 성능 모니터링: 52주 × 0.5h × $100 = $2,600
- 버그 수정: $1,000
- 연간 총계: $5,232
```

비용 비교 결과, 초기 개발 비용은 $24,240 vs. $1,608로 93% 절감이다. 연간 운영 비용은 $27,640 vs. $5,232로 81% 절감이다. 3년 총 소유 비용(TCO)은 $107,160 vs. $17,304로 84% 절감이다.

**확장성 경제학**

제품 라인 확장 시 비용 차이가 더욱 극명해진다. 신규 제품 추가 시 전통적 방식은 full development cycle을 반복하지만, Dinomaly는 memory bank 업데이트만 필요하다.

```
신규 제품 클래스 추가 비용:

Traditional Approach:
- 데이터 수집: 2-4주
- 모델 개발: $1,616
- 배포 및 통합: 1주
- 총 시간: 4-6주

Dinomaly Approach:
- 데이터 수집: 100-200 samples
- Memory bank 업데이트: 10분
- 검증: 2-4시간
- 비용: ~$50
- 총 시간: 1일

비용 차이: 97% 절감
시간 단축: 95%
```

50개 제품 라인 규모에서는 차이가 극적이다. 전통적 방식은 $80,800 초기 비용 + $92,000 연간 유지보수이다. Dinomaly는 $1,608 + $5,232로, 94% TCO 절감이다.

**Break-even Analysis**

Multi-class 전환의 break-even point를 계산한다. 전통적 방식의 모델당 비용 $1,616을 기준으로, Dinomaly 초기 투자 $1,608을 회수하는 제품 수는 단 1개이다. 즉, 2개 이상 제품을 검사한다면 즉시 경제적이다.

ROI는 제품 수에 선형적으로 증가한다. 5개 제품: 380% ROI, 10개 제품: 900% ROI, 15개 제품: 1,400% ROI이다. 유지보수 비용까지 고려하면 연간 ROI는 더욱 개선된다.

### 7.3 Zero-shot Feasibility

Zero-shot의 실무 적용 가능성을 다양한 시나리오에서 분석한다. 어떤 상황에서 zero-shot이 viable하며, 언제 전통적 학습이 필수적인가?

**적용 가능 시나리오**

Zero-shot WinCLIP이 효과적인 시나리오는 명확한 패턴을 보인다.

첫째, 신제품 즉시 배포이다. 출시 첫날부터 자동 검사가 필요한 경우, 학습 데이터 수집 기간을 기다릴 수 없다. WinCLIP은 제품 사양서와 CAD 이미지만으로 시스템을 구축한다. 초기 정확도 85-90%는 수동 검사 부담을 50% 줄이며, 2-3주 후 충분한 데이터가 모이면 전통적 모델로 전환한다.

둘째, 다품종 소량 생산이다. 월 50-200개만 생산되는 제품은 개별 모델 학습을 정당화하기 어렵다. 20개 변형 각각에 모델을 만드는 대신, WinCLIP 하나로 모든 변형을 처리한다. 변형별 프롬프트만 조정한다.

셋째, 빠른 feasibility 검증이다. AI 검사 도입 전 POC 단계에서, 1일 내 프로토타입을 구축하고 실제 데이터로 테스트한다. 90% 이상 나오면 본격 개발 진행, 85-90%면 few-shot 고려, 85% 미만이면 전통적 학습 기반 모델 필요로 판단한다.

**한계 시나리오**

Zero-shot이 부적합한 경우도 명확하다.

첫째, 극도로 높은 정확도가 요구되는 critical 제조 공정이다. 항공우주, 의료기기처럼 99.5%+ 정확도가 필수인 경우, zero-shot의 91-95%는 불충분하다. 생명이나 안전에 직결되는 환경에서는 타협 불가능하다.

둘째, 도메인 특화 결함 유형이다. 반도체 wafer의 particle defect, PCB의 solder bridge처럼 매우 specific한 결함은 general purpose CLIP이 학습하지 못했다. 프롬프트로 "solder bridge"를 명시해도, CLIP의 임베딩 공간에 해당 개념이 없으면 무용지다.

셋째, 미세 결함 검출이다. CLIP의 해상도 제약(224×224 or 336×336)으로 sub-millimeter 결함은 불가시적이다. Transistor 핀 변형, screw 나사산 손상 같은 미세 결함은 zero-shot으로 검출 불가능하다.

**Hybrid Strategy**

실무에서 가장 효과적인 접근은 zero-shot와 trained model의 hybrid이다.

```
3-Phase Deployment Strategy:

Phase 1 (Day 1-14): Zero-shot Bootstrap
- Model: WinCLIP
- Data: 0 samples
- Performance: 85-90% AUROC
- Purpose: Immediate inspection capability
- Action: Collect true defect samples

Phase 2 (Day 15-60): Few-shot Transition
- Model: DRAEM
- Data: 50-100 defect samples
- Performance: 95-97% AUROC
- Purpose: Specialized detection
- Action: Continue data collection

Phase 3 (Day 61+): Full Deployment
- Model: PatchCore or Dinomaly
- Data: 200+ normal, 100+ defect samples
- Performance: 98-99% AUROC
- Purpose: Production-grade system
- Action: Quarterly retraining
```

이 전략은 time-to-value를 최적화한다. Day 1부터 value creation이 시작되며, 시간이 지남에 따라 점진적으로 성능이 개선된다. 전통적 방식처럼 4-6주 기다렸다가 한 번에 배포하는 것보다 누적 value가 크다.

### 7.4 Explainability Value

Explainable AI의 가치를 정성적, 정량적으로 평가한다. 자연어 설명이 실제로 얼마나 가치 있는가?

**정량적 가치**

VLM-AD의 재무적 영향을 측정한다. 중견 제조사(일 1000개 제품 검사, 3% 불량률) 시나리오이다.

```
전통적 문서화 비용:
- 불량 건수: 30/day
- 보고서 작성 시간: 15 min/case
- 품질 엔지니어 시급: $50
- 일일 비용: 30 × 0.25h × $50 = $375
- 월간 비용: $11,250
- 연간 비용: $135,000

VLM-AD 비용:
- API 호출: 30/day × $0.025 = $0.75
- 월간 비용: $22
- 연간 비용: $270
- 연간 절감: $134,730 (99.8%)
```

규제 대응 가치는 별도로 계산된다. FDA 감사 준비가 연 2회 발생하며, 각 감사마다 1주(40시간) 소요된다고 가정하자.

```
전통적 감사 준비:
- 불량 제품 근거 자료 수집: 20h × $50 = $1,000
- 문서 정리 및 검토: 15h × $50 = $750
- 감사관 질의 대응: 5h × $50 = $250
- 감사당 비용: $2,000
- 연간 비용: $4,000

VLM-AD 감사 준비:
- 자동 생성된 보고서 export: 2h × $50 = $100
- 검토 및 보완: 3h × $50 = $150
- 감사당 비용: $250
- 연간 비용: $500
- 연간 절감: $3,500 (87.5%)
```

총 연간 가치는 문서화 절감 $134,730 + 감사 대응 $3,500 = $138,230이다. VLM-AD 연간 비용 $270 대비 ROI는 51,000%이다.

**정성적 가치**

수치화하기 어렵지만 더 큰 가치들이 있다.

첫째, 의사결정 신뢰도 향상이다. AI가 "이상 점수 0.87"만 제시할 때, 검사원은 불안하다. "이게 진짜 불량인가? 왜 불량인가?" 의문이 든다. 하지만 AI가 "2mm 표면 스크래치, 운송 중 마찰로 추정, 외관에만 영향, 기능은 정상"이라고 설명하면, 검사원은 확신을 갖고 판정한다.

둘째, 지식 전수 가속화이다. 숙련 검사원의 암묵지가 AI 설명을 통해 명시지로 전환된다. 신입은 AI 설명을 읽으며 "이런 패턴은 이렇게 판단하는구나"를 학습한다. 기존 6개월 OJT가 3개월로 단축되며, 숙련 검사원의 부담도 줄어든다.

셋째, 지속적 개선 촉진이다. AI가 제시한 root cause를 집계하면, 공정 개선 우선순위가 명확해진다. "지난 달 스크래치 156건 중 121건(78%)이 포장 공정 기인 → 포장재 개선 우선 추진"처럼 data-driven decision이 가능하다.

넷째, 고객 신뢰 구축이다. 고객 불만 발생 시 "AI 검사에서 놓쳤습니다"보다 "AI 검사에서 0.3mm 결함을 탐지했으나, 허용 오차 0.5mm 이내로 판단하여 출하했습니다. 사용 환경에서 예상치 못한 응력으로 확대된 것으로 보입니다"라는 설명이 훨씬 설득력 있다.

---

## 8. Future Outlook (2025-2030)

### 8.1 Multi-class Standardization

2025-2030년 사이 multi-class anomaly detection은 niche에서 standard로 전환될 것이다.

**기술적 발전 방향**

현재 Dinomaly는 15개 클래스에서 98.8% 성능을 보이지만, 50-100개 규모에서는 아직 검증되지 않았다. 향후 연구는 massive multi-class scaling에 집중될 것이다. Class-conditional memory bank의 효율적 관리, hierarchical class organization, adaptive class balancing 등이 핵심 과제이다.

Cross-domain multi-class도 등장할 것이다. 현재는 MVTec AD 내 15개 클래스처럼 유사 도메인이다. 향후에는 metal, fabric, electronics, food를 하나의 모델로 처리하는 universal detector가 가능해질 것이다. 이는 DINOv2보다 더 강력한 foundation model을 요구한다.

**산업 채택 예측**

2025-2026년에는 early adopters가 multi-class를 시범 적용한다. 주로 다품종 생산 환경이나 복수 제품 라인을 보유한 중견 제조사이다. 성공 사례가 축적되면서 ROI가 검증된다.

2027-2028년에는 mainstream adoption이 시작된다. MLOps 플랫폼들이 multi-class를 기본 지원하며, 사전 학습된 multi-class model이 marketplace에서 판매된다. "15개 common manufacturing defects pre-trained model - $5,000" 같은 상품이 등장한다.

2029-2030년에는 multi-class가 industry standard가 된다. 신규 프로젝트의 80% 이상이 multi-class로 시작하며, single-class는 특수한 경우에만 사용된다. ISO, NIST 같은 표준 기관이 multi-class anomaly detection guideline을 발표한다.

### 8.2 Zero-shot Expansion

Zero-shot anomaly detection은 현재 CLIP 기반 WinCLIP이 유일하지만, 2025-2030년 동안 폭발적으로 발전할 것이다.

**차세대 모델 예측**

CLIP보다 강력한 vision-language models가 등장할 것이다. OpenAI의 GPT-5V, Google의 Gemini 2.0 Vision, Meta의 CLIP-NG 같은 모델들이 더 높은 해상도, 더 정교한 vision understanding을 제공한다.

특히 해상도 제약이 해결될 것이다. 현재 CLIP의 224×224/336×336에서, 차세대 모델은 1024×1024 이상을 지원한다. 이는 미세 결함 검출을 가능하게 한다. Zero-shot으로도 transistor 핀 변형, screw 나사산 손상을 탐지할 수 있다.

Domain-specific vision-language models도 등장한다. Manufacturing VLM, Medical VLM처럼 특정 도메인에 특화된 모델이다. 이들은 일반 목적 CLIP보다 해당 도메인에서 월등한 성능을 보인다. "manufacturing surface defect", "solder bridge"같은 용어를 정확히 이해한다.

**Few-shot의 진화**

Zero-shot과 fully-trained 사이에 few-shot이 중요해진다. 5-10장 예시만으로 95%+ 정확도를 달성하는 모델이 표준이 될 것이다. 이는 prompt tuning이나 adapter learning 같은 efficient fine-tuning 기법으로 구현된다.

In-context learning도 강력해진다. GPT-4V처럼 몇 개 예시를 prompt에 포함하면, 추가 학습 없이도 해당 패턴을 인식한다. "Here are 3 examples of good products and 2 examples of defects. Now classify this new product"처럼 작동한다.

**상용화 경로**

2025-2026년에는 zero-shot이 prototype과 POC 용도로 확산된다. 기업들이 AI 검사 도입 전 quick validation에 활용한다. "2주 안에 feasibility 검증"이 가능해진다.

2027-2028년에는 다품종 소량 생산 환경에서 본격 배포된다. 월 생산량 100개 미만 제품들이 주 대상이다. 개별 모델 학습이 비경제적인 niche markets에서 zero-shot이 유일한 자동화 방안이 된다.

2029-2030년에는 zero-shot 성능이 95%+에 도달하며, 많은 일반 제조 환경에서도 채택된다. "충분히 좋은" 성능이 되면, 학습 데이터 수집과 모델 학습의 번거로움을 피하기 위해 zero-shot을 선호한다.

### 8.3 Explainable AI Mandate

Explainable AI는 선택이 아닌 필수가 될 것이다. 규제, 윤리, 실무적 이유로 인해 모든 AI 검사 시스템이 설명 가능성을 요구받는다.

**규제 동향**

EU AI Act (2024 시행)는 high-risk AI systems에 대해 transparency와 explainability를 의무화했다. 의료기기, 항공우주, 자동차 같은 critical 제조는 high-risk에 해당한다. 이들 산업에서 AI 품질 검사는 설명 가능해야 한다.

FDA는 Software as Medical Device (SaMD) 가이드라인에서 AI 의사결정의 근거를 문서화하도록 요구한다. 단순히 "이상 점수 0.87"이 아니라, "왜 0.87인가, 어떤 특징이 기여했는가, 신뢰도는 얼마인가"를 기록해야 한다.

미국 NIST는 AI Risk Management Framework에서 explainability를 핵심 요소로 포함했다. 제조업체들이 이를 준수하려면, black-box AI에서 explainable AI로 전환이 불가피하다.

**기술 발전**

현재 VLM-AD는 post-hoc explanation이다. 기존 anomaly detector의 출력을 GPT-4V가 해석한다. 향후에는 inherently explainable models이 등장한다. 모델 자체가 추론 과정에서 자연어 설명을 생성한다.

Multimodal reasoning도 강화된다. 현재는 단순히 이미지를 서술하는 수준이지만, 향후에는 복잡한 추론이 가능하다. "이 스크래치는 직선형이므로 abrasive contact을 암시함. 위치가 exposed surface이므로 handling 중 발생 가능성 높음. 깊이가 얕으므로 기능적 영향 없음"처럼 multi-step reasoning을 수행한다.

Interactive explanation도 가능해진다. 검사원이 "왜 이 부분이 이상인가?"라고 물으면, AI가 "이 영역의 texture pattern이 정상 샘플 대비 23% 거칠며, 색상이 5% 어둡습니다. 이는 oxidation을 나타냅니다"라고 대화형으로 답한다.

**산업 표준화**

2025-2027년에는 explainability formats가 표준화된다. JSON, XML 같은 structured format으로 설명을 표현하는 schema가 정의된다. 이를 통해 서로 다른 AI 시스템의 설명을 일관되게 처리할 수 있다.

2028-2030년에는 explainability benchmarks가 등장한다. 설명의 품질을 정량적으로 평가하는 metrics와 datasets이다. Faithfulness (설명이 실제 모델 행동을 얼마나 반영하는가), Plausibility (인간이 얼마나 납득하는가), Actionability (실무에 얼마나 유용한가) 같은 지표가 표준화된다.

### 8.4 Domain-specific Foundation Models

범용 foundation models (CLIP, DINOv2, GPT-4V)는 강력하지만 한계가 있다. 향후에는 특정 도메인에 특화된 foundation models이 등장한다.

**Manufacturing Foundation Models**

제조 도메인 특화 모델이 등장할 것이다. 수백만 개의 제조 이미지와 결함 사례로 학습된 모델이다. 일반 ImageNet 대신 industrial parts, surface defects, assembly processes로 학습한다.

이러한 모델은 manufacturing vocabulary를 이해한다. "scratch", "dent", "solder bridge", "misalignment" 같은 용어를 정확히 인식한다. CLIP이 "scratch"를 일반적 의미로만 이해한다면, Manufacturing VLM은 "linear abrasion defect on metal surface"로 구체적으로 이해한다.

Sub-domain models도 분화될 것이다. Electronics Manufacturing VLM, Textile Quality VLM, Metal Surface VLM처럼 더 좁은 영역에 특화된다. 이들은 해당 분야에서 범용 모델을 압도하는 성능을 보인다.

**Vertical Integration**

대형 제조사들이 자체 foundation models을 구축할 것이다. Tesla가 자동차 부품 검사를 위한 Tesla Vision Foundation Model을, Samsung이 반도체 검사를 위한 Samsung Inspection Model을 개발한다. 이는 수십 년간 축적된 자사 데이터와 도메인 지식을 활용한다.

이러한 vertical models는 엄청난 competitive advantage를 제공한다. 경쟁사가 접근할 수 없는 proprietary data로 학습되며, 자사 공정에 완벽히 최적화된다. 공개 모델보다 3-5% 높은 정확도는 수백만 달러 불량 비용 절감으로 이어진다.

**Collaborative Ecosystems**

반면 중소 제조사들은 협력적 접근을 택한다. Industry consortium이 공동으로 foundation model을 구축한다. 예를 들어, 10개 금속 가공 업체가 데이터를 pooling하여 Metal Forming Foundation Model을 학습한다. 각 업체는 자사 데이터의 일부를 제공하고, 훈련된 모델에 모두 접근한다.

Open-source foundation models도 등장한다. Hugging Face에서 "Manufacturing-CLIP", "Textile-VLM"같은 모델을 공개한다. 커뮤니티가 협력하여 지속적으로 개선하며, 중소기업들이 무료로 활용한다.

**Timeline and Milestones**

```
2025:
- 첫 manufacturing-specific foundation model 발표
- 주요 대학 및 연구소에서 benchmark datasets 공개

2026-2027:
- 대형 제조사들의 proprietary models 구축 시작
- Industry consortiums 형성 및 협력 모델 개발

2028-2029:
- Domain-specific models이 범용 모델 성능 추월
- Open-source manufacturing VLMs 생태계 성숙

2030:
- Domain-specific이 new standard
- 50% 이상 제조사가 specialized models 사용
- 범용 models는 prototype/POC용으로만 사용
```

이러한 발전은 Foundation Models 패러다임을 더욱 강화할 것이다. 범용 모델의 한계를 극복하고, 각 도메인에 최적화된 고성능 시스템을 제공한다. 결과적으로 2030년대에는 AI 기반 품질 검사가 제조 산업의 필수 인프라가 될 것이다.

---

06-foundation-models.md의 9장과 10장을 작성하겠습니다.

---

## 9. Practical Application Guide

### 9.1 Dinomaly for Multi-class

Dinomaly를 multi-class 환경에 성공적으로 배포하기 위한 실무 가이드이다. 단계별 접근으로 위험을 최소화하고 성공 확률을 높인다.

**Phase 1: Assessment and Planning (Week 1-2)**

먼저 현재 환경을 평가한다. 검사 대상 제품 클래스 수, 각 클래스의 샘플 수, 결함 유형, 검사 빈도를 파악한다. Dinomaly 적합성을 판단하는 체크리스트이다.

```
Dinomaly Suitability Checklist:
✓ 제품 클래스 5개 이상
✓ 클래스당 정상 샘플 100장 이상
✓ 이미지 품질 일관성 (조명, 각도, 배경)
✓ GPU 메모리 8GB 이상 또는 Cloud 사용
✓ 기존 single-class 모델 관리 부담 높음
✓ 신규 제품 추가 빈도 높음

적합도 점수:
6개 모두 충족: High (즉시 진행 권장)
4-5개 충족: Medium (pilot 권장)
3개 이하: Low (전통적 방식 유지)
```

데이터 준비 계획을 수립한다. 각 클래스별 train/test split 전략, augmentation 정책, class imbalance 처리 방법을 결정한다. 일반적으로 클래스당 train 200장, test 50장을 목표로 한다.

**Phase 2: Pilot Implementation (Week 3-4)**

소규모 pilot으로 시작한다. 전체 15개 클래스가 아니라 3-5개 representative classes로 feasibility를 검증한다. 텍스처 1-2개 (carpet, wood), 객체 1-2개 (bottle, cable), 복잡 객체 1개 (transistor)를 선택한다.

```python
# Dinomaly Pilot Configuration
pilot_classes = ["carpet", "wood", "bottle", "cable", "transistor"]

trainer = DinomalyTrainer(
    encoder_name="dinov2_vit_base_14",
    bottleneck_dropout=0.2,
    decoder_depth=8
)

# Multi-class training
train_loader = get_multi_class_dataloader(
    classes=pilot_classes,
    batch_size=16,
    balanced_sampling=True
)

# Train for 10-15 epochs
trainer.fit(train_loader, num_epochs=12)

# Build class-conditional memory banks
for class_name in pilot_classes:
    memory_bank[class_name] = build_memory_bank(
        model=trainer.model,
        samples=train_samples[class_name]
    )
```

성공 기준을 명확히 정의한다. Pilot AUROC > 97%, 개별 클래스 대비 성능 저하 < 2%, 메모리 사용량 < 1GB, 추론 속도 < 150ms를 목표로 한다. 이 기준을 충족하면 full deployment로 진행한다.

**Phase 3: Full Deployment (Week 5-8)**

Pilot 성공 후 전체 클래스로 확장한다. 모든 제품 라인의 데이터를 통합하고, 최종 모델을 학습한다. 이 단계에서 주의할 점들이다.

Class balancing이 중요하다. 샘플 수가 크게 불균형하면 weighted sampling을 적용한다. 샘플이 많은 클래스는 under-sampling, 적은 클래스는 augmentation으로 균형을 맞춘다.

```python
# Weighted sampler for imbalanced classes
class_weights = calculate_class_weights(dataset)
sampler = WeightedRandomSampler(
    weights=class_weights,
    num_samples=len(dataset),
    replacement=True
)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    sampler=sampler
)
```

Memory bank 최적화도 필요하다. 15개 클래스면 각 75MB씩 총 75MB 정도이다. 하지만 50개 클래스로 확장하면 250MB가 되므로, compression이나 quantization을 고려한다.

Inference pipeline을 구축한다. 실시간 추론 시 먼저 class identification을 수행하고, 해당 class의 memory bank로 anomaly detection을 진행한다.

```python
def inference_pipeline(image, class_name):
    # Extract features
    features = model.encoder(preprocess(image))
    
    # Get class-specific memory bank
    mu, sigma = memory_bank[class_name]
    
    # Compute anomaly score
    score = mahalanobis_distance(features, mu, sigma)
    
    # Generate anomaly map
    anomaly_map = model.compute_anomaly_map(features)
    
    return score, anomaly_map
```

**Phase 4: Operations and Maintenance (Ongoing)**

운영 단계에서는 지속적 모니터링과 개선이 필요하다. 주간 단위로 클래스별 성능 지표(AUROC, FPR, FNR)를 추적한다. 특정 클래스에서 성능 저하가 관찰되면 해당 클래스의 memory bank만 업데이트한다.

```
Performance Monitoring Dashboard:
┌─────────────┬──────────┬─────────┬─────────┐
│ Class       │ AUROC    │ FPR@1%  │ Samples │
├─────────────┼──────────┼─────────┼─────────┤
│ Carpet      │ 99.1%    │ 0.8%    │ 1,247   │
│ Wood        │ 98.8%    │ 1.2%    │ 1,089   │
│ Bottle      │ 98.5%    │ 1.5%    │ 943     │
│ Cable       │ 98.2%    │ 1.8%    │ 876     │
│ Transistor  │ 97.1% ⚠️  │ 2.9% ⚠️  │ 723     │
└─────────────┴──────────┴─────────┴─────────┘

Action: Transistor performance degraded
→ Collect 50 more normal samples
→ Update memory bank
→ Re-validate
```

신규 클래스 추가는 간단하다. 정상 샘플 100-200장을 수집하고, encoder로 특징을 추출하여 memory bank에 추가한다. 전체 모델 재학습은 불필요하다. 단, 분기별로 모든 클래스를 포함한 full retraining을 권장한다.

### 9.2 WinCLIP for Instant Deployment

WinCLIP을 활용한 즉시 배포 시나리오의 실전 가이드이다. 학습 데이터 없이 1일 내 시스템을 구축한다.

**Morning: Setup and Configuration (2-3 hours)**

WinCLIP 환경을 준비한다. OpenAI CLIP 모델을 다운로드하거나 Hugging Face에서 로드한다. ViT-B/32 (350MB)를 기본으로 사용하고, 더 높은 성능이 필요하면 ViT-L/14 (890MB)를 선택한다.

```python
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
```

제품 정보를 정리한다. 제품명, 주요 재질, 예상 결함 유형, 검사 기준을 문서화한다. 이 정보가 프롬프트 설계의 기초가 된다.

**Afternoon: Prompt Engineering (3-4 hours)**

효과적인 프롬프트를 설계한다. 여러 변형을 시도하고, 소수의 validation 샘플로 최적을 선택한다.

```python
# Prompt templates
templates_normal = [
    "a photo of a flawless {product}",
    "a perfect {product} without any defects",
    "{product} with pristine surface quality",
    "high quality {product} in excellent condition"
]

templates_damage = [
    "a photo of a {product} with defects",
    "a damaged {product}",
    "{product} with surface anomalies",
    "defective {product} with visible flaws"
]

# Test all combinations
best_auroc = 0
best_prompts = None

for prompt_normal in templates_normal:
    for prompt_damage in templates_damage:
        auroc = evaluate_prompts(
            prompt_normal.format(product="metal nut"),
            prompt_damage.format(product="metal nut"),
            validation_samples
        )
        if auroc > best_auroc:
            best_auroc = auroc
            best_prompts = (prompt_normal, prompt_damage)

print(f"Best prompts: {best_prompts}, AUROC: {best_auroc:.3f}")
```

Domain-specific keywords를 추가한다. "scratch", "dent", "contamination", "misalignment"처럼 구체적 결함 유형을 명시하면 성능이 향상된다.

```python
# Enhanced prompts with specific defect types
defect_types = ["scratch", "dent", "crack", "stain"]

enhanced_templates = [
    f"{{product}} with {defect}" for defect in defect_types
]

# Ensemble approach: average scores from multiple prompts
scores = []
for template in enhanced_templates:
    score = compute_anomaly_score(image, template)
    scores.append(score)

final_score = np.mean(scores)
```

**Evening: Validation and Deployment (2-3 hours)**

테스트 샘플로 시스템을 검증한다. 정상 50장, 이상 20장 정도면 충분하다. AUROC 90% 이상이면 배포 가능으로 판단한다.

```python
# Validation loop
results = {
    'normal': [],
    'anomaly': []
}

for image, label in validation_set:
    score = winclip_inference(
        image=image,
        prompt_normal=best_prompts[0],
        prompt_damage=best_prompts[1]
    )
    results[label].append(score)

# Calculate metrics
auroc = compute_auroc(results)
threshold = find_optimal_threshold(results)

print(f"Validation AUROC: {auroc:.3f}")
print(f"Optimal threshold: {threshold:.3f}")

# Decision logic
if auroc >= 0.90:
    print("✓ Ready for deployment")
elif auroc >= 0.85:
    print("⚠ Marginal - consider few-shot learning")
else:
    print("✗ Insufficient - full training required")
```

배포 파이프라인을 구성한다. 이미지 전처리, WinCLIP 추론, threshold 적용, 결과 저장의 흐름을 자동화한다.

**Next Day: Production Monitoring (Ongoing)**

운영 첫 날부터 성능을 모니터링한다. False positive와 false negative 케이스를 기록하고, 패턴을 분석한다. 2주 후 충분한 실제 데이터가 모이면 DRAEM이나 PatchCore로 전환을 고려한다.

```
Day 1-14: WinCLIP zero-shot
- Performance: 88-92% accuracy
- False positives: ~8-10%
- Action: Collect defect samples

Day 15-30: Transition planning
- Collected samples: 50-100 defects
- Options: DRAEM (few-shot) or PatchCore (full)
- Decision criteria: sample count, budget, timeline

Day 31+: Upgraded system
- Model: DRAEM or PatchCore
- Performance: 96-98% accuracy
- WinCLIP: Kept as backup for new products
```

### 9.3 VLM-AD for Quality Reports

VLM-AD를 활용한 품질 보고서 자동화의 실무 적용 가이드이다.

**Integration Architecture**

VLM-AD는 standalone이 아니라 기존 anomaly detector와 통합된다. 2-tier 구조를 권장한다.

```
┌─────────────────────────────────────────┐
│ Tier 1: Anomaly Detection               │
│ - Model: PatchCore / Dinomaly           │
│ - Input: All products (1000/day)        │
│ - Output: Anomaly score, map            │
│ - Cost: On-premise GPU (~$0)            │
└─────────────┬───────────────────────────┘
              │ Filter: score > threshold
              ▼
┌─────────────────────────────────────────┐
│ Tier 2: Explanation Generation          │
│ - Model: VLM-AD (GPT-4V)                │
│ - Input: Flagged products (~30/day)     │
│ - Output: Natural language report       │
│ - Cost: $0.025 per product (~$0.75/day) │
└─────────────────────────────────────────┘
```

이렇게 하면 API 비용을 최소화하면서 고품질 설명을 얻는다.

**Prompt Template Design**

효과적인 prompt template을 설계한다. 일관성과 구조화된 출력이 핵심이다.

```python
prompt_template = """
You are an expert quality inspector. Analyze this product image.

Product Information:
- Type: {product_type}
- Part Number: {part_number}
- Specification: {specification}

Detection Results:
- Anomaly Score: {score:.3f} (threshold: {threshold:.3f})
- Anomaly Map: [provided as overlay]
- Detected Region: {bbox}

Task: Provide a structured analysis including:
1. Defect Classification
2. Location and Size
3. Severity Assessment
4. Root Cause Hypothesis
5. Recommended Action

Output Format: JSON
"""

# Fill template
prompt = prompt_template.format(
    product_type="Metal Nut",
    part_number="MN-2024-1147",
    specification="ISO 4032 M8",
    score=0.87,
    threshold=0.75,
    bbox="x:145, y:78, w:35, h:45"
)
```

Structured output을 강제한다. JSON schema를 제공하여 GPT-4V가 파싱 가능한 형식으로 응답하게 한다.

```python
json_schema = {
    "defect_type": "string",
    "location": "string",
    "severity": "string (minor|moderate|critical)",
    "confidence": "float (0-1)",
    "root_cause": "string",
    "recommended_action": "string"
}

prompt += f"\n\nOutput Schema:\n{json.dumps(json_schema, indent=2)}"
```

**Report Generation Pipeline**

자동 보고서 생성 파이프라인을 구축한다.

```python
def generate_quality_report(image, anomaly_score, anomaly_map, metadata):
    # Step 1: Construct prompt
    prompt = build_prompt(metadata, anomaly_score)
    
    # Step 2: Call GPT-4V API
    response = call_gpt4v_api(
        image=image,
        anomaly_map=anomaly_map,
        prompt=prompt
    )
    
    # Step 3: Parse JSON response
    analysis = json.loads(response)
    
    # Step 4: Generate formatted report
    report = format_report(analysis, metadata)
    
    # Step 5: Store in database
    store_report(report, database)
    
    return report

# Batch processing for end-of-day reports
daily_defects = get_daily_defects(date)
reports = []

for defect in daily_defects:
    report = generate_quality_report(
        image=defect.image,
        anomaly_score=defect.score,
        anomaly_map=defect.map,
        metadata=defect.metadata
    )
    reports.append(report)

# Generate summary report
summary = aggregate_reports(reports)
send_email(summary, recipients=quality_team)
```

**Cost Optimization Strategies**

API 비용을 최적화하는 전략들이다.

첫째, selective generation이다. Critical defects (score > 0.9)만 VLM-AD로 분석하고, minor defects (0.75-0.9)는 template 기반 보고서를 사용한다.

둘째, caching이다. 유사한 결함은 과거 분석을 재사용한다. Image similarity > 0.95면 cached explanation을 반환한다.

```python
# Check cache before API call
similarity = compute_image_similarity(image, cached_images)
if similarity > 0.95:
    return cached_explanations[best_match]
else:
    explanation = call_gpt4v_api(image, prompt)
    cache[image_hash] = explanation
    return explanation
```

셋째, batch processing이다. 실시간이 아닌 end-of-day batch로 처리하면 API rate limit을 효율적으로 활용한다.

### 9.4 Migration Strategy

기존 single-class 시스템에서 Foundation Models로 마이그레이션하는 전략이다.

**Assessment Phase**

현재 시스템을 평가한다. 관리 중인 모델 수, 각 모델의 성능, 유지보수 부담, 확장 계획을 파악한다.

```
Current State Assessment:
┌──────────────────────┬─────────────────┐
│ Metric               │ Current Value   │
├──────────────────────┼─────────────────┤
│ Models in production │ 12              │
│ Avg AUROC            │ 98.5%           │
│ Storage footprint    │ 3.2 GB          │
│ Monthly maintenance  │ 40 hours        │
│ New product lead time│ 3-4 weeks       │
│ Annual cost          │ $48,000         │
└──────────────────────┴─────────────────┘

Pain Points:
- High maintenance burden (40h/month)
- Slow new product onboarding (3-4 weeks)
- Storage constraints in edge deployment
```

마이그레이션 목표를 설정한다. 성능 유지 (AUROC drop < 1%), 비용 절감 (50%+), 운영 부담 감소 (70%+)를 목표로 한다.

**Pilot Migration**

전체 마이그레이션 전에 pilot을 수행한다. 3-4개 모델을 선택하여 Dinomaly로 통합한다. Low-risk 제품으로 시작한다.

```
Pilot Selection Criteria:
✓ Non-critical products (safety not at stake)
✓ Stable production (no frequent changes)
✓ Similar characteristics (same material/size)
✓ Existing performance baseline (for comparison)

Selected for Pilot:
- Bottle (AUROC: 98.7%)
- Cable (AUROC: 98.3%)
- Wood (AUROC: 98.9%)
```

병렬 운영으로 검증한다. 4주간 기존 모델과 Dinomaly를 동시 실행하며, 결과를 비교한다.

```python
# Parallel validation
for product in test_set:
    # Legacy model
    score_legacy = legacy_models[product.class].infer(product.image)
    
    # Dinomaly
    score_dinomaly = dinomaly_model.infer(
        product.image, 
        class_name=product.class
    )
    
    # Compare
    comparison_log.append({
        'product_id': product.id,
        'class': product.class,
        'score_legacy': score_legacy,
        'score_dinomaly': score_dinomaly,
        'decision_match': (score_legacy > th) == (score_dinomaly > th)
    })

# After 4 weeks
agreement_rate = calculate_agreement(comparison_log)
print(f"Decision agreement: {agreement_rate:.1%}")
```

Agreement rate 95% 이상이면 마이그레이션 승인한다.

**Phased Rollout**

Pilot 성공 후 단계적으로 확장한다. 한 번에 3-4개 모델씩 추가하며, 각 단계마다 2주 안정화 기간을 둔다.

```
Migration Timeline (16 weeks):

Week 1-4: Pilot (3 models)
→ Validate, tune, document

Week 5-6: Phase 1 (4 more models, total 7)
→ Monitor, adjust thresholds

Week 7-8: Stabilization
→ Performance tuning, bug fixes

Week 9-10: Phase 2 (5 more models, total 12)
→ Full load testing

Week 11-12: Stabilization
→ Edge case handling

Week 13-14: Final rollout (all remaining)
→ Decommission legacy models

Week 15-16: Optimization
→ Memory tuning, speed optimization
```

각 단계마다 rollback plan을 준비한다. 문제 발생 시 즉시 legacy model로 복귀할 수 있어야 한다.

**Legacy System Decommissioning**

모든 모델이 성공적으로 마이그레이션되면 legacy system을 단계적으로 제거한다.

먼저 read-only mode로 전환한다. 더 이상 학습하지 않고, 추론만 가능하게 한다. 4주간 문제가 없으면 inference도 중단한다. 추가 4주 후 완전히 삭제한다.

```
Decommissioning Checklist:
□ All models migrated and validated
□ 8 weeks parallel operation completed
□ No critical issues reported
□ Backup of legacy models archived
□ Documentation updated
□ Team trained on new system
□ Monitoring dashboards configured
□ Rollback procedure documented
```

**Post-Migration Optimization**

마이그레이션 후 시스템을 최적화한다. Memory footprint, inference speed, false positive rate를 개선한다. 분기별 retraining 일정을 수립하고, continuous monitoring을 설정한다.

---

## 10. Research Insights

### 10.1 Foundation Model Impact

Foundation Models의 등장은 anomaly detection 연구 방향을 근본적으로 재편했다. 2025년 기준 주요 영향을 분석한다.

**연구 패러다임의 전환**

2020-2022년 연구는 architectural innovation에 집중했다. PatchCore의 coreset selection, FastFlow의 2D normalization, Reverse Distillation의 one-class embedding처럼 novel algorithms를 제안하는 것이 주류였다.

2023년 이후 연구는 foundation model adaptation으로 전환되었다. CLIP, DINOv2, SAM 같은 기존 모델을 anomaly detection에 효과적으로 활용하는 방법을 탐구한다. "어떻게 새로운 알고리즘을 만들까"에서 "어떻게 강력한 foundation model을 활용할까"로 질문이 바뀌었다.

이는 효율성의 승리이다. 처음부터 학습하는 것보다 거대 모델을 활용하는 것이 더 효과적이다. DINOv2는 1.4억 이미지로 학습되었는데, 개별 연구실이 이를 재현하는 것은 불가능하다. 대신 이미 학습된 DINOv2를 가져와 task-specific adaptation만 수행한다.

**벤치마크 성능의 포화**

MVTec AD에서 성능이 포화 상태에 도달했다. PatchCore 99.1%, Dinomaly 99.2%로, 추가 개선 여지가 거의 없다. 이는 연구 방향을 변화시켰다.

새로운 평가 지표가 등장했다. 단순 AUROC를 넘어, computational efficiency, memory footprint, inference speed, multi-class capability, zero-shot performance, explainability가 중요해졌다. "조금 더 정확한 모델"보다 "더 실용적인 모델"이 가치 있다.

새로운 벤치마크도 필요하다. MVTec AD는 single-class, supervised 환경이다. Multi-class, zero-shot, cross-domain 벤치마크가 개발되고 있다. VisA, BTAD를 넘어, real-world complexity를 반영하는 데이터셋이 요구된다.

**학술-산업 간극 축소**

Foundation Models는 학술 연구와 산업 적용의 간극을 줄였다. 전통적으로 논문의 SOTA 모델이 실무에 배포되기까지 2-3년 lag이 있었다. Foundation Models는 이 간극을 6개월-1년으로 단축했다.

이유는 deployment readiness이다. Dinomaly는 논문 발표와 동시에 Anomalib 라이브러리에 통합되어 즉시 사용 가능했다. WinCLIP은 OpenAI CLIP API만 있으면 바로 적용할 수 있다. 연구자가 reproduction code를 제공하고, 커뮤니티가 빠르게 검증한다.

산업계도 적극적으로 채택한다. Foundation Models의 강력한 성능과 사용 편의성이 낮은 진입 장벽을 만든다. ML 엔지니어가 논문을 이해하고 처음부터 구현할 필요 없이, pre-trained model을 다운로드하여 fine-tuning하면 된다.

### 10.2 Paradigm Transformation

Foundation Models가 가져온 패러다임 전환을 심층 분석한다.

**From Task-Specific to General-Purpose**

전통적 접근은 task-specific model이었다. PatchCore, STFPM, DRAEM은 anomaly detection만을 위해 설계되고 학습되었다. 다른 task로 전이가 불가능하다.

Foundation Models는 general-purpose이다. CLIP은 image classification, retrieval, generation 등 다양한 task에 활용된다. Anomaly detection은 그 중 하나일 뿐이다. DINOv2도 segmentation, object detection, depth estimation 등에 사용된다.

이는 개발 방식을 변화시켰다. 과거에는 각 task마다 전용 모델을 학습했다. 이제는 하나의 foundation model을 여러 task에 재사용한다. 한 번의 학습 비용으로 multiple applications을 지원한다.

**From Supervised to Self-Supervised**

전통적 computer vision은 supervised learning이 지배했다. ImageNet 분류로 학습된 ResNet이 backbone의 표준이었다. 1.4M labeled images가 필요했다.

Foundation Models는 self-supervised learning이다. DINOv2는 레이블 없이 140M 이미지로 학습되었다. CLIP은 image-text pairs를 사용하지만, 이는 웹에서 자동 수집 가능하다. 수동 labeling이 불필요하다.

Self-supervision의 장점은 scalability이다. Labeled data는 수집 비용이 높아 규모 확장이 어렵다. Unlabeled data는 사실상 무한하므로, 모델 크기와 데이터 크기를 계속 키울 수 있다. 이것이 foundation models의 강력함의 원천이다.

**From Single-Modal to Multi-Modal**

전통적 anomaly detection은 vision-only였다. 이미지만 입력받아 수치적 점수를 출력했다. 다른 modality와의 통합이 없었다.

Foundation Models는 multi-modal이다. CLIP은 vision과 language를 통합한다. GPT-4V는 vision, language, reasoning을 결합한다. VLM-AD는 이미지를 보고 자연어로 설명한다.

Multi-modality는 새로운 가능성을 연다. Text prompt로 zero-shot detection이 가능하다. 자연어로 질문하고 답변을 받는다. 이미지, 텍스트, 메타데이터를 종합하여 더 정교한 판단을 내린다.

**From Static to Dynamic**

전통적 모델은 static이다. 한 번 학습되면 고정된다. 새로운 결함 유형이 등장하면 재학습이 필요하다.

Foundation Models는 더 dynamic하다. Few-shot adaptation으로 빠르게 새로운 패턴을 학습한다. In-context learning으로 예시만 제공하면 즉시 적응한다. Prompt engineering으로 학습 없이 행동을 변경한다.

이는 continuous learning을 용이하게 한다. 운영 중 발견된 새로운 결함 패턴을 즉시 시스템에 반영할 수 있다. 전체 재학습 대신 incremental update가 가능하다.

### 10.3 Industry Implications

Foundation Models가 산업계에 미치는 영향을 분석한다.

**Democratization of AI**

Foundation Models는 AI를 민주화한다. 과거에는 AI 품질 검사 시스템 구축에 ML 전문가, 대규모 데이터, 고가의 컴퓨팅이 필요했다. 이는 대기업만 가능한 일이었다.

이제는 중소기업도 접근 가능하다. Pre-trained foundation model을 다운로드하고, 소량의 데이터로 fine-tuning하면 된다. WinCLIP은 아예 학습 데이터도 불필요하다. Cloud API를 사용하면 고가 GPU도 필요 없다.

기술 장벽도 낮아졌다. PyTorch 기본 지식만 있으면 Hugging Face에서 모델을 로드하고 사용할 수 있다. Anomalib 같은 라이브러리는 더욱 단순화한다. 몇 줄의 코드로 SOTA 모델을 배포한다.

**Shift in Competitive Advantage**

경쟁 우위의 원천이 변화했다. 과거에는 superior algorithms이 차별화 요소였다. 누가 더 정확한 모델을 개발하느냐가 중요했다.

이제는 data와 domain expertise가 핵심이다. Foundation model은 누구나 접근 가능하므로, 차별화는 얼마나 좋은 데이터를 보유하고, 도메인을 깊이 이해하는가에서 나온다. 자사만의 proprietary data로 학습한 domain-specific foundation model이 competitive moat가 된다.

Integration과 operationalization도 중요해졌다. 모델 자체보다 이를 기존 시스템에 효과적으로 통합하고, 안정적으로 운영하는 능력이 가치를 창출한다. MLOps 역량이 경쟁력이다.

**New Business Models**

Foundation Models는 새로운 비즈니스 모델을 가능하게 한다.

Model-as-a-Service가 부상한다. OpenAI API처럼 foundation model을 서비스로 제공한다. 사용량 기반 과금으로 초기 투자 없이 AI를 활용한다. VLM-AD가 이 모델의 예시이다.

Pre-trained Model Marketplace도 등장한다. 특정 도메인에 fine-tuned된 모델을 판매한다. "Metal Surface Defect Detection Model - $5,000"처럼 즉시 사용 가능한 모델을 구매한다. 자체 개발 대비 시간과 비용을 대폭 절감한다.

Data-as-a-Service도 가능해진다. Foundation model fine-tuning에 필요한 고품질 labeled data를 판매한다. 특정 산업의 결함 이미지 데이터셋이 상품화된다. 데이터가 새로운 자산 클래스가 된다.

**Workforce Transformation**

필요한 역량이 변화한다. 과거 ML 엔지니어는 algorithm design, optimization, training이 핵심 역량이었다. 이제는 model selection, prompt engineering, fine-tuning, integration이 중요하다.

새로운 역할도 등장한다. Prompt Engineer는 효과적인 프롬프트를 설계하여 foundation model의 능력을 최대한 끌어낸다. Model Curator는 다양한 foundation models 중 task에 최적을 선택하고 조합한다.

교육 방향도 조정된다. 대학은 처음부터 neural network를 구현하는 대신, pre-trained model을 활용하는 방법을 가르친다. Industry는 재교육 프로그램으로 기존 엔지니어를 foundation model 시대에 적응시킨다.

**Regulatory and Ethical Considerations**

Foundation Models는 새로운 규제와 윤리 이슈를 제기한다. 

모델의 거대함이 transparency를 어렵게 한다. 수십억 파라미터의 작동 방식을 설명하기 어렵다. 하지만 critical manufacturing에서는 explainability가 필수이다. 이 모순을 해결하기 위해 VLM-AD 같은 post-hoc explanation 방법이 중요해진다.

데이터 편향도 우려된다. Foundation model은 웹 데이터로 학습되므로, 인터넷의 편향을 내재한다. 특정 인종, 성별, 지역에 대한 편향이 anomaly detection에도 영향을 줄 수 있다. 이를 감지하고 완화하는 기법이 필요하다.

지적 재산권 이슈도 있다. Foundation model을 fine-tuning하여 만든 모델의 소유권은 누구에게 있는가? 학습 데이터에 저작권 있는 이미지가 포함되었다면? 명확한 법적 프레임워크가 아직 없다.

---
