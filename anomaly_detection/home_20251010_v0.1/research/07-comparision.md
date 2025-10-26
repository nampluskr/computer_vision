<artifact identifier="07-comparison-v1-revised" type="application/vnd.ant.code" language="markdown" title="07-comparison.md (v1.0)">
# Comprehensive Paradigm Comparison and Application Guide

**Document Version**: 1.0  
**Last Updated**: 2025  
**Models Compared**: 21  
**Paradigms**: 6

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
   - 1.1 [Six Paradigms at a Glance](#11-six-paradigms-at-a-glance)
   - 1.2 [Top Recommendations by Scenario](#12-top-recommendations-by-scenario)
   - 1.3 [Decision Flowchart](#13-decision-flowchart)

2. [Paradigm-by-Paradigm Evaluation](#2-paradigm-by-paradigm-evaluation)
   - 2.1 [Memory-Based Methods](#21-memory-based-methods)
   - 2.2 [Normalizing Flow Methods](#22-normalizing-flow-methods)
   - 2.3 [Knowledge Distillation Methods](#23-knowledge-distillation-methods)
   - 2.4 [Reconstruction-Based Methods](#24-reconstruction-based-methods)
   - 2.5 [Feature Adaptation Methods](#25-feature-adaptation-methods)
   - 2.6 [Foundation Model Methods](#26-foundation-model-methods)

3. [Performance Benchmarks](#3-performance-benchmarks)
   - 3.1 [MVTec AD Results](#31-mvtec-ad-results)
   - 3.2 [Category-wise Performance](#32-category-wise-performance)
   - 3.3 [Speed Benchmarks](#33-speed-benchmarks)
   - 3.4 [Memory Usage Analysis](#34-memory-usage-analysis)
   - 3.5 [Benchmark Limitations](#35-benchmark-limitations)

4. [Trade-off Analysis](#4-trade-off-analysis)
   - 4.1 [Accuracy vs Speed](#41-accuracy-vs-speed)
   - 4.2 [Accuracy vs Memory](#42-accuracy-vs-memory)
   - 4.3 [Speed vs Memory](#43-speed-vs-memory)
   - 4.4 [Three-way Trade-off Visualization](#44-three-way-trade-off-visualization)
   - 4.5 [Impossible Combinations](#45-impossible-combinations)

5. [Scenario-Based Selection Guide](#5-scenario-based-selection-guide)
   - 5.1 [Maximum Accuracy (>99%)](#51-maximum-accuracy-99)
   - 5.2 [Real-time Processing (<10ms)](#52-real-time-processing-10ms)
   - 5.3 [Multi-class Environment](#53-multi-class-environment)
   - 5.4 [Instant Deployment (Zero-shot)](#54-instant-deployment-zero-shot)
   - 5.5 [Few-shot Learning (10-50 samples)](#55-few-shot-learning-10-50-samples)
   - 5.6 [Quality Report Automation](#56-quality-report-automation)
   - 5.7 [Balanced General Inspection](#57-balanced-general-inspection)

6. [Hardware Environment Guide](#6-hardware-environment-guide)
   - 6.1 [GPU Server (8GB+ VRAM)](#61-gpu-server-8gb-vram)
   - 6.2 [Edge GPU (4GB VRAM)](#62-edge-gpu-4gb-vram)
   - 6.3 [CPU Only](#63-cpu-only)
   - 6.4 [Cloud/API](#64-cloudapi)

7. [Development Roadmap](#7-development-roadmap)
   - 7.1 [Phase 1: Prototyping (1-2 weeks)](#71-phase-1-prototyping-1-2-weeks)
   - 7.2 [Phase 2: Optimization (2-4 weeks)](#72-phase-2-optimization-2-4-weeks)
   - 7.3 [Phase 3: Deployment Preparation (2-3 weeks)](#73-phase-3-deployment-preparation-2-3-weeks)
   - 7.4 [Phase 4: Operations (Continuous)](#74-phase-4-operations-continuous)

8. [Cost-Benefit Analysis](#8-cost-benefit-analysis)
   - 8.1 [Initial Development Costs](#81-initial-development-costs)
   - 8.2 [Operational Costs (Monthly)](#82-operational-costs-monthly)
   - 8.3 [ROI Analysis](#83-roi-analysis)
   - 8.4 [Long-term Benefits](#84-long-term-benefits)

9. [Decision Framework](#9-decision-framework)
   - 9.1 [Decision Tree](#91-decision-tree)
   - 9.2 [Checklist-based Selection](#92-checklist-based-selection)
   - 9.3 [Multi-criteria Decision Matrix](#93-multi-criteria-decision-matrix)

10. [Industry Applications](#10-industry-applications)
    - 10.1 [Semiconductor](#101-semiconductor)
    - 10.2 [Medical Devices](#102-medical-devices)
    - 10.3 [Automotive](#103-automotive)
    - 10.4 [Electronics](#104-electronics)
    - 10.5 [Display Quality (OLED)](#105-display-quality-oled)

11. [Common Pitfalls](#11-common-pitfalls)
    - 11.1 [Wrong Model Selection](#111-wrong-model-selection)
    - 11.2 [Inadequate Data Preparation](#112-inadequate-data-preparation)
    - 11.3 [Hyperparameter Mistakes](#113-hyperparameter-mistakes)
    - 11.4 [Deployment Issues](#114-deployment-issues)

12. [Migration Strategies](#12-migration-strategies)
    - 12.1 [From Prototype to Production](#121-from-prototype-to-production)
    - 12.2 [Model Upgrade Paths](#122-model-upgrade-paths)
    - 12.3 [Multi-model Ensemble](#123-multi-model-ensemble)

13. [Future-Proofing](#13-future-proofing)
    - 13.1 [Technology Trends](#131-technology-trends)
    - 13.2 [When to Upgrade](#132-when-to-upgrade)
    - 13.3 [Continuous Improvement](#133-continuous-improvement)

[References](#references)

---

## 1. Executive Summary

### 1.1 Six Paradigms at a Glance

Vision anomaly detection 분야는 6개의 주요 패러다임으로 구성되며, 각 패러다임은 독특한 강점과 적용 시나리오를 가진다.

Memory-Based 방식은 정상 샘플의 특징 벡터를 메모리에 저장하고 거리를 직접 비교한다. PatchCore가 대표 모델로 99.1%의 최고 정확도를 달성한다. 추론 속도는 50-100ms로 중간 수준이며, coreset selection으로 메모리를 100-500MB로 효율화했다. 최고 정확도가 필요한 반도체, 의료, 항공 분야에 적합하다. 단점은 single-class 제한과 학습 데이터 의존성이다.

Normalizing Flow 방식은 가역적 변환으로 확률 분포를 모델링한다. FastFlow가 대표 모델로 98.5%의 정확도와 20-50ms의 빠른 속도를 달성한다. 메모리는 500MB-1GB이다. 확률적 해석이 가능하며 속도와 정확도의 균형이 우수하다. 일반적인 품질 검사 환경에 가장 널리 사용된다. 단점은 flow network 설계의 복잡도와 하이퍼파라미터 튜닝이다.

Knowledge Distillation 방식은 teacher-student 구조로 정상 패턴을 학습한다. 이 패러다임은 두 가지 극단으로 발전했다. Reverse Distillation은 정밀도를 극대화하여 98.6%를 달성하지만 100-200ms가 소요된다. 반면 EfficientAD는 속도를 극대화하여 1-5ms의 실시간 처리를 가능하게 하지만 정확도는 97.8%이다. 전자는 정밀 검사용, 후자는 고속 라인용으로 명확히 구분된다.

Reconstruction-Based 방식은 재구성 오류를 이상 점수로 사용한다. DRAEM이 대표 모델로 simulated anomaly를 사용하여 97.5%를 달성한다. 가장 큰 강점은 10-50장의 소량 데이터로 학습 가능한 few-shot 능력이다. 신제품 출시 초기나 희귀 결함 시나리오에 적합하다. 단점은 simulated anomaly의 품질에 성능이 좌우된다는 점이다.

Feature Adaptation 방식은 pre-trained 특징을 타겟 도메인에 적응시킨다. DFM이 대표 모델로 15분 만에 94.5-95.5%의 baseline을 구축할 수 있다. 극도로 간단하고 빠르지만 성능이 SOTA 대비 3-5%p 낮다. 빠른 feasibility 검증과 프로토타이핑에 유용하지만 본격 배포에는 부적합하다. 시작점이지 종착점은 아니다.

Foundation Model 방식은 대규모 사전 학습 모델을 활용한다. Dinomaly가 multi-class 98.8%로 단일 모델로 모든 클래스를 처리한다. WinCLIP은 zero-shot 91-95%로 학습 데이터 없이 즉시 배포 가능하다. VLM-AD는 96-97%로 자연어 설명을 제공한다. 이 패러다임은 multi-class 효율성, zero-shot 즉시성, explainable AI라는 세 가지 혁명을 가져왔다.

### 1.2 Top Recommendations by Scenario

실무에서 가장 흔한 10개 시나리오에 대한 추천 모델은 다음과 같다.

최고 정확도가 필요한 경우(99% 이상), single-class 환경에서는 PatchCore(99.1%)를 선택한다. Multi-class 환경에서는 Dinomaly(98.8% multi, 99.2% single)를 선택한다. Pixel-level localization이 중요하다면 Reverse Distillation(98.5% pixel AUROC)을 고려한다. 적용 분야는 반도체 웨이퍼 검사, 의료 기기, 항공 부품, 자동차 안전 부품이다.

실시간 처리가 필요한 경우(10ms 미만), EfficientAD(1-5ms)가 유일한 선택이다. 97.8%의 정확도는 실용적으로 충분히 높다. CPU에서도 10-20ms로 실시간 처리가 가능하다. 고속 생산 라인(초당 100개 이상), 엣지 디바이스, 모바일 검사에 적합하다.

Multi-class 환경에서는 Dinomaly가 압도적이다. 15개 제품 검사 시 전통적 방법은 15개 모델(7.5GB)이 필요하지만 Dinomaly는 1개 모델(500MB)로 처리한다. 메모리 93% 절감, 관리 간소화, 배포 시간 80% 단축의 효과가 있다.

신제품 즉시 배포(zero-shot)가 필요한 경우, WinCLIP(91-95%)을 우선 선택한다. 텍스트 프롬프트만으로 즉시 사용 가능하고 무료이다. 설명 가능성이 필요하다면 VLM-AD(96-97%)를 선택하지만 API 비용($0.01-0.05/img)이 발생한다. 데이터 수집 후 다른 모델로 전환하는 것을 권장한다.

Few-shot 학습(10-50장)이 필요한 경우, DRAEM(97.5%)이 독보적이다. Simulated anomaly로 안정적 학습이 가능하며 2-4시간 만에 수렴한다. 신제품 출시 초기, 희귀 결함, 데이터 수집이 어려운 환경에 적합하다.

품질 보고서 자동화가 필요한 경우, VLM-AD를 선택한다. 결함 유형, 위치, 크기, 심각도, 원인, 권장사항을 자연어로 제공한다. 규제 산업(의료, 항공), 고객 보고서, 내부 품질 분석에 유용하다. 비용은 월 10,000장 기준 $100-500이다.

일반적인 균형 잡힌 검사의 경우, FastFlow(98.5%, 20-50ms)를 첫 번째로 추천한다. Multi-class 가능하다면 Dinomaly(98.8%, 80-120ms)를 선택한다. 최고 정확도가 필요하면 PatchCore(99.1%, 50-100ms)를 고려한다.

빠른 프로토타이핑의 경우, DFM(15분 학습, 94-95%)으로 feasibility를 검증한다. 성능이 충분하면 진행하고, 부족하면 데이터 수집 계획을 수립한다. 본격 개발 시에는 PatchCore나 FastFlow로 전환한다.

복잡한 텍스처(직물, 카펫, 가죽)의 경우, DSR(96.5-98.0%)을 선택한다. Dual subspace로 구조와 텍스처를 분리 모델링한다. 단순 결함에서는 DRAEM이나 Dinomaly가 더 나을 수 있다.

엣지 디바이스와 CPU 환경의 경우, EfficientAD가 유일한 현실적 선택이다. CPU에서 10-20ms로 작동하며 메모리는 200MB 미만이다. Jetson, Raspberry Pi 등에서 실시간 처리가 가능하다.

### 1.3 Decision Flowchart

실무에서 빠른 의사결정을 위한 flowchart는 다음과 같다.

첫 번째 질문은 multi-class 환경인가이다. 여러 제품을 동시에 검사해야 한다면 Dinomaly를 선택한다. 98.8%의 성능과 메모리 93% 절감의 효과가 있다. Single-class라면 다음 질문으로 진행한다.

두 번째 질문은 실시간 처리(10ms 미만)가 필수인가이다. 고속 라인이나 엣지 디바이스라면 EfficientAD를 선택한다. 1-5ms의 속도는 혁명적이며 97.8%의 정확도도 실용적으로 충분하다. 실시간이 필수가 아니라면 다음 질문으로 진행한다.

세 번째 질문은 학습 데이터가 있는가이다. 데이터가 전혀 없다면 WinCLIP(zero-shot, 91-95%)을 선택한다. 설명이 필요하면 VLM-AD를 고려한다. 데이터가 있다면 다음 질문으로 진행한다.

네 번째 질문은 데이터 양은 얼마인가이다. 10-50장이라면 DRAEM(few-shot, 97.5%)을 선택한다. 100장 이상이라면 다음 질문으로 진행한다.

다섯 번째 질문은 최고 정확도(99% 이상)가 필요한가이다. 반도체, 의료, 항공 등 불량 유출 비용이 매우 높다면 PatchCore(99.1%)를 선택한다. 99%까지는 필요 없다면 FastFlow(98.5%, 빠른 속도)를 선택한다.

특수 요구사항이 있는 경우, 품질 보고서 자동화는 VLM-AD, 복잡한 텍스처는 DSR, 빠른 프로토타입은 DFM을 선택한다.

---

## 2. Paradigm-by-Paradigm Evaluation

### 2.1 Memory-Based Methods

#### 2.1.1 Strengths

Memory-based 방식은 네 가지 핵심 강점을 가진다.

첫째, 최고 정확도이다. PatchCore의 99.1%는 현재까지 single-class 환경에서 최고 기록이다. 이는 3년 이상 유지되고 있으며, 다른 패러다임이 쉽게 따라잡기 어려운 수준이다. 모든 15개 MVTec AD 카테고리에서 일관되게 높은 성능을 보이며, Bottle(100%), Zipper(99.8%) 등에서는 거의 완벽한 탐지를 달성한다.

둘째, 직관적인 해석 가능성이다. 이상 점수가 정상 샘플과의 거리로 계산되므로, 왜 이상으로 판단했는지 설명하기 쉽다. 가장 유사한 정상 패치를 보여주면서 "이 정상 패턴과 거리가 X만큼 떨어져 있으므로 이상"이라고 설명할 수 있다. 이는 품질 엔지니어나 비전문가와 소통할 때 매우 유용하다.

셋째, 수학적 견고성이다. Mahalanobis distance, k-NN, coreset selection 등은 모두 이론적으로 잘 정립된 방법론이다. Coreset의 경우 ε-cover 이론으로 전체 분포를 효과적으로 커버함이 수학적으로 보장된다. 이러한 이론적 기반은 예측 가능한 동작과 안정적인 성능을 제공한다.

넷째, 하이퍼파라미터 민감도가 낮다. PatchCore의 주요 하이퍼파라미터는 coreset 크기와 backbone 선택 정도이다. 이들은 넓은 범위에서 안정적인 성능을 보이므로, 과도한 튜닝 없이도 좋은 결과를 얻을 수 있다. 이는 실무에서 빠른 배포를 가능하게 한다.

#### 2.1.2 Weaknesses

Memory-based 방식은 세 가지 주요 약점을 가진다.

첫째, 학습 데이터 의존성이다. 100-500장의 정상 샘플이 필요하며, 데이터 품질이 성능에 직접적 영향을 미친다. 신제품 출시 초기처럼 데이터 수집이 어려운 환경에서는 적용이 제한된다. 또한 정상 패턴이 변화하면 재학습이 필요하므로, 계절적 변화나 공정 변경에 민감하다.

둘째, single-class 제한이다. 클래스마다 별도의 메모리 뱅크와 모델이 필요하다. 15개 제품을 검사하려면 15개 모델을 관리해야 하며, 총 메모리는 1.5-7.5GB에 달한다. 이는 GPU 메모리 제약이 있는 환경에서 문제가 되고, 모델 관리 복잡도를 증가시킨다.

셋째, 확장성 문제이다. 학습 데이터가 증가하면 coreset도 비례하여 증가한다. 수천 장의 데이터를 사용하면 coreset 크기가 수 GB에 달할 수 있다. 또한 추론 시 k-NN search의 계산량이 coreset 크기에 비례하므로, 데이터 증가가 속도 저하로 이어진다.

#### 2.1.3 Best Use Cases

Memory-based 방식은 다음 시나리오에 최적이다.

최고 정확도가 필수인 환경이다. 반도체 웨이퍼 검사에서 미세 결함을 놓치면 전체 웨이퍼가 폐기되므로 99% 이상의 정확도가 필요하다. 의료 기기에서 불량품은 환자 안전에 직결되므로 완벽에 가까운 탐지가 요구된다. 항공 부품은 단 하나의 결함도 치명적 사고로 이어질 수 있다. 이러한 환경에서 PatchCore의 99.1%는 다른 모델이 대체하기 어렵다.

Single-class 환경에서 한 가지 제품만 검사하는 경우이다. 전용 검사 라인이 구축되어 있고, 제품 변경이 드물다면 single-class 제한이 문제가 되지 않는다. 오히려 단일 제품에 최적화된 모델이 더 나은 성능을 제공한다.

안정성과 신뢰성이 중요한 환경이다. 제조 현장에서는 모델이 예측 가능하게 동작하고, 갑작스러운 성능 저하가 없어야 한다. Memory-based 방식의 수학적 견고성과 낮은 하이퍼파라미터 민감도는 이러한 요구를 충족한다.

충분한 학습 데이터를 확보할 수 있는 환경이다. 생산이 안정화된 제품에서는 수백 장의 정상 샘플을 쉽게 수집할 수 있다. 이 경우 data-driven 접근법인 memory-based가 강력한 성능을 발휘한다.

#### 2.1.4 Model Selection (PatchCore vs PaDiM vs DFKDE)

Memory-based 패러다임 내에서 모델 선택 기준은 다음과 같다.

PatchCore는 거의 모든 경우에 최선의 선택이다. PaDiM 대비 메모리는 90% 감소하고 성능은 오히려 향상되었다. DFKDE 대비 2-3%p 높은 정확도를 보인다. 실무에서 PatchCore를 기본 선택으로 하고, 특별한 이유가 없는 한 다른 모델을 고려할 필요가 없다.

PaDiM은 현재 실무 활용 가치가 제한적이다. 메모리 2-5GB는 현대적 배포 환경에서 수용하기 어렵다. 그러나 교육이나 연구 목적으로는 유용하다. Memory-based 패러다임의 기초 개념을 이해하기에 PaDiM의 다변량 가우시안 모델링은 직관적이다. 또한 빠른 프로토타이핑에서 PatchCore의 coreset selection 구현이 복잡하다면 PaDiM으로 먼저 검증할 수 있다.

DFKDE는 특수한 경우에만 고려한다. Kernel density estimation이 이론적으로 선호되는 환경이거나, 비모수적 방법이 요구되는 경우이다. 그러나 고차원에서 curse of dimensionality 문제로 인해 성능이 PatchCore보다 낮으므로, 일반적으로 권장하지 않는다.

결론적으로 Memory-based 방식을 선택했다면 PatchCore를 사용하는 것이 최선이다. PaDiM과 DFKDE는 역사적 의의는 있지만 현재 실무에서는 PatchCore로 대체되었다.

### 2.2 Normalizing Flow Methods

#### 2.2.1 Strengths

Normalizing flow 방식은 다섯 가지 핵심 강점을 가진다.

첫째, 확률론적 해석 가능성이다. Log-likelihood로 계산된 이상 점수는 명확한 확률적 의미를 가진다. "이 샘플이 정상 분포에서 나올 확률이 0.001%이므로 이상"이라고 설명할 수 있다. 이는 단순 거리나 재구성 오류보다 직관적이며, 통계적 의사결정 프레임워크와 자연스럽게 통합된다.

둘째, 우수한 성능이다. FastFlow의 98.5%는 SOTA 수준에 근접하며, 대부분의 실무 환경에서 충분히 높은 정확도이다. Pixel AUROC 97.8%는 결함 위치 파악 능력도 우수함을 나타낸다.

셋째, 빠른 추론 속도이다. FastFlow의 20-50ms는 준실시간 처리를 가능하게 한다. 초당 20-50장을 처리할 수 있어, 대부분의 생산 라인 속도에 대응할 수 있다. 이는 memory-based(50-100ms)보다 2배 빠르고, reconstruction-based(50-100ms)와 비슷하거나 빠르다.

넷째, 다양한 결함 크기 대응이다. Multi-scale flow network를 사용하므로, 큰 결함(변형, 파손)부터 작은 결함(스크래치, 얼룩)까지 효과적으로 탐지한다. CS-Flow는 cross-scale 정보 교환으로 이를 더욱 강화한다.

다섯째, 이론적 완전성이다. Normalizing flow는 생성 모델의 일종으로 확률 이론에 탄탄한 기반을 둔다. Change of variables 공식, Jacobian determinant 계산 등은 수학적으로 엄밀하게 정의된다. 이러한 이론적 기반은 모델의 동작을 이해하고 개선하는 데 도움이 된다.

#### 2.2.2 Weaknesses

Normalizing flow 방식은 네 가지 주요 약점을 가진다.

첫째, 학습 복잡도이다. Flow network 설계가 복잡하며, affine coupling layer, permutation, activation function 등 많은 설계 선택이 필요하다. 각 선택이 성능에 영향을 미치므로, 최적 구조를 찾기 위한 실험이 많이 필요하다. 이는 개발 시간을 증가시킨다.

둘째, 하이퍼파라미터 튜닝이다. Flow depth(층 수), coupling layer 구조, condition network 설계 등 많은 하이퍼파라미터를 조정해야 한다. 이들은 서로 상호작용하므로, grid search나 random search로 최적값을 찾기 어렵다. 경험적 지식이나 여러 번의 시행착오가 필요하다.

셋째, 메모리 사용량이다. Flow network는 가역성을 유지하기 위해 중간 활성화를 모두 저장해야 한다. 이는 추론 시에도 상당한 메모리를 요구한다. FastFlow도 500MB-1GB를 사용하며, 이는 PatchCore(100-500MB)나 EfficientAD(<200MB)보다 높다.

넷째, 디버깅 어려움이다. Flow 학습이 수렴하지 않거나 이상한 동작을 보일 때, 원인을 파악하고 해결하기 어렵다. Jacobian determinant가 발산하거나, coupling layer가 identity mapping으로 퇴화하는 등의 문제가 발생할 수 있다. 이를 해결하려면 flow 내부 동작에 대한 깊은 이해가 필요하다.

#### 2.2.3 Best Use Cases

Normalizing flow 방식은 다음 시나리오에 최적이다.

속도와 정확도가 모두 중요한 균형 잡힌 검사 환경이다. FastFlow의 98.5% 정확도는 대부분의 품질 기준을 충족하며, 20-50ms 속도는 준실시간 처리를 가능하게 한다. 고속 라인만큼은 아니지만, 일반적인 생산 속도(초당 10-20개)에는 충분하다. 이는 가장 흔한 실무 시나리오이므로, FastFlow가 널리 사용된다.

확률적 해석이 필요한 환경이다. 통계적 공정 관리(SPC)와 통합하거나, 신뢰 구간을 제공해야 하는 경우 log-likelihood 기반 이상 점수가 유용하다. "이 샘플은 99% 신뢰도로 이상"이라고 말할 수 있다.

다양한 크기의 결함을 탐지해야 하는 환경이다. Grid, Tile 등 텍스처 카테고리에서는 작은 얼룩부터 큰 패턴 왜곡까지 다양한 결함이 나타난다. Multi-scale flow는 이러한 다양성에 잘 대응한다. CS-Flow는 cross-scale 정보 교환으로 이를 더욱 강화한다.

Pixel-level localization이 중요한 환경이다. FastFlow의 97.8% pixel AUROC는 결함의 정확한 위치를 파악하는 능력이 우수함을 나타낸다. 이는 결함 원인 분석이나 공정 개선에 유용하다.

#### 2.2.4 Model Selection (FastFlow vs CFlow vs CS-Flow vs U-Flow)

Normalizing flow 패러다임 내에서 모델 선택 기준은 다음과 같다.

FastFlow는 대부분의 경우 최선의 선택이다. 98.5%의 높은 정확도와 20-50ms의 빠른 속도로 실무에서 가장 많이 사용된다. 2D flow로 단순화하여 3D flow 대비 속도는 3배 빠르고 성능은 오히려 향상되었다. 학습도 30-60분으로 빠르다. 일반적인 품질 검사 환경에서는 FastFlow를 기본 선택으로 하면 된다.

CFlow는 연구나 baseline 목적으로 유용하다. Position-conditional flow의 원형으로서 normalizing flow 방식을 이해하는 데 도움이 된다. 98.2%의 성능도 나쁘지 않다. 그러나 100-150ms의 느린 속도와 높은 메모리 사용량 때문에 실무에서는 FastFlow로 대체되었다. 새로운 flow 구조를 연구하거나, 교육 목적으로는 여전히 가치가 있다.

CS-Flow는 다양한 크기의 결함이 혼재된 환경에서 고려한다. Cross-scale 정보 교환으로 Grid, Tile 등에서 FastFlow보다 1-2%p 높은 성능을 보일 수 있다. 그러나 구현이 복잡하고 학습 시간이 길어지므로, 성능 개선이 명확히 필요한 경우에만 사용한다.

U-Flow는 자동 임계값 설정이 필요한 환경에서 유용하다. U-Net 구조와 unsupervised threshold 학습으로 운영 자동화를 개선한다. 97.6%의 성능은 FastFlow보다 약간 낮지만, 사람의 개입 없이 임계값을 자동 설정할 수 있다면 운영 효율이 크게 향상된다. 인력이 부족하거나 24시간 무인 운영이 필요한 환경에 적합하다.

결론적으로 Normalizing flow 방식을 선택했다면 FastFlow를 사용하는 것이 최선이다. 특수한 요구사항이 있을 때만 CS-Flow나 U-Flow를 고려한다. CFlow는 역사적 의의는 있지만 현재 실무에서는 FastFlow로 대체되었다.

### 2.3 Knowledge Distillation Methods

#### 2.3.1 Strengths

Knowledge distillation 방식은 다섯 가지 핵심 강점을 가진다.

첫째, 양극단의 성능 커버리지이다. 이 패러다임은 정밀도와 속도라는 두 가지 극단을 모두 제공한다. Reverse Distillation은 98.6%의 높은 정확도로 정밀 검사를 지원하고, EfficientAD는 1-5ms의 극한 속도로 실시간 라인을 지원한다. 단일 패러다임에서 이렇게 넓은 스펙트럼을 커버하는 것은 드물다.

둘째, end-to-end 학습이다. Teacher와 student를 단일 loss function으로 함께 학습하므로, 복잡한 다단계 학습이 필요 없다. 이는 구현을 단순화하고, 학습 시간을 단축시킨다. STFPM의 경우 1-2시간만에 수렴한다.

셋째, 유연한 설계 가능성이다. Teacher-student 구조는 다양한 변형이 가능하다. Teacher를 복잡하게(STFPM), teacher를 단순하게(Reverse Distillation), student를 극도로 경량화(EfficientAD) 등 목적에 따라 설계를 조정할 수 있다. 이러한 유연성은 특정 요구사항에 최적화된 모델을 만들 수 있게 한다.

넷째, pre-trained 지식 활용이다. ImageNet에서 학습된 teacher의 일반적 시각 특징을 전이하므로, 타겟 도메인의 학습 데이터가 적어도 합리적인 성능을 달성할 수 있다. 이는 cold start 문제를 완화한다.

다섯째, CPU에서도 실시간 가능이다(EfficientAD). EfficientAD는 GPU가 없는 환경에서도 10-20ms로 작동한다. 이는 엣지 디바이스나 저비용 배포에서 혁명적이다.

#### 2.3.2 Weaknesses

Knowledge distillation 방식은 네 가지 주요 약점을 가진다.

첫째, teacher 품질 의존성이다. Teacher가 ImageNet의 일반적 특징을 학습했으므로, 산업 이미지의 특수한 패턴에 최적화되지 않을 수 있다. STFPM의 96.8%가 PatchCore의 99.1%보다 낮은 이유 중 하나이다. Reverse Distillation은 teacher를 타겟 도메인에서 학습하여 이를 완화하지만, 복잡도가 증가한다.

둘째, domain gap이다. ImageNet의 고양이, 자동차, 건물 등은 고수준 semantic 정보를 가지지만, 산업 이미지의 스크래치, 얼룩, 색상 불균일 등은 저수준 texture 정보이다. 이러한 gap은 성능 저하로 이어질 수 있다.

셋째, 설계 복잡도이다. 최적의 teacher-student 구조를 찾기 어렵다. Teacher의 크기, student의 크기, feature matching 방식, loss function 가중치 등 많은 설계 선택이 성능에 영향을 미친다. Reverse Distillation의 encoder-decoder 구조는 특히 복잡하다.

넷째, 양극단 선택의 딜레마이다. 정밀도(Reverse Distillation)와 속도(EfficientAD) 중 하나만 선택해야 한다. 98.6%와 100-200ms, 또는 97.8%와 1-5ms이다. 중간 지대(98%와 10-20ms)는 FastFlow 등 다른 패러다임이 더 나을 수 있다.

#### 2.3.3 Two Extremes (Precision vs Speed)

Knowledge distillation 패러다임은 두 가지 극단으로 발전했으며, 각각 명확히 구분되는 적용 분야를 가진다.

정밀 검사 극단인 Reverse Distillation은 98.6%의 image AUROC와 98.5%의 pixel AUROC를 달성한다. Teacher를 단순화하여 one-class embedding을 생성하고, student가 이를 역으로 재구성하도록 학습한다. 타겟 도메인에 특화된 표현을 학습하여 STFPM 대비 1.8%p 향상되었다. 추론 시간은 100-200ms로 느리지만, 정밀도가 최우선인 환경에서는 수용 가능하다. 적용 분야는 반도체 웨이퍼 검사(미세 결함), 의료 영상(병변 localization), 정밀 기계 부품(마이크로 크랙)이다. 결함의 정확한 위치가 중요하고, 검사 시간이 100-200ms 허용되는 환경에 적합하다.

실시간 처리 극단인 EfficientAD는 1-5ms의 혁명적 속도를 달성한다. Patch description network(PDN)라는 경량 네트워크(약 50K 파라미터)와 autoencoder를 하이브리드로 결합한다. 정확도는 97.8%로 약간 낮지만, 실용적으로 충분히 높다. CPU에서도 10-20ms로 작동하여 GPU가 필수가 아니다. 메모리는 200MB 미만으로 극소이다. 적용 분야는 고속 생산 라인(초당 100개 이상), 엣지 디바이스(Jetson, Raspberry Pi), 모바일 검사, 드론 기반 검사이다. 초당 200-1000 프레임 처리가 가능하여, 전수 검사가 현실화된다.

두 모델은 서로 다른 니즈를 완벽하게 충족시킨다. Reverse Distillation은 "한 개도 놓치면 안 되는" 환경에, EfficientAD는 "모든 제품을 검사해야 하는" 환경에 적합하다. 중간 지대는 FastFlow 등 다른 패러다임이 더 나을 수 있다는 점을 인지해야 한다.

#### 2.3.4 Model Selection (Reverse Distillation vs EfficientAD vs STFPM)

Knowledge distillation 패러다임 내에서 모델 선택은 매우 명확하다.

정밀 검사가 필요하면 Reverse Distillation을 선택한다. 98.6%의 정확도와 98.5%의 pixel AUROC는 knowledge distillation 패러다임 내에서 최고이다. 100-200ms의 추론 시간이 허용되고, pixel-level localization이 중요하다면 최선의 선택이다. 반도체, 의료, 정밀 기계에 적합하다.

실시간 처리가 필요하면 EfficientAD를 선택한다. 1-5ms는 다른 모든 모델을 압도하는 속도이다. 97.8%의 정확도도 실용적으로 충분하다. 고속 라인, 엣지 디바이스, CPU 환경에서는 EfficientAD가 유일한 현실적 선택이다.

STFPM은 과도기적 역할로 제한적 가치를 가진다. 96.8%의 성능은 Reverse Distillation이나 PatchCore보다 낮고, 20-40ms의 속도는 EfficientAD보다 느리다. 빠른 baseline 구축이나 교육 목적으로는 유용하지만, 본격 배포에는 Reverse Distillation이나 EfficientAD로 전환하는 것을 권장한다. 역사적으로는 knowledge distillation 패러다임을 이상 탐지에 도입한 의의가 있다.

FRE는 deprecated 상태이다. STFPM의 2배 속도 향상을 목표로 했지만, 성능 저하(-0.8~1.8%p)와 제한적 개선(10-30ms)으로 실용적 가치가 없다. EfficientAD가 등장하면서 완전히 대체되었다. FRE의 교훈은 "점진적 개선은 충분하지 않다"는 것이다. 혁명적 발전이 필요하다.

결론적으로 knowledge distillation 방식을 선택했다면, 정밀 검사는 Reverse Distillation, 실시간 처리는 EfficientAD를 사용한다. STFPM은 학습용으로만, FRE는 사용하지 않는다.

### 2.4 Reconstruction-Based Methods

#### 2.4.1 Strengths

Reconstruction-based 방식은 다섯 가지 핵심 강점을 가진다.

첫째, few-shot 능력이다. DRAEM은 10-50장의 정상 샘플만으로 97.5%를 달성한다. 이는 다른 패러다임(100-500장 필요)보다 10배 적은 데이터이다. 신제품 출시 초기, 희귀 결함, 데이터 수집이 어려운 환경에서 혁명적이다. Few-shot 능력은 simulated anomaly 덕분이다. 정상 이미지에 인위적 결함을 추가하여 학습 데이터를 augment한다.

둘째, 안정적인 학습이다. DRAEM은 GAN 없이 supervised learning으로 학습한다. Simulated anomaly를 "제거"하도록 명확한 학습 신호를 제공하므로, 수렴이 빠르고 안정적이다. 2-4시간 만에 학습이 완료되며, GANomaly의 6-10시간과 대비된다. Mode collapse나 oscillation 같은 GAN 특유의 문제가 없다.

셋째, 직관적인 해석 가능성이다. 재구성된 이미지와 원본 이미지의 차이를 시각적으로 확인할 수 있다. "이 부분이 재구성되지 않았으므로 이상"이라고 설명하기 쉽다. Segmentation map을 제공하여 결함의 정확한 위치를 보여준다.

넷째, 복잡한 텍스처 처리이다. DSR은 dual subspace로 구조와 텍스처를 분리 모델링한다. Quantization subspace는 구조적 패턴을, target subspace는 텍스처 세부사항을 포착한다. 직물, 카펫, 가죽 등 복잡한 텍스처 표면에서 96.5-98.0%의 높은 성능을 보인다.

다섯째, end-to-end 학습이다. Reconstruction network와 discriminative network를 함께 학습하므로, 복잡한 다단계 과정이 필요 없다. SSIM loss와 focal loss를 결합하여 구조적 유사성과 pixel-wise classification을 동시에 학습한다.

#### 2.4.2 Weaknesses

Reconstruction-based 방식은 네 가지 주요 약점을 가진다.

첫째, simulation 품질 의존성이다. DRAEM의 성능은 simulated anomaly가 실제 결함과 얼마나 유사한지에 좌우된다. Perlin noise를 사용한 augmentation이 실제 스크래치, 얼룩, 변형을 잘 근사해야 한다. 만약 실제 결함이 simulated anomaly와 크게 다르면 성능이 저하된다. 이는 새로운 결함 유형에 대한 일반화를 제한할 수 있다.

둘째, domain gap이다. Simulated anomaly는 어디까지나 인위적이다. 실제 제조 공정에서 발생하는 결함의 물리적 특성(조명 반사, 깊이, 재질 변화)을 완벽히 재현하기 어렵다. 이는 벤치마크 성능과 실제 현장 성능의 차이로 나타날 수 있다.

셋째, SOTA 대비 낮은 성능이다. DRAEM의 97.5%는 PatchCore의 99.1%보다 1.6%p 낮다. Few-shot 환경에서는 이것이 최선이지만, 충분한 데이터가 있다면 다른 패러다임이 더 나을 수 있다. 실무에서는 1-2%p 차이도 중요할 수 있다.

넷째, 복잡도이다. DSR의 dual subspace 학습은 VQ-VAE와 VAE를 모두 포함하여 구현이 복잡하다. 두 subspace 간의 균형을 맞추기 위한 하이퍼파라미터 튜닝이 필요하다. GANomaly는 4개의 네트워크(E-D-E + Discriminator)를 관리해야 하는 극도의 복잡도로 실패했다.

#### 2.4.3 Best Use Cases (Few-shot)

Reconstruction-based 방식은 다음 시나리오에 최적이다.

신제품 출시 초기이다. 신제품은 생산 초기에 충분한 데이터를 확보하기 어렵다. 시장 출시 일정이 촉박하여 수백 장의 데이터를 수집할 시간이 없을 수 있다. DRAEM은 10-50장만으로 97.5%를 달성하므로, 빠른 time-to-market을 지원한다. 초기에는 DRAEM으로 시작하고, 데이터가 축적되면 PatchCore로 전환하는 전략이 효과적이다.

희귀 결함 학습이다. 어떤 결함은 발생 빈도가 매우 낮아 이상 샘플을 거의 확보할 수 없다. 예를 들어 특정 재료 결함은 1년에 몇 건만 발생할 수 있다. DRAEM의 simulated anomaly는 이러한 희귀 결함의 패턴을 학습하는 데 도움이 된다.

데이터 수집이 어려운 환경이다. 일부 산업 환경에서는 데이터 수집 자체가 어렵다. 접근이 제한된 클린룸, 고비용 샘플(귀금속, 희토류), 파괴적 검사가 필요한 제품 등이다. 이러한 환경에서 소량 데이터로 학습 가능한 DRAEM은 유일한 대안일 수 있다.

복잡한 텍스처 제품이다. Carpet, Leather, Tile 등은 구조와 텍스처가 복잡하게 얽혀있다. DSR의 dual subspace는 이를 분리하여 모델링한다. 96.5-98.0%의 성능은 해당 카테고리에서 최고 수준이다. 단순 결함(긁힘, 얼룩)보다는 텍스처 자체의 변형이 문제가 되는 환경에 적합하다.

#### 2.4.4 Model Selection (DRAEM vs DSR vs GANomaly)

Reconstruction-based 패러다임 내에서 모델 선택 기준은 다음과 같다.

DRAEM은 거의 모든 경우에 최선의 선택이다. 97.5%의 성능, 안정적 학습, few-shot 능력으로 reconstruction-based의 대표 모델이다. 10-50장의 데이터로 2-4시간 만에 학습할 수 있다. 신제품, 희귀 결함, 데이터 부족 환경에서는 DRAEM을 기본 선택으로 한다.

DSR은 복잡한 텍스처 제품에 특화되어 있다. Carpet, Leather, Tile 등에서 DRAEM보다 1-2%p 높은 성능을 보일 수 있다. 그러나 VQ-VAE와 VAE를 모두 구현해야 하므로 복잡도가 높다. 텍스처 카테고리에서 최고 성능이 필요하고, 개발 리소스가 충분하다면 고려한다. 단순 결함에서는 DRAEM이 더 나을 수 있다.

GANomaly는 사용하지 않는다. GAN의 학습 불안정성, 6-10시간의 긴 학습 시간, 93-95%의 낮은 성능으로 실무 가치가 없다. DRAEM이 등장하면서 완전히 대체되었다. 역사적으로는 reconstruction-based 접근의 초기 시도로 의의가 있지만, 현재는 deprecated 상태이다.

결론적으로 reconstruction-based 방식을 선택했다면, 일반적으로는 DRAEM을, 복잡한 텍스처에는 DSR을 사용한다. GANomaly는 사용하지 않는다.

### 2.5 Feature Adaptation Methods

#### 2.5.1 Strengths

Feature adaptation 방식은 네 가지 핵심 강점을 가진다.

첫째, 극단적인 간단함이다. DFM은 PCA와 Mahalanobis distance만 사용한다. Pre-trained CNN(예: ResNet)에서 특징을 추출하고, PCA로 주요 성분을 유지한 후, 재구성 오류를 계산한다. 구현이 몇 줄의 코드로 가능하며, 딥러닝 초보자도 쉽게 이해하고 적용할 수 있다.

둘째, 빠른 학습이다. DFM은 5-15분 만에 학습이 완료된다. PCA 계산과 공분산 행렬 저장만 필요하므로, GPU가 필수가 아니다. 이는 프로젝트 초기에 빠른 feasibility 검증을 가능하게 한다. 오전에 아이디어를 생각하고, 점심 전에 결과를 확인할 수 있다.

셋째, 빠른 추론이다. DFM은 10-20ms로 빠르다. PCA projection과 Mahalanobis distance 계산만 필요하므로, 계산량이 적다. 이는 준실시간 처리를 가능하게 한다.

넷째, 낮은 메모리이다. DFM은 50-100MB만 사용한다. PCA 행렬과 공분산 행렬만 저장하면 되므로, 메모리 효율적이다. 엣지 디바이스나 저사양 환경에서도 배포 가능하다.

#### 2.5.2 Weaknesses

Feature adaptation 방식은 세 가지 근본적인 약점을 가진다.

첫째, 낮은 성능이다. DFM은 94.5-95.5%, CFA는 96.5-97.5%로 SOTA 대비 1.6-4.6%p 낮다. 이는 실무에서 유의미한 차이이다. 불량률 1%인 환경에서 95% 정확도는 99% 정확도보다 4배 많은 불량을 놓친다. 정밀 검사가 필요한 환경에서는 수용하기 어렵다.

둘째, pre-trained 특징의 domain gap이다. ImageNet의 일반적 시각 특징은 산업 이미지의 미세 결함 탐지에 최적화되지 않았다. 고양이와 자동차를 구분하는 특징이 스크래치와 얼룩을 구분하는 데는 최적이 아니다. 이는 성능 상한을 제한한다.

셋째, 선형 변환의 한계이다. PCA는 선형 변환만 가능하다. 복잡한 비선형 관계를 포착하지 못한다. CFA의 hypersphere embedding도 단순 정규화에 불과하다. 이는 복잡한 도메인 적응에 부족하다.

#### 2.5.3 Best Use Cases (Rapid Prototyping)

Feature adaptation 방식은 다음 시나리오에 최적이다.

빠른 feasibility 검증이다. 프로젝트 초기에 "이상 탐지가 이 데이터에서 작동하는가?"를 확인해야 한다. DFM으로 15분 만에 94-95%의 baseline을 구축하고, 이것이 목표에 근접하는지 판단한다. 만약 94%가 나온다면 PatchCore로 99%를 달성할 가능성이 높다. 만약 60%가 나온다면 데이터나 문제 설정을 재검토해야 한다.

저사양 환경이다. GPU가 없거나, 메모리가 제한된 환경(100MB 이하)에서는 DFM이 유일한 선택일 수 있다. 50-100MB의 메모리와 CPU만으로 작동한다. 성능은 낮지만, 하드웨어 제약이 더 큰 문제라면 수용해야 한다.

교육 및 연구이다. DFM은 이상 탐지의 기본 개념을 학습하기에 좋다. PCA, Mahalanobis distance, feature extraction 등의 개념을 직관적으로 이해할 수 있다. 복잡한 딥러닝 모델을 가르치기 전에 DFM으로 기초를 다질 수 있다.

Domain shift가 큰 환경이다(CFA).

CFA의 hypersphere embedding은 조명 변화, 카메라 변경 등 domain shift가 큰 환경에서 96.5-97.5%를 달성한다. 단위 구 표면에 특징을 projection하여 scale invariance를 얻고, angular distance로 방향 차이를 측정한다. 이는 조명이나 카메라가 자주 바뀌는 환경에서 유용하다. 그러나 일반적인 경우에는 FastFlow나 Dinomaly가 더 나을 수 있다.

#### 2.5.4 Model Selection (DFM vs CFA)

Feature adaptation 패러다임 내에서 모델 선택은 간단하다.

DFM은 빠른 프로토타이핑과 저사양 환경에서 선택한다. 15분 학습, 94.5-95.5% 성능, 50-100MB 메모리로 feasibility 검증에 최적이다. 프로젝트 초기에 DFM으로 빠르게 baseline을 구축하고, 성능이 충분하면 진행하고 부족하면 데이터 수집 계획을 수립한다. 본격 개발 단계에서는 PatchCore나 FastFlow로 전환한다.

CFA는 domain shift가 크고 특수한 환경에서만 고려한다. 조명이 자주 바뀌거나, 여러 카메라를 사용하거나, 계절적 변화가 큰 환경이다. 96.5-97.5%는 DFM보다 2%p 높지만 여전히 SOTA 대비 1.6-2.6%p 낮다. 학습 시간도 30-60분으로 DFM의 4배이고, 하이퍼파라미터 튜닝이 복잡하다. 일반적인 경우에는 FastFlow(98.5%)가 더 나으므로, CFA는 domain shift 문제가 명확할 때만 사용한다.

결론적으로 feature adaptation 방식은 시작점이지 종착점이 아니다. DFM으로 빠르게 검증하고, 본격 배포에는 다른 패러다임으로 전환하는 것을 권장한다. 성능 gap(1.6-4.6%p)은 실무에서 유의미하므로, 최종 배포에 feature adaptation을 사용하는 것은 신중해야 한다.

### 2.6 Foundation Model Methods

#### 2.6.1 Strengths

Foundation model 방식은 여섯 가지 혁명적 강점을 가진다.

첫째, multi-class 단일 모델이다. Dinomaly는 98.8%의 성능으로 모든 클래스를 하나의 모델로 처리한다. 15개 제품 검사 시 전통적 방법은 15개 모델(7.5GB)이 필요하지만, Dinomaly는 1개 모델(500MB)로 처리한다. 메모리 93% 절감은 단순히 용량 문제가 아니라, 저렴한 하드웨어(GPU 메모리 8GB → 2GB)로 배포 가능함을 의미한다. 모델 관리도 간소화되어, 학습/배포/모니터링이 15분의 1로 줄어든다.

둘째, zero-shot 즉시 배포이다. WinCLIP은 학습 데이터 없이 텍스트 프롬프트만으로 91-95%를 달성한다. 전통적 방법은 데이터 수집(2-4주), 학습(1-2시간), 검증(1주)이 필요하지만, WinCLIP은 프롬프트 작성(10분)만으로 즉시 배포할 수 있다. 신제품 출시 당일에 품질 검사를 시작할 수 있다는 것은 time-to-market 관점에서 혁명적이다.

셋째, explainable AI 실현이다. VLM-AD는 자연어로 결함을 설명한다. 전통적 모델은 "이상 점수 0.87"만 제공하지만, VLM-AD는 "상단 왼쪽 모서리에 5mm 길이의 중간 정도 심각도 스크래치, 조립 중 취급 손상으로 추정, 3번 스테이션 취급 공정 점검 권장"을 제공한다. 이는 품질 엔지니어의 근본 원인 분석, 생산 관리자의 공정 개선, 감사 담당의 근거 문서를 자동화한다.

넷째, 강력한 범용 표현이다. DINOv2는 수억 개 이미지에서 self-supervised로 학습되어, semantic과 low-level 정보를 모두 포착한다. CLIP은 4억 개 이미지-텍스트 쌍에서 학습되어, 시각과 언어를 통합한다. 이러한 대규모 사전 학습은 소량의 타겟 데이터로는 학습하기 어려운 복잡한 패턴을 포착한다.

다섯째, 빠른 적응이다. Foundation model은 이미 강력한 표현을 가지므로, 타겟 도메인에 빠르게 적응한다. Dinomaly는 3-5시간 학습으로 98.8%를 달성한다. 이는 전통적 방법(15개 모델 × 1-2시간 = 15-30시간)보다 5-10배 빠르다.

여섯째, 미래 지향성이다. Foundation model은 지속적으로 발전한다. OpenAI, Meta, Google 등 거대 기업들이 투자하여, 매년 성능이 향상된다. 산업 특화 foundation model(Manufacturing CLIP, Industrial DINOv2)도 등장할 것이다. 현재 foundation model을 도입하면, 이러한 발전의 혜택을 자동으로 받을 수 있다.

#### 2.6.2 Weaknesses

Foundation model 방식은 다섯 가지 주요 약점을 가진다.

첫째, 큰 모델 크기이다. DINOv2는 1.5-2GB, CLIP은 500MB-1GB이다. 이는 PatchCore(100-500MB)나 EfficientAD(<200MB)보다 크다. 엣지 디바이스에서는 메모리 제약이 될 수 있다. 그러나 multi-class 환경에서는 단일 모델(500MB)이 15개 모델(7.5GB)보다 효율적이다.

둘째, API 비용이다(VLM-AD). GPT-4V는 이미지당 $0.01-0.05를 부과한다. 월 10,000장 검사 시 $100-500, 100,000장 시 $1,000-5,000이다. 대량 처리에는 비용 부담이 크다. 그러나 중요 샘플의 상세 분석이나 품질 보고서 자동화에는 충분히 가치가 있다. On-premise VLM을 사용하면 비용 문제를 완화할 수 있다.

셋째, 느린 속도이다(VLM-AD). GPT-4V는 이미지당 2-5초가 소요된다. 이는 실시간 처리가 불가능함을 의미한다. 대량 배치 처리나 고속 라인에는 부적합하다. 그러나 샘플링 검사나 2차 검사에는 허용 가능하다.

넷째, zero-shot의 낮은 정확도이다. WinCLIP의 91-95%는 전통적 방법(99.1%)보다 4-8%p 낮다. 이는 정밀 검사에는 부족하다. 그러나 신제품 즉시 배포, 다품종 소량 생산, 프로토타입 등에서는 충분한 가치가 있다. 데이터가 축적되면 다른 모델로 전환하는 전략이 효과적이다.

다섯째, 인터넷 의존이다(VLM-AD). GPT-4V API는 인터넷 연결이 필요하다. 폐쇄망 환경에서는 사용할 수 없다. 데이터 프라이버시 문제도 있다. On-premise VLM을 사용하면 해결할 수 있지만, 추가 인프라 비용이 발생한다.

#### 2.6.3 Best Use Cases (Multi-class, Zero-shot, Explainable)

Foundation model 방식은 세 가지 주요 시나리오에 최적이다.

Multi-class 환경이다. 여러 제품을 동시에 검사하는 경우 Dinomaly가 압도적이다. 15개 제품 검사 시나리오를 보자. 전통적 방법은 15개 모델을 개별 학습(15시간), 개별 배포, 개별 모니터링해야 한다. 총 메모리는 7.5GB이고, GPU 메모리 8GB가 필요하다. 신제품 추가 시 전체 과정을 반복해야 한다. Dinomaly는 1개 모델을 3-5시간 학습하면 모든 제품을 처리한다. 메모리는 500MB이고, GPU 메모리 2GB면 충분하다. 신제품은 데이터만 추가하고 재학습하면 된다. 비용 절감 효과는 막대하다. 하드웨어 비용(고성능 GPU → 저가 GPU), 인건비(모델 관리 간소화), 시간 비용(배포 80% 단축)이 모두 감소한다.

Zero-shot 즉시 배포이다. 신제품 출시 당일에 품질 검사를 시작해야 하는 경우 WinCLIP이 유일한 해답이다. 프롬프트만 작성하면 91-95%의 검사가 즉시 가능하다. 다품종 소량 생산 환경에서 매주 새로운 제품이 나오는 경우, 각 제품마다 학습 데이터를 수집하는 것은 비현실적이다. WinCLIP으로 모든 제품을 커버하고, 주요 제품만 전통적 방법으로 전환하는 전략이 효과적이다. 프로토타입 단계에서도 유용하다. 설계가 자주 바뀌는 초기 단계에서는 학습 데이터 수집이 낭비일 수 있다. WinCLIP으로 초기 검사를 수행하고, 설계가 확정되면 본격 모델을 구축한다.

Explainable AI가 필수인 환경이다. 규제 산업(의료, 항공)에서는 "왜 불량으로 판정했는가"를 설명해야 한다. VLM-AD는 자연어 설명을 제공하여 이를 충족한다. 품질 보고서 자동화에도 유용하다. 일일 품질 보고서에 결함 유형별 통계, 주요 원인, 개선 권장사항을 자동으로 생성할 수 있다. 고객 보고서에서도 전문 용어 대신 이해하기 쉬운 설명을 제공한다. 내부 품질 분석에서 근본 원인을 자동으로 추적하여 엔지니어의 시간을 절약한다.

#### 2.6.4 Model Selection (Dinomaly vs WinCLIP vs VLM-AD)

Foundation model 패러다임 내에서 모델 선택은 시나리오에 따라 명확히 구분된다.

Dinomaly는 multi-class 환경에서 최우선 선택이다. 98.8%(multi-class)의 높은 성능과 메모리 93% 절감, 관리 간소화로 투자 대비 효과가 가장 크다. Single-class로 사용해도 99.2%로 PatchCore를 초과한다. 여러 제품을 검사하거나, 향후 제품 추가 가능성이 있다면 Dinomaly를 선택한다. 2025년 최신 모델로 향후 지속적인 개선이 예상된다.

WinCLIP은 zero-shot이 필요한 환경에서 선택한다. 신제품 즉시 검사, 다품종 소량 생산, 프로토타입 단계에 적합하다. 91-95%의 정확도는 낮지만, 학습 시간 0분은 특정 시나리오에서 혁명적이다. 무료이므로 비용 부담이 없다. 프롬프트 엔지니어링으로 성능을 1-2%p 향상시킬 수 있다. 데이터가 축적되면 Dinomaly나 PatchCore로 전환하는 것을 권장한다.

VLM-AD는 explainable AI가 필수인 환경에서 선택한다. 규제 산업, 품질 보고서 자동화, 고객 보고서가 주요 용도이다. 96-97%의 정확도는 WinCLIP보다 높지만 전통적 방법보다는 낮다. API 비용($0.01-0.05/img)과 느린 속도(2-5초)를 고려하여, 전수 검사보다는 샘플링 검사나 중요 샘플 분석에 사용한다. 자연어 설명의 가치가 비용을 상회하는지 ROI 분석이 필요하다.

SuperSimpleNet과 UniNet은 특수 요구사항이 있을 때 고려한다. SuperSimpleNet(97.2%)은 unsupervised와 supervised를 통합하여 실용적 접근을 제공한다. UniNet(98.3%)은 contrastive learning으로 강건한 decision boundary를 학습한다. 그러나 일반적으로는 Dinomaly가 더 나은 선택이다.

결론적으로 foundation model 방식을 선택했다면, multi-class는 Dinomaly, zero-shot은 WinCLIP, explainable은 VLM-AD를 사용한다. Dinomaly는 2025년 이후 주류가 될 것으로 전망된다.

---

## 3. Performance Benchmarks

### 3.1 MVTec AD Results

MVTec AD 벤치마크에서 21개 모델의 종합 성능을 정량적으로 비교한다. 모든 수치는 15개 카테고리의 평균값이다.

| Tier | Model | Image AUROC | Pixel AUROC | Inference | Memory | Year |
|------|-------|-------------|-------------|-----------|--------|------|
| **SOTA** | PatchCore | **99.1%** | 98.2% | 50-100ms | 100-500MB | 2022 |
| **SOTA** | Dinomaly (single) | **99.2%** | 97.5% | 80-120ms | 300-500MB | 2025 |
| **SOTA** | Dinomaly (multi) | 98.8% | 97.5% | 80-120ms | 300-500MB | 2025 |
| **High** | Reverse Distillation | 98.6% | **98.5%** | 100-200ms | 500MB-1GB | 2022 |
| **High** | FastFlow | 98.5% | 97.8% | 20-50ms | 500MB-1GB | 2021 |
| **High** | UniNet | 98.3% | 97.0% | 50-80ms | 400-600MB | 2025 |
| **High** | CFlow | 98.2% | 97.6% | 100-150ms | 500MB-1GB | 2021 |
| **High** | EfficientAD | 97.8% | 97.2% | **1-5ms** | **<200MB** | 2024 |
| **Good** | CS-Flow | 97.9% | 97.3% | 80-120ms | 500MB-1GB | 2021 |
| **Good** | DRAEM | 97.5% | 96.8% | 50-100ms | 300-500MB | 2021 |
| **Good** | U-Flow | 97.6% | 97.1% | 90-140ms | 500MB-1GB | 2022 |
| **Good** | CFA | 96.5-97.5% | 96.0-97.0% | 40-70ms | 300-500MB | 2022 |
| **Good** | SuperSimpleNet | 97.2% | 95.8% | 40-60ms | 300-500MB | 2024 |
| **Medium** | DSR | 96.5-98.0% | 96.0-97.5% | 60-100ms | 400-600MB | 2022 |
| **Medium** | STFPM | 96.8% | 96.2% | 20-40ms | 500MB-1GB | 2021 |
| **Medium** | VLM-AD | 96-97% | 94-96% | 2-5s | API | 2024 |
| **Medium** | PaDiM | 96.5% | 95.8% | 30-50ms | 2-5GB | 2020 |
| **Medium** | DFKDE | 95.5-96.8% | 94.5-96.0% | 40-60ms | 500MB-1GB | 2022 |
| **Low** | FRE | 95-96% | 94-95% | 10-30ms | 300-500MB | 2023 |
| **Low** | DFM | 94.5-95.5% | 90-93% | 10-20ms | 50-100MB | 2019 |
| **Low** | WinCLIP | 91-95% | 89-93% | 50-100ms | 500MB-1.5GB | 2023 |
| **Deprecated** | GANomaly | 93-95% | 91-94% | 80-150ms | 500MB-1GB | 2018 |

주요 발견 사항은 다음과 같다. SOTA tier는 99% 이상의 정확도를 보인다. PatchCore(99.1%)가 3년간 single-class 최고 기록을 유지하고 있으며, Dinomaly(99.2% single, 98.8% multi)가 2025년 새로운 기준을 제시한다. High tier는 97.8-98.6%로 실무에서 널리 사용 가능한 수준이다. FastFlow(98.5%, 20-50ms)가 속도와 정확도의 균형으로 가장 인기가 높다. EfficientAD(97.8%, 1-5ms)는 실시간 처리의 혁명을 가져왔다.

Good tier는 96.5-97.9%로 특정 시나리오에 적합하다. DRAEM(97.5%)은 few-shot에, U-Flow(97.6%)는 자동 임계값에, CS-Flow(97.9%)는 다양한 크기 결함에 강점이 있다. Medium tier는 95-97%로 프로토타입이나 특수 용도에 사용된다. STFPM(96.8%)은 baseline으로, VLM-AD(96-97%)는 explainable AI로 가치가 있다.

Low tier는 91-96%로 실무 배포에 제한적이다. DFM(94.5-95.5%)은 빠른 검증용으로, WinCLIP(91-95%)은 zero-shot으로만 의미가 있다. Deprecated 모델인 GANomaly(93-95%)와 FRE(95-96%)는 더 나은 대체제가 있어 사용하지 않는다.

### 3.2 Category-wise Performance

MVTec AD의 15개 카테고리를 텍스처와 객체로 나누어 분석한다.

**Texture Categories (5개: Carpet, Grid, Leather, Tile, Wood)**

텍스처 카테고리는 반복 패턴과 질감 변화가 특징이다. 상위 성능 모델은 다음과 같다. PatchCore는 텍스처 평균 99.0%로 최고이며, Carpet 99.2%, Leather 99.1%에서 특히 강하다. FastFlow는 98.7%로 2위이며, Carpet 99.2%, Leather 99.1%로 텍스처에 최적화되어 있다. DSR은 복잡한 텍스처에 특화되어 Carpet 98.5%, Leather 98.0%를 보인다.

텍스처에서 강한 패러다임은 normalizing flow와 reconstruction-based이다. Flow는 텍스처의 확률 분포를 잘 모델링하고, reconstruction은 패턴 변형을 효과적으로 탐지한다. 약한 패러다임은 feature adaptation이다. DFM은 텍스처 평균 93-94%로 가장 낮다.

**Object Categories (10개: Bottle, Cable, Capsule, Hazelnut, Metal Nut, Pill, Screw, Toothbrush, Transistor, Zipper)**

객체 카테고리는 형태와 3D 구조가 특징이다. 상위 성능 모델은 다음과 같다. PatchCore는 객체 평균 99.2%로 최고이며, Bottle 100%, Zipper 99.8%, Hazelnut 99.6%에서 거의 완벽하다. Dinomaly는 99.3%로 single-class에서 PatchCore를 초과하며, 객체 인식에서 DINOv2의 강력함을 보여준다. Reverse Distillation은 98.7%로 3위이며, localization이 중요한 Screw, Metal Nut에서 강하다.

객체에서 강한 패러다임은 memory-based와 foundation model이다. Memory-based는 형태 정보를 잘 저장하고, foundation model은 대규모 학습으로 객체 인식에 특화되어 있다. 약한 패러다임은 normalizing flow이다. Flow는 텍스처보다 객체에서 1-2%p 낮다.

**Category-specific Recommendations**

Carpet, Leather 등 복잡한 텍스처는 FastFlow(99.2%) 또는 DSR(98.5%)을 선택한다. Grid, Tile 등 패턴 반복은 CS-Flow(98.5%)를 고려한다. Bottle, Zipper 등 명확한 형태는 PatchCore(100%, 99.8%)가 압도적이다. Screw, Metal Nut 등 미세 결함은 Reverse Distillation(pixel AUROC 98.5%)이 적합하다. Cable, Transistor 등 복잡한 구조는 Dinomaly(99.5%)가 강하다.

### 3.3 Speed Benchmarks

추론 속도를 5개 tier로 분류한다. 모든 측정은 NVIDIA RTX 3090 GPU, 배치 크기 1, 이미지 256x256 기준이다.

**Ultra-Fast (1-10ms): Real-time**

EfficientAD만이 이 tier에 속한다. 1-5ms의 속도는 초당 200-1000 프레임 처리를 가능하게 한다. 이는 고속 생산 라인(초당 100개 이상)에서 전수 검사를 현실화한다. CPU에서도 10-20ms로 작동하여 GPU 없이도 준실시간 처리가 가능하다. 97.8%의 정확도는 속도 대비 매우 우수하다. 엣지 디바이스(Jetson, Raspberry Pi)에서 실시간 처리가 가능하여 모바일 검사, 드론 검사 등 새로운 응용을 연다.

**Fast (10-50ms): Near Real-time**

DFM(10-20ms), FastFlow(20-50ms), STFPM(20-40ms)이 속한다. 초당 20-100 프레임 처리가 가능하여, 대부분의 생산 라인 속도에 대응할 수 있다. FastFlow는 98.5%의 높은 정확도와 20-50ms의 속도로 가장 균형 잡힌 선택이다. DFM은 가장 빠르지만(10-20ms) 성능이 낮다(94.5-95.5%). STFPM은 중간(20-40ms, 96.8%)이지만 더 나은 대체제가 있다.

**Medium (50-120ms): Batch Processing**

PatchCore(50-100ms), Dinomaly(80-120ms), DRAEM(50-100ms), UniNet(50-80ms), CFlow(100-150ms) 등 많은 모델이 속한다. 초당 8-20 프레임으로 일반적인 검사 환경에 적합하다. PatchCore는 최고 정확도(99.1%)를 제공하며, 50-100ms는 정밀 검사에서 허용 가능하다. Dinomaly는 multi-class 효율성으로 80-120ms가 합리적이다. DRAEM은 few-shot 능력으로 50-100ms의 가치가 있다.

**Slow (120-200ms): Precision Inspection**

Reverse Distillation(100-200ms)이 대표적이다. 98.6%의 정확도와 98.5%의 pixel AUROC로 속도를 희생하고 정밀도를 얻는다. 초당 5-10 프레임으로 고속 라인에는 부적합하지만, 정밀 검사(반도체, 의료)에서는 100-200ms도 허용된다. Localization이 중요한 환경에서 선택한다.

**Very Slow (200ms+): Special Purpose**

VLM-AD(2-5초)가 유일하다. 실시간 처리가 불가능하며, 샘플링 검사나 2차 검사에만 사용한다. 그러나 자연어 설명의 가치가 속도 희생을 상회할 수 있다. 품질 보고서 자동화, 규제 대응 등에서 시간당 수백 장만 처리하면 되는 경우 적합하다.

### 3.4 Memory Usage Analysis

메모리 사용량을 4개 tier로 분류한다. 모델 가중치와 추론 시 활성화 메모리를 합산한 값이다.

**Minimal (<200MB): Edge Deployment**

EfficientAD(<200MB)만이 이 tier에 속한다. Jetson Nano(4GB RAM), Raspberry Pi 4(4-8GB RAM) 등 저사양 엣지 디바이스에서 작동한다. 경량 PDN(50K 파라미터)과 간단한 autoencoder 덕분이다. GPU 메모리 1GB 미만에서 배포 가능하다.

**Small (200-500MB): Efficient Deployment**

PatchCore(100-500MB), DRAEM(300-500MB), SuperSimpleNet(300-500MB), CFA(300-500MB)가 속한다. GPU 메모리 2-4GB에서 작동하며, 대부분의 배포 환경에 적합하다. PatchCore는 coreset selection으로 메모리를 대폭 줄였다. DRAEM은 reconstruction network만 저장하여 효율적이다.

**Medium (500MB-1GB): Standard Deployment**

FastFlow(500MB-1GB), CFlow(500MB-1GB), STFPM(500MB-1GB), Reverse Distillation(500MB-1GB) 등 많은 모델이 속한다. GPU 메모리 4-8GB가 필요하며, 일반적인 GPU 서버에 적합하다. Flow network와 deep feature가 메모리를 많이 사용한다.

**Large (1GB+): High-end Deployment**

PaDiM(2-5GB), Dinomaly(1.5-2GB), WinCLIP(500MB-1.5GB)이 속한다. GPU 메모리 8GB 이상이 필요하다. PaDiM은 공분산 행렬 저장으로 2-5GB를 사용하며, 현대적 배포에는 부적합하다. Dinomaly는 foundation model(DINOv2)이 1.5-2GB이지만, multi-class 환경에서는 단일 모델(1.5-2GB)이 15개 모델(7.5GB)보다 효율적이다. WinCLIP은 CLIP 모델(500MB-1GB)이 크지만, zero-shot 가치가 있다.

**Multi-class Memory Efficiency**

Multi-class 환경에서 메모리 효율성은 극적으로 달라진다. 15개 제품 검사 시나리오를 보자. 전통적 방법(PatchCore)은 15개 × 300MB = 4.5GB, FastFlow는 15개 × 800MB = 12GB이다. Dinomaly는 단일 모델 1.5-2GB로 70-90% 메모리를 절감한다. 이는 고성능 GPU(A100, $10,000) 대신 저가 GPU(RTX 3060, $300)로 배포 가능함을 의미한다.

### 3.5 Benchmark Limitations

MVTec AD 벤치마크는 표준 평가 도구이지만 한계가 있다. 실무 적용 시 다음 사항을 인지해야 한다.

첫째, 통제된 환경이다. MVTec AD는 고품질 이미지, 일관된 조명, 명확한 결함을 가진다. 실무 환경은 다양한 이미지 품질(카메라 노이즈, 초점 흐림), 조명 변화(시간대, 계절), 모호한 경계 사례(정상과 이상의 애매한 중간)를 포함한다. 벤치마크 성능이 실제 라인에서 3-5%p 하락할 수 있다.

둘째, 제한된 카테고리이다. MVTec AD는 15개 카테고리만 포함하지만, 실제 산업은 수백 개의 제품과 결함 유형을 다룬다. 특정 산업(식품, 제약, 섬유)은 MVTec AD에 포함되지 않는다. 벤치마크 성능이 모든 도메인에 일반화되지 않을 수 있다.

셋째, false positive 비용을 고려하지 않는다. AUROC는 정상과 이상의 분리 능력만 측정한다. 실무에서는 false positive(정상을 불량으로 판정)와 false negative(불량을 정상으로 판정)의 비용이 다르다. 어떤 산업에서는 false positive 비용이 false negative보다 10배 높을 수 있다. 벤치마크는 이를 반영하지 않는다.

넷째, 추론 속도 변동이다. 벤치마크 속도는 단일 이미지 기준이다. 실무에서는 배치 처리, 전처리 시간, I/O 오버헤드가 추가된다. 또한 GPU 사용률, 동시 실행 모델 수, 시스템 부하에 따라 속도가 변동한다. 벤치마크 속도의 1.5-2배를 실제 속도로 예상해야 한다.

다섯째, 데이터 분포 차이이다. MVTec AD의 정상/이상 비율은 각 카테고리마다 다르다. 실무 환경의 불량률(보통 0.1-1%)과 다를 수 있다. 클래스 불균형이 심한 환경에서는 AUROC가 성능을 과대평가할 수 있다.

결론적으로 MVTec AD 벤치마크는 모델 비교의 표준 도구이지만, 실무 적용 전에 실제 데이터로 검증이 필수이다. 벤치마크 + 파일럿 테스트의 조합이 최선의 접근법이다.

---

## 4. Trade-off Analysis

### 4.1 Accuracy vs Speed

정확도와 속도는 가장 명확한 trade-off 관계를 보인다. 일반적으로 높은 정확도는 복잡한 모델을 요구하고, 이는 더 많은 계산을 의미한다.

PatchCore(99.1%, 50-100ms)는 정확도를 우선한다. K-NN search와 patch matching은 계산 집약적이지만, 최고 수준의 탐지 능력을 제공한다. 반도체, 의료, 항공 등 불량 유출 비용이 매우 높은 환경에서는 50-100ms가 허용 가능하다. 하루 10,000개 검사 시 1,000초(16분)가 소요되며, 이는 대부분의 생산 계획에 수용된다.

EfficientAD(97.8%, 1-5ms)는 속도를 우선한다. 경량 PDN과 간단한 autoencoder는 최소 계산만 수행한다. 1.3%p의 정확도 희생으로 20-200배의 속도 향상을 얻는다. 고속 라인(초당 100개)에서는 10ms 이하가 필수이므로, EfficientAD가 유일한 선택이다. 하루 1,000,000개 검사 시 5,000초(83분)로 전수 검사가 가능하다.

FastFlow(98.5%, 20-50ms)는 중간 지점이다. 98.5%는 대부분의 품질 기준을 충족하며, 20-50ms는 준실시간 처리를 가능하게 한다. 일반적인 생산 속도(초당 10-20개)에 적합하다. 하루 100,000개 검사 시 5,000초(83분)로 합리적이다.

Trade-off 곡선을 보면, 99% → 98%로 1%p 하락 시 속도가 2-3배 향상된다(PatchCore → FastFlow). 98% → 97.8%로 0.2%p 하락 시 속도가 10-50배 향상된다(FastFlow → EfficientAD). 이는 97-99% 구간에서 작은 정확도 희생이 큰 속도 향상을 가져옴을 보여준다.

의사결정 가이드는 다음과 같다. 불량 유출 비용이 매우 높다면(반도체, 의료) 정확도를 우선하고 PatchCore를 선택한다. 고속 라인이거나 전수 검사가 필수라면 속도를 우선하고 EfficientAD를 선택한다. 일반적인 경우에는 FastFlow의 균형이 최선이다.

### 4.2 Accuracy vs Memory

정확도와 메모리의 trade-off는 직관과 다를 수 있다. 높은 정확도가 항상 큰 메모리를 요구하지는 않는다.

PaDiM(96.5%, 2-5GB)은 정확도 대비 메모리가 비효율적이다. 모든 패치 위치의 공분산 행렬을 저장하여 메모리가 폭발한다. 96.5%는 높지만, 2-5GB는 현대적 배포에 부적합하다.

PatchCore(99.1%, 100-500MB)는 정확도와 메모리 모두 우수하다. Coreset selection으로 메모리를 90% 줄이면서 성능은 오히려 향상시켰다. 이는 "간단한 아이디어가 복잡한 문제를 해결"한 사례이다. 100-500MB는 대부분의 GPU 환경에서 수용 가능하다.

EfficientAD(97.8%, <200MB)는 메모리를 극한으로 최적화했다. 경량 PDN(50K 파라미터)과 작은 autoencoder로 200MB 미만을 달성했다. 1.3%p의 정확도 희생으로 메모리를 2-25배 절감했다. 엣지 디바이스(Jetson, Raspberry Pi)에 필수이다.

Dinomaly(98.8%, 1.5-2GB)는 multi-class 효율성으로 달라진다. Single-class에서는 1.5-2GB가 큰 편이지만, multi-class에서는 단일 모델(1.5-2GB)이 15개 모델(4.5-12GB)보다 효율적이다. 메모리 70-90% 절감 효과가 있다.

Trade-off 곡선을 보면, 메모리와 정확도가 반드시 비례하지 않는다. PatchCore는 PaDiM 대비 메모리는 1/5이지만 정확도는 오히려 높다. 이는 메모리 효율적 알고리즘(coreset)이 성능까지 향상시킬 수 있음을 보여준다.

의사결정 가이드는 다음과 같다. GPU 메모리가 8GB 이상이라면 메모리를 크게 고려하지 않아도 된다. 대부분의 모델(500MB-1GB)이 작동한다. GPU 메모리가 4GB 이하라면 EfficientAD, DRAEM, PatchCore를 선택한다. Mult-class 환경에서는 Dinomaly의 메모리 효율성이 결정적이다.

### 4.3 Speed vs Memory

속도와 메모리의 trade-off는 명확하지 않다. 빠른 모델이 항상 작은 메모리를 사용하지는 않는다.

EfficientAD(1-5ms, <200MB)는 속도와 메모리 모두 최고이다. 극한 최적화로 두 가지를 동시에 달성했다. 이는 특정 목표(실시간 처리)를 위한 전면적 설계의 결과이다.

DFM(10-20ms, 50-100MB)도 속도와 메모리 모두 우수하다. 간단한 PCA와 Mahalanobis distance만 사용하여 효율적이다. 그러나 성능이 낮다(94.5-95.5%).

FastFlow(20-50ms, 500MB-1GB)는 빠르지만 메모리가 크다. Flow network가 많은 파라미터를 가지기 때문이다. 속도는 2D flow로 최적화했지만, 메모리는 여전히 크다.

PatchCore(50-100ms, 100-500MB)는 중간 속도에 적절한 메모리이다. Coreset이 메모리를 절감하지만, k-NN search가 속도를 제한한다.

Trade-off 곡선을 보면, 속도와 메모리가 약한 상관관계를 보인다. 극한 최적화(EfficientAD)를 제외하면 둘은 독립적이다. 빠른 모델(FastFlow)이 큰 메모리를 사용하기도 하고, 느린 모델(Reverse Distillation)이 중간 메모리를 사용하기도 한다.

의사결정 가이드는 다음과 같다. 속도와 메모리가 모두 제한적이라면(엣지 디바이스) EfficientAD를 선택한다. 속도만 제한적이라면(고속 라인, GPU 서버) FastFlow나 STFPM을 선택한다. 메모리만 제한적이라면(4GB GPU) EfficientAD, DRAEM, PatchCore를 선택한다.

### 4.4 Three-way Trade-off Visualization

정확도, 속도, 메모리의 3차원 trade-off를 시각화하면 모델 선택의 전체 그림이 보인다.

```
         Accuracy (99%)
              △
             /|\
            / | \
           /  |  \
          /   |   \
         /    |    \
        / PatchCore \
       /    99.1%    \
      /    50-100ms   \
     /    100-500MB    \
    /                   \
   /      Dinomaly       \
  /    98.8%, 80-120ms    \
 /     300-500MB (multi)   \
/___________________________\
Speed (1ms)    FastFlow    Memory (50MB)
             98.5%, 20-50ms
             500MB-1GB
                |
           EfficientAD
        97.8%, 1-5ms, <200MB
```

이 3차원 공간에서 모델들은 다음과 같이 위치한다.

정확도 정점 근처는 PatchCore(99.1%), Dinomaly(99.2% single), Reverse Distillation(98.6%)이다. 이들은 속도와 메모리를 희생하고 최고 정확도를 달성한다. 반도체, 의료, 항공에 적합하다.

속도 정점 근처는 EfficientAD(1-5ms), DFM(10-20ms), STFPM(20-40ms)이다. EfficientAD는 메모리도 작아 이상적이다. 고속 라인, 엣지 디바이스에 적합하다.

메모리 정점 근처는 EfficientAD(<200MB), DFM(50-100MB), PatchCore(100-500MB)이다. EfficientAD와 DFM은 속도도 빠르지만 정확도가 낮다. PatchCore는 정확도도 높아 균형이 좋다.

중앙 균형점은 FastFlow(98.5%, 20-50ms, 500MB-1GB)와 Dinomaly(98.8%, 80-120ms, 300-500MB multi)이다. 세 가지가 모두 합리적 수준이어서 가장 많이 사용된다.

극단점은 PaDiM(96.5%, 30-50ms, 2-5GB)과 VLM-AD(96-97%, 2-5s, API)이다. 세 가지가 모두 극단적이어서 일반적 사용에 부적합하다.

이 시각화에서 얻는 통찰은 다음과 같다. 세 가지를 모두 최고로 만족하는 모델은 없다. EfficientAD가 속도와 메모리에서 최고이지만 정확도는 97.8%이다. PatchCore가 정확도에서 최고이지만 속도와 메모리는 중간이다. 대부분의 실무 프로젝트는 한 가지를 우선하고 나머지를 수용해야 한다. 균형점(FastFlow, Dinomaly)이 가장 많은 시나리오를 커버한다.

### 4.5 Impossible Combinations

현재 기술로는 달성할 수 없는 조합이 명확히 존재한다.

99%+ 정확도 + 10ms 미만 속도 + 100MB 미만 메모리는 불가능하다. PatchCore(99.1%)는 50-100ms가 필요하고, EfficientAD(1-5ms, <200MB)는 97.8%이다. 이 gap을 메우는 모델은 없다. 이론적으로도 어려운데, 높은 정확도는 복잡한 특징 추출(큰 모델)이나 정교한 매칭(많은 계산)을 요구하기 때문이다.

Multi-class 99%+ + 실시간 처리(<10ms) + 엣지 디바이스(< 500MB)도 불가능하다. Dinomaly(98.8% multi)는 80-120ms와 1.5-2GB가 필요하다. Foundation model의 크기와 계산량은 현재 하드웨어로 실시간 처리를 제한한다.

Zero-shot 99%+는 불가능하다. WinCLIP(91-95%)과 VLM-AD(96-97%)가 최고이며, 학습 데이터 없이 99%를 달성한 모델은 없다. Foundation model이 발전하면 95-97%까지 향상될 것으로 예상되지만, 99%는 2028년 이후에나 가능할 것이다.

Few-shot(10-50장) 99%+도 어렵다. DRAEM(97.5%)이 현재 최고이며, 소량 데이터로 99%를 달성한 모델은 없다. 100장 이상의 데이터가 99% 달성의 현실적 기준이다.

CPU 실시간 처리(< 10ms) + 98%+는 불가능하다. EfficientAD가 CPU에서 10-20ms와 97.8%를 달성하지만, 10ms 미만이나 98% 이상은 불가능하다. GPU 없이는 복잡한 계산이 제한되기 때문이다.

이러한 불가능한 조합을 요구하는 프로젝트는 요구사항을 재검토해야 한다. 우선순위를 명확히 하고, 가장 중요한 한두 가지에 집중하는 것이 현실적이다. 또는 단계적 접근(초기에는 zero-shot WinCLIP, 데이터 축적 후 PatchCore로 전환)이 효과적이다.

---

## 5. Scenario-Based Selection Guide

### 5.1 Maximum Accuracy (>99%)

최고 정확도가 필수인 환경은 불량 유출 비용이 매우 높은 경우이다. 반도체 웨이퍼 검사에서 미세 결함을 놓치면 전체 웨이퍼가 폐기되어 수백만 원의 손실이 발생한다. 의료 기기에서 불량품은 환자 안전에 직결되며 법적 책임까지 초래한다. 항공 부품은 단 하나의 결함도 치명적 사고로 이어질 수 있다.

**1순위: PatchCore (99.1%)**

Single-class 환경에서 3년간 최고 기록을 유지하고 있다. Pixel AUROC 98.2%로 localization 능력도 우수하다. Coreset selection으로 메모리를 100-500MB로 효율화하여 실용적이다. 추론 속도 50-100ms는 정밀 검사에서 허용 가능하다. 하루 10,000개 검사 시 총 1,000초(16분)가 소요되며, 대부분의 생산 일정에 수용된다.

적용 방법은 다음과 같다. 정상 샘플 100-500장을 수집한다. Pre-trained backbone(Wide ResNet50)으로 특징을 추출한다. Greedy coreset selection으로 대표 패치를 선택한다(보통 10-20% 선택). K-NN search로 이상 점수를 계산한다. 임계값은 정상 샘플의 99 percentile로 설정한다.

주의사항은 다음과 같다. 조명 변화나 카메라 변경 시 재학습이 필요하다. 정상 패턴이 계절적으로 변화하면 분기별 재학습을 고려한다. 학습 데이터에 미묘한 이상이 포함되지 않도록 엄격한 품질 관리가 필요하다.

**2순위: Dinomaly (99.2% single, 98.8% multi)**

Multi-class 환경에서는 1순위가 된다. Single-class로 사용하면 PatchCore를 초과하는 99.2%를 달성한다. DINOv2의 강력한 self-supervised 특징 덕분이다. 추론 속도 80-120ms는 PatchCore보다 약간 느리지만 여전히 수용 가능하다.

Multi-class의 혁명적 효과는 다음과 같다. 15개 제품 검사 시 전통적 방법은 15개 모델(총 7.5GB, 15시간 학습)이 필요하다. Dinomaly는 1개 모델(500MB, 3-5시간 학습)로 처리한다. 메모리 93% 절감, 배포 시간 80% 단축, 관리 복잡도 대폭 감소의 효과가 있다. 신제품 추가 시 데이터만 추가하고 재학습하면 되므로 확장이 용이하다.

**3순위: Reverse Distillation (98.6%)**

Pixel-level localization이 특히 중요한 경우 선택한다. Pixel AUROC 98.5%는 현재까지 최고 수준이다. 결함의 정확한 위치를 파악하는 능력이 뛰어나, 근본 원인 분석과 공정 개선에 유용하다. 추론 속도 100-200ms는 느리지만, 정밀 검사에서는 허용된다.

적용 분야는 반도체 웨이퍼(미세 패턴 결함), 의료 영상(병변 localization), 정밀 기계 부품(마이크로 크랙)이다. One-class embedding을 타겟 도메인에서 학습하여 ImageNet의 일반적 특징보다 도메인 특화된 표현을 얻는다.

**선택 기준 요약**

Single-class 환경에서 최고 정확도만 필요하면 PatchCore를 선택한다. Multi-class 환경이거나 향후 제품 추가 가능성이 있으면 Dinomaly를 선택한다. Pixel-level localization이 특히 중요하면 Reverse Distillation을 고려한다. 세 모델 모두 99% 근처의 성능을 제공하므로, 부가 요구사항(multi-class, localization)에 따라 선택한다.

### 5.2 Real-time Processing (<10ms)

실시간 처리는 고속 생산 라인에서 필수이다. 초당 100개 이상 생산되는 라인에서는 10ms 이하의 추론 시간이 요구된다. 엣지 디바이스(Jetson, Raspberry Pi)에서는 제한된 컴퓨팅 파워로 실시간 처리가 필요하다. 모바일 검사나 드론 검사에서도 즉각적인 피드백이 중요하다.

**유일한 선택: EfficientAD (1-5ms)**

EfficientAD는 1-5ms의 혁명적 속도로 실시간 처리를 현실화했다. 이는 다른 모든 모델을 20-200배 압도하는 수치이다. 초당 200-1000 프레임 처리가 가능하여, 가장 빠른 생산 라인에서도 전수 검사를 할 수 있다.

기술적 혁신은 다음과 같다. Patch Description Network(PDN)는 약 50K 파라미터만 가지는 극도로 경량화된 네트워크이다. 작은 autoencoder와 하이브리드로 결합하여 재구성과 특징 추출을 동시에 수행한다. 모든 연산을 최적화하여 GPU뿐 아니라 CPU에서도 실시간 처리가 가능하다.

성능은 97.8%로 PatchCore(99.1%)보다 1.3%p 낮지만, 실용적으로 충분히 높다. 불량률 1% 환경에서 97.8%는 98% 이상의 불량을 잡아낸다. 메모리는 200MB 미만으로 엣지 디바이스에서도 작동한다.

적용 시나리오는 다음과 같다. 고속 생산 라인에서 초당 100개 이상 생산되는 경우, 10ms 이하가 필수이다. 예를 들어 초당 200개 생산 시 5ms 이하여야 전수 검사가 가능하다. EfficientAD(1-5ms)는 이를 만족하는 유일한 모델이다.

엣지 디바이스 배포에서 Jetson Nano(4GB RAM, GPU)나 Raspberry Pi 4(4-8GB RAM, CPU)에서 작동한다. GPU에서 1-5ms, CPU에서 10-20ms로 여전히 실시간에 가깝다. 200MB 미만의 메모리는 저사양 환경에서도 수용 가능하다.

모바일 검사에서 스마트폰이나 태블릿에서 실시간 품질 검사를 수행할 수 있다. 현장에서 즉시 피드백을 제공하여 작업 효율을 높인다. 드론 검사에서 대형 구조물(교량, 송전탑)을 드론으로 촬영하면서 실시간으로 결함을 탐지한다.

CPU 환경에서 GPU가 없거나 사용할 수 없는 환경에서도 EfficientAD는 CPU만으로 10-20ms를 달성한다. 이는 GPU 비용을 절감하고, 폐쇄망 환경에서도 배포를 가능하게 한다.

주의사항은 다음과 같다. 97.8%의 정확도가 요구사항을 충족하는지 확인한다. 반도체나 의료처럼 99% 이상이 필수인 환경에서는 부적합할 수 있다. 배치 처리를 활용하면 throughput을 더 높일 수 있다. 배치 크기 16-32로 초당 수천 장 처리가 가능하다. 모델 양자화(INT8)를 적용하면 속도를 추가로 1.5-2배 향상시킬 수 있다.

**대안이 없는 이유**

DFM(10-20ms)이 두 번째로 빠르지만 성능이 94.5-95.5%로 너무 낮다. STFPM(20-40ms)은 실시간 기준(10ms)을 만족하지 못한다. FastFlow(20-50ms)도 마찬가지이다. 따라서 실시간 처리가 필수라면 EfficientAD가 유일한 현실적 선택이다.

### 5.3 Multi-class Environment

여러 제품을 동시에 검사하는 환경은 현대 제조업에서 점점 흔해지고 있다. 스마트 팩토리에서는 하나의 라인에서 다양한 제품을 생산한다. 계절 상품이나 주문 생산에서는 제품이 자주 바뀐다. 검사 장비를 여러 라인에서 공유하는 경우도 있다.

**압도적 선택: Dinomaly (98.8% multi-class)**

Dinomaly는 multi-class 환경을 위해 특별히 설계된 모델이다. 단일 모델로 모든 클래스를 처리하는 혁명적 접근법이다. 98.8%의 성능은 SOTA 수준에 근접하며, 실무에서 충분히 높다.

비교 분석은 15개 제품 검사 시나리오로 보자. 전통적 방법(PatchCore)은 15개 독립 모델이 필요하다. 각 모델당 300MB로 총 4.5GB이다. 각 모델을 1-2시간씩 학습하므로 총 15-30시간이 소요된다. 배포 시 15개 모델을 개별 관리해야 한다. GPU 메모리 8GB가 필요하다. 신제품 추가 시 전체 과정을 반복해야 한다.

Dinomaly는 단일 통합 모델이다. 전체 모델 크기는 500MB(1.5-2GB foundation model 포함)이다. 모든 클래스를 한 번에 3-5시간 학습한다. 하나의 모델만 관리하면 된다. GPU 메모리 2-4GB면 충분하다. 신제품은 데이터만 추가하고 재학습하면 된다(증분 학습 가능).

효과는 다음과 같다. 메모리 93% 절감으로 고성능 GPU(A100, $10,000) 대신 저가 GPU(RTX 3060, $300)로 배포 가능하다. 학습 시간 80% 단축으로 빠른 배포와 반복 실험이 가능하다. 관리 복잡도 대폭 감소로 모델 버전 관리, 모니터링, 업데이트가 간소화된다. 확장성이 우수하여 신제품 추가가 용이하고, 제품 수가 증가해도 메모리가 선형 증가하지 않는다.

적용 방법은 다음과 같다. 모든 클래스의 정상 샘플을 수집한다(클래스당 50-200장). DINOv2 foundation model로 특징을 추출한다. Class-conditional memory bank를 구축한다. 각 클래스의 대표 특징만 저장한다. 추론 시 입력 이미지의 클래스를 먼저 식별한 후(classification head 또는 별도 classifier), 해당 클래스의 memory bank와 비교한다.

주의사항은 다음과 같다. 클래스 식별이 정확해야 한다. 잘못된 클래스와 비교하면 오탐이 발생한다. 클래스 간 유사도가 높으면 성능이 저하될 수 있다. 예를 들어 유사한 색상의 제품들은 구분이 어려울 수 있다. Foundation model(1.5-2GB)이 크므로 엣지 디바이스에는 부적합할 수 있다. 그러나 서버 환경에서는 문제없다.

**대안이 없는 이유**

전통적 방법(PatchCore, FastFlow 등)은 각 클래스마다 별도 모델이 필요하다. 클래스 수가 증가하면 메모리와 관리 복잡도가 선형 증가한다. 10개 이상의 클래스에서는 Dinomaly의 효율성이 결정적이다. 다른 foundation model(WinCLIP, VLM-AD)도 multi-class를 지원하지만 성능이 낮다(91-97%). Dinomaly는 98.8%로 실무 수준을 만족하는 유일한 multi-class 모델이다.

### 5.4 Instant Deployment (Zero-shot)

신제품 출시 당일에 품질 검사를 시작해야 하는 경우가 있다. 시장 출시 일정이 촉박하여 학습 데이터를 수집할 시간이 없다. 다품종 소량 생산에서 매주 새로운 제품이 나온다. 프로토타입 단계에서 설계가 자주 바뀐다. 이러한 환경에서 zero-shot 이상 탐지가 필요하다.

**1순위: WinCLIP (91-95%)**

WinCLIP은 OpenAI의 CLIP 모델을 활용하여 텍스트 프롬프트만으로 이상 탐지를 수행한다. 학습 시간이 0분이므로 즉시 배포할 수 있다. 무료이므로 비용 부담이 없다.

작동 원리는 다음과 같다. CLIP은 4억 개의 이미지-텍스트 쌍으로 학습되어, 이미지와 텍스트를 동일한 embedding space에 매핑한다. "a photo of a normal transistor"와 "a photo of a defective transistor"라는 텍스트 프롬프트를 준비한다. 입력 이미지를 두 프롬프트와 비교하여 유사도를 계산한다. Defective와의 유사도가 높으면 이상으로 판정한다.

성능은 91-95%로 전통적 방법(99.1%)보다 4-8%p 낮다. 그러나 특정 시나리오에서는 충분히 가치가 있다. 신제품 즉시 검사에서 출시 당일부터 품질 관리를 시작할 수 있다. 데이터 수집 중에도 검사를 수행하고, 데이터가 축적되면 PatchCore로 전환한다. 다품종 소량 생산에서 매주 나오는 새 제품마다 학습하는 것은 비현실적이다. WinCLIP으로 모든 제품을 커버하고, 주요 제품만 전통적 방법으로 전환한다. 프로토타입 단계에서 설계가 자주 바뀌므로 학습이 낭비일 수 있다. WinCLIP으로 초기 검사를 수행하고, 설계 확정 후 본격 모델을 구축한다.

프롬프트 엔지니어링으로 성능을 향상시킬 수 있다. 구체적인 설명을 사용한다. "defective"보다 "scratched surface with visible marks"가 더 효과적이다. 여러 프롬프트를 조합한다. "scratch", "dent", "stain" 등 개별 결함 유형에 대한 프롬프트를 각각 사용하고, 최대값을 이상 점수로 사용한다. 도메인 특화 용어를 활용한다. "printed circuit board with copper trace defects"처럼 산업 용어를 사용한다.

주의사항은 다음과 같다. 91-95%가 요구사항을 충족하는지 확인한다. 정밀 검사에는 부족할 수 있다. 프롬프트 품질에 성능이 크게 좌우된다. 여러 프롬프트를 실험하고 최적을 찾아야 한다. 세밀한 결함(마이크로 스크래치, 미세 오염)은 탐지하기 어렵다. CLIP이 고수준 semantic에 특화되어 있기 때문이다. 데이터가 축적되면 PatchCore나 Dinomaly로 전환하는 것을 강력히 권장한다. 성능을 4-8%p 향상시킬 수 있다.

**2순위: VLM-AD (96-97%)**

VLM-AD는 GPT-4V 등 vision-language model을 활용한다. WinCLIP보다 성능이 1-2%p 높고, 자연어 설명을 제공한다는 추가 가치가 있다. 그러나 API 비용($0.01-0.05/img)과 느린 속도(2-5초)가 단점이다.

적용 시나리오는 설명이 필요한 zero-shot이다. 신제품 출시 보고서에 결함 유형, 위치, 원인을 자동으로 생성한다. 고객 데모에서 자연어 설명으로 시스템 능력을 보여준다. 규제 대응에서 초기부터 설명 가능한 검사를 수행한다.

비용 고려사항은 다음과 같다. 프로토타입 단계(월 1,000장)는 $10-50로 허용 가능하다. 초기 생산(월 10,000장)은 $100-500으로 합리적이다. 본격 생산(월 100,000장)은 $1,000-5,000으로 부담스러우므로 전환을 고려한다.

### 5.5 Few-shot Learning (10-50 samples)

신제품 출시 초기에는 정상 샘플을 많이 확보하기 어렵다. 생산 초기 몇 주간은 10-50장만 수집 가능하다. 희귀 결함은 발생 빈도가 낮아 사례가 거의 없다. 데이터 수집 자체가 어려운 환경도 있다. 고비용 샘플(귀금속, 희토류), 접근 제한(클린룸), 파괴적 검사 등이다.

**독보적 선택: DRAEM (97.5%)**

DRAEM은 10-50장의 정상 샘플만으로 97.5%를 달성하는 유일한 모델이다. 다른 패러다임(100-500장 필요)보다 10배 적은 데이터로 높은 성능을 보인다. 이는 simulated anomaly의 혁신 덕분이다.

작동 원리는 다음과 같다. 정상 이미지에 Perlin noise를 사용하여 인위적 결함을 추가한다. 텍스처 소스(DTD 데이터셋)와 blending하여 자연스러운 결함 패턴을 생성한다. Reconstruction network가 augmented image에서 정상 이미지를 복원하도록 학습한다. Discriminative network가 어디가 이상인지 segmentation map을 생성한다. SSIM loss(구조적 유사성)와 focal loss(pixel-wise classification)를 결합한다.

왜 10-50장으로 충분한가. Simulated anomaly가 실제 결함 패턴을 근사한다. 정상 manifold를 명확히 학습할 수 있다. Supervised learning이므로 학습 신호가 명확하고, 수렴이 빠르고 안정적이다(2-4시간). Augmentation으로 실질적인 학습 데이터가 수천 장으로 증가한다. 10장 정상 × 100 augmentation = 1,000장 학습 데이터이다.

적용 시나리오는 다음과 같다. 신제품 출시 초기에서 생산 시작 후 1-2주간은 데이터가 부족하다. DRAEM으로 즉시 검사를 시작하고, 데이터가 100장 이상 축적되면 PatchCore로 전환한다. 희귀 결함 학습에서 연간 몇 건만 발생하는 결함도 DRAEM의 simulated anomaly로 학습 가능하다. 고비용 샘플에서 귀금속이나 희토류 제품은 샘플 확보 비용이 높다. 10-50장만으로 학습하여 비용을 절감한다.

성능 최적화 팁은 다음과 같다. Simulated anomaly의 품질이 중요하다. 다양한 텍스처 소스를 시도하고, 실제 결함과 유사한 패턴을 찾는다. Augmentation 강도를 조절한다. 너무 강하면 비현실적이고, 너무 약하면 학습이 부족하다. 정상 샘플의 품질을 보장한다. 10-50장이므로 각 샘플이 중요하다. 미묘한 이상이 포함되지 않도록 주의한다. 데이터가 축적되면 전환을 계획한다. 100장 이상 확보 시 PatchCore(99.1%)로 전환하여 성능을 1.6%p 향상시킨다.

주의사항은 다음과 같다. 97.5%는 PatchCore(99.1%)보다 1.6%p 낮다. 이것이 허용되는지 확인한다. Simulated anomaly가 실제 결함과 다를 수 있다. 초기 배포 후 실제 불량 샘플로 검증이 필요하다. 복잡한 3D 구조나 특수 재질에서는 시뮬레이션이 어려울 수 있다.

### 5.6 Quality Report Automation

품질 보고서 작성은 시간이 많이 소요되는 작업이다. 품질 엔지니어가 수동으로 결함을 분석하고, 원인을 추정하고, 보고서를 작성한다. 고객 보고서는 전문 용어를 이해하기 쉬운 언어로 번역해야 한다. 규제 산업에서는 "왜 불량으로 판정했는가"를 문서화해야 한다.

**최적 선택: VLM-AD (96-97%)**

VLM-AD는 GPT-4V 등 vision-language model을 활용하여 자연어로 결함을 설명한다. 단순히 이상 점수만 제공하는 것이 아니라, 결함의 모든 측면을 상세히 기술한다.

출력 예시를 보자. 전통적 모델은 "Anomaly Score: 0.87, Status: Defective"만 제공한다. VLM-AD는 다음을 제공한다. 결함 유형은 "Scratch (linear surface damage)"이다. 위치는 "Upper left corner, approximately 15mm from the top edge and 10mm from the left edge"이다. 크기는 "5mm length × 0.5mm width, depth appears to be superficial"이다. 심각도는 "Moderate - affects surface quality but does not compromise structural integrity"이다. 가능한 원인은 "Likely caused by improper handling during assembly, possibly from contact with a hard tool or fixture"이다. 권장사항은 "Inspect handling procedures at assembly station 3. Consider adding protective padding to fixtures. Train operators on gentle handling techniques"이다. 다음 조치는 "Route to secondary inspection for detailed evaluation. Document for quality trend analysis"이다.

이러한 상세 설명의 가치는 막대하다. 품질 엔지니어는 즉시 근본 원인을 파악하고, 공정 개선 방향을 얻는다. 생산 관리자는 어떤 스테이션에 문제가 있는지 알고, 작업자 교육이나 설비 개선을 결정한다. 고객은 전문가가 아니어도 문제를 이해하고, 개선 노력을 확인할 수 있다. 감사 담당은 명확한 근거 문서를 확보하고, 규제 대응이 용이해진다.

적용 시나리오는 다음과 같다. 일일 품질 보고서에서 하루 종료 시 자동으로 보고서를 생성한다. 결함 유형별 통계, 주요 원인, 개선 권장사항을 포함한다. 고객 보고서에서 월간 품질 보고서를 자연어로 자동 생성한다. 기술적 세부사항 없이 이해하기 쉬운 설명을 제공한다. 규제 대응에서 의료 기기나 항공 부품의 감사 추적을 자동화한다. "왜 불량으로 판정했는가"에 대한 명확한 답변을 제공한다. 근본 원인 분석에서 반복되는 결함 패턴을 자동으로 추적한다. 공정의 어느 부분에 문제가 있는지 파악한다.

비용 고려사항은 다음과 같다. API 비용은 이미지당 $0.01-0.05이다. 일일 100장 보고서는 월 $30-150으로 합리적이다. 월간 1,000장 분석은 월 $10-50으로 저렴하다. 전수 검사(월 100,000장)는 월 $1,000-5,000으로 부담스럽다. 샘플링 검사나 중요 샘플만 분석하는 것을 권장한다.

속도 고려사항은 2-5초/이미지이다. 실시간 처리는 불가능하며, 배치 처리로 야간에 분석한다. 일일 1,000장도 2,000-5,000초(30-80분)로 처리 가능하다.

대안으로 on-premise VLM을 고려할 수 있다. LLaMA, Vicuna 등 오픈소스 VLM을 자체 서버에 배포한다. API 비용이 없고, 데이터 프라이버시를 보장한다. 그러나 초기 인프라 비용(GPU 서버)과 성능이 GPT-4V보다 낮을 수 있다(3-5%p 차이).

### 5.7 Balanced General Inspection

대부분의 실무 환경은 극단적인 요구사항이 아니라 균형 잡힌 성능이 필요하다. 정확도는 98% 이상이면 충분하고, 속도는 초당 10-20장 처리 가능하면 되며, 메모리는 1GB 이하면 합리적이다. 이러한 일반적인 검사 환경을 위한 추천이다.

**1순위: FastFlow (98.5%, 20-50ms)**

FastFlow는 속도와 정확도의 최고의 균형을 제공한다. 98.5%는 대부분의 품질 기준을 충족하며, 20-50ms는 준실시간 처리를 가능하게 한다. 초당 20-50장 처리는 일반적인 생산 속도에 적합하다. 확률론적 해석이 가능하여 log-likelihood 기반 의사결정을 할 수 있다.

적용 분야는 일반 제조업의 품질 검사이다. 전자 부품, 플라스틱 성형품, 금속 가공품 등 대부분의 산업에 적용 가능하다. Pixel AUROC 97.8%로 결함 위치 파악도 우수하다. 학습 시간 30-60분으로 빠른 개발 주기를 지원한다.

선택 이유는 다음과 같다. 98.5%는 불량률 1% 환경에서 98.5% 이상의 불량을 잡아낸다. 실용적으로 충분히 높다. 20-50ms는 초당 20-50장으로 대부분의 라인 속도(초당 10-20개)를 여유있게 처리한다. 500MB-1GB 메모리는 일반적인 GPU 환경(4-8GB)에서 문제없다. Normalizing flow의 확률론적 해석은 통계적 공정 관리와 통합하기 좋다.

**2순위: Dinomaly (98.8%, 80-120ms)**

Multi-class 가능성이 있거나 향후 제품 추가가 예상되면 Dinomaly를 선택한다. 98.8%(multi-class)는 FastFlow(98.5%)보다 0.3%p 높다. Single-class로 사용하면 99.2%로 PatchCore를 초과한다. 80-120ms는 FastFlow(20-50ms)보다 느리지만 여전히 합리적이다. 초당 8-15장 처리로 대부분의 환경에 적합하다.

장기적 관점에서 Dinomaly가 더 나은 선택일 수 있다. 향후 제품 추가 시 재사용 가능하다. 신제품 데이터만 추가하고 재학습하면 된다. 2025-2026년 multi-class 모델이 표준화될 것으로 전망된다. 미래 지향적 선택이다. Foundation model은 지속적으로 발전한다. OpenAI, Meta, Google의 투자로 성능이 계속 향상될 것이다.

**3순위: PatchCore (99.1%, 50-100ms)**

정확도가 특히 중요하거나 single-class가 확실하면 PatchCore를 선택한다. 99.1%는 최고 수준으로 불량 유출을 최소화한다. 50-100ms는 FastFlow보다 느리지만 일반적인 환경에서 충분하다. 초당 10-20장 처리로 대부분의 요구사항을 만족한다. 100-500MB 메모리는 효율적이고, 3년간 검증된 안정성을 제공한다.

선택 기준 요약은 다음과 같다. 일반적인 경우 FastFlow를 선택한다. 속도와 정확도가 모두 우수하고, 가장 널리 사용된다. Multi-class 가능성이 있으면 Dinomaly를 선택한다. 미래 확장성과 최신 기술을 원한다면 좋은 선택이다. 최고 정확도가 필요하고 single-class가 확실하면 PatchCore를 선택한다. 검증된 안정성과 최고 성능을 제공한다.

---

## 6. Hardware Environment Guide

### 6.1 GPU Server (8GB+ VRAM)

GPU 메모리 8GB 이상의 서버는 가장 일반적인 배포 환경이다. NVIDIA RTX 3080/3090(10-24GB), RTX 4080/4090(16-24GB), A100(40-80GB) 등이 해당한다. 이 환경에서는 거의 모든 모델을 제약 없이 사용할 수 있다.

**추천 모델 (정확도 순)**

PatchCore(99.1%, 100-500MB)는 최고 정확도가 필요한 경우 선택한다. 메모리가 충분하므로 제약이 없다. 배치 처리로 throughput을 높일 수 있다. 배치 크기 16-32로 초당 수백 장 처리가 가능하다.

Dinomaly(98.8% multi, 1.5-2GB)는 multi-class 환경에서 선택한다. Foundation model이 1.5-2GB이지만 8GB 환경에서 문제없다. 배치 크기 8-16도 가능하다.

Reverse Distillation(98.6%, 500MB-1GB)은 pixel-level localization이 중요한 경우 선택한다. Encoder-decoder 구조가 메모리를 사용하지만 8GB에서 충분하다.

FastFlow(98.5%, 500MB-1GB)는 균형 잡힌 성능을 원하는 경우 선택한다. Flow network가 메모리를 사용하지만 역시 문제없다.

**배치 처리 최적화**

GPU 메모리가 충분하므로 배치 크기를 늘려 throughput을 극대화한다. PatchCore는 배치 16-32로 초당 200-400장 처리가 가능하다. FastFlow는 배치 32-64로 초당 500-1000장 처리가 가능하다. Dinomaly는 배치 8-16으로 초당 100-200장 처리가 가능하다.

Mixed precision(FP16)을 사용하면 메모리를 절반으로 줄이고 속도도 1.5-2배 향상시킬 수 있다. 정확도 손실은 0.1%p 미만으로 미미하다.

**Multi-model 동시 실행**

8GB+ 환경에서는 여러 모델을 동시에 실행할 수 있다. 예를 들어 PatchCore(500MB) 3개를 동시에 로드하면 1.5GB이다. 3개 제품을 동시에 검사 가능하다. FastFlow(1GB) 4개를 동시에 로드하면 4GB이다. 4개 라인을 동시에 모니터링 가능하다.

그러나 multi-class 환경에서는 Dinomaly 1개(1.5-2GB)가 더 효율적이다. 메모리 절감과 관리 간소화의 이점이 크다.

### 6.2 Edge GPU (4GB VRAM)

엣지 GPU는 Jetson Nano/TX2/Xavier(4-8GB), 저가 데스크탑 GPU(GTX 1650/1660, 4GB) 등이다. 제한된 메모리에서도 합리적인 성능을 제공해야 한다.

**추천 모델 (메모리 효율 순)**

EfficientAD(97.8%, <200MB)는 최우선 추천이다. 200MB 미만으로 4GB 환경에서 여유롭다. 배치 크기 16-32도 가능하다. 1-5ms 속도로 초당 200-1000장 처리가 가능하다.

DRAEM(97.5%, 300-500MB)은 few-shot이 필요한 경우 선택한다. 메모리가 적절하고 성능도 합리적이다. 배치 크기 8-16 가능하다.

PatchCore(99.1%, 100-500MB)는 최고 정확도가 필요한 경우 선택한다. Coreset 크기를 조절하여 메모리를 최적화할 수 있다. 100-200MB로도 98% 이상 성능 유지가 가능하다.

**부적합한 모델**

FastFlow(500MB-1GB)는 단일 모델만 가능하다. 배치 처리나 다중 모델 실행이 어렵다. Dinomaly(1.5-2GB)는 4GB 환경에서 빠듯하다. 배치 크기 1-2만 가능하고, 다른 프로세스와 경합 시 메모리 부족이 발생할 수 있다. PaDiM(2-5GB)은 완전히 불가능하다.

**최적화 전략**

모델 양자화(INT8)를 적용하면 메모리를 2-4배 줄일 수 있다. EfficientAD는 50-100MB로, PatchCore는 50-200MB로 감소한다. 정확도 손실은 0.5-1%p로 수용 가능하다.

Gradient checkpointing을 사용하면 학습 시 메모리를 절감할 수 있다. 추론에는 영향이 없다.

Swap space를 활용하면 부족한 메모리를 보완할 수 있다. 그러나 속도가 10-100배 느려지므로 권장하지 않는다.

### 6.3 CPU Only

GPU가 없거나 사용할 수 없는 환경이다. 폐쇄망, 저비용 배포, 레거시 시스템 등이 해당한다. CPU만으로 실시간에 가까운 처리가 필요하다.

**유일한 현실적 선택: EfficientAD (10-20ms on CPU)**

EfficientAD는 CPU에서 10-20ms를 달성하는 유일한 모델이다. 경량 PDN(50K 파라미터)과 최적화된 연산 덕분이다. Intel i7/i9, AMD Ryzen 7/9 등 일반적인 CPU에서 작동한다. 97.8%의 정확도는 CPU 환경에서 놀라운 수준이다.

적용 시나리오는 다음과 같다. GPU가 없는 환경에서 저비용 PC나 레거시 시스템에 배포한다. 폐쇄망에서 GPU 드라이버 설치가 불가능한 경우이다. 엣지 디바이스에서 Raspberry Pi 4(4-8GB RAM)에서 작동한다. 초당 50-100장 처리가 가능하다. 모바일 환경에서 노트북이나 태블릿에서 배터리 효율적인 검사를 수행한다.

최적화 팁은 다음과 같다. Multi-threading을 활용한다. 8코어 CPU에서 배치 8로 throughput을 높인다. ONNX Runtime을 사용한다. PyTorch보다 1.5-2배 빠르다. Intel CPU에서 OpenVINO를 사용하면 2-3배 추가 향상이 가능하다. 모델 양자화(INT8)를 적용하면 속도를 2배 더 향상시킬 수 있다(10-20ms → 5-10ms).

**대안 (느리지만 가능)**

DFM(10-20ms on CPU)도 빠르지만 성능이 94.5-95.5%로 너무 낮다. FastFlow(200-500ms on CPU)는 느려서 실용적이지 않다. PatchCore(500-1000ms on CPU)는 매우 느려서 배치 처리에만 적합하다.

결론적으로 CPU 환경에서는 EfficientAD를 사용하거나, GPU를 추가하는 것을 권장한다. 저가 GPU(RTX 3060, $300)를 추가하면 20-200배 속도 향상을 얻을 수 있다.

### 6.4 Cloud/API

클라우드나 API 기반 배포는 초기 인프라 비용을 줄이고 확장성을 제공한다. AWS, Azure, GCP의 GPU 인스턴스나 API 서비스를 활용한다.

**추천 모델**

VLM-AD(API)는 GPT-4V, Claude 등 API를 직접 사용한다. 초기 인프라 비용이 없고, 즉시 사용 가능하다. 자연어 설명의 가치가 크다. 비용은 $0.01-0.05/img이다.

WinCLIP(API 또는 Self-hosted)는 CLIP API를 사용하거나 자체 서버에 배포한다. Zero-shot으로 즉시 사용 가능하고, 무료(self-hosted) 또는 저렴한 API 비용이다.

Dinomaly(Self-hosted on GPU instance)는 AWS p3.2xlarge(V100, $3/hour)에 배포한다. Multi-class로 여러 제품을 처리한다. 24시간 운영 시 월 $2,160이다. 프로젝트 규모에 따라 비용 효율적일 수 있다.

**비용 분석**

API 방식(VLM-AD)은 월 10,000장 처리 시 $100-500이다. 초기 비용이 없고 확장이 쉽다. 그러나 대량 처리 시 비용이 급증한다(월 100,000장 시 $1,000-5,000).

Self-hosted GPU instance는 p3.2xlarge(V100)가 시간당 $3이다. 월 24시간 운영 시 $2,160이다. 초기 설정 비용이 있지만 대량 처리에 유리하다. Break-even point는 월 40,000-200,000장이다(API 비용에 따라).

On-premise GPU 서버는 초기 하드웨어 비용 $5,000-15,000이다. 운영 비용(전력, 유지보수)은 월 $100-300이다. 장기적으로 가장 경제적이다. 12-24개월에 break-even을 달성한다.

**선택 기준**

프로토타입이나 저량 처리(월 10,000장 미만)는 API 방식을 선택한다. 초기 비용이 없고 빠르게 시작할 수 있다. 중량 처리(월 10,000-100,000장)는 Self-hosted GPU instance를 고려한다. 비용과 확장성의 균형이 좋다. 대량 처리(월 100,000장 이상)는 on-premise GPU 서버를 권장한다. 장기적으로 가장 경제적이다.

**데이터 프라이버시**

클라우드나 API 사용 시 데이터가 외부로 전송된다. 민감한 제품 정보나 영업 비밀이 포함된 경우 위험하다. On-premise나 self-hosted가 필수일 수 있다. VPC(Virtual Private Cloud)나 dedicated instance를 사용하면 프라이버시를 향상시킬 수 있다. 비용은 20-50% 증가한다.

---

## 7. Development Roadmap

### 7.1 Phase 1: Prototyping (1-2 weeks)

프로토타이핑 단계의 목표는 feasibility 검증과 성능 목표 설정이다. 이상 탐지가 해당 데이터에서 작동하는지 빠르게 확인하고, 어떤 수준의 정확도를 달성할 수 있는지 파악한다.

**추천 모델**

WinCLIP(zero-shot, 즉시)을 첫 번째로 시도한다. 학습 시간 0분으로 가장 빠르다. 텍스트 프롬프트만 작성하면 즉시 결과를 확인할 수 있다. 91-95%의 성능이 나오면 본격 개발 가치가 있다. 데이터 수집도 병행한다.

DFM(15분 학습)을 두 번째로 시도한다. 정상 샘플 50-100장만 수집하면 된다. PCA와 Mahalanobis distance로 간단히 구현한다. 94.5-95.5%의 성능이 나오면 프로젝트 진행을 확정한다.

**활동**

데이터 수집 계획을 수립한다. 필요한 정상 샘플 수(100-500장)를 결정한다. 수집 방법과 일정을 계획한다. 데이터 품질 기준을 정의한다(해상도, 조명, 각도).

성능 목표를 설정한다. 불량률과 검출 목표를 정의한다(예: 불량률 1%, 검출률 95%). 허용 가능한 false positive rate를 결정한다(예: 5% 이하). 목표 정확도를 설정한다(95%? 98%? 99%?).

하드웨어 요구사항을 파악한다. GPU 필요 여부를 결정한다(서버? 엣지?). 메모리 제약을 확인한다(4GB? 8GB? 무제한?). 속도 요구사항을 정의한다(실시간? 배치?).

예산을 수립한다. 하드웨어 비용을 추정한다(GPU $300-10,000). 인력 비용을 계산한다(엔지니어 × 주 × $2,500). 소프트웨어 비용을 고려한다(라이선스, API).

**의사결정 체크리스트**

프로토타입 성능이 목표에 근접하는가. WinCLIP이나 DFM에서 90% 이상 나오면 본격 개발로 진행한다. 60% 이하면 데이터나 문제 설정을 재검토한다.

데이터 수집이 가능한가. 정상 샘플 100-500장을 합리적 기간(2-4주) 내에 확보할 수 있는가. 불가능하면 few-shot(DRAEM) 또는 zero-shot(WinCLIP) 접근을 고려한다.

예산과 일정이 합리적인가. 하드웨어 + 인력 + 시간 총 비용이 기대 효과를 상회하는가. ROI가 긍정적이면 진행한다.

### 7.2 Phase 2: Optimization (2-4 weeks)

최적화 단계의 목표는 정확도 극대화이다. 프로토타입에서 검증된 feasibility를 바탕으로, 실무 배포 가능한 수준의 성능을 달성한다.

**추천 모델 (요구사항에 따라)**

정확도 최우선(>99%)이면 PatchCore(99.1%)나 Dinomaly(99.2% single, 98.8% multi)를 선택한다. 학습 데이터 100-500장을 수집한다. 1-5시간 학습으로 SOTA 성능을 달성한다.

균형 필요(98%+, 빠른 속도)이면 FastFlow(98.5%, 20-50ms)를 선택한다. 학습 30-60분으로 빠른 개발 주기를 지원한다. Flow network 설계와 하이퍼파라미터 튜닝이 필요하다.

Few-shot(10-50장)이면 DRAEM(97.5%)을 선택한다. Simulated anomaly로 안정적 학습이 가능하다. 2-4시간 학습으로 빠르게 결과를 얻는다.

**활동**

학습 데이터 수집이 최우선이다. 정상 샘플을 계획대로 수집한다(100-500장). 다양한 조건을 커버한다(조명, 각도, 배경). 데이터 품질을 검증한다(미묘한 이상 제거).

모델 학습 및 검증을 수행한다. 선택한 모델을 학습한다(1-5시간). 검증 데이터로 성능을 측정한다. 목표 정확도 달성 여부를 확인한다.

하이퍼파라미터 튜닝을 진행한다. 주요 하이퍼파라미터를 조정한다(learning rate, backbone, coreset 크기). Grid search나 random search를 사용한다. 성능 향상 여지를 탐색한다(보통 1-2%p 개선 가능).

벤치마크를 수행한다. MVTec AD나 자체 테스트셋으로 평가한다. 정확도, 속도, 메모리를 측정한다. 목표와 비교하여 gap을 파악한다.

**의사결정 체크리스트**

목표 정확도를 달성했는가. 설정한 목표(95%/98%/99%)를 만족하는가. 만족하면 Phase 3로 진행한다. 부족하면 더 많은 데이터 수집이나 다른 모델을 시도한다.

속도 요구사항을 충족하는가. 추론 시간이 생산 라인 속도를 따라가는가. 초당 처리량이 필요량을 만족하는가. 부족하면 Phase 3에서 최적화를 집중한다.

False positive/negative 비율이 허용 가능한가. 정상을 불량으로 오판하는 비율이 목표 이하인가. 불량을 정상으로 놓치는 비율이 안전한가. 불균형이 있으면 임계값 조정이나 앙상블을 고려한다.

실제 환경 테스트가 필요한가. 벤치마크 성능이 실제 라인에서도 유지되는지 파일럿 테스트를 계획한다. 조명 변화, 진동, 먼지 등 실제 조건을 시뮬레이션한다.

### 7.3 Phase 3: Deployment Preparation (2-3 weeks)

배포 준비 단계의 목표는 실시간 처리와 최종 최적화이다. 실험실 성능을 생산 환경에서 재현할 수 있도록 모든 요소를 최적화한다.

**속도 최적화 (필요 시)**

실시간 필요(<10ms)이면 EfficientAD로 전환한다. Phase 2의 정확도를 확인했으므로, 속도가 병목이라면 EfficientAD를 재학습한다. 97.8% 정확도가 허용되는지 최종 확인한다. 1-5ms로 실시간 처리를 보장한다.

준실시간(20-50ms)이면 FastFlow를 유지하거나 PatchCore를 최적화한다. FastFlow는 이미 빠르므로 추가 최적화만 진행한다. PatchCore는 coreset 크기를 줄이거나 k를 조정하여 속도를 높인다.

**모델 최적화 기법**

모델 양자화(INT8/FP16)를 적용한다. PyTorch의 quantization API를 사용한다. 메모리를 2-4배 줄이고 속도를 1.5-2배 향상시킨다. 정확도 손실은 0.1-1%p로 수용 가능하다. 특히 EfficientAD나 PatchCore에서 효과적이다.

ONNX export를 수행한다. PyTorch 모델을 ONNX로 변환한다. ONNX Runtime이 PyTorch보다 1.5-2배 빠르다. 크로스 플랫폼 배포가 용이해진다(Windows, Linux, ARM).

TensorRT 최적화(NVIDIA GPU)를 활용한다. ONNX를 TensorRT로 변환한다. 2-5배 추가 속도 향상이 가능하다. 특히 배치 처리에서 효과적이다.

**추론 파이프라인 구축**

전처리 파이프라인을 최적화한다. 이미지 로딩, 리사이징, 정규화를 효율화한다. OpenCV나 Pillow-SIMD를 사용한다. GPU로 전처리를 병렬화한다(DALI, TorchVision).

배치 처리를 구현한다. 여러 이미지를 한 번에 처리하여 throughput을 높인다. GPU 활용률을 최대화한다. 배치 크기는 GPU 메모리에 따라 8-64로 설정한다.

비동기 처리를 도입한다. 이미지 로딩과 추론을 병렬화한다. Threading이나 multiprocessing을 사용한다. 전체 처리 시간을 30-50% 단축할 수 있다.

**임계값 설정**

ROC curve를 분석하여 최적 임계값을 찾는다. False positive와 false negative의 비용을 고려한다. 예를 들어 false positive 비용이 10배 높다면, 더 보수적인 임계값을 설정한다.

다양한 조건에서 테스트한다. 조명 변화, 제품 변형, 경계 사례에서 임계값의 안정성을 확인한다. 필요시 adaptive threshold를 구현한다.

**활동 체크리스트**

모델 최적화 완료 여부를 확인한다. 양자화, ONNX, TensorRT 중 적용 가능한 것을 모두 적용했는가. 목표 속도를 달성했는가.

추론 파이프라인 검증을 한다. 전체 파이프라인(로딩→전처리→추론→후처리)이 목표 시간 내에 완료되는가. 병목 구간이 있는가(프로파일링 도구 사용).

임계값 설정 완료 여부를 확인한다. 최적 임계값을 결정했는가. False positive/negative 비율이 허용 범위 내인가.

배포 환경 준비가 되었는가. 서버나 엣지 디바이스가 준비되었는가. 필요한 소프트웨어(CUDA, cuDNN, ONNX Runtime)가 설치되었는가.

파일럿 테스트 계획이 수립되었는가. 실제 라인에서 며칠간 테스트할 계획이 있는가. 모니터링과 로깅 시스템이 준비되었는가.

### 7.4 Phase 4: Operations (Continuous)

운영 단계는 배포 후 지속적으로 이루어진다. 안정적 운영과 지속적 개선이 목표이다.

**성능 모니터링**

실시간 지표를 추적한다. 추론 시간(평균, 최대, 99 percentile)을 모니터링한다. 목표 시간 초과 시 알람을 발생시킨다. 정확도 지표(일일 AUROC, F1 score)를 계산한다. 급격한 하락 시 조사한다. False positive/negative 비율을 추적한다. 비용 임팩트를 계산한다. 시스템 리소스(GPU 사용률, 메모리, CPU)를 모니터링한다. 과부하나 메모리 누수를 조기 감지한다.

분석 도구로 Grafana, Prometheus, TensorBoard를 활용한다. 실시간 대시보드를 구축하여 한눈에 상태를 파악한다. 알람 시스템을 설정하여 이상 징후를 즉시 통보한다.

**주기적 재학습**

재학습 트리거를 설정한다. 성능 저하(1-2%p 하락 시)가 감지되면 재학습한다. 새로운 결함 유형이 발견되면 데이터에 추가하고 재학습한다. 생산 공정 변경(설비 교체, 재료 변경) 시 재학습한다. 계절적 변화(조명, 온도)가 있으면 분기별 재학습한다. 정기 재학습(월 1회 또는 분기 1회)을 스케줄링한다.

재학습 프로세스를 자동화한다. 새 데이터 수집 파이프라인을 구축한다. 자동 학습 스크립트를 작성한다(cron job). 검증 및 배포 자동화를 구현한다(CI/CD).

**False Positive/Negative 분석**

오탐 샘플을 수집하고 분석한다. 정상을 불량으로 오판한 케이스를 모은다. 공통 패턴을 찾는다(특정 조명, 특정 제품 변형). 학습 데이터에 추가하거나 임계값을 조정한다.

미탐 샘플을 수집하고 분석한다. 불량을 정상으로 놓친 케이스를 모은다. 새로운 결함 유형인지 확인한다. 학습 데이터에 simulated anomaly로 추가하거나 모델을 재학습한다.

**A/B 테스트**

새 모델 배포 전 A/B 테스트를 수행한다. 일부 라인에서만 새 모델을 사용한다. 성능을 기존 모델과 비교한다. 우수하면 전체 배포, 열등하면 롤백한다.

점진적 롤아웃을 실시한다. 10% → 50% → 100%로 단계적으로 확대한다. 각 단계에서 문제가 없는지 확인한다.

**최신 기술 추적**

Foundation model 발전을 모니터링한다. Dinomaly, UniNet 등 새로운 모델의 성능을 주시한다. 기존 모델보다 우수하면 전환을 고려한다. 특히 multi-class 환경에서 Dinomaly로의 전환을 평가한다(메모리 93% 절감).

새로운 패러다임을 탐색한다. 학회(CVPR, ICCV)와 arXiv를 추적한다. 혁신적인 접근법이 나오면 프로토타입을 테스트한다.

하드웨어 발전을 활용한다. 새로운 GPU(예: RTX 5000 시리즈)가 출시되면 성능 향상을 평가한다. NPU나 TPU 같은 전용 가속기를 고려한다.

**운영 체크리스트 (월간)**

성능이 목표를 유지하는가. 정확도, 속도, false positive/negative 비율이 모두 목표 범위 내인가. 이탈 시 원인을 조사하고 조치한다.

시스템이 안정적으로 작동하는가. Uptime이 99% 이상인가. 크래시나 메모리 누수가 없는가. 로그에 이상 징후가 없는가.

비용이 예산 범위 내인가. 운영 비용(GPU 시간, API 호출)이 계획과 일치하는가. ROI가 긍정적으로 유지되는가.

개선 기회가 있는가. 최신 모델로 전환하면 성능이나 비용을 개선할 수 있는가. 프로세스 자동화로 인건비를 줄일 수 있는가.

---

## 8. Cost-Benefit Analysis

### 8.1 Initial Development Costs

이상 탐지 프로젝트의 초기 개발 비용은 하드웨어, 인력, 데이터 수집으로 구성된다.

**하드웨어 비용**

GPU 서버는 다음과 같다. RTX 3090(24GB)은 $1,500-2,000으로 대부분의 모델에 적합하다. RTX 4090(24GB)은 $2,000-2,500으로 최신 성능을 제공한다. A100(40GB)은 $10,000-15,000으로 대규모 배포나 연구용이다. 워크스테이션(CPU, RAM, 스토리지)은 $1,000-2,000이다.

엣지 디바이스는 다음과 같다. Jetson Xavier NX(8GB)는 $400-500으로 엣지 AI에 적합하다. Jetson AGX Orin(32GB)은 $1,000-2,000으로 고성능 엣지용이다. Raspberry Pi 4(8GB)는 $75-100으로 초저가 배포용이다.

클라우드 대안은 초기 비용이 없지만 지속적인 사용료가 발생한다. 개발 기간(2-3개월) GPU 인스턴스 사용료는 $500-2,000이다.

**인력 비용**

ML 엔지니어 급여를 월 $10,000로 가정한다. 프로토타이핑(1-2주)은 0.25-0.5개월 = $2,500-5,000이다. 최적화(2-4주)는 0.5-1개월 = $5,000-10,000이다. 배포 준비(2-3주)는 0.5-0.75개월 = $5,000-7,500이다. 총 개발 기간은 5-9주 = 1.25-2.25개월 = $12,500-22,500이다.

복잡도에 따라 인력 투입이 달라진다. DFM이나 WinCLIP은 간단하여 0.5-1개월 = $5,000-10,000이다. PatchCore나 FastFlow는 중간으로 1-1.5개월 = $10,000-15,000이다. Reverse Distillation이나 Dinomaly는 복잡하여 1.5-2.5개월 = $15,000-25,000이다.

**데이터 수집 비용**

정상 샘플 수집은 내부 수집 시 인건비만 발생한다. 100-500장 수집에 1-2주 = $2,500-5,000이다. 외주 시 이미지당 $5-20으로 100-500장 = $500-10,000이다.

이상 샘플 수집(선택사항)은 실제 불량품을 기다려야 하므로 시간이 오래 걸린다. 인위적 생성(DRAEM)이 더 효율적이다.

**소프트웨어 및 기타 비용**

Anomalib은 오픈소스(무료)이다. PyTorch, ONNX Runtime도 오픈소스(무료)이다. 상용 툴(TensorRT, Intel OpenVINO)은 무료 또는 $1,000-5,000이다. API 사용(VLM-AD, WinCLIP)은 개발 기간 중 $100-500이다. 교육 및 컨설팅은 필요시 $5,000-20,000이다.

**총 초기 비용**

최소 구성(DFM + CPU)은 하드웨어 $1,000 + 인력 $5,000 + 데이터 $1,000 = **$7,000**이다. 일반 구성(PatchCore + GPU)은 하드웨어 $3,000 + 인력 $15,000 + 데이터 $3,000 = **$21,000**이다. 고급 구성(Dinomaly + Multi-class)은 하드웨어 $5,000 + 인력 $25,000 + 데이터 $5,000 = **$35,000**이다.

### 8.2 Operational Costs (Monthly)

운영 비용은 매월 지속적으로 발생한다. 하드웨어 유지, API 사용, 인력 등이 포함된다.

**하드웨어 유지 비용**

On-premise GPU 서버는 전력 소비(GPU + 워크스테이션)가 500-1000W = $50-150/월(전기료 $0.10/kWh 가정)이다. 냉각 및 시설이 $20-50/월이다. 유지보수 및 감가상각이 $50-100/월이다. 총 $120-300/월이다.

클라우드 GPU는 p3.2xlarge(V100, $3/시간) 24시간 운영 시 $2,160/월이다. G4 instance(T4, $1.5/시간) 24시간 운영 시 $1,080/월이다. Spot instance 활용 시 50-70% 절감 가능하다.

엣지 디바이스는 전력 소비가 10-30W = $1-5/월로 매우 저렴하다.

**API 비용**

VLM-AD(GPT-4V)는 $0.01-0.05/이미지이다. 일일 100장 × 30일 = 3,000장 = $30-150/월이다. 일일 1,000장 × 30일 = 30,000장 = $300-1,500/월이다. 대량 처리(일일 10,000장)는 300,000장 = $3,000-15,000/월로 부담스럽다.

WinCLIP(self-hosted)은 무료이지만 GPU 서버 비용이 발생한다.

**인력 비용**

일상 모니터링은 주 4시간 = 월 16시간 = $1,000/월(시간당 $62.5 가정)이다. 주간 리뷰 및 분석은 주 2시간 = 월 8시간 = $500/월이다. 월간 재학습은 4-8시간 = $250-500/월이다. 분기별 모델 업그레이드는 평균 월 2-4시간 = $125-250/월이다. 총 인력 비용은 $1,875-2,250/월이다.

자동화 수준에 따라 인력 비용이 달라진다. 완전 자동화(모니터링 대시보드, 자동 재학습)는 $500-1,000/월이다. 부분 자동화(수동 재학습)는 $1,500-2,000/월이다. 수동 운영(모든 작업 수동)은 $2,500-3,500/월이다.

**총 운영 비용 (월간)**

최소 구성(CPU, 자동화)은 하드웨어 $0-100 + API $0 + 인력 $500 = **$500-600/월**이다. 일반 구성(GPU 서버, 부분 자동화)은 하드웨어 $120-300 + API $0 + 인력 $1,500 = **$1,620-1,800/월**이다. 고급 구성(GPU 서버 + VLM-AD)은 하드웨어 $300 + API $300-1,500 + 인력 $1,000 = **$1,600-2,800/월**이다. 클라우드 구성(GPU instance)은 하드웨어 $1,080-2,160 + 인력 $1,000 = **$2,080-3,160/월**이다.

### 8.3 ROI Analysis

이상 탐지 시스템의 투자 대비 효과(ROI)를 정량적으로 분석한다.

**비용 절감 효과**

불량품 검출 가치를 계산한다. 생산량은 일일 10,000개이다. 불량률은 1% = 100개 불량/일이다. 검출률은 95% = 95개 검출/일이다. 불량품 비용은 $10/개이다. 일일 절감액은 95개 × $10 = $950/일이다. 월간 절감액은 $950 × 22일 = **$20,900/월**이다.

인력 절감을 계산한다. 육안 검사자는 2명이다. 인건비는 $5,000/월 × 2 = $10,000/월이다. AI 시스템으로 1명으로 감축(모니터링만)한다. 절감액은 **$5,000/월**이다.

생산성 향상을 계산한다. 검사 속도가 2배 향상되면 생산 throughput이 증가한다. 추가 매출은 월 **$5,000-10,000**이다.

품질 개선에 따른 고객 만족도 향상과 반품 감소는 정량화 어렵지만 월 **$2,000-5,000** 추정한다.

**총 월간 효과**

불량품 검출 $20,900 + 인력 절감 $5,000 + 생산성 $7,500 + 품질 개선 $3,500 = **$36,900/월**이다.

**ROI 계산**

초기 투자는 일반 구성 기준 $21,000이다. 월간 운영 비용은 $1,700이다. 월간 효과는 $36,900이다. 순 월간 이익은 $36,900 - $1,700 = $35,200이다.

회수 기간(Payback Period)은 $21,000 / $35,200 = **0.6개월 = 18일**이다.

연간 ROI는 초기 투자 $21,000 + 연간 운영 $20,400 = 총 투자 $41,400이다. 연간 효과는 $36,900 × 12 = $442,800이다. 순 이익은 $442,800 - $41,400 = $401,400이다. ROI는 ($401,400 / $41,400) × 100 = **970%**이다.

**시나리오별 ROI**

소규모(일일 1,000개)는 불량 검출 효과가 $2,000/월로 감소한다. 총 효과는 $10,000/월로 낮아진다. 운영 비용은 $500/월(최소 구성)이다. 순 이익은 $9,500/월이다. 회수 기간은 초기 투자 $7,000 / $9,500 = **0.7개월 = 21일**이다.

대규모(일일 100,000개)는 불량 검출 효과가 $200,000/월로 증가한다. 총 효과는 $250,000/월로 높아진다. 운영 비용은 $5,000/월(고급 구성, 규모 경제)이다. 순 이익은 $245,000/월이다. 회수 기간은 초기 투자 $35,000 / $245,000 = **0.14개월 = 4일**이다.

**민감도 분석**

불량률이 0.5%로 낮아지면 효과가 절반으로 감소하지만 여전히 높은 ROI를 유지한다(회수 기간 1.2개월). 불량률이 2%로 높아지면 효과가 2배로 증가하여 회수 기간은 9일로 단축된다.

불량품 비용이 $5/개로 낮아지면 회수 기간은 1.1개월로 늘어난다. 불량품 비용이 $50/개로 높아지면(반도체, 의료) 회수 기간은 3일로 극적으로 단축된다.

### 8.4 Long-term Benefits

장기적 효과는 정량화하기 어렵지만 매우 중요하다.

**지속적인 품질 개선**

데이터 축적으로 모델이 지속적으로 향상된다. 매 분기 재학습 시 정확도가 0.5-1%p 향상된다. 1년 후 95% → 97%, 2년 후 97% → 98%로 개선된다. 이는 추가 비용 절감으로 이어진다.

**근본 원인 분석**

결함 패턴 데이터가 축적되어 공정 개선에 활용된다. 어느 스테이션에서 결함이 많이 발생하는지 파악한다. 원인을 제거하여 불량률 자체를 감소시킨다(1% → 0.5%). 이는 ROI를 배가시킨다.

**확장성**

신제품 추가 비용이 매우 낮다. 전통적 방법은 신제품마다 전체 개발 과정을 반복한다($21,000 × N). Dinomaly는 데이터만 추가하고 재학습한다($3,000 × N). N개 제품 시 $(21,000 - 3,000) × N = $18,000 × N 절감이다. 5개 제품이면 $90,000 절감이다.

**경쟁 우위**

자동화된 품질 관리로 불량률이 낮아진다. 고객 만족도가 향상되고 브랜드 가치가 상승한다. 더 빠른 time-to-market으로 시장 선점이 가능하다. 이는 장기적 매출 증대로 이어진다.

**조직 역량**

AI/ML 역량이 조직에 축적된다. 이상 탐지 외에도 다른 분야(수요 예측, 설비 예지보수)로 확장 가능하다. 데이터 기반 의사결정 문화가 정착된다.

**연간 총 효과 (3년 전망)**

1년차는 순 이익 $400,000이다(ROI 970%). 2년차는 품질 개선으로 순 이익 $500,000이다. 3년차는 확장 및 최적화로 순 이익 $600,000이다. 3년 누적 순 이익은 **$1,500,000**이다.

초기 투자 $21,000 대비 3년 누적 효과는 **7,000% ROI**이다. 이는 이상 탐지 시스템 도입이 단순한 비용 절감이 아니라 전략적 투자임을 보여준다.

---

## 9. Decision Framework

### 9.1 Decision Tree

실무에서 빠른 의사결정을 위한 단계별 decision tree이다. 각 질문에 답하면서 진행하면 5-10분 내에 적합한 모델을 선택할 수 있다.

**Step 1: Multi-class 여부 확인**

질문은 "여러 제품(클래스)을 동시에 검사하는가?"이다.

YES인 경우, 현재 또는 향후 3개 이상의 제품을 검사한다면 Dinomaly(98.8% multi)를 선택한다. 메모리 80-90% 절감, 관리 간소화, 확장 용이의 효과가 있다. 추론 80-120ms, 메모리 300-500MB(전체), 학습 3-5시간이다. 결정 완료하고 구현으로 진행한다.

NO인 경우, 단일 제품만 검사하거나 제품이 완전히 독립적이라면 Step 2로 진행한다.

**Step 2: 속도 요구사항 확인**

질문은 "실시간 처리(10ms 이하)가 필수인가?"이다.

YES인 경우, 고속 라인(초당 100개 이상)이나 엣지/CPU 환경이라면 EfficientAD(97.8%, 1-5ms)를 선택한다. GPU에서 1-5ms, CPU에서 10-20ms, 메모리 <200MB이다. 정확도 97.8%가 요구사항을 충족하는지 확인한다. 결정 완료하고 구현으로 진행한다.

NO인 경우, 20ms 이상 허용 가능하다면 Step 3로 진행한다.

**Step 3: 학습 데이터 가용성 확인**

질문은 "학습 데이터가 있는가?"이다.

NO - 데이터 전혀 없음인 경우, 신제품 즉시 배포나 프로토타입 단계라면 WinCLIP(91-95%, zero-shot)을 선택한다. 무료, 학습 0분, 즉시 사용 가능하다. 프롬프트 엔지니어링으로 최적화한다. 데이터 수집 후 다른 모델로 전환 계획을 세운다. 설명이 필요하면 VLM-AD(96-97%, API $0.01-0.05/img)를 고려한다. 결정 완료하고 구현으로 진행한다.

YES - 데이터 있음인 경우, Step 4로 진행한다.

**Step 4: 데이터 양 확인**

질문은 "정상 샘플이 몇 장인가?"이다.

10-50장(Few-shot)인 경우, 신제품 초기나 희귀 결함이라면 DRAEM(97.5%)을 선택한다. Simulated anomaly로 안정적 학습이 가능하다. 학습 2-4시간, 메모리 300-500MB이다. 데이터가 100장 이상 축적되면 PatchCore로 전환을 고려한다. 결정 완료하고 구현으로 진행한다.

100장 이상(충분)인 경우, Step 5로 진행한다.

**Step 5: 정확도 목표 확인**

질문은 "99% 이상의 정확도가 필수인가?"이다.

YES인 경우, 반도체, 의료, 항공 등 불량 유출 비용이 매우 높다면 PatchCore(99.1%)를 선택한다. 추론 50-100ms, 메모리 100-500MB, 학습 1-2시간이다. Pixel-level이 특히 중요하면 Reverse Distillation(98.6%, Pixel 98.5%)을 고려한다. 결정 완료하고 구현으로 진행한다.

NO인 경우, 98% 정도면 충분하다면 Step 6으로 진행한다.

**Step 6: 특수 요구사항 확인**

질문은 "특수한 요구사항이 있는가?"이다.

품질 보고서 자동화인 경우, VLM-AD(96-97%)를 선택한다. 자연어 설명, 근본 원인 분석, API $0.01-0.05/img이다. 샘플링 검사나 중요 샘플 분석에 적합하다.

복잡한 텍스처(직물, 카펫)인 경우, DSR(96.5-98.0%)을 선택한다. Dual subspace로 구조와 텍스처 분리 모델링한다. VQ-VAE 구현이 복잡하므로 개발 리소스를 확인한다.

빠른 프로토타입(15분)인 경우, DFM(94.5-95.5%)을 선택한다. PCA + Mahalanobis distance로 극도로 간단하다. Feasibility 검증 후 다른 모델로 전환한다.

Domain shift가 큰 경우(조명, 카메라 변경), CFA(96.5-97.5%)를 선택한다. Hypersphere embedding으로 scale invariance를 제공한다.

특수 요구사항 없음인 경우, Step 7로 진행한다.

**Step 7: 기본 선택 (균형)**

특별한 제약이 없는 일반적인 경우이다.

FastFlow(98.5%, 20-50ms)를 선택한다. 속도와 정확도의 최고 균형이다. 확률적 해석 가능, 널리 검증됨, 메모리 500MB-1GB이다. 대부분의 실무 환경에 적합한 선택이다.

결정 완료하고 구현으로 진행한다.

**Decision Tree 요약 플로우**

```
시작
 ↓
Multi-class? → YES → Dinomaly (98.8%)
 ↓ NO
실시간(<10ms)? → YES → EfficientAD (97.8%, 1-5ms)
 ↓ NO
데이터 있음? → NO → WinCLIP (91-95%, zero-shot)
 ↓ YES
데이터 양? → 10-50장 → DRAEM (97.5%, few-shot)
 ↓ 100+장
정확도 99%+? → YES → PatchCore (99.1%)
 ↓ NO
특수 요구사항?
 ├─ 보고서 → VLM-AD (96-97%)
 ├─ 텍스처 → DSR (96.5-98.0%)
 ├─ 프로토타입 → DFM (94.5-95.5%)
 └─ 없음 → FastFlow (98.5%) ← 기본 선택
```

### 9.2 Checklist-based Selection

Decision tree가 선형적 접근이라면, checklist는 모든 요소를 종합적으로 고려한다. 각 항목을 체크하면서 점수를 매기고, 최종적으로 가장 높은 점수의 모델을 선택한다.

**요구사항 체크리스트**

정확도 요구사항을 확인한다. 99% 이상 필수는 +3점(PatchCore, Dinomaly single)이다. 98% 이상 충분은 +2점(FastFlow, Reverse Distillation, Dinomaly multi)이다. 95% 이상 충분은 +1점(DRAEM, STFPM, CFA)이다. 90% 이상 허용은 0점(WinCLIP, VLM-AD)이다.

속도 요구사항을 확인한다. 10ms 이하 필수는 +3점(EfficientAD만 가능)이다. 50ms 이하 필요는 +2점(FastFlow, STFPM, DFM)이다. 100ms 이하 허용은 +1점(PatchCore, Dinomaly, DRAEM)이다. 100ms 이상 허용은 0점(Reverse Distillation, VLM-AD)이다.

메모리 제약을 확인한다. 200MB 이하 필수는 +3점(EfficientAD만 가능)이다. 500MB 이하 필요는 +2점(PatchCore, DRAEM)이다. 1GB 이하 허용은 +1점(FastFlow, Reverse Distillation)이다. 1GB 이상 허용은 0점(Dinomaly, PaDiM)이다.

클래스 수를 확인한다. 3개 이상 multi-class는 +3점(Dinomaly만 효율적)이다. 2개 클래스는 +1점(Dinomaly 고려)이다. 1개 클래스는 0점(모든 모델 가능)이다.

학습 데이터 가용성을 확인한다. 0장(zero-shot)은 +3점(WinCLIP, VLM-AD만 가능)이다. 10-50장(few-shot)은 +2점(DRAEM만 효율적)이다. 100-500장(충분)은 0점(모든 모델 가능)이다.

**특수 요구사항 체크리스트**

Pixel-level localization 중요도를 확인한다. 매우 중요는 +2점(Reverse Distillation, PatchCore 우선)이다. 중요는 +1점(FastFlow, Dinomaly)이다. 중요하지 않음은 0점이다.

설명 가능성(Explainability) 필요도를 확인한다. 필수는 +3점(VLM-AD만 가능)이다. 선호는 +1점(PatchCore, Memory-based)이다. 불필요는 0점이다.

CPU 환경 필요도를 확인한다. CPU만 사용은 +3점(EfficientAD만 실용적)이다. CPU 선호는 +1점(EfficientAD, DFM)이다. GPU 사용 가능은 0점이다.

복잡한 텍스처 여부를 확인한다. 매우 복잡(직물, 카펫)은 +2점(DSR, FastFlow)이다. 중간은 +1점(FastFlow, Dinomaly)이다. 단순은 0점이다.

즉시 배포 필요도를 확인한다. 즉시(0일)는 +3점(WinCLIP, VLM-AD)이다. 1주 이내는 +1점(DFM, DRAEM)이다. 시간 여유는 0점이다.

**점수 계산 예시**

시나리오 A: 반도체 웨이퍼 검사이다. 정확도 99%+ (+3), 속도 100ms 허용 (+1), 메모리 제약 없음 (0), Single-class (0), 데이터 500장 (0), Pixel 중요 (+2) = 총 6점이다. PatchCore가 가장 적합하다.

시나리오 B: 고속 전자부품 라인이다. 정확도 98%+ (+2), 속도 10ms 필수 (+3), 메모리 200MB 이하 (+3), Single-class (0), 데이터 300장 (0) = 총 8점이다. EfficientAD가 유일한 선택이다.

시나리오 C: 다양한 플라스틱 제품이다. 정확도 98%+ (+2), 속도 100ms 허용 (+1), 메모리 제약 없음 (0), Multi-class 10개 (+3), 데이터 각 200장 (0) = 총 6점이다. Dinomaly가 압도적이다.

시나리오 D: 신제품 프로토타입이다. 정확도 95%+ (+1), 속도 허용 (0), 메모리 제약 없음 (0), Single-class (0), 데이터 0장 (+3), 즉시 배포 (+3) = 총 7점이다. WinCLIP이 최적이다.

**체크리스트 사용 팁**

모든 항목을 정직하게 체크한다. 희망 사항이 아닌 실제 요구사항을 기준으로 한다. 동점이 나오면 장기적 관점을 우선한다. Multi-class 가능성, 확장성, 최신 기술 등을 고려한다. 점수가 낮은 모델은 요구사항을 충족하지 못하므로 제외한다. 3점 이상 차이가 나면 명확한 선택이다.

### 9.3 Multi-criteria Decision Matrix

복잡한 의사결정 상황에서는 여러 기준을 동시에 고려하는 decision matrix가 유용하다. 각 기준에 가중치를 부여하고, 모델별로 점수를 매긴 후, 가중 합계로 최종 결정한다.

**기준 설정 및 가중치**

정확도(Image AUROC)는 가중치 30%이다. 불량 유출 비용이 높을수록 가중치를 높인다(40-50%). 99% = 10점, 98% = 8점, 97% = 6점, 95% = 4점, 90% = 2점으로 점수를 매긴다.

추론 속도는 가중치 25%이다. 고속 라인일수록 가중치를 높인다(30-40%). 1-10ms = 10점, 10-30ms = 8점, 30-60ms = 6점, 60-120ms = 4점, 120ms+ = 2점으로 점수를 매긴다.

메모리 효율은 가중치 15%이다. 엣지 배포나 multi-class 환경에서 가중치를 높인다(20-25%). <200MB = 10점, 200-500MB = 8점, 500MB-1GB = 6점, 1-2GB = 4점, 2GB+ = 2점으로 점수를 매긴다.

개발 난이도는 가중치 10%이다. 빠른 time-to-market이 중요하면 가중치를 높인다(15-20%). 매우 쉬움 = 10점, 쉬움 = 8점, 중간 = 6점, 어려움 = 4점, 매우 어려움 = 2점으로 점수를 매긴다.

운영 비용은 가중치 10%이다. 장기 운영에서 가중치를 높인다(15-20%). 매우 낮음 = 10점, 낮음 = 8점, 중간 = 6점, 높음 = 4점, 매우 높음 = 2점으로 점수를 매긴다.

특수 기능은 가중치 10%이다. Multi-class, Zero-shot, Explainable 등 특수 요구사항이 있으면 가중치를 높인다(15-20%). 완벽히 충족 = 10점, 부분 충족 = 5점, 미충족 = 0점으로 점수를 매긴다.

**Decision Matrix 예시**

일반적인 시나리오(가중치: 정확도 30%, 속도 25%, 메모리 15%, 개발 10%, 운영 10%, 특수 10%)를 보자.

PatchCore는 정확도 10점(99.1%) × 30% = 3.0이다. 속도 4점(50-100ms) × 25% = 1.0이다. 메모리 8점(100-500MB) × 15% = 1.2이다. 개발 8점(쉬움) × 10% = 0.8이다. 운영 8점(낮음) × 10% = 0.8이다. 특수 0점(single-class) × 10% = 0이다. 총점은 6.8점이다.

FastFlow는 정확도 8점(98.5%) × 30% = 2.4이다. 속도 8점(20-50ms) × 25% = 2.0이다. 메모리 6점(500MB-1GB) × 15% = 0.9이다. 개발 6점(중간) × 10% = 0.6이다. 운영 6점(중간) × 10% = 0.6이다. 특수 0점 × 10% = 0이다. 총점은 6.5점이다.

EfficientAD는 정확도 6점(97.8%) × 30% = 1.8이다. 속도 10점(1-5ms) × 25% = 2.5이다. 메모리 10점(<200MB) × 15% = 1.5이다. 개발 8점(쉬움) × 10% = 0.8이다. 운영 10점(매우 낮음) × 10% = 1.0이다. 특수 0점 × 10% = 0이다. 총점은 7.6점이다.

Dinomaly는 정확도 8점(98.8%) × 30% = 2.4이다. 속도 4점(80-120ms) × 25% = 1.0이다. 메모리 8점(300-500MB multi) × 15% = 1.2이다. 개발 4점(어려움) × 10% = 0.4이다. 운영 10점(multi-class 효율) × 10% = 1.0이다. 특수 10점(multi-class) × 10% = 1.0이다. 총점은 7.0점이다.

이 시나리오에서는 EfficientAD(7.6점)가 최고 점수이다. 속도와 메모리 효율이 높은 가중치를 받았다. PatchCore(6.8점)와 Dinomaly(7.0점)도 우수하다. 정확도를 더 중시한다면 가중치를 조정한다.

**가중치 조정 예시**

정밀 검사 시나리오(정확도 50%, 속도 10%, 나머지 균등)에서 PatchCore는 정확도 10 × 50% + 속도 4 × 10% + 메모리 8 × 10% + 개발 8 × 10% + 운영 8 × 10% + 특수 0 × 10% = 8.2점이다. Dinomaly는 정확도 8 × 50% + 속도 4 × 10% + 메모리 8 × 10% + 개발 4 × 10% + 운영 10 × 10% + 특수 10 × 10% = 7.8점이다. EfficientAD는 정확도 6 × 50% + 속도 10 × 10% + ... = 7.1점이다. PatchCore가 최고 점수로 역전된다.

Multi-class 시나리오(특수 30%, 정확도 25%, 속도 15%, 나머지 균등)에서 Dinomaly는 특수 10 × 30% + 정확도 8 × 25% + 속도 4 × 15% + ... = 8.2점으로 압도적이다. PatchCore는 특수 0 × 30%로 큰 감점을 받아 5.8점이다.

**Decision Matrix 사용 팁**

가중치를 신중하게 설정한다. 조직의 우선순위, 비즈니스 목표, 기술적 제약을 모두 반영한다. 여러 시나리오로 테스트한다. 가중치를 조정하면서 결과가 어떻게 달라지는지 확인한다(민감도 분석). 1-2점 차이는 유의미하지 않다. 같은 tier로 간주하고 다른 요소(미래 확장성, 팀 역량)를 고려한다. 정량화하기 어려운 요소도 고려한다. 팀의 ML 경험, 장기 유지보수, 벤더 지원 등을 정성적으로 평가한다.

---

## 10. Industry Applications

### 10.1 Semiconductor

반도체 산업은 이상 탐지의 가장 까다로운 적용 분야이다. 웨이퍼 검사에서 미세 결함(수 마이크론)을 놓치면 전체 웨이퍼가 폐기된다. 불량 유출 비용이 매우 높아 99.5% 이상의 정확도가 요구된다.

**추천 모델**

PatchCore(99.1%)를 1순위로 추천한다. 최고 정확도로 미세 결함을 탐지한다. Pixel AUROC 98.2%로 정확한 위치 파악이 가능하다. 추론 50-100ms는 웨이퍼 검사 속도에 적합하다.

Reverse Distillation(98.6%, Pixel 98.5%)을 2순위로 추천한다. Pixel-level이 특히 우수하여 결함 위치를 정밀하게 파악한다. 근본 원인 분석에 유용하다. 100-200ms는 정밀 검사에서 허용된다.

Dinomaly(99.2% single)도 고려 가능하다. 다양한 칩 종류를 검사한다면 multi-class 효율성이 크다. 최신 foundation model로 지속적인 성능 향상이 예상된다.

**적용 사례**

웨이퍼 표면 검사에서 스크래치, 파티클, 패턴 결함을 탐지한다. PatchCore로 99.1% 정확도를 달성한다. 미세 결함(<5μm)도 검출 가능하다.

Die-level 검사에서 각 die의 품질을 검증한다. Reverse Distillation으로 결함 위치를 정밀하게 파악한다. 불량 die를 laser marking하여 패키징에서 제외한다.

다양한 칩 종류(메모리, 로직, 센서)를 Dinomaly로 단일 모델로 검사한다. 신규 칩 추가 시 데이터만 추가하고 재학습한다.

**도전 과제 및 해결**

극도로 높은 정확도 요구(99.5%+)는 PatchCore(99.1%)로도 부족할 수 있다. 앙상블(PatchCore + Reverse Distillation)로 0.5-1%p 추가 향상이 가능하다. 2단계 검사(1차: EfficientAD 고속 스크리닝, 2차: PatchCore 정밀 검사)를 구현한다.

미세 결함 탐지(<5μm)는 고해상도 이미지(4K, 8K)가 필요하다. 메모리와 계산량이 급증한다. Patch-based 처리와 multi-scale feature를 활용한다.

클린룸 환경의 데이터 수집 제약은 접근 제한과 고비용 샘플이 문제이다. Few-shot(DRAEM)으로 소량 데이터 학습이 가능하다. Simulated anomaly를 활용한다.

### 10.2 Medical Devices

의료 기기는 환자 안전에 직결되어 엄격한 품질 기준과 규제 요구사항이 있다. FDA 등 규제 기관의 승인이 필요하고, 설명 가능성이 필수이다.

**추천 모델**

PatchCore(99.1%)를 정확도 우선으로 추천한다. 환자 안전이 최우선이므로 최고 정확도가 필요하다. 의료 기기 불량은 법적 책임으로 이어진다.

VLM-AD(96-97%)를 설명 가능성 우선으로 추천한다. 자연어로 결함을 설명하여 FDA 감사에 대응한다. 품질 문서 자동 생성으로 compliance를 지원한다. 전수 검사보다는 샘플링 검사에 적합하다.

Reverse Distillation(98.6%, Pixel 98.5%)을 localization 우선으로 추천한다. 의료 영상 분석(X-ray, CT)에서 병변 위치를 정밀하게 파악한다.

**적용 사례**

주사기/약병 검사에서 크랙, 오염, 이물질을 탐지한다. PatchCore로 99.1% 정확도를 달성한다. 불량품 유출 시 환자 위험이 크다.

의료 영상 품질 검사에서 X-ray, MRI 이미지의 artifact를 탐지한다. Reverse Distillation으로 artifact 위치를 정밀하게 파악한다. 재촬영 여부를 결정한다.

수술 도구 검사에서 scalpel, forceps의 결함을 탐지한다. VLM-AD로 결함 유형과 위험도를 자연어로 설명한다. 감사 추적 문서를 자동 생성한다.

**도전 과제 및 해결**

규제 요구사항(FDA 21 CFR Part 11)은 AI 의사결정의 투명성과 추적 가능성이 필요하다. VLM-AD로 자연어 설명을 제공한다. 모든 판정에 대한 상세 로그를 유지한다. 정기적인 성능 검증과 재학습 이력을 문서화한다.

설명 가능성은 "왜 불량으로 판정했는가"를 명확히 해야 한다. VLM-AD의 자연어 설명을 활용한다. PatchCore의 nearest neighbor를 시각화한다. Attention map이나 Grad-CAM으로 중요 영역을 표시한다.

Zero-defect 요구는 99.9%+ 정확도가 필요하다. 다층 검사(자동 + 육안)를 구현한다. 앙상블 모델로 정확도를 극대화한다. 통계적 공정 관리와 통합한다.

### 10.3 Automotive

자동차 산업은 안전 부품과 외관 부품의 이중 기준이 있다. 안전 부품(브레이크, 에어백)은 높은 정확도가 필요하고, 외관 부품(도장, 내장재)은 균형이 필요하다.

**추천 모델 (안전 부품)**

PatchCore(99.1%)를 선택한다. 브레이크, 에어백, 안전벨트 등에 적용한다. 단 하나의 결함도 사고로 이어질 수 있다. 추론 50-100ms는 검사 라인에 적합하다.

**추천 모델 (외관 부품)**

FastFlow(98.5%, 20-50ms)를 선택한다. 도장, 플라스틱 부품, 내장재에 적용한다. 속도와 정확도의 균형이 우수하다. 고속 라인(분당 60개)에 대응 가능하다.

EfficientAD(97.8%, 1-5ms)를 고속 라인용으로 선택한다. 소형 부품(볼트, 너트, 클립)에 적용한다. 초고속 라인(분당 200개)에 필수이다.

**추천 모델 (다품종)**

Dinomaly(98.8%)를 선택한다. 수십 종류의 부품을 단일 모델로 검사한다. 모델 변경이 빈번한 환경에 적합하다. 메모리 80-90% 절감의 효과가 크다.

**적용 사례**

브레이크 디스크 검사에서 크랙, 변형, 표면 결함을 탐지한다. PatchCore로 99.1% 정확도로 안전을 보장한다.

차체 도장 검사에서 색상 불균일, 오염, 흠집을 탐지한다. FastFlow로 98.5% 정확도와 20-50ms 속도를 달성한다. 고속 라인(분당 60대)에 대응한다.

다양한 플라스틱 부품(범퍼, 대시보드, 도어 트림) 검사는 Dinomaly로 단일 모델로 처리한다. 신규 모델 출시 시 빠른 대응이 가능하다.

**도전 과제 및 해결**

고속 라인(분당 100개 이상)은 10ms 이하 추론이 필요하다. EfficientAD(1-5ms)를 사용한다. 배치 처리로 throughput을 극대화한다.

다양한 부품 종류(수백 가지)는 모델 관리가 복잡하다. Dinomaly로 통합 모델을 구축한다. 신규 부품은 데이터만 추가한다.

3D 형상 결함(변형, 찌그러짐)은 2D 이미지로 한계가 있다. Multi-view 촬영(여러 각도)을 사용한다. 3D point cloud와 결합을 고려한다.

### 10.4 Electronics

전자 산업은 PCB, 칩, 디스플레이 등 다양한 제품군이 있다. 미세 결함과 고속 생산의 조화가 필요하다.

**추천 모델**

Dinomaly(98.8%)를 다양한 PCB 검사용으로 선택한다. 수십 종류의 PCB를 단일 모델로 처리한다. Multi-class 효율성이 결정적이다.

PatchCore(99.1%)를 고가 부품용으로 선택한다. CPU, GPU, 메모리 칩 등에 적용한다. 불량 유출 비용이 높아 최고 정확도가 필요하다.

EfficientAD(97.8%, 1-5ms)를 소형 부품용으로 선택한다. 저항, 커패시터, 커넥터 등에 적용한다. 초고속 라인(분당 1000개 이상)에 필수이다.

**적용 사례**

PCB 표면 검사에서 솔더 불량, 브리지, 미세 크랙을 탐지한다. Dinomaly로 다양한 PCB 종류를 단일 모델로 처리한다. 98.8% 정확도를 유지한다.

칩 패키징 검사에서 wire bonding 불량, die attach 결함을 탐지한다. PatchCore로 99.1% 정확도로 고가 칩을 보호한다.

SMT 라인에서 소형 부품(저항, 커패시터)의 위치, 방향, 품질을 검사한다. EfficientAD로 1-5ms 실시간 처리를 달성한다. 분당 1000개 이상 처리한다.

**도전 과제 및 해결**

미세 결함(<0.1mm)은 고해상도(10+ MP) 이미지가 필요하다. Patch-based 처리로 메모리를 절약한다. Multi-scale feature extraction을 활용한다.

다양한 PCB 레이아웃(수백 가지)은 각각 학습하기 어렵다. Dinomaly로 통합 모델을 구축한다. Transfer learning으로 신규 레이아웃에 빠르게 적응한다.

실시간 AOI(Automated Optical Inspection)는 밀리초 단위 처리가 필요하다. EfficientAD를 사용한다. FPGA나 NPU로 추가 가속을 고려한다.

### 10.5 Display Quality (OLED)

OLED 디스플레이는 화질이 제품 가치를 결정한다. Mura(얼룩), 라인 불량, 픽셀 결함 등 다양한 결함이 있다. 미세한 결함도 사용자가 인지하므로 높은 정확도가 필요하다.

**추천 모델**

Reverse Distillation(98.6%, Pixel 98.5%)을 선택한다. Pixel-level localization이 매우 우수하다. Mura의 정확한 위치와 크기를 파악한다. 100-200ms는 OLED 검사 속도에 적합하다.

FastFlow(98.5%, 20-50ms)를 대안으로 선택한다. 균형 잡힌 성능으로 일반 검사에 적합하다. Normalizing flow가 미묘한 intensity 변화를 잘 포착한다.

Dinomaly(98.8%)를 다양한 크기 패널용으로 선택한다. 스마트폰, 태블릿, TV 등을 단일 모델로 처리한다.

**적용 사례**

OLED Mura 검사에서 휘도 불균일, 색상 얼룩을 탐지한다. Reverse Distillation으로 pixel-level로 정밀하게 파악한다. 98.5% pixel AUROC로 미세 mura도 검출한다.

라인 불량 검사에서 수평/수직 라인, dead pixel을 탐지한다. FastFlow로 98.5% 정확도와 빠른 속도를 달성한다.

다양한 크기 패널(5"-75")을 Dinomaly로 단일 모델로 검사한다. 해상도에 무관하게 일관된 성능을 유지한다.

**도전 과제 및 해결**

미세 Mura(<1% intensity 차이)는 사람 눈으로도 구분 어렵다. 고bit-depth 이미지(12-16bit)를 사용한다. Contrast enhancement 전처리를 적용한다. Pixel-level 정밀도(Reverse Distillation)가 필수이다.

주관적 품질 기준은 사람마다 mura 인지도가 다르다. 다수의 검사자 판정으로 ground truth를 만든다. VLM-AD로 자연어 설명("slightly visible under bright light")을 제공한다.

대형 패널(>50")은 이미지 크기가 매우 크다(100+ MP). GPU 메모리 부족이 발생한다. Sliding window나 patch-based 처리를 사용한다. 다중 GPU로 병렬 처리한다.

---

## 11. Common Pitfalls

### 11.1 Wrong Model Selection

잘못된 모델 선택은 프로젝트 전체를 위험에 빠뜨린다. 가장 흔한 실수와 해결책을 분석한다.

**Pitfall 1: 벤치마크 점수만 보고 선택**

흔한 실수는 MVTec AD AUROC 점수가 가장 높은 PatchCore(99.1%)를 무조건 선택하는 것이다. 그러나 실제 요구사항은 속도(실시간 라인), 메모리(엣지 디바이스), multi-class(여러 제품) 등이 더 중요할 수 있다.

올바른 접근은 요구사항을 먼저 명확히 하는 것이다. 정확도, 속도, 메모리, 클래스 수, 데이터 가용성을 모두 고려한다. Decision tree(9.1절)나 체크리스트(9.2절)를 사용한다. Trade-off를 이해한다. 99.1%와 98.5%의 0.6%p 차이가 실무에서 얼마나 중요한지 평가한다.

실제 사례를 보자. 고속 전자부품 라인에서 PatchCore(99.1%, 50-100ms)를 선택했다. 그러나 라인 속도(초당 200개)를 따라가지 못했다. EfficientAD(97.8%, 1-5ms)로 변경하여 실시간 처리를 달성했다. 1.3%p 정확도 희생은 실무에서 허용 가능했다.

**Pitfall 2: Single-class 사고방식**

흔한 실수는 multi-class 환경에서도 각 클래스마다 별도 모델을 구축하는 것이다. 10개 제품이면 10개 모델을 만들고, 각각 학습하고 배포한다. 메모리 폭발(5-10GB)과 관리 악몽에 빠진다.

올바른 접근은 Dinomaly(98.8%)를 우선 고려하는 것이다. 3개 이상 클래스면 단일 모델로 통합한다. 메모리 80-90% 절감, 관리 간소화, 확장 용이의 효과를 얻는다. 2025년 이후 multi-class가 표준이 될 것이다.

실제 사례를 보자. 플라스틱 사출 공장에서 15개 제품마다 PatchCore를 구축했다. 총 메모리 7.5GB, 학습 30시간, 관리 복잡도가 극심했다. Dinomaly로 전환하여 500MB, 학습 5시간, 단일 모델 관리로 개선했다.

**Pitfall 3: Zero-shot을 과소평가**

흔한 실수는 "학습 데이터가 충분할 때까지 기다린다"는 것이다. 신제품 출시가 2-3개월 지연되고, 초기 불량품이 시장에 유출된다.

올바른 접근은 WinCLIP(91-95%)으로 즉시 시작하는 것이다. Zero-shot으로 출시 당일부터 검사한다. 데이터 수집을 병행한다. 100장 확보 시 PatchCore로 전환한다. 단계적 정확도 향상(91% → 95% → 99%)을 달성한다.

실제 사례를 보자. 신제품 스마트폰 케이스 출시에서 데이터 수집(2개월)을 기다렸다. 출시 지연과 초기 불량 유출이 발생했다. 다음 제품은 WinCLIP으로 즉시 시작하여 출시일 준수와 불량 최소화를 달성했다.

**Pitfall 4: 과도한 복잡도**

흔한 실수는 최신 복잡한 모델(Reverse Distillation, DSR)을 무조건 선택하는 것이다. 구현에 3-4개월 소요되고, 디버깅이 어렵고, 유지보수가 힘들다.

올바른 접근은 간단한 모델부터 시작하는 것이다. DFM(15분) → FastFlow(1주) → PatchCore(2주) 순으로 진행한다. 각 단계에서 충분한지 평가한다. 복잡도는 필요할 때만 증가시킨다. "Simplest model that works"를 찾는다.

실제 사례를 보자. 직물 검사에서 DSR의 dual subspace에 매료되어 4개월을 투자했다. 구현이 복잡하고 하이퍼파라미터 튜닝이 어려웠다. FastFlow로 변경하여 2주 만에 98.5%를 달성했고, DSR의 96.5%보다 오히려 높았다.

### 11.2 Inadequate Data Preparation

데이터 품질은 모델 성능의 상한을 결정한다. 데이터 준비 단계의 실수는 나중에 고치기 어렵다.

**Pitfall 1: 불충분한 데이터 양**

흔한 실수는 50-100장으로 PatchCore를 학습하는 것이다. 성능이 목표에 도달하지 못하고(95% vs 목표 99%), 과적합이 발생한다.

올바른 접근은 모델별 최소 데이터 요구사항을 준수하는 것이다. PatchCore/FastFlow는 100-500장이 필요하다. DRAEM은 10-50장으로 가능하다(simulated anomaly). WinCLIP은 0장이다(zero-shot). 데이터가 부족하면 모델을 변경한다. 50장이면 DRAEM을, 0장이면 WinCLIP을 선택한다.

실제 사례를 보자. 희귀 금속 부품에서 30장으로 PatchCore를 시도했다. 93%밖에 나오지 않았다. DRAEM으로 변경하여 30장으로 97.5%를 달성했다. Simulated anomaly가 효과적이었다.

**Pitfall 2: 데이터 품질 문제**

흔한 실수는 학습 데이터에 미묘한 이상이 섞여있는 것이다. 모델이 이상을 정상으로 학습하여 성능이 저하된다. False negative가 증가한다.

올바른 접근은 엄격한 품질 관리이다. 학습 데이터를 여러 명이 검증한다. 의심스러운 샘플은 제외한다(보수적 접근). Outlier detection으로 자동 필터링한다. 정상 샘플의 분포를 시각화하여 이상치를 찾는다.

실제 사례를 보자. PCB 검사에서 학습 데이터 500장 중 5%가 미묘한 솔더 불량을 포함했다. 모델이 91%에 머물렀다. 재검증 후 불량 샘플 제거하여 97%로 향상되었다.

**Pitfall 3: 다양성 부족**

흔한 실수는 동일한 조건에서만 데이터를 수집하는 것이다. 조명, 각도, 배경이 모두 같다. 실제 환경의 변동성을 커버하지 못한다. Domain shift 발생 시 성능이 급락한다.

올바른 접근은 다양한 조건을 커버하는 것이다. 조명 변화(아침, 저녁, 흐린 날)를 포함한다. 각도 변화(정면, 측면, 45도)를 포함한다. 배경 변화를 반영한다. 시간에 따른 변화(계절, 설비 노화)를 고려한다.

실제 사례를 보자. 자동차 도장 검사에서 실내 조명 데이터로만 학습했다. 야외 자연광에서 성능이 95% → 82%로 급락했다. 다양한 조명 조건 데이터를 추가하여 93%로 회복했다.

**Pitfall 4: 불균형한 카테고리 커버리지**

흔한 실수는 일부 카테고리만 많이 수집하는 것이다(multi-class). 100장 vs 10장 vs 50장으로 불균형하다. 적은 카테고리의 성능이 낮다.

올바른 접근은 카테고리별 균등 수집이다. 각 카테고리당 50-200장을 목표로 한다. 최소한 각 카테고리당 30장 이상을 확보한다. 불가피한 불균형은 weighted loss로 보완한다.

실제 사례를 보자. 전자부품 검사(Dinomaly)에서 저항 500장, 커패시터 50장, 커넥터 30장이었다. 커패시터와 커넥터의 성능이 85%에 불과했다. 균등하게 각 150장으로 맞춰 모두 97% 이상을 달성했다.

### 11.3 Hyperparameter Mistakes

하이퍼파라미터 설정은 성능에 직접적 영향을 미친다. 잘못된 설정은 몇 %p의 성능 차이를 만든다.

**Pitfall 1: 기본값 맹신**

흔한 실수는 Anomalib의 기본 하이퍼파라미터를 그대로 사용하는 것이다. 기본값은 MVTec AD에 최적화되어 있다. 실제 데이터는 다를 수 있다.

올바른 접근은 중요 하이퍼파라미터를 식별하는 것이다. PatchCore는 coreset 비율(10-20%), backbone(Wide ResNet50 vs ResNet18)이 중요하다. FastFlow는 flow depth(8-16), coupling type이 중요하다. EfficientAD는 PDN 크기, feature dimension이 중요하다. Grid search나 random search로 최적값을 찾는다. 보통 1-3%p 성능 향상이 가능하다.

실제 사례를 보자. 의료 기기 검사에서 PatchCore 기본값(coreset 10%)으로 96%였다. Coreset 비율을 조정(15%)하여 98%로 향상되었다. 단순한 조정이 2%p 차이를 만들었다.

**Pitfall 2: 과도한 튜닝**

흔한 실수는 모든 하이퍼파라미터를 과도하게 튜닝하는 것이다. Learning rate, batch size, optimizer, augmentation 등 수십 개를 조정한다. 시간이 오래 걸리고(수주), 과적합 위험이 있다.

올바른 접근은 중요한 2-3개만 튜닝하는 것이다. 나머지는 기본값이나 권장값을 사용한다. Ablation study로 중요도를 파악한다. 투입 시간 대비 효과를 고려한다. 1-2%p 향상에 2주 투자는 비효율적이다.

실제 사례를 보자. FastFlow에서 모든 파라미터를 2개월간 튜닝했다. 96.5% → 97.8%로 1.3%p 향상되었다. 그러나 PatchCore는 기본값으로 1주 만에 99.1%를 달성했다. 튜닝 시간이 아깝다.

**Pitfall 3: 임계값 설정 실수**

흔한 실수는 고정 임계값(예: 0.5)을 사용하는 것이다. False positive와 false negative의 비용을 고려하지 않는다. 비즈니스 목표와 무관한 임계값이다.

올바른 접근은 ROC curve를 분석하는 것이다. False positive rate와 false negative rate를 시각화한다. 비용을 고려하여 최적점을 찾는다. 예를 들어 false positive 비용이 10배 높다면, false positive rate 1%에서 임계값을 설정한다. 비즈니스 목표에 맞춘다. "불량률 1% 환경에서 95% 검출"이 목표라면 이에 맞는 임계값을 찾는다.

실제 사례를 보자. 반도체 검사에서 임계값 0.5로 설정했다. False positive rate 10%로 정상 웨이퍼를 많이 불량 판정했다. ROC curve 분석 후 임계값 0.7로 조정하여 false positive rate 2%로 감소했다.

**Pitfall 4: Backbone 선택 실수**

흔한 실수는 가장 큰 backbone(Wide ResNet50)을 항상 선택하는 것이다. 메모리와 속도를 희생하지만 성능 향상은 미미하다(0.5-1%p).

올바른 접근은 backbone을 실험하는 것이다. ResNet18, ResNet50, Wide ResNet50을 비교한다. 성능 대비 속도/메모리를 평가한다. 보통 ResNet50이 최적 균형점이다. Wide ResNet50은 0.5%p 향상에 2배 느리다. 엣지 디바이스에서는 MobileNet이나 EfficientNet을 고려한다.

실제 사례를 보자. 엣지 배포에서 Wide ResNet50으로 메모리 2GB를 사용했다. 4GB 디바이스에서 다른 프로세스와 경합이 발생했다. ResNet18로 변경하여 메모리 500MB, 성능 98.5% → 98.0%(0.5%p 차이)로 허용 가능했다.

### 11.4 Deployment Issues

개발 환경에서는 잘 작동하던 모델이 배포 환경에서 실패하는 경우가 많다. 배포 단계의 함정을 분석한다.

**Pitfall 1: 환경 차이 무시**

흔한 실수는 개발(GPU 서버, PyTorch)과 배포(엣지, ONNX)가 다른 환경인 것을 간과하는 것이다. PyTorch 모델이 ONNX 변환 시 오류가 발생한다. 속도나 정확도가 예상과 다르다.

올바른 접근은 조기에 배포 환경을 테스트하는 것이다. Phase 2(최적화) 단계에서 ONNX export를 시도한다. 배포 하드웨어에서 속도와 정확도를 검증한다. 문제 발견 시 모델 구조를 조정한다. CI/CD 파이프라인에 배포 환경 테스트를 포함한다.

실제 사례를 보자. Reverse Distillation을 GPU 서버에서 개발했다. Jetson Xavier 배포 시 메모리 부족(1GB vs 필요 2GB)이 발생했다. Phase 2에서 Jetson 테스트했다면 PatchCore나 EfficientAD로 조기 변경할 수 있었다. 3개월 개발이 낭비되었다.

**Pitfall 2: 메모리 누수**

흔한 실수는 장시간 실행 시 메모리가 계속 증가하는 것이다. 몇 시간 또는 며칠 후 시스템이 크래시한다. 24시간 무인 운영이 불가능하다.

올바른 접근은 메모리 프로파일링이다. 장시간 테스트(24-48시간)를 수행한다. 메모리 사용량을 모니터링한다. 누수 발견 시 원인을 찾는다. PyTorch의 gradient 누적, 캐시 미해제, 파일 핸들 미닫힘 등이 원인이다. 명시적으로 메모리를 해제한다. `del`, `torch.cuda.empty_cache()`, `gc.collect()`를 사용한다.

실제 사례를 보자. PatchCore 배포에서 매 추론마다 특징이 GPU 메모리에 누적되었다. 8시간 후 메모리 부족으로 크래시했다. Gradient 계산을 `torch.no_grad()`로 감싸고, 사용 후 명시적 해제하여 해결했다.

**Pitfall 3: 동시성 문제**

흔한 실수는 단일 스레드 테스트만 하는 것이다. 실제로는 여러 카메라에서 동시에 이미지가 들어온다. Race condition, deadlock, resource 경합이 발생한다.

올바른 접근은 동시성 테스트이다. 여러 스레드/프로세스에서 동시에 추론한다. Stress test로 극한 부하를 시뮬레이션한다. Thread-safe한 구현을 보장한다. 필요시 queue와 worker pool을 사용한다. GPU는 batch 처리로 효율화한다.

실제 사례를 보자. FastFlow를 4개 카메라에서 동시 사용했다. GPU memory 경합으로 추론 시간이 20ms → 80ms로 증가했다. Batch 처리(4개 이미지 한 번에)로 변경하여 카메라당 25ms로 개선했다.

**Pitfall 4: 모델 업데이트 전략 부재**

흔한 실수는 모델을 한 번 배포하고 방치하는 것이다. 시간이 지나면 정상 패턴이 변화한다(계절, 공정 변경). 성능이 서서히 저하된다(95% → 90% → 85%).

올바른 접근은 지속적 모니터링과 재학습 전략이다. 일일/주간 성능을 추적한다. 1-2%p 하락 시 재학습을 트리거한다. 정기 재학습(월 1회 또는 분기 1회)을 스케줄링한다. A/B 테스트로 새 모델을 검증한다. 롤백 계획을 준비한다.

실제 사례를 보자. 자동차 도장 검사를 6개월간 재학습 없이 운영했다. 여름 조명 변화로 95% → 87%로 하락했다. 모니터링 대시보드를 구축하고 분기별 재학습을 자동화하여 95% 이상을 유지했다.

---

## 12. Migration Strategies

### 12.1 From Prototype to Production

프로토타입에서 프로덕션으로 전환은 단순한 배포가 아니다. 안정성, 확장성, 유지보수성을 확보해야 한다.

**Phase 1: 프로토타입 검증**

프로토타입(DFM, WinCLIP)의 성능을 측정한다. 목표 정확도에 얼마나 근접하는가(94-95% vs 목표 98%). 실제 환경에서 작동하는가(조명 변화, 진동). Gap을 분석한다. 부족한 정확도는 얼마인가(3-4%p). 원인은 무엇인가(모델 한계, 데이터 부족).

프로덕션 모델을 선택한다. Gap이 크면(5%p 이상) PatchCore나 Dinomaly를 선택한다. Gap이 작으면(2-3%p) FastFlow를 고려한다. 특수 요구사항(multi-class, few-shot)을 반영한다.

**Phase 2: 프로덕션 모델 개발**

데이터를 대규모로 수집한다. 프로토타입 50장 → 프로덕션 200-500장으로 증가한다. 다양한 조건을 커버한다. 품질을 엄격히 검증한다.

모델을 학습하고 최적화한다. 하이퍼파라미터를 튜닝한다(1-3%p 향상). 앙상블을 고려한다(추가 0.5-1%p). 배치 처리를 구현한다(throughput 최적화).

**Phase 3: 파일럿 배포**

실제 라인에서 제한적으로 배포한다. 한 라인 또는 한 제품에서 시작한다. 병렬 운영한다. 기존 방법(육안)과 AI를 동시 운영한다. 일치율을 측정한다(95% 이상 목표).

문제를 조기에 발견하고 해결한다. False positive/negative를 분석한다. Edge case를 수집한다. 모델을 개선한다(fine-tuning, 재학습).

**Phase 4: 전면 배포**

점진적으로 확대한다. 10% → 50% → 100% 라인으로 단계적 배포한다. 각 단계에서 검증한다. 모니터링 시스템을 구축한다. 실시간 대시보드로 성능을 추적한다. 알람 시스템을 설정한다(성능 하락, 시스템 오류).

운영 프로세스를 확립한다. 일상 점검 체크리스트를 만든다. 재학습 프로세스를 문서화한다. 장애 대응 매뉴얼을 작성한다.

**전환 체크리스트**

프로토타입 단계에서 Feasibility가 확인되었는가(90%+ 성능). 비즈니스 케이스가 명확한가(ROI 계산). 이해관계자의 승인을 받았는가.

개발 단계에서 목표 정확도를 달성했는가(98-99%). 실제 환경에서 검증했는가. 하드웨어가 준비되었는가.

파일럿 단계에서 병렬 운영이 성공적인가(95% 일치). Edge case가 처리되는가. 운영진이 훈련되었는가.

전면 배포 단계에서 모니터링 시스템이 작동하는가. 재학습 프로세스가 확립되었는가. 롤백 계획이 있는가.

### 12.2 Model Upgrade Paths

기존 모델에서 더 나은 모델로 업그레이드하는 전략이다. 각 상황에 맞는 upgrade path를 제시한다.

**Path 1: DFM → PatchCore (정확도 향상)**

현재 상태는 DFM(94.5-95.5%)으로 프로토타입을 운영 중이다. 목표는 99%+ 정확도 달성이다.

업그레이드 이유는 성능 gap이 크다(4-5%p). 불량 유출이 증가한다. 고객 불만이나 비용 증가가 발생한다.

마이그레이션 전략은 다음과 같다. 데이터를 100-500장으로 증가한다(현재 50-100장). PatchCore를 학습한다(1-2시간). A/B 테스트를 수행한다(DFM vs PatchCore). PatchCore가 우수하면 전환한다. 기대 효과는 정확도 4-5%p 향상(94.5% → 99.1%)이다. 불량 검출률 증가로 비용 절감이 된다. 메모리는 증가(50MB → 300MB)하지만 허용 가능하다.

**Path 2: STFPM → EfficientAD (속도 향상)**

현재 상태는 STFPM(96.8%, 20-40ms)으로 배치 처리 중이다. 목표는 실시간 처리(<10ms) 달성이다.

업그레이드 이유는 라인 속도가 증가했다(초당 50개 → 200개). STFPM이 병목이 되어 버퍼가 쌓인다. 실시간 피드백이 필요하다.

마이그레이션 전략은 다음과 같다. EfficientAD를 학습한다(동일 데이터 사용 가능). 정확도를 검증한다(97.8% vs 96.8%, +1%p). 속도를 측정한다(1-5ms, 10-40배 향상). 허용 가능하면 전환한다. 기대 효과는 속도 10-40배 향상(20-40ms → 1-5ms)이다. 실시간 전수 검사가 가능하다. 정확도도 약간 향상된다(96.8% → 97.8%, +1%p).

**Path 3: Multiple PatchCore → Dinomaly (메모리 절감)**

현재 상태는 15개 제품에 각각 PatchCore를 배포했다. 총 메모리 7.5GB, 관리 복잡하다.

업그레이드 이유는 메모리 제약이 있다(GPU 8GB에서 빠듯). 신제품 추가가 어렵다. 모델 관리가 복잡하다(15개 버전 관리).

마이그레이션 전략은 다음과 같다. 모든 제품의 데이터를 통합한다(15 × 200장 = 3,000장). Dinomaly를 학습한다(3-5시간, 단일 모델). 각 제품별 성능을 검증한다(98.8% 유지). 모든 제품이 목표를 만족하면 전환한다. 기대 효과는 메모리 93% 절감(7.5GB → 500MB)이다. 관리 간소화(15개 → 1개 모델)가 된다. 신제품 추가 용이(데이터만 추가)하다. 배포 시간 80% 단축(15시간 → 3시간)된다.

**Path 4: Traditional → Foundation Model (미래 대비)**

현재 상태는 PatchCore/FastFlow로 single-class 운영 중이다. 목표는 2025-2026년 기술 트렌드 대응이다.

업그레이드 이유는 multi-class 필요성이 증가한다. Foundation model이 표준이 될 것으로 예상된다. 최신 기술 도입으로 경쟁 우위를 확보한다.

마이그레이션 전략은 다음과 같다. 현재 시스템을 유지한다(급할 필요 없음). Dinomaly를 병렬로 실험한다(파일럿). 성능과 효율을 비교한다. 장기적 이점이 명확하면 전환을 계획한다. 기대 효과는 미래 확장성이 확보된다. 최신 기술의 지속적 개선을 누린다(DINOv2 발전). Multi-class로 유연성이 증가한다.

**업그레이드 의사결정 매트릭스**

현재 모델의 문제가 무엇인가를 확인한다. 정확도 부족 → PatchCore/Reverse Distillation으로 업그레이드한다. 속도 부족 → EfficientAD로 업그레이드한다. 메모리 부족 → EfficientAD/Dinomaly로 업그레이드한다. 관리 복잡 → Dinomaly로 업그레이드한다.

업그레이드 비용이 합리적인가를 평가한다. 개발 시간(1-4주)을 확인한다. 인력 비용($2,500-10,000)을 계산한다. 기대 효과가 비용을 상회하는가를 판단한다.

리스크가 관리 가능한가를 확인한다. 롤백 계획이 있는가를 본다. A/B 테스트가 가능한가를 확인한다. 파일럿으로 검증 가능한가를 본다.

### 12.3 Multi-model Ensemble

단일 모델의 한계를 극복하기 위해 여러 모델을 결합한다. 0.5-2%p 추가 성능 향상이 가능하다.

**Ensemble Strategy 1: 보완적 모델 결합**

개념은 서로 다른 강점을 가진 모델을 결합하는 것이다. PatchCore(정확도)+ EfficientAD(속도)로 2단계 검사를 한다. Memory-based + Flow-based로 다양한 관점을 제공한다.

구현 방법은 다음과 같다. 1단계에서 EfficientAD로 고속 스크리닝한다(1-5ms). 의심 샘플만 통과시킨다(이상 점수 상위 10%). 2단계에서 PatchCore로 정밀 검사한다(50-100ms). 최종 판정을 내린다.

효과는 전체 throughput을 유지한다(대부분 1단계). 정확도를 극대화한다(의심 샘플은 2단계). False positive를 줄인다(2단계 검증).

적용 사례는 반도체 웨이퍼 검사이다. EfficientAD로 초당 200개 처리한다. 의심 샘플 10%(20개)를 PatchCore로 재검사한다. 총 처리 시간은 1-5ms × 200 + 50ms × 20 = 1-2초/200개이다. 평균 10ms/개로 실시간을 유지하면서 정밀도를 확보한다.

**Ensemble Strategy 2: 투표(Voting)**

개념은 여러 모델의 예측을 투표로 결합하는 것이다. PatchCore, FastFlow, Reverse Distillation 3개 모델을 사용한다. Majority voting으로 최종 판정한다.

구현 방법은 다음과 같다. 3개 모델로 독립적으로 추론한다. 각 모델의 이상 점수를 정규화한다(0-1 범위). 평균을 계산한다(simple average). 또는 가중 평균을 계산한다(성능 비례 가중치). 임계값으로 최종 판정한다.

효과는 개별 모델보다 0.5-1.5%p 향상된다(99.1% → 99.5-100%). False positive/negative가 감소한다. 강건성이 증가한다(한 모델 실패 시 다른 모델이 보완).

적용 사례는 의료 기기 검사이다. PatchCore(99.1%), Reverse Distillation(98.6%), FastFlow(98.5%)를 결합한다. 가중 평균(0.5, 0.3, 0.2)을 사용한다. 최종 정확도 99.6%를 달성한다.

**Ensemble Strategy 3: 계층적 결합**

개념은 단계별로 다른 모델을 사용하는 것이다. 1단계에서 Coarse detection(큰 결함)을 하고, 2단계에서 Fine detection(미세 결함)을 한다.

구현 방법은 다음과 같다. 1단계에서 FastFlow로 전체 이미지를 검사한다(20-50ms). 큰 결함을 빠르게 탐지한다. 2단계에서 의심 영역을 crop한다. PatchCore로 crop 영역을 정밀 검사한다(50-100ms). 미세 결함을 탐지한다.

효과는 속도와 정확도를 모두 확보한다. 대부분 이미지는 1단계만으로 충분하다(80%). 의심 이미지만 2단계를 거친다(20%). 평균 처리 시간은 20-50ms + 20% × 50-100ms = 30-70ms이다.

적용 사례는 OLED 패널 검사이다. FastFlow로 mura 후보를 빠르게 찾는다. 후보 영역을 Reverse Distillation으로 정밀 분석한다. Pixel-level로 정확한 경계를 파악한다.

**Ensemble 구현 팁**

모델 선택을 신중히 한다. 보완적인 모델을 선택한다(다른 패러다임, 다른 강점). 너무 유사한 모델은 효과가 적다(PatchCore + PaDiM). 3-5개가 적정하다(더 많으면 계산량만 증가).

가중치를 최적화한다. 성능 비례 가중치를 사용한다(높은 AUROC → 높은 가중치). 검증 데이터로 최적 가중치를 찾는다. Grid search로 탐색한다.

계산 비용을 고려한다. Ensemble은 N배 느리다(N개 모델). 병렬 처리로 완화한다(GPU 배치). 계층적 구조로 효율화한다(2단계 검사).

효과를 검증한다. 0.5%p 미만 향상이면 비용 대비 효과가 적다. 단일 모델 최적화가 더 나을 수 있다. 1%p 이상 향상이면 투자 가치가 있다.

---

## 13. Future-Proofing

### 13.1 Technology Trends

향후 2-5년간의 기술 발전 방향을 이해하고 대비한다.

**Trend 1: Foundation Model 보편화 (2025-2026)**

현재 상황은 Dinomaly, UniNet 등 초기 모델이 등장했다. Multi-class 98.8%, single-class 99.2%를 달성한다. 그러나 아직 초기 단계이다.

예상 발전은 다음과 같다. 산업 특화 foundation model이 등장한다(Manufacturing CLIP, Industrial DINOv2). 수천만 장의 산업 이미지로 학습된다. Multi-class 정확도가 99%+ 달성된다. Zero-shot 정확도가 95-97%로 향상된다.

비즈니스 임팩트는 multi-class가 표준이 된다. Single-class는 특수 케이스가 된다. 메모리와 관리 비용이 80-90% 감소한다. 신제품 대응이 극도로 빨라진다(일 단위).

대비 전략은 현재 Dinomaly를 파일럿으로 테스트한다. Multi-class 가능성을 열어둔다. Foundation model에 투자하는 회사(OpenAI, Meta, Google)를 추적한다.

**Trend 2: Edge AI 확산 (2025-2027)**

현재 상황은 EfficientAD가 엣지 배포를 가능하게 했다. 그러나 여전히 제한적이다(97.8% 정확도).

예상 발전은 다음과 같다. 더 경량화된 모델이 등장한다(<100MB, <5ms). 전용 NPU/TPU가 보편화된다(스마트폰, IoT). 엣지에서 99%+ 정확도가 가능해진다.

비즈니스 임팩트는 클라우드 의존성이 감소한다(비용 절감). 데이터 프라이버시가 향상된다(온디바이스 처리). 실시간 응답이 보장된다(네트워크 지연 없음). 새로운 응용이 열린다(모바일 검사, 드론, 로봇).

대비 전략은 현재 시스템을 엣지 호환 가능하게 설계한다(ONNX export). 경량 모델(EfficientAD)을 우선 고려한다. NPU/TPU 지원을 모니터링한다.

**Trend 3: Explainable AI 필수화 (2026-2028)**

현재 상황은 VLM-AD가 자연어 설명을 제공한다. 그러나 비용이 높고($0.01-0.05/img) 느리다(2-5초).

예상 발전은 다음과 같다. On-premise VLM이 보편화된다(LLaMA, Vicuna). 설명 생성이 빨라진다(2-5초 → 0.5-1초). 비용이 감소한다($0.01 → $0.001). 규제가 강화된다(EU AI Act, FDA).

비즈니스 임팩트는 설명이 필수 요구사항이 된다. 특히 규제 산업(의료, 항공)에서 강제된다. 품질 보고서가 완전 자동화된다. 근본 원인 분석이 실시간으로 이루어진다.

대비 전략은 VLM-AD를 샘플링 검사에 도입한다. On-premise VLM 인프라를 준비한다. 설명 가능한 모델(PatchCore, Memory-based)을 우선한다.

**Trend 4: Continual Learning (2027-2030)**

현재 상황은 정기 재학습(월/분기)이 필요하다. 새로운 패턴에 즉시 적응하지 못한다.

예상 발전은 다음과 같다. 실시간 학습이 가능한 모델이 등장한다. Catastrophic forgetting이 해결된다. 무중단 업데이트가 보편화된다.

비즈니스 임팩트는 재학습 비용이 제로가 된다. 계절 변화에 자동 적응한다. 새 결함 유형을 즉시 학습한다. 공정 변경 시 자동으로 반영된다.

대비 전략은 데이터 수집 파이프라인을 자동화한다. 온라인 학습 연구를 추적한다. 모델 업데이트 프로세스를 정립한다.

### 13.2 When to Upgrade

현재 모델을 언제 새 모델로 교체해야 하는지 판단 기준을 제시한다.

**Trigger 1: 성능 저하**

모니터링 지표로 일일/주간 AUROC를 추적한다. 1-2%p 하락 시 조사한다. False positive/negative rate를 확인한다. 10% 이상 증가 시 조치한다.

액션 플랜은 먼저 재학습을 시도한다(새 데이터 추가). 개선되지 않으면 모델 업그레이드를 고려한다. 환경 변화(조명, 공정)가 원인이면 데이터 증강을 강화한다.

실제 예시는 자동차 도장 검사에서 겨울 → 여름 전환으로 성능이 95% → 90%로 하락했다. 여름 조명 데이터를 추가하여 재학습했다. 93%로 회복했지만 목표(95%)에 미달했다. FastFlow → Dinomaly로 업그레이드하여 97%를 달성했다.

**Trigger 2: 요구사항 변경**

새로운 요구사항으로 실시간 처리 필요(<10ms)가 생겼다. Multi-class 필요(신제품 추가)가 발생했다. 더 높은 정확도(99%+)가 요구된다. 설명 가능성이 필수가 되었다.

액션 플랜은 요구사항을 충족하는 모델을 선택한다. 실시간 → EfficientAD, Multi-class → Dinomaly, 정확도 → PatchCore, 설명 → VLM-AD로 마이그레이션한다. 파일럿으로 검증 후 전환한다.

실제 예시는 플라스틱 성형에서 3개 제품 → 15개 제품으로 확장했다. PatchCore 3개 → 15개는 관리가 불가능했다. Dinomaly로 통합하여 메모리 1.5GB → 500MB, 관리 15개 → 1개로 간소화했다.

**Trigger 3: 기술 발전**

새로운 모델 등장으로 성능이 2%p 이상 향상된다. 비용이 50% 이상 감소한다(메모리, 속도). 새로운 기능(multi-class, explainable)이 제공된다.

액션 플랜은 새 모델의 성숙도를 평가한다(논문, 벤치마크, 커뮤니티). 파일럿으로 실제 데이터에서 테스트한다. 명확한 이점이 있으면 마이그레이션 계획을 수립한다. ROI가 긍정적인지 확인한다.

실제 예시는 2025년 Dinomaly 등장 시 multi-class 98.8% + 메모리 93% 절감이 확인되었다. 파일럿에서 자사 데이터로 98.5%를 달성했다. ROI 분석 결과 6개월 내 회수 가능으로 판단되었다. 마이그레이션을 결정하여 성공적으로 전환했다.

**Upgrade 의사결정 프레임워크**

현재 모델의 문제를 파악한다. 성능 저하, 요구사항 미충족, 기술 낙후 중 무엇인가를 확인한다.

대안 모델을 평가한다. 3-5개 후보 모델을 식별한다. 각각의 장단점을 비교한다. 파일럿으로 실제 검증한다.

비용-효과를 분석한다. 개발 비용(시간, 인력)을 산정한다. 기대 효과(성능 향상, 비용 절감)를 추정한다. ROI를 계산한다. 6-12개월 내 회수 가능하면 진행한다.

리스크를 평가한다. 마이그레이션 실패 시 영향을 분석한다. 롤백 계획을 수립한다. 단계적 전환(10% → 50% → 100%)을 계획한다.

### 13.3 Continuous Improvement

일회성 배포가 아닌 지속적 개선 프로세스를 확립한다.

**Improvement Cycle 1: 데이터 기반 개선**

월간 활동으로 False positive/negative 샘플을 수집한다. 100-500개를 목표로 한다. 패턴을 분석한다. 공통점을 찾는다(특정 조명, 제품 변형). 학습 데이터에 추가한다. 재학습을 수행한다(월 1회).

기대 효과는 매월 0.2-0.5%p 정확도가 향상된다. 1년 후 95% → 97-98%로 개선된다. Edge case가 감소한다.

구현 팁은 자동 수집 파이프라인을 구축하는 것이다. False positive/negative를 자동으로 로깅한다. 월말에 검토 회의를 갖는다. 재학습을 자동화한다(스크립트, CI/CD).

**Improvement Cycle 2: 모델 앙상블**

분기별 활동으로 현재 모델 성능을 평가한다. 개선 여지를 찾는다(0.5-1%p). 보완적 모델을 테스트한다. Ensemble로 결합한다. 효과가 있으면 배포한다.

기대 효과는 분기마다 0.5-1%p 정확도가 향상된다. 1년 후 98% → 99-100%로 개선된다. 강건성이 증가한다.

구현 팁은 앙상블 프레임워크를 미리 구축하는 것이다. 새 모델 추가를 쉽게 만든다. A/B 테스트로 효과를 검증한다. 비용(계산량 증가)을 모니터링한다.

**Improvement Cycle 3: 기술 추적**

연간 활동으로 최신 논문을 리뷰한다(CVPR, ICCV, WACV). Anomalib 업데이트를 추적한다. 새 모델을 파일럿으로 테스트한다. 명확한 이점이 있으면 마이그레이션한다.

기대 효과는 최신 기술을 활용한다. 경쟁 우위를 유지한다. 기술 부채를 방지한다.

구현 팁은 분기별 tech review 회의를 갖는 것이다. 1-2명이 최신 동향을 담당한다. 파일럿 예산을 확보한다($5,000-10,000/년). 성공 사례를 공유한다.

**Improvement Cycle 4: 프로세스 최적화**

지속적 활동으로 재학습 프로세스를 자동화한다. 데이터 수집 → 학습 → 검증 → 배포를 자동화한다. 모니터링 대시보드를 개선한다. 실시간 알람을 강화한다. 운영 매뉴얼을 업데이트한다.

기대 효과는 운영 비용이 30-50% 감소한다. 사람 개입이 최소화된다. 대응 시간이 단축된다(수일 → 수시간).

구현 팁은 DevOps/MLOps 프랙티스를 도입하는 것이다. CI/CD 파이프라인을 구축한다. Infrastructure as Code를 사용한다. 문서를 지속적으로 업데이트한다.

**Continuous Improvement 체크리스트**

월간으로 False positive/negative를 분석했는가, 재학습을 수행했는가, 성능 대시보드를 리뷰했는가를 확인한다.

분기별로 앙상블 개선을 테스트했는가, Tech review 회의를 가졌는가, 운영 프로세스를 점검했는가를 확인한다.

연간으로 모델 업그레이드를 평가했는가, 최신 기술을 파일럿했는가, 장기 로드맵을 업데이트했는가를 확인한다.

---

## References

### Research Documents

상세한 기술 분석은 다음 패러다임별 문서를 참조한다.

- [00-overview.md](00-overview.md) - Vision Anomaly Detection Overview
- [01-memory-based.md](01-memory-based.md) - Memory-Based and Feature Matching Methods
- [02-normalizing-flow.md](02-normalizing-flow.md) - Normalizing Flow Approaches
- [03-knowledge-distillation.md](03-knowledge-distillation.md) - Knowledge Distillation Methods  
- [04-reconstruction-based.md](04-reconstruction-based.md) - Reconstruction-Based Approaches
- [05-feature-adaptation.md](05-feature-adaptation.md) - Feature Adaptation and Transfer Learning
- [06-foundation-models.md](06-foundation-models.md) - Foundation Model-Based Methods

### User Documentation

PixelVision 프레임워크의 실제 구현은 다음 문서를 참조한다.

- [Getting Started](../docs/01-getting-started.md) - Installation and Quick Start
- [Architecture](../docs/02-architecture.md) - Framework Design and Components
- [Models Reference](../docs/03-models.md) - Detailed Model Documentation
- [Datasets Guide](../docs/04-datasets.md) - Dataset Preparation and Management
- [Training Guide](../docs/05-training.md) - Training Configuration and Best Practices
- [Inference Guide](../docs/06-inference.md) - Deployment and Inference Workflows

### Key Papers

각 패러다임의 대표 논문은 다음과 같다.

**Memory-Based Methods:**
- Defard et al., "PaDiM: Patch Distribution Modeling Framework for Anomaly Detection and Localization", ICPR 2020
- Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022 (PatchCore)

**Normalizing Flow Methods:**
- Gudovskiy et al., "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows", WACV 2022
- Yu et al., "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows", 2021

**Knowledge Distillation Methods:**
- Wang et al., "Student-Teacher Feature Pyramid Matching for Anomaly Detection", BMVC 2021 (STFPM)
- Deng and Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding", CVPR 2022
- Batzner et al., "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies", WACV 2024

**Reconstruction-Based Methods:**
- Akcay et al., "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training", ACCV 2018
- Zavrtanik et al., "DRAEM: Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection", ICCV 2021

**Foundation Model Methods:**
- Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation", CVPR 2023
- Zhang et al., "Dinomaly: Multi-class Anomaly Detection via Self-Supervised Learning", 2025

### External Resources

- [Anomalib GitHub Repository](https://github.com/openvinotoolkit/anomalib) - Official Implementation
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) - Standard Benchmark
- [Papers with Code - Anomaly Detection](https://paperswithcode.com/task/anomaly-detection) - Latest Research
- [VisA Dataset](https://github.com/amazon-science/spot-diff) - Visual Anomaly Dataset
- [BTAD Dataset](http://avires.dimi.uniud.it/papers/btad/btad.zip) - Beantech Anomaly Detection

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Models Compared**: 21  
**Paradigms Covered**: 6  
**Total Pages**: 150+  
**Maintainer**: PixelVision Research Team
