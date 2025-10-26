# PixelVision Research Documentation

학술 연구 자료 및 SOTA 모델 분석 문서이다.

---

## Overview

본 디렉토리는 Vision Anomaly Detection 분야의 최신 연구 동향과 주요 패러다임을 분석한 학술 자료를 포함한다. PixelVision에 구현된 20개 모델의 이론적 배경, 기술적 원리, 그리고 실무 적용 가이드를 제공한다. 각 패러다임별로 대표 모델들의 핵심 알고리즘과 성능 벤치마크를 상세히 다루며, 실무 환경에서의 적용 전략을 제시한다.

---

## Survey Documents

### 패러다임별 심층 분석

이상 감지 분야를 6개의 주요 패러다임으로 분류하고, 각 패러다임의 핵심 원리와 대표 모델을 분석한다.

#### 1. [Overview](00-overview.md) - 전체 개요

이상 감지 기술의 발전 과정과 주요 도전 과제를 다룬다.

**주요 내용:**
- 산업 이상 감지의 도전 과제
- 6개 패러다임 소개
- 시간순 기술 발전 (2018-2025)
- 주요 기술적 전환점
- 향후 연구 방향

전체 분야를 조망하여 독자가 패러다임 간의 상호 연관성과 발전 방향을 이해할 수 있도록 구성되어 있다.

#### 2. [Memory-Based Methods](01-memory-based.md) - 메모리 기반 방식

정상 샘플의 특징을 메모리에 저장하고 거리 기반으로 이상을 탐지하는 방법론이다.

**분석 모델:**
- PaDiM (2020): Patch Distribution Modeling
- PatchCore (2022): Coreset Selection
- DFKDE (2022): Kernel Density Estimation

**핵심 성과:**
- 최고 정확도: 99.1% (PatchCore)
- Coreset으로 메모리 90% 절감
- Single-class 환경의 표준

**수식 및 알고리즘:**
- Mahalanobis distance
- Greedy Coreset Selection
- k-NN anomaly scoring

특히 PatchCore의 Coreset Selection 알고리즘은 메모리 효율성 측면에서 혁신적인 성과를 달성했다.

#### 3. [Normalizing Flow](02-normalizing-flow.md) - 정규화 플로우

가역적 변환을 통해 정상 데이터의 확률 분포를 모델링하는 생성 모델 기반 접근법이다.

**분석 모델:**
- CFlow (2021): Conditional Normalizing Flow
- FastFlow (2021): 2D Normalizing Flow
- CS-Flow (2021): Cross-Scale Flow
- U-Flow (2022): U-shaped Flow

**핵심 성과:**
- 확률적 해석 가능 (log-likelihood)
- FastFlow: 98.5% AUROC, 20-50ms
- 속도-정확도 균형의 대표 모델

**기술적 혁신:**
- 3D → 2D 단순화로 속도 3배 향상
- Multi-scale feature fusion

FastFlow의 단순화 전략은 복잡도를 낮추면서도 성능을 유지할 수 있음을 입증했다.

#### 4. [Knowledge Distillation](03-knowledge-distillation.md) - 지식 증류

Teacher-Student 구조로 정상 패턴을 학습하며, 이상 샘플은 모방에 실패한다는 원리를 활용한다.

**분석 모델:**
- STFPM (2021): Student-Teacher Feature Pyramid Matching
- FRE (2023): Feature Reconstruction Error
- Reverse Distillation (2022): One-Class Embedding
- EfficientAD (2024): Millisecond-Level Detection

**핵심 성과:**
- 정밀: Reverse Distillation 98.6%
- 실시간: EfficientAD 1-5ms (20-200배 향상)
- CPU 환경에서도 10-20ms 가능

**패러다임의 양극단:**
- 정밀 검사 (Reverse Distillation)
- 실시간 처리 (EfficientAD)

EfficientAD는 산업 현장에서 요구하는 밀리초 단위 추론 속도를 달성한 획기적인 모델이다.

#### 5. [Reconstruction-Based](04-reconstruction-based.md) - 재구성 기반

정상 데이터로 학습된 재구성 모델이 이상 샘플을 제대로 재구성하지 못한다는 원리를 활용한다.

**분석 모델:**
- GANomaly (2018): GAN-based Semi-Supervised
- DRAEM (2021): Discriminatively Trained Reconstruction
- DSR (2022): Dual Subspace Re-Projection
- Autoencoder (Baseline): Vanilla Autoencoder

**핵심 성과:**
- Few-shot: DRAEM 10-50장으로 97.5%
- Simulated Anomaly 패러다임 전환
- 신제품 및 희귀 결함 대응

**패러다임 전환:**
- GANomaly (GAN 불안정) → DRAEM (Supervised 안정)

DRAEM의 Simulated Anomaly 기법은 실제 결함 샘플 없이도 효과적인 학습을 가능케 한다.

#### 6. [Feature Adaptation](05-feature-adaptation.md) - 특징 적응

Pre-trained 모델의 특징을 타겟 도메인에 적응시켜 활용하는 전이 학습 방법이다.

**분석 모델:**
- DFM (2019): Deep Feature Modeling
- CFA (2022): Coupled-hypersphere Feature Adaptation

**핵심 성과:**
- 빠른 프로토타입: DFM 15분 학습
- 94.5-95.5% 정확도
- 저사양 환경 적합

**실무 역할:**
- 빠른 Feasibility 검증
- Baseline 구축
- 본격 배포 전 프로토타입

빠른 학습 속도로 인해 초기 개념 검증 단계에서 유용하게 활용된다.

#### 7. [Foundation Models](06-foundation-models.md) - 파운데이션 모델

대규모 사전 학습 모델(CLIP, DINOv2, GPT-4V)을 활용한 차세대 접근법이다.

**분석 모델:**
- WinCLIP (2023): Zero-shot with CLIP
- Dinomaly (2025): Multi-class with DINOv2
- VLM-AD (2024): Vision-Language Model
- SuperSimpleNet (2024): Unified Framework
- UniNet (2025): Contrastive Learning

**핵심 성과:**
- Multi-class: Dinomaly 98.8% (단일 모델)
- Zero-shot: WinCLIP 91-95% (학습 0분)
- Explainable: VLM-AD 자연어 설명

**패러다임 전환:**
- Single-class → Multi-class
- 학습 필요 → Zero-shot 가능
- 수치만 → 자연어 설명

Foundation Model의 등장으로 전통적인 Single-class 학습 패러다임이 Multi-class Zero-shot 방식으로 진화하고 있다.

#### 8. [Comprehensive Comparison](07-comparison.md) - 종합 비교

6개 패러다임의 장단점, 성능 벤치마크, 그리고 실무 적용 가이드를 제공한다.

**주요 내용:**
- 패러다임별 종합 평가
- MVTec AD 벤치마크 비교
- 성능-속도-메모리 Trade-off
- 시나리오별 최적 모델 선택
- 하드웨어 환경별 권장사항
- 개발 단계별 로드맵
- 비용-효과 분석

실무 의사결정을 위한 정량적, 정성적 비교 분석을 종합적으로 제시한다.

---

## Quick Reference

### 성능 비교 (MVTec AD Benchmark)

| 패러다임 | 대표 모델 | Image AUROC | 추론 속도 | 메모리 | 주요 장점 |
|---------|----------|-------------|-----------|--------|----------|
| Memory-Based | PatchCore | **99.1%** | 50-100ms | 100-500MB | 최고 정확도 |
| Foundation | Dinomaly | 98.8% | 80-120ms | 300-500MB | Multi-class |
| Distillation | Reverse Distillation | 98.6% | 100-200ms | 500MB-1GB | Pixel-level |
| Flow | FastFlow | 98.5% | 20-50ms | 500MB-1GB | 균형 |
| Distillation | EfficientAD | 97.8% | **1-5ms** | **<200MB** | 실시간 |
| Reconstruction | DRAEM | 97.5% | 50-100ms | 300-500MB | Few-shot |

### 시나리오별 추천 모델

**최고 정확도 필요 (>99%)**
- Single-class: [Memory-Based](01-memory-based.md) - PatchCore (99.1%)
- Multi-class: [Foundation Models](06-foundation-models.md) - Dinomaly (98.8%)

**실시간 처리 (<10ms)**
- [Knowledge Distillation](03-knowledge-distillation.md) - EfficientAD (1-5ms)

**균형잡힌 성능**
- [Normalizing Flow](02-normalizing-flow.md) - FastFlow (98.5%, 20-50ms)

**Few-shot (10-50장)**
- [Reconstruction](04-reconstruction-based.md) - DRAEM (97.5%)

**Zero-shot (학습 0분)**
- [Foundation Models](06-foundation-models.md) - WinCLIP (91-95%)

**빠른 프로토타입 (15분)**
- [Feature Adaptation](05-feature-adaptation.md) - DFM (94.5-95.5%)

---

## Document Structure

각 서베이 문서는 다음 구조로 작성되어 있다:

1. **패러다임 개요**: 핵심 원리와 수학적 정식화
2. **모델별 상세 분석**: 
   - 기본 정보 (논문, 저자, 링크)
   - 핵심 원리 및 수식
   - 기술적 세부사항
   - 성능 및 벤치마크
   - 장단점 분석
3. **종합 비교**: 패러다임 내 모델 간 비교
4. **실무 적용 가이드**: 시나리오별 선택 기준
5. **참고 자료**: 관련 논문 및 구현 링크

일관된 구조를 통해 독자가 패러다임 간 비교와 모델 선택을 체계적으로 수행할 수 있도록 한다.

---

## Reading Guide

### 처음 읽는 독자

1. [Overview](00-overview.md) - 전체 그림 이해
2. [Comparison](07-comparison.md) - 패러다임 비교
3. 관심 있는 특정 패러다임 문서

이 순서로 읽으면 전체적인 맥락을 파악한 후 세부 내용으로 진입할 수 있다.

### 특정 모델 연구

1. [Models 문서](../docs/03-models.md) - 모델 기본 정보
2. 해당 패러다임 서베이 - 이론적 배경
3. 원 논문 - 상세 내용

연구자를 위한 심화 학습 경로를 제시한다.

### 실무 적용 목적

1. [Comparison](07-comparison.md) - 시나리오별 추천
2. [Training Guide](../docs/05-training.md) - 실제 학습
3. 필요시 특정 패러다임 문서 참조

실무자는 빠른 의사결정을 위해 비교 문서부터 시작하는 것을 권장한다.

---

## Key Insights

### 기술적 전환점

**PaDiM → PatchCore (2020-2022)**
- 문제: 메모리 2-5GB
- 해결: Coreset Selection
- 효과: 메모리 90% 절감 + 성능 향상

메모리 효율성을 크게 개선하면서도 정확도를 유지한 대표적인 사례다.

**CFlow → FastFlow (2021)**
- 문제: 3D flow, 100-150ms
- 해결: 2D 단순화
- 효과: 속도 3배 + 성능 유지

복잡도를 낮추면서 성능을 유지할 수 있음을 보여준 모델이다.

**STFPM → Reverse Distillation (2021-2022)**
- 문제: Teacher의 일반적 특징
- 해결: 패러다임 역전
- 효과: 96.8% → 98.6%

전통적인 지식 증류 방향을 뒤집어 더 나은 성능을 달성했다.

**GANomaly → DRAEM (2018-2021)**
- 문제: GAN 학습 불안정
- 해결: Simulated Anomaly
- 효과: 안정적 학습 + Few-shot

GAN의 불안정성 문제를 Simulated Anomaly로 해결하며 패러다임을 전환했다.

**전통적 → Foundation Models (2023-2025)**
- 문제: Single-class, 학습 데이터 필요
- 해결: 대규모 사전 학습 활용
- 효과: Multi-class, Zero-shot 가능

Foundation Model의 등장으로 이상 감지 분야가 근본적으로 재편되고 있다.

### 설계 원칙

**단순화의 힘**
- FastFlow: 3D → 2D로 속도 3배, 성능 향상
- 교훈: 불필요한 복잡도 제거가 개선

과도한 복잡도는 오히려 성능 저하를 유발할 수 있다.

**점진적 vs 혁명적 개선**
- FRE: 2배 개선 (실패)
- EfficientAD: 20-200배 개선 (성공)
- 교훈: 충분한 임팩트가 필요

점진적 개선보다는 혁명적 변화가 실무에 채택될 가능성이 높다.

**실용성의 승리**
- 이론적 우아함 < 실무 효과
- GAN < Simulated Anomaly
- 3D Flow < 2D Flow

실무에서는 이론적 우아함보다 실제 효과가 중요하다.

---

## Citation

본 연구 자료를 참고할 경우, 각 서베이 문서에 명시된 원 논문을 인용하기 바란다.

### BibTeX 예시

```bibtex
@inproceedings{roth2022patchcore,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{yu2021fastflow,
  title={FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows},
  author={Yu, Jiawei and Zheng, Ye and Wang, Xiang and Li, Wei and Wu, Yushuang and Zhao, Rui and Wu, Liwei},
  year={2021}
}
```

---

## Related Resources

### PixelVision Documentation

- [User Guide](../docs/) - 사용자 가이드
- [Models Reference](../docs/03-models.md) - 모델 구현 상세
- [Training Guide](../docs/05-training.md) - 학습 설정
- [Inference Guide](../docs/06-inference.md) - 배포 가이드

PixelVision의 전체 문서 체계와 연계하여 학습할 수 있다.

### External Resources

- [Anomalib GitHub](https://github.com/openvinotoolkit/anomalib) - 원본 라이브러리
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) - 벤치마크 데이터셋
- [Papers with Code](https://paperswithcode.com/task/anomaly-detection) - 최신 논문

외부 리소스를 통해 최신 연구 동향을 지속적으로 파악할 수 있다.

---

## Contributing

새로운 논문 분석이나 벤치마크 결과 기여를 환영한다.

**기여 방법:**
1. 새로운 모델/논문 발견 시 Issue 생성
2. 서베이 문서 개선 사항 제안
3. 벤치마크 결과 공유
4. 오류 수정 및 업데이트

자세한 내용은 [CONTRIBUTING.md](../CONTRIBUTING.md)를 참조하라.