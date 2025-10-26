# PixelVision Research Documentation - 목차 구조

- 블릿 사용을 하지 않고, 학술 리뷰 논문 형식으로 작성
- 균형잡힌 길이와 일관된 형식으로 작성

## 파일 구조 및 넘버링

```
research/
├── README.md
├── 00-overview.md
├── 01-memory-based.md
├── 02-normalizing-flow.md
├── 03-knowledge-distillation.md
├── 04-reconstruction-based.md
├── 05-feature-adaptation.md
├── 06-foundation-models.md
└── 07-comparison.md
```

---

## 00-overview.md - Overview

**제목:** Vision Anomaly Detection: Overview and Paradigm Evolution

### 목차

```
1. Introduction
   1.1 Challenges in Industrial Anomaly Detection
   1.2 Deep Learning Revolution
   1.3 Document Scope and Organization

2. Six Major Paradigms
   2.1 Paradigm Classification Framework
   2.2 Evolution Timeline (2018-2025)
   2.3 Paradigm Comparison Matrix

3. Key Technical Transitions
   3.1 Memory Efficiency Breakthrough (PaDiM → PatchCore)
   3.2 Speed Optimization (CFlow → FastFlow)
   3.3 Paradigm Inversion (STFPM → Reverse Distillation)
   3.4 Learning Stability (GANomaly → DRAEM)
   3.5 Multi-class Revolution (Traditional → Foundation Models)

4. Performance Landscape
   4.1 MVTec AD Benchmark Overview
   4.2 Top-5 Models by Accuracy
   4.3 Speed-Accuracy-Memory Trade-offs

5. Future Directions
   5.1 Short-term (2025-2026)
   5.2 Mid-term (2026-2028)
   5.3 Long-term (2028-2030)
   5.4 Zero-Defect Manufacturing Vision

6. Reading Guide
   6.1 For Beginners
   6.2 For Researchers
   6.3 For Practitioners

References
```

---

## 01-memory-based.md - Memory-Based Methods

**제목:** Memory-Based and Feature Matching Approaches

### 목차

```
1. Paradigm Overview
   1.1 Core Principle
   1.2 Mathematical Formulation
   1.3 Key Assumptions
   1.4 Historical Context

2. PaDiM (2020)
   2.1 Basic Information
   2.2 Core Algorithm
       2.2.1 Patch Distribution Modeling
       2.2.2 Multivariate Gaussian Assumption
       2.2.3 Mahalanobis Distance
   2.3 Technical Details
       2.3.1 Multi-scale Feature Extraction
       2.3.2 Covariance Matrix Computation
       2.3.3 Dimensionality Reduction
   2.4 Performance Analysis
   2.5 Advantages and Limitations
   2.6 Implementation Considerations

3. PatchCore (2022)
   3.1 Basic Information
   3.2 Coreset Selection Algorithm
       3.2.1 Greedy Subsampling
       3.2.2 Coverage Guarantee (ε-cover)
       3.2.3 Complexity Analysis
   3.3 Technical Innovations
       3.3.1 Locally Aware Patch Features
       3.3.2 Neighborhood Aggregation
       3.3.3 k-NN Anomaly Scoring
   3.4 PaDiM vs PatchCore Comparison
   3.5 Performance Analysis
   3.6 Memory Efficiency Breakthrough
   3.7 Implementation Guide

4. DFKDE (2022)
   4.1 Basic Information
   4.2 Kernel Density Estimation
       4.2.1 KDE Fundamentals
       4.2.2 Gaussian Kernel
       4.2.3 Bandwidth Selection
   4.3 Deep Feature Integration
   4.4 Comparison with Parametric Methods
   4.5 Performance and Limitations

5. Comprehensive Comparison
   5.1 Technical Evolution
   5.2 Detailed Comparison Table
   5.3 Memory Usage Analysis
   5.4 Computational Complexity
   5.5 Scalability Considerations

6. Practical Application Guide
   6.1 Model Selection Criteria
   6.2 Hyperparameter Tuning
   6.3 Training Pipeline
   6.4 Deployment Checklist
   6.5 Common Pitfalls

7. Research Insights
   7.1 Why Memory-Based Works
   7.2 Theoretical Guarantees
   7.3 Open Research Questions
```

---

## 02-normalizing-flow.md - Normalizing Flow Methods

**제목:** Normalizing Flow for Anomaly Detection

### 목차

```
1. Paradigm Overview
   1.1 Core Principle
   1.2 Mathematical Foundation
       1.2.1 Change of Variables
       1.2.2 Invertible Transformations
       1.2.3 Log-Likelihood Computation
   1.3 Advantages of Probabilistic Modeling

2. CFlow (2021)
   2.1 Basic Information
   2.2 Conditional Normalizing Flow
       2.2.1 Position-Conditional Architecture
       2.2.2 Multi-scale Processing
       2.2.3 Coupling Layers
   2.3 Technical Details
   2.4 Performance Analysis
   2.5 Limitations (Speed, Memory)

3. FastFlow (2021)
   3.1 Basic Information
   3.2 2D Normalizing Flow Innovation
       3.2.1 3D → 2D Simplification
       3.2.2 Channel Independence Assumption
       3.2.3 Speed-Accuracy Trade-off
   3.3 Architecture Design
   3.4 Performance Breakthrough
   3.5 Why Simplification Works
   3.6 Implementation Guide

4. CS-Flow (2021)
   4.1 Basic Information
   4.2 Cross-Scale Information Exchange
   4.3 Multi-resolution Features
   4.4 Performance and Use Cases

5. U-Flow (2022)
   5.1 Basic Information
   5.2 U-Net Integration
   5.3 Automatic Threshold Selection
   5.4 Operational Automation

6. Comprehensive Comparison
   6.1 Flow Architecture Evolution
   6.2 Performance Comparison
   6.3 Computational Analysis
   6.4 Design Trade-offs

7. Practical Application Guide
   7.1 Model Selection by Scenario
   7.2 Hyperparameter Tuning
   7.3 Training Strategies
   7.4 Deployment Considerations

8. Research Insights
   8.1 Why FastFlow Succeeded
   8.2 Channel vs Spatial Information
   8.3 Future Directions
```

---

## 03-knowledge-distillation.md - Knowledge Distillation Methods

**제목:** Knowledge Distillation for Anomaly Detection

### 목차

```
1. Paradigm Overview
   1.1 Core Principle
   1.2 Teacher-Student Framework
   1.3 Knowledge Transfer Mechanism
   1.4 Anomaly as Imitation Failure

2. STFPM (2021)
   2.1 Basic Information
   2.2 Student-Teacher Architecture
       2.2.1 Feature Pyramid Matching
       2.2.2 Multi-scale Knowledge Transfer
       2.2.3 Loss Functions
   2.3 Technical Details
   2.4 Performance Analysis
   2.5 Baseline Establishment

3. FRE (2023)
   3.1 Basic Information
   3.2 Feature Reconstruction Approach
   3.3 Lightweight Architecture
   3.4 Speed Optimization Attempt
   3.5 Why FRE Failed
       3.5.1 Insufficient Improvement
       3.5.2 Incremental vs Revolutionary
       3.5.3 Lessons Learned

4. Reverse Distillation (2022)
   4.1 Basic Information
   4.2 Paradigm Inversion
       4.2.1 One-Class Embedding
       4.2.2 Encoder-Decoder Structure
       4.2.3 Domain-Specific Teacher
   4.3 Technical Innovation
   4.4 SOTA Performance (98.6%)
   4.5 Pixel-Level Excellence
   4.6 Trade-offs (Speed vs Accuracy)

5. EfficientAD (2024)
   5.1 Basic Information
   5.2 Real-time Revolution
       5.2.1 Patch Description Network (PDN)
       5.2.2 Autoencoder Integration
       5.2.3 Extreme Optimization
   5.3 Architecture Design (~50K parameters)
   5.4 Performance Analysis
       5.4.1 1-5ms Inference
       5.4.2 CPU Capability
       5.4.3 97.8% Accuracy
   5.5 Edge Deployment
   5.6 Implementation Guide

6. Comprehensive Comparison
   6.1 Evolution Timeline
   6.2 Two Extremes
       6.2.1 Precision (Reverse Distillation)
       6.2.2 Speed (EfficientAD)
   6.3 Performance-Speed Analysis
   6.4 Use Case Matrix

7. Practical Application Guide
   7.1 Precision vs Speed Decision
   7.2 Model Selection by Requirements
   7.3 Hardware Considerations
   7.4 Deployment Strategies

8. Research Insights
   8.1 Knowledge Distillation Duality
   8.2 The Middle Ground Trap
   8.3 Revolutionary Optimization
```

---

## 04-reconstruction-based.md - Reconstruction-Based Methods

**제목:** Reconstruction-Based Anomaly Detection

### 목차

```
1. Paradigm Overview
   1.1 Core Principle
   1.2 Reconstruction Error as Anomaly Signal
   1.3 Normal Manifold Learning
   1.4 Historical Development

2. GANomaly (2018)
   2.1 Basic Information
   2.2 GAN-based Architecture
       2.2.1 Encoder-Decoder-Encoder Structure
       2.2.2 Adversarial Training
       2.2.3 Latent Space Representation
   2.3 Technical Challenges
   2.4 Training Instability Issues
   2.5 Why GANomaly Failed
       2.5.1 Mode Collapse
       2.5.2 Convergence Problems
       2.5.3 Long Training Time
   2.6 Lessons Learned

3. DRAEM (2021)
   3.1 Basic Information
   3.2 Paradigm Shift: Simulated Anomaly
       3.2.1 Synthetic Defect Generation
       3.2.2 Perlin Noise Augmentation
       3.2.3 Supervised Learning Effect
   3.3 Technical Architecture
       3.3.1 Reconstruction Network
       3.3.2 Discriminative Network
       3.3.3 Loss Functions (SSIM + Focal)
   3.4 Few-shot Capability (10-50 samples)
   3.5 Performance Analysis (97.5%)
   3.6 Training Stability
   3.7 Implementation Guide

4. DSR (2022)
   4.1 Basic Information
   4.2 Dual Subspace Architecture
       4.2.1 Quantization Subspace (Structure)
       4.2.2 Target Subspace (Texture)
       4.2.3 VQ-VAE Integration
   4.3 Texture Specialization
   4.4 Performance on Complex Surfaces
   4.5 Use Cases (Fabric, Carpet, Leather)

5. Autoencoder (Baseline)
   5.1 Vanilla Autoencoder
   5.2 Bottleneck Architecture
   5.3 Reconstruction Loss
   5.4 Baseline Performance
   5.5 Limitations

6. Comprehensive Comparison
   6.1 Paradigm Evolution
       6.1.1 Unsupervised (GANomaly)
       6.1.2 Supervised (DRAEM)
       6.1.3 Hybrid (DSR)
   6.2 Performance Comparison
   6.3 Training Stability Analysis
   6.4 Data Efficiency

7. Practical Application Guide
   7.1 Few-shot Scenarios
   7.2 Simulated Anomaly Design
   7.3 Model Selection
   7.4 Training Strategies
   7.5 Domain Adaptation

8. Research Insights
   8.1 Supervised vs Unsupervised
   8.2 Simulated Anomaly Effectiveness
   8.3 GAN Instability Lessons
```

---

## 05-feature-adaptation.md - Feature Adaptation Methods

**제목:** Feature Adaptation and Transfer Learning

### 목차

```
1. Paradigm Overview
   1.1 Core Principle
   1.2 Transfer Learning Foundation
   1.3 Pre-trained Feature Utilization
   1.4 Domain Adaptation

2. DFM (2019)
   2.1 Basic Information
   2.2 Deep Feature Modeling
       2.2.1 Pre-trained CNN Features
       2.2.2 PCA Dimensionality Reduction
       2.2.3 Mahalanobis Distance
   2.3 Extreme Simplicity
   2.4 Performance Analysis (94.5-95.5%)
   2.5 Fast Prototyping (15 minutes)
   2.6 Advantages and Limitations

3. CFA (2022)
   3.1 Basic Information
   3.2 Coupled-hypersphere Adaptation
       3.2.1 Hypersphere Projection
       3.2.2 Angular Distance
       3.2.3 Scale Invariance
   3.3 Domain Shift Robustness
   3.4 Performance Analysis (96.5-97.5%)
   3.5 Illumination/Camera Variation Handling

4. Comprehensive Comparison
   4.1 DFM vs CFA
   4.2 Performance Gap Analysis
   4.3 Computational Efficiency
   4.4 Domain Adaptation Capability

5. Fundamental Limitations
   5.1 Pre-trained Feature Domain Gap
   5.2 Linear Transformation Constraints
   5.3 SOTA Performance Gap
   5.4 When Not to Use

6. Practical Application Guide
   6.1 Rapid Prototyping Workflow
   6.2 Feasibility Testing
   6.3 Low-resource Environments
   6.4 Transition Strategy to SOTA Models

7. Research Insights
   7.1 Role as Entry Point
   7.2 Starting Point, Not Destination
   7.3 Value in Speed
```

---

## 06-foundation-models.md - Foundation Model Methods

**제목:** Foundation Models for Anomaly Detection

### 목차

```
1. Paradigm Overview
   1.1 Paradigm Shift
   1.2 Large-scale Pre-training
   1.3 Three Revolutions
       1.3.1 Multi-class
       1.3.2 Zero-shot
       1.3.3 Explainable AI

2. WinCLIP (2023)
   2.1 Basic Information
   2.2 Zero-shot with CLIP
       2.2.1 Text-Image Similarity
       2.2.2 Prompt Engineering
       2.2.3 No Training Required
   2.3 Performance Analysis (91-95%)
   2.4 Instant Deployment
   2.5 Use Cases
       2.5.1 New Product Launch
       2.5.2 Multi-variant Production
       2.5.3 Rapid Prototyping
   2.6 Limitations and Prompt Sensitivity

3. Dinomaly (2025)
   3.1 Basic Information
   3.2 Multi-class Revolution
       3.2.1 DINOv2 Foundation
       3.2.2 Single Unified Model
       3.2.3 Class-conditional Memory Bank
   3.3 Performance Analysis
       3.3.1 Multi-class: 98.8%
       3.3.2 Single-class: 99.2%
   3.4 Memory Efficiency (93% reduction)
   3.5 Business Impact
   3.6 Implementation Guide

4. VLM-AD (2024)
   4.1 Basic Information
   4.2 Vision-Language Models
       4.2.1 GPT-4V Integration
       4.2.2 Natural Language Explanation
       4.2.3 Root Cause Analysis
   4.3 Explainable AI Realization
   4.4 Output Examples
   4.5 Use Cases (Regulatory, Quality Reports)
   4.6 Cost Considerations

5. SuperSimpleNet (2024)
   5.1 Basic Information
   5.2 Unified Framework
   5.3 Unsupervised + Supervised Fusion
   5.4 Performance Analysis (97.2%)

6. UniNet (2025)
   6.1 Basic Information
   6.2 Contrastive Learning
   6.3 Robust Decision Boundaries
   6.4 Performance Analysis (98.3%)

7. Comprehensive Comparison
   7.1 Three Dimensions of Revolution
   7.2 Multi-class Economics
   7.3 Zero-shot Feasibility
   7.4 Explainability Value

8. Future Outlook (2025-2030)
   8.1 Multi-class Standardization
   8.2 Zero-shot Expansion
   8.3 Explainable AI Mandate
   8.4 Domain-specific Foundation Models

9. Practical Application Guide
   9.1 Dinomaly for Multi-class
   9.2 WinCLIP for Instant Deployment
   9.3 VLM-AD for Quality Reports
   9.4 Migration Strategy

10. Research Insights
    10.1 Foundation Model Impact
    10.2 Paradigm Transformation
    10.3 Industry Implications
```

---

## 07-comparison.md - Comprehensive Comparison

**제목:** Comprehensive Paradigm Comparison and Application Guide

### 목차

```
1. Executive Summary
   1.1 Six Paradigms at a Glance
   1.2 Top Recommendations by Scenario
   1.3 Decision Flowchart

2. Paradigm-by-Paradigm Evaluation
   2.1 Memory-Based Methods
       2.1.1 Strengths
       2.1.2 Weaknesses
       2.1.3 Best Use Cases
       2.1.4 Model Selection (PatchCore vs PaDiM vs DFKDE)
   2.2 Normalizing Flow Methods
       2.2.1 Strengths
       2.2.2 Weaknesses
       2.2.3 Best Use Cases
       2.2.4 Model Selection (FastFlow vs CFlow vs CS-Flow vs U-Flow)
   2.3 Knowledge Distillation Methods
       2.3.1 Strengths
       2.3.2 Weaknesses
       2.3.3 Two Extremes (Precision vs Speed)
       2.3.4 Model Selection (Reverse Distillation vs EfficientAD vs STFPM)
   2.4 Reconstruction-Based Methods
       2.4.1 Strengths
       2.4.2 Weaknesses
       2.4.3 Best Use Cases (Few-shot)
       2.4.4 Model Selection (DRAEM vs DSR vs GANomaly)
   2.5 Feature Adaptation Methods
       2.5.1 Strengths
       2.5.2 Weaknesses
       2.5.3 Best Use Cases (Rapid Prototyping)
       2.5.4 Model Selection (DFM vs CFA)
   2.6 Foundation Model Methods
       2.6.1 Strengths
       2.6.2 Weaknesses
       2.6.3 Best Use Cases (Multi-class, Zero-shot, Explainable)
       2.6.4 Model Selection (Dinomaly vs WinCLIP vs VLM-AD)

3. Performance Benchmarks
   3.1 MVTec AD Results
   3.2 Category-wise Performance
   3.3 Speed Benchmarks
   3.4 Memory Usage Analysis
   3.5 Benchmark Limitations

4. Trade-off Analysis
   4.1 Accuracy vs Speed
   4.2 Accuracy vs Memory
   4.3 Speed vs Memory
   4.4 Three-way Trade-off Visualization
   4.5 Impossible Combinations

5. Scenario-Based Selection Guide
   5.1 Maximum Accuracy (>99%)
   5.2 Real-time Processing (<10ms)
   5.3 Multi-class Environment
   5.4 Instant Deployment (Zero-shot)
   5.5 Few-shot Learning (10-50 samples)
   5.6 Quality Report Automation
   5.7 Balanced General Inspection

6. Hardware Environment Guide
   6.1 GPU Server (8GB+ VRAM)
   6.2 Edge GPU (4GB VRAM)
   6.3 CPU Only
   6.4 Cloud/API

7. Development Roadmap
   7.1 Phase 1: Prototyping (1-2 weeks)
   7.2 Phase 2: Optimization (2-4 weeks)
   7.3 Phase 3: Deployment Preparation (2-3 weeks)
   7.4 Phase 4: Operations (Continuous)

8. Cost-Benefit Analysis
   8.1 Initial Development Costs
   8.2 Operational Costs (Monthly)
   8.3 ROI Analysis
   8.4 Long-term Benefits

9. Decision Framework
   9.1 Decision Tree
   9.2 Checklist-based Selection
   9.3 Multi-criteria Decision Matrix

10. Industry Applications
    10.1 Semiconductor
    10.2 Medical Devices
    10.3 Automotive
    10.4 Electronics
    10.5 Display Quality (OLED)

11. Common Pitfalls
    11.1 Wrong Model Selection
    11.2 Inadequate Data Preparation
    11.3 Hyperparameter Mistakes
    11.4 Deployment Issues

12. Migration Strategies
    12.1 From Prototype to Production
    12.2 Model Upgrade Paths
    12.3 Multi-model Ensemble

13. Future-Proofing
    13.1 Technology Trends
    13.2 When to Upgrade
    13.3 Continuous Improvement

Appendix A: Complete Benchmark Tables
Appendix B: Hardware Requirements
Appendix C: Glossary
References
```

---

## 문서 간 상호 참조 구조

```
00-overview.md (시작점)
    ├─→ 01-memory-based.md (최고 정확도)
    ├─→ 02-normalizing-flow.md (균형)
    ├─→ 03-knowledge-distillation.md (양극단)
    ├─→ 04-reconstruction-based.md (Few-shot)
    ├─→ 05-feature-adaptation.md (빠른 시작)
    ├─→ 06-foundation-models.md (미래)
    └─→ 07-comparison.md (종합) ← 모든 문서 참조
```

---

## 각 문서의 역할

| 파일 | 역할 | 주요 독자 | 길이 |
|-----|------|----------|------|
| 00-overview.md | 전체 지도 | 모든 독자 | 중간 |
| 01-memory-based.md | 기술 심층 분석 | 연구자, 개발자 | 매우 긴 |
| 02-normalizing-flow.md | 기술 심층 분석 | 연구자, 개발자 | 긴 |
| 03-knowledge-distillation.md | 기술 심층 분석 | 연구자, 개발자 | 긴 |
| 04-reconstruction-based.md | 기술 심층 분석 | 연구자, 개발자 | 중간 |
| 05-feature-adaptation.md | 빠른 참조 | 초보자, 실무자 | 짧음 |
| 06-foundation-models.md | 미래 기술 | 모든 독자 | 중간 |
| 07-comparison.md | 의사결정 가이드 | 실무자, 관리자 | 매우 긴 |

---
