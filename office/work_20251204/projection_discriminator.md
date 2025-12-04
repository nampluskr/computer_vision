## Projection Discriminator란?  

**Projection Discriminator**(투영 판별기)는 **조건부 GAN(Conditional GAN)** 에서 라벨(또는 임베딩) 정보를 **단순히 concat(연결)** 하는 대신, **내부에 내적(inner‑product) 연산**을 이용해 라벨과 이미지 특징을 “투영”시켜 판별에 활용하는 구조입니다.  

> **핵심 아이디어**  
> - 이미지 특징 벡터 **f(x)** 와 라벨 임베딩 **e(y)** 를 **점곱(·)** 으로 결합 → 라벨이 이미지 특징 공간에 **정렬(alignment)** 되는 정도를 점수에 반영한다.  
> - 이렇게 하면 라벨과 이미지 사이의 **상호 정보량(mutual information)** 을 직접적으로 최적화할 수 있어, 조건부 생성 품질과 라벨 일관성이 크게 향상된다.  

아래에서는 원리를 수식으로 정리하고, 기존 **concat‑discriminator** 와 비교한 뒤, 구현 시 주의사항과 활용 사례를 정리합니다.  

---

## 1️⃣ 수식적 정의  

조건부 GAN에서 판별기의 출력은 **real/fake** 를 나타내는 스칼라 **D(x, y)** 로 정의됩니다.  
Projection Discriminator는 다음과 같이 구성됩니다.

$$\boxed{
D(x, y) = \underbrace{c^\top f(x)}_{\text{이미지 스코어}} 
          + \underbrace{e(y)^\top f(x)}_{\text{라벨‑이미지 투영}} 
          + b(y)
}$$

| 기호 | 의미 |
|------|------|
| \(x\) | 입력 이미지 |
| \(y\) | 조건 라벨 (정수형 클래스 인덱스 등) |
| \(f(x) \in \mathbb{R}^d\) | 이미지 특징 추출기(Convolutional backbone) → **global‑average‑pooled** 후 얻은 벡터 |
| \(e(y) \in \mathbb{R}^d\) | 라벨 임베딩(학습 가능한 **Embedding** 레이어) |
| \(c \in \mathbb{R}^d\) | 이미지만으로 판단하는 **bias vector** (보통 `nn.Linear(d, 1, bias=False)` 로 구현) |
| \(b(y) \in \mathbb{R}\) | 라벨별 **bias term** (선택적, `nn.Embedding(n_classes, 1)` 로 구현) |

### 1‑1. 왜 내적(inner‑product)인가?  

- **점곱**은 두 벡터가 같은 방향을 가질 때 큰 값을, 서로 다른 방향이면 작은 값을 반환합니다.  
- 따라서 **\(e(y)^\top f(x)\)** 가 크게 나오면, 이미지 특징이 해당 라벨의 임베딩과 잘 정렬(일치)된 것으로 해석됩니다.  
- 라벨이 없는 **unconditional** GAN에서는 두 번째 항과 \(b(y)\) 를 제거하고, 첫 번째 항만 남게 됩니다.

### 1‑2. Loss  

보통 **시그모이드 BCE** 혹은 **Wasserstein** 손실을 사용합니다.

- **시그모이드 BCE**  
  $$
  \mathcal{L}_D = -\mathbb{E}_{x\sim p_{\text{real}}}\big[\log\sigma(D(x,y))\big]
                -\mathbb{E}_{\tilde{x}\sim G(z,y)}\big[\log(1-\sigma(D(\tilde{x},y)))\big]
  $$

- **Wasserstein** (Spectral Normalization 적용 시)  
  $$
  \mathcal{L}_D = -\mathbb{E}_{x\sim p_{\text{real}}}[D(x,y)] + \mathbb{E}_{\tilde{x}\sim G(z,y)}[D(\tilde{x},y)] + \lambda\cdot\text{GP}
  $$

---

## 2️⃣ 기존 “Concat‑Discriminator”와의 차이  

| 구조 | 입력 결합 방식 | 파라미터 효율성 | 라벨‑이미지 상호작용 | 대표 논문 |
|------|----------------|----------------|----------------------|-----------|
| **Concat‑Discriminator** | \([f(x); e(y)]\) 를 `Linear` 로 바로 연결 | 라벨 차원만큼 **추가 파라미터** (예: `Linear(d+e_dim, 1)`) | 라벨 정보가 **선형 결합**에만 사용 → 라벨‑이미지 관계를 충분히 표현하기 어려움 | Mirza & Osindero, 2014 |
| **Projection Discriminator** | \(c^\top f(x) + e(y)^\top f(x) + b(y)\) | 라벨 임베딩 차원 **d** 와 동일하게 공유 → **파라미터 증가가 거의 없음** | 라벨과 이미지 특징이 **점곱**으로 직접 상호작용 → 라벨‑이미지 정렬을 명시적으로 학습 | Miyato & Koyama, 2018 (“cGANs with Projection Discriminator”) |

- **파라미터 절감**: 라벨 차원이 커져도 `e(y)` 가 `d` 차원(이미지 특징 차원)만 가지므로, `concat` 방식보다 훨씬 적은 파라미터를 사용합니다.  
- **학습 안정성**: 라벨과 이미지가 같은 공간에 투영되므로, 라벨이 없는 경우에도 `cᵀf(x)` 로 충분히 판별이 가능해 **gradient flow** 가 원활합니다.  

---

## 3️⃣ 구현 팁 (PyTorch 예시)

아래는 **Spectral Normalization**을 적용한 Projection Discriminator의 최소 구현입니다.

```python
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class ProjectionDiscriminator(nn.Module):
    """
    이미지 x와 라벨 y를 받아서 D(x, y)를 반환한다.
    - f(x) : Conv backbone + GlobalAvgPool → (B, d)
    - e(y) : nn.Embedding(n_classes, d)
    - c    : nn.Linear(d, 1, bias=False)   (이미지 전용 스코어)
    - b(y) : nn.Embedding(n_classes, 1)   (optional bias)
    """
    def __init__(self, img_channels=3, img_size=64,
                 n_classes=10, feat_dim=128, use_bias=True):
        super().__init__()
        self.feat_dim = feat_dim

        # 1) 이미지 특징 추출기 (간단히 Conv + GAP)
        self.backbone = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),  # (B,64,32,32)
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),           # (B,128,16,16)
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, feat_dim, 4, 2, 1)),      # (B,feat_dim,8,8)
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))                         # (B,feat_dim,1,1)
        )
        # (B, feat_dim)
        self.flatten = nn.Flatten()

        # 2) 라벨 임베딩 (학습 가능한 파라미터)
        self.embed_y = nn.Embedding(n_classes, feat_dim)

        # 3) 이미지 전용 bias vector cᵀf(x)
        self.fc = spectral_norm(nn.Linear(feat_dim, 1, bias=False))

        # 4) 라벨별 bias term b(y) (optional)
        self.use_bias = use_bias
        if use_bias:
            self.bias_y = nn.Embedding(n_classes, 1)

        # 초기화 (Xavier)
        nn.init.xavier_uniform_(self.embed_y.weight)
        if use_bias:
            nn.init.zeros_(self.bias_y.weight)

    def forward(self, img, label):
        """
        img   : (B, C, H, W)
        label : (B,)  long tensor (0~n_classes-1)
        반환값: (B, 1)   – D(x, y) 스칼라 로그it
        """
        # 이미지 특징
        h = self.backbone(img)          # (B, feat_dim, 1, 1)
        h = self.flatten(h)             # (B, feat_dim)

        # 라벨 임베딩
        y_emb = self.embed_y(label)     # (B, feat_dim)

        # D(x) = cᵀ f(x)
        img_score = self.fc(h)          # (B, 1)

        # 라벨‑이미지 투영 term = e(y)ᵀ f(x)
        proj_score = torch.sum(h * y_emb, dim=1, keepdim=True)   # (B, 1)

        out = img_score + proj_score

        if self.use_bias:
            out = out + self.bias_y(label)   # (B, 1)

        return out
```

### 구현 시 주의사항  

| 항목 | 설명 |
|------|------|
| **Spectral Normalization** | `spectral_norm`을 모든 `Conv`와 `Linear`에 적용하면 **Wasserstein GAN** 형태에서도 안정적인 학습이 가능하다. |
| **feat_dim 선택** | 라벨 임베딩 차원과 이미지 특징 차원을 동일하게 맞춰야 한다 (`feat_dim`). 일반적으로 64~256 사이가 많이 쓰인다. |
| **라벨 임베딩 초기화** | `nn.init.xavier_uniform_` 혹은 `normal_(0, 0.02)` 로 초기화하면 초반에 과도한 bias를 방지한다. |
| **bias_y 사용 여부** | 라벨별 **bias term** 은 선택 사항이며, `b(y)` 를 넣으면 **class‑conditional prior** 를 더 명시적으로 반영한다. |
| **다중 라벨(멀티‑핫)** | 라벨이 여러 개일 경우 `e(y)` 를 각 라벨 임베딩을 평균/합산한 뒤 사용한다. |
| **연속형 코드** | 연속형 조건을 사용할 경우 `e(y)` 를 `nn.Linear` 로 만든 뒤 `proj_score` 에 그대로 사용하면 된다. |
| **GPU 메모리** | `feat_dim`이 클수록 메모리 사용량이 증가한다. `feat_dim=128` 정도면 64×64 이미지 기준 ~200 MB 정도. |
| **학습 안정성** | `D`와 `G` 모두 **Adam(β1=0.5, β2=0.999)** 혹은 **RMSProp** 로 학습하고, `D` 업데이트를 `G`보다 1~5배 더 많이 수행한다. |

---

## 4️⃣ 왜 Projection Discriminator가 좋은가? (실험적 근거)

| 논문 / 구현 | 주요 결과 |
|-------------|-----------|
| **Miyato & Koyama, 2018 – “cGANs with Projection Discriminator”** | CIFAR‑10/100, ImageNet‑64 에서 **Concat‑Discriminator 대비 2~5%** 높은 Inception Score / FID 개선. |
| **BigGAN (Brock et al., 2019)** | 대규모 클래스(1000)에서 **Projection** 방식을 기본 채택 → 클래스 일관성 크게 향상. |
| **SAGAN / Self‑Attention GAN** | 라벨‑조건부 self‑attention과 결합했을 때 **시각적 다양성**과 **라벨 정확도**가 크게 상승. |
| **StyleGAN2‑ADA (Karras et al., 2020)** | 라벨이 있는 경우 **Projection**을 사용하면 **Fidelity**와 **Conditional Consistency**가 크게 개선됨. |
| **Diffusion 모델 (ADM, 2021)** | 조건부 diffusion에서도 **Projection** 형태의 판별기(또는 classifier‑free guidance) 가 **샘플 품질**을 높이는 데 기여. |

> **핵심 포인트**  
> - 라벨과 이미지 특징을 **같은 벡터 공간**에 투영함으로써, **라벨‑이미지 정합성**을 직접적인 스칼라 점수에 반영한다.  
> - 파라미터 효율성 + 학습 안정성 → 대규모 클래스(수천~수만)에서도 사용 가능.  

---

## 5️⃣ 활용 예시 (실제 프로젝트에 적용하기)

### 5‑1. 작은 데이터셋 (MNIST, CIFAR‑10)  
- **feat_dim = 64** 정도로 설정하고, `spectral_norm` 없이 `nn.Linear` 로도 충분.  
- `b(y)` 를 **사용**하면 클래스별 bias가 학습돼 **Faster convergence** 를 경험할 수 있다.

### 5‑2. 대규모 이미지 (ImageNet‑128, CelebA‑HQ)  
- **feat_dim = 256~512** 로 늘리고, **Spectral Normalization** + **Adam(β1=0.0)** 로 학습.  
- 라벨이 1000개 이상이면 **Embedding을 sparse** 로 선언하고 `torch.optim.SparseAdam` 사용 → 메모리 절감.

### 5‑3. 멀티‑조건 (예: CelebA 속성 + 클래스)  
```python
# 여러 라벨을 각각 임베딩 → 평균 후 concat
attr_emb = self.attr_embed(attr_idx)          # (B, n_attr, embed_dim)
attr_vec = attr_emb.mean(dim=1)               # (B, embed_dim)

class_emb = self.class_embed(class_idx)      # (B, embed_dim)

cond_vec = torch.cat([attr_vec, class_emb], dim=1)   # (B, 2*embed_dim)
# 이후 proj_score = torch.sum(f(x) * cond_vec, dim=1, keepdim=True)
```

### 5‑4. 연속형 조건 (예: 회전 각도, 색상)  
- 연속형 변수를 **선형 변환** (`nn.Linear(1, d)`) 으로 임베딩하고, 위와 동일하게 점곱한다.  
- `proj_score` 가 연속형 변수와 이미지 특징 사이의 **상관관계**를 직접 학습한다.

---

## 6️⃣ 한계와 주의점  

| 한계 | 설명 | 해결 방안 |
|------|------|----------|
| **라벨 임베딩 차원 선택** | 차원이 너무 작으면 라벨 정보를 충분히 표현하지 못한다. | `feat_dim` 과 비슷하거나 약간 큰 차원(예: `feat_dim*1.5`)을 사용. |
| **라벨 불균형** | 일부 라벨이 거의 등장하지 않으면 `e(y)` 가 충분히 학습되지 않는다. | **클래스 가중치** 혹은 **oversampling**을 적용하거나, `b(y)` 로 보완. |
| **연속형 코드와 정규화** | 연속형 라벨을 그대로 임베딩하면 스케일 차이로 학습이 불안정할 수 있다. | 라벨을 **표준화(z‑score)** 후 임베딩하거나, `nn.LayerNorm` 적용. |
| **스파스 임베딩과 GPU 효율** | `sparse=True` 를 쓰면 `SparseAdam` 은 CPU‑GPU 전송 오버헤드가 있다. | 클래스가 10 k 이상일 때만 사용하고, **embedding bucket**(예: hierarchical) 로 대체. |
| **다중 라벨(멀티‑핫) 결합** | 단순 평균/합산은 라벨 간 상호작용을 무시한다. | **Self‑Attention** 혹은 **FiLM** 같은 메커니즘을 라벨 임베딩에 적용. |

---

## 7️⃣ 요약  

1. **Projection Discriminator**는  
   $$
   D(x,y)=c^\top f(x)+e(y)^\top f(x)+b(y)
   $$  
   형태로 라벨과 이미지 특징을 **점곱**해 결합한다.  

2. **장점**  
   - 파라미터 효율성 (라벨 차원에 비례하지 않음)  
   - 라벨‑이미지 정합성을 직접 스코어에 반영 → **조건부 일관성** 향상  
   - **Spectral Normalization**과 잘 어울려 **학습 안정성**을 제공  

3. **구현 포인트**  
   - 이미지 backbone → global‑avg‑pool → `f(x)`  
   - 라벨 임베딩 `e(y)` 와 `b(y)` (optional) 를 정의  
   - `torch.sum(f(x) * e(y), dim=1, keepdim=True)` 로 투영 점수 계산  
   - `nn.Linear` 혹은 `spectral_norm(nn.Linear)` 로 이미지 전용 bias `c` 구현  

4. **활용 사례**  
   - BigGAN, SAGAN, StyleGAN2‑ADA 등 대규모 클래스 조건부 GAN에서 표준으로 사용  
   - 작은 데이터셋에서도 **Concat**보다 **Projection**이 더 빠르게 수렴  

> **결론**: 라벨 정보를 단순히 연결하는 것이 아니라 **같은 특징 공간에 투영**함으로써, 조건부 GAN에서 **품질·일관성**을 크게 끌어올릴 수 있는 강력하고 효율적인 설계가 바로 **Projection Discriminator**입니다.  

궁금한 점이나 실제 코드 적용 중 겪는 문제(예: 메모리, 학습 불안정 등)가 있으면 언제든 알려 주세요! 🚀
