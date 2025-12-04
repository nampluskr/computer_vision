## GAN·CGAN·ACGAN·InfoGAN 비교 (텍스트 다이어그램)

아래 표와 간단한 **블록 다이어그램**을 통해 네 모델의 **아키텍처**, **입력·출력 형태**, **핵심 차이점**을 한눈에 확인할 수 있습니다.  
예시에서는 **64 × 64 × 3** RGB 이미지를 생성한다고 가정하고, **노이즈 벡터** `z ∈ ℝ^d (d=100)` 를 사용합니다.  

---

### 1️⃣ 전체 비교 표

| 모델 | Generator 입력 | Generator 출력 | Discriminator 입력 | Discriminator 출력 | 주요 특징 |
|------|----------------|----------------|--------------------|--------------------|-----------|
| **GAN** | `z` (100‑dim) | `x̂ = G(z)` → (3, 64, 64) | `x` 혹은 `x̂` (3, 64, 64) | `D(x)` → 실/가짜 확률 (scalar) | 조건 없음, 가장 기본 형태 |
| **CGAN** | `z` + **조건 라벨** `y` (one‑hot, C‑dim) | `x̂ = G(z, y)` → (3, 64, 64) | `x` 혹은 `x̂` + **조건 라벨** `y` | `D(x, y)` → 실/가짜 확률 (scalar) | 라벨을 **concat** 혹은 **embedding** 후 입력 |
| **ACGAN** | `z` + **조건 라벨** `y` (one‑hot) | `x̂ = G(z, y)` → (3, 64, 64) | `x` 혹은 `x̂` (3, 64, 64) | ① `D_src(x)` → 실/가짜 확률 (scalar) <br>② `D_cls(x)` → 클래스 확률 (C‑dim) | Discriminator가 **보조 분류기**를 갖고, 라벨을 **예측** (real/fake와 별도) |
| **InfoGAN** | `z` + **잠재 코드** `c = (c_cat, c_cont)` <br> - `c_cat` : one‑hot (K‑dim) <br> - `c_cont` : 연속형 (M‑dim) | `x̂ = G(z, c)` → (3, 64, 64) | `x` 혹은 `x̂` (3, 64, 64) | ① `D_src(x)` → 실/가짜 확률 (scalar) <br>② `Q(x)` → `ĉ` (ĉ_cat, ĉ_cont) | Discriminator에 **Q‑head**를 추가해 **Mutual Information** I(c; G(z,c)) 를 최대화 |

---

### 2️⃣ 블록 다이어그램 (ASCII)

#### (a) Vanilla GAN
```
   z ──► [ Generator G ] ──► x̂ (3×64×64)
                                 │
                                 ▼
   ──────────────────────► [ Discriminator D ] ──► D(x̂) ∈ [0,1]
   (real image x also fed to D)
```

#### (b) Conditional GAN (CGAN)
```
   z ──►┐
        │  (concat / embed) ──► [ Generator G ] ──► x̂
   y ──►┘                                 │
                                          ▼
   ──────────────────────► [ Discriminator D ] ──► D(x̂, y)
   (real image x + same y also fed)
```

#### (c) Auxiliary Classifier GAN (ACGAN)
```
   z ──►┐
        │  (concat / embed) ──► [ Generator G ] ──► x̂
   y ──►┘                                 │
                                          ▼
   ──────────────────────► [ Discriminator D ]
                         │
                         ├─► D_src(x)  (real/fake scalar)
                         └─► D_cls(x)  (C‑dim softmax)
   (real image x + its true label y also fed)
```

#### (d) InfoGAN
```
   z ──►┐
        │  (concat) ──► [ Generator G ] ──► x̂
   c ──►┘                                 │
                                          ▼
   ──────────────────────► [ Discriminator D ]
                         │
                         ├─► D_src(x)  (real/fake scalar)
                         └─► Q(x)      (predict ĉ_cat, ĉ_cont)
   (real image x also fed → D_src(x) only)
```

---

### 3️⃣ 입력·출력 상세 (예시)

| 모델 | `z` (noise) | `y` (class) | `c_cat` | `c_cont` | `G` 출력 | `D` 출력 (real/fake) | `D`/`Q` 추가 출력 |
|------|-------------|-------------|---------|----------|----------|----------------------|-------------------|
| GAN | 100‑dim | – | – | – | (3, 64, 64) | scalar | – |
| CGAN | 100‑dim | C‑dim one‑hot (ex. 10) | – | – | (3, 64, 64) | scalar | – |
| ACGAN | 100‑dim | C‑dim one‑hot | – | – | (3, 64, 64) | scalar | C‑dim softmax (class) |
| InfoGAN | 100‑dim | – | K‑dim one‑hot (ex. 10) | M‑dim (ex. 2) | (3, 64, 64) | scalar | (K‑dim logits, M‑dim μ, M‑dim logσ) |

> **Tip**  
> - `y` 혹은 `c` 를 **Embedding → concat** 하는 경우, `z` 와 같은 차원(`d`) 로 임베딩을 맞추면 `torch.cat([z, embed])` 로 간단히 연결할 수 있습니다.  
> - ACGAN·InfoGAN에서는 **Discriminator**가 두 개(또는 세 개)의 헤드를 갖게 되므로, `nn.Module` 안에 `self.adv_head`, `self.aux_head`(ACGAN) 혹은 `self.q_head`(InfoGAN) 로 구현하면 깔끔합니다.

---

### 4️⃣ 핵심 차이점 요약

| 구분 | 라벨/코드 사용 방식 | Discriminator 역할 | 학습 목표 |
|------|-------------------|-------------------|-----------|
| **GAN** | 없음 | 진위 판별만 | `min_G max_D V(D,G)` |
| **CGAN** | **조건 라벨**을 **입력**에 직접 concat/embedding | 진위 판별 (조건 포함) | 라벨‑조건 일관성 확보 |
| **ACGAN** | **조건 라벨**을 **입력**에 concat + **보조 분류** (출력) | ① 진위 판별 ② 클래스 예측 | 라벨‑조건을 **두 개의 손실**(BCE + CE) 로 동시에 최적화 |
| **InfoGAN** | **잠재 코드** `c` (카테고리·연속) → **입력** | ① 진위 판별 ② `Q`‑head 로 `c` 복원 | **Mutual Information** `I(c; G(z,c))` 를 최대화 (Info‑loss) |

---

## 5️⃣ 구현 시 참고 코드 스니펫 (PyTorch)

```python
# ---------- GAN ----------
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 4*4*512),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, img_ch, 4, 2, 1), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 0)   # (B,1,1,1)
        )
    def forward(self, x): return self.net(x).view(-1, 1)
```

*CGAN, ACGAN, InfoGAN 은 위 `Generator`/`Discriminator` 에 **조건/보조 헤드**만 추가하면 됩니다.*  

---

### 마무리

- **GAN** → 가장 단순한 “노이즈 → 이미지” 구조.  
- **CGAN** → 라벨을 **입력**에 넣어 **조건부** 생성.  
- **ACGAN** → 라벨을 **입력**에 넣고, **Discriminator**가 라벨을 **예측**하는 보조 분류기를 가짐.  
- **InfoGAN** → 라벨 대신 **잠재 코드** `c` 를 넣고, **Q‑head** 로 `c` 를 복원해 **정보량**을 최대화.

각 모델은 **조건부 생성**이라는 공통 목표를 가지고 있지만, **조건을 어디에 두고 어떻게 활용하느냐**에 따라 아키텍처와 학습 목표가 달라집니다. 필요에 따라 위 표와 다이어그램을 참고해 구현을 선택하면 됩니다. 🚀
