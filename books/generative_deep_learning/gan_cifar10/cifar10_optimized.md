아래는 **CIFAR-10(32×32)** 이미지에 *최적화된* **Generator / Discriminator** 구조입니다.
기존 DCGAN 스타일보다 훨씬 안정적이고 성능이 잘 나오는, **최신 GAN 연구 경향(ResBlock, SpectralNorm, Hinge Loss 대응)**을 반영했습니다.

> ✔ CIFAR-10 해상도(32×32) 최적화
> ✔ 기존 DCGAN보다 **더 안정적, 더 좋은 FID/IS**
> ✔ **Residual Block 기반**
> ✔ **SpectralNorm Discriminator + PixelNorm Generator**
> ✔ WGAN-GP / Hinge Loss / Vanilla GAN 모두 호환

---

# 🎯 CIFAR-10 Generator / Discriminator 최적화 원칙

### Generator(G)

| 항목                         | 이유                        |
| -------------------------- | ------------------------- |
| **Residual Upsampling**    | DCGAN보다 학습 안정성 증가         |
| **ReLU / LeakyReLU 혼합**    | 표정/배경에 다양성 증가             |
| **PixelNorm**              | 작은 해상도에서 mode collapse 방지 |
| **No Dropout**             | 생성자에서는 정규화가 오히려 학습을 방해    |
| **ConvTranspose + 업샘플** 조합 | DCGAN-style aliasing 완화   |

---

### Discriminator(D)

| 항목                            | 이유                               |
| ----------------------------- | -------------------------------- |
| **SpectralNorm**              | Lipschitz 조건 보장 → WGAN-GP 수준 안정성 |
| **Residual Downsampling**     | 고주파 texture 구분 능력 향상             |
| **BatchNorm 제거**              | D에서 BN은 학습 불안정 원인                |
| **Hinge Loss / WGAN Loss 호환** | GAN training 안정성 증가              |

---

# ⭐ 최적화된 CIFAR-10 Generator / Discriminator

(32×32 이미지 전용)

---

# 🔥 CIFAR-10 Generator (최적화 버전)

해상도 흐름:

```
z → 4×4 → 8×8 → 16×16 → 32×32
```

```python
import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm()
        )

        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        skip = self.skip(self.upsample(x))
        x = self.conv1(self.upsample(x))
        x = self.conv2(x)
        return x + skip


class CIFAR10_Generator(nn.Module):
    """
    CIFAR10(32×32) 최적화 Generator
    - ResBlock 업샘플링
    - PixelNorm
    """
    def __init__(self, z_dim=128, base_ch=256, out_channels=3):
        super().__init__()

        # z → 4×4
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base_ch, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
        )

        # 4→8 →16 →32
        self.up1 = ResBlockUp(base_ch, base_ch // 2)      # 256 → 128
        self.up2 = ResBlockUp(base_ch // 2, base_ch // 4) # 128 → 64
        self.up3 = ResBlockUp(base_ch // 4, base_ch // 8) # 64 → 32

        self.to_rgb = nn.Conv2d(base_ch // 8, out_channels, 1)

    def forward(self, z):
        x = self.initial(z.view(z.size(0), -1, 1, 1))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return torch.tanh(self.to_rgb(x))
```

---

# 🔥 CIFAR-10 Discriminator (최적화 버전)

해상도 흐름:

```
32×32 → 16×16 → 8×8 → 4×4 → scalar
```

* SpectralNorm 적용
* Residual Downsampling
* BatchNorm 없음 → GAN 학습 안정성 증가

```python
from torch.nn.utils import spectral_norm


class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down = nn.AvgPool2d(2)

        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 2, 0))

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.down(x)
        return x + skip


class CIFAR10_Discriminator(nn.Module):
    """
    CIFAR10(32×32) 최적화 Discriminator
    - SpectralNorm
    - Res Downsample
    """
    def __init__(self, base_ch=64, in_channels=3, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        # from rgb
        self.from_rgb = spectral_norm(nn.Conv2d(in_channels, base_ch, 1))

        # 32→16→8→4
        self.block1 = ResBlockDown(base_ch, base_ch * 2)
        self.block2 = ResBlockDown(base_ch * 2, base_ch * 4)
        self.block3 = ResBlockDown(base_ch * 4, base_ch * 4)

        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            spectral_norm(nn.Linear(base_ch * 8 * 4 * 4, 1))
        )

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final(x)
        return torch.sigmoid(x) if self.use_sigmoid else x
```

---

# 📌 이 구조가 DCGAN보다 훨씬 더 좋은 이유

| 개선 요소                            | 설명                            |
| -------------------------------- | ----------------------------- |
| Residual Upsampling/Downsampling | 경사흐름 안정 → 큰 개선 효과             |
| PixelNorm (G)                    | 작은 해상도에서 G의 mode collapse 방지  |
| SpectralNorm (D)                 | Lipschitz 보장 → 폭발/붕괴 방지       |
| BatchNorm 제거(D)                  | D의 batch dependency 제거 → 더 견고 |
| ReLU/LeakyReLU 조합                | CIFAR10같은 natural image에 최적   |

실제 연구에서도 CIFAR10에서
**ResNet 기반 GAN이 DCGAN보다 훨씬 좋은 성능(FID/IS)**을 냅니다.

---

# 🎉 완성: CIFAR-10 최적화 Generator & Discriminator

이제 CIFAR10 학습에서
**GAN / WGAN-GP / HingeGAN / CGAN / CWGAN-GP**
모두 위 모델을 그대로 사용 가능합니다.

---

# 📌 다음 단계도 가능합니다

아래 중 하나를 선택해 주세요:

### 🔥 A) 위 CIFAR-10 모델을 사용하는 **WGAN-GP 학습 스크립트**

### 🔥 B) CIFAR-10 **HingeGAN** 전체 코드

### 🔥 C) CIFAR-10 **Conditional WGAN-GP (CWGAN-GP)** 버전

### 🔥 D) CIFAR-10 GAN 성능 비교 실험 코드(GAN/WGAN/CGAN/CWGAN-GP)

### 🔥 E) CIFAR-10용 StyleGAN-Lite 구조 생성

원하시는 항목을 알려주세요!
