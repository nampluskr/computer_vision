아래는 **현재 만든 HighRes-Optimized GAN(GAN-N)** 구조를 유지하면서
**CIFAR10(32×32) → CELEBA(64×64) 확장** 버전입니다.

핵심 요구사항:

* 기존 구조(ResBlockUp/Down + PixelNorm(G) + SN(D) + Hinge Loss)는 그대로 유지
* **해상도를 32 → 64 로 확장**
* 4×4에서 시작해서 **4→8→16→32→64** 로 업샘플링
* Discriminator도 반대로 64→32→16→8→4 로 다운샘플링

즉, **Upsample/Downsample 블록 1개씩 추가**만 하면 됩니다.

---

# ⭐ 전체 구조 차이 (32 vs 64)

| Dataset | Resolution | Generator Blocks | Discriminator Blocks |
| ------- | ---------- | ---------------- | -------------------- |
| CIFAR10 | 32×32      | 4→8→16→32        | 32→16→8→4            |
| CELEBA  | 64×64      | **4→8→16→32→64** | **64→32→16→8→4**     |

그래서 각 네트워크에 **ResBlockUp 1개, ResBlockDown 1개 추가**하면 됩니다.

---

# ⭐ 수정된 64×64 CELEBA 버전 코드

아래는 **GAN-N 구조 유지 + 64×64 확장**된 최종 버전입니다.

---

# ⭐ PixelNorm (변경 없음)

```python
from torch.nn.utils import spectral_norm

class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
```

---

# ⭐ ResBlockUp (변경 없음)

```python
class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        skip = self.skip(self.upsample(x))
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out + skip
```

---

# ⭐ ResBlockDown (변경 없음)

```python
class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 2, 0))

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        skip = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.down(out)
        return out + skip
```

---

# ⭐ ★ CELEBA-64용 Generator (확장 버전)

💡 **추가된 부분: res4 = 32→64 업샘플링**

```python
class ResGenerator64(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim

        # z: (B, latent_dim, 1, 1) → (B, base*4, 4, 4)
        self.fc = nn.ConvTranspose2d(latent_dim, base * 4, 4, 1, 0)

        # 4→8→16→32→64
        self.res1 = ResBlockUp(base * 4, base * 2)   # 4→8
        self.res2 = ResBlockUp(base * 2, base)       # 8→16
        self.res3 = ResBlockUp(base, base // 1)      # 16→32 (same channel)
        self.res4 = ResBlockUp(base // 1, base // 2) # 32→64

        self.out_conv = nn.Conv2d(base // 2, out_channels, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.out_conv(x)
        return torch.tanh(x)
```

---

# ⭐ ★ CELEBA-64용 Discriminator (확장 버전)

💡 **추가된 부분: res4 = 64→32 다운샘플링**

```python
class ResDiscriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()

        # 64→32→16→8→4
        self.res1 = ResBlockDown(in_channels, base // 2)   # 64→32
        self.res2 = ResBlockDown(base // 2, base)          # 32→16
        self.res3 = ResBlockDown(base, base * 2)           # 16→8
        self.res4 = ResBlockDown(base * 2, base * 4)       # 8→4

        # global sum pooling 후 linear
        self.linear = spectral_norm(nn.Linear(base * 4, 1))

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = nn.LeakyReLU(0.2)(x)
        x = x.sum(dim=[2, 3])  # (B, C)

        return self.linear(x)
```

---

# ⭐ 한눈에 요약

### CIFAR10 (32×32)

```
G: z→4→8→16→32
D: 32→16→8→4
```

### CELEBA (64×64) 확장

```
G: z→4→8→16→32→64   (ResBlockUp 4개)
D: 64→32→16→8→4     (ResBlockDown 4개)
```

### 구조는 유지하면서 블록만 1개씩 추가하며 자연스럽게 확장.

---

# ⭐ 다음 작업도 가능합니다

원하시는 다음 항목을 말씀해 주세요:

### (1) **train_celeba64_highresgan.py**

* dataloader 포함
* FID/IS 자동 계산
* 학습/샘플링 스크립트 완성

### (2) DiffAug + HighResGAN (GAN-L + GAN-N 통합)

### (3) Multi-resolution (32/64/128/256) scalable GAN 구조 설계

### (4) StyleGAN2-lite 업그레이드 버전

어떤 걸 진행할까요?
