
다음은 **InfoGAN**을 PyTorch로 구현한 예제 코드입니다.  
CelebA 데이터셋과 함께 사용할 수 있으며, **연속적인 잠재 코드(latent code)** 와 **이산적인 잠재 코드**를 학습하여 의미 있는 특성(예: 회전, 웃음 등)을 제어할 수 있도록 설계되었습니다.

---

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

# CelebA Dataset (이전에 정의한 클래스 사용)
# from celeba_dataset import CelebA  # 위에서 정의한 CelebA 클래스 사용

# 하이퍼파라미터
BATCH_SIZE = 64
IMAGE_SIZE = 64
NZ = 64  # 노이즈 벡터 크기
NC = 3  # 채널 수 (RGB)
NGF = 64  # Generator 필터 수
NDF = 64  # Discriminator 필터 수
N_CONT = 2  # 연속적 잠재 코드 (예: 각도, 스마일 정도 등)
N_DISC = 10  # 이산적 잠재 코드 (예: 10가지 얼굴 특성)
LR = 1e-4
BETA1 = 0.5
EPOCHS = 10
LOG_INTERVAL = 100

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 변환
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CelebA 데이터셋 로드 (root_dir 경로 수정 필요)
dataset = CelebA(root_dir="./data/celeba", split="train", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Generator 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nz = NZ + N_CONT + N_DISC
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 2, NC, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(input.size(0), input.size(1), 1, 1))

# Discriminator 및 Q 네트워크 (InfoGAN용)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d_layer = nn.Linear(NDF * 8 * 4 * 4, 1)
        self.q_layer = nn.Linear(NDF * 8 * 4 * 4, N_CONT + N_DISC)
        self.q_disc = nn.Linear(N_CONT + N_DISC, N_DISC)  # 이산 코드 예측
        self.q_cont = nn.Linear(N_CONT + N_DISC, N_CONT)  # 연속 코드 예측

    def forward(self, input):
        x = self.main(input).view(input.size(0), -1)
        d_out = torch.sigmoid(self.d_layer(x))
        q_out = self.q_layer(x)
        disc_logits = self.q_disc(q_out)
        cont_mu = self.q_cont(q_out)
        cont_var = torch.ones_like(cont_mu)  # 간단화: 분산 고정
        return d_out, disc_logits, cont_mu, cont_var

# 모델 초기화
netG = Generator().to(device)
netD = Discriminator().to(device)

# 손실 함수
criterion_d = nn.BCELoss()
criterion_q_disc = nn.CrossEntropyLoss()
criterion_q_cont = nn.MSELoss()

# 최적화
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

# 고정된 테스트 노이즈 (시각화용)
fixed_noise = torch.randn(64, NZ, device=device)
fixed_cont = torch.randn(64, N_CONT, device=device)
fixed_disc = torch.LongTensor([i % N_DISC for i in range(64)]).to(device)
fixed_input = torch.cat([fixed_noise, fixed_cont, nn.functional.one_hot(fixed_disc, N_DISC).float()], 1)

# 학습 루프
def train():
    for epoch in range(EPOCHS):
        for i, data in enumerate(tqdm(dataloader)):
            real_images = data['image'].to(device)
            b_size = real_images.size(0)

            # 레이블
            real_label = torch.ones(b_size, 1, device=device)
            fake_label = torch.zeros(b_size, 1, device=device)

            # ----------------------------
            # 판별자(D) 업데이트
            # ----------------------------
            netD.zero_grad()
            # 진짜 이미지
            output, _, _, _ = netD(real_images)
            loss_d_real = criterion_d(output, real_label)
            loss_d_real.backward()

            # 가짜 이미지
            noise = torch.randn(b_size, NZ, device=device)
            cont_code = torch.randn(b_size, N_CONT, device=device)
            disc_code = torch.randint(0, N_DISC, (b_size,), device=device)
            one_hot_disc = nn.functional.one_hot(disc_code, N_DISC).float()
            fake_input = torch.cat([noise, cont_code, one_hot_disc], 1)
            fake_images = netG(fake_input.detach())
            output, _, _, _ = netD(fake_images.detach())
            loss_d_fake = criterion_d(output, fake_label)
            loss_d_fake.backward()
            optimizerD.step()

            # ----------------------------
            # 생성자(G) 및 Q 업데이트
            # ----------------------------
            netG.zero_grad()
            output, q_disc_logits, q_cont_mu, q_cont_var = netD(fake_images)
            loss_g = criterion_d(output, real_label)

            # Q 손실 (이산 + 연속)
            loss_q_disc = criterion_q_disc(q_disc_logits, disc_code)
            loss_q_cont = criterion_q_cont(q_cont_mu, cont_code)
            loss_q = loss_q_disc + loss_q_cont

            loss_g_total = loss_g + loss_q
            loss_g_total.backward()
            optimizerG.step()

            if i % LOG_INTERVAL == 0:
                print(f"[{epoch}/{EPOCHS}] [{i}/{len(dataloader)}] "
                      f"Loss_D: {(loss_d_real + loss_d_fake):.4f}, "
                      f"Loss_G: {loss_g:.4f}, "
                      f"Loss_Q: {loss_q:.4f}")

        # 시각화
        with torch.no_grad():
            fake = netG(fixed_input).cpu()
            grid = make_grid(fake, nrow=8, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            plt.title(f"Generated Faces - Epoch {epoch}")
            plt.show()

train()
```

---

### 🔍 InfoGAN 핵심 요소
- **G(z, c)**: `z`는 순수 노이즈, `c`는 의미 있는 잠재 코드 (연속 + 이산)
- **Q 네트워크**: D의 출력을 이용해 `c`를 재구성 (정보 손실 최소화)
- **Mutual Information 최대화**: `c`와 `Q(c|x)` 간 정보 보존

---

### ✅ 사용 팁
- `N_CONT`: 연속적 특성 (예: 미소 정도, 머리 기울기)
- `N_DISC`: 이산적 특성 (예: 성별, 안경 착용 여부)
- 학습 후 `c`를 조작해 이미지 특성 제어 가능

---

> 💡 **참고**: InfoGAN은 GAN의 mode collapse 문제 완화와 더불어 **해석 가능한 생성**을 가능하게 합니다.  
> CelebA로 학습 시, `c`를 변화시켜 "웃는 정도", "성별", "머리 스타일" 등을 조절할 수 있습니다.



다음은 InfoGAN 학습 후, **잠재 코드 `c`를 조작하여 생성 이미지의 특성(예: 미소, 성별 등)을 제어하는 예제**입니다.

---

### ✅ 조건별 이미지 생성 예제 (학습 후)

```python
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def interpolate_code(netG, fixed_noise, cont_code_start, cont_code_end, disc_code, num_steps=8):
    """
    연속 코드를 점진적으로 변화시켜 이미지 생성 (예: 미소 정도 조절)
    """
    with torch.no_grad():
        vectors = []
        for alpha in torch.linspace(0, 1, num_steps):
            cont_code = alpha * cont_code_end + (1 - alpha) * cont_code_start
            one_hot_disc = torch.nn.functional.one_hot(
                torch.tensor([disc_code] * num_steps), N_DISC
            ).float().to(fixed_noise.device)
            noise = fixed_noise[:num_steps]
            input_vec = torch.cat([noise, cont_code, one_hot_disc], dim=1)
            img = netG(input_vec)
            vectors.append(img.cpu())
        return torch.cat(vectors, 0)

# ----------------------------------------
# 설정
# ----------------------------------------
netG.eval()  # 평가 모드
device = next(netG.parameters()).device
num_steps = 10
fixed_noise = torch.randn(num_steps, NZ, device=device)

# 이산 코드: 예를 들어 "남성(1)" 또는 "여성(0)"
DISC_CODE = 1  # 예: 남성

# 연속 코드 조작 예시 1: 미소 정도 (가정: c[0]이 'Smiling'에 해당)
cont_start = torch.randn(1, N_CONT, device=device)
cont_start[:, 0] = -2.0  # '안 웃는다'
cont_end = cont_start.clone()
cont_end[:, 0] = 2.0     # '크게 웃는다'

# 연속 코드 조작 예시 2: 성별 느낌 조절 (가정: c[1]이 'Male/Female' 느낌)
cont_start2 = torch.randn(1, N_CONT, device=device)
cont_start2[:, 1] = -2.0  # 여성스러운 특성
cont_end2 = cont_start2.clone()
cont_end2[:, 1] = 2.0     # 남성스러운 특성

# ----------------------------------------
# 1. 연속 코드 변화 → 미소 정도 변화
# ----------------------------------------
print("🔄 연속 코드 조작: '미소 정도' 변화")
images_smile = interpolate_code(
    netG, fixed_noise,
    cont_start.expand(num_steps, -1),
    cont_end.expand(num_steps, -1),
    disc_code=DISC_CODE,
    num_steps=num_steps
)

grid_smile = make_grid(images_smile, nrow=num_steps, normalize=True)
plt.figure(figsize=(12, 3))
plt.imshow(grid_smile.permute(1, 2, 0))
plt.axis("off")
plt.title("👉 연속 코드 조작: 미소 정도 증가 (좌: 안 웃음 → 우: 크게 웃음)")
plt.show()

# ----------------------------------------
# 2. 연속 코드 변화 → 성별 느낌 조절
# ----------------------------------------
print("🔄 연속 코드 조작: '성별 느낌' 변화")
images_gender = interpolate_code(
    netG, fixed_noise,
    cont_start2.expand(num_steps, -1),
    cont_end2.expand(num_steps, -1),
    disc_code=DISC_CODE,
    num_steps=num_steps
)

grid_gender = make_grid(images_gender, nrow=num_steps, normalize=True)
plt.figure(figsize=(12, 3))
plt.imshow(grid_gender.permute(1, 2, 0))
plt.axis("off")
plt.title("👉 연속 코드 조작: 여성 → 남성 느낌 (DISC 코드 고정)")
plt.show()

# ----------------------------------------
# 3. 이산 코드 변화 → 다른 속성 (예: 안경 착용)
# ----------------------------------------
print("🔄 이산 코드 변화: 다른 범주적 특성 비교")
with torch.no_grad():
    rows = []
    for d in range(min(N_DISC, 8)):  # 처음 8개 클래스만 시각화
        one_hot = torch.nn.functional.one_hot(
            torch.tensor([d] * 8), N_DISC
        ).float().to(device)
        noise = torch.randn(8, NZ, device=device)
        cont = torch.randn(8, N_CONT, device=device)
        input_vec = torch.cat([noise, cont, one_hot], dim=1)
        row = netG(input_vec).cpu()
        rows.append(row)
    all_images = torch.cat(rows, 0)
    grid_disc = make_grid(all_images, nrow=8, normalize=True)

plt.figure(figsize=(12, 12))
plt.imshow(grid_disc.permute(1, 2, 0))
plt.axis("off")
plt.title("👉 이산 코드 변화: 8가지 다른 범주적 특성")
plt.show()
```

---

### 🔍 해석
- **연속 코드(`c_cont`)**: 점진적 변화 → `표정`, `기울기`, `조명` 등 **아날로그 특성** 제어
- **이산 코드(`c_disc`)**: 범주 전환 → `성별`, `안경`, `머리스타일` 등 **디지털 특성** 제어

---

### 💡 팁
- 실제 어떤 코드가 어떤 특성에 대응하는지는 **학습 후 탐색** 필요 (t-SNE, 조작 실험 등)
- `cont_code`를 `-3 → +3` 범위로 변화시키면 의미 있는 변화가 잘 나타남
- 고정된 `noise`와 `disc_code`를 사용하면 비교 실험이 더 명확

---

> ✅ **결론**: InfoGAN은 단순 생성이 아닌, **의미 있는 조작 가능 생성**(controllable generation)을 가능하게 합니다.
