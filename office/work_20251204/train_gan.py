import os
import numpy as np

import torch
from torchvision import transforms as T

from datasets import CIFAR10, CelebA
from datasets import get_train_loader, get_test_loader
from trainer import train_gan, fit
from utils import set_seed, plot_images
from gan_v2 import GAN, InfoGAN
from gan_v2 import Generator32, Discriminator32
from gan_v2 import Generator64, Discriminator64
from gan_v2 import InfoGenerator64, InfoDiscriminator64


SEED = 42
BATCH_SIZE = 128
NUM_SAMPLES = 100

LATENT_DIM = 100
IN_CHANNELS = 3
OUT_CHANNELS = 3
BASE = 64

NUM_EPOCHS = 5
TOTAL_EPOCHS = 20


if __name__ == "__main__":

    if 0:   # (32, 32, 3)
        OUTPUT_DIR = "./outputs_gan_cifar10"
        FILENAME = f"cifar10_gan"

        set_seed(SEED)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        dataset = CIFAR10(root_dir="/home/namu/myspace/NAMU/datasets/cifar10", split="train",
                          transform=transform)
        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
        noises = np.random.normal(size=(NUM_SAMPLES, LATENT_DIM, 1, 1))

        discriminator = Discriminator32(in_channels=IN_CHANNELS, base=BASE)
        generator = Generator32(latent_dim=LATENT_DIM, out_channels=OUT_CHANNELS, base=BASE)
        gan = GAN(discriminator, generator)
        history = train_gan(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS,
                            noises=noises, output_dir=OUTPUT_DIR, filename=FILENAME)

    if 0:   # (64, 64, 3)
        OUTPUT_DIR = "./outputs_gan_cifar10"
        FILENAME = f"cifar10_gan-64"

        set_seed(SEED)
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        dataset = CIFAR10(root_dir="/home/namu/myspace/NAMU/datasets/cifar10", split="train",
                          transform=transform)
        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
        noises = np.random.normal(size=(NUM_SAMPLES, LATENT_DIM, 1, 1))

        discriminator = Discriminator64(in_channels=IN_CHANNELS, base=BASE)
        generator = Generator64(latent_dim=LATENT_DIM, out_channels=OUT_CHANNELS, base=BASE)
        gan = GAN(discriminator, generator)
        history = train_gan(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS,
                            noises=noises, output_dir=OUTPUT_DIR, filename=FILENAME)

    if 0:   # (64, 64)
        OUTPUT_DIR = "./outputs_gan_celeba"
        FILENAME = f"celeba_gan"

        set_seed(SEED)
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        dataset = CelebA(root_dir="/home/namu/myspace/NAMU/datasets/celeba", split="train",
                         attributes=["Male", "Smiling"], transform=transform)
        dataset = dataset.filter(Smiling=0)

        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
        noises = np.random.normal(size=(NUM_SAMPLES, LATENT_DIM, 1, 1))

        discriminator = Discriminator64(in_channels=IN_CHANNELS, base=BASE)
        generator = Generator64(latent_dim=LATENT_DIM, out_channels=OUT_CHANNELS, base=BASE)
        gan = GAN(discriminator, generator)
        history = train_gan(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS,
                            noises=noises, output_dir=OUTPUT_DIR, filename=FILENAME)

    if 0:   # (64, 64)
        OUTPUT_DIR = "./outputs_infogan_celeba_hinge"
        FILENAME = "celeba_infogan"
        NUM_SAMPLES = 8          # 보간 이미지 샘플 수 (행 수)
        NUM_EPOCHS = 1
        TOTAL_EPOCHS = 5

        LATENT_DIM = 100
        NUM_DISCRETE = 2         # Male/Female (one-hot)
        NUM_CONTINUOUS = 1       # Smiling intensity (연속 코드)

        BATCH_SIZE = 128
        IN_CHANNELS = 3
        OUT_CHANNELS = 3
        BASE = 64

        SEED = 42
        set_seed(SEED)

        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        dataset = CelebA(
            root_dir="/home/namu/myspace/NAMU/datasets/celeba",
            split="train",
            attributes=["Male", "Smiling"],  # label: [Male, Smiling]
            transform=transform
        )
        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)

        z = np.random.randn(NUM_SAMPLES, LATENT_DIM, 1, 1).astype(np.float32)

        # c_discrete: 성별 클래스 (0: Male, 1: Female)
        # one-hot: [1,0] = Male, [0,1] = Female
        gender_cycle = [0, 1]  # [Male, Female]
        gender_indices = [gender_cycle[i % 2] for i in range(NUM_SAMPLES)]

        discrete_codes = []
        for idx in gender_indices:
            if idx == 0:
                discrete_codes.append([1, 0])  # Male
            else:
                discrete_codes.append([0, 1])  # Female

        fixed_disc = np.array(discrete_codes, dtype=np.float32)[:, :, np.newaxis, np.newaxis]

        discriminator = InfoDiscriminator64(
            num_continuous=NUM_CONTINUOUS,
            num_discrete=NUM_DISCRETE,
            in_channels=IN_CHANNELS,
            base=BASE
        )

        generator = InfoGenerator64(
            latent_dim=LATENT_DIM,
            num_continuous=NUM_CONTINUOUS,
            num_discrete=NUM_DISCRETE,
            out_channels=OUT_CHANNELS,
            base=BASE
        )

        gan = InfoGAN(
            discriminator=discriminator,
            generator=generator,
            latent_dim=LATENT_DIM,
            num_continuous=NUM_CONTINUOUS,
            num_discrete=NUM_DISCRETE,
            lambda_continuous=1.0,
            lambda_discrete=1.0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        @torch.no_grad()
        def generate_interpolation(gan, z, fixed_disc, n_steps=10):
            generator = gan.generator
            device = next(generator.parameters()).device
            generator.eval()

            images = []
            z_tensor = torch.from_numpy(z).float().to(device)           # (N, 100, 1, 1)
            c_disc_tensor = torch.from_numpy(fixed_disc).float().to(device)  # (N, 2, 1, 1)

            for i in range(z.shape[0]):
                z_i = z_tensor[i:i+1]      # (1, 100, 1, 1)
                c_disc_i = c_disc_tensor[i:i+1]  # (1, 2, 1, 1)

                for alpha in np.linspace(-1, 1, n_steps):  # Smiling intensity
                    c_cont_i = torch.full((1, NUM_CONTINUOUS, 1, 1), alpha, device=device)
                    zc = torch.cat([z_i, c_cont_i, c_disc_i], dim=1)  # (1, 103, 1, 1)
                    img = generator(zc).cpu()
                    img = (img[0].permute(1, 2, 0).numpy() + 1) / 2  # [-1,1] → [0,1]
                    images.append(img)

            generator.train()
            return np.array(images)

        # --------------------------------------------------------------
        # 학습 루프
        # --------------------------------------------------------------

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        history = {}

        for epoch in range(0, TOTAL_EPOCHS, NUM_EPOCHS):
            epoch_history = fit(gan, train_loader, num_epochs=NUM_EPOCHS)
            for k, v in epoch_history["train"].items():
                history.setdefault(k, []).extend(v)

            interpolated_images = generate_interpolation(gan, z=z, fixed_disc=fixed_disc, n_steps=10)
            image_path = os.path.join(OUTPUT_DIR, f"{FILENAME}_epoch{epoch + NUM_EPOCHS:03d}.png")
            plot_images(*interpolated_images, ncols=10, xunit=1.5, yunit=1.5, save_path=image_path)

            print(f"Epoch [{epoch + NUM_EPOCHS}/{TOTAL_EPOCHS}] interpolation saved.")

    if 1:
        OUTPUT_DIR = "./outputs_infogan_celeba_cont2d"
        FILENAME = "celeba_infogan_2d"
        NUM_SAMPLES = 1          # 고정된 스타일 1개만 사용 (10x10 생성용)
        NUM_EPOCHS = 1
        TOTAL_EPOCHS = 10

        LATENT_DIM = 100
        NUM_DISCRETE = 2         # Male/Female
        NUM_CONTINUOUS = 2       # 예: [Smiling intensity, Face angle]

        BATCH_SIZE = 128
        IN_CHANNELS = 3
        OUT_CHANNELS = 3
        BASE = 64

        SEED = 42
        set_seed(SEED)

        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        dataset = CelebA(
            root_dir="/home/namu/myspace/NAMU/datasets/celeba",
            split="train",
            attributes=["Male", "Smiling"],
            transform=transform
        )

        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)

        # 고정된 z (하나의 잠재 스타일)
        z = np.random.randn(NUM_SAMPLES, LATENT_DIM, 1, 1).astype(np.float32)

        # 이산 코드 고정 (예: Female)
        fixed_gender = 1  # 0: Male, 1: Female
        c_disc = np.zeros((NUM_SAMPLES, NUM_DISCRETE), dtype=np.float32)
        c_disc[0, fixed_gender] = 1.0
        fixed_disc = c_disc[:, :, np.newaxis, np.newaxis]  # (1, 2, 1, 1)

        discriminator = InfoDiscriminator64(
            num_continuous=NUM_CONTINUOUS,
            num_discrete=NUM_DISCRETE,
            in_channels=IN_CHANNELS,
            base=BASE
        )

        generator = InfoGenerator64(
            latent_dim=LATENT_DIM,
            num_continuous=NUM_CONTINUOUS,
            num_discrete=NUM_DISCRETE,
            out_channels=OUT_CHANNELS,
            base=BASE
        )

        gan = InfoGAN(
            discriminator=discriminator,
            generator=generator,
            latent_dim=LATENT_DIM,
            num_continuous=NUM_CONTINUOUS,
            num_discrete=NUM_DISCRETE,
            lambda_continuous=1.0,
            lambda_discrete=1.0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        @torch.no_grad()
        def generate_2d_interpolation_grid(gan, z, fixed_disc, n_steps=10, cont_range=(-1, 1)):
            generator = gan.generator
            device = next(generator.parameters()).device
            generator.eval()

            z_tensor = torch.from_numpy(z).float().to(device)  # (1, 100, 1, 1)
            c_disc_tensor = torch.from_numpy(fixed_disc).float().to(device)  # (1, 2, 1, 1)

            # ✅ float32 보장
            values = np.linspace(cont_range[0], cont_range[1], n_steps).astype(np.float32)

            images = []
            for c1 in values:
                row = []
                for c2 in values:
                    # ✅ dtype=torch.float32 강제
                    c_cont = torch.tensor([[[[c1]], [[c2]]]], dtype=torch.float32, device=device)
                    zc = torch.cat([z_tensor, c_cont, c_disc_tensor], dim=1)
                    img = generator(zc).cpu()
                    img = (img[0].permute(1, 2, 0).numpy() + 1) / 2
                    row.append(img)
                images.extend(row)

            generator.train()
            return np.array(images)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        history = {}

        for epoch in range(0, TOTAL_EPOCHS, NUM_EPOCHS):
            # 학습
            epoch_history = fit(gan, train_loader, num_epochs=NUM_EPOCHS)
            for k, v in epoch_history["train"].items():
                history.setdefault(k, []).extend(v)

            # 10x10 그리드 생성
            grid_images = generate_2d_interpolation_grid(
                gan=gan, z=z, fixed_disc=fixed_disc, n_steps=10)

            # 저장
            image_path = os.path.join(OUTPUT_DIR, f"{FILENAME}_epoch{epoch + NUM_EPOCHS:03d}.png")
            plot_images(*grid_images, ncols=10, xunit=1.5, yunit=1.5, 
                save_path=image_path,
                # title=f"Epoch {epoch + NUM_EPOCHS}: c1=Smile, c2=Angle"
            )

            print(f"Epoch [{epoch + NUM_EPOCHS}/{TOTAL_EPOCHS}] 10x10 interpolation saved.")
