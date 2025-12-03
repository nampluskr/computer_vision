import numpy as np

import torch
from torchvision import transforms as T

from datasets import CIFAR10, CelebA
from datasets import get_train_loader, get_test_loader
from trainer import train_gan
from utils import set_seed
from gan import GAN, InfoGAN
from gan import Generator32, Discriminator32
from gan import Generator64, Discriminator64
from gan import InfoGenerator64, InfoDiscriminator64


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

    if 1:   # (64, 64)
        OUTPUT_DIR = "./outputs_infogan_celeba"
        FILENAME = f"celeba_infogan"
        NUM_SAMPLES = 8
        NUM_EPOCHS = 2
        TOTAL_EPOCHS = 10

        set_seed(SEED)
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        dataset = CelebA(root_dir="/home/namu/myspace/NAMU/datasets/celeba", split="train",
                         attributes=["Male", "Smiling"], transform=transform)

        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
        noises = np.random.normal(size=(NUM_SAMPLES, LATENT_DIM, 1, 1))
        codes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * (NUM_SAMPLES // 4)).reshape(-1, 2, 1, 1)
        zc = np.concatenate([noises, codes], axis=1)

        discriminator = InfoDiscriminator64(in_channels=IN_CHANNELS, base=BASE)
        generator = InfoGenerator64(z_dim=LATENT_DIM, c_dim=2, out_channels=OUT_CHANNELS, base=BASE)
        gan = InfoGAN(discriminator, generator, latent_dim=100, c_dim_discrete=2)
        history = train_gan(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS,
                            noises=zc, output_dir=OUTPUT_DIR, filename=FILENAME)
