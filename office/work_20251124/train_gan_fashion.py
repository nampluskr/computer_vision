import numpy as np

import torch
from torchvision import transforms as T

from datasets import FashionMNIST, FashionMNIST, FashionMNIST
from datasets import get_train_loader, get_test_loader
from trainer import train_gan
from utils import set_seed
from gan import GAN, CGAN, ACGAN
from gan import Generator32, CGenerator32
from gan import Discriminator32, CDiscriminator32, ACDiscriminator32


SEED = 42
BATCH_SIZE = 128
NUM_SAMPLES = 100

LATENT_DIM = 100
IN_CHANNELS = 1
OUT_CHANNELS = 1
BASE = 64

NUM_EPOCHS = 5
TOTAL_EPOCHS = 20


if __name__ == "__main__":

    if 1:
        LOSS_TYPE = "hinge"
        OUTPUT_DIR = "./outputs_gan_fashion"
        FILENAME = f"fashion_gan-{LOSS_TYPE}"

        set_seed(SEED)
        dataset = FashionMNIST(root_dir="/home/namu/myspace/NAMU/datasets/fashion_mnist", split="train",
                        transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]))
        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
        noises = np.random.normal(size=(NUM_SAMPLES, LATENT_DIM, 1, 1))

        discriminator = Discriminator32(in_channels=IN_CHANNELS, base=BASE)
        generator = Generator32(latent_dim=LATENT_DIM, out_channels=OUT_CHANNELS, base=BASE)
        gan = GAN(discriminator, generator, loss_type=LOSS_TYPE)
        history = train_gan(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS, 
                            noises=noises, output_dir=OUTPUT_DIR, filename=FILENAME)

    if 1:
        NUM_CLASSES = 10
        LOSS_TYPE = "hinge"
        OUTPUT_DIR = "./outputs_cgan_fashion"
        FILENAME = f"fashion_cgan-{LOSS_TYPE}"

        set_seed(SEED)
        dataset = FashionMNIST(root_dir="/home/namu/myspace/NAMU/datasets/fashion_mnist", split="train",
                        transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]))
        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
        noises = np.random.normal(size=(NUM_SAMPLES, LATENT_DIM, 1, 1))
        labels = np.tile(np.arange(NUM_CLASSES), 10)

        discriminator = CDiscriminator32(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, base=BASE)
        generator = CGenerator32(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, out_channels=OUT_CHANNELS, base=BASE)
        gan = CGAN(discriminator, generator, loss_type=LOSS_TYPE)
        history = train_gan(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS, 
                            noises=noises, labels=labels, output_dir=OUTPUT_DIR, filename=FILENAME)

    if 1:
        NUM_CLASSES = 10
        LOSS_TYPE = "hinge"
        OUTPUT_DIR = "./outputs_acgan_fashion"
        FILENAME = f"fashion_acgan-{LOSS_TYPE}"

        set_seed(SEED)
        dataset = FashionMNIST(root_dir="/home/namu/myspace/NAMU/datasets/fashion_mnist", split="train",
                        transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]))
        train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
        noises = np.random.normal(size=(NUM_SAMPLES, LATENT_DIM, 1, 1))
        labels = np.tile(np.arange(NUM_CLASSES), 10)

        discriminator = ACDiscriminator32(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, base=BASE)
        generator = CGenerator32(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, out_channels=OUT_CHANNELS, base=BASE)
        gan = ACGAN(discriminator, generator, loss_type=LOSS_TYPE)
        history = train_gan(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS, 
                            noises=noises, labels=labels, output_dir=OUTPUT_DIR, filename=FILENAME)
