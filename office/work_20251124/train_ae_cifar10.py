import torch
from torchvision import transforms as T

from datasets import CIFAR10, get_train_loader, get_test_loader
from autoencoder import Encoder32, Decoder32, AutoEncoder
from autoencoder import VEncoder32, VAE
from trainer import fit
from utils import set_seed


SEED = 42
BATCH_SIZE = 64
LATENT_DIM = 128
BASE = 64
NUM_EPOCHS = 10


if __name__ == "__main__":

    if 1:
        print(f"\n>> Autoencoder")
        set_seed(SEED)

        root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
        train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=BATCH_SIZE)
        test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=BATCH_SIZE)

        encoder = Encoder32(in_channels=3, latent_dim=LATENT_DIM)
        decoder = Decoder32(out_channels=3, latent_dim=LATENT_DIM)
        ae = AutoEncoder(encoder, decoder)
        history = fit(ae, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

    if 1:
        print(f"\n>> Variational Autoencoder")
        set_seed(SEED)

        root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
        train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=BATCH_SIZE)
        test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=BATCH_SIZE)

        encoder = VEncoder32(in_channels=3, latent_dim=LATENT_DIM)
        decoder = Decoder32(out_channels=3, latent_dim=LATENT_DIM)
        vae = VAE(encoder, decoder, beta=0.0001)
        history = fit(vae, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
