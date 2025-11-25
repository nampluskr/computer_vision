import torch
from torchvision import transforms as T

from datasets import FashionMNIST, MNIST, get_train_loader, get_test_loader
from classifier import Encoder32, MulticlassClassifier, BinaryClassifier
from trainer import fit
from utils import set_seed


SEED = 42
BATCH_SIZE = 128
BASE = 32
NUM_EPOCHS = 5


if __name__ == "__main__":

    if 1:
        print(f"\n>> Multiclass Classification (CrossEntropyLoss)")
        NUM_CLASSES = 10
        IN_CHANNELS = 1

        set_seed(SEED)
        train_loader = get_train_loader(
            dataset=FashionMNIST(root_dir="/home/namu/myspace/NAMU/datasets/fashion_mnist",
                split="train", transform=T.Compose([T.ToTensor()])),
            batch_size=BATCH_SIZE)
        test_loader = get_test_loader(
            dataset=FashionMNIST(root_dir="/home/namu/myspace/NAMU/datasets/fashion_mnist",
                split="test", transform=T.Compose([T.ToTensor()])),
            batch_size=BATCH_SIZE)

        encoder = Encoder32(in_channels=IN_CHANNELS, latent_dim=NUM_CLASSES, base=BASE)
        clf = MulticlassClassifier(encoder)
        history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

    if 1:
        print(f"\n>> Binary Classification (CrossEntropyLoss)")
        NUM_CLASSES = 2
        IN_CHANNELS = 1

        set_seed(SEED)
        train_loader = get_train_loader(
            dataset=MNIST(root_dir="/home/namu/myspace/NAMU/datasets/mnist",
                split="train", transform=T.Compose([T.ToTensor()]), binary=True),
            batch_size=BATCH_SIZE)
        test_loader = get_test_loader(
            dataset=MNIST(root_dir="/home/namu/myspace/NAMU/datasets/mnist",
                split="test", transform=T.Compose([T.ToTensor()]), binary=True),
            batch_size=BATCH_SIZE)

        encoder = Encoder32(in_channels=IN_CHANNELS, latent_dim=NUM_CLASSES, base=BASE)
        clf = MulticlassClassifier(encoder)
        history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

    if 1:
        print(f"\n>> Binary Classification (BCEWithLogitsLoss)")
        NUM_CLASSES = 1
        IN_CHANNELS = 1

        set_seed(SEED)
        train_loader = get_train_loader(
            dataset=MNIST(root_dir="/home/namu/myspace/NAMU/datasets/mnist",
                split="train", transform=T.Compose([T.ToTensor()]), binary=True),
            batch_size=BATCH_SIZE)
        test_loader = get_test_loader(
            dataset=MNIST(root_dir="/home/namu/myspace/NAMU/datasets/mnist",
                split="test", transform=T.Compose([T.ToTensor()]), binary=True),
            batch_size=BATCH_SIZE)

        encoder = Encoder32(in_channels=IN_CHANNELS, latent_dim=NUM_CLASSES, base=BASE)
        clf = BinaryClassifier(encoder)
        history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

