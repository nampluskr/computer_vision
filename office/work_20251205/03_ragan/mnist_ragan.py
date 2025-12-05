import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.join(os.path.dirname(current_dir), 'common')
model_dir = os.path.join(os.path.dirname(current_dir), 'models')

for path in [common_dir, model_dir]:
    if path not in sys.path:
        sys.path.append(path)

#####################################################################
import torch
import torchvision.transforms as T


from datasets import MNIST, get_train_loader
from utils import set_seed, create_images, sample_latent, update_history, plot_images
from generator import Generator32
from discriminator import Discriminator32
from ragan import RaGAN
from trainer import fit


if __name__ == "__main__":

    SEED = 42
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/mnist"
    BATCH_SIZE = 128

    set_seed(SEED)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
    dataset = MNIST(root_dir=DATA_DIR, split="train", transform=transform)
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)

    LATENT_DIM = 100
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BASE = 64

    generator = Generator32(latent_dim=LATENT_DIM, out_channels=OUT_CHANNELS, base=BASE)
    discriminator = Discriminator32(in_channels=IN_CHANNELS, base=BASE)
    gan = RaGAN(discriminator, generator, loss_type="ra")

    NUM_EPOCHS = 2
    TOTAL_EPOCHS = 10
    NUM_SAMPLES = 100

    FILENAME = os.path.splitext(os.path.basename(__file__))[0]
    OUTPUT_DIR = f"./outputs_{FILENAME}"
    IMAGE_NAME = FILENAME + ""

    noises = sample_latent(NUM_SAMPLES, LATENT_DIM)
    history = {}
    epoch = 0
    for _ in range(TOTAL_EPOCHS // NUM_EPOCHS):
        epoch_history = fit(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS)
        update_history(history, epoch_history)

        images = create_images(gan.generator, noises)
        epoch += NUM_EPOCHS
        image_path = os.path.join(OUTPUT_DIR, f"{IMAGE_NAME}_epoch{epoch:03d}.png")
        plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)



