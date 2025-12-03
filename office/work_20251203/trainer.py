## filename: trainer.py

import os
import sys
from tqdm import tqdm
from utils import update_history, create_images, plot_images


def train_gan(gan, train_loader, num_epochs, total_epochs, noises, labels=None, 
              output_dir=None, filename=None):
    history = {}
    if filename is None:
        filename = os.path.splitext(os.path.basename(__file__))[0]

    epoch = 0
    for _ in range(total_epochs // num_epochs):
        epoch += num_epochs
        epoch_history = fit(gan, train_loader, num_epochs=num_epochs)
        update_history(history, epoch_history)

        images = create_images(gan.generator, noises, labels=labels)

        if output_dir is not None:
            image_path = os.path.join(output_dir, f"{filename}_epoch{epoch:03d}.png")
            plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=image_path)
        else:
            plot_images(*images, ncols=10, xunit=1, yunit=1, save_path=None)
    return history


def train(model, dataloader):
    model.train()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Train", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            batch_size = batch["image"].shape[0]
            total += batch_size

            outputs = model.train_step(batch)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def evaluate(model, dataloader):
    model.eval()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Evaluate", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            batch_size = batch["image"].shape[0]
            total += batch_size

            outputs = model.eval_step(batch)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def fit(model, train_loader, num_epochs, valid_loader=None):
    history = {"train": {}, "valid": {}}
    for epoch in range(1, num_epochs + 1):
        epoch_info = f"[{epoch:3d}/{num_epochs}]"
        train_results = train(model, train_loader)
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in train_results.items()])

        for name, value in train_results.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)
            valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in valid_results.items()])

            for name, value in valid_results.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)
            print(f"{epoch_info} {train_info} | (val) {valid_info}")
        else:
            print(f"{epoch_info} {train_info}")

    return history
