import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def sample_latent(num_samples, latent_dim):
    # noises = torch.randn(num_samples, latent_dim)
    noises = np.random.normal(size=(num_samples, latent_dim))
    return noises


@torch.no_grad()
def create_images(generator, noises, labels=None):
    training = generator.training
    generator.eval()

    device = next(generator.parameters()).device
    noises = torch.tensor(noises).float().to(device)

    if labels is not None:
        labels = torch.tensor(labels).long().to(device)
        outputs = generator(noises, labels)
    else:
        outputs = generator(noises)

    images = (outputs + 1) / 2  # [-1,1] → [0,1]
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    if training:
        generator.train()
    return images


def plot_images(*images, ncols=5, xunit=3, yunit=3, cmap='gray',
                titles=[], vmin=None, vmax=None, save_path=None):
    n_images = len(images)
    ncols = n_images if ncols > n_images else ncols
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * xunit, nrows * yunit))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        if vmin is None or vmax is None:
            axes[row, col].imshow(img, cmap=cmap)
        else:
            axes[row, col].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, col].set_axis_off()
        if len(titles) > 0:
            axes[row, col].set_title(titles[idx])

    for idx in range(n_images, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_axis_off()

    fig.tight_layout()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f">> {save_path} is saved.\n")
    else:
        plt.show()


def update_history(history, epoch_history):
    for split_name, metrics in epoch_history.items():
        history.setdefault(split_name, {})
        for metric_name, metric_values in metrics.items():
            history[split_name].setdefault(metric_name, [])
            history[split_name][metric_name].extend(metric_values)


def plot_history(history, metric_names=None, save_path=None):
    if metric_names is None:
        metric_names = list(history.keys())

    num_metrics = len(metric_names)
    fig, axes = plt.subplots(ncols=num_metrics, figsize=(3 * num_metrics, 3))
    if num_metrics == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_names):
        metric_values = history[metric_name]
        num_epochs = len(metric_values)
        epochs = range(1, num_epochs + 1)

        ax.plot(epochs, metric_values, 'k')
        ax.set_title(metric_name)
        ax.set_xlabel('Epoch')
        ax.grid(True)

    fig.tight_layout()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f">> {save_path} is saved.\n")
    else:
        plt.show()
