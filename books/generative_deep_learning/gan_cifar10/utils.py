import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch


def set_seed(seed, deterministic=True, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic)

    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def create_images(generator, z_sample):
    device = next(generator.parameters()).device
    z_tensor = torch.tensor(z_sample).float().to(device)
    outputs = generator(z_tensor).cpu()

    images = (outputs + 1) / 2
    images = images.permute(0, 2, 3, 1).numpy()
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
        print(f">> {save_path} is saved.\n")
    else:
        plt.show()


def plot_history(history, metric_names=None):
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
        # ax.set_ylabel(metric_name)
        ax.grid(True)

    fig.tight_layout()
    plt.show()