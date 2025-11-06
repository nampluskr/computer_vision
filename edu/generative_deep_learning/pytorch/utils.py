import numpy as np
import matplotlib.pyplot as plt


def show_images(*images, ncols=5, xunit=3, yunit=3, cmap='gray', titles=[], vmin=None, vmax=None):
    n_images = len(images)
    ncols = n_images if ncols > n_images else ncols
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * xunit, nrows * yunit))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        if vmin is None or vmax is None:
            axes[row, col].imshow(img, cmap=cmap if img.ndim == 2 else None)
        else:
            axes[row, col].imshow(img, cmap=cmap if img.ndim == 2 else None, vmin=vmin, vmax=vmax)
        axes[row, col].set_axis_off()
        if len(titles) > 0:
            axes[row, col].set_title(titles[idx])

    for idx in range(n_images, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_axis_off()

    fig.tight_layout()
    plt.show()