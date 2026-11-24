"""Plotting helpers reused across modules."""
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_image_grid(images, labels=None, nrow: int = 8, figsize_per: float = 1.4, cmap="gray", title=None):
    """Show a batch of images in a grid.

    images: tensor/array shaped (N, C, H, W) or (N, H, W), values in [0, 1] (or roughly).
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    images = np.asarray(images)
    if images.ndim == 4:
        # (N, C, H, W) -> (N, H, W, C) because matplotlib expects channels last
        images = images.transpose(0, 2, 3, 1)
        if images.shape[-1] == 1:
            images = images[..., 0]
    n = len(images)
    ncol = min(nrow, n)
    nrows = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * figsize_per, nrows * figsize_per), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(np.clip(images[i], 0, 1), cmap=cmap if images[i].ndim == 2 else None)
            if labels is not None:
                ax.set_title(str(labels[i]), fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_loss_curves(history: dict, title: str = "Training curves"):
    """Plot every series in a history dict like {"train_loss": [...], "val_loss": [...], "val_acc": [...]}.

    Losses and accuracies go on separate axes because their scales differ.
    """
    loss_keys = [k for k in history if "loss" in k]
    other_keys = [k for k in history if k not in loss_keys]
    ncols = 2 if other_keys else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)
    for k in loss_keys:
        axes[0, 0].plot(history[k], label=k)
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_title("loss"); axes[0, 0].legend()
    if other_keys:
        for k in other_keys:
            axes[0, 1].plot(history[k], label=k)
        axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_title("metrics"); axes[0, 1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(predict_fn, X, y, resolution: int = 200, title: str = ""):
    """Plot a 2-D classifier's decision regions.

    predict_fn: takes an (M, 2) NumPy array and returns class labels or probabilities of class 1, shape (M,).
    """
    X = np.asarray(X); y = np.asarray(y)
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]  # every grid point as an (x, y) row
    zz = np.asarray(predict_fn(grid)).reshape(xx.shape)
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=20)
    plt.title(title)
    plt.show()
