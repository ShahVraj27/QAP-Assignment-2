"""Histogram plotting: measured frequencies as bars, exact probabilities as markers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import bits


def plot_panel(ax, measured, exact, m, title):
    """Draw one histogram panel on ``ax``."""
    labels = [bits(y, m) for y in range(2**m)]
    x = np.arange(2**m)
    ax.bar(x, measured, color="#4C72B0", alpha=0.85, label="measured (shots)")
    ax.scatter(x, exact, color="#DD8452", marker="D", s=30, zorder=3, label="exact")
    for xi, value in zip(x, measured):
        if value > 0.005:
            ax.text(xi, value + 0.02, f"{value:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90 if m >= 3 else 0)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("y (counting register)")
    ax.set_ylabel("probability")
    ax.set_title(title, fontsize=10)


def plot_side_by_side(panels, suptitle, filename=None):
    """Plot several panels in one row.

    ``panels`` is a list of dicts with keys: measured, exact, m, title.
    If ``filename`` is given, the figure is also saved there (PNG, 150 dpi).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        plot_panel(ax, panel["measured"], panel["exact"], panel["m"], panel["title"])
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig
