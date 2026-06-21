"""
data/visualize.py
SANITY CHECK — run this BEFORE training every time you change preprocessing
or switch datasets. Plots a few SAR / Optical-preview / per-band MS patches
side by side so you can catch bugs (wrong band order, bad normalization,
mismatched SAR/MS pairs) visually instead of finding out after a 2-hour
training run produces garbage embeddings.

Usage (in a notebook cell):
    from data.visualize import sanity_check_grid
    sanity_check_grid(df, n_samples=6)
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config.config import PLOTS_DIR, MS_BANDS
from data.preprocessing import load_sar, load_optical_rgb, load_ms, load_ms_rgb_preview


def sanity_check_grid(df, n_samples=6, seed=0, save=True, show=True):
    """
    For n_samples random rows of df, plot:
      col 1: SAR VV
      col 2: SAR VH
      col 3: Optical RGB preview (B04/B03/B02)
      col 4: One MS band (B08, NIR) as a quick multispectral check
    """
    sample = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.2 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for i, row in tqdm(sample.iterrows(), total=len(sample), desc="Rendering sanity-check grid"):
        sar = load_sar(row["s1_name"]).numpy()          # (2, H, W)
        rgb_preview = load_ms_rgb_preview(row["patch_id"])  # (H, W, 3) uint8
        optical = load_optical_rgb(row["patch_id"]).numpy()
        optical = np.clip(optical * np.array([.229, .224, .225])[:, None, None] + np.array([.485, .456, .406])[:, None, None], 0, 1).transpose(1, 2, 0)
        ms = load_ms(row["patch_id"]).numpy()            # (10, H, W) normalized

        axes[i, 0].imshow(sar[0], cmap="gray")
        axes[i, 0].set_title(f"SAR VV\n{row['primary_label']}", fontsize=9)

        axes[i, 1].imshow(sar[1], cmap="gray")
        axes[i, 1].set_title("SAR VH", fontsize=9)

        axes[i, 2].imshow(optical)
        axes[i, 2].set_title("Optical encoder input", fontsize=9)

        nir_idx = MS_BANDS.index("B08")
        axes[i, 3].imshow(ms[nir_idx], cmap="viridis")
        axes[i, 3].set_title(f"MS band {MS_BANDS[nir_idx]} (normalized)", fontsize=9)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    if save:
        out_path = PLOTS_DIR / "sanity_check_grid.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved sanity-check figure to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def class_distribution_plot(df, save=True, show=True):
    """Bar chart of primary_label counts per split — quick balance check."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby(["primary_label", "split"]).size().unstack().plot(kind="bar", ax=ax)
    ax.set_title("Class distribution per split")
    ax.set_ylabel("Patch count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save:
        out_path = PLOTS_DIR / "class_distribution.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved class distribution figure to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
