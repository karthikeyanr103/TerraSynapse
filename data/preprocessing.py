"""
data/preprocessing.py
Per-modality loading + normalization. Used by both the Dataset class and the
sanity-check visualizer, so there's exactly one source of truth for how a
raw .tif becomes a model-ready tensor.
"""

import numpy as np
import torch
import rasterio
from pathlib import Path
from tqdm import tqdm

from config.config import S1_ROOT, S2_ROOT, MS_BANDS, OPTICAL_BANDS

# Placeholder stats — REPLACE these by running scripts in data/compute_stats.py
# (or the notebook cell that calls compute_ms_stats below) on the TRAIN split only.
MS_MEAN = np.array([340, 430, 420, 490, 740, 960, 1020, 1060, 820, 540], dtype=np.float32)
MS_STD  = np.array([180, 180, 200, 210, 290, 330, 360, 360, 320, 260], dtype=np.float32)


def load_sar(s1_name: str) -> torch.Tensor:
    """Load a SAR patch -> tensor (2, H, W), normalized to [0, 1]."""
    folder = S1_ROOT / s1_name
    tensors = []
    for pol in ["VV", "VH"]:
        tif_path = folder / f"{s1_name}_{pol}.tif"
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        arr = 10.0 * np.log10(np.abs(arr) + 1e-10)            # to dB
        arr = np.clip((arr + 30.0) / 30.0, 0.0, 1.0)            # clip to [-30,0] dB -> [0,1]
        tensors.append(arr)
    return torch.tensor(np.stack(tensors, axis=0))


def load_ms(patch_id: str) -> torch.Tensor:
    """Load a multispectral patch -> tensor (10, H, W), per-band z-score normalized."""
    folder = S2_ROOT / patch_id
    tensors = []
    for band in MS_BANDS:
        tif_path = folder / f"{patch_id}_{band}.tif"
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        tensors.append(arr)
    arr = np.stack(tensors, axis=0)
    arr = np.clip(arr, 0, 10000)
    arr = (arr - MS_MEAN[:, None, None]) / (MS_STD[:, None, None] + 1e-6)
    return torch.tensor(arr)


def load_optical_rgb(patch_id: str) -> torch.Tensor:
    """Load B04/B03/B02, percentile-stretch, and ImageNet-normalize."""
    folder = S2_ROOT / patch_id
    bands = []
    for band in OPTICAL_BANDS:
        with rasterio.open(folder / f"{patch_id}_{band}.tif") as src:
            bands.append(src.read(1).astype(np.float32))
    rgb = np.stack(bands, axis=0)
    lo, hi = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return torch.from_numpy((rgb - mean) / std)


def load_ms_rgb_preview(patch_id: str) -> np.ndarray:
    """
    Load only B04/B03/B02 and return an (H, W, 3) uint8 array for quick
    human-viewable previews — used only by the visualizer, not the model.
    """
    folder = S2_ROOT / patch_id
    bands = []
    for band in ["B04", "B03", "B02"]:
        tif_path = folder / f"{patch_id}_{band}.tif"
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        bands.append(arr)
    rgb = np.stack(bands, axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def compute_ms_stats(patch_ids, sample_size=2000, seed=0):
    """
    Compute per-band mean/std from a random sample of the TRAINING split only.
    Run this once, print the result, and hardcode it into MS_MEAN / MS_STD above.

    Example:
        from data.preprocessing import compute_ms_stats
        train_ids = df[df['split']=='train']['patch_id'].tolist()
        mean, std = compute_ms_stats(train_ids)
    """
    rng = np.random.default_rng(seed)
    ids = patch_ids if len(patch_ids) <= sample_size else rng.choice(
        patch_ids, size=sample_size, replace=False
    )

    stacks = []
    for pid in tqdm(ids, desc="Computing MS band statistics", unit="patch"):
        folder = S2_ROOT / pid
        bands = []
        for band in MS_BANDS:
            with rasterio.open(folder / f"{pid}_{band}.tif") as src:
                bands.append(src.read(1).astype(np.float32))
        stacks.append(np.stack(bands, axis=0))

    stack = np.stack(stacks)  # (N, 10, H, W)
    mean = stack.mean(axis=(0, 2, 3))
    std = stack.std(axis=(0, 2, 3))
    print("MS_MEAN =", mean.round(2).tolist())
    print("MS_STD  =", std.round(2).tolist())
    return mean, std
