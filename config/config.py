"""
config.py
All paths, hyperparameters, and the SWITCHABLE PRETRAINED MODEL LIST live here.
This file is imported by every other module — never hardcode paths elsewhere.
"""

import os
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SWITCH — change this one line when moving AWS → Kaggle
# ════════════════════════════════════════════════════════════════════════
ENV = os.getenv("BEN_ENV", "kaggle")

if ENV == "ec2":
    DATA_ROOT  = Path.home() / "ben_subset_data"
    OUTPUT_DIR = Path.home() / "TerraSynapse_outputs"
elif ENV == "kaggle":
    DATA_ROOT  = Path("/kaggle/input/bigearth-mm-subset-47k")
    OUTPUT_DIR = Path("/kaggle/working/outputs")
else:
    DATA_ROOT = Path(__file__).resolve().parents[1] / "deploy" / "artifacts"
    OUTPUT_DIR = DATA_ROOT

S1_ROOT  = DATA_ROOT / "BigEarthNet-S1"
S2_ROOT  = DATA_ROOT / "BigEarthNet-S2"
METADATA = DATA_ROOT / "ben_subset.csv"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "sanity_check_plots"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [OUTPUT_DIR, CKPT_DIR, PLOTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# PRETRAINED BACKBONE REGISTRY  (configilm / BigEarthNet v2.0 weights)
# ════════════════════════════════════════════════════════════════════════
# These are loaded via:
#   from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
#   model = BigEarthNetv2_0_ImageClassifier.from_pretrained("<hf_repo_id>")
#
# IMPORTANT: verify the exact repo id strings on the model hub before running —
# https://huggingface.co/BIFOLD-BigEarthNetv2-0
# Repo ids occasionally get renamed between releases, and the list below is only
# as accurate as the day it was written. Treat it as a starting point, not gospel.
#
# Pick ONE entry per modality per run via BACKBONE_S1 / BACKBONE_S2 below.
# To try a different backbone, just change the string — nothing else changes.

S2_BACKBONE_REGISTRY = {
    "resnet50":   "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
    "resnet101":  "BIFOLD-BigEarthNetv2-0/resnet101-s2-v0.2.0",
    "resnet152":  "BIFOLD-BigEarthNetv2-0/resnet152-s2-v0.2.0",
    "vit_small":  "BIFOLD-BigEarthNetv2-0/vit_small_patch16_224-s2-v0.2.0",
    "vit_tiny":   "BIFOLD-BigEarthNetv2-0/vit_tiny_patch16_224-s2-v0.2.0",
    "convnext":   "BIFOLD-BigEarthNetv2-0/convnext_base-s2-v0.2.0",
}

S1_BACKBONE_REGISTRY = {
    "resnet50":   "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0",
    "resnet101":  "BIFOLD-BigEarthNetv2-0/resnet101-s1-v0.2.0",
    "resnet152":  "BIFOLD-BigEarthNetv2-0/resnet152-s1-v0.2.0",
    "vit_small":  "BIFOLD-BigEarthNetv2-0/vit_small_patch16_224-s1-v0.2.0",
    "vit_tiny":   "BIFOLD-BigEarthNetv2-0/vit_tiny_patch16_224-s1-v0.2.0",
    "convnext":   "BIFOLD-BigEarthNetv2-0/convnext_base-s1-v0.2.0",
}

OPTICAL_BACKBONE_REGISTRY = {
    "resnet50_imagenet": "torchvision://resnet50",
    "resnet101_imagenet": "torchvision://resnet101",
}

# ── ACTIVE SELECTION — change these two lines to swap backbones ──────────
BACKBONE_KEY_S1 = "resnet50"
BACKBONE_KEY_S2 = "resnet50"
BACKBONE_KEY_OPTICAL = "resnet50_imagenet"

BACKBONE_S1 = S1_BACKBONE_REGISTRY[BACKBONE_KEY_S1]
BACKBONE_S2 = S2_BACKBONE_REGISTRY[BACKBONE_KEY_S2]
BACKBONE_OPTICAL = OPTICAL_BACKBONE_REGISTRY[BACKBONE_KEY_OPTICAL]

# ════════════════════════════════════════════════════════════════════════
# Modality settings
# ════════════════════════════════════════════════════════════════════════
SAR_CHANNELS = 2          # VV, VH
OPTICAL_CHANNELS = 3
MS_CHANNELS  = 10         # B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
PATCH_SIZE   = 120

MS_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
OPTICAL_BANDS = ["B04", "B03", "B02"]
MODALITIES = ["SAR", "Optical", "Multispectral"]

# ════════════════════════════════════════════════════════════════════════
# Model / training hyperparameters
# ════════════════════════════════════════════════════════════════════════
EMBED_DIM        = 256          # output dim of the shared projection head
FREEZE_BACKBONE  = True         # True = use pretrained backbone as a fixed feature
                                 # extractor (fast, good for limited time/compute);
                                 # set False to fine-tune end-to-end if time allows

BATCH_SIZE       = 32
EPOCHS_PROJ_ONLY = 10            # train only the projection head (backbone frozen)
EPOCHS_FINETUNE  = 10            # optional second phase if FREEZE_BACKBONE=False
LR_PROJ          = 1e-3
LR_ENC           = 1e-5
MARGIN           = 0.3
CROSS_MODAL_RATIO = 0.5

TOP_K           = [5, 10]
N_QUERY_PER_CLS = 50

# ════════════════════════════════════════════════════════════════════════
# Classes
# ════════════════════════════════════════════════════════════════════════
KEEP_CLASSES = [
    "Arable land", "Broad-leaved forest", "Coniferous forest",
    "Mixed forest", "Pastures", "Urban fabric",
    "Industrial or commercial units", "Inland waters",
    "Complex cultivation patterns", "Transitional woodland, shrub",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(KEEP_CLASSES)}
N_CLASSES    = len(KEEP_CLASSES)

RANDOM_STATE = 42
