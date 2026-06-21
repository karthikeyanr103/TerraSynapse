# CODEX PROMPT — Cross-Modal Satellite Image Retrieval (v2)
## Full Pipeline: BigEarthNet-MM Subset → GitHub → Kaggle (train/eval) → Streamlit Deployment

---

## CONTEXT FOR CODEX

You are building a **Cross-Modal Satellite Image Retrieval** system for an internship project
(ISRO Internship 2026, Problem Statement 11). The goal: retrieve semantically similar remote
sensing images both within and across sensor modalities, using deep metric learning on
pretrained BigEarthNet v2.0 backbones, then ship a working **Streamlit inference demo**.

The dataset is a curated subset of **BigEarthNet-MM** with **three modalities derived from
two sensors**:
- **SAR** — Sentinel-1 (VV, VH)
- **Optical (RGB)** — Sentinel-2 bands B04/B03/B02 (3 channels)
- **Multispectral** — Sentinel-2 bands B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 (10 channels)

Optical and Multispectral come from the *same* Sentinel-2 source patch — they are not
independently captured. Treat them as separate modalities only at the **representation
level**: two different encoders, never sharing weights, trained with their own projection
heads, so the retrieval task is non-trivial rather than a literal same-image lookup.

This satisfies all **7 required retrieval directions**:
- Same-modal: optical→optical, SAR→SAR, multispectral→multispectral
- Cross-modal: optical→SAR, SAR→optical, optical→multispectral, multispectral→optical

Workflow: build the data subset on **AWS EC2** → push code to **GitHub** → run three
separate **Kaggle notebooks** (data-check, train, evaluate) → export trained
weights+embeddings → build a **Streamlit app** for interactive inference.

Read every section before writing code. Do not invent alternative paths, names, or schemas.

---

## SECTION 1 — EC2 SOURCE DATA (unchanged from BigEarthNet-MM layout)

```
~/BigEarthNet-MM/
├── metadata.parquet
├── BigEarthNet-S1/{s1_name}/{s1_name}_VV.tif, {s1_name}_VH.tif
├── BigEarthNet-S2/{patch_id}/{patch_id}_B02.tif ... {patch_id}_B12.tif
└── Reference_Maps/        # not used
```

metadata.parquet columns: `patch_id`, `labels` (list[str]), `split`, `country`,
`s1_name`, `s2v1_name`, `contains_seasonal_snow`, `contains_cloud_or_shadow`.

---

## SECTION 2 — SUBSET SELECTION (fixed parameters, do not change)

```python
KEEP_CLASSES = [
    "Arable land", "Broad-leaved forest", "Coniferous forest",
    "Mixed forest", "Pastures", "Urban fabric",
    "Industrial or commercial units", "Inland waters",
    "Complex cultivation patterns", "Transitional woodland, shrub",
]
N_CAP_PER_CLASS = 5000
RANDOM_STATE = 42
```
Use BigEarthNet's pre-defined `split` column — never re-split (avoids spatial leakage
between geographically adjacent patches). Expected output: ~47,498 patches
(train ~48.6%, validation ~26.2%, test ~25.2%).

---

## SECTION 3 — REPOSITORY STRUCTURE (build exactly this)

```
ben_retrieval/
├── README.md
├── requirements.txt
├── .gitignore                      # exclude kaggle.json, outputs/, __pycache__, raw rasters
│
├── scripts/                        # run on EC2
│   ├── 01_select_subset.py         # builds ben_subset.csv (tqdm progress bars)
│   ├── 02_copy_subset.py           # copies only selected S1+S2 folders (tqdm)
│   └── 03_upload_kagglehub.py      # uploads to Kaggle via kagglehub (argparse + private)
│
├── config/
│   └── config.py                   # paths + SWITCHABLE per-modality backbone registry
│
├── data/
│   ├── preprocessing.py            # load_sar, load_optical_rgb, load_ms, stats (tqdm)
│   ├── dataset.py                  # triplet sampler across 3 modalities
│   └── visualize.py                # sanity-check grid (SAR/Optical/MS side by side)
│
├── models/
│   └── encoders.py                 # pretrained-backbone wrapper, swappable per modality
│
├── training/
│   ├── loss.py                     # cosine triplet loss
│   └── trainer.py                  # tqdm progress bars; frozen-backbone aware
│
├── evaluation/
│   └── metrics.py                  # F1@5/10, Recall@5/10, mAP@5/10, retrieval time (FAISS)
│
├── notebooks/                      # cell-by-cell .md guides, one per Kaggle notebook
│   ├── 01_data_check.md
│   ├── 02_train_retrieval_model.md
│   └── 03_evaluate_retrieval.md
│
├── deploy/                         # NEW — Streamlit inference app
│   ├── app.py                      # main Streamlit entry point
│   ├── export_artifacts.py         # run in Kaggle after training: dumps embeddings+index+thumbnails
│   ├── inference.py                # shared encode + FAISS-search logic used by app.py
│   ├── artifacts/                  # populated by export_artifacts.py, gitignored if large
│   │   ├── checkpoints/{backbone_key}_{modality}.pt
│   │   ├── gallery_embeddings/{modality}_{backbone_key}.npy
│   │   ├── gallery_labels/{modality}_{backbone_key}.csv
│   │   ├── faiss_index/{modality}_{backbone_key}.index
│   │   └── sample_thumbnails/{modality}/{patch_id}.png   # small previews for the picker UI
│   └── requirements_deploy.txt     # streamlit, faiss-cpu, torch (cpu), pillow, rasterio
│
└── utils/
    └── __init__.py
```

---

## SECTION 4 — config/config.py (extend to 3 modalities)

```python
from pathlib import Path

ENV = "kaggle"   # "ec2" | "kaggle" | "local_deploy"

if ENV == "ec2":
    DATA_ROOT = Path.home() / "ben_subset_data"
elif ENV == "kaggle":
    DATA_ROOT = Path("/kaggle/input/bigearth-mm-subset-47k")
else:
    DATA_ROOT = Path("./deploy/artifacts")   # streamlit app reads only exported artifacts

S1_ROOT  = DATA_ROOT / "BigEarthNet-S1"
S2_ROOT  = DATA_ROOT / "BigEarthNet-S2"
METADATA = DATA_ROOT / "ben_subset.csv"

# ── Per-modality pretrained backbone registries ──────────────────────────
# VERIFY exact HF repo ids at https://huggingface.co/BIFOLD-BigEarthNetv2-0
# before trusting any run — names can change between releases.
S2_BACKBONE_REGISTRY = {
    "resnet50":  "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
    "resnet101": "BIFOLD-BigEarthNetv2-0/resnet101-s2-v0.2.0",
    "vit_small": "BIFOLD-BigEarthNetv2-0/vit_small_patch16_224-s2-v0.2.0",
    "convnext":  "BIFOLD-BigEarthNetv2-0/convnext_base-s2-v0.2.0",
}
S1_BACKBONE_REGISTRY = {
    "resnet50":  "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0",
    "resnet101": "BIFOLD-BigEarthNetv2-0/resnet101-s1-v0.2.0",
    "vit_small": "BIFOLD-BigEarthNetv2-0/vit_small_patch16_224-s1-v0.2.0",
    "convnext":  "BIFOLD-BigEarthNetv2-0/convnext_base-s1-v0.2.0",
}
# Optical (RGB-only, 3-band): if the BEN-v2.0 registry has no RGB-specific
# checkpoint, fall back to a standard ImageNet-pretrained torchvision backbone
# (e.g. resnet50). A from-scratch BEN checkpoint is not required for 3-channel
# natural-looking imagery — ImageNet weights transfer reasonably well here.
OPTICAL_BACKBONE_REGISTRY = {
    "resnet50_imagenet":  "torchvision://resnet50",
    "resnet101_imagenet": "torchvision://resnet101",
}

BACKBONE_KEY_S1      = "resnet50"
BACKBONE_KEY_S2      = "resnet50"
BACKBONE_KEY_OPTICAL = "resnet50_imagenet"

BACKBONE_S1      = S1_BACKBONE_REGISTRY[BACKBONE_KEY_S1]
BACKBONE_S2      = S2_BACKBONE_REGISTRY[BACKBONE_KEY_S2]
BACKBONE_OPTICAL = OPTICAL_BACKBONE_REGISTRY[BACKBONE_KEY_OPTICAL]

MODALITIES = ["SAR", "Optical", "Multispectral"]   # canonical order used everywhere

SAR_CHANNELS = 2
OPTICAL_CHANNELS = 3
MS_CHANNELS = 10
MS_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
OPTICAL_BANDS = ["B04", "B03", "B02"]

EMBED_DIM = 256
FREEZE_BACKBONE = True
BATCH_SIZE = 32
EPOCHS_PROJ_ONLY = 10
EPOCHS_FINETUNE = 10
LR_PROJ = 1e-3
LR_ENC = 1e-5
MARGIN = 0.3
CROSS_MODAL_RATIO = 0.5
TOP_K = [5, 10]
N_QUERY_PER_CLS = 50

KEEP_CLASSES = [ ... ]   # same 10 classes as Section 2
RANDOM_STATE = 42
```

---

## SECTION 5 — data/preprocessing.py (extend with `load_optical_rgb`)

Add a third loader alongside the existing `load_sar` / `load_ms`:

```python
def load_optical_rgb(patch_id: str) -> torch.Tensor:
    """
    Load Sentinel-2 B04/B03/B02 only -> tensor (3, H, W), ImageNet-style
    normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] on a
    percentile-stretched [0,1] image), since this feeds an ImageNet-pretrained
    backbone, not a BEN-specific one.
    """
```
Keep `load_ms` reading the full 10-band stack as before — `load_optical_rgb` and
`load_ms` must NOT share a code path beyond the raw `.tif` read, to keep the two
modalities conceptually and numerically distinct downstream.

---

## SECTION 6 — data/dataset.py (extend triplet sampler to 3 modalities)

Anchor modality is now sampled from `["SAR", "Optical", "Multispectral"]`. When
generating a cross-modal positive, pick any *other* modality at random (not just
the one opposite). Negative modality is also sampled from all three. Same-class
same-modal positives still draw from `cls_idx[class]` as before.

---

## SECTION 7 — models/encoders.py (3 encoders)

`build_encoders(device)` now returns a dict:
```python
{
    "SAR": PretrainedModalityEncoder(BACKBONE_S1, in_channels=2),
    "Optical": PretrainedModalityEncoder(BACKBONE_OPTICAL, in_channels=3),
    "Multispectral": PretrainedModalityEncoder(BACKBONE_S2, in_channels=10),
}
```
Keep the existing fallback feature-extraction logic (`forward_features` +
`forward_head(pre_logits=True)` for timm-style models, forward-hook on the last
`nn.Linear` otherwise) — it must work for both the BEN checkpoints and a plain
torchvision ImageNet ResNet (which has `.fc` as its final Linear; the hook
fallback already handles this case generically).

Each encoder's `inspect_model_structure()` should still be callable per
modality before a full training run.

---

## SECTION 8 — training/trainer.py & evaluation/metrics.py

Adapt the training loop's `encode_batch` to route each item through
`encoders[modality]` using a dict lookup instead of an `if SAR else MS` branch.
Evaluation must run **all 7 directions**:
optical→optical, SAR→SAR, multispectral→multispectral,
optical→SAR, SAR→optical, optical→multispectral, multispectral→optical.
Report F1@5, F1@10, Recall@5, Recall@10, mAP@5, mAP@10, and avg retrieval
time (ms/query) per direction, via FAISS `IndexFlatIP` on L2-normalized embeddings.

---

## SECTION 9 — EC2 → GITHUB → KAGGLE WORKFLOW

```bash
# One-time: push repo
cd ben_retrieval && git init && git add . && git commit -m "init"
git remote add origin https://github.com/<USER>/<REPO>.git && git push -u origin main

# On EC2: build + upload subset
python3 scripts/01_select_subset.py
python3 scripts/02_copy_subset.py
pip install kagglehub --quiet
python3 scripts/03_upload_kagglehub.py \
    --kaggle_username "$KAGGLE_USERNAME" --kaggle_key "$KAGGLE_KEY" \
    --handle "<USER>/bigearth-mm-subset-47k" \
    --local_dir ~/ben_subset_data --version_notes "Initial 47K subset"
# -> confirm Private on kaggle.com dataset page manually
```
`scripts/03_upload_kagglehub.py` sets `os.environ["KAGGLE_USERNAME"]` and
`os.environ["KAGGLE_KEY"]` (exact names kagglehub expects) from argparse args.

Each Kaggle notebook's first cell clones the GitHub repo fresh:
```python
!git clone https://github.com/<USER>/<REPO>.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -r requirements.txt
```

Three notebooks, in order: **01-data-check** (sanity visualization, no GPU needed)
→ **02-train-retrieval-model** (GPU, trains projection heads for all 3 encoders,
saves `best_model.pt` containing all three state_dicts) → **03-evaluate-retrieval**
(attaches Notebook 2's output, runs all 7 directions, saves results CSV).

---

## SECTION 10 — NEW: STREAMLIT DEPLOYMENT APP (`deploy/`)

### 10.1 deploy/export_artifacts.py (run in Kaggle, AFTER Notebook 2/3 complete)

Purpose: shrink everything needed for inference down to a small, portable
artifact bundle that doesn't require the full 47K-patch dataset at deploy time.

```python
"""
Run inside the Kaggle evaluation notebook (or a dedicated 04-export notebook)
once training is done. With tqdm progress bars throughout, this script:

1. Loads the trained checkpoint (all 3 encoders).
2. Encodes the FULL test-split gallery for each modality -> saves as
   gallery_embeddings/{modality}_{backbone_key}.npy (float32 arrays)
   and gallery_labels/{modality}_{backbone_key}.csv (patch_id, primary_label).
3. Builds and saves a FAISS index per modality
   (faiss.write_index) to faiss_index/{modality}_{backbone_key}.index.
4. Saves small (128x128) PNG thumbnails of a representative sample
   (e.g. 20 per class per modality) into sample_thumbnails/{modality}/
   for the Streamlit image picker — using the existing
   load_optical_rgb-style preview logic for SAR (dB-scaled grayscale->RGB)
   and MS (NIR/Red/Green false color) so thumbnails are human-interpretable.
5. Copies best_model.pt into deploy/artifacts/checkpoints/.
6. Zips deploy/artifacts/ into one file for download from Kaggle's Output tab.
"""
```

### 10.2 deploy/inference.py (shared logic, imported by app.py)

```python
"""
- load_artifacts(): loads the 3 encoders + their FAISS indices + gallery
  labels + thumbnails into memory ONCE (use st.cache_resource in app.py
  around this call, not inside this module, to keep this module UI-free).
- encode_query(image_path_or_array, modality, backbone_key) -> np.ndarray
  Runs the appropriate preprocessing (load_sar / load_optical_rgb / load_ms
  equivalents, but accepting an uploaded file instead of a known patch_id)
  then the matching encoder -> L2-normalized embedding.
- search(query_embedding, target_modality, backbone_key, k)
  -> list of (patch_id, similarity_score, thumbnail_path)
  Looks up the right FAISS index for target_modality, returns top-k with
  cosine similarity scores and the elapsed search time in ms.
"""
```

### 10.3 deploy/app.py (Streamlit UI — exact flow required)

```python
"""
Streamlit app with this exact interaction flow, in this order:

1. TASK SELECTOR (st.selectbox)
   "Choose retrieval task:"
   -> one of the 7 directions, displayed as readable labels:
      "Optical -> Optical", "SAR -> SAR", "Multispectral -> Multispectral",
      "Optical -> SAR", "SAR -> Optical",
      "Optical -> Multispectral", "Multispectral -> Optical"
   Selecting this fixes BOTH the source modality (for the query) and the
   target modality (for the gallery/FAISS index to search).

2. MODEL SELECTOR (st.selectbox)
   "Choose backbone:" -> populated from whichever backbone_keys have
   exported artifacts available (scan deploy/artifacts/checkpoints/ at
   startup, don't hardcode the list -- if only resnet50 was exported,
   only resnet50 should appear as an option).

3. IMAGE SELECTOR (st.radio: "Pick a sample image" vs "Upload my own")
   - Sample mode: st.selectbox of thumbnails from
     sample_thumbnails/{source_modality}/, rendered as an image grid
     (st.columns + st.image) the user clicks/picks from.
   - Upload mode: st.file_uploader accepting .tif/.jpg/.png matching the
     source modality's expected channel count; show a friendly error via
     st.error if the uploaded file's band count doesn't match what the
     source modality needs (e.g. uploading a 3-band JPG for a 'SAR' query
     should fail clearly, not crash).

4. "Run Inference" button (st.button)
   On click:
   - show a st.spinner while encoding + searching
   - call inference.encode_query(...) then inference.search(...)
   - display the query image on the left (st.image)
   - display top-5 AND top-10 results as two rows of thumbnails
     (st.tabs(["Top-5", "Top-10"]) is a clean way to do this), each
     thumbnail captioned with its similarity score and predicted class
   - display total retrieval time in ms below the results
     (st.metric("Avg retrieval time", f"{elapsed:.2f} ms"))

5. Sidebar: short static explanation of what same-modal vs cross-modal
   retrieval means, and a link/note about which dataset subset and
   backbone produced the currently loaded artifacts (read this from a
   small metadata.json saved alongside the artifacts by export_artifacts.py,
   don't hardcode it in app.py).

Use st.cache_resource for loading encoders/indices (expensive, load once)
and st.cache_data for thumbnail listings (cheap, but re-scanning the
filesystem every rerun is wasteful).
"""
```

### 10.4 deploy/requirements_deploy.txt

```
streamlit>=1.30.0
torch>=2.0.0          # CPU build is fine for inference-only deployment
faiss-cpu>=1.7.4
rasterio>=1.3.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
configilm
```

### 10.5 Local run / deployment instructions

```bash
cd deploy
pip install -r requirements_deploy.txt
streamlit run app.py
```
For sharing: deploy via Streamlit Community Cloud pointed at the GitHub repo's
`deploy/` subfolder, with `deploy/artifacts/` either committed via Git LFS (if
small enough, since artifacts are now just embeddings + a few hundred
thumbnails + checkpoints, NOT the raw dataset) or downloaded at app startup
from a private Kaggle dataset/notebook output via `kagglehub.dataset_download()`
if Git LFS quota is a concern — implement whichever fits, but document the
choice in `deploy/README.md`.

---

## SECTION 11 — requirements.txt (root, training-side)

```
torch>=2.0.0
torchvision>=0.15.0
rasterio>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
faiss-cpu>=1.7.4
tqdm>=4.65.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
configilm
kagglehub
```

---

## SECTION 12 — EXPECTED RESULTS (internship targets, unchanged in spirit)

| Metric | Baseline (random init) | Target (trained) |
|---|---|---|
| F1@5 same-modal | ~0.10 (1/N_CLASSES) | ≥ 0.65 |
| F1@5 cross-modal | ~0.10 | ≥ 0.50 |
| F1@10 same-modal | ~0.10 | ≥ 0.75 |
| F1@10 cross-modal | ~0.10 | ≥ 0.60 |
| Avg retrieval time | < 5 ms | < 10 ms (requirement) |

Cross-modal performance will be lower than same-modal — expected and acceptable;
weight cross-modal results appropriately when reporting.

---

## SECTION 13 — CODEX INSTRUCTIONS

1. Implement the exact directory structure in SECTION 3. Do not rename paths,
   classes, or config variables.
2. Use `rasterio` for all GeoTIFF reads. Never use PIL/OpenCV for `.tif` satellite bands.
3. All encoder outputs must be L2-normalized before any similarity computation.
4. FAISS indices must be `IndexFlatIP` (inner product = cosine sim on unit vectors).
5. Never re-split the data — always use the pre-defined `split` column.
6. Implement Optical, SAR, and Multispectral as three fully independent encoders
   with no shared weights and no shared preprocessing code path.
7. Every loop that touches disk, trains, or runs inference over >1 item must show
   a `tqdm` progress bar — this includes the Streamlit export script.
8. Add a visual sanity-check step (`data/visualize.py`) and require it be run and
   manually confirmed before any training run is trusted.
9. MS_MEAN/MS_STD must be computed from the training split only, via the provided
   `compute_ms_stats` utility — never hardcode guessed values into a final run.
10. Evaluate and report all 7 retrieval directions separately; never average across
    directions in a way that hides cross-modal underperformance.
11. The Streamlit app must read available backbone options dynamically from
    exported artifacts on disk — never hardcode a model list in `app.py`.
12. The Streamlit app must handle a mismatched-modality upload (e.g. wrong band
    count) with a clear `st.error`, not an unhandled exception.
13. All output files (checkpoints, results CSVs, exported artifacts) go under
    `OUTPUT_DIR` / `deploy/artifacts/` as defined in `config/config.py` — never
    write outside these locations.
14. Before trusting `models/encoders.py`'s automatic feature extraction against
    a *new* backbone, call `inspect_model_structure()` on it and confirm by eye
    that the located feature dimension is sensible.
