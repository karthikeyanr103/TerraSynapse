# Cross-Modal Satellite Image Retrieval — Step-by-Step Guide

End-to-end three-modality pipeline: AWS EC2 data prep -> GitHub -> Kaggle train/evaluate -> Streamlit deployment.

---

## PHASE 0 — One-time setup

### 0.1 Push this code to GitHub (from EC2 or your laptop)
```bash
cd TerraSynapse
git init
git add .
git commit -m "Initial cross-modal retrieval pipeline"
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git branch -M main
git push -u origin main
```
Every time you change a `.py` file, commit + push again — each Kaggle
notebook re-clones the repo at the start, so Kaggle always sees your latest code.

### 0.2 Get your Kaggle API key
Go to https://www.kaggle.com/settings -> API -> "Create New Token".
Open the downloaded `kaggle.json` — note the `"username"` and `"key"` values.
You'll pass these as command-line arguments on EC2 (script 03), never commit
this file to GitHub (it's already in `.gitignore`).

---

## PHASE 1 — On AWS EC2: build and upload the dataset subset

```bash
cd ~/BigEarthNet-MM

# Step 1: build the 47K-patch manifest (progress bars included)
python3 ~/TerraSynapse/scripts/01_select_subset.py

# Step 2: copy only those patch folders into a clean staging dir
python3 ~/TerraSynapse/scripts/02_copy_subset.py \
    --src_s1 ./BigEarthNet-S1 \
    --src_s2 ./BigEarthNet-S2

# Step 3: upload to Kaggle as a PRIVATE dataset
pip install --upgrade kaggle --quiet
python3 ~/TerraSynapse/scripts/03_upload_kagglehub.py \
    --kaggle_username YOUR_USERNAME \
    --kaggle_key YOUR_API_KEY \
    --handle YOUR_USERNAME/bigearth-mm-subset-47k \
    --local_dir ~/ben_subset_data \
    --version_notes "Initial 47K balanced subset"
```

`--src_s1` and `--src_s2` point to the extracted dataset directories. BigEarthNet
stores patches as `<dataset>/<acquisition>/<patch>`; the script detects this grouped
layout from the first `s1_name` and `patch_id` in `ben_subset.csv` and derives the
acquisition directory for every patch. It also supports a flat `<dataset>/<patch>`
layout. The script stops before copying when the layout cannot be resolved and
fails if any source patches are missing. Do not continue to Step 3 unless both
reported staged-folder counts are nonzero and the missing-source counts are zero.

The upload script passes `-r zip` to `kaggle datasets create/version` so the
`BigEarthNet-S1` and `BigEarthNet-S2` directories are uploaded as ZIP archives
instead of being skipped. With this Kaggle CLI, use `-r zip`; the long option
`--dir-mode zip` is not supported. Do not add the directory option to `du` or
`kaggle datasets init`.

**After this finishes:** open `https://www.kaggle.com/datasets/YOUR_USERNAME/bigearth-mm-subset-47k`
and manually confirm the dataset is marked **Private** before continuing.

---

## PHASE 2 — On Kaggle: three separate notebooks

### Notebook 1 — `01-data-check`
Full cell-by-cell instructions: `notebooks/01_data_check.md`
Purpose: load the manifest, plot class balance, **visually verify** SAR/optical/MS
patches are correctly paired and correctly normalized, compute real MS
normalization stats. **Do not skip this** — catching a band-mismatch bug
here costs 5 minutes; catching it after a failed training run costs hours.

### Notebook 2 — `02-train-retrieval-model`
Full cell-by-cell instructions: `notebooks/02_train_retrieval_model.md`
Purpose: load pretrained BigEarthNet v2.0 backbones (configilm), train only
the projection heads (fast — backbone stays frozen by default), save
`best_model.pt`. Commit the notebook so its Output (the checkpoint) becomes
available to Notebook 3.

Before training, smoke-test direct ConfigILM loading for both pretrained models:

```bash
python3 scripts/05_test_configilm_loader.py \
    --model_id BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0
python3 scripts/05_test_configilm_loader.py \
    --model_id BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0
```

Each command downloads `config.json` and `model.safetensors` from Hugging Face,
constructs ConfigILM without importing `reben_publication`, loads all weights
strictly, and checks a random inference batch.

To try a different backbone (ResNet101, ViT, ConvNeXt, ...): edit
`BACKBONE_KEY_S1` / `BACKBONE_KEY_S2` in `config/config.py`, push to GitHub,
re-run from Cell 1 of Notebook 2. Everything downstream picks it up automatically.

### Notebook 3 — `03-evaluate-retrieval`
Full cell-by-cell instructions: `notebooks/03_evaluate_retrieval.md`
Purpose: encode the test gallery and queries for all three modalities and run
all 7 required retrieval directions through FAISS. Report F1@5, F1@10, Recall@5, Recall@10, mAP@5,
mAP@10, and average retrieval time per query. Saves a results CSV.

---

## PHASE 3 — Comparing multiple backbones (if time allows)

Run Notebook 2 + Notebook 3 once per backbone choice (e.g. resnet50, then
vit_small, then convnext), saving each results CSV with a distinct name
(e.g. `retrieval_results_resnet50.csv`). Combine them into one comparison
table for your final report — this is a strong, easy way to show rigor in
a competition submission without much extra work, since the pipeline
already supports it via the config switch.

---

## Quick reference: what to verify before trusting any number

1. Notebook 1, Cell 5 sanity-check grid — do SAR/optical/MS actually look
   like the same place and the labeled class?
2. Notebook 2 training log — does loss actually decrease over epochs? If
   it's flat near the margin value from epoch 1, the embeddings probably
   aren't separating classes — recheck `models/encoders.py`'s feature
   extraction against the real model structure via `inspect_model_structure()`.
3. Notebook 3 results — is same-modal (MS->MS) F1/Recall noticeably higher
   than cross-modal (SAR->MS)? That's the expected, healthy pattern. If
   cross-modal is *higher* than same-modal, something is leaking (e.g. SAR
   and MS embeddings collapsed to the same vector) and needs investigation.
