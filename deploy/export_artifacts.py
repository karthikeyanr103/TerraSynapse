"""Export a compact test-gallery bundle after Kaggle training/evaluation."""
import argparse
import json
import shutil
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from config.config import METADATA, S1_ROOT, S2_ROOT, KEEP_CLASSES, BACKBONE_KEY_S1
from data.preprocessing import load_sar, load_optical_rgb, load_ms
from models.encoders import build_encoders

LOADERS = {"SAR": lambda r: load_sar(r.s1_name), "Optical": lambda r: load_optical_rgb(r.patch_id), "Multispectral": lambda r: load_ms(r.patch_id)}
SLUG = {"SAR": "sar", "Optical": "optical", "Multispectral": "multispectral"}

def _stretch(array):
    lo, hi = np.percentile(array, (2, 98)); return (np.clip((array-lo)/(hi-lo+1e-6), 0, 1)*255).astype("uint8")

def _thumbnail(row, modality):
    import rasterio
    if modality == "SAR":
        with rasterio.open(S1_ROOT / row.s1_name / f"{row.s1_name}_VV.tif") as src: raw = src.read(1).astype("float32")
        gray = _stretch(10*np.log10(np.abs(raw)+1e-10)); return np.repeat(gray[..., None], 3, axis=2)
    bands = ["B04", "B03", "B02"] if modality == "Optical" else ["B08", "B04", "B03"]
    arrays = []
    for band in bands:
        with rasterio.open(S2_ROOT / row.patch_id / f"{row.patch_id}_{band}.tif") as src: arrays.append(src.read(1).astype("float32"))
    return _stretch(np.stack(arrays, axis=-1))

def export(checkpoint_path, output, backbone_key=BACKBONE_KEY_S1, per_class=20):
    output = Path(output); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = build_encoders(device); checkpoint = torch.load(checkpoint_path, map_location=device)
    states = checkpoint.get("encoders")
    if states is None: raise ValueError("Checkpoint uses the old two-encoder schema; retrain with the v2 pipeline.")
    for modality, encoder in encoders.items(): encoder.load_state_dict(states[modality]); encoder.eval()
    test = pd.read_csv(METADATA).query("split == 'test'").reset_index(drop=True)
    for name in ("checkpoints", "gallery_embeddings", "gallery_labels", "faiss_index", "sample_thumbnails"):
        (output / name).mkdir(parents=True, exist_ok=True)
    for modality, encoder in encoders.items():
        embeddings = []
        with torch.no_grad():
            for row in tqdm(test.itertuples(index=False), total=len(test), desc=f"Encoding {modality}", unit="patch"):
                embeddings.append(encoder(LOADERS[modality](row).unsqueeze(0).to(device)).cpu().numpy()[0])
        matrix = np.asarray(embeddings, dtype="float32"); faiss.normalize_L2(matrix); slug = SLUG[modality]
        np.save(output / "gallery_embeddings" / f"{slug}_{backbone_key}.npy", matrix)
        test[["patch_id", "primary_label"]].to_csv(output / "gallery_labels" / f"{slug}_{backbone_key}.csv", index=False)
        index = faiss.IndexFlatIP(matrix.shape[1]); index.add(matrix)
        faiss.write_index(index, str(output / "faiss_index" / f"{slug}_{backbone_key}.index"))
        torch.save(encoder.state_dict(), output / "checkpoints" / f"{backbone_key}_{modality}.pt")
        chosen = pd.concat([group.sample(min(per_class, len(group)), random_state=42) for _, group in test.groupby("primary_label")])
        thumb_dir = output / "sample_thumbnails" / modality; thumb_dir.mkdir(parents=True, exist_ok=True)
        for row in tqdm(chosen.itertuples(index=False), total=len(chosen), desc=f"Thumbnails {modality}", unit="image"):
            Image.fromarray(_thumbnail(row, modality)).resize((128, 128)).save(thumb_dir / f"{row.patch_id}.png")
    (output / "metadata.json").write_text(json.dumps({"dataset": "BigEarthNet-MM curated 47K subset", "backbone_key": backbone_key, "classes": KEEP_CLASSES}, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    print(f"Export complete: {archive}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", default="deploy/artifacts"); parser.add_argument("--backbone-key", default=BACKBONE_KEY_S1); parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args(); export(args.checkpoint, args.output, args.backbone_key, args.per_class)
