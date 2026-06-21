"""UI-free artifact loading, uploaded-image encoding, and FAISS search."""
import json
import os
import time
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rasterio.io import MemoryFile

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
SLUG = {"SAR": "sar", "Optical": "optical", "Multispectral": "multispectral"}
CHANNELS = {"SAR": 2, "Optical": 3, "Multispectral": 10}

def available_backbones(root=ARTIFACT_ROOT):
    files = list((Path(root) / "checkpoints").glob("*_SAR.pt"))
    return sorted(path.name[:-7] for path in files)

def load_artifacts(backbone_key, root=ARTIFACT_ROOT, device="cpu"):
    os.environ.setdefault("BEN_ENV", "local_deploy")
    from models.encoders import build_encoders
    root, encoders = Path(root), build_encoders(torch.device(device))
    indices, labels, embeddings = {}, {}, {}
    for modality, encoder in encoders.items():
        checkpoint = torch.load(root / "checkpoints" / f"{backbone_key}_{modality}.pt", map_location=device)
        encoder.load_state_dict(checkpoint); encoder.eval()
        slug = SLUG[modality]
        indices[modality] = faiss.read_index(str(root / "faiss_index" / f"{slug}_{backbone_key}.index"))
        labels[modality] = pd.read_csv(root / "gallery_labels" / f"{slug}_{backbone_key}.csv")
        embeddings[modality] = np.load(root / "gallery_embeddings" / f"{slug}_{backbone_key}.npy", mmap_mode="r")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"encoders": encoders, "indices": indices, "labels": labels, "embeddings": embeddings,
            "metadata": metadata, "root": root, "backbone_key": backbone_key, "device": device}

def _read_upload(upload, modality):
    raw = upload.getvalue() if hasattr(upload, "getvalue") else Path(upload).read_bytes()
    name = getattr(upload, "name", str(upload)).lower()
    if name.endswith((".tif", ".tiff")):
        with MemoryFile(raw) as mem, mem.open() as src: array = src.read().astype(np.float32)
    else:
        if modality != "Optical": raise ValueError(f"{modality} requires a {CHANNELS[modality]}-band GeoTIFF; JPG/PNG is RGB only.")
        import io
        array = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    if array.shape[0] != CHANNELS[modality]:
        raise ValueError(f"Expected {CHANNELS[modality]} bands for {modality}, but received {array.shape[0]}.")
    if modality == "SAR":
        array = np.clip((10 * np.log10(np.abs(array) + 1e-10) + 30) / 30, 0, 1)
    elif modality == "Optical":
        lo, hi = np.percentile(array, (2, 98)); array = np.clip((array - lo) / (hi - lo + 1e-6), 0, 1)
        array = (array - np.array([.485, .456, .406])[:, None, None]) / np.array([.229, .224, .225])[:, None, None]
    else:
        from data.preprocessing import MS_MEAN, MS_STD
        array = (np.clip(array, 0, 10000) - MS_MEAN[:, None, None]) / (MS_STD[:, None, None] + 1e-6)
    return torch.from_numpy(array.astype(np.float32))

def encode_query(upload, modality, backbone_key, artifacts):
    if backbone_key != artifacts["backbone_key"]: raise ValueError("Loaded artifacts do not match the selected backbone")
    tensor = _read_upload(upload, modality).unsqueeze(0).to(artifacts["device"])
    with torch.no_grad(): embedding = artifacts["encoders"][modality](tensor).cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)
    return embedding

def sample_embedding(patch_id, modality, artifacts):
    rows = artifacts["labels"][modality]
    matches = np.flatnonzero(rows["patch_id"].astype(str).to_numpy() == str(patch_id))
    if not len(matches): raise ValueError(f"Sample {patch_id} is not present in the {modality} gallery")
    return np.asarray(artifacts["embeddings"][modality][matches[0]:matches[0]+1], dtype="float32")

def search(query_embedding, target_modality, backbone_key, k, artifacts):
    if backbone_key != artifacts["backbone_key"]: raise ValueError("Loaded artifacts do not match the selected backbone")
    started = time.perf_counter(); scores, ids = artifacts["indices"][target_modality].search(query_embedding, k)
    elapsed = (time.perf_counter() - started) * 1000; table = artifacts["labels"][target_modality]
    results = []
    for score, idx in zip(scores[0], ids[0]):
        row = table.iloc[int(idx)]; patch_id = str(row["patch_id"])
        thumb = artifacts["root"] / "sample_thumbnails" / target_modality / f"{patch_id}.png"
        results.append((patch_id, float(score), str(row.get("primary_label", "Unknown")), thumb if thumb.exists() else None))
    return results, elapsed
