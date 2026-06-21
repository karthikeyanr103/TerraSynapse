"""Triplet sampling across three independent modalities."""
import numpy as np
from torch.utils.data import Dataset
from config.config import KEEP_CLASSES, CLASS_TO_IDX, CROSS_MODAL_RATIO, MODALITIES
from data.preprocessing import load_sar, load_optical_rgb, load_ms

class BigEarthMMDataset(Dataset):
    def __init__(self, df, split, cross_modal_ratio=CROSS_MODAL_RATIO, seed=0):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.ratio, self.rng = cross_modal_ratio, np.random.default_rng(seed)
        self.cls_idx = {c: self.df.index[self.df["primary_label"] == c].tolist() for c in KEEP_CLASSES}
        missing = [c for c, ids in self.cls_idx.items() if not ids]
        if missing: raise ValueError(f"Split {split!r} has no samples for {missing}")
    def __len__(self): return len(self.df)
    @staticmethod
    def _load(row, modality):
        if modality == "SAR": return load_sar(row["s1_name"])
        if modality == "Optical": return load_optical_rgb(row["patch_id"])
        if modality == "Multispectral": return load_ms(row["patch_id"])
        raise ValueError(f"Unknown modality: {modality}")
    def __getitem__(self, idx):
        row, cls = self.df.iloc[idx], self.df.iloc[idx]["primary_label"]
        anchor_mod = str(self.rng.choice(MODALITIES))
        if self.rng.random() < self.ratio:
            pos_mod = str(self.rng.choice([m for m in MODALITIES if m != anchor_mod])); pos_row = row
        else:
            candidates = [i for i in self.cls_idx[cls] if i != idx] or self.cls_idx[cls]
            pos_mod, pos_row = anchor_mod, self.df.iloc[self.rng.choice(candidates)]
        neg_cls = str(self.rng.choice([c for c in KEEP_CLASSES if c != cls]))
        neg_row = self.df.iloc[self.rng.choice(self.cls_idx[neg_cls])]
        neg_mod = str(self.rng.choice(MODALITIES))
        return {"anchor": self._load(row, anchor_mod), "positive": self._load(pos_row, pos_mod),
                "negative": self._load(neg_row, neg_mod), "anchor_cls": CLASS_TO_IDX[cls],
                "anchor_mod": anchor_mod, "pos_mod": pos_mod, "neg_mod": neg_mod}
