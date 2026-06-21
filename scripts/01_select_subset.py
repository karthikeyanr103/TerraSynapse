"""
scripts/01_select_subset.py
Run this FIRST, on EC2, from the BigEarthNet-MM root directory.

Builds ben_subset.csv — the manifest of the ~47,498 patches you'll actually use.
Does NOT copy any image files yet (that's script 02).

Usage:
    cd ~/BigEarthNet-MM
    python3 /path/to/scripts/01_select_subset.py
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

KEEP_CLASSES = [
    "Arable land", "Broad-leaved forest", "Coniferous forest",
    "Mixed forest", "Pastures", "Urban fabric",
    "Industrial or commercial units", "Inland waters",
    "Complex cultivation patterns", "Transitional woodland, shrub",
]
N_CAP_PER_CLASS = 5000
RANDOM_STATE = 42


def main():
    print("Loading metadata.parquet ...")
    df = pd.read_parquet("metadata.parquet")
    print(f"  Total patches in full dataset: {len(df)}")

    print("\nFiltering patches that contain at least one KEEP class ...")
    tqdm.pandas(desc="Scanning labels")
    mask = df["labels"].progress_apply(lambda lbls: any(l in KEEP_CLASSES for l in lbls))
    focused = df[mask].copy()
    print(f"  Patches with >=1 keep-class label: {len(focused)}")

    print("\nSampling up to N_CAP_PER_CLASS per class ...")
    selected_ids = set()
    for cls in tqdm(KEEP_CLASSES, desc="Sampling classes"):
        cls_df = focused[focused["labels"].apply(lambda x: cls in x)]
        n = min(N_CAP_PER_CLASS, len(cls_df))
        sampled = cls_df["patch_id"].sample(n, random_state=RANDOM_STATE).tolist()
        selected_ids.update(sampled)

    subset = focused[focused["patch_id"].isin(selected_ids)].copy()

    print("\nAssigning primary_label ...")
    tqdm.pandas(desc="Assigning labels")
    subset["primary_label"] = subset["labels"].progress_apply(
        lambda x: next((l for l in x if l in KEEP_CLASSES), x[0])
    )
    subset["labels_str"] = subset["labels"].apply(lambda x: "|".join(x))

    out_cols = ["patch_id", "s1_name", "s2v1_name", "split", "country",
                "primary_label", "labels_str"]
    subset[out_cols].to_csv("ben_subset.csv", index=False)

    print(f"\nSaved ben_subset.csv: {len(subset)} patches")
    print(subset["split"].value_counts())
    print("\nPer-class counts:")
    print(subset["primary_label"].value_counts())


if __name__ == "__main__":
    main()
