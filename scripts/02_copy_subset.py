"""
scripts/02_copy_subset.py
Run SECOND, on EC2. Copies only the patch folders listed in ben_subset.csv
into a clean staging directory, ready to be zipped/uploaded to Kaggle.

Usage:
    cd ~/BigEarthNet-MM
    python3 /path/to/scripts/02_copy_subset.py
"""

import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

SRC_S1 = Path("BigEarthNet-S1")
SRC_S2 = Path("BigEarthNet-S2")
DST_ROOT = Path.home() / "ben_subset_data"
DST_S1 = DST_ROOT / "BigEarthNet-S1"
DST_S2 = DST_ROOT / "BigEarthNet-S2"


def main():
    DST_S1.mkdir(parents=True, exist_ok=True)
    DST_S2.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("ben_subset.csv")

    print(f"Copying {len(df)} SAR (S1) folders ...")
    missing_s1 = 0
    for s1_name in tqdm(df["s1_name"], desc="S1 folders", unit="patch"):
        src = SRC_S1 / s1_name
        dst = DST_S1 / s1_name
        if not src.exists():
            missing_s1 += 1
            continue
        if not dst.exists():
            shutil.copytree(src, dst)

    print(f"\nCopying {len(df)} Multispectral (S2) folders ...")
    missing_s2 = 0
    for patch_id in tqdm(df["patch_id"], desc="S2 folders", unit="patch"):
        src = SRC_S2 / patch_id
        dst = DST_S2 / patch_id
        if not src.exists():
            missing_s2 += 1
            continue
        if not dst.exists():
            shutil.copytree(src, dst)

    # Copy manifest into the staging dir too
    shutil.copy("ben_subset.csv", DST_ROOT / "ben_subset.csv")

    print("\nDone. Verifying counts ...")
    s1_count = len(list(DST_S1.iterdir()))
    s2_count = len(list(DST_S2.iterdir()))
    print(f"  S1 folders copied: {s1_count}  (missing source: {missing_s1})")
    print(f"  S2 folders copied: {s2_count}  (missing source: {missing_s2})")

    print("\nChecking total staged size ...")
    import subprocess
    subprocess.run(["du", "-sh", str(DST_ROOT)])
    subprocess.run(["du", "-sh", str(DST_S1)])
    subprocess.run(["du", "-sh", str(DST_S2)])


if __name__ == "__main__":
    main()
