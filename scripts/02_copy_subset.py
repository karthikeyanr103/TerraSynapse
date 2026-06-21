"""
scripts/02_copy_subset.py
Run SECOND, on EC2. Copies only the patch folders listed in ben_subset.csv
into a clean staging directory, ready to be zipped/uploaded to Kaggle.

Usage:
    cd ~/BigEarthNet-MM
    python3 /path/to/scripts/02_copy_subset.py \
        --src_s1 BigEarthNet-S1-v1.0 \
        --src_s2 BigEarthNet-v1.0
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy the selected BigEarthNet-MM patches into a staging directory"
    )
    parser.add_argument(
        "--src_s1",
        type=Path,
        default=Path("BigEarthNet-S1-v1.0"),
        help="Directory containing the S1 patch folders",
    )
    parser.add_argument(
        "--src_s2",
        type=Path,
        default=Path("BigEarthNet-v1.0"),
        help="Directory containing the S2 v1 patch folders",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ben_subset.csv"),
        help="Subset CSV produced by 01_select_subset.py",
    )
    parser.add_argument(
        "--dst_root",
        type=Path,
        default=Path.home() / "ben_subset_data",
        help="Kaggle upload staging directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_s1 = args.src_s1.expanduser().resolve()
    src_s2 = args.src_s2.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()
    dst_s1 = dst_root / "BigEarthNet-S1"
    dst_s2 = dst_root / "BigEarthNet-S2"

    for label, path in (("S1", src_s1), ("S2", src_s2)):
        if not path.is_dir():
            raise NotADirectoryError(
                f"{label} source directory not found: {path}. "
                f"Pass the correct location with --src_{label.lower()}."
            )
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    dst_s1.mkdir(parents=True, exist_ok=True)
    dst_s2.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)

    print(f"Copying {len(df)} SAR (S1) folders ...")
    missing_s1 = 0
    for s1_name in tqdm(df["s1_name"], desc="S1 folders", unit="patch"):
        src = src_s1 / s1_name
        dst = dst_s1 / s1_name
        if not src.exists():
            missing_s1 += 1
            continue
        if not dst.exists():
            shutil.copytree(src, dst)

    print(f"\nCopying {len(df)} Multispectral (S2) folders ...")
    missing_s2 = 0
    for s2_name in tqdm(df["s2v1_name"], desc="S2 folders", unit="patch"):
        src = src_s2 / s2_name
        dst = dst_s2 / s2_name
        if not src.exists():
            missing_s2 += 1
            continue
        if not dst.exists():
            shutil.copytree(src, dst)

    # Copy manifest into the staging dir too
    shutil.copy(manifest, dst_root / "ben_subset.csv")

    print("\nDone. Verifying counts ...")
    s1_count = sum(1 for path in dst_s1.iterdir() if path.is_dir())
    s2_count = sum(1 for path in dst_s2.iterdir() if path.is_dir())
    print(f"  S1 folders copied: {s1_count}  (missing source: {missing_s1})")
    print(f"  S2 folders copied: {s2_count}  (missing source: {missing_s2})")

    if missing_s1 or missing_s2:
        raise RuntimeError(
            "Some source patches were not found; refusing to stage an incomplete dataset. "
            "Check --src_s1 and --src_s2."
        )

    print("\nChecking total staged size ...")
    import subprocess
    subprocess.run(["du", "-sh", str(dst_root)])
    subprocess.run(["du", "-sh", str(dst_s1)])
    subprocess.run(["du", "-sh", str(dst_s2)])


if __name__ == "__main__":
    main()
