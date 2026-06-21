"""
scripts/02_copy_subset.py
Run SECOND, on EC2. Copies only the patch folders listed in ben_subset.csv
into a clean staging directory, ready to be zipped/uploaded to Kaggle.

Usage:
    cd ~/BigEarthNet-MM
    python3 /path/to/scripts/02_copy_subset.py \
        --src_s1 BigEarthNet-S1 \
        --src_s2 BigEarthNet-S2
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
        default=Path("~/BigEarthNet-MM/BigEarthNet-S1"),
        help="Directory containing the S1 patch folders",
    )
    parser.add_argument(
        "--src_s2",
        type=Path,
        default=Path("~/BigEarthNet-MM/BigEarthNet-S2"),
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


def detect_grouping_depth(source_root, first_patch_name, label):
    """Detect whether patches are direct or grouped below an acquisition folder."""
    direct_patch = source_root / first_patch_name
    if direct_patch.is_dir():
        print(f"Detected flat {label} patch layout under {source_root}")
        return 0

    name_parts = first_patch_name.split("_")
    for trailing_parts in range(1, min(8, len(name_parts))):
        group_name = "_".join(name_parts[:-trailing_parts])
        grouped_patch = source_root / group_name / first_patch_name
        if grouped_patch.is_dir():
            print(
                f"Detected grouped {label} layout: "
                f"{source_root}/<acquisition>/<patch>"
            )
            return trailing_parts

    raise FileNotFoundError(
        f"Could not resolve {label} patch folder {first_patch_name!r} under "
        f"{source_root}. Check the manifest and extracted dataset."
    )


def source_patch_path(source_root, patch_name, trailing_parts):
    """Build a patch path using the layout detected from the first manifest row."""
    if trailing_parts == 0:
        return source_root / patch_name
    group_name = "_".join(patch_name.split("_")[:-trailing_parts])
    return source_root / group_name / patch_name


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
    required_columns = {"s1_name", "patch_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )
    if df.empty:
        raise ValueError(f"Manifest contains no rows: {manifest}")

    s1_grouping = detect_grouping_depth(
        src_s1, str(df.iloc[0]["s1_name"]), "S1"
    )
    s2_grouping = detect_grouping_depth(
        src_s2, str(df.iloc[0]["patch_id"]), "S2"
    )

    print(f"Copying {len(df)} SAR (S1) folders ...")
    missing_s1 = 0
    for s1_name in tqdm(df["s1_name"], desc="S1 folders", unit="patch"):
        src = source_patch_path(src_s1, s1_name, s1_grouping)
        dst = dst_s1 / s1_name
        if not src.exists():
            missing_s1 += 1
            continue
        if not dst.exists():
            shutil.copytree(src, dst)

    print(f"\nCopying {len(df)} Multispectral (S2) folders ...")
    missing_s2 = 0
    for patch_id in tqdm(df["patch_id"], desc="S2 folders", unit="patch"):
        src = source_patch_path(src_s2, patch_id, s2_grouping)
        dst = dst_s2 / patch_id
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
