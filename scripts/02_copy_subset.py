"""
scripts/02_copy_subset.py
Run SECOND, on EC2. Reads the selected patch folders listed in ben_subset.csv
and writes them STRAIGHT into compressed tar.gz archives — no intermediate
uncompressed copy — ready to upload to Kaggle via kagglehub.

Auto-detects whether S1/S2 patches sit directly under the source root or are
grouped one level down under an acquisition-level folder (reBEN's actual
layout is grouped — see detect_grouping_depth below).

Output (in --dst_root):
    ben_subset.csv
    BigEarthNet-S1.tar.gz   (internal paths flattened to BigEarthNet-S1/{s1_name}/...)
    BigEarthNet-S2.tar.gz   (internal paths flattened to BigEarthNet-S2/{patch_id}/...)

The flattening (via tarfile's arcname) means the rest of the pipeline can
always expect S1_ROOT/{s1_name}/ and S2_ROOT/{patch_id}/ after extraction,
regardless of how the source dataset nests its acquisition folders.

Usage:
    cd ~/BigEarthNet-MM
    python3 /path/to/scripts/02_copy_subset.py \\
        --src_s1 BigEarthNet-S1 \\
        --src_s2 BigEarthNet-S2
"""

import argparse
import tarfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tar+compress the selected BigEarthNet-MM patches into a staging directory"
    )
    parser.add_argument("--src_s1", type=Path, default=Path("BigEarthNet-S1"),
                         help="Directory containing the S1 patch folders")
    parser.add_argument("--src_s2", type=Path, default=Path("BigEarthNet-S2"),
                         help="Directory containing the S2 patch folders")
    parser.add_argument("--manifest", type=Path, default=Path("ben_subset.csv"),
                         help="Subset CSV produced by 01_select_subset.py")
    parser.add_argument("--dst_root", type=Path, default=Path.home() / "ben_subset_data",
                         help="Kaggle upload staging directory")
    parser.add_argument("--compresslevel", type=int, default=6,
                         help="gzip compression level 1 (fastest) - 9 (smallest). Default 6.")
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
            print(f"Detected grouped {label} layout: {source_root}/<acquisition>/<patch>")
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


def tar_patches(patch_names, source_root, trailing_parts, tar_path, arc_prefix, desc):
    """
    Stream each selected patch folder straight into a gzip-compressed tar,
    flattening its arcname to f"{arc_prefix}/{patch_name}/...". Returns the
    count of patches that were missing from the source (not added).
    """
    missing = 0
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tar:
        for patch_name in tqdm(patch_names, desc=desc, unit="patch"):
            src = source_patch_path(source_root, patch_name, trailing_parts)
            if not src.exists():
                missing += 1
                continue
            tar.add(src, arcname=f"{arc_prefix}/{patch_name}")
    return missing


def main():
    args = parse_args()
    src_s1 = args.src_s1.expanduser().resolve()
    src_s2 = args.src_s2.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    tar_s1_path = dst_root / "BigEarthNet-S1.tar.gz"
    tar_s2_path = dst_root / "BigEarthNet-S2.tar.gz"

    for label, path in (("S1", src_s1), ("S2", src_s2)):
        if not path.is_dir():
            raise NotADirectoryError(
                f"{label} source directory not found: {path}. "
                f"Pass the correct location with --src_{label.lower()}."
            )
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest)
    # NOTE: S2 folders on disk are named by patch_id (e.g.
    # S2A_MSIL2A_20180506T100031_N9999_R122_T33UWP_83_87), NOT s2v1_name
    # (a shorter, legacy BigEarthNet-v1-style identifier that does not
    # correspond to any folder in the reBEN/v2.0 layout) — always use
    # patch_id for S2 lookups.
    required_columns = {"s1_name", "patch_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError(f"Manifest contains no rows: {manifest}")

    s1_grouping = detect_grouping_depth(src_s1, str(df.iloc[0]["s1_name"]), "S1")
    s2_grouping = detect_grouping_depth(src_s2, str(df.iloc[0]["patch_id"]), "S2")

    print(f"\nCompressing {len(df)} SAR (S1) folders -> {tar_s1_path.name} ...")
    missing_s1 = tar_patches(
        df["s1_name"].tolist(), src_s1, s1_grouping, tar_s1_path,
        arc_prefix="BigEarthNet-S1", desc="S1 -> tar.gz",
    )

    print(f"\nCompressing {len(df)} Multispectral (S2) folders -> {tar_s2_path.name} ...")
    missing_s2 = tar_patches(
        df["patch_id"].tolist(), src_s2, s2_grouping, tar_s2_path,
        arc_prefix="BigEarthNet-S2", desc="S2 -> tar.gz",
    )

    # Copy manifest into the staging dir too
    import shutil
    shutil.copy(manifest, dst_root / "ben_subset.csv")

    print(f"\nDone.")
    print(f"  S1 patches missing from source: {missing_s1}")
    print(f"  S2 patches missing from source: {missing_s2}")

    if missing_s1 or missing_s2:
        raise RuntimeError(
            "Some source patches were not found; refusing to ship an incomplete dataset. "
            "Check --src_s1 and --src_s2."
        )

    print("\nFinal archive sizes:")
    for p in (tar_s1_path, tar_s2_path, dst_root / "ben_subset.csv"):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}: {size_mb:,.1f} MB")


if __name__ == "__main__":
    main()
