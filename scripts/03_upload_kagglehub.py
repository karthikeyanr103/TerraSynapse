"""
scripts/03_upload_kagglehub.py
Run THIRD, on EC2, after 01_select_subset.py and 02_copy_subset.py have completed.

Uploads ~/ben_subset_data/ to Kaggle as a private dataset using kagglehub
(no kaggle.json file needed — credentials are passed as CLI args and set as
the env vars kagglehub actually expects: KAGGLE_USERNAME and KAGGLE_KEY).

Usage:
    python3 scripts/03_upload_kagglehub.py \\
        --kaggle_username YOUR_USERNAME \\
        --kaggle_key YOUR_API_KEY \\
        --handle YOUR_USERNAME/bigearth-mm-subset-47k \\
        --local_dir ~/ben_subset_data \\
        --version_notes "Initial 47K balanced subset"

Get your API key from: https://www.kaggle.com/settings -> API -> Create New Token
(open the downloaded kaggle.json — it contains "username" and "key")

NOTE ON PRIVACY:
Kaggle datasets are created as PRIVATE by default when made via the API/kagglehub.
After the first upload finishes, open the dataset page on kaggle.com and confirm
the visibility toggle still says "Private" before you move on — don't rely on
this script alone to guarantee it, double-check manually once.
"""

import argparse
import os
from pathlib import Path

import kagglehub


def parse_args():
    p = argparse.ArgumentParser(description="Upload a local dataset folder to Kaggle via kagglehub")
    p.add_argument("--kaggle_username", required=True, help="Your Kaggle username")
    p.add_argument("--kaggle_key", required=True, help="Your Kaggle API key")
    p.add_argument("--handle", required=True,
                   help="Kaggle dataset handle, e.g. myusername/bigearth-mm-subset-47k")
    p.add_argument("--local_dir", required=True, help="Local folder to upload (e.g. ~/ben_subset_data)")
    p.add_argument("--version_notes", default="dataset update", help="Version notes for this upload")
    p.add_argument("--ignore", nargs="*", default=["*.tmp", "*.log"],
                   help="Extra glob patterns to exclude from the upload")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Set the credentials kagglehub / kaggle actually look for ──────────
    # (corrects the common typo of KAGLE_SUERNAME / KAGLE_KEY -> these exact names)
    os.environ["KAGGLE_USERNAME"] = args.kaggle_username
    os.environ["KAGGLE_KEY"] = args.kaggle_key

    local_dir = Path(args.local_dir).expanduser()
    if not local_dir.exists():
        raise FileNotFoundError(f"local_dir does not exist: {local_dir}")

    print(f"Uploading '{local_dir}' to Kaggle dataset handle '{args.handle}' ...")
    print("This can take a while for large folders — progress is shown by kagglehub itself.")

    try:
        # First-time creation of the dataset
        kagglehub.dataset_upload(
            args.handle,
            str(local_dir),
            ignore_patterns=args.ignore,
        )
        print("\nDataset created successfully.")
    except Exception as e:
        # If it already exists, push a new version instead
        print(f"\ndataset_upload (create) failed or dataset already exists: {e}")
        print("Trying to upload a new version instead ...")
        kagglehub.dataset_upload(
            args.handle,
            str(local_dir),
            version_notes=args.version_notes,
            ignore_patterns=args.ignore,
        )
        print("\nNew dataset version uploaded successfully.")

    print(f"\nDone. View it at: https://www.kaggle.com/datasets/{args.handle}")
    print("Reminder: confirm visibility is still set to PRIVATE on that page.")


if __name__ == "__main__":
    main()
