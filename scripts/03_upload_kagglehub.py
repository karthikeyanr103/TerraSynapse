"""
scripts/03_upload_kaggle_cli.py

Run THIRD, on EC2, after 01_select_subset.py and
02_copy_subset.py have completed.

Creates a low-memory ZIP of ~/ben_subset_data/, then uploads that ZIP with the
Kaggle CLI. The native zip process avoids Kaggle CLI directory archiving, which
can exhaust memory on small EC2 instances.

First upload:
    kaggle datasets create

Later uploads:
    kaggle datasets version

Usage:
    python3 scripts/03_upload_kaggle_cli.py \
        --kaggle_username YOUR_USERNAME \
        --kaggle_key YOUR_API_KEY \
        --handle YOUR_USERNAME/bigearth-mm-subset-47k \
        --local_dir ~/ben_subset_data \
        --version_notes "Initial 47K balanced subset"

The dataset is created as private because the command does not use --public.
Confirm the visibility manually after upload.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a local dataset folder to Kaggle using Kaggle CLI"
    )

    parser.add_argument(
        "--kaggle_username",
        required=True,
        help="Your Kaggle username",
    )

    parser.add_argument(
        "--kaggle_key",
        required=True,
        help="Your Kaggle API key",
    )

    parser.add_argument(
        "--handle",
        required=True,
        help=(
            "Kaggle dataset handle, for example "
            "myusername/bigearth-mm-subset-47k"
        ),
    )

    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local folder to upload, for example ~/ben_subset_data",
    )

    parser.add_argument(
        "--version_notes",
        default="dataset update",
        help="Version notes for this upload",
    )

    return parser.parse_args()


def run_command(command, cwd=None):
    """Run a shell command and stop if it fails."""

    print("\nRunning command:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
        text=True,
        cwd=cwd,
    )


def kaggle_dataset_exists(handle):
    """
    Check whether the Kaggle dataset already exists.

    Returns True when the dataset can be accessed.
    """

    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "files",
            handle,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    return result.returncode == 0


def create_or_update_metadata(local_dir, handle):
    """
    Create dataset-metadata.json if it does not exist,
    then set the correct dataset ID and title.
    """

    metadata_path = local_dir / "dataset-metadata.json"

    if not metadata_path.exists():
        run_command(
            [
                "kaggle",
                "datasets",
                "init",
                "-p",
                str(local_dir),
            ]
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    dataset_slug = handle.split("/", 1)[1]

    metadata["id"] = handle
    metadata["title"] = dataset_slug

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nDataset metadata:")
    print(json.dumps(metadata, indent=2))

    return metadata_path


def prepare_zip_upload(local_dir, metadata_path):
    """Create one streaming ZIP and a flat directory for Kaggle upload."""
    if shutil.which("zip") is None:
        raise RuntimeError(
            "The native zip command is required. Install it with: sudo apt install zip"
        )

    upload_dir = local_dir.parent / f"{local_dir.name}_kaggle_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    archive_path = upload_dir / f"{local_dir.name}.zip"

    if archive_path.exists():
        print(f"\nRemoving previous archive: {archive_path}")
        archive_path.unlink()

    shutil.copy2(metadata_path, upload_dir / "dataset-metadata.json")

    print(
        f"\nCreating low-memory ZIP: {archive_path}\n"
        "The archive contains the dataset contents directly, without an extra "
        f"{local_dir.name}/ directory."
    )
    run_command(
        [
            "zip",
            "-q",
            "-r",
            "-1",
            str(archive_path),
            ".",
        ],
        cwd=str(local_dir),
    )

    print("\nPrepared archive size:")
    run_command(["du", "-sh", str(archive_path)])
    return upload_dir, archive_path


def main():
    args = parse_args()

    # Kaggle CLI reads these exact environment variables.
    os.environ["KAGGLE_USERNAME"] = args.kaggle_username
    os.environ["KAGGLE_KEY"] = args.kaggle_key

    local_dir = Path(args.local_dir).expanduser().resolve()

    if not local_dir.exists():
        raise FileNotFoundError(
            f"local_dir does not exist: {local_dir}"
        )

    if not local_dir.is_dir():
        raise NotADirectoryError(
            f"local_dir is not a directory: {local_dir}"
        )

    if "/" not in args.handle:
        raise ValueError(
            "--handle must use the format username/dataset-name"
        )

    handle_username = args.handle.split("/", 1)[0]

    if handle_username != args.kaggle_username:
        print(
            "\nWarning: the username in --handle is different "
            "from --kaggle_username."
        )

    print(f"\nLocal dataset folder: {local_dir}")
    print(f"Kaggle dataset handle: {args.handle}")

    print("\nDataset size:")
    run_command(
        [
            "du",
            "-sh",
            str(local_dir),
        ]
    )

    metadata_path = create_or_update_metadata(
        local_dir=local_dir,
        handle=args.handle,
    )

    print(f"\nMetadata file created at: {metadata_path}")

    upload_dir, archive_path = prepare_zip_upload(local_dir, metadata_path)
    print(f"\nKaggle upload directory: {upload_dir}")
    print(f"Kaggle archive file: {archive_path}")

    if kaggle_dataset_exists(args.handle):
        print(
            "\nThe Kaggle dataset already exists."
            "\nUploading a new dataset version..."
        )

        run_command(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(upload_dir),
                "-m",
                args.version_notes,
                "-r",
                "skip",
            ]
        )

        print("\nNew Kaggle dataset version uploaded successfully.")

    else:
        print(
            "\nThe Kaggle dataset does not exist."
            "\nCreating a new private dataset..."
        )

        run_command(
            [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(upload_dir),
                "-r",
                "skip",
            ]
        )

        print("\nKaggle dataset created successfully.")

    print(
        f"\nDataset page:\n"
        f"https://www.kaggle.com/datasets/{args.handle}"
    )

    print(
        "\nThe script did not use --public. "
        "Confirm that the Kaggle dataset visibility is Private."
    )


if __name__ == "__main__":
    main()
