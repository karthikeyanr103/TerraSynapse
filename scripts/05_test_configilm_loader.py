"""Download one BigEarthNet checkpoint and smoke-test the direct ConfigILM loader."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoders import load_pretrained_classifier


DEFAULT_MODEL = "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_pretrained_classifier(args.model_id).eval()
    config = model.config
    sample = torch.randn(
        args.batch_size,
        config.channels,
        config.image_size,
        config.image_size,
    )

    with torch.inference_mode():
        logits = model(sample)

    expected_shape = (args.batch_size, config.classes)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(
            f"Expected output shape {expected_shape}, got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise AssertionError("Model output contains NaN or infinite values")

    print(f"Loaded: {args.model_id}")
    print(f"Input shape:  {tuple(sample.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print("Direct ConfigILM pretrained-weight test passed.")


if __name__ == "__main__":
    main()
