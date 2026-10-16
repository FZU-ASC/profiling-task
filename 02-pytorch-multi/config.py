"""Arguments shared by the image-training pipeline."""
import argparse
from pathlib import Path


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Profile a modular residual CNN training pipeline.")
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--steps", type=positive, default=8)
    parser.add_argument("--warmup", type=positive, default=2)
    parser.add_argument("--image-size", type=positive, default=32)
    parser.add_argument("--threads", type=positive, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/pytorch-multi"))
    args = parser.parse_args()
    if args.image_size < 8:
        parser.error("--image-size must be at least 8")
    return args
