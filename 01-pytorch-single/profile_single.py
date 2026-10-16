"""A small MLP inference workload; run on Linux/macOS, CPU by default."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive, default=64)
    parser.add_argument("--steps", type=positive, default=12)
    parser.add_argument("--warmup", type=positive, default=4)
    parser.add_argument("--threads", type=positive, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/pytorch-single"))
    args = parser.parse_args()
    if platform.system() not in ("Linux", "Darwin"):
        parser.error("Please run this experiment on Linux/macOS (Windows: use WSL2).")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or configure CUDA-enabled PyTorch.")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 64))
    model = model.to(args.device).eval()
    inputs = torch.randn(args.batch_size, 256, device=args.device)
    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            parser.error("CUDA profiling unavailable in this PyTorch environment.")
        activities.append(ProfilerActivity.CUDA)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs)
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
            for _ in range(args.steps):
                with record_function("inference_step"):
                    with record_function("mlp_forward"):
                        output = model(inputs)
            if args.device == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    (args.output / "operators.txt").write_text(table, encoding="utf-8")
    prof.export_chrome_trace(str(args.output / "trace.json"))
    summary = {
        "workload": "single_mlp_inference", "device": args.device,
        "batch_size": args.batch_size, "steps": args.steps,
        "warmup": args.warmup, "threads": args.threads, "seed": 42,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "platform": platform.platform(), "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "output_shape": list(output.shape), "checksum": output.sum().item(),
        "profiled_wall_seconds": elapsed,
        "timing_note": "Includes profiler overhead; not a benchmark.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table)
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
