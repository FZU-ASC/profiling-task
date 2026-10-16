"""Entry point for a deeper data -> training -> residual model -> evaluation pipeline."""
import json
import platform
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from config import parse_args
from data import make_loader
from engine import evaluate, train_step
from model import TinyClassifier


def main():
    args = parse_args()
    if platform.system() not in ("Linux", "Darwin"):
        raise SystemExit("Please run this experiment on Linux/macOS (Windows: use WSL2).")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use --device cpu.")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    loader = make_loader(args.batch_size, args.image_size)
    model = TinyClassifier().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batches = iter(loader)

    def next_batch():
        nonlocal batches
        try:
            return next(batches)
        except StopIteration:
            batches = iter(loader)
            return next(batches)

    for _ in range(args.warmup):
        train_step(model, optimizer, next_batch(), args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise SystemExit("CUDA profiling unavailable in this PyTorch environment.")
        activities.append(ProfilerActivity.CUDA)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        with record_function("training_pipeline"):
            for _ in range(args.steps):
                with record_function("load_batch"):
                    batch = next_batch()
                loss = train_step(model, optimizer, batch, args.device)
            accuracy = evaluate(model, next_batch(), args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    (args.output / "operators.txt").write_text(table, encoding="utf-8")
    prof.export_chrome_trace(str(args.output / "trace.json"))
    summary = {
        "workload": "multi_residual_cnn_training", "device": args.device,
        "batch_size": args.batch_size, "image_size": args.image_size,
        "steps": args.steps, "warmup": args.warmup, "threads": args.threads,
        "seed": 42, "python": sys.version.split()[0], "torch": torch.__version__,
        "platform": platform.platform(), "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "last_train_loss": loss.item(), "eval_accuracy": accuracy.item(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "profiled_wall_seconds": elapsed,
        "timing_note": "Includes profiler overhead; not a benchmark.",
        "data_note": "Synthetic in-memory data, num_workers=0; no disk/network I/O measured.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table)
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
