"""Exercise all four workloads, including numeric correctness and nested trace ranges."""
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, success=True):
    result = subprocess.run([str(arg) for arg in args], cwd=ROOT, text=True, capture_output=True)
    if success and result.returncode:
        raise AssertionError(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")
    if not success and not result.returncode:
        raise AssertionError(f"Invalid input was accepted: {args}")
    return result.stdout


def check_trace(output, names, nesting):
    summary = json.loads((output / "summary.json").read_text())
    assert summary["device"] == "cpu"
    events = json.loads((output / "trace.json").read_text())["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X" and "dur" in event]
    assert set(names) <= {event["name"] for event in spans}
    for parent_name, child_name in nesting:
        parents = [event for event in spans if event["name"] == parent_name]
        children = [event for event in spans if event["name"] == child_name]
        assert any(
            p.get("tid") == c.get("tid") and p["ts"] <= c["ts"]
            and c["ts"] + c["dur"] <= p["ts"] + p["dur"] + 1
            for p in parents for c in children
        ), (parent_name, child_name)
    assert "Self CPU" in (output / "operators.txt").read_text()
    return summary


def main():
    if platform.system() not in ("Linux", "Darwin"):
        raise SystemExit("Run tests on Linux/macOS; Windows users should use WSL2.")
    subprocess.run(["make", "all", "build/numeric_test"], cwd=ROOT, check=True)
    run(ROOT / "build/numeric_test")
    matrix = json.loads(run(ROOT / "build/matrix_demo", "--size", 3, "--rounds", 2))
    expected = 2 * sum(
        sum(((i * 3 + k) % 17) / 17 * (((k * 3 + j) % 13) / 13) for k in range(3))
        for i in range(3) for j in range(3)
    )
    assert math.isclose(matrix["checksum"], expected, rel_tol=1e-12)
    base = [ROOT / "build/signal_pipeline", "--records", 12, "--length", 32,
            "--bins", 4, "--queries", 4]
    first = json.loads(run(*base, "--rounds", 1))
    second = json.loads(run(*base, "--rounds", 2))
    assert first["energy"] > 0 and first["distance"] >= 0
    for key in ("energy", "distance"):
        assert math.isclose(second[key], 2 * first[key], rel_tol=1e-12)
    run(ROOT / "build/matrix_demo", "--size", 0, success=False)
    run(ROOT / "build/signal_pipeline", "--length", 8, "--bins", 9, success=False)
    with tempfile.TemporaryDirectory(prefix="profiling-task-test-") as temporary:
        single, multi = Path(temporary) / "single", Path(temporary) / "multi"
        run(sys.executable, ROOT / "01-pytorch-single/profile_single.py",
            "--steps", 2, "--warmup", 1, "--batch-size", 2, "--threads", 1, "--output", single)
        small = check_trace(single, ["inference_step", "mlp_forward"],
                            [("inference_step", "mlp_forward")])
        assert small["output_shape"] == [2, 64] and math.isfinite(small["checksum"])
        run(sys.executable, ROOT / "02-pytorch-multi/main.py", "--steps", 2,
            "--warmup", 1, "--batch-size", 2, "--image-size", 16, "--threads", 1, "--output", multi)
        deep = check_trace(multi,
            ["training_pipeline", "load_batch", "train_step", "preprocess", "augmentation",
             "normalize", "device_transfer", "forward", "backbone", "residual_block_1",
             "residual_block_2", "conv_path", "loss", "backward", "optimizer_step", "evaluation"],
            [("training_pipeline", "train_step"), ("train_step", "forward"),
             ("forward", "backbone"), ("backbone", "residual_block_1"),
             ("residual_block_1", "conv_path"), ("evaluation", "eval_forward")])
        assert math.isfinite(deep["last_train_loss"]) and 0 <= deep["eval_accuracy"] <= 1
        run(sys.executable, ROOT / "01-pytorch-single/profile_single.py",
            "--steps", 0, success=False)
    print("PASS: four workloads, numeric checks, invalid input, and real nested trace spans")


if __name__ == "__main__":
    main()
