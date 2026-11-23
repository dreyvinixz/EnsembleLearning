"""Synthetic CUDA throughput benchmark; never reads project data or labels."""

from __future__ import annotations

import argparse
import json
import platform
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ensemble_learning.neural_training import _WindowDataset
from ensemble_learning.sequence_models import (
    NEURAL_ARCHITECTURES,
    MultiTaskSequenceClassifier,
)
from ensemble_learning.sequences import CausalWindows


def benchmark_case(
    architecture: str,
    batch_size: int,
    precision: str,
    allow_tf32: bool,
    fused_adamw: bool,
    warmup_steps: int,
    timed_steps: int,
    repeat: int,
    hidden_size: int,
    feature_count: int,
):
    device = torch.device("cuda")
    torch.manual_seed(420 + repeat)
    torch.cuda.manual_seed_all(420 + repeat)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    model = MultiTaskSequenceClassifier(
        architecture,
        input_size=feature_count,
        hidden_size=hidden_size,
        sequence_length=60 if architecture == "mlp" else None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused_adamw)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    features = torch.randn(batch_size, 60, feature_count, device=device)
    labels = torch.randint(0, 2, (batch_size, 3), device=device)
    dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            outputs = model(features)
            loss = torch.stack([
                F.cross_entropy(outputs[horizon].float(), labels[:, index])
                for index, horizon in enumerate((5, 10, 15))
            ]).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.train()
    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(timed_steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "architecture": architecture,
        "strategy": "mtl",
        "batch_size": batch_size,
        "precision": precision,
        "allow_tf32": allow_tf32,
        "fused_adamw": fused_adamw,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "repeat": repeat,
        "elapsed_seconds": elapsed,
        "steps_per_second": timed_steps / elapsed,
        "samples_per_second": timed_steps * batch_size / elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def benchmark_loader(worker_counts: list[int], feature_count: int, repeats: int):
    rng = np.random.default_rng(420)
    context_rows, anchors_count, window_minutes = 12_000, 8_000, 60
    values = rng.normal(size=(context_rows, feature_count - 1)).astype(np.float32)
    row_indices = rng.integers(0, context_rows, size=(anchors_count, window_minutes))
    row_indices[rng.random(row_indices.shape) < 0.1] = -1
    anchors = pd.DataFrame({
        f"y_{horizon}m": rng.integers(0, 2, size=anchors_count)
        for horizon in (5, 10, 15)
    })
    dataset = _WindowDataset(
        CausalWindows(row_indices, anchors, (row_indices >= 0).sum(axis=1)),
        values,
        ["y_5m", "y_10m", "y_15m"],
    )
    results = []
    for workers in worker_counts:
        for repeat in range(repeats):
            loader = DataLoader(
                dataset, batch_size=256, shuffle=False, num_workers=workers,
                pin_memory=True, persistent_workers=workers > 0,
            )
            started = time.perf_counter()
            for _ in range(2):
                for features, labels in loader:
                    features.to("cuda", non_blocking=True)
                    labels.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results.append({
                "num_workers": workers, "pin_memory": True,
                "passes": 2, "samples": len(dataset) * 2,
                "repeat": repeat, "elapsed_seconds": elapsed,
                "samples_per_second": len(dataset) * 2 / elapsed,
            })
            del loader
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gpu-benchmarks"))
    parser.add_argument("--architectures", nargs="+", choices=NEURAL_ARCHITECTURES,
                        default=list(NEURAL_ARCHITECTURES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "bf16", "fp16"),
                        default=["fp32", "bf16", "fp16"])
    parser.add_argument("--optimizer-modes", nargs="+", choices=("fused", "unfused"),
                        default=["fused", "unfused"])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--feature-count", type=int, default=22,
                        help="21 market features plus the missing-candle mask")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required; no benchmark was run")
    if any(batch < 1 for batch in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.warmup_steps < 1 or args.timed_steps < 1 or args.repeats < 1:
        parser.error("warmup, timed steps and repeats must be positive")
    if "bf16" in args.precisions and not torch.cuda.is_bf16_supported():
        parser.error("The selected GPU does not support bfloat16")
    if any(worker < 0 for worker in args.worker_counts):
        parser.error("worker counts cannot be negative")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_id = f"gpu-bench-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for architecture in args.architectures:
        for batch_size in args.batch_sizes:
            for precision in args.precisions:
                tf32_options = (False, True) if precision == "fp32" else (False,)
                for allow_tf32 in tf32_options:
                    for optimizer_mode in args.optimizer_modes:
                        for repeat in range(args.repeats):
                            rows.append(benchmark_case(
                                architecture, batch_size, precision, allow_tf32,
                                optimizer_mode == "fused", args.warmup_steps,
                                args.timed_steps, repeat, args.hidden_size,
                                args.feature_count,
                            ))
    loader_results = benchmark_loader(
        args.worker_counts, args.feature_count, args.repeats
    )
    pd.DataFrame(rows).to_csv(output / "throughput.csv", index=False)
    frame = pd.DataFrame(rows)
    group_columns = [
        "architecture", "batch_size", "precision", "allow_tf32", "fused_adamw"
    ]
    summary = frame.groupby(group_columns, dropna=False).agg(
        samples_per_second_median=("samples_per_second", "median"),
        samples_per_second_std=("samples_per_second", "std"),
        peak_cuda_allocated_bytes=("peak_cuda_allocated_bytes", "max"),
    ).reset_index()
    summary.to_csv(output / "throughput_summary.csv", index=False)
    properties = torch.cuda.get_device_properties(0)
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "synthetic GPU throughput only; no BovDB/project data or labels read",
        "scope": "model forward/backward/optimizer; excludes preprocessing and DataLoader",
        "loader_scope": "synthetic _WindowDataset assembly and pinned host-to-device transfer",
        "device": properties.name,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "parameters": vars(args) | {"output": str(args.output)},
        "results": len(rows),
        "repeats": args.repeats,
        "loader_results": loader_results,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(output)
    print(json.dumps({
        "model_throughput_median": summary.to_dict(orient="records"),
        "loader_throughput": loader_results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
