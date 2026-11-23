"""Export aggregate neural training histories without copying predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_RUNS = {
    f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{seed}"
    for architecture in ("MLP", "CNN1D", "LSTM", "GRU")
    for strategy in ("INDEPENDENT", "MTL")
    for seed in (42, 43)
}
REQUIRED = {
    "fold", "architecture", "strategy", "seed", "horizon", "epoch", "train_loss",
    "validation_loss", "validation_mcc", "validation_accuracy", "phase",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(results_root: Path, output_csv: Path, manifest_path: Path) -> None:
    frames = []
    source_files: dict[str, str] = {}
    seen: set[str] = set()
    for run_dir in sorted(results_root.glob("EXP_STAGE11_REFIT_*")):
        if run_dir.name not in EXPECTED_RUNS:
            continue
        path = run_dir / "training_history.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing history: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Empty history: {path}")
        frame.insert(0, "run_id", run_dir.name)
        frames.append(frame)
        seen.add(run_dir.name)
        source_files[run_dir.name] = sha256(path)
    if seen != EXPECTED_RUNS:
        raise ValueError(f"Expected 16 runs; missing={sorted(EXPECTED_RUNS-seen)}, extra={sorted(seen-EXPECTED_RUNS)}")
    history = pd.concat(frames, ignore_index=True)
    if history.run_id.nunique() != 16:
        raise ValueError("History export did not preserve all 16 run identifiers")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_csv, index=False)
    manifest = {
        "source": "Linux results/EXP_STAGE11_REFIT_*/training_history.csv",
        "run_count": len(seen),
        "row_count": len(history),
        "source_sha256_by_run": source_files,
        "aggregate_csv_sha256": sha256(output_csv),
        "contains_predictions": False,
        "June_accessed": False,
        "interpretation": "Training histories only; descriptive and not inferential.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    export(args.results_root, args.output_csv, args.manifest)
    print(f"Exported aggregate training histories to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
