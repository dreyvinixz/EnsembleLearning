"""Summarize MCC and accuracy on the exact shared Stage 12 populations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

KEYS = ["instrument", "session_date", "decision_time", "scale", "fold"]
PERIODS = {
    "OOF March-April": ["fold02", "fold03"],
    "May validation": ["fold04"],
}
RUNS = {
    "Logistic Regression": ("EXP_STAGE11_PAIR_CLASSICAL_42", None),
    "Voting": ("EXP_STAGE12_AVERAGING_MEMBERAUDIT_20261123",
               "soft_voting_all_architectures_strategies"),
    "Block Bagging": ("EXP_STAGE12_BLOCKBAG_LOGREG_B5_M10_V2_20261123",
                      "temporal_block_bagging_logreg"),
    "Stacking": ("EXP_STAGE12_STACKING_20261123", "chronological_oof_stacking"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tn, fp, fn, tp = counts.T.astype(float)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator,
                    out=np.zeros_like(denominator), where=denominator > 0)
    accuracy = (tp + tn) / np.maximum(counts.sum(axis=1), 1)
    return mcc, accuracy


def load_run(results_root: Path, run_id: str, ensemble: str | None) -> tuple[pd.DataFrame, str]:
    run_dir = results_root / run_id
    predictions_path = run_dir / "predictions.parquet"
    predictions = pd.read_parquet(predictions_path)
    if ensemble is not None:
        predictions = predictions.loc[predictions["ensemble"].eq(ensemble)].copy()
    return predictions, sha256(predictions_path)


def summarize(results_root: Path, output_csv: Path, *, iterations: int, seed: int) -> Path:
    if output_csv.exists() or output_csv.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite summary at {output_csv.parent}")
    loaded, hashes = {}, {}
    for method, (run_id, ensemble) in RUNS.items():
        loaded[method], hashes[method] = load_run(results_root, run_id, ensemble)

    rows = []
    for period_name, folds in PERIODS.items():
        subsets = {
            method: frame.loc[frame["fold"].isin(folds)].copy()
            for method, frame in loaded.items()
        }
        reference = subsets["Logistic Regression"]
        reference = reference.sort_values(KEYS).reset_index(drop=True)
        for method, frame in subsets.items():
            candidate = frame.sort_values(KEYS).reset_index(drop=True)
            if candidate.duplicated(KEYS).any():
                raise ValueError(f"Duplicate sample keys for {method} in {period_name}")
            if not candidate[KEYS].equals(reference[KEYS]):
                raise ValueError(f"Population mismatch for {method} in {period_name}")
            if not np.array_equal(candidate["y_true"].to_numpy(),
                                  reference["y_true"].to_numpy()):
                raise ValueError(f"y_true mismatch for {method} in {period_name}")
            subsets[method] = candidate

        for scale_index, scale in enumerate(("5m", "10m", "15m")):
            common = reference.loc[reference["scale"].eq(scale)].copy()
            sessions = sorted(common["session_date"].astype(str).unique())
            if not sessions:
                raise ValueError(f"No observations for {period_name}, {scale}")
            draw_indices = np.random.default_rng(seed + scale_index + len(rows)).integers(
                0, len(sessions), size=(iterations, len(sessions))
            )
            for method, all_predictions in subsets.items():
                predictions = all_predictions.loc[all_predictions["scale"].eq(scale)].copy()
                y_true = predictions["y_true"].to_numpy(dtype=int)
                y_pred = predictions["y_pred"].to_numpy(dtype=int)
                point = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                session_counts = []
                for session in sessions:
                    sample = predictions.loc[
                        predictions["session_date"].astype(str).eq(session)
                    ]
                    session_counts.append(
                        confusion_matrix(sample["y_true"], sample["y_pred"],
                                         labels=[0, 1]).ravel()
                    )
                session_counts_array = np.asarray(session_counts, dtype=np.int64)
                boot_counts = session_counts_array[draw_indices].sum(axis=1)
                boot_mcc, boot_accuracy = metric_from_counts(boot_counts)
                point_mcc, point_accuracy = metric_from_counts(point[None, :])
                mcc_low, mcc_high = np.quantile(boot_mcc, [0.025, 0.975])
                acc_low, acc_high = np.quantile(boot_accuracy, [0.025, 0.975])
                rows.append({
                    "period": period_name,
                    "scale": scale,
                    "method": method,
                    "n": len(predictions),
                    "n_sessions": len(sessions),
                    "mcc": float(point_mcc[0]),
                    "mcc_ci95_low": float(mcc_low),
                    "mcc_ci95_high": float(mcc_high),
                    "accuracy": float(point_accuracy[0]),
                    "accuracy_ci95_low": float(acc_low),
                    "accuracy_ci95_high": float(acc_high),
                })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    manifest = {
        "periods": PERIODS,
        "methods": {method: run for method, run in RUNS.items()},
        "source_prediction_sha256": hashes,
        "snapshot_sha256": "a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91",
        "keys_checked": KEYS,
        "y_true_equality_checked": True,
        "bootstrap": {"unit": "session_date", "iterations": iterations,
                      "seed": seed, "ci": "percentile 95%"},
        "june_accessed": False,
        "note": "OOF comparison restricted to March-April (folds 02-03) so stacking and all baselines share identical samples.",
    }
    output_csv.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(summarize(args.results_root, args.output_csv,
                    iterations=args.iterations, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
