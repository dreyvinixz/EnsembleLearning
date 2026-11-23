"""Recompute Stage 11 metrics and paired session-bootstrap contrasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_learning.contracts import timestamp_column
from ensemble_learning.pairing import KEYS, paired_metrics
from ensemble_learning.tracking import classification_metrics

ARCHITECTURES = ("MLP", "CNN1D", "LSTM", "GRU")
STRATEGIES = ("INDEPENDENT", "MTL")
SEEDS = (42, 43)
FOLDS = ("fold01", "fold02", "fold03", "fold04")
SCALES = ("5m", "10m", "15m")
SOURCE_SHA256 = "a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91"
CLASSICAL_ID = "EXP_STAGE11_PAIR_CLASSICAL_42"


def _load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(KEYS + ["y_true", "p_up"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing prediction columns: {sorted(missing)}")
    if frame[KEYS + ["y_true", "p_up"]].isna().any().any():
        raise ValueError(f"{path} has null prediction keys, labels or probabilities")
    frame["decision_time"] = timestamp_column(frame, "decision_time")
    frame["session_date"] = frame.session_date.astype(str)
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} has duplicate prediction keys")
    if not frame.y_true.isin((0, 1)).all() or not frame.p_up.between(0, 1).all():
        raise ValueError(f"{path} has invalid B0 classes or probabilities")
    if frame.session_date.min() < "2026-01-01" or frame.session_date.max() > "2026-05-31":
        raise ValueError(f"{path} includes rows outside Jan-May development")
    if set(frame.fold.astype(str)) != set(FOLDS) or set(frame.scale.astype(str)) != set(SCALES):
        raise ValueError(f"{path} does not contain the expected folds and scales")
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _mcc_from_confusion(counts: np.ndarray) -> float:
    tp, tn, fp, fn = (int(value) for value in counts)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def paired_session_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    iterations: int = 5000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap MCC(left)-MCC(right), resampling whole paired sessions."""
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations")
    left = left.sort_values(KEYS, kind="stable").reset_index(drop=True)
    right = right.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if not left[KEYS].equals(right[KEYS]):
        raise ValueError("Paired bootstrap requires identical complete key populations")
    if not left.y_true.reset_index(drop=True).equals(right.y_true.reset_index(drop=True)):
        raise ValueError("Paired bootstrap requires equal y_true for every key")
    if left.empty:
        raise ValueError("Paired bootstrap cannot use an empty population")

    truth = left.y_true.to_numpy(dtype=np.int8)
    pred_left = (left.p_up.to_numpy(dtype=float) >= 0.5).astype(np.int8)
    pred_right = (right.p_up.to_numpy(dtype=float) >= 0.5).astype(np.int8)
    sessions = left.session_date.astype(str).to_numpy()
    session_ids = pd.unique(sessions)
    if len(session_ids) < 2:
        raise ValueError("At least two distinct sessions are required for cluster bootstrap")

    per_session = np.zeros((len(session_ids), 8), dtype=np.int64)
    for index, session in enumerate(session_ids):
        selected = sessions == session
        y, a, b = truth[selected], pred_left[selected], pred_right[selected]
        per_session[index] = (
            np.count_nonzero((y == 1) & (a == 1)),
            np.count_nonzero((y == 0) & (a == 0)),
            np.count_nonzero((y == 0) & (a == 1)),
            np.count_nonzero((y == 1) & (a == 0)),
            np.count_nonzero((y == 1) & (b == 1)),
            np.count_nonzero((y == 0) & (b == 0)),
            np.count_nonzero((y == 0) & (b == 1)),
            np.count_nonzero((y == 1) & (b == 0)),
        )

    observed_left = _mcc_from_confusion(per_session[:, :4].sum(axis=0))
    observed_right = _mcc_from_confusion(per_session[:, 4:].sum(axis=0))
    rng = np.random.default_rng(seed)
    sampled_sessions = rng.integers(0, len(session_ids), size=(iterations, len(session_ids)))
    sampled = per_session[sampled_sessions].sum(axis=1)
    a, b = sampled[:, :4], sampled[:, 4:]
    def scores(confusion: np.ndarray) -> np.ndarray:
        tp, tn, fp, fn = confusion.T
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return np.divide(tp * tn - fp * fn, denominator,
                         out=np.zeros(len(confusion), dtype=float), where=denominator > 0)
    differences = scores(a) - scores(b)
    return {
        "n": len(left), "n_sessions": len(session_ids),
        "mcc_left": observed_left, "mcc_right": observed_right,
        "mcc_difference": observed_left - observed_right,
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_probability_difference_positive": float(np.mean(differences > 0)),
        "bootstrap_iterations": iterations,
    }


def _metric_rows(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run_id, frame in predictions.items():
        if run_id == CLASSICAL_ID:
            architecture, strategy, seed = "LogisticRegression", "INDEPENDENT", 42
        else:
            _, _, _, architecture, strategy, seed = run_id.split("_")
            seed = int(seed)
        groups = [(str(fold), group) for fold, group in frame.groupby("fold", sort=True)]
        groups += [("oof_feb_apr", frame[frame.fold.isin(FOLDS[:3])]),
                   ("validation_may", frame[frame.fold.eq(FOLDS[3])])]
        for fold, group in groups:
            for scale, part in group.groupby("scale", sort=True):
                metrics = classification_metrics(part.y_true, part.p_up)
                rows.append({
                    "run_id": run_id, "architecture": architecture, "strategy": strategy,
                    "seed": seed, "fold": fold, "scale": scale,
                    "period": ("oof_feb_apr" if fold == "oof_feb_apr" else
                               "validation_may" if fold == "validation_may" else
                               "oof" if fold in FOLDS[:3] else "validation"),
                    **metrics,
                })
    return pd.DataFrame(rows)


def consolidate(results_root: Path, output_dir: Path, *, iterations: int = 5000,
                seed: int = 42) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite result directory {output_dir}")
    baseline_dir = results_root / CLASSICAL_ID
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    if baseline_manifest.get("data_sha256") != SOURCE_SHA256:
        raise ValueError("Classical baseline snapshot hash differs from approved Stage 11 data")
    if baseline_manifest.get("test_june_accessed") is not False:
        raise ValueError("Classical baseline manifest does not attest that June was untouched")
    predictions = {CLASSICAL_ID: _load_prediction(baseline_dir / "predictions.parquet")}
    expected_runs = []
    for architecture in ARCHITECTURES:
        for strategy in STRATEGIES:
            for model_seed in SEEDS:
                expected_runs.append(
                    f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{model_seed}"
                )
    manifests = {}
    for run_id in expected_runs:
        run_dir = results_root / run_id
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.yaml").read_text(encoding="utf-8"))
        _, _, _, architecture, strategy, model_seed = run_id.split("_")
        if manifest.get("data_sha256") != SOURCE_SHA256:
            raise ValueError(f"{run_id} has a different data hash")
        if manifest.get("test_june_accessed") is not False:
            raise ValueError(f"{run_id} does not attest that June was untouched")
        expected_config = {
            "architecture": architecture.lower(), "strategy": strategy.lower(),
            "seed": int(model_seed), "epochs": 5, "early_stopping_patience": 2,
            "batch_size": 256, "precision": "fp32", "device_requested": "cuda",
        }
        if any(config.get(key) != value for key, value in expected_config.items()):
            raise ValueError(f"{run_id} configuration differs from the audited matrix")
        environment = manifest.get("environment", {})
        if (environment.get("device") != "cuda" or not environment.get("cuda_available")
                or "RTX 5060 Ti" not in environment.get("gpu_name", "")):
            raise ValueError(f"{run_id} manifest does not confirm the expected CUDA host")
        expected_fits = 12 if strategy == "INDEPENDENT" else 4
        if len(manifest.get("folds", [])) != expected_fits:
            raise ValueError(f"{run_id} manifest has an unexpected fold/model fit count")
        summary_path = run_dir / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fits = summary.get("fold_fits", [])
        expected_fold_fits = 12 if strategy == "INDEPENDENT" else 4
        if len(fits) != expected_fold_fits:
            raise ValueError(f"{run_id} has an unexpected number of fold fits")
        for fit in fits:
            if not fit.get("refit_after_early_stopping"):
                raise ValueError(f"{run_id} contains a fit without full-block refit")
            if fit.get("refit_epochs") != fit.get("best_epoch"):
                raise ValueError(f"{run_id} refit epochs differ from selected inner epoch")
            if fit.get("refit_train_end", "") >= fit.get("oof_start", ""):
                raise ValueError(f"{run_id} refit training reaches its OOF boundary")
        frame = _load_prediction(run_dir / "predictions.parquet")
        paired, _ = paired_metrics(predictions[CLASSICAL_ID], frame)
        if len(paired) != len(frame) or len(paired) != len(predictions[CLASSICAL_ID]):
            raise ValueError(f"{run_id} does not have complete population equality with baseline")
        if len(paired) != 170_100:
            raise ValueError(f"Unexpected prediction count for {run_id}: {len(paired)}")
        coverage = pd.read_csv(run_dir / "coverage.csv")
        for fold in FOLDS:
            accepted = coverage.loc[
                coverage.fold.eq(fold) & coverage.partition.eq("oof"), "accepted"
            ]
            if len(accepted) != 1:
                raise ValueError(f"{run_id} has no unique OOF coverage row for {fold}")
            for scale in SCALES:
                actual = int((frame.fold.eq(fold) & frame.scale.eq(scale)).sum())
                if actual != int(accepted.iloc[0]):
                    raise ValueError(f"{run_id} {fold}/{scale} differs from accepted coverage")
        predictions[run_id] = frame
        manifests[run_id] = manifest

    output_dir.mkdir(parents=True)
    metric_table = _metric_rows(predictions)
    metric_table.to_csv(output_dir / "metrics_by_run_fold_scale.csv", index=False)
    comparison_rows = []

    def append_comparison(label: str, left_id: str, right_id: str,
                          scale: str, period: str, folds: tuple[str, ...]) -> None:
        left = predictions[left_id]
        right = predictions[right_id]
        left = left[left.scale.eq(scale) & left.fold.isin(folds)].copy()
        right = right[right.scale.eq(scale) & right.fold.isin(folds)].copy()
        result = paired_session_bootstrap(
            left, right, iterations=iterations,
            seed=seed + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16),
        )
        comparison_rows.append({
            "contrast": label, "period": period, "scale": scale,
            "left_run": left_id, "right_run": right_id, **result,
        })

    oof_folds, may_fold = tuple(FOLDS[:3]), (FOLDS[3],)
    periods = (("oof_feb_apr", oof_folds), ("validation_may", may_fold))
    for architecture in ARCHITECTURES:
        for model_seed in SEEDS:
            independent = f"EXP_STAGE11_REFIT_{architecture}_INDEPENDENT_{model_seed}"
            mtl = f"EXP_STAGE11_REFIT_{architecture}_MTL_{model_seed}"
            for scale in SCALES:
                for period, folds in periods:
                    append_comparison(f"{architecture}: MTL - Independent, seed {model_seed}",
                                      mtl, independent, scale, period, folds)
        for strategy in STRATEGIES:
            for scale in SCALES:
                for period, folds in periods:
                    append_comparison(f"{architecture}/{strategy}: seed 43 - seed 42",
                                      f"EXP_STAGE11_REFIT_{architecture}_{strategy}_43",
                                      f"EXP_STAGE11_REFIT_{architecture}_{strategy}_42",
                                      scale, period, folds)
        for strategy in STRATEGIES:
            for model_seed in SEEDS:
                model_id = f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{model_seed}"
                for scale in SCALES:
                    for period, folds in periods:
                        append_comparison(f"{model_id} - LogisticRegression", model_id,
                                          CLASSICAL_ID, scale, period, folds)

    pd.DataFrame(comparison_rows).to_csv(output_dir / "paired_mcc_session_bootstrap.csv", index=False)
    metadata = {
        "snapshot_sha256": SOURCE_SHA256,
        "experiments": expected_runs,
        "baseline": CLASSICAL_ID,
        "all_populations_equal": True,
        "predictions_per_experiment": 170100,
        "June_accessed": False,
        "primary_metric": "MCC, fixed threshold 0.5; exploratory until protocol freeze",
        "secondary_metrics": ["macro_f1", "roc_auc", "average_precision", "brier", "log_loss",
                              "balanced_accuracy", "accuracy"],
        "statistical_method": "paired cluster bootstrap resampling session_date with replacement",
        "bootstrap_iterations": iterations,
        "bootstrap_seed_base": seed,
        "multiplicity_adjustment": None,
        "interpretation": "Exploratory intervals; do not use June for selection or tuning.",
        "source_commits": sorted({m.get("git_commit", "unknown") for m in manifests.values()}),
        "training_configuration": {
            "device": "NVIDIA GeForce RTX 5060 Ti (CUDA)", "precision": "fp32",
            "batch_size": 256, "epochs_max": 5, "early_stopping_patience": 2,
            "seeds": list(SEEDS), "folds_per_run": len(FOLDS),
            "refit_after_early_stopping": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(consolidate(args.results_root, args.output_dir,
                      iterations=args.iterations, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
