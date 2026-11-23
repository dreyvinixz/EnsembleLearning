"""Paired session-block bootstrap with family-wise max-T intervals.

Input files must each contain predictions for one method and the approved
development periods only. June is rejected before any metric is calculated.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["instrument", "session_date", "decision_time", "scale", "fold"]
REQUIRED = [*KEYS, "y_true", "p_up"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_methods(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if len(frames) < 2:
        raise ValueError("At least two methods are required")
    cleaned = {}
    for name, frame in frames.items():
        missing = sorted(set(REQUIRED) - set(frame.columns))
        if missing:
            raise ValueError(f"{name}: missing prediction columns {missing}")
        part = frame[REQUIRED].copy()
        part["session_date"] = part.session_date.astype(str)
        if part.session_date.gt("2026-05-31").any() or part.session_date.lt("2026-01-01").any():
            raise ValueError(f"{name}: dates must be limited to Jan-May 2026; June is forbidden")
        part["decision_time"] = pd.to_datetime(part.decision_time, utc=True, errors="raise")
        if part.duplicated(KEYS).any():
            raise ValueError(f"{name}: duplicate sample keys")
        if not part.y_true.isin([0, 1]).all():
            raise ValueError(f"{name}: labels must be binary 0/1")
        part["y_true"] = part.y_true.astype(int)
        part["p_up"] = pd.to_numeric(part.p_up, errors="raise")
        if not np.isfinite(part.p_up).all() or not part.p_up.between(0, 1).all():
            raise ValueError(f"{name}: p_up must be finite and lie in [0,1]")
        cleaned[name] = part.sort_values(KEYS).reset_index(drop=True)

    common = cleaned[next(iter(cleaned))][KEYS]
    for frame in list(cleaned.values())[1:]:
        common = common.merge(frame[KEYS], on=KEYS, how="inner", validate="one_to_one")
    if common.empty:
        raise ValueError("Methods have no common sample keys")
    output = {}
    reference_labels = None
    for name, frame in cleaned.items():
        aligned = common.merge(frame, on=KEYS, how="left", validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
        labels = aligned.y_true.to_numpy()
        if reference_labels is not None and not np.array_equal(reference_labels, labels):
            raise ValueError(f"{name}: y_true differs on common keys")
        reference_labels = labels
        output[name] = aligned
    return output


def mcc_from_confusion(counts: np.ndarray) -> np.ndarray:
    tp, tn, fp, fn = np.moveaxis(counts, -1, 0)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    numerator = tp * tn - fp * fn
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def select_ensemble_predictions(frame: pd.DataFrame, ensemble: str) -> pd.DataFrame:
    """Select one named ensemble from a multi-ensemble prediction artifact."""
    if "ensemble" not in frame.columns:
        raise ValueError("Ensemble artifact must include an 'ensemble' column")
    selected = frame.loc[frame["ensemble"].astype(str).eq(ensemble)].copy()
    if selected.empty:
        raise ValueError(f"No predictions found for ensemble={ensemble!r}")
    return selected


def max_t_comparisons(
    frames: dict[str, pd.DataFrame], *, iterations: int = 5000, seed: int = 42,
) -> pd.DataFrame:
    aligned = align_methods(frames)
    methods = list(aligned)
    results = []
    rng = np.random.default_rng(seed)
    period_fold_rules = {
        "oof_feb_apr": lambda frame: frame.loc[frame.fold.ne("fold04")],
        "oof_mar_apr": lambda frame: frame.loc[frame.fold.isin(("fold02", "fold03"))],
        "validation_may": lambda frame: frame.loc[frame.fold.eq("fold04")],
    }
    scales = sorted(set.intersection(*(set(frame.scale) for frame in aligned.values())))
    for period_name, filter_period in period_fold_rules.items():
        for scale in scales:
            group_frames = {
                method: filter_period(frame).loc[filter_period(frame).scale.eq(scale)].copy()
                for method, frame in aligned.items()
            }
            if any(frame.empty for frame in group_frames.values()):
                continue
            common_group = align_methods(group_frames)
            reference = common_group[methods[0]]
            if period_name == "oof_feb_apr" and not reference.fold.eq("fold01").any():
                # A method such as OOF stacking has no prediction for its first fold.
                # Its exact common OOF population is reported separately as March-April.
                continue
            sessions = sorted(reference.session_date.unique())
            if len(sessions) < 2:
                raise ValueError(f"Need at least two sessions for {period_name}/{scale}")
            per_method = []
            for method in methods:
                current = common_group[method]
                daily = []
                for session in sessions:
                    part = current.loc[current.session_date.eq(session)]
                    truth = part.y_true.to_numpy(dtype=int)
                    pred = (part.p_up.to_numpy(dtype=float) >= .5).astype(int)
                    daily.append([
                        np.count_nonzero((truth == 1) & (pred == 1)),
                        np.count_nonzero((truth == 0) & (pred == 0)),
                        np.count_nonzero((truth == 0) & (pred == 1)),
                        np.count_nonzero((truth == 1) & (pred == 0)),
                        len(truth),
                    ])
                per_method.append(np.asarray(daily, dtype=np.int64))
            counts = np.stack(per_method)  # [method, session, TP/TN/FP/FN/N]
            observed_counts = counts.sum(axis=1)
            observed_mcc = mcc_from_confusion(observed_counts[:, :4])
            observed_accuracy = (observed_counts[:, 0] + observed_counts[:, 1]) / observed_counts[:, 4]
            sampled_sessions = rng.integers(0, len(sessions), size=(iterations, len(sessions)))
            boot_counts = counts[:, sampled_sessions, :].sum(axis=2).transpose(1, 0, 2)
            boot_mcc = mcc_from_confusion(boot_counts[:, :, :4])
            boot_accuracy = (boot_counts[:, :, 0] + boot_counts[:, :, 1]) / boot_counts[:, :, 4]

            pair_indices = list(itertools.combinations(range(len(methods)), 2))
            observed_deltas = []
            boot_deltas = []
            pair_metric = []
            for metric_name, observed_values, boot_values in (
                ("mcc", observed_mcc, boot_mcc),
                ("accuracy", observed_accuracy, boot_accuracy),
            ):
                for left, right in pair_indices:
                    observed_delta = observed_values[left] - observed_values[right]
                    draws = boot_values[:, left] - boot_values[:, right]
                    observed_deltas.append(observed_delta)
                    boot_deltas.append(draws)
                    pair_metric.append((metric_name, left, right, observed_delta))
            boot_matrix = np.stack(boot_deltas, axis=1)
            standard_errors = boot_matrix.std(axis=0, ddof=1)
            safe_se = np.where(standard_errors > 0, standard_errors, 1.0)
            max_abs_t = np.max(np.abs((boot_matrix - np.asarray(observed_deltas)) / safe_se), axis=1)
            critical = float(np.quantile(max_abs_t, .95))
            for pair_position, (metric_name, left, right, delta) in enumerate(pair_metric):
                se = float(standard_errors[pair_position])
                scores = observed_mcc if metric_name == "mcc" else observed_accuracy
                results.append({
                    "period": period_name, "scale": scale, "metric": metric_name,
                    "left_method": methods[left], "right_method": methods[right],
                    "n": len(reference), "n_sessions": len(sessions),
                    "left_score": float(scores[left]), "right_score": float(scores[right]),
                    "difference_left_minus_right": float(delta), "bootstrap_se": se,
                    "simultaneous_ci95_low": float(delta - critical * se),
                    "simultaneous_ci95_high": float(delta + critical * se),
                    "max_t_critical_95": critical, "bootstrap_iterations": iterations,
                    "session_block_unit": "session_date; all instruments within a date resampled together",
                })
    return pd.DataFrame(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=PARQUET",
                        help="One source per method; repeat. Each file must contain one method.")
    parser.add_argument("--ensemble-prediction", action="append", default=[],
                        metavar="NAME=PARQUET,ENSEMBLE_KEY",
                        help="Select one named ensemble from a multi-ensemble prediction Parquet.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 1000:
        parser.error("At least 1000 session bootstrap iterations are required")
    if args.output_csv.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite output artifacts")
    sources = {}
    paths: dict[str, list[Path]] = {}
    source_descriptors: dict[str, dict[str, str]] = {}
    for item in args.prediction:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path or name in sources:
            parser.error(f"Invalid/duplicate --prediction entry: {item}")
        path = Path(raw_path)
        sources[name] = pd.read_parquet(path)
        paths[name] = [path]
        source_descriptors[name] = {"path": str(path)}
    for item in args.ensemble_prediction:
        name, separator, source = item.partition("=")
        if not separator or not name or name in sources:
            parser.error(f"Invalid/duplicate --ensemble-prediction entry: {item}")
        raw_path, comma, ensemble_key = source.rpartition(",")
        if not comma or not raw_path or not ensemble_key:
            parser.error("--ensemble-prediction must use NAME=PARQUET,ENSEMBLE_KEY")
        path = Path(raw_path)
        artifact = pd.read_parquet(path)
        sources[name] = select_ensemble_predictions(artifact, ensemble_key)
        paths[name] = [path]
        source_descriptors[name] = {"path": str(path), "ensemble_key": ensemble_key}
    summary = max_t_comparisons(sources, iterations=args.iterations, seed=args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    manifest = {
        "method": "paired session bootstrap max-T simultaneous intervals",
        "sources": {
            name: {
                **source_descriptors[name],
                "sha256": sha256(paths[name][0]),
            }
            for name in sources
        },
        "methods": list(sources), "iterations": args.iterations, "seed": args.seed,
        "primary_metric": "MCC", "secondary_metric": "accuracy", "decision_threshold": 0.5,
        "multiplicity": "95% max absolute studentized bootstrap deviation across all pairwise method contrasts and both metrics within each period-scale family",
        "common_keys": KEYS, "y_true_equality_required": True,
        "script_sha256": sha256(Path(__file__).resolve()),
        "June_accessed": False, "contains_row_level_predictions": False,
        "output_sha256": sha256(args.output_csv),
        "interpretation": "Simultaneous intervals are exploratory unless contrast families and protocol are frozen in advance.",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
