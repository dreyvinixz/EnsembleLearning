"""Run a fixed temporal block-bagging Logistic Regression baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from ensemble_learning.baselines import make_classical_model
from ensemble_learning.block_bagging import sample_session_blocks
from ensemble_learning.contracts import feature_columns, validate_sequence_frame
from ensemble_learning.ensembles import mean_probability_ensemble
from ensemble_learning.pairing import paired_metrics
from ensemble_learning.sequences import build_causal_windows
from ensemble_learning.tracking import classification_metrics, walk_forward_folds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_stage11_results import paired_session_bootstrap

SNAPSHOT_SHA256 = "a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91"
BASELINE_ID = "EXP_STAGE11_PAIR_CLASSICAL_42"
VOTING_RUN_ID = "EXP_STAGE12_AVERAGING_MEMBERAUDIT_20261123"
HORIZONS = (5, 10, 15)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_block_bagging(
    data_path: Path, results_root: Path, output_dir: Path, *,
    members: int = 10, block_size: int = 5, seed: int = 42,
    bootstrap_iterations: int = 5000,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if members < 2 or block_size < 1 or bootstrap_iterations < 100:
        raise ValueError("Require >=2 members, positive block size and >=100 bootstrap draws")
    if _hash(data_path) != SNAPSHOT_SHA256:
        raise ValueError("Input data hash differs from the audited Jan-May snapshot")
    frame = validate_sequence_frame(pd.read_csv(data_path))
    columns = feature_columns(frame)
    target_columns = [f"y_{horizon}m" for horizon in HORIZONS]
    member_frames, aggregate_frames, fit_records = [], [], []
    started = time.perf_counter()

    for fold_index, fold in enumerate(walk_forward_folds(frame)):
        train_windows = build_causal_windows(
            fold["train"], feature_columns=columns, target_columns=target_columns,
            window_minutes=60, min_observed=48,
        )
        prediction_windows = build_causal_windows(
            fold["oof"], feature_columns=columns, target_columns=target_columns,
            window_minutes=60, min_observed=48,
        )
        train = train_windows.anchors.copy()
        predict = prediction_windows.anchors.copy()
        if train.empty or predict.empty:
            raise ValueError(f"No eligible train or prediction anchors in {fold['fold']}")
        session_dates = sorted(train.session_date.astype(str).unique())
        for horizon in HORIZONS:
            label = f"y_{horizon}m"
            member_outputs = []
            for replicate in range(members):
                replicate_seed = seed + fold_index * 10_000 + horizon * 100 + replicate
                sampled_dates = sample_session_blocks(
                    session_dates, block_size=block_size, seed=replicate_seed,
                )
                sampled = pd.concat([
                    train.loc[train.session_date.astype(str).eq(day)]
                    for day in sampled_dates
                ], ignore_index=True)
                if set(sampled[label].unique()) != {0, 1}:
                    raise ValueError(
                        f"Block bootstrap member lacks both classes: {fold['fold']}/{label}/{replicate}"
                    )
                model = make_classical_model("logreg", replicate_seed)
                fit_started = time.perf_counter()
                model.fit(sampled[columns], sampled[label].astype(int))
                fit_seconds = time.perf_counter() - fit_started
                probabilities = model.predict_proba(predict[columns])[
                    :, list(model.classes_).index(1)
                ]
                output = predict[["instrument", "session_date", "decision_time"]].copy()
                output["fold"] = fold["fold"]
                output["scale"] = f"{horizon}m"
                output["y_true"] = predict[label].to_numpy(dtype=int)
                output["p_up"] = probabilities
                output["replicate"] = replicate
                output["model"] = "logistic_regression"
                member_outputs.append(output)
                member_frames.append(output)
                fit_records.append({
                    "fold": fold["fold"], "scale": f"{horizon}m",
                    "replicate": replicate, "seed": replicate_seed,
                    "block_size_sessions": min(block_size, len(session_dates)),
                    "training_sessions_available": len(session_dates),
                    "training_session_draws": len(sampled_dates),
                    "training_anchors_after_resampling": len(sampled),
                    "n_prediction_anchors": len(predict),
                    "fit_seconds": fit_seconds,
                    "train_end": str(sampled.session_date.max()),
                    "prediction_start": str(predict.session_date.min()),
                })
            aggregate = mean_probability_ensemble(
                {f"block{index:02d}": output for index, output in enumerate(member_outputs)},
                ensemble_name="temporal_block_bagging_logreg",
            )
            aggregate_frames.append(aggregate)

    aggregated = pd.concat(aggregate_frames, ignore_index=True)
    replicate_predictions = pd.concat(member_frames, ignore_index=True)
    if aggregated.empty or replicate_predictions.empty:
        raise RuntimeError("Block-bagging outputs are incomplete")
    baseline_dir = results_root / BASELINE_ID
    baseline = pd.read_parquet(baseline_dir / "predictions.parquet")
    paired, _ = paired_metrics(baseline, aggregated)
    if len(paired) != len(baseline):
        raise ValueError("Block-bagging predictions differ from the complete baseline population")
    voting_path = results_root / VOTING_RUN_ID / "predictions.parquet"
    voting = pd.read_parquet(voting_path)
    voting = voting.loc[voting.ensemble.eq("soft_voting_all_architectures_strategies")]

    output_dir.mkdir(parents=True)
    aggregated.to_parquet(output_dir / "predictions.parquet", index=False)
    replicate_predictions.to_parquet(output_dir / "member_predictions.parquet", index=False)
    pd.DataFrame(fit_records).to_csv(output_dir / "fit_records.csv", index=False)
    by_fold, by_period, paired_rows = [], [], []
    for (fold_name, scale), group in aggregated.groupby(["fold", "scale"], sort=True):
        by_fold.append({"fold": fold_name, "scale": scale,
                        **classification_metrics(group.y_true, group.p_up)})
    for period, folds in (("development_oof_feb_apr", ("fold01", "fold02", "fold03")),
                          ("may_validation", ("fold04",))):
        for scale in ("5m", "10m", "15m"):
            left = aggregated.loc[aggregated.scale.eq(scale) & aggregated.fold.isin(folds)]
            for comparator_name, comparator_all in (("logistic_regression", baseline),
                                                    ("soft_voting", voting)):
                right = comparator_all.loc[
                    comparator_all.scale.eq(scale) & comparator_all.fold.isin(folds)
                ]
                contrast = paired_session_bootstrap(
                    left, right, iterations=bootstrap_iterations,
                    seed=seed + len(paired_rows),
                )
                paired_rows.append({"period": period, "scale": scale,
                                    "right": comparator_name, **contrast})
            by_period.append({"period": period, "scale": scale,
                              **classification_metrics(left.y_true, left.p_up)})
    pd.DataFrame(by_fold).to_csv(output_dir / "metrics_by_fold_scale.csv", index=False)
    pd.DataFrame(by_period).to_csv(output_dir / "metrics_by_period_scale.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_mcc_bootstrap.csv", index=False)
    manifest = {
        "method": "Logistic Regression with session-level moving block bootstrap",
        "snapshot_sha256": SNAPSHOT_SHA256, "input_data_sha256": _hash(data_path),
        "baseline_manifest_sha256": _hash(baseline_dir / "manifest.json"),
        "voting_manifest_sha256": _hash(results_root / VOTING_RUN_ID / "manifest.json"),
        "members_per_fold_scale": members, "block_size_sessions": block_size,
        "seed_base": seed, "folds": ["fold01", "fold02", "fold03", "fold04"],
        "rows": len(aggregated), "member_prediction_rows": len(replicate_predictions),
        "exact_population_match_with_baseline": True,
        "may_used_for_fit_or_selection": False, "june_accessed": False,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "sum_individual_fit_seconds": sum(record["fit_seconds"] for record in fit_records),
        "bootstrap": {"unit": "session_date", "iterations": bootstrap_iterations,
                      "multiplicity_adjustment": None},
        "fit_records": fit_records,
        "interpretation": "Exploratory fixed block length; no block-size or model tuning on May/June.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(run_block_bagging(
        args.csv, args.results_root, args.output_dir, members=args.members,
        block_size=args.block_size, seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
