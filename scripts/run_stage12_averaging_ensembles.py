"""Build fixed-uniform seed and soft-voting ensembles from Stage 11 runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from ensemble_learning.ensembles import mean_probability_ensemble
from ensemble_learning.pairing import KEYS, paired_metrics
from ensemble_learning.tracking import classification_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_stage11_results import paired_session_bootstrap

ARCHITECTURES = ("MLP", "CNN1D", "LSTM", "GRU")
STRATEGIES = ("INDEPENDENT", "MTL")
SEEDS = (42, 43)
SCALES = ("5m", "10m", "15m")
FOLDS = ("fold01", "fold02", "fold03", "fold04")
SNAPSHOT_SHA256 = "a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91"
BASELINE_ID = "EXP_STAGE11_PAIR_CLASSICAL_42"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(results_root: Path, run_id: str) -> tuple[pd.DataFrame, dict]:
    run_dir = results_root / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_sha256") != SNAPSHOT_SHA256:
        raise ValueError(f"{run_id} uses a different training snapshot")
    if manifest.get("test_june_accessed") is not False:
        raise ValueError(f"{run_id} does not affirm that June remained untouched")
    if run_id != BASELINE_ID:
        config = json.loads((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if config.get("refit_after_early_stopping") is not True:
            raise ValueError(f"{run_id} does not use the full-fold refit protocol")
        summary = json.loads((run_dir / "training_summary.json").read_text(
            encoding="utf-8"))
        if not all(fit.get("refit_after_early_stopping") for fit in summary["fold_fits"]):
            raise ValueError(f"{run_id} has a fold without full-fold refit")
    frame = pd.read_parquet(run_dir / "predictions.parquet")
    if frame.session_date.astype(str).max() > "2026-05-31":
        raise ValueError(f"{run_id} includes data beyond May")
    if len(frame) != 170_100 or frame.duplicated(KEYS).any():
        raise ValueError(f"{run_id} has invalid prediction count or duplicate keys")
    if set(frame.fold.astype(str)) != set(FOLDS) or set(frame.scale.astype(str)) != set(SCALES):
        raise ValueError(f"{run_id} is missing a fold or scale")
    return frame, manifest


def _metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_fold, by_session, by_period = [], [], []
    for ensemble_name, frame in predictions.groupby("ensemble", sort=True):
        for (fold, scale), group in frame.groupby(["fold", "scale"], sort=True):
            by_fold.append({"ensemble": ensemble_name, "fold": fold, "scale": scale,
                            **classification_metrics(group.y_true, group.p_up)})
            for session_date, session in group.groupby("session_date", sort=True):
                by_session.append({"ensemble": ensemble_name, "fold": fold,
                                   "scale": scale, "session_date": session_date,
                                   **classification_metrics(session.y_true, session.p_up)})
        for period, folds in (("development_oof_feb_apr", FOLDS[:3]),
                              ("may_validation", (FOLDS[3],))):
            part = frame[frame.fold.isin(folds)]
            for scale, group in part.groupby("scale", sort=True):
                by_period.append({"ensemble": ensemble_name, "period": period,
                                  "scale": scale,
                                  **classification_metrics(group.y_true, group.p_up)})
    return pd.DataFrame(by_fold), pd.DataFrame(by_session), pd.DataFrame(by_period)


def build_averaging_ensembles(
    results_root: Path, output_dir: Path, *, bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 42,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    loaded: dict[str, pd.DataFrame] = {}
    manifests = {}
    for architecture in ARCHITECTURES:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                run_id = f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{seed}"
                loaded[run_id], manifests[run_id] = _load_run(results_root, run_id)
    baseline, _ = _load_run(results_root, BASELINE_ID)
    for run_id, frame in loaded.items():
        paired, _ = paired_metrics(baseline, frame)
        if len(paired) != len(baseline):
            raise ValueError(f"{run_id} differs from the complete baseline population")

    ensembles = []
    for architecture in ARCHITECTURES:
        for strategy in STRATEGIES:
            members = {
                f"seed{seed}": loaded[
                    f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{seed}"
                ] for seed in SEEDS
            }
            ensemble_name = f"seed_ensemble_{architecture.lower()}_{strategy.lower()}"
            ensembles.append(mean_probability_ensemble(members, ensemble_name=ensemble_name))
    seed_ensembles = {name: frame for name, frame in zip(
        (f"seed_ensemble_{a.lower()}_{s.lower()}"
         for a in ARCHITECTURES for s in STRATEGIES), ensembles, strict=True
    )}
    voting = mean_probability_ensemble(
        seed_ensembles, ensemble_name="soft_voting_all_architectures_strategies"
    )
    ensembles.append(voting)
    predictions = pd.concat(ensembles, ignore_index=True)

    # Verify every emitted member against the same source labels and complete keys.
    for ensemble_name, frame in predictions.groupby("ensemble", sort=True):
        reference, _ = paired_metrics(baseline, frame)
        if len(reference) != len(baseline):
            raise ValueError(f"{ensemble_name} lost or added prediction observations")

    output_dir.mkdir(parents=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    by_fold, by_session, by_period = _metrics(predictions)
    by_fold.to_csv(output_dir / "metrics_by_fold_scale.csv", index=False)
    by_session.to_csv(output_dir / "metrics_by_session.csv", index=False)
    by_period.to_csv(output_dir / "metrics_by_period_scale.csv", index=False)

    paired_rows = []
    baseline = baseline.sort_values(KEYS, kind="stable").reset_index(drop=True)
    for ensemble_name, frame in predictions.groupby("ensemble", sort=True):
        for period, folds in (("development_oof_feb_apr", FOLDS[:3]),
                              ("may_validation", (FOLDS[3],))):
            for scale in SCALES:
                left = frame[frame.scale.eq(scale) & frame.fold.isin(folds)]
                right = baseline[baseline.scale.eq(scale) & baseline.fold.isin(folds)]
                contrast = paired_session_bootstrap(
                    left, right, iterations=bootstrap_iterations,
                    seed=bootstrap_seed + len(paired_rows),
                )
                paired_rows.append({"ensemble": ensemble_name, "period": period,
                                    "scale": scale, "baseline": BASELINE_ID, **contrast})
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "paired_mcc_vs_logistic_bootstrap.csv", index=False
    )

    member_rows = []
    for ensemble_name, ensemble_frame in seed_ensembles.items():
        _, _, architecture, strategy = ensemble_name.split("_")
        for seed in SEEDS:
            member_id = f"EXP_STAGE11_REFIT_{architecture.upper()}_{strategy.upper()}_{seed}"
            for period, folds in (("development_oof_feb_apr", FOLDS[:3]),
                                  ("may_validation", (FOLDS[3],))):
                for scale in SCALES:
                    left = ensemble_frame[
                        ensemble_frame.scale.eq(scale) & ensemble_frame.fold.isin(folds)
                    ]
                    right = loaded[member_id][
                        loaded[member_id].scale.eq(scale)
                        & loaded[member_id].fold.isin(folds)
                    ]
                    contrast = paired_session_bootstrap(
                        left, right, iterations=bootstrap_iterations,
                        seed=bootstrap_seed + len(member_rows),
                    )
                    member_rows.append({
                        "ensemble": ensemble_name, "member": member_id,
                        "period": period, "scale": scale, **contrast,
                    })
    voting_frame = voting[voting.ensemble.eq("soft_voting_all_architectures_strategies")]
    for member_name, member_frame in seed_ensembles.items():
        for period, folds in (("development_oof_feb_apr", FOLDS[:3]),
                              ("may_validation", (FOLDS[3],))):
            for scale in SCALES:
                left = voting_frame[
                    voting_frame.scale.eq(scale) & voting_frame.fold.isin(folds)
                ]
                right = member_frame[
                    member_frame.scale.eq(scale) & member_frame.fold.isin(folds)
                ]
                contrast = paired_session_bootstrap(
                    left, right, iterations=bootstrap_iterations,
                    seed=bootstrap_seed + len(member_rows),
                )
                member_rows.append({
                    "ensemble": "soft_voting_all_architectures_strategies",
                    "member": member_name, "period": period, "scale": scale,
                    **contrast,
                })
    pd.DataFrame(member_rows).to_csv(
        output_dir / "paired_mcc_vs_members_bootstrap.csv", index=False
    )

    source_paths = {
        run_id: str(results_root / run_id / "predictions.parquet")
        for run_id in loaded
    }
    source_paths[BASELINE_ID] = str(results_root / BASELINE_ID / "predictions.parquet")
    metadata = {
        "method": "uniform mean of probabilities; fixed weights, no fitted meta-model",
        "seed_ensemble_members": len(SEEDS),
        "soft_voting_members": len(seed_ensembles),
        "soft_voting_sources": "uniform seed ensembles across 4 architectures x 2 strategies",
        "fitted_on_oof": False,
        "fit_period": None,
        "oof_reported_as_development": "fold01-fold03, February-April",
        "may_used_for_fitting_or_weight_selection": False,
        "june_accessed": False,
        "snapshot_sha256": SNAPSHOT_SHA256,
        "baseline_manifest_sha256": _hash(results_root / BASELINE_ID / "manifest.json"),
        "source_prediction_sha256": {name: _hash(Path(path))
                                      for name, path in source_paths.items()},
        "source_model_commits": sorted({
            manifest.get("git_commit", "unknown") for manifest in manifests.values()
        }),
        "prediction_rows_per_ensemble": 170_100,
        "ensemble_count": len(ensembles),
        "metrics": ["MCC", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
                    "average_precision", "brier", "log_loss", "ECE"],
        "paired_bootstrap": {"unit": "session_date", "iterations": bootstrap_iterations,
                             "seed_base": bootstrap_seed, "multiplicity_adjustment": None},
        "interpretation": "Exploratory; equal-weight aggregation is pre-specified, not tuned on May or June.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(build_averaging_ensembles(
        args.results_root, args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.seed,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
