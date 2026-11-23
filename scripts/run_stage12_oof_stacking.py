"""Fit per-scale stackers on earlier OOF folds and predict later folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ensemble_learning.pairing import KEYS, paired_metrics
from ensemble_learning.stacking import chronological_fold_schedule, stacking_feature_frame
from ensemble_learning.tracking import classification_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_stage11_results import paired_session_bootstrap

OOF_FOLDS = ("fold01", "fold02", "fold03")
EVALUATION_FOLDS = (*OOF_FOLDS, "fold04")
SCALES = ("5m", "10m", "15m")
ENSEMBLE_RUN_ID = "EXP_STAGE12_AVERAGING_MEMBERAUDIT_20261123"
BASELINE_ID = "EXP_STAGE11_PAIR_CLASSICAL_42"
SNAPSHOT_SHA256 = "a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_oof_stacking(
    results_root: Path, output_dir: Path, *, bootstrap_iterations: int = 5000,
    seed: int = 42,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    ensemble_dir = results_root / ENSEMBLE_RUN_ID
    ensemble_manifest = json.loads((ensemble_dir / "manifest.json").read_text())
    if ensemble_manifest.get("snapshot_sha256") != SNAPSHOT_SHA256:
        raise ValueError("Averaging ensemble has the wrong source snapshot")
    if ensemble_manifest.get("june_accessed") is not False:
        raise ValueError("Averaging ensemble did not preserve the June holdout")
    all_predictions = pd.read_parquet(ensemble_dir / "predictions.parquet")
    base_members = {
        name: frame.copy()
        for name, frame in all_predictions.groupby("ensemble", sort=True)
        if name.startswith("seed_ensemble_")
    }
    if len(base_members) != 8:
        raise ValueError(f"Expected 8 seed-ensemble base features; found {len(base_members)}")
    baseline_dir = results_root / BASELINE_ID
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text())
    if baseline_manifest.get("data_sha256") != SNAPSHOT_SHA256:
        raise ValueError("Classical comparison uses a different snapshot")
    baseline = pd.read_parquet(baseline_dir / "predictions.parquet")
    features = stacking_feature_frame(base_members)
    if features.session_date.astype(str).max() > "2026-05-31":
        raise ValueError("Stacking inputs extend beyond May")

    schedule = chronological_fold_schedule(EVALUATION_FOLDS, EVALUATION_FOLDS)
    rows, fits = [], []
    for scale in SCALES:
        scale_frame = features.loc[features.scale.eq(scale)].copy()
        for item in schedule:
            prediction_fold = item["prediction_fold"]
            train_folds = item["meta_train_folds"]
            train = scale_frame.loc[scale_frame.fold.isin(train_folds)]
            predict = scale_frame.loc[scale_frame.fold.eq(prediction_fold)]
            if train.empty or predict.empty:
                raise ValueError(f"Empty stacking train or prediction fold: {scale}/{prediction_fold}")
            if set(train.fold.astype(str)) - set(OOF_FOLDS):
                raise ValueError("Stacker training contains a non-OOF fold")
            if not all(EVALUATION_FOLDS.index(fold) < EVALUATION_FOLDS.index(prediction_fold)
                       for fold in train.fold.astype(str).unique()):
                raise ValueError("Stacker training reaches the prediction fold or later")
            y_train = train.y_true.to_numpy(dtype=int)
            if set(y_train) != {0, 1}:
                raise ValueError(f"Stacker training lacks both classes: {scale}/{prediction_fold}")
            feature_columns = [column for column in features.columns if column.startswith("p_")]
            model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
            model.fit(train[feature_columns], y_train)
            probability = model.predict_proba(predict[feature_columns])[:, 1]
            prediction = predict[KEYS + ["y_true"]].copy()
            prediction["p_up"] = probability
            prediction["p_down"] = 1.0 - probability
            prediction["y_pred"] = (probability >= 0.5).astype("int8")
            prediction["ensemble"] = "chronological_oof_stacking"
            prediction["member_count"] = len(feature_columns)
            prediction["members"] = ",".join(feature_columns)
            prediction["meta_train_folds"] = ",".join(train_folds)
            rows.append(prediction)
            fits.append({
                "scale": scale, "prediction_fold": prediction_fold,
                "meta_train_folds": list(train_folds), "n_meta_train": len(train),
                "meta_train_start": str(train.session_date.min()),
                "meta_train_end": str(train.session_date.max()),
                "prediction_start": str(predict.session_date.min()),
                "prediction_end": str(predict.session_date.max()),
                "features": feature_columns, "coefficient": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]), "C": 1.0,
                "class_weight": None, "random_state": seed,
            })
    predictions = pd.concat(rows, ignore_index=True)
    if predictions.session_date.astype(str).max() > "2026-05-31":
        raise ValueError("Stacking output extends beyond May")

    output_dir.mkdir(parents=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    by_fold, by_session, by_period = [], [], []
    for (fold, scale), group in predictions.groupby(["fold", "scale"], sort=True):
        by_fold.append({"fold": fold, "scale": scale,
                        **classification_metrics(group.y_true, group.p_up)})
        for session_date, session in group.groupby("session_date", sort=True):
            by_session.append({"fold": fold, "scale": scale, "session_date": session_date,
                               **classification_metrics(session.y_true, session.p_up)})
    evaluation_sets = (
        ("development_oof_march_april", predictions.fold.isin(("fold02", "fold03"))),
        ("may_validation", predictions.fold.eq("fold04")),
    )
    for period, selected in evaluation_sets:
        part = predictions.loc[selected]
        for scale, group in part.groupby("scale", sort=True):
            by_period.append({"period": period, "scale": scale,
                              **classification_metrics(group.y_true, group.p_up)})
    pd.DataFrame(by_fold).to_csv(output_dir / "metrics_by_fold_scale.csv", index=False)
    pd.DataFrame(by_session).to_csv(output_dir / "metrics_by_session.csv", index=False)
    pd.DataFrame(by_period).to_csv(output_dir / "metrics_by_period_scale.csv", index=False)

    bootstrap_rows = []
    voting = all_predictions.loc[
        all_predictions.ensemble.eq("soft_voting_all_architectures_strategies")
    ]
    for period, folds in (("development_oof_march_april", ("fold02", "fold03")),
                          ("may_validation", ("fold04",))):
        for scale in SCALES:
            stacked = predictions.loc[predictions.scale.eq(scale)
                                      & predictions.fold.isin(folds)]
            for comparator_name, comparator_all in (("logistic_regression", baseline),
                                                    ("soft_voting", voting)):
                comparator = comparator_all.loc[
                    comparator_all.scale.eq(scale) & comparator_all.fold.isin(folds)
                ]
                paired, _ = paired_metrics(stacked, comparator)
                if len(paired) != len(stacked):
                    raise ValueError("Stacking comparison is not on an exact paired population")
                contrast = paired_session_bootstrap(
                    stacked, comparator, iterations=bootstrap_iterations,
                    seed=seed + len(bootstrap_rows),
                )
                bootstrap_rows.append({
                    "period": period, "scale": scale,
                    "left": "chronological_oof_stacking", "right": comparator_name,
                    **contrast,
                })
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "paired_mcc_bootstrap.csv", index=False
    )
    manifest = {
        "method": "per-scale LogisticRegression stacker on earlier-fold seed-ensemble probabilities",
        "base_prediction_manifest_sha256": _hash(ensemble_dir / "manifest.json"),
        "base_predictions_sha256": _hash(ensemble_dir / "predictions.parquet"),
        "baseline_manifest_sha256": _hash(baseline_dir / "manifest.json"),
        "snapshot_sha256": SNAPSHOT_SHA256,
        "meta_features": sorted(column for column in features if column.startswith("p_")),
        "chronological_schedule": schedule,
        "fold01_predictions": "not produced: no prior OOF folds exist for meta-training",
        "may_used_for_meta_training": False,
        "june_accessed": False,
        "rows": len(predictions),
        "prediction_periods": ["fold02 March", "fold03 April", "fold04 May validation"],
        "fit_records": fits,
        "bootstrap": {"unit": "session_date", "iterations": bootstrap_iterations,
                      "seed_base": seed, "multiplicity_adjustment": None},
        "interpretation": "Exploratory; meta-model is trained only on predictions from strictly earlier OOF folds.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(run_oof_stacking(
        args.results_root, args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations, seed=args.seed,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
