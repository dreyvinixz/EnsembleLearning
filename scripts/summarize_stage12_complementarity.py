"""Compute paired prediction diversity without exporting row-level predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["instrument", "session_date", "decision_time", "scale", "fold"]
REQUIRED = {*KEYS, "y_true", "p_up"}
SNAPSHOT_SHA256 = "a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mcc(y: np.ndarray, predicted: np.ndarray) -> float:
    tp = int(np.count_nonzero((y == 1) & (predicted == 1)))
    tn = int(np.count_nonzero((y == 0) & (predicted == 0)))
    fp = int(np.count_nonzero((y == 0) & (predicted == 1)))
    fn = int(np.count_nonzero((y == 1) & (predicted == 0)))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def load_predictions(results_root: Path) -> tuple[
    dict[str, pd.DataFrame], dict[str, str], dict[str, str]
]:
    files: dict[str, Path] = {
        "LogisticRegression": results_root / "EXP_STAGE11_PAIR_CLASSICAL_42/predictions.parquet",
        "Voting": results_root / "EXP_STAGE12_AVERAGING_MEMBERAUDIT_20261123/predictions.parquet",
        "BlockBagging": results_root / "EXP_STAGE12_BLOCKBAG_LOGREG_B5_M10_V2_20261123/predictions.parquet",
        "Stacking": results_root / "EXP_STAGE12_STACKING_20261123/predictions.parquet",
    }
    for architecture in ("MLP", "CNN1D", "LSTM", "GRU"):
        for strategy in ("INDEPENDENT", "MTL"):
            for seed in (42, 43):
                name = f"{architecture}_{strategy}_seed{seed}"
                run = f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{seed}"
                files[name] = results_root / run / "predictions.parquet"
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_snapshot = next((manifest[key] for key in
                                ("data_sha256", "snapshot_sha256", "input_data_sha256")
                                if manifest.get(key)), None)
        if source_snapshot != SNAPSHOT_SHA256:
            raise ValueError(f"{name} has a different data snapshot: {source_snapshot}")
        june_flag = next((manifest[key] for key in
                          ("test_june_accessed", "june_accessed", "June_accessed")
                          if key in manifest), None)
        if june_flag is not False:
            raise ValueError(f"{name} manifest does not affirm that June remained untouched")
        frame = pd.read_parquet(path)
        seed_ensembles: dict[str, pd.DataFrame] = {}
        if name == "Voting":
            if "ensemble" not in frame.columns:
                raise ValueError("Voting prediction file lacks ensemble identifiers")
            for ensemble_name in frame.ensemble.astype(str).unique():
                if ensemble_name.startswith("seed_ensemble_"):
                    derived_name = ensemble_name.removeprefix("seed_ensemble_").upper()
                    seed_ensembles["SeedEnsemble_" + derived_name] = frame.loc[
                        frame.ensemble.eq(ensemble_name)
                    ].copy()
            frame = frame.loc[
                frame.ensemble.eq("soft_voting_all_architectures_strategies")
            ].copy()
        elif name == "Stacking" and "ensemble" in frame.columns:
            frame = frame.loc[frame.ensemble.eq("chronological_oof_stacking")].copy()
        elif name == "BlockBagging" and "ensemble" in frame.columns:
            frame = frame.loc[
                frame.ensemble.eq("temporal_block_bagging_logreg")
            ].copy()
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame[KEYS + ["y_true", "p_up"]].copy()
        frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
        frame["session_date"] = pd.to_datetime(frame.session_date).dt.strftime("%Y-%m-%d")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate sample keys in {path}")
        frames[name] = frame
        hashes[name] = sha256(path)
        manifest_hashes[name] = sha256(manifest_path)
        for seed_name, member in seed_ensembles.items():
                member = member[KEYS + ["y_true", "p_up"]].copy()
                member["decision_time"] = pd.to_datetime(member.decision_time, utc=True)
                member["session_date"] = pd.to_datetime(member.session_date).dt.strftime("%Y-%m-%d")
                if member.duplicated(KEYS).any():
                    raise ValueError(f"Duplicate sample keys in {seed_name}")
                frames[seed_name] = member
                hashes[seed_name] = hashes[name]
                manifest_hashes[seed_name] = manifest_hashes[name]
    return frames, hashes, manifest_hashes


def _pair_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | int]:
    # Inputs are pre-aligned and validated once per period/scale in summarize().
    y = left.y_true.to_numpy(dtype=int)
    p_left = left.p_up.to_numpy(dtype=float)
    p_right = right.p_up.to_numpy(dtype=float)
    pred_left, pred_right = p_left >= .5, p_right >= .5
    err_left, err_right = pred_left != y, pred_right != y
    union = np.count_nonzero(err_left | err_right)
    ensemble = ((p_left + p_right) / 2 >= .5).astype(int)
    return {
        "n": len(left),
        "probability_pearson": float(np.corrcoef(p_left, p_right)[0, 1])
        if np.std(p_left) and np.std(p_right) else None,
        "prediction_agreement": float(np.mean(pred_left == pred_right)),
        "error_jaccard": float(np.count_nonzero(err_left & err_right) / union) if union else 1.0,
        "both_wrong_rate": float(np.mean(err_left & err_right)),
        "left_wrong_right_correct": float(np.mean(err_left & ~err_right)),
        "right_wrong_left_correct": float(np.mean(err_right & ~err_left)),
        "left_mcc": _mcc(y, pred_left),
        "right_mcc": _mcc(y, pred_right),
        "mean_probability_mcc": _mcc(y, ensemble),
        "left_accuracy": float(np.mean(y == pred_left)),
        "right_accuracy": float(np.mean(y == pred_right)),
        "mean_probability_accuracy": float(np.mean(y == ensemble)),
    }


def summarize(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    scopes = {
        "oof_feb_apr": lambda d: d.fold.isin(["fold01", "fold02", "fold03"]),
        "oof_mar_apr": lambda d: d.fold.isin(["fold02", "fold03"]),
        "validation_may": lambda d: d.fold.eq("fold04"),
    }
    for period, selector in scopes.items():
        for scale in ("5m", "10m", "15m"):
            members = {}
            for name, frame in frames.items():
                if period == "oof_feb_apr" and name == "Stacking":
                    continue  # stacking has no February OOF fold by design
                part = frame.loc[selector(frame) & frame.scale.eq(scale)].copy()
                if not part.empty:
                    members[name] = part.sort_values(KEYS, kind="stable").reset_index(drop=True)
            if period == "oof_mar_apr" and "Stacking" not in members:
                raise ValueError("Stacking is required in the March-April paired population")
            if members:
                reference = next(iter(members.values()))
                for name, candidate in members.items():
                    if not reference[KEYS].equals(candidate[KEYS]):
                        raise ValueError(
                            f"{period}/{scale} {name} differs from the common sample keys"
                        )
                    if not reference.y_true.equals(candidate.y_true):
                        raise ValueError(
                            f"{period}/{scale} {name} differs in ground-truth labels"
                        )
            for (left_name, left), (right_name, right) in combinations(members.items(), 2):
                records.append({"period": period, "scale": scale,
                                "model_left": left_name, "model_right": right_name,
                                **_pair_metrics(left, right)})
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    frames, source_hashes, source_manifest_hashes = load_predictions(args.results_root)
    summary = summarize(frames)
    if args.output_csv.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite complementarity artifacts")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    manifest = {
        "source_prediction_sha256": source_hashes,
        "source_manifest_sha256": source_manifest_hashes,
        "snapshot_sha256": SNAPSHOT_SHA256,
        "methods": len(frames), "pair_records": len(summary),
        "key_columns": KEYS, "label_equality_required": True,
        "periods": ["OOF February-April", "OOF March-April", "May validation"],
        "June_accessed": False,
        "output_is_aggregate_only": True,
        "metrics": ["probability correlation", "prediction agreement", "error Jaccard",
                    "both wrong", "one right when other wrong", "paired mean-probability MCC and accuracy"],
        "limitations": "Exploratory descriptive diagnostics; no confidence intervals or multiplicity correction.",
        "output_sha256": sha256(args.output_csv),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} paired aggregate records across {len(frames)} methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
