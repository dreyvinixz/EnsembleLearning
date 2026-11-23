"""Run a leakage-aware feature-group screen on the Jan-May development snapshot.

Each variant uses the existing expanding walk-forward runner and Logistic
Regression baseline. The script does not load or inspect June.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


GROUP_RULES = {
    "returns": lambda name: name == "x_return_1bar" or bool(re.fullmatch(r"x_return_\d+bars", name)),
    "candles_price_shape": lambda name: name in {
        "x_range_fraction", "x_body_fraction", "x_close_location", "x_average_close_gap",
    },
    "volatility": lambda name: name.startswith("x_return_volatility_"),
    "trend_vs_sma": lambda name: name.startswith("x_close_vs_sma_"),
    "volume_activity": lambda name: name in {
        "x_log_volume", "x_log_business", "x_log_amount", "x_log_volume_zscore_30bars",
    },
}
TARGETS = ["y_5m", "y_10m", "y_15m"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {group: [name for name in columns if rule(name)] for group, rule in GROUP_RULES.items()}
    assigned = [name for values in groups.values() for name in values]
    if sorted(assigned) != sorted(columns) or len(assigned) != len(set(assigned)):
        unassigned = sorted(set(columns) - set(assigned))
        duplicates = sorted(name for name in set(assigned) if assigned.count(name) > 1)
        raise ValueError(f"Feature taxonomy must be exhaustive and disjoint; unassigned={unassigned}, duplicated={duplicates}")
    if any(not values for values in groups.values()):
        raise ValueError("Every pre-specified feature group must contain at least one feature")
    return groups


def verify_common_eligibility(frame: pd.DataFrame, columns: list[str], groups: dict[str, list[str]]) -> int:
    from ensemble_learning.sequences import build_causal_windows

    reference = build_causal_windows(frame, feature_columns=columns, target_columns=TARGETS)
    keys = ["instrument", "session_date", "decision_time"]
    expected = reference.anchors[keys].reset_index(drop=True)
    for feature_set in [*groups.values(), *(
        [name for name in columns if name not in dropped]
        for dropped in groups.values()
    )]:
        candidate = build_causal_windows(frame, feature_columns=feature_set, target_columns=TARGETS)
        observed = candidate.anchors[keys].reset_index(drop=True)
        if not expected.equals(observed):
            raise AssertionError("Feature ablation changed the eligible anchor population")
    return len(expected)


def run_ablation(snapshot: Path, results_root: Path, report_dir: Path, *, seed: int = 42) -> Path:
    from ensemble_learning.contracts import feature_columns, validate_sequence_frame
    from ensemble_learning.tracking import run_walk_forward

    manifest_path = snapshot.with_suffix(".manifest.json")
    if not snapshot.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Snapshot CSV and its source manifest are required")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    period = source_manifest.get("period", {})
    if period.get("start", "9999") < "2026-01-01" or period.get("end", "0000") > "2026-05-31":
        raise ValueError("Feature ablation accepts development data Jan-May 2026 only")
    if source_manifest.get("test_policy") and "not queried or read" not in source_manifest["test_policy"]:
        raise ValueError("Source manifest does not prove the June boundary")
    if report_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {report_dir}")

    frame = validate_sequence_frame(pd.read_csv(snapshot))
    columns = feature_columns(frame)
    groups = assign_feature_groups(columns)
    eligible = verify_common_eligibility(frame, columns, groups)
    snapshot_hash = sha256(snapshot)
    report_dir.mkdir(parents=True, exist_ok=False)
    records = []
    variants: dict[str, list[str]] = {"full": columns}
    for group, members in groups.items():
        variants[f"only_{group}"] = members
        variants[f"without_{group}"] = [name for name in columns if name not in members]

    for variant, selected in variants.items():
        variant_frame = frame.drop(columns=[name for name in columns if name not in selected])
        experiment_id = f"EXP_FGROUP_{variant.upper()}_S{seed}"
        output = run_walk_forward(
            variant_frame, model_name="logreg", seed=seed, data_path=snapshot,
            results_root=results_root, experiment_id=experiment_id,
        )
        run_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if run_manifest.get("data_sha256") != snapshot_hash or run_manifest.get("test_june_accessed") is not False:
            raise ValueError(f"Experiment manifest failed source/holdout check: {experiment_id}")
        records.append({
            "variant": variant,
            "experiment_id": experiment_id,
            "feature_count": len(selected),
            "features": selected,
            "eligible_anchors_per_variant": eligible,
            "result_manifest_sha256": sha256(output / "manifest.json"),
        })

    result = {
        "experiment_family": "feature_group_ablation_v1",
        "snapshot_sha256": snapshot_hash,
        "source_manifest_sha256": sha256(manifest_path),
        "date_boundary": {"start": "2026-01-01", "end": "2026-05-31", "June_accessed": False},
        "model": "LogisticRegression",
        "seed": seed,
        "metric_primary": "MCC at fixed threshold 0.5",
        "design": "full, each group only, and full minus one group; identical chronological folds and anchor eligibility",
        "feature_groups": groups,
        "variants": records,
        "interpretation": "Exploratory screening. Group ablations are not causal feature importance and may interact; evaluate planned contrasts with session-block paired intervals and multiplicity control.",
    }
    (report_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (report_dir / "feature_groups.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in records]).to_csv(
        report_dir / "variants.csv", index=False
    )
    return report_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Jan-May sequence CSV with adjacent manifest")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_ablation(args.snapshot, args.results_root, args.report_dir, seed=args.seed)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
