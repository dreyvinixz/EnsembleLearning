"""Run the predeclared 28-source Stage 12 paired max-T comparison."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from paired_session_max_t import REQUIRED, max_t_comparisons, select_ensemble_predictions, sha256


def build_sources(results_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], dict[str, str]]:
    """Load all Stage 12 sources and verify their development-only manifests."""
    sources: dict[str, pd.DataFrame] = {}
    source_map: dict[str, dict[str, str]] = {}
    experiment_names = ["EXP_STAGE11_PAIR_CLASSICAL_42"]
    experiment_names += [
        f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{seed}"
        for architecture in ("MLP", "CNN1D", "LSTM", "GRU")
        for strategy in ("INDEPENDENT", "MTL")
        for seed in (42, 43)
    ]
    experiment_names += [
        "EXP_STAGE12_AVERAGING_20261123",
        "EXP_STAGE12_BLOCKBAG_LOGREG_B5_M10_V2_20261123",
        "EXP_STAGE12_STACKING_20261123",
    ]

    manifest_hashes: dict[str, str] = {}
    snapshot_hashes = set()
    for experiment in experiment_names:
        folder = results_root / experiment
        manifest_path = folder / "manifest.json"
        prediction_path = folder / "predictions.parquet"
        if not manifest_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Missing manifest or predictions for {experiment}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        june_accessed = manifest.get("test_june_accessed", manifest.get("june_accessed"))
        if june_accessed is not False:
            raise ValueError(f"June holdout gate failed for {experiment}: {june_accessed!r}")
        snapshot_hash = manifest.get("data_sha256", manifest.get("snapshot_sha256"))
        if not snapshot_hash:
            raise ValueError(f"Missing snapshot hash in {experiment}")
        snapshot_hashes.add(snapshot_hash)
        manifest_hashes[experiment] = sha256(manifest_path)

    if len(snapshot_hashes) != 1:
        raise ValueError(f"Stage 12 sources do not share one snapshot: {sorted(snapshot_hashes)}")

    def add_direct(method: str, experiment: str) -> None:
        path = results_root / experiment / "predictions.parquet"
        sources[method] = pd.read_parquet(path, columns=REQUIRED)
        source_map[method] = {"experiment": experiment, "artifact": "predictions.parquet"}

    add_direct("LogisticRegression", "EXP_STAGE11_PAIR_CLASSICAL_42")
    for architecture in ("MLP", "CNN1D", "LSTM", "GRU"):
        for strategy in ("INDEPENDENT", "MTL"):
            for seed in (42, 43):
                method = f"{architecture}_{strategy}_seed{seed}"
                add_direct(method, f"EXP_STAGE11_REFIT_{architecture}_{strategy}_{seed}")

    averaging_experiment = "EXP_STAGE12_AVERAGING_20261123"
    averaging_path = results_root / averaging_experiment / "predictions.parquet"
    averaging_frame = pd.read_parquet(averaging_path, columns=[*REQUIRED, "ensemble"])
    ensemble_keys = []
    for architecture in ("MLP", "CNN1D", "LSTM", "GRU"):
        for strategy in ("INDEPENDENT", "MTL"):
            key = f"seed_ensemble_{architecture.lower()}_{strategy.lower()}"
            method = f"SeedEnsemble_{architecture}_{strategy}"
            sources[method] = select_ensemble_predictions(averaging_frame, key)
            source_map[method] = {
                "experiment": averaging_experiment,
                "artifact": "predictions.parquet",
                "ensemble_key": key,
            }
            ensemble_keys.append(key)
    voting_key = "soft_voting_all_architectures_strategies"
    sources["Voting"] = select_ensemble_predictions(averaging_frame, voting_key)
    source_map["Voting"] = {
        "experiment": averaging_experiment,
        "artifact": "predictions.parquet",
        "ensemble_key": voting_key,
    }

    add_direct("BlockBagging", "EXP_STAGE12_BLOCKBAG_LOGREG_B5_M10_V2_20261123")
    add_direct("Stacking", "EXP_STAGE12_STACKING_20261123")
    if len(sources) != 28 or len(source_map) != 28:
        raise AssertionError(f"Expected 28 methods, got {len(sources)}")
    if len(ensemble_keys) != 8 or len(set(ensemble_keys)) != 8:
        raise AssertionError("Expected eight unique seed-ensemble keys")
    return sources, source_map, {
        "snapshot_sha256": next(iter(snapshot_hashes)),
        "manifest_sha256_by_experiment": manifest_hashes,
    }


def run(results_root: Path, output_dir: Path, *, iterations: int, seed: int) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    sources, source_map, source_metadata = build_sources(results_root)
    summary = max_t_comparisons(sources, iterations=iterations, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_csv = output_dir / "paired_max_t.csv"
    summary.to_csv(output_csv, index=False)
    repo_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    unique_sources = sorted({source_map[method]["experiment"] for method in source_map})
    prediction_hashes = {
        experiment: sha256(results_root / experiment / "predictions.parquet")
        for experiment in unique_sources
    }
    manifest = {
        "method": "paired session bootstrap max-T simultaneous intervals",
        "source_commit": commit,
        **source_metadata,
        "sources": source_map,
        "prediction_sha256_by_experiment": prediction_hashes,
        "methods": list(sources),
        "method_count": len(sources),
        "iterations": iterations,
        "seed": seed,
        "primary_metric": "MCC",
        "secondary_metric": "accuracy",
        "decision_threshold": 0.5,
        "multiplicity": "95% max absolute studentized bootstrap deviation across all pairwise method contrasts and both metrics within each period-scale family",
        "period_method_counts": {
            f"{period}/{scale}": int(group.method_count.iloc[0])
            for (period, scale), group in summary.groupby(["period", "scale"], sort=True)
        },
        "period_sample_counts": {
            f"{period}/{scale}": {
                "n": int(group.n.iloc[0]), "sessions": int(group.n_sessions.iloc[0])
            }
            for (period, scale), group in summary.groupby(["period", "scale"], sort=True)
        },
        "common_keys": ["instrument", "session_date", "decision_time", "scale", "fold"],
        "y_true_equality_required": True,
        "June_accessed": False,
        "contains_row_level_predictions": False,
        "script_sha256": sha256(Path(__file__).resolve()),
        "comparison_script_sha256": sha256(Path(__file__).with_name("paired_session_max_t.py")),
        "output_sha256": sha256(output_csv),
        "interpretation": "Exploratory simultaneous intervals; method families and contrasts were not frozen as confirmatory hypotheses.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 1000:
        parser.error("At least 1000 session bootstrap iterations are required")
    print(run(args.results_root, args.output_dir, iterations=args.iterations, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
