"""Entry point for the first reproducible validation-only experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from ensemble_learning import __version__
from ensemble_learning.baselines import CLASSICAL_MODELS, run_independent_classical
from ensemble_learning.contracts import validate_frame
from ensemble_learning.splits import dated_holdout
from ensemble_learning.tracking import run_walk_forward


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audited independent BovDB baseline")
    parser.add_argument("--csv", type=Path, required=True, help="Approved snapshot, no DB writes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=CLASSICAL_MODELS, default="logreg")
    parser.add_argument("--output", type=Path, default=Path("results/independent_logreg.json"))
    parser.add_argument("--evaluate-test", action="store_true", help="One-time final holdout evaluation")
    parser.add_argument("--protocol-id", default=None, help="Frozen preregistered protocol reference")
    args = parser.parse_args(argv)
    if args.evaluate_test and not args.protocol_id:
        parser.error("--evaluate-test requires --protocol-id (explicit final-test access)")
    raw = args.csv.read_bytes()
    data_hash = hashlib.sha256(raw).hexdigest()
    frame = validate_frame(pd.read_csv(args.csv))
    split = dated_holdout(frame)
    if args.evaluate_test and split.test.empty:
        parser.error("Final test evaluation requires June data")
    outcome = run_independent_classical(
        split, model_name=args.model, seed=args.seed, evaluate_test=args.evaluate_test
    )
    report = {
        "version": __version__,
        "source_sha256": data_hash,
        "source_file": args.csv.name,
        "packages": {"pandas": version("pandas"), "scikit-learn": version("scikit-learn"),
                     "xgboost": version("xgboost")},
        "test_evaluated": bool(args.evaluate_test),
        "protocol_id": args.protocol_id,
        "split": {
            "train_sessions": split.train_sessions,
            "validation_sessions": split.validation_sessions,
            "test_sessions": split.test_sessions,
            "train_rows_after_purge": len(split.train),
            "validation_rows_after_purge": len(split.validation),
            "test_rows_reserved": len(split.test),
            "purged_train_rows": split.purged_train,
            "purged_validation_rows": split.purged_validation,
        },
        "experiment": outcome,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new path")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written: {args.output}")
    print("TEST was evaluated" if args.evaluate_test else "TEST reserved and NOT evaluated")
    return 0


def walk_forward_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tracked expanding walk-forward classical experiment")
    parser.add_argument("--csv", type=Path, required=True, help="Development snapshot, Jan-May 2026 only")
    parser.add_argument("--model", choices=CLASSICAL_MODELS, default="logreg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    frame = validate_frame(pd.read_csv(args.csv))
    result = run_walk_forward(frame, model_name=args.model, seed=args.seed,
        data_path=args.csv, results_root=args.results, experiment_id=args.experiment_id)
    print(f"Experiment artifacts: {result}")
    return 0


def neural_walk_forward_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tracked PyTorch B0 sequence experiments")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Full-context B0 snapshot, Jan-May 2026; unlabeled candles are retained")
    parser.add_argument("--architecture", choices=("all", "mlp", "cnn1d", "lstm", "gru"),
                        default="all")
    parser.add_argument("--strategy", choices=("all", "independent", "mtl"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--experiment-id", help="Only valid for a single architecture/strategy pair")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args(argv)

    from ensemble_learning.contracts import validate_sequence_frame
    from ensemble_learning.neural_training import (
        NEURAL_ARCHITECTURES,
        STRATEGIES,
        run_neural_experiment,
    )

    frame = validate_sequence_frame(pd.read_csv(args.csv))
    architectures = NEURAL_ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    if args.experiment_id and (len(architectures) != 1 or len(strategies) != 1):
        parser.error("--experiment-id requires one architecture and one strategy")
    outputs = []
    for architecture in architectures:
        for strategy in strategies:
            result = run_neural_experiment(
                frame, architecture=architecture, strategy=strategy,
                data_path=args.csv, results_root=args.results, seed=args.seed,
                experiment_id=args.experiment_id, device_name=args.device,
                epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                hidden_size=args.hidden_size, learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            outputs.append(str(result))
    print("\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
