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


if __name__ == "__main__":
    raise SystemExit(main())
