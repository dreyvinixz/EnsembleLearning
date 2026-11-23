"""Paired evaluation of predictions from independently recorded experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ensemble_learning.contracts import timestamp_column
from ensemble_learning.tracking import classification_metrics

KEYS = ["instrument", "session_date", "decision_time", "scale", "fold"]


def paired_metrics(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare identical prediction populations; reject partial intersections."""
    normalized = []
    for name, original in (("left", left), ("right", right)):
        required = set(KEYS + ["y_true", "p_up"])
        if missing := required - set(original.columns):
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        frame = original.copy()
        if frame[KEYS + ["y_true", "p_up"]].isna().any().any():
            raise ValueError(f"{name} has null keys, labels or probabilities")
        frame["decision_time"] = timestamp_column(frame, "decision_time")
        frame["session_date"] = frame.session_date.astype(str)
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} contains duplicate prediction keys")
        if not frame.y_true.isin([0, 1]).all() or not frame.p_up.between(0, 1).all():
            raise ValueError(f"{name} has invalid labels or probabilities")
        normalized.append(frame[KEYS + ["y_true", "p_up"]])
    paired = normalized[0].merge(
        normalized[1], on=KEYS, how="outer", suffixes=("_left", "_right"),
        validate="one_to_one", indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        only_left = int(paired["_merge"].eq("left_only").sum())
        only_right = int(paired["_merge"].eq("right_only").sum())
        raise ValueError(
            "Prediction populations differ: "
            f"{only_left} keys only on left, {only_right} keys only on right"
        )
    if not paired.y_true_left.eq(paired.y_true_right).all():
        raise ValueError("Conflicting y_true in paired predictions")
    paired = paired.drop(columns="_merge")
    records = []
    for (fold, scale), group in paired.groupby(["fold", "scale"], sort=True):
        for name in ("left", "right"):
            records.append({"fold": fold, "scale": scale, "experiment": name,
                            "available_left": len(normalized[0].loc[
                                normalized[0].fold.eq(fold) & normalized[0].scale.eq(scale)]),
                            "available_right": len(normalized[1].loc[
                                normalized[1].fold.eq(fold) & normalized[1].scale.eq(scale)]),
                            **classification_metrics(group.y_true_left, group[f"p_up_{name}"])})
    return paired, pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two recorded prediction sets")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path, help="New directory for paired results")
    args = parser.parse_args()
    paired, metrics = paired_metrics(pd.read_parquet(args.left), pd.read_parquet(args.right))
    args.output.mkdir(parents=True, exist_ok=False)
    paired.to_parquet(args.output / "paired_predictions.parquet", index=False)
    metrics.to_json(args.output / "metrics.json", orient="records", indent=2)
    return 0
