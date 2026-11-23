"""Compare historical and single-month protocol sample populations through May."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ensemble_learning.contracts import feature_columns
from ensemble_learning.sequences import build_causal_windows

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/processed/win_2026_development_jan_may_v1.csv"
NEW = ROOT / "data/processed/win_2026_single_contract_sequence_jan_may_v1.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile(path: Path) -> dict:
    frame = pd.read_csv(path)
    frame["month"] = frame.session_date.astype(str).str[:7]
    if frame.session_date.max() > "2026-05-31":
        raise ValueError("Comparison input exceeds May; June access is forbidden")
    if len([column for column in frame if column.startswith("x_")]) != 21:
        raise ValueError(f"Expected 21 features in {path}")
    features = feature_columns(frame)
    scales = {}
    for horizon in (5, 10, 15):
        target = f"y_{horizon}m"
        windows = build_causal_windows(frame, feature_columns=features, target_columns=[target])
        anchors = windows.anchors
        anchors["month"] = anchors.session_date.astype(str).str[:7]
        scales[target] = {
            "samples": len(anchors),
            "classes": {str(int(k)): int(v) for k, v in anchors[target].value_counts().sort_index().items()},
            "by_month": {
                month: {"samples": len(group), "classes": {
                    str(int(k)): int(v) for k, v in group[target].value_counts().sort_index().items()
                }}
                for month, group in anchors.groupby("month", sort=True)
            },
            "by_contract": {
                ticker: {"samples": len(group), "classes": {
                    str(int(k)): int(v) for k, v in group[target].value_counts().sort_index().items()
                }}
                for ticker, group in anchors.groupby("instrument", sort=True)
            },
        }
    joint = build_causal_windows(
        frame, feature_columns=features, target_columns=[f"y_{h}m" for h in (5, 10, 15)]
    ).anchors
    joint["month"] = joint.session_date.astype(str).str[:7]
    return {
        "sha256": sha256(path), "rows": len(frame),
        "date_range": [frame.session_date.min(), frame.session_date.max()],
        "contracts": sorted(frame.instrument.unique().tolist()),
        "contracts_by_month": {
            month: sorted(group.instrument.unique().tolist())
            for month, group in frame.groupby("month", sort=True)
        },
        "session_dates_by_month": frame.groupby("month").session_date.nunique().astype(int).to_dict(),
        "common_three_target_samples": len(joint),
        "common_samples_by_month": joint.groupby("month").size().astype(int).to_dict(),
        "scale_samples": scales,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--new", type=Path, default=NEW)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/win_single_contract_v1/protocol_comparison.json")
    args = parser.parse_args()
    paths = [value if value.is_absolute() else ROOT / value for value in (args.old, args.new, args.output)]
    old, new, output = paths
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    old_profile, new_profile = profile(old), profile(new)
    old_samples = old_profile["common_three_target_samples"]
    new_samples = new_profile["common_three_target_samples"]
    result = {
        "period": "through 2026-05-31 only; June not read",
        "old_protocol": old_profile,
        "new_protocol": new_profile,
        "common_sample_change": {
            "absolute": new_samples - old_samples,
            "fraction": round((new_samples - old_samples) / old_samples, 6),
            "interpretation": "population/coverage change only; no model performance comparison was run",
        },
        "caveats": [
            "The old input is the retained historical Jan-May development CSV; it is not rewritten.",
            "Anchor counts apply the same causal 60-minute window and 48-candle minimum to both CSVs.",
            "Metrics of model performance require rerunning every method on the same frozen population.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["common_sample_change"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
