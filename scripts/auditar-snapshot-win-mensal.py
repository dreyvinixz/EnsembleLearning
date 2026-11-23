"""Audit sample coverage, sessions and B0 classes in the single-contract base."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "data/processed/win_2026_single_contract_jan_may_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/win_single_contract_v1/snapshot_audit.json")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "data/processed/win_2026_single_contract_sequence_jan_may_v1.csv")
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    output = args.output if args.output.is_absolute() else ROOT / args.output
    snapshot = args.snapshot if args.snapshot.is_absolute() else ROOT / args.snapshot
    manifest_copy = output.parent / "sequence_snapshot_manifest.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if manifest_copy.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_copy}")
    coverage = pd.read_csv(base / "session_coverage.csv", dtype={"session_date": str})
    targets = pd.read_parquet(base / "b0_targets_retrospective.parquet")
    market = pd.read_parquet(base / "market_candles.parquet")
    schedule = pd.read_csv(ROOT / "reports/win_single_contract_v1/monthly_schedule.csv", dtype=str)
    schedule = schedule.loc[schedule.contrato.ne("") & schedule.data.le("2026-05-31")]
    coverage["month"] = coverage.session_date.str[:7]
    targets["month"] = targets.session_date.astype(str).str[:7]
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    from prepare_b0_baseline_snapshot import make_snapshot

    from ensemble_learning.sequences import build_causal_windows

    market_for_windows = market.copy()
    target_frame = targets.copy()
    context = make_snapshot(market_for_windows, target_frame, include_context_bars=True)
    context["month"] = context.session_date.astype(str).str[:7]

    def label_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
        return {str(int(k)): int(v) for k, v in frame[column].value_counts(dropna=True).sort_index().items()}

    common = build_causal_windows(
        context, feature_columns=[name for name in context if name.startswith("x_")],
        target_columns=[f"y_{h}m" for h in (5, 10, 15)],
    ).anchors.copy()
    common["month"] = common.session_date.astype(str).str[:7]
    rows = []
    for (month, instrument), group in coverage.groupby(["month", "instrument"], sort=True):
        selected_sessions = int((schedule.mes == month).sum())
        rows.append({
            "month": month, "instrument": instrument,
            "expected_regular_sessions": selected_sessions,
            "coverage_audit_sessions": len(group),
            "sessions_with_any_80pct_window": int(group.has_any_80pct_window.sum()),
            "sessions_without_source_candles": int(group.observed_candles.eq(0).sum()),
            "observed_candles": int(group.observed_candles.sum()),
            "label_rows": len(targets.loc[(targets.month == month) & (targets.instrument == instrument)]),
        })
    scale_distribution = {}
    for horizon in (5, 10, 15):
        column = f"y_b0_{horizon}m"
        present = targets.loc[targets[column].notna()]
        sequence_column = f"y_{horizon}m"
        windows = build_causal_windows(
            context, feature_columns=[name for name in context if name.startswith("x_")],
            target_columns=[sequence_column],
        )
        accepted = windows.anchors.copy()
        accepted["month"] = accepted.session_date.astype(str).str[:7]
        scale_distribution[f"{horizon}m"] = {
            "non_null_rows": len(present), "classes": label_counts(present, column),
            "accepted_prediction_samples_60m_48": len(accepted),
            "accepted_classes": label_counts(accepted, sequence_column),
            "accepted_common_three_targets": len(common),
            "common_classes": label_counts(common, sequence_column),
            "by_contract": {
                instrument: {"non_null_rows": len(part), "classes": label_counts(part, column),
                             "accepted_samples": int((accepted.instrument == instrument).sum()),
                             "accepted_classes": label_counts(accepted.loc[accepted.instrument == instrument], sequence_column)}
                for instrument, part in present.groupby("instrument", sort=True)
            },
            "by_month": {
                month: {"non_null_rows": len(part), "classes": label_counts(part, column),
                        "accepted_samples": int((accepted.month == month).sum()),
                        "accepted_classes": label_counts(accepted.loc[accepted.month == month], sequence_column)}
                for month, part in present.groupby("month", sort=True)
            },
        }
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "protocol_id": "win_single_contract_monthly_v1",
        "source_period_cap": "2026-05-31",
        "january_selection": "WING26 fixed by study owner; December coverage gate cannot be verified from local history",
        "base_manifest_sha256": sha256(base / "manifest.json"),
        "sequence_snapshot_sha256": sha256(snapshot),
        "sequence_snapshot_manifest_sha256": sha256(snapshot.with_suffix(".manifest.json")),
        "sequence_snapshot_manifest_copy": manifest_copy.name,
        "figure_sha256": {
            name: sha256(ROOT / "figures" / name) for name in (
                "contratos-win-contrato-mensal-v1.png",
                "contratos-win-contrato-mensal-v1.pdf",
                "contratos-win-contrato-mensal-v1.csv",
            )
        },
        "parquet_sha256": {name: sha256(base / name) for name in
                           ("market_candles.parquet", "b0_targets_retrospective.parquet")},
        "source_hashes": {row.get("table", ",".join(row.get("tables", []))): row["sha256"]
                          for row in manifest["sources"]},
        "coverage_by_month_contract": rows,
        "market_rows": len(market), "target_rows": len(targets),
        "common_complete_prediction_samples_60m_48": len(common),
        "contracts": sorted(targets.instrument.unique().tolist()),
        "features_preserved": "21 OHLCV-derived causal features; 60-minute window, minimum 48 candles",
        "label_distributions": scale_distribution,
        "limitations": [
            "Counts describe retrospective source coverage and do not establish real-time vintage availability.",
            "B0 release/confirmation time is not observed; the training pipeline uses the documented simulated post-session policy.",
            "Monthly coverage gate is a selection criterion; individual prediction anchors still require 48 of 60 candles.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_copy.write_bytes(snapshot.with_suffix(".manifest.json").read_bytes())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output), "market_rows": len(market), "target_rows": len(targets),
                      "coverage_months": sorted(coverage.month.unique()), "contracts": result["contracts"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
