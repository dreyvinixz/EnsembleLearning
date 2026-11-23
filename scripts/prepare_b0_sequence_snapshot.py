"""Build Jan-May B0 targets with all observed candles retained for window context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from prepare_b0_baseline_snapshot import make_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data/processed/win_2026_jan_may_v1"
DEFAULT_OUT = ROOT / "data/processed/win_2026_sequence_jan_may_v1.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help="Read-only Parquet base already extracted through May only")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--end-date", default="2026-05-31")
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.end_date > "2026-05-31" or args.end_date < "2026-01-01":
        parser.error("Stage 11 development data must end between Jan 1 and May 31, 2026")
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {manifest_path}")
    market_source = base / "market_candles.parquet"
    label_source = base / "b0_targets_retrospective.parquet"
    base_manifest = base / "manifest.json"
    for source in (market_source, label_source, base_manifest):
        if not source.is_file():
            raise FileNotFoundError(source)

    source_manifest = json.loads(base_manifest.read_text(encoding="utf-8"))
    source_end = source_manifest.get("period_inclusive", {}).get("end")
    if source_end is None or source_end > "2026-05-31":
        parser.error("Base manifest does not prove that the source extraction ends by May 31")
    market = pd.read_parquet(market_source)
    labels = pd.read_parquet(label_source)
    market["session_date"] = market.session_date.astype(str)
    labels["session_date"] = labels.session_date.astype(str)
    market = market.loc[market.session_date.between("2026-01-01", args.end_date)].copy()
    labels = labels.loc[labels.session_date.between("2026-01-01", args.end_date)].copy()
    snapshot = make_snapshot(market, labels, include_context_bars=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = snapshot.copy()
    for column in serializable.select_dtypes(include=["datetimetz"]).columns:
        serializable[column] = serializable[column].map(lambda value: value.isoformat())
    serializable.to_csv(output, index=False)
    manifest = {
        "dataset_id": "win_2026_sequence_jan_may_v1",
        "status": "development_context_and_retrospective_targets",
        "snapshot": output.name,
        "snapshot_sha256": sha256(output),
        "source_manifest_sha256": sha256(base_manifest),
        "market_source_sha256": sha256(market_source),
        "label_source_sha256": sha256(label_source),
        "source_period": source_manifest["period_inclusive"],
        "period": {"start": snapshot.session_date.min(), "end": snapshot.session_date.max()},
        "rows": len(snapshot),
        "instruments": sorted(snapshot.instrument.unique().tolist()),
        "features": [column for column in snapshot if column.startswith("x_")],
        "targets": ["y_5m", "y_10m", "y_15m"],
        "label_non_null": {
            f"y_{h}m": int(snapshot[f"y_{h}m"].notna().sum()) for h in (5, 10, 15)
        },
        "context_policy": "All source candles in selected regular sessions are retained; prediction anchors must independently pass the causal 60-minute/48-candle gate.",
        "window_policy": "Exact 60-minute clock grid ending at decision time; no cross-contract/session history; absent candles represented by a mask.",
        "feature_policy": "OHLCV-derived features use current closed candle and prior same-session candles only; labels and label metadata are not features.",
        "label_availability": "Conservative simulated post-session release 18:02 local, not historically observed; whole-session label intervals are purged.",
        "test_policy": "The source database extraction and Parquet base are capped at May 31; June prices and B0 labels are not queried or read.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
