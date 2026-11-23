import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _selector():
    path = ROOT / "scripts/selecionar-win-contrato-mensal.py"
    spec = importlib.util.spec_from_file_location("monthly_win_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_evidence(tmp_path):
    import csv

    rows = [
        # target month, symbol, eligible prior-month session coverage, volume
        (2, "WING26", 21, 900), (2, "WINJ26", 21, 100),
        (3, "WING26", 10, 900), (3, "WINJ26", 17, 800), (3, "WINM26", 5, 100),
        (4, "WINJ26", 22, 900), (4, "WINM26", 22, 800), (4, "WINQ26", 5, 100),
        (5, "WINJ26", 10, 900), (5, "WINM26", 20, 800), (5, "WINQ26", 5, 100),
        (6, "WINM26", 20, 900), (6, "WINQ26", 5, 800), (6, "WINV26", 5, 100),
    ]
    path = tmp_path / "prior_month_evidence.csv"
    fields = ["mes_selecao", "contrato", "sessoes_com_janela_80",
              "linhas_volume_nulo", "volume_bovdb_bruto", "negocios"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for month, symbol, coverage, volume in rows:
            writer.writerow({"mes_selecao": f"2026-{month:02d}", "contrato": symbol,
                             "sessoes_com_janela_80": coverage, "linhas_volume_nulo": 0,
                             "volume_bovdb_bruto": volume, "negocios": volume})
    return path


def test_prior_month_selector_uses_coverage_expiry_and_liquidity_without_mcc(tmp_path):
    evidence_path = _synthetic_evidence(tmp_path)
    _, schedule = _selector().select(evidence_path)
    chosen = {}
    for row in schedule:
        chosen.setdefault(row["mes"], row["contrato"])
    assert chosen == {
        "2026-01": "WING26", "2026-02": "WINJ26", "2026-03": "WINJ26",
        "2026-04": "WINM26", "2026-05": "WINM26", "2026-06": "WINQ26",
    }
    assert "mcc" not in " ".join(schedule[0]).lower()
    for month in range(1, 7):
        contracts = {row["contrato"] for row in schedule if row["mes"] == f"2026-{month:02d}"}
        assert len(contracts) == 1


def test_schedule_has_no_intramonth_replacement_and_uses_prior_month_evidence(tmp_path):
    _, schedule = _selector().select(_synthetic_evidence(tmp_path))
    for month in range(2, 7):
        rows = [row for row in schedule if row["mes"] == f"2026-{month:02d}"]
        assert len({row["contrato"] for row in rows}) == 1
        assert {row["mes_evidencia"] for row in rows} == {f"2026-{month - 1:02d}"}


def test_causal_windows_and_features_do_not_cross_monthly_contract_rollover():
    import pandas as pd
    from prepare_b0_baseline_snapshot import make_snapshot

    from ensemble_learning.sequences import build_causal_windows

    chunks = []
    for instrument, day, offset in [("WINJ26", "2026-03-31", 0), ("WINM26", "2026-04-01", 1000)]:
        stamps = pd.date_range(f"{day} 09:00", periods=90, freq="min")
        for i, stamp in enumerate(stamps):
            price = offset + 100 + i
            chunks.append({
                "instrument": instrument, "asset_id": 1 if instrument == "WINJ26" else 2,
                "session_date": day, "candle_start_local": stamp,
                "open": price, "high": price + 1, "low": price - 1,
                "close": price, "average": price, "volume": i + 1,
                "business": i + 1, "amount_stock": i + 1,
                "y_b0_5m": i % 2, "y_b0_10m": i % 2, "y_b0_15m": i % 2,
            })
    source = pd.DataFrame(chunks)
    market = source.drop(columns=[f"y_b0_{h}m" for h in (5, 10, 15)])
    labels = source[["instrument", "asset_id", "session_date", "candle_start_local",
                     *[f"y_b0_{h}m" for h in (5, 10, 15)]]]
    snapshot = make_snapshot(market, labels, include_context_bars=True)
    assert len([c for c in snapshot if c.startswith("x_")]) == 21
    march = snapshot.loc[snapshot.instrument == "WINJ26"]
    april = snapshot.loc[snapshot.instrument == "WINM26"]
    assert march.x_return_1bar.isna().iloc[0]
    assert april.x_return_1bar.isna().iloc[0]

    windows = build_causal_windows(
        snapshot, feature_columns=["x_return_1bar"], target_columns=["y_5m", "y_10m", "y_15m"],
    )
    for row_indices, anchor in zip(windows.row_indices, windows.anchors.itertuples(), strict=True):
        valid = row_indices[row_indices >= 0]
        assert set(snapshot.iloc[valid].instrument) == {anchor.instrument}
        assert set(snapshot.iloc[valid].session_date) == {anchor.session_date}
