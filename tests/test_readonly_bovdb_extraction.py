import hashlib
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd


def _builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_win_parquet.py"
    spec = importlib.util.spec_from_file_location("readonly_win_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readonly_extraction_end_date_excludes_june_rows(tmp_path, monkeypatch):
    market_db = tmp_path / "market.sqlite"
    labels_db = tmp_path / "labels.sqlite"
    schedule = tmp_path / "schedule.csv"
    output = tmp_path / "jan_may"

    with sqlite3.connect(market_db) as connection:
        connection.executescript("""
            CREATE TABLE ticker (ticker TEXT, id_ticker INTEGER);
            CREATE TABLE price1 (
                id_ticker INTEGER, date TEXT, time TEXT, open REAL, high REAL, low REAL,
                close REAL, average REAL, volume REAL, business INTEGER, amount_stock REAL
            );
            INSERT INTO ticker VALUES ('WINM26', 7);
            INSERT INTO price1 VALUES (7,'2026-05-04','09:00:00',100,101,99,100,100,1,1,100);
            INSERT INTO price1 VALUES (7,'2026-06-01','09:00:00',200,201,199,200,200,1,1,200);
        """)
    with sqlite3.connect(labels_db) as connection:
        for horizon in (5, 10, 15):
            connection.execute(f"""CREATE TABLE trend_{horizon}m (
                id_ticker INTEGER, ticker TEXT, date TEXT, time TEXT, trend_label INTEGER
            )""")
            connection.executemany(
                f"INSERT INTO trend_{horizon}m VALUES (7,'WINM26',?,?,?)",
                [("2026-05-04", "09:00:00", 1), ("2026-06-01", "09:00:00", 0)],
            )
    pd.DataFrame([{
        "data": "2026-05-04", "slot": 1, "contrato": "WINM26",
        "tipo_decisao": "approved", "mes_evidencia": "2026-04",
    }, {
        "data": "2026-06-01", "slot": 1, "contrato": "WINM26",
        "tipo_decisao": "approved", "mes_evidencia": "2026-05",
    }]).to_csv(schedule, index=False)
    before = (_sha(market_db), _sha(labels_db))

    monkeypatch.setattr("sys.argv", [
        "build_win_parquet", "--market-db", str(market_db), "--labels-db", str(labels_db),
        "--selection", str(schedule), "--output", str(output), "--end-date", "2026-05-31",
    ])
    _builder().main()

    market = pd.read_parquet(output / "market_candles.parquet")
    targets = pd.read_parquet(output / "b0_targets_retrospective.parquet")
    assert set(market.session_date.astype(str)) == {"2026-05-04"}
    assert set(targets.session_date.astype(str)) == {"2026-05-04"}
    assert (_sha(market_db), _sha(labels_db)) == before
