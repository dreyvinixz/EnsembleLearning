"""Prior-month WIN selection evidence, read-only; not final training roster."""
from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/db/bovdb-v3.1-internal.2.sqlite"
DEFAULT_OUT = ROOT / "data/b3_win/evidencia-selecao-mensal-2026.csv"
CLOSED = {
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-04-03",
    "2026-04-21", "2026-05-01", "2026-06-04",
}
EXPIRY = {
    "WING26": "2026-02-18", "WINJ26": "2026-04-15", "WINM26": "2026-06-17",
    "WINQ26": "2026-08-12", "WINV26": "2026-10-14", "WINZ26": "2026-12-16",
}


def count_windows(minutes: set[int]) -> int:
    count = sum(minute in minutes for minute in range(540, 600))
    good = 0
    for decision_minute in range(600, 1081):
        if decision_minute > 600:
            count += int(decision_minute - 1 in minutes) - int(decision_minute - 61 in minutes)
        good += count >= 48
    return good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Read-only BovDB market SQLite")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    database = args.db if args.db.is_absolute() else ROOT / args.db
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not database.is_file():
        raise FileNotFoundError(database)
    if output.exists():
        raise FileExistsError(output)

    groups = defaultdict(lambda: {
        "minutes": set(), "volume": Decimal(0), "missing_volume": 0, "trades": 0,
    })
    uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute("""SELECT t.ticker,p.date,p.time,p.volume,p.business FROM price1 p
            JOIN ticker t ON t.id_ticker=p.id_ticker WHERE t.ticker LIKE 'WIN%26'
            AND p.date >= '2026-01-01' AND p.date < '2026-06-01'
            AND p.time >= '09:00:00' AND p.time < '18:00:00' ORDER BY p.date,p.time""")
        for symbol, day, timestamp, volume, trades in rows:
            if date.fromisoformat(day).weekday() >= 5 or day in CLOSED:
                continue
            if symbol not in EXPIRY:
                raise ValueError(f"Unknown expiration: {symbol}")
            hour, minute, second = map(int, timestamp.split(":"))
            if second:
                raise ValueError((symbol, day, timestamp))
            entry = groups[(symbol, day)]
            entry["minutes"].add(hour * 60 + minute)
            if volume is None:
                entry["missing_volume"] += 1
            else:
                entry["volume"] += Decimal(str(volume))
            entry["trades"] += int(trades or 0)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "mes_selecao", "contrato", "mes_historico", "vencimento", "elegivel_inicio_mes",
            "sessoes_com_candle", "sessoes_com_janela_80", "janelas_80", "janelas_possiveis",
            "minutos_com_candle", "volume_bovdb_bruto", "linhas_volume_nulo", "negocios",
        ])
        for month in range(2, 7):
            target, prior = f"2026-{month:02d}", f"2026-{month - 1:02d}"
            for symbol in sorted(EXPIRY):
                days = [value for (ticker, day), value in groups.items()
                        if ticker == symbol and day.startswith(prior)]
                good = [count_windows(value["minutes"]) for value in days]
                writer.writerow([
                    target, symbol, prior, EXPIRY[symbol], int(EXPIRY[symbol] >= target + "-01"),
                    len(days), sum(value > 0 for value in good), sum(good), 481 * len(days),
                    sum(len(value["minutes"]) for value in days),
                    str(sum((value["volume"] for value in days), Decimal(0))),
                    sum(value["missing_volume"] for value in days),
                    sum(value["trades"] for value in days),
                ])
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
