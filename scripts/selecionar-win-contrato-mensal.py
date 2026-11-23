"""Freeze one ex-ante WIN contract per month from prior-month evidence."""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSED = {
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18",
    "2026-04-03", "2026-04-21", "2026-05-01", "2026-06-04",
}
EXPIRY = {
    "WING26": date(2026, 2, 18), "WINJ26": date(2026, 4, 15),
    "WINM26": date(2026, 6, 17), "WINQ26": date(2026, 8, 12),
    "WINV26": date(2026, 10, 14), "WINZ26": date(2026, 12, 16),
}


def regular_sessions(year: int, month: int) -> list[date]:
    return [
        date(year, month, day)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if date(year, month, day).weekday() < 5
        and date(year, month, day).isoformat() not in CLOSED
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(source: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with source.open(encoding="utf-8", newline="") as stream:
        evidence = list(csv.DictReader(stream))
    # Explicit study-owner roster for January; December coverage is unavailable.
    selected: dict[int, str] = {1: "WING26"}
    ranking: list[dict[str, str]] = []
    for target_month in range(2, 7):
        prior_month = target_month - 1
        rows = [row for row in evidence if int(row["mes_selecao"][5:7]) == target_month]
        eligible = []
        target_sessions = regular_sessions(2026, target_month)
        prior_sessions = regular_sessions(2026, prior_month)
        for row in rows:
            symbol = row["contrato"]
            expiry = EXPIRY.get(symbol)
            coverage_sessions = int(row["sessoes_com_janela_80"])
            volume_missing = int(row["linhas_volume_nulo"])
            volume = int(row["volume_bovdb_bruto"])
            valid_all_month = expiry is not None and expiry >= target_sessions[-1]
            enough_history = len(prior_sessions) > 0 and 4 * coverage_sessions >= len(prior_sessions)
            is_eligible = valid_all_month and enough_history and volume > 0 and volume_missing == 0
            row.update({
                "pregoes_regulares_mes_anterior": str(len(prior_sessions)),
                "cobertura_80_pregoes_mes_anterior": str(coverage_sessions),
                "cobertura_25_aprovada": str(int(enough_history)),
                "vigente_mes_alvo_inteiro": str(int(valid_all_month)),
                "elegivel": str(int(is_eligible)),
                "rank_liquidez": "",
            })
            if is_eligible:
                eligible.append(row)
        eligible.sort(key=lambda row: (-int(row["volume_bovdb_bruto"]), -int(row["negocios"]), row["contrato"]))
        for rank, row in enumerate(eligible, 1):
            row["rank_liquidez"] = str(rank)
            ranking.append(row)
        if eligible:
            selected[target_month] = eligible[0]["contrato"]

    schedule: list[dict[str, str]] = []
    for month in range(1, 7):
        contract = selected.get(month, "")
        reason = (
            "fixado pelo responsavel; cobertura de dezembro indisponivel" if month == 1
            else "" if contract
            else "nenhum contrato atende a todos os criterios"
        )
        for session in regular_sessions(2026, month):
            schedule.append({
                "data": session.isoformat(), "mes": f"2026-{month:02d}", "slot": "1",
                "contrato": contract, "fonte_selecao": "mes_anterior" if month > 1 else "decisao_responsavel",
                "mes_evidencia": f"2026-{month - 1:02d}" if month > 1 else "",
                "tipo_decisao": "criterios_mes_anterior" if month > 1 else "fixado_pelo_responsavel",
                "observacao": reason,
            })
    return evidence, schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/b3_win/evidencia-selecao-mensal-2026.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/win_single_contract_v1")
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    evidence, schedule = select(source)
    output.mkdir(parents=True)
    ranking_path = output / "candidate_ranking.csv"
    fields = list(evidence[0]) + [
        "pregoes_regulares_mes_anterior", "cobertura_80_pregoes_mes_anterior",
        "cobertura_25_aprovada", "vigente_mes_alvo_inteiro", "elegivel", "rank_liquidez",
    ]
    with ranking_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evidence)
    schedule_path = output / "monthly_schedule.csv"
    with schedule_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(schedule[0]))
        writer.writeheader()
        writer.writerows(schedule)
    manifest = {
        "protocol_id": "win_single_contract_monthly_v1",
        "rule": "one contract for the full target month; expiry on/after last regular session; at least 25% of previous month's regular sessions have a causal 60-minute window with >=48 closed candles; rank eligible contracts by previous-month BovDB volume, then business count, then ticker",
        "january": "WING26 fixed by explicit study-owner decision; December historical coverage evidence unavailable, so the 25% gate cannot be independently verified for January",
        "volume_metric": "Previous-month price1 volume, ranked within BovDB; equivalence of units to B3 BDI volume is unresolved",
        "availability_warning": "Historical source aggregation is retrospective and does not prove that the source vintage was available before each target month. B0 confirmation/processing timestamps are unobserved.",
        "selection_uses_mcc": False,
        "intramonth_replacement": False,
        "month_schedule_includes_june_dates_only": "calendar schedule derived from May evidence; no June source data queried",
        "source": {"path": str(source.relative_to(ROOT)), "sha256": sha256(source)},
        "outputs": {ranking_path.name: sha256(ranking_path), schedule_path.name: sha256(schedule_path)},
        "monthly_selection": {f"2026-{m:02d}": next((row["contrato"] for row in schedule if row["mes"] == f"2026-{m:02d}" and row["contrato"]), None) for m in range(1, 7)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["monthly_selection"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
