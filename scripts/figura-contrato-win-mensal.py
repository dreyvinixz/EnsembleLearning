"""Refresh the supplied monthly WIN calendar figure from the frozen roster."""
import calendar
import csv
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "reports/win_single_contract_v1/monthly_schedule.csv"
COVERAGE = ROOT / "figures/contratos-win-80porcento-dados.csv"
OUT = ROOT / "figures/contratos-win-contrato-mensal-v1.png"
OUT_PDF = OUT.with_suffix(".pdf")
FIGURE_DATA = ROOT / "figures/contratos-win-contrato-mensal-v1.csv"
MONTHS = ["JANEIRO", "FEVEREIRO", "MARÇO", "ABRIL", "MAIO", "JUNHO"]
COLORS = {
    "fim_semana": "#d5d9de", "b3_fechada": "#f28e2b", "pregao_especial_excluido": "#a67ad6",
    "sem_price1": "#df4a46", "sem_janela_80": "#f4df4e", "com_candle": "#ffffff",
    "sem_selecao": "#ffffff", "nao_consultado": "#ffffff",
}
HATCH = {"fim_semana": "", "b3_fechada": "//", "pregao_especial_excluido": "xx",
         "sem_price1": "++", "sem_janela_80": "--", "com_candle": "",
         "sem_selecao": "\\\\", "nao_consultado": ".."}
HOLIDAYS = {"2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-04-03",
            "2026-04-21", "2026-05-01", "2026-06-04"}
SPECIAL = {"2026-02-18"}

with SCHEDULE.open(encoding="utf-8", newline="") as stream:
    schedule = list(csv.DictReader(stream))
roster = {int(month[-2:]): next((row["contrato"] for row in schedule
                                if row["mes"] == month), "")
          for month in {row["mes"] for row in schedule}}
observed = {}
with COVERAGE.open(encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream):
        observed[(row["data"], row["contrato"])] = row["situacao"]

figure_rows = []
for month in range(1, 7):
    symbol = roster.get(month, "")
    for day in range(1, calendar.monthrange(2026, month)[1] + 1):
        key = date(2026, month, day).isoformat()
        if date(2026, month, day).weekday() >= 5:
            status = "fim_semana"
        elif key in SPECIAL:
            status = "pregao_especial_excluido"
        elif key in HOLIDAYS:
            status = "b3_fechada"
        elif not symbol:
            status = "sem_selecao"
        else:
            status = observed.get((key, symbol), "sem_price1")
            if status == "sem_candle_contrato":
                status = "sem_price1"
        figure_rows.append({"data": key, "mes": f"2026-{month:02d}", "contrato": symbol,
                            "status": status, "programado": int(bool(symbol)),
                            "dados_consultados": 1})
with FIGURE_DATA.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(figure_rows[0]))
    writer.writeheader()
    writer.writerows(figure_rows)

fig, ax = plt.subplots(figsize=(16, 8.8))
fig.patch.set_facecolor("white")
for month in range(1, 7):
    symbol = roster.get(month, "")
    y = 5 - month + 1
    ax.text(-10.7, y, MONTHS[month - 1], fontsize=10.2, fontweight="bold", va="center", color="#273545")
    ax.text(-6.3, y, symbol or ("SEM SELEÇÃO" if month == 1 else ""), fontsize=10,
            fontweight="bold", va="center", color="#005AB5" if month == 1 else "#007A52")
    available = []
    for day in range(1, calendar.monthrange(2026, month)[1] + 1):
        key = date(2026, month, day).isoformat()
        if date(2026, month, day).weekday() >= 5:
            status = "fim_semana"
        elif key in SPECIAL:
            status = "pregao_especial_excluido"
        elif key in HOLIDAYS:
            status = "b3_fechada"
        elif not symbol:
            status = "sem_selecao"
        else:
            status = observed.get((key, symbol), "sem_price1")
            if status == "sem_candle_contrato":
                status = "sem_price1"
            if status == "com_candle":
                available.append(day)
        rect = Rectangle((day - .47, y - .37), .94, .74, facecolor=COLORS[status],
                         edgecolor="#626a73" if HATCH[status] else "white", hatch=HATCH[status], lw=.35)
        ax.add_patch(rect)
        if status == "com_candle":
            ax.scatter(day, y, s=20, color="#005AB5" if month == 1 else "#007A52", zorder=3)
    if month == 1:
        label = "WING26 | cobertura de dezembro não verificada"
    else:
        label = f"{min(available):02d}–{max(available):02d} | {len(available)} dias" if available else "Sem candle"
    ax.text(32.0, y, label, fontsize=9, color="#465666", va="center")
    if month < 6:
        ax.axhline(y - .5, color="#dce3ea", lw=.7)

ax.set_xlim(-11, 43)
ax.set_ylim(-.6, 5.7)
ax.set_yticks([])
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 31])
ax.tick_params(axis="x", length=0, pad=8)
ax.set_xlabel("Dia do mês", labelpad=8, color="#526170")
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(.07, .965, "Mini-índice: um contrato WIN por mês", fontsize=19, fontweight="bold", color="#172636")
fig.text(.07, .93, "Elegibilidade e ranking definidos com dados do mês anterior | janeiro a junho de 2026",
         fontsize=10.5, color="#5b6876")
legend = [Patch(facecolor=COLORS[k], edgecolor="#626a73" if HATCH[k] else "white", hatch=HATCH[k], label=label)
          for k, label in [("fim_semana", "Fim de semana"), ("b3_fechada", "B3 fechada"),
                           ("pregao_especial_excluido", "Pregão especial excluído"),
                           ("sem_price1", "Sem dados price1"),
                           ("sem_janela_80", "Sem janela ≥ 80%")]]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .13), ncol=3, frameon=False, fontsize=9)
fig.text(.07, .04, "Ponto: ao menos uma janela causal de 60 minutos com 48 candles fechados no contrato escolhido.\n"
         "Janeiro WING26 foi fixado pelo responsável; cobertura de dezembro não verificável. Junho mostra cobertura observada de WINQ26.",
         fontsize=9, color="#586a7b")
fig.subplots_adjust(left=.07, right=.97, top=.89, bottom=.24)
fig.savefig(OUT, dpi=200, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
plt.close(fig)
print(OUT)
