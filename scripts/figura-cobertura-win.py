"""Plot observed WIN contract sessions; no inference of continuous coverage."""
from pathlib import Path
import csv
import sqlite3
from datetime import date, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
database = ROOT / 'data/db/bovdb-v3.1-internal.2.sqlite'
uri = database.resolve().as_uri() + '?mode=ro'
with sqlite3.connect(uri, uri=True) as db:
    rows = db.execute('''
        SELECT t.ticker, p.date, MIN(p.time), MAX(p.time), COUNT(*)
        FROM price1 p JOIN ticker t ON t.id_ticker = p.id_ticker
        WHERE t.ticker IN ('WING26', 'WINJ26')
          AND p.date >= '2026-01-01' AND p.date < '2026-07-01'
        GROUP BY t.ticker, p.date ORDER BY t.ticker, p.date
    ''').fetchall()
if not rows:
    raise RuntimeError('Nenhuma sessão encontrada; confira o banco de origem.')
output = ROOT / 'figures'
output.mkdir(exist_ok=True)
with (output / 'sessoes-wing26-winj26-2026.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['contrato', 'data', 'primeiro_candle', 'ultimo_candle', 'candles'])
    writer.writerows(rows)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, ax = plt.subplots(figsize=(14, 4.6), layout='constrained')
fig.patch.set_facecolor('white')
colors = {'WING26': '#27679f', 'WINJ26': '#df8a33'}
for sym, y in [('WING26', 1), ('WINJ26', 0)]:
    sessions = [datetime.strptime(row[1], '%Y-%m-%d') for row in rows if row[0] == sym]
    ax.scatter(sessions, [y] * len(sessions), s=34, color=colors[sym], zorder=3)
    for dt, align, offset in [(sessions[0], 'left', 5), (sessions[-1], 'right', -5)]:
        ax.annotate(dt.strftime('%d/%m'), (dt, y), xytext=(offset, 16),
                    textcoords='offset points', ha=align, color=colors[sym], fontweight='bold')
ax.set_xlim(date(2026, 1, 1), date(2026, 7, 1))
ax.set_ylim(-0.65, 1.65)
ax.set_yticks([1, 0], ['WING26', 'WINJ26'])
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.set_xticks([datetime(2026, m, 15) for m in range(1, 7)],
              ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN'])
for m in range(2, 7):
    ax.axvline(datetime(2026, m, 1), color='#d9dfe6', linewidth=1, zorder=0)
ax.tick_params(axis='both', length=0, pad=12)
ax.grid(axis='y', color='#edf0f4', linewidth=1)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title('Mini-índice: sessões observadas no banco', loc='left', fontsize=18, fontweight='bold', pad=26)
fig.text(0.075, 0.92, 'WING26 e WINJ26  |  janeiro a junho de 2026', color='#606c78', fontsize=11)
fig.text(0.075, 0.025, 'Cada ponto representa uma data com pelo menos um candle de 1 minuto.\nInício e fim indicam os dados observados, não garantem cobertura contínua nem representam, por si só, o vencimento.', color='#606c78', fontsize=9)
fig.savefig(output / 'presenca-wing26-winj26-jan-jun-2026.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(output / 'presenca-wing26-winj26-jan-jun-2026.pdf', bbox_inches='tight', facecolor='white')
print('Sessions:',len(rows),'| Figure:',output / 'presenca-wing26-winj26-jan-jun-2026.png')
