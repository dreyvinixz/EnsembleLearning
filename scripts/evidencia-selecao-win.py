"""Prior-month WIN selection evidence, read-only; not final training roster."""
from __future__ import annotations
import csv
import sqlite3
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/db/bovdb-v3.1-internal.2.sqlite'
OUT = ROOT / 'data/b3_win/evidencia-selecao-mensal-2026.csv'
CLOSED = {'2026-01-01','2026-02-16','2026-02-17','2026-02-18','2026-04-03',
          '2026-04-21','2026-05-01','2026-06-04'}
EXPIRY = {'WING26':'2026-02-18','WINJ26':'2026-04-15','WINM26':'2026-06-17',
          'WINQ26':'2026-08-12','WINV26':'2026-10-14','WINZ26':'2026-12-16'}
if OUT.exists(): raise FileExistsError(OUT)
groups = defaultdict(lambda: {'minutes':set(),'volume':Decimal(0),'missing_volume':0,'trades':0})
with sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True) as db:
 rows = db.execute("""SELECT t.ticker,p.date,p.time,p.volume,p.business FROM price1 p
    JOIN ticker t ON t.id_ticker=p.id_ticker WHERE t.ticker LIKE 'WIN%26'
    AND p.date >= '2026-01-01' AND p.date < '2026-06-01'
    AND p.time >= '09:00:00' AND p.time < '18:00:00' ORDER BY p.date,p.time""")
 for sym,day,tm,vol,trades in rows:
  if date.fromisoformat(day).weekday()>=5 or day in CLOSED: continue
  if sym not in EXPIRY: raise ValueError(f'Unknown expiration: {sym}')
  h,m,s = map(int,tm.split(':'))
  if s: raise ValueError((sym,day,tm))
  entry = groups[(sym,day)]; entry['minutes'].add(h*60+m)
  if vol is None: entry['missing_volume'] += 1
  else: entry['volume'] += Decimal(str(vol))
  entry['trades'] += int(trades or 0)

def count_windows(minutes):
 count=sum(i in minutes for i in range(540,600));good=0
 for t in range(600,1081):
  if t>600: count += int(t-1 in minutes)-int(t-61 in minutes)
  good += count>=48
 return good

with OUT.open('x',encoding='utf-8',newline='') as f:
 writer=csv.writer(f);writer.writerow(['mes_selecao','contrato','mes_historico','vencimento',
  'elegivel_inicio_mes','sessoes_com_candle','sessoes_com_janela_80','janelas_80','janelas_possiveis',
  'minutos_com_candle','volume_bovdb_bruto','linhas_volume_nulo','negocios'])
 for month in range(2,7):
  target=f'2026-{month:02d}'; prior=f'2026-{month-1:02d}'
  for sym in sorted(EXPIRY):
   days=[v for (s,d),v in groups.items() if s==sym and d.startswith(prior)]
   good=[count_windows(v['minutes']) for v in days]
   writer.writerow([target,sym,prior,EXPIRY[sym],int(EXPIRY[sym]>=target+'-01'),len(days),
    sum(x>0 for x in good),sum(good),481*len(days),sum(len(v['minutes']) for v in days),
    str(sum((v['volume'] for v in days),Decimal(0))),sum(v['missing_volume'] for v in days),
    sum(v['trades'] for v in days)])
print('Wrote',OUT)
