"""Read-only WIN session coverage audit; no retrospective eligibility decisions."""
from __future__ import annotations
import csv
import hashlib
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/db/bovdb-v3.1-internal.2.sqlite'
OUT = ROOT / 'data/b3_win/auditoria-cobertura-win-2026.csv'
CLOSED = {'2026-01-01','2026-02-16','2026-02-17','2026-04-03','2026-04-21','2026-05-01','2026-06-04','2026-02-18'}

def audit(minutes):
    observed = {m for m in minutes if 540 <= m < 1080}
    if not observed:
        return 0, 540, 540, '', '', 0
    longest = run = 0
    for minute in range(540,1080):
        run = 0 if minute in observed else run + 1
        longest = max(longest,run)
    blocks = sum(1 for start in range(540,1080,60) if any(start <= m < start+60 for m in observed))
    return len(observed),540-len(observed),longest,min(observed),max(observed),blocks

def clock(m):
    return '' if m == '' else f'{m//60:02d}:{m%60:02d}'

if OUT.exists(): raise FileExistsError(OUT)
groups = defaultdict(list)
with sqlite3.connect(DB.resolve().as_uri()+'?mode=ro',uri=True) as db:
    for ticker,day,time in db.execute("""SELECT t.ticker,p.date,p.time FROM price1 p JOIN ticker t USING(id_ticker)
       WHERE t.ticker LIKE 'WIN%' AND p.date >= '2026-01-01' AND p.date < '2026-07-01'
       ORDER BY t.ticker,p.date,p.time"""):
        if date.fromisoformat(day).weekday()>=5 or day in CLOSED: continue
        h,m,s=map(int,time.split(':'))
        if s != 0: raise ValueError((ticker,day,time))
        groups[(ticker,day)].append(h*60+m)
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('x',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['contrato','data','minutos_com_candle_09_18','minutos_sem_candle_09_18','maior_lacuna_min','primeiro_minuto','ultimo_minuto','horas_com_candle','candle_1800_excluido'])
    for (ticker,day),minutes in sorted(groups.items()):
        n,missing,gap,first,last,blocks=audit(minutes)
        w.writerow([ticker,day,n,missing,gap,clock(first),clock(last),blocks,int(1080 in minutes)])
print('sessoes',len(groups),'sha256',hashlib.sha256(OUT.read_bytes()).hexdigest())
