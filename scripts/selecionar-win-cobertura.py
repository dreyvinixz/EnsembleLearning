"""Build provisional causal monthly rosters from previous-month evidence only."""
from __future__ import annotations
import calendar
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/b3_win/evidencia-selecao-mensal-2026.csv'
RANKS = ROOT/'data/b3_win/selecao-win-50-criterios.csv'
SCHEDULE = ROOT/'data/b3_win/selecao-win-50-por-pregao.csv'
MANIFEST = ROOT/'data/b3_win/selecao-win-50-manifest.json'
CLOSED = {'2026-01-01','2026-02-16','2026-02-17','2026-02-18','2026-04-03',
          '2026-04-21','2026-05-01','2026-06-04'}
JANUARY = [('WING26','2026-02-18'),('WINJ26','2026-04-15'),('WINM26','2026-06-17')]


def sessions(month: int) -> list[date]:
 return [date(2026,month,d) for d in range(1,calendar.monthrange(2026,month)[1]+1)
         if date(2026,month,d).weekday()<5 and date(2026,month,d).isoformat() not in CLOSED]

if any(p.exists() for p in (RANKS,SCHEDULE,MANIFEST)):
 raise FileExistsError('Output already exists; use a new version instead of overwriting')
with SOURCE.open(encoding='utf-8',newline='') as f: source=list(csv.DictReader(f))
candidates=defaultdict(list)
for r in source:
 month=int(r['mes_selecao'][5:7]);prior=month-1;expiry=date.fromisoformat(r['vencimento'])
 # Conservative denominator: all prior-month sessions until expiration.
 expected=sum(day<=expiry for day in sessions(prior))
 qualified=int(r['sessoes_com_janela_80']); observed=int(r['sessoes_com_candle'])
 if qualified>observed or observed>expected: raise ValueError(('invalid counts',r,expected))
 valid_date=expiry>=date(2026,month,1)
 volume=float(r['volume_bovdb_bruto']); missing=int(r['linhas_volume_nulo'])
 coverage_pass=expected>0 and 2*qualified>=expected
 eligible=valid_date and coverage_pass and volume>0 and missing==0
 r.update(pregoes_anteriores_elegiveis=expected,cobertura_50_aprovada=int(coverage_pass),
          elegivel_criterios=int(eligible),rank_entre_elegiveis='')
 candidates[month].append(r)
for month,rows in candidates.items():
 eligible=sorted([r for r in rows if r['elegivel_criterios']],
                 key=lambda r:(-float(r['volume_bovdb_bruto']),-int(r['negocios']),r['contrato']))
 for i,r in enumerate(eligible,1):r['rank_entre_elegiveis']=i

with RANKS.open('x',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(source[0]));writer.writeheader();writer.writerows(source)
with SCHEDULE.open('x',encoding='utf-8',newline='') as f:
 writer=csv.writer(f);writer.writerow(['data','slot','contrato','tipo_decisao','mes_evidencia','observacao'])
 for month in range(1,7):
  if month==1:
   active=[(sym,date.fromisoformat(exp)) for sym,exp in JANUARY];reserve=[];basis='manual_usuario'
  else:
   ranked=sorted((r for r in candidates[month] if r['elegivel_criterios']),
                 key=lambda r:int(r['rank_entre_elegiveis']))
   active=[(r['contrato'],date.fromisoformat(r['vencimento'])) for r in ranked[:3]]
   reserve=[(r['contrato'],date.fromisoformat(r['vencimento'])) for r in ranked[3:]]
   basis=f'2026-{month-1:02d}'
  for day in sessions(month):
   for i,(sym,expiry) in enumerate(active):
    if expiry<day:
     available=next((x for x in reserve if x[1]>=day and x[0] not in {a[0] for a in active}),None)
     active[i]=available if available else ('',expiry)
     if available: reserve.remove(available)
   for slot in range(3):
    symbol=active[slot][0] if slot<len(active) else ''
    writer.writerow([day.isoformat(),slot+1,symbol,'manual' if month==1 else 'criterios_mes_anterior',basis,
      '' if symbol else 'sem_candidato_elegivel'])
manifest={'status':'diagnostico_nao_treino','source':str(SOURCE.relative_to(ROOT)),
 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'rule':'previous calendar month regular eligible sessions, >= 50% with at least one 80% window; volume > 0 and no null volume; no forward substitution evidence',
 'assumptions':['Denominator counts all prior-month regular sessions until expiry; historical listing start not verified.',
 'Availability and data vintage assumed as documented; current DB generated later.',
 'At least one 80% window per session does not guarantee sufficient coverage at prediction time.',
 'Jan manually approved and does not satisfy proven December historical coverage.',
 'No maximum intraday gap configured; volumes rank only within BovDB data.'],
 'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (RANKS,SCHEDULE)}}
with MANIFEST.open('x',encoding='utf-8') as f: json.dump(manifest,f,indent=2,ensure_ascii=False)
print('saved',RANKS,SCHEDULE,MANIFEST,sep='\n')
