"""Calendar-oriented view of approved and provisional WIN candidates."""
from pathlib import Path
import calendar
import csv
import sqlite3
from collections import defaultdict
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT/'data/db/bovdb-v3.1-internal.2.sqlite'
OUT = ROOT/'figures'
MONTHS = ['JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO']
# February–June are candidate illustrations, not an approved roster.
ROSTERS = [('WING26','WINJ26','WINM26'), ('WING26','WINJ26','WINM26'),
           ('WINJ26','WINM26','WINQ26'), ('WINJ26','WINM26','WINQ26'),
           ('WINM26','WINQ26','WINV26'), ('WINM26','WINQ26','WINV26')]
EXPIRIES = {'WING26':'2026-02-18','WINJ26':'2026-04-15',
            'WINM26':'2026-06-17','WINQ26':'2026-08-12','WINV26':'2026-10-14'}
CLOSED = {'2026-01-01','2026-02-16','2026-02-17','2026-04-03',
          '2026-04-21','2026-05-01','2026-06-04'}
SPECIAL = {'2026-02-18'}
SYMBOL_COLORS = {'WING26':'#005AB5','WINJ26':'#B04A00','WINM26':'#007A52','WINQ26':'#762A9D','WINV26':'#A50050'}
COLORS = {'fim_semana':'#d5d9de','b3_fechada':'#f28e2b','pregao_especial_excluido':'#a67ad6',
          'sem_price1':'#df4a46','fora_vigencia':'#9cbad1','sem_candle_contrato':'#f4df4e',
          'com_candle':'#ffffff','sem_janela_80':'#f4df4e'}
HATCHES = {'fim_semana':'', 'b3_fechada':'//', 'pregao_especial_excluido':'xx',
           'sem_price1':'++', 'fora_vigencia':'..', 'sem_candle_contrato':'--',
           'com_candle':'', 'sem_janela_80':'--'}
with sqlite3.connect(DB.resolve().as_uri()+'?mode=ro',uri=True) as db:
    dates = {r[0] for r in db.execute("SELECT DISTINCT date FROM price1 WHERE date >= '2026-01-01' AND date < '2026-07-01'")}
    observed = {(r[0],r[1]) for r in db.execute("""SELECT DISTINCT t.ticker,p.date FROM price1 p
        JOIN ticker t USING(id_ticker) WHERE t.ticker LIKE 'WIN%' AND p.date >= '2026-01-01'
        AND p.date < '2026-07-01' AND p.time >= '09:00:00' AND p.time < '18:00:00'""")}
# Each decision minute is assessed using only the 60 preceding completed candles.
minutes = defaultdict(set)
with sqlite3.connect(DB.resolve().as_uri()+'?mode=ro',uri=True) as db:
 for sym,day,stamp in db.execute("""SELECT t.ticker,p.date,p.time FROM price1 p
      JOIN ticker t USING(id_ticker) WHERE t.ticker LIKE 'WIN%'
      AND p.date >= '2026-01-01' AND p.date < '2026-07-01'
      AND p.time >= '09:00:00' AND p.time < '18:00:00'"""):
  h,m,sec=map(int,stamp.split(':'))
  if sec: raise ValueError((sym,day,stamp))
  minutes[(sym,day)].add(h*60+m)
windows={}
for key,values in minutes.items():
 count=sum(m in values for m in range(540,600))
 qualified=[]
 for decision in range(600,1081):
  if decision>600:
   count += int(decision-1 in values)-int(decision-61 in values)
  if count>=48: qualified.append(decision)
 windows[key]=qualified
records=[]
for month, roster in enumerate(ROSTERS, 1):
 for symbol in roster:
  for day in range(1,calendar.monthrange(2026,month)[1]+1):
   dt=date(2026,month,day);key=dt.isoformat()
   if dt.weekday()>=5: status='fim_semana'
   elif key in CLOSED: status='b3_fechada'
   elif key in SPECIAL: status='pregao_especial_excluido'
   elif key>EXPIRIES[symbol]: status='fora_vigencia'
   elif key not in dates: status='sem_price1'
   elif (symbol,key) not in observed: status='sem_candle_contrato'
   elif not windows.get((symbol,key)): status='sem_janela_80'
   else: status='com_candle'
   records.append((key,symbol,status,'aprovado' if month==1 else 'candidato_preliminar',len(windows.get((symbol,key),[])) if status in ('sem_janela_80','com_candle') else ''))
OUT.mkdir(exist_ok=True)
with (OUT/'contratos-win-80porcento-dados.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.writer(f);writer.writerow(['data','contrato','situacao','selecao','janelas_aptas_80']);writer.writerows(records)
fig,ax=plt.subplots(figsize=(16,11.4));fig.patch.set_facecolor('white')
for idx,roster in enumerate(ROSTERS):
 for slot,symbol in enumerate(roster):
  y=17-(idx*3+slot)
  ax.text(-10.7,y,MONTHS[idx] if slot==0 else '',fontsize=10.2,fontweight='bold',va='center',color='#273545')
  ax.text(-6.3,y,symbol,fontsize=10,fontweight='bold',va='center',color=SYMBOL_COLORS[symbol])
  available=[]
  for key,sym,status,_,_ in records:
   dt=date.fromisoformat(key)
   if dt.month != idx+1 or sym!=symbol:continue
   x=dt.day
   ax.add_patch(Rectangle((x-.47,y-.37),.94,.74,facecolor=COLORS[status],edgecolor='#626a73' if HATCHES[status] else 'white', hatch=HATCHES[status],lw=.35,zorder=0))
   if status=='com_candle':
    available.append(x);ax.scatter(x,y,s=20,color=SYMBOL_COLORS[symbol],zorder=3)
  text=f'{min(available):02d}–{max(available):02d} | {len(available)} dias' if available else 'Sem janela'
  ax.text(32.0,y,text,fontsize=9,color='#465666',va='center')
 if idx<5:ax.axhline(14.5-idx*3,color='#dce3ea',lw=.7)
ax.set_xlim(-11,39);ax.set_ylim(-.6,17.7);ax.set_yticks([])
ax.set_xticks([1,5,10,15,20,25,30,31]);ax.tick_params(axis='x',length=0,pad=8)
ax.set_xlabel('Dia do mês',labelpad=8,color='#526170')
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.07,.965,'Mini-índice: janelas com 80% de cobertura',fontsize=19,fontweight='bold',color='#172636')
fig.text(.07,.939,'Três contratos por mês  |  Sessões regulares no BovDB  |  janeiro a junho de 2026',fontsize=11,color='#5b6876')
legend=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#526c87',markersize=7,label='Pelo menos uma janela ≥ 80%'),
        *[Patch(facecolor=COLORS[k],edgecolor='#626a73' if HATCHES[k] else 'white',hatch=HATCHES[k],label=l) for k,l in [('fim_semana','Fim de semana'),('b3_fechada','B3 fechada'),('pregao_especial_excluido','Pregão especial excluído'),('sem_price1','Sem dados price1'),('sem_janela_80','Sem janela ≥ 80%'),('fora_vigencia','Após vencimento')]]]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.12),ncol=4,frameon=False,fontsize=9)
fig.text(.07,.046,'Ponto = pelo menos uma janela de 60 min com 48 candles fechados; não significa que o dia inteiro passou.\nSem limite de lacuna definido. Janeiro: aprovados; fevereiro–junho: candidatos preliminares, sem substituições calculadas.',fontsize=9,color='#586a7b')
fig.subplots_adjust(left=.07,right=.97,top=.90,bottom=.21)
for ext in ['png','pdf']:
 fig.savefig(OUT/f'contratos-win-80porcento-2026.{ext}',dpi=200,facecolor='white')
print('Calendar rows',len(records))
