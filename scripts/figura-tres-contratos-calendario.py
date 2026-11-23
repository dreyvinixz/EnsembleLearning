"""Calendar-oriented view of approved and provisional WIN candidates."""
from pathlib import Path
import calendar
import csv
import sqlite3
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
SYMBOL_COLORS = {'WING26':'#24699f','WINJ26':'#d48637','WINM26':'#409175',
                 'WINQ26':'#805dab','WINV26':'#ba5d76'}
COLORS = {'fim_semana':'#e7ebf0','b3_fechada':'#f2cf8f','pregao_especial_excluido':'#c9bce2',
          'sem_price1':'#f4b4b4','fora_vigencia':'#d4dbe3','sem_candle_contrato':'#f6e9df',
          'com_candle':'#ffffff'}
with sqlite3.connect(DB.resolve().as_uri()+'?mode=ro',uri=True) as db:
    dates = {r[0] for r in db.execute("SELECT DISTINCT date FROM price1 WHERE date >= '2026-01-01' AND date < '2026-07-01'")}
    observed = {(r[0],r[1]) for r in db.execute("""SELECT DISTINCT t.ticker,p.date FROM price1 p
        JOIN ticker t USING(id_ticker) WHERE t.ticker LIKE 'WIN%' AND p.date >= '2026-01-01'
        AND p.date < '2026-07-01' AND p.time >= '09:00:00' AND p.time < '18:00:00'""")}
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
   else: status='com_candle'
   records.append((key,symbol,status,'aprovado' if month==1 else 'candidato_preliminar'))
OUT.mkdir(exist_ok=True)
with (OUT/'contratos-win-tres-calendario-dados.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.writer(f);writer.writerow(['data','contrato','situacao','selecao']);writer.writerows(records)
fig,ax=plt.subplots(figsize=(16,11.4));fig.patch.set_facecolor('white')
for idx,roster in enumerate(ROSTERS):
 for slot,symbol in enumerate(roster):
  y=17-(idx*3+slot)
  ax.text(-10.7,y,MONTHS[idx] if slot==0 else '',fontsize=10.2,fontweight='bold',va='center',color='#273545')
  ax.text(-6.3,y,symbol,fontsize=10,fontweight='bold',va='center',color=SYMBOL_COLORS[symbol])
  available=[]
  for key,sym,status,_ in records:
   dt=date.fromisoformat(key)
   if dt.month != idx+1 or sym!=symbol:continue
   x=dt.day
   ax.add_patch(Rectangle((x-.47,y-.37),.94,.74,facecolor=COLORS[status],edgecolor='white',lw=.4,zorder=0))
   if status=='com_candle':
    available.append(x);ax.scatter(x,y,s=20,color=SYMBOL_COLORS[symbol],zorder=3)
  text=f'{min(available):02d}–{max(available):02d} | {len(available)} dias' if available else 'Sem candle'
  ax.text(32.0,y,text,fontsize=9,color='#465666',va='center')
 if idx<5:ax.axhline(14.5-idx*3,color='#dce3ea',lw=.7)
ax.set_xlim(-11,39);ax.set_ylim(-.6,17.7);ax.set_yticks([])
ax.set_xticks([1,5,10,15,20,25,30,31]);ax.tick_params(axis='x',length=0,pad=8)
ax.set_xlabel('Dia do mês',labelpad=8,color='#526170')
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.07,.965,'Mini-índice: três contratos e calendário',fontsize=19,fontweight='bold',color='#172636')
fig.text(.07,.939,'Sessões regulares observadas no BovDB  |  janeiro a junho de 2026',fontsize=11,color='#5b6876')
legend=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#526c87',markersize=7,label='Pelo menos um candle'),
        *[Patch(facecolor=COLORS[k],label=l) for k,l in [('fim_semana','Fim de semana'),('b3_fechada','B3 fechada'),('pregao_especial_excluido','Pregão especial excluído'),('sem_price1','Sem dados price1'),('sem_candle_contrato','Sem candle do contrato'),('fora_vigencia','Após vencimento')]]]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.12),ncol=4,frameon=False,fontsize=9)
fig.text(.07,.046,'Cada ponto indica ao menos um candle 09h–18h; não indica janela aprovada pelo critério de 80%.\nJaneiro: contratos aprovados. Fevereiro–junho: candidatos preliminares; substituições após vencimento ainda não calculadas.',fontsize=9,color='#586a7b')
fig.subplots_adjust(left=.07,right=.97,top=.90,bottom=.21)
for ext in ['png','pdf']:
 fig.savefig(OUT/f'contratos-win-tres-calendario-2026.{ext}',dpi=200,facecolor='white')
print('Calendar rows',len(records))
