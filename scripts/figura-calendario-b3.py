"""Calendar of B3 market days and WIN observations, January–June 2026."""
from pathlib import Path
import calendar
import csv
import sqlite3
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

ROOT = Path(__file__).resolve().parents[1]
closed = {'2026-01-01':'Ano Novo','2026-02-16':'Carnaval','2026-02-17':'Carnaval',
          '2026-04-03':'Sexta-feira Santa','2026-04-21':'Tiradentes',
          '2026-05-01':'Dia do Trabalho','2026-06-04':'Corpus Christi'}
special = {'2026-02-18':'Quarta-feira de Cinzas: horário especial'}
with sqlite3.connect((ROOT/'data/db/bovdb-v3.1-internal.2.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
    win_dates={r[0] for r in db.execute("SELECT DISTINCT p.date FROM price1 p JOIN ticker t USING(id_ticker) WHERE t.ticker LIKE 'WIN%' AND p.date >= '2026-01-01' AND p.date < '2026-07-01'")}
    any_dates={r[0] for r in db.execute("SELECT DISTINCT date FROM price1 WHERE date >= '2026-01-01' AND date < '2026-07-01'")}
records=[]
for m in range(1,7):
 for day in range(1,calendar.monthrange(2026,m)[1]+1):
    dt=date(2026,m,day); key=dt.isoformat()
    if dt.weekday()>=5: status='fim_de_semana'
    elif key in closed: status='fechado_b3'
    elif key not in any_dates: status='sem_dados_price1'
    elif key not in win_dates: status='sem_win_no_banco'
    elif key in special: status='pregao_especial_com_win'
    else: status='pregao_com_win'
    records.append((key,status,closed.get(key,special.get(key,''))))

out=ROOT/'figures';out.mkdir(exist_ok=True)
with (out/'calendario-b3-bovdb-jan-jun-2026.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['data','classificacao','observacao']);w.writerows(records)
labels=['JANEIRO','FEVEREIRO','MARÇO','ABRIL','MAIO','JUNHO']
colors={'pregao_com_win':'#2b8c79','pregao_especial_com_win':'#a8d8c9','fim_de_semana':'#e5eaf0','fechado_b3':'#e4ae58','sem_dados_price1':'#d35d5d','sem_win_no_banco':'#8b6daf'}
fig,ax=plt.subplots(figsize=(16,6.3));fig.patch.set_facecolor('white')
for key,status,note in records:
 dt=date.fromisoformat(key);x=dt.day-1;y=6-dt.month
 ax.add_patch(Rectangle((x+.04,y+.07),.90,.82,facecolor=colors[status],edgecolor='white',linewidth=1))
 ax.text(x+.49,y+.48,str(dt.day),ha='center',va='center',fontsize=9,color='white' if status in ('pregao_com_win','sem_dados_price1','sem_win_no_banco') else '#334354',fontweight='bold' if status in ('sem_dados_price1','pregao_especial_com_win') else 'normal')
ax.set_xlim(0,31);ax.set_ylim(0,6);ax.set_yticks([5.5,4.5,3.5,2.5,1.5,.5],labels);ax.set_xticks([]);ax.tick_params(length=0,pad=10)
for s in ax.spines.values():s.set_visible(False)
fig.text(.075,.94,'Calendário de pregões e cobertura do BovDB',fontsize=20,fontweight='bold',color='#172636')
fig.text(.075,.89,'Mini-índice WIN  |  janeiro a junho de 2026',fontsize=11,color='#5b6876')
legend=[Patch(facecolor=colors[k],label=l) for k,l in [('pregao_com_win','Pregão com WIN no banco'),('pregao_especial_com_win','Pregão especial com WIN'),('fim_de_semana','Fim de semana'),('fechado_b3','B3 fechada'),('sem_dados_price1','Sem dados price1'),('sem_win_no_banco','Sem WIN no banco')]]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.145),ncol=3,frameon=False,fontsize=10)
fig.text(.075,.055,'Datas sem dados price1: 13, 21 e 25 de maio.  |  18 de fevereiro: horário especial.\nPresença de WIN indica ao menos um candle em qualquer contrato WIN; não comprova sessão completa.',fontsize=9,color='#5b6876')
fig.subplots_adjust(left=.12,right=.98,top=.83,bottom=.25)
for ext in ('png','pdf'):fig.savefig(out/f'calendario-b3-bovdb-jan-jun-2026.{ext}',dpi=200,facecolor='white')
print('Sem price1:',[r[0] for r in records if r[1]=='sem_dados_price1'])
print('Sem WIN:',[r[0] for r in records if r[1]=='sem_win_no_banco'])
