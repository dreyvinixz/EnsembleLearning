"""Monthly WIN coverage using the proposed previous-month selection."""
from pathlib import Path
import csv
import sqlite3
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
selections = {'2026-01': ('WING26','WINJ26'), '2026-02': ('WING26','WINJ26'),
              '2026-03': ('WINJ26','WINM26'), '2026-04': ('WINJ26','WINM26'),
              '2026-05': ('WINM26','WINQ26'), '2026-06': ('WINM26','WINQ26')}
months = {'2026-01':'JANEIRO','2026-02':'FEVEREIRO','2026-03':'MARÇO',
          '2026-04':'ABRIL','2026-05':'MAIO','2026-06':'JUNHO'}
colors = {'WING26':'#24699f','WINJ26':'#d48637','WINM26':'#409175','WINQ26':'#805dab'}
p = ROOT / 'data/db/bovdb-v3.1-internal.2.sqlite'
with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
    sessions=db.execute('''SELECT t.ticker,p.date,MIN(p.time),MAX(p.time),COUNT(*)
        FROM price1 p JOIN ticker t ON t.id_ticker=p.id_ticker
        WHERE t.ticker IN ('WING26','WINJ26','WINM26','WINQ26')
        AND p.date >= '2026-01-01' AND p.date < '2026-07-01'
        GROUP BY t.ticker,p.date ORDER BY p.date''').fetchall()
rows=[]
for month, symbols in selections.items():
    for sym in symbols:
        found=[r for r in sessions if r[0]==sym and r[1].startswith(month)]
        if not found: raise ValueError(f'Sem dados: {month} {sym}')
        rows.extend((month,*r) for r in found)
out=ROOT/'figures';out.mkdir(exist_ok=True)
with (out/'sessoes-win-selecao-mensal-2026.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['mes','contrato','data','primeiro_candle','ultimo_candle','candles']);w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(14,8.2));fig.patch.set_facecolor('white')
for idx,(month,symbols) in enumerate(selections.items()):
    for sub,sym in enumerate(symbols):
        y=11-(idx*2+sub); found=[r for r in rows if r[0]==month and r[1]==sym]
        days=[int(r[2][8:10]) for r in found]; color=colors[sym]
        ax.scatter(days,[y]*len(days),s=22,color=color,zorder=3)
        ax.text(-0.1,y,months[month] if sub==0 else '',ha='left',va='center',fontweight='bold',color='#273545',fontsize=10)
        ax.text(5.4,y,sym,ha='left',va='center',color=color,fontweight='bold',fontsize=10)
        ax.text(42,y,f'{min(days):02d}–{max(days):02d}  |  {len(days)} pregões',va='center',fontsize=9,color='#465666')
    if idx<5: ax.axhline(9.5-idx*2,color='#e4e9ef',lw=.8)
# Calendar day positions shifted to leave labels at left.
for collection in ax.collections:
    points=collection.get_offsets(); points[:,0]+=9; collection.set_offsets(points)
ax.set_xlim(-.5,55);ax.set_ylim(-.8,11.8)
ax.set_xticks([10,14,19,24,29,34,39,40],['1','5','10','15','20','25','30','31'])
ax.set_xlabel('Dia do mês',labelpad=10,color='#526170')
ax.set_yticks([]);ax.tick_params(axis='x',length=0,pad=7)
for x in [10,14,19,24,29,34,39]: ax.axvline(x,color='#edf0f4',lw=.7,zorder=0)
for s in ax.spines.values():s.set_visible(False)
fig.text(.08,.95,'Mini-índice: os dois contratos de cada mês',fontsize=19,fontweight='bold',color='#172636')
fig.text(.08,.915,'Datas observadas no BovDB  |  janeiro a junho de 2026',fontsize=11,color='#5b6876')
fig.text(.08,.055,'Cada ponto é um pregão com candle; os números mostram a primeira e a última data observadas no mês.\nJaneiro foi escolhido; fevereiro a junho são seleção preliminar pelo volume do mês anterior, sujeita à validação.',fontsize=9,color='#5b6876')
fig.subplots_adjust(left=.08,right=.97,top=.88,bottom=.16)
for ext in ['png','pdf']:fig.savefig(out/f'contratos-win-seis-meses-2026.{ext}',dpi=200,facecolor='white')
print('Gerado:',len(rows),'sessões de contrato')
