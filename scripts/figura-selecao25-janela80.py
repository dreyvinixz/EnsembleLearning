"""Plot v2 scheduled contracts against 80% coverage and the B3 calendar."""
import calendar
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
SOURCE = OUT / 'contratos-win-80porcento-dados.csv'
SCHEDULE = ROOT / 'data/b3_win/selecao-win-25-v3-por-pregao.csv'
MONTHS = ['JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO']
SYMBOL_COLORS = {'WING26':'#005AB5','WINJ26':'#B04A00','WINM26':'#007A52',
                 'WINQ26':'#762A9D','WINV26':'#A50050'}
COLORS = {'fim_semana':'#d5d9de','b3_fechada':'#f28e2b',
          'pregao_especial_excluido':'#a67ad6','sem_price1':'#df4a46',
          'fora_vigencia':'#9cbad1','nao_selecionado':'#b2a9a2',
          'sem_candle_contrato':'#f4df4e','sem_janela_80':'#f4df4e',
          'com_candle':'#ffffff'}
HATCHES = {'fim_semana':'','b3_fechada':'//','pregao_especial_excluido':'xx',
           'sem_price1':'++','fora_vigencia':'..','nao_selecionado':'\\\\',
           'sem_candle_contrato':'--','sem_janela_80':'--','com_candle':''}
with SOURCE.open(encoding='utf-8',newline='') as f:
    coverage={(r['data'],r['contrato']):r for r in csv.DictReader(f)}
with SCHEDULE.open(encoding='utf-8',newline='') as f:
    schedule=list(csv.DictReader(f))
selected={(r['data'],r['contrato']) for r in schedule if r['contrato']}
by_month=defaultdict(set)
for day,sym in selected:by_month[int(day[5:7])].add(sym)
rosters={month:sorted(by_month[month],key=lambda sym:min((r['data'], int(r['slot'])) for r in schedule if r['contrato']==sym and int(r['data'][5:7])==month)) for month in range(1,7)}
records=[]
for month, roster in rosters.items():
    for symbol in roster:
        for day in range(1,calendar.monthrange(2026,month)[1]+1):
            key=date(2026,month,day).isoformat()
            item=coverage.get((key,symbol))
            if item is None: raise ValueError(f'Missing coverage evidence: {key} {symbol}')
            status=item['situacao']
            if status in ('com_candle','sem_janela_80','sem_candle_contrato') and (key,symbol) not in selected:
                status='nao_selecionado'
            records.append((key,symbol,status,item['janelas_aptas_80'],int((key,symbol) in selected)))
OUT.mkdir(exist_ok=True)
with (OUT/'contratos-win-selecao25-janela80-dados.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['data','contrato','situacao','janelas_aptas_80','selecionado']);w.writerows(records)
fig,ax=plt.subplots(figsize=(16,8.8));fig.patch.set_facecolor('white')
total=sum(map(len,rosters.values())); row_index=0
for month,roster in rosters.items():
    for slot,symbol in enumerate(roster):
        y=total-1-row_index;row_index+=1
        ax.text(-10.7,y,MONTHS[month-1] if slot==0 else '',fontsize=10.2,fontweight='bold',va='center',color='#273545')
        ax.text(-6.3,y,symbol,fontsize=10,fontweight='bold',va='center',color=SYMBOL_COLORS[symbol])
        available=[]
        for key,sym,status,_,_ in records:
            dt=date.fromisoformat(key)
            if dt.month != month or sym!=symbol: continue
            x=dt.day
            ax.add_patch(Rectangle((x-.47,y-.37),.94,.74,facecolor=COLORS[status],
                edgecolor='#626a73' if HATCHES[status] else 'white',hatch=HATCHES[status],lw=.35,zorder=0))
            if status=='com_candle':
                available.append(x);ax.scatter(x,y,s=20,color=SYMBOL_COLORS[symbol],zorder=3)
        label=f'{min(available):02d}–{max(available):02d} | {len(available)} dias' if available else 'Sem janela'
        ax.text(32.0,y,label,fontsize=9,color='#465666',va='center')
    if month<6:ax.axhline(total-row_index-.5,color='#dce3ea',lw=.7)
ax.set_xlim(-11,39);ax.set_ylim(-.6,total-.3);ax.set_yticks([])
ax.set_xticks([1,5,10,15,20,25,30,31]);ax.tick_params(axis='x',length=0,pad=8)
ax.set_xlabel('Dia do mês',labelpad=8,color='#526170')
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.07,.965,'Mini-índice: seleção mensal (25%) e janelas de 80%',fontsize=19,fontweight='bold',color='#172636')
fig.text(.07,.93,'Seleção: 25% dos pregões elegíveis do mês anterior  |  janeiro a junho de 2026',fontsize=10.5,color='#5b6876')
legend=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#526c87',markersize=7,label='Ao menos uma janela ≥ 80%')]
legend += [Patch(facecolor=COLORS[k],edgecolor='#626a73' if HATCHES[k] else 'white',hatch=HATCHES[k],label=l)
           for k,l in [('fim_semana','Fim de semana'),('b3_fechada','B3 fechada'),
                       ('pregao_especial_excluido','Pregão especial excluído'),('sem_price1','Sem dados price1'),
                       ('sem_janela_80','Sem janela ≥ 80%'),('fora_vigencia','Após vencimento'),
                       ('nao_selecionado','Não selecionado')]]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.13),ncol=4,frameon=False,fontsize=9)
fig.text(.07,.04,'Ponto: ao menos uma janela de 60 minutos com 48 candles fechados; não significa que o dia inteiro passou.\nJaneiro tem escolha manual retrospectiva; a programação posterior é diagnóstica e depende da validação da listagem histórica.',fontsize=9,color='#586a7b')
fig.subplots_adjust(left=.07,right=.97,top=.89,bottom=.24)
for ext in ('png','pdf'):fig.savefig(OUT/f'contratos-win-selecao25-janela80-2026.{ext}',dpi=200,facecolor='white')
print('Monthly roster',dict(rosters),'rows',len(records))
