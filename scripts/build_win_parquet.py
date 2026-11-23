"""Build the local WIN research base from read-only BovDB SQLite inputs."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sqlite3, sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ensemble_learning.coverage import recent_coverage
SRC_M=ROOT/'data/db/bovdb-v3.1-internal.2.sqlite'
SRC_L=ROOT/'data/db/bovdb-labels-B4-v1.sqlite'
SEL=ROOT/'data/b3_win/selecao-win-25-v3-por-pregao.csv'
OUT=ROOT/'data/processed/win_2026_v1'
TMP=ROOT/'data/processed/.win_2026_v1.building'
TZ=ZoneInfo('America/Sao_Paulo')
START,END='2026-01-02','2026-06-30'

def ro(path):
    return sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro&immutable=1',uri=True)
def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save_parquet(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression='zstd',version='2.6')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,help='New output directory; existing directories are never overwritten')
    args=parser.parse_args()
    output=args.output if args.output.is_absolute() else ROOT/args.output
    tmp=output.with_name('.'+output.name+'.building')
    if output.exists() or tmp.exists(): raise FileExistsError('Refusing to overwrite '+str(output if output.exists() else tmp))
    for p in (SRC_M,SRC_L,SEL):
        if not p.is_file(): raise FileNotFoundError(p)
    sched=[r for r in csv.DictReader(SEL.open(encoding='utf-8',newline='')) if r['contrato'] and START<=r['data']<=END]
    sessions=pd.DataFrame(sched).drop_duplicates(['data','contrato'])
    symbols=sorted(sessions.contrato.unique().tolist())
    marks=','.join('?' for _ in symbols)
    c=ro(SRC_M)
    ticker=pd.read_sql_query(f'SELECT ticker,id_ticker FROM ticker WHERE ticker IN ({marks})',c,params=symbols)
    if set(ticker.ticker)!=set(symbols) or ticker.ticker.duplicated().any(): raise ValueError('Ticker mapping mismatch')
    market=pd.read_sql_query(f"""SELECT t.ticker instrument,p.id_ticker asset_id,p.date session_date,p.time source_time,
        p.open,p.high,p.low,p.close,p.average,p.volume,p.business,p.amount_stock
        FROM price1 p JOIN ticker t ON t.id_ticker=p.id_ticker
        WHERE t.ticker IN ({marks}) AND p.date BETWEEN ? AND ? AND p.time>='09:00:00' AND p.time<'18:00:00'
        ORDER BY p.date,t.ticker,p.time""",c,params=[*symbols,START,END])
    c.close()
    market.session_date=market.session_date.astype(str)
    market=market.merge(sessions[['data','contrato']],left_on=['session_date','instrument'],right_on=['data','contrato'],how='inner',validate='many_to_one')
    market=market.drop(columns=['data','contrato'])
    market['candle_start_local']=pd.to_datetime(market.session_date+' '+market.source_time.astype(str))
    market=market.drop(columns='source_time')
    market=market[['instrument','asset_id','session_date','candle_start_local','open','high','low','close','average','volume','business','amount_stock']]
    key=['asset_id','session_date','candle_start_local']
    if market.duplicated(key).any() or market.empty: raise ValueError('Duplicate or empty selected market data')
    labels_all=market[key+['instrument']].copy()
    labels_all['label_time']=labels_all.candle_start_local.dt.strftime('%H:%M:%S')
    lc=ro(SRC_L)
    for h in (5,10,15):
        tab=f'trend_{h}m'
        df=pd.read_sql_query(f"""SELECT id_ticker asset_id,date session_date,time label_time,trend_label y_b0_{h}m
            FROM {tab} WHERE ticker IN ({marks}) AND date BETWEEN ? AND ? AND time>='09:00:00' AND time<'18:00:00'""",
            lc,params=[*symbols,START,END])
        df.session_date=df.session_date.astype(str); df.label_time=df.label_time.astype(str)
        df=df.merge(ticker,left_on='asset_id',right_on='id_ticker',how='left',validate='many_to_one').drop(columns='id_ticker')
        df=df[df.ticker.isin(symbols)].drop(columns='ticker')
        if df.duplicated(['asset_id','session_date','label_time']).any(): raise ValueError(f'Duplicate target keys: {tab}')
        if not set(df[f'y_b0_{h}m'].dropna().unique()).issubset({0,1}): raise ValueError(f'Invalid binary labels in {tab}')
        labels_all=labels_all.merge(df,on=['asset_id','session_date','label_time'],how='left',validate='one_to_one')
    lc.close()
    targets=labels_all.drop(columns='label_time')
    for h in (5,10,15): targets[f'y_b0_{h}m']=pd.array(targets[f'y_b0_{h}m'],dtype='Int8')
    targets['b0_label_type']='retrospective_target_no_known_at'
    targets=targets[['instrument','asset_id','session_date','candle_start_local','y_b0_5m','y_b0_10m','y_b0_15m','b0_label_type']]
    # Session audit includes scheduled sessions with no records and computes only causal completed-bar windows.
    bars=defaultdict(list)
    for x in market[['instrument','session_date','candle_start_local']].itertuples(index=False): bars[(x.instrument,x.session_date)].append(x.candle_start_local.to_pydatetime())
    label_counts=targets.groupby(['instrument','session_date'])[['y_b0_5m','y_b0_10m','y_b0_15m']].count()
    audit=[]
    for s in sessions.itertuples(index=False):
        symbol,day=s.contrato,s.data; stamps=bars.get((symbol,day),[])
        mins={x.hour*60+x.minute for x in stamps}; run=longest=0
        for m in range(540):
            if 540+m in mins: run=0
            else: run+=1; longest=max(longest,run)
        # Prefix sums compute each causal 60-minute window in O(1), equivalent
        # to recent_coverage over completed bars in [decision-60m, decision).
        bitmap=[int(540+m in mins) for m in range(540)]
        prefix=[0]
        for present in bitmap: prefix.append(prefix[-1]+present)
        qualifying=0
        for m in range(600,1080):
            end=m-540
            qualifying+=int(prefix[end]-prefix[end-60] >= 48)
        # Spot-check the first three selected sessions against the canonical implementation.
        if len(audit)<3:
            aware=[x.replace(tzinfo=TZ) for x in stamps]
            for m in (600, 721, 1079):
                decision=datetime.combine(date.fromisoformat(day),time(m//60,m%60),TZ)
                expected=recent_coverage(aware,decision_time=decision,
                    session_start=datetime.combine(date.fromisoformat(day),time(9,0),TZ),
                    window_minutes=60,minimum_fraction=.8).eligible
                actual=prefix[m-540]-prefix[m-600] >= 48
                if actual != expected: raise AssertionError((symbol,day,m,actual,expected))
        counts=label_counts.loc[(symbol,day)].to_dict() if (symbol,day) in label_counts.index else {}
        audit.append({'session_date':day,'instrument':symbol,'selection_slot':s.slot,'selection_type':s.tipo_decisao,'evidence_month':s.mes_evidencia,
          'expected_clock_minutes':540,'observed_candles':len(mins),'missing_clock_minutes':540-len(mins),
          'coverage_fraction':round(len(mins)/540,6),'first_candle_start':min((x.strftime('%H:%M:%S') for x in stamps),default=''),
          'last_candle_start':max((x.strftime('%H:%M:%S') for x in stamps),default=''),'longest_gap_including_edges_minutes':longest,
          'decision_minutes_with_80pct_history':qualifying,'has_any_80pct_window':int(qualifying>0),
          'labels_nonnull_5m':int(counts.get('y_b0_5m',0)),'labels_nonnull_10m':int(counts.get('y_b0_10m',0)),'labels_nonnull_15m':int(counts.get('y_b0_15m',0))})
    coverage=pd.DataFrame(audit).sort_values(['session_date','instrument'])
    tmp.mkdir(parents=True)
    save_parquet(market,tmp/'market_candles.parquet'); save_parquet(targets,tmp/'b0_targets_retrospective.parquet')
    coverage.to_csv(tmp/'session_coverage.csv',index=False,encoding='utf-8')
    def sha(p):return hash_file(p)
    manifest={'dataset_id':'win_2026_v1','status':'research_base_with_session_bound_release_policy_historical_release_unobserved',
      'period_inclusive':{'start':START,'end':END},'timezone':'America/Sao_Paulo',
      'session':{'start':'09:00','end_exclusive':'18:00','included_candle_starts':'09:00 through 17:59'},
      'universe':symbols,'selection_rule':{'source_file':str(SEL.relative_to(ROOT)),'sha256':sha(SEL),'coverage_threshold_monthly':0.25,
        'coverage_window':'60 min with at least 48 closed one-minute candles','max_gap_minutes':None,'January':'manual roster'},
      'sources':[{'file':str(SRC_M.relative_to(ROOT)),'sha256':sha(SRC_M),'table':'price1','read_mode':'read-only immutable'},
        {'file':str(SRC_L.relative_to(ROOT)),'sha256':sha(SRC_L),'tables':['trend_5m','trend_10m','trend_15m'],'read_mode':'read-only immutable'}],
      'market_candles':{'rows':len(market),'schema':list(market.columns),'description':'Observed source candles only; no synthetic fill or indicators.'},
      'b0_targets_retrospective':{'rows':len(targets),'schema':list(targets.columns),
        'non_null':{f'y_b0_{h}m':int(targets[f'y_b0_{h}m'].notna().sum()) for h in (5,10,15)},
        'known_at_observed':None,'label_info_intervals':{'conservative_start':'session open 09:00 local','conservative_end':'session input completion; includes the candle starting 18:00, which closes at 18:01','scope':'full contract-session, shared by 5m/10m/15m labels'},'release_policy':'After session source finalization and B0 job completion; historical release timestamps are not observed. Any offline walk-forward must declare the simulated post-session availability assumption and purge whole sessions.','warning':'Base is not a direct trainer snapshot: no observed label_known_at exists, and the session-level simulation policy must be applied by the training adapter.'},
      'session_coverage':{'rows':len(coverage),'sessions_without_source_candles':int((coverage.observed_candles==0).sum()),
        'sessions_with_any_causal_80pct_window':int(coverage.has_any_80pct_window.sum()),'description':'Descriptive audit; does not remove rows from raw market data.'},
      'excluded':['weekends','B3 holidays','special session 2026-02-18','candle starts at 18:00 or later','non-selected contract sessions',
        'SIDEWAYS features','derived trend_mean','segment_end_time/type'],
      'label_policy':'Null labels preserved. No label_known_at inferred from horizon or segment_end_time. Conservative information bound is the full session; release event requires finalized source and completed B0 job.',
      'build_script':'scripts/build_win_parquet.py','output_files':{}}
    for p in sorted(tmp.iterdir()):
        if p.is_file():manifest['output_files'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    (tmp/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    readme='''# Base WIN 2026 em Parquet\n\nCandles observados do BovDB para a programação v3 (corte mensal de 25%), no período inclusivo 2026-01-02 a 2026-06-30.\n\n- `market_candles.parquet`: OHLC, média, volume e contadores originais; um registro por candle observado entre 09:00 e 17:59. Sem imputação ou indicadores técnicos.\n- `b0_targets_retrospective.parquet`: alvos B0 de 5/10/15 minutos alinhados por ativo/data/início do minuto; nulos preservados. Alvos são retrospectivos, não entradas. A auditoria estabelece um limite informacional conservador de sessão inteira; B4 não registra `label_known_at` nem tempos reais de ingestão/conclusão. Para walk-forward, declarar a hipótese de liberação pós-sessão e purgar sessões completas.\n- `session_coverage.csv`: cobertura de cada contrato selecionado em cada pregão, inclusive pregões sem candles; janela causal de 60 minutos com pelo menos 48 candles concluídos. Sem limite máximo de lacuna definido.\n- `manifest.json`: hashes das fontes e saídas, esquemas, critérios e limitações.\n\nTimestamps são hora local B3 sem fuso embutido (America/Sao_Paulo no manifesto). `candle_start_local` indica o início do minuto; OHLCV só pode ser usado após o minuto fechar e chegar ao sistema. Construa atributos de previsão somente com candles já encerrados antes da decisão.\n'''
    (tmp/'README.md').write_text(readme,encoding='utf-8')
    if output.exists():raise FileExistsError('Output appeared while building')
    output.parent.mkdir(parents=True,exist_ok=True); os.replace(tmp,output)
    print(json.dumps({'output':str(output),'symbols':symbols,'sessions':len(sessions),'market_rows':len(market),'target_rows':len(targets),
      'target_nonnull':{f'y_b0_{h}m':int(targets[f'y_b0_{h}m'].notna().sum()) for h in (5,10,15)},
      'sessions_without_source':int((coverage.observed_candles==0).sum()),'sessions_with_80_window':int(coverage.has_any_80pct_window.sum()),
      'files':sorted(p.name for p in output.iterdir())},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

