"""Read-only reproduction check for B0/B4 labels on the selected WIN sessions."""
from __future__ import annotations
import argparse,csv,json,sqlite3,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def ro(path): return sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--pipeline',type=Path,default=Path(r'C:\mysystems\services\bovdb\bovdb-pipeline'))
 a=p.parse_args(); repo=a.pipeline.resolve(); scripts=repo/'rotulos'/'scripts'
 if not (scripts/'build_b4_experimental.py').is_file(): raise FileNotFoundError(scripts/'build_b4_experimental.py')
 sys.path.insert(0,str(scripts))
 import build_b4_experimental as b4
 pairs=sorted({(r['contrato'],r['data']) for r in csv.DictReader((ROOT/'data/b3_win/selecao-win-25-v3-por-pregao.csv').open(encoding='utf-8',newline='')) if r['contrato']})
 source=ro(ROOT/'data/db/bovdb-v3.1-internal.2.sqlite'); labels=ro(ROOT/'data/db/bovdb-labels-B4-v1.sqlite')
 totals={str(h):{'compared_rows':0,'label_mismatches':0,'boundary_mismatches':0,'type_mismatches':0,'missing_keys':0} for h in (5,10,15)}
 session_mismatches=[]
 for symbol,day in pairs:
  rows=source.execute("""SELECT p.id_ticker,t.ticker,p.date,p.time,p.open,p.close,p.high,p.low,p.volume,p.business
   FROM price1 p JOIN ticker t USING(id_ticker) WHERE t.ticker=? AND p.date=?
   AND p.time>='09:00:00' AND p.time<='18:00:00' ORDER BY p.time""",(symbol,day)).fetchall()
  if not rows: continue
  asset_id=rows[0][0]; frame=b4.build_session(rows)
  for h in (5,10,15):
   tops,bottoms,_=b4.b3.detectar_topos_fundos(frame,f'{h}min')
   pivots=b4.b3.pivots_from_b0(tops,bottoms)
   calculated=b4.b3.pivots_to_binary_trend_labels(frame,pivots)
   expected={r[0]:r[1:] for r in labels.execute(f"SELECT time,trend_label,segment_end_time,segment_end_type FROM trend_{h}m WHERE id_ticker=? AND date=?",(asset_id,day))}
   got={ts.strftime('%H:%M:%S'):(None if pd.isna(calculated.loc[ts,'trend_label']) else int(calculated.loc[ts,'trend_label']),
      None if pd.isna(calculated.loc[ts,'segment_end_time']) else pd.Timestamp(calculated.loc[ts,'segment_end_time']).strftime('%H:%M:%S'),
      None if pd.isna(calculated.loc[ts,'segment_end_type']) else str(calculated.loc[ts,'segment_end_type'])) for ts in calculated.index}
   s=totals[str(h)]; s['compared_rows']+=len(got); s['missing_keys']+=len(set(expected)^set(got))
   mismatches={'label':0,'boundary':0,'type':0}
   for key in set(expected)&set(got):
    e,g=expected[key],got[key]
    mismatches['label']+=int(e[0]!=g[0]); mismatches['boundary']+=int(e[1]!=g[1]); mismatches['type']+=int(e[2]!=g[2])
   s['label_mismatches']+=mismatches['label']; s['boundary_mismatches']+=mismatches['boundary']; s['type_mismatches']+=mismatches['type']
   if any(mismatches.values()) or set(expected)^set(got): session_mismatches.append({'symbol':symbol,'date':day,'horizon':h,**mismatches,'missing_keys':len(set(expected)^set(got))})
 source.close();labels.close()
 print(json.dumps({'selected_sessions':len(pairs),'timeframe_results':totals,'sessions_with_mismatches':session_mismatches[:20],'mismatch_session_count':len(session_mismatches)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
