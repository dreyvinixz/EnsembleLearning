import csv, hashlib, re, requests, fitz
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
D='01 02 03 04 05 08 09 10 11 12 15 16 17 18 19 22 23 26 29 30'.split()
def parse(day):
 part='03-1' if int(day)<=11 else '03-4'; url=f'https://arquivos.b3.com.br/bdi/download/bdi/2025-12-{day}/BDI_{part}_202512{day}.pdf'
 r=requests.get(url,timeout=60);r.raise_for_status(); doc=fitz.open(stream=r.content,filetype='pdf'); digest=hashlib.sha256(r.content).hexdigest(); found={}
 for i,p in enumerate(doc):
  if part=='03-1':
   t=p.get_text(); x=t.find('WIN: Ibovespa Míni')
   if x<0: continue
   t=re.split(r'\n[A-Z0-9]{2,5}:',t[x:],maxsplit=1)[0]
   for m in re.finditer(r'(?m)^([A-Z]\d\d)\s+([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+|-)(?=\s)',t):
    if m.group(1) in ('G26','J26','M26'): found[m.group(1)]=(m.group(3),m.group(4),m.group(5),i+1)
  else:
   words=p.get_text('words')
   for w in words:
    if w[4] not in ('WING26','WINJ26','WINM26') or w[0]>80 or w[4][3:] in found: continue
    row=[v for v in words if v[0]>700 and abs(v[1]-w[1])<5.5]
    vals=[]
    for lo,hi in [(700,737),(737,774),(774,816)]:
     vals.append(''.join(v[4] for v in sorted((v for v in row if lo<=v[2]<hi),key=lambda v:(v[1],v[0]))))
    found[w[4][3:]]=(*vals,i+1)
 if set(found)!={'G26','J26','M26'}: raise ValueError((day,found))
 rows=[]
 for sym,(trades,contracts,volume,page) in found.items():
  def n(v):
   if v=='-':return 0
   if not re.fullmatch(r'\d+(?:\.\d{3})*',v):raise ValueError((day,sym,v))
   return int(v.replace('.',''))
  rows.append((f'2025-12-{day}','WIN'+sym,n(trades),n(contracts),n(volume),page,url,digest))
 return rows
with ThreadPoolExecutor(max_workers=5) as pool: rows=[r for batch in pool.map(parse,D) for r in batch]
p=Path('data/b3-win-dezembro-2025.csv')
if p.exists():raise FileExistsError(p)
with p.open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['data','contrato','negocios','contratos_negociados','volume_bdi','pagina','fonte','sha256_pdf']);w.writerows(rows)
for symbol in ['WING26','WINJ26','WINM26']:
 a=[r for r in rows if r[1]==symbol];print(symbol,len(a),sum(r[2] for r in a),sum(r[3] for r in a),sum(r[4] for r in a))
