from pathlib import Path
import json, subprocess, sys
import pandas as pd, requests
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(parents=True,exist_ok=True)
URL='https://home.treasury.gov/system/files/256/Investment-Transactions-Report-as-of-02-16-21.xlsx'
p=ROOT/'tarp.xlsx'
r=requests.get(URL,timeout=180); r.raise_for_status(); p.write_bytes(r.content)
try:
    import openpyxl
except Exception:
    subprocess.run([sys.executable,'-m','pip','install','openpyxl'],check=True)
book=pd.ExcelFile(p,engine='openpyxl')
summary={'url':URL,'bytes':p.stat().st_size,'sheets':book.sheet_names,'samples':{}}
for sh in book.sheet_names:
    raw=pd.read_excel(p,sheet_name=sh,header=None,nrows=30,engine='openpyxl')
    vals=raw.fillna('').astype(str).values.tolist()
    summary['samples'][sh]=vals[:15]
(OUT/'treasury_workbook_schema.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2)[:50000])
