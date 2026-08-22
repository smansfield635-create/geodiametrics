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
raw=pd.read_excel(p,sheet_name='CPP',header=None,nrows=35,engine='openpyxl')
block=raw.iloc[12:30].fillna('').astype(str)
summary={'url':URL,'shape_first35':list(raw.shape),'cpp_rows_13_30':[]}
for idx,row in block.iterrows():
    summary['cpp_rows_13_30'].append({'excel_row':int(idx+1),'cells':[{'col':int(j+1),'value':v} for j,v in enumerate(row.tolist()) if v!='']})
(OUT/'cpp_column_schema.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
