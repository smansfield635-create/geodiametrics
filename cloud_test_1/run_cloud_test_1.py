from __future__ import annotations
import json, re, subprocess, shutil, sys
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; OUT=ROOT/'outputs'; RAW.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
PDF='https://home.treasury.gov/system/files/256/Investment-Transactions-Report-as-of-02-16-21.pdf'
MAP='''UST,CERT,NAME
UST0458,58134,Bank of Commerce
UST0063,28489,Banner Bank
UST0418,34579,California Oaks State Bank
UST0061,59049,Capital Bank Corporation
UST0597,57026,Carolina Trust Bank
UST0647,57684,CedarStone Bank
UST1034,35117,CenterBank
UST0573,35035,Centrix Bank & Trust
UST0164,35326,Citizens Community Bank
UST0171,57566,Commerce National Bank
UST0057,57873,Commonwealth Business Bank
UST0354,34210,Community Bank of the Bay
UST0681,58159,Community Business Bank
UST0466,58154,DeSoto County Bank
UST0177,8468,Exchange Bank
UST0406,2429,Farmers Bank
UST0625,57514,First Bank of Charleston Inc
UST0649,57966,First Choice Bank
UST0687,57967,First Resource Bank
UST0137,57799,First Sound Bank
UST1010,35527,Fort Lee Federal Savings Bank FSB
UST1144,58523,Georgia Primary Bank
UST1254,58066,Gold Canyon Bank
UST1243,58073,GulfSouth Private Bank
UST0689,58371,Hyperion Bank
UST0203,57379,Independence Bank
UST0278,34937,Magna Bank
UST0860,34233,Marine Bank & Trust Company
UST1047,57821,Maryland Financial Bank
UST0759,57449,Medallion Bank
UST0601,58181,Metro City Bank
UST0138,9889,Mid Penn Bank
UST0883,57580,Midtown Bank & Trust Company
UST0600,57942,MONUMENT BANK
UST0804,57191,Northwest Commercial Bank
UST0386,57850,Ojai Community Bank
UST1196,58238,One Georgia Bank
UST0162,57065,Pacific Commerce Bank
UST0808,57059,Premier Service Bank
UST0165,58325,Presidio Bank
UST1215,58239,Providence Bank
UST0424,57955,Puget Sound Bank
UST1339,22746,Randolph Bank & Trust Company
UST0033,57974,California International Bank NA
UST0540,34806,Santa Clara Valley Bank National Association
UST0104,57053,Signature Bank
UST0099,26849,Carolina First Bank
UST0148,32203,Summit State Bank
UST0643,2039,The Bank of Currituck
UST0652,17909,First State Bank of Mobeetie
UST0470,18067,The Freeport State Bank
UST0500,58099,The Private Bank of California
UST0824,57831,Tifton Banking Company
UST0153,35095,TowneBank
UST0610,16511,Tri-State Bank of Memphis
UST0933,58467,TriSummit Bank
UST1150,58245,Union Bank & Trust Company
UST0664,57447,United American Bank
UST0499,58310,US Metro Bank
UST0254,34689,Valley Community Bank
UST1231,58147,Virginia Company Bank
UST0732,58447,Vision Bank - Texas
UST0156,27009,Wainwright Bank & Trust Company
UST1120,57377,Traditions Bank
'''

def pdf_text(path:Path)->str:
    txt=RAW/'tarp2021.txt'
    if shutil.which('pdftotext'):
        subprocess.run(['pdftotext','-layout',str(path),str(txt)],check=True)
        return txt.read_text(errors='ignore')
    subprocess.run([sys.executable,'-m','pip','install','pypdf'],check=True)
    from pypdf import PdfReader
    t='\n'.join((p.extract_text() or '') for p in PdfReader(str(path)))
    txt.write_text(t); return t

def main():
    mp=pd.read_csv(pd.io.common.StringIO(MAP))
    r=requests.get(PDF,timeout=180); r.raise_for_status(); p=RAW/'tarp2021.pdf'; p.write_bytes(r.content)
    text=pdf_text(p); lines=text.splitlines()
    rows=[]
    for x in mp.itertuples():
        hits=[i for i,l in enumerate(lines) if x.UST in l]
        for j,i in enumerate(hits):
            rows.append({'UST':x.UST,'CERT':x.CERT,'NAME':x.NAME,'occurrence':j+1,'line_no':i,'line':re.sub(r'\s+',' ',lines[i]).strip(),'prev2':re.sub(r'\s+',' ',lines[i-2]).strip() if i>=2 else '', 'prev1':re.sub(r'\s+',' ',lines[i-1]).strip() if i>=1 else '', 'next1':re.sub(r'\s+',' ',lines[i+1]).strip() if i+1<len(lines) else '', 'next2':re.sub(r'\s+',' ',lines[i+2]).strip() if i+2<len(lines) else ''})
    df=pd.DataFrame(rows); df.to_csv(OUT/'ust_exit_source_context.csv',index=False)
    summary={'status':'SOURCE_STRUCTURE_ONLY','matched_population':len(mp),'ust_with_hits':int(df.UST.nunique()),'total_occurrences':len(df),'pdf_bytes':len(r.content)}
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(df.head(20).to_json(orient='records',indent=2))
if __name__=='__main__': main()
