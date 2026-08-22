from __future__ import annotations
import hashlib, json, re, shutil, subprocess, sys
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; OUT=ROOT/'outputs'; RAW.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
FDIC='https://api.fdic.gov/banks'
TREASURY='https://home.treasury.gov/system/files/256/Investment-Transactions-Report-as-of-02-16-21.pdf'
EPS=.05
START='2007-01-01'; END='2020-12-31'
FIN_FIELDS=['CERT','REPDTE','NAME','CITY','STALP','ASSET','EQ','RBC1RWAJ','NCLNLSR','ROA','LNLSNET','DEP']
FAIL_FIELDS=['CERT','NAME','FAILDATE']
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

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
    return h.hexdigest()

def rows(payload):
    return [x.get('data',x) if isinstance(x,dict) else x for x in payload.get('data',[])]

def get_pages(endpoint,params,limit=10000):
    out=[]; rec=[]; offset=0
    while True:
        p=dict(params); p.update({'limit':limit,'offset':offset,'format':'json'})
        r=requests.get(f'{FDIC}/{endpoint}',params=p,timeout=120); rec.append({'source':'FDIC','endpoint':endpoint,'offset':offset,'status':r.status_code,'url':r.url}); r.raise_for_status()
        chunk=rows(r.json()); out.extend(chunk)
        if len(chunk)<limit: break
        offset+=limit
        if offset>1500000: raise RuntimeError('FDIC pagination guard')
    return pd.DataFrame(out),rec

def pdf_text(path):
    txt=RAW/'tarp2021.txt'
    if shutil.which('pdftotext'):
        subprocess.run(['pdftotext','-layout',str(path),str(txt)],check=True); return txt.read_text(errors='ignore')
    subprocess.run([sys.executable,'-m','pip','install','pypdf'],check=True)
    from pypdf import PdfReader
    reader=PdfReader(str(path)); t='\n'.join((p.extract_text() or '') for p in reader.pages); txt.write_text(t); return t

def date_from(line):
    m=re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b',line)
    return pd.to_datetime(m.group(1),errors='coerce') if m else pd.NaT

def positive_dollars(line):
    return re.findall(r'(?<!\()\$\s*[0-9][0-9,]*(?:\.\d+)?',line)

def summary_row(line):
    u=line.upper()
    return ('PREFERRED STOCK' in u) and ('$0.00' in line) and any(x in u for x in ['REDEEMED, IN FULL','SOLD, IN FULL','EXITED BANKRUPTCY/RECEIVERSHIP'])

def extract_exit_ledger(text,mp):
    lines=[re.sub(r'\s+',' ',x).strip() for x in text.splitlines()]
    out=[]; context=[]
    for m in mp.itertuples():
        occ=[(i,lines[i]) for i in range(len(lines)) if m.UST in lines[i]]
        sums=[(i,l) for i,l in occ if summary_row(l)]
        if not sums:
            out.append({'UST':m.UST,'CERT':int(m.CERT),'NAME':m.NAME,'status':'NO_CPP_ZERO_CAPITAL_SUMMARY'}); continue
        si,sl=sums[0]; purchase=date_from(sl); u=sl.upper()
        disposition='BANKRUPTCY_RECEIVERSHIP' if 'EXITED BANKRUPTCY/RECEIVERSHIP' in u else ('REDEEMED_IN_FULL' if 'REDEEMED, IN FULL' in u else 'SOLD_IN_FULL')
        next_summary=min([i for i,l in sums[1:] if i>si],default=10**9)
        tx=[(i,l) for i,l in occ if i>si and i<next_summary and not summary_row(l)]
        capital=[]
        if disposition in ['REDEEMED_IN_FULL','SOLD_IN_FULL']:
            for i,l in tx:
                d=date_from(l)
                if pd.notna(d) and len(positive_dollars(l))>=2: capital.append((i,l,d))
            chosen=capital[-1] if capital else None
            method='LAST_PRINCIPAL_REDEMPTION_OR_SALE_ROW'
        else:
            loss=[]
            for i,l in tx:
                d=date_from(l)
                if pd.notna(d) and re.search(r'\(\$\s*[0-9]',l): loss.append((i,l,d))
            chosen=loss[0] if loss else None; method='BANKRUPTCY_RECEIVERSHIP_FULL_CAPITAL_DISPOSITION_ROW'
        for i,l in occ:
            context.append({'UST':m.UST,'CERT':int(m.CERT),'line_no':i,'line':l,'is_first_summary':int(i==si),'in_first_segment':int(i>=si and i<next_summary)})
        if chosen is None or pd.isna(chosen[2]) or pd.isna(purchase) or chosen[2]<=purchase:
            out.append({'UST':m.UST,'CERT':int(m.CERT),'NAME':m.NAME,'purchase_date':purchase,'disposition':disposition,'summary_line':sl,'status':'UNEVALUABLE_EXIT_EXTRACTION'}); continue
        out.append({'UST':m.UST,'CERT':int(m.CERT),'NAME':m.NAME,'purchase_date':purchase,'exit_date':chosen[2],'disposition':disposition,'method':method,'summary_line':sl,'selected_exit_line':chosen[1],'capital_candidate_count':len(capital) if disposition!='BANKRUPTCY_RECEIVERSHIP' else len(loss),'status':'VALID_ZERO_CAPITAL_EXIT'})
    return pd.DataFrame(out),pd.DataFrame(context)

def rank01(s,higher=True):
    x=pd.to_numeric(s,errors='coerce'); r=x.rank(method='average'); n=x.notna().sum(); out=(r-1)/(n-1) if n>1 else pd.Series(np.nan,index=x.index); return out if higher else 1-out

def nearest(g,target,max_days=100):
    if g.empty: return None
    q=g.copy(); q['dist']=(q.REPDTE-target).abs().dt.days; q=q[q.dist<=max_days]
    if q.empty:return None
    return q.sort_values(['dist','REPDTE']).iloc[0]

def main():
    mp=pd.read_csv(StringIO(MAP)); receipts=[]
    tr=requests.get(TREASURY,timeout=180); tr.raise_for_status(); tp=RAW/'tarp2021.pdf'; tp.write_bytes(tr.content); text=pdf_text(tp)
    receipts.append({'source':'US_TREASURY','url':TREASURY,'status':tr.status_code,'bytes':len(tr.content),'sha256':sha256(tp)})
    ledger,ctx=extract_exit_ledger(text,mp); ledger.to_csv(OUT/'cpp_exit_ledger.csv',index=False); ctx.to_csv(OUT/'treasury_source_context.csv',index=False)
    valid=ledger[ledger.status=='VALID_ZERO_CAPITAL_EXIT'].copy()
    if len(valid)<20:
        findings={'status':'FAIL_CLOSED_SOURCE_COVERAGE','valid_exit_dates':int(len(valid)),'frozen_population':int(len(mp))}; (OUT/'metrics.json').write_text(json.dumps(findings,indent=2,default=str)); print(json.dumps(findings,indent=2)); return

    fin,rec=get_pages('financials',{'filters':f'REPDTE:[{START} TO {END}]','fields':','.join(FIN_FIELDS),'sort_by':'REPDTE','sort_order':'ASC'}); receipts+=rec
    failures,rec=get_pages('failures',{'fields':','.join(FAIL_FIELDS),'sort_by':'FAILDATE','sort_order':'ASC'}); receipts+=rec
    pd.DataFrame(receipts).to_csv(OUT/'source_receipts.csv',index=False)
    fin.columns=[str(c).upper() for c in fin.columns]; failures.columns=[str(c).upper() for c in failures.columns]
    fin['CERT']=pd.to_numeric(fin.CERT,errors='coerce').astype('Int64'); fin['REPDTE']=pd.to_datetime(fin.REPDTE.astype(str),errors='coerce')
    for c in ['ASSET','EQ','RBC1RWAJ','NCLNLSR','ROA','LNLSNET','DEP']: fin[c]=pd.to_numeric(fin[c],errors='coerce')
    fin=fin[fin.CERT.notna()].drop_duplicates().copy()
    if fin.duplicated(['CERT','REPDTE']).any(): raise RuntimeError('Conflicting duplicate FDIC bank-quarter keys')
    fin['EQ_ASSET']=np.where(fin.ASSET>0,fin.EQ/fin.ASSET,np.nan); fin['LTD']=np.where(fin.DEP>0,fin.LNLSNET/fin.DEP,np.nan); fin['LOAN_OUTPUT']=np.where(fin.ASSET>0,fin.LNLSNET/fin.ASSET,np.nan)
    parts=[]
    for dt,g in fin.groupby('REPDTE',sort=True):
        g=g.copy(); g['A_EQ']=rank01(g.EQ_ASSET,True); g['A_RBC']=rank01(g.RBC1RWAJ,True); g['CAPITAL']=g[['A_EQ','A_RBC']].min(axis=1,skipna=False); g['ASSET_QUALITY']=rank01(g.NCLNLSR,False); g['EARNINGS']=rank01(g.ROA,True); g['LIQUIDITY']=rank01(g.LTD,False); parts.append(g)
    panel=pd.concat(parts,ignore_index=True).sort_values(['CERT','REPDTE']); domains=['CAPITAL','ASSET_QUALITY','EARNINGS','LIQUIDITY']; panel['IMI']=panel[domains].prod(axis=1,min_count=4); panel['WMI']=panel[domains].min(axis=1,skipna=False); panel['ADD']=panel[domains].mean(axis=1,skipna=False)
    frozen_certs=set(mp.CERT.astype(int)); panel[panel.CERT.astype(int).isin(frozen_certs)].to_csv(OUT/'matched_bank_quarter_panel.csv',index=False)
    failures['CERT']=pd.to_numeric(failures.CERT,errors='coerce').astype('Int64'); failures['FAILDATE']=pd.to_datetime(failures.FAILDATE.astype(str),errors='coerce'); fmap=failures.dropna(subset=['CERT','FAILDATE']).groupby('CERT').FAILDATE.min().to_dict()

    outs=[]
    for r in valid.itertuples():
        cert=int(r.CERT); exitd=pd.Timestamp(r.exit_date); purchase=pd.Timestamp(r.purchase_date); g=panel[panel.CERT==cert].copy(); fail=fmap.get(cert,pd.NaT)
        pre=g[(g.REPDTE<exitd)&g.IMI.notna()].sort_values('REPDTE').tail(1)
        row={'UST':r.UST,'CERT':cert,'NAME':r.NAME,'purchase_date':purchase,'exit_date':exitd,'disposition':r.disposition,'failure_date':fail}
        if pre.empty or (exitd-pre.iloc[0].REPDTE).days>100:
            row['post_exit_status']='UNEVALUABLE_PRE_EXIT'; outs.append(row); continue
        b=pre.iloc[0]; row.update({'pre_exit_date':b.REPDTE,'pre_exit_IMI':b.IMI,'pre_exit_WMI':b.WMI,'pre_exit_ADD':b.ADD,'pre_exit_output':b.LOAN_OUTPUT})
        p4=nearest(g[g.IMI.notna()],b.REPDTE+pd.offsets.QuarterEnd(4)); p8=nearest(g[g.IMI.notna()],b.REPDTE+pd.offsets.QuarterEnd(8))
        for h,x in [(4,p4),(8,p8)]:
            target=b.REPDTE+pd.offsets.QuarterEnd(h); row[f'target{h}_date']=target
            failed=int(pd.notna(fail) and fail<=target); row[f'failed_by_{h}q']=failed
            if x is not None:
                row[f'post{h}_date']=x.REPDTE; row[f'post{h}_IMI']=x.IMI; row[f'post{h}_WMI']=x.WMI; row[f'post{h}_ADD']=x.ADD; row[f'post{h}_output']=x.LOAN_OUTPUT; row[f'delta{h}_IMI']=x.IMI-b.IMI; row[f'delta{h}_WMI']=x.WMI-b.WMI; row[f'delta{h}_ADD']=x.ADD-b.ADD; row[f'continuity_{h}q']=int((not failed) and x.ASSET>0 and pd.notna(x.LOAN_OUTPUT))
            else: row[f'continuity_{h}q']=0
        d4=row.get('delta4_IMI',np.nan); d8=row.get('delta8_IMI',np.nan)
        row['persistent_intrinsic_restoration']=int(row.get('continuity_8q',0)==1 and pd.notna(d4) and pd.notna(d8) and d4>=EPS and d8>=EPS)
        row['transient_restoration_candidate']=int(row.get('continuity_8q',0)==1 and pd.notna(d4) and pd.notna(d8) and d4>=EPS and d8<EPS)
        row['post_exit_continuity_without_restoration']=int(row.get('continuity_8q',0)==1 and row['persistent_intrinsic_restoration']==0)
        row['post_exit_digression']=int(row.get('continuity_8q',0)==1 and pd.notna(d8) and d8<=-EPS)
        row['stable_intrinsic_post_exit']=int(row.get('continuity_8q',0)==1 and pd.notna(d8) and abs(d8)<EPS)
        row['post_exit_failure']=int(row.get('failed_by_8q',0)==1)
        if row['post_exit_failure']: cls='POST_EXIT_FAILURE'
        elif row['persistent_intrinsic_restoration']: cls='PERSISTENT_INTRINSIC_RESTORATION'
        elif row['transient_restoration_candidate']: cls='TRANSIENT_RESTORATION_CANDIDATE'
        elif row['post_exit_digression']: cls='POST_EXIT_DIGRESSION'
        elif row['stable_intrinsic_post_exit']: cls='STABLE_INTRINSIC_POST_EXIT'
        elif row['post_exit_continuity_without_restoration']: cls='POST_EXIT_CONTINUITY_WITHOUT_RESTORATION'
        else: cls='UNEVALUABLE_OR_OTHER_POST_EXIT'
        row['post_exit_status']=cls
        # Reconstruct prior support-era disposition under the immediately preceding frozen rules.
        sb=g[(g.REPDTE<purchase)&g.IMI.notna()].sort_values('REPDTE').tail(1)
        if not sb.empty:
            s=sb.iloc[0]; s4=nearest(g[g.IMI.notna()],s.REPDTE+pd.offsets.QuarterEnd(4)); s8=nearest(g[g.IMI.notna()],s.REPDTE+pd.offsets.QuarterEnd(8)); sd4=s4.IMI-s.IMI if s4 is not None else np.nan; sd8=s8.IMI-s.IMI if s8 is not None else np.nan
            out8=int(s8 is not None and (pd.isna(fail) or fail>s8.REPDTE) and pd.notna(s8.LOAN_OUTPUT) and pd.notna(s.LOAN_OUTPUT) and s8.LOAN_OUTPUT>=.90*s.LOAN_OUTPUT)
            if pd.notna(sd4) and pd.notna(sd8) and sd4>=EPS and sd8>=EPS: prior='SUPPORT_ERA_PERSISTENT_RESTORATION'
            elif out8 and pd.notna(sd8) and sd8<EPS: prior='SUPPORT_ERA_PERSISTENT_CONTINUITY'
            elif pd.notna(sd4) and sd4>=EPS: prior='SUPPORT_ERA_4Q_RESTORATION_CANDIDATE'
            else: prior='OTHER_SUPPORT_ERA'
            row['support_era_status']=prior; row['support_delta4_IMI']=sd4; row['support_delta8_IMI']=sd8
        outs.append(row)
    out=pd.DataFrame(outs); out.to_csv(OUT/'post_exit_outcomes.csv',index=False)
    trans=pd.crosstab(out.get('support_era_status',pd.Series(dtype=str)),out.get('post_exit_status',pd.Series(dtype=str)),dropna=False); trans.to_csv(OUT/'transition_matrix.csv')

    eval8=out[(out.continuity_8q==1)&out.delta4_IMI.notna()&out.delta8_IMI.notna()] if 'continuity_8q' in out else pd.DataFrame()
    findings={'status':'VALID_EXECUTION','protocol':'IMI_v3_CPP_EXIT_AWARE_RESTORATION_PROTOCOL_v1','source_realization_rule':'IMI_v3_CPP_EXIT_SOURCE_REALIZATION_RULE_v1','frozen_population':int(len(mp)),'valid_zero_capital_exits':int(len(valid)),'pre_exit_evaluable':int(out.pre_exit_IMI.notna().sum()) if 'pre_exit_IMI' in out else 0,'intrinsic_8q_evaluable_continuing':int(len(eval8)),'post_exit_failures_by_8q':int(out.get('post_exit_failure',pd.Series(dtype=int)).sum())}
    for h in [4,8]:
        for op in ['IMI','WMI','ADD']:
            s=out.get(f'delta{h}_{op}',pd.Series(dtype=float)).dropna(); findings[f'delta{h}_{op}_n']=int(len(s)); findings[f'delta{h}_{op}_median']=float(s.median()) if len(s) else None
    findings['disposition_counts']={k:int(out.get(k,pd.Series(dtype=int)).sum()) for k in ['persistent_intrinsic_restoration','transient_restoration_candidate','post_exit_continuity_without_restoration','post_exit_digression','stable_intrinsic_post_exit','post_exit_failure']}
    findings['post_exit_status_counts']=out.post_exit_status.value_counts(dropna=False).to_dict()
    findings['persistent_restoration_rate_intrinsic_evaluable']=float(out.persistent_intrinsic_restoration.sum()/len(eval8)) if len(eval8) else None
    findings['median_WMI_minus_IMI_change_8q']=float((out.delta8_WMI-out.delta8_IMI).dropna().median()) if 'delta8_WMI' in out else None
    pc=out[out.support_era_status=='SUPPORT_ERA_4Q_RESTORATION_CANDIDATE'] if 'support_era_status' in out else pd.DataFrame(); findings['support_era_4q_restoration_candidates_with_post_exit']=int(len(pc)); findings['support_era_candidate_post_exit_status']=pc.post_exit_status.value_counts().to_dict() if len(pc) else {}
    findings['operator_discrimination_authorized']=bool(findings['disposition_counts']['persistent_intrinsic_restoration']>=20 and (len(eval8)-findings['disposition_counts']['persistent_intrinsic_restoration'])>=20)
    findings['claim_boundary']='Retrospective exit-aware disposition study within frozen high-confidence CPP-to-FDIC matches. No causal TARP effect, dependency, or predictability-loss claim.'
    (OUT/'metrics.json').write_text(json.dumps(findings,indent=2,default=str))
    receipt={'treasury_pdf_sha256':sha256(tp),'exit_ledger_sha256':sha256(OUT/'cpp_exit_ledger.csv'),'outcomes_sha256':sha256(OUT/'post_exit_outcomes.csv'),'metrics_sha256':sha256(OUT/'metrics.json')}; (OUT/'run_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(findings,indent=2,default=str))
if __name__=='__main__': main()
