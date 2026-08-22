from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
BASE = 'https://api.fdic.gov/banks'
START = pd.Timestamp('2004-01-01')
END = pd.Timestamp('2020-12-31')
FIN_FIELDS = ['CERT','REPDTE','ASSET','EQ','RBC1RWAJ','NCLNLSR','ROA','LNLSNET','DEP']
FAIL_FIELDS = ['CERT','NAME','FAILDATE','RESTYPE','RESTYPE1','FIN']


def extract_rows(payload):
    rows = []
    for x in payload.get('data', []):
        rows.append(x.get('data', x) if isinstance(x, dict) else x)
    return rows


def get_pages(endpoint, params, limit=10000):
    rows=[]; offset=0; receipts=[]
    while True:
        p=dict(params); p['limit']=limit; p['offset']=offset; p['format']='json'
        r=requests.get(f'{BASE}/{endpoint}', params=p, timeout=120)
        receipts.append({'endpoint':endpoint,'offset':offset,'status':r.status_code,'url':r.url})
        r.raise_for_status()
        chunk=extract_rows(r.json())
        rows.extend(chunk)
        if len(chunk)<limit: break
        offset += limit
        if offset > 2000000: raise RuntimeError('pagination guard')
    return pd.DataFrame(rows), receipts


def rank01(s, higher=True):
    x=pd.to_numeric(s, errors='coerce')
    r=x.rank(method='average')
    n=x.notna().sum()
    if n <= 1:
        out=pd.Series(np.nan,index=x.index)
    else:
        out=(r-1)/(n-1)
    return out if higher else 1-out


def safe_metric(y,p):
    out={'n':int(len(y)),'events':int(np.sum(y)),'prevalence':float(np.mean(y)) if len(y) else np.nan}
    out['brier']=float(brier_score_loss(y,p))
    out['auroc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
    out['ap']=float(average_precision_score(y,p)) if np.sum(y)>0 else np.nan
    return out


def model_pipeline():
    return Pipeline([
        ('scale', StandardScaler()),
        ('logit', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=256))
    ])


def main():
    fin_filter='REPDTE:[2004-01-01 TO 2020-12-31]'
    fin, rec1=get_pages('financials', {'filters':fin_filter,'fields':','.join(FIN_FIELDS),'sort_by':'REPDTE','sort_order':'ASC'})
    failures, rec2=get_pages('failures', {'fields':','.join(FAIL_FIELDS),'sort_by':'FAILDATE','sort_order':'ASC'})
    pd.DataFrame(rec1+rec2).to_csv(OUT/'source_receipts.csv',index=False)
    fin.to_csv(OUT/'financials_raw.csv',index=False)
    failures.to_csv(OUT/'failures_raw.csv',index=False)
    if fin.empty: raise RuntimeError('No financial rows returned')

    # Normalize and source-QC.
    fin.columns=[str(c).upper() for c in fin.columns]
    for c in FIN_FIELDS:
        if c not in fin.columns: fin[c]=np.nan
    fin['CERT']=pd.to_numeric(fin['CERT'],errors='coerce').astype('Int64')
    fin['REPDTE']=pd.to_datetime(fin['REPDTE'].astype(str), errors='coerce')
    for c in ['ASSET','EQ','RBC1RWAJ','NCLNLSR','ROA','LNLSNET','DEP']:
        fin[c]=pd.to_numeric(fin[c],errors='coerce')
    fin=fin[(fin['REPDTE']>=START)&(fin['REPDTE']<=END)&fin['CERT'].notna()].copy()
    dup=fin.duplicated(['CERT','REPDTE'],keep=False)
    dup_summary=fin.loc[dup].groupby(['CERT','REPDTE']).size().reset_index(name='n') if dup.any() else pd.DataFrame(columns=['CERT','REPDTE','n'])
    dup_summary.to_csv(OUT/'duplicate_key_qc.csv',index=False)
    # Exact duplicate rows are safely collapsed; conflicting duplicate keys fail closed.
    exact_before=len(fin)
    fin=fin.drop_duplicates().copy()
    conflicts=fin.duplicated(['CERT','REPDTE'],keep=False)
    if conflicts.any():
        fin.loc[conflicts].to_csv(OUT/'conflicting_duplicate_keys.csv',index=False)
        raise RuntimeError('Conflicting duplicate CERT-quarter rows')

    # Domain raw measures.
    fin['EQ_ASSET']=np.where(fin['ASSET']>0,fin['EQ']/fin['ASSET'],np.nan)
    fin['LTD']=np.where(fin['DEP']>0,fin['LNLSNET']/fin['DEP'],np.nan)
    # Availability ranks are contemporaneous only.
    grouped=[]
    for dt,g in fin.groupby('REPDTE',sort=True):
        g=g.copy()
        g['A_EQ']=rank01(g['EQ_ASSET'],True)
        g['A_RBC']=rank01(g['RBC1RWAJ'],True)
        g['CAPITAL']=g[['A_EQ','A_RBC']].min(axis=1,skipna=False)
        g['ASSET_QUALITY']=rank01(g['NCLNLSR'],False)
        g['EARNINGS']=rank01(g['ROA'],True)
        g['LIQUIDITY']=rank01(g['LTD'],False)
        grouped.append(g)
    panel=pd.concat(grouped,ignore_index=True)
    domains=['CAPITAL','ASSET_QUALITY','EARNINGS','LIQUIDITY']
    panel['ROUTE_COMPLETE']=panel[domains].notna().all(axis=1)
    panel['IMI']=panel[domains].prod(axis=1,min_count=len(domains))
    panel['WMI']=panel[domains].min(axis=1,skipna=False)
    panel['ADD']=panel[domains].mean(axis=1,skipna=False)
    panel['LOG_ASSET']=np.log1p(panel['ASSET'].where(panel['ASSET']>=0))
    panel=panel.sort_values(['CERT','REPDTE'])
    prev=panel[['CERT','REPDTE','IMI']].copy()
    prev['REPDTE']=prev['REPDTE']+pd.offsets.QuarterEnd(1)
    prev=prev.rename(columns={'IMI':'IMI_PREV'})
    panel=panel.merge(prev,on=['CERT','REPDTE'],how='left')
    panel['DELTA_IMI']=panel['IMI']-panel['IMI_PREV']

    # Failure labels from independently adjudicated FDIC failures.
    failures.columns=[str(c).upper() for c in failures.columns]
    if 'CERT' not in failures.columns or 'FAILDATE' not in failures.columns:
        raise RuntimeError('Failure schema missing CERT/FAILDATE')
    failures['CERT']=pd.to_numeric(failures['CERT'],errors='coerce').astype('Int64')
    failures['FAILDATE']=pd.to_datetime(failures['FAILDATE'].astype(str),errors='coerce')
    failure_map=failures.dropna(subset=['CERT','FAILDATE']).groupby('CERT')['FAILDATE'].apply(list).to_dict()
    def label(row,days):
        dates=failure_map.get(row.CERT,[])
        return int(any((d>row.REPDTE) and (d<=row.REPDTE+pd.Timedelta(days=days)) for d in dates))
    panel['FAILURE_4Q']=[label(r,365) for r in panel.itertuples()]
    panel['FAILURE_8Q']=[label(r,730) for r in panel.itertuples()]
    panel['YEAR']=panel['REPDTE'].dt.year
    panel.to_csv(OUT/'bank_quarter_panel.csv',index=False)

    coverage=panel.groupby('YEAR').agg(
        rows=('CERT','size'), route_complete=('ROUTE_COMPLETE','sum'), delta_complete=('DELTA_IMI','count'),
        failures_4q=('FAILURE_4Q','sum'), failures_8q=('FAILURE_8Q','sum'),
        rbc_nonmissing=('RBC1RWAJ','count'), aq_nonmissing=('NCLNLSR','count'), roa_nonmissing=('ROA','count')
    ).reset_index()
    coverage.to_csv(OUT/'year_coverage.csv',index=False)

    model_features={
        'BASE_RAW':['CAPITAL','ASSET_QUALITY','EARNINGS','LIQUIDITY','LOG_ASSET'],
        'ADD_STATE':['CAPITAL','ASSET_QUALITY','EARNINGS','LIQUIDITY','LOG_ASSET','ADD'],
        'IMI_STATE':['CAPITAL','ASSET_QUALITY','EARNINGS','LIQUIDITY','LOG_ASSET','IMI'],
        'WMI_STATE':['CAPITAL','ASSET_QUALITY','EARNINGS','LIQUIDITY','LOG_ASSET','WMI'],
        'TRAJ':['CAPITAL','ASSET_QUALITY','EARNINGS','LIQUIDITY','LOG_ASSET','IMI','DELTA_IMI']
    }
    all_metrics=[]; all_preds=[]
    for outcome in ['FAILURE_4Q','FAILURE_8Q']:
        # Primary comparisons use the identical TRAJ-complete population.
        matched=panel.dropna(subset=model_features['TRAJ']+[outcome]).copy()
        for yr in sorted(matched['YEAR'].unique()):
            train=matched[matched['YEAR']<yr]
            test=matched[matched['YEAR']==yr]
            if train['YEAR'].nunique()<4 or train[outcome].nunique()<2 or test[outcome].nunique()<2:
                continue
            for name,features in model_features.items():
                m=model_pipeline(); m.fit(train[features],train[outcome].astype(int))
                p=m.predict_proba(test[features])[:,1]
                met=safe_metric(test[outcome].astype(int).to_numpy(),p)
                met.update({'outcome':outcome,'year':int(yr),'model':name})
                all_metrics.append(met)
                tmp=pd.DataFrame({'CERT':test['CERT'].astype(str).to_numpy(),'REPDTE':test['REPDTE'].astype(str).to_numpy(),'YEAR':yr,'outcome':outcome,'model':name,'y':test[outcome].astype(int).to_numpy(),'p':p,'IMI':test['IMI'].to_numpy(),'WMI':test['WMI'].to_numpy(),'DELTA_IMI':test['DELTA_IMI'].to_numpy()})
                all_preds.append(tmp)
    metrics=pd.DataFrame(all_metrics); preds=pd.concat(all_preds,ignore_index=True) if all_preds else pd.DataFrame()
    metrics.to_csv(OUT/'holdout_metrics.csv',index=False); preds.to_csv(OUT/'holdout_predictions.csv',index=False)

    pooled=[]
    if len(preds):
        for (outcome,model),g in preds.groupby(['outcome','model']):
            met=safe_metric(g['y'].to_numpy(),g['p'].to_numpy()); met.update({'outcome':outcome,'model':model}); pooled.append(met)
    pooled=pd.DataFrame(pooled); pooled.to_csv(OUT/'pooled_metrics.csv',index=False)

    # Structural enrichment on exact-delta complete route, independent of model fit.
    structural=[]
    s=panel.dropna(subset=['IMI','WMI','DELTA_IMI']).copy()
    for outcome in ['FAILURE_4Q','FAILURE_8Q']:
        for metric in ['IMI','WMI']:
            s['decile']=s.groupby('REPDTE')[metric].transform(lambda x: pd.qcut(x.rank(method='first'),10,labels=False,duplicates='drop'))
            low=s[s['decile']==0]; rest=s[s['decile']>0]
            lr=float(low[outcome].mean()) if len(low) else np.nan; rr=float(rest[outcome].mean()) if len(rest) else np.nan
            structural.append({'outcome':outcome,'metric':metric,'low_decile_n':len(low),'low_decile_events':int(low[outcome].sum()),'low_decile_rate':lr,'rest_n':len(rest),'rest_events':int(rest[outcome].sum()),'rest_rate':rr,'risk_ratio':float(lr/rr) if rr>0 else np.nan})
    structural=pd.DataFrame(structural); structural.to_csv(OUT/'structural_enrichment.csv',index=False)

    findings={'status':'VALID_EXECUTION','raw_financial_rows':int(exact_before),'deduplicated_rows':int(len(fin)),'failures_rows':int(len(failures)),'route_complete_rows':int(panel['ROUTE_COMPLETE'].sum()),'delta_complete_rows':int(panel['DELTA_IMI'].notna().sum()),'outcomes':{}}
    for outcome in ['FAILURE_4Q','FAILURE_8Q']:
        pp=pooled[pooled['outcome']==outcome] if len(pooled) else pd.DataFrame()
        mm=metrics[metrics['outcome']==outcome] if len(metrics) else pd.DataFrame()
        if len(pp):
            pmap=pp.set_index('model').to_dict('index')
            non=[m for m in ['BASE_RAW','ADD_STATE','IMI_STATE','WMI_STATE'] if m in pmap]
            best=min(non,key=lambda m:pmap[m]['brier']) if non else None
            traj=pmap.get('TRAJ')
            improvement=(pmap[best]['brier']-traj['brier'])/pmap[best]['brier'] if best and traj else np.nan
            wins=0; eligible=0
            if best and len(mm):
                for yr,g in mm.groupby('year'):
                    d=g.set_index('model')['brier'].to_dict()
                    if best in d and 'TRAJ' in d:
                        eligible+=1; wins+=int(d['TRAJ']<d[best])
            win_rate=wins/eligible if eligible else np.nan
            material=bool(best and traj and improvement>=0.10 and win_rate>=0.70 and traj['ap']>=pmap[best]['ap'])
            findings['outcomes'][outcome]={'best_nontrajectory':best,'pooled':pmap,'traj_brier_relative_improvement':float(improvement),'eligible_years':eligible,'traj_year_wins':wins,'traj_win_rate':float(win_rate) if eligible else None,'material_predictive_superiority':material}
    findings['structural_enrichment']=structural.to_dict('records')
    (OUT/'findings.json').write_text(json.dumps(findings,indent=2,default=str))
    print(json.dumps(findings,indent=2,default=str))

if __name__=='__main__':
    main()
