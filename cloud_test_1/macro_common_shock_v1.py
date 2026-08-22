import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
SEED=256; BASE='https://api.worldbank.org/v2'; OUT=Path(__file__).parent/'outputs'; OUT.mkdir(exist_ok=True)
IND={'unemp':'SL.UEM.TOTL.ZS','infl':'FP.CPI.TOTL.ZG','debt':'GC.DOD.TOTL.GD.ZS','tourism':'ST.INT.RCPT.XP.ZS','trade':'NE.TRD.GNFS.ZS','reserves':'FI.RES.TOTL.MO','savings':'NY.GNS.ICTR.ZS','capital':'NE.GDI.TOTL.ZS','spi':'IQ.SPI.OVRL','sci':'IQ.SCI.OVRL','gdp_growth':'NY.GDP.MKTP.KD.ZG','employment':'SL.EMP.TOTL.SP.ZS'}
def gj(u):
 r=requests.get(u,timeout=90); r.raise_for_status(); return r.json()
def countries():
 d=gj(f'{BASE}/country?format=json&per_page=400'); return {x['id'] for x in d[1] if x.get('region',{}).get('id')}
def fetch(code):
 u=f'{BASE}/country/all/indicator/{code}?date=2017:2021&format=json&per_page=20000'; d=gj(u)
 if not isinstance(d,list) or len(d)<2 or not isinstance(d[1],list): return pd.DataFrame(columns=['country','year','value']),{'url':u,'status':'NO_DATA','message':d}
 a=[{'country':x.get('countryiso3code'),'year':int(x['date']),'value':float(x['value'])} for x in d[1] if x.get('value') is not None and x.get('countryiso3code')]
 return pd.DataFrame(a),{'url':u,'status':'OK','rows':len(a)}
def pr(s,hi=True):
 r=s.rank(method='average',pct=True); return r if hi else 1-r
def split(c): return 'development' if int(hashlib.sha256(f'{SEED}:{c}'.encode()).hexdigest()[:8],16)/0xffffffff<.75 else 'evaluation'
def sc(y,p):
 y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
 return {'n':len(y),'events':int(y.sum()),'brier':float(brier_score_loss(y,p)),'auroc':float(roc_auc_score(y,p)) if len(set(y))>1 else None,'ap':float(average_precision_score(y,p))}
def logit(tr,te,f,y):
 m=Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED))]); m.fit(tr[f],tr[y]); return m.predict_proba(te[f])[:,1]
def main():
 cs=countries(); fs=[]; reg=[]
 for n,c in IND.items():
  d,meta=fetch(c); d=d[d.country.isin(cs)].copy(); d['name']=n; fs.append(d); reg.append({'name':n,'indicator':c,'countries':int(d.country.nunique()),**meta})
 raw=pd.concat(fs,ignore_index=True); raw.to_csv(OUT/'wdi_raw_long.csv',index=False); (OUT/'source_registry.json').write_text(json.dumps(reg,indent=2,default=str))
 p=raw.pivot_table(index=['country','year'],columns='name',values='value',aggfunc='first').reset_index()
 spi_cov=0
 if 'spi' in p:
  q=p[p.year.isin([2017,2018,2019])].pivot(index='country',columns='year',values='spi'); spi_cov=int(q.dropna().shape[0])
 ii='spi' if spi_cov>=40 else 'sci'
 pre=p[p.year.isin([2017,2018,2019])].groupby('country').mean(numeric_only=True).reset_index(); sh=p[p.year==2020].set_index('country'); rc=p[p.year==2021].set_index('country')
 req=['unemp','infl','debt','tourism','trade','reserves','savings','capital',ii,'gdp_growth','employment']
 missing=[x for x in req if x not in pre.columns]
 if missing: raise RuntimeError(f'Frozen indicators unavailable after coverage fallback: {missing}; registry={reg}')
 e=pre.dropna(subset=req).copy(); e=e[e.country.isin(sh.index)&e.country.isin(rc.index)].copy(); e=e[sh.loc[e.country,'gdp_growth'].notna().to_numpy()&rc.loc[e.country,'gdp_growth'].notna().to_numpy()].copy()
 if len(e)<30: raise RuntimeError(f'Complete frozen route too sparse: n={len(e)}')
 e['B']=pd.concat([pr(e.unemp,False),pr(e.infl.abs(),False),pr(e.debt,False)],axis=1).mean(axis=1); e['P']=pd.concat([pr(e.tourism,False),pr(e.trade,False)],axis=1).mean(axis=1)
 e['E']=pd.concat([pr(e.reserves),pr(e.savings),pr(e.capital)],axis=1).mean(axis=1); e['I']=pr(e[ii]); e['V']=pd.concat([pr(e.gdp_growth),pr(e.employment)],axis=1).mean(axis=1)
 e['IMI']=e.E*e.I*e.V; e['WMI']=e[['E','I','V']].min(axis=1); e['ADD']=e[['E','I','V']].mean(axis=1); e['CS']=1-e.IMI; e['split']=e.country.map(split)
 e['gdp_2020']=sh.loc[e.country,'gdp_growth'].to_numpy(); e['gdp_2021']=rc.loc[e.country,'gdp_growth'].to_numpy(); e['contraction']=(e.gdp_2020<0).astype(int); e['severe']=(e.gdp_2020<=-5).astype(int); e['recovered']=((e.gdp_2020<0)&(e.gdp_2021>0)).astype(int); e['recovery_magnitude']=e.gdp_2021-e.gdp_2020
 for m in ['IMI','WMI']: e[m+'_decile']=pd.qcut(e[m].rank(method='first'),10,labels=False)+1
 structural=[]
 for y in ['contraction','severe']:
  for m in ['IMI','WMI']:
   lo=e[e[m+'_decile']==1]; rr=e[e[m+'_decile']!=1]; structural.append({'outcome':y,'metric':m,'low_n':len(lo),'low_events':int(lo[y].sum()),'low_rate':float(lo[y].mean()),'rest_n':len(rr),'rest_events':int(rr[y].sum()),'rest_rate':float(rr[y].mean()),'risk_ratio':float(lo[y].mean()/rr[y].mean()) if rr[y].mean()>0 else None})
 tr=e[e.split=='development']; te=e[e.split=='evaluation']; models={}
 for y in ['contraction','severe']:
  models[y]={m:sc(te[y],1-te[m]) for m in ['IMI','WMI','ADD','B','P']}
  for n,f in {'BP':['B','P'],'RAW':['B','P','E','I','V'],'BP_IMI':['B','P','IMI'],'BP_WMI':['B','P','WMI']}.items(): models[y][n]=sc(te[y],logit(tr,te,f,y))
 c=e[e.contraction==1]; ctr=c[c.split=='development']; cte=c[c.split=='evaluation']; recovery={'n':len(c),'events':int(c.recovered.sum()),'holdout_n':len(cte),'rho_IMI_magnitude':float(c.IMI.corr(c.recovery_magnitude,method='spearman')),'rho_WMI_magnitude':float(c.WMI.corr(c.recovery_magnitude,method='spearman'))}
 if len(cte)>=5 and ctr.recovered.nunique()==2 and cte.recovered.nunique()==2:
  for m in ['IMI','WMI','ADD']: recovery[m]=sc(cte.recovered,cte[m])
  for n,f in {'BP':['B','P'],'BP_IMI':['B','P','IMI'],'BP_WMI':['B','P','WMI']}.items(): recovery[n]=sc(cte.recovered,logit(ctr,cte,f,'recovered'))
 e.to_csv(OUT/'macro_panel.csv',index=False); pd.DataFrame(structural).to_csv(OUT/'structural_concentration.csv',index=False)
 res={'status':'VALID_EXECUTION','i_indicator_used':IND[ii],'spi_complete_pre_shock_country_count':spi_cov,'eligible_countries':len(e),'development_countries':len(tr),'evaluation_countries':len(te),'contraction_events':int(e.contraction.sum()),'severe_events':int(e.severe.sum()),'recovery_events':int(e.recovered.sum()),'structural_concentration':structural,'holdout_models':models,'recovery':recovery,'claim_boundary':'Retrospective country common-shock realization; no causal, supported-continuity, or predictability-loss claim.'}
 (OUT/'metrics.json').write_text(json.dumps(res,indent=2)); (OUT/'run_receipt.json').write_text(json.dumps({'seed':SEED,'protocol':'IMI_v3_GLOBAL_MACRO_COMMON_SHOCK_REALIZATION_PROTOCOL_v1','panel_sha256':hashlib.sha256((OUT/'macro_panel.csv').read_bytes()).hexdigest(),'metrics_sha256':hashlib.sha256((OUT/'metrics.json').read_bytes()).hexdigest()},indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
