from __future__ import annotations

import hashlib, json, re, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED=256
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; OUT=ROOT/'outputs'
RAW.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
README='https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/README.md'
EXPECTED={
 'pai_job_table.tar.gz':'5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0',
 'pai_task_table.tar.gz':'cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5',
 'pai_instance_table.tar.gz':'1bf1e423a7ce3f8d086699801c362fd56a7182abdb234139e5ebbed97995ca06',
 'pai_machine_spec.tar.gz':'cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d'}
JOB=['job_name','inst_id','user','status','start_time','end_time']
TASK=['job_name','task_name','inst_num','status','start_time','end_time','plan_cpu','plan_mem','plan_gpu','gpu_type']
INST=['job_name','task_name','inst_name','worker_name','inst_id','status','start_time','end_time','machine']
SPEC=['machine','gpu_type','cap_cpu','cap_mem','cap_gpu']

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()

def links():
 r=requests.get(README,timeout=60); r.raise_for_status(); text=r.text
 urls=re.findall(r'\[[^\]]+\]\((https?://[^)]+)\)',text)+re.findall(r'https?://[^\s<>"\)]+',text)
 out={}
 for u in urls:
  u=u.rstrip('.,)')
  for fn in EXPECTED:
   if fn in u: out.setdefault(fn,u)
 missing=sorted(set(EXPECTED)-set(out))
 if missing: raise RuntimeError(f'missing official links: {missing}')
 return out

def download(u,p):
 with requests.get(u,stream=True,timeout=300) as r:
  r.raise_for_status()
  with open(p,'wb') as f:
   for c in r.iter_content(8*1024*1024):
    if c: f.write(c)

def ecdf(train,vals):
 tr=np.sort(pd.Series(train).dropna().to_numpy(float))
 if not len(tr): return pd.Series(np.nan,index=vals.index)
 return vals.map(lambda x: np.nan if pd.isna(x) else np.searchsorted(tr,x,side='right')/len(tr))

def binary(y,p):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
 return {'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
  'sensitivity':float(recall_score(y,p,zero_division=0)),
  'specificity':float(tn/(tn+fp)) if tn+fp else None,
  'precision':float(precision_score(y,p,zero_division=0)),
  'balanced_accuracy':float(balanced_accuracy_score(y,p)),
  'f1':float(f1_score(y,p,zero_division=0)),'mcc':float(matthews_corrcoef(y,p))}

def prob(y,p):
 return {'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),
  'Brier':float(brier_score_loss(y,np.clip(p,0,1)))}

print('Resolving and downloading canonical archives...',flush=True)
ls=links(); registry=[]
for fn,expected in EXPECTED.items():
 p=RAW/fn; print(f'Downloading {fn}',flush=True); download(ls[fn],p)
 actual=digest(p); registry.append({'file':fn,'url':ls[fn],'bytes':p.stat().st_size,
  'sha256':actual,'expected_sha256':expected,'checksum_pass':actual==expected})
 if actual!=expected: raise RuntimeError(f'checksum failed: {fn}')
 with tarfile.open(p,'r:gz') as t: t.extractall(RAW)
(OUT/'source_registry.json').write_text(json.dumps(registry,indent=2))

print('Loading job, task, and machine specification tables...',flush=True)
jobs=pd.read_csv(RAW/'pai_job_table.csv',header=None,names=JOB)
tasks=pd.read_csv(RAW/'pai_task_table.csv',header=None,names=TASK)
spec=pd.read_csv(RAW/'pai_machine_spec.csv',header=None,names=SPEC)
jobs=jobs[jobs.status.isin(['Failed','Terminated'])].copy()
jobs['outcome']=(jobs.status=='Failed').astype(int)
jobs['split']=jobs.user.astype(str).map(lambda u:'development' if int(hashlib.sha256(f'{SEED}:{u}'.encode()).hexdigest()[:8],16)/0xffffffff<.75 else 'evaluation')

t=tasks.copy(); req=['job_name','task_name','inst_num','start_time','plan_cpu','plan_mem','plan_gpu','gpu_type']
t['task_complete']=t[req].notna().mean(axis=1)
tagg=t.groupby('job_name').agg(task_count=('task_name','size'),planned_instances=('inst_num','sum'),
 total_cpu=('plan_cpu',lambda x:np.nansum(x)),total_mem=('plan_mem',lambda x:np.nansum(x)),
 total_gpu=('plan_gpu',lambda x:np.nansum(x)),peak_cpu=('plan_cpu','max'),peak_mem=('plan_mem','max'),
 peak_gpu=('plan_gpu','max'),first_task_start=('start_time','min'),task_complete=('task_complete','mean'),
 gpu_type_mode=('gpu_type',lambda x:x.dropna().astype(str).mode().iloc[0] if len(x.dropna()) else 'MISSING')).reset_index()

print('Streaming and aggregating the 663 MB instance archive...',flush=True)
parts=[]
for chunk in pd.read_csv(RAW/'pai_instance_table.csv',header=None,names=INST,chunksize=1_000_000):
 chunk['inst_complete']=chunk[['job_name','task_name','inst_name','worker_name','start_time','machine']].notna().mean(axis=1)
 chunk=chunk.merge(spec,on='machine',how='left')
 a=chunk.groupby('job_name').agg(observed_instances=('worker_name','size'),unique_machines=('machine','nunique'),
  first_instance_start=('start_time','min'),last_instance_start=('start_time','max'),
  instance_launch_coverage=('start_time',lambda x:x.notna().mean()),instance_complete=('inst_complete','mean'),
  spec_coverage=('cap_cpu',lambda x:x.notna().mean()),sum_cap_cpu=('cap_cpu','sum'),
  sum_cap_mem=('cap_mem','sum'),sum_cap_gpu=('cap_gpu','sum')).reset_index()
 parts.append(a)
inst=pd.concat(parts,ignore_index=True).groupby('job_name').agg(observed_instances=('observed_instances','sum'),
 unique_machines=('unique_machines','sum'),first_instance_start=('first_instance_start','min'),
 last_instance_start=('last_instance_start','max'),instance_launch_coverage=('instance_launch_coverage','mean'),
 instance_complete=('instance_complete','mean'),spec_coverage=('spec_coverage','mean'),sum_cap_cpu=('sum_cap_cpu','sum'),
 sum_cap_mem=('sum_cap_mem','sum'),sum_cap_gpu=('sum_cap_gpu','sum')).reset_index()

d=jobs.merge(tagg,on='job_name',how='left').merge(inst,on='job_name',how='left')
d['task_wait']=d.first_task_start-d.start_time; d['instance_wait']=d.first_instance_start-d.start_time
d['launch_spread']=(d.last_instance_start-d.first_instance_start).clip(lower=0)
d['instance_ratio']=d.observed_instances/d.planned_instances.replace(0,np.nan)
d['cpu_capacity_ratio']=(d.total_cpu/100)/d.sum_cap_cpu.replace(0,np.nan)
d['mem_capacity_ratio']=d.total_mem/d.sum_cap_mem.replace(0,np.nan)
d['gpu_capacity_ratio']=(d.total_gpu/100)/d.sum_cap_gpu.replace(0,np.nan)
d['ordering_consistent']=((d.task_wait>=0)&(d.instance_wait>=0)).astype(float)
d=d.sort_values('start_time').reset_index(drop=True)
tm=d.start_time.to_numpy(float); left=np.searchsorted(tm,tm-300,side='left')
d['cluster_density_5m']=np.arange(len(d))-left; d['user_density_5m']=0
for _,idx in d.groupby('user').groups.items():
 idx=np.array(sorted(idx)); u=d.loc[idx,'start_time'].to_numpy(float); l=np.searchsorted(u,u-300,side='left')
 d.loc[idx,'user_density_5m']=np.arange(len(idx))-l

dev=d[d.split=='development'].copy(); freq=dev.gpu_type_mode.value_counts(normalize=True)
d['gpu_scarcity']=1-d.gpu_type_mode.map(freq).fillna(0); dev=d[d.split=='development'].copy()
Bcols=['total_cpu','total_mem','total_gpu','task_count','planned_instances','launch_spread']
Pcols=['task_wait','instance_wait','cluster_density_5m','user_density_5m','gpu_scarcity']
Cap=['cpu_capacity_ratio','mem_capacity_ratio','gpu_capacity_ratio']
R={c:ecdf(dev[c],d[c]) for c in Bcols+Pcols+Cap}; r=pd.DataFrame(R,index=d.index)
d['B']=r[Bcols].mean(axis=1,skipna=False); d['P']=r[Pcols].mean(axis=1,skipna=False)
d['E']=1-r[Cap].mean(axis=1,skipna=False)
d['I']=d[['task_complete','instance_complete','spec_coverage','ordering_consistent']].mean(axis=1,skipna=False)
ratio_dev=dev.instance_ratio.clip(upper=1); spread90=dev.launch_spread.quantile(.90)
d['V']=d.instance_ratio.clip(lower=0,upper=1)*d.instance_launch_coverage*(d.launch_spread<=spread90).astype(float)
d['W']=d[['E','I','V']].min(axis=1,skipna=False); d['coverage']=d[['B','P','E','I','V']].notna().mean(axis=1)
dev2=d[(d.split=='development')&(d.coverage==1)]; bq=dev2.B.quantile(.75); pq=dev2.P.quantile(.75); eps=dev2.W.quantile(.25)
d['B_norm']=d.B/bq; d['P_norm']=d.P/pq; d['Pi']=d.B_norm*d.P_norm
d['K']=(d.E*d.I*d.V*d.coverage).clip(lower=.05); d['PCR']=d.Pi/d.K; d['H_star']=d.PCR/(1+d.PCR)
d['MQ']=((d.B_norm>=1)&(d.P_norm>=1)&(d.W<=eps)).astype(int)

eval=d[d.coverage==1].copy(); train=eval[eval.split=='development']; test=eval[eval.split=='evaluation']; y=test.outcome.to_numpy()
axes=['B','P','E','I','V']; raw=Bcols+Pcols+Cap+['task_complete','instance_complete','spec_coverage','instance_ratio','launch_spread']
def model(cols,gb=False):
 pipe=Pipeline([('imp',SimpleImputer(strategy='median'))]+([] if gb else [('scale',StandardScaler())])+[
  ('model',HistGradientBoostingClassifier(random_state=SEED,max_iter=200) if gb else LogisticRegression(max_iter=2000,random_state=SEED,class_weight='balanced'))])
 pipe.fit(train[cols],train.outcome); return pipe.predict_proba(test[cols])[:,1]
print('Fitting axis, raw, and augmentation comparators...',flush=True)
p_axis=model(axes); p_raw=model(raw); p_raw_mq=model(raw+['MQ']); p_raw_h=model(raw+['H_star']); p_gb=model(raw,True)
gates={'B_only':test.B_norm>=1,'P_only':test.P_norm>=1,'W_only':test.W<=eps,
 'B_P':(test.B_norm>=1)&(test.P_norm>=1),'P_W':(test.P_norm>=1)&(test.W<=eps),'full_MQ':test.MQ.astype(bool)}
results={'artifact_id':'CLOUD_TEST_1B_ALIBABA_INSTANCE_CAPACITY_EMPIRICAL_RETURN_v1','status':'COMPLETE',
 'claim_boundary':'Post-placement pre-terminal held-out discrimination; no lifetime sensor metrics and no early-warning lead-time claim.',
 'thresholds':{'B_q75':float(bq),'P_q75':float(pq),'epsilon_d':float(eps)},
 'coverage':{'terminal_jobs':int(len(jobs)),'evaluable_jobs':int(len(eval)),'development_jobs':int(len(train)),
  'evaluation_jobs':int(len(test)),'evaluation_failure_prevalence':float(test.outcome.mean())},
 'hard_MQ':binary(y,test.MQ.to_numpy()),'continuous_H_star':prob(y,test.H_star.to_numpy()),
 'axis_logistic':prob(y,p_axis),'raw_logistic':prob(y,p_raw),'raw_plus_MQ':prob(y,p_raw_mq),
 'raw_plus_H_star':prob(y,p_raw_h),'raw_gradient_boosting':prob(y,p_gb),
 'components_AUROC':{'B':float(roc_auc_score(y,test.B)),'P':float(roc_auc_score(y,test.P)),
  'one_minus_E':float(roc_auc_score(y,1-test.E)),'one_minus_I':float(roc_auc_score(y,1-test.I)),
  'one_minus_V':float(roc_auc_score(y,1-test.V)),'one_minus_W':float(roc_auc_score(y,1-test.W)),
  'one_minus_mean_EIV':float(roc_auc_score(y,1-test[['E','I','V']].mean(axis=1))},
 'ablations':{k:binary(y,v.astype(int).to_numpy()) for k,v in gates.items()}}
OUT.mkdir(exist_ok=True); (OUT/'metrics.json').write_text(json.dumps(results,indent=2));
pd.DataFrame([results['hard_MQ']]).to_csv(OUT/'confusion_matrix.csv',index=False)
pd.DataFrame(results['ablations']).T.to_csv(OUT/'ablations.csv')
pd.DataFrame({'job_name':d.job_name,'split':d.split,'outcome':d.outcome,'B':d.B,'P':d.P,'E':d.E,'I':d.I,'V':d.V,'W':d.W,'MQ':d.MQ,'H_star':d.H_star}).to_csv(OUT/'compact_panel.csv',index=False)
receipt={'seed':SEED,'metrics_sha256':digest(OUT/'metrics.json'),'panel_sha256':digest(OUT/'compact_panel.csv')}; (OUT/'run_receipt.json').write_text(json.dumps(receipt,indent=2))
(OUT/'run_log.txt').write_text('Completed Cloud Test 1B instance and capacity run.\n')
print(json.dumps(results,indent=2),flush=True)
