#!/usr/bin/env python3
import json,subprocess,sys,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from resume_detector import classify_resume,semantic_fingerprint
raw=subprocess.check_output([sys.executable,str(HERE/"benchmark_generator_v1.py")],text=True)
b=json.loads(raw)
tp=fp=tn=fn=0
for d in b["documents"]:
 pred=classify_resume(d["text"])["isResume"]; pos=d["label"]=="RESUME"
 if pred and pos: tp+=1
 elif pred and not pos: fp+=1
 elif not pred and pos: fn+=1
 else: tn+=1
dup_tp=dup_fp=dup_tn=dup_fn=0
for p in b["dedupPairs"]:
 pred=semantic_fingerprint(p["a"])==semantic_fingerprint(p["b"]); pos=p["label"]=="DUPLICATE"
 if pred and pos:dup_tp+=1
 elif pred and not pos:dup_fp+=1
 elif not pred and pos:dup_fn+=1
 else:dup_tn+=1
prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
dp=dup_tp/(dup_tp+dup_fp) if dup_tp+dup_fp else 0; dr=dup_tp/(dup_tp+dup_fn) if dup_tp+dup_fn else 0
receipt={"schema":"LBS_RESUME_INSTRUMENT_FULL_BENCHMARK_RECEIPT_v1","result":"PASS_CLOSED" if min(prec,rec,dp,dr)>=0.95 else "FAIL_CLOSED","documentCount":len(b["documents"]),"resumeCount":sum(d["label"]=="RESUME" for d in b["documents"]),"negativeCount":sum(d["label"]=="NOT_RESUME" for d in b["documents"]),"dedupPairCount":len(b["dedupPairs"]),"detector":{"tp":tp,"fp":fp,"tn":tn,"fn":fn,"precision":prec,"recall":rec},"dedup":{"tp":dup_tp,"fp":dup_fp,"tn":dup_tn,"fn":dup_fn,"precision":dp,"recall":dr},"threshold":0.95,"realCandidateDataUsed":False,"detectorModifiedAfterFreeze":False}
print(json.dumps(receipt,sort_keys=True))
sys.exit(0 if receipt["result"]=="PASS_CLOSED" else 1)
