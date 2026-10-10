#!/usr/bin/env python3
from dataclasses import dataclass,field
import hashlib,re,json
@dataclass
class GlobalLedger:
 unique:dict=field(default_factory=dict)
 stats:dict=field(default_factory=dict)
 def ingest(self,source,text,is_resume=True,is_us=True,authorized=True):
  s=self.stats.setdefault(source,{"attempted":0,"accepted":0,"duplicates":0,"rejected":0})
  s["attempted"]+=1
  if not (authorized and is_resume and is_us):
   s["rejected"]+=1;return "REJECT"
  n=text.lower()
  n=re.sub(r'(?im)^\s*(?:email|e-mail|phone|telephone|mobile|cell)\s*[:|-].*$',' ',n)
  n=re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b',' ',n)
  n=re.sub(r'\+?\d[\d\s().-]{7,}\d',' ',n)
  n=re.sub(r'\s+',' ',n).strip()
  fp=hashlib.sha256(n.encode()).hexdigest()
  if fp in self.unique:
   s["duplicates"]+=1;return "DUPLICATE"
  self.unique[fp]={"firstSource":source};s["accepted"]+=1;return "ACCEPT"
 def receipt(self):
  ranked=[]
  for k,v in self.stats.items():
   attempted=v["attempted"]
   ranked.append({"source":k,**v,"netNewYield":v["accepted"]/attempted if attempted else 0})
  ranked.sort(key=lambda x:(-x["netNewYield"],-x["accepted"],x["source"]))
  return {"schema":"LBS_MULTISOURCE_ACQUISITION_RECEIPT_v1","globalUnique":len(self.unique),"goal":1000000,"goalReached":len(self.unique)>=1000000,"sources":ranked}
def validate_registry(reg):
 if reg.get("schema")!="LBS_RESUME_SOURCE_REGISTRY_v1":raise ValueError("BAD_REGISTRY_SCHEMA")
 ids=[x["id"] for x in reg["sources"]]
 if len(ids)!=len(set(ids)):raise ValueError("DUPLICATE_SOURCE_ID")
 allowed={"DISCOVERY_ACTIVE","PERMISSION_REVIEW","DISABLED","EXHAUSTED"}
 for s in reg["sources"]:
  if s["status"] not in allowed:raise ValueError("BAD_SOURCE_STATUS")
  if not 0<=int(s["priority"])<=100:raise ValueError("BAD_PRIORITY")
 return True
def ranked_registry(reg,ledger):
 rows=[]
 for src in reg["sources"]:
  st=ledger.stats.get(src["id"],{"attempted":0,"accepted":0})
  yield_rate=st["accepted"]/st["attempted"] if st["attempted"] else None
  rows.append({"id":src["id"],"status":src["status"],"basePriority":src["priority"],"measuredNetNewYield":yield_rate})
 return sorted(rows,key=lambda x:(x["status"]!="DISCOVERY_ACTIVE",-(x["measuredNetNewYield"] if x["measuredNetNewYield"] is not None else -1),-x["basePriority"],x["id"]))
if __name__=="__main__":
 print(json.dumps({"schema":"LBS_MULTISOURCE_ORCHESTRATOR_v2","result":"READY"},sort_keys=True))
