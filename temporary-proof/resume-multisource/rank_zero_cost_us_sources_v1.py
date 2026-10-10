#!/usr/bin/env python3
import json,sys
p=sys.argv[1] if len(sys.argv)>1 else "control-plane/resume-corpus/zero-cost-us-resume-book-leads.v1.json"
d=json.load(open(p))
def readiness(x):
    if x["cost"]=="NO_COST": cost=2
    elif x["cost"]=="NO_COST_LIMITED": cost=1
    else: cost=0
    direct=2 if any(k in x["access"].lower() for k in ("request","download","resume-book","resume book")) else 1
    return cost*100+direct*10+x["priority"]
rows=sorted(({**x,"readinessScore":readiness(x),"countableNow":0} for x in d["leads"]),key=lambda x:(-x["readinessScore"],x["id"]))
print(json.dumps({"schema":"LBS_ZERO_COST_US_SOURCE_QUEUE_RECEIPT_v1","leadCount":len(rows),"countableNow":0,"top":[{"id":x["id"],"score":x["readinessScore"],"reuseState":x["reuseState"]} for x in rows],"nextGate":"EXPLICIT_CORPUS_PROCESSING_PERMISSION"},sort_keys=True))
