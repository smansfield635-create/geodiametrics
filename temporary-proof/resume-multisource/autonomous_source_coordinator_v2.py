#!/usr/bin/env python3
import json,sys

ORDER=[
 "DISCOVERED","ACCESS_CHECK","PERMISSION_MEASURED","SAMPLE_ACQUIRED",
 "Q3_SAMPLE_QUALIFIED","Q4_METRICS_MEASURED","Q5_PROJECTED",
 "Q6_RIGHTS_GATE","Q7_MILLION_CAPABLE","Q8_SCALE_COLLECT"
]
KILL="EXHAUSTED_OR_KILLED"

def projected(I,r,u,d,p):
    return float(I)*float(r)*float(u)*float(d)*float(p)

def evaluate(s):
    state=s["state"]
    if state==KILL:
        return {**s,"nextAction":"NONE","actor":"SYSTEM"}
    if state not in ORDER:
        raise ValueError("INVALID_STATE")
    if state=="Q5_PROJECTED":
        need={"I","r","u","d","p"}
        if not need.issubset(s):
            return {**s,"nextAction":"MEASURE_MISSING_METRICS","actor":"SYSTEM"}
        e=projected(s["I"],s["r"],s["u"],s["d"],s["p"])
        return {**s,"E":e,"nextAction":"ADVANCE_Q6_RIGHTS_GATE" if e>=1000000 else "KILL_OR_COMBINE_SOURCE","actor":"SYSTEM"}
    if state=="Q6_RIGHTS_GATE":
        if s.get("rightsGate")=="PASS":
            return {**s,"nextAction":"ADVANCE_Q7_MILLION_CAPABLE","actor":"SYSTEM"}
        return {**s,"nextAction":"KILL_SOURCE","actor":"SYSTEM"}
    if state=="Q7_MILLION_CAPABLE":
        return {**s,"nextAction":"START_Q8_SCALE_COLLECTION","actor":"SYSTEM"}
    i=ORDER.index(state)
    return {**s,"nextAction":"ADVANCE_"+ORDER[i+1],"actor":"SYSTEM"}

def receipt(sources):
    rows=[evaluate(x) for x in sources]
    return {
      "schema":"LBS_AUTONOMOUS_SOURCE_COORDINATOR_RECEIPT_v2",
      "ownerActionRequired":False,
      "sources":rows,
      "systemActionCount":len(rows),
      "ownerActionCount":0
    }

if __name__=="__main__":
    d=json.load(open(sys.argv[1]))
    print(json.dumps(receipt(d["sources"]),sort_keys=True))
