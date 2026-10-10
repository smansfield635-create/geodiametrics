#!/usr/bin/env python3
import json,sys
d=json.load(open(sys.argv[1] if len(sys.argv)>1 else "control-plane/resume-corpus/source-discovery-ledger.v4.json"))
advance=[];hold=[];kill=[]
for s in d["sources"]:
    row={"id":s["id"],"priority":s["priority"],"decision":s["decision"],"countableNow":s["countableNow"]}
    if s["decision"].startswith("ADVANCE") or s["decision"].startswith("KEEP"): advance.append(row)
    elif s["decision"].startswith("KILL") or s["decision"].startswith("EXCLUDE"): kill.append(row)
    else: hold.append(row)
advance.sort(key=lambda x:(-x["priority"],x["id"]))
print(json.dumps({"schema":"LBS_SOURCE_DISCOVERY_RANKING_RECEIPT_v1","target":d["target"]["uniqueUSMarketActualResumes"],"advance":advance,"hold":hold,"kill":kill,"countableNow":sum(x["countableNow"] for x in d["sources"])},sort_keys=True))
