#!/usr/bin/env python3
"""Source-agnostic authorized export adapter. Never crawls sites or emits personal data."""
import csv,hashlib,json,argparse,pathlib,sys
REQUIRED={"source_id","record_id","permission","us_market","authentic_resume","normalized_resume_text"}
def digest(t):
    import re
    t=t.lower()
    t=re.sub(r"(?im)^\\s*(?:email|phone|mobile|cell|telephone)\\s*[:|-].*$"," ",t)
    t=re.sub(r"\\b[\\w.+-]+@[\\w.-]+\\.\\w+\\b"," ",t)
    t=re.sub(r"\\+?\\d[\\d\\s().-]{7,}\\d"," ",t)
    return hashlib.sha256(" ".join(t.split()).encode()).hexdigest()
def process(rows,approved_sources):
    seen=set();stats={};rejected_reasons={}
    for row in rows:
        source=row["source_id"]
        s=stats.setdefault(source,{"seen":0,"accepted":0,"duplicate":0,"rejected":0})
        s["seen"]+=1
        reason=None
        if source not in approved_sources:reason="UNAPPROVED_SOURCE"
        elif row["permission"]!="EXPLICIT_AUTHORIZED":reason="NO_REUSE_AUTHORITY"
        elif row["authentic_resume"]!="true":reason="NOT_VERIFIED_AUTHENTIC"
        elif row["us_market"]!="true":reason="NOT_VERIFIED_US"
        elif not row["normalized_resume_text"].strip():reason="EMPTY_DOCUMENT"
        if reason:
            s["rejected"]+=1;rejected_reasons[reason]=rejected_reasons.get(reason,0)+1;continue
        fp=digest(row["normalized_resume_text"])
        if fp in seen:s["duplicate"]+=1;continue
        seen.add(fp);s["accepted"]+=1
    return {"schema":"LBS_AUTHORIZED_MULTISOURCE_EXPORT_RECEIPT_v1","globalUnique":len(seen),"sourceCounts":stats,"rejectedReasons":rejected_reasons,"personalDataEmitted":False,"target":1000000,"targetMet":len(seen)>=1000000}
def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--approved-sources",required=True);a=p.parse_args()
    approved=json.loads(pathlib.Path(a.approved_sources).read_text())
    if approved.get("schema")!="LBS_AUTHORIZED_SOURCE_ALLOWLIST_v1":raise ValueError("INVALID_ALLOWLIST")
    with open(a.manifest,newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        if not REQUIRED.issubset(r.fieldnames or []):raise ValueError("INVALID_EXPORT_SCHEMA")
        result=process(r,set(approved["sourceIds"]))
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
