#!/usr/bin/env python3
"""Aggregate resume-like URL prevalence in one fixed Common Crawl partition. No URLs emitted."""
import gzip,json,os,re,sys,tempfile,urllib.request
BASE="https://data.commoncrawl.org/"
PATHS=BASE+"crawl-data/CC-MAIN-2026-39/cc-index-table.paths.gz"
out={"schema":"LBS_Q3_RESUME_PATH_PREVALENCE_RECEIPT_v1","crawl":"CC-MAIN-2026-39","scope":"FIRST_PARQUET_PARTITION","personalDataRetained":False,"result":"NOT_EXECUTED"}
try:
    h={"User-Agent":"LBS-aggregate-research/1.0"}
    with urllib.request.urlopen(urllib.request.Request(PATHS,headers=h),timeout=30) as r:
        paths=gzip.decompress(r.read()).decode().splitlines()
    keys=[x.strip() for x in paths if x.strip().endswith(".parquet")]
    key=keys[0]
    import pyarrow.parquet as pq
    fd,p=tempfile.mkstemp(suffix=".parquet"); os.close(fd)
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE+key,headers=h),timeout=180) as src,open(p,"wb") as dst:
            while True:
                b=src.read(8*1024*1024)
                if not b: break
                dst.write(b)
        t=pq.read_table(p,columns=["url_path","fetch_status","content_mime_type","content_mime_detected","content_digest"])
        n=t.num_rows
        lexical=docs=both=0
        digs=set()
        pat=re.compile(r'(?i)(^|[/_.-])(resume|résumé|curriculum[-_. ]?vitae|cv)([/_.-]|$)')
        docpat=re.compile(r'(?i)(pdf|msword|officedocument|word)')
        for pa,st,m,d,di in zip(t["url_path"].to_pylist(),t["fetch_status"].to_pylist(),t["content_mime_type"].to_pylist(),t["content_mime_detected"].to_pylist(),t["content_digest"].to_pylist()):
            try:
                ok=int(st)==200
            except (TypeError,ValueError):
                ok=False
            if not ok: continue
            lx=bool(pat.search(pa or ""))
            dc=bool(docpat.search((m or "")+" "+(d or ""))) or bool(re.search(r'(?i)\.(pdf|doc|docx)$',pa or ""))
            if lx: lexical+=1
            if dc: docs+=1
            if lx and dc:
                both+=1
                if di: digs.add(di)
        out.update(result="PARTIAL_PREVALENCE_MEASURED",partitionRows=n,resumeLexical200=lexical,documentLike200=docs,resumeLexicalDocument200=both,distinctContentDigestsInCandidateSet=len(digs),partitionCount=len(keys),naive900PartitionCandidateExtrapolation=both*len(keys),note="EXTRAPOLATION_ONLY_NOT_POPULATION_PROOF")
    finally:
        if os.path.exists(p): os.remove(p)
except Exception as e:
    out.update(result="FAIL_CLOSED",failureClass=type(e).__name__,failureDetail=str(e)[:300])
print(json.dumps(out,sort_keys=True))
sys.exit(0 if out["result"]=="PARTIAL_PREVALENCE_MEASURED" else 1)
