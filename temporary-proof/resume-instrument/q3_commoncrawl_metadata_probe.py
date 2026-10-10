#!/usr/bin/env python3
"""Bounded public Common Crawl URL Index metadata probe; aggregate only."""
import gzip, hashlib, json, os, sys, tempfile, urllib.request
BASE="https://data.commoncrawl.org/"
PATHS_URL=BASE+"crawl-data/CC-MAIN-2026-39/cc-index-table.paths.gz"
receipt={"schema":"LBS_Q3_COMMON_CRAWL_BOUNDED_METADATA_PROBE_v1","crawl":"CC-MAIN-2026-39","subset":"warc","result":"NOT_EXECUTED","scope":"ONE_PARQUET_FILE_ONLY","actualResumeCount":None,"usResumeCount":None,"uniqueResumeCount":None,"millionCapable":False,"personalDataRetained":False}
try:
    headers={"User-Agent":"LBS-aggregate-research/1.0"}
    with urllib.request.urlopen(urllib.request.Request(PATHS_URL,headers=headers),timeout=30) as r:
        paths=gzip.decompress(r.read()).decode("utf-8").splitlines()
    keys=[x.strip() for x in paths if x.strip().endswith(".parquet")]
    receipt["indexObjectCountReturned"]=len(keys)
    if not keys: raise RuntimeError("NO_PARQUET_KEYS_RETURNED")
    key=keys[0]
    import pyarrow.parquet as pq
    fd,tmp_path=tempfile.mkstemp(suffix=".parquet"); os.close(fd)
    try:
        req=urllib.request.Request(BASE+key,headers=headers)
        with urllib.request.urlopen(req,timeout=180) as src, open(tmp_path,"wb") as dst:
            while True:
                chunk=src.read(8*1024*1024)
                if not chunk: break
                dst.write(chunk)
        size=os.path.getsize(tmp_path)
        pf=pq.ParquetFile(tmp_path)
        receipt.update(result="PARTIAL_METADATA_MEASURED",sampleParquetBytes=size,sampleParquetRows=pf.metadata.num_rows,sampleRowGroups=pf.metadata.num_row_groups,columns=pf.schema.names,objectKeySha256=hashlib.sha256(key.encode()).hexdigest())
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
except Exception as e:
    receipt.update(result="FAIL_CLOSED",failureClass=type(e).__name__,failureDetail=str(e)[:300])
print(json.dumps(receipt,sort_keys=True))
sys.exit(0 if receipt["result"]=="PARTIAL_METADATA_MEASURED" else 1)
