#!/usr/bin/env python3
"""Bounded public Common Crawl URL Index metadata probe; aggregate only."""
import gzip, hashlib, io, json, sys, urllib.request
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
    with urllib.request.urlopen(urllib.request.Request(BASE+key,headers=headers),timeout=120) as r:
        data=r.read(120*1024*1024+1)
    if len(data)>120*1024*1024: raise RuntimeError("OBJECT_EXCEEDS_120MB_BUDGET")
    pf=pq.ParquetFile(io.BytesIO(data))
    receipt.update(result="PARTIAL_METADATA_MEASURED",sampleParquetBytes=len(data),sampleParquetRows=pf.metadata.num_rows,sampleRowGroups=pf.metadata.num_row_groups,columns=pf.schema.names,objectKeySha256=hashlib.sha256(key.encode()).hexdigest())
except Exception as e:
    receipt.update(result="FAIL_CLOSED",failureClass=type(e).__name__,failureDetail=str(e)[:200])
print(json.dumps(receipt,sort_keys=True))
sys.exit(0 if receipt["result"]=="PARTIAL_METADATA_MEASURED" else 1)
