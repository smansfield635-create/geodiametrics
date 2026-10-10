#!/usr/bin/env python3
"""Bounded public Common Crawl URL Index metadata probe; aggregate only."""
import json,urllib.request,urllib.parse,io,sys,re
PREFIX="cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-39/subset=warc/"\nPATHS_URL="https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-39/cc-index-table.paths.gz"
BASE="https://data.commoncrawl.org/"
receipt={"schema":"LBS_Q3_COMMON_CRAWL_BOUNDED_METADATA_PROBE_v1","crawl":"CC-MAIN-2026-39","subset":"warc","result":"NOT_EXECUTED","scope":"ONE_PARQUET_FILE_ONLY","actualResumeCount":None,"usResumeCount":None,"uniqueResumeCount":None,"millionCapable":False,"personalDataRetained":False}
try:
 u=BASE+"?"+urllib.parse.urlencode({"list-type":"2","prefix":PREFIX,"max-keys":2})
 with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"LBS-aggregate-research/1.0"}),timeout=30) as r: xml=r.read()
 text=xml.decode("utf-8","replace"); keys=re.findall(r"<Key>([^<]+\\.parquet)</Key>",text)
 receipt["indexObjectCountReturned"]=len(keys)
 if not keys: raise RuntimeError("NO_PARQUET_KEYS_RETURNED")
 key=keys[0]
 import pyarrow.parquet as pq
 with urllib.request.urlopen(urllib.request.Request(BASE+urllib.parse.quote(key),headers={"User-Agent":"LBS-aggregate-research/1.0"}),timeout=120) as r:
  data=r.read(120*1024*1024+1)
 if len(data)>120*1024*1024: raise RuntimeError("OBJECT_EXCEEDS_120MB_BUDGET")
 pf=pq.ParquetFile(io.BytesIO(data))
 receipt.update(result="PARTIAL_METADATA_MEASURED",sampleParquetBytes=len(data),sampleParquetRows=pf.metadata.num_rows,sampleRowGroups=pf.metadata.num_row_groups,columns=pf.schema.names,objectKeySha256=__import__("hashlib").sha256(key.encode()).hexdigest())
except Exception as e:
 receipt.update(result="FAIL_CLOSED",failureClass=type(e).__name__,failureDetail=str(e)[:200])
print(json.dumps(receipt,sort_keys=True))
sys.exit(0 if receipt["result"]=="PARTIAL_METADATA_MEASURED" else 1)

# trigger: corrected-listing-parser-v1
