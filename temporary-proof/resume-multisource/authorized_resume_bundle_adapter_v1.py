#!/usr/bin/env python3
"""Offline importer for authorized resume bundles. No site crawling, login automation, or raw-resume persistence."""
from __future__ import annotations
import argparse,hashlib,io,json,pathlib,re,sqlite3,zipfile
from xml.etree import ElementTree as ET
from resume_detector import classify_resume,semantic_fingerprint

SUPPORTED={".txt",".md",".html",".htm",".docx",".pdf"}

def _text_from_docx(data:bytes)->str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml=z.read("word/document.xml")
    root=ET.fromstring(xml)
    return " ".join((n.text or "") for n in root.iter() if n.tag.endswith("}t"))

def _text_from_pdf(data:bytes)->str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("PDF_REQUIRES_PYPDF") from e
    return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)

def extract_text(name:str,data:bytes)->str:
    ext=pathlib.Path(name).suffix.lower()
    if ext in {".txt",".md",".html",".htm"}:
        return data.decode("utf-8","ignore")
    if ext==".docx": return _text_from_docx(data)
    if ext==".pdf": return _text_from_pdf(data)
    raise ValueError("UNSUPPORTED_DOCUMENT_TYPE")

def iter_bundle(path:pathlib.Path):
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                yield p.name,p.read_bytes()
        return
    if path.suffix.lower()==".zip":
        with zipfile.ZipFile(path) as z:
            for info in sorted(z.infolist(),key=lambda x:x.filename):
                if info.is_dir(): continue
                pure=pathlib.PurePosixPath(info.filename)
                if ".." in pure.parts: raise ValueError("UNSAFE_ZIP_PATH")
                if pure.suffix.lower() in SUPPORTED:
                    yield pure.name,z.read(info)
        return
    yield path.name,path.read_bytes()

def open_ledger(path:pathlib.Path):
    db=sqlite3.connect(path)
    db.execute("create table if not exists resumes (fingerprint text primary key, first_source text not null, first_record_hash text not null)")
    db.execute("create table if not exists runs (run_id text primary key, source_id text not null, accepted integer, duplicates integer, rejected integer)")
    return db

def authority_ok(a,source_id):
    return (a.get("schema")=="LBS_SOURCE_AUTHORITY_ASSERTION_v1" and
            a.get("sourceId")==source_id and
            a.get("authorityState")=="EXPLICIT_AUTHORIZED" and
            a.get("usMarketState")=="VERIFIED_US_MARKET_CHANNEL" and
            bool(a.get("authorizationBasis")))

def process(source_id,bundle,authority,ledger_path):
    if not authority_ok(authority,source_id): raise PermissionError("SOURCE_AUTHORITY_NOT_ESTABLISHED")
    db=open_ledger(ledger_path)
    stats={"discovered":0,"extracted":0,"accepted":0,"duplicates":0,"rejected":0,"extractFailures":0}
    run_seed=[]
    for name,data in iter_bundle(bundle):
        stats["discovered"]+=1
        record_hash=hashlib.sha256((source_id+"\0"+name+"\0").encode()+hashlib.sha256(data).digest()).hexdigest()
        try: text=extract_text(name,data)
        except Exception:
            stats["extractFailures"]+=1;stats["rejected"]+=1;continue
        stats["extracted"]+=1
        c=classify_resume(text)
        if not c["isResume"]:
            stats["rejected"]+=1;continue
        fp=semantic_fingerprint(text)
        before=db.total_changes
        db.execute("insert or ignore into resumes(fingerprint,first_source,first_record_hash) values(?,?,?)",(fp,source_id,record_hash))
        if db.total_changes>before: stats["accepted"]+=1
        else: stats["duplicates"]+=1
        run_seed.append(record_hash)
    run_id=hashlib.sha256((source_id+"|"+("|".join(sorted(run_seed)))).encode()).hexdigest()
    db.execute("insert or replace into runs values(?,?,?,?,?)",(run_id,source_id,stats["accepted"],stats["duplicates"],stats["rejected"]))
    global_unique=db.execute("select count(*) from resumes").fetchone()[0]
    db.commit();db.close()
    return {"schema":"LBS_AUTHORIZED_BUNDLE_RECEIPT_v1","sourceId":source_id,"runId":run_id,"globalUnique":global_unique,**stats,"rawResumeContentPersisted":False,"contactDataRequired":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-id",required=True);p.add_argument("--bundle",required=True)
    p.add_argument("--authority",required=True);p.add_argument("--ledger-db",required=True)
    a=p.parse_args()
    authority=json.loads(pathlib.Path(a.authority).read_text())
    receipt=process(a.source_id,pathlib.Path(a.bundle),authority,pathlib.Path(a.ledger_db))
    print(json.dumps(receipt,sort_keys=True))
if __name__=="__main__":main()
