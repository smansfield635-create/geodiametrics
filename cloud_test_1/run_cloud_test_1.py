from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FDIC_BASE = "https://api.fdic.gov/banks"
TARP_PDF = "https://home.treasury.gov/system/files/256/Investment-Transactions-Report-as-of-02-16-21.pdf"
START = pd.Timestamp("2007-01-01")
END = pd.Timestamp("2013-12-31")
EPS = 0.05
FIN_FIELDS = ["CERT","REPDTE","NAME","CITY","STALP","ASSET","EQ","RBC1RWAJ","NCLNLSR","ROA","LNLSNET","DEP"]
FAIL_FIELDS = ["CERT","NAME","FAILDATE"]
INST_FIELDS = ["CERT","NAME","CITY","STALP"]
CORP_SUFFIX = {"HOLDING","HOLDINGS","BANCSHARES","BANCORP","CORPORATION","CORP","INC","GROUP","COMPANY","CO"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_rows(payload):
    return [x.get("data", x) if isinstance(x, dict) else x for x in payload.get("data", [])]


def get_pages(endpoint, params, limit=10000):
    rows, receipts, offset = [], [], 0
    while True:
        p = dict(params); p.update({"limit": limit, "offset": offset, "format": "json"})
        r = requests.get(f"{FDIC_BASE}/{endpoint}", params=p, timeout=120)
        receipts.append({"source":"FDIC","endpoint":endpoint,"offset":offset,"status":r.status_code,"url":r.url})
        r.raise_for_status()
        chunk = extract_rows(r.json())
        rows.extend(chunk)
        if len(chunk) < limit: break
        offset += limit
        if offset > 2000000: raise RuntimeError("FDIC pagination guard")
    return pd.DataFrame(rows), receipts


def norm(s: str) -> str:
    s = str(s or "").upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\bNATIONAL ASSOCIATION\b", " NA ", s)
    s = re.sub(r"\bN A\b", " NA ", s)
    s = re.sub(r"\bFEDERAL SAVINGS BANK\b", " FSB ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_jaccard(a: str, b: str) -> float:
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / len(A | B) if A | B else 0.0


def pdf_to_text(pdf_path: Path, txt_path: Path) -> str:
    if shutil.which("pdftotext"):
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=True)
        return txt_path.read_text(errors="ignore")
    try:
        from pypdf import PdfReader
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "pypdf"], check=True)
        from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    txt_path.write_text(text)
    return text


def download_tarp():
    p = RAW / "tarp_investment_transactions_2021.pdf"
    r = requests.get(TARP_PDF, timeout=180)
    r.raise_for_status(); p.write_bytes(r.content)
    text = pdf_to_text(p, RAW / "tarp.txt")
    return p, text, {"source":"US_TREASURY","url":TARP_PDF,"status":r.status_code,"bytes":p.stat().st_size,"sha256":sha256(p)}


def rank01(s, higher=True):
    x = pd.to_numeric(s, errors="coerce")
    r = x.rank(method="average"); n = x.notna().sum()
    if n <= 1: out = pd.Series(np.nan, index=x.index)
    else: out = (r - 1) / (n - 1)
    return out if higher else 1 - out


def parse_purchase_lines(text: str, inst: pd.DataFrame) -> pd.DataFrame:
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    # Only CPP section before Citigroup common-stock disposition / CDCI.
    try: start = next(i for i,x in enumerate(lines) if "CAPITAL PURCHASE PROGRAM" in x.upper())
    except StopIteration: start = 0
    stop_candidates = [i for i,x in enumerate(lines[start+1:], start+1) if "CAPITAL PURCHASE PROGRAM - CITIGROUP" in x.upper() or "COMMUNITY DEVELOPMENT CAPITAL INITIATIVE" in x.upper()]
    stop = min(stop_candidates) if stop_candidates else len(lines)
    lines = lines[start:stop]
    inst = inst.copy()
    for c in INST_FIELDS:
        if c not in inst.columns: inst[c] = ""
    inst["CERT"] = pd.to_numeric(inst["CERT"], errors="coerce").astype("Int64")
    inst["KEY"] = inst.apply(lambda r: norm(f"{r['NAME']} {r['CITY']} {r['STALP']}"), axis=1)
    inst["NAME_N"] = inst["NAME"].map(norm)
    out = []
    for line_no, line in enumerate(lines):
        if not re.search(r"\d{1,2}/\d{1,2}/20(?:08|09)", line):
            continue
        if not re.search(r"Preferred Stock|Subordinated|Senior Securities|Common Stock", line, re.I):
            continue
        ln = norm(line)
        dates = re.findall(r"\d{1,2}/\d{1,2}/20(?:08|09)", line)
        if not dates: continue
        purchase_date = pd.to_datetime(dates[0], errors="coerce")
        amts = re.findall(r"\$\s*([0-9][0-9,]*(?:\.\d+)?)", line)
        amount = float(amts[0].replace(",", "")) if amts else np.nan
        exact = inst[inst["KEY"].map(lambda k: bool(k) and k in ln)].copy()
        method = "EXACT_NAME_CITY_STATE"
        score = 1.0
        if len(exact) != 1:
            # Frozen secondary rule: state/city must agree and unique token similarity >= .85.
            state_matches = []
            for r in inst.itertuples():
                st = norm(getattr(r,"STALP","")); city = norm(getattr(r,"CITY","")); nm = norm(getattr(r,"NAME",""))
                if not st or not city or f" {st} " not in f" {ln} " or city not in ln: continue
                # infer seller prefix portion before city occurrence
                prefix = ln.split(city,1)[0]
                prefix = re.sub(r"^UST\d+\s+(?:\d+[A-Z]?(?:\s+\d+[A-Z]?)*\s+)?", "", prefix).strip()
                sc = token_jaccard(prefix, nm)
                # reject obvious holding-company continuation after a bank-name prefix
                if nm in prefix:
                    tail = prefix.split(nm,1)[1].strip().split()
                    if tail and tail[0] in CORP_SUFFIX: sc = 0.0
                if sc >= 0.85: state_matches.append((r, sc))
            state_matches.sort(key=lambda z:z[1], reverse=True)
            if len(state_matches) == 1 or (len(state_matches)>1 and state_matches[0][1] > state_matches[1][1]):
                r, score = state_matches[0]
                exact = pd.DataFrame([r._asdict()]); method = "SECONDARY_UNIQUE_JACCARD"
            else:
                out.append({"line_no":line_no,"line":line,"purchase_date":purchase_date,"amount":amount,"match_status":"UNMATCHED_OR_AMBIGUOUS","match_method":"NONE","score":np.nan})
                continue
        r = exact.iloc[0]
        # extra holding-company safety: if matched bank name is followed by corp suffix before city, reject.
        name_n, city_n = norm(r["NAME"]), norm(r["CITY"])
        pre_city = ln.split(city_n,1)[0] if city_n and city_n in ln else ln
        if name_n in pre_city:
            tail = pre_city.split(name_n,1)[1].strip().split()
            if tail and tail[0] in CORP_SUFFIX:
                out.append({"line_no":line_no,"line":line,"purchase_date":purchase_date,"amount":amount,"match_status":"EXCLUDED_HOLDING_COMPANY","match_method":method,"score":score})
                continue
        out.append({"line_no":line_no,"line":line,"purchase_date":purchase_date,"amount":amount,"match_status":"MATCHED","match_method":method,"score":score,"CERT":int(r["CERT"]),"NAME":r["NAME"],"CITY":r["CITY"],"STALP":r["STALP"]})
    return pd.DataFrame(out)


def nearest_row(g: pd.DataFrame, target: pd.Timestamp, max_q=1):
    if g.empty: return None
    gg = g.copy(); gg["dist"] = (gg["REPDTE"] - target).abs().dt.days
    gg = gg[gg["dist"] <= 100*max_q]
    if gg.empty: return None
    return gg.sort_values(["dist","REPDTE"]).iloc[0]


def bootstrap_median_diff(a, b, seed=256, n=2000):
    a=np.asarray(pd.Series(a).dropna(),float); b=np.asarray(pd.Series(b).dropna(),float)
    if len(a)<5 or len(b)<5: return [None,None,None]
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n): vals.append(np.median(rng.choice(a,len(a),replace=True))-np.median(rng.choice(b,len(b),replace=True)))
    return [float(np.median(a)-np.median(b)), float(np.quantile(vals,.025)), float(np.quantile(vals,.975))]


def main():
    receipts=[]
    inst, rec = get_pages("institutions", {"fields":",".join(INST_FIELDS),"sort_by":"CERT","sort_order":"ASC"}); receipts += rec
    fin, rec = get_pages("financials", {"filters":"REPDTE:[2007-01-01 TO 2013-12-31]","fields":",".join(FIN_FIELDS),"sort_by":"REPDTE","sort_order":"ASC"}); receipts += rec
    failures, rec = get_pages("failures", {"fields":",".join(FAIL_FIELDS),"sort_by":"FAILDATE","sort_order":"ASC"}); receipts += rec
    pdf_path, text, trec = download_tarp(); receipts.append(trec)
    pd.DataFrame(receipts).to_csv(OUT/"source_receipts.csv", index=False)

    for df in [inst, fin, failures]: df.columns=[str(c).upper() for c in df.columns]
    # Treasury-to-FDIC linkage frozen before outcome inspection.
    ledger = parse_purchase_lines(text, inst)
    ledger.to_csv(OUT/"tarp_match_ledger.csv", index=False)
    matched = ledger[ledger.get("match_status",pd.Series(dtype=str))=="MATCHED"].dropna(subset=["CERT","purchase_date"]).copy()
    if matched.empty: raise RuntimeError("No deterministic CPP-to-FDIC matches")
    matched = matched.sort_values("purchase_date").drop_duplicates("CERT", keep="first")

    fin["CERT"]=pd.to_numeric(fin["CERT"],errors="coerce").astype("Int64")
    fin["REPDTE"]=pd.to_datetime(fin["REPDTE"].astype(str),errors="coerce")
    for c in ["ASSET","EQ","RBC1RWAJ","NCLNLSR","ROA","LNLSNET","DEP"]: fin[c]=pd.to_numeric(fin[c],errors="coerce")
    fin=fin[(fin.REPDTE>=START)&(fin.REPDTE<=END)&fin.CERT.notna()].drop_duplicates().copy()
    if fin.duplicated(["CERT","REPDTE"]).any(): raise RuntimeError("Conflicting duplicate FDIC bank-quarter keys")
    fin["EQ_ASSET"]=np.where(fin.ASSET>0,fin.EQ/fin.ASSET,np.nan)
    fin["LTD"]=np.where(fin.DEP>0,fin.LNLSNET/fin.DEP,np.nan)
    fin["LOAN_OUTPUT"]=np.where(fin.ASSET>0,fin.LNLSNET/fin.ASSET,np.nan)
    parts=[]
    for dt,g in fin.groupby("REPDTE"):
        g=g.copy(); g["A_EQ"]=rank01(g.EQ_ASSET,True); g["A_RBC"]=rank01(g.RBC1RWAJ,True)
        g["CAPITAL"]=g[["A_EQ","A_RBC"]].min(axis=1,skipna=False)
        g["ASSET_QUALITY"]=rank01(g.NCLNLSR,False); g["EARNINGS"]=rank01(g.ROA,True); g["LIQUIDITY"]=rank01(g.LTD,False)
        parts.append(g)
    panel=pd.concat(parts,ignore_index=True).sort_values(["CERT","REPDTE"])
    domains=["CAPITAL","ASSET_QUALITY","EARNINGS","LIQUIDITY"]
    panel["IMI"]=panel[domains].prod(axis=1,min_count=4); panel["WMI"]=panel[domains].min(axis=1,skipna=False); panel["ADD"]=panel[domains].mean(axis=1,skipna=False)
    panel.to_csv(OUT/"bank_quarter_panel.csv", index=False)

    failures["CERT"]=pd.to_numeric(failures.get("CERT"),errors="coerce").astype("Int64")
    failures["FAILDATE"]=pd.to_datetime(failures.get("FAILDATE").astype(str),errors="coerce")
    fmap=failures.dropna(subset=["CERT","FAILDATE"]).groupby("CERT")["FAILDATE"].min().to_dict()

    outcomes=[]
    for s in matched.itertuples():
        cert=int(s.CERT); pdte=pd.Timestamp(s.purchase_date); g=panel[panel.CERT==cert].copy()
        base=g[g.REPDTE<pdte].dropna(subset=["IMI"]).sort_values("REPDTE").tail(1)
        if base.empty: continue
        b=base.iloc[0]; p4=nearest_row(g.dropna(subset=["IMI"]), b.REPDTE+pd.offsets.QuarterEnd(4),1); p8=nearest_row(g.dropna(subset=["IMI"]), b.REPDTE+pd.offsets.QuarterEnd(8),1)
        fail=fmap.get(cert,pd.NaT)
        row={"CERT":cert,"NAME":getattr(s,"NAME",None),"STALP":getattr(s,"STALP",None),"purchase_date":pdte,"support_amount":s.amount,"baseline_date":b.REPDTE,"baseline_asset":b.ASSET,"baseline_IMI":b.IMI,"baseline_WMI":b.WMI,"baseline_ADD":b.ADD,"baseline_output":b.LOAN_OUTPUT}
        row["support_intensity"] = s.amount/(b.ASSET*1000.0) if pd.notna(s.amount) and pd.notna(b.ASSET) and b.ASSET>0 else np.nan
        for label,x in [("post4",p4),("post8",p8)]:
            if x is not None:
                row[f"{label}_date"]=x.REPDTE; row[f"{label}_IMI"]=x.IMI; row[f"{label}_WMI"]=x.WMI; row[f"{label}_ADD"]=x.ADD; row[f"{label}_output"]=x.LOAN_OUTPUT
                row[f"delta{label[-1]}_IMI"]=x.IMI-b.IMI; row[f"delta{label[-1]}_WMI"]=x.WMI-b.WMI; row[f"delta{label[-1]}_ADD"]=x.ADD-b.ADD
                survived = pd.isna(fail) or fail > x.REPDTE
                row[f"output_maintained_{label[-1]}q"] = int(survived and pd.notna(x.LOAN_OUTPUT) and pd.notna(b.LOAN_OUTPUT) and x.LOAN_OUTPUT >= .90*b.LOAN_OUTPUT)
            else:
                row[f"output_maintained_{label[-1]}q"] = np.nan
        d4=row.get("delta4_IMI",np.nan); d8=row.get("delta8_IMI",np.nan)
        row["restoration_candidate_4q"] = int(pd.notna(d4) and d4>=EPS)
        row["restoration_persistent_8q"] = int(pd.notna(d4) and pd.notna(d8) and d4>=EPS and d8>=EPS)
        row["supported_continuity_candidate_4q"] = int(row.get("output_maintained_4q")==1 and pd.notna(d4) and d4<EPS)
        row["supported_continuity_persistent_8q"] = int(row.get("output_maintained_8q")==1 and pd.notna(d8) and d8<EPS)
        row["digression_4q"] = int(pd.notna(d4) and d4<=-EPS)
        row["stable_intrinsic_4q"] = int(pd.notna(d4) and abs(d4)<EPS)
        row["failed_by_8q"] = int(pd.notna(fail) and p8 is not None and fail<=p8.REPDTE)
        outcomes.append(row)
    supp=pd.DataFrame(outcomes); supp.to_csv(OUT/"supported_bank_outcomes.csv",index=False)
    if len(supp)<20: raise RuntimeError(f"Insufficient supported bank outcome rows: {len(supp)}")

    # Deterministic exact-match comparison cohort by baseline quarter, size decile, IMI decile, state if available; relax state only if needed.
    support_certs=set(supp.CERT.astype(int)); controls=[]
    base_pool=[]
    for r in supp.itertuples():
        q=pd.Timestamp(r.baseline_date); gp=panel[(panel.REPDTE==q)&(~panel.CERT.astype(int).isin(support_certs))].dropna(subset=["IMI","ASSET"]).copy()
        if gp.empty: continue
        gp["size_decile"]=pd.qcut(gp.ASSET.rank(method="first"),10,labels=False,duplicates="drop"); gp["imi_decile"]=pd.qcut(gp.IMI.rank(method="first"),10,labels=False,duplicates="drop")
        target_size=int(pd.qcut(panel[panel.REPDTE==q].ASSET.rank(method="first"),10,labels=False,duplicates="drop").loc[panel[(panel.REPDTE==q)&(panel.CERT==r.CERT)].index[0]]) if len(panel[(panel.REPDTE==q)&(panel.CERT==r.CERT)]) else None
        target_imi=int(pd.qcut(panel[panel.REPDTE==q].IMI.rank(method="first"),10,labels=False,duplicates="drop").loc[panel[(panel.REPDTE==q)&(panel.CERT==r.CERT)].index[0]]) if len(panel[(panel.REPDTE==q)&(panel.CERT==r.CERT)]) else None
        cand=gp[(gp.size_decile==target_size)&(gp.imi_decile==target_imi)]
        same=cand[cand.STALP.astype(str)==str(r.STALP)] if "STALP" in cand.columns else pd.DataFrame()
        if len(same): cand=same
        if len(cand)==0: continue
        c=cand.assign(dist=(np.log1p(cand.ASSET)-math.log1p(r.baseline_asset))**2+(cand.IMI-r.baseline_IMI)**2).sort_values(["dist","CERT"]).iloc[0]
        cg=panel[panel.CERT==c.CERT].copy(); c4=nearest_row(cg.dropna(subset=["IMI"]), q+pd.offsets.QuarterEnd(4),1); c8=nearest_row(cg.dropna(subset=["IMI"]), q+pd.offsets.QuarterEnd(8),1)
        controls.append({"supported_CERT":r.CERT,"control_CERT":int(c.CERT),"baseline_date":q,"control_baseline_IMI":c.IMI,"control_delta4_IMI":(c4.IMI-c.IMI if c4 is not None else np.nan),"control_delta8_IMI":(c8.IMI-c.IMI if c8 is not None else np.nan)})
    ctrl=pd.DataFrame(controls); ctrl.to_csv(OUT/"control_pairs.csv",index=False)

    metrics={"status":"VALID_EXECUTION","protocol":"IMI_v3_TARP_CPP_SUPPORT_RESTORATION_PROTOCOL_v1","treasury_candidate_lines":int(len(ledger)),"treasury_matched_unique_banks":int(len(matched)),"supported_evaluable_banks":int(len(supp)),"control_pairs":int(len(ctrl))}
    for h in [4,8]:
        d=supp.get(f"delta{h}_IMI",pd.Series(dtype=float)).dropna(); metrics[f"delta{h}_IMI_n"]=int(len(d)); metrics[f"delta{h}_IMI_median"]=float(d.median()) if len(d) else None
    metrics["disposition_4q"]={k:int(supp[k].sum()) for k in ["restoration_candidate_4q","supported_continuity_candidate_4q","digression_4q","stable_intrinsic_4q"]}
    metrics["disposition_8q"]={k:int(supp[k].sum()) for k in ["restoration_persistent_8q","supported_continuity_persistent_8q","failed_by_8q"]}
    for h in [4,8]:
        sub=supp.dropna(subset=["support_intensity",f"delta{h}_IMI"])
        if len(sub)>=10:
            rho,p=spearmanr(sub.support_intensity,sub[f"delta{h}_IMI"]); metrics[f"support_intensity_rho_delta{h}"]={"n":int(len(sub)),"rho":float(rho),"p":float(p),"material":bool(abs(rho)>=.20 and p<.05)}
    maintained=supp[(supp.output_maintained_8q==1)&supp.delta8_IMI.notna()].copy()
    if len(maintained)>=10 and maintained.restoration_persistent_8q.nunique()==2:
        y=maintained.restoration_persistent_8q.to_numpy()
        for m in ["baseline_IMI","baseline_WMI","baseline_ADD"]:
            if y.sum()>=1 and (len(y)-y.sum())>=1: metrics[f"{m}_auroc_restoration8"] = float(roc_auc_score(y, maintained[m]))
    if len(ctrl):
        j=supp.merge(ctrl,left_on="CERT",right_on="supported_CERT")
        for h in [4,8]:
            svals=j[f"delta{h}_IMI"]; cvals=j[f"control_delta{h}_IMI"]
            ok=svals.notna()&cvals.notna(); diff=(svals[ok]-cvals[ok])
            metrics[f"paired_delta{h}_difference"]={"n":int(ok.sum()),"median_supported_minus_control":float(diff.median()) if ok.sum() else None}
            if ok.sum()>=20:
                metrics[f"unpaired_bootstrap_delta{h}"]={"difference_ci":bootstrap_median_diff(svals[ok],cvals[ok])}
    metrics["materiality"]={"both_4q_classes_ge20":bool(metrics["disposition_4q"]["restoration_candidate_4q"]>=20 and metrics["disposition_4q"]["supported_continuity_candidate_4q"]>=20),"both_8q_classes_ge20":bool(metrics["disposition_8q"]["restoration_persistent_8q"]>=20 and metrics["disposition_8q"]["supported_continuity_persistent_8q"]>=20)}
    metrics["claim_boundary"]="Retrospective support-separated disposition study. CPP receipt is independently observed; no causal TARP effect claim. Post-exit intrinsic restoration is not authorized unless official exit dates are independently recovered."
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2,default=str))
    (OUT/"run_receipt.json").write_text(json.dumps({"protocol":metrics["protocol"],"tarp_pdf_sha256":sha256(pdf_path),"match_ledger_sha256":sha256(OUT/"tarp_match_ledger.csv"),"supported_outcomes_sha256":sha256(OUT/"supported_bank_outcomes.csv"),"metrics_sha256":sha256(OUT/"metrics.json")},indent=2))
    (OUT/"run_log.txt").write_text("Completed frozen IMI v3 TARP CPP support/restoration study.\n")
    print(json.dumps(metrics,indent=2,default=str))

if __name__ == "__main__":
    main()
