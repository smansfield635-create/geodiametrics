from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ARCHIVE = "https://raw.githubusercontent.com/klocey/hospitals-data-archive/main/dataframes/filtered_files/Hospital_Readmissions_Reduction_Program/Hospital_Readmissions_Reduction_Program_{year}.csv"
YEARS = list(range(2013, 2024))
EXCLUDED_DUPLICATE_RELEASE_YEARS = {2017}
CONDS = ["READM-30-AMI", "READM-30-CABG", "READM-30-COPD", "READM-30-HF", "READM-30-HIP-KNEE", "READM-30-PN"]


def norm(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def fid_from_hospital(x: object) -> str | None:
    m = re.search(r"\((\d{6})\)\s*$", str(x))
    return m.group(1) if m else None


def canonical_measure(x: object) -> str:
    s = str(x).upper().strip()
    for c in CONDS:
        if c in s:
            return c
    return s


def dedupe_facility_year(out: pd.DataFrame):
    before = len(out)
    out = out.dropna(subset=["facility_id"]).copy()
    # Source-QC Amendment B: identical duplicate facility-year route rows collapse to one.
    out = out.drop_duplicates(subset=["facility_id"] + CONDS, keep="first")
    return out, before - len(out)


def load_year(year: int):
    url = ARCHIVE.format(year=year)
    r = requests.get(url, timeout=90)
    if r.status_code == 404:
        return None, {"year": year, "status": "NOT_FOUND"}
    r.raise_for_status()
    raw_path = OUT / f"HRRP_{year}.csv"
    raw_path.write_bytes(r.content)
    df = pd.read_csv(raw_path, low_memory=False)
    original_cols = list(df.columns)
    d = df.rename(columns={c: norm(c) for c in df.columns})

    wide_map = {}
    for cond in CONDS:
        target = norm(cond + " (Excess Readmission Ratio)")
        if target in d.columns:
            wide_map[cond] = target
    if len(wide_map) == len(CONDS):
        if "facility_id" in d.columns:
            fid = d["facility_id"].astype(str).str.extract(r"(\d{6})", expand=False)
        elif "hospital" in d.columns:
            fid = d["hospital"].map(fid_from_hospital)
        else:
            cands = [c for c in d.columns if "facility" in c and "id" in c]
            fid = d[cands[0]].astype(str).str.extract(r"(\d{6})", expand=False) if cands else pd.Series([None] * len(d))
        out = pd.DataFrame({"facility_id": fid, "year": year})
        for cond, col in wide_map.items():
            out[cond] = pd.to_numeric(d[col], errors="coerce")
        out, removed = dedupe_facility_year(out)
        return out, {"year": year, "status": "OK_WIDE", "raw_rows": int(len(df)), "canonical_rows": int(len(out)), "duplicate_rows_removed": int(removed), "cols": int(len(original_cols)), "excluded_as_duplicate_release": year in EXCLUDED_DUPLICATE_RELEASE_YEARS}

    mcands = [c for c in d.columns if c in ("measure_name", "measure_id", "measure")]
    ecands = [c for c in d.columns if c == "excess_readmission_ratio" or ("excess" in c and "readmission" in c and "ratio" in c)]
    fcands = [c for c in d.columns if c == "facility_id" or ("facility" in c and "id" in c)]
    if not (mcands and ecands and fcands):
        return None, {"year": year, "status": "UNRECOGNIZED_SCHEMA", "columns": original_cols[:60]}
    mcol, ecol, fcol = mcands[0], ecands[0], fcands[0]
    tmp = d[[fcol, mcol, ecol]].copy()
    tmp["facility_id"] = tmp[fcol].astype(str).str.extract(r"(\d{6})", expand=False)
    tmp["measure"] = tmp[mcol].map(canonical_measure)
    tmp["err"] = pd.to_numeric(tmp[ecol], errors="coerce")
    tmp = tmp[tmp["measure"].isin(CONDS)]
    wide = tmp.pivot_table(index="facility_id", columns="measure", values="err", aggfunc="first").reset_index()
    wide["year"] = year
    for cond in CONDS:
        if cond not in wide.columns:
            wide[cond] = np.nan
    out = wide[["facility_id", "year"] + CONDS]
    out, removed = dedupe_facility_year(out)
    return out, {"year": year, "status": "OK_LONG", "raw_rows": int(len(df)), "canonical_rows": int(len(out)), "duplicate_rows_removed": int(removed), "matched_rows": int(len(tmp)), "excluded_as_duplicate_release": year in EXCLUDED_DUPLICATE_RELEASE_YEARS}


def availability(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return (1.0 / x).where(x > 0).clip(lower=0.0, upper=1.0)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    tr = train.dropna(subset=features + ["Y_next"])
    te = test.dropna(subset=features + ["Y_next"])
    if len(tr) < max(50, 10 * len(features)) or len(te) == 0:
        return None
    X = np.c_[np.ones(len(tr)), tr[features].to_numpy(float)]
    y = tr["Y_next"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt = np.c_[np.ones(len(te)), te[features].to_numpy(float)]
    pred = Xt @ beta
    actual = te["Y_next"].to_numpy(float)
    e = actual - pred
    return te, pred, actual, float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def make_pairs(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future = panel[["facility_id", "year", "Y"]].rename(columns={"year": "future_year", "Y": "Y_next"})
    cur = panel.copy()
    cur["future_year"] = cur["year"] + horizon
    p = cur.merge(future, on=["facility_id", "future_year"], how="inner", validate="one_to_one")
    p["dY"] = p["Y_next"] - p["Y"]
    return p


def rolling(pairs: pd.DataFrame, horizon: int):
    models = {
        "BASE0": ["Y"],
        "ADD": ["Y", "ADD"],
        "WMI": ["Y", "WMI"],
        "IMI": ["Y", "IMI"],
        "TRAJ": ["Y", "IMI", "Delta_IMI"],
    }
    metrics, preds = [], []
    for target_year in sorted(pairs["future_year"].dropna().astype(int).unique()):
        train = pairs[pairs["future_year"] < target_year]
        test = pairs[pairs["future_year"] == target_year]
        if train["year"].nunique() < 2:
            continue
        for model, feats in models.items():
            z = fit_predict(train, test, feats)
            if z is None:
                continue
            te, pred, actual, rmse, mae = z
            n_train = len(train.dropna(subset=feats + ["Y_next"]))
            metrics.append({"horizon": horizon, "target_year": int(target_year), "model": model, "n_train": int(n_train), "n_test": int(len(te)), "rmse": rmse, "mae": mae})
            for (_, r), p, a in zip(te.iterrows(), pred, actual):
                preds.append({"horizon": horizon, "target_year": int(target_year), "model": model, "facility_id": r["facility_id"], "source_year": int(r["year"]), "Y": float(r["Y"]), "Y_next": float(a), "pred": float(p), "abs_resid": float(abs(a-p)), "IMI": float(r["IMI"]), "WMI": float(r["WMI"]), "Delta_IMI": float(r["Delta_IMI"]) if pd.notna(r["Delta_IMI"]) else np.nan})
    return pd.DataFrame(metrics), pd.DataFrame(preds)


def pooled(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, model), g in preds.groupby(["horizon", "model"]):
        e = g["Y_next"] - g["pred"]
        rows.append({"horizon": int(h), "model": model, "n": int(len(g)), "rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e)))})
    return pd.DataFrame(rows)


def rank_corr(x: pd.Series, y: pd.Series):
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 20:
        return None
    rx = z["x"].rank(method="average").to_numpy(float)
    ry = z["y"].rank(method="average").to_numpy(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return {"rho": float(np.corrcoef(rx, ry)[0, 1]), "n": int(len(z))}


def main():
    receipts, pieces = [], []
    for year in YEARS:
        try:
            df, rec = load_year(year)
        except Exception as exc:
            df, rec = None, {"year": year, "status": "ERROR", "error": repr(exc)}
        receipts.append(rec)
        if df is not None and year not in EXCLUDED_DUPLICATE_RELEASE_YEARS:
            pieces.append(df)
    pd.DataFrame(receipts).to_csv(OUT / "source_receipts.csv", index=False)
    if not pieces:
        raise RuntimeError("No historical HRRP source loaded")

    raw = pd.concat(pieces, ignore_index=True)
    raw = raw.drop_duplicates(subset=["facility_id", "year"], keep="first")
    for cond in CONDS:
        raw[cond] = pd.to_numeric(raw[cond], errors="coerce")
        raw["a_" + cond] = availability(raw[cond])
    acols = ["a_" + c for c in CONDS]
    raw["evaluable"] = raw[acols].notna().all(axis=1)
    panel = raw[raw["evaluable"]].copy()
    panel["IMI"] = panel[acols].prod(axis=1)
    panel["CS"] = 1.0 - panel["IMI"]
    panel["WMI"] = panel[acols].min(axis=1)
    panel["ADD"] = panel[acols].mean(axis=1)
    panel["Y"] = panel[CONDS].mean(axis=1)
    panel = panel.sort_values(["facility_id", "year"])

    # Exact-one-year first derivative; no bridge across excluded duplicate-release gap.
    prev = panel[["facility_id", "year", "IMI"]].copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"IMI": "IMI_prev"})
    panel = panel.merge(prev, on=["facility_id", "year"], how="left", validate="one_to_one")
    panel["Delta_IMI"] = panel["IMI"] - panel["IMI_prev"]
    panel.to_csv(OUT / "evaluable_panel.csv", index=False)
    panel.groupby("year").agg(n=("facility_id", "size"), mean_IMI=("IMI", "mean"), median_IMI=("IMI", "median"), mean_Y=("Y", "mean"), delta_n=("Delta_IMI", "count")).reset_index().to_csv(OUT / "year_summary.csv", index=False)

    holdouts, allpred, pair_summary = [], [], []
    for h in (1, 3):
        pairs = make_pairs(panel, h)
        pairs.to_csv(OUT / f"pairs_h{h}.csv", index=False)
        pair_summary.append({"horizon": h, "pairs": int(len(pairs)), "source_years": int(pairs["year"].nunique()), "target_years": int(pairs["future_year"].nunique()), "delta_pairs": int(pairs["Delta_IMI"].notna().sum())})
        hm, hp = rolling(pairs, h)
        holdouts.append(hm); allpred.append(hp)
    hold = pd.concat(holdouts, ignore_index=True) if holdouts else pd.DataFrame()
    predictions = pd.concat(allpred, ignore_index=True) if allpred else pd.DataFrame()
    hold.to_csv(OUT / "holdout_metrics.csv", index=False)
    predictions.to_csv(OUT / "holdout_predictions.csv", index=False)
    pool = pooled(predictions) if len(predictions) else pd.DataFrame()
    pool.to_csv(OUT / "pooled_metrics.csv", index=False)

    findings = {
        "source_integrity_amendment": {"excluded_duplicate_release_years": sorted(EXCLUDED_DUPLICATE_RELEASE_YEARS), "within_year_identical_duplicates_removed": True, "delta_requires_exact_previous_year": True},
        "route_conditions": CONDS,
        "years_requested": YEARS,
        "source_receipts": receipts,
        "pair_summary": pair_summary,
        "horizons": {},
    }
    for h in (1, 3):
        ph = pool[pool["horizon"] == h] if len(pool) else pd.DataFrame()
        hh = hold[hold["horizon"] == h] if len(hold) else pd.DataFrame()
        if len(ph) == 0:
            findings["horizons"][str(h)] = {"status": "NO_EVALUATION"}
            continue
        pm = {r["model"]: r for _, r in ph.iterrows()}
        candidates = [m for m in ("BASE0", "ADD", "WMI", "IMI") if m in pm]
        best = min(candidates, key=lambda m: pm[m]["rmse"])
        traj = pm.get("TRAJ")
        rel = None; wins = None; win_prop = None; hold_count = None
        if traj is not None:
            rel = (pm[best]["rmse"] - traj["rmse"]) / pm[best]["rmse"]
            q = hh[hh["model"].isin([best, "TRAJ"])].pivot(index="target_year", columns="model", values="rmse").dropna()
            hold_count = int(len(q)); wins = int((q["TRAJ"] < q[best]).sum()); win_prop = wins / hold_count if hold_count else None
        pr = predictions[(predictions["horizon"] == h) & (predictions["model"] == best)]
        assoc = {v: rank_corr(pr[v], pr["abs_resid"]) for v in ("IMI", "WMI", "Delta_IMI")}
        findings["horizons"][str(h)] = {
            "pooled": {m: {"n": int(pm[m]["n"]), "rmse": float(pm[m]["rmse"]), "mae": float(pm[m]["mae"])} for m in pm},
            "best_nontrajectory": best,
            "trajectory_relative_rmse_improvement": float(rel) if rel is not None else None,
            "trajectory_holdout_wins": wins,
            "trajectory_holdout_count": hold_count,
            "trajectory_win_proportion": win_prop,
            "material_rule_pass": bool(rel is not None and rel >= 0.10 and win_prop is not None and win_prop >= 0.70),
            "predictability_loss_rank_correlation_with_best_nontrajectory_abs_resid": assoc,
        }
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
