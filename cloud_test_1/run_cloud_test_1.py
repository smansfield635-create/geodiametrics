from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 256
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://api.worldbank.org/v2"
YEARS = [2017, 2018, 2019, 2020, 2021]
IND = {
    "unemp": "SL.UEM.TOTL.ZS",
    "infl": "FP.CPI.TOTL.ZG",
    "debt": "GC.DOD.TOTL.GD.ZS",
    "tourism": "ST.INT.RCPT.XP.ZS",
    "trade": "NE.TRD.GNFS.ZS",
    "reserves": "FI.RES.TOTL.MO",
    "savings": "NY.GNS.ICTR.ZS",
    "capital": "NE.GDI.TOTL.ZS",
    "spi": "IQ.SPI.OVRL",
    "sci": "IQ.SCI.OVRL",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "employment": "SL.EMP.TOTL.SP.ZS",
}


def get_json(url: str):
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.json()


def country_set() -> set[str]:
    data = get_json(f"{BASE}/country?format=json&per_page=400")
    rows = data[1]
    return {r["id"] for r in rows if r.get("region", {}).get("id")}


def fetch_indicator(code: str) -> pd.DataFrame:
    url = f"{BASE}/country/all/indicator/{code}?date=2017:2021&format=json&per_page=20000"
    data = get_json(url)
    rows = data[1] or []
    out = []
    for r in rows:
        if r.get("value") is None:
            continue
        out.append({"country": r["countryiso3code"], "year": int(r["date"]), "value": float(r["value"])})
    df = pd.DataFrame(out)
    df["indicator"] = code
    return df


def pct_rank(s: pd.Series, higher=True) -> pd.Series:
    r = s.rank(method="average", pct=True)
    return r if higher else 1 - r


def hash_split(country: str) -> str:
    x = int(hashlib.sha256(f"{SEED}:{country}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "development" if x < 0.75 else "evaluation"


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def score(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, int)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "auroc": safe_auc(y, p),
        "average_precision": float(average_precision_score(y, p)),
    }


def fit_logit(train, test, features, outcome):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=SEED, class_weight="balanced")),
    ])
    pipe.fit(train[features], train[outcome])
    return pipe.predict_proba(test[features])[:, 1]


def main():
    countries = country_set()
    frames, registry = [], []
    for name, code in IND.items():
        df = fetch_indicator(code)
        df = df[df["country"].isin(countries)].copy()
        frames.append(df.assign(name=name))
        registry.append({"name": name, "indicator": code, "rows": int(len(df)), "countries": int(df.country.nunique())})
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(OUT / "wdi_raw_long.csv", index=False)
    (OUT / "source_registry.json").write_text(json.dumps(registry, indent=2))

    pivot = raw.pivot_table(index=["country", "year"], columns="name", values="value", aggfunc="first").reset_index()
    # Coverage-only prespecified I fallback.
    spi_cov = pivot[pivot.year.isin([2017, 2018, 2019])].groupby("country")["spi"].apply(lambda s: s.notna().all()).sum() if "spi" in pivot else 0
    i_name = "spi" if spi_cov >= 40 else "sci"

    pre = pivot[pivot.year.isin([2017, 2018, 2019])].groupby("country").mean(numeric_only=True).reset_index()
    shock = pivot[pivot.year == 2020].set_index("country")
    rec = pivot[pivot.year == 2021].set_index("country")
    required = ["unemp", "infl", "debt", "tourism", "trade", "reserves", "savings", "capital", i_name, "gdp_growth", "employment"]
    eligible = pre.dropna(subset=required).copy()
    eligible = eligible[eligible.country.isin(shock.index) & eligible.country.isin(rec.index)].copy()
    eligible = eligible[shock.loc[eligible.country, "gdp_growth"].notna().to_numpy() & rec.loc[eligible.country, "gdp_growth"].notna().to_numpy()].copy()

    # Domain scores are percentile availabilities after complete-route filter.
    eligible["B"] = pd.concat([
        pct_rank(eligible["unemp"], higher=False),
        pct_rank(eligible["infl"].abs(), higher=False),
        pct_rank(eligible["debt"], higher=False),
    ], axis=1).mean(axis=1)
    eligible["P"] = pd.concat([
        pct_rank(eligible["tourism"], higher=False),
        pct_rank(eligible["trade"], higher=False),
    ], axis=1).mean(axis=1)
    eligible["E"] = pd.concat([
        pct_rank(eligible["reserves"], higher=True),
        pct_rank(eligible["savings"], higher=True),
        pct_rank(eligible["capital"], higher=True),
    ], axis=1).mean(axis=1)
    eligible["I"] = pct_rank(eligible[i_name], higher=True)
    eligible["V"] = pd.concat([
        pct_rank(eligible["gdp_growth"], higher=True),
        pct_rank(eligible["employment"], higher=True),
    ], axis=1).mean(axis=1)
    eligible["IMI"] = eligible["E"] * eligible["I"] * eligible["V"]
    eligible["WMI"] = eligible[["E", "I", "V"]].min(axis=1)
    eligible["ADD"] = eligible[["E", "I", "V"]].mean(axis=1)
    eligible["CS"] = 1 - eligible["IMI"]
    eligible["split"] = eligible.country.map(hash_split)
    eligible["gdp_2020"] = shock.loc[eligible.country, "gdp_growth"].to_numpy()
    eligible["gdp_2021"] = rec.loc[eligible.country, "gdp_growth"].to_numpy()
    eligible["contraction"] = (eligible.gdp_2020 < 0).astype(int)
    eligible["severe"] = (eligible.gdp_2020 <= -5).astype(int)
    eligible["recovered"] = ((eligible.gdp_2020 < 0) & (eligible.gdp_2021 > 0)).astype(int)
    eligible["recovery_magnitude"] = eligible.gdp_2021 - eligible.gdp_2020
    eligible["IMI_decile"] = pd.qcut(eligible.IMI.rank(method="first"), 10, labels=False) + 1
    eligible["WMI_decile"] = pd.qcut(eligible.WMI.rank(method="first"), 10, labels=False) + 1
    eligible.to_csv(OUT / "macro_panel.csv", index=False)

    structural = []
    for outcome in ["contraction", "severe"]:
        for metric in ["IMI", "WMI"]:
            dec = eligible[f"{metric}_decile"]
            low = eligible[dec == 1]
            rest = eligible[dec != 1]
            rr = (low[outcome].mean() / rest[outcome].mean()) if rest[outcome].mean() > 0 else None
            structural.append({
                "outcome": outcome, "metric": metric,
                "low_decile_n": int(len(low)), "low_decile_events": int(low[outcome].sum()),
                "low_decile_rate": float(low[outcome].mean()),
                "rest_n": int(len(rest)), "rest_events": int(rest[outcome].sum()),
                "rest_rate": float(rest[outcome].mean()),
                "risk_ratio": None if rr is None else float(rr),
            })
    pd.DataFrame(structural).to_csv(OUT / "structural_concentration.csv", index=False)

    train = eligible[eligible.split == "development"].copy()
    test = eligible[eligible.split == "evaluation"].copy()
    model_results = {}
    for outcome in ["contraction", "severe"]:
        model_results[outcome] = {}
        # Single operator discrimination: adverse score = 1 - availability.
        for m in ["IMI", "WMI", "ADD", "B", "P"]:
            p = 1 - test[m].to_numpy() if m in ["IMI", "WMI", "ADD", "B", "P"] else test[m].to_numpy()
            # B/P are already availability (low burden/exposure=high), so 1-x is adverse.
            model_results[outcome][m] = score(test[outcome], p)
        comparisons = {
            "BP": ["B", "P"],
            "RAW": ["B", "P", "E", "I", "V"],
            "BP_IMI": ["B", "P", "IMI"],
            "BP_WMI": ["B", "P", "WMI"],
        }
        for name, feats in comparisons.items():
            p = fit_logit(train, test, feats, outcome)
            model_results[outcome][name] = score(test[outcome], p)

    # Recovery among countries that contracted in 2020.
    rc = eligible[eligible.contraction == 1].copy()
    rtrain = rc[rc.split == "development"]
    rtest = rc[rc.split == "evaluation"]
    recovery = {"n": int(len(rc)), "events": int(rc.recovered.sum()), "holdout_n": int(len(rtest))}
    if len(rtest) >= 5 and rtrain.recovered.nunique() == 2 and rtest.recovered.nunique() == 2:
        for m in ["IMI", "WMI", "ADD"]:
            recovery[m] = score(rtest.recovered, rtest[m])
        for name, feats in {"BP": ["B", "P"], "BP_IMI": ["B", "P", "IMI"], "BP_WMI": ["B", "P", "WMI"]}.items():
            recovery[name] = score(rtest.recovered, fit_logit(rtrain, rtest, feats, "recovered"))
    # Magnitude association across all contracted countries.
    recovery["corr_IMI_magnitude"] = float(rc.IMI.corr(rc.recovery_magnitude, method="spearman")) if len(rc) else None
    recovery["corr_WMI_magnitude"] = float(rc.WMI.corr(rc.recovery_magnitude, method="spearman")) if len(rc) else None

    results = {
        "status": "VALID_EXECUTION",
        "i_indicator_used": IND[i_name],
        "spi_complete_pre_shock_country_count": int(spi_cov),
        "eligible_countries": int(len(eligible)),
        "development_countries": int(len(train)),
        "evaluation_countries": int(len(test)),
        "contraction_events": int(eligible.contraction.sum()),
        "severe_events": int(eligible.severe.sum()),
        "recovery_events": int(eligible.recovered.sum()),
        "structural_concentration": structural,
        "holdout_models": model_results,
        "recovery": recovery,
        "claim_boundary": "Retrospective country-level common-shock realization; association/discrimination only; no causal or supported-continuity claim.",
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    (OUT / "run_receipt.json").write_text(json.dumps({
        "seed": SEED,
        "protocol": "IMI_v3_GLOBAL_MACRO_COMMON_SHOCK_REALIZATION_PROTOCOL_v1",
        "panel_sha256": hashlib.sha256((OUT / "macro_panel.csv").read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256((OUT / "metrics.json").read_bytes()).hexdigest(),
    }, indent=2))
    (OUT / "run_log.txt").write_text("Completed frozen IMI v3 global macro common-shock realization.\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
