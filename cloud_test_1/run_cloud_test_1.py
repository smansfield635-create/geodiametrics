from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 256
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
README = "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/README.md"
EXPECTED = {
    "pai_job_table.tar.gz": "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
    "pai_task_table.tar.gz": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
}
JOB_COLUMNS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
TASK_COLUMNS = [
    "job_name", "task_name", "inst_num", "status", "start_time", "end_time",
    "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_links() -> dict[str, str]:
    r = requests.get(README, timeout=60)
    r.raise_for_status()
    text = r.text
    links = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", text)
    links += re.findall(r'https?://[^\s<>"\)]+', text)
    out: dict[str, str] = {}
    for url in links:
        clean = url.rstrip(".,)")
        for fn in EXPECTED:
            if fn in clean:
                out.setdefault(fn, clean)
    missing = sorted(set(EXPECTED) - set(out))
    if missing:
        raise RuntimeError(f"Official README links not resolved: {missing}")
    return out


def download(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def ecdf(train: pd.Series, values: pd.Series) -> pd.Series:
    tr = np.sort(pd.Series(train).dropna().to_numpy(float))
    if len(tr) == 0:
        return pd.Series(np.nan, index=values.index)
    return values.map(lambda x: np.nan if pd.isna(x) else np.searchsorted(tr, x, side="right") / len(tr))


def score_binary(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def score_prob(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }


def main() -> None:
    links = resolve_links()
    registry = []
    for fn, expected in EXPECTED.items():
        p = RAW / fn
        download(links[fn], p)
        actual = sha256(p)
        registry.append({"file": fn, "url": links[fn], "bytes": p.stat().st_size,
                         "sha256": actual, "expected_sha256": expected,
                         "checksum_pass": actual == expected})
        if actual != expected:
            raise RuntimeError(f"Checksum failed for {fn}")
        with tarfile.open(p, "r:gz") as tar:
            tar.extractall(RAW)
    (OUT / "source_registry.json").write_text(json.dumps(registry, indent=2))

    jobs = pd.read_csv(RAW / "pai_job_table.csv", header=None, names=JOB_COLUMNS)
    tasks = pd.read_csv(RAW / "pai_task_table.csv", header=None, names=TASK_COLUMNS)
    jobs = jobs[jobs["status"].isin(["Failed", "Terminated"])].copy()
    jobs["outcome"] = (jobs["status"] == "Failed").astype(int)
    jobs["split"] = jobs["user"].astype(str).map(
        lambda u: "development" if int(hashlib.sha256(f"{SEED}:{u}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.75 else "evaluation"
    )

    t = tasks.copy()
    required = ["job_name", "task_name", "inst_num", "start_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
    t["field_complete"] = t[required].notna().mean(axis=1)
    agg = t.groupby("job_name").agg(
        task_count=("task_name", "size"), inst_count=("inst_num", "sum"),
        total_cpu=("plan_cpu", "sum"), total_mem=("plan_mem", "sum"), total_gpu=("plan_gpu", "sum"),
        peak_cpu=("plan_cpu", "max"), peak_mem=("plan_mem", "max"), peak_gpu=("plan_gpu", "max"),
        first_task_start=("start_time", "min"),
        task_launch_coverage=("start_time", lambda x: x.notna().mean()),
        field_complete=("field_complete", "mean"),
        gpu_type_mode=("gpu_type", lambda x: x.dropna().astype(str).mode().iloc[0] if len(x.dropna()) else "MISSING"),
    ).reset_index()

    d = jobs.merge(agg, on="job_name", how="left")
    d["wait_time"] = d["first_task_start"] - d["start_time"]
    d["ordering_consistent"] = ((d["wait_time"] >= 0) | d["wait_time"].isna()).astype(float)
    d = d.sort_values("start_time").reset_index(drop=True)
    times = d["start_time"].to_numpy(float)
    left = np.searchsorted(times, times - 300, side="left")
    d["cluster_launch_density_5m"] = np.arange(len(d)) - left
    d["user_launch_density_5m"] = 0
    for _, idx in d.groupby("user").groups.items():
        idx = np.array(sorted(idx))
        ut = d.loc[idx, "start_time"].to_numpy(float)
        ul = np.searchsorted(ut, ut - 300, side="left")
        d.loc[idx, "user_launch_density_5m"] = np.arange(len(idx)) - ul

    dev = d[d["split"] == "development"].copy()
    freq = dev["gpu_type_mode"].value_counts(normalize=True)
    d["gpu_scarcity"] = 1 - d["gpu_type_mode"].map(freq).fillna(0)
    B_cols = ["total_cpu", "total_mem", "total_gpu", "task_count", "inst_count"]
    P_cols = ["wait_time", "cluster_launch_density_5m", "user_launch_density_5m", "gpu_scarcity"]
    E_cols = ["peak_cpu", "peak_mem", "peak_gpu"]
    ranked = {c: ecdf(dev[c], d[c]) for c in B_cols + P_cols + E_cols}
    r = pd.DataFrame(ranked, index=d.index)
    d["B"] = r[B_cols].mean(axis=1, skipna=False)
    d["P"] = r[P_cols].mean(axis=1, skipna=False)
    d["E"] = 1 - r[E_cols].mean(axis=1, skipna=False)
    d["I"] = d[["field_complete", "ordering_consistent"]].mean(axis=1, skipna=False)
    wait90 = dev["wait_time"].quantile(0.90)
    d["V"] = d["task_launch_coverage"] * (d["wait_time"] <= wait90).astype(float)
    d["W"] = d[["E", "I", "V"]].min(axis=1, skipna=False)
    d["coverage"] = d[["B", "P", "E", "I", "V"]].notna().mean(axis=1)

    dev2 = d[(d["split"] == "development") & (d["coverage"] == 1)]
    bq, pq, eps = dev2["B"].quantile(.75), dev2["P"].quantile(.75), dev2["W"].quantile(.25)
    d["B_norm"] = d["B"] / bq
    d["P_norm"] = d["P"] / pq
    d["Pi"] = d["B_norm"] * d["P_norm"]
    d["K"] = (d["E"] * d["I"] * d["V"] * d["coverage"]).clip(lower=.05)
    d["PCR"] = d["Pi"] / d["K"]
    d["H_star"] = d["PCR"] / (1 + d["PCR"])
    d["MQ"] = ((d["B_norm"] >= 1) & (d["P_norm"] >= 1) & (d["W"] <= eps)).astype(int)

    evaluable = d[d["coverage"] == 1].copy()
    train = evaluable[evaluable["split"] == "development"]
    test = evaluable[evaluable["split"] == "evaluation"]
    y = test["outcome"].to_numpy()
    features = ["B", "P", "E", "I", "V"]

    logit = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"))])
    logit.fit(train[features], train["outcome"])
    p_logit = logit.predict_proba(test[features])[:, 1]
    gb = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("model", HistGradientBoostingClassifier(random_state=SEED, max_iter=200))])
    gb.fit(train[features], train["outcome"])
    p_gb = gb.predict_proba(test[features])[:, 1]

    gates = {
        "B_only": (test["B_norm"] >= 1).astype(int),
        "P_only": (test["P_norm"] >= 1).astype(int),
        "W_only": (test["W"] <= eps).astype(int),
        "B_P": ((test["B_norm"] >= 1) & (test["P_norm"] >= 1)).astype(int),
        "B_W": ((test["B_norm"] >= 1) & (test["W"] <= eps)).astype(int),
        "P_W": ((test["P_norm"] >= 1) & (test["W"] <= eps)).astype(int),
        "full_MQ": test["MQ"].astype(int),
    }
    ablations = {k: score_binary(y, v.to_numpy()) for k, v in gates.items()}
    results = {
        "artifact_id": "CLOUD_TEST_1_ALIBABA_GPU_2020_EMPIRICAL_RETURN_v1",
        "status": "COMPLETE",
        "thresholds": {"B_q75": float(bq), "P_q75": float(pq), "epsilon_d": float(eps)},
        "coverage": {"raw_jobs": int(len(jobs)), "evaluable_jobs": int(len(evaluable)),
                     "development_jobs": int(len(train)), "evaluation_jobs": int(len(test)),
                     "evaluation_failure_prevalence": float(test["outcome"].mean())},
        "hard_MQ": score_binary(y, test["MQ"].to_numpy()),
        "continuous_H_star": score_prob(y, test["H_star"].to_numpy()),
        "logistic": score_prob(y, p_logit),
        "gradient_boosting": score_prob(y, p_gb),
        "components_AUROC": {
            "B": float(roc_auc_score(y, test["B"])), "P": float(roc_auc_score(y, test["P"])),
            "one_minus_E": float(roc_auc_score(y, 1 - test["E"])),
            "one_minus_I": float(roc_auc_score(y, 1 - test["I"])),
            "one_minus_V": float(roc_auc_score(y, 1 - test["V"])),
            "one_minus_W": float(roc_auc_score(y, 1 - test["W"])),
            "one_minus_mean_EIV": float(roc_auc_score(y, 1 - test[["E", "I", "V"]].mean(axis=1))),
        },
        "ablations": ablations,
        "claim_boundary": "Launch-time held-out association/discrimination only; not a universal or full telemetry validation.",
    }

    d.to_csv(OUT / "panel.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame([results["hard_MQ"]]).to_csv(OUT / "confusion_matrix.csv", index=False)
    pd.DataFrame(ablations).T.to_csv(OUT / "ablations.csv")
    pd.DataFrame({"exclusion": ["incomplete_predictor_set"], "count": [int((d["coverage"] < 1).sum())]}).to_csv(OUT / "exclusions.csv", index=False)
    receipt = {"seed": SEED, "source_registry_sha256": sha256(OUT / "source_registry.json"),
               "panel_sha256": sha256(OUT / "panel.csv"), "metrics_sha256": sha256(OUT / "metrics.json")}
    (OUT / "run_receipt.json").write_text(json.dumps(receipt, indent=2))
    (OUT / "run_log.txt").write_text("Completed frozen Cloud Test 1 launch-time run.\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
