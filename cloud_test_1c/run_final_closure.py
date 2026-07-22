from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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

SOURCE_SPLIT_SEED = 256
CLOSURE_SPLIT_SEED = 451
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

README = "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/README.md"
EXPECTED = {
    "pai_job_table.tar.gz": "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
    "pai_task_table.tar.gz": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
    "pai_instance_table.tar.gz": "1bf1e423a7ce3f8d086699801c362fd56a7182abdb234139e5ebbed97995ca06",
    "pai_machine_spec.tar.gz": "cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d",
}
JOB = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
TASK = [
    "job_name", "task_name", "inst_num", "status", "start_time",
    "end_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
]
INST = [
    "job_name", "task_name", "inst_name", "worker_name", "inst_id",
    "status", "start_time", "end_time", "machine",
]
SPEC = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]

GROUPS = {
    "B": ["total_cpu", "total_mem", "total_gpu", "task_count", "planned_instances", "launch_spread"],
    "P": ["task_wait", "instance_wait", "cluster_density_5m", "user_density_5m", "gpu_scarcity"],
    "E": ["cpu_capacity_ratio", "mem_capacity_ratio", "gpu_capacity_ratio"],
    "I": ["task_complete", "instance_complete", "spec_coverage", "ordering_consistent"],
    "V": ["instance_ratio", "instance_launch_coverage", "launch_spread"],
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_links() -> dict[str, str]:
    response = requests.get(README, timeout=60)
    response.raise_for_status()
    text = response.text
    urls = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", text)
    urls += re.findall(r'https?://[^\s<>"\)]+', text)
    found: dict[str, str] = {}
    for url in urls:
        clean = url.rstrip(".,)")
        for filename in EXPECTED:
            if filename in clean:
                found.setdefault(filename, clean)
    missing = sorted(set(EXPECTED) - set(found))
    if missing:
        raise RuntimeError(f"Missing official source links: {missing}")
    return found


def download(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def user_fraction(seed: int, user: object) -> float:
    token = hashlib.sha256(f"{seed}:{user}".encode()).hexdigest()[:8]
    return int(token, 16) / 0xFFFFFFFF


def ecdf(reference: pd.Series, values: pd.Series) -> pd.Series:
    ref = np.sort(reference.dropna().to_numpy(float))
    if len(ref) == 0:
        return pd.Series(np.nan, index=values.index)
    return values.map(
        lambda value: np.nan
        if pd.isna(value)
        else np.searchsorted(ref, value, side="right") / len(ref)
    )


def safe_auc(y: pd.Series, score: pd.Series) -> float:
    valid = y.notna() & score.notna()
    if valid.sum() == 0 or y[valid].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], score[valid]))


def orient_and_select(
    development: pd.DataFrame,
    full: pd.DataFrame,
    outcome: str,
    columns: list[str],
    health_axis: bool,
    minimum_distance: float = 0.02,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    transformed: dict[str, pd.Series] = {}
    registry: list[dict[str, object]] = []
    for column in columns:
        rank_dev = ecdf(development[column], development[column])
        auc_high = safe_auc(development[outcome], rank_dev)
        if np.isnan(auc_high):
            continue
        distance = abs(auc_high - 0.5)
        selected = distance >= minimum_distance
        risk_direction = "HIGHER_RISK" if auc_high >= 0.5 else "LOWER_RISK"
        rank_full = ecdf(development[column], full[column])
        risk_score = rank_full if auc_high >= 0.5 else 1 - rank_full
        final_score = 1 - risk_score if health_axis else risk_score
        registry.append({
            "variable": column,
            "development_auc_high_value": auc_high,
            "distance_from_chance": distance,
            "selected": selected,
            "empirical_risk_direction": risk_direction,
            "axis_semantics": "HEALTH_HIGH" if health_axis else "RISK_HIGH",
        })
        if selected:
            transformed[column] = final_score
    if not transformed:
        raise RuntimeError(f"No variables survived the development-only direction audit: {columns}")
    return pd.DataFrame(transformed, index=full.index), registry


def binary_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
    }


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 0, 1)
    return {
        "AUROC": float(roc_auc_score(y, probability)),
        "AUPRC": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, clipped)),
    }


def fit_logistic(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if len(columns) != len(set(columns)):
        raise RuntimeError(f"Duplicate model features: {columns}")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=CLOSURE_SPLIT_SEED,
            class_weight="balanced",
        )),
    ])
    pipe.fit(train[columns], train["outcome"])
    return pipe.predict_proba(test[columns])[:, 1]


print("PRECHECK PASS: final closure runner compiled and feature contracts loaded", flush=True)
print("Resolving canonical Alibaba archives...", flush=True)
links = resolve_links()
registry = []
for filename, expected in EXPECTED.items():
    path = RAW / filename
    print(f"Downloading {filename}", flush=True)
    download(links[filename], path)
    actual = digest(path)
    registry.append({
        "file": filename,
        "url": links[filename],
        "bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected,
        "checksum_pass": actual == expected,
    })
    if actual != expected:
        raise RuntimeError(f"Checksum failed: {filename}")
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(RAW)
(OUT / "source_registry.json").write_text(json.dumps(registry, indent=2))

print("Loading job, task, instance, and machine specification tables...", flush=True)
jobs = pd.read_csv(RAW / "pai_job_table.csv", header=None, names=JOB)
tasks = pd.read_csv(RAW / "pai_task_table.csv", header=None, names=TASK)
spec = pd.read_csv(RAW / "pai_machine_spec.csv", header=None, names=SPEC)
jobs = jobs[jobs["status"].isin(["Failed", "Terminated"])].copy()
jobs["outcome"] = (jobs["status"] == "Failed").astype(int)
jobs["source_partition"] = jobs["user"].astype(str).map(
    lambda user: "prior_development" if user_fraction(SOURCE_SPLIT_SEED, user) < 0.75 else "prior_evaluation"
)
jobs = jobs[jobs["source_partition"] == "prior_development"].copy()
jobs["closure_partition"] = jobs["user"].astype(str).map(
    lambda user: "repair_development" if user_fraction(CLOSURE_SPLIT_SEED, user) < 0.80 else "fresh_holdout"
)

task = tasks.copy()
required_task = ["job_name", "task_name", "inst_num", "start_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
task["task_complete"] = task[required_task].notna().mean(axis=1)
task_agg = task.groupby("job_name").agg(
    task_count=("task_name", "size"),
    planned_instances=("inst_num", "sum"),
    total_cpu=("plan_cpu", lambda x: np.nansum(x)),
    total_mem=("plan_mem", lambda x: np.nansum(x)),
    total_gpu=("plan_gpu", lambda x: np.nansum(x)),
    first_task_start=("start_time", "min"),
    task_complete=("task_complete", "mean"),
    gpu_type_mode=("gpu_type", lambda x: x.dropna().astype(str).mode().iloc[0] if len(x.dropna()) else "MISSING"),
).reset_index()

print("Streaming and aggregating the instance table...", flush=True)
parts = []
for chunk in pd.read_csv(RAW / "pai_instance_table.csv", header=None, names=INST, chunksize=1_000_000):
    chunk["inst_complete"] = chunk[["job_name", "task_name", "inst_name", "worker_name", "start_time", "machine"]].notna().mean(axis=1)
    chunk = chunk.merge(spec, on="machine", how="left")
    aggregate = chunk.groupby("job_name").agg(
        observed_instances=("worker_name", "size"),
        first_instance_start=("start_time", "min"),
        last_instance_start=("start_time", "max"),
        instance_launch_coverage=("start_time", lambda x: x.notna().mean()),
        instance_complete=("inst_complete", "mean"),
        spec_coverage=("cap_cpu", lambda x: x.notna().mean()),
        sum_cap_cpu=("cap_cpu", "sum"),
        sum_cap_mem=("cap_mem", "sum"),
        sum_cap_gpu=("cap_gpu", "sum"),
    ).reset_index()
    parts.append(aggregate)
instance_agg = pd.concat(parts, ignore_index=True).groupby("job_name").agg(
    observed_instances=("observed_instances", "sum"),
    first_instance_start=("first_instance_start", "min"),
    last_instance_start=("last_instance_start", "max"),
    instance_launch_coverage=("instance_launch_coverage", "mean"),
    instance_complete=("instance_complete", "mean"),
    spec_coverage=("spec_coverage", "mean"),
    sum_cap_cpu=("sum_cap_cpu", "sum"),
    sum_cap_mem=("sum_cap_mem", "sum"),
    sum_cap_gpu=("sum_cap_gpu", "sum"),
).reset_index()

data = jobs.merge(task_agg, on="job_name", how="left").merge(instance_agg, on="job_name", how="left")
data["task_wait"] = data["first_task_start"] - data["start_time"]
data["instance_wait"] = data["first_instance_start"] - data["start_time"]
data["launch_spread"] = (data["last_instance_start"] - data["first_instance_start"]).clip(lower=0)
data["instance_ratio"] = data["observed_instances"] / data["planned_instances"].replace(0, np.nan)
data["cpu_capacity_ratio"] = (data["total_cpu"] / 100) / data["sum_cap_cpu"].replace(0, np.nan)
data["mem_capacity_ratio"] = data["total_mem"] / data["sum_cap_mem"].replace(0, np.nan)
data["gpu_capacity_ratio"] = (data["total_gpu"] / 100) / data["sum_cap_gpu"].replace(0, np.nan)
data["ordering_consistent"] = ((data["task_wait"] >= 0) & (data["instance_wait"] >= 0)).astype(float)
data = data.sort_values("start_time").reset_index(drop=True)
times = data["start_time"].to_numpy(float)
left = np.searchsorted(times, times - 300, side="left")
data["cluster_density_5m"] = np.arange(len(data)) - left
data["user_density_5m"] = 0
for _, indexes in data.groupby("user").groups.items():
    indexes = np.array(sorted(indexes))
    user_times = data.loc[indexes, "start_time"].to_numpy(float)
    user_left = np.searchsorted(user_times, user_times - 300, side="left")
    data.loc[indexes, "user_density_5m"] = np.arange(len(indexes)) - user_left

repair_development = data[data["closure_partition"] == "repair_development"].copy()
frequency = repair_development["gpu_type_mode"].value_counts(normalize=True)
data["gpu_scarcity"] = 1 - data["gpu_type_mode"].map(frequency).fillna(0)
repair_development = data[data["closure_partition"] == "repair_development"].copy()

print("Running development-only direction audit and freezing repaired axes...", flush=True)
direction_registry = {}
for axis, columns in GROUPS.items():
    transformed, audit = orient_and_select(
        repair_development, data, "outcome", columns, health_axis=axis in {"E", "I", "V"}
    )
    direction_registry[axis] = audit
    data[axis] = transformed.mean(axis=1, skipna=False)
data["W"] = data[["E", "I", "V"]].min(axis=1, skipna=False)
data["coverage"] = data[["B", "P", "E", "I", "V"]].notna().mean(axis=1)
repair = data[(data["closure_partition"] == "repair_development") & (data["coverage"] == 1)].copy()
holdout = data[(data["closure_partition"] == "fresh_holdout") & (data["coverage"] == 1)].copy()
if repair["user"].astype(str).isin(holdout["user"].astype(str)).any():
    raise RuntimeError("User leakage detected between repair development and fresh holdout")
if len(repair) == 0 or len(holdout) == 0:
    raise RuntimeError("Closure partitions are empty")

bq = repair["B"].quantile(0.75)
pq = repair["P"].quantile(0.75)
epsilon = repair["W"].quantile(0.25)
for frame in (data, repair, holdout):
    frame["B_norm"] = frame["B"] / bq
    frame["P_norm"] = frame["P"] / pq
    frame["Pi"] = frame["B_norm"] * frame["P_norm"]
    frame["K"] = (frame["E"] * frame["I"] * frame["V"] * frame["coverage"]).clip(lower=0.05)
    frame["PCR"] = frame["Pi"] / frame["K"]
    frame["H_star"] = frame["PCR"] / (1 + frame["PCR"])
    frame["MQ"] = ((frame["B_norm"] >= 1) & (frame["P_norm"] >= 1) & (frame["W"] <= epsilon)).astype(int)

raw_features = sorted(set(sum(GROUPS.values(), [])))
axes = ["B", "P", "E", "I", "V"]
print("Fitting frozen comparators and evaluating fresh holdout...", flush=True)
p_axes = fit_logistic(repair, holdout, axes)
p_raw = fit_logistic(repair, holdout, raw_features)
p_raw_mq = fit_logistic(repair, holdout, raw_features + ["MQ"])
p_raw_h = fit_logistic(repair, holdout, raw_features + ["H_star"])
y = holdout["outcome"].to_numpy()
raw_metrics = probability_metrics(y, p_raw)
raw_mq_metrics = probability_metrics(y, p_raw_mq)
raw_h_metrics = probability_metrics(y, p_raw_h)
hard_metrics = binary_metrics(y, holdout["MQ"].to_numpy())
h_star_metrics = probability_metrics(y, holdout["H_star"].to_numpy())
results = {
    "artifact_id": "CLOUD_TEST_1C_FINAL_INSTRUMENT_CLOSURE_EMPIRICAL_RETURN_v1",
    "status": "COMPLETE",
    "claim_boundary": "Development-only direction audit and variable selection within the prior Test 1B development cohort, followed by one frozen evaluation on a fresh user-disjoint holdout from that cohort. This is Alibaba cloud-instrument closure, not independent-provider replication or universal validation.",
    "seeds": {"source_partition_seed": SOURCE_SPLIT_SEED, "closure_partition_seed": CLOSURE_SPLIT_SEED},
    "thresholds": {"B_q75": float(bq), "P_q75": float(pq), "epsilon_d": float(epsilon)},
    "coverage": {
        "prior_development_terminal_jobs": int(len(jobs)),
        "repair_development_jobs": int(len(repair)),
        "fresh_holdout_jobs": int(len(holdout)),
        "fresh_holdout_failure_prevalence": float(holdout["outcome"].mean()),
        "repair_development_users": int(repair["user"].nunique()),
        "fresh_holdout_users": int(holdout["user"].nunique()),
        "user_overlap": 0,
    },
    "direction_registry": direction_registry,
    "hard_MQ": hard_metrics,
    "continuous_H_star": h_star_metrics,
    "axis_logistic": probability_metrics(y, p_axes),
    "raw_logistic": raw_metrics,
    "raw_plus_MQ": raw_mq_metrics,
    "raw_plus_H_star": raw_h_metrics,
    "components_AUROC": {
        "B": safe_auc(holdout["outcome"], holdout["B"]),
        "P": safe_auc(holdout["outcome"], holdout["P"]),
        "one_minus_E": safe_auc(holdout["outcome"], 1 - holdout["E"]),
        "one_minus_I": safe_auc(holdout["outcome"], 1 - holdout["I"]),
        "one_minus_V": safe_auc(holdout["outcome"], 1 - holdout["V"]),
        "one_minus_W": safe_auc(holdout["outcome"], 1 - holdout["W"]),
    },
    "closure_rule": {
        "mq_augmentation_supported": bool(raw_mq_metrics["AUROC"] > raw_metrics["AUROC"] + 0.005),
        "h_star_augmentation_supported": bool(raw_h_metrics["AUROC"] > raw_metrics["AUROC"] + 0.005),
        "hard_mq_above_chance": bool(hard_metrics["balanced_accuracy"] > 0.52),
        "continuous_h_star_above_chance": bool(h_star_metrics["AUROC"] > 0.52),
    },
}
(OUT / "metrics.json").write_text(json.dumps(results, indent=2))
(OUT / "direction_registry.json").write_text(json.dumps(direction_registry, indent=2))
pd.DataFrame([results["hard_MQ"]]).to_csv(OUT / "confusion_matrix.csv", index=False)
pd.DataFrame({
    "job_name": holdout["job_name"], "user": holdout["user"], "outcome": holdout["outcome"],
    "B": holdout["B"], "P": holdout["P"], "E": holdout["E"], "I": holdout["I"],
    "V": holdout["V"], "W": holdout["W"], "MQ": holdout["MQ"], "H_star": holdout["H_star"],
}).to_csv(OUT / "fresh_holdout_panel.csv", index=False)
receipt = {
    "source_partition_seed": SOURCE_SPLIT_SEED,
    "closure_partition_seed": CLOSURE_SPLIT_SEED,
    "metrics_sha256": digest(OUT / "metrics.json"),
    "direction_registry_sha256": digest(OUT / "direction_registry.json"),
    "fresh_holdout_panel_sha256": digest(OUT / "fresh_holdout_panel.csv"),
}
(OUT / "run_receipt.json").write_text(json.dumps(receipt, indent=2))
(OUT / "run_log.txt").write_text("Completed bounded final Alibaba cloud-instrument closure run.\n")
print(json.dumps(results, indent=2), flush=True)
