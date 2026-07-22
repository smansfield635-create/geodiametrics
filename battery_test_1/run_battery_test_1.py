from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 451
HORIZON = 20
EOL_FRACTION = 0.80
BASE = Path("battery_test_1")
RAW = BASE / "raw"
OUT = BASE / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "CS2_33": "https://web.calce.umd.edu/batteries/data/CS2_33.zip",
    "CS2_34": "https://web.calce.umd.edu/batteries/data/CS2_34.zip",
    "CS2_35": "https://web.calce.umd.edu/batteries/data/CS2_35.zip",
    "CS2_36": "https://web.calce.umd.edu/batteries/data/CS2_36.zip",
    "CS2_37": "https://web.calce.umd.edu/batteries/data/CS2_37.zip",
    "CS2_38": "https://web.calce.umd.edu/batteries/data/CS2_38.zip",
}
DEV_CELLS = {"CS2_33", "CS2_35", "CS2_37"}
EVAL_CELLS = {"CS2_34", "CS2_36", "CS2_38"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def norm_name(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_header(raw: pd.DataFrame) -> int | None:
    for i in range(min(35, len(raw))):
        vals = [norm_name(v) for v in raw.iloc[i].tolist()]
        joined = " ".join(vals)
        if "voltage" in joined and "current" in joined:
            return i
    return None


def numeric_series(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series | None:
    for c in df.columns:
        name = norm_name(c)
        if all(k in name for k in keys):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 3:
                return s
    return None


def extract_cycle(path: Path, cell: str, cycle_index: int) -> dict | None:
    try:
        book = pd.ExcelFile(path)
    except Exception:
        return None

    frames = []
    for sheet in book.sheet_names:
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None)
            h = find_header(raw)
            if h is None:
                continue
            df = pd.read_excel(path, sheet_name=sheet, header=h)
            df.columns = [norm_name(c) for c in df.columns]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None

    best = max(frames, key=len)
    v = numeric_series(best, ("voltage",))
    i = numeric_series(best, ("current",))
    t = numeric_series(best, ("test", "time")) or numeric_series(best, ("time",))
    temp = numeric_series(best, ("temperature",))
    cap = numeric_series(best, ("discharge", "capacity")) or numeric_series(best, ("capacity",))

    if v is None or i is None:
        return None
    mask = v.notna() & i.notna()
    v = v[mask].reset_index(drop=True)
    i = i[mask].reset_index(drop=True)
    if len(v) < 5:
        return None

    if t is not None:
        t = t[mask].reset_index(drop=True)
        dt = t.diff().abs().replace([np.inf, -np.inf], np.nan)
        med = float(dt[(dt > 0) & dt.notna()].median()) if ((dt > 0) & dt.notna()).any() else 1.0
        dt = dt.fillna(med).clip(lower=0, upper=max(10 * med, 1.0))
    else:
        dt = pd.Series(np.ones(len(v)))

    abs_i = i.abs()
    throughput_ah = float((abs_i * dt).sum() / 3600.0)
    energy_wh = float((abs_i * v.abs() * dt).sum() / 3600.0)
    duration_s = float(dt.sum())

    capacity = np.nan
    if cap is not None:
        vals = cap.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            q = float(vals.quantile(0.95))
            if 0.05 <= q <= 10:
                capacity = q
    if not np.isfinite(capacity):
        capacity = throughput_ah

    row = {
        "cell": cell,
        "cycle": cycle_index,
        "source_file": path.name,
        "capacity_ah": capacity,
        "throughput_ah": throughput_ah,
        "energy_wh": energy_wh,
        "duration_s": duration_s,
        "voltage_mean": float(v.mean()),
        "voltage_min": float(v.min()),
        "voltage_std": float(v.std(ddof=0)),
        "current_abs_mean": float(abs_i.mean()),
        "current_abs_max": float(abs_i.max()),
        "sample_count": int(len(v)),
        "coverage": float(mask.mean()),
    }
    if temp is not None:
        temp = temp[mask].reset_index(drop=True)
        row["temperature_mean"] = float(temp.mean())
        row["temperature_max"] = float(temp.max())
        row["temperature_std"] = float(temp.std(ddof=0))
    else:
        row["temperature_mean"] = np.nan
        row["temperature_max"] = np.nan
        row["temperature_std"] = np.nan
    return row


def ecdf(train: pd.Series, values: pd.Series) -> pd.Series:
    tr = np.sort(pd.to_numeric(train, errors="coerce").dropna().to_numpy(float))
    if len(tr) == 0:
        return pd.Series(np.nan, index=values.index)
    arr = pd.to_numeric(values, errors="coerce")
    return arr.map(lambda x: np.nan if pd.isna(x) else np.searchsorted(tr, x, side="right") / len(tr))


def prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def main() -> None:
    registry = []
    rows = []
    for cell, url in SOURCES.items():
        zpath = RAW / f"{cell}.zip"
        print(f"Downloading {cell}", flush=True)
        download(url, zpath)
        registry.append({"cell": cell, "url": url, "bytes": zpath.stat().st_size, "sha256": sha256(zpath)})
        cell_dir = RAW / cell
        cell_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(cell_dir)
        files = sorted(list(cell_dir.rglob("*.xlsx")) + list(cell_dir.rglob("*.xls")))
        print(f"Parsing {cell}: {len(files)} workbooks", flush=True)
        for idx, file in enumerate(files, start=1):
            row = extract_cycle(file, cell, idx)
            if row is not None:
                rows.append(row)

    panel = pd.DataFrame(rows).sort_values(["cell", "cycle"]).reset_index(drop=True)
    if panel.empty or panel["cell"].nunique() < 4:
        raise RuntimeError(f"Insufficient parsed battery data: rows={len(panel)}, cells={panel['cell'].nunique() if len(panel) else 0}")

    panel["initial_capacity"] = panel.groupby("cell")["capacity_ah"].transform(lambda s: s.head(5).median())
    panel["capacity_ratio"] = panel["capacity_ah"] / panel["initial_capacity"]
    panel["cum_throughput"] = panel.groupby("cell")["throughput_ah"].cumsum()
    panel["capacity_slope_5"] = panel.groupby("cell")["capacity_ratio"].transform(lambda s: s.diff().rolling(5, min_periods=3).mean())
    panel["voltage_min_slope_5"] = panel.groupby("cell")["voltage_min"].transform(lambda s: s.diff().rolling(5, min_periods=3).mean())
    panel["temp_rise"] = panel["temperature_max"] - panel["temperature_mean"]

    eol_cycle = {}
    for cell, g in panel.groupby("cell"):
        hit = g.loc[g["capacity_ratio"] <= EOL_FRACTION, "cycle"]
        eol_cycle[cell] = int(hit.iloc[0]) if len(hit) else int(g["cycle"].max() + 1)
    panel["eol_cycle"] = panel["cell"].map(eol_cycle)
    panel["cycles_to_eol"] = panel["eol_cycle"] - panel["cycle"]
    panel["failure_within_horizon"] = ((panel["cycles_to_eol"] > 0) & (panel["cycles_to_eol"] <= HORIZON)).astype(int)
    panel = panel[panel["cycle"] < panel["eol_cycle"]].copy()
    panel["split"] = np.where(panel["cell"].isin(DEV_CELLS), "development", "evaluation")

    dev = panel[panel["split"] == "development"].copy()
    test = panel[panel["split"] == "evaluation"].copy()
    if dev["failure_within_horizon"].nunique() < 2 or test["failure_within_horizon"].nunique() < 2:
        raise RuntimeError("Both development and evaluation require positive and negative horizon outcomes")

    # Direct physical realization.
    B_raw = dev["cum_throughput"]
    panel["B"] = ecdf(B_raw, panel["cum_throughput"])
    pressure_raw_dev = (
        ecdf(dev["current_abs_mean"], dev["current_abs_mean"]) +
        ecdf(dev["current_abs_max"], dev["current_abs_max"]) +
        ecdf(dev["temperature_max"], dev["temperature_max"])
    ) / 3
    panel["P"] = (
        ecdf(dev["current_abs_mean"], panel["current_abs_mean"]) +
        ecdf(dev["current_abs_max"], panel["current_abs_max"]) +
        ecdf(dev["temperature_max"], panel["temperature_max"])
    ) / 3

    panel["E"] = 1 - (
        ecdf(dev["voltage_std"], panel["voltage_std"]) +
        ecdf(dev["temp_rise"], panel["temp_rise"])
    ) / 2
    panel["I"] = (
        panel["coverage"].clip(0, 1) +
        (1 - ecdf(dev["voltage_min_slope_5"].abs(), panel["voltage_min_slope_5"].abs()))
    ) / 2
    panel["V"] = (
        panel["capacity_ratio"].clip(0, 1.2) / 1.2 +
        (1 - ecdf(dev["capacity_slope_5"].abs(), panel["capacity_slope_5"].abs()))
    ) / 2
    panel["W"] = panel[["E", "I", "V"]].min(axis=1)
    panel["C"] = panel[["B", "P", "E", "I", "V"]].notna().mean(axis=1)

    dev = panel[(panel["split"] == "development") & (panel["C"] == 1)].copy()
    test = panel[(panel["split"] == "evaluation") & (panel["C"] == 1)].copy()
    bq = float(dev["B"].quantile(0.75))
    pq = float(dev["P"].quantile(0.75))
    eps = float(dev["W"].quantile(0.25))
    for frame in (dev, test):
        frame["B_norm"] = frame["B"] / bq
        frame["P_norm"] = frame["P"] / pq
        frame["Pi"] = frame["B_norm"] * frame["P_norm"]
        frame["K"] = (frame["E"] * frame["I"] * frame["V"] * frame["C"]).clip(lower=0.05)
        frame["PCR"] = frame["Pi"] / frame["K"]
        frame["H_star"] = frame["PCR"] / (1 + frame["PCR"])
        frame["MQ"] = ((frame["B_norm"] >= 1) & (frame["P_norm"] >= 1) & (frame["W"] <= eps)).astype(int)

    y_train = dev["failure_within_horizon"].to_numpy()
    y_test = test["failure_within_horizon"].to_numpy()
    raw_features = [
        "cum_throughput", "capacity_ratio", "capacity_slope_5", "voltage_min", "voltage_std",
        "current_abs_mean", "current_abs_max", "temperature_max", "temp_rise", "duration_s", "energy_wh"
    ]
    axis_features = ["B", "P", "E", "I", "V"]

    def fit_prob(features: list[str]) -> np.ndarray:
        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
        ])
        model.fit(dev[features], y_train)
        return model.predict_proba(test[features])[:, 1]

    p_raw = fit_prob(raw_features)
    p_axis = fit_prob(axis_features)
    p_raw_mq = fit_prob(raw_features + ["MQ"])
    p_raw_h = fit_prob(raw_features + ["H_star"])

    results = {
        "artifact_id": "BATTERY_TEST_1_CALCE_CS2_COHERENCE_COLLAPSE_RETURN_v1",
        "status": "COMPLETE",
        "claim_boundary": "Cycle-resolved CALCE CS2 run-to-failure test. Predicts crossing 80% retained-capacity EOL within 20 cycles using only current and prior-cycle physical measurements; cell-disjoint evaluation.",
        "source": {"institution": "CALCE, University of Maryland", "cells": sorted(SOURCES)},
        "outcome": {"EOL_fraction": EOL_FRACTION, "prediction_horizon_cycles": HORIZON},
        "coverage": {
            "parsed_cycles": int(len(panel)),
            "development_cycles": int(len(dev)),
            "evaluation_cycles": int(len(test)),
            "development_cells": sorted(DEV_CELLS),
            "evaluation_cells": sorted(EVAL_CELLS),
            "evaluation_event_prevalence": float(y_test.mean()),
        },
        "thresholds": {"B_q75": bq, "P_q75": pq, "epsilon_d": eps},
        "hard_MQ": binary_metrics(y_test, test["MQ"].to_numpy()),
        "continuous_H_star": prob_metrics(y_test, test["H_star"].to_numpy()),
        "axis_logistic": prob_metrics(y_test, p_axis),
        "raw_logistic": prob_metrics(y_test, p_raw),
        "raw_plus_MQ": prob_metrics(y_test, p_raw_mq),
        "raw_plus_H_star": prob_metrics(y_test, p_raw_h),
        "components_AUROC": {
            "B": float(roc_auc_score(y_test, test["B"])),
            "P": float(roc_auc_score(y_test, test["P"])),
            "one_minus_E": float(roc_auc_score(y_test, 1 - test["E"])),
            "one_minus_I": float(roc_auc_score(y_test, 1 - test["I"])),
            "one_minus_V": float(roc_auc_score(y_test, 1 - test["V"])),
            "one_minus_W": float(roc_auc_score(y_test, 1 - test["W"])),
        },
    }
    results["augmentation"] = {
        "MQ_AUROC_delta": results["raw_plus_MQ"]["AUROC"] - results["raw_logistic"]["AUROC"],
        "H_star_AUROC_delta": results["raw_plus_H_star"]["AUROC"] - results["raw_logistic"]["AUROC"],
    }

    panel.to_csv(OUT / "cycle_panel.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    (OUT / "source_registry.json").write_text(json.dumps(registry, indent=2))
    receipt = {
        "seed": SEED,
        "metrics_sha256": sha256(OUT / "metrics.json"),
        "panel_sha256": sha256(OUT / "cycle_panel.csv"),
        "source_registry_sha256": sha256(OUT / "source_registry.json"),
    }
    (OUT / "run_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
