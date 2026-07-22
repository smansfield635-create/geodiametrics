from __future__ import annotations

import hashlib
import json
import re
import zipfile
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
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 451
HORIZON = 20
EOL_FRACTION = 0.80
BASE = Path("battery_test_1")
RAW = BASE / "raw_schema_locked"
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

REQUIRED = {
    "cycle_index", "test_time_s", "current_a", "voltage_v",
    "discharge_capacity_ah", "internal_resistance_ohm", "date_time"
}


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


def norm(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def cycle_rows_from_workbook(path: Path, cell: str) -> list[dict]:
    rows: list[dict] = []
    try:
        book = pd.ExcelFile(path)
    except Exception:
        return rows

    for sheet in book.sheet_names:
        if not str(sheet).lower().startswith("channel_"):
            continue
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=0)
        except Exception:
            continue
        df.columns = [norm(c) for c in df.columns]
        if not REQUIRED.issubset(df.columns):
            continue

        for c in [
            "cycle_index", "test_time_s", "current_a", "voltage_v",
            "discharge_capacity_ah", "internal_resistance_ohm"
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        df = df.dropna(subset=["cycle_index", "current_a", "voltage_v"])
        if df.empty:
            continue

        for local_cycle, g in df.groupby("cycle_index", sort=True):
            g = g.sort_values(["test_time_s", "date_time"], na_position="last")
            if len(g) < 5:
                continue
            current = g["current_a"].astype(float)
            voltage = g["voltage_v"].astype(float)
            tt = g["test_time_s"].astype(float)
            dt = tt.diff().abs().replace([np.inf, -np.inf], np.nan)
            valid_dt = dt[(dt > 0) & dt.notna()]
            med_dt = float(valid_dt.median()) if len(valid_dt) else 1.0
            dt = dt.fillna(med_dt).clip(lower=0, upper=max(10 * med_dt, 1.0))
            abs_i = current.abs()

            discharge = g["discharge_capacity_ah"].replace([np.inf, -np.inf], np.nan)
            discharge = discharge[discharge >= 0]
            capacity = float(discharge.max()) if discharge.notna().any() else np.nan
            if not np.isfinite(capacity) or capacity <= 0:
                continue

            ir = g["internal_resistance_ohm"].replace([np.inf, -np.inf], np.nan)
            ir = ir[ir > 0]
            start_dt = g["date_time"].dropna().min() if g["date_time"].notna().any() else pd.NaT

            rows.append({
                "cell": cell,
                "source_file": path.name,
                "source_sheet": str(sheet),
                "local_cycle_index": int(local_cycle),
                "cycle_start": start_dt,
                "capacity_ah": capacity,
                "throughput_ah": float((abs_i * dt).sum() / 3600.0),
                "energy_wh": float((abs_i * voltage.abs() * dt).sum() / 3600.0),
                "duration_s": float(dt.sum()),
                "voltage_mean": float(voltage.mean()),
                "voltage_min": float(voltage.min()),
                "voltage_std": float(voltage.std(ddof=0)),
                "current_abs_mean": float(abs_i.mean()),
                "current_abs_max": float(abs_i.max()),
                "internal_resistance": float(ir.median()) if len(ir) else np.nan,
                "sample_count": int(len(g)),
                "coverage": float(g[["current_a", "voltage_v", "discharge_capacity_ah", "test_time_s"]].notna().mean().mean()),
            })
    return rows


def ecdf(train: pd.Series, values: pd.Series) -> pd.Series:
    tr = np.sort(pd.to_numeric(train, errors="coerce").dropna().to_numpy(float))
    arr = pd.to_numeric(values, errors="coerce")
    if len(tr) == 0:
        return pd.Series(np.nan, index=values.index)
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
    all_rows: list[dict] = []

    for cell, url in SOURCES.items():
        zpath = RAW / f"{cell}.zip"
        print(f"Downloading {cell}", flush=True)
        download(url, zpath)
        registry.append({"cell": cell, "url": url, "bytes": zpath.stat().st_size, "sha256": sha256(zpath)})
        cell_dir = RAW / cell
        cell_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(cell_dir)
        workbooks = sorted(cell_dir.rglob("*.xlsx"))
        cell_rows: list[dict] = []
        for wb in workbooks:
            cell_rows.extend(cycle_rows_from_workbook(wb, cell))
        print(f"Raw parsed cycle groups {cell}: {len(cell_rows)}", flush=True)
        all_rows.extend(cell_rows)

    panel = pd.DataFrame(all_rows)
    if panel.empty:
        raise RuntimeError("Schema-locked parser produced no cycle rows")

    panel["cycle_start_key"] = panel["cycle_start"].astype(str)
    panel = panel.sort_values(["cell", "cycle_start", "source_file", "local_cycle_index"])
    panel = panel.drop_duplicates(subset=["cell", "cycle_start_key", "capacity_ah"], keep="last")
    panel["cycle"] = panel.groupby("cell").cumcount() + 1
    panel = panel.reset_index(drop=True)

    counts_per_cell = panel.groupby("cell").size().to_dict()
    print(f"Deduplicated cycles per cell: {counts_per_cell}", flush=True)
    if panel["cell"].nunique() != 6 or min(counts_per_cell.values()) < 50:
        raise RuntimeError(f"Schema-locked cycle support insufficient: {counts_per_cell}")

    panel["initial_capacity"] = panel.groupby("cell")["capacity_ah"].transform(lambda s: s.head(5).median())
    panel["capacity_ratio"] = panel["capacity_ah"] / panel["initial_capacity"]
    panel["cum_throughput"] = panel.groupby("cell")["throughput_ah"].cumsum()
    panel["capacity_slope_5"] = panel.groupby("cell")["capacity_ratio"].transform(lambda s: s.diff().rolling(5, min_periods=3).mean())
    panel["voltage_min_slope_5"] = panel.groupby("cell")["voltage_min"].transform(lambda s: s.diff().rolling(5, min_periods=3).mean())
    panel["resistance_slope_5"] = panel.groupby("cell")["internal_resistance"].transform(lambda s: s.diff().rolling(5, min_periods=3).mean())

    eol_cycle: dict[str, int] = {}
    censored = []
    for cell, g in panel.groupby("cell"):
        hit = g.loc[g["capacity_ratio"] <= EOL_FRACTION, "cycle"]
        if len(hit):
            eol_cycle[cell] = int(hit.iloc[0])
        else:
            censored.append(cell)
    if censored:
        raise RuntimeError(f"Cells without observed 80% EOL crossing: {censored}; no pseudo-EOL substitution allowed")

    panel["eol_cycle"] = panel["cell"].map(eol_cycle)
    panel["cycles_to_eol"] = panel["eol_cycle"] - panel["cycle"]
    panel["failure_within_horizon"] = ((panel["cycles_to_eol"] > 0) & (panel["cycles_to_eol"] <= HORIZON)).astype(int)
    panel = panel[panel["cycle"] < panel["eol_cycle"]].copy()
    panel["split"] = np.where(panel["cell"].isin(DEV_CELLS), "development", "evaluation")

    dev0 = panel[panel["split"] == "development"].copy()
    test0 = panel[panel["split"] == "evaluation"].copy()

    panel["B"] = ecdf(dev0["cum_throughput"], panel["cum_throughput"])
    panel["P"] = (
        ecdf(dev0["current_abs_mean"], panel["current_abs_mean"]) +
        ecdf(dev0["current_abs_max"], panel["current_abs_max"]) +
        ecdf(dev0["duration_s"], panel["duration_s"])
    ) / 3
    panel["E"] = 1 - (
        ecdf(dev0["voltage_std"], panel["voltage_std"]) +
        ecdf(dev0["internal_resistance"], panel["internal_resistance"])
    ) / 2
    panel["I"] = (
        panel["coverage"].clip(0, 1) +
        (1 - ecdf(dev0["voltage_min_slope_5"].abs(), panel["voltage_min_slope_5"].abs())) +
        (1 - ecdf(dev0["resistance_slope_5"].abs(), panel["resistance_slope_5"].abs()))
    ) / 3
    panel["V"] = (
        panel["capacity_ratio"].clip(0, 1.2) / 1.2 +
        (1 - ecdf(dev0["capacity_slope_5"].abs(), panel["capacity_slope_5"].abs()))
    ) / 2
    panel["W"] = panel[["E", "I", "V"]].min(axis=1)
    panel["C"] = panel[["B", "P", "E", "I", "V"]].notna().mean(axis=1)

    dev = panel[(panel["split"] == "development") & (panel["C"] == 1)].copy()
    test = panel[(panel["split"] == "evaluation") & (panel["C"] == 1)].copy()
    class_counts = {
        "development": dev["failure_within_horizon"].value_counts().to_dict(),
        "evaluation": test["failure_within_horizon"].value_counts().to_dict(),
    }
    print(f"Outcome support: {class_counts}", flush=True)
    if dev["failure_within_horizon"].nunique() < 2 or test["failure_within_horizon"].nunique() < 2:
        raise RuntimeError(f"Outcome class support insufficient after schema-locked parsing: {class_counts}")

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
        "current_abs_mean", "current_abs_max", "internal_resistance", "resistance_slope_5",
        "duration_s", "energy_wh"
    ]
    axis_features = ["B", "P", "E", "I", "V"]

    def fit_prob(features: list[str]) -> np.ndarray:
        if len(features) != len(set(features)):
            raise RuntimeError(f"Duplicate model features: {features}")
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
        "artifact_id": "BATTERY_TEST_1_CALCE_CS2_SCHEMA_LOCKED_RETURN_v1",
        "status": "COMPLETE",
        "claim_boundary": "Cycle-resolved CALCE CS2 run-to-failure test using schema-observed Arbin fields and cell-disjoint evaluation. Predicts observed 80% retained-capacity EOL crossing within 20 cycles.",
        "schema_lock": {
            "sheet_prefix": "Channel_",
            "header_row": 0,
            "cycle_field": "Cycle_Index",
            "time_field": "Test_Time(s)",
            "current_field": "Current(A)",
            "voltage_field": "Voltage(V)",
            "capacity_field": "Discharge_Capacity(Ah)",
            "resistance_field": "Internal_Resistance(Ohm)",
        },
        "coverage": {
            "cycles_per_cell": counts_per_cell,
            "development_cycles": int(len(dev)),
            "evaluation_cycles": int(len(test)),
            "evaluation_event_prevalence": float(y_test.mean()),
            "development_cells": sorted(DEV_CELLS),
            "evaluation_cells": sorted(EVAL_CELLS),
            "eol_cycle": eol_cycle,
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
