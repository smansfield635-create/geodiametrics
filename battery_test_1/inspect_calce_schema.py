from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

SOURCE_URL = "https://web.calce.umd.edu/batteries/data/CS2_33.zip"
BASE = Path("battery_test_1/schema_inspection")
RAW = BASE / "raw"
OUT = BASE / "results"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def download(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def row_summary(raw: pd.DataFrame, row_index: int) -> dict:
    values = raw.iloc[row_index].tolist()
    text = [str(v) for v in values if pd.notna(v)]
    normalized = [norm(v) for v in values if pd.notna(v)]
    joined = " ".join(normalized)
    keywords = [
        key for key in (
            "cycle", "cycle_index", "test_time", "time", "voltage", "current",
            "temperature", "capacity", "discharge_capacity", "charge_capacity",
            "step_index", "data_point", "date_time"
        ) if key in joined
    ]
    return {
        "row_index_zero_based": int(row_index),
        "non_null_count": int(sum(pd.notna(v) for v in values)),
        "values": text[:80],
        "normalized_values": normalized[:80],
        "matched_keywords": keywords,
    }


def inspect_workbook(path: Path) -> dict:
    result = {
        "file": path.name,
        "relative_path": str(path.relative_to(RAW)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "sheets": [],
        "open_error": None,
    }
    try:
        book = pd.ExcelFile(path)
    except Exception as exc:
        result["open_error"] = repr(exc)
        return result

    for sheet in book.sheet_names:
        sheet_result = {
            "sheet": str(sheet),
            "read_error": None,
            "shape_preview": None,
            "candidate_header_rows": [],
            "first_nonempty_rows": [],
            "header_trials": [],
        }
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=80)
            sheet_result["shape_preview"] = [int(raw.shape[0]), int(raw.shape[1])]
            nonempty = [i for i in range(len(raw)) if raw.iloc[i].notna().any()]
            sheet_result["first_nonempty_rows"] = [row_summary(raw, i) for i in nonempty[:12]]

            for i in nonempty:
                summary = row_summary(raw, i)
                keys = set(summary["matched_keywords"])
                if {"voltage", "current"}.issubset(keys) or "cycle" in keys or "cycle_index" in keys:
                    sheet_result["candidate_header_rows"].append(summary)

            trial_rows = []
            for item in sheet_result["candidate_header_rows"][:8]:
                trial_rows.append(item["row_index_zero_based"])
            for row_index in sorted(set(trial_rows)):
                try:
                    df = pd.read_excel(path, sheet_name=sheet, header=row_index, nrows=25)
                    columns = [str(c) for c in df.columns]
                    normalized_columns = [norm(c) for c in columns]
                    numeric_counts = {
                        str(c): int(pd.to_numeric(df[c], errors="coerce").notna().sum())
                        for c in df.columns[:80]
                    }
                    sheet_result["header_trials"].append({
                        "header_row_zero_based": int(row_index),
                        "columns": columns[:80],
                        "normalized_columns": normalized_columns[:80],
                        "preview_row_count": int(len(df)),
                        "numeric_non_null_counts": numeric_counts,
                    })
                except Exception as exc:
                    sheet_result["header_trials"].append({
                        "header_row_zero_based": int(row_index),
                        "error": repr(exc),
                    })
        except Exception as exc:
            sheet_result["read_error"] = repr(exc)
        result["sheets"].append(sheet_result)
    return result


def main() -> None:
    archive = RAW / "CS2_33.zip"
    print("Downloading one canonical CALCE archive: CS2_33", flush=True)
    download(SOURCE_URL, archive)

    extract_dir = RAW / "CS2_33"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        zf.extractall(extract_dir)

    workbooks = sorted(list(extract_dir.rglob("*.xlsx")) + list(extract_dir.rglob("*.xls")))
    if not workbooks:
        raise RuntimeError("No Excel workbooks found in CS2_33 archive")

    manifest = {
        "artifact_id": "BATTERY_TEST_1_CALCE_CS2_33_SCHEMA_MANIFEST_v1",
        "status": "SCHEMA_INSPECTION_COMPLETE",
        "source_url": SOURCE_URL,
        "archive": {
            "file": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "zip_member_count": len(members),
            "zip_members": members,
        },
        "workbook_count": len(workbooks),
        "workbooks": [],
        "empirical_equations_executed": False,
        "claim_boundary": "Schema inspection only. No battery coherence, collapse, prediction, threshold, or validation result.",
    }

    for workbook in workbooks:
        print(f"Inspecting workbook: {workbook.name}", flush=True)
        manifest["workbooks"].append(inspect_workbook(workbook))

    manifest_path = OUT / "schema_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    lines = [
        "BATTERY TEST 1 — CALCE CS2_33 SCHEMA INSPECTION",
        f"Archive SHA-256: {manifest['archive']['sha256']}",
        f"Workbook count: {manifest['workbook_count']}",
        "Empirical equations executed: FALSE",
        "",
    ]
    for workbook in manifest["workbooks"]:
        lines.append(f"WORKBOOK: {workbook['file']}")
        if workbook["open_error"]:
            lines.append(f"  OPEN ERROR: {workbook['open_error']}")
            continue
        for sheet in workbook["sheets"]:
            lines.append(f"  SHEET: {sheet['sheet']} preview_shape={sheet['shape_preview']}")
            if sheet["read_error"]:
                lines.append(f"    READ ERROR: {sheet['read_error']}")
                continue
            for candidate in sheet["candidate_header_rows"][:8]:
                lines.append(
                    f"    CANDIDATE HEADER row={candidate['row_index_zero_based']} "
                    f"keywords={candidate['matched_keywords']} values={candidate['values']}"
                )
            for trial in sheet["header_trials"][:8]:
                lines.append(
                    f"    HEADER TRIAL row={trial.get('header_row_zero_based')} "
                    f"columns={trial.get('columns', [])} error={trial.get('error')}"
                )
        lines.append("")

    (OUT / "schema_report.txt").write_text("\n".join(lines))
    receipt = {
        "schema_manifest_sha256": sha256(manifest_path),
        "schema_report_sha256": sha256(OUT / "schema_report.txt"),
        "archive_sha256": sha256(archive),
    }
    (OUT / "schema_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({
        "status": manifest["status"],
        "workbook_count": manifest["workbook_count"],
        "schema_manifest_sha256": receipt["schema_manifest_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
