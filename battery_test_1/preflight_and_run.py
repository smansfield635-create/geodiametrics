from pathlib import Path

runner = Path("battery_test_1/run_battery_test_1.py")
source = runner.read_text()

# Replace the original workbook-collapsing parser. The CALCE archives contain
# repeated cycle worksheets inside workbooks; the prior parser selected only
# the single largest worksheet and therefore produced too few cycle rows.
start = source.index("def extract_cycle(")
end = source.index("\ndef ecdf(", start)
replacement = r'''def extract_frame(df: pd.DataFrame, cell: str, cycle_index: int, source_name: str, sheet_name: str) -> dict | None:
    v = numeric_series(df, ("voltage",))
    i = numeric_series(df, ("current",))
    t = numeric_series(df, ("test", "time"))
    if t is None:
        t = numeric_series(df, ("time",))
    temp = numeric_series(df, ("temperature",))
    cap = numeric_series(df, ("discharge", "capacity"))
    if cap is None:
        cap = numeric_series(df, ("capacity",))

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
        valid_dt = dt[(dt > 0) & dt.notna()]
        med = float(valid_dt.median()) if len(valid_dt) else 1.0
        dt = dt.fillna(med).clip(lower=0, upper=max(10 * med, 1.0))
    else:
        dt = pd.Series(np.ones(len(v)))

    abs_i = i.abs()
    throughput_ah = float((abs_i * dt).sum() / 3600.0)
    energy_wh = float((abs_i * v.abs() * dt).sum() / 3600.0)
    duration_s = float(dt.sum())

    capacity = np.nan
    if cap is not None:
        vals = cap[mask].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            q = float(vals.quantile(0.95))
            if 0.05 <= q <= 10:
                capacity = q
    if not np.isfinite(capacity):
        capacity = throughput_ah

    row = {
        "cell": cell,
        "cycle": cycle_index,
        "source_file": source_name,
        "source_sheet": sheet_name,
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


def extract_workbook(path: Path, cell: str, cycle_start: int) -> list[dict]:
    try:
        book = pd.ExcelFile(path)
    except Exception:
        return []

    rows = []
    next_cycle = cycle_start
    for sheet in book.sheet_names:
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None)
            h = find_header(raw)
            if h is None:
                continue
            df = pd.read_excel(path, sheet_name=sheet, header=h)
            df.columns = [norm_name(c) for c in df.columns]
            row = extract_frame(df, cell, next_cycle, path.name, str(sheet))
            if row is not None:
                rows.append(row)
                next_cycle += 1
        except Exception:
            continue
    return rows
'''
source = source[:start] + replacement + source[end:]

old_loop = '''        for idx, file in enumerate(files, start=1):
            row = extract_cycle(file, cell, idx)
            if row is not None:
                rows.append(row)
'''
new_loop = '''        cycle_index = 1
        for file in files:
            workbook_rows = extract_workbook(file, cell, cycle_index)
            rows.extend(workbook_rows)
            cycle_index += len(workbook_rows)
        print(f"Parsed {cell}: {cycle_index - 1} cycle worksheets", flush=True)
'''
if old_loop not in source:
    raise RuntimeError("Expected original workbook loop was not found")
source = source.replace(old_loop, new_loop)

old_guard = '        raise RuntimeError("Both development and evaluation require positive and negative horizon outcomes")'
new_guard = '''        counts = {
            "development": dev["failure_within_horizon"].value_counts().to_dict(),
            "evaluation": test["failure_within_horizon"].value_counts().to_dict(),
            "cycles_per_cell": panel.groupby("cell").size().to_dict(),
            "eol_cycle": eol_cycle,
        }
        raise RuntimeError(f"Outcome class support still insufficient after worksheet-level parsing: {counts}")'''
source = source.replace(old_guard, new_guard)

compile(source, str(runner), "exec")
print("PRECHECK PASS: worksheet-level cycle parser installed; syntax compiled", flush=True)
namespace = {"__name__": "__main__", "__file__": str(runner)}
exec(compile(source, str(runner), "exec"), namespace)
