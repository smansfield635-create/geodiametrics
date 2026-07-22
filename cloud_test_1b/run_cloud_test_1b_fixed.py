from pathlib import Path

source_path = Path(__file__).with_name("run_cloud_test_1b.py")
source = source_path.read_text(encoding="utf-8")
old = "'one_minus_mean_EIV':float(roc_auc_score(y,1-test[['E','I','V']].mean(axis=1))},"
new = "'one_minus_mean_EIV':float(roc_auc_score(y,1-test[['E','I','V']].mean(axis=1)))},"
if old not in source:
    raise RuntimeError("Expected Cloud Test 1B syntax target was not found.")
fixed = source.replace(old, new, 1)
compile(fixed, str(source_path), "exec")
exec(compile(fixed, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
