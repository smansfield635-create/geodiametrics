from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RUNNER = Path(__file__).with_name("run_cloud_test_1b.py")
source = RUNNER.read_text(encoding="utf-8")

repairs = {
    "axes=['B','P','E','I','V']; raw=Bcols+Pcols+Cap+['task_complete','instance_complete','spec_coverage','instance_ratio','launch_spread']":
    "axes=['B','P','E','I','V']; raw=Bcols+Pcols+Cap+['task_complete','instance_complete','spec_coverage','instance_ratio']",
    "'one_minus_mean_EIV':float(roc_auc_score(y,1-test[['E','I','V']].mean(axis=1))},":
    "'one_minus_mean_EIV':float(roc_auc_score(y,1-test[['E','I','V']].mean(axis=1)))},",
}

for old, new in repairs.items():
    if old not in source:
        raise RuntimeError(f"Required verified repair target was not found: {old}")
    source = source.replace(old, new, 1)

# Compile the exact corrected source before any network or data work.
compiled = compile(source, str(RUNNER), "exec")

# Static feature-contract gate.
Bcols = ['total_cpu','total_mem','total_gpu','task_count','planned_instances','launch_spread']
Pcols = ['task_wait','instance_wait','cluster_density_5m','user_density_5m','gpu_scarcity']
Cap = ['cpu_capacity_ratio','mem_capacity_ratio','gpu_capacity_ratio']
axes = ['B','P','E','I','V']
raw = Bcols + Pcols + Cap + ['task_complete','instance_complete','spec_coverage','instance_ratio']
feature_sets = {
    'axes': axes,
    'raw': raw,
    'raw_plus_MQ': raw + ['MQ'],
    'raw_plus_H_star': raw + ['H_star'],
}
for name, cols in feature_sets.items():
    duplicates = sorted({c for c in cols if cols.count(c) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate features in {name}: {duplicates}")

# End-to-end model smoke test before any source download.
rng = np.random.default_rng(256)
X = pd.DataFrame(rng.normal(size=(64, len(raw))), columns=raw)
y = np.array([0, 1] * 32)
pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('model', LogisticRegression(max_iter=200, random_state=256)),
])
pipe.fit(X, y)
if pipe.predict_proba(X).shape != (64, 2):
    raise RuntimeError("Synthetic model smoke test failed")

print('PRECHECK PASS: syntax repaired, feature uniqueness verified, model smoke test passed', flush=True)

# Execute only the already-compiled corrected source.
namespace = {'__name__': '__main__', '__file__': str(RUNNER)}
exec(compiled, namespace, namespace)
