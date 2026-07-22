from __future__ import annotations

from pathlib import Path
import runpy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RUNNER = Path(__file__).with_name("run_cloud_test_1b.py")
source = RUNNER.read_text(encoding="utf-8")

old = "axes=['B','P','E','I','V']; raw=Bcols+Pcols+Cap+['task_complete','instance_complete','spec_coverage','instance_ratio','launch_spread']"
new = "axes=['B','P','E','I','V']; raw=Bcols+Pcols+Cap+['task_complete','instance_complete','spec_coverage','instance_ratio']"
if old not in source:
    raise RuntimeError("Expected Test 1B feature declaration was not found; refusing unverified execution.")
source = source.replace(old, new, 1)

# Syntax gate.
compile(source, str(RUNNER), "exec")

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

# Small end-to-end model smoke test before any source download.
rng = np.random.default_rng(256)
X = pd.DataFrame(rng.normal(size=(64, len(raw))), columns=raw)
y = np.array([0, 1] * 32)
pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('model', LogisticRegression(max_iter=200, random_state=256)),
])
pipe.fit(X, y)
assert pipe.predict_proba(X).shape == (64, 2)
print('PRECHECK PASS: syntax, feature uniqueness, and model smoke test', flush=True)

# Execute the corrected source in-process only after preflight passes.
namespace = {'__name__': '__main__', '__file__': str(RUNNER)}
exec(compile(source, str(RUNNER), 'exec'), namespace, namespace)
