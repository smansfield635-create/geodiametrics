from pathlib import Path

runner = Path("battery_test_1/run_battery_test_1.py")
source = runner.read_text()
source = source.replace(
    't = numeric_series(best, ("test", "time")) or numeric_series(best, ("time",))',
    't = numeric_series(best, ("test", "time"))\n    if t is None:\n        t = numeric_series(best, ("time",))',
)
source = source.replace(
    'cap = numeric_series(best, ("discharge", "capacity")) or numeric_series(best, ("capacity",))',
    'cap = numeric_series(best, ("discharge", "capacity"))\n    if cap is None:\n        cap = numeric_series(best, ("capacity",))',
)
compile(source, str(runner), "exec")
print("PRECHECK PASS: syntax compiled; pandas Series fallback defect removed", flush=True)
namespace = {"__name__": "__main__", "__file__": str(runner)}
exec(compile(source, str(runner), "exec"), namespace)
