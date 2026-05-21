"""Compare all MLflow experiment runs."""
import sys, mlflow
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
mlflow.set_tracking_uri("mlruns")
exp = mlflow.get_experiment_by_name("canadian-financial-sentiment")
if not exp:
    print("No experiments. Run log_run.py first."); exit()
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
if runs.empty:
    print("No runs found."); exit()
print(f"Total runs: {len(runs)}")
cols = [c for c in runs.columns if any(x in c for x in
    ["run_id","status","params.epochs","params.learning_rate",
     "metrics.validation_accuracy","metrics.billable_seconds","metrics.cost_usd"])]
print(runs[cols].to_string(index=False))
