"""Log existing training job to MLflow."""
import sys, mlflow, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("canadian-financial-sentiment")
job = json.loads((Path(__file__).parent.parent / "artifacts" / "training_job.json").read_text())
with mlflow.start_run(run_name=job["job_name"]) as run:
    mlflow.log_params({"mode":"supervised","epochs":10,"learning_rate":0.05,
        "word_ngrams":2,"vector_dim":10,"min_count":2,
        "instance_type":"ml.m4.xlarge","training_examples":108,"validation_examples":27})
    mlflow.log_metrics({"train_accuracy":0.4815,"validation_accuracy":0.4815,
        "billable_seconds":94,"cost_usd_estimate":0.007})
    mlflow.set_tags({"algorithm":"BlazingText","dataset":"Canadian Financial Headlines",
        "use_case":"SmartMoney Canada","framework":"AWS SageMaker"})
    mlflow.log_param("model_s3_uri", job["model_artifacts"])
    print(f"Logged — Run ID: {run.info.run_id}")
