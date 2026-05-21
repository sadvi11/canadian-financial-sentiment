"""
SageMaker Training Job — Canadian Financial Sentiment Classifier
Uses AWS BlazingText built-in algorithm for text classification.

Run after data_prep.py has uploaded data to S3.
"""

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")

# ── SageMaker session ─────────────────────────────────────────────────────────

boto_session = boto3.Session(region_name=AWS_REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)


def train():
    """Launch BlazingText training job on SageMaker."""

    # BlazingText container image for us-east-1
    container = image_uris.retrieve("blazingtext", AWS_REGION, version="1")
    print(f"BlazingText container: {container}")

    # Job name with timestamp
    job_name = f"canadian-sentiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Training job: {job_name}")

    # Data channels
    data_prefix = f"s3://{S3_BUCKET}/canadian-financial-sentiment/data"
    output_path = f"s3://{S3_BUCKET}/canadian-financial-sentiment/model"

    # Estimator
    estimator = Estimator(
        image_uri=container,
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m4.xlarge",    # ~$0.10/hour — cheapest non-free instance
        volume_size=5,                    # GB
        max_run=600,                      # 10 min max (safety limit)
        output_path=output_path,
        sagemaker_session=sagemaker_session,
        base_job_name="canadian-sentiment",
    )

    # BlazingText hyperparameters for text classification
    estimator.set_hyperparameters(
        mode="supervised",          # text classification mode
        epochs=10,
        min_count=2,                # ignore words appearing < 2 times
        learning_rate=0.05,
        word_ngrams=2,              # bigrams improve accuracy
        vector_dim=10,              # small = faster training
        early_stopping=True,
        patience=4,
        min_epochs=5,
        eval_metric="accuracy",
    )

    # Data input channels
    train_data = sagemaker.inputs.TrainingInput(
        s3_data=f"{data_prefix}/train.txt",
        content_type="text/plain",
    )
    val_data = sagemaker.inputs.TrainingInput(
        s3_data=f"{data_prefix}/validation.txt",
        content_type="text/plain",
    )

    print("Starting training job...")
    print(f"Training data: {data_prefix}/train.txt")
    print(f"Model output: {output_path}")
    print("This takes approximately 3-5 minutes...")

    # Launch training (wait=True blocks until complete)
    estimator.fit(
        {"train": train_data, "validation": val_data},
        job_name=job_name,
        wait=True,
        logs="All",
    )

    print(f"\nTraining complete!")
    print(f"Job name: {job_name}")
    print(f"Model artifacts: {output_path}/{job_name}/output/model.tar.gz")

    # Save job name for deploy.py
    with open("training_job.json", "w") as f:
        json.dump({
            "job_name": job_name,
            "model_artifacts": f"{output_path}/{job_name}/output/model.tar.gz",
            "container": container,
        }, f, indent=2)

    print("Saved job details to training_job.json")
    print("Run deploy.py next")
    return estimator


if __name__ == "__main__":
    print("Canadian Financial Sentiment — SageMaker Training")
    print(f"Role: {ROLE_ARN}")
    print(f"Bucket: {S3_BUCKET}")
    print(f"Region: {AWS_REGION}")
    print()
    train()
