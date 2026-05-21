"""
Deploy trained model to SageMaker real-time endpoint.
Run after train.py completes successfully.
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker import image_uris
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")

boto_session = boto3.Session(region_name=AWS_REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)


def deploy():
    """Deploy trained model to SageMaker endpoint."""

    # Load training job details
    if not os.path.exists("training_job.json"):
        raise FileNotFoundError("training_job.json not found — run train.py first")

    with open("training_job.json") as f:
        job_info = json.load(f)

    job_name = job_info["job_name"]
    model_artifacts = job_info["model_artifacts"]
    container = job_info["container"]

    print(f"Deploying model from job: {job_name}")
    print(f"Model artifacts: {model_artifacts}")

    # Endpoint name
    endpoint_name = f"canadian-sentiment-endpoint"

    # Create model
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)

    # Check if endpoint already exists — delete it first if so
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} exists — deleting and recreating...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
        print("Old endpoint deleted")
    except sm_client.exceptions.ClientError:
        pass  # Endpoint doesn't exist — fine

    # Create SageMaker model
    model_name = f"canadian-sentiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": container,
            "ModelDataUrl": model_artifacts,
        },
        ExecutionRoleArn=ROLE_ARN,
    )
    print(f"Model created: {model_name}")

    # Create endpoint config
    config_name = f"{model_name}-config"
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": "ml.t2.medium",   # Cheapest endpoint: ~$0.056/hour
            "InitialInstanceCount": 1,
        }],
    )
    print(f"Endpoint config created: {config_name}")

    # Create endpoint
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    print(f"\nEndpoint creating: {endpoint_name}")
    print("Waiting for endpoint to be InService (3-5 minutes)...")

    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 20},
    )

    print(f"\nEndpoint is LIVE: {endpoint_name}")
    print("Run predict.py to test it")

    # Save endpoint name
    with open("endpoint.json", "w") as f:
        json.dump({"endpoint_name": endpoint_name}, f, indent=2)

    print("Saved to endpoint.json")
    return endpoint_name


if __name__ == "__main__":
    deploy()
