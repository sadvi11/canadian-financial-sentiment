"""
Delete SageMaker endpoint to stop charges.
Run this when you are done testing.
ml.t2.medium costs ~$0.056/hour — delete when not in use.
"""

import boto3
import os
import json
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT_NAME = "canadian-sentiment-endpoint"


def delete_endpoint():
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleted endpoint: {ENDPOINT_NAME}")
        print("No more charges for this endpoint.")
    except sm.exceptions.ClientError as e:
        print(f"Endpoint not found or already deleted: {e}")


if __name__ == "__main__":
    confirm = input(f"Delete endpoint '{ENDPOINT_NAME}'? (yes/no): ")
    if confirm.lower() == "yes":
        delete_endpoint()
    else:
        print("Cancelled.")
