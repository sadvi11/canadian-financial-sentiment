import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "canadian-sentiment-endpoint")
S3_PREFIX = "canadian-financial-sentiment"
S3_DATA_PATH = f"s3://{S3_BUCKET}/{S3_PREFIX}/data"
S3_MODEL_PATH = f"s3://{S3_BUCKET}/{S3_PREFIX}/model"
ROOT_DIR = Path(__file__).parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
TRAINING_JOB_FILE = ARTIFACTS_DIR / "training_job.json"
ENDPOINT_FILE = ARTIFACTS_DIR / "endpoint.json"
API_PORT = int(os.getenv("PORT", 5003))

def validate():
    missing = [k for k, v in {"S3_BUCKET_NAME": S3_BUCKET, "SAGEMAKER_ROLE_ARN": SAGEMAKER_ROLE_ARN}.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing: {', '.join(missing)} — fill in .env file")
    print(f"Config OK — Region: {AWS_REGION}, Bucket: {S3_BUCKET}")
