"""
Test the deployed SageMaker endpoint with Canadian financial headlines.
Run after deploy.py completes successfully.
"""

import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ── Test headlines ────────────────────────────────────────────────────────────

TEST_HEADLINES = [
    # Should be POSITIVE
    "TD Bank reports record quarterly profit beating analyst expectations",
    "Shopify revenue surges 25 percent driven by strong merchant growth",
    "Canadian economy adds 40000 jobs in March beating all forecasts",

    # Should be NEGATIVE
    "TD Bank fined 3 billion for money laundering compliance failures",
    "Canadian housing market crashes as mortgage defaults rise sharply",
    "Shopify announces mass layoffs cutting 20 percent of global workforce",

    # Should be NEUTRAL
    "Bank of Canada holds overnight rate at 5 percent as expected",
    "Statistics Canada releases monthly employment report for April",
    "RBC announces Q2 2024 earnings release date of August 22",
]


def predict(endpoint_name: str, headline: str) -> dict:
    """Call SageMaker endpoint for sentiment prediction."""
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

    payload = {"instances": [headline.lower()]}

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode())
    return result[0]


def run_tests():
    """Run all test headlines through the endpoint."""

    # Load endpoint name
    if not os.path.exists("endpoint.json"):
        raise FileNotFoundError("endpoint.json not found — run deploy.py first")

    with open("endpoint.json") as f:
        endpoint_name = json.load(f)["endpoint_name"]

    print(f"Testing endpoint: {endpoint_name}")
    print("=" * 60)

    correct = 0
    total = len(TEST_HEADLINES)

    for i, headline in enumerate(TEST_HEADLINES):
        result = predict(endpoint_name, headline)
        label = result["label"][0].replace("__label__", "").upper()
        prob = result["prob"][0]

        # Expected label based on position
        if i < 3:
            expected = "POSITIVE"
        elif i < 6:
            expected = "NEGATIVE"
        else:
            expected = "NEUTRAL"

        match = "✅" if label == expected else "❌"
        if label == expected:
            correct += 1

        print(f"{match} [{label}] (confidence: {prob:.2%})")
        print(f"   {headline[:70]}")
        print()

    accuracy = correct / total
    print("=" * 60)
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"Endpoint: {endpoint_name}")
    print()
    print("⚠️  REMEMBER: Delete endpoint when done to avoid charges:")
    print(f"   python3 delete_endpoint.py")


if __name__ == "__main__":
    run_tests()
