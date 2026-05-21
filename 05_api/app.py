"""
Flask REST API — Canadian Financial Sentiment
Exposes the SageMaker endpoint via REST API.

Endpoints:
  POST /predict    Classify a financial headline
  GET  /examples   Run sample predictions
  GET  /health     Service health check
"""

import boto3
import json
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "canadian-sentiment-endpoint")

runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

EXAMPLES = [
    "TD Bank reports record quarterly profit beating analyst expectations",
    "Canadian housing market crashes as mortgage defaults rise sharply",
    "Bank of Canada holds overnight rate at 5 percent as expected",
]


def call_endpoint(headline: str) -> dict:
    """Call SageMaker endpoint and return prediction."""
    payload = {"instances": [headline.lower()]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode())
    return result[0]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": "canadian-financial-sentiment",
        "version": "1.0",
        "status": "healthy",
        "endpoint": ENDPOINT_NAME,
        "algorithm": "AWS BlazingText (supervised text classification)",
        "labels": ["positive", "negative", "neutral"],
        "use_case": "Canadian Financial News Sentiment — SmartMoney Canada",
        "region": AWS_REGION,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a Canadian financial headline.
    Request: {"headline": "TD Bank reports record profit"}
    """
    data = request.get_json()
    if not data or "headline" not in data:
        return jsonify({"error": "headline field required"}), 400

    headline = data["headline"].strip()
    if not headline:
        return jsonify({"error": "headline cannot be empty"}), 400

    logger.info(f"Predicting: {headline[:60]}")
    result = call_endpoint(headline)

    label = result["label"][0].replace("__label__", "")
    confidence = result["prob"][0]

    return jsonify({
        "status": "success",
        "headline": headline,
        "sentiment": label,
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence:.1%}",
        "model": "BlazingText via AWS SageMaker",
        "endpoint": ENDPOINT_NAME,
    }), 200


@app.route("/examples", methods=["GET"])
def examples():
    """Run sample predictions on Canadian financial headlines."""
    predictions = []
    for headline in EXAMPLES:
        result = call_endpoint(headline)
        label = result["label"][0].replace("__label__", "")
        confidence = result["prob"][0]
        predictions.append({
            "headline": headline,
            "sentiment": label,
            "confidence": f"{confidence:.1%}",
        })

    return jsonify({
        "status": "success",
        "examples": predictions,
        "powered_by": "AWS SageMaker BlazingText",
    }), 200


@app.route("/batch", methods=["POST"])
def batch_predict():
    """
    Classify multiple headlines at once.
    Request: {"headlines": ["headline 1", "headline 2"]}
    """
    data = request.get_json()
    if not data or "headlines" not in data:
        return jsonify({"error": "headlines field required (list)"}), 400

    headlines = data["headlines"]
    if not isinstance(headlines, list) or len(headlines) == 0:
        return jsonify({"error": "headlines must be a non-empty list"}), 400

    results = []
    for headline in headlines[:10]:        # max 10 per request
        result = call_endpoint(headline)
        label = result["label"][0].replace("__label__", "")
        confidence = result["prob"][0]
        results.append({
            "headline": headline,
            "sentiment": label,
            "confidence": f"{confidence:.1%}",
        })

    return jsonify({
        "status": "success",
        "count": len(results),
        "predictions": results,
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5003))
    logger.info(f"Starting Canadian Financial Sentiment API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
