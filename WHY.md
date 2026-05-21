# WHY.md — The Decisions Behind This Project

> Most ML portfolios show a model with good accuracy. This document explains why every architectural and technical decision was made — the engineering thinking behind the pipeline.

---

## Why I Built This Project

My SmartMoney Canada platform (100K+ views) teaches financial literacy to Canadians. My audience watches videos about investing in Canadian stocks — TD Bank, RBC, Shopify, Suncor. After every video, the same question appears in comments:

*"Is the news about this company good or bad for my investment?"*

They are reading headlines they cannot interpret. A sentiment classifier solves this — upload a headline, get a clear signal: positive, negative, or neutral. No finance degree required.

The secondary reason: every cloud + AI job posting in Canada mentions SageMaker. Most candidates claim SageMaker knowledge from reading documentation. I built an end-to-end pipeline — data preparation, training job, endpoint deployment, inference, API, and experiment tracking — and ran it in production on real AWS infrastructure.

---

## Why AWS SageMaker Instead of Training Locally

I could have trained a sentiment classifier on my laptop using scikit-learn or PyTorch in 5 minutes. I did not.

Local training is a demo. SageMaker is production.

When a Canadian bank, insurance company, or fintech trains ML models, they use managed infrastructure — SageMaker, Vertex AI, Azure ML. The data scientist who only trains locally cannot operate in those environments. SageMaker forces you to think about:

- Data pipelines (S3 storage, data channels)
- IAM permissions (least privilege for training jobs)
- Infrastructure as code (instance types, resource limits)
- Cost management (billable seconds, instance selection)
- Reproducibility (job names, artifact paths, model registry)

None of those exist when you run `model.fit(X_train, y_train)` on your laptop.

---

## Why BlazingText Instead of a Pre-Trained LLM

Using Claude or GPT-4 for sentiment classification would produce 95%+ accuracy on this dataset. I deliberately chose BlazingText instead.

Three reasons:

**Cost at scale.** An LLM API call costs $0.001–0.01 per classification. BlazingText inference on a deployed endpoint costs $0.000001 per call. For a platform processing thousands of headlines per day, the cost difference is 100–1000x.

**Latency.** BlazingText returns a prediction in under 10 milliseconds. LLM API calls take 500ms–2 seconds. For real-time financial news processing, latency matters.

**Learning value.** Using an LLM for classification is a one-line API call. Training a classifier teaches data pipeline design, hyperparameter tuning, evaluation metrics, and model lifecycle management — the skills ML Associate certification and ML engineering roles require.

The honest tradeoff: BlazingText accuracy is lower (66.7% on 135 examples vs 95%+ for LLM). That tradeoff is intentional and documented — it is a feature of the design, not a failure.

---

## Why 135 Training Examples

135 examples is not a production dataset. It is intentionally a starting point.

The goal of this project is to demonstrate a production-grade ML pipeline architecture, not to achieve maximum accuracy. 135 examples proves:

- The pipeline ingests, formats, and uploads data correctly
- BlazingText trains and converges without errors
- The endpoint serves predictions in production
- MLflow tracks experiments across runs

Accuracy scales with data. The pipeline is the hard part — adding more labeled examples is straightforward once the infrastructure works. This is the same pattern used at every ML team: build the pipeline first, improve the data second.

---

## Why Numbered Folders (01_data, 02_training...)

Standard ML project structures use flat folders (data/, models/, src/) with no implied order. A new engineer or recruiter reading the repository has to read documentation to understand what runs first.

Numbered folders make the pipeline order unambiguous without reading a single line of documentation:

```
01_data → 02_training → 03_deployment → 04_inference → 05_api → 06_experiments
```

The sequence is visible from the file tree alone. This is operational clarity — the same principle that makes Nokia's network function deployment procedures readable by any engineer, regardless of prior context.

---

## Why MLflow for Experiment Tracking

The first training run produced 48.15% accuracy. That number is useless without context.

Is 48.15% good or bad? Compared to what? What hyperparameters produced it? How long did it take? How much did it cost?

MLflow answers all of these questions and stores the answers persistently. Every training run logs:

- All hyperparameters (epochs, learning rate, word n-grams)
- All metrics (train accuracy, validation accuracy)
- Infrastructure config (instance type, training duration)
- Cost estimate (billable seconds × instance hourly rate)
- Model artifact location (S3 URI)

When the accuracy improves from 48% to 70% by adding more training data, MLflow shows exactly which run produced the improvement and what changed. Without experiment tracking, that knowledge lives only in memory — and memory is unreliable.

---

## Why Delete the Endpoint After Testing

ml.t2.medium costs $0.056/hour = $1.34/day = $490/year.

A portfolio project that is not serving production traffic should not run a production endpoint continuously. The responsible pattern is: deploy → test → delete. The `03_deployment/delete_endpoint.py` script exists because cost discipline is part of cloud engineering — not an afterthought.

In production, the endpoint would stay running because it serves real traffic. For a portfolio project, demonstrating that you understand the cost model is more valuable than leaving an endpoint running to prove it works.

---

## Why Canadian Financial Headlines Specifically

Generic sentiment datasets (Twitter sentiment, movie reviews, product reviews) train models that do not understand financial language.

The word "bearish" is negative in finance. "Yield" can be positive or negative depending on context. "Covenant breach" is catastrophic. "Dividend reinstatement" is excellent.

Financial text requires financial training data. Canadian financial text requires Canadian examples — TD Bank, RBC, Bank of Canada rate decisions, Alberta oil sands, SmartMoney Canada's audience.

Using generic data for a financial use case produces a model that cannot reliably distinguish between "Bank of Canada raises rates" (negative for homeowners, potentially positive for savers) and "bank robber raises cash." Domain-specific training data is not optional — it is the difference between a demo and a product.

---

## The Nokia Bridge — Why My Background Is Relevant

I spent 2.5 years at Nokia dimensioning compute, memory, and network resources for 5G Packet Core deployments serving 100K+ subscribers. That is capacity planning.

Choosing `ml.m4.xlarge` for a 135-example BlazingText training job that completes in 89 seconds and costs $0.007 is also capacity planning. The mental model is identical: translate a workload requirement into infrastructure specifications and cost estimates.

CBAM at Nokia manages VNF lifecycle — deploy, monitor, scale, update, terminate. SageMaker manages ML model lifecycle — train, evaluate, deploy, monitor, update, delete. The orchestration pattern is the same. The technology is different.

This is not a stretch. It is the same engineering discipline applied to a different domain.

---

*Architecture decisions without documentation are institutional knowledge that disappears when engineers leave. This document exists so the reasoning is permanent.*

**Sadhvi Sharma** | Calgary, AB | github.com/sadvi11
