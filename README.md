# SentinelGraph Member 2: AML Risk Baseline

A standalone preparation project for a transaction-only XGBoost binary classifier. The initial baseline uses exactly `amount` and `timestamp_step`, in that order. It excludes transaction/account identifiers, row indexes, `is_fraud`, alert/investigation outcomes, and `tx_behavior_id`.

## Runtime and Installation

Use Python 3.10–3.13. In a virtual environment:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
```

The current workspace interpreter may differ; use a supported runtime when installing the pinned-compatible dependency ranges.

## Colab / Jupyter

Open `notebooks/SentinelGraph_Member2_XGBoost.ipynb` in Jupyter or upload it to Google Colab. In Colab, install dependencies in the notebook's setup cell. Provide the real dataset by uploading a ZIP or mounting Drive and setting `DATA_DIR` to the directory containing the six required dataset files.

Required files: `transactions.parquet`, `labels.parquet`, `splits.parquet`, `accounts.parquet`, `manifest.json`, and `split_config.json`. No replacement or sample files are generated. If any are missing or invalid, the notebook reports the problem and stops before model training.

## Validate, Train, and Evaluate

Run the notebook cells in order. It loads the parquet files and JSON metadata, validates schemas, uniqueness and missing IDs, one-to-one join coverage, binary labels, numerical features, split names, configured chronological ranges, and actual timestamp-step membership. The only permitted splits are train `0–139`, validation `140–169`, and test `170–199`; conflicts stop execution. The notebook uses these chronological assignments, never a random split.

The baseline uses class weighting derived only from the training split and validation-based early stopping. It selects a threshold by validation F1, freezes that threshold, and evaluates the held-out test split once. Precision, recall, F1, average precision, Precision@K, Recall@K, confusion matrix, distributions, configuration, and limitations are written to `reports/ml_evaluation.md` only after successful training. Metrics that are undefined for a single-class split are recorded as unavailable.

Successful training creates `models/xgb-v1.json` and `models/model_metadata.json`. No model artifact or metrics report is checked in as a placeholder. The metadata records feature order, configuration, dataset identity when available, split definitions, validation threshold, package versions, metrics, and training timestamp.

## Prediction Interface

Add `src` to `PYTHONPATH` or run from the project root, then:

```python
from sentinelgraph.models import predict_transactions

scores = predict_transactions([
    {"transaction_id": "TX-AML1M-123", "amount": 1500.0, "timestamp_step": 42}
])
```

The result preserves the supplied ID and order and has the shape `{"transaction_id": original_id, "risk_score": float_0_to_1, "model_version": "xgb-v1"}`. The required inference features are exactly `amount`, then `timestamp_step`; `is_fraud` is not required. Empty input returns `[]`. Missing fields, malformed requests, negative amount, non-finite/nonnumeric features, and invalid time steps raise clear errors. A missing artifact raises `FileNotFoundError`; prediction never retrains and never fabricates a fallback score. The default artifact path is `models/xgb-v1.json`.

Run dataset-independent tests from this directory:

```bash
python -m pytest -q
```

The tests use a mocked model and do not require the AMLSim data or a trained production artifact.

## Graph / Temporal Work and Pending Deliverables

The standalone workspace does not include Member 3's `graph_features.py` or `temporal_features.py`. The notebook checks for those modules and skips Models B/C unless their implementations and a leakage-safe integration adapter are available. Do not recreate those pipelines independently. Historical features must be computed from permitted prior information, with a team-approved batch rule for transactions at the same timestamp. `tx_behavior_id` remains excluded from the baseline and requires a separately reviewed leakage experiment.

Pending until team inputs arrive: validation against the actual dataset and split configuration; model training, evaluation metrics, saved model and metadata; graph/temporal models; and backend integration confirmation.

## Backend Handoff

Prediction contract: `predict_transactions(records)` receives a list of dictionaries with `transaction_id`, `amount`, and `timestamp_step`; returns a list of ordinary dictionaries with `transaction_id`, `risk_score`, and `model_version`. The saved model is `models/xgb-v1.json`; feature order is `amount`, `timestamp_step`. No API files are modified here.

Confirm with Member 4 before integration:

- Does the backend expect `risk_score` or `model_score`, and where should any mapping occur?
- What transaction-ID type and serialization does the API guarantee?
- How will the artifact path be configured in deployed environments?
- Should predictions load the model once at service startup or per call?
- What batching, latency, concurrency, and missing-feature error conventions are required?
- Is a probability score calibrated/consumed directly, or is a separate thresholded decision field expected?
