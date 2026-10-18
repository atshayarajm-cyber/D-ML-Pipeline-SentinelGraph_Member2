"""XGBoost baseline training and prediction interface for SentinelGraph."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

MODEL_VERSION = "xgb-v1"
FEATURE_NAMES = ("amount", "timestamp_step")
REQUIRED_PREDICTION_FIELDS = ("transaction_id", *FEATURE_NAMES)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / f"{MODEL_VERSION}.json"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_PATH.with_name("model_metadata.json")


class PredictionInputError(ValueError):
    """Raised when a prediction request does not match the baseline contract."""


def _validated_records(records: Any) -> tuple[list[Any], list[list[float]]]:
    if not isinstance(records, list):
        raise PredictionInputError("records must be a list of transaction dictionaries")
    if not records:
        return [], []

    transaction_ids: list[Any] = []
    feature_rows: list[list[float]] = []
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PredictionInputError(f"records[{row_index}] must be a dictionary-like object")

        missing = [field for field in REQUIRED_PREDICTION_FIELDS if field not in record]
        if missing:
            raise PredictionInputError(f"records[{row_index}] is missing required fields: {', '.join(missing)}")

        transaction_id = record["transaction_id"]
        if transaction_id is None or (isinstance(transaction_id, str) and not transaction_id.strip()):
            raise PredictionInputError(f"records[{row_index}].transaction_id must not be empty")

        row_features: list[float] = []
        for feature_name in FEATURE_NAMES:
            value = record[feature_name]
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise PredictionInputError(f"records[{row_index}].{feature_name} must be a finite number")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise PredictionInputError(f"records[{row_index}].{feature_name} must be a finite number")
            if feature_name == "amount" and numeric_value < 0:
                raise PredictionInputError(f"records[{row_index}].amount must be non-negative")
            if feature_name == "timestamp_step" and (numeric_value < 0 or not numeric_value.is_integer()):
                raise PredictionInputError(f"records[{row_index}].timestamp_step must be a non-negative integer")
            row_features.append(numeric_value)

        transaction_ids.append(transaction_id)
        feature_rows.append(row_features)

    return transaction_ids, feature_rows


def _load_saved_model(model_path: str | Path) -> Any:
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found at '{path}'. Train the baseline on the real dataset first.")

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required to load the saved SentinelGraph model; install requirements.txt") from exc

    model = XGBClassifier()
    model.load_model(path)
    saved_feature_names = model.get_booster().feature_names
    if saved_feature_names is None:
        raise ValueError("Saved model has no feature names; refusing inference with an unverifiable feature order")
    if tuple(saved_feature_names) != FEATURE_NAMES:
        raise ValueError(
            f"Saved model feature order {tuple(saved_feature_names)!r} does not match required order {FEATURE_NAMES!r}"
        )
    return model


def predict_transactions(
    records: list[dict[str, Any]],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[dict[str, Any]]:
    """Score records with the saved model, preserving IDs and input order.

    Prediction never fits or retrains a model. The model is loaded from disk for
    every non-empty call, and labels are not part of the input contract.
    """
    transaction_ids, feature_rows = _validated_records(records)
    if not records:
        return []

    model = _load_saved_model(model_path)
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to construct named model features; install requirements.txt") from exc

    feature_frame = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    probabilities = model.predict_proba(feature_frame)[:, 1]
    return [
        {
            "transaction_id": transaction_id,
            "risk_score": float(score),
            "model_version": MODEL_VERSION,
        }
        for transaction_id, score in zip(transaction_ids, probabilities, strict=True)
    ]


def _validate_training_frame(frame: Any, split_name: str, pandas_module: Any, numpy_module: Any) -> tuple[Any, Any]:
    if not isinstance(frame, pandas_module.DataFrame):
        raise TypeError(f"{split_name} must be a pandas DataFrame")
    required = {"transaction_id", "is_fraud", *FEATURE_NAMES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{split_name} must contain at least one row")
    if frame["transaction_id"].isna().any():
        raise ValueError(f"{split_name} contains missing transaction_id values")
    if frame["transaction_id"].duplicated().any():
        raise ValueError(f"{split_name} contains duplicate transaction_id values")

    features = frame.loc[:, FEATURE_NAMES].apply(pandas_module.to_numeric, errors="coerce")
    if features.isna().any().any() or not numpy_module.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{split_name} contains missing, non-numeric, or non-finite model features")
    if (features["amount"] < 0).any():
        raise ValueError(f"{split_name} contains negative amounts")
    step_values = features["timestamp_step"].to_numpy(dtype=float)
    if (step_values < 0).any() or not numpy_module.equal(step_values, numpy_module.floor(step_values)).all():
        raise ValueError(f"{split_name} timestamp_step values must be non-negative integers")

    labels = pandas_module.to_numeric(frame["is_fraud"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"{split_name} is_fraud values must be exactly 0 or 1")
    return features.astype(float), labels.astype(int).to_numpy()


def _select_validation_threshold(labels: Any, probabilities: Any, sklearn_metrics: Any) -> tuple[float, str]:
    if not (labels == 1).any():
        return 0.5, "validation split has no positive examples; default threshold 0.5 used"

    precision, recall, thresholds = sklearn_metrics.precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5, "validation data did not yield candidate thresholds; default threshold 0.5 used"
    denominator = precision[:-1] + recall[:-1]
    f1_scores = numpy_safe_f1(precision[:-1], recall[:-1], denominator)
    best_f1 = max(f1_scores)
    best_threshold = max(
        threshold for threshold, score in zip(thresholds, f1_scores, strict=True) if score == best_f1
    )
    return float(best_threshold), "threshold maximizes validation F1; ties use the higher threshold"


def numpy_safe_f1(precision: Sequence[float], recall: Sequence[float], denominator: Sequence[float]) -> list[float]:
    return [
        0.0 if total == 0 else float(2 * precision_value * recall_value / total)
        for precision_value, recall_value, total in zip(precision, recall, denominator, strict=True)
    ]


def _evaluate_split(labels: Any, probabilities: Any, threshold: float, sklearn_metrics: Any) -> dict[str, Any]:
    import numpy as np

    predictions = (probabilities >= threshold).astype(int)
    positive_count = int((labels == 1).sum())
    class_count = int(len(np.unique(labels)))
    predicted_positive_count = int(predictions.sum())

    precision: float | None
    if predicted_positive_count == 0:
        precision = None
    else:
        precision = float(sklearn_metrics.precision_score(labels, predictions, zero_division=0))
    recall = (
        float(sklearn_metrics.recall_score(labels, predictions, zero_division=0))
        if positive_count
        else None
    )
    f1 = float(sklearn_metrics.f1_score(labels, predictions, zero_division=0)) if positive_count else None
    average_precision = (
        float(sklearn_metrics.average_precision_score(labels, probabilities))
        if positive_count and class_count == 2
        else None
    )

    k = min(len(labels), max(1, math.ceil(0.01 * len(labels))))
    top_indices = np.argsort(-probabilities, kind="stable")[:k]
    top_positive_count = int(labels[top_indices].sum())
    precision_at_k = float(top_positive_count / k)
    recall_at_k = float(top_positive_count / positive_count) if positive_count else None
    confusion = sklearn_metrics.confusion_matrix(labels, predictions, labels=[0, 1]).tolist()

    return {
        "row_count": int(len(labels)),
        "class_distribution": {"0": int((labels == 0).sum()), "1": positive_count},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "k": int(k),
        "confusion_matrix_labels_0_1": [[int(value) for value in row] for row in confusion],
        "unavailable_metric_notes": (
            ["recall, F1, average precision, and recall@k are unavailable because this split has no positive examples."]
            if not positive_count
            else (["average precision is unavailable because this split contains only one class."] if class_count == 1 else [])
        ),
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _metric_text(value: Any) -> str:
    return "unavailable" if value is None else f"{value:.6f}"


def _write_evaluation_report(path: Path, metadata: Mapping[str, Any]) -> None:
    split_metrics = metadata["evaluation_metrics"]
    lines = [
        "# SentinelGraph Member 2 XGBoost Evaluation",
        "",
        f"- Model: `{metadata['model_version']}` ({metadata['model_type']})",
        f"- Trained at (UTC): `{metadata['training_timestamp_utc']}`",
        f"- Features, in order: `{', '.join(metadata['feature_names'])}`",
        f"- Validation-selected threshold: `{metadata['selected_validation_threshold']:.6f}`",
        f"- Threshold selection: {metadata['threshold_selection_note']}",
        f"- Dataset identity: `{json.dumps(metadata['dataset_identity'], sort_keys=True)}`",
        "",
        "## Chronological Splits",
        "",
        "| Split | Rows | Class 0 | Class 1 |",
        "|---|---:|---:|---:|",
    ]
    for split_name in ("train", "validation", "test"):
        metrics = split_metrics[split_name]
        distribution = metrics["class_distribution"]
        lines.append(f"| {split_name} | {metrics['row_count']} | {distribution['0']} | {distribution['1']} |")

    test_metrics = split_metrics["test"]
    lines.extend(
        [
            "",
            "## Held-Out Test Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Precision | {_metric_text(test_metrics['precision'])} |",
            f"| Recall | {_metric_text(test_metrics['recall'])} |",
            f"| F1 | {_metric_text(test_metrics['f1'])} |",
            f"| Average precision (PR-AUC summary) | {_metric_text(test_metrics['average_precision'])} |",
            f"| Precision@K (K={test_metrics['k']}) | {_metric_text(test_metrics['precision_at_k'])} |",
            f"| Recall@K (K={test_metrics['k']}) | {_metric_text(test_metrics['recall_at_k'])} |",
            "",
            "Confusion matrix uses rows=true labels `[0, 1]` and columns=predicted labels `[0, 1]`:",
            "",
            f"`{test_metrics['confusion_matrix_labels_0_1']}`",
            "",
            "## Training Configuration",
            "",
            "```json",
            json.dumps(metadata["training_configuration"], indent=2, sort_keys=True),
            "```",
            "",
            "## Validation Metrics",
            "",
            "```json",
            json.dumps(split_metrics["validation"], indent=2, sort_keys=True),
            "```",
            "",
            "## Limitations and Integration",
            "",
            "- This is the transaction-only baseline; graph and historical temporal features are not included.",
            "- `tx_behavior_id`, IDs, alert fields, investigation outcomes, and labels are excluded from predictors.",
            "- The baseline uses only `amount` and `timestamp_step`; it does not compute historical features.",
            "- Precision@K and recall@K use `K = ceil(1% of split rows)`, with a minimum K of 1.",
            "- Any unavailable single-class metrics are reported as unavailable, not imputed.",
            "- Backend field-name compatibility remains for the integration team to confirm; this interface returns `risk_score`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_and_evaluate(
    train_frame: Any,
    validation_frame: Any,
    test_frame: Any,
    output_dir: str | Path,
    dataset_identity: Mapping[str, Any] | None = None,
    split_definitions: Mapping[str, Any] | None = None,
    training_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the baseline, select threshold on validation, evaluate once on test, and save outputs."""
    try:
        import numpy as np
        import pandas as pd
        from sklearn import metrics as sklearn_metrics
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Training requires numpy, pandas, scikit-learn, and xgboost; install requirements.txt") from exc

    split_frames = {"train": train_frame, "validation": validation_frame, "test": test_frame}
    validated: dict[str, tuple[Any, Any]] = {
        name: _validate_training_frame(frame, name, pd, np) for name, frame in split_frames.items()
    }
    train_features, train_labels = validated["train"]
    validation_features, validation_labels = validated["validation"]
    test_features, test_labels = validated["test"]

    train_positive = int((train_labels == 1).sum())
    train_negative = int((train_labels == 0).sum())
    if not train_positive or not train_negative:
        raise ValueError("Training split must contain both fraud and non-fraud examples for binary XGBoost training")

    parameters: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "scale_pos_weight": train_negative / train_positive,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    if training_configuration:
        parameters.update(dict(training_configuration))
        parameters["scale_pos_weight"] = training_configuration.get(
            "scale_pos_weight", train_negative / train_positive
        )

    model = XGBClassifier(**parameters)
    model.fit(
        train_features,
        train_labels,
        eval_set=[(validation_features, validation_labels)],
        verbose=False,
    )
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    selected_threshold, threshold_note = _select_validation_threshold(
        validation_labels, validation_probabilities, sklearn_metrics
    )
    test_probabilities = model.predict_proba(test_features)[:, 1]

    metrics_by_split = {
        "train": _evaluate_split(
            train_labels, model.predict_proba(train_features)[:, 1], selected_threshold, sklearn_metrics
        ),
        "validation": _evaluate_split(
            validation_labels, validation_probabilities, selected_threshold, sklearn_metrics
        ),
        "test": _evaluate_split(test_labels, test_probabilities, selected_threshold, sklearn_metrics),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{MODEL_VERSION}.json"
    metadata_path = output_path / "model_metadata.json"
    report_path = output_path.parent / "reports" / "ml_evaluation.md"

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "XGBoost binary classifier",
        "feature_names": list(FEATURE_NAMES),
        "training_configuration": parameters,
        "dataset_identity": dict(dataset_identity or {}),
        "chronological_splits": dict(split_definitions or {}),
        "selected_validation_threshold": selected_threshold,
        "threshold_selection_note": threshold_note,
        "package_versions": {
            package: _package_version(package)
            for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "joblib")
        },
        "evaluation_metrics": metrics_by_split,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    model.save_model(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_evaluation_report(report_path, metadata)
    return {
        "model": model,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "report_path": report_path,
        "metadata": metadata,
    }
