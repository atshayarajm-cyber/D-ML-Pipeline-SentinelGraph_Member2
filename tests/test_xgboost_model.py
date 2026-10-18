from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinelgraph.models import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_VERSION,
    PredictionInputError,
    predict_transactions,
)
from sentinelgraph.models import xgboost_model  # noqa: E402


VALID_RECORDS = [
    {"transaction_id": "TX-001", "amount": 1500.0, "timestamp_step": 42},
    {"transaction_id": "TX-002", "amount": 25, "timestamp_step": 43},
]


class StubBooster:
    feature_names = list(FEATURE_NAMES)


class StubModel:
    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.probabilities = probabilities or [0.87, 0.12]
        self.seen_columns: list[str] | None = None

    def get_booster(self) -> StubBooster:
        return StubBooster()

    def predict_proba(self, frame: object) -> np.ndarray:
        self.seen_columns = list(frame.columns)
        return np.asarray([[1 - score, score] for score in self.probabilities])


def test_prediction_output_schema_and_id_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = StubModel()
    monkeypatch.setattr(xgboost_model, "_load_saved_model", lambda _path: model)

    results = predict_transactions(VALID_RECORDS)

    assert [result["transaction_id"] for result in results] == ["TX-001", "TX-002"]
    assert all(set(result) == {"transaction_id", "risk_score", "model_version"} for result in results)
    assert all(isinstance(result["risk_score"], float) for result in results)
    assert all(0.0 <= result["risk_score"] <= 1.0 for result in results)
    assert all(result["model_version"] == MODEL_VERSION for result in results)
    assert model.seen_columns == list(FEATURE_NAMES)


def test_empty_batch_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_loaded(_path: object) -> object:
        raise AssertionError("empty batch must not load a model")

    monkeypatch.setattr(xgboost_model, "_load_saved_model", fail_if_loaded)
    assert predict_transactions([]) == []


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"transaction_id": "TX-001", "amount": 1.0}, "timestamp_step"),
        ({"amount": 1.0, "timestamp_step": 1}, "transaction_id"),
        ({"transaction_id": "TX-001", "amount": True, "timestamp_step": 1}, "amount"),
        ({"transaction_id": "TX-001", "amount": float("nan"), "timestamp_step": 1}, "amount"),
        ({"transaction_id": "TX-001", "amount": 1.0, "timestamp_step": float("inf")}, "timestamp_step"),
        ({"transaction_id": "TX-001", "amount": -1.0, "timestamp_step": 1}, "non-negative"),
        ({"transaction_id": "TX-001", "amount": 1.0, "timestamp_step": 1.5}, "integer"),
        (None, "dictionary-like"),
    ],
)
def test_invalid_records_raise_clear_errors(record: object, message: str) -> None:
    with pytest.raises(PredictionInputError, match=message):
        predict_transactions([record])  # type: ignore[list-item]


def test_missing_model_artifact_raises_without_fallback(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Train the baseline"):
        predict_transactions(VALID_RECORDS, model_path=missing_path)


def test_saved_model_loader_loads_artifact_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "saved.json"
    artifact.write_text("test fixture", encoding="utf-8")
    events: list[tuple[str, str]] = []

    class FakeXGBClassifier:
        def load_model(self, path: Path) -> None:
            events.append(("load", str(path)))

        def get_booster(self) -> StubBooster:
            return StubBooster()

        def fit(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("prediction loader must not train")

    fake_xgboost = types.ModuleType("xgboost")
    fake_xgboost.XGBClassifier = FakeXGBClassifier  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)

    loaded = xgboost_model._load_saved_model(artifact)

    assert isinstance(loaded, FakeXGBClassifier)
    assert events == [("load", str(artifact))]


def test_feature_order_is_the_training_and_inference_contract() -> None:
    assert FEATURE_NAMES == ("amount", "timestamp_step")
    model = StubModel()
    assert tuple(model.get_booster().feature_names) == FEATURE_NAMES


def test_malformed_top_level_request_is_rejected() -> None:
    with pytest.raises(PredictionInputError, match="records must be a list"):
        predict_transactions((VALID_RECORDS[0],))  # type: ignore[arg-type]
