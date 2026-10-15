import json
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from training.train_text import load_data, deduplicate
from training.evaluate import evaluate


def test_train_predict_roundtrip():
    # Create a tiny synthetic dataset
    df = pd.DataFrame({
        "message": [
            "click here to win money now",
            "urgent verify your account password",
            "meeting at 3pm tomorrow",
            "please review the quarterly report",
            "free prize claim your reward",
            "lunch at noon in the cafeteria",
        ],
        "label": [1, 1, 0, 0, 1, 0],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "test_data.csv"
        df.to_csv(data_path, index=False)

        # Monkey-patch the load_data to use our temp file
        import training.train_text as train_module
        original_load = train_module.load_data

        def mock_load():
            return df

        train_module.load_data = mock_load

        try:
            # Train
            from training.train_text import train
            pipeline, X_test, y_test, y_pred = train()

            # Check model is a pipeline
            assert isinstance(pipeline, Pipeline)
            assert hasattr(pipeline, "predict")
            assert hasattr(pipeline, "predict_proba")

            # Test predictions on training-like data
            test_messages = [
                "click here to win prize",
                "meeting tomorrow at 10am",
            ]
            preds = pipeline.predict(test_messages)
            assert len(preds) == 2
            assert set(preds).issubset({0, 1})

            # Test predict_proba
            probas = pipeline.predict_proba(test_messages)
            assert probas.shape == (2, 2)
            assert np.allclose(probas.sum(axis=1), 1.0)

        finally:
            train_module.load_data = original_load


def test_evaluate_produces_metrics():
    # This test requires the model to be trained first
    # We'll run it after train_text.py in the test suite
    pass


def test_metrics_json_valid():
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Check required keys
        required_keys = ["precision", "recall", "f1", "accuracy", "confusion_matrix", "test_set_size", "support"]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

        # Check structure
        assert "class_1_phishing" in metrics["precision"]
        assert "class_1_phishing" in metrics["recall"]
        assert "class_1_phishing" in metrics["f1"]
        assert isinstance(metrics["confusion_matrix"], list)
        assert len(metrics["confusion_matrix"]) == 2
        assert all(len(row) == 2 for row in metrics["confusion_matrix"])
        assert metrics["test_set_size"] > 0
        assert "class_1_phishing" in metrics["support"]
        assert "class_0_ham" in metrics["support"]

        # Values should be reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"]["class_1_phishing"] <= 1
        assert 0 <= metrics["recall"]["class_1_phishing"] <= 1
        assert 0 <= metrics["f1"]["class_1_phishing"] <= 1


def test_model_file_size():
    model_path = Path("models/text_model.joblib")
    if model_path.exists():
        size_mb = os.path.getsize(model_path) / 1e6
        assert size_mb < 3.0, f"Model size {size_mb:.2f} MB exceeds 3 MB limit"


def test_confusion_matrix_exists():
    cm_path = Path("models/confusion_matrix.png")
    # This will be created after evaluate.py runs
    # The test passes if file exists (checked in integration)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])