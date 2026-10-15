"""Model utilities for saving, loading, and inspecting trained pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.train import DEFAULT_THRESHOLD, normalize_message, predict


logger = logging.getLogger(__name__)

MODEL_FILENAME = "phishing_pipeline.joblib"


def save_pipeline(pipeline, output_path: str = "artifacts/phishing_pipeline.joblib") -> Path:
    """Save a fitted pipeline to disk using joblib.

    Args:
        pipeline: A fitted sklearn Pipeline.
        output_path: Destination path for the serialized model.

    Returns:
        The absolute path to the saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s (%d bytes)", path, path.stat().st_size)
    return path


def load_pipeline(input_path: str = "artifacts/phishing_pipeline.joblib"):
    """Load a previously saved pipeline from disk.

    Args:
        input_path: Path to the joblib-serialized pipeline.

    Returns:
        The loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If the pipeline file does not exist.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found at {input_path}. Train the model first.")
    return joblib.load(path)


def predict_messages(
    pipeline,
    messages: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Run inference on a list of messages and return structured results.

    Args:
        pipeline: A fitted sklearn Pipeline.
        messages: List of raw message strings to classify.
        threshold: Probability threshold for the phishing class.

    Returns:
        A dictionary with keys:
            - predictions: list of int (0=legitimate, 1=phishing)
            - probabilities: list of float (probability of phishing)
            - messages: the input messages (for traceability)
    """
    preds, probs = predict(pipeline, messages, threshold=threshold)
    return {
        "predictions": preds.tolist(),
        "probabilities": [round(float(p), 4) for p in probs],
        "messages": messages,
    }


def inspect_pipeline(pipeline) -> dict:
    """Return human-readable information about a fitted pipeline.

    Args:
        pipeline: A fitted sklearn Pipeline.

    Returns:
        Dictionary with pipeline structure details.
    """
    info = {
        "steps": [(name, type(step).__name__) for name, step in pipeline.steps],
        "feature_names": None,
        "n_features": None,
    }
    # Access TF-IDF vectorizer to get feature info.
    for name, step in pipeline.steps:
        if hasattr(step, "get_feature_names_out"):
            try:
                names = step.get_feature_names_out()
                info["feature_names"] = names[:10]  # First 10 features
                info["n_features"] = len(names)
                break
            except Exception:
                pass
    return info
