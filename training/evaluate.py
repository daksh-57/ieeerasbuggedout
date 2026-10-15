import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


MODELS_DIR = Path("models")
METRICS_PATH = MODELS_DIR / "metrics.json"
CM_PATH = MODELS_DIR / "confusion_matrix.png"
MODEL_PATH = MODELS_DIR / "text_model.joblib"


def evaluate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_text.py first.")

    pipeline = joblib.load(MODEL_PATH)

    X_test = np.load(MODELS_DIR / "X_test.npy", allow_pickle=True)
    y_test = np.load(MODELS_DIR / "y_test.npy", allow_pickle=True)

    y_pred = pipeline.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "precision": {
            "class_0_ham": float(precision[0]),
            "class_1_phishing": float(precision[1]),
            "macro": float(precision.mean()),
        },
        "recall": {
            "class_0_ham": float(recall[0]),
            "class_1_phishing": float(recall[1]),
            "macro": float(recall.mean()),
        },
        "f1": {
            "class_0_ham": float(f1[0]),
            "class_1_phishing": float(f1[1]),
            "macro": float(f1.mean()),
        },
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "test_set_size": int(len(y_test)),
        "support": {
            "class_0_ham": int(support[0]),
            "class_1_phishing": int(support[1]),
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham (0)", "Phishing (1)"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Text Model (Test Set)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {CM_PATH}")

    print("\n=== Final Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (phishing): {metrics['precision']['class_1_phishing']:.4f}")
    print(f"Recall (phishing):    {metrics['recall']['class_1_phishing']:.4f}")
    print(f"F1 (phishing):        {metrics['f1']['class_1_phishing']:.4f}")
    print(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    evaluate()