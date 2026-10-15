import os
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


RAW_DIR = Path("data/raw")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> pd.DataFrame:
    path = RAW_DIR / "spamassassin_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}. Run download_data.py first.")
    df = pd.read_csv(path)
    df = df.dropna(subset=["message", "label"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() >= 10]
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["message"])
    after = len(df)
    if before != after:
        print(f"Deduplicated: removed {before - after} duplicate messages")
    return df


def train():
    df = load_data()
    print(f"Loaded {len(df)} emails (phishing={int(df.label.sum())}, ham={len(df)-int(df.label.sum())})")

    df = deduplicate(df)

    X = df["message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(y), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=2,
    )
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    model_path = MODELS_DIR / "text_model.joblib"
    joblib.dump(pipeline, model_path)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"Model saved to {model_path} ({size_mb:.2f} MB)")

    if size_mb >= 3.0:
        warnings.warn(f"Model size ({size_mb:.2f} MB) exceeds 3 MB limit!")

    y_pred = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={acc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_test.npy", y_test)

    return pipeline, X_test, y_test, y_pred


if __name__ == "__main__":
    train()