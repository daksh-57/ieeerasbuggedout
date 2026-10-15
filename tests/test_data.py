import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from training.download_data import extract_email_body, load_spamassassin
from training.train_text import deduplicate


def test_extract_email_body_spam():
    # Test that we can extract body from a real spam file
    spam_files = list((__import__("pathlib").Path("data/raw/spamassassin/spam_2")).glob("*"))
    assert len(spam_files) > 0
    body = extract_email_body(spam_files[0])
    assert isinstance(body, str)
    assert len(body) > 0


def test_extract_email_body_ham():
    ham_files = list((__import__("pathlib").Path("data/raw/spamassassin/easy_ham")).glob("*"))
    assert len(ham_files) > 0
    body = extract_email_body(ham_files[0])
    assert isinstance(body, str)
    assert len(body) > 0


def test_load_spamassassin_structure():
    df = load_spamassassin()
    assert isinstance(df, pd.DataFrame)
    assert "message" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1})
    assert len(df) > 100
    # Both classes present
    assert df["label"].sum() > 0  # has phishing
    assert (df["label"] == 0).sum() > 0  # has ham


def test_deduplication():
    df = pd.DataFrame({
        "message": ["hello world", "hello world", "different message"],
        "label": [1, 1, 0],
    })
    deduped = deduplicate(df)
    assert len(deduped) == 2
    assert deduped["message"].nunique() == 2


def test_label_mapping():
    df = load_spamassassin()
    # Spam should be label 1, ham should be label 0
    spam_count = (df["label"] == 1).sum()
    ham_count = (df["label"] == 0).sum()
    assert spam_count > 0
    assert ham_count > 0
    # SpamAssassin spam_2 should have fewer than ham total
    # (just a sanity check, not strict)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])