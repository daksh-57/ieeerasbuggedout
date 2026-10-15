import os
import re
import hashlib
import warnings
import pandas as pd
from pathlib import Path
from email import policy
from email.parser import BytesParser


SPAMASSASSIN_DIR = Path("data/raw/spamassassin")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HAM_DIRS = ["easy_ham", "easy_ham_2", "hard_ham"]
SPAM_DIRS = ["spam_2"]

SECOND_DATASET_URL = "https://raw.githubusercontent.com/andrewtavis/phishing-dataset/main/data/phishing_email.csv"
SECOND_DATASET_PATH = RAW_DIR / "phishing_email_cross.csv"


def extract_email_body(filepath: Path) -> str:
    try:
        with open(filepath, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode(errors="ignore"))
            return "\n".join(parts)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode(errors="ignore")
            return ""
    except Exception:
        try:
            with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
                content = f.read()
            lines = content.split("\n")
            body_start = 0
            for i, line in enumerate(lines):
                if line.strip() == "":
                    body_start = i + 1
                    break
            return "\n".join(lines[body_start:])
        except Exception:
            return ""


def load_spamassassin() -> pd.DataFrame:
    rows = []
    seen_hashes = set()

    for subdir in SPAM_DIRS:
        dirpath = SPAMASSASSIN_DIR / subdir
        if not dirpath.exists():
            continue
        for fname in os.listdir(dirpath):
            fpath = dirpath / fname
            if not fpath.is_file():
                continue
            body = extract_email_body(fpath)
            if not body or len(body.strip()) < 10:
                continue
            h = hashlib.md5(body.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            rows.append({"message": body.strip(), "label": 1})

    for subdir in HAM_DIRS:
        dirpath = SPAMASSASSIN_DIR / subdir
        if not dirpath.exists():
            continue
        for fname in os.listdir(dirpath):
            fpath = dirpath / fname
            if not fpath.is_file():
                continue
            body = extract_email_body(fpath)
            if not body or len(body.strip()) < 10:
                continue
            h = hashlib.md5(body.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            rows.append({"message": body.strip(), "label": 0})

    df = pd.DataFrame(rows)
    print(f"SpamAssassin loaded: {len(df)} emails (phishing={sum(df.label)}, ham={len(df)-sum(df.label)})")
    return df


def download_second_dataset() -> pd.DataFrame | None:
    import requests
    try:
        print(f"Downloading cross-dataset from {SECOND_DATASET_URL}...")
        resp = requests.get(SECOND_DATASET_URL, timeout=30)
        resp.raise_for_status()
        SECOND_DATASET_PATH.write_bytes(resp.content)
        df = pd.read_csv(SECOND_DATASET_PATH)
        if "text" in df.columns and "label" in df.columns:
            df = df.rename(columns={"text": "message"})
        elif "email" in df.columns and "label" in df.columns:
            df = df.rename(columns={"email": "message"})
        elif "message" not in df.columns or "label" not in df.columns:
            for col in df.columns:
                if df[col].dtype == object:
                    df = df.rename(columns={col: "message"})
                    break
        df = df[["message", "label"]].dropna()
        df["message"] = df["message"].astype(str).str.strip()
        df = df[df["message"].str.len() >= 10]
        print(f"Cross-dataset loaded: {len(df)} emails (phishing={sum(df.label)}, ham={len(df)-sum(df.label)})")
        return df
    except Exception as e:
        warnings.warn(f"Failed to download cross-dataset: {e}. Continuing with SpamAssassin only.")
        return None


def main():
    df_sa = load_spamassassin()
    df_sa.to_csv(RAW_DIR / "spamassassin_processed.csv", index=False)
    print(f"Saved SpamAssassin processed data to {RAW_DIR / 'spamassassin_processed.csv'}")

    df_cross = download_second_dataset()
    if df_cross is not None:
        df_cross.to_csv(SECOND_DATASET_PATH, index=False)
        print(f"Saved cross-dataset to {SECOND_DATASET_PATH}")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()