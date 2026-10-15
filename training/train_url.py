"""Train URL phishing detection model."""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Ensure src module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.url_features import extract_url_features, url_to_feature_vector

print("Script started")  # Debug


def generate_synthetic_urls() -> Tuple[List[str], List[int]]:
    """Generate synthetic legitimate and phishing URLs for training.

    Returns:
        urls: list of URL strings
        labels: list of integers (0=legitimate, 1=phishing)
    """
    print("Generating synthetic URLs...")  # Debug
    # Legitimate domains and paths
    legit_domains = [
        "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com",
        "linkedin.com", "github.com", "stackoverflow.com", "reddit.com", "wikipedia.org",
        "amazon.com", "ebay.com", "paypal.com", "apple.com", "microsoft.com",
        "netflix.com", "spotify.com", "dropbox.com", "adobe.com", "yahoo.com"
    ]
    legit_paths = [
        "", "/", "/index.html", "/about", "/contact", "/products", "/services",
        "/blog", "/news", "/login", "/signup", "/settings", "/profile"
    ]

    # Phishing patterns: misspelled domains, added words, IP addresses, etc.
    phishing_patterns = [
        # Misspelled domains
        "http://gooogle.com/login",
        "http://amaz0n.com/signin",
        "http://faceb00k.com/secure",
        "http://paypa1.com/verify",
        "http://micros0ft.com/update",
        "http://appleid.com/account",
        "http://netflix.com/signin",
        # Hyphenated domains
        "http://google-login.com/",
        "http://paypal-verify.net/",
        "http://amazon-security.org/",
        # IP addresses
        "http://192.168.1.1/login",
        "http://10.0.0.1/secure",
        "http://8.8.8.8/account",
        # Punycode (xn--)
        "http://xn--pple-43d.com/",  # apple.com
        "http://xn--80ak6aa92e.com/",  #看起来像xn--80ak6aa92e.com is a punycode domain
        # URL shorteners (sometimes used in phishing)
        "http://bit.ly/abc123",
        "http://tinyurl.com/def456",
        # Suspicious TLDs
        "http://example.tk/login",
        "http://example.ml/secure",
        # Many subdomains
        "http://login.google.com.attacker.com/",
        "http://secure.paypal.com.fake.com/",
        # Random strings
        "http://randomstring123.com/login",
        # HTTPS phishing
        "https://paypa1.com/login",
    ]

    # Generate legitimate URLs
    legit_urls = []
    for domain in legit_domains:
        for path in legit_paths:
            # Both http and https
            legit_urls.append(f"http://{domain}{path}")
            legit_urls.append(f"https://{domain}{path}")

    # Generate phishing URLs from patterns
    phishing_urls = phishing_patterns.copy()

    # Add some variations by changing paths
    extra_paths = ["/login", "/signin", "/verify", "/secure", "/account", "/update"]
    for url in phishing_patterns[:20]:  # Take first 20 to avoid too many
        base_url = url.split("/")[0] + "//" + url.split("/")[2]  # scheme://domain
        for path in extra_paths:
            phishing_urls.append(f"{base_url}{path}")

    # Combine and label
    urls = legit_urls + phishing_urls
    labels = [0] * len(legit_urls) + [1] * len(phishing_urls)
    print(f"Generated {len(urls)} URLs")  # Debug
    return urls, labels


def extract_features(urls: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract features from a list of URLs.

    Returns:
        feature_matrix: numpy array of shape (n_urls, n_features)
        feature_names: list of feature names in order
    """
    print("Extracting features...")  # Debug
    # Extract features for first URL to get feature names
    first_features = extract_url_features(urls[0])
    feature_names = sorted(first_features.keys())  # consistent order

    # Build feature matrix
    feature_matrix = []
    for i, url in enumerate(urls):
        if i % 1000 == 0:
            print(f"  Processed {i} URLs")
        features = extract_url_features(url)
        feature_matrix.append([features[name] for name in feature_names])

    print(f"Extracted features for {len(urls)} URLs")
    return np.array(feature_matrix), feature_names


def train_model() -> None:
    """Train URL model and save the better of LogisticRegression and RandomForest."""
    print("In train_model")  # Debug
    # Generate synthetic data
    urls, labels = generate_synthetic_urls()
    print(f"Generated {len(urls)} URLs ({sum(labels)} phishing, {len(labels)-sum(labels)} legitimate)")

    # Extract features
    X, feature_names = extract_features(urls)
    y = np.array(labels)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred)
    print(f"Logistic Regression F1: {lr_f1:.4f}")

    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred)
    print(f"Random Forest F1: {rf_f1:.4f}")

    # Choose the better model
    if lr_f1 >= rf_f1:
        best_model = lr
        best_model_name = "LogisticRegression"
        best_f1 = lr_f1
    else:
        best_model = rf
        best_model_name = "RandomForest"
        best_f1 = rf_f1

    print(f"Best model: {best_model_name} with F1 {best_f1:.4f}")

    # Save the model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "url_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")

    # Save metadata
    meta = {
        "model_name": best_model_name,
        "feature_names": feature_names,
        "f1_score": best_f1,
    }
    meta_path = os.path.join(model_dir, "url_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Also print feature names for verification
    print(f"Feature names ({len(feature_names)}): {feature_names}")


if __name__ == "__main__":
    print("About to call train_model")  # Debug
    train_model()
    print("Done")  # Debug