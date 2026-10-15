"""URL feature extraction for phishing detection."""

from __future__ import annotations

import math
import re
from typing import Dict, List

import tldextract


# Initialize tldextract offline (no network calls)
_extractor = tldextract.TLDExtract(suffix_list_urls=())

# Suspicious TLDs (list from common phishing TLDs)
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "site",
    "win", "bid", "date", "review", "download", "stream", "faith", "racing",
    "cricket", "accountant", "loan", "lol", "party", "gdn", "link", "click",
    "help", "expert", "guru", "kim", "ltd", "red", "blue", "green", "black",
    "white", "ltd.uk", "co.uk"  # some may be legitimate but we keep for demo
}

# Known URL shortener domains (registered domains)
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
    "bit.do", "mcaf.ee", "tiny.cc", "url.ie", "v.gd", "tr.im", "tiny.pl",
    "short.to", "BudURL.com", "chk.me", "lnk.co", "urlazarus.com", "flip.lk"
}

# Keywords that often appear in phishing URLs
PHISHING_KEYWORDS = {
    "login", "verify", "secure", "update", "account", "signin", "auth",
    "confirm", "password", "winning", "claim", "prize"
}


def _entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    # Count frequency of each character
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    # Calculate entropy
    entropy = 0.0
    length = len(s)
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def extract_url_features(url: str) -> Dict[str, float]:
    """Extract lexical features from a URL.

    Args:
        url: The URL string to analyze.

    Returns:
        A dictionary of feature names to values.
    """
    # Ensure the URL has a scheme for proper parsing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    # Extract tldextract components
    extracted = _extractor(url)
    # extracted.subdomain is string like "sub.sub2" or ""
    # extracted.domain is the registered domain (without TLD)
    # extracted.suffix is the TLD (like "com", "co.uk")

    # Build host (subdomain + domain + suffix) but we may not need it
    host_parts = []
    if extracted.subdomain:
        host_parts.append(extracted.subdomain)
    host_parts.append(extracted.domain)
    if extracted.suffix:
        host_parts.append(extracted.suffix)
    host = ".".join(host_parts) if host_parts else ""

    # Features
    features: Dict[str, float] = {}

    # length
    features["length"] = len(url)

    # dot_count
    features["dot_count"] = url.count(".")

    # subdomain_depth: number of subdomain parts (split by '.')
    if extracted.subdomain:
        # Split by '.' and filter empty strings
        subdomain_parts = [part for part in extracted.subdomain.split(".") if part]
        features["subdomain_depth"] = len(subdomain_parts)
    else:
        features["subdomain_depth"] = 0

    # hyphen_count
    features["hyphen_count"] = url.count("-")

    # digits_ratio
    digit_count = sum(c.isdigit() for c in url)
    features["digits_ratio"] = digit_count / len(url) if len(url) > 0 else 0.0

    # has_at_sign
    features["has_at_sign"] = 1.0 if "@" in url else 0.0

    # is_ip_as_host: check if host is an IPv4 or IPv6 address
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    ipv6_pattern = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
    is_ip = (
        re.match(ipv4_pattern, host) is not None
        or re.match(ipv6_pattern, host) is not None
    )
    features["is_ip_as_host"] = 1.0 if is_ip else 0.0

    # has_punycode
    features["has_punycode"] = 1.0 if "xn--" in url else 0.0

    # entropy
    features["entropy"] = _entropy(url)

    # has_suspicious_tld
    features["has_suspicious_tld"] = 1.0 if extracted.suffix.lower() in SUSPICIOUS_TLDS else 0.0

    # has_https
    features["has_https"] = 1.0 if url.lower().startswith("https://") else 0.0

    # keyword_hits (case-insensitive)
    lower_url = url.lower()
    keyword_count = sum(lower_url.count(keyword) for keyword in PHISHING_KEYWORDS)
    features["keyword_hits"] = float(keyword_count)

    # is_shortener_domain: check if the registered domain (or subdomain+domain) is in shortener list
    # We'll check the registered domain (extracted.domain + '.' + extracted.suffix) if suffix exists
    registered_domain = ""
    if extracted.domain and extracted.suffix:
        registered_domain = f"{extracted.domain}.{extracted.suffix}"
    elif extracted.domain:
        registered_domain = extracted.domain
    # Also consider the case where the domain might be a shortener without subdomain
    features["is_shortener_domain"] = (
        1.0 if registered_domain.lower() in SHORTENER_DOMAINS else 0.0
    )

    return features


def url_to_feature_vector(url: str, feature_names: List[str]) -> List[float]:
    """Convert a URL to a feature vector in the order of feature_names.

    Args:
        url: The URL string.
        feature_names: List of feature names in the order expected by the model.

    Returns:
        List of feature values.
    """
    features = extract_url_features(url)
    return [features.get(name, 0.0) for name in feature_names]