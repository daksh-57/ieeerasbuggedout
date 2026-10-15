"""Unit tests for URL feature extraction."""

from src.url_features import extract_url_features


def test_legitimate_paypal():
    """Test legitimate PayPal URL."""
    url = "https://www.paypal.com/login"
    features = extract_url_features(url)
    assert features["has_https"] == 1.0
    assert features["has_punycode"] == 0.0
    # Keyword hits: contains "login" -> at least 1
    assert features["keyword_hits"] >= 1.0
    # Not an IP address
    assert features["is_ip_as_host"] == 0.0
    # Not a shortener
    assert features["is_shortener_domain"] == 0.0


def test_paypal_lookalike():
    """Test paypa1.com/verify (lookalike)."""
    url = "http://paypa1.com/verify"
    features = extract_url_features(url)
    # Should have keyword hit for "verify"
    assert features["keyword_hits"] >= 1.0
    # Has digits (maybe not, but we check)
    # Actually, paypa1 has a digit '1' in domain
    assert features["digits_ratio"] > 0.0
    # Not HTTPS
    assert features["has_https"] == 0.0


def test_micros0ft_login():
    """Test micros0ft-login.net/ (hyphen and keyword)."""
    url = "https://micros0ft-login.net/"
    features = extract_url_features(url)
    # Has hyphen
    assert features["hyphen_count"] >= 1.0
    # Has keyword "login"
    assert features["keyword_hits"] >= 1.0
    # Has digit '0' in micros0ft
    assert features["digits_ratio"] > 0.0


def test_ip_as_host():
    """Test IP address as host."""
    url = "http://192.168.1.1/login"
    features = extract_url_features(url)
    assert features["is_ip_as_host"] == 1.0
    assert features["has_https"] == 0.0
    assert features["keyword_hits"] >= 1.0  # login


def test_shortener_domain():
    """Test bit.ly shortener."""
    url = "https://bit.ly/abc"
    features = extract_url_features(url)
    assert features["is_shortener_domain"] == 1.0
    assert features["has_https"] == 1.0


def test_punycode():
    """Test punycode domain."""
    url = "https://xn--80ak6aa92e.com/"
    features = extract_url_features(url)
    assert features["has_punycode"] == 1.0
    # This domain is actually xn--80ak6aa92e.com which is a punycode for a Russian domain?
    # We just check that the feature is triggered.


def test_suspicious_tld():
    """Test suspicious TLD .tk."""
    url = "http://example.tk/login"
    features = extract_url_features(url)
    assert features["has_suspicious_tld"] == 1.0
    assert features["keyword_hits"] >= 1.0


def test_no_suspicious_tld():
    """Test legitimate TLD .com."""
    url = "https://google.com"
    features = extract_url_features(url)
    assert features["has_suspicious_tld"] == 0.0
    assert features["has_https"] == 1.0
    assert features["keyword_hits"] == 0.0  # no keywords