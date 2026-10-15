"""Phishing Detection — Streamlit entry point.

Phase 1 placeholder. The full UI (Analyze Message, Analyze URL, Model
Performance, About) is built in later phases. This file exists so the app can
be deployed and verified early on Streamlit Community Cloud.
"""

from __future__ import annotations


def main() -> None:
    """Entry point — runs the Streamlit UI."""
    import streamlit as st

    st.title("Phishing Detection")
    st.write(
        "Under construction. See `IMPLEMENTATION.md` for the roadmap and "
        "`COMPLETED.md` for what has been built so far."
    )


if __name__ == "__main__":
    main()