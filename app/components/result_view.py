"""Rendering helpers for the Streamlit demo (kept separate from app logic so they stay small and testable)."""
from __future__ import annotations

import streamlit as st

DISCLAIMER = "Research prototype \u2014 not intended for real-world financial/security decisions."


def render_result(result) -> None:
    """Show prediction, confidence, detected language and selected model."""
    is_scam = result.label_id == 1
    (st.error if is_scam else st.success)(f"Prediction: **{result.label}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{result.confidence:.1%}" if result.supported_language else "not meaningful")
    c2.metric("Detected language", result.language)
    c3.metric("Selected model", result.model_name)
    if result.confidence_type != "probability":
        st.caption("Confidence for this model is an uncalibrated margin-based score, not a true probability.")
    for w in result.warnings:
        st.warning(w)


def render_features(result) -> None:
    """Show the most influential features (model attributions, not causal explanations)."""
    if not result.top_features:
        st.caption("No feature attribution available for this model.")
        return
    st.markdown("**Most influential features** (model attributions, not causal evidence)")
    rows = [{"feature": f, "contribution": round(c, 4), "towards": "Scam" if c > 0 else "Legitimate"}
            for f, c in result.top_features]
    st.dataframe(rows, width="stretch", hide_index=True)
