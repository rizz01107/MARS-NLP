"""MARS-NLP Streamlit demo.  Run from the repository root:  streamlit run app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _p in (str(ROOT), str(APP_DIR)):   # repo root (for `src`) and app dir (for `components`)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

from components.result_view import DISCLAIMER, render_features, render_result  # noqa: E402
from src import config as C  # noqa: E402
from src.prediction import InvalidInput, available_models, best_model_name, load_predictor, predict_message  # noqa: E402

# Illustrative, hand-written examples (NOT rows from the dataset).
EXAMPLES = {
    "English": "URGENT! You have won a 1000 prize. Call 09061234567 now to claim your reward.",
    "Roman Urdu": "Aap ka account block ho gaya hai. Foran apna PIN is number par bhejein warna account band kar diya jayega.",
    "Urdu script": "\u0622\u067e \u06a9\u0627 \u0627\u06a9\u0627\u0624\u0646\u0679 \u0628\u0644\u0627\u06a9 \u06c1\u0648 \u06af\u06cc\u0627 \u06c1\u06d2",
}

st.set_page_config(page_title="MARS-NLP Scam Detector", page_icon="\U0001F6E1\ufe0f", layout="centered")


@st.cache_resource(show_spinner=False)
def get_predictor(name: str):
    return load_predictor(name)


def main() -> None:
    st.title("MARS-NLP: multilingual scam/spam detection")
    st.caption("English \u00b7 Roman Urdu \u00b7 code-switched messages")
    st.info(DISCLAIMER)

    models = available_models()
    if not models:
        st.error("No trained model found. Run `python -m src.train_baselines` first (see README).")
        return
    names = sorted(models)
    default = best_model_name()
    selected = st.selectbox("Model", names, index=names.index(default) if default in names else 0)

    if "msg" not in st.session_state:
        st.session_state["msg"] = ""
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        if col.button(f"Example: {label}", width="stretch"):
            st.session_state["msg"] = text
    message = st.text_area("Message", key="msg", height=140, placeholder="Paste an SMS-style message here\u2026")

    if st.button("Analyse", type="primary"):
        try:
            predictor = get_predictor(selected)
            result = predict_message(message, predictor)
        except InvalidInput as exc:
            st.warning(str(exc))
            return
        except FileNotFoundError as exc:
            st.error(f"Model could not be loaded: {exc}")
            return
        except Exception as exc:  # never crash on unexpected input
            st.error("Sorry, this message could not be analysed. Please try different text.")
            st.caption(f"Technical detail: {type(exc).__name__}")
            return
        render_result(result)
        render_features(result)

    st.divider()
    st.caption(f"{DISCLAIMER} Trained on SMS Spam Collection (UCI) and a Roman-Urdu phishing SMS dataset; "
               "no real Urdu-script training data. Seed %d." % C.SEED)


main()
