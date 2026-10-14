"""Streamlit smoke tests using Streamlit's AppTest (no browser needed)."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.prediction import available_models  # noqa: E402

from pathlib import Path  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


@pytest.fixture(scope="module")
def app():
    if not any(k.startswith("baseline_") for k in available_models()):
        pytest.skip("no trained model")
    return AppTest.from_file(APP, default_timeout=90).run()


def _analyse(at, text):
    at.text_area[0].set_value(text).run()
    [b for b in at.button if b.label == "Analyse"][0].click().run()
    return at


def test_app_loads_without_exception_and_shows_disclaimer(app):
    assert not app.exception
    assert any("not intended for real-world" in i.value for i in app.info)


@pytest.mark.parametrize("text", ["", "   ", "\U0001F600\U0001F600", "<script>alert(1)</script>", "a" * 9000,
                                  "\u0622\u067e \u06a9\u0627 \u0627\u06a9\u0627\u0624\u0646\u0679 \u0628\u0644\u0627\u06a9 \u06c1\u0648 \u06af\u06cc\u0627",
                                  "Aap ka account block ho gaya hai. Foran PIN bhejein."])
def test_app_never_crashes_on_odd_input(app, text):
    at = _analyse(app, text)
    assert not at.exception


def test_app_shows_prediction_fields(app):
    at = _analyse(app, "URGENT! You have won a prize. Call 09061234567 now to claim.")
    labels = {m.label for m in at.metric}
    assert {"Confidence", "Detected language", "Selected model"} <= labels
