"""Model loading and single-message prediction (shared by the Streamlit app, tests and experiments)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np

from . import config as C
from .preprocessing import clean_text, detect_language_with_confidence


class InvalidInput(ValueError):
    """Raised for input that cannot be classified (empty / non-text); message is user-friendly."""


@dataclass
class PredictionResult:
    label: str
    label_id: int
    p_scam: float
    confidence: float
    confidence_type: str
    language: str
    language_confidence: str
    model_name: str
    top_features: list = field(default_factory=list)
    truncated: bool = False
    warnings: list = field(default_factory=list)
    supported_language: bool = True   # False -> language not covered by training data; confidence is NOT meaningful

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def pipeline_scores(pipe, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (predicted_ids, p_scam, confidence_type) for a fitted sklearn pipeline.

    LogisticRegression / NaiveBayes expose probabilities. LinearSVC does not: its signed margin is passed
    through a logistic sigmoid (``margin_sigmoid``), which is an UNCALIBRATED pseudo-confidence.
    """
    texts = list(texts)
    clf = pipe.steps[-1][1]
    if hasattr(clf, "predict_proba"):
        p = pipe.predict_proba(texts)[:, 1]
        kind = "probability"
    else:
        margin = pipe.decision_function(texts)
        p = 1.0 / (1.0 + np.exp(-margin))
        kind = "margin_sigmoid"
    return (p >= 0.5).astype(int), p, kind


def validate_input(text: object) -> tuple[str, bool]:
    """Return (cleaned_text, truncated). Raises :class:`InvalidInput` for empty/non-string input."""
    if text is None:
        raise InvalidInput("Please enter a message to analyse.")
    if not isinstance(text, str):
        raise InvalidInput("The input must be text.")
    cleaned = clean_text(text)
    if not cleaned:
        raise InvalidInput("The message is empty after cleaning. Please enter some text.")
    if not any(ch.isalnum() for ch in cleaned):
        raise InvalidInput("The message contains no letters or digits to analyse.")
    truncated = len(cleaned) > C.MAX_INPUT_CHARS
    return cleaned[:C.MAX_INPUT_CHARS], truncated


class BaselinePredictor:
    """Wraps a saved baseline artefact ``{"pipeline": sklearn Pipeline, "meta": dict}``."""

    kind = "baseline"

    def __init__(self, path: Path, name: Optional[str] = None) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        art = joblib.load(path)   # only load artefacts you trust (joblib/pickle)
        if not isinstance(art, dict) or "pipeline" not in art:
            raise ValueError(f"{path} is not a valid MARS-NLP baseline artefact")
        self.pipeline = art["pipeline"]
        self.meta = art.get("meta", {})
        self.name = name or self.meta.get("name", path.stem)
        self.path = path

    def scores(self, texts: Sequence[str]):
        return pipeline_scores(self.pipeline, texts)

    def top_features(self, text: str, k: int = 8) -> list:
        from .explainability import linear_feature_contributions

        try:
            return linear_feature_contributions(self.pipeline, text, k)
        except Exception:
            return []


class TransformerPredictor:
    """Loads a fine-tuned Hugging Face sequence-classification directory (requires torch + transformers)."""

    kind = "transformer"

    def __init__(self, path: Path, name: Optional[str] = None, max_length: int = 128) -> None:
        from .train_transformer import load_finetuned  # lazy: heavy deps

        self.model, self.tokenizer, self.device = load_finetuned(Path(path))
        self.name = name or Path(path).name
        self.max_length = max_length

    def scores(self, texts: Sequence[str]):
        from .train_transformer import predict_proba_texts

        p = predict_proba_texts(self.model, self.tokenizer, list(texts), self.device, self.max_length)
        return (p >= 0.5).astype(int), p, "probability"

    def top_features(self, text: str, k: int = 8) -> list:
        return []


def available_models(models_dir: Path = C.MODELS_DIR) -> dict[str, Path]:
    """Discover loadable models: ``baseline_*.joblib`` files and ``transformers/<name>/`` directories."""
    found: dict[str, Path] = {}
    models_dir = Path(models_dir)
    if models_dir.exists():
        for p in sorted(models_dir.glob("baseline_*.joblib")):
            found[p.stem] = p
        tdir = models_dir / "transformers"
        if tdir.exists():
            for d in sorted(tdir.iterdir()):
                if d.is_dir() and (d / "config.json").exists() and (d / "mars_meta.json").exists():
                    found[f"transformer_{d.name}"] = d
    return found


def best_model_name(models_dir: Path = C.MODELS_DIR) -> Optional[str]:
    """Name of the best baseline according to ``models/best_model.json`` if present, else first available."""
    from .utils import load_json

    meta = Path(models_dir) / "best_model.json"
    avail = available_models(models_dir)
    if meta.exists():
        name = load_json(meta).get("model_id")
        if name in avail:
            return name
    return next(iter(avail), None)


def load_predictor(name_or_path: Optional[str] = None, models_dir: Path = C.MODELS_DIR):
    avail = available_models(models_dir)
    if name_or_path is None:
        name_or_path = best_model_name(models_dir)
        if name_or_path is None:
            raise FileNotFoundError("No trained model found. Run `python -m src.train_baselines` first.")
    if name_or_path in avail:
        path = avail[name_or_path]
    else:
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Unknown model {name_or_path!r}. Available: {sorted(avail)}")
    return TransformerPredictor(path, name_or_path) if path.is_dir() else BaselinePredictor(path, name_or_path)


def predict_message(text: object, predictor, with_features: bool = True, top_k: int = 8) -> PredictionResult:
    """Classify a single message. Raises :class:`InvalidInput` for unusable input; never fails on odd Unicode."""
    cleaned, truncated = validate_input(text)
    lang, lang_conf = detect_language_with_confidence(cleaned)
    ids, p, kind = predictor.scores([cleaned])
    label_id = int(ids[0])
    p_scam = float(p[0])
    conf = p_scam if label_id == 1 else 1.0 - p_scam
    warns = []
    if truncated:
        warns.append(f"Input truncated to {C.MAX_INPUT_CHARS} characters.")
    if lang_conf == "low":
        warns.append("Language detection is low-confidence for this message.")
    supported = lang in (C.LANG_ENGLISH, C.LANG_ROMAN_URDU, C.LANG_MIXED)
    if not supported:
        warns.append("The training data contains no Urdu-script (or unidentifiable-language) messages, so this prediction "
                     "is out of scope and its confidence is not meaningful.")
    feats = predictor.top_features(cleaned, top_k) if with_features else []
    return PredictionResult(C.ID2LABEL[label_id], label_id, p_scam, float(conf), kind, lang, lang_conf,
                            getattr(predictor, "name", "model"), feats, truncated, warns, supported)
