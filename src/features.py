"""TF-IDF feature extractors used by the classical baselines."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_KINDS = ("word", "char")


def make_vectorizer(kind: str) -> TfidfVectorizer:
    """Return an unfitted TF-IDF vectoriser.

    * ``word``: word 1-2-grams (Unicode-aware tokens, so Urdu script would also tokenise), lowercase.
    * ``char``: character 2-5-grams within word boundaries (``char_wb``), lowercase; more tolerant to spelling noise.
    Both use sublinear term frequency; ``min_df=2`` discards singletons (fit on the training split only).
    """
    if kind == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=2,
                               token_pattern=r"(?u)\b\w+\b")
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, min_df=2,
                               max_features=300_000)
    raise ValueError(f"Unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")
