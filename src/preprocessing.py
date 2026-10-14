"""Text cleaning, label/language normalisation, language heuristics, near-duplicate grouping.

Design principle: cleaning must NOT destroy linguistic information. ``message`` always keeps the
raw text; ``cleaned_message`` only receives lossless-in-spirit normalisation (Unicode NFKC,
HTML entities, control characters, whitespace). Case, punctuation and spelling variation are kept.
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Sequence

import numpy as np
import pandas as pd

from . import config as C

# Zero-width space / word joiner / BOM are removed; ZWNJ (U+200C) and ZWJ (U+200D) are kept
# because they are orthographically meaningful in Urdu script.
_REMOVE_CHARS = dict.fromkeys(map(ord, "\u200b\u2060\ufeff"), None)
_MOJIBAKE_RE = re.compile("[\u00c2\u00c3\u00e2][\u0080-\u00bf\u20ac\u2122\u201c\u201d\u2018\u2019\u2013\u2014\u2026]")
_WS_RE = re.compile(r"\s+")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


# --------------------------------------------------------------------------- text cleaning
def repair_mojibake(text: str) -> str:
    """Repair UTF-8 text mis-decoded as cp1252/latin-1 when it clearly reduces mojibake markers."""
    if not _MOJIBAKE_RE.search(text):
        return text
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if len(_MOJIBAKE_RE.findall(fixed)) < len(_MOJIBAKE_RE.findall(text)):
            return fixed
    return text


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_REMOVE_CHARS)


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _strip_controls(text: str) -> str:
    return "".join(" " if (unicodedata.category(ch) == "Cc") else ch for ch in text)


def clean_text(text: object) -> str:
    """Return the cleaned representation of ``text``; non-strings/NaN become an empty string."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = repair_mojibake(s)
    s = html.unescape(s)
    s = normalize_unicode(s)
    s = _strip_controls(s)
    return normalize_whitespace(s)


# --------------------------------------------------------------------------- label handling
_SCAM_LABELS = {"spam", "scam", "phishing", "smishing", "fraud", "1", "true", "scam/spam"}
_LEGIT_LABELS = {"ham", "legitimate", "legit", "non-scam", "non scam", "not spam", "benign", "0", "false",
                 "legitimate/ham"}


def normalize_label(value: object) -> str:
    """Map heterogeneous raw labels to ``Scam`` / ``Legitimate``. Raises ValueError if unknown."""
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value in (0.0, 1.0)):
        value = str(int(value))
    key = str(value).strip().lower()
    if key in _SCAM_LABELS:
        return "Scam"
    if key in _LEGIT_LABELS:
        return "Legitimate"
    raise ValueError(f"Unknown label: {value!r}")


_LANG_ALIASES = {
    "english": C.LANG_ENGLISH, "en": C.LANG_ENGLISH,
    "urdu": C.LANG_URDU, "ur": C.LANG_URDU,
    "roman urdu": C.LANG_ROMAN_URDU, "roman_urdu": C.LANG_ROMAN_URDU, "ru": C.LANG_ROMAN_URDU,
    "mixed/code-switched": C.LANG_MIXED, "mixed": C.LANG_MIXED, "code-switched": C.LANG_MIXED,
    "unknown": C.LANG_UNKNOWN,
}


def normalize_language(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _LANG_ALIASES:
        raise ValueError(f"Unknown language: {value!r}")
    return _LANG_ALIASES[key]


# --------------------------------------------------------------------------- language heuristic
# Small hand-curated lexicons of high-frequency function words. This is a transparent HEURISTIC,
# not a trained language-ID model; its accuracy is audited manually in reports/data_audit.md.
ROMAN_URDU_WORDS = frozenset("""
ka ki ke ko ne se main mein men hai hain hy hay hn hein ho hu hun hoon tha thi thay ga gi ge gya gaya gayi
apka apki apke aapka aapki aapke aap apna apni apne tum tumhara mera meri mere hamara hamari uska iska
ye yeh wo woh jo jis aur liye liya lie lye kar kr karo karen karein krein krin krain karain karin karna krna
karne krne kia kya kiya dia diya diye diyay nahi nahin nai nhi abhi warna agar bhi sirf bohat bohot bahut
zyada kal aaj kab kahan kyun kyu kuch koi sab sath saath wala wali wale waja wajah band hamesha jaye jayega
jaen bhejein bhejain bhejen bhej bhejo inam inaam mubarak jeeto jeetne moaziz taseeq hasil nikla nikli lga
laga chuka chuki raha rahi rahe haan theek thik acha accha achha yaar paisa paise paisay rupay janab sahab
sahib teri tera tumne unka humne yahan wahan isliye kyunke lekin magar bas bilkul jaldi din raat subah
shaam ghar kaam khata pehle phir fir baad meherbani shukriya dost pyaar waqt aao zaroori
""".split())

ENGLISH_WORDS = frozenset("""
the a an and or but if is are was were be been being to of in on at for from with by about as into this that
these those you your yours i me my we our us it its he she they them his her their not no do does did done
have has had will would can could should may might must please call text free now today tomorrow get got
send reply txt ur u im ill dont cant wont ok okay lol what when where why how who which there here then
than so just also very all any some one more most up out over only own same too again further once dear
customer account bank card number pin otp click link win won prize claim urgent msg thanks thank hi hello
hey good love going come back
""".split())


def language_scores(text: str) -> dict:
    """Return token counts used by :func:`detect_language_with_confidence`."""
    tokens = _TOKEN_RE.findall(text.lower())
    n = len(tokens)
    ru = sum(t in ROMAN_URDU_WORDS for t in tokens)
    en = sum(t in ENGLISH_WORDS for t in tokens)
    letters_ar = len(_ARABIC_RE.findall(text))
    letters_lat = len(_LATIN_RE.findall(text))
    return {"n_tokens": n, "ru_hits": ru, "en_hits": en, "arabic_letters": letters_ar, "latin_letters": letters_lat}


def detect_language_with_confidence(text: str) -> tuple[str, str]:
    """Heuristic language detector -> (language, confidence in {'high','low'}).

    * Perso-Arabic script dominant           -> Urdu
    * Perso-Arabic + Latin both substantial  -> Mixed/Code-Switched
    * Latin script: Roman-Urdu vs English function-word hits
        both >=2 hits                        -> Mixed/Code-Switched
        Roman-Urdu hits >=2                  -> Roman Urdu
        otherwise English (or Unknown when there is almost no alphabetic evidence)
    """
    if not isinstance(text, str) or not text.strip():
        return C.LANG_UNKNOWN, "low"
    s = language_scores(text)
    ar, lat = s["arabic_letters"], s["latin_letters"]
    total_letters = ar + lat
    if total_letters == 0:
        return C.LANG_UNKNOWN, "low"
    if ar / total_letters >= 0.30:
        if lat / total_letters >= 0.30:
            return C.LANG_MIXED, "high"
        return C.LANG_URDU, "high"
    ru, en, n = s["ru_hits"], s["en_hits"], s["n_tokens"]
    if ru >= 2 and en >= 2:
        return C.LANG_MIXED, "high"
    if ru >= 2:
        return C.LANG_ROMAN_URDU, "high"
    if ru == 1 and en == 0 and n <= 4:
        return C.LANG_ROMAN_URDU, "low"
    if en >= 2:
        return C.LANG_ENGLISH, "high"
    if n <= 2 and en == 0:
        return C.LANG_UNKNOWN, "low"
    return C.LANG_ENGLISH, "low"


def detect_language(text: str) -> str:
    return detect_language_with_confidence(text)[0]


# --------------------------------------------------------------------------- near-duplicate grouping
def template_key(text: str) -> str:
    """Key for template matching: lowercase, digit runs masked, whitespace collapsed."""
    t = re.sub(r"\d+", "0", text.lower())
    return normalize_whitespace(t)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_groups(texts: Sequence[str], threshold: float = C.NEAR_DUP_THRESHOLD,
                          chunk: int = 400) -> np.ndarray:
    """Cluster near-duplicate/templated messages (single-linkage over cosine similarity).

    Similarity is computed on character 3-5-gram TF-IDF of :func:`template_key`. Returns an integer
    cluster id per text (ids are consecutive, ordered by first occurrence -> deterministic).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    if n == 0:
        return np.array([], dtype=int)
    keys = [template_key(t) for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1, dtype=np.float32)
    X = vec.fit_transform(keys).tocsr()
    uf = _UnionFind(n)
    XT = X.T.tocsc()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = (X[start:stop] @ XT).toarray()
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows, cols):
            i = start + int(r)
            if c > i:
                uf.union(i, int(c))
    roots = np.array([uf.find(i) for i in range(n)])
    _, inverse = np.unique(roots, return_inverse=True)
    # relabel by first occurrence for determinism/readability
    order: dict[int, int] = {}
    out = np.empty(n, dtype=int)
    for i, g in enumerate(inverse):
        out[i] = order.setdefault(int(g), len(order))
    return out


# --------------------------------------------------------------------------- dataframe preprocessing
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "message", label_col: str = "label") -> pd.DataFrame:
    """Clean text, normalise labels, drop empty/invalid rows. Keeps the raw text column untouched."""
    out = df.copy()
    out[text_col] = out[text_col].astype("object")
    out["cleaned_message"] = out[text_col].map(clean_text)
    keep = out["cleaned_message"].str.len() > 0
    out = out.loc[keep].copy()
    if label_col in out.columns:
        out[label_col] = out[label_col].map(normalize_label)
    return out.reset_index(drop=True)


def drop_exact_duplicates(df: pd.DataFrame, col: str = "cleaned_message") -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicates (keep first). Messages duplicated with conflicting labels are removed entirely."""
    info = {"n_before": int(len(df))}
    nlab = df.groupby(col)["label"].nunique()
    conflicting = set(nlab[nlab > 1].index)
    info["conflicting_label_messages"] = len(conflicting)
    work = df[~df[col].isin(conflicting)] if conflicting else df
    out = work.drop_duplicates(subset=[col], keep="first").reset_index(drop=True)
    info["n_after"] = int(len(out))
    info["n_removed"] = info["n_before"] - info["n_after"]
    return out, info
