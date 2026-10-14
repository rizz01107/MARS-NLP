"""Controlled, seeded text perturbations (derived ONLY from real original messages).

Every function has the signature ``f(text, rng, language) -> str | None`` and returns ``None`` when the
perturbation is not applicable or would leave the text unchanged. Generation is deterministic given
(original_message_id, perturbation_type, seed), independent of processing order.

These are random *noise* perturbations, not optimised adversarial attacks.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Optional

from . import config as C
from .utils import stable_seed

_LETTERS = re.compile(r"[^\W\d_]+")
_VOWELS = "aeiou"
_KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl",
    "p": "ol", "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm",
    "k": "jil", "l": "kop", "z": "asx", "x": "zsc", "c": "xdv", "v": "cfb", "b": "vgn", "n": "bhm", "m": "nj",
}
_LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "l": "1", "t": "7"}

_ABBREVIATIONS = {
    "you": "u", "your": "ur", "you're": "ur", "are": "r", "to": "2", "too": "2", "for": "4", "before": "b4",
    "please": "pls", "because": "bcoz", "message": "msg", "messages": "msgs", "thanks": "thx", "thank": "thnx",
    "tomorrow": "tmrw", "tonight": "2nite", "today": "2day", "text": "txt", "great": "gr8", "late": "l8",
    "later": "l8r", "okay": "ok", "people": "ppl", "number": "no.", "minutes": "mins", "with": "wid",
    "what": "wat", "love": "luv", "about": "abt", "good": "gud", "between": "btwn", "without": "w/o",
    "customer": "cust", "account": "acct", "mobile": "mob", "request": "req", "information": "info",
}

_RU_VARIANTS = {
    "hai": ["hy", "hay", "ha"], "hain": ["hn", "hen", "hein"], "apka": ["aapka", "apkaa"], "apki": ["aapki", "apkee"],
    "aap": ["ap", "app"], "ke": ["kay", "k"], "ka": ["kaa"], "ki": ["kee"], "kar": ["kr"],
    "karein": ["krein", "karen", "krain"], "krin": ["karein", "krain", "karin"], "krain": ["karain", "krein"],
    "liye": ["liay", "lye", "lie"], "lie": ["liye", "liay"], "nahi": ["nahin", "nhi"], "abhi": ["abi", "abhee"],
    "main": ["mein", "men", "mai"], "mubarak": ["mubaarak", "mobarak"], "inam": ["inaam", "enam"],
    "hasil": ["haasil", "hasl"], "diye": ["diay", "diyay"], "gaya": ["gya", "gayaa"], "gya": ["gaya"],
    "par": ["pr", "pe"], "se": ["say", "sey"], "warna": ["warnah", "varna"], "bohat": ["bohot", "boht"],
    "kal": ["kl"], "aaj": ["aj"], "acha": ["achha", "accha"], "theek": ["thik", "theak"], "kuch": ["kch", "kuchh"],
    "ne": ["nay"], "ho": ["hoo"], "aur": ["or", "aor"], "kya": ["kia", "kiya"], "nikla": ["nikala"],
    "jaye": ["jae", "jaey"], "band": ["bnd"], "lga": ["laga"], "hamesha": ["hmesha", "humesha"],
}

_EN_TO_RU = {
    "and": "aur", "not": "nahi", "your": "apka", "you": "aap", "please": "meherbani", "now": "abhi",
    "today": "aaj", "tomorrow": "kal", "money": "paisa", "free": "muft", "message": "paigham", "send": "bhejein",
    "win": "jeeto", "won": "jeeta", "account": "khata", "tonight": "aaj raat", "friend": "dost", "home": "ghar",
    "work": "kaam", "good": "acha", "love": "pyaar", "time": "waqt", "come": "aao", "yes": "haan", "what": "kya",
    "when": "kab", "where": "kahan", "why": "kyun", "with": "saath", "but": "lekin", "very": "bohat",
    "thanks": "shukriya", "thank": "shukriya", "prize": "inam", "congratulations": "mubarak ho", "urgent": "zaroori",
}
_RU_TO_EN = {
    "aur": "and", "nahi": "not", "abhi": "now", "apka": "your", "aapka": "your", "aap": "you", "kal": "tomorrow",
    "aaj": "today", "paisa": "money", "paise": "money", "inam": "prize", "inaam": "prize", "mubarak": "congratulations",
    "bhejein": "send", "krin": "do", "karein": "do", "krain": "do", "band": "closed", "hai": "is", "hain": "are",
    "jeeto": "win", "warna": "otherwise", "sirf": "only", "bohat": "very", "lekin": "but", "haan": "yes",
    "kyun": "why", "kya": "what", "kab": "when", "saath": "with", "dost": "friend", "ghar": "home",
    "kaam": "work", "acha": "good", "shukriya": "thanks",
}


def _match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _spans(text: str, min_len: int) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _LETTERS.finditer(text) if m.end() - m.start() >= min_len]


def _finish(original: str, new: str) -> Optional[str]:
    new = new if new.strip() else original
    return new if new != original else None


# --------------------------------------------------------------------------- perturbation functions
def spelling_variation(text: str, rng: random.Random, language: str) -> Optional[str]:
    spans = _spans(text, 4)
    if not spans:
        return None
    k = max(1, round(0.15 * len(spans)))
    chosen = sorted(rng.sample(spans, min(k, len(spans))), reverse=True)
    out = text
    for s, e in chosen:
        w = out[s:e]
        op = rng.choice(["swap", "double", "vowel"])
        if op == "swap" and len(w) >= 4:
            i = rng.randrange(1, len(w) - 2)
            nw = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        elif op == "double":
            i = rng.randrange(1, len(w))
            nw = w[:i] + w[i] + w[i:]
        else:
            idx = [i for i, ch in enumerate(w) if ch.lower() in _VOWELS and i > 0]
            if not idx:
                nw = w[:1] + w[2] + w[1] + w[3:]
            else:
                i = rng.choice(idx)
                rep = rng.choice([v for v in _VOWELS if v != w[i].lower()])
                nw = w[:i] + (rep.upper() if w[i].isupper() else rep) + w[i + 1:]
        out = out[:s] + nw + out[e:]
    return _finish(text, out)


def character_deletion(text: str, rng: random.Random, language: str) -> Optional[str]:
    spans = _spans(text, 3)
    if not spans:
        return None
    positions = [i for s, e in spans for i in range(s + 1, e)]
    if not positions:
        return None
    chosen = {p for p in positions if rng.random() < 0.08}
    if not chosen:
        chosen = {rng.choice(positions)}
    return _finish(text, "".join(ch for i, ch in enumerate(text) if i not in chosen))


def character_substitution(text: str, rng: random.Random, language: str) -> Optional[str]:
    spans = _spans(text, 3)
    positions = [i for s, e in spans for i in range(s, e)
                 if text[i].lower() in _KEYBOARD_NEIGHBORS or text[i].lower() in _LEET]
    if not positions:
        return None
    chosen = [p for p in positions if rng.random() < 0.08] or [rng.choice(positions)]
    chars = list(text)
    for p in chosen:
        ch = chars[p]
        low = ch.lower()
        use_leet = low in _LEET and (low not in _KEYBOARD_NEIGHBORS or rng.random() < 0.5)
        rep = _LEET[low] if use_leet else rng.choice(_KEYBOARD_NEIGHBORS[low])
        chars[p] = rep.upper() if ch.isupper() and rep.isalpha() else rep
    return _finish(text, "".join(chars))


def word_spacing_variation(text: str, rng: random.Random, language: str) -> Optional[str]:
    out = text
    long_words = _spans(out, 6)
    if long_words:
        for s, e in sorted(rng.sample(long_words, min(2, len(long_words))), reverse=True):
            cut = rng.randrange(2, e - s - 1)
            out = out[:s + cut] + " " + out[s + cut:]
    gaps = [m.start() for m in re.finditer(r"(?<=[^\W\d_]) (?=[^\W\d_])", out)]
    if gaps:
        for g in sorted(rng.sample(gaps, min(2, len(gaps))), reverse=True):
            out = out[:g] + out[g + 1:]
    return _finish(text, out)


def punctuation_variation(text: str, rng: random.Random, language: str) -> Optional[str]:
    mode = rng.choice(["strip", "emphasise", "ellipsis", "stray"])
    if mode == "strip":
        out = re.sub(r"(?<!\d)[.,!?;:'\"](?!\d)", "", text)
    elif mode == "emphasise":
        out = re.sub(r"[.!?]", lambda m: m.group(0) * 3, text, count=2) if re.search(r"[.!?]", text) else text + "!!!"
    elif mode == "ellipsis":
        out = re.sub(r"\.(?!\d)", "...", text) if re.search(r"\.(?!\d)", text) else text + "..."
    else:
        words = text.split(" ")
        i = rng.randrange(len(words))
        words[i] = words[i] + rng.choice([",", ";", "!", "."])
        out = " ".join(words)
    if out == text:
        out = text + "!!"
    return _finish(text, out)


def casing_variation(text: str, rng: random.Random, language: str) -> Optional[str]:
    mode = rng.choice(["upper", "lower", "random"])
    if mode == "upper":
        out = text.upper()
    elif mode == "lower":
        out = text.lower()
    else:
        out = "".join(ch.swapcase() if (ch.isalpha() and rng.random() < 0.3) else ch for ch in text)
    if out == text:
        out = text.lower() if text != text.lower() else text.upper()
    return _finish(text, out)


def _replace_words(text: str, mapping: dict, rng: random.Random, prob: float, ensure_one: bool = True) -> Optional[str]:
    matches = [m for m in _LETTERS.finditer(text) if m.group(0).lower() in mapping]
    if not matches:
        return None
    chosen = [m for m in matches if rng.random() < prob]
    if not chosen and ensure_one:
        chosen = [rng.choice(matches)]
    out = text
    for m in sorted(chosen, key=lambda x: x.start(), reverse=True):
        word = m.group(0)
        rep = mapping[word.lower()]
        rep = rng.choice(rep) if isinstance(rep, list) else rep
        out = out[:m.start()] + _match_case(word, rep) + out[m.end():]
    return out


def informal_abbreviation(text: str, rng: random.Random, language: str) -> Optional[str]:
    out = _replace_words(text, _ABBREVIATIONS, rng, 0.7)
    return _finish(text, out) if out else None


def roman_urdu_spelling_variation(text: str, rng: random.Random, language: str) -> Optional[str]:
    if language not in (C.LANG_ROMAN_URDU, C.LANG_MIXED):
        return None
    out = _replace_words(text, _RU_VARIANTS, rng, 0.6)
    return _finish(text, out) if out else None


def code_switching(text: str, rng: random.Random, language: str) -> Optional[str]:
    if language == C.LANG_ENGLISH:
        mapping = _EN_TO_RU
    elif language == C.LANG_ROMAN_URDU:
        mapping = _RU_TO_EN
    elif language == C.LANG_MIXED:
        mapping = {**_RU_TO_EN, **_EN_TO_RU} if rng.random() < 0.5 else _RU_TO_EN
    else:
        return None
    out = _replace_words(text, mapping, rng, 0.35)
    return _finish(text, out) if out else None


def minor_noise(text: str, rng: random.Random, language: str) -> Optional[str]:
    chars = list(text)
    n_edits = rng.randint(1, 3)
    for _ in range(n_edits):
        idx = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if len(idx) < 4:
            break
        i = rng.choice(idx)
        op = rng.choice(["insert", "delete", "swap"])
        if op == "insert":
            chars.insert(i, chars[i])
        elif op == "delete":
            del chars[i]
        elif i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return _finish(text, "".join(chars))


PERTURBATIONS: dict[str, Callable[[str, random.Random, str], Optional[str]]] = {
    "spelling_variation": spelling_variation,
    "character_deletion": character_deletion,
    "character_substitution": character_substitution,
    "word_spacing_variation": word_spacing_variation,
    "punctuation_variation": punctuation_variation,
    "casing_variation": casing_variation,
    "informal_abbreviation": informal_abbreviation,
    "roman_urdu_spelling_variation": roman_urdu_spelling_variation,
    "code_switching": code_switching,
    "minor_noise": minor_noise,
}
assert tuple(PERTURBATIONS) == C.PERTURBATION_TYPES


def apply_perturbation(text: str, ptype: str, language: str, original_id: str, seed: int = C.SEED) -> Optional[str]:
    """Deterministically apply one perturbation type to ``text``."""
    if ptype not in PERTURBATIONS:
        raise KeyError(f"Unknown perturbation type: {ptype}")
    rng = random.Random(stable_seed(original_id, ptype, seed))
    return PERTURBATIONS[ptype](text, rng, language)


def generate_variants(text: str, language: str, original_id: str, seed: int = C.SEED,
                      types: Optional[tuple[str, ...]] = None) -> list[tuple[str, str]]:
    """Return ``[(perturbation_type, perturbed_text), ...]`` for all applicable types."""
    out = []
    for ptype in types or C.PERTURBATION_TYPES:
        new = apply_perturbation(text, ptype, language, original_id, seed)
        if new is not None and new != text:
            out.append((ptype, new))
    return out
