import pytest

from src import config as C
from src.perturbations import PERTURBATIONS, apply_perturbation, generate_variants
from src.preprocessing import (clean_text, detect_language, detect_language_with_confidence, drop_exact_duplicates,
                               near_duplicate_groups, normalize_label, normalize_language, repair_mojibake)
import pandas as pd


def test_clean_text_unicode_whitespace_html():
    assert clean_text("  Hello\u200b   world \n\t!  ") == "Hello world !"
    assert clean_text("a &lt;#&gt; b &amp; c") == "a <#> b & c"
    assert clean_text("\ufb01ne") == "fine"                       # NFKC ligature
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_clean_text_keeps_case_punctuation_and_urdu():
    s = "FREE!!! Aap ka account... \u0622\u067e \u06a9\u0627"
    out = clean_text(s)
    assert "FREE!!!" in out and "\u0622\u067e" in out and "..." in out


def test_zwnj_is_preserved():
    assert "\u200c" in clean_text("\u0645\u06cc\u200c\u062e\u0648\u0627\u06c1\u0645")


def test_mojibake_repair():
    broken = "caf\u00c3\u00a9"
    assert repair_mojibake(broken) == "caf\u00e9"
    assert repair_mojibake("plain text") == "plain text"


@pytest.mark.parametrize("raw,expected", [("ham", "Legitimate"), ("SPAM", "Scam"), (0, "Legitimate"), (1, "Scam"), ("1", "Scam"),
                                          ("phishing", "Scam"), ("Legitimate", "Legitimate")])
def test_label_mapping(raw, expected):
    assert normalize_label(raw) == expected


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        normalize_label("maybe")


def test_label_ids_consistent():
    assert C.LABEL2ID == {"Legitimate": 0, "Scam": 1} and C.ID2LABEL[1] == "Scam"


def test_language_normalisation():
    assert normalize_language("roman_urdu") == C.LANG_ROMAN_URDU and normalize_language("EN") == C.LANG_ENGLISH
    with pytest.raises(ValueError):
        normalize_language("klingon")


def test_language_detection_three_scripts():
    assert detect_language("Please call me now, your account is blocked") == C.LANG_ENGLISH
    assert detect_language("Aap ka account block ho gaya hai, apna PIN bhejein") in (C.LANG_ROMAN_URDU, C.LANG_MIXED)
    assert detect_language("\u0622\u067e \u06a9\u0627 \u0627\u06a9\u0627\u0624\u0646\u0679 \u0628\u0644\u0627\u06a9 \u06c1\u0648 \u06af\u06cc\u0627") == C.LANG_URDU
    assert detect_language("") == C.LANG_UNKNOWN and detect_language("1234 !!!") == C.LANG_UNKNOWN
    assert detect_language_with_confidence("ok")[1] == "low"


def test_exact_duplicate_removal_and_conflicts():
    df = pd.DataFrame({"cleaned_message": ["a", "a", "b", "c", "c"], "label": ["Scam", "Scam", "Legitimate", "Scam", "Legitimate"]})
    out, info = drop_exact_duplicates(df)
    assert list(out["cleaned_message"]) == ["a", "b"]            # 'c' has conflicting labels -> removed entirely
    assert info["n_removed"] == 3 and info["conflicting_label_messages"] == 1


def test_near_duplicate_groups_merge_templates_not_unrelated():
    texts = ["Your OTP for bank A is 12345. Do not share it.", "Your OTP for bank A is 99887. Do not share it.",
             "Lunch at noon tomorrow?", "Your OTP for bank A is 55555. Do not share it!"]
    g = near_duplicate_groups(texts, threshold=0.8)
    assert g[0] == g[1] == g[3] and g[2] != g[0]
    assert list(near_duplicate_groups([], 0.8)) == []


def test_perturbations_deterministic_and_change_text():
    msg = "Please call us now to claim your free prize, thanks"
    for ptype in PERTURBATIONS:
        a = apply_perturbation(msg, ptype, C.LANG_ENGLISH, "ID1")
        b = apply_perturbation(msg, ptype, C.LANG_ENGLISH, "ID1")
        assert a == b                                              # deterministic
        assert a is None or a != msg
    assert apply_perturbation(msg, "casing_variation", C.LANG_ENGLISH, "ID1") is not None
    assert apply_perturbation(msg, "roman_urdu_spelling_variation", C.LANG_ENGLISH, "ID1") is None   # not applicable
    with pytest.raises(KeyError):
        apply_perturbation(msg, "nope", C.LANG_ENGLISH, "ID1")


def test_roman_urdu_and_code_switch_variants():
    ru = "Aap ka account band ho gaya hai, inam hasil karein"
    types = dict(generate_variants(ru, C.LANG_ROMAN_URDU, "R1"))
    assert "roman_urdu_spelling_variation" in types and "code_switching" in types
    assert all(v != ru for v in types.values())
