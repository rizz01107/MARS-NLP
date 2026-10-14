# MARS-NLP data audit

All statistics below are computed by `python -m src.audit` from the files in `data/raw/` and `data/processed/`.

## 1. Originally provided files (scaffold)

* CSV: 14 columns, **0 data rows**; XLSX main sheet: **0 data rows**; headers identical: True.
* README states the scaffold contains **0 imported rows** - consistent with the files.
* **Authoritative source:** since the scaffold is empty, the raw public sources (UCI, Zenodo) supplied afterwards are authoritative; nothing was overwritten (the scaffold files are kept unchanged in `data/raw/scaffold/`).

## 2. Raw public sources

### 2a. UCI SMS Spam Collection

| Metric | Value |
|---|---|
| rows | 5574 |
| encoding | utf-8 |
| malformed_lines | 0 |
| ham | 4827 |
| spam | 747 |
| empty_messages | 0 |
| exact_duplicate_messages (raw text) | 403 |
| exact_duplicate_messages (after cleaning) | 414 |
| duplicate_groups_with_conflicting_labels | 0 |
| near_duplicates_beyond_exact (case/punct-insensitive) | 32 |
| messages_with_non_ascii | 483 |
| messages_with_html_entities (&lt; &amp; ...) | 309 |
| length_min/median/mean/max | 2/62/80.5/910 |

License: CC BY 4.0 (as reported on UCI page; bundled readme: free use, copyright + citation request). The bundled readme grants free use with no limitations except copyright (Almeida & Gomez Hidalgo) and a citation request; the CC BY 4.0 statement originates from the UCI web page and was **not re-verified offline**.

### 2b. Zenodo Roman Urdu phishing SMS (`phishing_dataset_1000.xlsx`)

| Metric | Value |
|---|---|
| rows | 1000 |
| columns | ID, Raw_text, Labels, Source_Type, Normalized_text |
| missing_values_total | 0 |
| legitimate(0) | 595 |
| phishing(1) | 405 |
| n_message_categories | 16 |
| categories_with_mixed_labels | 0 |
| unique_IDs | True |
| exact_duplicate_Raw_text | 0 |
| duplicate_Normalized_text | 91 |
| unique_after_digit_masking | 578 |
| unique_after_digit+punctuation_masking | 502 |
| templates_occurring_more_than_once | 130 |
| messages_in_repeated_templates | 552 |
| largest_template_size | 78 |
| messages_in_Perso-Arabic_script | 0 |
| license_in_file | none found |

License: UNVERIFIED (no license field in provided files; confirm on Zenodo record before redistribution).

**Finding:** the messages are highly templated (see counts above): near-identical frames with swapped brand names, amounts and IDs. Random splitting would leak templates, so splitting is done at the template-group level (Section 4).

## 3. Processed MARS-NLP corpus (`data/processed/mars_nlp_processed.csv`)

| Metric | Value |
|---|---|
| rows_total | 58239 |
| columns | 18 |
| originals | 6160 |
| perturbed_variants | 52079 |
| missing_values_total (originals_original_message_id excluded) | 0 |
| duplicate_message_id | 0 |
| duplicate_original_messages | 0 |
| n_near_duplicate_groups (originals) | 4970 |
| variants_per_original (mean) | 8.45 |

**Label distribution (originals)**

| label      |    n |   share |
|:-----------|-----:|--------:|
| Legitimate | 5113 |    0.83 |
| Scam       | 1047 |    0.17 |

**Language distribution (originals, heuristic)**

| language            |    n |
|:--------------------|-----:|
| English             | 5340 |
| Mixed/Code-Switched |  430 |
| Roman Urdu          |  359 |
| Unknown             |   31 |

**Language x label (originals)**

| language            |   Legitimate |   Scam |
|:--------------------|-------------:|-------:|
| English             |         4700 |    640 |
| Mixed/Code-Switched |          157 |    273 |
| Roman Urdu          |          227 |    132 |
| Unknown             |           29 |      2 |

**Source distribution (originals)**

| source_dataset                                                        |    n |
|:----------------------------------------------------------------------|-----:|
| SMS Spam Collection (UCI)                                             | 5160 |
| Roman Urdu Phishing SMS Dataset (Zenodo, DOI 10.5281/zenodo.21810885) | 1000 |

**Source x language (originals)**

| source_dataset           |   English |   Mixed/Code-Switched |   Roman Urdu |   Unknown |
|:-------------------------|----------:|----------------------:|-------------:|----------:|
| Roman Urdu Phishing SMS  |       216 |                   427 |          357 |         0 |
| SMS Spam Collection (UCI |      5124 |                     3 |            2 |        31 |

**Perturbation distribution (variants)**

| perturbation_type             |    n |
|:------------------------------|-----:|
| punctuation_variation         | 6160 |
| casing_variation              | 6157 |
| character_deletion            | 6148 |
| character_substitution        | 6148 |
| word_spacing_variation        | 6053 |
| spelling_variation            | 6042 |
| minor_noise                   | 5908 |
| code_switching                | 4640 |
| informal_abbreviation         | 4047 |
| roman_urdu_spelling_variation |  776 |

**Split x label (originals)**

| split      |   Legitimate |   Scam |
|:-----------|-------------:|-------:|
| train      |         3572 |    736 |
| validation |          761 |    156 |
| test       |          780 |    155 |

**Split x source (originals)**

| split      |   Roman Urdu Phishing SMS  |   SMS Spam Collection (UCI |
|:-----------|---------------------------:|---------------------------:|
| train      |                        696 |                       3612 |
| validation |                        143 |                        774 |
| test       |                        161 |                        774 |

**Split x language (originals)**

| split      |   English |   Mixed/Code-Switched |   Roman Urdu |   Unknown |
|:-----------|----------:|----------------------:|-------------:|----------:|
| train      |      3729 |                   228 |          328 |        23 |
| validation |       829 |                    60 |           24 |         4 |
| test       |       782 |                   142 |            7 |         4 |

**Split x label (all rows incl. variants)**

| split      |   Legitimate |   Scam |
|:-----------|-------------:|-------:|
| train      |        33320 |   7401 |
| validation |         7058 |   1577 |
| test       |         7306 |   1577 |

* Near-duplicate groups: **4970**; groups with >1 message: 144; largest group: 143 messages.
* Zenodo originals form only **19 groups** (largest 143 messages) -> every Zenodo test number rests on very few independent groups.

### Original / augmented relationship

* variants_whose_original_exists: 52079
* variants_total: 52079
* variants_with_split_equal_to_original: 52079
* variants_with_label_equal_to_original: 52079

### Language-heuristic spot checks

Language labels come from a transparent lexicon heuristic (`src/preprocessing.py`), not a trained model. Sanity checks:

* UCI messages flagged Roman Urdu/Mixed: **5** of 5160 (mostly false positives on short English texts, e.g. 'ho ho'; one or two are genuine Roman-Urdu/Hindi text):
  * `Goodmorning sleeping ga.`
  * `Ho ho - big belly laugh! See ya tomo`
  * `Waqt se pehle or naseeb se zyada kisi ko kuch nahi milta,Zindgi wo nahi he jo hum sochte h`
  * `Indians r poor but India is not a poor country. Says one of the swiss bank directors. He s`
  * `Am on a train back from northampton so i'm afraid not! I'm staying skyving off today ho ho`

* Zenodo messages labelled English: **216** (fully English bank/OTP-style notification templates; individual messages are not reproduced here because the Zenodo license is unverified). The source's metadata calls the whole set 'Roman Urdu', so **language is not equivalent to source** here.
* Urdu (Perso-Arabic script) messages in the corpus: **0** -> Urdu cannot be evaluated.
* Language 'Unknown' (too little alphabetic evidence): 31 originals.

### Sources and licenses

| source_dataset                                                        | source_license                                                                                  |    n |
|:----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|-----:|
| Roman Urdu Phishing SMS Dataset (Zenodo, DOI 10.5281/zenodo.21810885) | UNVERIFIED (no license field in provided files; confirm on Zenodo record before redistribution) | 1000 |
| SMS Spam Collection (UCI)                                             | CC BY 4.0 (as reported on UCI page; bundled readme: free use, copyright + citation request)     | 5160 |

## 4. Leakage methodology and findings

* Exact duplicates are removed first (keep first occurrence; none had conflicting labels).
* Near-duplicate/templated messages are merged into groups (character 3-5-gram TF-IDF cosine >= 0.8 on digit-masked, lower-cased text; single-linkage).
* Whole groups are assigned to train/validation/test (70/15/15 by message count, stratified by source and label); perturbed variants inherit the split of their original.
* `python -m src.validate_data` verifies: no group in more than one split, every variant in its original's split, no exact text overlap, and reports residual cross-split similarity (see `reports/tables/validation_report.md`).
