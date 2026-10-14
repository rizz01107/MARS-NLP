# MARS-NLP: Multilingual Adversarial Robustness for Scam Detection

## Status of this delivery

This package is an **audit-safe scaffold, not the completed message corpus**.

No message rows have been inserted because the execution environment was unable to retrieve the underlying raw UCI and Zenodo files reliably. The requirement was explicitly to use real, verifiable messages and never fabricate missing rows. An empty dataset is therefore preferable to an apparently complete but unverifiable dataset.

## Research objective

MARS-NLP is intended to support research on scam/spam detection and robustness across:

- English
- Urdu
- Roman Urdu
- Mixed/Code-Switched text

The intended binary labels are:

- Scam / Spam
- Legitimate / Ham

## Sources investigated

### 1. UCI SMS Spam Collection

- Dataset: SMS Spam Collection
- Authors: Tiago Almeida and Jos Hidalgo
- Public source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Reported size: 5,574 messages
- Language: English
- Original labels: ham / spam
- License: CC BY 4.0
- Intended use here: original real_public_dataset
- Verification: UCI's official repository identifies the corpus as a public SMS collection, gives 5,574 instances, identifies the authors, and states CC BY 4.0.

### 2. Roman Urdu Phishing SMS Dataset for Smishing Detection Research

- Creator: Ghufran Siddiq, University of Management and Technology
- Public source: https://zenodo.org/records/21810885
- Reported size: 1,000 unique messages
- Composition: 595 legitimate, 405 phishing
- Language metadata: English, Urdu; description identifies Roman Urdu/English SMS
- DOI: 10.5281/zenodo.21810885
- Intended use here: original real_public_dataset
- Verification limitation: the public record page exposed the dataset metadata and downloadable XLSX name, but the file download timed out in this execution environment. The license field was not rendered with a usable value on the record page, so no license permission is asserted here.

### 3. ScamBench

- Dataset: SCAMBENCH: A Multi-Perspective Benchmark for Online Scam Communication
- Public page: https://huggingface.co/datasets/Shouninger/ScamBench
- Reported size: 1,836 scam + 1,834 non-scam = 3,670 messages
- Language: English
- License: CC BY-NC 4.0
- Provenance: the dataset card states that scam messages are derived from publicly available scam reports and describes the messages as real-world scam messages.
- Access limitation: the repository is gated and requires account/contact-information agreement before file access. Therefore these rows were **not imported**.

## Sources deliberately excluded

### Synthetic Urdu spam datasets

Several Hugging Face Urdu spam datasets were explicitly marked by their dataset cards as synthetically generated. They are unsuitable for the REAL-DATA portion of MARS-NLP and were excluded.

### Machine-translated multilingual SMS datasets

Datasets that take an English SMS corpus and machine-translate it into Urdu or other languages were not counted as independently collected real Urdu messages. They may be useful for a separate translation/augmentation experiment, but including them as original Urdu observations would misrepresent provenance.

### Unlicensed or unclear-license GitHub datasets

Public visibility alone was not treated as permission to redistribute/repackage message text. Repositories without a sufficiently clear dataset license were not used as original rows.

## Intended schema

The final corpus will use exactly these columns:

`message_id`
`message`
`language`
`label`
`source_dataset`
`source_url`
`source_license`
`source_type`
`message_type`
`is_code_switched`
`is_perturbed`
`perturbation_type`
`original_message_id`
`split`

For original messages:

`source_type = real_public_dataset`

For controlled perturbations:

`source_type = augmented_from_real`

## Cleaning protocol

Once the raw source files are available, processing will include:

1. Empty-message removal.
2. Exact duplicate removal.
3. Near-duplicate detection without rewriting the original text.
4. Invalid-label removal.
5. Unicode normalization and encoding repair where necessary.
6. Language validation.
7. Label normalization to Scam/Spam vs Legitimate/Ham.
8. Preservation of the original message alongside any `cleaned_message` representation where normalization changes text.

## Language policy

Language will not be assigned simply from a filename or source label.

- `English`: predominantly English text.
- `Urdu`: Urdu written in Perso-Arabic script.
- `Roman Urdu`: Urdu written using Latin/Roman characters.
- `Mixed/Code-Switched`: meaningful combination of languages.
- `Unknown`: uncertain cases.

The real distribution will be preserved rather than artificially balanced.

## Augmentation protocol

Augmentation will be a separate research subset derived only from real original messages.

Possible controlled perturbations:

- spelling variation
- character deletion
- character substitution
- spacing changes
- Roman Urdu spelling variation
- punctuation variation
- casing variation
- code-switching
- common informal abbreviations

Every generated variant will have:

- `source_type = augmented_from_real`
- `is_perturbed = True`
- `original_message_id = ID of its real source`
- an explicit `perturbation_type`

No augmented message will be presented as an originally collected message.

## Leakage prevention

The split must be assigned at the original-message-group level:

- Train: 70%
- Validation: 15%
- Test: 15%

An original message and every perturbation derived from it must remain in the same split.

For datasets with source-level template IDs or related grouping information, those grouping identifiers will also be considered when preventing template leakage.

## Intended experiments

### Traditional ML
- TF-IDF + Logistic Regression
- TF-IDF + SVM

### Multilingual transformers
- mBERT
- XLM-R

### Cross-lingual evaluation
- English
- Urdu
- Roman Urdu
- Mixed/Code-Switched

### Robustness evaluation
- Original
- Spelling perturbations
- Character noise
- Code-switching
- Roman Urdu variations

### Metrics
- Accuracy
- Precision
- Recall
- Macro F1
- Confusion matrix
- Robustness degradation
- Inference time

## Ethical and privacy considerations

The final release should not reproduce private leaked datasets, personal communications obtained without an appropriate public/research basis, or unnecessary personal identifiers.

Public availability does not automatically mean that every downstream redistribution is ethically appropriate. Dataset-level licenses, source provenance, privacy considerations, and the terms attached to the original collection must be documented.

Where a source is unsuitable for redistribution but useful for methodological comparison, it should be cited rather than copied into the released corpus.

## Current verified counts

These are **source-level counts reported by the public dataset records**, not counts successfully imported into the attached dataset files:

- UCI SMS Spam Collection: 5,574
- Roman Urdu Phishing SMS Dataset: 1,000
- ScamBench: 3,670

Because the raw files could not all be retrieved and validated at message level in this execution environment, the attached MARS-NLP dataset contains **0 imported rows**.

This is intentional and prevents unsupported claims.

## Citation guidance

Use the original dataset citation in any paper/report that uses its messages. Do not cite MARS-NLP as if it were an independently collected corpus unless the provenance table and source-level transformations are retained.

## Next step for the completed corpus

To produce the actual 10k+ row MARS-NLP release without violating the real-data requirement, the raw source files need to be made available to the processing environment (for example, by uploading the UCI SMS file and the permitted Roman-Urdu dataset file). After that, the cleaning, deduplication, language validation, group-level split, perturbation generation, CSV/XLSX export, and final quality report can be performed reproducibly.
