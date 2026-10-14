**Requested comparison table**

| Model                                                              |   Macro-F1 |   Precision |   Recall | English-F1                 | Roman Urdu/Mixed-F1                                                     |   Robustness Degradation | Status                                                                       |
|:-------------------------------------------------------------------|-----------:|------------:|---------:|:---------------------------|:------------------------------------------------------------------------|-------------------------:|:-----------------------------------------------------------------------------|
| TFIDF-char + LinearSVM (best CPU baseline, selected on validation) |     0.9741 |      0.9931 |   0.9226 | 0.9594                     | 1.0 (descriptive only: 5 independent test groups)                       |                   0.0029 | Executed (CPU)                                                               |
| mBERT (bert-base-multilingual-cased)                               |     0.8772 |      0.875  |   0.7226 | N/A (raw run files needed) | N/A (raw run files needed; descriptive only: 5 independent test groups) |                  -0.0055 | Executed on NVIDIA Tesla T4 (owner-verified; raw files not in this snapshot) |
| XLM-RoBERTa (xlm-roberta-base)                                     |     0.9722 |      0.9862 |   0.9226 | N/A (raw run files needed) | N/A (raw run files needed; descriptive only: 5 independent test groups) |                   0.0339 | Executed on NVIDIA Tesla T4 (owner-verified; raw files not in this snapshot) |

**With 95% cluster-bootstrap CI (overall macro-F1)**

| Model                                                              |   Macro-F1 | Macro-F1 95% CI   |   Precision |   Recall |
|:-------------------------------------------------------------------|-----------:|:------------------|------------:|---------:|
| TFIDF-char + LinearSVM (best CPU baseline, selected on validation) |     0.9741 | [0.949, 0.988]    |      0.9931 |   0.9226 |
| mBERT (bert-base-multilingual-cased)                               |     0.8772 | [0.754, 0.970]    |      0.875  |   0.7226 |
| XLM-RoBERTa (xlm-roberta-base)                                     |     0.9722 | [0.946, 0.987]    |      0.9862 |   0.9226 |
