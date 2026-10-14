# Data validation report

Rows: 58191 (originals 6160, variants 52031)

**Critical failures: 0 | warnings: 0 | checks: 36**

| Check | Status | Detail | Count |
|---|---|---|---|
| required_columns | PASS | all present | 0 |
| boolean_columns_valid | PASS | 0 invalid boolean values | 0 |
| no_missing_message_id | PASS | 0 missing | 0 |
| no_missing_message | PASS | 0 missing | 0 |
| no_missing_cleaned_message | PASS | 0 missing | 0 |
| no_missing_label | PASS | 0 missing | 0 |
| no_missing_language | PASS | 0 missing | 0 |
| no_missing_source_type | PASS | 0 missing | 0 |
| no_missing_split | PASS | 0 missing | 0 |
| no_missing_group_id | PASS | 0 missing | 0 |
| no_missing_source_dataset | PASS | 0 missing | 0 |
| no_missing_perturbation_type | PASS | 0 missing | 0 |
| perturbed_rows_have_original_message_id | PASS | 0 perturbed rows without original_message_id | 0 |
| originals_have_no_original_message_id | PASS | 0 originals with original_message_id (each is its own source) | 0 |
| unique_message_id | PASS | 0 duplicated message_id | 0 |
| allowed_labels | PASS | invalid: [] | 0 |
| allowed_languages | PASS | invalid: [] | 0 |
| allowed_source_type | PASS | invalid: [] | 0 |
| source_type_consistent_with_is_perturbed | PASS | 0 inconsistent rows | 0 |
| allowed_perturbation_type | PASS | invalid: [] | 0 |
| originals_have_perturbation_type_none | PASS | 0 originals with a perturbation type | 0 |
| allowed_split | PASS | invalid: [] | 0 |
| original_message_id_refers_to_original | PASS | 0 variants point to a missing/non-original id | 0 |
| variant_label_equals_original_label | PASS | 0 variants with changed label | 0 |
| variant_group_equals_original_group | PASS | 0 mismatches | 0 |
| variants_differ_from_original_text | PASS | 0 identical to original | 0 |
| no_duplicate_original_messages | PASS | 0 duplicated original messages | 0 |
| perturbed_texts_not_identical_to_other_originals | PASS | 0 perturbed texts equal some original | 0 |
| no_group_in_multiple_splits | PASS | 0 groups, e.g. [] | 0 |
| variants_share_split_with_original | PASS | 0 variants | 0 |
| no_exact_text_overlap_between_splits | PASS | 0 shared messages | 0 |
| code_switched_flag_matches_language | PASS | 0 mismatches | 0 |
| split_train_has_both_classes | PASS | train: labels ['Legitimate', 'Scam'] | 4308 |
| split_validation_has_both_classes | PASS | validation: labels ['Legitimate', 'Scam'] | 917 |
| split_test_has_both_classes | PASS | test: labels ['Legitimate', 'Scam'] | 935 |
| residual_near_duplicate_similarity_test_vs_train | INFO | mean max-sim=0.329; share>=0.90: 0.000; share>=0.80: 0.000 | 0 |
