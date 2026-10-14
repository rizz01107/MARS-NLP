# Release manifest

**Author:** Muhammad Rizwan  
**Affiliation:** Independent Researcher  
**Email:** rizwan01107@gmail.com

Generated 2026-10-14 16:37 UTC. Release archive: `MARS-NLP-release.zip`.

* Files listed: 164; total uncompressed size: 8.5 MB; in release archive: 164 files, 4.83 MB (4,832,655 bytes).
* **Excluded from the archive:** third-party raw data (`data/raw/uci`, `data/raw/zenodo`; Zenodo license unverified), the processed corpus (rebuild with `python -m src.build_dataset` and verify `data/processed/mars_nlp_processed.sha256`), and fine-tuned transformer checkpoints (reproduce with `docs/COLAB_GPU.md`).
* Also excluded: files containing dataset message text (`experiments/results/error_analysis.csv`, `reports/tables/explain_examples.csv`), local virtual environments, backups and temporary files.
* `required_for_reproduction`: yes = needed to rerun/verify; evidence = saved outputs kept for audit; no = convenience.

| path | purpose | size_bytes | sha256 | required_for_reproduction | in_release_archive |
|:---|:---|---:|:---|:---|:---|
| .github/workflows/finalize-release.yml | project documentation / configuration | 11300 | dd26066f84fc17d090a6550f13e4bf7b90d3f179ccbc84da4f6150aa20a4e177 | no | yes |
| .gitignore | git ignore rules | 804 | 7d19349018ae236050522031ca4cfadba375984cf444d1c0719bb0d93f20d525 | no | yes |
| CITATION.cff | citation metadata | 875 | 3183401b2d688bfcad3a3568a15ad02a32b9b76ed64c789512d08f55e6d0e61a | no | yes |
| LICENSE | MIT license (code only) | 1297 | 27a3a518d5b4faf12e172f206b107db1a434127071b6ee3ce07a41643b267c4a | no | yes |
| README.md | project README | 14290 | bc6f90bf79283d284c642e2185666a7747e3ae5e4f0828984bf2127294398314 | no | yes |
| RELEASE_MANIFEST.md | project documentation / configuration | 25774 | omitted (self-referential) | no | yes |
| SUBMISSION_CHECKLIST.md | submission checklist | 6055 | 40869bb840f5b787054b93d6d117ecf6404003fe3d70b32485c0a6dc2b09f23b | no | yes |
| app/app.py | Streamlit demo | 3122 | abbb35a06c4024bbd0fabf7bb1f5d63e5c2d2df80151232e7add2e98141ff4c2 | yes | yes |
| app/components/__init__.py | Streamlit demo | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | yes | yes |
| app/components/result_view.py | Streamlit demo | 1492 | 9391c9ed7c1cb388a4c71045884cfafbfbd1afcaf07045987d4407a046a47684 | yes | yes |
| data/README.md | data documentation | 2904 | 7fed9c46696a8337c62da2d96ccf8a3d1d6a6ff7a0166a7c74d53102162473d5 | yes | yes |
| data/processed/.gitkeep | checksum / split fingerprint guarding the audited split | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | yes | yes |
| data/processed/mars_nlp_processed.sha256 | processed corpus (rebuild with `src.build_dataset`) | 89 | 5fb24ea1844536f055793974dc435c893b02e7b7f65c28b30fe34f9b95b0848c | yes | yes |
| data/processed/split_fingerprint.json | checksum / split fingerprint guarding the audited split | 346 | abb109898ad15fb7c48f8506775b00a9ae47fc9659882a97237d72234b61b8d1 | yes | yes |
| data/raw/scaffold/DATASET_README.md | originally supplied empty scaffold files (audit trail) | 8052 | cfc2a1721c82ac0390e4105b81e4c75daf5ff34633555c508d9b6d0937d9c1ec | no | yes |
| data/raw/scaffold/MARS_NLP_REAL_DATASET.csv | originally supplied empty scaffold files (audit trail) | 177 | 3d08494486b15a873193daf892470fea1b6671fb6191ad3bd89a22162a775665 | no | yes |
| data/raw/scaffold/MARS_NLP_REAL_DATASET.xlsx | originally supplied empty scaffold files (audit trail) | 6262 | 3f14a4faeee9d7f9b522c21f7498adf0f50a075cde1946e13b16f0782cf3aa59 | no | yes |
| docs/COLAB_GPU.md | documentation | 5465 | cdbe70baa2d3a423df759b472f0a067ba18be3fdaa2986eff76c358f94c49447 | yes | yes |
| environment.yml | conda environment | 116 | 581960964ddbccd6b92fc8eccca1468288fead35403654883ff515bf859b022b | yes | yes |
| experiments/configs/baselines.yaml | experiment configuration | 943 | 04c2b770d5dd3376698ac6c38f243944ceeeedd368da0faf24bf14d9e0ee62d0 | yes | yes |
| experiments/configs/transformers.yaml | experiment configuration | 795 | b4451facebebfbcfa9b054bb43ffa6b5eb67ca7192848e7711c5e46c6f4637be | yes | yes |
| experiments/results/.gitkeep | raw experiment outputs (audit evidence) | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | yes | yes |
| experiments/results/baseline_config.json | raw experiment outputs (audit evidence) | 11798 | 1513da90a3b1e8a1e6eb2dba6bf4545fcf1b8352bbf721811af6d89f30e728c9 | yes | yes |
| experiments/results/baseline_predictions.csv | raw experiment outputs (audit evidence) | 646265 | ef2f6842eed6eecf62cbcf436ac572a4a90680e506f206a90ee8ceb8b24a1c51 | yes | yes |
| experiments/results/baseline_results.csv | raw experiment outputs (audit evidence) | 30447 | 636469cd7c72a755dc4d9b828ffd863857d5326ab50d0ea65b0759ceb2104334 | yes | yes |
| experiments/results/cross_lingual_results.csv | raw experiment outputs (audit evidence) | 5206 | 5719b8c94a1a28899775abbfcdfb1bd38a749a8b58fba0975791970c383d48cb | yes | yes |
| experiments/results/final_results.csv | raw experiment outputs (audit evidence) | 8833 | 5e10cd290ce5e4214d4b5477fc226d59958d2db5b63d57b9497feb255394f7ac | yes | yes |
| experiments/results/gpu_preflight.json | raw experiment outputs (audit evidence) | 1606 | 11af223e91b733ad09196a96ad0d1fbe0bae298b423a5a1ccb73a84282598e33 | yes | yes |
| experiments/results/group_cv_config.json | raw experiment outputs (audit evidence) | 185 | 84af12d01db1b992136850ce7e96f9e3a924f9c4d3ac4c2f05777b01a44ce13b | yes | yes |
| experiments/results/group_cv_folds.csv | raw experiment outputs (audit evidence) | 836 | d660b8832ba212d18f353dca898bda3f02fa951ce875d0f08de2b62ef61b3e5d | yes | yes |
| experiments/results/group_cv_results.csv | raw experiment outputs (audit evidence) | 4285 | 081f8086ad9e3187ec34c782768b424048624e2b8b7814461d110480c8869a80 | yes | yes |
| experiments/results/group_cv_robustness.csv | raw experiment outputs (audit evidence) | 19432 | 115ca09e380c78f179ca93dee6a929964e571b5de01c44f75c85a812e930a97e | yes | yes |
| experiments/results/robustness_results.csv | raw experiment outputs (audit evidence) | 77594 | c4fd128fc1e9c967778fe351bfe2f82de174b993f0f9421b1ae12463e2cead8c | yes | yes |
| experiments/results/transformer_vs_baseline.csv | raw experiment outputs (audit evidence) | 1 | 01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b | yes | yes |
| figures/mars_nlp_pipeline.pdf | publication figures | 41587 | 65b828b773262c7e421f819c443572481da8a01d72d0fee614bfe9783ddd3390 | evidence | yes |
| figures/mars_nlp_pipeline.png | publication figures | 206927 | a468d0371477fdea69678ce5659b58c15aebe7320c2ba2d682235063905cf020 | evidence | yes |
| figures/model_macro_f1_comparison.pdf | publication figures | 37270 | 8d6d59d2b3c9de62492fa22533afe36bbe5e7cdacf3742336cc6bd1fc0291461 | evidence | yes |
| figures/model_macro_f1_comparison.png | publication figures | 198799 | e13082ca25ce0c605de611d41101857c027a47c7df2e2335fa9c587b4b6eedaa | evidence | yes |
| figures/robustness_degradation.pdf | publication figures | 28303 | bc261cc1ee35046c23851a3fb3280dd13d7120c0e741be097fc7688358dc0833 | evidence | yes |
| figures/robustness_degradation.png | publication figures | 115605 | 569f21ec62dbf68ce41b9e2aabf0b2d854e6082ed5c7dea0a2cdbfe114103592 | evidence | yes |
| models/.gitkeep | project file | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | no | yes |
| notebooks/01_data_audit.ipynb | executed analysis notebooks | 29833 | 88423e7510a1823cd320adb6f10c1df3805c1dd461323e45cc6b7685dc1389f6 | no | yes |
| notebooks/02_eda.ipynb | executed analysis notebooks | 600120 | 986326484399dc8e51d9c29a63d6100f8fb15a429218d3d43ee352209a1ef738 | no | yes |
| notebooks/03_baseline_models.ipynb | executed analysis notebooks | 393064 | 03bb391d5361f2522fc61589b917587c771bb9b96e338c0508256c8f70f1c4c0 | no | yes |
| notebooks/04_transformer_models.ipynb | executed analysis notebooks | 12084 | 0680cbd8d7e7f83d147661a220c56ae7b4b441450b5e946cdae9c7d925696591 | no | yes |
| notebooks/05_robustness_experiments.ipynb | executed analysis notebooks | 398211 | b9b35a52b2eea7bcf859ac715a7ed28e70c64073090c14482f6784053cc3ed89 | no | yes |
| notebooks/06_explainability.ipynb | executed analysis notebooks | 462537 | bf35ca3b1781b1b87949c877c782b2d16daa0ffb6d6cd7aba41d6b49a2ed5405 | no | yes |
| notebooks/07_final_evaluation.ipynb | executed analysis notebooks | 242279 | 3493475275612ce4840995ac37406b8880c8174674cdade585136262314676fb | no | yes |
| predictions/README.md | raw predictions | 492 | fa507f6f9d93992578be8d0c3dbf2ccfa18df191f8a014e6584d946fb0530a02 | evidence | yes |
| predictions/baseline_predictions.csv | raw predictions | 646265 | ef2f6842eed6eecf62cbcf436ac572a4a90680e506f206a90ee8ceb8b24a1c51 | evidence | yes |
| reports/FINAL_GPU_REVIEW.md | reports | 9347 | ce3064ef223f286ab40f5a8f83ebca85fbe420e763dad75f9f2b8f2f08203a7b | evidence | yes |
| reports/FINAL_REPRODUCIBILITY_REPORT.md | reports | 24928 | 5829c41dcdd53e6070b200d65e4455408cf967498c1e07fa6f767570197ae2dd | evidence | yes |
| reports/FINAL_REPRODUCIBILITY_REPORT.md.bak_20261013T205343Z | research outputs / audit evidence | 24928 | 39d9a24f928b8c4aa112fb073e0d1fe39007ce9f086a7285e16156e907920f06 | evidence | yes |
| reports/MARS_NLP_Final_Manuscript.md | reports | 35081 | d6126872d2c93448937ffdb70ef18fef6fe5bc6b31e8414ec4c6082d711acb68 | evidence | yes |
| reports/MARS_NLP_Final_Manuscript.md.bak_20261013T205343Z | research outputs / audit evidence | 33968 | 74ef940fcc655da88915343e3fb022b75aaf4550e7b3ee490f519144a2a80e2e | evidence | yes |
| reports/MARS_NLP_Final_Manuscript.md.bak_20261013T205728Z | research outputs / audit evidence | 33968 | fc4f3ac4bda4363dfe458007f181504a02d06574047e4af9122c8a97fd4850be | evidence | yes |
| reports/MARS_NLP_Final_Manuscript.pdf | reports | 1206548 | 8f6be96ef884aaf4faf14888ac2d8be5a3aabffa0bf45387b0ef8534696a631b | evidence | yes |
| reports/MARS_NLP_Research_Report.md | reports | 56851 | bb3fdb90a312f939624a86726a9183533553d37dab09ae3ae465129426d6c315 | evidence | yes |
| reports/QA_REPORT.md | reports | 11396 | 0d975b9a543b6c8332c25eb75069fc234a1e425fb851e876583e3e4e125d22b2 | evidence | yes |
| reports/data_audit.csv | reports | 6120 | 917860313a162494b1c403d6cd2075bef081adc9a8685d0f0742981c62d4c4e4 | evidence | yes |
| reports/data_audit.md | reports | 9202 | 4d158611ee098ae8dbb3d65c10affde4fdf5f4a4799500f60e39de8c07857968 | evidence | yes |
| reports/figures/baseline_confusion_matrices.png | analysis figures | 154548 | a29e70f9d005da7faca12a20a46dc3588d323097a0774e7d1dc13c286cee8ff4 | evidence | yes |
| reports/figures/eda_class_by_language.png | analysis figures | 67656 | b50cb036dcbd4a924c1aabb1c58dcbc2b2363457ecb10f17403c0052a42616c7 | evidence | yes |
| reports/figures/eda_class_distribution.png | analysis figures | 34165 | a601e7fb8c5f43eb006374fa0c83fe16f60aad8bd886b5a20d06db25c00ec9b7 | evidence | yes |
| reports/figures/eda_code_switch_distribution.png | analysis figures | 44968 | 5ed53d6000682fce723634bc245d4bbb34848432cf7d9028b5ecb2bec42ec45e | evidence | yes |
| reports/figures/eda_language_distribution.png | analysis figures | 61786 | 8c5ee524f7879fee93865c73aedde9f9586649b778fe046bb2dafa607a92417c | evidence | yes |
| reports/figures/eda_message_length.png | analysis figures | 49570 | 5cf031dae59fc33ddc3cd8cd9688864ea364757c7f7b75859c7b9c78197448f6 | evidence | yes |
| reports/figures/eda_perturbation_distribution.png | analysis figures | 83433 | d847ec385bd12d6ff3d6169e272599ad0618fa06a9f3e3567220c4a283039c1d | evidence | yes |
| reports/figures/eda_source_distribution.png | analysis figures | 55124 | 7ca93c1c29716d5e4b656cbac2038f0c819b0ffca1a45703fc07919e579aab11 | evidence | yes |
| reports/figures/eda_split_distribution.png | analysis figures | 46694 | 40acf6ffd984e9ec70acd42742714d3e0bf68550ca27ed437698fd529c4ddd83 | evidence | yes |
| reports/figures/explain_examples_char_lr.png | analysis figures | 98259 | 6602fe38f6724ad75702f96283bddeccb3c503d7c2069cfbae0b2449b75727d3 | evidence | yes |
| reports/figures/explain_examples_word_lr.png | analysis figures | 110451 | 8ee9b81e86c9a5dd10c667c2a3ed7b9f8ef286cdecfcc5a3a712b1a6a1cb50ee | evidence | yes |
| reports/figures/explain_global_char_lr.png | analysis figures | 58720 | bb2d560f87b15f3818f536c9f7bf407b75137381f5a4afd13ac1b563795f9823 | evidence | yes |
| reports/figures/explain_global_word_lr.png | analysis figures | 70888 | 8e49d144b8e05b2727875c8923d96a4f1f73379ab850a4b6c50702e92ec9247f | evidence | yes |
| reports/figures/results_cross_lingual.png | analysis figures | 75385 | 988b01e8c262a1464d26ed1ae06168763519303c795cbdc257d7eef50d0eb63a | evidence | yes |
| reports/figures/results_efficiency.png | analysis figures | 66777 | 2875ebc5b49d18b7c5a42aeecc95c79d02eee84ec4260b68844579c1e405e205 | evidence | yes |
| reports/figures/results_model_comparison.png | analysis figures | 96528 | d06128b270f0cd90ef4c690486be5d51290aeaca44774f544517a1989a9a0050 | evidence | yes |
| reports/figures/results_robustness_by_type.png | analysis figures | 96907 | f5357e2f60f087569306b8f7032528bbe6eca9ce0667efa720ec4224a5331f2c | evidence | yes |
| reports/figures/results_robustness_cv_heatmap.png | analysis figures | 136619 | 0a212b3a3cde94e8085f2a9683182c6bc81ff09c095a987655bcd5295386b47c | evidence | yes |
| reports/pre_rerun_backup_20261013_194241/baseline_config.json | reports | 11670 | 7c27074a75b1c608b42fc3361b787d51b193e8089ecba143579ed55ca7d3fda9 | evidence | yes |
| reports/pre_rerun_backup_20261013_194241/baseline_predictions.csv | reports | 645814 | 38d1de1d7969ae84eadbe1b01f92e914c59af7bf87bca0b82dd1950a1774e215 | evidence | yes |
| reports/pre_rerun_backup_20261013_194241/baseline_results.csv | reports | 30433 | e566f6b88e424d81bd6d8afcd0d0497d4e8e48bde099e88557bcafd6177dc400 | evidence | yes |
| reports/pre_rerun_backup_20261013_194241/final_results.csv | reports | 8832 | 0944142a6947afaa8a04deb09629caafcf0be00edf464f537498b25a10db9f85 | evidence | yes |
| reports/pre_rerun_backup_20261013_194241/group_cv_results.csv | reports | 4285 | 081f8086ad9e3187ec34c782768b424048624e2b8b7814461d110480c8869a80 | evidence | yes |
| reports/pre_rerun_backup_20261013_194241/group_cv_robustness.csv | reports | 19449 | 4878ddadab94e17ffafced328d50e7db569b21f6ae36d459ce12d44b38950a50 | evidence | yes |
| reports/pre_rerun_backup_20261013_194241/robustness_results.csv | reports | 77599 | f5774c7499f2286db63067149a138c1fe8f09e5f885a236a7b2b90da83d32313 | evidence | yes |
| reports/tables/build_summary.before_rebuild.json | generated tables / QA / validation reports | 1606 | 7db8331939e8e5cc81e9213e6a2814381a88a3f8036604153e08800dd9275b2f | evidence | yes |
| reports/tables/build_summary.json | generated tables / QA / validation reports | 1606 | ac31f0cd1a0e6044905c880cd7df29632e3ab5f8df5b3edf6974c7573d3fe279 | evidence | yes |
| reports/tables/eda_class_by_language.csv | generated tables / QA / validation reports | 111 | ded1b574b66467528a3ff43f158d6aac22d2bcfaff393ad35b98607cee3f261c | evidence | yes |
| reports/tables/eda_class_distribution.csv | generated tables / QA / validation reports | 83 | 8a744ca877de57f68edbb28fb30b033033a1d1beafe86c9b0768c8e730be8df7 | evidence | yes |
| reports/tables/eda_code_switch_distribution.csv | generated tables / QA / validation reports | 82 | 39996bc3f23f767847cf4401a52156e39f9deec3df9896468a5ca6c860799c60 | evidence | yes |
| reports/tables/eda_group_structure.csv | generated tables / QA / validation reports | 124 | 9bb3b06cd57115b18dbda97b8a1b33187339d6c1b515d49c4fe925a40fd0050d | evidence | yes |
| reports/tables/eda_language_distribution.csv | generated tables / QA / validation reports | 85 | a943896095f006a195769aa69499681e58b64f5952adde3d2f4a4e4affbba72b | evidence | yes |
| reports/tables/eda_length_stats.csv | generated tables / QA / validation reports | 337 | 272226700f463fe5846244a40bcde2b1c71b188eef912add24c8f041e94af217 | evidence | yes |
| reports/tables/eda_perturbation_distribution.csv | generated tables / QA / validation reports | 275 | 83844e5b72b76cd4c1a64dced17ef8fdad570f289ca66f9e2d73b4e7a90c7261 | evidence | yes |
| reports/tables/eda_source_distribution.csv | generated tables / QA / validation reports | 80 | a67cb7ea6821913d22b0fd770e729f4a45afd2424008f306809399be78a2b21e | evidence | yes |
| reports/tables/eda_split_distribution.csv | generated tables / QA / validation reports | 69 | 063b5f86df973e27d1788521ad3d56c1430f899767e96786acac9a89cbe94cb5 | evidence | yes |
| reports/tables/error_summary.csv | generated tables / QA / validation reports | 16259 | 7b2fa4d2a96178f68a0dda8db862cc5b6d23f408dec92c1ae6c6e21d88ce3883 | evidence | yes |
| reports/tables/global_features_char_lr.csv | generated tables / QA / validation reports | 1177 | 011281faf406cfa3617e971c5b97e27cccb227d0135801423039f935aec0aed1 | evidence | yes |
| reports/tables/global_features_word_lr.csv | generated tables / QA / validation reports | 1218 | de67758321a9c1f612bb90a707c599d20d6b5e617db072b052fae4e21bfd8e59 | evidence | yes |
| reports/tables/model_comparison.csv | generated tables / QA / validation reports | 805 | 43859663a11bcbf7ad28655e444fc1e53cb5c8d2d9a7634523c3a2b65b6f6679 | evidence | yes |
| reports/tables/model_comparison.md | generated tables / QA / validation reports | 2322 | 8ba3d78621fc75c416b98add870a89d0c283946c2e9682d8bdf4ec892edbd8d9 | evidence | yes |
| reports/tables/qa_report.json | generated tables / QA / validation reports | 13244 | 59da5fa3549df020d993da532e91f4069fec24be86ae58acdb2661a2424cb7dc | evidence | yes |
| reports/tables/qa_report.md | generated tables / QA / validation reports | 7230 | 0a6fba4978f39af87f37623dec4ed13d73ea9bb367e04a4f282c43caeed1148d | evidence | yes |
| reports/tables/robustness_comparison.csv | generated tables / QA / validation reports | 394 | 7ca03368ae291da11f4fbe4934a9ab92706a4b22186530f68860acc6164e7d45 | evidence | yes |
| reports/tables/validation_report.json | generated tables / QA / validation reports | 6725 | a423dd5f02bc670dc0fb12e7717774cb2cecd0327e354413860a70f8a895c962 | evidence | yes |
| reports/tables/validation_report.md | generated tables / QA / validation reports | 2628 | 929d7322fbbb1fddacdf7ae7ec000b6a1a954033f28a11e1a1d3fce360f8ab39 | evidence | yes |
| requirements-colab.txt | Colab dependencies | 424 | 6761d1c3cfabdf8a2198cdd2b3a2dccfbd568b435de1b583f32a55d7147b4c50 | yes | yes |
| requirements.txt | dependencies | 691 | 0535c0efba0b4736cbaa5f93d25942e18afc92b4a89e208becbc54db53c3a6fa | yes | yes |
| results/ERROR_COUNTS.csv | curated final results | 1432 | 2ea6d0b98f179843428cdb05eed17b84160462477c09a81e53423850b0747397 | evidence | yes |
| results/ERROR_COUNTS.csv.bak_20261013T205343Z | research outputs / audit evidence | 1412 | b2d01c4f1dede64d86f940d3cc60aa47d7e3ddfb490d725c3e432a0d49524aa0 | evidence | yes |
| results/FINAL_RESULTS.csv | curated final results | 1817 | 23b9c5e9b170fffa51bf06f9c8d79dd250ff32f4062f6383c6a03fcefbe59a1a | evidence | yes |
| results/README.md | curated final results | 1169 | 772ac1a3c2c6c04ac37ddb919fb0605bae79eaefd938c3fac7bf2ed1bfc9078b | evidence | yes |
| results/ROBUSTNESS_RESULTS.csv | curated final results | 11139 | 87220554c8bb79e8824c191074361b714fc3f03890df33589bd820336b446db4 | evidence | yes |
| results/STATISTICAL_COMPARISONS.csv | curated final results | 1223 | b2d94b5b26b866daaf2d60a378bb8e3ef1b63ee6a22133b6adbf24753905812f | evidence | yes |
| results/verified_transformer_results.json | curated final results | 6808 | 992532c946b06d26134e5ecf4a4d0636b710ffcf169804eaedd4f773fabf1447 | yes | yes |
| results/verified_transformer_results.json.backup_20261013_202529 | curated final results | 6513 | 1d1a79db76c800380074bf1ba96e43b4b3cd74db1fe4c49e10d54e329b77a699 | yes | yes |
| results/verified_transformer_results.json.bak_20261013T205343Z | research outputs / audit evidence | 6808 | 629435bfd3411799d89ad978c827afd3cadca618ae77b2fbdc17fe9caf7fed99 | evidence | yes |
| scripts/build_manuscript_pdf.sh | run scripts | 669 | ded55d86c556301fb5dc86fbee9365139348f043d194caf8d81264cde94c2eaf | yes | yes |
| scripts/colab_run_all.sh | run scripts | 1763 | ab7cc3b57f146ad27392b77c8f3d3e318ca8ee9a95eceb44da0417d20aa303b0 | yes | yes |
| scripts/manuscript.css | run scripts | 690 | 483fe1b45d3ec1c74a0e1a4396f7580d143c366137201e754f4487208f1e4671 | yes | yes |
| src/__init__.py | source code | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | yes | yes |
| src/audit.py | source code | 13276 | 5b8066ab6e4ea9f22f489f0048cfe6c354507df0cbff72e5c0ea5e27d5c943bb | yes | yes |
| src/build_dataset.py | source code | 7991 | d0eb994e1682545edd3c716dc8ed124e2426cde9ff2b31e037375b678a8b3874 | yes | yes |
| src/build_dataset.py.before_leakage_fix | source code | 7322 | 17bb0fb2fd82b91cb878857e5ac97f2a164c486abce0243294c6ee454d28374e | yes | yes |
| src/compare_models.py | source code | 7586 | c97fbf0b70c86dbe8fd380bdddd8144bd0bfa058b2965c268c882da61ca4beba | yes | yes |
| src/config.py | source code | 3566 | 52b30ac534df68d9ee44d7cea2fb18cab1ec35cfab88031b01afb50b5fb7c41c | yes | yes |
| src/config.py.before_license_update | source code | 3601 | dac27da8068cce54c2995c64271d057d500c012a071a398175f492ca26ef6e0a | yes | yes |
| src/cross_lingual.py | source code | 9655 | 747b01e7447bc1ae20b2c198b7f167fec458336094ba356a71269d562610c342 | yes | yes |
| src/data_loader.py | source code | 3871 | 86e07c9f4a5ac2225a64e72efb340ee4928494a9aa0e7eea701045aaa62efeb3 | yes | yes |
| src/eda.py | source code | 7050 | 7615a6d089c5a48dcffeb76380588c7e9ea75b014a62f9cf9d4076660dcace87 | yes | yes |
| src/error_analysis.py | source code | 5287 | 2e0f34ddece5a6293d18fea403ee032e61ba113145a872a76648824300835c2f | yes | yes |
| src/evaluation.py | source code | 8970 | 395f901f1c6c4c9a46946cd53679fb628b027a2f29cece75bd923c2ed991a5bd | yes | yes |
| src/explainability.py | source code | 7309 | 840d1ba79365930fcf76d8bed4aa7f525a103ab6ad83cf592a82ea54bdbee5cd | yes | yes |
| src/features.py | source code | 1105 | ba8eebfaf781a5e578d46dd91abe692beed7c155c7eb1440fbc4b72879ec82fb | yes | yes |
| src/final_tables.py | source code | 10313 | b01fabf3f8d4034f122f18341ddd782eabe58b0689cb6776a6cdd5408c98e394 | yes | yes |
| src/gpu_preflight.py | source code | 3422 | 450146e3ab464e05249ad2b809a35303b492764cc2c3bc1961cff60338b633c9 | yes | yes |
| src/make_final_package.py | source code | 31280 | 42c0068b4f9708b5c292b69df03f335c0631fe9118bb83922482c93b12e382ab | yes | yes |
| src/make_gpu_review.py | source code | 13584 | 5d29148076e7d48ad768d8c2e54e2fd66acb9382cc367ae43bae7b5ed04ef92b | yes | yes |
| src/make_manuscript.py | source code | 37997 | 347fe852ddbbb7c5ab47db0e81ab6423a80615cc6863bbcab780afdbae23e3e1 | yes | yes |
| src/make_publication_figures.py | source code | 6273 | 99ad058b900ae6009c36fae41ae1b7733999a4b6dc08901b23e940604ffdbe15 | yes | yes |
| src/make_report.py | source code | 59333 | cbb3265fab3cf22bc1fd51ccc6257ecf307743041ac80760954f134995318fad | yes | yes |
| src/make_results.py | source code | 9273 | e4f4138fc15bb8155ea0c8333729e5eb8388a68ed68173ae64cc15a0c042ae97 | yes | yes |
| src/perturbations.py | source code | 12648 | a5613b6fb96cfe878fdaaaf9fcf61b40fe4ee07d2957e0078c6adbac355c149a | yes | yes |
| src/plots.py | source code | 8100 | a3fdd9bdc48cba78bf3f4e32220333e83af64333ef47d7a75cfb7a48b2608702 | yes | yes |
| src/prediction.py | source code | 7477 | 24d6940148b358ac0998c36edebbdc295ac22ed02516ef59b458d8d1437eeea8 | yes | yes |
| src/preprocessing.py | source code | 11245 | feee4e8ea29dafe60dccdd544a661ebd958ca55c021b3bb15e32a1bb46b8210c | yes | yes |
| src/qa_check.py | source code | 35281 | baa1602bb3bd075b1f23bfcf5cc9b6310cd4a856b4ec0cfe103eac05389cfe5d | yes | yes |
| src/qa_check.py.bak_20261013T205343Z | source code / reproducibility tooling | 34993 | ad4761f57643e6882bf302e8bd67643842f59394eed07560e64f6559d9438af0 | yes | yes |
| src/robustness.py | source code | 6273 | 6fac3d153655f50f304b00087c23b25bcff7a58d62f2c9f62ae39e462da4c14c | yes | yes |
| src/run_pipeline.py | source code | 1666 | c35eb9b8f008eb4b7dff07b48db0e9046264dfc274fe7f7594026f34d7d6a15e | yes | yes |
| src/splitting.py | source code | 4757 | 9e45900564642531fd2b14000fa14b49381782aa881df64adda48d6f2239a8d7 | yes | yes |
| src/train_baselines.py | source code | 10833 | f91f9af5579b6184a9bebb3c545dfed031e6e30be35735a80df02cdc2a6dc699 | yes | yes |
| src/train_transformer.py | source code | 20505 | 3e6f94b3614db88f31ec048727bd6b3f14926cc05c9f125ba8b8dc7696a75988 | yes | yes |
| src/utils.py | source code | 4041 | 67cd167980f1dffe94f7276e58e9c2cf1d5e2bcc12c1aa2271ffc94bdd09c8d7 | yes | yes |
| src/validate_data.py | source code | 9394 | dc42983dbbe635e0d47d8e61fa8a4826ebb94b5d6032b90d6068829b750ed573 | yes | yes |
| src/verified.py | source code | 892 | 73bfd1e7e414c5a720ff249ec582b7986393d8f1e63f6961bf7f1ae5ce0ed2cd | yes | yes |
| tests/conftest.py | pytest suite | 2230 | 50f97fa242b2e29e4da2e1e3a234205339b7cacd66ad37bcc8cf671ff6fa3a60 | yes | yes |
| tests/test_app.py | pytest suite | 1598 | dffba6e21e4600441279011d9cfb928295ab9ed0b38858b19a805afc2a470ea0 | yes | yes |
| tests/test_data.py | pytest suite | 3158 | ed45cc9bd55ada1cf0c5d6adcc630022161ae2bb89254e9fd16ec1a76f4f6df0 | yes | yes |
| tests/test_prediction.py | pytest suite | 4322 | 0a6122523cea01e1629549fb9ab7937f926686f95a943f5ac258b66baf2bb622 | yes | yes |
| tests/test_preprocessing.py | pytest suite | 4405 | ccd0e88d94d4f1e6347c3691ca0405f4054018c627f136f34c51c6d6d788d6e0 | yes | yes |
| tests/test_splitting.py | pytest suite | 2558 | ee76b85202e4b336ae71a2e85bd878e49e46950f044b96971bbb85e23ecc56bd | yes | yes |
| tests/test_transformer_pipeline.py | pytest suite | 7858 | fa6674803a728a79b6063c6edc4a760465a10aa7af730f5e952e51dff01c3e26 | yes | yes |
