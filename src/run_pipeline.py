"""End-to-end CPU pipeline: build -> validate -> baselines -> robustness -> cross-lingual -> errors -> explain -> results.

Usage:  python -m src.run_pipeline [--skip-build]
Transformer fine-tuning is NOT part of this CPU pipeline (needs GPU): see README, `python -m src.train_transformer`.
"""
from __future__ import annotations

import argparse
import sys

from . import config as C
from .utils import get_logger

log = get_logger()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-build", action="store_true", help="reuse data/processed/mars_nlp_processed.csv")
    a = ap.parse_args(argv)

    from . import (build_dataset, cross_lingual, eda, error_analysis, explainability, make_results, robustness,
                   train_baselines, validate_data)

    if not a.skip_build or not C.PROCESSED_CSV.exists():
        log.info("[1/9] build dataset")
        build_dataset.build()
    log.info("[2/9] validate data")
    if validate_data.main([]) != 0:
        log.error("Data validation FAILED - aborting.")
        return 1
    log.info("[3/9] EDA figures")
    eda.run()
    log.info("[4/9] baselines")
    train_baselines.run()
    log.info("[5/9] robustness")
    robustness.run()
    log.info("[6/9] cross-lingual experiments + group CV")
    cross_lingual.experiments_abc()
    cross_lingual.group_cv()
    log.info("[7/9] error analysis")
    error_analysis.run()
    log.info("[8/9] explainability")
    explainability.run()
    log.info("[9/9] final results table")
    make_results.build()
    log.info("Pipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
