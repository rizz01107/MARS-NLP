"""Pre-flight check for the GPU run (run this FIRST in Colab):  python -m src.gpu_preflight

Verifies: GPU/CUDA/torch/transformers, model-hub access + tokenizers for mBERT and XLM-R, data checksum and split
fingerprints, disk space. Writes experiments/results/gpu_preflight.json. Exit code 1 if anything blocking fails.
"""
from __future__ import annotations

import shutil
import sys

from . import config as C
from .utils import environment_info, save_json, sha256_file, split_fingerprint


def main() -> int:
    checks, ok = [], True

    def add(name, passed, detail="", blocking=True):
        nonlocal ok
        checks.append({"check": name, "status": "PASS" if passed else ("FAIL" if blocking else "WARN"), "detail": detail})
        if not passed and blocking:
            ok = False
        print(f"[{'PASS' if passed else ('FAIL' if blocking else 'WARN')}] {name}: {detail}")

    env = environment_info()
    add("torch installed", env["torch"] != "not installed", str(env["torch"]))
    add("transformers installed", env["transformers"] != "not installed", str(env["transformers"]))
    add("CUDA GPU available", bool(env["cuda_available"]), f"{env.get('gpu_name', 'none')} | CUDA {env.get('cuda_version')} | {env.get('gpu_memory_gb', '?')} GB")
    try:
        import json

        want = json.loads((C.RESULTS_DIR / "baseline_config.json").read_text())["environment"]["scikit_learn"]
        add("scikit-learn version equals the one the saved baselines were pickled with", env["scikit_learn"] == want,
            f"installed {env['scikit_learn']} / required {want}  ->  pip install -q scikit-learn=={want}  (then restart the runtime)")
    except Exception as exc:
        add("scikit-learn version check", False, f"{type(exc).__name__}: {exc}")
    free_gb = shutil.disk_usage(C.ROOT_DIR).free / 1024 ** 3
    add("disk space >= 8 GB", free_gb >= 8, f"{free_gb:.1f} GB free", blocking=False)
    try:
        from .data_loader import load_processed

        expected = (C.PROCESSED_DIR / "mars_nlp_processed.sha256").read_text().split()[0]
        add("processed CSV checksum", sha256_file(C.PROCESSED_CSV) == expected, "matches audited file" if sha256_file(C.PROCESSED_CSV) == expected else "MISMATCH")
        fp = split_fingerprint(load_processed())
        import json

        ref = json.loads((C.PROCESSED_DIR / "split_fingerprint.json").read_text())
        add("split fingerprints unchanged", fp == ref, "train/validation/test message-id sets identical to the audited split")
    except Exception as exc:
        add("data integrity", False, f"{type(exc).__name__}: {exc}")
    try:
        from transformers import AutoTokenizer

        from .train_transformer import MODEL_ZOO

        for key in ("mbert", "xlmr"):
            tok = AutoTokenizer.from_pretrained(MODEL_ZOO[key])
            enc = tok(["Aap ka account block ho gaya hai", "free prize"], padding=True, truncation=True, max_length=16)
            add(f"hub access + tokenizer: {MODEL_ZOO[key]}", "input_ids" in enc and "attention_mask" in enc, f"vocab={len(tok)}")
    except Exception as exc:
        add("hub access + tokenizers", False, f"{type(exc).__name__}: {str(exc)[:200]}")
    save_json({"environment": env, "checks": checks, "all_blocking_passed": ok}, C.RESULTS_DIR / "gpu_preflight.json")
    print("PREFLIGHT", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
