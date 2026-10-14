"""Fine-tune multilingual transformers (mBERT, XLM-R, optional DistilmBERT) for scam/spam detection.

Usage (GPU recommended, e.g. Google Colab):
    python -m src.train_transformer --model mbert --epochs 4 --batch-size 32 --robustness
    python -m src.train_transformer --model xlmr  --epochs 4 --batch-size 32 --robustness
    python -m src.train_transformer --model xlmr  --train-languages English --eval-languages "Roman Urdu" "Mixed/Code-Switched"
    python -m src.train_transformer --model mbert --smoke   # tiny/fast run to check the pipeline

STATUS: the final mBERT / XLM-R results were produced with this script on a Google Colab NVIDIA Tesla T4 (see docs/COLAB_GPU.md and
results/verified_transformer_results.json). Runs with ``--smoke`` or ``--tiny-offline`` are pipeline checks only: their outputs are tagged
``*_smoke`` and are never used for results.

Design: pretrained encoder + classification head (never trained from scratch); fixed seeds; CUDA if available else
CPU; dynamic padding with attention masks; max_length truncation; class-weighted cross-entropy (as the baselines use
balanced class weights); AdamW + linear warm-up/decay; model selection and early stopping on validation macro-F1;
the test split is evaluated once at the end. Training uses ORIGINAL training messages (``--augment`` adds their
perturbed variants). torch/transformers are imported lazily so the rest of the repository works without them.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import get_split, load_processed
from .evaluation import benchmark_latency, compute_metrics
from .train_baselines import evaluate_subsets
from .utils import (ensure_dir, environment_info, get_logger, save_json, set_seed, sha256_file, split_fingerprint)

log = get_logger()

MODEL_ZOO = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "distilmbert": "distilbert-base-multilingual-cased",
}
PRETTY = {"mbert": "mBERT (bert-base-multilingual-cased)", "xlmr": "XLM-RoBERTa (xlm-roberta-base)",
          "distilmbert": "DistilmBERT (distilbert-base-multilingual-cased)"}
DEFAULTS = {"max_length": 128, "epochs": 4, "batch_size": 32, "lr": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1,
            "patience": 2, "seed": C.SEED}


# --------------------------------------------------------------------------- helpers usable without torch
def resolve_model_name(key_or_name: str) -> str:
    """Map a short key (mbert/xlmr/distilmbert) to a Hugging Face model id; other strings pass through."""
    return MODEL_ZOO.get(key_or_name, key_or_name)


def label_maps() -> tuple[dict, dict]:
    """(id2label, label2id) used for the classification head - identical to the baselines."""
    return dict(C.ID2LABEL), dict(C.LABEL2ID)


def select_device(prefer: Optional[str] = None) -> str:
    """'cuda' when available, otherwise 'cpu'. Never raises if torch or CUDA is missing."""
    if prefer == "cpu":
        return "cpu"
    try:
        import torch  # type: ignore

        ok = bool(torch.cuda.is_available())
    except Exception:
        ok = False
    if prefer == "cuda" and not ok:
        log.warning("CUDA requested but not available - falling back to CPU")
    return "cuda" if ok else "cpu"


def class_weights(labels: Sequence[int]) -> list[float]:
    """Balanced class weights n / (2 * n_c), as sklearn's class_weight='balanced'."""
    y = np.asarray(labels)
    n = len(y)
    out = []
    for c in (0, 1):
        k = int((y == c).sum())
        out.append(float(n / (2 * k)) if k else 1.0)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="xlmr", help=f"one of {list(MODEL_ZOO)} or a Hugging Face model id / local path")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--augment", action="store_true", help="also train on perturbed variants of training originals")
    ap.add_argument("--train-languages", nargs="*", default=None, help="restrict training languages (zero-shot transfer)")
    ap.add_argument("--eval-languages", nargs="*", default=None, help="restrict test languages")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run (<=256 train samples, 1 epoch, <=128 val/test); marked smoke_test")
    ap.add_argument("--tiny-offline", action="store_true",
                    help="PIPELINE TEST ONLY: random tiny BERT + locally trained tokenizer, no model download (implies --smoke)")
    ap.add_argument("--skip-checksum", action="store_true", help="do not verify data/processed checksum (not recommended)")
    ap.add_argument("--robustness", action="store_true", help="also evaluate perturbed test variants")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap


def run_tag(args) -> str:
    key = args.model if args.model in MODEL_ZOO else Path(str(args.model)).name.replace("/", "-")
    tag = key
    if args.train_languages:
        tag += "_train-" + "+".join(l.replace(" ", "").replace("/", "") for l in args.train_languages)
    if args.augment:
        tag += "_aug"
    if getattr(args, "tiny_offline", False):
        tag += "_tinyoffline"
    if args.smoke or getattr(args, "tiny_offline", False):
        tag += "_smoke"
    return tag


# --------------------------------------------------------------------------- torch-dependent parts
def _require_torch():
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("This step needs `torch` and `transformers` (pip install torch transformers); run it on a "
                           "GPU machine such as Google Colab.") from exc
    return torch, transformers


class TextDataset:
    """Texts + integer labels; tokenisation happens per batch (dynamic padding)."""

    def __init__(self, texts: Sequence[str], labels: Optional[Sequence[int]] = None):
        self.texts = list(texts)
        self.labels = None if labels is None else [int(x) for x in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return (self.texts[i], -1 if self.labels is None else self.labels[i])


def make_collate(tokenizer, max_length: int):
    torch, _ = _require_torch()

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def predict_proba_texts(model, tokenizer, texts: Sequence[str], device: str, max_length: int = 128,
                        batch_size: int = 64) -> np.ndarray:
    """P(Scam) for each text (softmax over logits; attention masks come from the tokenizer)."""
    torch, _ = _require_torch()
    model.eval()
    out = []
    texts = list(texts)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tokenizer(texts[s:s + batch_size], padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt").to(device)
            logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def load_finetuned(path: Path):
    """Load a model directory written by :func:`train` -> (model, tokenizer, device)."""
    torch, tf = _require_torch()
    device = select_device()
    tok = tf.AutoTokenizer.from_pretrained(str(path))
    model = tf.AutoModelForSequenceClassification.from_pretrained(str(path)).to(device)
    model.eval()
    return model, tok, device


def verify_data_integrity() -> dict:
    """Abort if data/processed/mars_nlp_processed.csv differs from the audited file (guards the split)."""
    sha_file = C.PROCESSED_DIR / "mars_nlp_processed.sha256"
    actual = sha256_file(C.PROCESSED_CSV)
    if not sha_file.exists():
        raise RuntimeError(f"{sha_file} missing - cannot prove the audited split is in use.")
    expected = sha_file.read_text().split()[0]
    if actual != expected:
        raise RuntimeError("Processed dataset checksum MISMATCH (expected %s..., got %s...). Do not train on a modified or "
                           "rebuilt dataset; re-upload the original data/processed files." % (expected[:12], actual[:12]))
    return {"sha256": actual, "verified": True}


def build_tiny_offline_model(texts: Sequence[str], out_dir: Path, seed: int = C.SEED) -> Path:
    """Create a tiny random BERT + WordPiece tokenizer trained on ``texts`` (pipeline test only, no downloads)."""
    _, tf = _require_torch()
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

    ensure_dir(out_dir)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(list(texts), trainers.WordPieceTrainer(
        vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = tf.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                      sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(out_dir)
    import torch

    torch.manual_seed(seed)
    cfg = tf.BertConfig(vocab_size=len(fast), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256, num_labels=2)
    tf.BertForSequenceClassification(cfg).save_pretrained(out_dir)
    return out_dir


class _Predictor:
    """Adapter exposing ``scores(texts)`` like the baseline predictors (used for robustness/explanations)."""

    def __init__(self, model, tokenizer, device, max_length):
        self.model, self.tokenizer, self.device, self.max_length = model, tokenizer, device, max_length

    def scores(self, texts):
        p = predict_proba_texts(self.model, self.tokenizer, list(texts), self.device, self.max_length)
        return (p >= 0.5).astype(int), p, "probability"


def _evaluate_df(model, tok, df: pd.DataFrame, device: str, max_length: int):
    p = predict_proba_texts(model, tok, df["cleaned_message"].tolist(), device, max_length)
    return (p >= 0.5).astype(int), p


def train(args) -> dict:
    """Full fine-tuning run. Returns a results dict (also written to experiments/results/)."""
    torch, tf = _require_torch()
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    tiny = bool(getattr(args, "tiny_offline", False))
    if tiny:
        args.smoke = True
    device = select_device(args.device)
    tag = run_tag(args)
    if device == "cpu" and not args.smoke:
        log.warning("Running a FULL training on CPU - this will be very slow. Use a GPU runtime (Colab: Runtime > Change runtime type).")
    default_dir = (C.MODELS_DIR / "_smoke" / tag) if args.smoke else (C.MODELS_DIR / "transformers" / tag)
    out_dir = Path(args.output_dir) if args.output_dir else default_dir
    ensure_dir(out_dir), ensure_dir(C.RESULTS_DIR)
    integrity = {"verified": False, "skipped": True} if args.skip_checksum else verify_data_integrity()
    log.info("Model=%s | device=%s | tag=%s | data checksum verified=%s", args.model, device, tag, integrity.get("verified"))

    df = load_processed()
    fingerprint = split_fingerprint(df)
    train_df = get_split(df, "train", include_perturbed=args.augment, languages=args.train_languages)
    val_df = get_split(df, "validation", languages=args.train_languages)
    test_df = get_split(df, "test", languages=args.eval_languages)
    if args.smoke:
        train_df, args.epochs = train_df.sample(n=min(256, len(train_df)), random_state=args.seed), 1
        val_df, test_df = val_df.head(128), test_df.head(128)
    elif args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)
    if train_df["label_id"].nunique() < 2:
        raise ValueError("Training data must contain both classes")
    # leakage guard: the three evaluation/training sets must be disjoint at the group level
    g_tr, g_va, g_te = (set(x["group_id"]) for x in (train_df, val_df, test_df))
    if (g_tr & g_va) or (g_tr & g_te) or (g_va & g_te):
        raise RuntimeError("Group leakage between train/validation/test detected - aborting.")
    log.info("train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    name = str(build_tiny_offline_model(train_df["cleaned_message"].tolist(), out_dir / "_tiny_base", args.seed)) if tiny \
        else resolve_model_name(args.model)
    id2label, label2id = label_maps()
    tok = tf.AutoTokenizer.from_pretrained(name)
    model = tf.AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, id2label=id2label,
                                                                  label2id=label2id).to(device)
    params = count_parameters(model)
    collate = make_collate(tok, args.max_length)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TextDataset(train_df["cleaned_message"], train_df["label_id"]), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate, generator=gen)
    steps = max(1, len(loader) * args.epochs)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": DEFAULTS["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    sched = tf.get_linear_schedule_with_warmup(opt, int(DEFAULTS["warmup_ratio"] * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(train_df["label_id"]), dtype=torch.float, device=device))

    best, best_epoch, best_state, bad, history = -1.0, 0, None, 0, []
    stopped_early = False
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += float(loss.item())
        vp, _ = _evaluate_df(model, tok, val_df, device, args.max_length)
        vf1 = compute_metrics(val_df["label_id"].values, vp)["macro_f1"]
        history.append({"epoch": epoch + 1, "train_loss": running / max(len(loader), 1), "val_macro_f1": vf1})
        log.info("epoch %d | loss %.4f | val macro-F1 %.4f", epoch + 1, history[-1]["train_loss"], vf1)
        if vf1 > best + 1e-9:
            best, best_epoch, bad = vf1, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                stopped_early = True
                log.info("early stopping (patience=%d)", args.patience)
                break
    if device == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test evaluation (touched once): originals (+ their perturbed variants for robustness) ----
    do_rob = bool(args.robustness) and not args.train_languages
    test_p = df[(df["split"] == "test") & df["is_perturbed"].astype(bool)
                & df["original_message_id"].isin(set(test_df["message_id"]))].reset_index(drop=True) if do_rob else test_df.iloc[0:0]
    pred_df = pd.concat([test_df, test_p], ignore_index=True)
    all_ids, all_p = _evaluate_df(model, tok, pred_df, device, args.max_length)
    n_o = len(test_df)
    ids = all_ids[:n_o]
    subsets = evaluate_subsets(test_df, ids)
    overall = next(r for r in subsets if r["subset"] == "Overall")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    size_mb = sum(f.stat().st_size for f in out_dir.glob("**/*") if f.is_file() and "_tiny_base" not in f.parts) / (1024 * 1024)
    texts = test_df["cleaned_message"].tolist()
    latency = {device: benchmark_latency(lambda t: predict_proba_texts(model, tok, t, device, args.max_length), texts,
                                         n_single=100, n_batch=256)}
    if device == "cuda":
        model.to("cpu")
        latency["cpu"] = benchmark_latency(lambda t: predict_proba_texts(model, tok, t, "cpu", args.max_length), texts,
                                           n_single=50, n_batch=128)
        model.to(device)

    pd.DataFrame({"message_id": pred_df["message_id"], "is_perturbed": pred_df["is_perturbed"].astype(bool),
                  "y_true": pred_df["label_id"], "y_pred": all_ids, "p_scam": all_p}).to_csv(
        C.RESULTS_DIR / f"transformer_predictions_{tag}.csv", index=False)
    pd.DataFrame(subsets).to_csv(C.RESULTS_DIR / f"transformer_subsets_{tag}.csv", index=False)
    if do_rob and len(test_p):
        from .robustness import robustness_rows

        rows = robustness_rows(test_df, test_p.assign(pred=all_ids[n_o:]),
                               pd.Series(ids, index=test_df["message_id"].values), tag)
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / f"transformer_robustness_{tag}.csv", index=False)
    meta = {"tag": tag, "hf_model_id": name, "label2id": label2id, "max_length": args.max_length, "seed": args.seed}
    save_json(meta, out_dir / "mars_meta.json")
    results = {
        "model": PRETTY.get(args.model, name) if not tiny else "tiny random BERT (pipeline test, NOT pretrained)",
        "hf_model_id": name if not tiny else "local-random-init", "tag": tag, "smoke_test": bool(args.smoke),
        "tiny_offline_random_init": tiny, "pretrained": not tiny,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": environment_info(), "device": device, "parameters": params, "model_size_mb": size_mb,
        "train_seconds": train_seconds, "best_val_macro_f1": best, "best_epoch": best_epoch, "stopped_early": stopped_early,
        "history": history, "n_train": len(train_df), "n_val": len(val_df), "n_test_originals": len(test_df),
        "n_test_perturbed": int(len(test_p)), "test_overall": overall, "test_subsets": subsets,
        "latency_computational_benchmark": latency, "data_integrity": integrity, "split_fingerprint": fingerprint,
        "processed_csv_sha256": sha256_file(C.PROCESSED_CSV),
        "note": "Latency is a computational benchmark on this machine, not an edge-device measurement."}
    save_json(results, C.RESULTS_DIR / f"transformer_{tag}.json")
    log.info("Done. test macro-F1=%.4f (smoke=%s)", overall["macro_f1"], args.smoke)
    return results


def main(argv: Optional[list[str]] = None) -> None:
    train(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
