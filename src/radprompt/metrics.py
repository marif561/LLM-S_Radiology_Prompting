"""Per-report metric computation.

All metrics are computed per test report so that the bootstrap in ``stats.py``
can resample reports. BLEU-4 is additionally reported at corpus level via
sacreBLEU (recomputed on each bootstrap resample).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def rouge_scores(preds: list[str], refs: list[str], use_stemmer: bool = True) -> dict[str, np.ndarray]:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    rows = [sc.score(r, p) for p, r in zip(preds, refs)]
    return {k: np.array([row[k].fmeasure for row in rows]) for k in ("rouge1", "rouge2", "rougeL")}


def meteor_scores(preds: list[str], refs: list[str]) -> np.ndarray:
    import nltk
    from nltk.translate.meteor_score import meteor_score

    for res in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    return np.array([meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)])


def sentence_bleu_scores(preds: list[str], refs: list[str]) -> np.ndarray:
    import sacrebleu

    return np.array(
        [sacrebleu.sentence_bleu(p, [r]).score / 100.0 for p, r in zip(preds, refs)]
    )


def corpus_bleu(preds: list[str], refs: list[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_bleu(preds, [refs]).score / 100.0


def bertscore_f1(preds: list[str], refs: list[str], model_type: str = "roberta-large",
                 rescale: bool = False) -> np.ndarray:
    from bert_score import score

    _, _, f1 = score(preds, refs, model_type=model_type, lang="en",
                     rescale_with_baseline=rescale, verbose=False)
    return f1.numpy()


def radgraph_f1(preds: list[str], refs: list[str]) -> np.ndarray:
    """RadGraph F1 (partial reward) via the ``radgraph`` package."""
    from radgraph import F1RadGraph

    scorer = F1RadGraph(reward_level="partial")
    _, per_report, _, _ = scorer(hyps=preds, refs=refs)
    return np.array(per_report, dtype=float)


def chexbert_f1(preds: list[str], refs: list[str]) -> np.ndarray:
    """Per-report micro-F1 over the 14 CheXpert labels (positive = present).

    Uses the ``f1chexbert`` package. Per-report F1 is 1.0 when both texts
    have no positive labels.
    """
    from f1chexbert import F1CheXbert

    scorer = F1CheXbert()
    y_pred = np.array([scorer.get_label(p) for p in preds])
    y_true = np.array([scorer.get_label(r) for r in refs])
    tp = (y_pred & y_true).sum(1)
    fp = (y_pred & ~y_true).sum(1)
    fn = (~y_pred & y_true).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom == 0, 1.0, 2 * tp / np.maximum(denom, 1))


METRICS: dict[str, Callable] = {
    "meteor": meteor_scores,
    "bleu4_sent": sentence_bleu_scores,
    "bertscore": bertscore_f1,
    "chexbert_f1": chexbert_f1,
    "radgraph_f1": radgraph_f1,
}


def compute_all(preds: list[str], refs: list[str], which: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if any(m.startswith("rouge") for m in which):
        out.update({k: v for k, v in rouge_scores(preds, refs).items() if k in which})
    for m in which:
        if m in METRICS:
            out[m] = METRICS[m](preds, refs)
    return out
