"""Bootstrap confidence intervals, paired bootstrap tests and Holm correction."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_indices(n: int, n_resamples: int = 10_000, seed: int = 42) -> np.ndarray:
    """Shared resample indices so paired comparisons use identical resamples."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_ci(scores: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> dict:
    means = scores[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(scores.mean()), "lo": float(lo), "hi": float(hi),
            "half_width": float((hi - lo) / 2)}


def bootstrap_ci_corpus(fn: Callable[[list, list], float], preds: list[str], refs: list[str],
                        idx: np.ndarray, alpha: float = 0.05) -> dict:
    """CI for a corpus-level metric, recomputed on every resample."""
    p, r = np.array(preds, dtype=object), np.array(refs, dtype=object)
    vals = np.array([fn(list(p[i]), list(r[i])) for i in idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(fn(preds, refs)), "lo": float(lo), "hi": float(hi),
            "half_width": float((hi - lo) / 2)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> dict:
    """Mean paired difference (b - a), its CI and a two-sided bootstrap p-value."""
    diff = b - a
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    centred = boot - diff.mean()
    p = float((np.abs(centred) >= abs(diff.mean())).mean())
    return {"diff": float(diff.mean()), "lo": float(lo), "hi": float(hi), "p": max(p, 1 / len(idx))}


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[dict]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return [{"p": pvals[i], "p_holm": float(adj[i]), "reject": bool(adj[i] < alpha)}
            for i in range(m)]
