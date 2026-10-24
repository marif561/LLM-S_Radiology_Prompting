"""Score all generations, compute bootstrap CIs, paired tests and Holm correction.

Writes outputs/results/{scores.csv, contrasts.csv, per_report/*.npz}.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radprompt.metrics import compute_all, corpus_bleu
from radprompt.stats import (bootstrap_ci, bootstrap_ci_corpus, bootstrap_indices,
                             holm_bonferroni, paired_bootstrap)

CONTRASTS = [  # (a, b): report b - a
    ("zero-shot", "R-5"), ("zero-shot", "R-10"), ("zero-shot", "B-5"), ("zero-shot", "B-10"),
    ("R-5", "R-10"), ("B-5", "B-10"),   # effect of k at fixed selection
    ("R-5", "B-5"), ("R-10", "B-10"),   # effect of selection at fixed k
]


def load(path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    rows.sort(key=lambda r: r["report_id"])
    return [r["report_id"] for r in rows], [r["prediction"] for r in rows], [r["reference"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gen-dir", default="outputs/generations")
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev = cfg["evaluation"]
    out = Path(args.out)
    (out / "per_report").mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.gen_dir).glob("*/*.jsonl"))
    idx, ids0, per = None, None, {}
    score_rows = []
    for f in files:
        system, cond = f.parent.name, f.stem
        ids, preds, refs = load(f)
        if ids0 is None:
            ids0, idx = ids, bootstrap_indices(len(ids), ev["bootstrap_resamples"], cfg["seed"])
        assert ids == ids0, f"{f} covers a different set of test reports"
        scores = compute_all(preds, refs, ev["metrics"])
        per[(system, cond)] = scores
        np.savez(out / "per_report" / f"{system}__{cond}.npz", **scores)
        for metric, s in scores.items():
            score_rows.append({"system": system, "condition": cond, "metric": metric,
                               **bootstrap_ci(s, idx, ev["alpha"])})
        score_rows.append({"system": system, "condition": cond, "metric": "bleu4_corpus",
                           **bootstrap_ci_corpus(corpus_bleu, preds, refs, idx[:1000], ev["alpha"])})
        out_len = np.array([len(p.split()) for p in preds], dtype=float)
        score_rows.append({"system": system, "condition": cond, "metric": "output_len",
                           **bootstrap_ci(out_len, idx, ev["alpha"])})

    pd.DataFrame(score_rows).to_csv(out / "scores.csv", index=False)

    contrast_rows = []
    systems = sorted({s for s, _ in per if s != "baseline"})
    for system, (a, b), metric in product(systems, CONTRASTS, ev["metrics"]):
        if (system, a) in per and (system, b) in per and metric in per[(system, a)]:
            res = paired_bootstrap(per[(system, a)][metric], per[(system, b)][metric], idx, ev["alpha"])
            contrast_rows.append({"system": system, "a": a, "b": b, "metric": metric, **res})
    if contrast_rows:
        for row, h in zip(contrast_rows, holm_bonferroni([r["p"] for r in contrast_rows], ev["alpha"])):
            row.update(p_holm=h["p_holm"], reject=h["reject"])
    pd.DataFrame(contrast_rows).to_csv(out / "contrasts.csv", index=False)
    print(f"Scored {len(files)} runs; {len(contrast_rows)} pre-specified contrasts tested.")


if __name__ == "__main__":
    main()
