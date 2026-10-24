"""Render Table 3 (lexical/semantic) and Table 4 (clinical factuality) as LaTeX.

Values are means with 95% bootstrap CI half-widths as subscripts; best per column in bold.
"""

import argparse
from pathlib import Path

import pandas as pd

LEX = ["rouge1", "rouge2", "rougeL", "meteor", "bleu4_corpus", "bertscore"]
FACT = ["chexbert_f1", "radgraph_f1"]
ORDER = ["zero-shot", "R-5", "R-10", "B-5", "B-10"]


def cell(mean, hw, best):
    s = f"{mean:.4f}_{{\\pm{hw:.3f}}}"
    return f"\\mathbf{{{s}}}" if best else s


def table(df, metrics):
    piv_m = df.pivot_table(index=["system", "condition"], columns="metric", values="mean")
    piv_h = df.pivot_table(index=["system", "condition"], columns="metric", values="half_width")
    metrics = [m for m in metrics if m in piv_m.columns]
    best = {m: piv_m[m].drop(index="baseline", level=0, errors="ignore").max() for m in metrics}
    lines = []
    for (system, cond) in piv_m.index:
        vals = [f"${cell(piv_m.loc[(system, cond), m], piv_h.loc[(system, cond), m], piv_m.loc[(system, cond), m] == best[m])}$"
                for m in metrics]
        lines.append(" & ".join([system, cond] + vals) + r" \\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/results/scores.csv")
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()
    df = pd.read_csv(args.scores)
    df["condition"] = pd.Categorical(df["condition"], ORDER + sorted(set(df.condition) - set(ORDER)))
    df = df.sort_values(["system", "condition"])
    Path(args.out, "table3.tex").write_text(table(df, LEX))
    Path(args.out, "table4.tex").write_text(table(df[df.system != "baseline"], FACT))
    print("Wrote table3.tex and table4.tex")


if __name__ == "__main__":
    main()
