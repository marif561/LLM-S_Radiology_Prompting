"""Generate impressions for every (model, condition) and for the lead-k baselines.

Usage:
    python scripts/run_generation.py --model llama3-8b
    python scripts/run_generation.py --baselines-only
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from radprompt.baselines import lead_k
from radprompt.data import read_jsonl
from radprompt.prompting import build_messages, load_system_prompt, postprocess
from radprompt.selection import make_selector


def write(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", help="key under `models` in the config")
    ap.add_argument("--conditions", nargs="*", help="subset of condition names")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--out", default="outputs/generations")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    proc = Path(cfg["data"]["processed_dir"])
    pool, test = read_jsonl(proc / "pool.jsonl"), read_jsonl(proc / "test.jsonl")
    out = Path(args.out)

    for k in (1, 2):
        write([{"report_id": r.report_id, "prediction": lead_k(r, k), "reference": r.impression}
               for r in test], out / "baseline" / f"lead-{k}.jsonl")
    if args.baselines_only:
        return

    from radprompt.generate import GenConfig, HFGenerator

    m = cfg["models"][args.model]
    gen = HFGenerator(m["hf_id"], m.get("revision"), GenConfig(**cfg["generation"]))
    system_prompt = load_system_prompt()
    sel_cfg = cfg["selection"]
    selectors = {
        "random": make_selector("random", pool, seed=cfg["seed"], per_query=sel_cfg["random_per_query"]),
        "bm25": make_selector("bm25", pool, k1=sel_cfg["bm25_k1"], b=sel_cfg["bm25_b"]),
    }

    for cond in cfg["conditions"]:
        if args.conditions and cond["name"] not in args.conditions:
            continue
        rows, batch, metas = [], [], []
        bs = cfg["generation"]["batch_size"]
        for r in tqdm(test, desc=f"{args.model}/{cond['name']}"):
            demos = [] if cond["k"] == 0 else selectors[cond["selector"]].select(r, cond["k"])
            batch.append(build_messages(r, demos, system_prompt, gen.supports_system_role))
            metas.append((r, [d.report_id for d in demos]))
            if len(batch) == bs or r is test[-1]:
                for (q, demo_ids), text in zip(metas, gen.generate(batch)):
                    rows.append({"report_id": q.report_id, "prediction": postprocess(text),
                                 "raw_output": text, "reference": q.impression,
                                 "demo_ids": demo_ids})
                batch, metas = [], []
        write(rows, out / args.model / f"{cond['name']}.jsonl")


if __name__ == "__main__":
    main()
