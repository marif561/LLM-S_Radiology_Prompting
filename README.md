<div align="center">

# 🩻 LLM-S Radiology Prompting

**Evaluating open-weight LLMs for radiology impression generation with zero-shot and few-shot prompting**

[![tests](https://github.com/marif561/LLM-S_Radiology_Prompting/actions/workflows/ci.yml/badge.svg)](https://github.com/marif561/LLM-S_Radiology_Prompting/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Models](https://img.shields.io/badge/models-Mistral%20%7C%20LLaMA--3%20%7C%20Qwen-purple)
![Fine-tuning](https://img.shields.io/badge/fine--tuning-none-lightgrey)

</div>

---

The **Impression** is the part of a radiology report most clinicians actually read. This repository contains the full, reproducible pipeline for measuring how well small (7–8B) instruction-tuned open-weight LLMs write impressions from radiologist findings **without any parameter updates**, and for separating *how many* demonstrations help from *which* demonstrations help.

```
Findings ──► prompt (0-shot | k ∈ {5,10} × {random, BM25}) ──► LLM (4-bit NF4, greedy) ──► Impression
                                                                                             │
          ROUGE · METEOR · BLEU-4 · BERTScore · CheXbert F1 · RadGraph F1 ◄── reference ─────┘
                         + bootstrap 95% CIs, paired tests, Holm correction
```

## ✨ Highlights

- **Controlled 2 × 2 design** — number of shots (5 vs 10) crossed with selection strategy (random vs BM25), plus zero-shot.
- **Leak-free** — patient-level split; demonstrations come only from the pool, never the test set, and BM25 never returns the query patient.
- **Clinical factuality, not just n-grams** — CheXbert F1 and RadGraph F1 catch flipped negations that ROUGE misses.
- **Proper statistics** — per-report bootstrap CIs (10 000 resamples), paired bootstrap contrasts on shared resamples, Holm–Bonferroni.
- **On-premises friendly** — every model fits on one GPU at 4-bit; no patient data leaves the machine.

## 📊 Preliminary results (IU X-ray, test set)

| Model | Prompt | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BLEU-4 | BERTScore |
|---|---|---|---|---|---|---|---|
| Mistral-7B-Instruct | 0-shot | 0.3812 | 0.1934 | 0.3421 | 0.3107 | 0.1523 | 0.8312 |
| | R-5 | 0.4217 | 0.2301 | 0.3812 | 0.3492 | 0.1901 | 0.8543 |
| | B-10 | 0.4489 | 0.2512 | 0.4078 | 0.3712 | 0.2089 | 0.8634 |
| **LLaMA-3-8B-Instruct** | 0-shot | 0.4023 | 0.2147 | 0.3689 | 0.3354 | 0.1712 | 0.8491 |
| | R-5 | 0.4512 | 0.2589 | 0.4102 | 0.3741 | 0.2134 | 0.8712 |
| | **B-10** | **0.4803** | **0.2841** | **0.4391** | **0.3989** | **0.2378** | **0.8821** |
| Qwen (7B) | 0-shot | 0.3601 | 0.1823 | 0.3198 | 0.2987 | 0.1389 | 0.8201 |
| | R-5 | 0.4089 | 0.2201 | 0.3712 | 0.3321 | 0.1812 | 0.8489 |
| | B-10 | 0.4312 | 0.2489 | 0.3989 | 0.3589 | 0.2012 | 0.8612 |

> ⚠️ These numbers come from the submitted version of the paper. The full 2 × 2 grid (R-10, B-5), lead-k baselines, CIs and clinical factuality scores are produced by `scripts/run_all.sh` and will replace this table.

## 🗂️ Repository layout

```
├── configs/default.yaml        # every experimental setting in one place
├── prompts/system_prompt.txt   # the exact instruction used in all conditions
├── src/radprompt/
│   ├── data.py                 # Open-i XML parsing, cleaning, patient-level split, Table 1 stats
│   ├── selection.py            # length-stratified random & BM25 demonstration selection
│   ├── prompting.py            # chat-template messages (handles models without a system role)
│   ├── generate.py             # HF Transformers, 4-bit NF4, greedy decoding
│   ├── baselines.py            # lead-1 / lead-2
│   ├── metrics.py              # ROUGE, METEOR, BLEU-4, BERTScore, CheXbert F1, RadGraph F1
│   └── stats.py                # bootstrap CIs, paired bootstrap, Holm–Bonferroni
├── scripts/
│   ├── download_data.sh        # fetch IU X-ray reports from Open-i
│   ├── prepare_data.py         # → data/processed/{pool,test}.jsonl + table1_stats.json
│   ├── run_generation.py       # one model × all conditions (+ baselines)
│   ├── evaluate.py             # → outputs/results/{scores,contrasts}.csv
│   ├── make_tables.py          # → LaTeX Tables 3 & 4
│   └── run_all.sh              # end-to-end reproduction
└── tests/                      # CPU-only unit tests (run in CI)
```

## 🚀 Quick start

```bash
git clone https://github.com/marif561/LLM-S_Radiology_Prompting.git
cd LLM-S_Radiology_Prompting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

bash scripts/download_data.sh        # IU X-ray report XMLs (text only)
huggingface-cli login                # LLaMA-3 is a gated model
bash scripts/run_all.sh              # data → generations → metrics → tables
```

Run a single model or condition:

```bash
python scripts/run_generation.py --model llama3-8b --conditions zero-shot B-10
python scripts/evaluate.py && python scripts/make_tables.py
```

## 🧪 Experimental setup

| | |
|---|---|
| **Dataset** | Indiana University Chest X-ray (Open-i), 3,955 reports; findings → impression pairs, empty sections removed |
| **Split** | Patient-level, 80 % demonstration pool / 20 % held-out test, seed 42 |
| **Models** | `mistralai/Mistral-7B-Instruct-v0.x`, `meta-llama/Meta-Llama-3-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` |
| **Inference** | 4-bit NF4 (bitsandbytes), Transformers 4.39.3, greedy, ≤ 256 new tokens, official chat templates |
| **Conditions** | zero-shot, R-5, R-10, B-5, B-10 |
| **Random selection** | stratified by findings length (short / medium / long); one fixed set per *k* (configurable per-query) |
| **BM25 selection** | Okapi BM25 (k1 = 1.5, b = 0.75) over findings; most similar example placed last |
| **Metrics** | ROUGE-1/2/L F (rouge-score, stemming), METEOR (NLTK), BLEU-4 (sacreBLEU, corpus), BERTScore F1 (roberta-large), CheXbert F1, RadGraph F1 |
| **Statistics** | 10 000-sample bootstrap over test reports; paired bootstrap; Holm–Bonferroni over pre-specified contrasts |

All settings live in [`configs/default.yaml`](configs/default.yaml). Pin model `revision` hashes there for an exactly reproducible run.

## ✅ Tests

```bash
pip install pytest && pytest -q
```

Unit tests cover XML parsing and cleaning, split disjointness, demonstration selection, prompt construction for models with and without a system role, baselines, and the statistics module.

## ⚕️ Intended use

This is a research codebase. Generated impressions are **drafts** and must be reviewed by a qualified radiologist; automatic metrics are not a substitute for clinical validation.

## 📚 Citation

```bibtex
@misc{llms_radiology_prompting_2026,
  title  = {Evaluating Open-Weight Large Language Models for Radiology Report Impression
            Generation with Zero-Shot and Few-Shot Prompting},
  author = {Arif, Muhammad and others},
  year   = {2026},
  note   = {Under review}
}
```

## 🙏 Acknowledgements

IU Chest X-ray collection: Demner-Fushman et al., *JAMIA* 23(2), 2016, distributed via the U.S. National Library of Medicine Open-i service.

## 📄 License

MIT — see [LICENSE](LICENSE).
