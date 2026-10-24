#!/usr/bin/env bash
# End-to-end reproduction: data -> baselines -> 3 models x 5 conditions -> metrics -> tables.
set -euo pipefail

CONFIG=${CONFIG:-configs/default.yaml}

python scripts/prepare_data.py --config "$CONFIG"
python scripts/run_generation.py --config "$CONFIG" --baselines-only
for MODEL in mistral-7b llama3-8b qwen-7b; do
  python scripts/run_generation.py --config "$CONFIG" --model "$MODEL"
done
python scripts/evaluate.py --config "$CONFIG"
python scripts/make_tables.py
