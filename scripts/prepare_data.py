"""Parse IU X-ray XML reports, clean them and write a patient-level split.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from radprompt.data import describe, load_iu_xray, patient_level_split, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]

    reports = load_iu_xray(d["report_dir"], lowercase=d.get("lowercase", False))
    pool, test = patient_level_split(reports, d["test_fraction"], cfg["seed"])
    assert not {r.patient_id for r in pool} & {r.patient_id for r in test}

    out = Path(d["processed_dir"])
    write_jsonl(pool, out / "pool.jsonl")
    write_jsonl(test, out / "test.jsonl")
    stats = {"pool": describe(pool), "test": describe(test), "total": describe(reports)}
    (out / "table1_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
