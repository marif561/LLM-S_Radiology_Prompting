"""Load, clean and split the Indiana University Chest X-ray reports (Open-i).

The Open-i release (``NLMCXR_reports.tgz``) unpacks to ``ecgen-radiology/*.xml``.
Each XML file is one report with ``<AbstractText Label="FINDINGS">`` and
``<AbstractText Label="IMPRESSION">`` sections.
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DEID_TOKEN = re.compile(r"\bX{2,}\b")
MULTISPACE = re.compile(r"\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Report:
    report_id: str
    patient_id: str
    findings: str
    impression: str


def clean_text(text: str | None, lowercase: bool = False) -> str:
    """Remove de-identification tokens and normalise whitespace.

    Clinical terms are never altered.
    """
    if not text:
        return ""
    text = DEID_TOKEN.sub(" ", text)
    text = text.replace(" ", " ")
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"([.,;:]){2,}", r"\1", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text.lower() if lowercase else text


def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text.strip()) if s]


def parse_report(xml_path: Path, lowercase: bool = False) -> Report | None:
    root = ET.parse(xml_path).getroot()
    sections = {
        (el.get("Label") or "").upper(): (el.text or "")
        for el in root.iter("AbstractText")
    }
    findings = clean_text(sections.get("FINDINGS"), lowercase)
    impression = clean_text(sections.get("IMPRESSION"), lowercase)
    if not findings or not impression:
        return None
    report_id = xml_path.stem
    uid = root.find(".//uId")
    # In IU X-ray each report corresponds to one study/patient (CXR<n>).
    patient_id = uid.get("id") if uid is not None and uid.get("id") else report_id
    return Report(report_id, patient_id, findings, impression)


def load_iu_xray(report_dir: str | Path, lowercase: bool = False) -> list[Report]:
    paths = sorted(Path(report_dir).glob("*.xml"))
    if not paths:
        raise FileNotFoundError(f"No XML reports found in {report_dir}")
    reports = [r for p in paths if (r := parse_report(p, lowercase)) is not None]
    return reports


def patient_level_split(
    reports: list[Report], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[Report], list[Report]]:
    """Split by patient so no patient appears in both pool and test set."""
    patients = sorted({r.patient_id for r in reports})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_test = round(len(patients) * test_fraction)
    test_patients = set(patients[:n_test])
    pool = [r for r in reports if r.patient_id not in test_patients]
    test = [r for r in reports if r.patient_id in test_patients]
    return pool, test


def write_jsonl(reports: Iterable[Report], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Report]:
    with Path(path).open(encoding="utf-8") as f:
        return [Report(**json.loads(line)) for line in f if line.strip()]


def describe(reports: list[Report]) -> dict:
    """Descriptive statistics for Table 1 (whitespace tokens, mean and SD)."""
    import statistics as st

    def ms(xs):
        return {"mean": round(st.mean(xs), 2), "sd": round(st.pstdev(xs), 2)}

    normal = re.compile(
        r"\b(no acute|normal chest|no active|negative|unremarkable)\b", re.I
    )
    return {
        "n_reports": len(reports),
        "findings_len": ms([len(r.findings.split()) for r in reports]),
        "impression_len": ms([len(r.impression.split()) for r in reports]),
        "normal_impression_pct": round(
            100 * sum(bool(normal.search(r.impression)) for r in reports) / len(reports), 2
        ),
    }
