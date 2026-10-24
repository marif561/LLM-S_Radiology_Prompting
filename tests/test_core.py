"""CPU-only tests for the parts of the pipeline that need no GPU or model download."""

import numpy as np
import pytest

from radprompt.baselines import lead_k
from radprompt.data import Report, clean_text, parse_report, patient_level_split
from radprompt.prompting import build_messages, postprocess
from radprompt.selection import BM25Selector, RandomSelector
from radprompt.stats import bootstrap_ci, bootstrap_indices, holm_bonferroni, paired_bootstrap

XML = """<?xml version="1.0"?>
<eCitation><uId id="CXR1"/><MedlineCitation><Article><Abstract>
<AbstractText Label="COMPARISON">None.</AbstractText>
<AbstractText Label="FINDINGS">The heart is normal in size. XXXX  The lungs are clear .</AbstractText>
<AbstractText Label="IMPRESSION">Normal chest x-XXXX.</AbstractText>
</Abstract></Article></MedlineCitation></eCitation>"""


def make_reports(n=40):
    texts = ["pleural effusion on the left", "cardiomegaly with edema", "clear lungs no effusion",
             "right lower lobe consolidation pneumonia"]
    return [Report(f"r{i}", f"p{i}", f"{texts[i % 4]}. sentence two. " * (1 + i % 4), f"imp {i}")
            for i in range(n)]


def test_clean_text():
    assert clean_text("The heart XXXX is  normal .") == "The heart is normal."
    assert clean_text(None) == ""


def test_parse_report(tmp_path):
    p = tmp_path / "1.xml"
    p.write_text(XML)
    r = parse_report(p)
    assert r.patient_id == "CXR1"
    assert r.findings == "The heart is normal in size. The lungs are clear."
    assert "XXXX" not in r.impression


def test_patient_split_disjoint():
    reports = make_reports() + [Report("dup", "p0", "x.", "y")]
    pool, test = patient_level_split(reports, 0.25, seed=1)
    assert not {r.patient_id for r in pool} & {r.patient_id for r in test}
    assert len(pool) + len(test) == len(reports)


def test_random_selector_fixed_and_stratified():
    pool = make_reports()
    sel = RandomSelector(pool, seed=0)
    q = Report("q", "pq", "x.", "y")
    a, b = sel.select(q, 5), sel.select(Report("q2", "pq2", "z.", "w"), 5)
    assert a == b and len(a) == 5
    assert len({len(d.findings.split(". ")) for d in a}) > 1  # covers several lengths


def test_bm25_selector_relevance():
    pool = make_reports()
    sel = BM25Selector(pool)
    q = Report("q", "pq", "cardiomegaly with pulmonary edema", "y")
    demos = sel.select(q, 3)
    assert len(demos) == 3
    assert "cardiomegaly" in demos[-1].findings  # most similar is last


def test_bm25_excludes_same_patient():
    pool = make_reports()
    sel = BM25Selector(pool)
    demos = sel.select(Report("q", "p1", pool[1].findings, "y"), 5)
    assert all(d.patient_id != "p1" for d in demos)


def test_build_messages():
    q, d = Report("q", "p", "f q", "i q"), Report("d", "p2", "f d", "i d")
    m = build_messages(q, [d], "SYS")
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    m2 = build_messages(q, [d], "SYS", supports_system_role=False)
    assert m2[0]["role"] == "user" and m2[0]["content"].startswith("SYS")


def test_postprocess_and_lead():
    assert postprocess("IMPRESSION: no acute disease.\nFINDINGS: x") == "no acute disease."
    assert lead_k(Report("a", "b", "One. Two. Three.", "x"), 2) == "One. Two."


def test_stats():
    rng = np.random.default_rng(0)
    a = rng.normal(0.4, 0.05, 200)
    b = a + 0.05
    idx = bootstrap_indices(200, 2000, seed=0)
    ci = bootstrap_ci(a, idx)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    res = paired_bootstrap(a, b, idx)
    assert res["diff"] == pytest.approx(0.05) and res["p"] < 0.01
    h = holm_bonferroni([0.01, 0.04, 0.03])
    assert [x["p_holm"] for x in h] == pytest.approx([0.03, 0.06, 0.06])
