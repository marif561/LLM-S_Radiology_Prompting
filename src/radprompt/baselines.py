"""Model-free extractive baselines."""

from .data import Report, split_sentences


def lead_k(report: Report, k: int) -> str:
    return " ".join(split_sentences(report.findings)[:k])
