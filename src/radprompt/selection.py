"""Demonstration selection: random (length-stratified) and BM25 retrieval.

Demonstrations are always drawn from the demonstration pool, never the test set.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from .data import Report, split_sentences

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class RandomSelector:
    """Sample k demonstrations stratified by number of findings sentences.

    ``per_query=False`` (default) uses one fixed demonstration set for every
    test report; ``per_query=True`` draws a fresh, seeded sample per report.
    """

    def __init__(self, pool: list[Report], seed: int = 42, per_query: bool = False):
        self.pool = pool
        self.seed = seed
        self.per_query = per_query
        self._fixed: dict[int, list[Report]] = {}
        self.strata: dict[str, list[Report]] = defaultdict(list)
        for r in pool:
            n = len(split_sentences(r.findings))
            self.strata["short" if n <= 3 else "medium" if n <= 6 else "long"].append(r)

    def _sample(self, k: int, rng: random.Random) -> list[Report]:
        names = [s for s in ("short", "medium", "long") if self.strata[s]]
        buckets = {s: rng.sample(self.strata[s], len(self.strata[s])) for s in names}
        out: list[Report] = []
        i = 0
        while len(out) < k and any(buckets.values()):
            s = names[i % len(names)]
            if buckets[s]:
                out.append(buckets[s].pop())
            i += 1
        return out

    def select(self, query: Report, k: int) -> list[Report]:
        if self.per_query:
            return self._sample(k, random.Random(f"{self.seed}-{query.report_id}-{k}"))
        if k not in self._fixed:
            self._fixed[k] = self._sample(k, random.Random(self.seed + k))
        return self._fixed[k]


class BM25Selector:
    """Retrieve the k pool reports whose findings are most similar (BM25 Okapi)."""

    def __init__(self, pool: list[Report], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.pool = pool
        self.bm25 = BM25Okapi([tokenize(r.findings) for r in pool], k1=k1, b=b)

    def select(self, query: Report, k: int) -> list[Report]:
        scores = self.bm25.get_scores(tokenize(query.findings))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        picked = [
            self.pool[i] for i in order if self.pool[i].patient_id != query.patient_id
        ][:k]
        # Most similar demonstration placed last, closest to the query.
        return list(reversed(picked))


def make_selector(name: str, pool: list[Report], **kw):
    if name == "random":
        return RandomSelector(pool, seed=kw.get("seed", 42), per_query=kw.get("per_query", False))
    if name == "bm25":
        return BM25Selector(pool, k1=kw.get("k1", 1.5), b=kw.get("b", 0.75))
    raise ValueError(f"Unknown selector: {name}")
