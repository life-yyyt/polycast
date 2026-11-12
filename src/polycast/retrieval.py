"""Retrieval over an in-memory corpus with recency-aware ranking.

Plain TF-IDF rather than an embedding index: for the corpus sizes a forecasting
panel actually reads (hundreds of documents, not millions) the difference in
recall is small, and having zero infrastructure between `git clone` and a
working retrieval stack is worth more. Swap `TfidfRetriever` for a vector store
by implementing the same `search` signature.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .schemas import Document, Evidence, utcnow

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "will", "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class TfidfRetriever:
    def __init__(self, documents: list[Document], half_life_days: float = 45.0) -> None:
        if not documents:
            raise ValueError("retriever needs at least one document")
        self.half_life_days = half_life_days
        self.documents = documents
        self._tokens = {d.doc_id: tokenize(d.text) for d in documents}
        self._tf = {d.doc_id: Counter(self._tokens[d.doc_id]) for d in documents}
        df: Counter[str] = Counter()
        for tokens in self._tokens.values():
            df.update(set(tokens))
        n = len(documents)
        self._idf = {term: math.log((1 + n) / (1 + freq)) + 1 for term, freq in df.items()}
        self._norms = {d.doc_id: self._norm(self._tf[d.doc_id]) for d in documents}

    def _norm(self, tf: Counter[str]) -> float:
        return math.sqrt(sum((count * self._idf.get(t, 0.0)) ** 2 for t, count in tf.items())) or 1.0

    def _similarity(self, query_tf: Counter[str], query_norm: float, doc_id: str) -> float:
        dot = 0.0
        for term, qcount in query_tf.items():
            if term not in self._idf:
                continue
            weight = self._idf[term] ** 2
            dot += qcount * self._tf[doc_id].get(term, 0) * weight
        return dot / (query_norm * self._norms[doc_id])

    def search(self, query: str, k: int = 5, now=None) -> list[Evidence]:
        now = now or utcnow()
        query_tf = Counter(tokenize(query))
        if not query_tf:
            return []
        query_norm = math.sqrt(
            sum((c * self._idf.get(t, 0.0)) ** 2 for t, c in query_tf.items())
        ) or 1.0

        scored: list[Evidence] = []
        for doc in self.documents:
            similarity = self._similarity(query_tf, query_norm, doc.doc_id)
            if similarity <= 0:
                continue
            age_days = max((now - doc.published_at).total_seconds() / 86400, 0.0)
            freshness = 0.5 ** (age_days / self.half_life_days)
            # Keep stale documents reachable but discounted; a total cutoff would
            # hide slow-moving structural evidence behind breaking news.
            score = similarity * (0.4 + 0.6 * freshness)
            scored.append(Evidence(**doc.model_dump(), score=score, freshness=freshness))

        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:k]
