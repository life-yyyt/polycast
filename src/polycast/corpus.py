"""Corpus loading helpers shared by the API, the CLI and the eval harness."""

from __future__ import annotations

import json
from pathlib import Path

from .retrieval import TfidfRetriever
from .schemas import Document

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus.json"


def load_corpus(path: str | Path | None = None) -> list[Document]:
    corpus_path = Path(path) if path else DEFAULT_CORPUS
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [Document(**item) for item in raw]


def build_retriever(
    path: str | Path | None = None, half_life_days: float = 45.0
) -> TfidfRetriever:
    return TfidfRetriever(load_corpus(path), half_life_days=half_life_days)
