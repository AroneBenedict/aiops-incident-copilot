"""Runbook retrieval: the "R" of a retrieval-augmented incident report.

Runbooks are split into sections, indexed, and searched with a free-text query built from
log lines and metric symptoms. Two retrievers share one interface: TF-IDF cosine similarity
and BM25. Both are lexical and run offline; a dense embedding model can be added behind the
same ``search`` method.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN = re.compile(r"[a-z][a-z0-9_]+")
_STOP = frozenset((
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are", "was", "for", "with", "that", "this",
    "it", "its", "as", "at", "by", "be", "from", "not", "has", "have", "while", "when", "than", "then", "them",
    "their", "there", "which", "who", "will", "can", "may", "so", "if", "into",
))
_STEP = re.compile(r"^\s*\d+\.\s+(.*)$")
_FAULT = re.compile(r"<!--\s*fault:\s*([a-z_]+)\s*-->")
APPROVAL_MARK = "(requires approval)"


@dataclass(frozen=True)
class Chunk:
    runbook: str   # file stem, e.g. "db-connection-exhaustion"
    title: str
    section: str
    text: str


@dataclass(frozen=True)
class Hit:
    runbook: str
    title: str
    section: str
    score: float


@dataclass(frozen=True)
class Step:
    text: str
    requires_approval: bool


@dataclass
class Runbook:
    name: str
    title: str
    fault_type: str | None
    sections: dict[str, str]

    def remediation(self) -> list[Step]:
        steps = []
        for line in self.sections.get("Remediation", "").splitlines():
            match = _STEP.match(line)
            if match:
                text = match.group(1).strip()
                needs = APPROVAL_MARK in text
                steps.append(Step(text.replace(APPROVAL_MARK, "").strip(), needs))
        return steps


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def parse_runbook(path: Path) -> Runbook:
    title, sections, current = path.stem, {}, None
    text = path.read_text(encoding="utf-8")
    fault = _FAULT.search(text)
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            current = line[3:].strip()
            sections[current] = ""
        elif current is not None and not line.startswith("<!--"):
            sections[current] += line + "\n"
    fault_type = fault.group(1) if fault and fault.group(1) != "none" else None
    return Runbook(path.stem, title, fault_type, {k: v.strip() for k, v in sections.items()})


def load_runbooks(directory: Path | str) -> dict[str, Runbook]:
    paths = sorted(Path(directory).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no runbooks found in {directory}")
    return {path.stem: parse_runbook(path) for path in paths}


def chunk_runbooks(runbooks: dict[str, Runbook], sections: tuple[str, ...] = ("Symptoms", "Diagnosis")
                   ) -> list[Chunk]:
    """Index the descriptive sections. Remediation text is returned, not searched."""
    chunks = []
    for book in runbooks.values():
        for section in sections:
            if book.sections.get(section):
                chunks.append(Chunk(book.name, book.title, section, f"{book.title}. {book.sections[section]}"))
    return chunks


def _best_per_runbook(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    best: dict[str, Hit] = {}
    for chunk, score in zip(chunks, scores, strict=True):
        if chunk.runbook not in best or score > best[chunk.runbook].score:
            best[chunk.runbook] = Hit(chunk.runbook, chunk.title, chunk.section, round(float(score), 4))
    return sorted(best.values(), key=lambda hit: (-hit.score, hit.runbook))[:k]


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return _best_per_runbook(self.chunks, scores, k)


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = np.array([sum(doc.values()) for doc in self.docs], dtype=float)
        self.avg_length = float(self.lengths.mean())
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in frequency.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = Counter(tokenize(query))
        scores = np.zeros(len(self.docs))
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return _best_per_runbook(self.chunks, scores, k)


def build_query(kind: str, deviations: dict[str, float], log_lines: list[str], max_lines: int = 6) -> str:
    """Compose a retrieval query from the evidence the agent has gathered so far."""
    words = {"latency_ms": "latency", "error_rate": "error rate", "cpu_pct": "cpu", "throughput_rps": "throughput"}
    symptoms = []
    for metric, value in deviations.items():
        if value >= 3:
            symptoms.append(f"high {words[metric]}")
        elif value <= -3:
            symptoms.append(f"{words[metric]} drops")
        else:
            symptoms.append(f"{words[metric]} normal")
    seen, distinct = set(), []
    for line in log_lines:
        key = re.sub(r"\d+", "#", line)
        if key not in seen:
            seen.add(key)
            distinct.append(line)
    return f"{kind}. {', '.join(symptoms)}. " + " ".join(distinct[:max_lines])
