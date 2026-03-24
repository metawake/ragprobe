"""Core scoring engine: inverted index, specificity, IDF, difficulty tiers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from ragprobe.models import (
    DifficultyTier,
    DomainReport,
    QueryDifficulty,
)
from ragprobe.profiles import BUILTIN_PROFILES, closest_profile

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "just", "let", "me", "might", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "re", "s", "same", "shan't", "she", "should", "shouldn't", "so", "some",
    "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won't", "would", "wouldn't",
    "you", "your", "yours", "yourself", "yourselves",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
RARE_THRESHOLD = 5


def tokenize(text: str, stopwords: Optional[Set[str]] = None) -> List[str]:
    """Lowercase tokenize, removing stopwords."""
    sw = stopwords if stopwords is not None else STOPWORDS
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in sw and len(t) > 1]


def build_inverted_index(passages: List[str],
                         stopwords: Optional[Set[str]] = None,
                         ) -> Dict[str, Set[int]]:
    """Map each term to the set of passage indices it appears in."""
    index: Dict[str, Set[int]] = {}
    for i, passage in enumerate(passages):
        for token in tokenize(passage, stopwords):
            index.setdefault(token, set()).add(i)
    return index


def _idf(term: str, index: Dict[str, Set[int]], n_docs: int) -> float:
    df = len(index.get(term, set()))
    if df == 0:
        return 0.0
    return math.log((n_docs + 1) / (df + 1)) + 1


def score_query(query: str,
                index: Dict[str, Set[int]],
                n_docs: int,
                stopwords: Optional[Set[str]] = None,
                ) -> QueryDifficulty:
    """Score a single query against a corpus inverted index."""
    tokens = tokenize(query, stopwords)
    if not tokens:
        return QueryDifficulty(
            text=query, specificity=0.0,
            difficulty=DifficultyTier.HARD, ambiguous_terms=[], idf_avg=0.0, idf_max=0.0,
        )

    specific_count = 0
    ambiguous: List[str] = []
    idfs: List[float] = []

    for token in tokens:
        df = len(index.get(token, set()))
        idfs.append(_idf(token, index, n_docs))
        if df < RARE_THRESHOLD:
            specific_count += 1
        elif df >= RARE_THRESHOLD:
            ambiguous.append(token)

    specificity = specific_count / len(tokens)
    idf_avg = sum(idfs) / len(idfs) if idfs else 0.0
    idf_max = max(idfs) if idfs else 0.0

    return QueryDifficulty(
        text=query,
        specificity=round(specificity, 4),
        difficulty=DifficultyTier.from_specificity(specificity),
        ambiguous_terms=ambiguous,
        idf_avg=round(idf_avg, 4),
        idf_max=round(idf_max, 4),
    )


def _top_ambiguous_terms(index: Dict[str, Set[int]],
                         limit: int = 20) -> List[Tuple[str, int]]:
    """Return the most frequent terms by document frequency."""
    counts = [(term, len(doc_set)) for term, doc_set in index.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts[:limit]


def _recommendations(specificity: float,
                     difficulty: DifficultyTier,
                     closest_ref: str,
                     recall_range: Optional[Tuple[float, float]] = None,
                     ) -> List[str]:
    recs: List[str] = []
    if difficulty == DifficultyTier.HARD:
        recs.append(
            "Domain is HARD — build domain-specific needle annotations "
            "before trusting any benchmark results."
        )
        recs.append(
            "Generic benchmarks (HotpotQA, FinanceBench) will NOT "
            "transfer to this domain."
        )
    elif difficulty == DifficultyTier.MEDIUM:
        recs.append(
            "Domain is MEDIUM — build 10-20 domain-specific test queries "
            "and validate retrieval before deploying."
        )
    else:
        recs.append(
            "Domain is EASY — standard benchmarks likely transfer. "
            "Verify with a small domain-specific sample."
        )
    recs.append(f"Closest reference profile: {closest_ref}")
    if recall_range and recall_range != (0.0, 0.0):
        lo = int(recall_range[0] * 100)
        hi = int(recall_range[1] * 100)
        recs.append(
            f"Expected Recall@5: {lo}–{hi}% "
            f"(reference value from benchmarks at similar specificity)"
        )
    return recs


def score_corpus(passages: List[str],
                 queries: List[str],
                 stopwords: Optional[Set[str]] = None,
                 compare_references: bool = True,
                 ) -> DomainReport:
    """Run the full domain difficulty diagnostic."""
    index = build_inverted_index(passages, stopwords)
    n_docs = len(passages)

    query_results = [score_query(q, index, n_docs, stopwords) for q in queries]

    if query_results:
        corpus_specificity = round(
            sum(q.specificity for q in query_results) / len(query_results), 4
        )
    else:
        corpus_specificity = 0.0

    difficulty = DifficultyTier.from_specificity(corpus_specificity)

    ref = closest_profile(corpus_specificity, BUILTIN_PROFILES) if compare_references else None
    ref_name = ref.name if ref else None
    recall_range = ref.expected_recall_range if ref else None

    ambiguous = _top_ambiguous_terms(index)
    recs = _recommendations(corpus_specificity, difficulty, ref_name or "N/A",
                            recall_range)

    return DomainReport(
        specificity=corpus_specificity,
        difficulty=difficulty,
        closest_reference=ref_name,
        queries=query_results,
        ambiguous_terms=ambiguous,
        expected_recall_range=recall_range,
        recommendations=recs,
    )
