"""ragprobe — pre-deployment domain difficulty diagnostic for RAG.

Know if your benchmark transfers before you deploy.
"""

from __future__ import annotations

from typing import List, Optional, Set, Union

from ragprobe.loaders import load_corpus, load_queries
from ragprobe.models import DifficultyTier, DomainReport, QueryDifficulty
from ragprobe.profiles import BUILTIN_PROFILES, ReferenceProfile
from ragprobe.scorer import score_corpus

__version__ = "0.1.0"

__all__ = [
    "DomainProbe",
    "DomainReport",
    "QueryDifficulty",
    "DifficultyTier",
    "ReferenceProfile",
    "BUILTIN_PROFILES",
    "__version__",
]


class DomainProbe:
    """High-level API for domain difficulty scoring.

    Example::

        probe = DomainProbe(
            corpus=["path/to/docs/"],
            queries=["What are the controller's obligations?"],
        )
        report = probe.score()
        print(report.specificity)   # 0.177
        print(report.difficulty)    # DifficultyTier.HARD
    """

    def __init__(
        self,
        corpus: Union[str, List[str]],
        queries: Union[str, List[str]],
        pre_chunked: bool = False,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        self._passages = load_corpus(corpus, pre_chunked=pre_chunked)
        self._queries = load_queries(queries) if isinstance(queries, str) else queries
        self._stopwords = stopwords
        self._pre_chunked = pre_chunked

    @property
    def passages(self) -> List[str]:
        return self._passages

    @property
    def queries(self) -> List[str]:
        return self._queries

    def score(self, compare_references: bool = True) -> DomainReport:
        """Run the full domain difficulty diagnostic."""
        return score_corpus(
            self._passages,
            self._queries,
            stopwords=self._stopwords,
            compare_references=compare_references,
        )
