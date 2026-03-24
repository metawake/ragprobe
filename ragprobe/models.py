"""Dataclasses for ragprobe domain difficulty reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class DifficultyTier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def from_specificity(cls, specificity: float) -> DifficultyTier:
        if specificity > 0.8:
            return cls.EASY
        if specificity >= 0.3:
            return cls.MEDIUM
        return cls.HARD


@dataclass
class QueryDifficulty:
    text: str
    specificity: float
    difficulty: DifficultyTier
    ambiguous_terms: List[str] = field(default_factory=list)
    idf_avg: float = 0.0
    idf_max: float = 0.0


@dataclass
class ReferenceProfile:
    name: str
    specificity: float
    difficulty: DifficultyTier
    description: str = ""
    expected_recall_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class DomainReport:
    specificity: float
    difficulty: DifficultyTier
    closest_reference: Optional[str]
    queries: List[QueryDifficulty]
    ambiguous_terms: List[Tuple[str, int]]
    expected_recall_range: Optional[Tuple[float, float]] = None
    recommendations: List[str] = field(default_factory=list)

    @property
    def hardest_queries(self) -> List[QueryDifficulty]:
        return sorted(self.queries, key=lambda q: q.specificity)

    @property
    def easiest_queries(self) -> List[QueryDifficulty]:
        return sorted(self.queries, key=lambda q: q.specificity, reverse=True)

    def summary_counts(self) -> dict:
        easy = sum(1 for q in self.queries if q.difficulty == DifficultyTier.EASY)
        medium = sum(1 for q in self.queries if q.difficulty == DifficultyTier.MEDIUM)
        hard = sum(1 for q in self.queries if q.difficulty == DifficultyTier.HARD)
        return {"easy": easy, "medium": medium, "hard": hard}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["difficulty"] = self.difficulty.value
        d["queries"] = [
            {**asdict(q), "difficulty": q.difficulty.value} for q in self.queries
        ]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
