"""Built-in reference profiles from measured corpora."""

from __future__ import annotations

from typing import List, Optional

from ragprobe.models import DifficultyTier, ReferenceProfile

BUILTIN_PROFILES: List[ReferenceProfile] = [
    ReferenceProfile("CaseHOLD (legal holdings)", 0.985, DifficultyTier.EASY,
                      "Case names and statute numbers act as unique identifiers",
                      (0.90, 0.99)),
    ReferenceProfile("Financial (SEC filings)", 0.95, DifficultyTier.EASY,
                      "Company names, dates, figures",
                      (0.85, 0.95)),
    ReferenceProfile("HotpotQA (Wikipedia)", 0.946, DifficultyTier.EASY,
                      "Named entities, dates, specific facts",
                      (0.85, 0.95)),
    ReferenceProfile("Technical docs", 0.80, DifficultyTier.MEDIUM,
                      "Product terms provide moderate specificity",
                      (0.55, 0.80)),
    ReferenceProfile("Medical (clinical)", 0.55, DifficultyTier.MEDIUM,
                      "Clinical terminology helps but overlaps",
                      (0.35, 0.60)),
    ReferenceProfile("GDPR (regulatory)", 0.177, DifficultyTier.HARD,
                      "Generic vocabulary: 'data', 'processing', 'controller' everywhere",
                      (0.15, 0.35)),
    ReferenceProfile("RFC (technical standards)", 0.024, DifficultyTier.HARD,
                      "'client', 'server', 'request', 'response' in every passage",
                      (0.05, 0.20)),
]


def closest_profile(specificity: float,
                    profiles: Optional[List[ReferenceProfile]] = None,
                    ) -> ReferenceProfile:
    """Return the reference profile with specificity closest to the given value."""
    pool = profiles or BUILTIN_PROFILES
    return min(pool, key=lambda p: abs(p.specificity - specificity))
