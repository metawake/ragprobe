"""Tests for ragprobe.models."""

import json

from ragprobe.models import DifficultyTier, DomainReport, QueryDifficulty


class TestDifficultyTier:
    def test_easy_above_08(self):
        assert DifficultyTier.from_specificity(0.9) == DifficultyTier.EASY
        assert DifficultyTier.from_specificity(0.81) == DifficultyTier.EASY
        assert DifficultyTier.from_specificity(1.0) == DifficultyTier.EASY

    def test_medium_03_to_08(self):
        assert DifficultyTier.from_specificity(0.5) == DifficultyTier.MEDIUM
        assert DifficultyTier.from_specificity(0.3) == DifficultyTier.MEDIUM
        assert DifficultyTier.from_specificity(0.8) == DifficultyTier.MEDIUM

    def test_hard_below_03(self):
        assert DifficultyTier.from_specificity(0.1) == DifficultyTier.HARD
        assert DifficultyTier.from_specificity(0.0) == DifficultyTier.HARD
        assert DifficultyTier.from_specificity(0.29) == DifficultyTier.HARD

    def test_str_enum_values(self):
        assert DifficultyTier.EASY.value == "easy"
        assert DifficultyTier.MEDIUM.value == "medium"
        assert DifficultyTier.HARD.value == "hard"


class TestDomainReport:
    def _make_report(self):
        queries = [
            QueryDifficulty("easy q", 0.9, DifficultyTier.EASY),
            QueryDifficulty("medium q", 0.5, DifficultyTier.MEDIUM),
            QueryDifficulty("hard q", 0.1, DifficultyTier.HARD),
        ]
        return DomainReport(
            specificity=0.5,
            difficulty=DifficultyTier.MEDIUM,
            closest_reference="HotpotQA (Wikipedia)",
            queries=queries,
            ambiguous_terms=[("data", 100), ("processing", 50)],
        )

    def test_hardest_queries_sorted(self):
        report = self._make_report()
        hardest = report.hardest_queries
        assert hardest[0].text == "hard q"
        assert hardest[-1].text == "easy q"

    def test_easiest_queries_sorted(self):
        report = self._make_report()
        easiest = report.easiest_queries
        assert easiest[0].text == "easy q"
        assert easiest[-1].text == "hard q"

    def test_summary_counts(self):
        report = self._make_report()
        counts = report.summary_counts()
        assert counts == {"easy": 1, "medium": 1, "hard": 1}

    def test_to_json_roundtrip(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        assert data["specificity"] == 0.5
        assert data["difficulty"] == "medium"
        assert len(data["queries"]) == 3
        assert data["queries"][0]["difficulty"] in ("easy", "medium", "hard")
