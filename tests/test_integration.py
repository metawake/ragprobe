"""Integration tests for the DomainProbe public API."""

import json
import tempfile
from pathlib import Path

from ragprobe import DomainProbe, DifficultyTier


class TestDomainProbeAPI:
    def test_score_with_string_lists(self):
        probe = DomainProbe(
            corpus=[
                "Quantum computing leverages superposition and entanglement",
                "Classical computers use binary transistor logic gates",
                "Machine learning models optimize loss functions via gradient descent",
            ],
            queries=[
                "How does quantum superposition work?",
                "What are gradient descent optimizers?",
            ],
        )
        report = probe.score()
        assert 0.0 <= report.specificity <= 1.0
        assert isinstance(report.difficulty, DifficultyTier)
        assert len(report.queries) == 2
        assert report.closest_reference is not None

    def test_easy_domain_detected(self):
        passages = [
            "Acme Corp reported Q3 revenue of $4.2B on October 15 2024",
            "Beta Inc acquired Gamma Ltd for $800M in a cash deal",
            "Delta Systems filed patent US-2024-0012345 for neural chip design",
        ]
        queries = [
            "What was Acme Corp Q3 revenue?",
            "How much did Beta Inc pay for Gamma Ltd?",
        ]
        probe = DomainProbe(corpus=passages, queries=queries)
        report = probe.score()
        assert report.difficulty == DifficultyTier.EASY

    def test_hard_domain_detected(self):
        passages = [
            f"controller shall process personal data subject regulation {i}"
            for i in range(30)
        ]
        queries = [
            "controller data processing regulation",
            "personal data subject regulation",
            "controller subject personal data",
        ]
        probe = DomainProbe(corpus=passages, queries=queries)
        report = probe.score()
        assert report.difficulty == DifficultyTier.HARD

    def test_score_from_directory(self, tmp_path):
        (tmp_path / "doc1.txt").write_text(
            "Specialized quantum chromodynamics research paper"
        )
        (tmp_path / "doc2.txt").write_text(
            "Neutrino oscillation measurements at CERN"
        )

        probe = DomainProbe(
            corpus=str(tmp_path),
            queries=["quantum chromodynamics cross section"],
        )
        report = probe.score()
        assert len(report.queries) == 1
        assert report.specificity >= 0.0

    def test_json_output_is_valid(self):
        probe = DomainProbe(
            corpus=["doc one alpha", "doc two beta"],
            queries=["alpha query"],
        )
        report = probe.score()
        data = json.loads(report.to_json())
        assert "specificity" in data
        assert "difficulty" in data
        assert data["difficulty"] in ("easy", "medium", "hard")
        assert "expected_recall_range" in data

    def test_recall_range_present_for_easy_domain(self):
        probe = DomainProbe(
            corpus=[
                "Acme Corp reported Q3 revenue of $4.2B on October 15 2024",
                "Beta Inc acquired Gamma Ltd for $800M in a cash deal",
                "Delta Systems filed patent US-2024-0012345 for neural chip design",
            ],
            queries=["What was Acme Corp Q3 revenue?"],
        )
        report = probe.score()
        assert report.expected_recall_range is not None
        lo, hi = report.expected_recall_range
        assert lo >= 0.80

    def test_recall_range_low_for_hard_domain(self):
        probe = DomainProbe(
            corpus=[f"controller shall process personal data subject regulation {i}"
                    for i in range(30)],
            queries=["controller data processing regulation"],
        )
        report = probe.score()
        assert report.expected_recall_range is not None
        _, hi = report.expected_recall_range
        assert hi <= 0.40

    def test_hardest_queries_ordering(self):
        passages = [
            "Alpha Centauri is a specific star system",
            f"data processing controller " * 5,
        ]
        queries = [
            "Alpha Centauri distance",
            "data processing requirements",
        ]
        probe = DomainProbe(corpus=passages, queries=queries)
        report = probe.score()
        hardest = report.hardest_queries
        assert hardest[0].specificity <= hardest[-1].specificity

    def test_custom_stopwords(self):
        probe = DomainProbe(
            corpus=["alpha beta gamma delta"],
            queries=["alpha beta"],
            stopwords={"alpha"},
        )
        report = probe.score()
        # "alpha" is now a stopword, so only "beta" counts
        assert len(report.queries) == 1
