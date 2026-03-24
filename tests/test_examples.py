"""Tests that shipped example corpora produce the expected difficulty tiers."""

from pathlib import Path

import pytest

from ragprobe import DomainProbe, DifficultyTier

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.skipif(not EXAMPLES_DIR.exists(), reason="examples/ not present")
class TestExampleCorpora:
    def test_wiki_is_easy(self):
        probe = DomainProbe(
            corpus=str(EXAMPLES_DIR / "wiki" / "corpus"),
            queries=str(EXAMPLES_DIR / "wiki" / "queries.txt"),
        )
        report = probe.score()
        assert report.difficulty == DifficultyTier.EASY
        assert report.specificity > 0.8

    def test_regulatory_is_hard(self):
        probe = DomainProbe(
            corpus=str(EXAMPLES_DIR / "regulatory" / "corpus"),
            queries=str(EXAMPLES_DIR / "regulatory" / "queries.txt"),
        )
        report = probe.score()
        assert report.difficulty == DifficultyTier.HARD
        assert report.specificity < 0.3

    def test_technical_is_medium(self):
        probe = DomainProbe(
            corpus=str(EXAMPLES_DIR / "technical" / "corpus"),
            queries=str(EXAMPLES_DIR / "technical" / "queries.txt"),
        )
        report = probe.score()
        assert report.difficulty == DifficultyTier.MEDIUM
        assert 0.3 <= report.specificity <= 0.8

    def test_all_three_tiers_differ(self):
        results = {}
        for domain in ("wiki", "regulatory", "technical"):
            probe = DomainProbe(
                corpus=str(EXAMPLES_DIR / domain / "corpus"),
                queries=str(EXAMPLES_DIR / domain / "queries.txt"),
            )
            results[domain] = probe.score()

        assert results["wiki"].specificity > results["technical"].specificity
        assert results["technical"].specificity > results["regulatory"].specificity
