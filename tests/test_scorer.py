"""Tests for ragprobe.scorer."""

from ragprobe.models import DifficultyTier
from ragprobe.scorer import (
    build_inverted_index,
    score_corpus,
    score_query,
    tokenize,
)


class TestTokenize:
    def test_removes_stopwords(self):
        tokens = tokenize("the quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "over" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_lowercases(self):
        tokens = tokenize("Python RAGPROBE Testing")
        assert "python" in tokens
        assert "ragprobe" in tokens

    def test_strips_single_chars(self):
        tokens = tokenize("I went to a shop")
        assert "went" in tokens
        for t in tokens:
            assert len(t) > 1

    def test_custom_stopwords(self):
        tokens = tokenize("custom words removed here", stopwords={"custom", "removed"})
        assert "custom" not in tokens
        assert "removed" not in tokens
        assert "words" in tokens
        assert "here" in tokens


class TestInvertedIndex:
    def test_basic_index(self):
        passages = ["cat sat on mat", "dog sat on log", "cat chased dog"]
        index = build_inverted_index(passages)
        assert 0 in index["cat"]
        assert 2 in index["cat"]
        assert 1 in index["dog"]
        assert 2 in index["dog"]

    def test_term_appears_in_correct_passages(self):
        passages = ["xalpha term", "common term", "xbeta term"]
        index = build_inverted_index(passages)
        assert index["xalpha"] == {0}
        assert index["xbeta"] == {2}
        assert index["term"] == {0, 1, 2}


class TestScoreQuery:
    def _easy_setup(self):
        """Corpus where query terms are highly specific (appear in <5 passages)."""
        passages = [
            "Quantum entanglement enables teleportation of quantum states",
            "Classical mechanics describes macroscopic motion",
            "Thermodynamics governs heat transfer processes",
        ]
        index = build_inverted_index(passages)
        return index, len(passages)

    def _hard_setup(self):
        """Corpus where query terms appear in many passages."""
        passages = [f"data processing controller subject personal item {i}"
                    for i in range(20)]
        index = build_inverted_index(passages)
        return index, len(passages)

    def test_specific_query_scores_easy(self):
        index, n = self._easy_setup()
        result = score_query("quantum entanglement teleportation", index, n)
        assert result.specificity > 0.8
        assert result.difficulty == DifficultyTier.EASY

    def test_ambiguous_query_scores_hard(self):
        index, n = self._hard_setup()
        result = score_query("data processing controller", index, n)
        assert result.specificity < 0.3
        assert result.difficulty == DifficultyTier.HARD
        assert len(result.ambiguous_terms) > 0

    def test_empty_query_returns_hard(self):
        index, n = self._easy_setup()
        result = score_query("the a an", index, n)
        assert result.specificity == 0.0
        assert result.difficulty == DifficultyTier.HARD


class TestScoreCorpus:
    def test_easy_corpus_report(self):
        passages = [
            "Alpha Centauri stellar observations",
            "Hubble telescope deep field imaging",
            "Mars rover geological sampling",
            "Jupiter magnetosphere exploration probe",
        ]
        queries = [
            "Alpha Centauri stellar distance",
            "Hubble deep field resolution",
        ]
        report = score_corpus(passages, queries)
        assert report.specificity > 0.5
        assert report.difficulty in (DifficultyTier.EASY, DifficultyTier.MEDIUM)
        assert len(report.queries) == 2
        assert report.closest_reference is not None

    def test_hard_corpus_report(self):
        passages = [f"data processing controller subject personal regulation {i}"
                    for i in range(50)]
        queries = [
            "data processing regulation",
            "controller subject personal",
            "personal data processing",
        ]
        report = score_corpus(passages, queries)
        assert report.specificity < 0.3
        assert report.difficulty == DifficultyTier.HARD
        assert len(report.ambiguous_terms) > 0
        assert len(report.recommendations) > 0
        assert report.expected_recall_range is not None
        lo, hi = report.expected_recall_range
        assert lo < hi

    def test_report_json_output(self):
        passages = ["unique document alpha", "unique document beta"]
        queries = ["alpha query"]
        report = score_corpus(passages, queries)
        import json
        data = json.loads(report.to_json())
        assert "specificity" in data
        assert "difficulty" in data
        assert "queries" in data

    def test_no_queries_returns_zero(self):
        passages = ["some text"]
        report = score_corpus(passages, [])
        assert report.specificity == 0.0

    def test_without_reference_comparison(self):
        passages = ["hello world doc"]
        queries = ["hello world"]
        report = score_corpus(passages, queries, compare_references=False)
        assert report.closest_reference is None
