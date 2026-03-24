"""Tests for ragprobe.profiles."""

from ragprobe.models import DifficultyTier
from ragprobe.profiles import BUILTIN_PROFILES, closest_profile


class TestBuiltinProfiles:
    def test_profiles_exist(self):
        assert len(BUILTIN_PROFILES) >= 7

    def test_profiles_sorted_descending_by_specificity(self):
        specs = [p.specificity for p in BUILTIN_PROFILES]
        assert specs == sorted(specs, reverse=True)

    def test_each_profile_has_required_fields(self):
        for p in BUILTIN_PROFILES:
            assert p.name
            assert 0.0 <= p.specificity <= 1.0
            assert isinstance(p.difficulty, DifficultyTier)
            lo, hi = p.expected_recall_range
            assert 0.0 <= lo <= hi <= 1.0


class TestClosestProfile:
    def test_exact_match(self):
        result = closest_profile(0.946)
        assert "HotpotQA" in result.name

    def test_close_to_gdpr(self):
        result = closest_profile(0.18)
        assert "GDPR" in result.name

    def test_very_low_matches_rfc(self):
        result = closest_profile(0.01)
        assert "RFC" in result.name

    def test_very_high_matches_casehold(self):
        result = closest_profile(0.99)
        assert "CaseHOLD" in result.name

    def test_medium_matches_medical_or_tech(self):
        result = closest_profile(0.6)
        assert result.difficulty == DifficultyTier.MEDIUM

    def test_recall_ranges_decrease_with_difficulty(self):
        easy = [p for p in BUILTIN_PROFILES if p.difficulty == DifficultyTier.EASY]
        hard = [p for p in BUILTIN_PROFILES if p.difficulty == DifficultyTier.HARD]
        easy_max = max(hi for _, hi in [p.expected_recall_range for p in easy])
        hard_max = max(hi for _, hi in [p.expected_recall_range for p in hard])
        assert easy_max > hard_max
