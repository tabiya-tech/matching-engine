"""CosineSkillMatcher component tests."""

from tests.components.conftest import _job_with_essential, _user_with_skills

_SKILL = "manage staff"


class TestSkillScorer:
    def test_score_pair_returns_expected_keys(self, cosine_matcher):
        user = _user_with_skills([_SKILL])
        job = _job_with_essential([_SKILL])
        out = cosine_matcher.score_pair(user, job)
        assert "mean_best_cosine" in out
        assert "per_job_skill" in out
        assert 0.0 <= out["mean_best_cosine"] <= 1.0

    def test_empty_user_skills_yield_zero_score(self, cosine_matcher):
        user = {"skills_vector": {"top_skills": []}}
        job = _job_with_essential([_SKILL])
        out = cosine_matcher.score_pair(user, job)
        assert out["mean_best_cosine"] == 0.0
        assert out["per_job_skill"] == []

    def test_empty_job_skills_yield_zero_score(self, cosine_matcher):
        user = _user_with_skills([_SKILL])
        job = {"essential_skills": [], "optional_skills": []}
        out = cosine_matcher.score_pair(user, job)
        assert out["mean_best_cosine"] == 0.0


class TestDisplayLabelLanguage:
    """The labels echoed back in a response follow the deployment's TARGET_LANGUAGE.

    Resolution is id-based and language-neutral — a Spanish profile and an English one land on
    the same embedding row — but ``job_skill_label`` / ``best_user_skill_label`` are strings the
    caller reads. They used to come from the canonical (English) pack only, so a Spanish
    deployment answered a Spanish request with English skill names.
    """

    def test_matcher_exposes_the_deployment_languages_labels(self, cosine_matcher):
        from app.languages import default_language

        expected = cosine_matcher._packs.display_labels(default_language())
        assert cosine_matcher.skill_labels == expected

    def test_every_language_pack_can_label_the_same_skill(self, cosine_matcher):
        # What makes the swap safe: each pack covers the canonical id space, so switching the
        # display language cannot blank a label.
        packs = cosine_matcher._packs
        for language in packs.loaded_languages:
            labels = packs.display_labels(language)
            assert len(labels) >= len(packs.skill_labels)
            assert all(labels.values())

    def test_non_canonical_packs_really_change_the_labels(self, cosine_matcher):
        # Guards the substance rather than the plumbing: switching the display language has to
        # produce different strings, not the canonical ones under a different key.
        packs = cosine_matcher._packs
        for language in packs.loaded_languages:
            if language == packs.canonical_language:
                continue
            labels = packs.display_labels(language)
            translated = sum(
                1 for sid, lab in labels.items() if packs.skill_labels.get(sid) != lab
            )
            assert translated > 1000, f"{language}: only {translated} labels differ"

    def test_scored_rows_use_that_language(self, cosine_matcher):
        user = _user_with_skills([_SKILL])
        job = _job_with_essential([_SKILL])
        rows = cosine_matcher.score_pair_v4(user, job)["per_job_skill"]
        assert rows, "expected the resolved skill to score"
        for row in rows:
            expected = cosine_matcher.skill_labels.get(row["job_skill_id"])
            assert row["job_skill_label"] == expected
