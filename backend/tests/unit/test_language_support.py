"""Language registry, per-language config, and cross-language skill resolution.

The load-bearing claim these tests defend: a Spanish job posting matched against a Spanish
user profile resolves through the *same* embedding rows as the English equivalent, so no
Spanish retrain is needed for matching to work. Only the cross-encoder checkpoint, the
stopwords and the display labels are language-specific, and those follow the deployment's
``TARGET_LANGUAGE`` — nothing about the language comes from the request.
"""

import csv
import inspect

import pytest

from app import config
from app.languages import (
    CANONICAL_LANGUAGE,
    LANGUAGE_REGISTRY,
    default_language,
    enabled_languages,
    get_language_config,
    normalise_language,
)
from app.services.skill_label_packs import SkillLabelPacks, oldest_uuid

csv.field_size_limit(10_000_000)


@pytest.fixture(autouse=True)
def _clear_language_caches(monkeypatch):
    """Language configs read env at import time; keep tests independent."""
    import app.languages as languages

    for var in ("TARGET_LANGUAGE", "MATCHING_LANGUAGE", "ENABLED_LANGUAGES"):
        monkeypatch.delenv(var, raising=False)
    languages.get_language_config.cache_clear()
    languages._locale_index.cache_clear()
    yield
    languages.get_language_config.cache_clear()
    languages._locale_index.cache_clear()


class TestRegistry:
    def test_every_registered_language_has_a_config(self):
        for code in LANGUAGE_REGISTRY:
            cfg = get_language_config(code)
            assert cfg["language"] == code
            for key in ("name", "locales", "resources_subdir", "cross_encoder_model"):
                assert cfg.get(key), f"{code} config is missing {key}"

    def test_canonical_language_is_registered(self):
        assert CANONICAL_LANGUAGE in LANGUAGE_REGISTRY

    def test_unregistered_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_language_config("fr")

    def test_english_cross_encoder_default_is_unchanged(self):
        # The pre-language default: existing deployments must score identically.
        assert (
            get_language_config("en")["cross_encoder_model"]
            == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def test_spanish_uses_a_multilingual_cross_encoder(self):
        # An English-only reranker on Spanish label text scores poorly.
        assert "mmarco" in get_language_config("es")["cross_encoder_model"]


class TestNormaliseLanguage:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("en", "en"),
            ("English", "en"),
            ("es", "es"),
            ("AR-es", "es"),  # taxonomy locale
            ("es_AR.UTF-8", "es"),
            ("spanish", "es"),
            ("argentina", "es"),
        ],
    )
    def test_resolves_locale_spellings(self, value, expected):
        assert normalise_language(value) == expected

    def test_unknown_language_falls_back_instead_of_raising(self):
        # A bad `language` on a request must not become a 500.
        assert normalise_language("klingon") == "en"
        assert normalise_language(None) == "en"


class TestDefaults:
    def test_defaults_to_english(self):
        assert default_language() == "en"

    def test_target_language_env(self, monkeypatch):
        monkeypatch.setenv("TARGET_LANGUAGE", "AR-es")
        assert default_language() == "es"

    def test_all_packs_are_loaded_by_default(self):
        # Loading every pack is what lets one deployment serve both languages.
        assert enabled_languages() == LANGUAGE_REGISTRY

    def test_enabled_languages_always_includes_the_canonical_one(self, monkeypatch):
        # It defines the id space the other packs are mapped onto.
        monkeypatch.setenv("ENABLED_LANGUAGES", "es")
        assert enabled_languages()[0] == CANONICAL_LANGUAGE
        assert "es" in enabled_languages()


class TestResourcePaths:
    def test_each_language_has_its_own_taxonomy_pack(self):
        for code in LANGUAGE_REGISTRY:
            paths = config.taxonomy_pack_paths(code)
            subdir = get_language_config(code)["resources_subdir"]
            for key, path in paths.items():
                assert f"/skill_taxonomy/{subdir}/" in path, f"{code}/{key} → {path}"

    def test_explicit_env_override_pins_every_language(self, monkeypatch):
        # Scripts that deliberately want one pack keep working.
        monkeypatch.setenv("SKILLS_CSV_PATH", "/tmp/pinned-skills.csv")
        assert config.taxonomy_pack_paths("es")["skills"] == "/tmp/pinned-skills.csv"

    def test_ignore_pins_returns_the_per_language_layout(self, monkeypatch):
        # How the resolver recovers when a pin lands off the embedding id space.
        monkeypatch.setenv("SKILLS_CSV_PATH", "/tmp/pinned-skills.csv")
        for code in LANGUAGE_REGISTRY:
            paths = config.taxonomy_pack_paths(code, ignore_pins=True)
            subdir = get_language_config(code)["resources_subdir"]
            assert f"/skill_taxonomy/{subdir}/" in paths["skills"]

    def test_occupation_db_falls_back_to_the_canonical_pack(self):
        # The occupation DB is keyed on ISCO codes / work activities, so English serves
        # a language that has not translated it.
        assert config.occupation_json_path("es") == config.occupation_json_path("en")

    def test_stopwords_differ_per_language(self):
        assert "the" in config.stopwords("en")
        assert "la" in config.stopwords("es")
        assert "la" not in config.stopwords("en")


class TestTaxonomyPacksShareOneIdSpace:
    """The invariant the whole design rests on."""

    @staticmethod
    def _oldest_uuids(language: str) -> set[str]:
        path = config.taxonomy_pack_paths(language)["skills"]
        with open(path, encoding="utf-8", newline="") as f:
            return {
                oldest_uuid(row.get("UUIDHISTORY") or "") for row in csv.DictReader(f)
            }

    def test_every_pack_joins_onto_the_canonical_one(self):
        canonical = self._oldest_uuids(CANONICAL_LANGUAGE)
        canonical.discard("")
        assert len(canonical) > 10_000
        for code in LANGUAGE_REGISTRY:
            if code == CANONICAL_LANGUAGE:
                continue
            other = self._oldest_uuids(code)
            other.discard("")
            unjoinable = other - canonical
            assert not unjoinable, (
                f"{code}: {len(unjoinable)} skills have no counterpart in the "
                f"{CANONICAL_LANGUAGE} id space — the packs are from different releases"
            )

    def test_pack_ids_are_per_locale(self):
        """Sanity check on the premise: ids are NOT shared, which is why we join on UUIDs."""

        def ids(language: str) -> set[str]:
            path = config.taxonomy_pack_paths(language)["skills"]
            with open(path, encoding="utf-8", newline="") as f:
                return {str(row.get("ID") or "") for row in csv.DictReader(f)}

        assert not (ids("en") & ids("es"))


class TestCrossLanguageResolution:
    """End-to-end over the real packs, with a synthetic embedding id space."""

    @pytest.fixture(scope="class")
    def sample_rows(self):
        path = config.taxonomy_pack_paths("en")["skills"]
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [next(reader) for _ in range(50)]

    @pytest.fixture(scope="class")
    def packs(self, sample_rows):
        # Only the sampled English ids are "in the embedding artefact"; that is enough to
        # prove the join and keeps this test fast.
        embedding_ids = {str(r["ID"]) for r in sample_rows}
        return SkillLabelPacks(embedding_ids, languages=("en", "es"))

    def test_both_packs_load(self, packs):
        assert packs.loaded_languages == ["en", "es"]
        assert not packs.unjoined_by_language

    def test_english_label_resolves_to_its_own_id(self, packs, sample_rows):
        row = sample_rows[0]
        assert packs.resolve_label(row["PREFERREDLABEL"]) == str(row["ID"])

    def test_spanish_label_resolves_to_the_same_canonical_id(self, packs, sample_rows):
        """The core claim: a Spanish label lands on the English row's vector."""
        row = sample_rows[0]
        canonical_id = str(row["ID"])
        spanish_label = packs.labels_by_language["es"][canonical_id]
        assert spanish_label != row["PREFERREDLABEL"], "expected a translated label"
        assert packs.resolve_label(spanish_label) == canonical_id

    def test_resolution_is_case_and_whitespace_insensitive(self, packs, sample_rows):
        canonical_id = str(sample_rows[0]["ID"])
        spanish_label = packs.labels_by_language["es"][canonical_id]
        assert packs.resolve_label(f"  {spanish_label.upper()}  ") == canonical_id

    def test_spanish_taxonomy_id_maps_to_the_canonical_id(self, packs, sample_rows):
        """Fallback path for a document whose label is missing but whose id is Spanish."""
        canonical_id = str(sample_rows[0]["ID"])
        spanish_ids = [
            local
            for local, canonical in packs.id_to_canonical.items()
            if canonical == canonical_id
        ]
        assert spanish_ids, "expected an es id mapped onto this canonical id"
        assert packs.resolve_id(spanish_ids[0]) == canonical_id

    def test_canonical_id_resolves_to_itself(self, packs, sample_rows):
        canonical_id = str(sample_rows[0]["ID"])
        assert packs.resolve_id(canonical_id) == canonical_id

    def test_unknown_label_and_id_return_none(self, packs):
        assert packs.resolve_label("definitely not a taxonomy skill") is None
        assert packs.resolve_label("") is None
        assert packs.resolve_id("not-an-id") is None
        assert packs.resolve_id(None) is None

    def test_uuid_resolution_works_across_locales(self, packs, sample_rows):
        row = sample_rows[0]
        assert packs.resolve_uuid(oldest_uuid(row["UUIDHISTORY"])) == str(row["ID"])

    def test_display_labels_follow_the_requested_language(self, packs, sample_rows):
        canonical_id = str(sample_rows[0]["ID"])
        assert packs.display_labels("en")[canonical_id] == row_label(sample_rows[0])
        assert packs.display_labels("es")[canonical_id] != row_label(sample_rows[0])
        # An unknown language falls back rather than returning blanks.
        assert packs.display_labels("fr")[canonical_id] == row_label(sample_rows[0])

    def test_labels_outside_the_embedding_space_are_dropped(self, packs):
        # A real skill that is not in this test's sampled id space must not resolve.
        path = config.taxonomy_pack_paths("en")["skills"]
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        outside = next(r for r in rows if str(r["ID"]) not in packs._embedding_ids)
        assert packs.resolve_label(outside["PREFERREDLABEL"]) is None


def row_label(row: dict) -> str:
    return (row.get("PREFERREDLABEL") or "").strip()


class TestPinnedPackOffTheIdSpace:
    """A deployment that pins the taxonomy must not end up with a silently empty resolver.

    ``SKILLS_CSV_PATH`` pins *every* language, canonical included. Pointing it at the Spanish
    pack therefore loaded per-locale AR ids as the canonical id space, none of which exist in
    the embedding artefact: all 13,896 rows were dropped, no label resolved, every posting came
    back with zero essential-coverage and (via the Phase-2 demotion) final_score 0.0. The pin is
    unusable in that state, so the resolver ignores it and loads the per-language packs.
    """

    @pytest.fixture(scope="class")
    def embedding_ids(self):
        path = config.taxonomy_pack_paths("en", ignore_pins=True)["skills"]
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return {str(next(reader)["ID"]) for _ in range(50)}

    def test_pin_off_the_id_space_is_ignored(self, monkeypatch, embedding_ids):
        monkeypatch.setenv(
            "SKILLS_CSV_PATH",
            config.taxonomy_pack_paths("es", ignore_pins=True)["skills"],
        )
        packs = SkillLabelPacks(embedding_ids, languages=("en", "es"))
        assert packs._skills_paths["en"].parent.name == "en"
        assert packs._skills_paths["es"].parent.name == "es"
        # The proof it recovered: labels resolve, in both languages, onto the canonical ids.
        canonical_id = min(embedding_ids)
        for language in ("en", "es"):
            label = packs.labels_by_language[language][canonical_id]
            assert packs.resolve_label(label) == canonical_id

    def test_a_usable_pin_is_still_honoured(self, monkeypatch, embedding_ids):
        # Scripts pinning the canonical pack itself must be unaffected by the recovery path.
        pinned = config.taxonomy_pack_paths("en", ignore_pins=True)["skills"]
        monkeypatch.setenv("SKILLS_CSV_PATH", pinned)
        packs = SkillLabelPacks(embedding_ids, languages=("en", "es"))
        assert [str(p) for p in packs._skills_paths.values()] == [pinned, pinned]


class TestLanguageIsNotPerRequest:
    """The language is deployment config, not request input."""

    def test_match_request_has_no_language_field(self):
        from app.schemas import MatchRequest

        assert "language" not in MatchRequest.model_fields
        # A consumer that still sends one is not rejected — the field is simply ignored.
        assert not hasattr(MatchRequest(language="es"), "language")

    def test_match_v4_takes_no_language_query_param(self):
        from app.routes import match_v4

        assert "language" not in inspect.signature(match_v4).parameters

    def test_the_engine_takes_no_language_argument(self):
        from app.services.match_concat_gemini_ce_service import (
            run_match_concat_gemini_ce,
        )
        from app.services.match_v3_full_service import run_match_v3_full
        from app.services.match_v4_full_service import run_match_v4_full

        for fn in (run_match_concat_gemini_ce, run_match_v3_full, run_match_v4_full):
            assert "language" not in inspect.signature(fn).parameters, fn.__name__


class TestCrossEncoderPerLanguage:
    def test_model_name_is_selected_by_language(self):
        assert config.cross_encoder_model_name("en") != config.cross_encoder_model_name(
            "es"
        )

    def test_env_override_per_language(self, monkeypatch):
        monkeypatch.setenv("CROSS_ENCODER_MODEL_NAME_ES", "some-org/custom-es-reranker")
        import app.languages as languages

        languages.get_language_config.cache_clear()
        assert config.cross_encoder_model_name("es") == "some-org/custom-es-reranker"

    def test_reranker_reports_its_language_without_loading_weights(self):
        from app.services.cross_encoder.reranker import CrossEncoderReranker

        # Construction must not download a checkpoint — the model is lazy.
        rr = CrossEncoderReranker(language="AR-es")
        assert rr.language == "es"
        assert rr.model_name == config.cross_encoder_model_name("es")
        assert rr._model is None

    @pytest.mark.parametrize(
        ("language", "checkpoint"),
        [
            ("en", "ms-marco-MiniLM-L-6-v2"),
            ("es", "mmarco-mMiniLMv2-L12-H384-v1"),
        ],
    )
    def test_vendored_lookup_matches_the_setup_sh_layout(self, language, checkpoint):
        # setup.sh writes resources/models/cross-encoder/<repo-name>/ per language.
        from app.services.cross_encoder.reranker import CrossEncoderReranker

        dirs = CrossEncoderReranker(language=language)._vendored_model_dirs()
        assert any(
            d.name == checkpoint and d.parent.name == "cross-encoder" for d in dirs
        ), dirs

    def test_a_non_canonical_language_never_falls_back_to_the_flat_dir(self):
        # The flat resources/models/cross-encoder/ held the English checkpoint before the
        # language split; Spanish must not pick it up (nor its parent).
        from app.services.cross_encoder.reranker import CrossEncoderReranker

        dirs = CrossEncoderReranker(language="es")._vendored_model_dirs()
        assert all(d.name != "cross-encoder" for d in dirs), dirs

    def test_the_shared_parent_directory_is_not_treated_as_a_checkpoint(self, tmp_path):
        from app.services.cross_encoder.reranker import CrossEncoderReranker

        parent = tmp_path / "cross-encoder"
        (parent / "ms-marco-MiniLM-L-6-v2").mkdir(parents=True)
        (parent / "ms-marco-MiniLM-L-6-v2" / "config.json").write_text("{}")

        assert not CrossEncoderReranker._is_checkpoint_dir(parent)
        assert CrossEncoderReranker._is_checkpoint_dir(
            parent / "ms-marco-MiniLM-L-6-v2"
        )

    def test_a_custom_org_checkpoint_is_looked_up_by_repo_name(self, monkeypatch):
        from app import languages
        from app.services.cross_encoder.reranker import CrossEncoderReranker

        monkeypatch.setenv("CROSS_ENCODER_MODEL_NAME_ES", "some-org/custom-es-reranker")
        languages.get_language_config.cache_clear()
        dirs = CrossEncoderReranker(language="es")._vendored_model_dirs()
        assert any(
            d.name == "custom-es-reranker" and d.parent.name == "cross-encoder"
            for d in dirs
        ), dirs


class TestTokenizerStopwords:
    def test_english_tokenisation_is_unchanged(self):
        from app.services.bm25_scoring.text_builders import tokenize

        assert tokenize("the manager of the kitchen") == ["manager", "kitchen"]

    def test_tokenisation_follows_the_deployment_language(self, monkeypatch):
        monkeypatch.setenv("TARGET_LANGUAGE", "es")
        from app.services.bm25_scoring.text_builders import tokenize

        assert tokenize("el gerente de la cocina") == ["gerente", "cocina"]

    def test_spanish_stopwords_apply_when_passed_explicitly(self):
        # The kwarg stays for scripts/tools that tokenise a known-language corpus.
        from app.services.bm25_scoring.text_builders import tokenize

        assert tokenize("el gerente de la cocina", language="es") == [
            "gerente",
            "cocina",
        ]

    def test_spanish_words_are_kept_when_tokenising_as_english(self):
        from app.services.bm25_scoring.text_builders import tokenize

        # Proof the two sets are not merged: merging would silently change English scores.
        assert "de" in tokenize("el gerente de la cocina")
