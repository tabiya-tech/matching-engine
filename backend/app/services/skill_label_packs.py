"""Cross-language skill label resolution.

Matching resolves skills by **label**: the user profile carries `preferredLabel` strings and
the job posting carries `{id, label}` pairs, and both must land on a row of the embedding
artefact. Labels are the trust anchor rather than UUIDs, which carry modelId-drift risk.

That works across languages because a taxonomy translation describes the *same* skills.
Skill ``ID`` values are per-taxonomy-model and share nothing across locales, but
``UUIDHISTORY``'s **oldest** entry is identical across them (13,896/13,896 skills between
the English model and AR-es). So this module loads every enabled language's label pack and
maps all of them onto one id space — the canonical language's, the one
``skill_to_row.json`` and the embedding artefacts are keyed on.

The consequence worth stating plainly: a Spanish job posting matched against a Spanish user
profile scores through exactly the same vectors as an English one, and neither side has to
say which language it is in. Nothing about the embeddings is language-specific; only the
label text used to reach them is.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.config import taxonomy_pack_paths
from app.languages import CANONICAL_LANGUAGE, enabled_languages

logger = logging.getLogger(__name__)

# Skill DESCRIPTION cells run past the default field size limit on some exports.
csv.field_size_limit(10_000_000)

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def canon(label: str) -> str:
    """Canonical form for label-based resolution: lowercase, whitespace-collapsed."""
    return " ".join(str(label or "").strip().lower().split())


def oldest_uuid(uuid_history: str) -> str:
    """``UUIDHISTORY``'s oldest entry — the locale-stable identity of a taxonomy entity."""
    entries = [u.strip() for u in (uuid_history or "").split("\n") if u.strip()]
    return entries[-1] if entries else ""


class SkillLabelPacks:
    """Label → canonical-id lookups built from one or more languages' taxonomy packs.

    ``embedding_ids`` is the set of ids present in the embedding artefact: labels that would
    resolve outside it are dropped, exactly as the single-language loader did.
    """

    def __init__(
        self,
        embedding_ids: set[str],
        *,
        languages: Iterable[str] | None = None,
        canonical_language: str = CANONICAL_LANGUAGE,
    ):
        self.canonical_language = canonical_language
        self.languages: tuple[str, ...] = tuple(languages or enabled_languages())
        if canonical_language not in self.languages:
            # The canonical pack defines the id space every other pack is joined onto.
            self.languages = (canonical_language, *self.languages)
        self._embedding_ids = embedding_ids

        # canonical_id -> preferredLabel, in the canonical language (display default).
        self.skill_labels: dict[str, str] = {}
        # language -> {canonical_id -> preferredLabel in that language}
        self.labels_by_language: dict[str, dict[str, str]] = {}
        # canonical preferredLabel (any language) -> canonical_id
        self.preferred_to_id: dict[str, str] = {}
        # canonical altLabel (any language) -> canonical_id
        self.altlabel_to_id: dict[str, str] = {}
        # per-locale taxonomy skill id -> canonical_id (for ids arriving on documents)
        self.id_to_canonical: dict[str, str] = {}
        # every ESCO origin / historical UUID seen in any pack -> canonical_id. A
        # drift-tolerant last-resort resolver for callers that only have a UUID.
        self.uuid_to_id: dict[str, str] = {}

        self.preferred_collisions = 0
        self.loaded_languages: list[str] = []
        self.unjoined_by_language: Counter = Counter()

        self._uuid_to_canonical: dict[str, str] = {}
        self._rows_cache: dict[Path, list[dict]] = {}
        self._skills_paths = self._resolve_skills_paths()
        for language in self.languages:
            self._load_language(language)
        self._rows_cache.clear()

        if not self.preferred_to_id:
            logger.error(
                "SkillLabelPacks: NO labels resolved from %d language pack(s) %s — every skill on "
                "every request will fail to resolve, so matched_skills comes back empty and "
                "essential-coverage is 0 for every posting. Check that the packs and the embedding "
                "artefact (skill_to_row.json) are from the same taxonomy release.",
                len(self.loaded_languages),
                ",".join(self.loaded_languages),
            )

        logger.info(
            "SkillLabelPacks: %d language(s) %s | %d preferredLabel keys, %d altLabel keys "
            "(preferred-collisions: %d) | %d non-canonical ids mapped",
            len(self.loaded_languages),
            ",".join(self.loaded_languages),
            len(self.preferred_to_id),
            len(self.altlabel_to_id),
            self.preferred_collisions,
            len(self.id_to_canonical),
        )
        for language, count in self.unjoined_by_language.items():
            logger.warning(
                "SkillLabelPacks: %d skill(s) in the %r pack have no counterpart in the %r "
                "id space and were skipped — labels resolving to them cannot be scored. "
                "The two packs are probably from different taxonomy releases.",
                count,
                language,
                self.canonical_language,
            )

    # ── loading ──────────────────────────────────────────────────────────────

    def _resolve_skills_paths(self) -> dict[str, Path]:
        """Per-language ``skills.csv`` to load, recovering from an unusable pin.

        ``SKILLS_CSV_PATH`` pins *every* language to one file. Pointing it at a non-canonical
        pack therefore loads that pack as the canonical one, and since skill ids are per-locale
        none of them exist in the embedding artefact: the join drops all 13,896 rows, no label
        resolves, and the service keeps serving requests with an empty resolver (every posting
        gets essential-coverage 0). The pin cannot be honoured in that state, so prefer the
        per-language layout and say loudly what happened.
        """
        pinned = {
            lang: Path(taxonomy_pack_paths(lang)["skills"]) for lang in self.languages
        }
        canonical = pinned[self.canonical_language]
        if self._pack_shares_id_space(canonical):
            return pinned

        unpinned = {
            lang: Path(taxonomy_pack_paths(lang, ignore_pins=True)["skills"])
            for lang in self.languages
        }
        if unpinned[self.canonical_language] == canonical:
            return pinned  # not a pin: the layout's own canonical pack is the mismatched one
        if not self._pack_shares_id_space(unpinned[self.canonical_language]):
            logger.error(
                "SkillLabelPacks: neither the pinned canonical pack %s nor %s shares ids with the "
                "embedding artefact; skill resolution will fail for every request.",
                canonical,
                unpinned[self.canonical_language],
            )
            return pinned
        logger.error(
            "SkillLabelPacks: the %r pack is pinned to %s (SKILLS_CSV_PATH / SKILL_GROUPS_CSV_PATH / "
            "SKILL_HIERARCHY_CSV_PATH pin EVERY language to one file), whose skill ids are not in the "
            "embedding artefact — loading it would leave the resolver empty. Ignoring the pin and "
            "using the per-language packs instead. Clear those env vars: the taxonomy now lives under "
            "resources/skill_taxonomy/<lang>/ and is selected by TARGET_LANGUAGE.",
            self.canonical_language,
            canonical,
        )
        return unpinned

    def _pack_shares_id_space(self, path: Path) -> bool:
        """True if any ``ID`` in this pack is a row of the embedding artefact."""
        try:
            rows = self._read_rows_cached(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            return False
        return any(
            str(row.get("ID") or "").strip() in self._embedding_ids for row in rows
        )

    def _read_rows_cached(self, path: Path) -> list[dict]:
        """``_read_rows`` memoised for this instance — the canonical pack is read twice
        (id-space preflight, then loading) and these files are tens of megabytes."""
        rows = self._rows_cache.get(path)
        if rows is None:
            rows = self._read_rows(path)
            self._rows_cache[path] = rows
        return rows

    def _load_language(self, language: str) -> None:
        path = self._skills_paths[language]
        try:
            rows = self._read_rows_cached(path)
        except FileNotFoundError:
            if language == self.canonical_language:
                logger.error(
                    "SkillLabelPacks: canonical skills.csv not found at %s — skill "
                    "resolution will fail for every request",
                    path,
                )
            else:
                logger.warning(
                    "SkillLabelPacks: no %r taxonomy pack at %s — skills submitted in that "
                    "language will not resolve",
                    language,
                    path,
                )
            return

        is_canonical = language == self.canonical_language
        labels: dict[str, str] = {}
        joined = 0
        for row in rows:
            local_id = str(row.get("ID") or "").strip()
            if not local_id:
                continue
            uuid = oldest_uuid(row.get("UUIDHISTORY") or "")

            if is_canonical:
                canonical_id = local_id
                if uuid:
                    self._uuid_to_canonical.setdefault(uuid, canonical_id)
            else:
                canonical_id = self._uuid_to_canonical.get(uuid, "")
                if not canonical_id:
                    self.unjoined_by_language[language] += 1
                    continue
                if local_id != canonical_id:
                    self.id_to_canonical.setdefault(local_id, canonical_id)

            if canonical_id not in self._embedding_ids:
                continue
            joined += 1

            label = (row.get("PREFERREDLABEL") or "").strip()
            if label:
                labels[canonical_id] = label
                if is_canonical:
                    self.skill_labels[canonical_id] = label
                self._register_preferred(canon(label), canonical_id)

            for alt in (row.get("ALTLABELS") or "").split("\n"):
                key = canon(alt)
                if not key or key in self.preferred_to_id:
                    # Never let an altLabel shadow a preferredLabel hit; first writer
                    # wins among altLabels.
                    continue
                self.altlabel_to_id.setdefault(key, canonical_id)

            # Current ESCO origin UUID and every historical UUID → the same canonical id.
            # Locales share most of their UUID history, so this map is consistent across
            # packs rather than competing between them.
            for column in ("ORIGINURI", "UUIDHISTORY"):
                for match in _UUID_RE.findall((row.get(column) or "").lower()):
                    self.uuid_to_id.setdefault(match, canonical_id)

        self.labels_by_language[language] = labels
        self.loaded_languages.append(language)
        log = logger.info if joined else logger.error
        log(
            "SkillLabelPacks: %r pack → %d skills usable (%s)%s",
            language,
            joined,
            path.name if is_canonical else f"joined onto {self.canonical_language}",
            ""
            if joined
            else f" — NOTHING in {path} lands in the embedding id space, so no label submitted "
            f"in {language!r} can be scored; the pack and the embedding artefact are from "
            "different taxonomy releases",
        )

    @staticmethod
    def _read_rows(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _register_preferred(self, key: str, canonical_id: str) -> None:
        if not key:
            return
        existing = self.preferred_to_id.get(key)
        if existing is None:
            self.preferred_to_id[key] = canonical_id
        elif existing != canonical_id:
            # Two skills share a label text (across or within languages). First writer
            # wins, as before; counted so the ambiguity is visible in the logs.
            self.preferred_collisions += 1

    # ── resolution ───────────────────────────────────────────────────────────

    def resolve_label(self, label: str | None) -> str | None:
        """Canonical id for a label in any loaded language, or None."""
        key = canon(label or "")
        if not key:
            return None
        found = self.preferred_to_id.get(key)
        if found is not None:
            return found
        return self.altlabel_to_id.get(key)

    def resolve_id(self, skill_id: str | None) -> str | None:
        """Canonical id for a taxonomy skill id from any language, or None.

        A fallback for documents whose label is missing: an id from a non-canonical
        taxonomy model still identifies the right skill.
        """
        sid = str(skill_id or "").strip()
        if not sid:
            return None
        if sid in self._embedding_ids:
            return sid
        return self.id_to_canonical.get(sid)

    def resolve_uuid(self, uuid: str | None) -> str | None:
        """Canonical id for an ESCO origin/historical UUID, or None."""
        key = str(uuid or "").strip().lower()
        if not key:
            return None
        return self.uuid_to_id.get(key)

    def display_labels(self, language: str | None = None) -> dict[str, str]:
        """canonical_id → preferredLabel in ``language``, falling back to the canonical one."""
        if not language or language == self.canonical_language:
            return self.skill_labels
        pack = self.labels_by_language.get(language)
        if not pack:
            return self.skill_labels
        # A language pack can be missing a skill the canonical one has; fill from canonical
        # so a response never shows a blank label.
        return {**self.skill_labels, **pack}
