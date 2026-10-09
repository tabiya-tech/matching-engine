"""`POST /match` full response: occupations + opportunities + skill-gaps via the Gemini engine.

Runs the v4 engine (Gemini concat cosine -> cross-encoder rerank -> u_hat x p_hat) over BOTH jobs
and occupations using a single shared user embedding, reuses the Node2Vec skill-gap analysis, and
assembles one `MatchResponse`-shaped dict per user. Per-item detail is best-effort from v4 outputs
(see match_v4_formatting).
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from app import observability
from app.artifacts.repository import get_artifacts_repository
from app.clients.gemini_embedding_client import EMBEDDING_DIM
from app.config import (
    FINAL_SCORE_COMBINER,
    LOCATION_HUB_CHAINS_PATH,
    LOCATION_TIER_ENABLED,
    LOCATION_TIER_W_NATIONAL,
    LOCATION_TIER_W_REGIONAL,
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V4_DISABLE_OCCUPATIONS,
    MATCH_V4_OCC_DEMAND_GAMMA,
    MATCH_V4_TOP_K_OCCUPATIONS,
    SKILL_RESCALE_TARGET,
    V4_FULL_COVERAGE_GAMMA,
    V4_FULL_MIN_ESS_SHARE,
    V4_FULL_RANK_DEMOTE,
    V4_FULL_SIM_THRESHOLD,
    V4_FULL_UNPARSED_COVERAGE,
    V4_FULL_WHITENED_GATE,
)
from app.ranking.retrieval import (
    index_by_uuid,
    is_prewhitened,
    stage1_vector,
    user_matches_any_county,
)
from app.ranking.scoring import RankOverrides, enriched_recommendations, skill_detail
from app.ranking.skill_gaps import skill_gaps_for
from app.server_dependencies.model_dependencies import (
    get_concat_whitener,
    get_preference_scorer,
    get_skill_matcher,
    get_skill_scorer,
    get_v4_skill_matcher,
)
from app.services import match_v4_formatting as fmt
from app.services.match_concat_gemini_ce_service import (
    embed_user_unit_vectors,
    run_match_concat_gemini_ce,
)

__all__ = ["run_match_v4_full"]

logger = logging.getLogger(__name__)


def _location_tier_overrides(
    user: dict[str, Any],
    v3_row: dict[str, Any] | None,
    item_index: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Per-uuid location-tier multiplier for a user's job shortlist (urban-pull Part B).

    local=1.0, regional hub=W_REGIONAL, national hub=W_NATIONAL, remote=1.0, off-chain=0.0. Returns ``{}``
    (a no-op) when the feature is disabled or the hub-chain data is unavailable. Runs independent of the
    Phase-2 coverage demotion (no whitening artifact needed)."""
    if not LOCATION_TIER_ENABLED:
        return {}
    hc = get_artifacts_repository().load_hub_chains(LOCATION_HUB_CHAINS_PATH)
    if hc is None:
        return {}
    county = user.get("province") or user.get("city") or ""
    tiers: dict[str, float] = {}
    for r in (v3_row or {}).get("concat_gemini_ce_recommendations") or []:
        if not isinstance(r, dict):
            continue
        uuid = str(r.get("job_uuid") or "")
        if not uuid or uuid in tiers:
            continue
        item = item_index.get(uuid)
        if not item:
            continue
        tiers[uuid] = hc.tier_factor_for_job(
            item,
            county,
            w_regional=LOCATION_TIER_W_REGIONAL,
            w_national=LOCATION_TIER_W_NATIONAL,
        )
    return tiers


def run_match_v4_full(
    users: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    occupations: list[dict[str, Any]],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    final_score_combiner: str | None = None,
    skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
    mongo_timing: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return one MatchResponse-shaped dict per user (occupations + opportunities + skill-gaps).

    The deployment's ``TARGET_LANGUAGE`` selects the stage-2 cross-encoder checkpoint, whose
    passages are skill-label text. Stage-1 retrieval and the per-skill gate are
    language-neutral: every language's labels resolve into the same shared embedding space
    (``ranking/skills``).
    """
    combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
    if combiner not in ("product", "geometric_mean"):
        raise ValueError("final_score_combiner must be 'product' or 'geometric_mean'")
    if not users:
        return []

    # Per-deployment kill-switch: with MATCH_V4_DISABLE_OCCUPATIONS every user gets an empty
    # occupation_recommendations list and NO occupation work runs at all (no stage-1 retrieval, no CE
    # rerank, no Phase-2 overrides). Dropping the corpus here makes every occupation-derived structure
    # below empty; the remaining guards skip the calls that would otherwise run on an empty corpus.
    occupations_enabled = not MATCH_V4_DISABLE_OCCUPATIONS
    if not occupations_enabled:
        occupations = []

    u_norm = embed_user_unit_vectors(users)  # embed users ONCE, reuse for both corpora
    pref_scorer = get_preference_scorer()
    # Per-skill GATE matcher: whitened (default) or — via the kill-switch — the legacy raw matcher.
    matcher = get_v4_skill_matcher() if V4_FULL_WHITENED_GATE else get_skill_matcher()

    job_index = index_by_uuid(jobs)
    occ_index = index_by_uuid(occupations)

    # Phase-2 (V4_FULL_RANK_DEMOTE) ranking inputs. Snapshot the concat embeddings NOW — stage-1
    # retrieval pops job_embedding off these dicts in place — and whiten the user vectors once.
    # When the toggle is off these stay empty and ranking is the Phase-1 behaviour.
    # Safety: if the toggle is on but the concat-whitening artifact is unavailable/incompatible
    # (target==0), DON'T half-apply Phase 2 (raw p_hat x coverage is count-biased — see notes); fall
    # back to pure Phase-1 (annotation-only, ranking unchanged) and log loudly.
    demote_active = V4_FULL_RANK_DEMOTE and get_concat_whitener().rescale_target() > 0
    if V4_FULL_RANK_DEMOTE and not demote_active:
        logger.error(
            "V4_FULL_RANK_DEMOTE is on but the concat-whitening artifact is unavailable; "
            "falling back to Phase-1 (no demotion, raw p_hat). Build/ship the artifact to enable Phase 2."
        )
    job_concat: dict[str, tuple] = {}  # uuid -> (stage1_vector, is_already_whitened)
    occ_concat: dict[str, tuple] = {}
    u_white_by_uid: dict[str, np.ndarray] = {}
    rank_overrides: RankOverrides | None = None
    if demote_active:
        whitener = get_concat_whitener()
        rank_overrides = RankOverrides(
            matcher=matcher,
            whitener=whitener,
            whitened_gate=V4_FULL_WHITENED_GATE,
            sim_threshold=V4_FULL_SIM_THRESHOLD,
            unparsed_coverage=V4_FULL_UNPARSED_COVERAGE,
        )
        with observability.stage("preference_scoring", step="phase2_inputs"):
            for j in jobs:
                v = stage1_vector(j, dim=EMBEDDING_DIM)
                if v is not None:
                    # (vector, is_already_whitened) — DB-whitened jobs are consumed directly in Phase-2;
                    # raw vectors are whitened in-process. Same artifact => identical result either way.
                    job_concat[str(j.get("uuid") or "")] = (v, is_prewhitened(j))
            for o in occupations:
                v = stage1_vector(o, dim=EMBEDDING_DIM)
                if v is not None:
                    # occupations are whitened once at cache-load (consumed directly) or raw (whitened
                    # in-process) — the flag set by app.occupations.loader.attach_occupation_embeddings tells which.
                    occ_concat[str(o.get("uuid") or "")] = (v, is_prewhitened(o))
            u_white = whitener.whiten_rows(u_norm)
            u_white_by_uid = {
                str(u.get("user_id") or ""): u_white[i] for i, u in enumerate(users)
            }

    job_v3 = run_match_concat_gemini_ce(
        users,
        jobs,
        retrieve_top_k=retrieve_top_k,
        final_top_k=final_top_k,
        mongo_timing=mongo_timing,
        user_unit_vectors=u_norm,
        # Urban-pull: weight the stage-1 cosine by location tier so relevant local jobs survive the
        # retrieve_top_k cutoff (occupations keep their own county scoping, so not applied there).
        apply_location_tier=LOCATION_TIER_ENABLED,
    )
    # Occupations are flattened into 4 identical-embedding county-rows per code (the fixed sample
    # counties Kilifi/Kitui/Mombasa/Nairobi). The per-user location filter (below) keeps only the
    # user's own county row per code, so the shortlist/CE slate must be wide enough that ~top_k
    # distinct codes survive: size at top_k * 4 (counties) * 2 (buffer). De-dup by code remains a
    # safety net.
    occ_breadth = max(retrieve_top_k, final_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
    occ_v3 = (
        run_match_concat_gemini_ce(
            users,
            occupations,
            retrieve_top_k=occ_breadth,
            final_top_k=occ_breadth,
            user_unit_vectors=u_norm,
            corpus="occupations",
        )
        if occupations_enabled
        else []
    )
    job_v3_by_uid = {str(r.get("user_id") or ""): r for r in job_v3}
    occ_v3_by_uid = {str(r.get("user_id") or ""): r for r in occ_v3}

    # Available occupation counties (Kilifi/Kitui/Mombasa/Nairobi). Safety net: if a user's province
    # matches none of them, fall back to a random available county so occupations still return.
    occ_counties = sorted(
        {str(o.get("province")) for o in occupations if o.get("province")}
    )

    def _skill_detail(user, item):
        return skill_detail(matcher, user, item, whitened_gate=V4_FULL_WHITENED_GATE)

    out: list[dict[str, Any]] = []
    cov_gamma = V4_FULL_COVERAGE_GAMMA if demote_active else 0.0
    for user in users:
        uid = str(user.get("user_id") or "")

        # Opportunities. Jobs keep the existing location scoping (Mongo prefilter via
        # JobsRepository.find_active(users)). Instead of a hard python location filter, urban-pull
        # applies a per-user SOFT location tier (local=1.0 > regional hub > national hub; off-chain=0)
        # as a final_score multiplier — local jobs preferred, hub jobs surface when better/needed, and
        # off-chain jobs (e.g. another batch user's locations) are dropped. Always-on (independent of
        # the Phase-2 coverage demotion); {} no-op when LOCATION_TIER_ENABLED is off.
        with observability.stage("preference_scoring", corpus="jobs"):
            # Phase-2 ranking overrides (whitened+rescaled p_hat + coverage demotion); empty dicts when
            # the toggle is off, in which case enriched_recommendations falls back to the stage-1 concat
            # cosine p_hat (itself whitened, just unrescaled) with no demotion (Phase 1).
            job_p, job_cov, job_det = ({}, {}, {})
            if rank_overrides is not None:
                job_p, job_cov, job_det = rank_overrides.compute(
                    user,
                    job_v3_by_uid.get(uid),
                    job_index,
                    job_concat,
                    u_white_by_uid.get(uid),
                )
            job_tiers = _location_tier_overrides(
                user, job_v3_by_uid.get(uid), job_index
            )
            job_recs = enriched_recommendations(
                user,
                job_v3_by_uid.get(uid),
                job_index,
                pref_scorer,
                combiner,
                location_filter=False,
                p_hat_by_uuid=job_p,
                coverage_by_uuid=job_cov,
                coverage_gamma=cov_gamma,
                location_tier_by_uuid=job_tiers,
            )
        opportunities: list[dict[str, Any]] = []
        with observability.stage("formatting", corpus="jobs"):
            for rec in job_recs:
                item = job_index.get(str(rec.get("job_uuid") or ""))
                if not item:
                    continue
                per, ess_ids = job_det.get(
                    str(rec.get("job_uuid") or "")
                ) or _skill_detail(user, item)
                opportunities.append(
                    fmt.build_opportunity_row(
                        rec,
                        item,
                        per,
                        ess_ids,
                        rank=len(opportunities) + 1,
                        sim_threshold=V4_FULL_SIM_THRESHOLD,
                        min_ess_share=V4_FULL_MIN_ESS_SHARE,
                    )
                )

        # Occupations: filter to the user's county; if the user's province matches no occupation
        # county, fall back to a random available county (location filter only — the user's real
        # preferences still drive u_hat). Then dedupe by code, keep best-ranked, take top-k.
        # Skipped entirely (empty list) when MATCH_V4_DISABLE_OCCUPATIONS is set.
        occupations_out: list[dict[str, Any]] = []
        if occupations_enabled:
            with observability.stage("preference_scoring", corpus="occupations"):
                occ_p, occ_cov, occ_det = ({}, {}, {})
                if rank_overrides is not None:
                    occ_p, occ_cov, occ_det = rank_overrides.compute(
                        user,
                        occ_v3_by_uid.get(uid),
                        occ_index,
                        occ_concat,
                        u_white_by_uid.get(uid),
                    )
                loc_user = None
                if occ_counties and not user_matches_any_county(user, occ_counties):
                    fallback = random.choice(occ_counties)
                    loc_user = {
                        "city": fallback,
                        "province": fallback,
                        "location": fallback,
                    }
                    logger.warning(
                        "User %r province=%r matches no occupation county %s; using random fallback county %r.",
                        uid,
                        user.get("province"),
                        occ_counties,
                        fallback,
                    )
                occ_recs = enriched_recommendations(
                    user,
                    occ_v3_by_uid.get(uid),
                    occ_index,
                    pref_scorer,
                    combiner,
                    location_user=loc_user,
                    include_demand=True,
                    demand_gamma=MATCH_V4_OCC_DEMAND_GAMMA,
                    p_hat_by_uuid=occ_p,
                    coverage_by_uuid=occ_cov,
                    coverage_gamma=cov_gamma,
                )
            with observability.stage("formatting", corpus="occupations"):
                seen_codes: set = set()
                for rec in occ_recs:
                    item = occ_index.get(str(rec.get("job_uuid") or ""))
                    if not item:
                        continue
                    code = str(item.get("originUuid") or item.get("uuid") or "")
                    if not code or code in seen_codes:
                        continue
                    seen_codes.add(code)
                    per, ess_ids = occ_det.get(
                        str(rec.get("job_uuid") or "")
                    ) or _skill_detail(user, item)
                    occupations_out.append(
                        fmt.build_occupation_row(
                            rec,
                            item,
                            per,
                            ess_ids,
                            rank=len(occupations_out) + 1,
                            sim_threshold=V4_FULL_SIM_THRESHOLD,
                            min_ess_share=V4_FULL_MIN_ESS_SHARE,
                        )
                    )
                    if len(occupations_out) >= MATCH_V4_TOP_K_OCCUPATIONS:
                        break

        with observability.stage("skill_gaps"):
            skill_gaps = skill_gaps_for(
                user,
                jobs,
                skill_gap_top_k,
                scorer=get_skill_scorer(),
                rescale_target=SKILL_RESCALE_TARGET,
            )

        out.append(
            {
                "user_id": uid,
                "occupation_recommendations": occupations_out,
                "opportunity_recommendations": opportunities,
                "skill_gap_recommendations": skill_gaps,
            }
        )

    return out
