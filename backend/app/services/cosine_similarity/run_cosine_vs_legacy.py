"""Compare pooled cosine skill match vs legacy ``compute_U_complete`` (full U stack).

Columns:

1. **Pooled cosine** — :meth:`CosineSkillMatcher.score_pair` /
   ``run_cosine_matching`` semantics: essential ∪ optional, row-wise max cosine per
   job skill, mean (no rescaling hook in cosine module; embeddings as loaded).

2. **Legacy utility** — :class:`app.services.skill_score.SkillScorer` /
   ``compute_U_complete``: location, groups, weighted ess/opt/grp blend, rescaling when
   the scorer hydrates targets from artefact metadata, gap penalty, etc.

Produces JSON; pair with ``build_cosine_legacy_compare_dashboard`` for HTML.

Usage::

    cd backend
    python -m app.services.cosine_similarity.run_cosine_vs_legacy \\
        --users …/njila_match_input.resolved.jsonl \\
        --from-mongo \\
        --output …/cosine_vs_legacy_results.json \\
        --top-compare 15

    python -m app.services.cosine_similarity.build_cosine_legacy_compare_dashboard \\
        --input …/cosine_vs_legacy_results.json \\
        --output …/cosine_vs_legacy_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from app.config import EMBEDDING_MODEL_PATH, SKILLS_CSV_PATH, SKILL_TO_ROW_PATH
from app.services.skill_score import SkillScorer

from .run_cosine_matching import load_jobs, _load_users
from .skill_score import CosineSkillMatcher


def _attach_global_ranks(
    rows: List[Dict[str, Any]],
    *,
    sort_key_primary: str,
    sort_key_secondary: str,
    rank_field: str,
) -> List[Dict[str, Any]]:
    keyed = sorted(
        rows,
        key=lambda x: (
            -float(x.get(sort_key_primary) or 0),
            -float(x.get(sort_key_secondary) or 0),
        ),
    )
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(keyed, 1):
        d = dict(row)
        d[rank_field] = i
        out.append(d)
    return out


def _user_profile_for_skill_scorer(user: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the same labeled skills cosine sees via ``resolved_skills`` to SkillScorer.

    SkillScorer only reads ``skills_vector.top_skills`` with ``preferredLabel``;
    :class:`CosineSkillMatcher` prefers ``resolved_skills[].label``.
    """

    resolved = user.get("resolved_skills") or []
    if not resolved:
        return user
    out = dict(user)
    top_skills: List[Dict[str, str]] = []
    for s in resolved:
        if isinstance(s, dict) and s.get("label"):
            top_skills.append({"preferredLabel": str(s["label"]).strip()})
    sv = dict(user.get("skills_vector") or {})
    sv["top_skills"] = top_skills
    out["skills_vector"] = sv
    return out


def _jacc_top_k(uuids_a: List[str], uuids_b: List[str]) -> float:
    aa, bb = set(uuids_a), set(uuids_b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / float(len(aa | bb))


def _legacy_summary(legacy: Dict[str, Any]) -> Dict[str, Any]:
    comps = legacy.get("components") or {}
    return {
        "legacy_U_final": float(legacy.get("U_final") or 0),
        "legacy_is_eligible": bool(legacy.get("is_eligible")),
        "legacy_penalty": float(legacy.get("penalty") or 0),
        "legacy_components": {
            k: comps.get(k) for k in ("loc", "ess", "opt", "grp") if k in comps
        },
        "legacy_sort_secondary": float(comps.get("ess") or 0),
    }


def _one_row_skeleton(
    matcher: CosineSkillMatcher,
    scorer: SkillScorer,
    user_raw: Dict[str, Any],
    user_legacy: Dict[str, Any],
    job: Dict[str, Any],
) -> Dict[str, Any]:
    uni = matcher.score_pair(user_raw, job)
    legacy = scorer.calculate_score(user_legacy, job)
    leg = _legacy_summary(legacy)
    jid = job.get("uuid") or job.get("_id")

    return {
        "job_uuid": str(jid) if jid is not None else "",
        "job_title": job.get("opportunity_title"),
        "employer": job.get("employer"),
        "location": job.get("location"),
        "mean_best_cosine_union_pool": uni["mean_best_cosine"],
        "min_best_cosine_union_pool": uni["min_best_cosine"],
        "n_job_skills_embedded_union_pool": uni["n_job_skills_embedded"],
        "n_user_skills_embedded": uni["n_user_skills_embedded"],
        **leg,
    }


def run_comparison(
    users_path: Path,
    jobs_source: Literal["file", "mongo"],
    jobs_path: Optional[Path],
    output_path: Optional[Path],
    *,
    top_compare: int = 15,
    mongo_filter_by_users: bool = True,
) -> Dict[str, Any]:
    users_raw = _load_users(users_path)
    jobs, mongo_timing = load_jobs(
        jobs_source, jobs_path, users_raw, mongo_filter_by_users
    )

    matcher = CosineSkillMatcher()
    scorer = SkillScorer()

    users_legacy = [_user_profile_for_skill_scorer(u) for u in users_raw]

    print(
        f"[cosine_vs_legacy] {len(users_raw)} users × {len(jobs)} jobs "
        f"(dim={matcher.W.shape[1]})",
        file=sys.stderr,
    )

    results_out: List[Dict[str, Any]] = []

    for user_raw, user_legacy in zip(users_raw, users_legacy):
        labels = matcher.resolved_user_skill_labels_ordered(user_raw)
        skinny: List[Dict[str, Any]] = []
        for job in jobs:
            skinny.append(_one_row_skeleton(matcher, scorer, user_raw, user_legacy, job))

        cos_ranked = _attach_global_ranks(
            skinny,
            sort_key_primary="mean_best_cosine_union_pool",
            sort_key_secondary="min_best_cosine_union_pool",
            rank_field="rank_pooled_cosine",
        )
        leg_ranked = _attach_global_ranks(
            skinny,
            sort_key_primary="legacy_U_final",
            sort_key_secondary="legacy_sort_secondary",
            rank_field="rank_legacy_U",
        )

        jid_to_legacy_rank = {
            str(r["job_uuid"]): int(r["rank_legacy_U"]) for r in leg_ranked
        }
        jid_to_cos_rank = {
            str(r["job_uuid"]): int(r["rank_pooled_cosine"]) for r in cos_ranked
        }

        cos_annotated: List[Dict[str, Any]] = []
        for r in cos_ranked:
            dd = dict(r)
            jid = str(dd["job_uuid"])
            ir = jid_to_legacy_rank.get(jid, 999999)
            dd["rank_legacy_U"] = ir
            dd["delta_rank_cosine_minus_legacy"] = int(dd["rank_pooled_cosine"]) - ir
            cos_annotated.append(dd)

        leg_annotated: List[Dict[str, Any]] = []
        for r in leg_ranked:
            dd = dict(r)
            jid = str(dd["job_uuid"])
            ic = jid_to_cos_rank.get(jid, 999999)
            dd["cross_rank_pooled_cosine"] = ic
            dd["delta_rank_cosine_minus_legacy"] = ic - int(dd["rank_legacy_U"])
            leg_annotated.append(dd)

        top_cosine = cos_annotated[: max(0, top_compare)]
        top_legacy = leg_annotated[: max(0, top_compare)]

        jacc = _jacc_top_k(
            [str(r["job_uuid"]) for r in top_cosine],
            [str(r["job_uuid"]) for r in top_legacy],
        )

        results_out.append({
            "user_id": user_raw.get("user_id"),
            "city": user_raw.get("city"),
            "province": user_raw.get("province"),
            "n_resolved_user_skills": len(labels),
            "resolved_user_skill_labels": labels,
            "overlap_jaccard_top_cosine_vs_legacy": round(jacc, 4),
            "top_pooled_cosine": top_cosine,
            "top_legacy_U": top_legacy,
        })

    config: Dict[str, Any] = {
        "users_path": str(users_path),
        "jobs_source": jobs_source,
        "top_compare": top_compare,
        "mode": "cosine_pooled_union_vs_legacy_skills_match_U",
        "embedding_model_path": str(EMBEDDING_MODEL_PATH),
        "skill_to_row_path": str(SKILL_TO_ROW_PATH),
        "skills_csv_path": str(SKILLS_CSV_PATH),
        "comparison_note": (
            "Left column ranks by pooled cosine (CosineSkillMatcher.score_pair: "
            "essential ∪ optional, mean of row-wise max cosines vs user skills). "
            "Right column ranks by SkillScorer.calculate_score → compute_U_complete "
            "(location + skill groups + weighted ess/opt/grp + gap penalty; embedding "
            "rescaling applies when SKILL_RESCALE_TARGET is set from artefact/env). "
            "Legacy user skills use the same labels as cosine when resolved_skills is "
            "present (mapped to skills_vector.top_skills for SkillScorer)."
        ),
    }
    if jobs_source == "file":
        config["jobs_path"] = str(jobs_path) if jobs_path else None
    else:
        config["mongo_filter_by_users"] = mongo_filter_by_users

    payload: Dict[str, Any] = {
        "config": config,
        "index_stats": {
            "n_jobs": len(jobs),
            "embedding_dim": int(matcher.W.shape[1]),
            "n_embedding_rows": int(matcher.W.shape[0]),
        },
        "n_users": len(users_raw),
        "n_jobs": len(jobs),
        "results": results_out,
    }
    if mongo_timing is not None:
        payload["mongo_timing"] = mongo_timing

    sc_stats = scorer.get_resolution_stats()
    payload["skill_scorer_resolution_stats"] = deepcopy(sc_stats)
    mr_stats = matcher.get_resolution_stats()
    payload["cosine_resolution_stats"] = mr_stats

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[cosine_vs_legacy] wrote {output_path}", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    examples = """\
Examples:
  %(prog)s --users …/njila_match_input.resolved.jsonl \\
      --from-mongo --output …/cosine_vs_legacy_results.json --top-compare 15
"""

    p = argparse.ArgumentParser(
        description=(
            "Compare pooled cosine embeddings vs legacy skills_match U_complete rankings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    p.add_argument("--users", required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jobs", type=Path)
    src.add_argument("--from-mongo", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--top-compare",
        type=int,
        default=15,
        help="Jobs per column in JSON (still scores all jobs globally).",
    )
    p.add_argument("--mongo-all-active", action="store_true")

    args = p.parse_args(argv)

    js: Literal["file", "mongo"] = "mongo" if args.from_mongo else "file"
    jp = None if js == "mongo" else args.jobs

    if js == "file" and jp is not None and not jp.expanduser().is_file():
        print(
            "[cosine_vs_legacy] jobs file not found — use --from-mongo for Mongo.",
            file=sys.stderr,
        )
        return 2

    run_comparison(
        users_path=args.users,
        jobs_source=js,
        jobs_path=jp,
        output_path=args.output,
        top_compare=args.top_compare,
        mongo_filter_by_users=not args.mongo_all_active,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
