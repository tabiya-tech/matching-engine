"""Prep the full njila supply-side dataset for the SHP matching service.

Inputs — either JSON exports on disk or the same data live in MongoDB (database
``compass-application-njila`` in Compass), collection names:

  - explore_experiences_director_state  (was *.explore_experiences_director_state.json)
  - user_preferences
  - job_preferences
  - plain_personal_data
  - programme_skills

Mongo usage (from ``backend/`` with ``backend/.env``): connection URI is
``NJILA_PREP_SOURCE_DB_URI`` when set, otherwise ``MONGO_URL``::

  NJILA_PREP_SOURCE=mongo python prep_njila_for_matching.py
  python prep_njila_for_matching.py --mongo

File usage (default): set ``NJILA_JSON_INPUT_DIR`` to the folder of the five
``compass-application-njila.*.json`` files, or place them under
``data/njila_supply_side/`` at the repo root.

Stitching pipeline per session:
  1. Aggregate top_skills + remaining_skills across explored_experiences
     (Compass SkillsExtractor formula: 0.6·avg_score + 0.25·freq_norm + 0.15·top_ratio).
  2. Look up user_id via user_preferences.sessions.
  3. Pull province + programme_name from plain_personal_data.
  4. Merge programme_skills (curriculum-derived) into top_skills, deduped by
     originUUID, at proficiency=0.40 — they are "exposure" not demonstrated experience.
  5. Pull job_preferences (BWS importance scores) and remap to matcher API attribute
     names (financial_importance → earnings_per_month, etc.).
  6. Build a MatchRequest payload with city="Unknown" (not in any source).

Output: ``njila_match_input.jsonl`` (default ``data/njila_match_input.jsonl`` under repo root;
override with ``NJILA_OUTPUT_JSONL``).
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
# The configured Mongo URI lives in backend/.env (see module docstring).
# Loading data/.env (parent of this file) silently fell back to localhost:27017.
load_dotenv(REPO_ROOT / "backend" / ".env")
ROOT = Path(os.getenv("NJILA_JSON_INPUT_DIR", str(REPO_ROOT / "data" / "njila_supply_side")))
OUT = Path(os.getenv("NJILA_OUTPUT_JSONL", str(REPO_ROOT / "data" / "njila_match_input.jsonl")))

# MongoDB (same collections as the JSON filenames; ``motor`` installs ``pymongo``)
def _mongo_uri() -> str:
    dedicated = (os.getenv("NJILA_PREP_SOURCE_DB_URI") or "").strip()
    if dedicated:
        return dedicated
    return os.getenv("MONGO_URL", "mongodb://localhost:27017")


def _mongo_client():
    from pymongo import MongoClient

    timeout_ms = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "30000"))
    return MongoClient(_mongo_uri(), serverSelectionTimeoutMS=timeout_ms)


def _mongo_db():
    db_name = (
        os.getenv("NJILA_MONGO_DB")
        or os.getenv("MONGO_DB_NAME")
        or "compass-application-njila"
    )
    return _mongo_client()[db_name]


def _collection_names():
    return {
        "exp": os.getenv("NJILA_COLL_EXPLORE", "explore_experiences_director_state"),
        "upref": os.getenv("NJILA_COLL_USER_PREFERENCES", "user_preferences"),
        "jpref": os.getenv("NJILA_COLL_JOB_PREFERENCES", "job_preferences"),
        "pers": os.getenv("NJILA_COLL_PLAIN_PERSONAL", "plain_personal_data"),
        "prog": os.getenv("NJILA_COLL_PROGRAMME_SKILLS", "programme_skills"),
    }


def load_inputs_from_mongo():
    db = _mongo_db()
    names = _collection_names()
    out = {}
    for key, coll in names.items():
        out[key] = list(db[coll].find({}))
    return (
        out["exp"],
        out["upref"],
        out["jpref"],
        out["pers"],
        out["prog"],
    )


def load_inputs_from_files():
    def load_json_file(local_name: str):
        path = ROOT / local_name
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return (
        load_json_file("compass-application-njila.explore_experiences_director_state.json"),
        load_json_file("compass-application-njila.user_preferences.json"),
        load_json_file("compass-application-njila.job_preferences.json"),
        load_json_file("compass-application-njila.plain_personal_data.json"),
        load_json_file("compass-application-njila.programme_skills.json"),
    )

# Compass SkillsExtractor weights
W_AVG = 0.60
W_FREQ = 0.25
W_TOP_RATIO = 0.15
DEFAULT_SCORE_TOP = 0.5
DEFAULT_SCORE_REMAINING = 0.3
PROGRAMME_SKILL_PROFICIENCY = 0.40  # exposure-tier; below CV-derived skills

# BWS importance → matcher API attribute mapping (semantic best fit; documented above)
PREF_MAP = {
    "earnings_per_month": "financial_importance",
    "task_content": "task_preference_importance",
    "physical_demand": "work_environment_importance",
    "work_flexibility": "work_life_balance_importance",
    "career_growth": "career_advancement_importance",
    "social_meaning": "social_impact_importance",
    # social_interaction: no clean BWS source → default 0.5
    # job_security_importance: dropped, no API target
}

# Province normalisation
PROVINCE_NORMALISE = {
    "North - Western": "North Western",
    "North-Western": "North Western",
    "north - western": "North Western",
    "northwestern": "North Western",
}


def extract_long(s):
    """Session / user ids from JSON export ($numberLong) or native BSON (int)."""
    if isinstance(s, dict):
        return s.get("$numberLong")
    if isinstance(s, int):
        return str(s)
    return str(s) if s is not None else None


def aggregate_skills(record):
    """Aggregate top_skills + remaining_skills across all explored_experiences."""
    experiences = record.get("explored_experiences") or []
    total_experiences = len(experiences)
    if total_experiences == 0:
        return []

    agg = {}
    for exp in experiences:
        exp_uuid = exp.get("uuid") or ""
        for src_field, is_top in (("top_skills", True), ("remaining_skills", False)):
            for entry in exp.get(src_field) or []:
                # Compass stores [rank, skill_dict] pairs; sometimes plain dicts
                if isinstance(entry, list) and len(entry) >= 2 and isinstance(entry[1], dict):
                    skill = entry[1]
                elif isinstance(entry, dict):
                    skill = entry
                else:
                    continue

                origin_uuid = skill.get("originUUID") or skill.get("UUID")
                if not origin_uuid:
                    continue

                a = agg.setdefault(origin_uuid, {
                    "originUUID": origin_uuid,
                    "preferredLabel": skill.get("preferredLabel") or "",
                    "scores": [],
                    "frequency": 0,
                    "from_top_skills": 0,
                    "from_remaining_skills": 0,
                    "source_experiences": set(),
                    "source": "experience",
                })
                a["scores"].append(skill.get("score") if skill.get("score") is not None
                                   else (DEFAULT_SCORE_TOP if is_top else DEFAULT_SCORE_REMAINING))
                a["frequency"] += 1
                if is_top:
                    a["from_top_skills"] += 1
                else:
                    a["from_remaining_skills"] += 1
                if exp_uuid:
                    a["source_experiences"].add(exp_uuid)

    out = []
    for a in agg.values():
        avg_score = sum(a["scores"]) / len(a["scores"]) if a["scores"] else 0.0
        freq_norm = min(a["frequency"] / total_experiences, 1.0)
        top_ratio = a["from_top_skills"] / a["frequency"] if a["frequency"] else 0.0
        proficiency = round(W_AVG * avg_score + W_FREQ * freq_norm + W_TOP_RATIO * top_ratio, 4)
        out.append({
            "preferredLabel": a["preferredLabel"],
            "originUUID": a["originUUID"],
            "proficiency": proficiency,
            "source": "experience",
        })
    return out


def merge_programme_skills(experience_skills, programme_record):
    """Merge programme curriculum skills into experience-derived skills.

    Dedupe by originUUID. Programme-only skills get proficiency=0.40 (exposure).
    Skills already present from experiences keep their CV-derived proficiency.
    """
    if not programme_record:
        return experience_skills, 0

    have = {s["originUUID"] for s in experience_skills}
    added = 0
    out = list(experience_skills)
    for skill in programme_record.get("skills") or []:
        origin = skill.get("originUUID") or skill.get("UUID")
        if not origin or origin in have:
            continue
        out.append({
            "preferredLabel": skill.get("preferredLabel") or "",
            "originUUID": origin,
            "proficiency": PROGRAMME_SKILL_PROFICIENCY,
            "source": "programme",
        })
        have.add(origin)
        added += 1
    # sort by proficiency desc
    out.sort(key=lambda x: x["proficiency"], reverse=True)
    return out, added


def normalise_province(p):
    if not p:
        return "Unknown"
    p = p.strip()
    norm = PROVINCE_NORMALISE.get(p)
    if norm:
        return norm
    # If it doesn't look like a real province name (e.g. "3 km off..."), bucket as Unknown
    if any(c.isdigit() for c in p) or len(p) > 30:
        return "Unknown"
    return p


def map_preferences(jp_record):
    """Map BWS importance fields → matcher API attribute fields."""
    out = {}
    for api_attr, bws_attr in PREF_MAP.items():
        v = jp_record.get(bws_attr) if jp_record else None
        try:
            out[api_attr] = float(v) if v is not None else 0.5
        except (TypeError, ValueError):
            out[api_attr] = 0.5
    out["social_interaction"] = 0.5  # no BWS source
    return out


NEUTRAL_PREFS = {
    "earnings_per_month": 0.5, "task_content": 0.5, "physical_demand": 0.5,
    "work_flexibility": 0.5, "social_interaction": 0.5, "career_growth": 0.5,
    "social_meaning": 0.5,
}


def main():
    parser = argparse.ArgumentParser(description="Build njila_match_input.jsonl for the matcher.")
    parser.add_argument(
        "--mongo",
        action="store_true",
        help="Load from MongoDB (compass-application-njila collections). Same as NJILA_PREP_SOURCE=mongo.",
    )
    args = parser.parse_args()

    use_mongo = args.mongo or (os.getenv("NJILA_PREP_SOURCE", "").strip().lower() in ("mongo", "db", "database"))

    print("Loading collections…")
    if use_mongo:
        exp, upref, jpref, pers, prog = load_inputs_from_mongo()
        uri_src = "NJILA_PREP_SOURCE_DB_URI" if (os.getenv("NJILA_PREP_SOURCE_DB_URI") or "").strip() else "MONGO_URL"
        print(
            f"  (source=MongoDB uri_env={uri_src} "
            f"db={os.getenv('NJILA_MONGO_DB') or os.getenv('MONGO_DB_NAME') or 'compass-application-njila'})"
        )
    else:
        exp, upref, jpref, pers, prog = load_inputs_from_files()
        print(f"  (source=files ROOT={ROOT})")
    print(f"  experiences: {len(exp)}, user_preferences: {len(upref)}, "
          f"job_preferences: {len(jpref)}, personal_data: {len(pers)}, "
          f"programme_skills: {len(prog)}")

    # Build maps
    session_to_user = {}
    for r in upref:
        uid = r.get("user_id")
        if not uid:
            continue
        for s in r.get("sessions", []):
            sid = extract_long(s)
            if sid:
                session_to_user[sid] = uid

    pers_by_uid = {r.get("user_id"): r for r in pers if r.get("user_id")}
    jpref_by_sid = {extract_long(r.get("session_id")): r for r in jpref}
    prog_by_name = {r.get("programme_name"): r for r in prog if r.get("programme_name")}

    # Counters
    n_no_skills = 0
    n_no_user = 0
    n_no_pers = 0
    n_with_prefs = 0
    n_with_programme_merge = 0
    programme_skills_added = 0
    written = 0
    skill_count_dist = []

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as out:
        for rec in exp:
            sid = extract_long(rec.get("session_id"))
            if not sid:
                continue

            # 1. Aggregate experience skills
            skills = aggregate_skills(rec)
            if not skills:
                n_no_skills += 1
                continue

            # 2. Map session → user
            uid = session_to_user.get(sid)
            if not uid:
                n_no_user += 1
                continue

            # 3. Personal data → province + programme_name
            p = pers_by_uid.get(uid) or {}
            pdata = p.get("data") or {}
            province = normalise_province(pdata.get("province"))
            programme_name = pdata.get("programme_name")
            if not pdata:
                n_no_pers += 1

            # 4. Merge programme skills if available
            n_added = 0
            if programme_name:
                prog_rec = prog_by_name.get(programme_name)
                if prog_rec:
                    skills, n_added = merge_programme_skills(skills, prog_rec)
                    if n_added > 0:
                        n_with_programme_merge += 1
                        programme_skills_added += n_added

            # 5. Preferences
            jp = jpref_by_sid.get(sid)
            preferences = map_preferences(jp) if jp else dict(NEUTRAL_PREFS)
            if jp:
                n_with_prefs += 1

            # 6. Build payload
            uid_obj = rec.get("_id")
            row_oid = uid_obj.get("$oid") if isinstance(uid_obj, dict) else str(uid_obj)
            payload = {
                "user_id": uid,
                "compass_session_state_id": row_oid,
                "session_id": sid,
                "city": "Unknown",
                "province": province,
                "country_of_user": rec.get("country_of_user"),
                "programme_name": programme_name,
                "institution_name": pdata.get("institution_name"),
                "school_year": pdata.get("school_year"),
                "skills_vector": {
                    "top_skills": [
                        {"preferredLabel": s["preferredLabel"],
                         "originUUID": s["originUUID"],
                         "proficiency": s["proficiency"]}
                        for s in skills
                    ],
                },
                "skill_groups_origin_uuids": [],
                "preference_vector": preferences,
                # diagnostics — useful for the dashboard but not consumed by API
                "n_user_skills": len(skills),
                "n_skills_from_programme": sum(1 for s in skills if s.get("source") == "programme"),
                "n_skills_from_experience": sum(1 for s in skills if s.get("source") == "experience"),
                "n_experiences": len(rec.get("explored_experiences") or []),
                "preferences_source": "job_preferences" if jp else "default_neutral",
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            skill_count_dist.append(len(skills))
            written += 1

    print(f"\n=== Summary ===")
    print(f"  Total session-state records:         {len(exp)}")
    print(f"  Skipped (no aggregated skills):      {n_no_skills}")
    print(f"  Skipped (no session->user mapping):  {n_no_user}")
    print(f"  Records without personal_data:       {n_no_pers}")
    print(f"  Records with job_preferences:        {n_with_prefs}")
    print(f"  Records where programme_skills merged: {n_with_programme_merge}")
    print(f"  Total programme skills added:        {programme_skills_added}")
    print(f"  Written to JSONL:                    {written}")

    if skill_count_dist:
        import statistics
        print(f"\nSkill-count distribution (per matcher-ready user, after merge):")
        print(f"  min={min(skill_count_dist)}  max={max(skill_count_dist)}  "
              f"mean={statistics.mean(skill_count_dist):.1f}  median={statistics.median(skill_count_dist)}")
        for lo, hi in [(1,10),(10,20),(20,30),(30,50),(50,80),(80,200)]:
            n = sum(1 for c in skill_count_dist if lo <= c < hi)
            print(f"  [{lo:>2}-{hi:>3}): {n}")
    print(f"\nWritten to: {OUT}")


if __name__ == "__main__":
    main()
