"""Embedding-only skill overlap: cosine similarity between user and job skill vectors.

**Production “legacy cosine” skill model** — used by ``run_cosine_matching`` and
``hybrid_scoring.run_bm25_cosine_hybrid`` for column 2. Not the full-rule
``SkillScorer`` stack (no location, skill groups, gap penalties on ``U``).

Loads the same ``.pt`` matrix, ``skill_to_row.json``, and ``skills.csv`` as the main
app (via ``app.config``). Resolves labels to internal IDs, then for each **job**
skill (essential ∪ optional, de-duplicated) takes the **row-wise maximum** cosine
against all **user** skills and ranks jobs by the **mean** of those maxima.

No location, skill groups, gap penalties, or feasibility — only vectors + cosine.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import csv

import numpy as np
import torch

from app.config import EMBEDDING_MODEL_PATH, SKILLS_CSV_PATH, SKILL_TO_ROW_PATH

logger = logging.getLogger(__name__)


def _canon(label: str) -> str:
    return " ".join(label.strip().lower().split())


class CosineSkillMatcher:
    """Row-normalised embedding matrix + label-based skill lookup."""

    def __init__(self) -> None:
        self.MODEL_PATH = Path(EMBEDDING_MODEL_PATH)
        self.MAPPING_PATH = Path(SKILL_TO_ROW_PATH)
        self.SKILLS_CSV_PATH = Path(SKILLS_CSV_PATH)

        state = torch.load(self.MODEL_PATH, map_location="cpu")
        W = state["state_dict"]["embedding.weight"].numpy()
        if W.dtype != np.float32:
            W = W.astype(np.float32)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self.W = W / np.where(norms > 0, norms, 1.0)

        with open(self.MAPPING_PATH, "r", encoding="utf-8") as f:
            self.skill_to_row = json.load(f)

        self._embedding_ids = set(self.skill_to_row.keys())
        self.skill_labels: Dict[str, str] = {}
        self._preferred_to_id: Dict[str, str] = {}
        self._altlabel_to_id: Dict[str, str] = {}
        self._preferred_collisions = 0

        try:
            with open(self.SKILLS_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("ID")
                    label = row.get("PREFERREDLABEL") or ""
                    if not sid or sid not in self._embedding_ids:
                        continue
                    if label:
                        self.skill_labels[str(sid)] = label
                        canon = _canon(label)
                        if canon and canon not in self._preferred_to_id:
                            self._preferred_to_id[canon] = str(sid)
                        elif canon and self._preferred_to_id.get(canon) != str(sid):
                            self._preferred_collisions += 1
                    for alt in (row.get("ALTLABELS") or "").split("\n"):
                        canon = _canon(alt)
                        if not canon or canon in self._preferred_to_id:
                            continue
                        self._altlabel_to_id.setdefault(canon, str(sid))
        except FileNotFoundError:
            self.skill_labels = {}

        self._missed_labels: Counter[str] = Counter()

        logger.info(
            "CosineSkillMatcher: %d embedding rows (model=%s, dim=%d); "
            "%d preferredLabel keys, %d altLabel keys (preferred-collisions: %d)",
            len(self._embedding_ids),
            state.get("model_name") or self.MODEL_PATH.name,
            self.W.shape[1],
            len(self._preferred_to_id),
            len(self._altlabel_to_id),
            self._preferred_collisions,
        )

    def _resolve_label(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        canon = _canon(label)
        if not canon:
            return None
        sid = self._preferred_to_id.get(canon)
        if sid is not None:
            return sid
        sid = self._altlabel_to_id.get(canon)
        if sid is not None:
            return sid
        self._missed_labels[label] += 1
        return None

    def get_resolution_stats(self) -> Dict[str, Any]:
        return {
            "total_misses": sum(self._missed_labels.values()),
            "distinct_missed_labels": len(self._missed_labels),
            "top_misses": self._missed_labels.most_common(20),
        }

    def _rows_with_ids(
        self, skill_ids: Sequence[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Return row matrix and list of IDs that exist in ``skill_to_row`` (stable order)."""
        valid_ids: List[str] = []
        for s in skill_ids:
            sid = str(s)
            if sid in self.skill_to_row:
                valid_ids.append(sid)
        if not valid_ids:
            return np.empty((0, self.W.shape[1]), dtype=np.float32), []
        idx = np.array([self.skill_to_row[s] for s in valid_ids], dtype=np.int64)
        return self.W[idx, :].astype(np.float32, copy=False), valid_ids

    @staticmethod
    def _ordered_unique_skill_ids(pairs: Iterable[Tuple[str, str]]) -> List[str]:
        """pairs: (internal_id, label) — preserve first-seen order, de-dupe by id."""
        seen: set[str] = set()
        out: List[str] = []
        for sid, _ in pairs:
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
        return out

    def _user_skill_pairs(self, user_profile: Dict[str, Any]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for s in user_profile.get("resolved_skills") or []:
            if not isinstance(s, dict):
                continue
            lab = s.get("label")
            sid = self._resolve_label(str(lab) if lab else None)
            if sid is not None:
                out.append((sid, str(lab)))
        if out:
            return out
        for s in (user_profile.get("skills_vector") or {}).get("top_skills") or []:
            if not isinstance(s, dict):
                continue
            lab = s.get("preferredLabel") or s.get("label")
            sid = self._resolve_label(str(lab) if lab else None)
            if sid is not None:
                out.append((sid, str(lab)))
        return out

    def _job_skill_pairs(self, job_posting: Dict[str, Any]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []

        def _consume(items: Any) -> None:
            for s in items or []:
                if not isinstance(s, dict):
                    continue
                lab = s.get("label")
                sid = self._resolve_label(str(lab) if lab else None)
                if sid is not None:
                    out.append((sid, str(lab)))

        _consume(job_posting.get("essential_skills"))
        _consume(job_posting.get("optional_skills"))
        return out

    def score_pair(self, user_profile: Dict[str, Any], job_posting: Dict[str, Any]) -> Dict[str, Any]:
        """Cosine similarity only: mean over job skills of (max over user skills).

        Returns ``mean_best_cosine`` in ``[0, 1]`` (zero if no overlap to score).
        """
        user_pairs = self._user_skill_pairs(user_profile)
        job_pairs = self._job_skill_pairs(job_posting)
        user_ids = self._ordered_unique_skill_ids(user_pairs)
        job_ids = self._ordered_unique_skill_ids(job_pairs)
        user_labels = {sid: lab for sid, lab in user_pairs}
        job_labels = {sid: lab for sid, lab in job_pairs}

        u_mat, u_valid = self._rows_with_ids(user_ids)
        j_mat, j_valid = self._rows_with_ids(job_ids)

        if j_mat.size == 0 or u_mat.size == 0:
            return {
                "mean_best_cosine": 0.0,
                "min_best_cosine": 0.0,
                "n_user_skills_embedded": int(u_mat.shape[0]),
                "n_job_skills_embedded": int(j_mat.shape[0]),
                "per_job_skill": [],
            }

        sims = j_mat @ u_mat.T
        np.maximum(sims, 0.0, out=sims)
        argmax = sims.argmax(axis=1)
        rowmax = sims.max(axis=1)

        per: List[Dict[str, Any]] = []
        for i, jid in enumerate(j_valid):
            ui = int(argmax[i])
            uid = u_valid[ui]
            per.append({
                "job_skill_id": jid,
                "job_skill_label": self.skill_labels.get(jid) or job_labels.get(jid),
                "best_user_skill_id": uid,
                "best_user_skill_label": self.skill_labels.get(uid) or user_labels.get(uid),
                "cosine_similarity": round(float(rowmax[i]), 4),
            })

        return {
            "mean_best_cosine": round(float(rowmax.mean()), 4),
            "min_best_cosine": round(float(rowmax.min()), 4),
            "n_user_skills_embedded": int(u_mat.shape[0]),
            "n_job_skills_embedded": int(j_mat.shape[0]),
            "per_job_skill": per,
        }

    def resolved_user_skill_labels_ordered(self, user_profile: Dict[str, Any]) -> List[str]:
        """Display labels for embedded user skills (order preserved, de-duplicated by id)."""
        pairs = self._user_skill_pairs(user_profile)
        seen: set[str] = set()
        out: List[str] = []
        for sid, lab in pairs:
            if sid in seen:
                continue
            seen.add(sid)
            txt = lab or self.skill_labels.get(sid) or sid
            out.append(txt)
        return out

    def rank_jobs(
        self,
        user_profile: Dict[str, Any],
        jobs: Sequence[Dict[str, Any]],
        *,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Score every job and return the top ``top_k`` by ``mean_best_cosine`` desc."""
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for job in jobs:
            detail = self.score_pair(user_profile, job)
            key = (detail["mean_best_cosine"], detail["min_best_cosine"])
            scored.append((key, {"job": job, "score": detail}))

        scored.sort(key=lambda x: (-x[0][0], -x[0][1]))
        out: List[Dict[str, Any]] = []
        for rank, (_, row) in enumerate(scored[: max(0, top_k)], 1):
            j = row["job"]
            out.append({
                "rank": rank,
                "job_uuid": j.get("uuid") or j.get("_id"),
                "job_title": j.get("opportunity_title"),
                "employer": j.get("employer"),
                "location": j.get("location"),
                **row["score"],
            })
        return out
