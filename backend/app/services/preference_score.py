import math
import re

# src/preference_scorer.py
from app.config import PREFERENCE_CONFIG, PREFERENCE_LEGACY_SCORE_SCALE, PREFERENCE_SIGMOID_NUMERATOR


class PreferenceScorer:

    @staticmethod
    def detect_bws_score_type(bws_scores: dict) -> str:
        """
        Detects if bws_scores keys are occupation IDs (2 digits) or Work Activity IDs (e.g., '4.A.2.a.4').
        Returns 'occupation_id', 'work_activity_id', or 'unknown'.
        """
        if not bws_scores:
            return "unknown"
        occupation_id_pattern = re.compile(r"^\d{2}$")
        work_activity_id_pattern = re.compile(r"^\d+(\.[A-Za-z0-9]+)+$")
        keys = list(bws_scores.keys())
        if all(occupation_id_pattern.match(k) for k in keys):
            return "occupation_id"
        if any(work_activity_id_pattern.match(k) for k in keys):
            return "work_activity_id"
        return "unknown"
    def __init__(self):
        self.config = PREFERENCE_CONFIG
        self.base_constant = self.config["base_constant"]

        self._enabled_attrs = {
            k: v for k, v in self.config["attributes"].items()
            if v.get("enabled", True)
        }

        # Dynamic sigmoid scaling: sigmoid(max_raw * factor) ≈ 0.98
        # so a perfect match on all enabled attributes reaches ~0.98.
        max_positive_sum = sum(abs(s["beta"]) for s in self._enabled_attrs.values())
        self._sigmoid_factor = (
            PREFERENCE_SIGMOID_NUMERATOR / max_positive_sum if max_positive_sum > 0 else 2.0
        )

    @staticmethod
    def _humanize_label(value: str):
        if value is None:
            return None
        text = str(value)
        prefixes = ("earn_", "task_", "phys_", "flex_", "soc_", "growth_", "mean_")
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):]
                break
        text = text.replace("_", " ").strip()
        return text.title() if text else None

    def calculate_score(self, user_profile: dict, job_posting: dict) -> dict:
        """S_pref = base_constant + raw_sum * scaling_factor. Optionally includes ONET BWS scores for work activities."""

        raw_score_sum = 0.0
        details = []

        user_weights = user_profile.get("preference_vector", {})
        job_attrs = job_posting.get("attributes", {})

        # BWS scores may be at top level or nested inside preference_vector
        bws_scores = (
            user_profile.get("bws_scores")
            or user_weights.get("bws_scores")
            or {}
        )
        top_10_bws = (
            user_profile.get("top_10_bws")
            or user_weights.get("top_10_bws")
            or []
        )
        bws_score_type = self.detect_bws_score_type(bws_scores)

        # Standard preference scoring (only enabled attributes)
        for attr_key, settings in self._enabled_attrs.items():
            beta = settings["beta"]
            user_weight = user_weights.get(attr_key, 0.0)
            job_value_raw = job_attrs.get(attr_key)

            encoded_value = 0.0
            if settings["type"] == "dummy":
                if job_value_raw == settings["active_level"]:
                    encoded_value = 1.0
            elif settings["type"] == "ordered_linear":
                mapping = settings.get("mapping", {})
                encoded_value = mapping.get(job_value_raw, 0.0)

            contribution = beta * user_weight * encoded_value
            raw_score_sum += contribution

            details.append(
                {
                    "attribute": attr_key,
                    "job_value": job_value_raw,
                    "job_value_label": self._humanize_label(job_value_raw),
                    "user_weight": round(float(user_weight), 4),
                    "beta": round(float(beta), 4),
                    "encoded_value": round(float(encoded_value), 4),
                    "contribution": round(float(contribution), 4),
                    "matched": encoded_value > 0,
                }
            )

        # If bws_scores are work activity IDs and occupation has onet_work_activities, add WA matching logic
        wa_contributions = []
        wa_details = []
        if bws_score_type == "work_activity_id":
            # job_posting should have 'onet_work_activities' as a list of dicts with WA_code, WA_Importance, WA_Level
            wa_list = job_posting.get("onet_work_activities", [])
            for wa in wa_list:
                wa_code = wa.get("WA_code")
                wa_importance = float(wa.get("WA_Importance", 0))
                wa_level = float(wa.get("WA_Level", 0))
                user_bws = float(bws_scores.get(wa_code, 0.0))
                # Example logic: contribution = user_bws * (importance/5) * (level/7)
                # (normalize importance and level to [0,1])
                norm_importance = wa_importance / 5.0 if wa_importance else 0.0
                norm_level = wa_level / 7.0 if wa_level else 0.0
                wa_contribution = user_bws * norm_importance * norm_level
                wa_contributions.append(wa_contribution)
                wa_details.append({
                    "wa_code": wa_code,
                    "user_bws": user_bws,
                    "wa_importance": wa_importance,
                    "wa_level": wa_level,
                    "norm_importance": round(norm_importance, 4),
                    "norm_level": round(norm_level, 4),
                    "wa_contribution": round(wa_contribution, 4),
                })
            # Aggregate WA contributions (mean or sum, here sum)
            wa_score_sum = sum(wa_contributions)
            raw_score_sum += wa_score_sum
            # Add WA details to output
            details.append({
                "attribute": "work_activity_bws",
                "wa_details": wa_details,
                "wa_score_sum": round(wa_score_sum, 4),
            })

        scaled_sum_legacy = raw_score_sum * PREFERENCE_LEGACY_SCORE_SCALE

        # --- u_hat: sigmoid-normalised utility in (0, 1) ---
        # Scaling factor is computed at init from enabled attributes so that
        # a perfect match always reaches ~0.98 regardless of how many
        # attributes are active.
        sigmoid_input = raw_score_sum * self._sigmoid_factor
        u_hat = 1.0 / (1.0 + math.exp(-sigmoid_input)) if abs(sigmoid_input) < 500 else (1.0 if sigmoid_input > 0 else 0.0)

        # Legacy score kept for backward compatibility during migration
        legacy_score = max(0.0, min(1.0, self.base_constant + scaled_sum_legacy))

        # Return BWS scores, type, and top_10_bws as part of the details for downstream use
        return {
            "u_hat": round(u_hat, 4),
            "score": legacy_score,
            "details": details,
            "bws_scores": bws_scores,
            "bws_score_type": bws_score_type,
            "top_10_bws": top_10_bws,
        }