# src/preference_scorer.py
from app.config import PREFERENCE_CONFIG


class PreferenceScorer:
    def __init__(self):
        self.config = PREFERENCE_CONFIG
        self.base_constant = self.config["base_constant"]

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
        """S_pref = base_constant + raw_sum * scaling_factor."""

        raw_score_sum = 0.0
        details = []

        user_weights = user_profile.get("preference_vector", {})
        job_attrs = job_posting.get("attributes", {})

        for attr_key, settings in self.config["attributes"].items():
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

        SCALING_FACTOR = 0.2
        scaled_sum = raw_score_sum * SCALING_FACTOR
        final_pref_score = self.base_constant + scaled_sum

        return {
            "score": max(0.0, min(1.0, final_pref_score)),
            "details": details,
        }