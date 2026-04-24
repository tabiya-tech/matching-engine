# src/demand_scorer.py
from app.config import DEMAND_SCORE_MAPPING

class DemandScorer:
    def __init__(self):
        self.mapping = DEMAND_SCORE_MAPPING

    def calculate_score(self, job_posting: dict) -> dict:
        """
        Computes the S_demand score.
        
        Logic:
        1. Checks job_posting['attributes']['expected_demand'] -> Maps text to float.
        2. Fallback to 0.5 if missing.
        """
        
        # 1. Safely extract the attributes dictionary
        attributes = job_posting.get("attributes", {})
        
        # 2. Get the specific key
        demand_label = attributes.get("expected_demand")
        
        # 3. Map label to value
        if demand_label and demand_label in self.mapping:
            return {
                "score": self.mapping[demand_label],
                "label": demand_label,
                "present": True,
            }

        # 4. Fallback / Default
        return {
            "score": 0.5,
            "label": demand_label,
            "present": False,
        }