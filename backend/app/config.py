# src/config.py


# 1. Global Weights (PDF Page 1, Section 1.1)
GLOBAL_WEIGHTS = {
    "w1_skills": 0.40,
    "w2_preference": 0.40,
    "w3_market": 0.20
}

# 2. Demand Score Mapping (PDF Page 2, Top Table)
DEMAND_SCORE_MAPPING = {
    "Very Low Expected Demand": 0.10,
    "Low Expected Demand": 0.25,
    "Moderate Expected Demand": 0.5,
    "High Expected Demand": 0.75,
    "Very High Expected Demand": 1.0
}

# 3. Preference Model Configuration (PDF Page 2 & 4)
PREFERENCE_CONFIG = {
    "base_constant": 0.5, # Fixed at 0.5 (PDF Page 2, Section 2)
    "attributes": {
        "earnings_per_month": {
            "type": "ordered_linear",
            "beta": 0.5, # From Schema & PDF Page 4
            "mapping": {
                "earn_15k": 0.10, 
                "earn_30k": 0.33, 
                "earn_50k": 0.67, 
                "earn_70k": 1.0
            }
        },
        "task_content": {
            "type": "dummy",
            "beta": 0.2, # From Schema & PDF Page 4
            "active_level": "task_creative"
        },
        "physical_demand": {
            "type": "dummy",
            "beta": -0.4, # From Schema & PDF Page 4
            "active_level": "phys_heavy"
        },
        "work_flexibility": {
            "type": "dummy",
            "beta": 0.4, # From Schema & PDF Page 4
            "active_level": "flex_high"
        },
        "social_interaction": {
            "type": "dummy",
            "beta": 0.1, # From Schema & PDF Page 4
            "active_level": "soc_people"
        },
        "career_growth": {
            "type": "dummy",
            "beta": 0.5, # From Schema & PDF Page 4
            "active_level": "growth_high"
        },
        "social_meaning": {
            "type": "dummy",
            "beta": 0.3, # From Schema & PDF Page 4
            "active_level": "mean_high"
        }
    }
}