"""
Success Propensity Scorer (p_hat)

Computes a cold-start hiring-probability proxy for a (seeker, job) pair.
This is the recruiter-side feasibility signal, distinct from the seeker-side
utility proxy (u_hat) produced by PreferenceScorer.

The score is multiplicative, not additive:

    p_hat = G_ij * E_ij^alpha * R_ij^beta * M_ij^gamma

where:
    G_ij  - hard feasibility gate (0 or 1)
    E_ij  - essential-skill coverage (geometric mean of Node2Vec cosine sims)
    R_ij  - recruiter-side readiness (optional skills + skill group recall)
    M_ij  - market opportunity (demand, freshness)

All per-skill similarities are Node2Vec graph-based cosine distances computed
via SimilarityEngine — identical to the existing skill matching pipeline.
"""

import app.config as c


class SuccessPropensityScorer:

    def __init__(self):
        self.demand_mapping = c.DEMAND_SCORE_MAPPING

    # ------------------------------------------------------------------
    # M_ij: market opportunity
    # ------------------------------------------------------------------
    def _market_opportunity(self, job_posting: dict) -> float:
        """Compute market-opportunity signal from demand label.

        Currently uses the same demand-label mapping as the legacy DemandScorer.
        When richer market data become available (tightness, penetration,
        absorptive capacity, freshness), this method is the place to extend.
        """
        attributes = job_posting.get("attributes", {})
        demand_label = attributes.get("expected_demand")
        if demand_label and demand_label in self.demand_mapping:
            return self.demand_mapping[demand_label]
        return 0.5  # neutral default

    # ------------------------------------------------------------------
    # R_ij: recruiter-side readiness
    # ------------------------------------------------------------------
    @staticmethod
    def _recruiter_readiness(
        optional_sim: float,
        skill_group_recall: float,
        has_optional_skills: bool,
        has_skill_groups: bool,
    ) -> float:
        """Combine optional-skill alignment and skill-group recall into a
        single recruiter-readiness signal.

        Both inputs are already in [0, 1].  When a dimension has no data
        (the job lists no optional skills / no skill groups), that dimension
        is excluded rather than counted as zero — otherwise data sparsity
        is penalised as poor fit.  If *neither* dimension has data, R_ij
        returns 1.0 (neutral).
        """
        terms = []
        weights = []
        if has_optional_skills:
            terms.append(optional_sim)
            weights.append(0.6)
        if has_skill_groups:
            terms.append(skill_group_recall)
            weights.append(0.4)
        if not terms:
            return 1.0  # no data → neutral, don't penalise
        return sum(w * t for w, t in zip(weights, terms)) / sum(weights)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def calculate_score(
        self,
        user_profile: dict,
        job_posting: dict,
        feasibility: dict,
    ) -> dict:
        """Compute p_hat from pre-computed feasibility signals.

        Parameters
        ----------
        user_profile : dict
            User profile (currently unused here, but kept for future
            confidence / shrinkage extensions).
        job_posting : dict
            Job or occupation record.
        feasibility : dict
            Output of ``compute_feasibility_signals()`` — contains gate,
            essential fit, optional sim, skill group recall, etc.

        Returns
        -------
        dict with keys:
            p_hat            – float in [0, 1]
            components       – dict of G, E, R, M values
            demand_label     – str, the raw demand label (for display)
        """
        cfg = c.SUCCESS_PROPENSITY_CONFIG
        alpha = cfg["alpha_essential"]
        beta = cfg["beta_readiness"]
        gamma = cfg["gamma_market"]

        # --- components ---
        g_ij = 1.0 if feasibility["gate_passed"] else 0.0
        e_ij = feasibility["essential_fit"]
        r_ij = self._recruiter_readiness(
            feasibility["optional_sim"],
            feasibility["skill_group_recall"],
            has_optional_skills=feasibility.get("has_optional_skills", False),
            has_skill_groups=feasibility.get("has_skill_groups", False),
        )

        attributes = job_posting.get("attributes", {})
        demand_label = attributes.get("expected_demand")
        has_demand = demand_label and demand_label in self.demand_mapping

        if has_demand:
            m_ij = self.demand_mapping[demand_label]
        else:
            m_ij = None

        # --- dynamic exponent redistribution ---
        # When a component's data is absent, redistribute its exponent
        # proportionally to the remaining components so that missing data
        # doesn't drag the score down.
        if m_ij is None:
            remaining = alpha + beta
            total = alpha + beta + gamma
            if remaining > 0:
                alpha = total * (alpha / remaining)
                beta = total * (beta / remaining)
            gamma = 0.0
            m_ij = 1.0

        # Floor at 0.01 so a zero component doesn't annihilate the product
        e_ij = max(e_ij, 0.01)
        r_ij = max(r_ij, 0.01)
        m_ij = max(m_ij, 0.01)

        p_hat = g_ij * (e_ij ** alpha) * (r_ij ** beta) * (m_ij ** gamma)

        return {
            "p_hat": round(p_hat, 4),
            "components": {
                "gate": round(g_ij, 4),
                "essential_fit": round(e_ij, 4),
                "recruiter_readiness": round(r_ij, 4),
                "market_opportunity": round(m_ij, 4),
            },
            "demand_label": demand_label or "Unknown",
        }
