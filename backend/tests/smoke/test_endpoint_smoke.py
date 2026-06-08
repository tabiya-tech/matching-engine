"""Smoke tests for match endpoints: payload guards, combiner validation,
ZQF annotation logic, and the unified MatchResponse contract.
"""

from app.routes import _zqf_annotation


MINIMAL_PAYLOAD = [
    {
        "user_id": "smoke-u1",
        "city": "Johannesburg",
        "province": "Gauteng",
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "customer service",
                }
            ]
        },
        "preference_vector": {"earnings_per_month": 0.5},
    }
]


class TestPayloadGuards:
    def test_empty_payload_returns_400(self, test_client):
        resp = test_client.post("/experiments/v2/match", json=[])
        assert resp.status_code == 400


class TestCombinerValidation:
    def test_invalid_combiner_returns_400(self, test_client):
        resp = test_client.post(
            "/match_v4?final_score_combiner=invalid",
            json=MINIMAL_PAYLOAD,
        )
        assert resp.status_code == 400


class TestZqfAnnotation:
    """_zqf_annotation is new on this branch. Pure function tested directly."""

    def test_eligible(self):
        eligible, gap = _zqf_annotation(user_zqf=5, job_zqf_min=3)
        assert eligible is True
        assert gap == 2

    def test_not_eligible(self):
        eligible, gap = _zqf_annotation(user_zqf=2, job_zqf_min=5)
        assert eligible is False
        assert gap == 3

    def test_missing_user_zqf(self):
        assert _zqf_annotation(user_zqf=None, job_zqf_min=3) == (None, None)

    def test_missing_job_zqf(self):
        assert _zqf_annotation(user_zqf=5, job_zqf_min=None) == (None, None)


class TestUnifiedResponseContract:
    """Every match endpoint must return responses with user_id and 3 list keys."""

    MATCH_ENDPOINTS = [
        ("/match", {"headers": {"x-api-key": "test-key"}}),
        ("/experiments/v2/match", {}),
    ]

    def test_response_shape(self, test_client):
        for path, extra_kwargs in self.MATCH_ENDPOINTS:
            resp = test_client.post(path, json=MINIMAL_PAYLOAD, **extra_kwargs)
            assert resp.status_code == 200, f"{path} returned {resp.status_code}"
            data = resp.json()
            assert isinstance(data, list) and len(data) >= 1, (
                f"{path} returned non-list or empty"
            )
            row = data[0]
            assert "user_id" in row, f"{path} missing user_id"
            assert "opportunity_recommendations" in row, (
                f"{path} missing opportunity_recommendations"
            )
            assert "occupation_recommendations" in row, (
                f"{path} missing occupation_recommendations"
            )
            assert "skill_gap_recommendations" in row, (
                f"{path} missing skill_gap_recommendations"
            )
