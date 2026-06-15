"""Tests for the cursor-paginated GET /jobs browse endpoint and its cursor helpers.

The endpoint reads from the same source/shape as the matched-jobs endpoints
(CORE-418); these tests cover the keyset cursor codec and the HTTP contract
(auth, page shape, invalid-cursor handling).
"""

from unittest.mock import patch

import pytest
from bson import ObjectId

from app.database import (
    InvalidCursor,
    _decode_jobs_cursor,
    _encode_jobs_cursor,
)


class TestCursorCodec:
    def test_round_trip(self):
        oid = ObjectId("69f10d2f3603f583ccfcc26c")
        cursor = _encode_jobs_cursor(oid)
        assert _decode_jobs_cursor(cursor) == oid

    def test_cursor_is_url_safe_and_opaque(self):
        oid = ObjectId("69f10d2f3603f583ccfcc26c")
        cursor = _encode_jobs_cursor(oid)
        # urlsafe base64 has no characters needing percent-encoding.
        assert all(c.isalnum() or c in "-_=" for c in cursor)
        assert str(oid) not in cursor  # not a raw, guessable id

    @pytest.mark.parametrize(
        "bad",
        ["not-base64!!", "", "YWJj", "%%%%"],  # garbage / valid-b64-but-not-an-oid
    )
    def test_malformed_cursor_raises(self, bad):
        with pytest.raises(InvalidCursor):
            _decode_jobs_cursor(bad)


def _fake_page(jobs, next_cursor, has_more, total=None):
    async def _impl(cursor=None, limit=20, **kwargs):
        return jobs, next_cursor, total, {"has_more": has_more, "limit": limit}

    return _impl


class TestJobsEndpoint:
    AUTH = {"headers": {"x-api-key": "test-key"}}

    def test_requires_api_key(self, test_client):
        resp = test_client.get("/jobs")
        assert resp.status_code in (401, 403)

    def test_first_page_shape(self, test_client):
        jobs = [
            {
                "uuid": "job-1",
                "opportunity_title": "Technical Manager",
                "employer": "RTI International",
                "location": "Nairobi Nairobi",
                "url": "https://example.com/job/1",
                # extra fields from build_job_dict_from_ranked must be ignored
                "essential_skills": [{"id": "x", "label": "y"}],
                "job_embedding": [0.1, 0.2],
            }
        ]
        with patch(
            "app.routes.get_jobs_page_with_timing",
            side_effect=_fake_page(jobs, "next-cursor-token", True),
        ):
            resp = test_client.get("/jobs?limit=1", **self.AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["next_cursor"] == "next-cursor-token"
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["uuid"] == "job-1"
        assert item["opportunity_title"] == "Technical Manager"
        # Internal matching fields are not exposed on the browse contract.
        assert "essential_skills" not in item
        assert "job_embedding" not in item

    def test_last_page_has_null_cursor(self, test_client):
        with patch(
            "app.routes.get_jobs_page_with_timing",
            side_effect=_fake_page([], None, False),
        ):
            resp = test_client.get("/jobs", **self.AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["next_cursor"] is None
        assert data["items"] == []

    def test_invalid_cursor_returns_400(self, test_client):
        async def _raise(cursor=None, limit=20, **kwargs):
            raise InvalidCursor("invalid cursor")

        with patch("app.routes.get_jobs_page_with_timing", side_effect=_raise):
            resp = test_client.get("/jobs?cursor=garbage", **self.AUTH)
        assert resp.status_code == 400

    @pytest.mark.parametrize("limit", [0, 101])
    def test_limit_out_of_range_returns_422(self, test_client, limit):
        resp = test_client.get(f"/jobs?limit={limit}", **self.AUTH)
        assert resp.status_code == 422

    def test_total_included_when_requested(self, test_client):
        with patch(
            "app.routes.get_jobs_page_with_timing",
            side_effect=_fake_page([], None, False, total=42),
        ):
            resp = test_client.get("/jobs?include_total=true", **self.AUTH)
        assert resp.status_code == 200
        assert resp.json()["total"] == 42

    def test_filters_forwarded_to_data_layer(self, test_client):
        captured = {}

        async def _impl(cursor=None, limit=20, **kwargs):
            captured.update(kwargs)
            return [], None, None, {"has_more": False, "limit": limit}

        with patch("app.routes.get_jobs_page_with_timing", side_effect=_impl):
            resp = test_client.get(
                "/jobs?search=nurse&category=Health&employment_type=full_time"
                "&location=Lusaka&skills=care&days=30",
                **self.AUTH,
            )
        assert resp.status_code == 200
        assert captured["search"] == "nurse"
        assert captured["category"] == "Health"
        assert captured["employment_type"] == "full_time"
        assert captured["location"] == "Lusaka"
        assert captured["skills"] == "care"
        assert captured["days"] == 30


class TestJobsStatsEndpoint:
    AUTH = {"headers": {"x-api-key": "test-key"}}

    def test_requires_api_key(self, test_client):
        resp = test_client.get("/jobs/stats")
        assert resp.status_code in (401, 403)

    def test_returns_stats(self, test_client):
        from app.schemas import JobsStats

        async def _stats():
            return JobsStats(total=10, sectors=3, platforms=2)

        with patch("app.routes.get_jobs_stats", side_effect=_stats):
            resp = test_client.get("/jobs/stats", **self.AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"total": 10, "sectors": 3, "platforms": 2}
