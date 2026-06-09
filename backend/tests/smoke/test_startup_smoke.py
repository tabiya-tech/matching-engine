"""Smoke tests for app startup and the /health endpoint.

The gateway health check hits GET /health with x-api-key. If auth breaks
or the endpoint is unregistered, the service gets marked unhealthy.
"""


class TestHealthEndpoint:
    def test_health_with_api_key(self, test_client):
        resp = test_client.get("/health", headers={"x-api-key": "test-key"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_without_api_key_is_rejected(self, test_client):
        resp = test_client.get("/health")
        # FastAPI/Starlette versions differ: missing API key → 401 or 403.
        assert resp.status_code in (401, 403)
