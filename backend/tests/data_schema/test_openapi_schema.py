"""Verify the OpenAPI schema exposes exactly the served API.

POST /match (the v4 engine) is the only matching endpoint; the retired v2/v3/v5/legacy
handlers are not registered. If a retired route is re-registered, or a registration breaks,
these tests fail.
"""

EXPECTED_ENDPOINTS = {
    "/health": "get",
    "/jobs": "get",
    "/jobs/stats": "get",
    "/match": "post",
}


def _openapi(test_client):
    resp = test_client.get("/openapi.json")
    assert resp.status_code == 200
    return resp.json()


class TestOpenAPIEndpoints:
    def test_exactly_expected_endpoints_registered(self, test_client):
        paths = _openapi(test_client)["paths"]
        assert set(paths) == set(EXPECTED_ENDPOINTS), (
            f"unexpected OpenAPI paths: {sorted(set(paths) ^ set(EXPECTED_ENDPOINTS))}"
        )
        for path, method in EXPECTED_ENDPOINTS.items():
            assert method in paths[path], f"{method.upper()} not registered on {path}"


class TestAuthBoundaries:
    """Every endpoint requires x-api-key."""

    def test_all_endpoints_require_api_key(self, test_client):
        paths = _openapi(test_client)["paths"]
        for path, method in EXPECTED_ENDPOINTS.items():
            assert paths[path][method].get("security"), (
                f"{path} should require x-api-key but has no security"
            )
