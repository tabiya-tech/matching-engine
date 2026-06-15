"""Verify the OpenAPI schema has the correct endpoints and auth boundaries.

This branch renamed routes (/match_v2 -> /experiments/v2/match, etc.).
If a router registration breaks, the endpoint silently vanishes.
"""

EXPECTED_ENDPOINTS = {
    "/health": "get",
    "/jobs": "get",
    "/match": "post",
    "/experiments/v2/match": "post",
    "/experiments/v3/match": "post",
    "/match_v4": "post",
    "/experiments/v5/match": "post",
}

AUTH_REQUIRED_PATHS = {"/health", "/jobs", "/match"}


class TestOpenAPIEndpoints:
    """All registered endpoints must be present with correct HTTP methods."""

    def test_all_endpoints_registered(self, test_client):
        resp = test_client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]

        for path, method in EXPECTED_ENDPOINTS.items():
            assert path in paths, f"{path} missing from OpenAPI schema"
            assert method in paths[path], f"{method.upper()} not registered on {path}"


class TestAuthBoundaries:
    """/health and /match require x-api-key; experiment endpoints do not."""

    def test_auth_split(self, test_client):
        resp = test_client.get("/openapi.json")
        paths = resp.json()["paths"]

        for path, method in EXPECTED_ENDPOINTS.items():
            operation = paths[path][method]
            has_security = bool(operation.get("security"))

            if path in AUTH_REQUIRED_PATHS:
                assert has_security, (
                    f"{path} should require x-api-key but has no security"
                )
            else:
                assert not has_security, f"{path} should be public but has security"
