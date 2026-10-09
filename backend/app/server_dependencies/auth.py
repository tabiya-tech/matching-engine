"""API-key authentication for the service's routes (the key itself is checked by the API gateway)."""

from fastapi.security import APIKeyHeader

api_key_auth = APIKeyHeader(
    scheme_name="gcp_api_key", name="x-api-key", auto_error=True
)
