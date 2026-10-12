from typing import Any

from pydantic import BaseModel, Field


class HTTPErrorResponse(BaseModel):
    """The body of every error response the API returns (4xx and 5xx, including 422)."""

    detail: str = Field(description="Human-readable description of the error.")


def error_response_doc(description: str, detail: str) -> dict[str, Any]:
    """OpenAPI ``responses`` entry for an error status: ``HTTPErrorResponse`` plus an example."""
    return {
        "model": HTTPErrorResponse,
        "description": description,
        "content": {"application/json": {"example": {"detail": detail}}},
    }


FORBIDDEN_RESPONSE_DOC = error_response_doc(
    "Forbidden: missing ``x-api-key`` header.", "Not authenticated"
)
INTERNAL_ERROR_RESPONSE_DOC = error_response_doc(
    "Internal Server Error.", "Internal server error: RuntimeError"
)
