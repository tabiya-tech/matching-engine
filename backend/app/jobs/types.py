"""Types for the jobs feature."""

from pydantic import BaseModel, Field


class JobsBrowseQuery(BaseModel):
    """One ``GET /jobs`` page request: keyset cursor, page size and the AND-ed filters."""

    cursor: str | None = Field(
        default=None,
        description="Opaque ``next_cursor`` from the previous page; None for the first page.",
    )
    limit: int = Field(description="Page size.")
    search: str | None = Field(
        default=None, description="Case-insensitive search on the job title."
    )
    category: str | None = Field(
        default=None,
        description="Sector/category (matches category, sector or ISCO group).",
    )
    employment_type: str | None = Field(
        default=None, description="Employment type (exact match)."
    )
    location: str | None = Field(
        default=None, description="Case-insensitive filter on city/county/province."
    )
    skills: str | None = Field(
        default=None, description="Case-insensitive filter on a skill label."
    )
    days: int | None = Field(
        default=None, description="Only jobs posted within the last N days."
    )
    include_total: bool = Field(
        default=False, description="Also count every job matching the filters."
    )
