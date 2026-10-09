"""Types for the jobs feature."""

from pydantic import BaseModel


class JobsBrowseQuery(BaseModel):
    """One ``GET /jobs`` page request: keyset cursor, page size and the AND-ed filters."""

    cursor: str | None = None
    """Opaque ``next_cursor`` from the previous page; None for the first page."""

    limit: int
    """Page size."""

    search: str | None = None
    category: str | None = None
    employment_type: str | None = None
    location: str | None = None
    skills: str | None = None
    days: int | None = None
    include_total: bool = False
