"""Types for the jobs feature."""

from typing import Optional

from pydantic import BaseModel


class JobsBrowseQuery(BaseModel):
    """One ``GET /jobs`` page request: keyset cursor, page size and the AND-ed filters."""

    cursor: Optional[str] = None
    """Opaque ``next_cursor`` from the previous page; None for the first page."""

    limit: int
    """Page size."""

    search: Optional[str] = None
    category: Optional[str] = None
    employment_type: Optional[str] = None
    location: Optional[str] = None
    skills: Optional[str] = None
    days: Optional[int] = None
    include_total: bool = False
