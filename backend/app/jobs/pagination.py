"""Keyset pagination for GET /jobs: an opaque cursor wrapping the last ``_id`` of a page."""

import base64
import binascii

from bson import ObjectId
from bson.errors import InvalidId


class InvalidCursor(ValueError):
    """Raised when a /jobs pagination cursor cannot be decoded into a Mongo _id."""


# GET /jobs is sorted by ``_id`` descending (newest first). ``_id`` is always indexed, so the
# keyset seek stays fast without a dedicated sort index.
JOBS_PAGE_SORT = [("_id", -1)]


def encode_jobs_cursor(object_id: ObjectId) -> str:
    """Opaque, URL-safe cursor wrapping a Mongo ``_id`` (the last item on the page)."""
    return base64.urlsafe_b64encode(str(object_id).encode("ascii")).decode("ascii")


def decode_jobs_cursor(cursor: str) -> ObjectId:
    """Inverse of encode_jobs_cursor. Raises InvalidCursor on any malformed input."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("ascii")
        return ObjectId(raw)
    except (ValueError, InvalidId, binascii.Error, UnicodeDecodeError) as e:
        raise InvalidCursor(f"invalid cursor: {cursor!r}") from e
