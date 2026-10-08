"""Errors raised by the jobs feature."""


class InvalidCursor(ValueError):
    """Raised when a /jobs pagination cursor cannot be decoded into a Mongo _id."""
