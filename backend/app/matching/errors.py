"""Errors raised by the matching feature."""


class InvalidMatchRequestError(Exception):
    """The match request cannot be served as sent (empty body, too many users, bad option)."""
