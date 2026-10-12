"""JSON log formatting for Cloud Logging (see README "Errors and logging")."""

import json
import logging
from datetime import datetime, timezone


class JsonLogFormatter(logging.Formatter):
    """One JSON object per record, so Cloud Logging indexes the fields and reads ``severity``.

    Structured fields go in ``extra={"fields": {...}}`` and are emitted as top-level keys.
    Never put user profile data (skills, preferences, location, education) in a message or a field.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                entry.setdefault(key, value)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)
