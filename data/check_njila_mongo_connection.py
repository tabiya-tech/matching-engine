"""Verify MongoDB connectivity for NJILA_PREP_SOURCE_DB_URI (secrets not printed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def main() -> int:
    uri = (os.getenv("NJILA_PREP_SOURCE_DB_URI") or "").strip()
    if not uri:
        print("FAIL: NJILA_PREP_SOURCE_DB_URI is empty or unset in backend/.env")
        return 1

    timeout_ms = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "15000"))

    try:
        from pymongo import MongoClient
        from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError

        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        print("OK: connected and ping succeeded.")
        client.close()
        return 0
    except ServerSelectionTimeoutError as e:
        err = str(e)
        print(f"FAIL: cannot reach cluster (timeout): {e}")
        if "CERTIFICATE_VERIFY_FAILED" in err or "SSL" in err:
            print(
                "Hint (macOS): run `/Applications/Python 3.x/Install Certificates.command` "
                "or fix corporate SSL inspection; not usually a wrong Mongo URI."
            )
        return 1
    except OperationFailure as e:
        if getattr(e, "code", None) == 13:
            print(
                "FAIL: Unauthorized after connect — URI works for login but lacks "
                "permission for admin ping or wrong authSource (Atlas: check roles)."
            )
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1
    except ConfigurationError as e:
        print(f"FAIL: bad URI / TLS config: {e}")
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
