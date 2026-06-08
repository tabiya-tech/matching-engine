import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    checks = [
        ("/docs", "Swagger Docs"),
        ("/openapi.json", "OpenAPI Schema"),
    ]

    for endpoint, name in checks:
        response = client.get(endpoint)

        if response.status_code != 200:
            print(f"FAIL | {name} | Status Code: {response.status_code}")
            sys.exit(1)

    print("PASS | FastAPI Startup Check")
    sys.exit(0)

except Exception as e:
    print(f"FAIL | FastAPI Startup Check | {e}")
    sys.exit(1)
