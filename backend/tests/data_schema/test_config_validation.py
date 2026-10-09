"""Verify that app.config rejects invalid enum values at import time.

`Settings` validates FINAL_SCORE_COMBINER and SCORING_MODE as `Literal` fields on a
pydantic-settings model. If those type annotations are loosened to `str`, invalid config
silently flows through scoring and produces wrong results.
"""

import os
import subprocess
import sys

import pytest


class TestConfigEnumRejection:
    @pytest.mark.parametrize(
        "env_var,bad_value",
        [
            ("FINAL_SCORE_COMBINER", "invalid_combiner"),
            ("SCORING_MODE", "invalid_mode"),
        ],
    )
    def test_invalid_config_raises(self, env_var, bad_value):
        """Invalid config enum must crash the import with ValueError."""
        env = {
            **os.environ,
            "MONGO_URL": "mongodb://localhost:27017",
            "MONGO_DB_NAME": "test",
            env_var: bad_value,
        }
        result = subprocess.run(
            [sys.executable, "-c", "import app.config"],
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        assert result.returncode != 0, (
            f"import app.config should fail with {env_var}={bad_value}"
        )
        assert "ValidationError" in result.stderr or bad_value in result.stderr
