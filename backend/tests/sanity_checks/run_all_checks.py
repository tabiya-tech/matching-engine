"""Full test suite runner with one-line PASS/FAIL per check.

Usage:
    python tests/sanity_checks/run_all_checks.py

Paste the output into PR descriptions as proof all checks pass.
"""

import os
import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
os.chdir(_BACKEND)
sys.path.insert(0, str(_BACKEND))

os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "test")

TOTAL = 7
results = []


def _dot_pad(label, width=44):
    dots = "." * (width - len(label))
    return f"{label} {dots}"


def _run_cmd(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def _run_pytest(test_dir):
    return _run_cmd([sys.executable, "-m", "pytest", test_dir, "-q", "--tb=no"])


def check_lint():
    return _run_cmd(["ruff", "check", "."])


def check_format():
    return _run_cmd(["ruff", "format", "--check", "."])


def check_data_validation():
    return _run_pytest("tests/data_validation/")


def check_data_schema():
    return _run_pytest("tests/data_schema/")


def check_smoke():
    return _run_pytest("tests/smoke/")


def check_job_dict_mapping():
    return _run_pytest("tests/unit/")


def check_ml_logic():
    return _run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/ml_logic/",
            "tests/components/",
            "tests/integration/",
            "-q",
            "--tb=no",
        ]
    )


def main():
    print("=" * 50)
    print("SHP MATCHING ALGORITHM — FULL TEST SUITE")
    print("=" * 50)
    print()

    checks = [
        ("Lint Check", check_lint),
        ("Format Check", check_format),
        ("Data Validation Check", check_data_validation),
        ("Data Schema Check", check_data_schema),
        ("Smoke Check", check_smoke),
        ("Job Dict Mapping Check", check_job_dict_mapping),
        ("ML Logic Check", check_ml_logic),
    ]

    for i, (name, fn) in enumerate(checks, 1):
        passed = fn()
        verdict = "PASS" if passed else "FAIL"
        results.append((name, passed))
        print(f"[{i}/{TOTAL}] {_dot_pad(name)} {verdict}")

    print()
    n_passed = sum(1 for _, p in results if p)
    if n_passed == TOTAL:
        print("=" * 50)
        print(f"ALL CHECKS PASSED ({n_passed}/{TOTAL})")
        print("=" * 50)
        sys.exit(0)
    else:
        print("=" * 50)
        print(f"CHECKS FAILED ({n_passed}/{TOTAL} passed)")
        print("=" * 50)
        for name, passed in results:
            if not passed:
                print(f"  FAILED: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
