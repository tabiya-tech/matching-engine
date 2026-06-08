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

TOTAL = 6
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


def check_classifier_metadata_migration():
    from app.database import build_job_dict_from_ranked

    def _base_doc():
        return {
            "job_id": "test-job-001",
            "job_fingerprint": "fp-abc123",
            "is_active": True,
            "classifier_metadata": {
                "title": "Software Engineer",
                "employer": "TechCorp",
                "city": "Lusaka",
                "county": "Lusaka",
                "employment_type": "full_time",
                "salary": "K8000/month",
                "closing_date": "2026-07-01",
                "application_url": "https://example.com/apply",
                "job_description": "Build things.",
            },
            "llm_classified_skills": {
                "essential": [{"tabiya_skill_id": "s1", "label": "Python"}],
                "optional": [],
            },
            "llm_job_attributes": {"attributes": {}},
            "onet_work_activities": [],
            "skill_groups_origin_uuids": [],
        }

    try:
        # 1: New format — province + ZQF in classifier_metadata
        doc = _base_doc()
        doc["classifier_metadata"]["province"] = "Copperbelt"
        doc["classifier_metadata"]["min_zqf_level"] = 5
        doc["classifier_metadata"]["max_zqf_level"] = 7
        doc["classifier_metadata"]["min_zqf_label"] = "Diploma / Technician"
        doc["classifier_metadata"]["max_zqf_label"] = "Bachelor's Degree"
        job = build_job_dict_from_ranked(doc)
        assert job["province"] == "Copperbelt"
        assert job["zqf_min"] == 5
        assert job["zqf_max"] == 7
        assert job["zqf_min_label"] == "Diploma / Technician"
        assert job["zqf_max_label"] == "Bachelor's Degree"

        # 2: Legacy format — county + root-level ZQF
        doc = _base_doc()
        doc["classifier_metadata"].pop("province", None)
        doc["zqf_min"] = 4
        doc["zqf_max"] = 9
        job = build_job_dict_from_ranked(doc)
        assert job["province"] == "Lusaka"
        assert job["zqf_min"] == 4
        assert job["zqf_max"] == 9
        assert job["zqf_min_label"] is None
        assert job["zqf_max_label"] is None

        # 3: New format overrides root-level
        doc = _base_doc()
        doc["zqf_min"] = 3
        doc["zqf_max"] = 6
        doc["classifier_metadata"]["min_zqf_level"] = 5
        doc["classifier_metadata"]["max_zqf_level"] = 8
        job = build_job_dict_from_ranked(doc)
        assert job["zqf_min"] == 5
        assert job["zqf_max"] == 8

        # 4: Province fallback (empty string -> county)
        doc = _base_doc()
        doc["classifier_metadata"]["province"] = ""
        doc["classifier_metadata"]["county"] = "Central"
        job = build_job_dict_from_ranked(doc)
        assert job["province"] == "Central"

        # 5: Both absent
        doc = _base_doc()
        doc["classifier_metadata"].pop("county", None)
        doc["classifier_metadata"].pop("province", None)
        job = build_job_dict_from_ranked(doc)
        assert job["province"] == ""
        assert job["zqf_min"] is None
        assert job["zqf_max"] is None

        return True
    except (AssertionError, Exception):
        return False


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
        ("Classifier Metadata Migration", check_classifier_metadata_migration),
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
