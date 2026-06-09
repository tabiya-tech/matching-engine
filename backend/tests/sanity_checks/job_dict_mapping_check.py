import subprocess
import sys


def main():
    print("=" * 50)
    print("SHP MATCHING ALGORITHM - JOB DICT MAPPING CHECK")
    print("=" * 50)

    print("\nRunning job dict mapping tests...\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("PASS | Job Dict Mapping Check")
        sys.exit(0)

    print("FAIL | Job Dict Mapping Check")
    sys.exit(1)


if __name__ == "__main__":
    main()
