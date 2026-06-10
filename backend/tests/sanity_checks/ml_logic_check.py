import subprocess
import sys


def main():
    print("=" * 50)
    print("SHP MATCHING ALGORITHM - ML LOGIC CHECK")
    print("=" * 50)

    print("\nRunning ML logic + component + integration tests...\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/ml_logic/",
            "tests/components/",
            "tests/integration/",
            "-q",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("PASS | ML Logic Check")
        sys.exit(0)

    print("FAIL | ML Logic Check")
    sys.exit(1)


if __name__ == "__main__":
    main()
