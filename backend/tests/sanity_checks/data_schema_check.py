import subprocess
import sys


def main():
    print("=" * 50)
    print("SHP MATCHING ALGORITHM - DATA SCHEMA CHECK")
    print("=" * 50)

    print("\nRunning data-schema tests...\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/data_schema/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("PASS | Data Schema Check")
        sys.exit(0)

    print("FAIL | Data Schema Check")
    sys.exit(1)


if __name__ == "__main__":
    main()
