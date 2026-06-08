import subprocess
import sys


def main():
    print("=" * 50)
    print("SHP MATCHING ALGORITHM - SMOKE CHECK")
    print("=" * 50)

    print("\nRunning smoke tests...\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/smoke/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("PASS | Smoke Check")
        sys.exit(0)

    print("FAIL | Smoke Check")
    sys.exit(1)


if __name__ == "__main__":
    main()
