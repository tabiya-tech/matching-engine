"""Format sanity check for the matching service codebase.

Default: check-only (no file modifications).
Pass --fix to auto-format before checking.

Usage:
    python tests/sanity_checks/formatter_check.py          # check only
    python tests/sanity_checks/formatter_check.py --fix     # auto-format then check
"""

import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 50)
    print("SHP MATCHING ALGORITHM - FORMAT SANITY CHECK")
    print("=" * 50)

    # Only auto-format when explicitly requested via --fix.
    # Without --fix, the check is read-only and safe to run in CI.
    fix_mode = "--fix" in sys.argv

    if fix_mode:
        print("\nRunning auto-format (ruff format)...")
        code, out, err = run(["ruff", "format", "."])
        if out:
            print(out)
        if err:
            print(err)

    # Verify all files match ruff's formatting rules.
    # Exit code 0 = everything formatted, non-zero = files would be reformatted.
    print("\nRunning format check (ruff format --check)...\n")
    code, out, err = run(["ruff", "format", "--check", "."])

    if code == 0:
        print("PASS | Format Check")
        sys.exit(0)

    print("FAIL | Format Check\n")
    if out:
        print(out)
    if err:
        print(err)

    # Hint to the developer how to fix the failures.
    if not fix_mode:
        print("Run with --fix to auto-format.")

    sys.exit(1)


if __name__ == "__main__":
    main()
