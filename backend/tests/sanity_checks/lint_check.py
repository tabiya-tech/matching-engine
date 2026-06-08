"""Lint sanity check for the matching service codebase.

Default: check-only (no file modifications).
Pass --fix to auto-fix safe lint issues before checking.

Usage:
    python tests/sanity_checks/lint_check.py          # check only
    python tests/sanity_checks/lint_check.py --fix     # auto-fix then check
"""

import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 50)
    print("SHP MATCHING ALGORITHM - LINT SANITY CHECK")
    print("=" * 50)

    # Only auto-fix when explicitly requested via --fix.
    # ruff --fix applies safe fixes only (e.g. removing unused imports).
    # Without --fix, the check is read-only and safe to run in CI.
    fix_mode = "--fix" in sys.argv

    if fix_mode:
        print("\nRunning auto-fix (ruff check --fix)...")
        code, out, err = run(["ruff", "check", ".", "--fix"])
        if out:
            print(out)
        if err:
            print(err)

    # Verify no lint errors remain.
    # Exit code 0 = all clean, non-zero = errors found.
    print("\nRunning lint check (ruff check)...\n")
    code, out, err = run(["ruff", "check", "."])

    if code == 0:
        print("PASS | Lint Check")
        sys.exit(0)

    print("FAIL | Lint Check\n")
    if out:
        print(out)
    if err:
        print(err)

    # Hint to the developer how to fix the failures.
    if not fix_mode:
        print("Run with --fix to auto-fix safe issues.")

    sys.exit(1)


if __name__ == "__main__":
    main()
