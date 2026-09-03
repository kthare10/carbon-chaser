#!/usr/bin/env python3
"""Run every test in this directory, and refuse to report success on nothing.

These tests are standalone scripts (each has a `__main__` block that runs its
checks and prints ALL PASS), not unittest.TestCase subclasses. That matters,
because the obvious commands lie:

    $ python -m unittest discover -s tests -q
    Ran 0 tests in 0.000s
    OK                          # <- collected nothing, still said OK

A test suite that reports success without running anything is worse than no
suite at all: it is a green light wired to nothing. So this runner treats an
empty discovery as a FAILURE, and requires each file to both exit 0 *and*
print its ALL PASS sentinel — an exit code alone would be satisfied by a
script that crashed before reaching its assertions and got tidied up by a
bare `except`.

Usage:
    python tests/run_all.py [--dir tests]
"""

import argparse
import glob
import os
import subprocess
import sys

SENTINEL = "ALL PASS"
# Every test file must run in well under this; a hang is a failure, not a wait.
PER_FILE_TIMEOUT = 300


def run_file(path):
    """(ok, detail) for one test script."""
    try:
        proc = subprocess.run([sys.executable, path], text=True,
                              capture_output=True, timeout=PER_FILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {PER_FILE_TIMEOUT}s"
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}\n" + output.strip()[-1200:]
    if SENTINEL not in output:
        # Exited 0 but never announced it finished its checks.
        return False, (f"exit 0 but never printed {SENTINEL!r} — did it "
                       f"return before asserting?\n" + output.strip()[-800:])
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory of test_*.py scripts")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "test_*.py")))
    if not paths:
        # The whole point of this runner.
        print(f"NO TESTS FOUND in {args.dir} — refusing to report success.\n"
              f"An empty suite is a failure: it would otherwise read as "
              f"'everything passes'.")
        return 1

    failures = []
    for path in paths:
        name = os.path.basename(path)
        ok, detail = run_file(path)
        print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)
        if not ok:
            failures.append((name, detail))

    print()
    if failures:
        for name, detail in failures:
            print(f"--- {name}\n{detail}\n")
        print(f"{len(failures)}/{len(paths)} test files FAILED")
        return 1
    print(f"{len(paths)}/{len(paths)} test files passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
