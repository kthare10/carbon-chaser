#!/usr/bin/env python3
"""Report a workflow result ONLY if THIS run actually produced it.

Runs on the submit node.

The problem this exists to solve: by default Pegasus stages outputs into ONE
shared directory (`wf-output/higgs_result.json`), so the file on disk carries
no evidence of which run wrote it. Three ways that misleads:

* a previous run's result is still sitting there and gets read as this one's;
* `rm -rf wf-runs` does NOT stop an already-running DAG, so an OLDER workflow
  can finish and write that file *after* the new submission — passing any
  "is the file newer than submit time?" check while belonging to the old run;
* the current run failed, and the stale file makes it look like it succeeded.

Timestamps cannot separate these, because recency is not provenance. Nor is
"this run's evaluate job exited 0" sufficient on its own: it says our job
succeeded, not that the file in a SHARED directory is the one our job wrote —
a concurrent run overwriting it afterwards satisfies both conditions.

So provenance is made structural rather than inferred, in two parts:

1. Each run is planned with its own `--output-dir`, so no other run can write
   there. Isolation by construction, not by checking afterwards.
2. This script does not take the caller's word for where that is. It reads
   `braindump.yml` from the run directory — written by the planner, recording
   `wf_uuid` and the exact `planner_arguments` — and recovers the output
   directory from those arguments. The run states where its own outputs went.

A run planned WITHOUT `--output-dir` is refused outright, because for that run
the question is genuinely unanswerable. The stampede DB check (did THIS run's
evaluate job succeed?) and the mtime check remain as corroboration.

Usage (on the submit node):
    python3 run_result.py --run-dir <dir> [--submitted-after EPOCH]
Exit status is 0 only when a genuine result for this run is printed.
"""

import argparse
import glob
import json
import os
import re
import sqlite3
import sys

RESULT_NAME = "higgs_result.json"

# Outputs are staged out AFTER the evaluate job terminates, so the file's mtime
# sits a little past the last recorded job state. Generous enough to cover
# stage-out, tight enough that a different run's write falls outside.
WINDOW_SLACK_S = 900.0


def run_identity(run_dir):
    """(wf_uuid, output_dir) as recorded by the run ITSELF, or (None, reason).

    `braindump.yml` is written by the planner into the run directory and
    records `wf_uuid` plus the exact `planner_arguments` used. Parsing the
    output directory back out of those arguments is what makes provenance
    structural: the run states where its own outputs went, rather than the
    caller asserting it.

    A run planned WITHOUT `--output-dir` staged into a shared, run-agnostic
    directory, and no amount of checking afterwards can prove which run wrote
    a file there — so that case is refused rather than guessed at.
    """
    path = os.path.join(run_dir, "braindump.yml")
    if not os.path.exists(path):
        return None, f"no braindump.yml in {run_dir}: cannot identify this run"
    text = open(path).read()

    uuid_match = re.search(r'^wf_uuid:\s*"?([0-9a-fA-F-]+)"?', text, re.M)
    args_match = re.search(r'^planner_arguments:\s*"(.*)"\s*$', text, re.M)
    if not uuid_match:
        return None, f"braindump.yml in {run_dir} records no wf_uuid"
    if not args_match:
        return None, f"braindump.yml in {run_dir} records no planner_arguments"

    planner_args = args_match.group(1).replace('\\"', '"')
    out_match = re.search(r'(?:--output-dir|-O)\s+(\S+)', planner_args)
    if not out_match:
        return None, (
            "this run was planned WITHOUT --output-dir, so its outputs went to "
            "a shared directory that any other run also writes to. Provenance "
            "cannot be established after the fact -- replan with "
            "--output-dir <per-run path>.")
    return (uuid_match.group(1), out_match.group(1).rstrip('"')), None


def base_dir(run_dir):
    """`basedir` as the run recorded it, e.g. /home/ubuntu/wf-runs."""
    path = os.path.join(run_dir, "braindump.yml")
    try:
        text = open(path).read()
    except OSError:
        return None
    match = re.search(r'^basedir:\s*"?([^"\n]+)"?', text, re.M)
    return match.group(1).strip() if match else None


def result_write_window(run_dir):
    """(earliest, latest) that THIS run could have written its result, or None.

    Bounds are deliberately asymmetric, because the two ends mean different
    things — an earlier version used one slack for both and let two files
    through that this run demonstrably did not write:

    * **Lower bound: when this run's evaluate job last began EXECUTING, with
      no slack.**
      The result is produced by evaluate and staged out afterwards, so a file
      written before evaluate began cannot be its output. Two things were
      Three things were wrong before, each demonstrated: the bound was the
      run's *first* timestamp (so a file written during the training segments
      passed, ~17 min early); it was relaxed downward by the 900s slack (so a
      previous run's file written 10 min before this run started passed); and
      it was the evaluate job's SUBMIT rather than its EXECUTE (so a file
      written while the job sat Idle awaiting a negotiation cycle passed).
      `max()` over EXECUTE transitions, so a retried job is anchored to the
      attempt that actually produced the output.
    * **Upper bound: the run's last recorded activity, plus slack**, because
      stage-out genuinely happens after the last job state is recorded.

    `jobstate.log` is the source: plain text in the run directory, one line
    per transition, `epoch jobname state ...` — the format observed on the
    real submit node. Falls back to the stampede DB's `jobstate` table,
    introspected rather than assumed. If neither can identify the evaluate
    job, returns None and the caller refuses: no record of when the result
    could have been written means no way to attribute it.
    """
    all_stamps, execute_stamps = [], []
    log = os.path.join(run_dir, "jobstate.log")
    if os.path.exists(log):
        for line in open(log):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                stamp = float(parts[0])
            except ValueError:
                continue
            all_stamps.append(stamp)
            # Only the EXECUTE transition, and only for the evaluate job. The
            # earliest evaluate timestamp is its SUBMIT, which can precede
            # execution by minutes -- the job sits Idle waiting for a
            # negotiation cycle and its stage-in. Using SUBMIT let a prior
            # run's file written during that idle gap be attributed to us.
            if parts[1].startswith("evaluate") and parts[2] == "EXECUTE":
                execute_stamps.append(stamp)

    if not execute_stamps:
        for db in sorted(glob.glob(os.path.join(run_dir, "*.stampede.db"))):
            try:
                conn = sqlite3.connect(db)
                cols = [r[1] for r in conn.execute("pragma table_info(jobstate)")]
                stamp_col = next((c for c in cols if "timestamp" in c.lower()),
                                 None)
                id_col = next((c for c in cols
                               if c.lower() == "job_instance_id"), None)
                state_col = next((c for c in cols if c.lower() == "state"), None)
                if stamp_col and id_col and state_col:
                    rows = conn.execute(
                        f"select js.{stamp_col} from jobstate js "
                        f"join job_instance ji on ji.job_instance_id = js.{id_col} "
                        f"join job j on j.job_id = ji.job_id "
                        f"where j.exec_job_id like 'evaluate%' "
                        f"and js.{state_col} = 'EXECUTE' "
                        f"and js.{stamp_col} is not null").fetchall()
                    execute_stamps = [float(r[0]) for r in rows]
                    every = conn.execute(
                        f"select {stamp_col} from jobstate "
                        f"where {stamp_col} is not null").fetchall()
                    all_stamps = [float(r[0]) for r in every]
                conn.close()
            except sqlite3.Error:
                continue
            if execute_stamps:
                break

    if not execute_stamps:
        return None
    # The LAST execution start, not the first: DAGMan retries mean an earlier
    # attempt may have run and failed, and a file from that attempt is not the
    # output of the attempt that succeeded.
    return max(execute_stamps), max(all_stamps or execute_stamps)


def runs_sharing_output_dir(run_dir, output_dir):
    """Other runs whose braindump names the SAME output directory.

    The isolation this script relies on comes from each run being planned with
    its own `--output-dir`. That holds when the caller derives the path from
    something unique (the notebook uses the submission epoch), but nothing
    forced it: plan two runs with the same `--output-dir` and both stage into
    one directory again, at which point "this run's evaluate job exited 0" is
    once more compatible with the file having been written by the other run.

    So check it rather than assume it. Sibling run directories are right there
    next to this one, and each records its own output directory.
    """
    # Scan the whole base directory, not just siblings. Sibling-only missed a
    # clashing run under a different workflow name entirely — verified.
    base = base_dir(run_dir) or os.path.dirname(os.path.abspath(run_dir))
    want = os.path.normpath(output_dir)
    clashes = []
    for entry in sorted(glob.glob(os.path.join(base, "**", "braindump.yml"),
                                  recursive=True)):
        entry = os.path.dirname(entry)
        if os.path.abspath(entry) == os.path.abspath(run_dir):
            continue
        identity, problem = run_identity(entry)
        if problem:
            continue
        _uuid, other_dir = identity
        if os.path.normpath(other_dir) == want:
            clashes.append(os.path.basename(entry))
    return clashes


def evaluate_job_status(run_dir):
    """(state, detail) for this run's evaluate job, from its OWN database."""
    matches = sorted(glob.glob(os.path.join(run_dir, "*.stampede.db")))
    if not matches:
        return "no-db", (f"no *.stampede.db in {run_dir} — this run never got "
                         f"far enough to record anything")
    connection = sqlite3.connect(matches[0])
    try:
        rows = connection.execute(
            "select j.exec_job_id, ji.exitcode from job_instance ji "
            "join job j on ji.job_id = j.job_id "
            "where j.exec_job_id like 'evaluate%' "
            "order by ji.job_instance_id").fetchall()
    except sqlite3.Error as exc:
        return "no-db", f"could not read {matches[0]}: {exc}"
    finally:
        connection.close()

    if not rows:
        return "not-run", ("this run has no evaluate job instance yet — the "
                           "workflow has not reached it")
    name, exitcode = rows[-1]
    if exitcode is None:
        return "running", f"{name} is still running (no exit code recorded)"
    if int(exitcode) != 0:
        return "failed", f"{name} exited {exitcode}"
    return "succeeded", f"{name} exited 0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="THIS submission's Pegasus run directory")
    # Deliberately NOT an --output-dir override. An earlier version of this
    # script had one, which defeated the whole point: a caller could hand it
    # any path and the braindump check was skipped entirely, while the output
    # still read "result for run ..." as though it had been attributed. A
    # caller-supplied path only relocates the assumption.
    #
    # This flag can therefore only ADD a constraint, never remove one: if
    # given, it must MATCH what the run itself recorded. That catches the case
    # where the planner did not honour the --output-dir it was passed.
    ap.add_argument("--expect-output-dir",
                    help="assert the run's own output directory equals this; "
                         "a mismatch is refused. Cannot be used to point the "
                         "tool somewhere else — attribution always comes from "
                         "the run's braindump.yml")
    ap.add_argument("--submitted-after", type=int, default=0,
                    help="epoch seconds; a result older than this cannot "
                         "belong to this run (secondary check)")
    args = ap.parse_args()

    # WHERE to look is decided by the run itself. There is no path around this.
    identity, problem = run_identity(args.run_dir)
    if problem:
        print(f"CANNOT ATTRIBUTE A RESULT TO THIS RUN: {problem}")
        return 1
    wf_uuid, output_dir = identity

    if args.expect_output_dir:
        want = os.path.normpath(args.expect_output_dir)
        got = os.path.normpath(output_dir)
        if want != got:
            print(f"OUTPUT DIRECTORY MISMATCH: this run recorded {got}, but "
                  f"{want} was expected. The planner may not have honoured "
                  f"the --output-dir it was given, which means outputs went "
                  f"somewhere shared. Refusing to report a result.")
            return 1

    shared_with = runs_sharing_output_dir(args.run_dir, output_dir)
    if shared_with:
        print(f"OUTPUT DIRECTORY IS NOT EXCLUSIVE TO THIS RUN: "
              f"{output_dir} is also the output directory of {shared_with}. "
              f"A result there cannot be attributed to either run — plan each "
              f"run with its own --output-dir.")
        return 1

    output = os.path.join(output_dir, RESULT_NAME)

    state, detail = evaluate_job_status(args.run_dir)
    if state != "succeeded":
        print(f"NO RESULT FOR THIS RUN ({state}): {detail}")
        print(f"Refusing to report anything from {output}.")
        return 1

    if not os.path.exists(output):
        # The DB says this run's evaluate job succeeded, but the file is gone.
        print(f"INCONSISTENT: this run's evaluate job exited 0 ({detail}) but "
              f"{output} does not exist.")
        return 1

    # NOT int(): truncating to whole seconds while the bounds below are floats
    # lets sub-second rounding push a genuine result under the lower bound —
    # a guard that fails a fraction of the time is worse than no guard, because
    # the failure looks like a real attribution problem.
    mtime = os.path.getmtime(output)
    if args.submitted_after and mtime < args.submitted_after:
        print(f"INCONSISTENT: this run's evaluate job exited 0, but "
              f"{output} was last written at {mtime:.0f}, before this run was "
              f"submitted ({args.submitted_after}). Refusing to report it.")
        return 1

    # The decisive check, and the only one that does not depend on other runs
    # leaving evidence behind: was this file written while THIS run was
    # actually working? A leftover file from a run whose directory has since
    # been deleted -- which `rm -rf wf-runs` does before every submission --
    # falls outside the window, and no clash scan could ever have seen it.
    window = result_write_window(args.run_dir)
    if window is None:
        print(f"CANNOT ATTRIBUTE A RESULT TO THIS RUN: nothing in "
              f"{args.run_dir} records when this run's evaluate job ran (no "
              f"jobstate.log, and no readable jobstate table), so there is no "
              f"way to tell whether this run wrote the file or something else "
              f"did.")
        return 1
    earliest, latest = window
    if mtime < earliest:
        print(f"NOT THIS RUN'S RESULT: {output} was written at {mtime:.0f}, "
              f"{earliest - mtime:.0f}s BEFORE this run's evaluate job "
              f"started ({earliest:.0f}). This run cannot have written it — "
              f"most likely an earlier run that shared this output directory. "
              f"No slack is allowed on this side: a file predating the job "
              f"that produces it is not that job's output.")
        return 1
    if mtime > latest + WINDOW_SLACK_S:
        print(f"NOT THIS RUN'S RESULT: {output} was written at {mtime:.0f}, "
              f"{mtime - latest:.0f}s after this run's last recorded activity "
              f"({latest:.0f}), beyond the {WINDOW_SLACK_S:.0f}s allowed for "
              f"stage-out. Something wrote it after this run finished.")
        return 1

    with open(output) as handle:
        payload = handle.read()
    try:
        json.loads(payload)
    except ValueError as exc:
        print(f"INCONSISTENT: {output} is not valid JSON: {exc}")
        return 1

    print(f"result for run {wf_uuid} ({detail}, written at {mtime:.0f}),")
    print(f"read from that run's OWN output directory {output_dir}:")
    print(payload.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
