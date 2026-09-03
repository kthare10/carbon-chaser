"""A result must be attributable to THIS run, not merely be recent.

Pegasus stages outputs into a fixed, run-agnostic directory, so the file on
disk says nothing about which run wrote it. The case that defeats a timestamp
check: `rm -rf wf-runs` does not stop an already-running DAG, so a PREVIOUS
workflow can finish and write the result *after* the new submission — newer
than submit time, and still the wrong run's answer.

These tests build synthetic stampede databases (the real schema: `job` joined
to `job_instance`) and check the decision comes from the run's own database.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "pegasus", "run_result.py")


def make_run_dir(root, evaluate_exitcode="omit", with_db=True,
                 output_dir="use-own", uuid="11111111-2222-3333-4444-555555555555",
                 basedir=None):
    """A run directory with a braindump + stampede DB, as Pegasus writes them.

    `output_dir="use-own"` gives the run its own isolated output directory (the
    correct setup); None omits --output-dir from planner_arguments, which is
    the unsound legacy case that must be refused.
    """
    run_dir = tempfile.mkdtemp(dir=root)
    if output_dir == "use-own":
        output_dir = os.path.join(run_dir, "outputs")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        planner_args = ('--submit -s condorpool --output-site local '
                        f'--output-dir {output_dir} --dir wf-runs higgs-wf.yml ')
    else:
        planner_args = ('--submit -s condorpool --output-site local '
                        '--dir wf-runs higgs-wf.yml ')
    with open(os.path.join(run_dir, "braindump.yml"), "w") as handle:
        handle.write(f'wf_uuid: "{uuid}"\n')
        handle.write(f'submit_dir: "{run_dir}"\n')
        handle.write(f'basedir: "{basedir or root}"\n')
        handle.write(f'planner_arguments: "\\"{planner_args}\\""\n')
    # A real run records when it was active; that window is what ties a file in
    # a shared output directory to THIS run, without needing any other run to
    # have left evidence behind.
    write_jobstate(run_dir)
    if not with_db:
        return run_dir
    db = os.path.join(run_dir, "carbon-aware-higgs-0.stampede.db")
    conn = sqlite3.connect(db)
    conn.execute("create table job (job_id integer, exec_job_id text)")
    conn.execute("create table job_instance (job_instance_id integer, "
                 "job_id integer, exitcode integer)")
    conn.execute("insert into job values (1, 'train_higgs_train_000')")
    conn.execute("insert into job_instance values (1, 1, 0)")
    if evaluate_exitcode != "omit":
        conn.execute("insert into job values (2, 'evaluate_higgs_evaluate')")
        conn.execute("insert into job_instance values (2, 2, ?)",
                     (evaluate_exitcode,))
    conn.commit()
    conn.close()
    return run_dir


def write_jobstate(run_dir, first=None, last=None):
    """A jobstate.log in the real format: epoch, job, state, ...

    Emits both of the evaluate job's transitions, and places its EXECUTE
    comfortably before `last`, because that is what a real run looks like: the
    evaluate job starts, terminates, and the output is staged out afterwards.
    An earlier version put the evaluate line AT `last` (= now), so a result
    file written moments later was a sub-second *before* the evaluate window
    began and got refused — the fixture, not the code, was wrong.
    """
    now = time.time()
    last = now - 5 if last is None else last
    first = last - 120 if first is None else first
    with open(os.path.join(run_dir, "jobstate.log"), "w") as handle:
        handle.write(f"{first:.0f} INTERNAL *** DAGMAN_STARTED ***\n")
        handle.write(f"{last - 30:.0f} evaluate_higgs_evaluate EXECUTE "
                     f"29.0 condorpool - 13\n")
        handle.write(f"{last:.0f} evaluate_higgs_evaluate JOB_TERMINATED "
                     f"29.0 condorpool - 13\n")


def result_path(run_dir):
    """Where THIS run's output lands, per its own braindump."""
    return os.path.join(run_dir, "outputs", "higgs_result.json")


def write_result(path, auc=0.8767, mtime=None):
    with open(path, "w") as handle:
        json.dump({"val_auc": auc, "trained_steps": 155608}, handle)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def run(run_dir, submitted_after=0):
    """Deliberately does NOT pass --output-dir: the script must work it out
    from the run's own braindump, which is what makes attribution sound."""
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--run-dir", run_dir,
         "--submitted-after", str(submitted_after)],
        text=True, capture_output=True, timeout=60)
    return proc.returncode, proc.stdout + proc.stderr


def test_result_is_reported_when_this_run_produced_it():
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=0)
        write_result(result_path(run_dir))
        code, text = run(run_dir, submitted_after=int(time.time()) - 600)
        assert code == 0, text
        assert "0.8767" in text, text
        print("  a genuine result for this run is reported")


def test_previous_runs_result_is_refused_even_though_it_is_newer():
    """The case a timestamp check cannot catch.

    The old DAG was still running, finished after the new submission, and
    wrote a result. The file is NEWER than submit time — but THIS run's
    evaluate job never ran, so it is not this run's answer.
    """
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode="omit")   # never ran
        submitted = int(time.time()) - 600
        write_result(result_path(run_dir), auc=0.5, mtime=submitted + 300)
        code, text = run(run_dir, submitted_after=submitted)
        assert code == 1, f"reported another run's result: {text}"
        assert "NO RESULT FOR THIS RUN" in text, text
        assert "0.5" not in text, f"leaked the wrong run's number: {text}"
        print("  a newer file from a DIFFERENT run is refused, not reported")


def test_failed_evaluate_job_is_not_reported_as_a_result():
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=1)
        write_result(result_path(run_dir))
        code, text = run(run_dir)
        assert code == 1, text
        assert "failed" in text, text
        print("  a failed evaluate job yields no result")


def test_still_running_is_distinguished_from_finished():
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=None)   # no exit code
        write_result(result_path(run_dir))
        code, text = run(run_dir)
        assert code == 1, text
        assert "running" in text, text
        print("  a still-running evaluate job is reported as running")


def test_missing_database_is_refused():
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, with_db=False)
        write_result(result_path(run_dir))
        code, text = run(run_dir)
        assert code == 1, text
        assert "no-db" in text, text
        print("  a run with no database reports no result")


def test_result_older_than_submission_is_refused():
    """Secondary check: DB says success but the file predates the run."""
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=0)
        submitted = int(time.time())
        write_result(result_path(run_dir), mtime=submitted - 3600)
        code, text = run(run_dir, submitted_after=submitted)
        assert code == 1, text
        assert "INCONSISTENT" in text, text
        print("  a result predating the submission is refused")


def test_run_planned_without_an_isolated_output_dir_is_refused():
    """The genuinely unanswerable case, refused rather than guessed.

    Planned without --output-dir, a run stages into a directory shared with
    every other run. Nothing checkable afterwards can establish which run
    wrote a file there — so the honest answer is to say the question cannot
    be answered, not to pick the file and hope.
    """
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=0, output_dir=None)
        shared = os.path.join(root, "wf-output")
        os.makedirs(shared, exist_ok=True)
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.42)
        code, text = run(run_dir)
        assert code == 1, text
        assert "CANNOT ATTRIBUTE" in text, text
        assert "0.42" not in text, f"leaked an unattributable number: {text}"
        print("  a run without an isolated output dir is refused")


def test_one_runs_result_is_never_read_for_another():
    """Two runs, two output directories: no cross-reading, by construction."""
    with tempfile.TemporaryDirectory() as root:
        run_a = make_run_dir(root, evaluate_exitcode=0, uuid="aaaaaaaa-0000-0000-0000-000000000001")
        run_b = make_run_dir(root, evaluate_exitcode=0, uuid="bbbbbbbb-0000-0000-0000-000000000002")
        write_result(result_path(run_a), auc=0.8767)      # only A produced one
        code, text = run(run_b)
        assert code == 1, f"read run A's result for run B: {text}"
        assert "0.8767" not in text, f"leaked run A's number: {text}"
        assert "INCONSISTENT" in text, text
        # ...and A still reports its own correctly.
        code_a, text_a = run(run_a)
        assert code_a == 0 and "0.8767" in text_a, text_a
        assert "aaaaaaaa" in text_a, f"result not labelled with its run: {text_a}"
        print("  one run's result is never reported for another")


def test_no_flag_can_point_the_tool_away_from_the_run():
    """There must be no caller-supplied path that skips attribution.

    An earlier version accepted --output-dir and, when given, skipped the
    braindump check entirely while still printing "result for run ...". That
    is the same unsoundness the braindump check was added to remove, reachable
    by anyone who passes a flag. So the override is gone: the only path-shaped
    flag can add a constraint, not remove one.
    """
    import subprocess as sp
    helptext = sp.run([sys.executable, SCRIPT, "--help"], text=True,
                      capture_output=True, timeout=60).stdout
    assert "--output-dir" not in helptext.replace("--expect-output-dir", ""), (
        f"an --output-dir override is back; it bypasses provenance:\n{helptext}")

    # And passing it is rejected outright rather than silently ignored.
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=0)
        write_result(result_path(run_dir))
        elsewhere = os.path.join(root, "elsewhere")
        os.makedirs(elsewhere, exist_ok=True)
        write_result(os.path.join(elsewhere, "higgs_result.json"), auc=0.11)
        proc = sp.run([sys.executable, SCRIPT, "--run-dir", run_dir,
                       "--output-dir", elsewhere],
                      text=True, capture_output=True, timeout=60)
        assert proc.returncode != 0, "accepted an --output-dir override"
        assert "0.11" not in (proc.stdout + proc.stderr), "read the wrong dir"
        print("  no flag can point the tool away from the run's own outputs")


def test_expect_output_dir_can_only_add_a_constraint():
    """Matching is allowed; mismatching is refused, never followed."""
    import subprocess as sp
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=0)
        write_result(result_path(run_dir))
        own = os.path.join(run_dir, "outputs")

        # Matching: proceeds.
        proc = sp.run([sys.executable, SCRIPT, "--run-dir", run_dir,
                       "--expect-output-dir", own],
                      text=True, capture_output=True, timeout=60)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "0.8767" in proc.stdout, proc.stdout

        # Mismatching: refused, and the other directory is NOT read.
        elsewhere = os.path.join(root, "elsewhere")
        os.makedirs(elsewhere, exist_ok=True)
        write_result(os.path.join(elsewhere, "higgs_result.json"), auc=0.22)
        proc = sp.run([sys.executable, SCRIPT, "--run-dir", run_dir,
                       "--expect-output-dir", elsewhere],
                      text=True, capture_output=True, timeout=60)
        assert proc.returncode == 1, proc.stdout
        assert "MISMATCH" in proc.stdout, proc.stdout
        assert "0.22" not in proc.stdout and "0.8767" not in proc.stdout, proc.stdout
        print("  --expect-output-dir verifies, and never redirects")


def test_two_runs_sharing_an_output_dir_are_refused():
    """Isolation is relied upon, so it must be verified rather than assumed.

    `--output-dir` makes the output exclusive only if the caller picks a unique
    path. Plan two runs with the same one and both stage into it, at which
    point "this run's evaluate exited 0" is again compatible with the file
    having been written by the other run — the exact hole --output-dir closed.
    """
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared-output")
        run_a = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                             uuid="aaaaaaaa-0000-0000-0000-00000000000a")
        run_b = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                             uuid="bbbbbbbb-0000-0000-0000-00000000000b")
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.9999)
        for run_dir in (run_a, run_b):
            code, text = run(run_dir)
            assert code == 1, f"approved a shared output dir: {text}"
            assert "NOT EXCLUSIVE" in text, text
            assert "0.9999" not in text, f"leaked an unattributable value: {text}"
        print("  two runs sharing an output directory are both refused")


def test_a_leftover_file_from_a_DELETED_run_is_refused():
    """The hole the clash scan structurally could not see.

    The notebook does `rm -rf wf-runs` before every submission, which deletes
    exactly the braindump a clash scan needs — while the previous run's file
    sits untouched in the shared output directory. Verified: before the
    activity-window check, this reported the deleted run's 0.6666 as the
    current run's result.
    """
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        gone = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                            uuid="cccccccc-0000-0000-0000-00000000000c")
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.6666,
                     mtime=time.time() - 7200)        # written long before
        shutil.rmtree(gone)                            # evidence removed
        mine = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                            uuid="dddddddd-0000-0000-0000-00000000000d")
        code, text = run(mine, submitted_after=1)
        assert code == 1, f"attributed a deleted run's file: {text}"
        assert "NOT THIS RUN'S RESULT" in text, text
        assert "0.6666" not in text, f"leaked the other run's value: {text}"
        print("  a leftover file from a deleted run is refused")


def test_a_clash_under_a_different_workflow_name_is_found():
    """The scan used to look only at siblings, so this was invisible."""
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        tree_a = os.path.join(root, "u", "pegasus", "wf-A")
        tree_b = os.path.join(root, "u", "pegasus", "wf-B")
        os.makedirs(tree_a)
        os.makedirs(tree_b)
        # basedir is `root` for both, as it is on a real submit node, so the
        # scan covers the whole tree rather than one workflow's subtree.
        run_a = make_run_dir(tree_a, evaluate_exitcode=0, output_dir=shared,
                             uuid="aaaaaaaa-0000-0000-0000-00000000000a",
                             basedir=root)
        run_b = make_run_dir(tree_b, evaluate_exitcode=0, output_dir=shared,
                             uuid="bbbbbbbb-0000-0000-0000-00000000000b",
                             basedir=root)
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.7777)
        code, text = run(run_a)
        assert code == 1, f"missed a clash in another subtree: {text}"
        assert "NOT EXCLUSIVE" in text, text
        assert "0.7777" not in text, text
        print("  a clash under a different workflow name is found")


def test_a_run_with_no_activity_record_is_refused():
    """No record of when the run worked means no way to attribute a file."""
    with tempfile.TemporaryDirectory() as root:
        run_dir = make_run_dir(root, evaluate_exitcode=0)
        os.remove(os.path.join(run_dir, "jobstate.log"))
        write_result(result_path(run_dir))
        code, text = run(run_dir)
        assert code == 1, f"attributed without an activity record: {text}"
        assert "records when this run's evaluate job ran" in text, text
        print("  a run with no activity record is refused")


def test_a_file_written_before_this_runs_evaluate_started_is_refused():
    """A file predating the job that produces it is not that job's output.

    Two bugs made this pass. The lower bound was the run's FIRST timestamp, so
    a file written during the training segments qualified; and it was relaxed
    downward by the stage-out slack, so a previous run's file written minutes
    before this run even started qualified. Slack belongs only on the upper
    bound, where stage-out actually happens.
    """
    now = time.time()
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        run_dir = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                               uuid="bbbbbbbb-0000-0000-0000-00000000000b")
        # This run: DAG started 20 min ago, evaluate ran in the last minute.
        with open(os.path.join(run_dir, "jobstate.log"), "w") as handle:
            handle.write(f"{now - 1200:.0f} INTERNAL *** DAGMAN_STARTED ***\n")
            handle.write(f"{now - 60:.0f} evaluate_higgs_evaluate EXECUTE "
                         f"29.0 condorpool - 13\n")
            handle.write(f"{now:.0f} evaluate_higgs_evaluate JOB_TERMINATED "
                         f"29.0 condorpool - 13\n")
        # Written during training, 17 min before evaluate started.
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.4444,
                     mtime=now - 1100)
        code, text = run(run_dir, submitted_after=1)
        assert code == 1, f"attributed a file written before evaluate: {text}"
        assert "BEFORE this run's evaluate job started" in text, text
        assert "0.4444" not in text, f"leaked the value: {text}"
        print("  a file predating this run's evaluate job is refused")


def test_a_file_written_after_the_run_finished_is_refused():
    """Beyond stage-out, a later write is someone else's."""
    now = time.time()
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        run_dir = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                               uuid="bbbbbbbb-0000-0000-0000-00000000000b")
        write_jobstate(run_dir, first=now - 3600, last=now - 3000)
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.3333,
                     mtime=now)                     # long after it finished
        code, text = run(run_dir, submitted_after=1)
        assert code == 1, f"attributed a later write: {text}"
        assert "after this run's last recorded activity" in text, text
        assert "0.3333" not in text, text
        print("  a file written after the run finished is refused")


def test_a_genuine_result_written_during_stage_out_is_accepted():
    """The bound must not be so tight that real runs fail.

    Outputs are staged out after the evaluate job terminates, so a genuine
    result is written slightly LATER than the last recorded job state.
    """
    now = time.time()
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        run_dir = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                               uuid="bbbbbbbb-0000-0000-0000-00000000000b")
        with open(os.path.join(run_dir, "jobstate.log"), "w") as handle:
            handle.write(f"{now - 1200:.0f} INTERNAL *** DAGMAN_STARTED ***\n")
            handle.write(f"{now - 120:.0f} evaluate_higgs_evaluate EXECUTE "
                         f"29.0 condorpool - 13\n")
            handle.write(f"{now - 60:.0f} evaluate_higgs_evaluate "
                         f"JOB_TERMINATED 29.0 condorpool - 13\n")
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.8767,
                     mtime=now - 30)                # staged out after the job
        code, text = run(run_dir, submitted_after=1)
        assert code == 0, f"refused a genuine result: {text}"
        assert "0.8767" in text, text
        print("  a genuine result staged out after the job is accepted")


def test_slack_is_not_allowed_on_the_lower_bound():
    """Pins the asymmetry, with a gap INSIDE the stage-out slack.

    Found by mutation testing: the sibling test used a 1040s gap, larger than
    the 900s slack, so restoring the slack on the lower bound still refused it
    and the mutant survived. The property that actually matters is that NO
    slack applies downward, so the gap here is deliberately smaller than the
    slack — a prior run's file written 600s before this run's evaluate job
    began. With slack applied downward this is attributed; without, refused.
    """
    now = time.time()
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        gone = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                            uuid="aaaaaaaa-0000-0000-0000-00000000000a")
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.5555,
                     mtime=now - 600)              # inside the 900s slack
        shutil.rmtree(gone)                         # its evidence is gone
        mine = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                            uuid="bbbbbbbb-0000-0000-0000-00000000000b")
        write_jobstate(mine, first=now - 120, last=now)   # evaluate: now-30
        code, text = run(mine, submitted_after=1)
        assert code == 1, f"slack on the lower bound attributed it: {text}"
        assert "BEFORE this run's evaluate job started" in text, text
        assert "0.5555" not in text, f"leaked the value: {text}"
        print("  no slack is allowed below the evaluate job's start")


def test_a_file_written_while_evaluate_sat_idle_is_refused():
    """The bound is EXECUTE, not SUBMIT.

    A job is SUBMITted, then waits: a negotiation cycle, a stage-in. With
    CLAIM_WORKLIFE=0 and a 20s negotiator interval that gap is routine. Using
    SUBMIT as the lower bound admitted anything written during it — verified:
    a prior run's 0.2222 was reported as this run's result.
    """
    now = time.time()
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        gone = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                            uuid="aaaaaaaa-0000-0000-0000-00000000000a")
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.2222,
                     mtime=now - 300)               # written during the wait
        shutil.rmtree(gone)
        mine = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                            uuid="bbbbbbbb-0000-0000-0000-00000000000b")
        with open(os.path.join(mine, "jobstate.log"), "w") as handle:
            handle.write(f"{now - 900:.0f} INTERNAL *** DAGMAN_STARTED ***\n")
            handle.write(f"{now - 600:.0f} evaluate_higgs_evaluate SUBMIT "
                         f"29.0 condorpool - 13\n")
            handle.write(f"{now - 60:.0f} evaluate_higgs_evaluate EXECUTE "
                         f"29.0 condorpool - 13\n")
            handle.write(f"{now - 10:.0f} evaluate_higgs_evaluate "
                         f"JOB_TERMINATED 29.0 condorpool - 13\n")
        code, text = run(mine, submitted_after=1)
        assert code == 1, f"attributed a file written while Idle: {text}"
        assert "BEFORE this run's evaluate job started" in text, text
        assert "0.2222" not in text, f"leaked the value: {text}"
        print("  a file written while evaluate sat Idle is refused")


def test_a_retried_evaluate_anchors_to_the_attempt_that_succeeded():
    """DAGMan retries, so there can be several EXECUTE transitions.

    A file from the FAILED attempt is not the output of the attempt that
    succeeded, so the bound is the LAST execution start, not the first.
    """
    now = time.time()
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "shared")
        mine = make_run_dir(root, evaluate_exitcode=0, output_dir=shared,
                            uuid="bbbbbbbb-0000-0000-0000-00000000000b")
        with open(os.path.join(mine, "jobstate.log"), "w") as handle:
            handle.write(f"{now - 900:.0f} INTERNAL *** DAGMAN_STARTED ***\n")
            handle.write(f"{now - 700:.0f} evaluate_higgs_evaluate EXECUTE "
                         f"29.0 condorpool - 13\n")          # attempt 1
            handle.write(f"{now - 650:.0f} evaluate_higgs_evaluate "
                         f"JOB_FAILURE 1 condorpool - 13\n")
            handle.write(f"{now - 60:.0f} evaluate_higgs_evaluate EXECUTE "
                         f"29.0 condorpool - 13\n")           # attempt 2
            handle.write(f"{now - 10:.0f} evaluate_higgs_evaluate "
                         f"JOB_TERMINATED 29.0 condorpool - 13\n")
        # Written by the FAILED attempt, between the two executions.
        write_result(os.path.join(shared, "higgs_result.json"), auc=0.1111,
                     mtime=now - 660)
        code, text = run(mine, submitted_after=1)
        assert code == 1, f"attributed the failed attempt's file: {text}"
        assert "0.1111" not in text, f"leaked the value: {text}"
        print("  a retried evaluate anchors to the successful attempt")


if __name__ == "__main__":
    for fn in (test_result_is_reported_when_this_run_produced_it,
               test_previous_runs_result_is_refused_even_though_it_is_newer,
               test_failed_evaluate_job_is_not_reported_as_a_result,
               test_still_running_is_distinguished_from_finished,
               test_missing_database_is_refused,
               test_result_older_than_submission_is_refused,
               test_run_planned_without_an_isolated_output_dir_is_refused,
               test_one_runs_result_is_never_read_for_another,
               test_no_flag_can_point_the_tool_away_from_the_run,
               test_expect_output_dir_can_only_add_a_constraint,
               test_two_runs_sharing_an_output_dir_are_refused,
               test_a_leftover_file_from_a_DELETED_run_is_refused,
               test_a_clash_under_a_different_workflow_name_is_found,
               test_a_run_with_no_activity_record_is_refused,
               test_a_file_written_before_this_runs_evaluate_started_is_refused,
               test_a_file_written_after_the_run_finished_is_refused,
               test_a_genuine_result_written_during_stage_out_is_accepted,
               test_slack_is_not_allowed_on_the_lower_bound,
               test_a_file_written_while_evaluate_sat_idle_is_refused,
               test_a_retried_evaluate_anchors_to_the_attempt_that_succeeded):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
