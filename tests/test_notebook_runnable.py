"""Check the generated notebook can actually run top-to-bottom.

A notebook is the shareable artifact — someone else runs it on their own
FABRIC credentials — so "it looks right" is not good enough. These checks
cannot execute it (that needs a testbed and ~2 hours), but they catch the
failures that make it fall over in the first seconds:

* a cell calling a helper defined in a LATER cell (a real bug: the offline
  checks cell was inserted above the setup cell that defines `sh`, so the
  notebook died with NameError on its first code cell),
* a syntax error in any cell,
* running a script locally that only works on the submit node — `workflow_generator.py`
  imports the Pegasus API at module scope, which is not installed on a
  laptop, so generating the workflow locally raises ModuleNotFoundError.

Deliberately conservative: it only tracks names the notebook itself defines,
so it flags use-before-definition without trying to be a type checker.
"""

import ast
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "pegasus",
                        "carbon-aware-pegasus.ipynb")

# Helpers the notebook defines for itself and then relies on.
TRACKED = {"sh", "submit", "sl", "fablib", "REPO", "run_dir"}

# Scripts that import the Pegasus API at module scope: submit-node only.
SUBMIT_NODE_ONLY = ("pegasus/workflow_generator.py",)


def code_cells():
    nb = json.load(open(NOTEBOOK))
    return [("".join(c["source"]), i)
            for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def executable(src):
    """`src` with comment-only lines removed.

    These tests assert on what the notebook DOES. Prose that explains a
    hazard names it, so a naive substring scan flags the explanation as
    the hazard — both of this file's path-attribution checks did exactly
    that once the submit cell started documenting why it no longer wipes.
    '#' opens a comment in Python cells and in the embedded shell script
    alike, so one rule covers both.
    """
    return "\n".join(line for line in src.splitlines()
                      if not line.strip().startswith("#"))


def test_every_cell_compiles():
    for src, index in code_cells():
        try:
            compile(src, f"<cell {index}>", "exec")
        except SyntaxError as exc:
            raise AssertionError(f"cell {index} has a syntax error: {exc}")
    print("  every code cell compiles")


def test_no_helper_is_used_before_it_is_defined():
    """The bug that made the notebook fail on its very first code cell."""
    defined = set()
    for src, index in code_cells():
        # Statement by statement, in order: a name assigned earlier in the
        # SAME cell is legitimately available later in it (REPO = ...; then
        # os.chdir(REPO)), so a whole-cell scan would cry wolf.
        for statement in ast.parse(src).body:
            used = {n.id for n in ast.walk(statement)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            premature = (used & TRACKED) - defined
            assert not premature, (
                f"cell {index} uses {sorted(premature)} before it is defined "
                f"— the notebook raises NameError here. Move the setup cell "
                f"above it.")
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.alias):
                    defined.add((node.asname or node.name).split(".")[0])
    print("  no cell uses a helper before it is defined")


def test_submit_node_only_scripts_are_not_run_locally():
    """workflow_generator.py imports Pegasus.api, which is not installed off the submit node."""
    for src, index in code_cells():
        for script in SUBMIT_NODE_ONLY:
            for line in src.splitlines():
                # `sh("python pegasus/workflow_generator.py ...")` runs it HERE.
                if f'sh("python {script}' in line or \
                        f"sh('python {script}" in line:
                    raise AssertionError(
                        f"cell {index} runs {script} locally via sh(), but it "
                        f"imports the Pegasus API at module scope and fails "
                        f"with ModuleNotFoundError off the submit node. "
                        f"Upload it and run it on the submit node instead.")
    print("  submit-node-only scripts are not invoked locally")


def test_run_dir_is_resolved_not_guessed():
    """`wf-runs` is not the run directory; Pegasus nests it several levels deep.

    The real path is wf-runs/<user>/pegasus/<wf-name>/run0001, so
    `pegasus-status -l wf-runs` and `cat wf-runs/*/higgs_result.json` both
    silently find nothing.
    """
    joined = "\n".join(src for src, _ in code_cells())
    for bad in ("pegasus-status -l wf-runs\"", "pegasus-status -l wf-runs ",
                "wf-runs/*/higgs_result.json",
                "pegasus-statistics -s all wf-runs"):
        assert bad not in joined, (
            f"notebook uses the un-resolved path {bad!r}; the run directory is "
            f"nested (wf-runs/<user>/pegasus/<wf>/run0001). Resolve it, e.g. "
            f"R=$(ls -d wf-runs/*/pegasus/*/run* | tail -1).")
    print("  run directory is resolved, not assumed to be wf-runs/")


def test_planning_failure_is_not_silent():
    """An empty run directory must stop the notebook, not flow onwards.

    `pegasus-plan` can fail (or `workflow_generator.py` can fail first, short-circuiting
    the `&&` chain), leaving no run directory at all. The resolver then yields
    an EMPTY string, and every later cell interpolates it into a command that
    quietly does the wrong thing — `pegasus-status -l` with no argument prints
    "(No matching jobs found in Condor Q)", which reads like a finished
    workflow rather than a failed submission.
    """
    for src, index in code_cells():
        if "run_dir =" not in src:
            continue
        assert "raise" in src, (
            f"cell {index} resolves run_dir but never checks it. If planning "
            f"failed, run_dir is '' and the following cells silently report "
            f"on nothing. Raise instead.")
        print("  an empty run directory raises instead of flowing onward")
        return
    raise AssertionError("no cell resolves run_dir")


def test_stale_results_cannot_masquerade_as_this_run():
    """A result must be attributable to THIS run — by namespacing, not by
    destroying the previous one.

    Outputs once landed at a fixed path, so any search found the PREVIOUS
    run's AUC, and the fix was to wipe wf-output before submitting. That
    wipe then destroyed a real artifact: the first run ever observed
    migrating had its report deleted by the next submission. Per-run
    epoch directories give the same guarantee without the collateral
    damage, so the invariant is now "namespaced and verified", and a
    reintroduced wipe is a regression.
    """
    joined = "\n".join(src for src, _ in code_cells())
    assert "find /home/ubuntu -name higgs_result.json" not in joined, (
        "notebook searches the home directory for the result, which finds a "
        "previous run's file. Read the explicit output path instead.")
    assert "wf-output/{submit_epoch}" in joined, (
        "outputs must be staged into a per-run epoch directory — that is "
        "what makes a result attributable without deleting anything")
    offenders = [line for line in executable(joined).splitlines()
                 if "rm -rf wf-runs" in line or "rm -rf wf-output" in line]
    assert not offenders, (
        "wiping wf-runs/wf-output destroys previous runs' evidence and buys "
        f"no isolation now that outputs are epoch-namespaced: {offenders}")
    assert "run_result.py" in joined, (
        "the notebook must verify the result via run_result.py, which reads "
        "the run's OWN braindump rather than trusting a path")
    print("  results are attributable by namespacing, not by wiping")


def submit_cell():
    for src, index in code_cells():
        if "pegasus-plan" in src:
            return src, index
    raise AssertionError("no cell runs pegasus-plan")


def test_planning_success_is_verified_not_inferred_from_a_directory():
    """A run directory existing does NOT mean planning and submission worked.

    `pegasus-plan` creates the run directory early and can fail afterwards —
    during submission, for instance — leaving a directory behind with no DAG
    in the queue. Checking only "did a directory appear" therefore passes on a
    failed submit. Worse, the command is piped into `tail`, so even a shell
    exit status would be tail's, not pegasus-plan's.

    So the notebook must capture the planner's own exit code AND confirm the
    submission marker in its output.
    """
    src, index = submit_cell()
    assert "PLAN_EXIT" in src, (
        f"cell {index} never captures pegasus-plan's exit code. It is piped "
        f"into `tail`, so the status is masked; write the log to a file and "
        f"echo $? instead.")
    assert "submitted to cluster" in src, (
        f"cell {index} does not confirm the submission marker. Planning can "
        f"exit 0 having produced a directory but no queued DAG.")
    print("  planning success is verified from exit code + submission marker")


def test_result_is_checked_for_freshness():
    """Clearing wf-output is necessary but not sufficient.

    The result path is fixed, so if the clear ever fails — or the result cell
    is re-run against an older state — a previous run's AUC is printed as this
    run's. Comparing the file's mtime against the submission time settles it
    regardless of whether the clear worked.
    """
    joined = "\n".join(src for src, _ in code_cells())
    assert "submit_epoch" in joined, (
        "notebook never records when the workflow was submitted, so it cannot "
        "tell a fresh result from a stale one")
    # The comparison itself lives in run_result.py (and is covered directly by
    # tests/test_run_result.py); what the notebook must do is PASS the
    # submission time, otherwise that check silently degrades to a no-op.
    assert "--submitted-after" in joined, (
        "notebook never passes the submission time to run_result.py, so its "
        "staleness check is disabled and an older higgs_result.json would be "
        "accepted")
    print("  result freshness is checked against the submission time")


def test_dag_failure_is_reported_as_failure():
    """A failed workflow must not read as an unfinished one.

    If jobs exhaust their retries the DAG ends in Failure, and a result cell
    that only says "no result yet" describes that as though it were still
    running. `pegasus-status` reports the terminal state; use it.
    """
    joined = "\n".join(src for src, _ in code_cells())
    assert "Failure" in joined, (
        "notebook never distinguishes a FAILED dag from an unfinished one — "
        "check the pegasus-status state and say so explicitly")
    print("  a failed DAG is reported as failed, not as 'not finished yet'")


def test_result_is_attributed_to_this_run_not_merely_recent():
    """Recency is not provenance.

    `rm -rf wf-runs` does not stop an already-running DAG, and outputs stage
    into a fixed run-agnostic directory — so a PREVIOUS workflow can finish
    and write the result AFTER the new submission. It is newer than submit
    time and still the wrong run's answer. Only this run's own stampede DB
    settles it, which is what pegasus/run_result.py consults.
    """
    joined = "\n".join(src for src, _ in code_cells())
    assert "run_result.py" in joined, (
        "the result cell does not use run_result.py, so it cannot tell this "
        "run's output from a concurrent older run's")
    assert "--run-dir" in joined, (
        "run_result.py must be given THIS run's directory — that is what ties "
        "the result to the submission")
    src, index = submit_cell()
    assert "--output-dir" in src, (
        f"cell {index} plans without --output-dir, so every run stages into "
        f"the SAME wf-output directory and no later check can prove which run "
        f"wrote a result. run_result.py refuses such runs outright.")
    # And it must not hand run_result.py a path: that flag is gone precisely
    # because it skipped attribution, so a notebook still passing one would be
    # relying on a bypass.
    for src, index in code_cells():
        live = executable(src)
        if "run_result.py" not in live:
            continue
        assert "--output-dir" not in live or "--expect-output-dir" in live, (
            f"cell {index} passes a bare --output-dir to run_result.py, which "
            f"would bypass braindump attribution")
    print("  the result is attributed to this run, not merely recent")


def test_a_previous_workflow_is_stopped_before_resubmitting():
    """Otherwise the old DAG keeps running and writes into the shared output."""
    src, index = submit_cell()
    assert "condor_rm" in src or "pegasus-remove" in src, (
        f"cell {index} wipes wf-runs/wf-output but never stops a workflow that "
        f"is still running; that DAG keeps going and stages its outputs into "
        f"the same directory, after this submission.")
    print("  any previously running workflow is stopped before resubmitting")


if __name__ == "__main__":
    for fn in (test_every_cell_compiles,
               test_no_helper_is_used_before_it_is_defined,
               test_submit_node_only_scripts_are_not_run_locally,
               test_run_dir_is_resolved_not_guessed,
               test_planning_failure_is_not_silent,
               test_stale_results_cannot_masquerade_as_this_run,
               test_planning_success_is_verified_not_inferred_from_a_directory,
               test_result_is_checked_for_freshness,
               test_dag_failure_is_reported_as_failure,
               test_result_is_attributed_to_this_run_not_merely_recent,
               test_a_previous_workflow_is_stopped_before_resubmitting):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
