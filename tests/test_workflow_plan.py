"""Guard the invariants that make the Pegasus workflow plannable.

Pegasus builds its DAG from which job *produces* each file, so two jobs
declaring the same output is a fatal planning error. The first version of the
carbon-aware workflow gave every training segment the same `checkpoint.pt`
and `progress.json`, and could not be planned at all:

    FATAL: Output file progress.json found for job train_1 has already been
           associated as output for a previous job train_0

The generator self-checks, so the failure surfaces locally with an
actionable message instead of after uploading to a submit node. This tests
that check directly — no Pegasus install needed (everything is loaded from
workflow_generator.py by AST, because it imports Pegasus.api at module
scope and that only exists on the submit node).
"""

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GENERATOR = os.path.join(os.path.dirname(__file__), "..", "pegasus",
                         "workflow_generator.py")


def load_function(name, prelude=""):
    """Extract one module-level function from the generator without
    importing it (Pegasus.api only exists on the submit node)."""
    tree = ast.parse(open(GENERATOR).read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    body = ast.parse(prelude).body if prelude else []
    module = ast.Module(body=body + [fn], type_ignores=[])
    namespace = {}
    exec(compile(module, "<workflow_generator>", "exec"), namespace)
    return namespace[name]


def test_distinct_outputs_are_accepted():
    check = load_function("check_unique_producers",
                          "from collections import defaultdict")
    plan = [(f"train_{i:03d}",
             [f"ckpt_{i+1:03d}.pt", f"progress_{i+1:03d}.json"])
            for i in range(5)]
    plan += [(f"predict_{i:03d}", [f"predictions_{i+1:03d}.npz"])
             for i in range(5)]
    plan.append(("evaluate", ["higgs_result.json"]))
    plan.append(("report", ["higgs_report.html"]))
    assert check(plan) is True
    print("  a chain with per-segment filenames passes")


def test_shared_checkpoint_name_is_rejected():
    """The exact bug: every segment writing checkpoint.pt."""
    check = load_function("check_unique_producers",
                          "from collections import defaultdict")
    plan = [(f"train_{i}", ["checkpoint.pt"]) for i in range(3)]
    try:
        check(plan)
    except ValueError as exc:
        assert "checkpoint.pt" in str(exc)
        assert "one producing job" in str(exc)
        print("  shared checkpoint.pt rejected with an actionable message")
        return
    raise AssertionError("accepted a workflow Pegasus would refuse to plan")


def test_shared_progress_name_is_rejected():
    """progress.json tripped first in the real failure."""
    check = load_function("check_unique_producers",
                          "from collections import defaultdict")
    plan = [("train_0", ["ckpt_001.pt", "progress.json"]),
            ("train_1", ["ckpt_002.pt", "progress.json"])]
    try:
        check(plan)
    except ValueError as exc:
        assert "progress.json" in str(exc)
        print("  shared progress.json rejected")
        return
    raise AssertionError("accepted duplicate progress.json")


def test_error_names_every_clashing_file():
    check = load_function("check_unique_producers",
                          "from collections import defaultdict")
    plan = [("a", ["x.pt", "y.json"]), ("b", ["x.pt", "y.json"])]
    try:
        check(plan)
    except ValueError as exc:
        assert "x.pt" in str(exc) and "y.json" in str(exc), str(exc)
        print("  all clashing files are named, not just the first")
        return
    raise AssertionError("accepted duplicates")


def test_kickstart_timeout_is_not_the_segment_boundary():
    """kickstart's SIGTERM must land well after the trainer's own deadline.

    Regression test for a run that failed three times over. Per the kickstart
    manpage, a job running past `-k` "exits with a non-zero exit status" no
    matter how gracefully it handles SIGTERM — so if that timeout coincides
    with the trainer's `--max-seconds`, every segment fails and DAGMan burns
    its retries. Pegasus puts SIGTERM at `checkpoint.time`, so that value must
    exceed the segment budget by a real margin, not by one second.
    """
    segment_timeouts = load_function("segment_timeouts")
    for minutes in (1, 2, 5, 20, 60):
        checkpoint_time, maxwalltime = segment_timeouts(minutes)
        assert checkpoint_time >= minutes * 2, (
            f"{minutes}min segment: SIGTERM at {checkpoint_time}min leaves no "
            f"headroom for dataset transfer+load, final eval and checkpoint "
            f"save — those sit outside the trainer's own clock")
        assert maxwalltime > checkpoint_time, (
            f"{minutes}min: SIGKILL must come after SIGTERM, got "
            f"maxwalltime={maxwalltime} <= checkpoint.time={checkpoint_time}")
        # Pegasus: SIGKILL at checkpoint.time + (maxwalltime-checkpoint.time)/2
        kill_at = checkpoint_time + (maxwalltime - checkpoint_time) / 2
        assert kill_at > checkpoint_time, "no window to write a checkpoint"
    print("  kickstart SIGTERM/SIGKILL both clear the segment budget")


def transformation_calls():
    """Every Transformation(...) call in the generator, plus every name the
    `stageable(...)` helper is invoked with."""
    tree = ast.parse(open(GENERATOR).read())
    calls, names = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) == "Transformation":
            calls.append({kw.arg: kw.value for kw in node.keywords})
        if (getattr(node.func, "id", None) == "stageable"
                and node.args and isinstance(node.args[0], ast.Constant)):
            names.append(node.args[0].value)
    return calls, names


def test_transformations_are_stageable_from_local():
    """Under condorio the executables exist ONLY on the submit node, so
    every transformation must be stageable with its PFN on site `local`.

    Both halves of this were real failures, one each way:

    * `is_stageable=True` with the PFN resolved against a WORKER-only path
      left `stage_in_local_local_*` jobs retrying forever — the PFN's site
      must be where the file actually is.
    * `is_stageable=False` (the previous design) means "installed on the
      worker", which is exactly the per-node pre-staged state the condorio
      restructure removes; regressing to it would need the fan-out deployer
      back.
    """
    calls, names = transformation_calls()
    assert calls, "found no Transformation(...) call in the generator"
    for kwargs in calls:
        stageable_kw = kwargs.get("is_stageable")
        assert isinstance(stageable_kw, ast.Constant), "not a literal"
        assert stageable_kw.value is True, (
            "is_stageable must be True: under condorio the scripts exist "
            "only on the submit node and HTCondor ships them per job")
        site = kwargs.get("site")
        assert isinstance(site, ast.Constant) and site.value == "local", (
            "transformation PFNs must be declared on site 'local' — that is "
            "where stage_submit_node.py puts the scripts; any other site "
            "points the planner at a path that does not exist there")
    assert sorted(names) == ["evaluate_higgs", "predict_higgs",
                             "report_higgs", "train_higgs"], names
    print("  all 4 transformations stageable from the submit node (local)")


def test_every_job_runs_in_the_container():
    """The container is the worker runtime: without it, jobs depend on each
    worker's system python — the drift class a live pool actually hit, with
    every segment dying on `import numpy` while an interactive check (as
    ubuntu, whose --user site-packages the condor user cannot see) looked
    healthy. Two invariants:

    * every Transformation is built with a container, and the GPU/CPU split
      is intentional — train/evaluate get `--nv` (host driver bound in, so
      CUDA and the real NVML sensor work), predict/report get the SAME sif
      without it, so the CPU-only visual jobs cannot touch a GPU a
      concurrent trainer is using;
    * the .def the sif is built from exists in the repo, so the image is
      reproducible rather than an artifact someone once built.
    """
    tree = ast.parse(open(GENERATOR).read())
    containers = {}          # var name -> has --nv
    stageable_args = {}      # transformation name -> container var name
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) == "Container":
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            args_kw = kwargs.get("arguments")
            containers[node.args[0].value] = (
                isinstance(args_kw, ast.Constant)
                and "--nv" in args_kw.value)
        if (getattr(node.func, "id", None) == "stageable"
                and len(node.args) >= 2):
            stageable_args[node.args[0].value] = node.args[1].id
    assert len(containers) == 2, containers
    nv = [n for n, has in containers.items() if has]
    assert len(nv) == 1, f"exactly one container should carry --nv: {containers}"

    calls, _ = transformation_calls()
    for kwargs in calls:
        assert "container" in kwargs, (
            "a Transformation without a container runs on the worker's "
            "system python — the exact drift the container closes")
    gpu_var, cpu_var = "gpu_container", "cpu_container"
    assert stageable_args.get("train_higgs") == gpu_var
    assert stageable_args.get("evaluate_higgs") == gpu_var
    assert stageable_args.get("predict_higgs") == cpu_var
    assert stageable_args.get("report_higgs") == cpu_var

    def_path = os.path.join(os.path.dirname(__file__), "..", "Apptainer",
                            "higgs_container.def")
    assert os.path.exists(def_path), "Apptainer/higgs_container.def missing"
    def_text = open(def_path).read()
    for needed in ("torch", "numpy", "pandas", "nvidia-ml-py"):
        assert needed in def_text or needed.replace("-", "_") in def_text, (
            f"container def does not provide {needed}")
    print("  all 4 transformations containerized; --nv only on the GPU pair")


def test_job_ads_capture_carbon_and_power_at_match():
    """job_machine_attrs is what turns slot ClassAds into per-job accounting:
    without it, condor_history has no record of the carbon intensity or GPU
    power of the machine each segment matched to, and the dashboard's
    gCO2-per-segment numbers would have to be reconstructed by timestamp
    correlation instead of read from the job ad."""
    text = open(GENERATOR).read()
    tree = ast.parse(text)
    attrs = next((n.value.value for n in tree.body
                  if isinstance(n, ast.Assign)
                  and any(getattr(t, "id", None) == "JOB_MACHINE_ATTRS"
                          for t in n.targets)), None)
    assert attrs, "JOB_MACHINE_ATTRS constant missing"
    for needed in ("CarbonIntensity", "GPUWatts", "FabricSite",
                   "CarbonInjected"):
        assert needed in attrs, (
            f"{needed} not captured into job ads"
            + (" — a segment matched against an injected demo price would "
               "record the number WITHOUT the label, i.e. a fabricated "
               "measurement in the permanent record"
               if needed == "CarbonInjected" else ""))
    assert 'key="job_machine_attrs"' in text, (
        "no transformation sets the job_machine_attrs condor profile")
    print("  job ads record carbon/power/site at every match")


if __name__ == "__main__":
    for fn in (test_distinct_outputs_are_accepted,
               test_shared_checkpoint_name_is_rejected,
               test_shared_progress_name_is_rejected,
               test_error_names_every_clashing_file,
               test_transformations_are_stageable_from_local,
               test_every_job_runs_in_the_container,
               test_job_ads_capture_carbon_and_power_at_match,
               test_kickstart_timeout_is_not_the_segment_boundary):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
