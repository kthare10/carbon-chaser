"""Keep the workflow generator and the workload deployment in step.

The Pegasus workflow passes flags to the trainer/evaluator. If a worker has an
older copy of those scripts the job fails at RUNTIME (`unrecognized
arguments: --in-checkpoint`) — after planning, submission and stage-in.

Two ways that goes wrong, both guarded:

1. The generator starts passing a flag nobody declared, so the deployer never
   verifies it and a stale worker is reported healthy. Checked statically here.
2. The installed script on the submit node predates the flag. Checked
   there by pegasus/stage_submit_node.py, which runs `--help` and refuses
   to report success if a declared flag is missing.
"""

import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pegasus"))

from workload_contract import (REQUIRED_EVALUATOR_FLAGS,  # noqa: E402
                              REQUIRED_PREDICTOR_FLAGS,
                              REQUIRED_REPORTER_FLAGS,
                              REQUIRED_TRAINER_FLAGS, WORKLOAD_SCRIPTS,
                              missing_flags)

WORKFLOW = os.path.join(HERE, "..", "pegasus", "workflow_generator.py")
WORKLOAD = os.path.join(HERE, "..", "carbon_chaser", "workload")

# Which local variable in create_workflow() builds which kind of job. The
# attribution matters: a flag passed to the predictor must be verified
# against predict_higgs.py, not the trainer.
JOB_VARS = {"job": "trainer", "pjob": "predictor",
            "final": "evaluator", "report": "reporter"}


def flags_emitted_by_workflow():
    """Flags the generator passes via add_args, attributed to the right task.

    AST-based and scoped to create_workflow(): a regex over the whole file
    also picks up the generator's OWN argparse options (--segments, ...)
    and mis-attributes them.
    """
    tree = ast.parse(open(WORKFLOW).read())
    create = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "create_workflow")
    per_kind = {}
    for node in ast.walk(create):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_args"
                and isinstance(node.func.value, ast.Name)):
            continue
        kind = JOB_VARS.get(node.func.value.id)
        assert kind is not None, (
            f"add_args on unknown variable '{node.func.value.id}' — add it "
            f"to JOB_VARS so its flags are verified against the right script")
        bucket = per_kind.setdefault(kind, set())
        for arg in node.args:
            if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    and arg.value.startswith("--")):
                bucket.add(arg.value)
    for kind in JOB_VARS.values():
        assert per_kind.get(kind), (
            f"found no {kind} add_args calls in create_workflow()")
    return per_kind


def flags_supported_by_script(path):
    """Flags an argparse-based script actually defines."""
    tree = ast.parse(open(path).read())
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith("--")):
                    found.add(arg.value)
    return found


def test_every_flag_the_workflow_passes_is_declared():
    per_kind = flags_emitted_by_workflow()
    contracts = {"trainer": REQUIRED_TRAINER_FLAGS,
                 "evaluator": REQUIRED_EVALUATOR_FLAGS,
                 "predictor": REQUIRED_PREDICTOR_FLAGS,
                 "reporter": REQUIRED_REPORTER_FLAGS}
    for kind, declared in contracts.items():
        undeclared = per_kind[kind] - set(declared)
        assert not undeclared, (
            f"workflow passes {sorted(undeclared)} to the {kind} but the "
            f"contract does not declare them, so stage_submit_node.py will "
            f"not verify them and a stale script would be reported healthy")
    print("  workflow flags all declared: "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(per_kind.items())))


def test_scripts_implement_the_contract():
    for script, required in WORKLOAD_SCRIPTS.items():
        supported = flags_supported_by_script(os.path.join(WORKLOAD, script))
        gaps = [f for f in required if f not in supported]
        assert not gaps, f"{script} does not implement {gaps}"
        print(f"  {script} implements all {len(required)} contract flags")


def test_contract_covers_the_checkpoint_chain():
    """These four flags are what make resume-across-segments work, and what
    the first (unplannable) design lacked."""
    for flag in ("--in-checkpoint", "--out-checkpoint", "--progress-file",
                 "--max-seconds"):
        assert flag in REQUIRED_TRAINER_FLAGS, flag
    assert "--checkpoint" in REQUIRED_EVALUATOR_FLAGS
    print("  chain flags are part of the verified contract")


def test_missing_flags_helper_detects_stale_help_output():
    """What stage_submit_node.py relies on when probing the scripts."""
    stale = ("usage: train_higgs.py [-h] [--workdir WORKDIR] [--data DATA]\n"
             "                      [--steps STEPS]\n")
    gaps = missing_flags(stale, REQUIRED_TRAINER_FLAGS)
    assert "--in-checkpoint" in gaps and "--out-checkpoint" in gaps, gaps
    fresh = " ".join(REQUIRED_TRAINER_FLAGS)
    assert missing_flags(f"usage: x {fresh}", REQUIRED_TRAINER_FLAGS) == []
    print("  stale --help output is detected as a contract gap")


if __name__ == "__main__":
    for fn in (test_every_flag_the_workflow_passes_is_declared,
               test_scripts_implement_the_contract,
               test_contract_covers_the_checkpoint_chain,
               test_missing_flags_helper_detects_stale_help_output):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
