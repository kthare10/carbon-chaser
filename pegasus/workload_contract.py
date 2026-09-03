"""The CLI contract between the workflow and the workload scripts.

The Pegasus workflow passes specific flags to each workload script. A stale
script fails the job at *runtime* with `unrecognized arguments:
--in-checkpoint` — after planning, submission and a stage-in, which is an
expensive and confusing place to discover it.

So the contract lives here, in one place, and is used three ways:

* `stage_submit_node.py` verifies the scripts it installs on the submit node
  actually accept these flags (via `--help`) and refuses to report success
  otherwise. (Under condorio that one copy is THE copy: HTCondor ships it
  into every job sandbox, so there is no per-worker drift to check.)
* `tests/test_workload_contract.py` checks that every flag the workflow
  generator emits is declared here, so the generator and the staging script
  cannot drift apart.
* Anyone reading it can see what each script must support.

Add a flag here in the same change that starts passing it.
"""

# Flags train_higgs.py must accept for the checkpoint chain to work.
REQUIRED_TRAINER_FLAGS = (
    "--workdir",
    "--data",
    "--in-checkpoint",      # resume point; absent only for the first segment
    "--out-checkpoint",     # this segment's product, the next one's input
    "--progress-file",      # per-segment, so no two jobs share an output
    "--max-seconds",        # deterministic segment boundary (exit 0)
    "--width",
    "--depth",
    "--batch",
)

REQUIRED_EVALUATOR_FLAGS = (
    "--workdir",
    "--data",
    "--checkpoint",         # the final checkpoint of the chain
    "--width",
    "--depth",
)

# Per-segment prediction: scores a small fixed validation sample against one
# segment's checkpoint, so the report can show the model improving (and
# moving between sites) over the run.
REQUIRED_PREDICTOR_FLAGS = (
    "--checkpoint",
    "--sample",             # higgs_val_sample.npz, built on the submit node
    "--out",
    "--width",
    "--depth",
)

REQUIRED_REPORTER_FLAGS = (
    "--predictions",        # one npz per segment, in chain order
    "--progress",           # one progress json per segment, in chain order
    "--result",             # higgs_result.json from the evaluator
    "--out",                # self-contained HTML report
)

WORKLOAD_SCRIPTS = {
    "train_higgs.py": REQUIRED_TRAINER_FLAGS,
    "evaluate_higgs.py": REQUIRED_EVALUATOR_FLAGS,
    "predict_higgs.py": REQUIRED_PREDICTOR_FLAGS,
    "report_higgs.py": REQUIRED_REPORTER_FLAGS,
}

# condorio layout: everything lives on the SUBMIT node only, and HTCondor
# file transfer carries it into each job's sandbox. Workers hold no
# pre-staged state at all.
SUBMIT_WORKLOAD_DIR = "/home/ubuntu/workload"   # stageable executables
SUBMIT_DATA_DIR = "/home/ubuntu/higgs-data"     # .npy cache + val sample


def missing_flags(help_text, required):
    """Flags absent from a script's --help output."""
    return [flag for flag in required if flag not in (help_text or "")]
