#!/usr/bin/env python3
"""Generate carbon-aware-pegasus.ipynb — the shareable end-to-end notebook.

Written as a generator rather than by hand so the notebook and the scripts
cannot drift: every code cell calls the same modules used in anger.

The notebook is kept deliberately short — it is the guided tour. The
post-mortems behind each design choice live in ../README.md.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "carbon-aware-pegasus.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Carbon-Aware Scientific Workflows on FABRIC

Train a real HEP model as a **Pegasus workflow** whose tasks run at whichever
FABRIC site currently has the **cleanest electricity** — with both halves of
the carbon arithmetic measured rather than assumed.

| Quantity | Where it comes from |
|---|---|
| Grid carbon intensity | EIA hourly generation mix × published emission factors, per balancing authority |
| Node power | NVML board power, read in-guest (FABRIC GPUs are PCI passthrough) |
| Science | UCI **HIGGS** — 11M simulated collision events ([doi:10.24432/C5V312](https://doi.org/10.24432/C5V312)); Baldi, Sadowski & Whiteson, *Nature Communications* **5**, 4308 (2014), [doi:10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308) |

**There is no simulator in this stack.** With no carbon feed the run refuses
to start; with no NVML reading, energy accounting for that site pauses rather
than guessing. The single demo affordance is a *disclosed* carbon override
(the dashboard's inject button): it replaces one site's matchmaking input,
advertises `CarbonInjected = true` alongside the real trace value, and expires
by itself. Power and energy stay measurement-only, so a triggered migration
still has a real, measured cost.

## How the carbon-awareness works

Pegasus and HTCondor already contain the whole mechanism — nothing here is a
custom scheduler:

1. **The workflow is a chain of time-boxed training segments** linked by
   *distinct* checkpoint files: `train_000 → ckpt_001.pt → train_001 → …`.
   The data dependency **is** the resume mechanism, and each segment is a
   separate placement decision.
2. **Each worker advertises `CarbonIntensity` and `GPUWatts`** as HTCondor
   ClassAd attributes (a `STARTD_CRON` job), so placement is ordinary
   matchmaking: `RANK = -CarbonIntensity`.
3. **A "migration" is just the next segment landing at a cleaner site** with
   its predecessor's checkpoint. There is no external control loop.

## Three defaults that quietly un-do the policy

Each of these was hit for real on this pool, and none of them reports an
error — jobs run, exit 0, and land somewhere plausible while the carbon
policy is never consulted. `provision.py` sets all three correctly; they are
listed here because they are *how* a carbon-aware scheduler turns out not to
be one. Full post-mortems, with the measured numbers, are in the
[README](../README.md).

| Default | What goes wrong | Fix |
|---|---|---|
| `CLAIM_WORKLIFE = 1200` | the schedd reuses its claim, so a whole workflow fits inside one match and `RANK` is evaluated once, for the first segment | `CLAIM_WORKLIFE = 0` **in the worker config** — it is a startd knob, so setting it on the submit node changes nothing |
| `NEGOTIATOR_PRE_JOB_RANK` | consulted *before* the job's rank, and its default prefers a smaller slot — measured at ~212,000 points of bias against a carbon difference worth ~13 | replaced with a slot-size-neutral expression in the negotiator (submit-node) config |
| per-zone trace replay | EIA coverage differs per balancing authority, so sites drift to different replayed hours (7.9 and 15.9 apart, measured) and then rank on phase rather than carbon | replay position derived from the *global* trace bounds plus absolute wall-clock, so every node evaluates the **same function of time** |

If your pool predates these fixes, re-run `provision.py --skip-create` before
expecting migrations.

**Same function of time does not mean equal timestamps**, and expecting it to
will send you chasing a non-bug. `STARTD_CRON` samples each node every 60 s
and the nodes are not in phase, so their `CarbonTimestamp` values legitimately
differ by up to `60 × CARBON_ACCEL` of replayed time — 5 hours at the default
`ACCEL = 300` — and two nodes either side of the replay wrap differ by a whole
span. What must hold is that each node's position is what *its own*
`CarbonSampledAt` implies, on one shared span and accel; that is what
`check_pool_match.py` verifies, and comparing the raw values to each other
instead reports a healthy pool as broken.

Two more things that look broken and are not:

* `condor_q -af Rank` *evaluates* the expression with no target machine, so a
  perfectly healthy rank prints as `undefined`. Use `-af:r` for the raw
  expression before concluding it went missing.
* A worker with no carbon feed publishes **no** `CarbonIntensity` attribute
  rather than `0` — absent is deliberate, since `0 gCO₂/kWh` would rank as the
  cleanest grid in the pool and attract every job.
"""),

    md("""
## 0. Prerequisites

* FABRIC credentials configured (`fabric_rc`, bastion key, slice keys) — see
  FABRIC's [configure_and_validate](https://github.com/fabric-testbed/jupyter-examples/blob/main/configure_and_validate/configure_and_validate.ipynb) notebook.
* A **free EIA API key**: https://www.eia.gov/opendata/register.php
* This repo checked out, with `fabrictestbed-extensions` installed.

Keep the key in `~/.secrets/eia`:

```bash
mkdir -p ~/.secrets && echo YOUR_KEY > ~/.secrets/eia && chmod 600 ~/.secrets/eia
```

The fetch cell reads it with `--key-file`, so it appears in neither argv nor
shell history, and request failures are reported with the key redacted so it
cannot land in this notebook's saved output. (It is still sent to EIA as an
HTTPS query parameter — that is their auth design.) `EIA_API_KEY` also works.
"""),
    code("""
import os, shlex, subprocess, sys

def find_repo():
    "Locate the checkout by MARKER, so this works from any subdirectory."
    probe = os.path.abspath(os.getcwd())
    while probe != os.path.dirname(probe):
        if os.path.exists(os.path.join(probe, "pegasus",
                                       "workflow_generator.py")):
            return probe
        probe = os.path.dirname(probe)
    raise RuntimeError("run this notebook from inside the carbon-chaser "
                       "checkout (any subdirectory works)")

REPO = find_repo()
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

def sh(cmd, timeout=None):
    "Run a repo script and stream its output."
    # THIS kernel's interpreter: on macOS `python` often does not exist, and
    # a system python lacks the repo's deps (yaml, fablib) when it does.
    if cmd.startswith("python "):
        cmd = shlex.quote(sys.executable) + cmd[len("python"):]
    print(f"$ {cmd}")
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True,
                       timeout=timeout)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        print("STDERR:", p.stderr[-2000:])
    return p.returncode

print(sys.version.split()[0], "|", REPO)
"""),

    md("""
The offline checks need no testbed and no API key, so run them first. They
cover what would otherwise waste a provisioning cycle: an unplannable DAG, a
workload CLI contract that has drifted from what the generator passes, and the
scheduling profiles that decide whether placement is carbon-aware at all.
"""),
    code("""
sh("python tests/run_all.py")
"""),

    md("""
## 1. Fetch real grid-carbon data

`fetch_traces.py` pulls the hourly generation mix per balancing authority from
the EIA API and converts it with published emission factors. The intensity is
therefore **derived** (measured fuel mix × standard factors), not a measured
CO₂ value — the sidecar it writes says so, and the dashboard repeats it.

Zone ids are validated against Electricity Maps' public zone list; a wrong id
otherwise yields a silently empty feed.
"""),
    code("""
sh("python fabric/fetch_traces.py --source eia --days 7 "
   "--key-file ~/.secrets/eia")
sh("head -3 config/traces/eia.csv; cat config/traces/eia.csv.meta.json")
"""),

    md("""
## 2. Provision the cluster

One submit node (HTCondor central manager + Pegasus) plus one **GPU worker per
site**. GPUs are mandatory: measured power is the point, and there is no
assumed-power fallback.

`--sites auto` picks worker sites from **live availability** (free GPUs per
model, cores/ram/disk), and then ranks the candidate sets by the only property
that actually produces migrations: **how often the cleanest site switches** in
the trace. That criterion replaced "maximize grid-zone diversity", which
sounds equivalent and is not — a perfectly diverse set can have one zone that
is cleanest 100% of the time, in which case the right answer never changes and
nothing ever migrates. The selector also excludes, with printed reasons, any
candidate whose zone the trace does not cover: such a worker provisions fine
and then never wins a match.

`python pegasus/select_sites.py` is the read-only dry run of the same choice;
an explicit `--sites CLEM,TACC,UTAH` still works.

**Re-running this cell is safe.** An existing slice is reused and only the
configuration is re-applied, which is also how config fixes reach a live pool.
Site selection happens only when the slice is actually created.
"""),
    code("""
sh("python pegasus/provision.py --sites auto --submit-site STAR",
   timeout=7200)
"""),

    md("""
### Slice inventory

fablib's tables show what you need when a node has to be poked by hand: the
management IP that SSH reaches (via the FABRIC bastion, with the full ssh
command per node) and the FABNetv4 dataplane addresses the pool itself uses.
"""),
    code("""
from fabrictestbed_extensions.fablib.fablib import FablibManager
fablib = FablibManager()
sl = fablib.get_slice(name="carbon-chaser-pegasus")
sl.list_nodes();
sl.list_networks();
sl.list_interfaces();
"""),

    md("""
## 3. Install the GPU stack

Do **not** use `ubuntu-drivers autoinstall` or the CUDA repo's `cuda`
metapackage here. Three traps, each of which cost a debugging cycle:

1. **DKMS/kernel skew** — DKMS builds against the running kernel, but apt often
   pulls a *newer* one in the same transaction, so the reboot lands on a kernel
   with no module at all.
2. **Newest ≠ usable** — the latest driver often has no prebuilt module for the
   running kernel, giving `Driver/library version mismatch`.
3. **Variant mismatch** — `linux-modules-nvidia-535-open` with the proprietary
   `nvidia-driver-535` gives *"held broken packages"*.

`install_gpu_stack.py` picks the highest driver that **has a prebuilt module
for the running kernel**, matches the userspace variant to it, and is
idempotent (a no-op once NVML reports watts).
"""),
    code("""
# --nodes auto = every *-gpu node in the slice, so a dynamically chosen site
# set needs no hand-maintained list. --slice-name is REQUIRED: without it the
# script falls back to the legacy slice in sites.yaml.
sh("python fabric/install_gpu_stack.py --reboot "
   "--slice-name carbon-chaser-pegasus --nodes auto",
   timeout=7200)
"""),

    md("""
## 4. Repair name resolution after the reboot

FABRIC images set cloud-init `manage_etc_hosts: True`, so `/etc/hosts` is
**regenerated on every boot** — and step 3 reboots the nodes. Workers then
cannot resolve the collector and never join, with a symptom that does not look
like a hosts problem:

```
StartLog:  Can't resolve collector submit; skipping update
MasterLog: WARNING: Saw slow DNS query ... getaddrinfo(submit) took 5.09s
```

This step disables cloud-init's hosts management and writes the pool entries
into both `/etc/hosts` and the cloud-init template. **Run it after anything
that reboots nodes.**
"""),
    code("""
sh("python pegasus/fix_pool_hosts.py", timeout=3600)
"""),

    md("""
## 5. Stage the submit node — the workers get nothing

The workflow runs with `pegasus.data.configuration = condorio`: the submit node
is the staging site, and **HTCondor file transfer carries every job's inputs
into its sandbox and its outputs back**. So this is the only machine that holds
pre-positioned state:

1. **Workload scripts** → `/home/ubuntu/workload`, always re-uploaded, then
   verified against the CLI contract via `--help`
   (`pegasus/workload_contract.py` is the single source of truth). A script
   that landed but does not accept the workflow's flags is reported as a
   failure, not a success.
2. **Dataset** → `HIGGS.csv`, 2.6 GB from UCI, downloaded on the node and
   skipped when present. No job reads the CSV; it is only the source for (3).
3. **The derived inputs jobs actually consume**, built once here because a
   fresh condorio sandbox could never amortize them: `HIGGS.csv.raw.npy`
   (~1 GB) and the ~50k-event `higgs_val_sample.npz` the predict jobs score.
4. **The Apptainer container every job runs in** → `higgs_container.sif`,
   built from `Apptainer/higgs_container.def` (torch, numpy, pandas, pynvml)
   and smoke-tested. The container *is* the worker runtime, so a worker needs
   only HTCondor, Apptainer and the NVIDIA driver — there is no system python
   to drift. Built here rather than pulled `docker://` per job so Docker Hub
   rate limits stay off the critical path.

The trade-off is explicit and **measured**: each training segment pulls ~1 GB
across the FABNet dataplane into its sandbox instead of reading a node-local
copy — and HTCondor accounts for every byte (`TransferInputSizeMB`,
`BytesRecvd`), which is what lets the dashboard show a migration's true
transfer cost instead of assuming it.
"""),
    code("""
sh("python pegasus/stage_submit_node.py", timeout=7200)
"""),

    md("""
`--verify-only` asks whether the submit node is consistent *without* uploading
or downloading anything: the contract check plus a listing of the derived
inputs the replica catalog will point at.
"""),
    code("""
sh("python pegasus/stage_submit_node.py --verify-only", timeout=1800)
"""),

    md("""
## 6. Confirm the pool is carbon-aware

Each worker should report its site, a live `CarbonIntensity` from the EIA
trace, and measured `GPUWatts`. A worker whose feed is missing publishes **no**
attribute — deliberately absent rather than `0`, because `0 gCO₂/kWh` would
rank as the cleanest grid in the pool and attract every job.
"""),
    code("""
from fabrictestbed_extensions.fablib.fablib import FablibManager
fablib = FablibManager()
sl = fablib.get_slice(name="carbon-chaser-pegasus")
submit = sl.get_node(name="submit")

out, _ = submit.execute(
    "condor_status -af Name FabricSite CarbonIntensity GPUWatts DetectedGPUs "
    "| sort -k3 -n", quiet=True)
print(out)
print("Cleanest site (what RANK = -CarbonIntensity picks):")
out, _ = submit.execute(
    'condor_status -af Name CarbonIntensity '
    '-constraint "CarbonIntensity =!= UNDEFINED" | sort -k2 -n | head -1',
    quiet=True)
print(out)
"""),

    md("""
Then ask the pool whether the workflow's requirements can match anything
**before** submitting. This prevents the quietest failure in HTCondor: the pool
once advertised `HasGPU = ifThenElse(DetectedGPUs > 0, ...)`, but
`DetectedGPUs` is a *string* of GPU ids (`"GPU-1cae1040"`), so the comparison
evaluated to **error** — and an error in a requirements expression means NEVER
MATCH. The job was submitted, accepted, and sat Idle forever: nothing held,
nothing failed, nothing logged.

The check evaluates the same requirement constants the generator emits, so the
two cannot drift apart. It also checks **replay coherence**: one shared
`CarbonSpan` and `CarbonAccel`, each node's position matching its own
`CarbonSampledAt`, and each node's clock agreeing with the collector's (from
`MyCurrentTime` vs `LastHeardFrom`, since a skewed clock produces a *self
-consistent* node that is nonetheless hours away from its peers). A mismatched
`CARBON_ACCEL` or a bad clock is caught before submission, rather than
inverting a ranking on phase alone.
"""),
    code("""
sh("python pegasus/check_pool_match.py")
"""),

    md("""
## 7. Generate and submit the workflow

`segments × minutes` sets how often placement is reconsidered: each segment is
one matchmaking decision, so more segments means more carbon responsiveness at
the cost of more transfers — and under condorio that cost is real and measured
(~1 GB into each segment's sandbox). That is exactly the trade-off worth
studying.

Besides the training chain the generator emits a **predict job per segment**
(scoring a fixed 50k-event sample against that segment's checkpoint, CPU-only
so it never competes for a GPU) and a final **report job** that renders a
self-contained HTML page.

The generator self-checks that no two jobs declare the same output file, so an
unplannable workflow fails here with an actionable message rather than after
upload. It writes `pegasus.properties` (condorio, stated explicitly) plus
`sites.yml`, `replicas.yml` and `transformations.yml`; the plan command passes
`--conf` so a stray user-level properties file cannot silently change the
staging model.
"""),
    code("""
# workflow_generator.py imports the Pegasus API at module scope, so it runs on
# the SUBMIT NODE, not here -- a laptop has no Pegasus install.
submit.upload_file("pegasus/workflow_generator.py", "workflow_generator.py")

SEGMENTS, MINUTES = 6, 2

# Timestamp the submission: it namespaces THIS run's output directory, which
# is what makes a result attributable without deleting anything.
submit_epoch, _ = submit.execute("date +%s", quiet=True)
submit_epoch = int(submit_epoch.strip())

script = f'''
cd /home/ubuntu || exit 9
# Stop anything still running FIRST: a live DAG from a previous submission
# keeps competing for the same GPUs, muddling both the demo and the
# placement story.
for OLD in $(ls -d wf-runs/*/pegasus/*/run* 2>/dev/null); do
    pegasus-remove $OLD > /dev/null 2>&1 || true
done
condor_rm -all > /dev/null 2>&1 || true
for _ in 1 2 3 4 5 6 7 8 9 10; do
    test -z "$(condor_q -af ClusterId 2>/dev/null)" && break
    sleep 3
done
# Deliberately NOT 'rm -rf wf-runs wf-output': output dirs are epoch-
# namespaced and run_result.py attributes via each run's own braindump, so a
# wipe buys no isolation -- it only destroys prior runs' evidence.
python3 workflow_generator.py --segments {SEGMENTS} --minutes {MINUTES} \\
    --out higgs-wf.yml > gen.log 2>&1
GEN=$?
if [ $GEN -ne 0 ]; then echo "GEN_EXIT=$GEN PLAN_EXIT=skipped";
    tail -15 gen.log; exit 0; fi
pegasus-plan --submit -s condorpool --output-site local \\
    --conf pegasus.properties \\
    --output-dir /home/ubuntu/wf-output/{submit_epoch} \\
    --dir wf-runs higgs-wf.yml > plan.log 2>&1
PLAN=$?
echo "GEN_EXIT=$GEN PLAN_EXIT=$PLAN"
tail -20 plan.log
'''
out, err = submit.execute(script, quiet=True)
plan_output = out or err
print(plan_output)

# Three independent things must all hold. A run DIRECTORY is not enough:
# pegasus-plan creates it early and can fail afterwards, during submission.
if "GEN_EXIT=0" not in plan_output:
    raise RuntimeError(
        "workflow_generator.py FAILED - see gen.log above. A common cause is "
        "two jobs declaring the same output file.")
if "PLAN_EXIT=0" not in plan_output:
    raise RuntimeError(
        "pegasus-plan FAILED - see plan.log above. Common causes are a "
        "transformation whose executable cannot be resolved, or a site "
        "catalog entry that does not exist.")
if "submitted to cluster" not in plan_output:
    raise RuntimeError(
        "pegasus-plan exited 0 but nothing reached the queue: planning "
        "succeeded and submission did not.")

# Pegasus nests the run directory several levels deep; resolve it rather than
# assuming "wf-runs", which silently matches nothing.
run_dir, _ = submit.execute(
    "ls -d /home/ubuntu/wf-runs/*/pegasus/*/run* 2>/dev/null | tail -1",
    quiet=True)
run_dir = run_dir.strip()
if not run_dir:
    raise RuntimeError(
        "no run directory despite a clean plan - refusing to continue, "
        "because every later cell would report on nothing.")
print("submitted at", submit_epoch, "| run dir:", run_dir)
"""),

    md("""
### Prove the transfers are wired before waiting on the run

Under condorio the evidence is checkable the moment planning finishes, so check
it: every train job's submit file must list the executable, the dataset and
(from segment 1 on) the predecessor checkpoint in `transfer_input_files`, and
the DAG must contain a `stage_in` job. An earlier run "worked" while nothing
was actually being moved.
"""),
    code("""
out, _ = submit.execute(
    f"SUB=$(ls {run_dir}/00/00/train_higgs_train_001* 2>/dev/null "
    f"|| ls {run_dir}/00/00/train_*001*.sub 2>/dev/null | head -1); "
    f"echo \\"--- $SUB\\"; grep -h '^transfer_input_files' $SUB "
    f"| tr ',' '\\\\n' | head; "
    f"echo '--- stage-in jobs in the DAG:'; "
    f"grep -c 'stage_in' {run_dir}/*.dag", quiet=True)
print(out)
if "transfer_input_files" not in out and "HIGGS" not in out:
    raise RuntimeError(
        "no transfer_input_files in the train submit file - condorio is not "
        "in effect; check pegasus.properties reached pegasus-plan via --conf")
"""),

    md("""
## 8. Watch the workflow chase clean energy

The next cell (re)starts the dashboard **on the submit node** and prints the
tunnel command to run in a local terminal; then open http://localhost:8799.

It shows per-site carbon and power, every job's placement, the carbon intensity
recorded in the job ad *at match time*, and the bytes HTCondor actually moved —
plus two demo controls: **start workflow**, and per-site **inject/clear**
buttons that override one site's carbon with a disclosed, expiring value so a
migration can be triggered on demand and watched happen for real.
(`python pegasus/dashboard.py` from a laptop also works, via fablib.)
"""),
    code("""
# Two hard-won details:
# * the subshell around setsid is load-bearing -- without it the SSH channel
#   stays open until the dashboard exits, so this cell would hang forever
#   looking like a failed start (measured 40s+ vs 0.2s);
# * the pkill pattern is ANCHORED (^python3), because the remote shell running
#   the pkill has 'dashboard.py' in its OWN command line.
submit.upload_file("pegasus/dashboard.py", "dashboard.py")
out, _ = submit.execute(
    "pkill -f '^python3 dashboard\\\\.py' 2>/dev/null; sleep 1; "
    "( setsid nohup python3 dashboard.py > dashboard.log 2>&1 & ); "
    "sleep 3; "
    "curl -s -o /dev/null -w 'dashboard answering: HTTP %{http_code}' "
    "http://localhost:8799/ || (echo 'NOT RUNNING - dashboard.log:'; "
    "tail -5 dashboard.log)", quiet=True)
print(out.strip())

# fablib's own ssh command (bastion hop included) plus a port forward.
tunnel = submit.get_ssh_command().replace(
    "ssh ", "ssh -L 8799:localhost:8799 ", 1)
print("\\nrun this locally, then open http://localhost:8799 :\\n")
print(" ", tunnel)
"""),

    md("""
`pegasus-status` shows progress; the `condor_q` line shows *where* each segment
landed. Over a run, later segments should prefer whichever site the EIA trace
currently makes cleanest.
"""),
    code("""
import time

state = None
for _ in range(30):
    out, _ = submit.execute(
        f"date -u +%H:%M:%S; "
        f"pegasus-status -l {run_dir} 2>/dev/null | tail -3; "
        f"condor_status -af Name CarbonIntensity | sort -k2 -n; "
        f"condor_q -af pegasus_wf_dag_job_id JobStatus RemoteHost "
        f"2>/dev/null | grep -E 'train|predict|evaluate|report'", quiet=True)
    print(out, flush=True)
    # Stop on a TERMINAL state and remember which one: a DAG that ended in
    # Failure must not be described later as "not finished yet".
    if "Failure" in out:
        state = "Failure"; break
    if "Success" in out:
        state = "Success"; break
    time.sleep(60)
print("terminal state:", state or "still running when polling stopped")
"""),

    md("""
## 9. The science result

`evaluate_higgs.py` reports held-out AUC plus signal efficiency at fixed
background rejection — the operating points an analysis would quote. The 2014
paper reached AUC ≈ 0.88 on the 21 low-level features with a deep network, so a
run in that neighbourhood is evidence the training genuinely worked rather than
merely completed. It **refuses** to emit a result if the checkpoint is missing,
rather than printing a plausible number.
"""),
    code("""
# WHERE each segment ran, WITH the carbon intensity and GPU power the job ad
# recorded at match time (job_machine_attrs) and the bytes HTCondor moved --
# the placement story is in the job ads, not reconstructed.
out, _ = submit.execute(
    "condor_history -af pegasus_wf_dag_job_id MachineAttrFabricSite0 "
    "MachineAttrCarbonIntensity0 MachineAttrGPUWatts0 RemoteWallClockTime "
    "BytesRecvd ExitCode "
    "-constraint 'pegasus_wf_dag_job_id =!= UNDEFINED' | tac", quiet=True)
print("job | site | gCO2/kWh@match | W@match | wall_s | bytes_in | exit")
print(out)

# State FIRST, so a failed DAG is never described as an unfinished one.
status, _ = submit.execute(f"pegasus-status -l {run_dir} 2>&1 | tail -4",
                           quiet=True)
print(status)

# The output file on disk carries no evidence of which run wrote it, and a
# timestamp cannot supply that evidence either (an older DAG left running
# would finish AFTER this submission). run_result.py decides from THIS run's
# own stampede database, and refuses rather than guesses.
submit.upload_file("pegasus/run_result.py", "run_result.py")
# --expect-output-dir is a cross-check, not a redirect: the location always
# comes from the run's own braindump, and a disagreement is a refusal. That
# catches a planner that ignored --output-dir and staged somewhere shared.
out, err = submit.execute(
    f"python3 /home/ubuntu/run_result.py --run-dir {run_dir} "
    f"--expect-output-dir /home/ubuntu/wf-output/{submit_epoch} "
    f"--submitted-after {submit_epoch} 2>&1", quiet=True)
print(out or err)

out, _ = submit.execute(f"pegasus-statistics -s all {run_dir} 2>/dev/null "
                        f"| head -25", quiet=True)
print(out)
"""),

    md("""
### The visual deliverable

`report_higgs.html` is produced *by the workflow itself* (the `report` job) and
staged out with the result: AUC per segment **colored by the site that trained
it** — a color change on that line is a carbon migration, and the line not
resetting is the checkpoint hand-off working — plus the ROC curve and the
signal/background score distributions of the final model.
"""),
    code("""
from IPython.display import HTML, display

local_report = "higgs_report.html"
submit.download_file(local_report,
                     f"/home/ubuntu/wf-output/{submit_epoch}/higgs_report.html")
with open(local_report) as handle:
    display(HTML(handle.read()))
"""),

    md("""
## 10. Housekeeping

The slice holds real GPUs — release it when you are done, or extend the lease
if you need longer.
"""),
    code("""
# Extend:
# from datetime import datetime, timedelta
# from dateutil import tz
# sl.renew((datetime.now(tz=tz.tzutc()) + timedelta(days=7))
#          .strftime("%Y-%m-%d %H:%M:%S %z"))

# Release (destructive):
# sl.delete()
"""),

    md("""
## What to take away

* **Pegasus needed no modification.** The chain of checkpoint-linked segments
  is the migration mechanism, ClassAd `RANK` is the carbon policy, and
  condorio + `job_machine_attrs` make every byte moved and every match's
  carbon and power a recorded fact in the job ad. All documented features.
* **Workers hold zero pre-staged state — runtime included.** Executables,
  dataset, sample and the Apptainer image all ride HTCondor file transfer from
  the submit node into each sandbox, so the workflow is portable to any pool
  with HTCondor + Apptainer + a GPU driver. The price is the dataset plus the
  image per segment — measured, not assumed.
* **Placement happens at task granularity**, not by preempting a running task.
  Responsiveness is tuned by segment length, which is a limitation worth
  stating rather than hiding.
* **Measured beats modelled, and refusing beats inventing.** Absent data is
  reported absent; a missing power reading pauses accounting instead of
  substituting a guess.

### Verified on the testbed, and what is still open

**Verified** — evidence for the run below is kept in
[`runs/migration-run-3x/`](../runs/migration-run-3x):

* **Carbon-driven migration happens.** One 6-segment run placed its segments
  `STAR → TACC → UTAH → UTAH → STAR → STAR` — **3 migrations**, each one just
  the next segment matching at a cleaner site.
* **The checkpoint chain resumes across those moves**: held-out AUC climbs
  monotonically straight through them
  (0.8082 → 0.8401 → 0.8562 → 0.8631 → 0.8674 → 0.8709), so each segment
  continued from its predecessor's checkpoint rather than restarting.
* **The science is real**: final val-AUC **0.8709** on 200k held-out events
  after 71,923 steps, against the ≈0.88 the 2014 paper reports for these
  21 low-level features (earlier runs: 0.8702 on a Tesla T4, 0.8767 on an
  RTX 6000 at 155,608 steps).
* **Energy is measured per segment**: NVML integrals of 2.22–2.29 Wh per
  segment, with the working GPU at 67–72 W against 12–14 W idle elsewhere.
* **The sites are comparable**: all three workers agree on one `CarbonSpan`
  and one `CarbonAccel`, and each one's replay position matches its own
  `CarbonSampledAt`, despite the zones holding 156/157/158 samples — so the
  per-zone drift of 7.9 and 15.9 replayed hours is gone. (Their raw
  `CarbonTimestamp` values still differ, by design: see the note above.)

Getting there needed two fixes worth repeating, because both produced runs
that looked healthy and migrated nothing:

* `CLAIM_WORKLIFE = 0` had been set on the submit node instead of the workers,
  so a whole workflow fit inside one claim and `RANK` was consulted once.
* **Site selection maximized grid-zone diversity, which is not the same thing
  as migration opportunity** — one chosen zone was the cleanest in 100% of the
  trace, so the correct answer never changed. `select_sites.py` now ranks
  candidate site sets by how often the *cleanest* site switches.

**Still open — do not claim these:**

* **Carbon-ad freshness bounds how responsive placement can be.** The
  negotiator ranks the *collector's* copy of each worker's ad, which was
  measured at 208–268 s old — at `CARBON_ACCEL = 300` that is 17–22 replayed
  hours. Worker `UPDATE_INTERVAL` is now 60 s, and both `check_pool_match.py`
  and the dashboard's pool badge report ad age separately from coherence — a
  pool can be perfectly self-consistent and still be ranking prices the trace
  has moved past. The right `CARBON_ACCEL` for a demo remains a judgement
  call: more acceleration means more visible migrations and staler prices
  behind them.
* **No controlled A/B against a carbon-blind scheduler.** The run above shows
  the mechanism working; it does not quantify how much CO₂ the policy saves
  versus a default placement of the same workflow.
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(OUT, "w") as handle:
        json.dump(nb, handle, indent=1)
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
