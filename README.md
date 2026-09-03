# Carbon Chaser 🌱

**Carbon-aware workload migration across FABRIC sites.** A long-running,
checkpointable HEP training job automatically follows clean energy around
the country, while dashboards show the job chasing low-carbon grids and
count the CO₂ per training segment — from measured quantities only.

Built for the MERIF26 Demo & Poster Night.

There are two implementations in this repo:

* **The current demo — a Pegasus/HTCondor workflow** (`pegasus/`): placement
  is ordinary HTCondor matchmaking against live carbon ClassAds, data moves
  by HTCondor file I/O (condorio), and nothing in Pegasus or HTCondor is
  modified. This is what the MERIF demo runs. See the next section and the
  end-to-end notebook `pegasus/carbon-aware-pegasus.ipynb`.
* **The legacy hand-rolled orchestrator** (`carbon_chaser/`): an external
  control loop that stops/checkpoints/resumes a trainer over SSH. Kept as
  the no-testbed local fallback (`--mode sim` replays measured traces on a
  laptop) and for the measured migration-cost history, whose findings shaped
  how the current demo prices a migration. Documented separately in
  [README-legacy.md](README-legacy.md).

## The demo: a carbon-aware Pegasus workflow

The workload is real science, not a benchmark loop: signal-vs-background
classification on the **UCI HIGGS** dataset — 11M simulated ATLAS-like
collision events, 28 features (21 low-level kinematic quantities plus 7
hand-engineered high-level ones). Training uses the 21 low-level features
only, which is the regime where the published result showed deep networks
beating shallow ones, so architecture and training genuinely matter. Both the
dataset and that result are cited under [Citation](#citation); the target to
beat is **AUC ≈ 0.88**.

Training runs as a chain of time-boxed segments linked by their checkpoints
(`train_000 → ckpt_001.pt → train_001 → …`), so every segment is a fresh
HTCondor matchmaking decision. Workers advertise their grid's live carbon
intensity and measured GPU power as ClassAds, and a one-line policy makes
each decision carbon-aware:

```
requirements = GPUs >= 1 && CarbonIntensity =!= UNDEFINED
rank         = -CarbonIntensity
```

A **migration** is simply the next segment landing at a cleaner site with
its predecessor's checkpoint staged in.

Everything a job needs rides **HTCondor file transfer** (`condorio`): the
executables, the parsed ~1 GB dataset, and the checkpoints all ship from the
submit node into each job sandbox, so **workers hold zero pre-staged
state** — and HTCondor accounts for every byte (`TransferInputSizeMB`,
`BytesRecvd`), which makes the transfer cost of a migration a measurement,
not an estimate. **Every job runs inside an Apptainer container**
(`Apptainer/higgs_container.def`: torch + numpy + pandas + pynvml), built
once on the submit node and staged like any other input — so a worker
needs only HTCondor, Apptainer and the NVIDIA driver, plus one host
package for the carbon cron (pynvml, installed and verified **as the
condor user** by `provision.py` — a `pip install --user` under ubuntu
passes an interactive check while `GPUWatts` silently never appears). GPU
jobs run with `--nv`; the CPU-only predict/report jobs use the same image
without it, so they cannot touch a GPU a trainer is using. Each segment also records the carbon intensity and GPU
power of the machine it matched (`job_machine_attrs`), and the trainer
integrates NVML power over the segment (`gpu_wh` in the progress file), so
gCO₂-per-segment is computed from measured energy only.

Alongside training, a CPU-only **predict** job scores each segment's
checkpoint against a fixed validation sample, and a final **report** job
renders a self-contained HTML page — AUC per segment *colored by the site
that trained it* (a color change is a migration), ROC curve, and score
distributions.

Run order (details and rationale in the notebook):

```bash
# `--sites auto` surveys LIVE availability (free GPUs per model, cores,
# ram, disk) and picks grid-zone-diverse sites; select_sites.py is the
# read-only dry run of the same choice. Explicit lists still work.
python pegasus/provision.py --sites auto --submit-site STAR
# --slice-name is REQUIRED here: without it install_gpu_stack falls back to
# config/sites.yaml (slice_name: carbon-chaser, the legacy orchestrator's
# slice) and installs drivers on the wrong VMs.
python fabric/install_gpu_stack.py --reboot \
    --slice-name carbon-chaser-pegasus --nodes auto   # auto = every *-gpu node in the slice
python pegasus/fix_pool_hosts.py
python pegasus/stage_submit_node.py      # dataset + inputs + container + tools, submit node ONLY
python pegasus/check_pool_match.py       # can the policy match anything, right now?
# dashboard runs ON the submit node (start button submits the workflow,
# per-site inject/clear buttons drive demo migrations):
#   ssh -L 8799:localhost:8799 ubuntu@<submit-mgmt-ip>
#   python3 dashboard.py                 # on the submit node; then open http://localhost:8799
```

Two HTCondor defaults silently defeat a carbon rank and are set by the
provisioning scripts — `CLAIM_WORKLIFE = 0` **in the worker (startd)
config** so every segment is re-negotiated, and a de-biased
`NEGOTIATOR_PRE_JOB_RANK`. A pool provisioned before these fixes needs
`provision.py --skip-create` re-run. The notebook documents both failure
modes with the measured evidence.

## Data provenance — what is measured vs. modeled

The headline "CO₂ saved" is `power × time × carbon_intensity`, and **both
factors are now measured**. There is deliberately no simulator and no
assumed power anywhere in the codebase: every earlier class of bug here
came from synthetic values being mistaken for measurements, and the surest
fix is for no synthetic path to exist. With no real carbon source
configured the run **refuses to start**; with no NVML reading for a site,
emissions accounting for that site **pauses** rather than substituting a
guess.

One narrow, loudly-disclosed exception exists for demos: the dashboard can
**inject a carbon value for one site** (`/opt/carbon/override`) to trigger
a real migration on demand. The injected number replaces only the
matchmaking input, is published with `CarbonInjected = true` (amber badge
on the dashboard, recorded into the job ads of anything matched against
it), keeps the true trace value visible alongside, and **expires by
itself**. Power and energy are not injectable — the measured half of the
arithmetic cannot be faked.

| Quantity | Status | Source |
|---|---|---|
| Carbon intensity | **real (derived)** | EIA hourly generation mix per balancing authority × published emission factors, replayed at demo speed. Verified against grid character: PACE (coal) highest at 598 g mean, CISO lowest at 228 with the real solar dip (147 midday vs 287 evening). Derived, not a measured CO₂ value — the sidecar and dashboard say so |
| Node power | **measured** | NVML board power, sampled in-guest by the workload (FABRIC GPUs are PCI passthrough, so the card's own sensor is readable). Measured live: Quadro RTX 6000 ≈12 W idle, Tesla T4 ≈9 W idle. No assumed-power mode exists |
| Migration cost | **measured** | real dataplane transfers, real downtime, calibrated on the testbed |
| Inter-site link state | **measured** | FABRIC public metrics API (`dataplaneIn/OutBits`) |
| Training progress | **measured** | the job really runs and really migrates |

So the mechanism, the migrations, and the network measurements are real. The
carbon arithmetic depends on which sources you feed it, and the code will
not let a run imply more than its inputs support.

### Making the intensity real

```bash
# free key: https://www.eia.gov/opendata/register.php — keep it in
# ~/.secrets/eia (chmod 600); used automatically, or pass --key-file
python fabric/fetch_traces.py --source eia --days 7 --key-file ~/.secrets/eia
# or, with an Electricity Maps token (their model, ~24h on the free tier)
EMAPS_TOKEN=... python fabric/fetch_traces.py --source electricitymaps
```

The current demo picks that CSV up automatically: `provision.py` uploads it
to `/opt/carbon/eia.csv` on each worker, where `carbon_classad.py` reads it
(override with `CARBON_TRACE`). The legacy orchestrator instead points
`carbon.trace_file` in `config/sites.yaml` at it. Replay is what makes
measured data usable at demo speed: a live feed moves hourly, so at 1× a
two-hour booth would show almost no migrations, while the simulator moves
fast but invents the numbers. Replaying a real trace at 300× keeps every
value measured and compresses a day into minutes — and the same machinery
replays a year of history for offline policy evaluation.

The six configured zones were validated against Electricity Maps' public
zone list: all exist and are Tier A (Duke Energy Carolinas, ERCOT,
PacifiCorp East, PJM, CAISO, NYISO). A wrong zone id otherwise yields a
silently empty feed.

**Provenance travels with the data.** `fetch_traces.py` writes a
`<trace>.meta.json` sidecar recording the source, whether it is measured, and
any caveats (EIA intensity is *derived* — measured generation mix × published
emission factors — not a measured CO₂ value). Ship the sidecar with the trace:
a trace whose provenance is unknown should be reported as unverified, not
quoted as measurement.

Two rules the accounting follows in **both** implementations: **unknown is
`null`, never `0`** (zero is a legitimate reading — a fully renewable grid —
so it must never double as "no data"), and **stale is a third state, not a
flavour of fresh**. Both came from real bugs; the post-mortems, with the
measured evidence against the legacy engine and dashboard that produced them,
are in
[README-legacy.md](README-legacy.md#carbon-feed-handling--the-engine-and-its-dashboard).

**Where the two differ — and the current demo's honest caveat.** The legacy
engine refuses to let a stale reading move the emissions counters at all
(`policy.carbon_max_stale_s`). The workflow instead prices each segment with
the intensity **recorded in its own job ad at match time**, which is
auditable afterwards from `condor_history` alone but does lag the trace: the
negotiator ranks the *collector's* copy of a machine ad, measured 140–200 s
old (with `UPDATE_INTERVAL = 60`; it was 208–268 s at the default), while at
`CARBON_ACCEL = 300` a single trace row lasts 12 s. So the defensible claim
for the workflow is "the site advertising the lowest carbon intensity", not
"the cleanest site at this instant". `check_pool_match.py` measures and warns
about the gap.

Do the arithmetic before picking an acceleration. One hourly trace row lasts
`3600/ACCEL` real seconds, so an ad `S` seconds old is `S × ACCEL / 3600`
**replayed hours** behind:

| `CARBON_ACCEL` | a trace row lasts | ad lag at S = 140–200 s | inside one row? |
|---|---|---|---|
| 300× | 12 s | 11.7–16.7 h | no |
| 60× | 60 s | 2.3–3.3 h | no |
| 30× | 120 s | 1.2–1.7 h | no |
| 20× | 180 s | 0.8–1.1 h | borderline |
| 12× | 300 s | 0.5–0.7 h | yes |

So dropping to 30–60× shrinks the lag by roughly 5–10× but does **not**
eliminate it — the advertised value is still at least a full trace row
behind, and CISO alone swings 147 → 287 g between midday and evening, so a
two-hour-stale ranking can still be the wrong ranking. Bounding the lag to
about one row needs `ACCEL < 3600/S`, i.e. under ~20× at the staleness we
measure.

**Even then the claim stays "advertised", and no acceleration changes that.**
Two things stop a lower `ACCEL` from buying "the cleanest site at this
instant". An ad younger than one row can still be the *previous* row's — the
bound is on age, not on which row it came from. More fundamentally,
matchmaking ranks the collector's copies of ads that each worker sampled on
its own unsynchronised 60 s `STARTD_CRON`, so the values being compared are
up to `60 × ACCEL` seconds of replayed time apart — 5 h at 300×, and still
20 min at 20×. There is no single-instant snapshot to rank: each node's
position is only required to match *its own* `CarbonSampledAt`, which is
exactly what `replay_coherence()` in the dashboard and `check_pool_match.py`
verify. Lowering the acceleration narrows the error; it does not turn "lowest
advertised" into "lowest". That is why the dashboard, the report and the
abstract all say **advertised** — and why the remaining choice is a booth
trade-off (livelier migrations vs tighter prices), not a correctness fix.

### Measured power (GPUs are mandatory)

Every worker carries a GPU and the workload samples NVML, so power is a
reading rather than a constant. This also makes the *network* term real: a
GPU model checkpoints at tens of MB (measured: 35 MB) instead of the 12 KB
CPU toy, where transfer time was pure connection setup.

```bash
# current demo: provision.py creates the pool (GPU per worker), then
python fabric/install_gpu_stack.py --reboot \
    --slice-name carbon-chaser-pegasus --nodes auto   # driver + torch + pynvml
# legacy orchestrator uses fabric/create_slice.py instead — see README-legacy.md
```

`install_gpu_stack.py` is separate and idempotent because this is the
failure-prone step. Three traps it now handles, each of which cost a cycle:

* **DKMS/kernel skew** — DKMS builds against the kernel running at install
  time, but apt often pulls a newer kernel in the same transaction, so the
  reboot lands on a kernel with no module at all.
* **Newest driver ≠ usable driver** — `ubuntu-drivers install` picks the
  latest version, which frequently has no prebuilt module for the running
  kernel, giving `Driver/library version mismatch`. The script instead picks
  the highest driver that *has* a prebuilt module for the running kernel.
* **Re-running on a healthy node breaks it** — reinstalling reintroduces the
  skew, so the script does nothing when NVML already reports watts.

The success criterion is a number: an earlier version accepted any
non-empty `nvidia-smi` output and reported OK while the node was printing
`Failed to initialize NVML`.

GPU models are heterogeneous by necessity (RTX 6000 at CLEM, T4 at
TACC/UTAH), so sites differ in both carbon *and* power. That is realistic,
and it makes the migration decision genuinely two-dimensional — worth
stating explicitly in a paper rather than leaving as a confound.

## Booth runbook

1. Before doors: provision the pool, run `stage_submit_node.py` and
   `check_pool_match.py`, submit a short workflow (`--segments 6
   --minutes 2`) and confirm one migration end-to-end with the dashboard
   open; keep the legacy `--mode sim` orchestrator in a second terminal as
   the no-network fallback ([README-legacy.md](README-legacy.md)).
2. Pitch (30 s): "This training workflow is chasing clean electricity
   across the country. Every few minutes HTCondor re-decides where the next
   segment runs based on live grid carbon; it has migrated N times today,
   never lost a training step, and every joule and every transferred byte
   on screen is measured."
3. Involve the visitor: point at the dashboard's cleanest-site highlight and
   let them predict where the next segment lands, then watch the match. The
   report's AUC-per-segment chart shows the same story after the fact —
   each color change is a migration, and the curve never resets.
4. Talking points: nothing in Pegasus or HTCondor is modified — the policy
   is one RANK expression; workers hold zero pre-staged state (condorio),
   so the workflow is portable to any pool; carbon, power, and transfer
   bytes are all in the job ads, so the accounting is auditable with
   `condor_history` alone; FABRIC's slice API + nationwide GPU footprint is
   what makes multi-grid placement a one-liner.

## Draft sign-up abstract

> **Carbon Chaser: Carbon-Aware Scientific Workflows on FABRIC.**
> We demonstrate a real high-energy-physics training workload that
> automatically follows clean energy across the FABRIC testbed, using only
> unmodified Pegasus and HTCondor. Training runs as a chain of
> checkpoint-linked workflow segments; FABRIC GPU workers advertise live
> grid carbon intensity and measured GPU power as HTCondor ClassAds, so
> each segment is placed by ordinary matchmaking with a one-line
> carbon-aware rank. All data movement rides HTCondor file I/O, making the
> byte cost of every migration a measured quantity, and per-segment energy
> is integrated from NVML on the worker. A live dashboard shows placement
> decisions, per-site carbon, and measured gCO₂ per segment; the workflow
> itself renders a report in which training accuracy climbs across
> segments while color changes mark carbon-driven migrations. The demo
> highlights how FABRIC's programmable, geographically distributed
> infrastructure enables sustainability-aware systems research.

## Citation

The science workload is not ours — it is a published high-energy-physics
benchmark. Cite the paper for the result, and the UCI record for the data.

**Paper.** P. Baldi, P. Sadowski and D. Whiteson, "Searching for exotic
particles in high-energy physics with deep learning," *Nature
Communications* **5**, 4308 (2014).
doi:[10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308) ·
arXiv:[1402.4735](https://arxiv.org/abs/1402.4735) [hep-ph]

**Dataset.** D. Whiteson, *HIGGS* [Dataset], UCI Machine Learning
Repository (2014).
doi:[10.24432/C5V312](https://doi.org/10.24432/C5V312) ·
[archive.ics.uci.edu/dataset/280/higgs](https://archive.ics.uci.edu/dataset/280/higgs)

> Note the DOI: Nature Communications article **4308** carries DOI
> `ncomms5308`. `10.1038/ncomms4308` is a *different* paper (Boczkowska
> et al., on Arp2/3 complex activation) — an easy and citable-looking
> mistake, so it is spelled out here.

```bibtex
@article{baldi2014searching,
  title   = {Searching for exotic particles in high-energy physics with deep learning},
  author  = {Baldi, Pierre and Sadowski, Peter and Whiteson, Daniel},
  journal = {Nature Communications},
  volume  = {5},
  pages   = {4308},
  year    = {2014},
  doi     = {10.1038/ncomms5308},
  eprint  = {1402.4735},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph}
}

@misc{whiteson2014higgs,
  title     = {{HIGGS}},
  author    = {Whiteson, Daniel},
  year      = {2014},
  howpublished = {UCI Machine Learning Repository},
  doi       = {10.24432/C5V312}
}
```

The published AUC on the 21 low-level features is ≈0.88; this demo reports
its own measured AUC per run so the two can be compared directly.

## Layout

```
config/sites.yaml                  sites, zones, policy, fabric settings
config/traces/                     measured EIA carbon traces (+ provenance sidecars)
pegasus/                           THE CURRENT DEMO (Pegasus/HTCondor, condorio)
  workflow_generator.py            builds the workflow: train chain + predict fan-out + report
  workload_contract.py             CLI contract between generator and workload scripts
  stage_submit_node.py             one-time staging: dataset, derived inputs, scripts (submit node only)
  provision.py                     HTCondor+Pegasus pool on FABRIC, carbon ClassAds
  node_tools/                      condor configs (CLAIM_WORKLIFE=0 on workers, de-biased pre-job rank)
  carbon_classad.py                STARTD_CRON: publish CarbonIntensity/GPUWatts per slot
  check_pool_match.py              can the workflow's requirements match the live pool?
  fix_pool_hosts.py                re-pin /etc/hosts after reboots (cloud-init rewrites it)
  dashboard.py                     live run view on :8799 (carbon, placements, measured Wh/gCO2, bytes)
  run_result.py                    attribute a result to THIS run via its stampede DB
  build_notebook.py                generates carbon-aware-pegasus.ipynb (the shareable walkthrough)
carbon_chaser/
  workload/train_higgs.py          HIGGS trainer: checkpoint chain + NVML energy integral
  workload/evaluate_higgs.py       final held-out AUC + signal efficiencies
  workload/predict_higgs.py        per-segment scoring of a fixed validation sample
  workload/report_higgs.py         self-contained HTML report (AUC by site, ROC, distributions)
  main.py / engine.py / ...        LEGACY orchestrator -> see README-legacy.md
fabric/create_slice.py             legacy: 1 VM per site via fablib
fabric/fetch_traces.py             EIA / Electricity Maps -> measured trace CSVs
fabric/install_gpu_stack.py        kernel-safe NVIDIA driver + torch + pynvml (idempotent)
tests/                             offline suite (run tests/run_all.py before provisioning)
```
