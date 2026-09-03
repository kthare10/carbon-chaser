#!/usr/bin/env python3
"""Carbon-aware HIGGS training as a Pegasus workflow — condorio edition.

Structured per the pegasus-scaffold conventions: a `CarbonHiggsWorkflow`
class building properties, site/replica/transformation catalogs and the DAG,
with `main()` doing argparse + validation.

## The shape

A chain of time-boxed training segments linked by their checkpoints, with a
prediction fan-out and a final visual report:

    train_000 ──ckpt_001──> train_001 ──ckpt_002──> ... ──ckpt_N──> evaluate
        │                       │                            │          │
        └──> predict_000        └──> predict_001    predict_N-1        │
                 └────────────────────┴──── predictions ───┴──> report

Each training segment is a separate matchmaking decision. Workers advertise
grid carbon intensity (carbon_classad.py), so a one-line policy makes every
decision carbon-aware:

    condor.requirements = GPUs >= 1 && CarbonIntensity =!= UNDEFINED
    condor.rank         = -CarbonIntensity

A "migration" is the next segment landing at a cleaner site with its
predecessor's checkpoint staged in. The per-segment predict jobs score a
small fixed validation sample against each checkpoint, and `report` renders
a self-contained HTML page whose headline chart is AUC-per-segment colored
by the site that trained it — the migration story as a picture.

## condorio: nothing is pre-staged on workers

`pegasus.data.configuration = condorio` (set explicitly in the properties
this generator writes, though it is also Pegasus 5's default): the submit
host is the staging site and HTCondor file transfer carries every job's
inputs into its sandbox and its outputs back. Concretely:

* **Executables are stageable from site `local`.** The previous design
  declared them installed at /home/ubuntu/bin on every worker and needed a
  fan-out deployer + contract verifier per node. Now they live ONLY in
  {workload_dir} on the submit node; each job's submit file lists them in
  `transfer_input_files`. (The historical failure that motivated
  installed-executables — stage-in jobs retrying against a worker-only path
  — was a wrong PFN *site*, not a problem with staging itself: a stageable
  PFN must name where the file actually is, which is `local`.)
* **The dataset rides HTCondor I/O too.** Jobs consume the parsed
  `HIGGS.csv.raw.npy` (~1 GB), registered in the replica catalog at
  {data_dir} on the submit node and declared as an input of every training
  job. `train_higgs.load_higgs()` finds the cache next to `--data HIGGS.csv`
  in the sandbox and never touches a CSV. The 2.6 GB CSV download and the
  one-time 8 GB parse happen ONCE, on the submit node
  (stage_submit_node.py) — shipping the raw URL to jobs instead would make
  every fresh sandbox re-parse 8 GB of text.
* **The runtime is an Apptainer container, staged the same way.** Every job
  executes inside Apptainer/higgs_container.def (torch + numpy + pandas +
  pynvml), built ONCE on the submit node and registered as a file://
  container on `local` — so the workers need only HTCondor, Apptainer and
  the NVIDIA driver, and there is no system-python to drift. GPU jobs get
  `--nv`; the CPU-only predict/report jobs deliberately do not, so they
  cannot touch a GPU a concurrent trainer is using.
* **The trade-off is explicit**: each segment now pulls ~1 GB of dataset
  plus the multi-GB container image across the FABNet dataplane instead of
  reading node-local copies, and in exchange the workers hold zero
  pre-staged state — no deployer, no cache prewarm, no contract or python
  drift between nodes. HTCondor accounts for every byte
  (TransferInputSizeMB, BytesRecvd in the job ad), which is what makes the
  transfer cost measurable instead of assumed.

## Power/carbon accounting lands in the JOB ad

Every transformation carries

    job_machine_attrs = CarbonIntensity, GPUWatts, FabricSite, CarbonTimestamp

so at each match HTCondor copies those machine attributes into the job ad
(MachineAttrCarbonIntensity0, ...). `condor_history` then holds, per
segment: where it ran, the grid intensity AND measured GPU power at match
time, wall time, and bytes transferred — enough to compute gCO2 per segment
from measured quantities only. The dashboard reads exactly that.

## Two planning invariants, learned by breaking them

* No two jobs may produce the same filename (Pegasus derives the DAG from
  producers) — `check_unique_producers` fails generation with the fix in
  the message.
* `add_checkpoint` is NOT the segment hand-off: it only engages when a job
  is killed, whereas a segment that checkpoints and exits 0 is a success.
  The explicit ckpt_NNN chain is the resume mechanism.

Usage (on the submit node):
    python3 workflow_generator.py --segments 12 --minutes 20
    pegasus-plan --submit -s condorpool --output-site local higgs-wf.yml
"""

import argparse
import sys
from collections import defaultdict

# The matchmaking policy, as named constants so check_pool_match.py can
# verify the live pool against the SAME expressions this generator emits.
#
# `GPUs >= 1` deliberately uses HTCondor's own integer slot attribute: the
# pool once advertised `HasGPU = ifThenElse(DetectedGPUs > 0, ...)`, but
# DetectedGPUs is a STRING of GPU ids, so the comparison evaluated to ERROR
# — and an error in requirements means NEVER MATCH, silently.
TRAIN_REQUIREMENTS = "GPUs >= 1 && CarbonIntensity =!= UNDEFINED"
TRAIN_RANK = "-CarbonIntensity"
EVAL_REQUIREMENTS = "GPUs >= 1"
# Predict jobs need torch but not a GPU. FabricSite marks worker slots (the
# submit node also runs a startd, and a job with no requirements can match
# it — where torch is not installed and the failure surfaces mid-DAG).
PREDICT_REQUIREMENTS = "FabricSite =!= UNDEFINED"

# Machine attributes copied into the job ad at every match — the mechanism
# that turns per-slot carbon/power ClassAds into per-JOB accounting that
# condor_history keeps after the job is gone. CarbonInjected rides along
# so a segment matched against a demo-injected price carries the label
# PERMANENTLY: without it, MachineAttrCarbonIntensity0 would present an
# injected number as a trace reading forever after.
JOB_MACHINE_ATTRS = ("CarbonIntensity, GPUWatts, FabricSite, "
                     "CarbonTimestamp, CarbonInjected")

try:
    from Pegasus.api import (Arch, Container, Directory, File, FileServer,
                             Job, Namespace, Operation, OS, Properties,
                             ReplicaCatalog, Site, SiteCatalog,
                             Transformation, TransformationCatalog, Workflow)
except ImportError:
    print("Pegasus API not importable here — run this on the submit node.",
          file=sys.stderr)
    raise


def segment_timeouts(minutes):
    """(checkpoint_time, maxwalltime) in minutes, with headroom over `minutes`.

    Getting this wrong cost a whole run, three times over: every segment
    failed with kickstart `status 14` and DAGMan burned both retries.

    Per the kickstart manpage, `-k` is when kickstart sends SIGTERM — and a
    job that runs longer than `-k` "exits with a non-zero exit status" BY
    CONSTRUCTION, even one that catches the signal, checkpoints and exits 0.
    Pegasus derives `-k` from `checkpoint.time` and sends SIGKILL at
    `checkpoint.time + (maxwalltime - checkpoint.time)/2`; leaving
    checkpoint.time to its default (maxwalltime/2) once put SIGTERM exactly
    on the trainer's own --max-seconds deadline.

    So kickstart's timeout must NOT be the segment boundary: the trainer
    ends itself and exits 0, and these values only bound a genuine hang.
    The headroom also covers work outside the trainer's own clock — under
    condorio that now includes the ~1 GB dataset transfer into the sandbox,
    which happens before the process starts but kickstart never sees;
    dataset LOAD and the final eval/checkpoint save still do sit inside
    kickstart's clock.
    """
    if minutes < 1:
        raise ValueError("segment minutes must be >= 1")
    checkpoint_time = minutes * 2      # SIGTERM at twice the self-budget
    maxwalltime = minutes * 3          # => SIGKILL at 2.5x
    return checkpoint_time, maxwalltime


def check_unique_producers(plan):
    """Fail generation if any file is produced by more than one job.

    Pegasus derives the DAG from which job produces each file, so a
    duplicate producer is a FATAL planning error — far cheaper to catch at
    generation time with a clear message than after uploading:

        FATAL: Output file progress.json found for job train_1 has already
               been associated as output for a previous job train_0

    `plan` is [(job_id, [output_lfn, ...]), ...].
    """
    producers = defaultdict(list)
    for job_id, outputs in plan:
        for lfn in outputs:
            producers[lfn].append(job_id)
    clashes = {lfn: jobs for lfn, jobs in producers.items() if len(jobs) > 1}
    if clashes:
        detail = "; ".join(f"{lfn} <- {', '.join(jobs)}"
                           for lfn, jobs in sorted(clashes.items()))
        raise ValueError(
            f"each output file needs exactly one producing job, but: {detail}. "
            f"Give every segment its own filenames (ckpt_001.pt, "
            f"ckpt_002.pt, ...) — a shared name cannot be planned.")
    return True


class CarbonHiggsWorkflow:
    """Carbon-aware HIGGS training chain + prediction fan-out + HTML report."""

    wf_name = "carbon-aware-higgs"

    def __init__(self, args):
        self.args = args
        self.wf = None
        self.sc = None
        self.tc = None
        self.rc = None
        self.props = None
        # LFNs shared between catalogs and jobs. The dataset LFN must be the
        # exact cache name load_higgs() derives from `--data HIGGS.csv`
        # (tag "raw", all rows), so the shipped file is found in the sandbox.
        self.data_npy = File("HIGGS.csv.raw.npy")
        self.val_sample = File("higgs_val_sample.npz")
        # evaluate/predict import build_model/auc_score from train_higgs.py;
        # under condorio the module reaches their sandboxes as a plain input
        # file, not via any worker install.
        self.train_module = File("train_higgs.py")

    # ------------------------------------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        # Explicit even though it is the Pegasus 5 default: this file is the
        # single statement of the staging model, and the notebook passes it
        # via --conf so a submit node with different user-level properties
        # cannot silently change the design.
        self.props["pegasus.data.configuration"] = "condorio"

    def create_sites_catalog(self):
        """local = submit node (staging site under condorio) + condorpool.

        Written explicitly rather than relying on Pegasus 5's built-in
        defaults so the scratch/storage paths are visible and versioned
        with the generator instead of implied.
        """
        args = self.args
        self.sc = SiteCatalog()
        local = Site("local").add_directories(
            Directory(Directory.SHARED_SCRATCH, args.scratch_dir)
            .add_file_servers(FileServer(f"file://{args.scratch_dir}",
                                         Operation.ALL)),
            Directory(Directory.LOCAL_STORAGE, args.storage_dir)
            .add_file_servers(FileServer(f"file://{args.storage_dir}",
                                         Operation.ALL)),
        )
        condorpool = (Site("condorpool")
                      .add_condor_profile(universe="vanilla")
                      .add_pegasus_profile(style="condor"))
        self.sc.add_sites(local, condorpool)

    def create_replica_catalog(self):
        """Everything a job consumes that no job produces, all on `local`.

        These are the ONLY pre-positioned files in the whole system, and
        they live on the submit node alone (stage_submit_node.py puts them
        there): the parsed dataset, the fixed validation sample, and the
        train_higgs module for the jobs that import it.
        """
        args = self.args
        self.rc = ReplicaCatalog()
        self.rc.add_replica("local", self.data_npy,
                            f"file://{args.data_dir}/{self.data_npy.lfn}")
        self.rc.add_replica("local", self.val_sample,
                            f"file://{args.data_dir}/{self.val_sample.lfn}")
        self.rc.add_replica("local", self.train_module,
                            f"file://{args.workload_dir}/{self.train_module.lfn}")

    def create_transformation_catalog(self):
        """All four executables, stageable from the submit node, each run
        inside the project Apptainer container.

        `site="local"` + `is_stageable=True` means "this file lives on the
        submit host; ship it to the job" — the condorio-correct declaration.
        (An earlier failed submission taught the converse lesson: a
        stageable PFN pointing at a path that only exists on the WORKERS
        left stage-in jobs retrying forever. The PFN site must be where the
        file actually is.)

        The CONTAINER is the runtime: workers need only HTCondor, Apptainer
        and the NVIDIA driver, and the entire python stack travels in the
        image. This closes the drift class a live pool actually hit —
        workers whose system python lacked numpy killed every segment three
        seconds in, and a `pip --user` install under ubuntu was invisible
        to the condor user the jobs run as. Two Container entries share ONE
        sif (built once on the submit node, file:// on `local`, shipped
        like any other input):

        * `higgs_gpu` adds `--nv`, binding the host driver in, so training
          sees CUDA and pynvml reads the real board sensor from inside.
        * `higgs_cpu` omits it, so the deliberately CPU-only predict/report
          jobs cannot even see the GPU a concurrent trainer is using —
          torch.cuda.is_available() is false by construction, not by hope.
        """
        args = self.args
        self.tc = TransformationCatalog()

        sif = f"file://{args.data_dir}/higgs_container.sif"
        gpu_container = Container("higgs_gpu", Container.SINGULARITY,
                                  image=sif, image_site="local",
                                  arguments="--nv")
        cpu_container = Container("higgs_cpu", Container.SINGULARITY,
                                  image=sif, image_site="local")
        self.tc.add_containers(gpu_container, cpu_container)

        def stageable(name, container):
            return Transformation(
                name, site="local",
                pfn=f"{args.workload_dir}/{name}.py",
                is_stageable=True, arch=Arch.X86_64, os_type=OS.LINUX,
                container=container)

        train = stageable("train_higgs", gpu_container)
        # Carbon-aware placement. `=!= UNDEFINED` is load-bearing: a worker
        # whose carbon feed is missing advertises no attribute at all
        # (unknown is absent, never 0), and this keeps it out of the running
        # rather than letting an undefined RANK silently rank it best.
        train.add_profiles(Namespace.CONDOR, key="requirements",
                           value=TRAIN_REQUIREMENTS)
        train.add_profiles(Namespace.CONDOR, key="rank", value=TRAIN_RANK)
        train.add_profiles(Namespace.CONDOR, key="request_gpus", value="1")
        # Explicit, because Pegasus's default request_memory lands at
        # ~128 MB and HTCondor enforces it as a cgroup limit: the job is not
        # slow, it is HELD. Units: request_memory MB, request_disk KB —
        # disk must now also hold the ~1 GB dataset the sandbox receives.
        train.add_profiles(Namespace.CONDOR, key="request_memory",
                           value=str(args.memory_mb))
        train.add_profiles(Namespace.CONDOR, key="request_disk",
                           value=str(args.disk_gb * 1024 * 1024))
        train.add_profiles(Namespace.CONDOR, key="request_cpus", value="2")

        evaluate = stageable("evaluate_higgs", gpu_container)
        evaluate.add_profiles(Namespace.CONDOR, key="requirements",
                              value=EVAL_REQUIREMENTS)
        evaluate.add_profiles(Namespace.CONDOR, key="request_gpus", value="1")
        evaluate.add_profiles(Namespace.CONDOR, key="request_memory",
                              value=str(args.memory_mb))
        evaluate.add_profiles(Namespace.CONDOR, key="request_disk",
                              value=str(args.disk_gb * 1024 * 1024))

        # Prediction is deliberately GPU-free: a forward pass over the
        # ~50k-event sample is seconds on CPU, and requesting a GPU would
        # make the visual layer compete with training for the pool's
        # scarcest resource. But "no GPU" must not mean "any slot": the
        # submit node runs a startd too, and torch exists only on the
        # workers — FabricSite is advertised by every worker's startd (all
        # slot types, dynamic included) and by nothing else, so it is the
        # worker marker that request_gpus cannot provide here.
        predict = stageable("predict_higgs", cpu_container)
        predict.add_profiles(Namespace.CONDOR, key="requirements",
                             value=PREDICT_REQUIREMENTS)
        predict.add_profiles(Namespace.CONDOR, key="request_memory",
                             value="2048")
        predict.add_profiles(Namespace.CONDOR, key="request_cpus", value="1")

        report = stageable("report_higgs", cpu_container)
        report.add_profiles(Namespace.CONDOR, key="request_memory",
                            value="1024")
        report.add_profiles(Namespace.CONDOR, key="request_cpus", value="1")

        for tr in (train, evaluate, predict, report):
            # Per-match carbon/power snapshot into the job ad; length covers
            # DAGMan retries (each retry is a fresh match).
            tr.add_profiles(Namespace.CONDOR, key="job_machine_attrs",
                            value=JOB_MACHINE_ATTRS)
            tr.add_profiles(Namespace.CONDOR,
                            key="job_machine_attrs_history_length", value="4")
            self.tc.add_transformations(tr)

    # ------------------------------------------------------------------
    def create_workflow(self):
        args = self.args
        # Dependencies are NOT added by hand: every edge is a file edge
        # (ckpt chain, predictions, progress, result), so the planner infers
        # the DAG from producers/consumers and a hand-written edge could
        # only ever disagree with the data flow.
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        seconds = int(args.minutes * 60)
        checkpoint_time, maxwalltime = segment_timeouts(args.minutes)

        previous_ckpt = None
        progress_files = []
        prediction_files = []
        declared = []          # (job_id, [output lfns]) for the self-check

        for index in range(args.segments):
            out_ckpt = File(f"ckpt_{index + 1:03d}.pt")
            progress = File(f"progress_{index + 1:03d}.json")

            job = Job("train_higgs", _id=f"train_{index:03d}",
                      node_label=f"train_{index:03d}")
            job.add_args("--workdir", ".", "--data", "HIGGS.csv",
                         "--out-checkpoint", out_ckpt.lfn,
                         "--progress-file", progress.lfn,
                         "--max-seconds", str(seconds),
                         "--width", str(args.width),
                         "--depth", str(args.depth),
                         "--batch", str(args.batch))
            job.add_inputs(self.data_npy)
            if previous_ckpt is not None:
                job.add_args("--in-checkpoint", previous_ckpt.lfn)
                job.add_inputs(previous_ckpt)
            job.add_outputs(out_ckpt, stage_out=args.keep_checkpoints,
                            register_replica=False)
            job.add_outputs(progress, stage_out=True, register_replica=False)

            # Timeouts are the HANG BACKSTOP, not the segment boundary: the
            # segment ends itself via --max-seconds and exits 0.
            job.add_profiles(Namespace.PEGASUS, key="checkpoint.time",
                             value=str(checkpoint_time))
            job.add_profiles(Namespace.PEGASUS, key="maxwalltime",
                             value=str(maxwalltime))
            job.add_profiles(Namespace.DAGMAN, key="retry", value="2")
            self.wf.add_jobs(job)
            declared.append((f"train_{index:03d}",
                             [out_ckpt.lfn, progress.lfn]))

            # The visual layer: score this segment's checkpoint against the
            # fixed sample. Runs concurrently with the NEXT segment (its
            # only parent is this segment, via the checkpoint file).
            prediction = File(f"predictions_{index + 1:03d}.npz")
            pjob = Job("predict_higgs", _id=f"predict_{index:03d}",
                       node_label=f"predict_{index:03d}")
            pjob.add_args("--checkpoint", out_ckpt.lfn,
                          "--sample", self.val_sample.lfn,
                          "--out", prediction.lfn,
                          "--width", str(args.width),
                          "--depth", str(args.depth))
            pjob.add_inputs(out_ckpt, self.val_sample, self.train_module)
            pjob.add_outputs(prediction, stage_out=False,
                             register_replica=False)
            pjob.add_profiles(Namespace.PEGASUS, key="maxwalltime", value="20")
            pjob.add_profiles(Namespace.DAGMAN, key="retry", value="2")
            self.wf.add_jobs(pjob)
            declared.append((f"predict_{index:03d}", [prediction.lfn]))

            progress_files.append(progress)
            prediction_files.append(prediction)
            previous_ckpt = out_ckpt

        result = File("higgs_result.json")
        final = Job("evaluate_higgs", _id="evaluate", node_label="evaluate")
        final.add_args("--workdir", ".", "--data", "HIGGS.csv",
                       "--width", str(args.width), "--depth", str(args.depth),
                       "--checkpoint", previous_ckpt.lfn)
        final.add_inputs(previous_ckpt, self.data_npy, self.train_module)
        final.add_outputs(result, stage_out=True, register_replica=False)
        final.add_profiles(Namespace.PEGASUS, key="maxwalltime", value="30")
        self.wf.add_jobs(final)
        declared.append(("evaluate", [result.lfn]))

        report_out = File("higgs_report.html")
        report = Job("report_higgs", _id="report", node_label="report")
        report.add_args("--predictions",
                        *[f.lfn for f in prediction_files],
                        "--progress", *[f.lfn for f in progress_files],
                        "--result", result.lfn, "--out", report_out.lfn)
        report.add_inputs(result, *prediction_files, *progress_files)
        report.add_outputs(report_out, stage_out=True, register_replica=False)
        # Render on the submit host: every input is already in the staging
        # directory there (zero transfer), numpy is guaranteed by
        # stage_submit_node.py, and the visual deliverable never waits on —
        # or fails with — a worker slot.
        report.add_profiles(Namespace.SELECTOR, key="execution.site",
                            value="local")
        report.add_profiles(Namespace.PEGASUS, key="maxwalltime", value="20")
        self.wf.add_jobs(report)
        declared.append(("report", [report_out.lfn]))

        check_unique_producers(declared)

    # ------------------------------------------------------------------
    def write(self):
        """Write properties + all catalogs as separate files in cwd.

        pegasus-plan picks up ./pegasus.properties, ./sites.yml,
        ./replicas.yml and ./transformations.yml automatically when run in
        the same directory (and the notebook passes --conf explicitly so a
        stray user-level properties file cannot override the staging model).
        """
        self.props.write()               # ./pegasus.properties
        self.sc.write()                  # ./sites.yml
        self.rc.write()                  # ./replicas.yml
        self.tc.write()                  # ./transformations.yml
        self.wf.write(self.args.out)


def main():
    ap = argparse.ArgumentParser(
        description="Generate the carbon-aware HIGGS Pegasus workflow "
                    "(condorio: no pre-staged worker state)")
    ap.add_argument("--segments", type=int, default=12,
                    help="training segments; each is one placement decision, "
                         "so more segments = more carbon responsiveness")
    ap.add_argument("--minutes", type=int, default=20,
                    help="walltime per segment before it checkpoints itself")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--memory-mb", type=int, default=12288,
                    help="request_memory in MB; enforced as a cgroup limit, "
                         "so too little means HELD, not slow")
    ap.add_argument("--disk-gb", type=int, default=8,
                    help="request_disk for the job sandbox, which under "
                         "condorio receives the ~1 GB dataset too")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="stage every intermediate checkpoint out (large); "
                         "by default only the final results matter")
    ap.add_argument("--workload-dir", default="/home/ubuntu/workload",
                    help="where stage_submit_node.py installed the workload "
                         "scripts ON THE SUBMIT NODE")
    ap.add_argument("--data-dir", default="/home/ubuntu/higgs-data",
                    help="where stage_submit_node.py built the .npy cache "
                         "and validation sample ON THE SUBMIT NODE")
    ap.add_argument("--scratch-dir", default="/home/ubuntu/wf-scratch",
                    help="local site shared-scratch (staging area)")
    ap.add_argument("--storage-dir", default="/home/ubuntu/wf-output",
                    help="local site storage (final outputs; pegasus-plan "
                         "--output-dir overrides per run)")
    ap.add_argument("--out", default="higgs-wf.yml")
    args = ap.parse_args()

    if args.segments < 1:
        ap.error("--segments must be >= 1")
    for name in ("workload_dir", "data_dir", "scratch_dir", "storage_dir"):
        value = getattr(args, name)
        if not value.startswith("/"):
            ap.error(f"--{name.replace('_', '-')} must be an absolute path "
                     f"(got {value!r}): it becomes a file:// PFN, and a "
                     f"relative one plans into a URL that resolves nowhere")

    generator = CarbonHiggsWorkflow(args)
    generator.create_pegasus_properties()
    generator.create_sites_catalog()
    generator.create_replica_catalog()
    generator.create_transformation_catalog()
    generator.create_workflow()
    generator.write()
    print(f"wrote {args.out} + pegasus.properties: {args.segments} segments "
          f"x {args.minutes} min, carbon-ranked, condorio staging, "
          f"{args.segments} predict jobs + report", file=sys.stderr)


if __name__ == "__main__":
    main()
