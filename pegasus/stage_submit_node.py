#!/usr/bin/env python3
"""Provision the SUBMIT NODE with everything the condorio workflow stages.

This replaced three per-worker fan-out scripts (stage_higgs, prewarm_cache,
deploy_workload — since deleted): under `pegasus.data.configuration =
condorio` the submit node is the staging site and HTCondor file transfer
carries each job's inputs into its sandbox, so the ONLY machine that needs
pre-positioned state is this one. Workers hold nothing, which also removes
the whole class of "one stale worker fails at job runtime" contract-drift
failures.

Four idempotent steps, in dependency order:

1. **Workload scripts** -> {WORKLOAD_DIR}. Always re-uploaded (they are KB
   and change on every edit — the old deployer once skipped this inside a
   dataset short-circuit and every worker silently kept the old CLI), then
   verified against the CLI contract via `--help`. A script that landed but
   does not accept the flags the workflow passes is a failure, not a success.
2. **Dataset** -> {DATA_DIR}/HIGGS.csv. 2.6 GB from UCI, downloaded on the
   node (not through the laptop), skipped when present. The raw CSV is kept
   only as the source for step 3 — no job ever reads it.
3. **Derived inputs** the jobs actually consume, built once here because a
   fresh condorio sandbox could never amortize them:
   * `HIGGS.csv.raw.npy` — the parsed float32 cache (~1 GB). Exactly the
     file `train_higgs.load_higgs()` derives from `--data HIGGS.csv`, so a
     job finds it beside the data path and never parses text.
   * `higgs_val_sample.npz` — ~50k events from the SAME held-out tail with
     the SAME full-dataset standardisation load_higgs applies (torch's
     unbiased std, hence ddof=1). The per-segment predict jobs score this;
     a sample scaled differently would draw a smooth, wrong learning curve.
4. **The Apptainer container** every job runs in ->
   {DATA_DIR}/higgs_container.sif, built here from
   Apptainer/higgs_container.def (skipped when present) and smoke-tested by
   importing the full stack. Building on the submit node rather than
   pulling docker:// per job keeps Docker Hub's rate limits out of the
   demo's critical path, and the workers consequently need NO python at
   all — the numpy-missing-on-one-worker failure class cannot recur.

Usage:
    python pegasus/stage_submit_node.py [--verify-only]
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbon_chaser.executor import call_with_timeout  # noqa: E402

from fabrictestbed_extensions.fablib.fablib import FablibManager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workload_contract import (SUBMIT_DATA_DIR, SUBMIT_WORKLOAD_DIR,  # noqa: E402
                              WORKLOAD_SCRIPTS, missing_flags)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKLOAD_SRC = os.path.join(HERE, "..", "carbon_chaser", "workload")
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
       "HIGGS.csv.gz")

VAL_ROWS = 200_000          # must match train_higgs.VAL_ROWS
SAMPLE_ROWS = 50_000

# Runs on the submit node. Mirrors load_higgs(): label + 21 low-level
# features as float32, cache saved next to the CSV; the sample is the
# held-out tail, standardised with FULL-dataset statistics (ddof=1 to match
# torch.std), evenly subsampled so the signal/background mix is preserved.
BUILD_DERIVED = f"""
python3 - <<'EOF'
import numpy as np, os, sys
data_dir = "{SUBMIT_DATA_DIR}"
csv = os.path.join(data_dir, "HIGGS.csv")
cache = csv + ".raw.npy"
sample_path = os.path.join(data_dir, "higgs_val_sample.npz")

if os.path.exists(cache):
    data = np.load(cache, mmap_mode="r")
    if data.shape != (11_000_000, 22):
        sys.exit(f"cache has shape {{data.shape}}, expected (11000000, 22): "
                 f"delete it and re-run")
    print(f"cache present: {{data.shape}}", flush=True)
else:
    import pandas as pd
    print("parsing HIGGS.csv (one-time, minutes)...", flush=True)
    frame = pd.read_csv(csv, header=None, dtype=np.float32)
    data = frame.to_numpy()[:, :22]
    np.save(cache, data)
    print(f"cache built: {{data.shape}}", flush=True)

if os.path.exists(sample_path):
    print("sample present", flush=True)
else:
    data = np.asarray(np.load(cache, mmap_mode="r"))
    feats = data[:, 1:]
    mean = feats.mean(axis=0, keepdims=True, dtype=np.float64)
    std = feats.std(axis=0, keepdims=True, ddof=1, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    tail = data[-{VAL_ROWS}:]
    idx = np.linspace(0, len(tail) - 1, {SAMPLE_ROWS}).astype(int)
    x = ((tail[idx, 1:] - mean) / std).astype(np.float32)
    y = (tail[idx, 0] > 0.5).astype(np.uint8)
    np.savez_compressed(sample_path, x=x, y=y)
    print(f"sample built: {{x.shape}}, {{int(y.sum())}} signal", flush=True)
print("DERIVED_OK", flush=True)
EOF
"""


def run(node, cmd, label, budget=600):
    out, err = call_with_timeout(
        lambda: node.execute(cmd, quiet=True, retry=1), budget, label)
    return (out or ""), (err or "")


def install_scripts(submit, verify_only):
    if not verify_only:
        # The submit node also gets the generator (the dashboard's
        # start-workflow button runs it there) and the dashboard itself
        # (runs in --mode local on this node; laptops just tunnel to it).
        for tool in ("workflow_generator.py", "dashboard.py"):
            submit.upload_file(os.path.join(HERE, tool), tool)
        run(submit, f"mkdir -p {SUBMIT_WORKLOAD_DIR}", "mkdir:workload")
        for script in WORKLOAD_SCRIPTS:
            submit.upload_file(os.path.join(WORKLOAD_SRC, script), script)
        moves = " && ".join(f"mv -f {s} {SUBMIT_WORKLOAD_DIR}/{s}"
                            for s in WORKLOAD_SCRIPTS)
        out, err = run(submit,
                       f"{moves} && chmod +x {SUBMIT_WORKLOAD_DIR}/*.py && "
                       f"echo MOVED", "install:workload")
        if "MOVED" not in out:
            sys.exit(f"script install FAILED: {(err or out)[-200:]}")

    problems = []
    for script, required in WORKLOAD_SCRIPTS.items():
        out, err = run(submit,
                       f"python3 {SUBMIT_WORKLOAD_DIR}/{script} --help 2>&1 "
                       f"| head -60", f"help:{script}")
        text = out or err
        if "usage:" not in text.lower():
            problems.append(f"{script}: --help did not run "
                            f"({text.strip()[:80]})")
            continue
        gaps = missing_flags(text, required)
        if gaps:
            problems.append(f"{script}: missing {', '.join(gaps)}")
    if problems:
        sys.exit("CONTRACT NOT MET on the submit node: "
                 + "; ".join(problems))
    print(f"  scripts: {len(WORKLOAD_SCRIPTS)} installed in "
          f"{SUBMIT_WORKLOAD_DIR}, contract verified", flush=True)


def stage_dataset(submit):
    out, _ = run(submit,
                 f"test -s {SUBMIT_DATA_DIR}/HIGGS.csv && echo PRESENT "
                 f"|| echo ABSENT", "check:csv")
    if "PRESENT" in out:
        print("  dataset: HIGGS.csv already on the submit node", flush=True)
        return
    print("  dataset: downloading 2.6 GB from UCI (on the node)...",
          flush=True)
    out, err = run(submit,
                   f"sudo mkdir -p {SUBMIT_DATA_DIR} && "
                   f"sudo chown ubuntu:ubuntu {SUBMIT_DATA_DIR} && "
                   f"cd {SUBMIT_DATA_DIR} && "
                   f"curl -fsSL -o HIGGS.csv.gz '{URL}' && "
                   f"gunzip -f HIGGS.csv.gz && echo DOWNLOAD_OK",
                   "dl:csv", 5400)
    if "DOWNLOAD_OK" not in out:
        sys.exit(f"dataset download FAILED: {(err or out)[-200:]}")
    print("  dataset: downloaded and unpacked", flush=True)


def build_derived(submit):
    # pip can be absent on a FRESH submit node (node_tools installs
    # python3-pip on the workers; the submit role never needed it before
    # this step), and the previous form here swallowed every failure
    # (`2>/dev/null || ... ; echo DEPS` prints DEPS unconditionally) — the
    # same silent-deps class that once let every train job die on `import
    # numpy`. Install loudly, then prove it by importing: the import IS
    # the check the derive step will repeat.
    out, err = run(submit,
                   "python3 -m pip --version >/dev/null 2>&1 || "
                   "sudo apt-get install -y -q python3-pip; "
                   "(sudo python3 -m pip install -q --break-system-packages "
                   "pandas numpy 2>/dev/null || "
                   "sudo python3 -m pip install -q pandas numpy); "
                   "python3 -c 'import numpy, pandas' && echo DEPS_OK",
                   "deps", 1800)
    if "DEPS_OK" not in out:
        sys.exit(f"numpy/pandas did not install on the submit node — the "
                 f"derived-input build cannot run: {(err or out)[-300:]}")
    out, err = run(submit, BUILD_DERIVED, "derive", 3600)
    if "DERIVED_OK" not in out:
        sys.exit(f"derived-input build FAILED: {(err or out)[-300:]}")
    for line in out.strip().splitlines():
        print(f"  derive: {line}", flush=True)


SIF = f"{SUBMIT_DATA_DIR}/higgs_container.sif"
STAMP = f"{SIF}.def.sha256"


def local_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sif_verdict(submit, upload_def=False):
    """ABSENT / STALE / CURRENT for the container against the LOCAL def.

    Shared by the build (to decide whether to rebuild) and by
    --verify-only (so 'verified' means the same thing in both places).
    The reference hash is computed locally, so the check itself is
    READ-ONLY on the node — a verifier that uploads files is quietly a
    stager. Only the build path passes upload_def=True, because
    `apptainer build` needs the def present remotely.
    """
    def_path = os.path.join(HERE, "..", "Apptainer", "higgs_container.def")
    want = local_sha256(def_path)
    if upload_def:
        submit.upload_file(def_path, "higgs_container.def")
    out, _ = run(submit,
                 f"HAVE=$(cat {STAMP} 2>/dev/null); "
                 f"if [ ! -s {SIF} ]; then echo VERDICT=ABSENT; "
                 f"elif [ \"{want}\" != \"$HAVE\" ]; then echo VERDICT=STALE; "
                 f"else echo VERDICT=CURRENT; fi",
                 "check:sif")
    return next((line.split("=", 1)[1] for line in out.splitlines()
                 if line.startswith("VERDICT=")), "ABSENT")


def smoke_test_sif(submit, sif):
    """The imports must work WITHOUT --nv — the submit node has no GPU, and
    the CPU-only predict/report jobs run exactly that way."""
    out, _ = run(submit,
                 f"apptainer exec {sif} python3 -c "
                 f"'import torch, numpy, pandas, pynvml' && echo STACK_OK",
                 "smoke:sif", 900)
    return "STACK_OK" in out


def build_container(submit):
    """Build the job container; every job sandbox receives a copy.

    "The sif exists" is not the invariant that matters — "the sif matches
    the CURRENT def and its stack imports" is. So the skip path is earned,
    not assumed: the def's hash is recorded next to the sif at build time,
    a changed def forces a rebuild (otherwise editing the def silently
    ships the OLD image forever), and even an up-to-date sif is smoke
    tested, because a truncated or corrupt image would otherwise surface
    as every job in the run dying inside the sandbox.
    """
    sif, stamp = SIF, STAMP

    # node_tools installs apptainer, but a fresh provision was observed
    # without it (the install section can fail silently inside the role
    # script) — and this step is the first thing that would notice.
    # Self-heal rather than instruct: same PPA the role scripts use.
    out, _ = run(submit, "command -v apptainer >/dev/null && echo HAVE "
                         "|| echo MISSING", "check:apptainer")
    if "HAVE" not in out:
        print("  container: apptainer missing on the submit node — "
              "installing (PPA)...", flush=True)
        out, err = run(submit,
                       "sudo add-apt-repository -y ppa:apptainer/ppa "
                       "> /tmp/apptainer-install.log 2>&1 && "
                       "sudo apt-get update -q >> /tmp/apptainer-install.log "
                       "2>&1 && sudo apt-get install -y -q apptainer "
                       ">> /tmp/apptainer-install.log 2>&1 && "
                       "apptainer --version && echo APPTAINER_OK",
                       "install:apptainer", 1200)
        if "APPTAINER_OK" not in out:
            sys.exit(f"apptainer install FAILED on the submit node: "
                     f"{(err or out)[-200:]} (log: "
                     f"/tmp/apptainer-install.log there)")

    verdict = sif_verdict(submit, upload_def=True)

    if verdict == "CURRENT":
        if smoke_test_sif(submit, sif):
            print("  container: sif matches the current def, stack imports",
                  flush=True)
            return
        print("  container: sif matches the def but its stack does NOT "
              "import — rebuilding rather than staging a broken image",
              flush=True)
    elif verdict == "STALE":
        print("  container: def changed since the sif was built — "
              "rebuilding", flush=True)
    else:
        print("  container: building (one-time, pulls the pytorch base "
              "image; 10-30 min)...", flush=True)

    out, err = run(submit,
                   f"sudo apptainer build --force {sif} higgs_container.def "
                   f"> /tmp/sif-build.log 2>&1 && echo BUILD_OK; "
                   f"tail -3 /tmp/sif-build.log",
                   "build:sif", 3600)
    if "BUILD_OK" not in out:
        sys.exit(f"container build FAILED: {(err or out)[-300:]} "
                 f"(full log: /tmp/sif-build.log on the submit node)")
    if not smoke_test_sif(submit, sif):
        sys.exit("container built but the tool stack does not import — "
                 "refusing to stage an image the jobs would die inside")
    # Stamp only AFTER the smoke test: a recorded hash claims "this sif was
    # built from this def AND verified", so a failed build can never earn
    # the skip path on the next run. The write itself is verified by
    # reading it back — a silently failed stamp would not break anything
    # today, it would condemn every future staging run to a needless
    # 10-30 minute rebuild, which is exactly the kind of quiet cost this
    # script exists to avoid.
    out, err = run(submit,
                   f"WANT=$(sha256sum higgs_container.def | cut -d' ' -f1); "
                   f"echo \"$WANT\" > {stamp} && "
                   f"[ \"$(cat {stamp})\" = \"$WANT\" ] && echo STAMP_OK",
                   "stamp:sif")
    if "STAMP_OK" not in out:
        sys.exit(f"could not record the def hash at {stamp}: "
                 f"{(err or out)[-160:]} — without it every future staging "
                 f"run will rebuild the image from scratch")
    size, _ = run(submit, f"du -h {sif} | cut -f1", "du:sif")
    print(f"  container: built and smoke-tested ({size.strip()})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-name", default="carbon-chaser-pegasus")
    ap.add_argument("--verify-only", action="store_true",
                    help="check contract + presence without uploading or "
                         "downloading anything")
    args = ap.parse_args()

    fablib = FablibManager()
    sl = fablib.get_slice(name=args.slice_name)
    submit = sl.get_node(name="submit")

    print("Staging the submit node (workers get NOTHING — condorio ships "
          "per-job):", flush=True)
    install_scripts(submit, args.verify_only)
    if args.verify_only:
        # A verifier that prints an ls and exits 0 is not a verifier: a
        # missing dataset or stale container "passed" it. Every artifact
        # a submission needs is checked, the container against the SAME
        # currency verdict the build uses, and any gap is a non-zero exit
        # naming what to do about it.
        problems = []
        # Code artifacts are verified by CONTENT (local hash vs remote
        # sha256sum, read-only on the node): a present-but-stale copy of
        # the generator, dashboard or a workload script certified by a
        # mere `test -s` is exactly how a pool ends up running last
        # week's code while "verified". The derived data files have no
        # local reference (they are built on the node), so presence and
        # non-emptiness is the strongest honest check for them.
        code_files = {
            "/home/ubuntu/workflow_generator.py":
                os.path.join(HERE, "workflow_generator.py"),
            "/home/ubuntu/dashboard.py": os.path.join(HERE, "dashboard.py"),
        }
        for script in WORKLOAD_SCRIPTS:
            code_files[f"{SUBMIT_WORKLOAD_DIR}/{script}"] = os.path.join(
                WORKLOAD_SRC, script)
        remote_cmd = "; ".join(
            f"echo \"$(sha256sum {remote} 2>/dev/null | cut -d' ' -f1) "
            f"{remote}\"" for remote in code_files)
        out, _ = run(submit, remote_cmd, "verify:code")
        remote_hashes = {}
        for line in out.strip().splitlines():
            parts = line.split()
            if parts:
                remote_hashes[parts[-1]] = parts[0] if len(parts) == 2 else ""
        for remote, local in sorted(code_files.items()):
            have = remote_hashes.get(remote, "")
            if not have:
                print(f"  MISSING {remote}", flush=True)
                problems.append(remote)
            elif have != local_sha256(local):
                print(f"  STALE   {remote}", flush=True)
                problems.append(f"{remote} (stale)")
            else:
                print(f"  OK      {remote}", flush=True)
        out, _ = run(submit,
                     f"for f in {SUBMIT_DATA_DIR}/HIGGS.csv.raw.npy "
                     f"{SUBMIT_DATA_DIR}/higgs_val_sample.npz; do "
                     f'test -s "$f" && echo "OK      $f" || '
                     f'echo "MISSING $f"; done', "verify:data")
        for line in out.strip().splitlines():
            print(f"  {line}", flush=True)
            if line.startswith("MISSING"):
                problems.append(line.split()[-1])
        verdict = sif_verdict(submit)
        print(f"  container: {verdict}", flush=True)
        if verdict != "CURRENT":
            problems.append(f"higgs_container.sif ({verdict})")
        if problems:
            sys.exit("verify-only FAILED — missing or stale: "
                     + ", ".join(problems)
                     + ". Run without --verify-only to stage them.")
        print("verify-only: all artifacts present and current (container "
              "smoke test not run — full staging runs it)", flush=True)
        return
    stage_dataset(submit)
    build_derived(submit)
    build_container(submit)
    print("submit node staged; generate + plan the workflow next", flush=True)


if __name__ == "__main__":
    main()
