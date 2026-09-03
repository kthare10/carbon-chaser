#!/usr/bin/env python3
"""Real science: HIGGS signal-vs-background classification on GPU.

The dataset is the UCI HIGGS benchmark (11M simulated ATLAS-like collision
events, 28 features) from Baldi, Sadowski & Whiteson, "Searching for exotic
particles in high-energy physics with deep learning", Nature Communications
5:4308 (2014), doi:10.1038/ncomms5308, arXiv:1402.4735. The dataset itself is
Whiteson, HIGGS, UCI Machine Learning Repository, doi:10.24432/C5V312.
The task is real: separate Higgs-producing signal processes
from the background that mimics them. The published contribution of that
paper was precisely that deep networks beat shallow ones on the *raw* 21
kinematic features, without the 7 hand-engineered high-level features — so
this is a benchmark where architecture and training actually matter, not a
toy fit.

Why it suits a carbon-migration demo:

* Genuinely long-running (hours on one GPU), so migrating mid-training is a
  real operation rather than a contrivance.
* Checkpoints are tens of MB, so the transfer cost of a migration is a
  measurable network event instead of pure connection setup.
* AUC on a held-out split is a meaningful progress metric a physicist would
  recognise, so "the job kept its progress across migrations" is a claim
  about science output, not just a step counter.

Pegasus contract (see `add_checkpoint` + `checkpoint.time` / `maxwalltime`):
kickstart sends SIGTERM when the task's slice of walltime is up; we write a
checkpoint and exit 0 so Pegasus stages the checkpoint out and resubmits the
task — possibly at a cleaner site. Resuming is therefore the normal path,
not an error path.

Files written in --workdir:
    checkpoint.pt   model + optimiser + epoch/step + RNG state (atomic)
    progress.json   step, loss, AUC, host, done flag, and gpu_wh — the
                    NVML-integrated energy of THIS segment (null if NVML
                    is absent; never a wall-clock estimate)
    power.json      NVML board power: rolling mean + integrated wh
"""

import argparse
import fcntl
import json
import os
import signal
import sys
import threading
import time

EXIT_ALREADY_RUNNING = 3
PROGRESS_EVERY = 200
CHECKPOINT_EVERY = 2000
POWER_WINDOW_S = 30.0
POWER_SAMPLE_S = 1.0
N_FEATURES = 28
VAL_ROWS = 200_000          # held-out tail, as in common HIGGS protocol


def acquire_lock(workdir):
    """One trainer per workdir, enforced by the kernel.

    Two trainers on one checkpoint diverge silently, and the guarantee has
    to live in the workload rather than in whatever launched it.
    """
    handle = open(os.path.join(workdir, "train.lock"), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[train] another trainer holds this workdir; exiting", flush=True)
        sys.exit(EXIT_ALREADY_RUNNING)
    return handle


def workflow_uuid():
    """The Pegasus workflow this job belongs to, or None.

    Read from the HTCondor job ad (`_CONDOR_JOB_AD`, which every job gets)
    rather than passed in: the uuid is minted by the PLANNER, after the
    workflow generator has already emitted this job's arguments, so there
    is no earlier point at which it could be handed over.

    Stamping it into progress.json is what lets a consumer prove a
    measurement belongs to a given run. Filenames cannot: every run writes
    progress_001.json, so two runs sharing an output directory produce
    indistinguishable, mutually overwriting files, and anything keyed on
    the path alone would attribute one run's energy to another's segments.
    """
    path = os.environ.get("_CONDOR_JOB_AD")
    if not path:
        return None
    try:
        with open(path) as handle:
            for line in handle:
                if line.split("=")[0].strip() == "pegasus_wf_uuid":
                    return line.partition("=")[2].strip().strip('"') or None
    except OSError:
        return None
    return None


def atomic_write(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


class PowerSampler(threading.Thread):
    """Rolling-mean NVML board power, plus MEASURED energy.

    `joules` integrates the sampled watts over the sampled intervals, so
    the per-segment energy staged out in the progress file is an integral
    of measurements — not a power snapshot multiplied by wall-clock, which
    is an estimate wearing a measurement's clothes (match-time watts say
    nothing about the other N minutes, and wall-clock includes transfer
    and load time the GPU spent idle). Absent NVML publishes nothing:
    joules stays 0 and the progress field is null, never a guess.
    """

    daemon = True

    def __init__(self, workdir):
        super().__init__(name="nvml-sampler")
        self.workdir = workdir
        self.stop_flag = threading.Event()
        self.handles = []
        self.device = None
        self.joules = 0.0
        self._last_sample_ts = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            for index in range(pynvml.nvmlDeviceGetCount()):
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(index))
            if self.handles:
                name = pynvml.nvmlDeviceGetName(self.handles[0])
                self.device = name.decode() if isinstance(name, bytes) else name
                print(f"[train] NVML active on {self.device} "
                      f"({len(self.handles)} device(s))", flush=True)
        except Exception as exc:
            print(f"[train] NVML unavailable ({exc}); power will be reported "
                  f"as unavailable, not estimated", flush=True)

    def run(self):
        if not self.handles:
            return
        samples = []
        while not self.stop_flag.is_set():
            try:
                watts = sum(
                    self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    for h in self.handles)
                now = time.time()
                samples.append((now, watts))
                # Rectangle-rule integral over the actually-sampled
                # intervals. A gap (sampler stalled, NVML hiccup) simply
                # contributes nothing — an undercount is honest, a
                # gap-filled value would be invented.
                if (self._last_sample_ts is not None
                        and now - self._last_sample_ts <= 5 * POWER_SAMPLE_S):
                    self.joules += watts * (now - self._last_sample_ts)
                self._last_sample_ts = now
            except Exception:
                self._last_sample_ts = None
            cutoff = time.time() - POWER_WINDOW_S
            samples = [(t, w) for t, w in samples if t >= cutoff]
            if samples:
                mean = sum(w for _, w in samples) / len(samples)
                atomic_write(os.path.join(self.workdir, "power.json"), {
                    "watts": round(mean, 1), "ts": time.time(),
                    "device": self.device, "samples": len(samples),
                    "window_s": POWER_WINDOW_S,
                    "wh": round(self.joules / 3600.0, 3),
                })
            self.stop_flag.wait(POWER_SAMPLE_S)

    def measured_wh(self):
        """Integrated energy so far, or None when nothing was measured."""
        if not self.handles or self.joules <= 0:
            return None
        return round(self.joules / 3600.0, 3)


def load_higgs(path, device, high_level, max_rows=0):
    """Load HIGGS into tensors. Column 0 is the label; 1..21 are the raw
    kinematic features; 22..28 are the hand-engineered high-level ones.

    Parsed once into a .npy cache: the CSV is ~8 GB of text and re-parsing it
    on every migration would dwarf the training time. `np.loadtxt` is far too
    slow for 11M rows (tens of minutes), so pandas does the parse when
    available.
    """
    import numpy as np
    import torch

    tag = "hl" if high_level else "raw"
    tag += f".{max_rows}" if max_rows else ""
    cache = f"{path}.{tag}.npy"
    if os.path.exists(cache):
        data = np.load(cache, mmap_mode=None)
    else:
        print(f"[train] parsing {path} (one-time; then cached as {cache})",
              flush=True)
        try:
            import pandas as pd
            frame = pd.read_csv(path, header=None, dtype=np.float32,
                                nrows=max_rows or None)
            raw = frame.to_numpy()
        except ImportError:
            raw = np.loadtxt(path, delimiter=",", dtype=np.float32,
                             max_rows=max_rows or None)
        data = raw if high_level else raw[:, :22]
        np.save(cache, data)
    labels = torch.from_numpy(data[:, 0]).float()
    feats = torch.from_numpy(data[:, 1:]).float()
    # Standardise once; HIGGS features have very different scales.
    mean, std = feats.mean(0, keepdim=True), feats.std(0, keepdim=True)
    feats = (feats - mean) / std.clamp_min(1e-6)
    split = len(labels) - VAL_ROWS
    return (feats[:split].to(device), labels[:split].to(device),
            feats[split:].to(device), labels[split:].to(device))


def build_model(in_features, width, depth, device):
    import torch.nn as nn
    layers, size = [], in_features
    for _ in range(depth):
        layers += [nn.Linear(size, width), nn.BatchNorm1d(width),
                   nn.ReLU(), nn.Dropout(0.1)]
        size = width
    layers += [nn.Linear(size, 1)]
    return nn.Sequential(*layers).to(device)


def auc_score(scores, labels):
    """AUC via the rank identity — no sklearn dependency on the workers."""
    import torch
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device,
                                dtype=torch.float32)
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--data", default="HIGGS.csv",
                    help="dataset path; under condorio the parsed "
                         "HIGGS.csv.raw.npy cache is shipped into the "
                         "sandbox next to this name, so the CSV itself is "
                         "never read by a job")
    # Distinct per-segment filenames: a Pegasus workflow derives its DAG from
    # file producers, so two jobs cannot declare the same output. The chain
    # ckpt_001 -> ckpt_002 -> ... IS the resume mechanism.
    ap.add_argument("--in-checkpoint", default=None,
                    help="checkpoint to resume from (absent for segment 0)")
    ap.add_argument("--out-checkpoint", default="checkpoint.pt")
    ap.add_argument("--progress-file", default="progress.json")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="end the segment cleanly after this long. Preferred "
                         "over relying on SIGTERM: a segment that ends itself "
                         "exits 0 with a complete checkpoint, whereas a "
                         "SIGKILLed one is a DAGMan failure.")
    ap.add_argument("--steps", type=int, default=0,
                    help="0 = run until stopped (SIGTERM checkpoints)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=5,
                    help="the Nature Comms result needed depth, not width")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = all 11M events; a subset makes the first run "
                         "quick without changing the task")
    ap.add_argument("--high-level", action="store_true",
                    help="include the 7 engineered features (the paper's "
                         "point was that deep nets do not need them)")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    lock = acquire_lock(args.workdir)          # noqa: F841

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[train] WARNING: CUDA unavailable; running on CPU", flush=True)

    x_tr, y_tr, x_val, y_val = load_higgs(args.data, device, args.high_level,
                                          args.max_rows)
    print(f"[train] HIGGS: {len(y_tr):,} train / {len(y_val):,} val, "
          f"{x_tr.shape[1]} features, device={device}", flush=True)

    model = build_model(x_tr.shape[1], args.width, args.depth, device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    ckpt = os.path.join(args.workdir, args.out_checkpoint)
    resume_from = (os.path.join(args.workdir, args.in_checkpoint)
                   if args.in_checkpoint else None)
    step = 0
    if resume_from and os.path.exists(resume_from):
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        print(f"[train] resumed from {args.in_checkpoint} at step {step} "
              f"(AUC was {state.get('auc', float('nan')):.4f})", flush=True)
    elif resume_from:
        sys.exit(f"--in-checkpoint {resume_from} missing: refusing to silently "
                 f"restart training from scratch and report it as progress")
    else:
        print("[train] starting from step 0 (first segment)", flush=True)

    sampler = PowerSampler(args.workdir)
    sampler.start()
    wf_uuid = workflow_uuid()

    stop = {"flag": False}

    def on_term(signum, frame):
        # NOT the normal path, despite what an earlier version of this comment
        # claimed. kickstart sends SIGTERM at its `-k` timeout, and per its
        # manpage a job that runs past `-k` "exits with a non-zero exit
        # status" REGARDLESS of how gracefully it handles the signal — so
        # arriving here means the segment has already been marked failed and
        # DAGMan will retry it. The normal boundary is the --max-seconds
        # deadline below, which exits 0 on its own well before this fires.
        # Checkpointing here is still worth doing: the retry resumes from it.
        print(f"[train] signal {signum}: checkpointing (kickstart timeout — "
              f"this segment will be retried)", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    target = args.steps or float("inf")
    deadline = (time.time() + args.max_seconds) if args.max_seconds else None
    gen = torch.Generator(device="cpu").manual_seed(1234 + step)
    auc = float("nan")
    loss_val = 0.0

    def evaluate():
        model.eval()
        with torch.no_grad():
            chunks = [model(x_val[i:i + 65536]).squeeze(-1)
                      for i in range(0, len(x_val), 65536)]
            scores = torch.cat(chunks)
        model.train()
        return auc_score(scores, y_val)

    def progress_payload(done):
        # gpu_wh is the NVML integral for THIS segment (the sampler starts
        # with the process), staged out in the progress file so per-segment
        # energy downstream is a measurement, not watts-at-match x wall.
        return {
            "step": step, "loss": round(float(loss_val), 4),
            "auc": None if auc != auc else round(auc, 4),
            "ts": time.time(), "host": os.uname().nodename,
            "device": device, "done": done,
            "gpu_wh": sampler.measured_wh(),
            # Which run produced this measurement. Absent outside HTCondor.
            "wf_uuid": wf_uuid,
            "metric": "val_auc", "dataset": "UCI HIGGS (11M events)",
        }

    def save(done=False):
        tmp = ckpt + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "auc": auc,
                    "cpu_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, ckpt)
        atomic_write(os.path.join(args.workdir, args.progress_file),
                     progress_payload(done))

    model.train()
    while step < target and not stop["flag"]:
        if deadline is not None and time.time() >= deadline:
            print(f"[train] segment time budget reached at step {step}",
                  flush=True)
            break
        idx = torch.randint(0, len(y_tr), (args.batch,), generator=gen)
        idx = idx.to(device)
        logits = model(x_tr[idx]).squeeze(-1)
        loss = loss_fn(logits, y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        loss_val = loss.item()

        if step % PROGRESS_EVERY == 0:
            atomic_write(os.path.join(args.workdir, args.progress_file),
                         progress_payload(False))
        if step % CHECKPOINT_EVERY == 0:
            auc = evaluate()
            save()
            print(f"[train] step {step} loss {loss_val:.4f} val-AUC {auc:.4f}",
                  flush=True)

    auc = evaluate()
    finished = step >= target
    save(done=finished)
    # Always exit 0 with a complete checkpoint: in this workflow the segment
    # boundary is normal, and the NEXT job resumes from this file.
    sampler.stop_flag.set()
    print(f"[train] {'completed' if finished else 'checkpointed'} at step "
          f"{step}, val-AUC {auc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
