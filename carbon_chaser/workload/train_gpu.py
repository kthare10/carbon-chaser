#!/usr/bin/env python3
"""GPU training workload with NVML power sampling.

Same contract as the CPU trainer (checkpoint.npz + progress.json + an
exclusive workdir lock), plus a third file:

    power.json  {"watts": 71.4, "ts": ..., "device": "Tesla T4",
                 "samples": 30, "window_s": 30}

Why this matters: FABRIC compute is virtualised, so RAPL is unavailable and
CPU power can only be modelled. GPUs are PCI passthrough, so NVML reads the
board's own power sensor *inside* the VM — the dominant consumer becomes a
measurement rather than an assumption, which is the last synthetic term in
the CO2 arithmetic.

A real model also gives realistically large checkpoints, so the bandwidth
term in the migration-cost model stops being noise (the CPU toy checkpoints
were 12 KB, where transfer time is pure connection setup).

Falls back to CPU if CUDA is absent, and to no power reporting if NVML is
absent — reporting nothing is correct there; the engine then says the figure
is assumed.
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
PROGRESS_EVERY = 20
CHECKPOINT_EVERY = 100
POWER_WINDOW_S = 30.0          # averaging window for the NVML samples
POWER_SAMPLE_S = 1.0


def acquire_lock(workdir):
    handle = open(os.path.join(workdir, "train.lock"), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[train] another trainer already holds this workdir; exiting",
              flush=True)
        sys.exit(EXIT_ALREADY_RUNNING)
    return handle


def atomic_write(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


class PowerSampler(threading.Thread):
    """Averages NVML board power over a rolling window.

    An instantaneous reading is noisy and would make the emissions integral
    jump around; the engine wants average draw over its tick interval.
    """

    daemon = True

    def __init__(self, workdir):
        super().__init__(name="nvml-sampler")
        self.workdir = workdir
        self.stop_flag = threading.Event()
        self.handle = None
        self.device = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self.handle)
            self.device = name.decode() if isinstance(name, bytes) else name
            print(f"[train] NVML active on {self.device}", flush=True)
        except Exception as e:
            print(f"[train] NVML unavailable ({e}); power will be reported as "
                  f"assumed, not measured", flush=True)

    def run(self):
        if self.handle is None:
            return
        samples = []
        while not self.stop_flag.is_set():
            try:
                milliwatts = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
                samples.append((time.time(), milliwatts / 1000.0))
            except Exception:
                pass
            cutoff = time.time() - POWER_WINDOW_S
            samples = [(t, w) for t, w in samples if t >= cutoff]
            if samples:
                mean = sum(w for _, w in samples) / len(samples)
                atomic_write(os.path.join(self.workdir, "power.json"), {
                    "watts": round(mean, 1), "ts": time.time(),
                    "device": self.device, "samples": len(samples),
                    "window_s": POWER_WINDOW_S,
                })
            self.stop_flag.wait(POWER_SAMPLE_S)


def build_model(device, width):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(256, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, 10),
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--target-steps", type=int, default=0,
                    help="0 = run until stopped")
    ap.add_argument("--width", type=int, default=2048,
                    help="hidden width; drives checkpoint size so the "
                         "migration bandwidth term is non-trivial")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    lock = acquire_lock(args.workdir)          # noqa: F841

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[train] CUDA unavailable; running on CPU", flush=True)

    torch.manual_seed(0)
    model = build_model(device, args.width)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    ckpt = os.path.join(args.workdir, "checkpoint.pt")

    step = 0
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        print(f"[train] resumed at step {step}", flush=True)
    else:
        print("[train] starting from step 0", flush=True)

    sampler = PowerSampler(args.workdir)
    sampler.start()

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    gen = torch.Generator(device="cpu").manual_seed(1234)
    centres = torch.randn(10, 256, generator=gen) * 0.9
    target = args.target_steps or float("inf")
    loss_val = acc_val = 0.0

    while step < target and not stop["flag"]:
        y = torch.randint(0, 10, (args.batch,), generator=gen)
        x = centres[y] + torch.randn(args.batch, 256, generator=gen) * 2.2
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        loss_val = float(loss.item())
        acc_val = float((out.argmax(1) == y).float().mean().item())

        if step % PROGRESS_EVERY == 0:
            atomic_write(os.path.join(args.workdir, "progress.json"), {
                "step": step, "loss": round(loss_val, 4),
                "acc": round(acc_val, 4), "ts": time.time(),
                "host": os.uname().nodename, "done": False,
                "device": device,
            })
        if step % CHECKPOINT_EVERY == 0:
            tmp = ckpt + ".tmp"
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(), "step": step}, tmp)
            os.replace(tmp, ckpt)

    tmp = ckpt + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, tmp)
    os.replace(tmp, ckpt)
    finished = step >= target
    atomic_write(os.path.join(args.workdir, "progress.json"), {
        "step": step, "loss": round(loss_val, 4), "acc": round(acc_val, 4),
        "ts": time.time(), "host": os.uname().nodename, "done": finished,
        "device": device,
    })
    sampler.stop_flag.set()
    print(f"[train] {'completed' if finished else 'stopped'} at step {step}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
