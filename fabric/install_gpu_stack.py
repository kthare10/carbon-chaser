#!/usr/bin/env python3
"""Install the NVIDIA driver + torch + pynvml on every worker, in parallel.

Split out of create_slice.py because this is the slow, failure-prone step
(GBs of downloads, a kernel module, and possibly a reboot) and it needs to be
re-runnable without touching the slice. It also must NOT swallow errors: an
earlier version used `|| true`, which reported success while leaving the
nodes with no driver at all and NVML unavailable — meaning no measured power,
which this demo treats as fatal rather than falling back to a guess.

Usage:  python fabric/install_gpu_stack.py [--reboot]
"""

import argparse
import os
import sys
import threading

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbon_chaser.executor import call_with_timeout  # noqa: E402

from fabrictestbed_extensions.fablib.fablib import FablibManager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def nvml_watts(text):
    """Parse a power figure out of the NVML probe, or None.

    Requires a line that is ENTIRELY a number (optionally with a W suffix).
    Two earlier versions were too loose and both reported success on failing
    nodes:

      * "any non-empty output" accepted
        "Failed to initialize NVML: Driver/library version mismatch";
      * "contains a number" accepted
        "bash: line 1: nvidia-smi: command not found" — the 1 from "line 1"
        became a 1.0 W reading, and three GPU-less nodes were reported OK.

    Error text is prose; a measurement is a bare number.
    """
    import re
    for line in (text or "").splitlines():
        token = line.strip().rstrip("W").strip()
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
            continue
        value = float(token)
        if 0.0 < value < 2000.0:          # plausible board watts
            return value
    return None


def wait_for_ssh(node, site, budget=600):
    """Nodes are unreachable while they reboot; poll until they answer."""
    import time
    deadline = time.time() + budget
    while time.time() < deadline:
        try:
            out, _ = call_with_timeout(
                lambda: node.execute("echo UP", quiet=True, retry=1), 90,
                f"ping:{site}")
            if "UP" in (out or ""):
                return True
        except Exception:
            pass
        time.sleep(20)
    return False


def run(node, cmd, label, budget=1500):
    out, err = call_with_timeout(
        lambda: node.execute(cmd, quiet=True, retry=1), budget, label)
    return (out or ""), (err or "")


def probe_watts(node, site):
    """Current NVML board power, or None. The only success criterion.

    Gates on the tool existing first: a missing nvidia-smi produces shell
    error text, and prose must never be mistaken for a measurement.
    """
    out, _ = run(node,
                 "command -v nvidia-smi >/dev/null || { echo NO_SMI; exit 0; }; "
                 "nvidia-smi --query-gpu=power.draw --format=csv,noheader "
                 "2>&1 | head -1", f"probe:{site}", 240)
    low = out.lower()
    if "no_smi" in low or "mismatch" in low or "failed" in low:
        return None
    return nvml_watts(out)


def setup(node, site, results, do_reboot):
    """Make NVML report power on `site`, idempotently.

    The recipe matters and was learned the hard way:

    * DKMS builds against the kernel running at install time, but apt often
      pulls a newer kernel in the same transaction — so the reboot lands on a
      kernel with no module ("Module nvidia not found in /lib/modules/<new>").
    * `ubuntu-drivers install` picks the newest driver, which frequently has
      NO prebuilt module for the running kernel, giving userspace/module
      version mismatch.
    * Re-running a driver install on a WORKING node re-introduces that skew.

    So: do nothing if power already reads; otherwise pick the highest driver
    version that has a prebuilt module for the RUNNING kernel, align
    userspace to it, and load it. No DKMS build required.
    """
    try:
        watts = probe_watts(node, site)
        if watts is not None:
            results[site] = f"already OK | NVML {watts} W (untouched)"
            return

        kern, _ = run(node, "uname -r", f"kernel:{site}", 120)
        kern = kern.strip().splitlines()[-1].strip()

        # Which driver versions ship a prebuilt module for THIS kernel?
        # The module variant and the userspace driver package MUST match:
        # pairing linux-modules-nvidia-535-OPEN with the proprietary
        # nvidia-driver-535 yields "held broken packages".
        out, _ = run(node,
                     f"apt-cache search 'linux-modules-nvidia-.*-{kern}' "
                     f"2>/dev/null | sed -E 's/^linux-modules-nvidia-(.*)-"
                     f"{kern} .*/\\1/'", f"avail:{site}", 300)
        candidates = []
        for line in out.strip().splitlines():
            token = line.strip()
            if not token or not token[0].isdigit():
                continue
            head, _, variant = token.partition("-")
            if not head.isdigit():
                continue
            candidates.append((int(head), variant))     # variant may be ""
        if not candidates:
            results[site] = f"no prebuilt nvidia module for kernel {kern}"
            return

        # Prefer the plain proprietary variant, then -server, then -open:
        # plain is the best-tested pairing on these images.
        rank = {"": 0, "server": 1, "open": 2, "server-open": 3}
        candidates.sort(key=lambda c: (-c[0], rank.get(c[1], 9)))
        base, variant = candidates[0]
        version = f"{base}-{variant}" if variant else str(base)
        driver_pkg = f"nvidia-driver-{version}"
        module_pkg = f"linux-modules-nvidia-{version}-{kern}"

        results[site] = f"aligning on driver {version} for {kern}"
        run(node,
            "sudo DEBIAN_FRONTEND=noninteractive apt-get -y purge "
            "'*nvidia*' >/dev/null 2>&1; sudo apt-get -y autoremove "
            ">/dev/null 2>&1; sudo dpkg --configure -a >/dev/null 2>&1; "
            "echo cleaned", f"purge:{site}", 1800)
        out, err = run(node,
                       f"sudo DEBIAN_FRONTEND=noninteractive apt-get -y "
                       f"install {module_pkg} {driver_pkg} && echo INSTALL_OK",
                       f"install:{site}", 3000)
        if "INSTALL_OK" not in out:
            results[site] = f"install FAILED: {(err or out)[-160:]}"
            return
        run(node, "sudo modprobe nvidia 2>&1 | tail -1", f"modprobe:{site}", 300)

        watts = probe_watts(node, site)
        if watts is None and do_reboot:
            results[site] = "rebooting to load the module"
            run(node, "sudo nohup reboot &>/dev/null & echo rebooting",
                f"reboot:{site}", 120)
            if not wait_for_ssh(node, site):
                results[site] = "did not come back after reboot"
                return
            watts = probe_watts(node, site)
        if watts is None:
            results[site] = ("module installed but NVML still silent"
                             + ("" if do_reboot else "; re-run with --reboot"))
            return

        # python side (idempotent, cheap if already present)
        out, err = run(node,
                       "python3 -m pip -q install --user nvidia-ml-py && "
                       "echo PYNVML_OK", f"pynvml:{site}", 900)
        if "PYNVML_OK" not in out:
            results[site] = f"pynvml FAILED: {(err or out)[-160:]}"
            return
        out, _ = run(node, "python3 -c 'import torch' 2>/dev/null && "
                           "echo HAVE_TORCH || echo NO_TORCH",
                     f"torchchk:{site}", 300)
        if "NO_TORCH" in out:
            out, err = run(node,
                           "python3 -m pip -q install --user torch "
                           "--index-url https://download.pytorch.org/whl/cu121"
                           " && echo TORCH_OK", f"torch:{site}", 3600)
            if "TORCH_OK" not in out:
                results[site] = f"torch FAILED: {(err or out)[-160:]}"
                return
        name, _ = run(node, "nvidia-smi --query-gpu=name --format=csv,noheader "
                            "| head -1", f"name:{site}", 240)
        cuda, _ = run(node, "python3 -c 'import torch;"
                            "print(torch.cuda.is_available())' 2>&1 | tail -1",
                      f"cuda:{site}", 600)
        results[site] = (f"OK | {name.strip()} | driver {version} | "
                         f"NVML {watts} W | torch.cuda={cuda.strip()[:6]}")
    except Exception as e:
        results[site] = f"EXCEPTION {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reboot", action="store_true",
                    help="reboot nodes whose driver needs it")
    ap.add_argument("--config", default=os.path.join(ROOT, "config",
                                                     "sites.yaml"))
    ap.add_argument("--slice-name", default=None,
                    help="override the slice from config")
    ap.add_argument("--nodes", default=None,
                    help="comma-separated node names (default: config sites). "
                         "The Pegasus cluster names GPU workers '<SITE>-gpu'.")
    args = ap.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    fablib = FablibManager()
    slice_ = fablib.get_slice(
        name=args.slice_name or cfg["fabric"]["slice_name"])
    if args.nodes == "auto":
        # Every *-gpu node in the slice — the pegasus pool's worker naming,
        # so a dynamically-chosen site set (provision.py --sites auto)
        # needs no hand-maintained node list here.
        sites = sorted(n.get_name() for n in slice_.get_nodes()
                       if n.get_name().endswith("-gpu"))
        if not sites:
            sys.exit("--nodes auto found no *-gpu nodes in this slice")
    elif args.nodes:
        sites = [n.strip() for n in args.nodes.split(",") if n.strip()]
    else:
        sites = cfg["fabric"]["sites"]
    results = {}
    threads = []
    for site in sites:
        node = slice_.get_node(name=site)
        t = threading.Thread(target=setup,
                             args=(node, site, results, args.reboot),
                             daemon=True)
        t.start()
        threads.append(t)
        print(f"  {site}: started")
    for t in threads:
        t.join()

    print("\nResults:")
    ok = 0
    for site in sites:
        line = results.get(site, "no result")
        print(f"  {site}: {line}")
        ok += line.startswith("OK") or line.startswith("already OK")
    print(f"\n{ok}/{len(sites)} nodes have measured GPU power available")
    if ok < len(sites):
        print("Nodes without NVML cannot contribute to the CO2 arithmetic — "
              "the engine pauses accounting for them rather than guessing.")


if __name__ == "__main__":
    main()
