#!/usr/bin/env python3
"""Make HTCondor name resolution survive a reboot, then restart the pool.

FABRIC images ship cloud-init with `manage_etc_hosts: True`, which
*regenerates* /etc/hosts on every boot. Appending pool entries therefore
works right up until something reboots the node — and installing NVIDIA
drivers reboots the node. The symptom is not obviously a hosts problem:

    StartLog: Can't resolve collector submit; skipping update
    MasterLog: WARNING: Saw slow DNS query ... getaddrinfo(submit) took 5.09s

i.e. the worker silently falls back to DNS, times out, and never joins.

Fix, applied idempotently: turn off cloud-init's hosts management, write the
FABNetv4 pool entries into BOTH /etc/hosts and cloud-init's template (so a
future reboot cannot undo it), then restart condor and verify the pool.

Run this AFTER any step that reboots nodes.
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
MARK = "# carbon-chaser pool"


def run(node, cmd, label, budget=600):
    out, err = call_with_timeout(
        lambda: node.execute(cmd, quiet=True, retry=1), budget, label)
    return (out or ""), (err or "")


def node_addr(node):
    for iface in node.get_interfaces():
        addr = iface.get_ip_addr()
        if addr:
            return str(addr)
    return None


def fix(node, name, block, results):
    try:
        # 1. stop cloud-init from rewriting /etc/hosts on boot
        run(node,
            "sudo sed -i 's/^manage_etc_hosts:.*/manage_etc_hosts: false/' "
            "/etc/cloud/cloud.cfg; "
            "grep -q '^manage_etc_hosts' /etc/cloud/cloud.cfg || "
            "echo 'manage_etc_hosts: false' | sudo tee -a /etc/cloud/cloud.cfg "
            ">/dev/null; echo CLOUDINIT", f"cloudinit:{name}")

        # 2. write the block to /etc/hosts and to the cloud-init template, so
        #    a reboot re-creates it rather than erasing it
        script = (
            f"set -e\n"
            f"for target in /etc/hosts "
            f"/etc/cloud/templates/hosts.debian.tmpl; do\n"
            f"  [ -f \"$target\" ] || continue\n"
            f"  sudo sed -i '/{MARK}/,$ d' \"$target\"\n"
            f"  printf '%s\\n' '{MARK}' | sudo tee -a \"$target\" >/dev/null\n"
            f"{block}"
            f"done\n"
            f"sudo sed -i 's/^127\\.0\\.1\\.1 /#127.0.1.1 /' /etc/hosts\n"
            f"echo HOSTS_OK\n")
        out, err = run(node, script, f"hosts:{name}")
        if "HOSTS_OK" not in out:
            results[name] = f"hosts FAILED: {(err or out)[-160:]}"
            return

        # 3. prove resolution works before blaming condor
        out, _ = run(node, "getent hosts submit || echo NO_RESOLVE",
                     f"resolve:{name}")
        if "NO_RESOLVE" in out:
            results[name] = "submit still does not resolve"
            return

        run(node, "sudo systemctl restart condor && sleep 5 && echo RESTARTED",
            f"restart:{name}")
        results[name] = f"ok ({out.strip().split()[0]} -> submit)"
    except Exception as exc:
        results[name] = f"EXCEPTION {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-name", default="carbon-chaser-pegasus")
    args = ap.parse_args()

    fablib = FablibManager()
    sl = fablib.get_slice(name=args.slice_name)
    nodes = {n.get_name(): n for n in sl.get_nodes()}
    addrs = {name: node_addr(n) for name, n in nodes.items()}
    if any(a is None for a in addrs.values()):
        sys.exit(f"missing FABNetv4 addresses: {addrs}")

    print("Pool addresses:")
    for name, addr in sorted(addrs.items()):
        print(f"  {addr:16s} {name}")

    block = "".join(
        f"  printf '%s %s\\n' '{addr}' '{name}' | sudo tee -a \"$target\" "
        f">/dev/null\n"
        for name, addr in sorted(addrs.items()))

    results = {}
    threads = [threading.Thread(target=fix, args=(n, name, block, results),
                                daemon=True)
               for name, n in nodes.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\nResults:")
    for name in sorted(results):
        print(f"  {name}: {results[name]}")

    print("\nPool status (allow ~30s for workers to report):")
    out, _ = run(nodes["submit"],
                 "sleep 25; condor_status -af Name FabricSite CarbonIntensity "
                 "GPUWatts DetectedGPUs 2>&1 | head -10", "status:submit", 900)
    print(out)


if __name__ == "__main__":
    main()
