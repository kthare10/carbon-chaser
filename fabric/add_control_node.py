#!/usr/bin/env python3
"""Add a control node to the slice and deploy the orchestrator onto it.

Puts the dashboard and the migration policy inside FABRIC instead of on the
operator's laptop: the control node reaches workers over FABNetv4 with the
shared relay key, so it needs no FABRIC token and no bastion hop. Checkpoints
still move worker-to-worker and never transit this node.

Usage:  python fabric/add_control_node.py [--site STAR] [--port 8080]
"""

import argparse
import os
import shlex

import yaml
from fabrictestbed_extensions.fablib.fablib import FablibManager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
IPS_FILE = os.path.join(HERE, "dataplane_ips.yaml")
CTRL = "CTRL"

# Files the orchestrator needs on the control node.
PAYLOAD = [
    ("carbon_chaser/__init__.py", "carbon_chaser/__init__.py"),
    ("carbon_chaser/clock.py", "carbon_chaser/clock.py"),
    ("carbon_chaser/carbon.py", "carbon_chaser/carbon.py"),
    ("carbon_chaser/engine.py", "carbon_chaser/engine.py"),
    ("carbon_chaser/executor.py", "carbon_chaser/executor.py"),
    ("carbon_chaser/dashboard.py", "carbon_chaser/dashboard.py"),
    ("carbon_chaser/fabric_metrics.py", "carbon_chaser/fabric_metrics.py"),
    ("carbon_chaser/main.py", "carbon_chaser/main.py"),
    ("carbon_chaser/power.py", "carbon_chaser/power.py"),
    ("carbon_chaser/workload/train_gpu.py",
     "carbon_chaser/workload/train_gpu.py"),
    ("fabric/fetch_traces.py", "fabric/fetch_traces.py"),
    ("carbon_chaser/static/index.html", "carbon_chaser/static/index.html"),
    ("carbon_chaser/static/us-map.js", "carbon_chaser/static/us-map.js"),
    ("config/sites.yaml", "config/sites.yaml"),
    ("fabric/dataplane_ips.yaml", "fabric/dataplane_ips.yaml"),
    ("config/traces/eia.csv", "config/traces/eia.csv"),
    ("config/traces/eia.csv.meta.json", "config/traces/eia.csv.meta.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="STAR", help="site for the control node")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "sites.yaml"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    slice_name = cfg["fabric"]["slice_name"]

    fablib = FablibManager()
    slice = fablib.get_slice(name=slice_name)

    existing = [n.get_name() for n in slice.get_nodes()]
    if CTRL not in existing:
        print(f"Adding {CTRL} at {args.site} to slice '{slice_name}'…")
        node = slice.add_node(name=CTRL, site=args.site, cores=2, ram=4,
                              disk=20, image=cfg["fabric"].get(
                                  "image", "default_ubuntu_22"))
        iface = node.add_component(model="NIC_Basic",
                                   name=f"{CTRL}-nic").get_interfaces()[0]
        slice.add_l3network(name=f"net-{CTRL}", interfaces=[iface], type="IPv4")
        slice.submit()
        slice = fablib.get_slice(name=slice_name)
    else:
        print(f"{CTRL} already present; redeploying onto it")

    node = slice.get_node(name=CTRL)

    # Dataplane addressing, so it can reach the workers.
    try:
        net = slice.get_network(name=f"net-{CTRL}")
        iface = node.get_interface(network_name=f"net-{CTRL}")
        addr = net.get_available_ips()[0]
        iface.ip_addr_add(addr=addr, subnet=net.get_subnet())
        node.ip_route_add(subnet=fablib.FABNETV4_SUBNET,
                          gateway=net.get_gateway())
        print(f"  {CTRL}: dataplane {addr}")
    except Exception as e:
        print(f"  {CTRL}: dataplane already configured or failed: {e}")

    print("Installing dependencies…")
    node.execute(
        "sudo apt-get -qq update && "
        "DEBIAN_FRONTEND=noninteractive sudo apt-get -qq install -y "
        "python3-pip python3-numpy util-linux && "
        "python3 -m pip -q install --user fastapi uvicorn pyyaml requests",
        quiet=True)

    print("Installing the relay key (to reach workers over FABNetv4)…")
    worker = next(n for n in slice.get_nodes()
                  if n.get_name() in cfg["fabric"]["sites"])
    priv, _ = worker.execute("cat ~/.ssh/ccrelay", quiet=True)
    node.execute(
        f"mkdir -p ~/.ssh && printf '%s' {shlex.quote(priv)} > ~/.ssh/ccrelay "
        f"&& chmod 600 ~/.ssh/ccrelay", quiet=True)

    print("Uploading the orchestrator…")
    node.execute("mkdir -p carbon-chaser/carbon_chaser/workload "
                 "carbon-chaser/carbon_chaser/static carbon-chaser/config "
                 "carbon-chaser/config/traces "
                 "carbon-chaser/fabric carbon-chaser/runs", quiet=True)
    for local, remote in PAYLOAD:
        node.upload_file(os.path.join(ROOT, local), f"carbon-chaser/{remote}")
    print(f"  uploaded {len(PAYLOAD)} files")

    with open(IPS_FILE) as f:
        ips = yaml.safe_load(f)
    print("Verifying the control node can reach every worker…")
    ok = True
    for site, ip in ips.items():
        out, _ = node.execute(
            f"ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
            f"-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "
            f"-i ~/.ssh/ccrelay ubuntu@{ip} 'echo REACHED' 2>/dev/null || true",
            quiet=True)
        state = "OK" if "REACHED" in (out or "") else "UNREACHABLE"
        ok = ok and state == "OK"
        print(f"  {CTRL} -> {site} ({ip}): {state}")
    if not ok:
        print("WARNING: some workers unreachable from the control node")

    mgmt = node.get_management_ip()
    print(f"""
Control node ready.

Start the dashboard on it:
  ssh {node.get_username()}@{mgmt} \\
    'cd carbon-chaser && nohup python3 -m carbon_chaser.main --mode ssh \\
       --accel 300 --port {args.port} > orchestrator.log 2>&1 &'

View it from your laptop (management is IPv6-only on FABRIC):
  ssh -L {args.port}:localhost:{args.port} {node.get_username()}@{mgmt}
  then open http://localhost:{args.port}
""")


if __name__ == "__main__":
    main()
