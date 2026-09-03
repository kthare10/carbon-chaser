#!/usr/bin/env python3
"""Create the carbon-chaser FABRIC slice: one GPU VM per site.

GPUs are not optional here. Power is measured from NVML (FABRIC GPUs are PCI
passthrough, so the board's own sensor is readable inside the VM), and the
demo refuses to invent a power figure — so a node without a GPU cannot
contribute to the CO2 arithmetic at all.

Reads config/sites.yaml (fabric section). Each node is named exactly like
its site (CLEM, TACC, ...) — FabricExecutor relies on that. Every node gets
a FABNetv4 interface so checkpoints move node-to-node across the FABRIC
dataplane — the same inter-rack links the network model prices — rather than
being relayed through the orchestrator's management connection.

Writes fabric/dataplane_ips.yaml (site -> FABNetv4 address) for the
executor, and installs a shared inter-node SSH key so scp between VMs works
unattended.

Usage:  python fabric/create_slice.py [--config config/sites.yaml]
"""

import argparse
import os
import shlex

import yaml
from fabrictestbed_extensions.fablib.fablib import FablibManager

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(HERE, "..", "carbon_chaser", "workload",
                        "train_gpu.py")
IPS_FILE = os.path.join(HERE, "dataplane_ips.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config", "sites.yaml"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    fab = cfg["fabric"]

    fablib = FablibManager()
    slice_name = fab["slice_name"]
    print(f"Creating slice '{slice_name}' with nodes at {fab['sites']}")

    slice = fablib.new_slice(name=slice_name)
    for site in fab["sites"]:
        node = slice.add_node(
            name=site, site=site,
            cores=fab.get("cores", 8), ram=fab.get("ram", 32),
            disk=fab.get("disk", 100),
            image=fab.get("image", "default_ubuntu_22"),
        )
        gpu_model = (fab.get("gpus") or {}).get(site)
        if not gpu_model:
            raise SystemExit(
                f"no GPU model configured for {site}; measured power is "
                f"mandatory, so every worker needs a GPU (see fabric.gpus)")
        node.add_component(model=gpu_model, name=f"{site}-gpu")
        print(f"  {site}: requesting {gpu_model}")
        # FABNetv4 gives every node a routable dataplane address, so
        # checkpoints cross the FABRIC backbone instead of the mgmt network.
        iface = node.add_component(model="NIC_Basic",
                                   name=f"{site}-nic").get_interfaces()[0]
        slice.add_l3network(name=f"net-{site}", interfaces=[iface],
                            type="IPv4")
    slice.submit()

    print("Provisioned; configuring dataplane addressing…")
    slice = fablib.get_slice(name=slice_name)
    dataplane_ips = {}
    for site in fab["sites"]:
        node = slice.get_node(name=site)
        net = slice.get_network(name=f"net-{site}")
        iface = node.get_interface(network_name=f"net-{site}")
        addr = net.get_available_ips()[0]
        iface.ip_addr_add(addr=addr, subnet=net.get_subnet())
        # Route to the rest of FABNetv4 via this network's gateway.
        node.ip_route_add(subnet=fablib.FABNETV4_SUBNET,
                          gateway=net.get_gateway())
        dataplane_ips[site] = str(addr)
        print(f"  {site}: dataplane {addr}")

    with open(IPS_FILE, "w") as f:
        yaml.safe_dump(dataplane_ips, f)
    print(f"Wrote {IPS_FILE}")

    print("Installing workload and inter-node SSH key…")
    # One keypair shared by the slice's nodes so scp between VMs is
    # non-interactive. Scoped to this throwaway demo slice.
    keygen = ("test -f ~/.ssh/ccrelay || ssh-keygen -q -t ed25519 "
              "-f ~/.ssh/ccrelay -N ''")
    first = slice.get_node(name=fab["sites"][0])
    first.execute(keygen, quiet=True)
    priv, _ = first.execute("cat ~/.ssh/ccrelay", quiet=True)
    pub, _ = first.execute("cat ~/.ssh/ccrelay.pub", quiet=True)

    for site in fab["sites"]:
        node = slice.get_node(name=site)
        node.upload_file(TRAIN_PY, os.path.basename(TRAIN_PY))
        # util-linux (flock) enforces one trainer per node. nvidia drivers +
        # torch + pynvml are what make power a measurement rather than a
        # guess; this is the slow step (GBs) and the main failure risk.
        node.execute(
            "sudo apt-get -qq update && "
            "DEBIAN_FRONTEND=noninteractive sudo apt-get -qq install -y "
            "util-linux python3-pip ubuntu-drivers-common && "
            "sudo ubuntu-drivers install --gpgpu || true",
            quiet=True,
        )
        node.execute(
            "python3 -m pip -q install --user nvidia-ml-py torch "
            "--index-url https://download.pytorch.org/whl/cu121 "
            "|| python3 -m pip -q install --user nvidia-ml-py torch",
            quiet=True,
        )
        out, _ = node.execute(
            "nvidia-smi --query-gpu=name,power.draw --format=csv,noheader "
            "2>/dev/null || echo NO_GPU", quiet=True)
        print(f"  {site}: nvidia-smi -> {out.strip()[:60]}")
        out, _ = node.execute(
            "python3 -c \'import pynvml;pynvml.nvmlInit();"
            "h=pynvml.nvmlDeviceGetHandleByIndex(0);"
            "print(pynvml.nvmlDeviceGetPowerUsage(h)/1000.0,\"W\")\' "
            "2>/dev/null || echo NVML_UNAVAILABLE", quiet=True)
        print(f"  {site}: NVML -> {out.strip()[:60]}")
        out, _ = node.execute("command -v flock >/dev/null && echo HAVE_FLOCK",
                              quiet=True)
        if "HAVE_FLOCK" not in (out or ""):
            print(f"  WARNING: {site} has no flock — duplicate-trainer "
                  f"protection will not work on this node")
        node.execute(
            f"mkdir -p ~/.ssh && printf '%s' {shlex.quote(priv)} > ~/.ssh/ccrelay && "
            f"chmod 600 ~/.ssh/ccrelay && "
            f"printf '%s\\n' {shlex.quote(pub.strip())} >> ~/.ssh/authorized_keys && "
            f"sort -u -o ~/.ssh/authorized_keys ~/.ssh/authorized_keys",
            quiet=True,
        )
        print(f"  {site}: ready")

    print("Verifying dataplane reachability between nodes…")
    ok = True
    for site in fab["sites"]:
        node = slice.get_node(name=site)
        for peer, ip in dataplane_ips.items():
            if peer == site:
                continue
            out, _ = node.execute(f"ping -c1 -W3 {ip} >/dev/null && echo OK",
                                  quiet=True)
            state = "OK" if "OK" in (out or "") else "UNREACHABLE"
            if state != "OK":
                ok = False
            print(f"  {site} -> {peer} ({ip}): {state}")
    if not ok:
        print("WARNING: some pairs unreachable — those migrations will fall "
              "back to relay-through-orchestrator (not a dataplane path).")
    print("Done. Run:  python -m carbon_chaser.main --mode fabric")


if __name__ == "__main__":
    main()
