#!/usr/bin/env python3
"""Provision a Pegasus/HTCondor cluster on FABRIC with carbon-aware GPU workers.

Structure and the node_tools scripts come from ~/pegasus/pegasus
(pegasus-fabric.ipynb) — submit node as HTCondor central manager plus workers
across sites, FABNetv4 via add_fabnet(), SSH key exchange, /etc/hosts. That
part is proven; this script is the automated form of it.

Two deliberate differences:

1. NVIDIA drivers are NOT installed by node_tools. The notebook path uses
   `ubuntu-drivers autoinstall` / the CUDA repo metapackage, which pull the
   newest driver — frequently with no prebuilt module for the running kernel
   ("Driver/library version mismatch"), and DKMS compounds it because apt
   often lands a newer kernel in the same transaction. Drivers come from
   fabric/install_gpu_stack.py, which picks the highest driver that HAS a
   prebuilt module for the running kernel and is idempotent.
2. Workers advertise `CarbonIntensity` and `GPUWatts` as ClassAd attributes
   (STARTD_CRON -> carbon_classad.py), so HTCondor matchmaking itself becomes
   carbon-aware and a job only needs `RANK = -CarbonIntensity`.

Usage:
    python pegasus/provision.py --sites CLEM,TACC,UTAH --submit-site STAR
"""

import argparse
import os
import re
import sys
import threading
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbon_chaser.executor import call_with_timeout  # noqa: E402

from fabrictestbed_extensions.fablib.fablib import FablibManager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
IMAGE = "default_ubuntu_22"      # matches our tested driver recipe


def log(msg):
    print(msg, flush=True)


def create_slice(fablib, name, submit_site, gpu_sites, gpu_models, specs):
    log(f"Creating slice '{name}': submit@{submit_site}, GPU workers "
        f"{gpu_sites}")
    sl = fablib.new_slice(name=name)

    submit = sl.add_node(name="submit", site=submit_site, image=IMAGE,
                         cores=specs["submit_cores"], ram=specs["submit_ram"],
                         disk=specs["submit_disk"])
    submit.add_fabnet()

    for site in gpu_sites:
        model = gpu_models[site]
        node = sl.add_node(name=f"{site}-gpu", site=site, image=IMAGE,
                           cores=specs["worker_cores"],
                           ram=specs["worker_ram"], disk=specs["worker_disk"])
        node.add_component(model=model, name="gpu1")
        node.add_fabnet()
        log(f"  {site}: {model}, {specs['worker_cores']}c/"
            f"{specs['worker_ram']}G/{specs['worker_disk']}G")

    for ifc in sl.get_interfaces():
        ifc.set_mode("auto")          # let fablib assign FABNet addresses
    sl.submit()
    return fablib.get_slice(name=name)


def node_addr(node):
    """FABNetv4 address of the node's dataplane interface."""
    for iface in node.get_interfaces():
        addr = iface.get_ip_addr()
        if addr:
            return str(addr)
    return None


def run(node, cmd, label, budget=2400):
    out, err = call_with_timeout(
        lambda: node.execute(cmd, quiet=True, retry=1), budget, label)
    return (out or ""), (err or "")


def install_base(node, name, results):
    """HTCondor + Pegasus + hostname fix, from the proven node_tools.

    Idempotent: re-running the installers on a configured node is not just
    wasted time, it can leave apt half-configured. Always refresh node_tools
    (the scripts change) but skip the installs when both are already present.
    """
    try:
        run(node, r"sudo sed -i 's/^127\.0\.1\.1 /#127.0.1.1 /' /etc/hosts",
            f"hosts:{name}", 300)
        node.upload_directory(os.path.join(HERE, "node_tools"), ".")
        out, _ = run(node,
                     "command -v condor_status >/dev/null && "
                     "command -v pegasus-version >/dev/null && echo HAVE_BOTH",
                     f"probe:{name}", 300)
        if "HAVE_BOTH" in out:
            versions, _ = run(node,
                              "condor_version | head -1; pegasus-version",
                              f"ver:{name}", 300)
            results[name] = ("base ok (already installed: "
                             + " / ".join(v.strip() for v in
                                          versions.strip().splitlines()[:2])
                             + ")")
            return
        out, _ = run(node,
                     "cd node_tools && chmod +x *.sh && "
                     "sudo ./htcondor.sh --no-dry-run > /tmp/htcondor.log 2>&1"
                     " && sudo ./pegasus.sh --no-dry-run > /tmp/pegasus.log "
                     "2>&1 && echo BASE_OK", f"base:{name}", 3600)
        if "BASE_OK" not in out:
            tail, _ = run(node, "tail -5 /tmp/htcondor.log /tmp/pegasus.log",
                          f"baselog:{name}", 300)
            results[name] = f"base install FAILED: {tail[-200:]}"
            return
        results[name] = "base ok"
    except Exception as exc:
        results[name] = f"base EXCEPTION {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-name", default="carbon-chaser-pegasus")
    ap.add_argument("--sites", default="CLEM,TACC,UTAH",
                    help="GPU worker sites (comma separated), or `auto` / "
                         "`auto:N` to choose N sites from LIVE availability "
                         "and grid-zone diversity (see select_sites.py)")
    ap.add_argument("--submit-site", default="STAR")
    ap.add_argument("--config", default=os.path.join(ROOT, "config",
                                                     "sites.yaml"))
    ap.add_argument("--skip-create", action="store_true",
                    help="(kept for compatibility; reuse is now automatic — "
                         "an existing slice is detected and only "
                         "configuration is re-applied)")
    args = ap.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    specs = {"submit_cores": 8, "submit_ram": 16, "submit_disk": 100,
             # Workers must fit on the GPU's own host, not the site total.
             "worker_cores": 4, "worker_ram": 16, "worker_disk": 100}

    fablib = FablibManager()

    # Reuse before create: re-running this script against a live pool is
    # the normal way to push config fixes, so an existing slice means
    # "re-apply configuration", never "fail" and never "make another one".
    # Site selection is skipped for an existing slice — its sites are a
    # fact recorded in its node names, not a choice still open.
    try:
        sl = fablib.get_slice(name=args.slice_name)
    except Exception:
        sl = None
    if sl is not None:
        existing = sorted(n.get_name() for n in sl.get_nodes())
        log(f"Slice '{args.slice_name}' already exists "
            f"(lease ends {sl.get_lease_end()}) — reusing it and "
            f"re-applying configuration to: {existing}")
        if args.sites != ap.get_default("sites"):
            log(f"  NOTE: --sites {args.sites!r} is ignored for an existing "
                f"slice; delete it first to change the site set")
    else:
        if args.skip_create:
            sys.exit(f"--skip-create given but slice '{args.slice_name}' "
                     f"does not exist")
        if args.sites.startswith("auto"):
            # Dynamic: survey live availability, require trace zone
            # coverage, maximize grid-zone diversity. The GPU model per
            # site comes from what is actually free, not from the static
            # fabric.gpus map.
            from select_sites import choose_worker_sites
            count = (int(args.sites.split(":", 1)[1])
                     if ":" in args.sites else 3)
            trace = os.path.join(ROOT, "config", "traces", "eia.csv")
            if not os.path.exists(trace):
                sys.exit(f"missing {trace} — run fabric/fetch_traces.py "
                         f"first; --sites auto checks zone coverage "
                         f"against it")
            log("Choosing worker sites from live availability:")
            gpu_models = choose_worker_sites(
                fablib.get_available_resources(), cfg, count,
                args.submit_site, trace, log=log)
            sites = sorted(gpu_models)
        else:
            gpu_models = cfg["fabric"]["gpus"]
            sites = [s.strip() for s in args.sites.split(",") if s.strip()]
            missing = [s for s in sites if s not in gpu_models]
            if missing:
                sys.exit(f"no GPU model configured for {missing} (see "
                         f"fabric.gpus in {args.config}, or use "
                         f"--sites auto)")
        sl = create_slice(fablib, args.slice_name, args.submit_site, sites,
                          gpu_models, specs)

    nodes = {n.get_name(): n for n in sl.get_nodes()}
    submit = nodes["submit"]
    workers = {name: n for name, n in nodes.items() if name != "submit"}
    log(f"\nNodes: {sorted(nodes)}")

    addrs = {name: node_addr(n) for name, n in nodes.items()}
    log("FABNetv4 addresses:")
    for name, addr in addrs.items():
        log(f"  {name}: {addr}")
    if any(a is None for a in addrs.values()):
        sys.exit("some nodes have no FABNetv4 address; cannot form a pool")

    # --- base software, in parallel -------------------------------------
    log("\nInstalling HTCondor + Pegasus on all nodes (parallel, slow)…")
    results = {}
    threads = [threading.Thread(target=install_base, args=(n, name, results),
                                daemon=True)
               for name, n in nodes.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for name in sorted(results):
        log(f"  {name}: {results[name]}")
    if any(not v.startswith("base ok") for v in results.values()):
        sys.exit("base install failed somewhere; fix before continuing")

    # --- /etc/hosts so HTCondor can resolve pool members ----------------
    log("\nWriting /etc/hosts on every node…")
    hosts_lines = "\n".join(f"{addr} {name}" for name, addr in addrs.items())
    for name, node in nodes.items():
        run(node,
            f"sudo sed -i '/# carbon-chaser pool/,+{len(addrs)}d' /etc/hosts; "
            f"printf '# carbon-chaser pool\\n{hosts_lines}\\n' | "
            f"sudo tee -a /etc/hosts >/dev/null && echo HOSTS_OK",
            f"hosts:{name}", 300)

    # --- carbon trace + zone marker on each worker ----------------------
    log("\nStaging the carbon trace and zone markers…")
    trace = os.path.join(ROOT, "config", "traces", "eia.csv")
    if not os.path.exists(trace):
        sys.exit(f"missing {trace} — run fabric/fetch_traces.py first; there "
                 f"is no simulator to fall back on")
    zones = {name: cfg["sites"][name.split("-")[0]]["zone"]
             for name in workers}
    for name, node in workers.items():
        run(node, "sudo mkdir -p /opt/carbon && sudo chmod 755 /opt/carbon",
            f"mkdir:{name}", 300)
        node.upload_file(trace, "eia.csv")
        node.upload_file(os.path.join(HERE, "carbon_classad.py"),
                         "carbon_classad.py")
        run(node,
            f"sudo mv eia.csv /opt/carbon/eia.csv && "
            f"echo '{zones[name]}' | sudo tee /opt/carbon/zone >/dev/null && "
            f"sudo mv carbon_classad.py /usr/local/bin/carbon_classad.py && "
            f"sudo chmod 755 /usr/local/bin/carbon_classad.py && echo STAGED",
            f"stage:{name}", 900)
        # The HOST python needs exactly one package: pynvml, for the
        # STARTD_CRON (carbon_classad) — the jobs themselves run inside the
        # workflow's Apptainer image and never touch system python. It must
        # be SYSTEM-WIDE and verified as the condor user, because that is
        # who the cron runs as: a `--user` install under ubuntu satisfies an
        # interactive check while GPUWatts silently never appears (unknown
        # is absent by design, so nothing errors). The old
        # `2>/dev/null || true` here swallowed exactly that failure on a
        # live pool; now it is checked.
        out, _ = run(node,
                     # pip itself can be absent on a fresh node (the role
                     # script that installs it runs LATER in this flow),
                     # so bootstrap it here rather than fail on the very
                     # dependency check meant to be reliable.
                     "python3 -m pip --version >/dev/null 2>&1 || "
                     "sudo apt-get install -y -q python3-pip; "
                     "(sudo python3 -m pip install -q --break-system-packages "
                     "nvidia-ml-py 2>/dev/null || "
                     "sudo python3 -m pip install -q nvidia-ml-py)"
                     " && sudo -u condor python3 -c 'import pynvml' "
                     "&& echo PYDEPS_OK",
                     f"pydeps:{name}", 1800)
        if "PYDEPS_OK" not in out:
            sys.exit(f"{name}: pynvml did not install system-wide — the "
                     f"carbon cron would advertise no GPUWatts, silently")
        log(f"  {name}: zone {zones[name]}, pynvml verified as condor")

    # --- HTCondor roles -------------------------------------------------
    log("\nConfiguring HTCondor roles…")
    # HTCondor must bind to the FABNetv4 dataplane interface, not whatever
    # carries the default route (that is the management path, and on these
    # nodes the default route may not even be v4).
    def dataplane_iface(node, name, addr):
        out, _ = run(node,
                     f"ip -o -4 addr show | awk '$4 ~ /^{addr}\\//"
                     f" {{print $2; exit}}'", f"iface:{name}", 300)
        iface = out.strip().splitlines()
        if not iface:
            sys.exit(f"could not find the interface holding {addr} on {name}; "
                     f"HTCondor would bind to the wrong network")
        return iface[-1].strip()

    # Role failures ABORT, with the role script's own log tail. An earlier
    # version only PRINTED "FAILED" and carried on to "Provisioned." — a
    # pool whose worker role never applied looks fine at a glance (condor
    # runs, the node responds) but each worker is its own mini-pool
    # (stock 00-minicondor, CONDOR_HOST = itself) and nothing ever joins
    # the collector. That exact pool shipped once.
    def apply_role(node, name, script, args_str):
        logfile = f"/tmp/{name}-role-cfg.log"
        # The nonce is written BEFORE the role script runs, so the
        # script's own condor restart advertises it: the pool check then
        # requires THIS RUN's value in the ad. Freshness by wall-clock was
        # tried and cannot work — skew tolerance in one direction is
        # stale acceptance in the other — whereas a stale ad cannot
        # contain a nonce minted after it was published.
        out, _ = run(node,
                     f"printf 'STARTD_ATTRS = $(STARTD_ATTRS) "
                     f"ProvisionNonce\\nProvisionNonce = \"{provision_nonce}\"\\n' "
                     f"| sudo tee /etc/condor/config.d/"
                     f"60-provision-nonce.config >/dev/null && echo NONCE_OK",
                     f"nonce:{name}", 300)
        if "NONCE_OK" not in out:
            sys.exit(f"{name}: could not write the provision nonce — the "
                     f"pool check below would have nothing to verify")
        out, _ = run(node,
                     f"cd node_tools && sudo ./{script} {args_str} "
                     f"> {logfile} 2>&1 && echo ROLE_OK",
                     f"rolecfg:{name}", 3000)
        if "ROLE_OK" not in out:
            tail, _ = run(node, f"tail -8 {logfile}", f"rolelog:{name}", 300)
            sys.exit(f"{name}: {script} FAILED — the node would run a "
                     f"stock single-machine condor and never join the "
                     f"pool. Log tail ({logfile}):\n{tail}")
        # Early spot-check only (exact match — a substring test passed on
        # any output that happened to contain the word "submit"). This
        # reads the config FILES, so it can pass while the daemon restart
        # failed; the authoritative check is pool membership, below.
        out, _ = run(node, "condor_config_val CONDOR_HOST", f"chost:{name}",
                     300)
        if out.strip() not in ("submit", addrs["submit"]):
            sys.exit(f"{name}: role script reported success but "
                     f"CONDOR_HOST={out.strip()!r} does not point at the "
                     f"submit node — config not in effect")
        log(f"  {name}: role applied, CONDOR_HOST={out.strip()}")

    # Minted BEFORE any role applies. A true nonce (kernel UUID), not a
    # timestamp: a one-second-resolution epoch is shared by any two runs
    # inside the same second, which quietly turns "minted by THIS run"
    # back into a clock comparison — the exact property class the nonce
    # exists to escape.
    out, _ = run(submit, "cat /proc/sys/kernel/random/uuid", "nonce", 300)
    provision_nonce = out.strip().splitlines()[-1]
    # Canonical 8-4-4-4-12 shape, not merely "36 uuid-ish characters" — a
    # loose character-class check would bless any garbage line of the
    # right length (including all hyphens), and the whole membership
    # proof rides on this value being a genuine kernel UUID.
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                        r"[0-9a-f]{4}-[0-9a-f]{12}", provision_nonce):
        sys.exit(f"could not mint a provision nonce (got "
                 f"{provision_nonce!r}) — membership verification would "
                 f"be meaningless")

    submit_iface = dataplane_iface(submit, "submit", addrs["submit"])
    log(f"  submit dataplane interface: {submit_iface}")
    apply_role(submit, "submit", "fabric-submit.sh",
               f"{submit_iface} {addrs['submit']} submit")

    for name, node in workers.items():
        wiface = dataplane_iface(node, name, addrs[name])
        site = name.split("-")[0]
        apply_role(node, name, "fabric-worker-gpu.sh",
                   f"{wiface} {addrs['submit']} submit {site}")

    # --- pool-internal SSH: submit -> workers ------------------------------
    # The dashboard's inject/clear controls write carbon overrides on the
    # workers over plain ssh from the submit node, and operators debug the
    # same way. The original pegasus-fabric flow had a key exchange; these
    # node_tools do NOT (discovered on a live pool: "Permission denied
    # (publickey)" from submit, ~/.ssh holding only authorized_keys). Set
    # it up idempotently and PROVE it with a BatchMode round-trip — a key
    # that exists but was never authorized fails identically to no key.
    log("\nSetting up pool-internal SSH (submit -> workers)…")
    out, _ = run(submit,
                 "test -f ~/.ssh/id_ed25519 || ssh-keygen -q -t ed25519 "
                 "-N '' -f ~/.ssh/id_ed25519; cat ~/.ssh/id_ed25519.pub",
                 "poolkey", 300)
    pool_pubkey = out.strip().splitlines()[-1]
    if not pool_pubkey.startswith("ssh-ed25519 "):
        sys.exit(f"could not create/read the submit node's pool key "
                 f"(got {pool_pubkey[:60]!r})")
    for name, node in workers.items():
        run(node,
            f"grep -qxF '{pool_pubkey}' ~/.ssh/authorized_keys 2>/dev/null "
            f"|| echo '{pool_pubkey}' >> ~/.ssh/authorized_keys",
            f"authkey:{name}", 300)
    for name in workers:
        out, _ = run(submit,
                     f"ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
                     f"-o ConnectTimeout=10 {name} 'echo SSH_OK'",
                     f"sshprobe:{name}", 300)
        if "SSH_OK" not in out:
            sys.exit(f"submit -> {name} ssh does not work — the dashboard "
                     f"inject controls and operator debugging both depend "
                     f"on it: {out.strip()[-120:]}")
    log(f"  submit can reach all {len(workers)} workers over pool ssh")

    # The ONLY proof a worker's role is in effect is the collector serving
    # an ad that carries THIS RUN's ProvisionNonce. Config files can
    # read perfectly (condor_config_val passes) while the daemon failed to
    # restart; mere presence in condor_status is not enough (the collector
    # serves stale ads from a previous registration for ~15-20 min); and a
    # DaemonStartTime-vs-wall-clock comparison needs a skew grace that is
    # itself a stale-acceptance window. The nonce has none of those
    # failure modes: a stale ad cannot contain a value minted after it
    # was published. Poll with a settling window — a startd's first
    # update can take up to its update interval.
    log("\nVerifying pool membership at the collector…")
    missing = set(workers)
    for _ in range(12):                      # up to ~2 minutes
        out, _ = run(submit,
                     "condor_status -af Machine ProvisionNonce | sort -u",
                     "poolcheck", 300)
        fresh = set()
        for line in out.splitlines():
            parts = line.split()
            if (len(parts) == 2 and parts[0] in workers
                    and parts[1] == provision_nonce):
                fresh.add(parts[0])
        missing = set(workers) - fresh
        if not missing:
            break
        time.sleep(10)
    if missing:
        sys.exit(f"no ad carrying this run's nonce from: {sorted(missing)} "
                 f"— either the worker never joined, or only a stale "
                 f"pre-existing ad represents it. Check "
                 f"/var/log/condor/StartLog on those nodes (binding to the "
                 f"management interface and a failed daemon restart are "
                 f"the two observed causes).")
    log(f"  all {len(workers)} workers advertise this run's nonce")

    log("\nProvisioned. Next:")
    log("  1. python fabric/install_gpu_stack.py --reboot   # our driver recipe")
    log("  2. python pegasus/provision.py --skip-create     # if roles need redo")
    log("  3. check the pool:  condor_status -af Name FabricSite "
        "CarbonIntensity GPUWatts")


if __name__ == "__main__":
    main()
