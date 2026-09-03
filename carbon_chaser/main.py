"""Entry point.

  python -m carbon_chaser.main --mode sim              # laptop demo / fallback
  python -m carbon_chaser.main --mode fabric           # drive the FABRIC slice
  EMAPS_TOKEN=... python -m carbon_chaser.main ...     # live carbon data

Dashboard at http://localhost:8080
"""

import argparse
import os
import threading

import uvicorn
import yaml

from .carbon import build_provider, plan_carbon_source
from .power import make_power_provider
from .clock import Clock, SimClock
from .dashboard import create_app
from .engine import Engine
from .executor import LocalExecutor

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "sites.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "fabric", "ssh"], default="sim",
                    help="sim: all local; fabric: drive the slice via fablib "
                         "from outside; ssh: run ON a slice control node and "
                         "drive workers over FABNetv4")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--force-start", action="store_true",
                    help="start even if a node cannot be verified idle "
                         "(risks a duplicate trainer; use only when you know "
                         "those nodes are clear)")
    ap.add_argument("--accel", type=float, default=None,
                    help="override simulator acceleration (sim-s per real-s)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sites = cfg["sites"]
    policy = cfg["policy"]
    replay_cfg = cfg.get("replay", cfg.get("simulator", {}))

    # Decide the carbon source first: the clock depends on it, and a trace
    # that does not cover every configured zone has to be rejected before
    # the engine starts rather than failing on every tick.
    plan = plan_carbon_source(cfg.get("carbon"),
                              {s["zone"] for s in sites.values()})
    if plan["kind"] == "none":
        raise SystemExit(
            f"[carbon-chaser] refusing to start: {plan['reason']}\n"
            f"  There is no simulator to fall back on by design — a "
            f"carbon-aware scheduler running on invented carbon data is not "
            f"one.\n  Fetch a trace:  python fabric/fetch_traces.py "
            f"--source eia --days 7")
    if plan["reason"]:
        print(f"[carbon-chaser] NOTE {plan['reason']}", flush=True)
    if plan["kind"] == "live":
        clock = Clock()  # real time; live data moves at grid speed
    else:
        accel = args.accel or replay_cfg.get("accel", 300)
        clock = SimClock(accel=accel,
                         start_hour=replay_cfg.get("start_hour", 6.0))
    provider = build_provider(clock, plan)

    if args.mode == "ssh":
        # Running on a slice control node: workers are reached directly over
        # FABNetv4, so no fablib/token/bastion is needed here.
        import yaml as _yaml
        from .executor import SshExecutor
        fab = cfg["fabric"]
        ips_file = os.path.join(os.path.dirname(args.config), "..", "fabric",
                                "dataplane_ips.yaml")
        with open(ips_file) as f:
            ips = _yaml.safe_load(f)
        managed = [s for s in fab["sites"] if s in ips]
        executor = SshExecutor(managed, ips,
                               transfer_timeout_s=cfg.get("network", {}).get(
                                   "transfer_timeout_s", 300),
                               workload=fab.get("workload", "train_gpu.py"),
                               checkpoint_file=fab.get("checkpoint_file",
                                                       "checkpoint.pt"))
    elif args.mode == "fabric":
        from .executor import FabricExecutor
        fab = cfg["fabric"]
        ips_file = os.path.join(os.path.dirname(args.config), "..", "fabric",
                                "dataplane_ips.yaml")
        executor = FabricExecutor(
            fab["slice_name"], fab["sites"], dataplane_ips_file=ips_file,
            transfer_timeout_s=cfg.get("network", {}).get(
                "transfer_timeout_s", 300))
        if not executor.dataplane_ips:
            print("[carbon-chaser] WARNING: no fabric/dataplane_ips.yaml — "
                  "checkpoint transfers will relay through this host and will "
                  "NOT traverse the priced dataplane path.", flush=True)
        managed = fab["sites"]
    else:
        executor = LocalExecutor()
        managed = list(sites)

    net_cfg = cfg.get("network", {})
    net_model = None
    if net_cfg.get("enabled", False):
        from .fabric_metrics import FabricNetworkModel
        net_model = FabricNetworkModel()

    power = make_power_provider(policy, executor)
    pw = power.describe()
    print(f"[carbon-chaser] power={pw['kind']} ({pw['detail']})", flush=True)

    engine = Engine(sites, cfg["start_site"], provider, executor, clock,
                    policy, managed_sites=managed,
                    net_model=net_model, net_cfg=net_cfg,
                    state_file=os.path.join("runs", "engine_state.json"),
                    force_start=args.force_start,
                    power_provider=power)
    threading.Thread(target=engine.run, daemon=True).start()

    prov = provider.describe()
    print(f"[carbon-chaser] mode={args.mode} "
          f"carbon={prov['kind']} ({prov['detail']}) "
          f"dashboard=http://localhost:{args.port}", flush=True)
    try:
        uvicorn.run(create_app(engine), host="0.0.0.0", port=args.port,
                    log_level="warning")
    finally:
        engine.stop()
        if hasattr(executor, "shutdown"):
            executor.shutdown()


if __name__ == "__main__":
    main()
