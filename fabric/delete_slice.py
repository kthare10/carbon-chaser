#!/usr/bin/env python3
"""Tear down a carbon-chaser slice.

Takes the name explicitly. It used to read `fabric.slice_name` from
config/sites.yaml, which said `carbon-chaser` long after the working slice had
become `carbon-chaser-pegasus` — so the script silently targeted a slice that
did not exist, and would have reported success for deleting nothing. This
account also holds unrelated slices (other projects), so a script that
resolves a name indirectly and then calls delete() is a footgun.

It therefore: shows what it is about to delete, refuses if the name does not
exist, and waits for the slice to actually leave the listing — FABRIC's
delete() is asynchronous, and a slice can sit in Closing.

Usage:
    python fabric/delete_slice.py --slice-name carbon-chaser-pegasus
    python fabric/delete_slice.py --list
"""

import argparse
import os
import time

import yaml
from fabrictestbed_extensions.fablib.fablib import FablibManager

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-name",
                    help="the slice to delete. Defaults to fabric.slice_name "
                         "in the config, which may be stale — prefer passing "
                         "it explicitly.")
    ap.add_argument("--config",
                    default=os.path.join(HERE, "..", "config", "sites.yaml"))
    ap.add_argument("--list", action="store_true",
                    help="list slices and exit, deleting nothing")
    ap.add_argument("--wait", type=int, default=300,
                    help="seconds to wait for the slice to disappear")
    args = ap.parse_args()

    fablib = FablibManager()
    existing = {s.get_name(): s.get_state() for s in fablib.get_slices()}

    if args.list:
        for name, state in existing.items():
            print(f"{name:30s} {state}")
        return 0

    name = args.slice_name
    if not name:
        with open(args.config) as handle:
            name = yaml.safe_load(handle)["fabric"]["slice_name"]
        print(f"--slice-name not given; config says {name!r}")

    if name not in existing:
        # Refusing beats printing "deleted" for a slice that was never there.
        print(f"No slice named {name!r}. Existing: {sorted(existing)}\n"
              f"Nothing deleted.")
        return 1

    print(f"Deleting {name!r} (state {existing[name]}).")
    slice_obj = fablib.get_slice(name=name)
    try:
        print("  nodes:", [n.get_name() for n in slice_obj.get_nodes()])
    except Exception:
        pass
    slice_obj.delete()

    # delete() returns before the slice is gone, so confirm rather than assume.
    deadline = time.time() + args.wait
    while time.time() < deadline:
        states = {s.get_name(): s.get_state()
                  for s in FablibManager().get_slices()}
        if name not in states:
            print(f"Slice {name!r} is gone.")
            return 0
        print(f"  still present: {states[name]}", flush=True)
        time.sleep(20)
    print(f"Slice {name!r} did not disappear within {args.wait}s — it may be "
          f"stuck in Closing, which can leak resources. Check the portal.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
