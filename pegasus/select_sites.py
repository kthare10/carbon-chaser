#!/usr/bin/env python3
"""Choose GPU worker sites from LIVE testbed availability, not a hardcoded list.

`config/sites.yaml` remains the CANDIDATE UNIVERSE — the site -> grid-zone
mapping is curated knowledge (which balancing authority powers which rack)
that cannot be measured off the testbed — but which N candidates actually
get a VM is decided here, from measured facts at provision time:

1. **The zone must be in the carbon trace.** A worker whose zone the trace
   does not cover advertises no CarbonIntensity (absent, never zero), so
   `CarbonIntensity =!= UNDEFINED` keeps it from EVER winning a match — a
   perfectly healthy VM that can never run a segment. Cheaper to exclude
   here, with the reason printed, than to discover via an idle pool.
2. **A GPU must actually be available.** FABRIC's inventory is
   heterogeneous and contended; the advertised availability per model
   (GPU_RTX6000, GPU_A30, GPU_A40, GPU_TeslaT4) is queried live and the
   site's model is picked from what is free — not from a config entry that
   rots as the testbed changes.
3. **The host must fit the worker spec** (cores/ram/disk), for the same
   reason.

Selection then maximizes MIGRATION OPPORTUNITY: the set whose grids
actually cross over most often in the trace (see `count_switches`), with
free-GPU headroom as a tie-break.

That criterion replaced "maximize grid-zone diversity", which sounds
equivalent and is not. Diversity picked CISO+ERCO+PACE — three balancing
authorities, three time zones — in which CISO is the cleanest grid in
**100% of the 157-hour trace**. The resulting pool ran a full workflow in
which every placement correctly chose the minimum and no migration was
possible at any run length. Crossovers are the property the demo needs;
diversity is only a proxy for it, and a lossy one.

Every exclusion is reported with its reason. An empty or thin result is
stated, not padded: if only two eligible sites exist, the answer is two
sites and a printed explanation, never a silently degraded three.

Usage (dry run — prints the survey and the choice, provisions nothing):
    python pegasus/select_sites.py [--count 3] [--submit-site STAR]

provision.py consumes this via `--sites auto` (or `auto:4` for a count).
"""

import argparse
import csv
import itertools
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# Any of these satisfies "has a GPU with a measurable NVML sensor". Order is
# only a tie-break when a site offers several with equal availability.
#
# Two naming namespaces in fablib, discovered the hard way (every site
# reported 0 available): slice REQUESTS use the left-hand names
# (node.add_component(model="GPU_TeslaT4")), while the resources aggregate
# advertises the right-hand names ("GPU-Tesla T4"). Survey with the
# aggregate name, return the request name.
GPU_MODELS = {
    "GPU_RTX6000": "GPU-RTX6000",
    "GPU_A30": "GPU-A30",
    "GPU_A40": "GPU-A40",
    "GPU_TeslaT4": "GPU-Tesla T4",
}
GPU_PREFERENCE = list(GPU_MODELS)


def load_trace_series(trace_path):
    """{timestamp: {zone: intensity}} from the measured trace."""
    series = {}
    with open(trace_path) as handle:
        for row in csv.DictReader(handle):
            zone = row.get("zone")
            if not zone:
                continue
            try:
                value = float(row["carbon_intensity_gco2_kwh"])
            except (KeyError, TypeError, ValueError):
                continue
            series.setdefault(row["timestamp"], {})[zone] = value
    return series


def trace_zones(series):
    """Zones the measured carbon trace actually covers."""
    return {zone for row in series.values() for zone in row}


def count_switches(series, zones):
    """How many times the CLEANEST zone changes across the trace.

    This is the number that decides whether a pool can migrate at all,
    and it is NOT the same as zone diversity — the property this selector
    used to maximize. Measured on a live pool: CISO/ERCO/PACE is maximally
    zone-diverse (three balancing authorities, three time zones) and has
    **zero** switches, because CISO is cleanest in 100% of the trace. That
    pool ran six segments, every placement correctly chose the minimum,
    and no migration was possible at any run length.

    Two sites in the same zone contribute no switches between themselves,
    so this metric subsumes the diversity heuristic instead of discarding
    it: distinct zones are preferred because only they can cross over.
    """
    zones = tuple(zones)
    winners = [min(zones, key=lambda z: row[z])
               for row in series.values() if all(z in row for z in zones)]
    return sum(1 for a, b in zip(winners, winners[1:]) if a != b)


def survey(resources, candidates, specs, covered_zones):
    """Measure each candidate. Returns (eligible, excluded) where eligible is
    [{site, zone, gpu_model, gpu_available, cores, ram, disk}, ...] and
    excluded is [(site, reason), ...] — every exclusion carries its reason.

    `resources` needs get_component_available / get_core_available /
    get_ram_available / get_disk_available (fablib's ResourcesV2); a query
    failure excludes the site with the error named, because "could not
    measure" must never be scored as "available".
    """
    eligible, excluded = [], []
    for site, zone in sorted(candidates.items()):
        if zone not in covered_zones:
            excluded.append((site, f"zone {zone} not in the carbon trace — "
                                   f"would never win a match"))
            continue
        try:
            per_model = {
                m: int(resources.get_component_available(site, aggregate))
                for m, aggregate in GPU_MODELS.items()}
            cores = int(resources.get_core_available(site))
            ram = int(resources.get_ram_available(site))
            disk = int(resources.get_disk_available(site))
        except Exception as exc:
            excluded.append((site, f"availability query failed "
                                   f"({type(exc).__name__}: {exc})"))
            continue
        available = {m: n for m, n in per_model.items() if n > 0}
        if not available:
            excluded.append((site, "no GPU of any accepted model available"))
            continue
        gaps = [f"{name} {have}<{need}" for name, have, need in
                (("cores", cores, specs["cores"]),
                 ("ram", ram, specs["ram"]),
                 ("disk", disk, specs["disk"])) if have < need]
        if gaps:
            excluded.append((site, "too thin: " + ", ".join(gaps)))
            continue
        # Most-available model wins; preference order breaks ties, so a
        # site with one RTX6000 and one T4 free prefers the RTX6000.
        model = max(available,
                    key=lambda m: (available[m], -GPU_PREFERENCE.index(m)))
        eligible.append({"site": site, "zone": zone, "gpu_model": model,
                         "gpu_available": available[model], "cores": cores,
                         "ram": ram, "disk": disk})
    return eligible, excluded


def choose(eligible, count, series, log=print):
    """Pick the `count` sites whose grids ACTUALLY CROSS OVER most often.

    Exhaustive over combinations (six candidates choose three is twenty
    sets, scored over ~157 hourly rows — free), ranked by:

      1. cleanest-site switches in the trace — the only property that
         decides whether the pool can migrate at all;
      2. total free GPUs, as a tie-break, so a crossover-equivalent set
         with more headroom wins.

    Never pads: if fewer than `count` sites are eligible, the shorter list
    IS the answer. A best score of zero is reported loudly rather than
    passed off as a choice — it means no organic migration is possible
    with any subset of what is available, and the demo would need the
    dashboard's injection control instead.
    """
    def warn_if_unmigratable(switches):
        # Hoisted so BOTH paths below get it. An earlier version put this
        # only in the ranked branch, so a pool with exactly `count`
        # eligible sites — the common case, and the one that shipped the
        # unmigratable CISO pool — skipped the warning entirely and
        # reported a bare "0 switches" line nobody reads as fatal.
        if switches == 0:
            log("  WARNING: this site set NEVER crosses over in the trace "
                "— the pool cannot migrate organically at any run length. "
                "Every placement will correctly pick the same site. Use "
                "the dashboard's (disclosed) carbon injection to "
                "demonstrate migration, add candidate sites whose grids "
                "actually cross, or extend the trace to cover more "
                "diurnal variation.")

    if len(eligible) <= count:
        chosen = sorted(eligible, key=lambda e: e["site"])
        if chosen:
            switches = count_switches(series, [e["zone"] for e in chosen])
            log(f"  only {len(chosen)} eligible site(s), so no choice to "
                f"make: {switches} cleanest-site switch(es) in the trace")
            warn_if_unmigratable(switches)
        return chosen

    scored = []
    for combo in itertools.combinations(
            sorted(eligible, key=lambda e: e["site"]), count):
        switches = count_switches(series, [e["zone"] for e in combo])
        headroom = sum(e["gpu_available"] for e in combo)
        scored.append((switches, headroom, combo))
    scored.sort(key=lambda s: (-s[0], -s[1],
                               tuple(e["site"] for e in s[2])))

    log("  candidate sets by migration opportunity "
        "(cleanest-site switches in the trace):")
    for switches, headroom, combo in scored[:5]:
        names = "+".join(e["site"] for e in combo)
        log(f"    {names:22s} {switches:3d} switches, "
            f"{headroom} GPU(s) free")
    best_switches, _, best = scored[0]
    if best_switches == 0:
        log("  (no available set crosses over — the best score is zero)")
    warn_if_unmigratable(best_switches)
    return sorted(best, key=lambda e: e["site"])


def choose_worker_sites(resources, cfg, count, submit_site, trace_path,
                        log=print):
    """The full pipeline provision.py calls for `--sites auto`.

    Returns {site: gpu_model}. Exits (via sys.exit) when fewer than two
    sites are eligible: a one-site pool cannot migrate, and a demo that
    silently cannot do the thing it demonstrates is worse than no demo.
    """
    candidates = {site: meta["zone"] for site, meta in cfg["sites"].items()
                  if meta.get("zone") and site != submit_site}
    series = load_trace_series(trace_path)
    covered = trace_zones(series)
    specs = {"cores": 4, "ram": 16, "disk": 100}   # provision.py worker spec
    eligible, excluded = survey(resources, candidates, specs, covered)

    for site, reason in excluded:
        log(f"  {site}: EXCLUDED — {reason}")
    for entry in eligible:
        log(f"  {entry['site']}: eligible — {entry['gpu_model']} x"
            f"{entry['gpu_available']}, {entry['cores']}c/"
            f"{entry['ram']}G/{entry['disk']}G free, zone {entry['zone']}")

    chosen = choose(eligible, count, series, log=log)
    if len(chosen) < 2:
        sys.exit(f"only {len(chosen)} eligible site(s) — a carbon-migration "
                 f"pool needs at least 2 distinct grids. Fix the exclusions "
                 f"above (extend the trace, free capacity, or add candidate "
                 f"sites to config/sites.yaml).")
    if len(chosen) < count:
        log(f"  NOTE: asked for {count} sites, only {len(chosen)} eligible — "
            f"proceeding with {len(chosen)}, not padding")
    zones = {e["zone"] for e in chosen}
    if len(zones) < len(chosen):
        log(f"  NOTE: only {len(zones)} distinct zones across {len(chosen)} "
            f"sites — same-zone sites cannot migrate meaningfully between "
            f"each other")
    log(f"  migration opportunity: "
        f"{count_switches(series, [e['zone'] for e in chosen])} "
        f"cleanest-site switch(es) across the trace")
    log("  chosen: " + ", ".join(f"{e['site']}({e['gpu_model']})"
                                 for e in chosen))
    return {e["site"]: e["gpu_model"] for e in chosen}


def main():
    ap = argparse.ArgumentParser(description="Dry-run site selection: "
                                 "survey live availability and print the "
                                 "choice; provisions nothing.")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--submit-site", default="STAR")
    ap.add_argument("--config",
                    default=os.path.join(ROOT, "config", "sites.yaml"))
    ap.add_argument("--trace",
                    default=os.path.join(ROOT, "config", "traces", "eia.csv"))
    args = ap.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    if not os.path.exists(args.trace):
        sys.exit(f"missing {args.trace} — run fabric/fetch_traces.py first; "
                 f"zone coverage cannot be checked without the trace")

    from fabrictestbed_extensions.fablib.fablib import FablibManager
    resources = FablibManager().get_available_resources()
    print("Surveying live availability:")
    chosen = choose_worker_sites(resources, cfg, args.count,
                                 args.submit_site, args.trace)
    print("\nprovision with:\n  python pegasus/provision.py --sites "
          + ",".join(sorted(chosen)) + f" --submit-site {args.submit_site}"
          + "\nor let provision.py do this itself with --sites auto")


if __name__ == "__main__":
    main()
