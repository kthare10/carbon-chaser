#!/usr/bin/env python3
"""Ask the live pool whether the workflow's requirements can match ANYTHING.

The failure this exists to prevent is the quietest one in HTCondor. The pool
advertised:

    HasGPU = ifThenElse(DetectedGPUs > 0, true, false)

but `DetectedGPUs` is a *string* of GPU ids ("GPU-1cae1040"), not a count, so
the comparison evaluated to **error** — and an error in a requirements
expression means NEVER MATCH. The job was submitted, accepted, and sat Idle
indefinitely. Nothing was held, nothing failed, nothing was logged; the DAG
simply never progressed. Only `condor_q -better-analyze` revealed
`[0] 0 HasGPU == true`.

So before submitting, ask the pool directly: how many slots satisfy this
expression? Zero is a hard failure, and the answer arrives in seconds instead
of after a confused wait.

This uses the SAME expression constants the generator emits (imported from
workflow_generator.py by AST, since the Pegasus API is only installed on the submit
node), so the check cannot drift from what is actually submitted.

Usage:
    python pegasus/check_pool_match.py                  # via fablib, remote
    python pegasus/check_pool_match.py --local          # on the submit node
"""

import argparse
import ast
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Attributes whose evaluation we report per-slot, because "error" and
# "undefined" here are exactly what silently breaks matchmaking.
DIAGNOSTIC_ATTRS = ("Name", "GPUs", "HasGPU", "CarbonIntensity",
                    "CarbonTimestamp", "CarbonAccel", "GPUWatts")

# Coherence is checked PER NODE against that node's own sample instant, not
# as a spread across nodes. A spread has two false-positive modes that both
# reject a perfectly healthy pool, and both were reproduced:
#   * the wrap boundary — two nodes either side of it differ by a whole span
#     (157.9h on the real trace), once every span/ACCEL = ~31.6 min;
#   * cron phase — STARTD_CRON samples each node every 60s, unsynchronised,
#     so healthy nodes routinely differ by 60s x ACCEL = 5h of replayed time.
# Raising the tolerance past those would exceed the 7.9h divergence the check
# exists to catch, so the comparison itself had to change.
#
# What remains is genuine clock skew between nodes, amplified by ACCEL.
CLOCK_SKEW_ALLOWANCE_S = 5.0            # wall-clock seconds between nodes


def policy_from_workflow():
    """Read the requirement/rank constants out of workflow_generator.py without importing.

    workflow_generator.py imports the Pegasus API at module scope, which only exists on
    the submit node — but these are plain string literals, so AST is enough.
    """
    tree = ast.parse(open(os.path.join(HERE, "workflow_generator.py")).read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "TRAIN_REQUIREMENTS", "TRAIN_RANK", "EVAL_REQUIREMENTS"):
                    found[target.id] = node.value.value
    missing = {"TRAIN_REQUIREMENTS", "TRAIN_RANK", "EVAL_REQUIREMENTS"} - set(found)
    if missing:
        raise SystemExit(f"workflow_generator.py no longer defines {sorted(missing)} — "
                         f"this check would be testing a stale policy.")
    return found


def make_runner(args):
    """A function that runs a shell command on the submit node."""
    if args.local:
        def run_local(cmd):
            proc = subprocess.run(cmd, shell=True, text=True,
                                  capture_output=True, timeout=300)
            return (proc.stdout or "") + (proc.stderr or "")
        return run_local

    sys.path.insert(0, os.path.join(HERE, ".."))
    from carbon_chaser.executor import call_with_timeout
    from fabrictestbed_extensions.fablib.fablib import FablibManager
    node = (FablibManager().get_slice(name=args.slice_name)
            .get_node(name=args.submit_node))

    def run_remote(cmd):
        out, err = call_with_timeout(
            lambda: node.execute(cmd, quiet=True, retry=1), 600, "check")
        return (out or "") + (err or "")
    return run_remote


def count_matching(run, expression):
    """How many slots satisfy `expression`, per condor_status itself."""
    escaped = expression.replace("'", "'\\''")
    out = run(f"condor_status -constraint '{escaped}' -af Name 2>&1")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # A malformed expression makes condor_status complain rather than list.
    bad = [ln for ln in lines
           if "error" in ln.lower() or "can't" in ln.lower()
           or "unable" in ln.lower()]
    if bad:
        return None, bad
    return len(lines), lines



def circular_delta(a, b, span):
    """Distance between two positions on a wrapping timeline.

    `abs(a - b)` is wrong here: the replay wraps, so a position just after the
    wrap and one just before it are ADJACENT in replayed time while being a
    whole span apart numerically.
    """
    gap = (a - b) % span
    return min(gap, span - gap)


def check_replay_coherence(run):
    """True if the sites are NOT reading the same moment of the carbon trace.

    Comparing intensities across sites is only meaningful if every site
    evaluates the same function of absolute time. An earlier publisher wrapped
    the replay with a PER-ZONE period, and because EIA returns different
    coverage per balancing authority the sites drifted into different phases —
    measured at 7.9 and 15.9 replayed HOURS apart. Every value was real; they
    were readings from different times of day, so `RANK = -CarbonIntensity`
    ranked phase as much as carbon.

    Checked per node, against that node's own `CarbonSampledAt`:

    * every node must agree on `CarbonSpan` — differing spans IS the per-zone
      bug, structurally;
    * every node must agree on `CarbonAccel`, or they advance at different
      rates and will diverge regardless of what they read now;
    * each node's position must be what its own sample time implies, to within
      clock skew, compared on a circle.

    That is immune to the two things a cross-node spread mistakes for a fault:
    out-of-phase 60s crons, and the wrap boundary.
    """
    # Start from the nodes that actually PARTICIPATE IN RANKING, not from the
    # nodes that happen to be checkable. Three false approvals came from
    # getting that backwards, all reproduced:
    #   * a pool running the OLD publisher emits CarbonIntensity but no
    #     span/sampled_at, so nothing was checkable and the gate returned
    #     "no failure" -- approving the exact state it exists to detect;
    #   * a node that ranks but is unverifiable was silently skipped, and the
    #     remaining nodes were declared coherent;
    #   * a single node was reported as "intensities are comparable" with
    #     nothing to compare it to.
    # An unverifiable node is not a node that passes. It is a node whose clock
    # nothing here can vouch for, while it competes for jobs on carbon.
    ranking = set()
    for line in (run("condor_status -af Name -constraint "
                     "'CarbonIntensity =!= UNDEFINED' 2>&1") or "").splitlines():
        name = line.strip()
        if name and " " not in name and "rror" not in name:
            ranking.add(name)

    out = run("condor_status -af Name CarbonAccel CarbonSpan CarbonSampledAt "
              "CarbonTimestamp -constraint 'CarbonTimestamp =!= UNDEFINED' 2>&1")
    rows = []
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) != 5 or any(p in ("undefined", "error") for p in parts[1:]):
            continue
        try:
            rows.append((parts[0], float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    verifiable = {name for name, _, _, _, _ in rows}

    # Clock offsets, measured rather than inferred. HTCondor stamps every ad
    # with the startd's own clock (`MyCurrentTime`) and, on arrival, the
    # collector's (`LastHeardFrom`) — their difference is the node's offset,
    # independent of how old the ad is. Queried separately so the row parsing
    # above keeps its shape; a pool that publishes neither simply yields no
    # entry here and falls back to the weaker future-sample test below.
    offsets = {}
    for line in (run("condor_status -af Name MyCurrentTime LastHeardFrom "
                     "-constraint 'CarbonTimestamp =!= UNDEFINED' 2>&1")
                 or "").splitlines():
        parts = line.split()
        if len(parts) != 3 or any(p in ("undefined", "error")
                                  for p in parts[1:]):
            continue
        try:
            offsets[parts[0]] = float(parts[1]) - float(parts[2])
        except ValueError:
            continue

    if not ranking:
        print("  no site publishes CarbonIntensity, so carbon ranks nothing "
              "(the matchmaking check below is what will catch that)")
        return False

    failed = False
    unverifiable = sorted(ranking - verifiable)
    if unverifiable:
        print(f"  UNVERIFIABLE SITES {unverifiable}: these publish "
              f"CarbonIntensity — so they compete for jobs on carbon — but "
              f"not the CarbonSpan/CarbonSampledAt needed to check they are "
              f"reading the same moment of the trace. They may be running an "
              f"older carbon_classad.py. Coherence here is UNKNOWN, which is "
              f"not the same as fine: redeploy carbon_classad.py to them "
              f"(pegasus/provision.py does this) and restart condor.")
        failed = True

    if not rows:
        return failed

    for name, accel, span, sampled, pos in sorted(rows):
        print(f"  {name:22s} accel={accel:>5.0f} span={span:>8.0f} "
              f"sampled_at={sampled:.0f} position={pos:.0f}")

    accels = {accel for _, accel, _, _, _ in rows}
    spans = {span for _, _, span, _, _ in rows}
    if len(spans) > 1:
        print(f"  MISMATCHED CarbonSpan {sorted(spans)}: the nodes are "
              f"wrapping the trace over different periods, so they drift into "
              f"different phases — this is exactly the per-zone replay bug. "
              f"Redeploy carbon_classad.py everywhere.")
        failed = True
    if len(accels) > 1:
        print(f"  MISMATCHED CarbonAccel {sorted(accels)}: nodes advance "
              f"through the trace at different speeds, so their intensities "
              f"are not comparable even if they agree right now.")
        failed = True

    if not failed:
        span, accel = spans.pop(), accels.pop()
        allowance = accel * CLOCK_SKEW_ALLOWANCE_S
        ref_name, _, _, ref_sampled, ref_pos = rows[0]
        worst = 0.0
        for name, _, _, sampled, pos in rows[1:]:
            expected = (sampled - ref_sampled) * accel
            error = circular_delta(pos - ref_pos, expected, span)
            worst = max(worst, error)
            if error > allowance:
                print(f"  {name} is out of step with {ref_name} by "
                      f"{error:.0f}s of replayed time after accounting for "
                      f"their {sampled - ref_sampled:.0f}s sampling gap "
                      f"(allowance {allowance:.0f}s). Check the node clocks: "
                      f"skew is amplified {accel:.0f}x.")
                failed = True
        if not failed:
            # The comparison above verifies each node's replay FORMULA, and it
            # is blind to clock skew: carbon_classad derives the sample time
            # AND the position from the same time.time(), so a node whose
            # clock is off is internally consistent and scores a ZERO error
            # while sitting offset x ACCEL of replayed time from its peers
            # (600s of offset at ACCEL=300 is 50 replayed hours) — in EITHER
            # direction.
            #
            # `offsets` measures that directly, from the two clocks in the
            # ad, so a SLOW clock fails here exactly like a fast one. An
            # earlier version inferred skew from ad age, which cannot: this
            # pool's healthy ads run 208-268s old, so a lagging clock and
            # slow updates are the same observation. Age is not evidence of
            # a clock fault; the offset is.
            for name, offset in sorted(offsets.items(),
                                       key=lambda kv: -abs(kv[1])):
                if abs(offset) <= CLOCK_SKEW_ALLOWANCE_S:
                    continue
                direction = "AHEAD OF" if offset > 0 else "BEHIND"
                print(f"  {name} CLOCK IS {abs(offset):.0f}s {direction} the "
                      f"collector's — it advertises a replay position "
                      f"{abs(offset) * accel / 3600.0:.1f} replayed hours "
                      f"from its peers while looking perfectly "
                      f"self-consistent, because its position and its sample "
                      f"time come from that same clock. Fix time sync on "
                      f"that node before trusting the ranking.")
                failed = True

            # A node whose offset cannot be MEASURED is unverifiable, and
            # unverifiable is not fine — the same rule this check already
            # applies to a site publishing CarbonIntensity without
            # CarbonSpan. There is deliberately no fallback to "is the
            # sample time in our future?": that only ever sees a clock
            # running AHEAD, so it would pass a slow clock silently, which
            # is precisely the hole being closed here.
            now = time.time()
            unmeasured = sorted(name for name, _, _, _, _ in rows
                                if name not in offsets)
            if unmeasured:
                print(f"  UNVERIFIABLE CLOCKS {unmeasured}: these rank on "
                      f"carbon but publish no MyCurrentTime/LastHeardFrom "
                      f"pair, so their clock offset cannot be measured — and "
                      f"a skewed clock is invisible to every other check "
                      f"here, since a node's position and its sample time "
                      f"come from that same clock. Check "
                      f"`condor_status -l <node> | grep -E "
                      f"'MyCurrentTime|LastHeardFrom'`.")
                failed = True

        if not failed:
            if len(rows) < 2:
                # Vacuously true is not the same as verified.
                print(f"  only one site ({rows[0][0]}) publishes a checkable "
                      f"position, so there is nothing to compare — coherence "
                      f"is trivially satisfied, not demonstrated")
            else:
                print(f"  coherent: all {len(rows)} ranking sites match their "
                      f"own sample times to within {worst:.0f}s of replayed "
                      f"time (allowance {allowance:.0f}s), one span, one "
                      f"accel — intensities are comparable")

            # Internal consistency is not freshness. Each node can be
            # perfectly consistent with its OWN sample time while the
            # collector serves an ad minutes old — and the negotiator ranks
            # on the collector's copy, not on the node's current reading.
            # Measured on a live pool: ads 208-268s old, which at
            # ACCEL=300 is 17-22 REPLAYED HOURS. Two segments were then
            # placed on a site whose advertised value exceeded another
            # site's trace-wide maximum: the rank was faithfully applied
            # to stale inputs. The clock check runs FIRST, because ages are
            # only meaningful once every sample time is on one clock: each
            # node's is in its own, so the measured offset is what converts
            # it to the collector's. Without that correction a healthy node
            # with a slow clock is misreported as serving a stale ad.
            ages = [(name, now - (sampled - offsets.get(name, 0.0)))
                    for name, _, _, sampled, _ in rows]
            worst_age = max(age for _, age in ages)
            skew = worst_age - min(age for _, age in ages)
            replayed_h = worst_age * accel / 3600.0
            print(f"\n  Ad freshness (the negotiator ranks the COLLECTOR's "
                  f"copy, not the node's):")
            for name, age in sorted(ages, key=lambda a: -a[1]):
                print(f"    {name:22s} {age:5.0f}s old = "
                      f"{age * accel / 3600.0:5.1f} replayed h")
            # One trace row is an hour, so an ad older than 3600/accel
            # seconds is ranking on a value the trace has already moved past.
            budget = 3600.0 / accel if accel else float("inf")
            if worst_age > budget:
                print(f"    WARNING: worst ad is {worst_age:.0f}s old but at "
                      f"accel={accel:.0f} a trace ROW lasts only "
                      f"{budget:.0f}s — placement ranks values the trace has "
                      f"moved past by {replayed_h:.1f} replayed hours, and "
                      f"sites are compared {skew:.0f}s apart "
                      f"({skew * accel / 3600.0:.1f} replayed h). Lower "
                      f"CARBON_ACCEL to <= {3600.0 / max(worst_age, 1):.0f} "
                      f"or shorten UPDATE_INTERVAL/STARTD_CRON_CARBON_PERIOD. "
                      f"Placement stays HONEST (it ranks what is advertised) "
                      f"but 'cleanest site' becomes 'cleanest recently "
                      f"advertised'.")
            else:
                print(f"    ads are fresher than one trace row "
                      f"({budget:.0f}s at accel={accel:.0f}) — 'cleanest "
                      f"advertised' and 'cleanest' coincide")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-name", default="carbon-chaser-pegasus")
    ap.add_argument("--submit-node", default="submit")
    ap.add_argument("--local", action="store_true",
                    help="run condor commands here (i.e. on the submit node)")
    ap.add_argument("--expression",
                    help="check an arbitrary expression instead of the "
                         "workflow's (useful for proving this check can fail)")
    args = ap.parse_args()

    run = make_runner(args)

    if args.expression:
        checks = [("--expression", args.expression)]
    else:
        policy = policy_from_workflow()
        checks = [("train jobs", policy["TRAIN_REQUIREMENTS"]),
                  ("evaluate job", policy["EVAL_REQUIREMENTS"])]

    print("Slot attributes as the negotiator sees them:")
    print(run("condor_status -af " + " ".join(DIAGNOSTIC_ATTRS)
              + " 2>&1").rstrip() or "  (no slots)")
    print("\nReplay coherence (are the sites even comparable?):")
    coherence_failed = check_replay_coherence(run)

    print("\nMatchmaking:")

    failed = False
    for label, expression in checks:
        count, detail = count_matching(run, expression)
        if count is None:
            print(f"  {label}: MALFORMED EXPRESSION -> {detail[:2]}")
            failed = True
            continue
        print(f"  {label}: {count} slot(s) match  [{expression}]")
        if count == 0:
            failed = True
        else:
            for name in detail:
                print(f"      {name}")

    if coherence_failed:
        print("\nCARBON VALUES ARE NOT COMPARABLE across the sites that will "
              "be ranked (see above). Submitting now would still run — the "
              "jobs would just be placed by a comparison that does not mean "
              "what it says, which is worse than an outright failure.")
        return 1
    if failed:
        print("\nNO SLOTS MATCH. Submitting now would leave the job Idle "
              "forever with nothing logged — HTCondor does not report an "
              "unsatisfiable requirement as an error.\n"
              "Check for attributes evaluating to 'error' or 'undefined' "
              "above; `condor_q -better-analyze <id>` names the failing "
              "clause once a job is queued.")
        return 1
    print("\nRequirements are satisfiable — safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
