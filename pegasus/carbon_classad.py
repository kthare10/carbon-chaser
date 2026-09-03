#!/usr/bin/env python3
"""Publish grid carbon intensity and measured GPU power as HTCondor ClassAds.

Run by `condor_startd` as a STARTD_CRON job (period 60s). It prints
`Attr = value` lines on stdout; HTCondor merges them into the slot's ClassAd.
That makes carbon a *matchmaking* input, so a Pegasus job can say

    RANK = -CarbonIntensity          # prefer the cleanest available site

and get carbon-aware placement on every submission, with no workflow
replanning and no custom site selector.

## The replay clock, and why it is shared

The trace is days of hourly data driving a minutes-long demo, so wall-clock is
mapped onto it at `CARBON_ACCEL`. Every site must be evaluated at the SAME
replayed instant, or the comparison between sites is meaningless — and that is
exactly what an earlier version got wrong, twice over:

* It wrapped with a PER-ZONE span (`position = start + elapsed % span`), and
  EIA returns different coverage per balancing authority — 156/157/158/159
  rows here, so spans of 155/156/157/158 h. Different periods put each zone at
  a different phase. Measured on the live pool: `CarbonTimestamp` differed by
  **7.9 and 15.9 replayed hours** between three sites. Each value was real EIA
  data; they were just readings from different times of day, and carbon
  intensity swings diurnally, so the ranking could invert on phase alone.
* It anchored to each node's own boot time (`/proc/stat` btime). That happened
  to be coherent because the nodes rebooted together, but any single-node
  reboot would silently desynchronise it by hours.

Both are fixed by deriving the position from quantities that are identical on
every node: the GLOBAL timeline across all zones in the (identical) trace
file, and absolute wall-clock time. No per-node state, nothing to stage, and
a reboot changes nothing. `CarbonTimestamp` is therefore the same on every
site, which makes coherence directly observable in `condor_status`.

Residual sensitivity, stated rather than hidden: clock skew between nodes is
amplified by `CARBON_ACCEL`, so 1s of skew is `ACCEL` seconds of replayed
time. With NTP (sub-100ms) that is well under the trace's hourly resolution.
`CarbonAccel` is published so a node configured with a different accel — which
would desynchronise it — is detectable rather than silent.

## Honesty rules carried over from the rest of this project

* Unknown is *absent*, never zero. A missing attribute is visibly unknown to
  a RANK expression, whereas `CarbonIntensity = 0` would read as the
  cleanest possible grid and attract every job.
* A position outside this zone's own coverage publishes NOTHING. Zones end at
  different times, so the shared clock can land past a given zone's last
  sample; the previous code clamped to the final value, which republishes an
  old reading as if it were current. (An earlier docstring also claimed a
  `MAX_AGE_S` staleness rule that the code never implemented — removed rather
  than left as a false claim. Coverage is the real mechanism.)
* Nothing is invented: no simulator, no assumed power. If the trace or NVML
  is unavailable, the attribute simply does not appear.
"""

import argparse
import csv
import datetime
import os
import sys
import time

TRACE = os.environ.get("CARBON_TRACE", "/opt/carbon/eia.csv")
ZONE_FILE = os.environ.get("CARBON_ZONE_FILE", "/opt/carbon/zone")
# Demo injection hook (dashboard "inject" button): a file holding
# "<gCO2/kWh> <expiry-epoch>". While present and unexpired it REPLACES this
# site's CarbonIntensity — and is DISCLOSED as CarbonInjected = true, so
# the dashboard shows an amber badge, the job ads record that the match was
# made against an injected value, and a demo can never quietly contaminate
# a measured run. Expiry is mandatory: an injection that never lapses is a
# lie that outlives the person who told it. Power is deliberately NOT
# injectable — GPUWatts and the in-job energy integral are the measured
# half of the story, and matchmaking only needs carbon to move.
OVERRIDE_FILE = os.environ.get("CARBON_OVERRIDE_FILE", "/opt/carbon/override")
# Replay speed. MUST be the same on every node: the position is derived from
# absolute wall-clock, so a differing accel puts a node at a different phase.
ACCEL = float(os.environ.get("CARBON_ACCEL", "300"))


def read_zone():
    try:
        with open(ZONE_FILE) as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def load_trace(path=None):
    """({zone: [(epoch_seconds, gco2_per_kwh)]}, global_start, global_end).

    Loads EVERY zone, not just this node's, because the replay timeline has to
    be a property of the trace as a whole. Deriving it from one zone's rows is
    what made the sites incomparable: zones have different coverage, so each
    node computed a different period and drifted into a different phase.

    The trace file is identical on every node (staged by provision.py), so the
    returned bounds are identical everywhere.
    """
    by_zone = {}
    try:
        with open(path or TRACE, newline="") as handle:
            for row in csv.DictReader(handle):
                zone = (row.get("zone") or "").strip()
                stamp = (row.get("timestamp") or "").strip()
                value = (row.get("carbon_intensity_gco2_kwh") or "").strip()
                if not zone or not stamp or not value:
                    continue
                try:
                    when = datetime.datetime.fromisoformat(
                        stamp.replace("Z", "+00:00"))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=datetime.timezone.utc)
                    by_zone.setdefault(zone, []).append(
                        (when.timestamp(), float(value)))
                except ValueError:
                    continue
    except OSError:
        return {}, None, None
    if not by_zone:
        return {}, None, None
    for series in by_zone.values():
        series.sort()
    global_start = min(series[0][0] for series in by_zone.values())
    global_end = max(series[-1][0] for series in by_zone.values())
    return by_zone, global_start, global_end


def replay_span(global_start, global_end):
    """The wrap period. Published, because two nodes computing DIFFERENT spans
    is the exact signature of the per-zone bug this replaced."""
    return max(1.0, global_end - global_start)


def replay_position(global_start, global_end, now=None):
    """The single replayed instant at which EVERY site is evaluated.

    Depends only on the global trace bounds and absolute wall-clock, both of
    which are the same on every node — so all sites are compared at the same
    moment of the trace. Deliberately NOT time-since-boot: that made the
    anchor per-node, and a single reboot would have shifted one site by hours
    while every value still looked plausible.
    """
    span = replay_span(global_start, global_end)
    stamp = time.time() if now is None else now
    return global_start + ((stamp * ACCEL) % span)


def intensity_at(series, position):
    """This zone's intensity at `position`, or None if it does not cover it.

    Returning None (rather than clamping to the first or last sample) is the
    point: zones end at different times, so the shared clock can land beyond a
    zone's last row, and clamping would publish a stale reading as current.
    Absent means unknown, and a job requiring `CarbonIntensity =!= UNDEFINED`
    then simply skips that site.
    """
    if not series:
        return None
    first, last = series[0][0], series[-1][0]
    if position < first or position > last:
        return None

    lo, hi = 0, len(series) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if series[mid][0] <= position:
            lo = mid
        else:
            hi = mid
    t0, v0 = series[lo]
    t1, v1 = series[hi]
    if position <= t0:
        return v0
    frac = 0.0 if t1 == t0 else (position - t0) / (t1 - t0)
    return v0 + (v1 - v0) * frac


def read_override(now):
    """(value, expiry) from the injection file, or None.

    Malformed or expired content reads as no override — a stale injection
    must lapse back to measured data by itself, without an operator
    remembering to clean up.
    """
    try:
        with open(OVERRIDE_FILE) as handle:
            parts = handle.read().split()
        value, expiry = float(parts[0]), float(parts[1])
    except (OSError, ValueError, IndexError):
        return None
    if now >= expiry or value < 0:
        return None
    return value, expiry


def gpu_watts():
    """Measured NVML board power, or None. Never a guess."""
    try:
        import pynvml
        pynvml.nvmlInit()
        total = 0.0
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            total += pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        return round(total, 1)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Print carbon/power ClassAd attributes for STARTD_CRON.")
    ap.add_argument("--at", type=float, default=None,
                    help="evaluate at this wall-clock epoch instead of now — "
                         "for inspecting what the pool would see at a given "
                         "moment, and for tests that need two zones compared "
                         "at exactly the same instant")
    args = ap.parse_args()

    zone = read_zone()
    if zone:
        print(f'CarbonZone = "{zone}"')
        by_zone, global_start, global_end = load_trace()
        if global_start is not None:
            sampled_at = time.time() if args.at is None else args.at
            position = replay_position(global_start, global_end, sampled_at)
            value = intensity_at(by_zone.get(zone), position)
            # Published even when the value is not, so a node that is reading a
            # different phase can be SPOTTED rather than quietly skewing the
            # ranking. All three are needed to check coherence per node:
            #   CarbonAccel    - the replay speed
            #   CarbonSpan     - the wrap period (differing spans WAS the bug)
            #   CarbonSampledAt- the wall-clock instant this node used
            # With those, a checker can verify each node's position against its
            # OWN sample time, so neither the 60s cron phase nor the wrap
            # boundary looks like a fault. A cross-node spread cannot: at the
            # wrap it reads as a full span of divergence, and out-of-phase
            # crons read as CRON_PERIOD x ACCEL.
            print(f"CarbonAccel = {ACCEL:.0f}")
            print(f"CarbonSpan = {replay_span(global_start, global_end):.0f}")
            print(f"CarbonSampledAt = {sampled_at:.0f}")
            override = read_override(sampled_at)
            if override is not None:
                # Injected value REPLACES the trace reading, loudly: the
                # disclosure attributes are published in the same breath,
                # so no consumer can see the number without the caveat.
                injected, expiry = override
                print(f"CarbonIntensity = {injected:.1f}")
                print("CarbonInjected = true")
                print(f"CarbonInjectedUntil = {int(expiry)}")
                if value is not None:
                    print(f"CarbonTraceIntensity = {value:.1f}")
                print(f"CarbonTimestamp = {int(position)}")
                print(f"CarbonSamples = {len(by_zone.get(zone) or [])}")
            elif value is not None:
                print(f"CarbonIntensity = {value:.1f}")
                print(f"CarbonTimestamp = {int(position)}")
                print(f"CarbonSamples = {len(by_zone.get(zone) or [])}")
            # else: attributes absent — unknown must not look like 0, and a
            # position past this zone's coverage must not be clamped.

    watts = gpu_watts()
    if watts is not None:
        print(f"GPUWatts = {watts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
