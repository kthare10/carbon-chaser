"""Carbon-intensity providers — measured sources only.

There is deliberately no simulator and no spike injection. Every earlier
class of bug in this system came from synthetic values being mistaken for
measurements; the surest fix is for no synthetic path to exist. If no real
source is configured, the run refuses to start rather than inventing
numbers.

- TraceProvider: replays measured history (from fetch_traces.py) at the
  clock's speed, so real data is usable at demo pace.
- ElectricityMapsProvider: live gCO2eq/kWh (needs EMAPS_TOKEN).
"""

import math
import os
import random
import threading
import time
from typing import Dict, List, Optional

import requests

from .clock import Clock




class CarbonProvider:
    def get_intensity(self, zone: str) -> float:
        """Current carbon intensity for a zone, in gCO2eq/kWh."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Provenance, so a run can state what its numbers actually are.

        A carbon figure is only as good as its inputs, and "simulated",
        "replayed from measured history", and "live" are three very
        different claims. The dashboard and the state API surface this
        verbatim rather than letting a plausible-looking number imply
        measurement.
        """
        return {"kind": "unknown", "detail": ""}


class ElectricityMapsProvider(CarbonProvider):
    """Live data; caches per zone for `ttl` seconds to respect rate limits."""

    URL = "https://api.electricitymap.org/v3/carbon-intensity/latest"

    def __init__(self, token: Optional[str] = None, ttl: float = 300.0):
        self.token = token or os.environ["EMAPS_TOKEN"]
        self.ttl = ttl
        self._cache: Dict[str, tuple] = {}  # zone -> (value, fetched_at)
        self._lock = threading.Lock()

    def describe(self) -> dict:
        return {"kind": "live",
                "detail": f"Electricity Maps live feed ({len(self._cache)} "
                          f"zones cached, ttl {int(self.ttl)}s)"}

    def get_intensity(self, zone: str) -> float:
        with self._lock:
            hit = self._cache.get(zone)
            if hit and time.time() - hit[1] < self.ttl:
                return hit[0]
        resp = requests.get(
            self.URL, params={"zone": zone},
            headers={"auth-token": self.token}, timeout=10,
        )
        resp.raise_for_status()
        value = float(resp.json()["carbonIntensity"])
        with self._lock:
            self._cache[zone] = (value, time.time())
        return value


class TraceProvider(CarbonProvider):
    """Replay *measured* carbon-intensity history at the sim clock's speed.

    This is the honest way to get both real numbers and demo-speed dynamics.
    A live feed moves hourly, so at 1x a two-hour booth would show almost no
    migrations; the simulator moves fast but invents the data. Replaying a
    real trace at 300x keeps every value measured while compressing a day
    into minutes — and the same machinery replays a year of history for
    offline policy evaluation.

    CSV schema (one row per zone per hour):
        timestamp,zone,carbon_intensity_gco2_kwh
        2026-08-01T00:00:00Z,US-CAL-CISO,243.0

    Values between hourly samples are linearly interpolated, which is a
    display choice on top of measured endpoints, not extra measurement.
    """

    def __init__(self, clock: Clock, path: str, loop: bool = True):
        self.clock = clock
        self.path = path
        self.loop = loop
        # A trace is only as trustworthy as its origin, and the replay code
        # cannot tell measured data from something someone generated. So the
        # claim lives with the file: fetch_traces.py writes a sidecar, and a
        # trace without one is reported as unverified rather than assumed
        # real.
        self.meta = self._load_meta(path)
        self._series: Dict[str, List[tuple]] = {}   # zone -> [(epoch_s, value)]
        self._load()

    @staticmethod
    def _load_meta(path: str) -> dict:
        import json
        for candidate in (path + ".meta.json",
                          os.path.splitext(path)[0] + ".meta.json"):
            try:
                with open(candidate) as handle:
                    return json.load(handle)
            except (OSError, ValueError):
                continue
        return {}

    def _load(self):
        import csv
        from datetime import datetime, timezone
        with open(self.path, newline="") as handle:
            for row in csv.DictReader(handle):
                zone = (row.get("zone") or "").strip()
                raw_ts = (row.get("timestamp") or "").strip()
                raw_val = (row.get("carbon_intensity_gco2_kwh") or "").strip()
                if not zone or not raw_ts or not raw_val:
                    continue
                try:
                    when = datetime.fromisoformat(
                        raw_ts.replace("Z", "+00:00"))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    value = float(raw_val)
                except ValueError:
                    continue
                self._series.setdefault(zone, []).append(
                    (when.timestamp(), value))
        if not self._series:
            raise ValueError(f"no usable rows in {self.path}")
        for zone in self._series:
            self._series[zone].sort()
        starts = [s[0][0] for s in self._series.values()]
        ends = [s[-1][0] for s in self._series.values()]
        self.trace_start = min(starts)
        self.trace_end = max(ends)
        self.span_s = max(1.0, self.trace_end - self.trace_start)

    def _trace_time(self) -> float:
        """Map elapsed sim time onto the trace's own timeline."""
        elapsed = self.clock.now()
        if self.loop:
            elapsed = elapsed % self.span_s
        else:
            elapsed = min(elapsed, self.span_s)
        return self.trace_start + elapsed

    def get_intensity(self, zone: str) -> float:
        series = self._series.get(zone)
        if not series:
            raise KeyError(f"zone {zone} not present in trace {self.path}")
        now = self._trace_time()

        # Binary search for the bracketing samples.
        lo, hi = 0, len(series) - 1
        if now <= series[0][0]:
            value = series[0][1]
        elif now >= series[-1][0]:
            value = series[-1][1]
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if series[mid][0] <= now:
                    lo = mid
                else:
                    hi = mid
            t0, v0 = series[lo]
            t1, v1 = series[hi]
            frac = 0.0 if t1 == t0 else (now - t0) / (t1 - t0)
            value = v0 + (v1 - v0) * frac

        return max(0.0, value)

    def describe(self) -> dict:
        from datetime import datetime, timezone
        fmt = lambda t: datetime.fromtimestamp(  # noqa: E731
            t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        source = self.meta.get("source")
        measured = bool(self.meta.get("measured"))
        if source:
            origin = f"{source}"
            if not measured:
                origin += " (SYNTHETIC — not measured)"
        else:
            origin = "provenance UNVERIFIED (no .meta.json sidecar)"

        note = (f"{origin}, replayed at "
                f"{getattr(self.clock, 'accel', 1.0):g}x — "
                f"{os.path.basename(self.path)}, "
                f"{fmt(self.trace_start)} to {fmt(self.trace_end)}, "
                f"{len(self._series)} zones")
        if self.meta.get("note"):
            note += f"; {self.meta['note']}"
        kind = "trace-replay" if measured else "trace-replay-unverified"
        return {"kind": kind, "detail": note}


def trace_zones(path: str) -> set:
    """Zones present in a trace file (cheap scan, no parsing of values)."""
    import csv
    zones = set()
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                zone = (row.get("zone") or "").strip()
                if zone:
                    zones.add(zone)
    except OSError:
        pass
    return zones


def plan_carbon_source(carbon_cfg: Optional[dict],
                       required_zones: Optional[set] = None) -> dict:
    """Decide the carbon source BEFORE the clock exists.

    Two reasons this is separate from building the provider:

    * The clock depends on the source. A live feed must run at 1x (real
      grids move hourly); a trace should run accelerated. Choosing the clock
      from the presence of EMAPS_TOKEN — while the provider prefers a
      configured trace — replays measured history at 1x and silently
      discards --accel.
    * A trace that does not cover every configured zone must be rejected
      here, not discovered per-tick. get_intensity() raises for an unknown
      zone, and the engine builds all intensities in one comprehension, so a
      single missing zone blanks every reading: the demo shows zeros and no
      history while still reporting itself healthy.

    Returns {kind, path, loop, reason} where kind is trace|live|simulated
    and reason explains any downgrade.
    """
    cfg = carbon_cfg or {}
    trace = cfg.get("trace_file")
    reason = None

    if trace:
        path = trace if os.path.isabs(trace) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), trace)
        if not os.path.exists(path):
            reason = f"trace_file {trace} not found"
        else:
            have = trace_zones(path)
            missing = sorted((required_zones or set()) - have)
            if missing:
                reason = (f"trace {os.path.basename(path)} is missing "
                          f"{len(missing)} configured zone(s): "
                          f"{', '.join(missing)}")
            else:
                return {"kind": "trace", "path": path,
                        "loop": cfg.get("trace_loop", True), "reason": None}

    if os.environ.get("EMAPS_TOKEN"):
        return {"kind": "live", "path": None, "loop": False, "reason": reason}
    # No measured source: there is nothing honest to fall back to.
    return {"kind": "none", "path": None, "loop": False,
            "reason": reason or "no carbon source configured: set "
                                "carbon.trace_file (see fabric/"
                                "fetch_traces.py) or EMAPS_TOKEN"}


class NoCarbonSource(RuntimeError):
    """Raised instead of substituting invented data."""


def build_provider(clock: Clock, plan: dict) -> CarbonProvider:
    """Instantiate the provider chosen by plan_carbon_source().

    Raises when no measured source is available. Refusing to run is the
    honest behaviour: a carbon-aware scheduler with invented carbon data is
    not a carbon-aware scheduler.
    """
    if plan["kind"] == "trace":
        return TraceProvider(clock, plan["path"], loop=plan.get("loop", True))
    if plan["kind"] == "live":
        return ElectricityMapsProvider()
    raise NoCarbonSource(plan.get("reason") or "no carbon source available")


def make_provider(clock: Clock, carbon_cfg: Optional[dict] = None,
                  required_zones: Optional[set] = None) -> CarbonProvider:
    """Convenience wrapper for callers that already have a clock."""
    return build_provider(clock, plan_carbon_source(carbon_cfg, required_zones))
