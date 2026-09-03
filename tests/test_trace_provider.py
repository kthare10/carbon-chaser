"""Trace-replay tests.

Replay is what makes measured carbon data usable at demo speed, so the
mapping from sim time onto the trace timeline has to be right — and the
provenance it reports has to stay honest about what is measured.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carbon_chaser.carbon import (NoCarbonSource,  # noqa: E402
                                  TraceProvider, build_provider,
                                  make_provider, plan_carbon_source)
from carbon_chaser.clock import SimClock  # noqa: E402

ZONE_A, ZONE_B = "US-CAL-CISO", "US-NW-PACE"


def write_trace(path, hours=24):
    """Two zones, hourly, with a deliberate shape we can assert on:
    zone A swings 100..500, zone B is flat at 600."""
    rows = ["timestamp,zone,carbon_intensity_gco2_kwh"]
    for h in range(hours):
        a = 100 + 400 * (h / max(1, hours - 1))     # ramps 100 -> 500
        rows.append(f"2026-08-01T{h:02d}:00:00Z,{ZONE_A},{a:.1f}")
        rows.append(f"2026-08-01T{h:02d}:00:00Z,{ZONE_B},600.0")
    with open(path, "w") as handle:
        handle.write("\n".join(rows) + "\n")
    return path


class FrozenClock:
    """Sim clock we can step by hand."""
    accel = 3600.0

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def hour_of_day(self):
        return (self.t / 3600.0) % 24


def test_replays_measured_values_exactly_on_sample_points(tmp):
    path = write_trace(os.path.join(tmp, "t.csv"))
    clock = FrozenClock()
    p = TraceProvider(clock, path)

    clock.t = 0                                   # first sample
    assert abs(p.get_intensity(ZONE_A) - 100.0) < 0.01
    clock.t = 3600 * 5                            # sixth hourly sample
    expected = 100 + 400 * (5 / 23)
    assert abs(p.get_intensity(ZONE_A) - expected) < 0.05, p.get_intensity(ZONE_A)
    assert abs(p.get_intensity(ZONE_B) - 600.0) < 0.01
    print("  measured sample points reproduced exactly")


def test_interpolates_between_hourly_samples(tmp):
    path = write_trace(os.path.join(tmp, "t.csv"))
    clock = FrozenClock()
    p = TraceProvider(clock, path)
    clock.t = 3600 * 0.5                          # halfway to the next hour
    lo, hi = 100.0, 100 + 400 * (1 / 23)
    got = p.get_intensity(ZONE_A)
    assert lo < got < hi, f"{got} not between {lo} and {hi}"
    assert abs(got - (lo + hi) / 2) < 0.05
    print("  linear interpolation between measured endpoints")


def test_loops_so_a_booth_run_never_runs_dry(tmp):
    path = write_trace(os.path.join(tmp, "t.csv"), hours=6)
    clock = FrozenClock()
    p = TraceProvider(clock, path, loop=True)
    clock.t = 0
    first = p.get_intensity(ZONE_A)
    clock.t = p.span_s                            # exactly one span later
    assert abs(p.get_intensity(ZONE_A) - first) < 0.01, "did not wrap"
    print("  trace loops cleanly")


def test_no_loop_holds_the_final_value(tmp):
    path = write_trace(os.path.join(tmp, "t.csv"), hours=6)
    clock = FrozenClock()
    p = TraceProvider(clock, path, loop=False)
    clock.t = p.span_s * 10                       # far past the end
    assert abs(p.get_intensity(ZONE_A) - 500.0) < 0.01
    print("  non-looping trace clamps to the last measured value")


def test_trace_without_sidecar_is_reported_unverified(tmp):
    """The replay code cannot tell measured data from something someone
    generated, so an unlabelled trace must not be presented as measured."""
    path = write_trace(os.path.join(tmp, "bare.csv"))
    d = TraceProvider(FrozenClock(), path).describe()
    assert d["kind"] == "trace-replay-unverified", d["kind"]
    assert "UNVERIFIED" in d["detail"], d["detail"]
    print("  unlabelled trace -> reported as unverified")


def test_synthetic_sidecar_is_labelled_synthetic(tmp):
    import json
    path = write_trace(os.path.join(tmp, "fake.csv"))
    with open(path + ".meta.json", "w") as handle:
        json.dump({"source": "generated for testing", "measured": False}, handle)
    d = TraceProvider(FrozenClock(), path).describe()
    assert d["kind"] == "trace-replay-unverified"
    assert "SYNTHETIC" in d["detail"], d["detail"]
    print("  synthetic trace labelled SYNTHETIC, not measured")


def test_measured_sidecar_is_trusted_and_described(tmp):
    import json
    path = write_trace(os.path.join(tmp, "real.csv"))
    with open(path + ".meta.json", "w") as handle:
        json.dump({"source": "EIA hourly fuel mix", "measured": True,
                   "note": "derived from measured generation mix"}, handle)
    p = TraceProvider(FrozenClock(), path)
    d = p.describe()
    assert d["kind"] == "trace-replay", d["kind"]
    assert "EIA" in d["detail"] and "2026-08-01" in d["detail"]
    assert "derived from measured" in d["detail"]
    print("  measured trace described with source and span")


def test_make_provider_uses_a_trace_and_refuses_without_one(tmp):
    """There is no simulator to fall back to: refusing beats inventing."""
    path = write_trace(os.path.join(tmp, "t.csv"))
    p = make_provider(SimClock(), {"trace_file": path})
    assert isinstance(p, TraceProvider), type(p)

    raised = False
    try:
        make_provider(SimClock(), {"trace_file": os.path.join(tmp, "no.csv")})
    except NoCarbonSource as e:
        raised = True
        assert "trace_file" in str(e) or "no carbon source" in str(e), e
    assert raised, "invented a carbon source instead of refusing"
    print("  trace used when present; refuses outright when absent")


def test_malformed_rows_are_skipped_not_fatal(tmp):
    path = os.path.join(tmp, "messy.csv")
    with open(path, "w") as handle:
        handle.write("timestamp,zone,carbon_intensity_gco2_kwh\n")
        handle.write("not-a-date,US-CAL-CISO,100\n")       # bad timestamp
        handle.write("2026-08-01T00:00:00Z,US-CAL-CISO,\n")  # empty value
        handle.write("2026-08-01T01:00:00Z,US-CAL-CISO,abc\n")  # bad value
        handle.write("2026-08-01T02:00:00Z,US-CAL-CISO,222.5\n")  # good
    p = TraceProvider(FrozenClock(), path)
    assert abs(p.get_intensity("US-CAL-CISO") - 222.5) < 0.01
    print("  malformed rows skipped, good rows kept")


# --- source selection & zone coverage -----------------------------------

ALL_ZONES = {ZONE_A, ZONE_B}


def test_partial_trace_is_rejected_before_the_run(tmp):
    """A trace missing a configured zone must be refused up front. Left to
    run, get_intensity raises for that zone and (previously) blanked EVERY
    reading, so the demo showed zeros and no history while reporting itself
    healthy."""
    path = os.path.join(tmp, "partial.csv")
    with open(path, "w") as handle:
        handle.write("timestamp,zone,carbon_intensity_gco2_kwh\n")
        handle.write(f"2026-08-01T00:00:00Z,{ZONE_A},200\n")
        handle.write(f"2026-08-01T01:00:00Z,{ZONE_A},210\n")
    plan = plan_carbon_source({"trace_file": path}, ALL_ZONES)
    assert plan["kind"] != "trace", "accepted a trace missing a zone"
    assert ZONE_B in plan["reason"], plan["reason"]
    # and with no other real source, that means refusing to run
    raised = False
    try:
        build_provider(SimClock(), plan)
    except NoCarbonSource:
        raised = True
    assert raised, "fell back to something synthetic"
    print("  partial trace rejected by name, and the run refuses")


def test_complete_trace_is_accepted(tmp):
    path = write_trace(os.path.join(tmp, "full.csv"))
    plan = plan_carbon_source({"trace_file": path}, ALL_ZONES)
    assert plan["kind"] == "trace" and plan["reason"] is None
    print("  complete trace accepted")


def test_trace_beats_live_token_so_accel_is_honoured(tmp):
    """The clock is chosen from the plan. If a token flipped the plan to
    'live' while the provider still used the trace, measured history would
    replay at 1x and --accel would be silently discarded."""
    path = write_trace(os.path.join(tmp, "full.csv"))
    os.environ["EMAPS_TOKEN"] = "dummy"
    try:
        plan = plan_carbon_source({"trace_file": path}, ALL_ZONES)
    finally:
        del os.environ["EMAPS_TOKEN"]
    assert plan["kind"] == "trace", plan
    print("  trace wins over a live token, so the accelerated clock is used")


def test_live_selected_only_without_a_usable_trace(tmp):
    os.environ["EMAPS_TOKEN"] = "dummy"
    try:
        plan = plan_carbon_source({"trace_file": os.path.join(tmp, "no.csv")},
                                  ALL_ZONES)
    finally:
        del os.environ["EMAPS_TOKEN"]
    assert plan["kind"] == "live", plan
    print("  live feed used when no usable trace exists")


# --- carbon-feed outage handling ----------------------------------------
#
# Data quality is a different question from whether the job is running.
# These pin that separation: a dead feed must not be laundered into a
# healthy-looking run, and it must not keep the CO2 counter moving.

class _DyingProvider:
    def __init__(self):
        self.alive = True

    def get_intensity(self, zone):
        if not self.alive:
            raise RuntimeError("feed unreachable")
        return 300.0

    def describe(self):
        return {"kind": "live", "detail": "stub", "injected_events": 0}


class _StubPower:
    """Measured watts, always fresh — power is a separate concern from the
    carbon-feed behaviour these tests exercise."""

    def get_kw(self, site):
        return 0.15

    def describe(self):
        return {"kind": "measured", "detail": "stub", "measured": True}


def _outage_engine(tmp, max_stale=3):
    import yaml
    from carbon_chaser.engine import Engine
    from carbon_chaser.executor import LocalExecutor
    cfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), "..", "config", "sites.yaml")))
    prov = _DyingProvider()
    clock = SimClock(accel=600)
    policy = dict(cfg["policy"], stall_after_s=999,
                  carbon_max_stale_s=max_stale, min_dwell_min=0)
    eng = Engine(cfg["sites"], cfg["start_site"], prov,
                 LocalExecutor(run_root=tmp), clock, policy,
                 managed_sites=list(cfg["sites"]), net_cfg={},
                 power_provider=_StubPower())
    return eng, prov


def test_total_outage_is_reported_not_hidden(tmp):
    """Advancing training steps prove the JOB is alive and say nothing about
    the carbon feed. Sharing one health field let progress clear an outage."""
    eng, prov = _outage_engine(tmp)
    eng.status = "running"
    eng._tick()
    assert eng.carbon_status == "ok"
    prov.alive = False
    time.sleep(3.2)                      # exceed the staleness bound
    eng._tick()
    eng._tick()

    assert eng.carbon_status == "unavailable", eng.carbon_status
    assert "unavailable" in (eng.carbon_note or "")
    # a liveness check must not wipe a data-quality verdict
    eng._check_liveness({"step": 999})
    assert eng.carbon_status == "unavailable", "liveness cleared the outage"
    print("  outage reported and survives a liveness check")


def test_emissions_do_not_accrue_on_stale_data(tmp):
    """Integrating a stale intensity invents CO2 that was never measured."""
    eng, prov = _outage_engine(tmp)
    eng.status = "running"
    for _ in range(3):
        eng._tick()
        time.sleep(0.2)
    grew = eng.emissions_g
    assert grew > 0, "should accrue while the feed is healthy"

    prov.alive = False
    time.sleep(3.2)
    for _ in range(4):
        eng._tick()
        time.sleep(0.2)
    assert eng.emissions_g == grew, (
        f"emissions moved on stale data: {grew} -> {eng.emissions_g}")

    prov.alive = True
    eng._tick()
    time.sleep(0.3)
    eng._tick()
    assert eng.emissions_g > grew, "did not resume once data returned"
    print("  counter freezes on stale data and resumes on recovery")


def test_no_migration_while_blind(tmp):
    eng, prov = _outage_engine(tmp)
    eng.status = "running"
    eng._tick()
    prov.alive = False
    time.sleep(3.2)
    for _ in range(3):
        eng._tick()
    assert eng.migrations == [], f"migrated on stale data: {eng.migrations}"
    print("  no migration decided while the feed is down")


def test_stale_values_stop_being_presented_as_readings(tmp):
    """Within the bound a last-known value is still shown (labelled stale);
    beyond it the reading is withdrawn rather than displayed indefinitely as
    if current."""
    eng, prov = _outage_engine(tmp, max_stale=3)
    eng.status = "running"
    eng._tick()
    prov.alive = False

    time.sleep(1)
    eng._tick()
    shown = [k for k, v in eng.state()["sites"].items()
             if v["intensity"] is not None]
    assert len(shown) == len(eng.sites), f"dropped too early: {shown}"

    time.sleep(3)
    eng._tick()
    shown = [k for k, v in eng.state()["sites"].items()
             if v["intensity"] is not None]
    assert shown == [], f"still presenting stale readings: {shown}"
    print("  stale readings withdrawn once past the age bound")


def test_tick_survives_a_missing_active_reading(tmp):
    """The tick must keep the dashboard alive even with no usable reading —
    integrating inside the lock before the history append means a raise here
    blanks everything."""
    eng, prov = _outage_engine(tmp, max_stale=1)
    eng.status = "running"
    eng._tick()
    prov.alive = False
    time.sleep(1.5)
    before = len(eng.state()["history"])
    eng._tick()
    eng._tick()
    after = len(eng.state()["history"])
    assert after > before, "history stopped advancing during the outage"
    print("  history keeps advancing through a total outage")


def test_no_emissions_during_the_stale_but_shown_window(tmp):
    """The subtle window: the feed has died but the last value is still
    within the age bound, so it is still displayed. Displaying it is fine;
    COUNTING it is not — that is CO2 that was never measured. This is what
    the freshness gate (rather than mere presence) protects."""
    eng, prov = _outage_engine(tmp, max_stale=60)   # generous bound
    eng.status = "running"
    eng._tick()
    time.sleep(0.3)
    eng._tick()
    accrued = eng.emissions_g
    assert accrued > 0

    prov.alive = False
    for _ in range(4):                 # well inside the 60s bound
        time.sleep(0.3)
        eng._tick()

    shown = [k for k, v in eng.state()["sites"].items()
             if v["intensity"] is not None]
    assert shown, "precondition: values should still be displayed here"
    assert eng.emissions_g == accrued, (
        f"counted stale-but-shown data: {accrued} -> {eng.emissions_g}")
    assert eng.carbon_status in ("stale", "unavailable"), eng.carbon_status
    print("  stale-but-displayed values are shown, never counted")


def test_missing_reading_is_null_not_zero(tmp):
    """Zero is a value: reporting a missing reading as 0 painted the site as
    the CLEANEST possible grid (lightest colour, "0 g"), which is the most
    flattering possible lie about data we do not have."""
    eng, prov = _outage_engine(tmp, max_stale=1)
    eng.status = "running"
    eng._tick()
    fresh = eng.state()["sites"]["CLEM"]
    assert fresh["intensity"] is not None and fresh["data"] == "fresh"

    prov.alive = False
    time.sleep(1.5)
    eng._tick()
    for name, site in eng.state()["sites"].items():
        assert site["intensity"] is None, (
            f"{name} reported {site['intensity']} for missing data")
        assert site["data"] == "missing", (name, site["data"])
    print("  missing readings reported as null with data='missing'")


def test_stale_reading_is_labelled_but_still_shown(tmp):
    eng, prov = _outage_engine(tmp, max_stale=60)
    eng.status = "running"
    eng._tick()
    prov.alive = False
    eng._tick()
    sites = eng.state()["sites"]
    assert all(s["intensity"] is not None for s in sites.values())
    assert all(s["data"] == "stale" for s in sites.values()), (
        {k: v["data"] for k, v in sites.items()})
    print("  stale readings shown but labelled data='stale'")


def test_history_gaps_are_absent_not_zero(tmp):
    """The chart breaks its line across a gap; a zero would draw a spike to
    the floor as though the grid had gone carbon-free."""
    eng, prov = _outage_engine(tmp, max_stale=1)
    eng.status = "running"
    eng._tick()
    prov.alive = False
    time.sleep(1.5)
    eng._tick()
    last = eng.state()["history"][-1]["intensities"]
    assert all(v != 0 for v in last.values()), last
    assert "CLEM" not in last, "missing site should be absent, not zero"
    print("  history omits missing samples rather than zeroing them")


def test_history_marks_which_samples_were_stale(tmp):
    """A stale value still has a number, so 'has a value' cannot be the test
    for 'was measured'. Without a per-point record the chart draws a stale
    plateau as a solid measured line."""
    eng, prov = _outage_engine(tmp, max_stale=600)   # keep values, mark stale
    eng.status = "running"
    eng._tick()
    first = eng.state()["history"][-1]
    assert first["stale"] == [], first["stale"]

    prov.alive = False
    eng._tick()
    point = eng.state()["history"][-1]
    assert sorted(point["stale"]) == sorted(eng.sites), point["stale"]
    # the values are still present — displayable, but flagged
    assert point["intensities"], "values should still be carried for display"
    print("  history flags stale samples while still carrying their values")


def test_site_state_distinguishes_all_three_conditions(tmp):
    """fresh / stale / missing must be separable by a consumer; collapsing
    stale into fresh is what made stale look measured."""
    eng, prov = _outage_engine(tmp, max_stale=600)
    eng.status = "running"
    eng._tick()
    assert {s["data"] for s in eng.state()["sites"].values()} == {"fresh"}

    prov.alive = False
    eng._tick()
    states = {k: v for k, v in eng.state()["sites"].items()}
    assert all(v["data"] == "stale" for v in states.values())
    assert all(v["intensity"] is not None for v in states.values()), (
        "stale readings should still be available for display")

    eng._intensity_seen = {k: 0.0 for k in eng.sites}   # force past any bound
    eng._tick()
    after = eng.state()["sites"]
    assert all(v["data"] == "missing" and v["intensity"] is None
               for v in after.values()), {k: v["data"] for k, v in after.items()}
    print("  fresh, stale and missing are all distinguishable")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="cc-trace-")
    for fn in (test_replays_measured_values_exactly_on_sample_points,
               test_interpolates_between_hourly_samples,
               test_loops_so_a_booth_run_never_runs_dry,
               test_no_loop_holds_the_final_value,
               test_trace_without_sidecar_is_reported_unverified,
               test_synthetic_sidecar_is_labelled_synthetic,
               test_measured_sidecar_is_trusted_and_described,
               test_make_provider_uses_a_trace_and_refuses_without_one,
               test_malformed_rows_are_skipped_not_fatal,
               test_partial_trace_is_rejected_before_the_run,
               test_complete_trace_is_accepted,
               test_trace_beats_live_token_so_accel_is_honoured,
               test_live_selected_only_without_a_usable_trace,
               test_total_outage_is_reported_not_hidden,
               test_emissions_do_not_accrue_on_stale_data,
               test_no_migration_while_blind,
               test_stale_values_stop_being_presented_as_readings,
               test_tick_survives_a_missing_active_reading,
               test_no_emissions_during_the_stale_but_shown_window,
               test_missing_reading_is_null_not_zero,
               test_stale_reading_is_labelled_but_still_shown,
               test_history_gaps_are_absent_not_zero,
               test_history_marks_which_samples_were_stale,
               test_site_state_distinguishes_all_three_conditions):
        print(fn.__name__)
        fn(tmp)
    print("\nALL PASS")
