"""Power-provider tests.

`CO2 = power x time x intensity`. Intensity is now real, so power is the last
synthetic term — and the same rule applies to it: a modelled watt must never
be presented as a measured one, and a stale sample must not be integrated as
if current.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carbon_chaser.power import (MeasuredPowerProvider,  # noqa: E402
                                 make_power_provider)


class FakeExec:
    def __init__(self, sample=None):
        self.sample = sample

    def get_power(self, site):
        return self.sample


def test_there_is_no_assumed_power_mode():
    """The whole point: no code path produces a power number that was not
    measured."""
    import carbon_chaser.power as mod
    assert not hasattr(mod, "AssumedPowerProvider")
    src = open(mod.__file__).read()
    assert "node_power_kw" not in src
    print("  no assumed-power provider exists")


def test_measured_provider_uses_nvml_watts():
    ex = FakeExec({"watts": 71.4, "ts": time.time(), "device": "Tesla T4"})
    p = MeasuredPowerProvider(ex)
    assert abs(p.get_kw("A") - 0.0714) < 1e-6, p.get_kw("A")
    d = p.describe()
    assert d["measured"] is True and "Tesla T4" in d["detail"]
    print("  NVML watts used and reported as measured")


def test_stale_power_sample_is_not_integrated():
    """Same rule as carbon: stale is not current."""
    ex = FakeExec({"watts": 300.0, "ts": time.time() - 999, "device": "T4"})
    p = MeasuredPowerProvider(ex, max_age_s=60)
    assert p.get_kw("A") is None, "used a stale power sample"
    d = p.describe()
    assert d["measured"] is False and d["kind"] == "unavailable"
    print("  stale NVML sample yields None, reported unavailable")


def test_missing_nvml_yields_no_number():
    p = MeasuredPowerProvider(FakeExec(None))
    assert p.get_kw("A") is None, "invented a power figure"
    d = p.describe()
    assert d["measured"] is False and d["kind"] == "unavailable"
    print("  absent NVML yields no number at all")


def test_factory_always_returns_the_measured_provider():
    for cfg in ({}, {"power_max_age_s": 30}, None):
        assert isinstance(make_power_provider(cfg, FakeExec(None)),
                          MeasuredPowerProvider)
    print("  factory always measured — no flag can turn it synthetic")


def test_engine_integrates_measured_power():
    """The headline number must actually move with measured watts."""
    from carbon_chaser.clock import SimClock
    from carbon_chaser.engine import Engine

    sites = {"A": {"display": "A", "lat": 0, "lon": 0, "zone": "z"}}

    class Prov:
        def get_intensity(self, zone):
            return 400.0          # gCO2/kWh

        def describe(self):
            return {"kind": "test", "detail": "", "injected_events": 0}

    class Ex:
        def get_progress(self, site):
            return {"step": 1, "loss": 0.1, "acc": 0.9}

        def checkpoint_bytes(self, site):
            return 1e6

        def is_running(self, site):
            return True

        def start(self, site):
            pass

    def emissions_after(kw_provider):
        eng = Engine(sites, "A", Prov(), Ex(), SimClock(accel=3600),
                     {"stall_after_s": 999},
                     net_cfg={}, power_provider=kw_provider)
        eng.status = "running"
        eng._tick()
        time.sleep(0.4)
        eng._tick()
        return eng.emissions_g

    class Fixed:
        def __init__(self, kw):
            self.kw = kw

        def get_kw(self, site):
            return self.kw

        def describe(self):
            return {"kind": "measured", "detail": "", "measured": True}

    low = emissions_after(Fixed(0.05))
    high = emissions_after(Fixed(0.50))
    assert high > low * 5, (low, high)
    print(f"  emissions scale with power: {low:.2f} g at 50W vs "
          f"{high:.2f} g at 500W")


def test_engine_reports_power_provenance():
    from carbon_chaser.clock import SimClock
    from carbon_chaser.engine import Engine
    sites = {"A": {"display": "A", "lat": 0, "lon": 0, "zone": "z"}}

    class Prov:
        def get_intensity(self, zone):
            return 400.0

        def describe(self):
            return {"kind": "test", "detail": "", "injected_events": 0}

    ex = FakeExec({"watts": 80.0, "ts": time.time(), "device": "RTX6000"})
    power = MeasuredPowerProvider(ex)
    eng = Engine(sites, "A", Prov(), ex, SimClock(), {}, net_cfg={},
                 power_provider=power)
    power.get_kw("A")                     # take a sample
    src = eng.state()["power_source"]
    assert src["measured"] is True and "RTX6000" in src["detail"]
    print("  state API reports power provenance")


if __name__ == "__main__":
    for fn in (test_there_is_no_assumed_power_mode,
               test_measured_provider_uses_nvml_watts,
               test_stale_power_sample_is_not_integrated,
               test_missing_nvml_yields_no_number,
               test_factory_always_returns_the_measured_provider,
               test_engine_integrates_measured_power,
               test_engine_reports_power_provenance):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
