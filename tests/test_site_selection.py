"""Dynamic site selection: measured availability in, honest choices out.

select_sites.py decides WHICH candidate sites get VMs from live testbed
facts. The failure modes worth guarding are all silent ones: a site whose
zone the carbon trace does not cover would provision fine and then never
win a match; a site with no free GPU would fail at slice submission after
minutes of waiting; and padding a thin result to the requested count would
hide that the pool cannot actually demonstrate migration.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pegasus"))

from select_sites import (GPU_PREFERENCE, choose,  # noqa: E402
                          count_switches, survey)

SPECS = {"cores": 4, "ram": 16, "disk": 100}


class FakeResources:
    """Stands in for fablib's ResourcesV2 with scripted availability."""

    def __init__(self, table, broken=()):
        self.table = table          # site -> dict(gpus={model: n}, cores...)
        self.broken = set(broken)   # sites whose queries raise

    def _row(self, site):
        if site in self.broken:
            raise RuntimeError("aggregate unreachable")
        return self.table[site]

    def get_component_available(self, site, model):
        return self._row(site)["gpus"].get(model, 0)

    def get_core_available(self, site):
        return self._row(site)["cores"]

    def get_ram_available(self, site):
        return self._row(site)["ram"]

    def get_disk_available(self, site):
        return self._row(site)["disk"]


def rich(gpus, cores=32, ram=256, disk=2000):
    return {"gpus": gpus, "cores": cores, "ram": ram, "disk": disk}


def test_exclusions_carry_reasons():
    resources = FakeResources({
        "CLEM": rich({"GPU-RTX6000": 2}),
        "TACC": rich({}),                                  # no GPU free
        "UTAH": rich({"GPU-Tesla T4": 1}, cores=2),         # too thin
        "UCSD": rich({"GPU-Tesla T4": 3}),                  # zone not traced
    }, broken=("NEWY",))
    candidates = {"CLEM": "Z-A", "TACC": "Z-B", "UTAH": "Z-C",
                  "UCSD": "Z-D", "NEWY": "Z-E"}
    covered = {"Z-A", "Z-B", "Z-C", "Z-E"}
    eligible, excluded = survey(resources, candidates, SPECS, covered)
    assert [e["site"] for e in eligible] == ["CLEM"]
    reasons = dict(excluded)
    assert "no GPU" in reasons["TACC"]
    assert "too thin" in reasons["UTAH"] and "cores" in reasons["UTAH"]
    assert "not in the carbon trace" in reasons["UCSD"]
    assert "query failed" in reasons["NEWY"], (
        "an unmeasurable site must be excluded WITH the error, never "
        "scored as available")
    print("  every exclusion carries its reason")


# A synthetic trace: DOMINANT is always cleanest; CROSSA and CROSSB trade
# places every hour; TWIN tracks DOMINANT exactly (same grid shape).
SERIES = {}
for _hour in range(24):
    SERIES[f"t{_hour:02d}"] = {
        "Z-DOMINANT": 100.0,
        "Z-TWIN": 100.0,
        "Z-CROSSA": 200.0 + (50.0 if _hour % 2 else -50.0),
        "Z-CROSSB": 200.0 - (50.0 if _hour % 2 else -50.0),
    }


def test_switch_counting_matches_the_definition():
    assert count_switches(SERIES, ["Z-CROSSA", "Z-CROSSB"]) == 23
    assert count_switches(SERIES, ["Z-DOMINANT", "Z-CROSSA"]) == 0
    assert count_switches(SERIES, ["Z-DOMINANT", "Z-TWIN"]) == 0, (
        "identical grids never cross, so they offer no migration")
    print("  switch counting: 23 for alternating grids, 0 for a dominated "
          "or duplicated one")


def test_crossover_beats_raw_availability():
    """The set that can actually migrate wins, even when a dominated site
    has far more GPUs free — this is the CISO/ERCO/PACE lesson: a
    perfectly diverse, perfectly provisioned pool that cannot migrate is
    the wrong answer."""
    resources = FakeResources({
        "DOM": rich({"GPU-Tesla T4": 9}),      # always cleanest, lots free
        "CRA": rich({"GPU-Tesla T4": 1}),      # scarce, but crosses over
        "CRB": rich({"GPU-Tesla T4": 1}),
    })
    candidates = {"DOM": "Z-DOMINANT", "CRA": "Z-CROSSA", "CRB": "Z-CROSSB"}
    eligible, _ = survey(resources, candidates, SPECS, set(SERIES["t00"]))
    chosen = choose(eligible, 2, SERIES, log=lambda *a: None)
    assert {e["site"] for e in chosen} == {"CRA", "CRB"}, chosen
    print("  the crossing pair wins over the better-provisioned dominant "
          "site")


def test_headroom_breaks_crossover_ties():
    resources = FakeResources({
        "CRA": rich({"GPU-Tesla T4": 1}),
        "CRB": rich({"GPU-Tesla T4": 1}),
        "CRB2": rich({"GPU-Tesla T4": 7}),     # same zone as CRB, more free
    })
    candidates = {"CRA": "Z-CROSSA", "CRB": "Z-CROSSB", "CRB2": "Z-CROSSB"}
    eligible, _ = survey(resources, candidates, SPECS, set(SERIES["t00"]))
    chosen = choose(eligible, 2, SERIES, log=lambda *a: None)
    assert {e["site"] for e in chosen} == {"CRA", "CRB2"}, chosen
    print("  equal crossovers -> the set with more free GPUs wins")


def test_zero_crossover_warns_on_both_code_paths():
    """A pool that cannot migrate must SAY so — that failure shipped once
    and cost a full run to diagnose.

    Both paths must warn, and the assertion demands the WARNING itself:
    an earlier version only warned in the ranked branch, and a weaker
    assertion here (accepting any line containing "switch") passed on the
    bare factual count that nobody reads as fatal. The
    eligible-count == requested-count case is exactly how the
    unmigratable pool shipped.
    """
    dominated = {"DOM": "Z-DOMINANT", "TWIN": "Z-TWIN"}

    # Path 1: len(eligible) <= count — no choice to make.
    resources = FakeResources({"DOM": rich({"GPU-Tesla T4": 4}),
                               "TWIN": rich({"GPU-Tesla T4": 4})})
    eligible, _ = survey(resources, dominated, SPECS, set(SERIES["t00"]))
    lines = []
    choose(eligible, 2, SERIES, log=lines.append)
    assert any("WARNING" in line and "cannot migrate organically" in line
               for line in lines), ("no-choice path did not warn", lines)

    # Path 2: ranked branch — more eligible sites than requested, but no
    # combination crosses over.
    resources = FakeResources({"DOM": rich({"GPU-Tesla T4": 4}),
                               "TWIN": rich({"GPU-Tesla T4": 4}),
                               "TWIN2": rich({"GPU-Tesla T4": 4})})
    eligible, _ = survey(resources,
                         {**dominated, "TWIN2": "Z-TWIN"}, SPECS,
                         set(SERIES["t00"]))
    lines = []
    choose(eligible, 2, SERIES, log=lines.append)
    assert any("WARNING" in line and "cannot migrate organically" in line
               for line in lines), ("ranked path did not warn", lines)

    # And a set that DOES cross must not cry wolf.
    resources = FakeResources({"CRA": rich({"GPU-Tesla T4": 1}),
                               "CRB": rich({"GPU-Tesla T4": 1})})
    eligible, _ = survey(resources,
                         {"CRA": "Z-CROSSA", "CRB": "Z-CROSSB"}, SPECS,
                         set(SERIES["t00"]))
    lines = []
    choose(eligible, 2, SERIES, log=lines.append)
    assert not any("WARNING" in line for line in lines), lines
    print("  zero-crossover warns on BOTH paths; a crossing set does not")


def test_thin_results_are_not_padded():
    resources = FakeResources({"A1": rich({"GPU-Tesla T4": 1})})
    eligible, _ = survey(resources, {"A1": "Z-DOMINANT"}, SPECS,
                         set(SERIES["t00"]))
    chosen = choose(eligible, 3, SERIES, log=lambda *a: None)
    assert len(chosen) == 1, "choose() must never invent sites"
    print("  a thin result stays thin — no padding")


def test_model_choice_prefers_free_capacity_then_preference_order():
    resources = FakeResources({
        "A1": rich({"GPU-Tesla T4": 5, "GPU-RTX6000": 1}),
        "B1": rich({"GPU-Tesla T4": 1, "GPU-RTX6000": 1}),
    })
    candidates = {"A1": "Z-A", "B1": "Z-B"}
    eligible, _ = survey(resources, candidates, SPECS, {"Z-A", "Z-B"})
    models = {e["site"]: e["gpu_model"] for e in eligible}
    assert models["A1"] == "GPU_TeslaT4"      # 5 free beats 1 free
    assert models["B1"] == "GPU_RTX6000", (
        "on a tie, the preference order decides — "
        f"order is {GPU_PREFERENCE}")
    print("  GPU model picked from free capacity, preference breaks ties")


if __name__ == "__main__":
    for fn in (test_exclusions_carry_reasons,
               test_switch_counting_matches_the_definition,
               test_crossover_beats_raw_availability,
               test_headroom_breaks_crossover_ties,
               test_zero_crossover_warns_on_both_code_paths,
               test_thin_results_are_not_padded,
               test_model_choice_prefers_free_capacity_then_preference_order):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
