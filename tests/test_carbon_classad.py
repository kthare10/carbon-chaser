"""Tests for the HTCondor carbon ClassAd publisher.

This is the carbon-aware matchmaking input, so the same honesty rules apply
as everywhere else — and here they have teeth: `CarbonIntensity = 0` would
not merely mislabel a reading, it would make an unmeasured worker rank as the
cleanest site and attract every job in the pool.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "pegasus", "carbon_classad.py")


def write_trace(path, zone="US-CAL-CISO", rows=6):
    with open(path, "w") as handle:
        handle.write("timestamp,zone,carbon_intensity_gco2_kwh\n")
        for hour in range(rows):
            handle.write(f"2026-08-01T{hour:02d}:00:00Z,{zone},{200 + hour}\n")
    return path


def write_multizone_trace(path, coverage):
    """A trace whose zones have DIFFERENT coverage, as EIA really returns.

    `coverage` maps zone -> number of hourly rows. The real trace had
    156/157/158/159 rows across six balancing authorities, and that difference
    is precisely what desynchronised the per-zone replay clock.
    """
    with open(path, "w") as handle:
        handle.write("timestamp,zone,carbon_intensity_gco2_kwh\n")
        for zone, rows in coverage.items():
            for hour in range(rows):
                stamp = (f"2026-08-{1 + hour // 24:02d}"
                         f"T{hour % 24:02d}:00:00Z")
                handle.write(f"{stamp},{zone},{200 + hour}\n")
    return path


def run(env, args=()):
    full = dict(os.environ)
    full.update(env)
    out = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True,
                         text=True, env=full, timeout=120)
    ads = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            ads[key.strip()] = value.strip()
    return ads, out


def test_publishes_intensity_for_a_known_zone(tmp):
    trace = write_trace(os.path.join(tmp, "eia.csv"))
    zone_file = os.path.join(tmp, "zone")
    open(zone_file, "w").write("US-CAL-CISO\n")
    ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file})
    assert ads.get("CarbonZone") == '"US-CAL-CISO"', ads
    assert "CarbonIntensity" in ads, ads
    value = float(ads["CarbonIntensity"])
    assert 195 <= value <= 210, value
    print(f"  publishes CarbonIntensity = {value}")


def test_unknown_zone_publishes_no_intensity(tmp):
    """Absent, not zero. An undefined attribute keeps a worker out of a
    RANK comparison; 0 would make it the most attractive slot in the pool."""
    trace = write_trace(os.path.join(tmp, "eia.csv"))
    zone_file = os.path.join(tmp, "zone")
    open(zone_file, "w").write("US-NOT-A-ZONE\n")
    ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file})
    assert "CarbonIntensity" not in ads, ads
    assert "0" not in ads.get("CarbonIntensity", ""), ads
    print("  unknown zone -> attribute absent, never 0")


def test_missing_trace_publishes_no_intensity(tmp):
    zone_file = os.path.join(tmp, "zone")
    open(zone_file, "w").write("US-CAL-CISO\n")
    ads, out = run({"CARBON_TRACE": os.path.join(tmp, "nope.csv"),
                    "CARBON_ZONE_FILE": zone_file})
    assert "CarbonIntensity" not in ads, ads
    assert out.returncode == 0, "must not crash the startd cron"
    print("  missing trace -> no attribute, clean exit")


def test_missing_zone_file_is_survivable(tmp):
    trace = write_trace(os.path.join(tmp, "eia.csv"))
    ads, out = run({"CARBON_TRACE": trace,
                    "CARBON_ZONE_FILE": os.path.join(tmp, "no-zone")})
    assert out.returncode == 0
    assert "CarbonIntensity" not in ads
    print("  missing zone file -> no attribute, clean exit")


def test_no_nvml_publishes_no_watts(tmp):
    """Power must never be invented for matchmaking either."""
    trace = write_trace(os.path.join(tmp, "eia.csv"))
    zone_file = os.path.join(tmp, "zone")
    open(zone_file, "w").write("US-CAL-CISO\n")
    ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file})
    # pynvml is absent in this venv, so GPUWatts must simply not appear
    assert "GPUWatts" not in ads, ads
    print("  no NVML -> GPUWatts absent (not 0)")


def test_output_is_valid_classad_syntax(tmp):
    """STARTD_CRON parses `Attr = value`; malformed output poisons the slot ad."""
    trace = write_trace(os.path.join(tmp, "eia.csv"))
    zone_file = os.path.join(tmp, "zone")
    open(zone_file, "w").write("US-CAL-CISO\n")
    _, out = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file})
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        assert " = " in line, f"not a ClassAd assignment: {line!r}"
        key = line.split(" = ")[0]
        assert key.isidentifier(), f"bad attribute name: {key!r}"
    print("  every line is a valid ClassAd assignment")


def test_all_zones_are_evaluated_at_the_same_replayed_instant(tmp):
    """The bug this fix exists for.

    The old code wrapped with a PER-ZONE span, and zones have different
    coverage, so each site sat at a different phase of the trace. Measured on
    the live pool: CarbonTimestamp differed by 7.9 and 15.9 replayed HOURS
    between three sites, which makes "site X is cleaner than site Y"
    meaningless — the values were real, but from different times of day.

    CarbonTimestamp is the published replay position, so equality across zones
    IS the coherence property.
    """
    trace = write_multizone_trace(os.path.join(tmp, "multi.csv"),
                                  {"ZONE-A": 40, "ZONE-B": 33, "ZONE-C": 27})
    stamps = {}
    for zone in ("ZONE-A", "ZONE-B", "ZONE-C"):
        zone_file = os.path.join(tmp, f"zone-{zone}")
        open(zone_file, "w").write(zone + "\n")
        # --at must be LARGE enough that the replay wraps. Caught by
        # mutation testing: with `--at 36000` (10h, below every zone's span)
        # the modulo never wrapped, so all three positions coincided by
        # accident and this test passed with the per-zone bug still live.
        # 1e6 s exceeds the shortest span (26h), so a per-zone period gives
        # three different remainders -- 17200 / 78400 / 64000 -- while one
        # global period gives a single value.
        ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file,
                      "CARBON_ACCEL": "1"}, args=("--at", "1000000"))
        assert "CarbonTimestamp" in ads, (zone, ads)
        assert "CarbonIntensity" in ads, (
            f"{zone} published no value, so this test would pass vacuously "
            f"on absence rather than on agreement: {ads}")
        stamps[zone] = int(ads["CarbonTimestamp"])
    assert len(set(stamps.values())) == 1, (
        f"sites are reading different moments of the trace: {stamps} — "
        f"their intensities are not comparable")
    print(f"  all zones evaluated at the same instant ({stamps['ZONE-A']})")


def test_position_outside_a_zones_coverage_publishes_nothing(tmp):
    """Zones end at different times; clamping would republish a stale value.

    With the shared clock, the position can land past a short zone's last
    sample. The old code returned `series[-1][1]` — an old reading presented
    as current. Absent is the honest answer.
    """
    trace = write_multizone_trace(os.path.join(tmp, "cover.csv"),
                                  {"LONG": 12, "SHORT": 6})
    # ACCEL=1 and --at 28800 puts the shared position 8h in: inside LONG
    # (0..11h), past the end of SHORT (0..5h).
    long_file = os.path.join(tmp, "zone-long")
    open(long_file, "w").write("LONG\n")
    ads_long, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": long_file,
                       "CARBON_ACCEL": "1"}, args=("--at", "28800"))
    short_file = os.path.join(tmp, "zone-short")
    open(short_file, "w").write("SHORT\n")
    ads_short, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": short_file,
                        "CARBON_ACCEL": "1"}, args=("--at", "28800"))

    assert "CarbonIntensity" in ads_long, ads_long
    assert "CarbonIntensity" not in ads_short, (
        f"SHORT does not cover this instant but published anyway: {ads_short}")
    # And it must not have been clamped to its last sample (205).
    assert ads_short.get("CarbonIntensity") != "205.0", ads_short
    print("  a zone that does not cover the instant publishes nothing")


def test_replay_clock_advances(tmp):
    """A shared clock still has to move, or every segment sees one snapshot."""
    trace = write_multizone_trace(os.path.join(tmp, "adv.csv"),
                                  {"ZONE-A": 40})
    zone_file = os.path.join(tmp, "zone-a")
    open(zone_file, "w").write("ZONE-A\n")
    seen = []
    for at in ("3600", "18000"):
        ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file,
                      "CARBON_ACCEL": "1"}, args=("--at", at))
        seen.append((int(ads["CarbonTimestamp"]), float(ads["CarbonIntensity"])))
    assert seen[0] != seen[1], f"clock did not advance: {seen}"
    assert seen[1][0] - seen[0][0] == 14400, seen
    print(f"  clock advances: {seen[0]} -> {seen[1]}")


def test_accel_is_published_so_a_misconfigured_node_is_visible(tmp):
    """Position derives from absolute wall-clock, so a node with a different
    CARBON_ACCEL reads a different phase. Publishing it makes that detectable
    instead of a silent skew in the ranking."""
    trace = write_multizone_trace(os.path.join(tmp, "accel.csv"),
                                  {"ZONE-A": 40})
    zone_file = os.path.join(tmp, "zone-a")
    open(zone_file, "w").write("ZONE-A\n")
    ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file,
                  "CARBON_ACCEL": "300"})
    assert ads.get("CarbonAccel") == "300", ads
    print("  CarbonAccel is published for cross-node comparison")


def test_injected_carbon_is_disclosed_not_smuggled(tmp):
    """The demo injection hook must be impossible to mistake for a reading:
    the override replaces CarbonIntensity (that is its job — it must move
    matchmaking), but CarbonInjected/CarbonInjectedUntil are published in
    the same breath and the trace's own value stays visible alongside."""
    import time as _time
    trace = write_trace(os.path.join(tmp, "inj.csv"))
    zone_file = os.path.join(tmp, "inj-zone")
    open(zone_file, "w").write("US-CAL-CISO\n")
    override = os.path.join(tmp, "override")
    open(override, "w").write(f"900.0 {int(_time.time()) + 600}\n")
    ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file,
                  "CARBON_OVERRIDE_FILE": override})
    assert float(ads["CarbonIntensity"]) == 900.0, ads
    assert ads.get("CarbonInjected") == "true", (
        "an injected value without the disclosure attribute is a fabricated "
        "measurement — exactly what this project deleted its simulator over")
    assert "CarbonInjectedUntil" in ads, ads
    assert "CarbonTraceIntensity" in ads, (
        "the real trace value must stay visible next to the injected one")
    print("  injected carbon replaces the value AND discloses itself")


def test_expired_or_malformed_override_lapses_to_measured(tmp):
    """An injection must clean up after itself: past expiry (or garbage in
    the file) the trace value returns with NO injection attributes."""
    import time as _time
    trace = write_trace(os.path.join(tmp, "exp.csv"))
    zone_file = os.path.join(tmp, "exp-zone")
    open(zone_file, "w").write("US-CAL-CISO\n")
    for content in (f"900.0 {int(_time.time()) - 5}\n",   # expired
                    "not-a-number whenever\n",            # malformed
                    "900.0\n"):                           # missing expiry
        override = os.path.join(tmp, "override-bad")
        open(override, "w").write(content)
        ads, _ = run({"CARBON_TRACE": trace, "CARBON_ZONE_FILE": zone_file,
                      "CARBON_OVERRIDE_FILE": override})
        assert "CarbonInjected" not in ads, (content, ads)
        assert 195 <= float(ads["CarbonIntensity"]) <= 210, (content, ads)
    print("  expired/malformed overrides lapse back to the measured trace")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="cc-classad-")
    for fn in (test_publishes_intensity_for_a_known_zone,
               test_unknown_zone_publishes_no_intensity,
               test_missing_trace_publishes_no_intensity,
               test_missing_zone_file_is_survivable,
               test_no_nvml_publishes_no_watts,
               test_output_is_valid_classad_syntax,
               test_all_zones_are_evaluated_at_the_same_replayed_instant,
               test_position_outside_a_zones_coverage_publishes_nothing,
               test_replay_clock_advances,
               test_accel_is_published_so_a_misconfigured_node_is_visible,
               test_injected_carbon_is_disclosed_not_smuggled,
               test_expired_or_malformed_override_lapses_to_measured):
        print(fn.__name__)
        fn(tmp)
    print("\nALL PASS")
