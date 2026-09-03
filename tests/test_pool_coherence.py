"""The replay-coherence check, in both places that implement it.

`pegasus/check_pool_match.py` is the pre-submit gate; `pegasus/dashboard.py`
carries a stdlib-only copy (`replay_coherence`) because it is uploaded to the
submit node alone and cannot import the gate. Duplicated logic drifts, so both
are held to the same scenarios here, including that they share one tolerance.

Comparing carbon intensity across sites is only meaningful if every site
evaluates the same function of absolute time. The publisher used to wrap the
replay with a per-zone period, and because EIA returns different coverage per
balancing authority the sites drifted into different phases — measured at 7.9
and 15.9 replayed HOURS apart on the live pool. Every number was real EIA
data; they were readings from different times of day, so `RANK` was ranking
phase as much as carbon.

The check must catch that, and must NOT cry wolf on a healthy pool. Both
false-positive modes below were real defects in the first version of this
check, reproduced before being fixed:

* **the wrap boundary** — the replay wraps, so two nodes either side of it are
  adjacent in replayed time while being a whole span apart numerically
  (157.9h on the real trace, once every ~31.6 min);
* **cron phase** — `STARTD_CRON` samples each node every 60s and the nodes are
  not in phase, so healthy nodes routinely differ by 60s x ACCEL = 5h.

Raising the tolerance past those would have exceeded the 7.9h divergence the
check exists to catch, so the comparison itself had to change: each node is
checked against its OWN sample instant, on a circle.

That per-node comparison verifies the replay FORMULA and is blind to a skewed
clock — carbon_classad derives the sample time and the position from the same
`time.time()`, so a node whose clock is fast is internally consistent and
scores a ZERO error while advertising a price 50 replayed hours from its peers
(600s of offset at ACCEL=300), in EITHER direction. Catching it needs the two
clocks HTCondor puts in every ad — `MyCurrentTime` from the startd and
`LastHeardFrom` from the collector — whose difference measures the offset
directly. Inferring it from ad AGE instead is what made an earlier version
asymmetric: it failed a fast clock and let a slow one pass as merely stale,
because this pool's healthy ads run 208-268s old and age cannot tell a lagging
clock from slow updates.

A node that publishes no clock pair is therefore UNVERIFIABLE, and fails — the
same rule already applied to a site publishing CarbonIntensity without
CarbonSpan. There is no age-based fallback for it, because a fallback that can
only see a fast clock reintroduces exactly that asymmetry.

`run` is injected, so no pool is needed — the fake returns the same text
`condor_status -af` would.
"""

import io
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "..", "pegasus", "check_pool_match.py")

SPAN = 158 * 3600          # the real trace's span: 568800s
ACCEL = 300.0
BASE = 1787000000.0        # an arbitrary replay position


def load():
    """Compile check_pool_match from SOURCE, never from cached bytecode.

    This matters because the file gets swapped under this test during mutation
    testing. Going through importlib's file loader, a stale `__pycache__`
    entry was executed after the source had been restored: the guard reported
    a 15.9h skew as coherent, and I spent a while debugging correct code.
    The same trap can hide a surviving mutant, which is worse.
    """
    source = open(MODULE).read()
    module = types.ModuleType("check_pool_match")
    module.__file__ = MODULE
    exec(compile(source, MODULE, "exec"), module.__dict__)
    return module


def row(name, sampled_at, position, accel=ACCEL, span=SPAN):
    return f"slot1@{name} {accel:.0f} {span:.0f} {sampled_at:.0f} {position:.0f}"


def healthy(name, sampled_at, t0=1000000.0, pos0=BASE, span=SPAN):
    """A node behaving exactly as carbon_classad.py does."""
    position = pos0 + (((sampled_at - t0) * ACCEL) % span)
    return row(name, sampled_at, position, span=span)


def capture(module, rows, ranking=None, clocks=None):
    """Run the check.

    Three queries are issued: the set of slots that PARTICIPATE IN RANKING
    (publish CarbonIntensity), the verifiable rows, and each node's two
    clocks. `ranking` defaults to exactly the rows' own names — i.e. every
    ranking node is verifiable, the healthy case. Pass it explicitly to model
    a node that ranks without being checkable, which is the state that used
    to be approved.

    `clocks` maps a slot name to its clock offset from the collector, in
    seconds (positive = that node's clock is fast). It defaults to a
    perfectly synchronised pool. Pass `clocks={}` to model a pool that
    publishes neither MyCurrentTime nor LastHeardFrom.
    """
    if ranking is None:
        # Faithful default: the real query is constrained to
        # `CarbonIntensity =!= UNDEFINED`, so a slot with undefined fields
        # (the submit node, say) is NOT in the ranking set. Deriving it from
        # every row regardless made the fixture claim a slot competes for jobs
        # when condor would never have offered it — and the test then failed
        # for a reason the pool could not produce.
        ranking = [r.split()[0] for r in rows
                   if len(r.split()) == 5
                   and not any(f in ("undefined", "error")
                               for f in r.split()[1:])]

    if clocks is None:
        clocks = {r.split()[0]: 0.0 for r in rows}

    def run(cmd):
        if "-af Name -constraint" in cmd:
            return "\n".join(ranking) + "\n"
        if "MyCurrentTime" in cmd:
            # The collector stamps LastHeardFrom on arrival; the startd
            # stamps MyCurrentTime from its own clock, so the offset is the
            # difference. Anchored to the real clock, since that is what the
            # checker compares against.
            heard = time.time()
            return "\n".join(
                f"{name} {heard + off:.0f} {heard:.0f}"
                for name, off in clocks.items()) + "\n"
        return "\n".join(rows) + "\n"

    held, sys.stdout = sys.stdout, io.StringIO()
    try:
        failed = module.check_replay_coherence(run)
        return failed, sys.stdout.getvalue()
    finally:
        sys.stdout = held


def test_healthy_pool_is_coherent():
    module = load()
    failed, text = capture(module, [
        healthy("CLEM-gpu", 1000000.0),
        healthy("TACC-gpu", 1000001.4),
        healthy("UTAH-gpu", 1000002.0),
    ])
    assert failed is False, text
    assert "coherent" in text, text
    print("  a healthy pool reports coherent")


def test_out_of_phase_crons_are_not_a_fault():
    """STARTD_CRON is 60s per node and unsynchronised — that is normal.

    The first version of this check flagged it as 5h of skew and would have
    blocked submission on a healthy pool.
    """
    module = load()
    failed, text = capture(module, [
        healthy("CLEM-gpu", 1000000.0),
        healthy("TACC-gpu", 1000000.0 + 59.0),   # sampled a cron period later
    ])
    assert failed is False, f"rejected a healthy out-of-phase pool: {text}"
    print("  out-of-phase 60s crons are not reported as skew")


def test_wrap_boundary_is_not_a_fault():
    """The reported bug: nodes either side of the wrap are ADJACENT in time.

    One node samples just before the replay wraps, the next just after. Their
    positions are a whole span apart numerically and ~1.4s apart in reality.
    """
    module = load()
    # Choose t0 so the first node sits 100 replayed seconds before the wrap.
    t_first = 1000000.0
    pos_first = BASE + SPAN - 100.0
    t_second = t_first + 1.4                      # 420 replayed seconds later
    pos_second = BASE + ((SPAN - 100.0 + 420.0) % SPAN)   # wrapped: BASE + 320
    failed, text = capture(module, [
        row("CLEM-gpu", t_first, pos_first),
        row("TACC-gpu", t_second, pos_second),
    ])
    assert pos_second < pos_first, "test did not actually straddle the wrap"
    assert failed is False, f"rejected a healthy pool at the wrap: {text}"
    print("  nodes straddling the replay wrap are not reported as skew")


def test_mismatched_span_is_caught():
    """Differing spans ARE the per-zone bug, structurally."""
    module = load()
    failed, text = capture(module, [
        healthy("CLEM-gpu", 1000000.0),
        row("TACC-gpu", 1000001.0, BASE + 500, span=156 * 3600),
    ])
    assert failed is True, f"accepted differing wrap periods: {text}"
    assert "MISMATCHED CarbonSpan" in text, text
    print("  differing spans (the per-zone bug) are caught")


def test_mismatched_accel_is_caught():
    module = load()
    failed, text = capture(module, [
        healthy("CLEM-gpu", 1000000.0),
        row("TACC-gpu", 1000001.0, BASE + 300, accel=600.0),
    ])
    assert failed is True, f"accepted mismatched accel: {text}"
    assert "MISMATCHED CarbonAccel" in text, text
    print("  a node with a different accel is caught")


def test_a_genuinely_skewed_clock_is_caught():
    """Same span and accel, but the position does not match its sample time.

    This is what a badly skewed node looks like, and it is the case the check
    exists for. The divergence used is the 7.9h originally measured.
    """
    module = load()
    good = healthy("CLEM-gpu", 1000000.0)
    bad = row("TACC-gpu", 1000001.0, BASE + 300 + 28475)   # 7.9h out of step
    failed, text = capture(module, [good, bad])
    assert failed is True, f"missed a 7.9h divergence: {text}"
    assert "out of step" in text, text
    print("  a genuinely out-of-step node is caught (7.9h, as measured)")


def test_undefined_values_do_not_become_numbers():
    module = load()
    failed, text = capture(module, [
        healthy("CLEM-gpu", 1000000.0),
        "slot1@submit undefined undefined undefined undefined",
    ])
    assert failed is False, text
    assert "submit" not in text, f"parsed an undefined slot as data: {text}"
    print("  undefined slots are skipped, not parsed as 0")


def test_a_pool_on_the_old_publisher_is_not_approved():
    """The state this check exists to detect must not pass it.

    Before this fix, a pool where nothing published CarbonSpan/CarbonSampledAt
    returned "no failure" and check_pool_match printed "safe to submit" — so
    the gate approved exactly the configuration that produced the 15.9h
    divergence. Nothing checkable is UNKNOWN, not fine.
    """
    module = load()
    failed, text = capture(
        module,
        rows=[],                                        # nothing verifiable
        ranking=["slot1@CLEM-gpu", "slot1@TACC-gpu", "slot1@UTAH-gpu"])
    assert failed is True, f"approved a pool it cannot verify: {text}"
    assert "UNVERIFIABLE SITES" in text, text
    print("  a pool running the old publisher is not approved")


def test_a_ranking_node_that_cannot_be_verified_fails():
    """Skipping the unverifiable node and passing the rest is a false approval.

    That node still competes for jobs on carbon, so its clock matters just as
    much as the ones that can be checked.
    """
    module = load()
    failed, text = capture(
        module,
        rows=[healthy("CLEM-gpu", 1000000.0), healthy("TACC-gpu", 1000001.0)],
        ranking=["slot1@CLEM-gpu", "slot1@TACC-gpu", "slot1@UTAH-gpu"])
    assert failed is True, f"passed while a ranking node was unchecked: {text}"
    assert "slot1@UTAH-gpu" in text, text
    print("  a ranking node that cannot be verified is a failure")


def test_a_single_site_is_not_claimed_comparable():
    """Vacuously true is not verified, and must not be reported as verified."""
    module = load()
    failed, text = capture(module, rows=[healthy("CLEM-gpu", 1000000.0)])
    assert failed is False, text
    assert "intensities are comparable" not in text, (
        f"claimed comparability with nothing to compare: {text}")
    assert "not demonstrated" in text, text
    print("  one site is reported as trivially satisfied, not demonstrated")


def test_no_carbon_anywhere_is_left_to_the_matchmaking_check():
    """If nothing publishes CarbonIntensity, carbon ranks nothing at all.

    That is a real problem, but it is the matchmaking check's to report (0
    slots match `CarbonIntensity =!= UNDEFINED`), and reporting it twice with
    different wording would be confusing.
    """
    module = load()
    failed, text = capture(module, rows=[], ranking=[])
    assert failed is False, text
    assert "ranks nothing" in text, text
    print("  no carbon at all defers to the matchmaking check")


# ---------------------------------------------------------------------------
# The dashboard runs the SAME test on the same attributes.
#
# pegasus/dashboard.py is uploaded to the submit node on its own and must stay
# stdlib-only, so it cannot import check_pool_match — the comparison is
# duplicated there as `replay_coherence()`. Duplicated logic drifts, and the
# first version of that function shipped the very bug this file exists to
# prevent: it compared raw CarbonTimestamp values across nodes, so healthy
# out-of-phase crons read as "sites not comparable" and a pool where only one
# node published a position read as "same on every site". The cases below hold
# the dashboard to the scenarios above.
DASHBOARD = os.path.join(HERE, "..", "pegasus", "dashboard.py")
NOW = 1000100.0


def load_dashboard():
    """Compile dashboard.py from source, for the same reason `load` does."""
    source = open(DASHBOARD).read()
    module = types.ModuleType("dashboard")
    module.__file__ = DASHBOARD
    exec(compile(source, DASHBOARD, "exec"), module.__dict__)
    return module


def dsite(name, sampled_at, t0=1000000.0, pos0=BASE, span=SPAN, accel=ACCEL,
          drift=0.0, carbon=300.0, offset=0.0, clocks=True):
    """A site dict as the poller builds one, behaving like carbon_classad.py.

    `drift` displaces the replay position away from what this node's own
    sample time implies — a broken replay formula, not a clock fault.
    `offset` is that node's CLOCK offset from the collector, published the
    way HTCondor does it: both clocks on the ad. `clocks=False` models an ad
    carrying neither, which is unverifiable rather than fine.
    """
    position = pos0 + (((sampled_at - t0) * accel) % span) + drift
    site = {"site": name, "carbon": carbon, "carbon_accel": accel,
            "carbon_span": span, "carbon_sampled_at": sampled_at,
            "carbon_ts": position}
    if clocks:
        site["ad_heard_at"] = NOW
        site["ad_node_time"] = NOW + offset
    return site


def test_dashboard_does_not_cry_wolf_on_out_of_phase_crons():
    """The bug this whole file exists to prevent, on the dashboard side.

    Three healthy nodes, crons up to 50s out of phase: at ACCEL=300 that is
    over 4 replayed hours of legitimate difference in CarbonTimestamp.
    """
    d = load_dashboard()
    verdict = d.replay_coherence([dsite("CLEM-gpu", 1000000.0),
                                  dsite("TACC-gpu", 1000032.0),
                                  dsite("UTAH-gpu", 1000050.0)], NOW)
    assert verdict["coherent"] is True, verdict
    assert verdict["state"] != "bad", verdict
    print("  out-of-phase crons are not reported as incomparable")


def test_dashboard_survives_the_wrap_boundary():
    d = load_dashboard()
    # pos0 within one cron period of the wrap, so the nodes land either side.
    verdict = d.replay_coherence(
        [dsite("CLEM-gpu", 1000000.0, pos0=BASE + SPAN - 600),
         dsite("TACC-gpu", 1000040.0, pos0=BASE + SPAN - 600)], NOW)
    assert verdict["coherent"] is True, verdict
    print("  the replay wrap is not reported as incomparable")


def test_dashboard_catches_a_genuinely_skewed_clock():
    """A displacement above accel x CLOCK_SKEW_ALLOWANCE_S must fail."""
    d = load_dashboard()
    over = d.CLOCK_SKEW_ALLOWANCE_S * ACCEL * 2
    verdict = d.replay_coherence([dsite("CLEM-gpu", 1000000.0),
                                  dsite("TACC-gpu", 1000010.0, drift=over)],
                                 NOW)
    assert verdict["coherent"] is False, verdict
    assert verdict["state"] == "bad", verdict
    print("  genuine clock skew is caught")


def test_dashboard_catches_mismatched_span_and_accel():
    d = load_dashboard()
    for label, other in (("span", dsite("TACC-gpu", 1000000.0,
                                        span=SPAN - 3600)),
                         ("accel", dsite("TACC-gpu", 1000000.0, accel=600.0))):
        verdict = d.replay_coherence([dsite("CLEM-gpu", 1000000.0), other],
                                     NOW)
        assert verdict["coherent"] is False, (label, verdict)
        assert label.capitalize() in verdict["detail"] or label in \
            verdict["detail"].lower(), (label, verdict)
    print("  mismatched span and accel are both caught")


def test_dashboard_does_not_approve_an_unverifiable_ranking_site():
    """A node ranking on carbon with no checkable position is not 'fine'.

    This is the false-ALL-CLEAR half of the original bug: filtering such a
    node out left one position behind, which then trivially 'agreed with
    itself' and was reported as coherent.
    """
    d = load_dashboard()
    verdict = d.replay_coherence(
        [dsite("CLEM-gpu", 1000000.0), dsite("TACC-gpu", 1000010.0),
         {"site": "OLD-gpu", "carbon": 250.0, "carbon_ts": BASE,
          "carbon_sampled_at": None, "carbon_span": None,
          "carbon_accel": None}], NOW)
    assert verdict["coherent"] is False, verdict
    assert "OLD-gpu" in verdict["detail"], verdict
    print("  an unverifiable ranking site is not approved")


def test_dashboard_does_not_claim_a_single_site_is_comparable():
    d = load_dashboard()
    verdict = d.replay_coherence([dsite("CLEM-gpu", 1000000.0)], NOW)
    assert verdict["coherent"] is None, verdict
    assert "not demonstrated" in verdict["headline"], verdict
    print("  one site is trivially satisfied, not demonstrated")


def test_dashboard_reports_freshness_separately_from_coherence():
    """Internally consistent is not current, and one must not imply the other.

    The negotiator ranks the collector's copy of an ad, so a coherent pool
    whose ads are older than one trace row (3600/accel seconds) is ranking
    values the trace has moved past.
    """
    d = load_dashboard()
    stale = d.replay_coherence([dsite("CLEM-gpu", NOW - 300),
                                dsite("TACC-gpu", NOW - 320)], NOW)
    assert stale["coherent"] is True and stale["fresh"] is False, stale
    assert stale["state"] == "warn", stale
    fresh = d.replay_coherence([dsite("CLEM-gpu", NOW - 3),
                                dsite("TACC-gpu", NOW - 8)], NOW)
    assert fresh["coherent"] is True and fresh["fresh"] is True, fresh
    assert fresh["state"] == "ok", fresh
    print("  freshness is reported separately from coherence")


def skewed_clock(name, offset, sampled_ago=3.0, now=NOW, clocks=True):
    """A node whose CLOCK is `offset` seconds off the collector's.

    The sample time AND the position come from that same skewed clock,
    exactly as carbon_classad.py produces them — which is why this case is
    invisible to the per-node comparison and needs the ad's two clocks.
    `clocks=False` withholds them, i.e. an unverifiable node.
    """
    sampled = (now + offset) - sampled_ago
    site = {"site": name, "carbon": 300.0, "carbon_accel": ACCEL,
            "carbon_span": SPAN, "carbon_sampled_at": sampled,
            "carbon_ts": BASE + ((sampled * ACCEL) % SPAN)}
    if clocks:
        site["ad_heard_at"] = now
        site["ad_node_time"] = now + offset
    return site


def test_a_self_consistent_but_skewed_clock_is_not_green_lit():
    """The per-node check scores this ZERO error, and it must still fail.

    carbon_classad derives `CarbonSampledAt` and `CarbonTimestamp` from the
    same `time.time()`, so a node whose clock is 600s fast is *internally
    perfect* while advertising a price 600 x 300 / 3600 = 50 replayed hours
    from its peers. The first version of the dashboard badge reported this
    as `coherent, fresh, state=ok` — a green light on an incomparable pool.
    """
    d = load_dashboard()
    verdict = d.replay_coherence(
        [skewed_clock("CLEM-gpu", 0.0), skewed_clock("TACC-gpu", 600.0)], NOW)
    assert verdict["coherent"] is False, verdict
    assert verdict["state"] == "bad", verdict
    assert "TACC-gpu" in verdict["headline"], verdict
    # The cost must be stated in replayed hours, not merely flagged. Exactly
    # 50.0 now: the offset is MEASURED from the ad's two clocks rather than
    # inferred from the sample age, which used to shave 3s off it.
    assert "replayed hours" in verdict["detail"], verdict
    assert "50.0 replayed hours" in verdict["detail"], verdict
    print("  a self-consistent node with a fast clock is caught")


def test_a_single_site_with_a_bad_clock_is_still_caught():
    """No peer to compare with is not a reason to trust the clock."""
    d = load_dashboard()
    verdict = d.replay_coherence([skewed_clock("CLEM-gpu", 600.0)], NOW)
    assert verdict["state"] == "bad", verdict
    print("  a lone node with a fast clock is still caught")


def test_normal_ntp_jitter_is_not_a_clock_fault():
    """The check must not fire on sub-allowance disagreement."""
    d = load_dashboard()
    small = d.CLOCK_SKEW_ALLOWANCE_S / 2.0
    verdict = d.replay_coherence(
        [skewed_clock("CLEM-gpu", 0.0, sampled_ago=3.0),
         skewed_clock("TACC-gpu", small, sampled_ago=3.0)], NOW)
    assert verdict["coherent"] is True, verdict
    assert verdict["state"] == "ok", verdict
    print("  sub-allowance clock jitter is not reported as a fault")


def test_checker_names_only_the_nodes_whose_clocks_it_cannot_measure():
    """A whole-pool failure must still point at the right node.

    Blaming a node that published its clocks correctly sends whoever reads
    this to the wrong machine, and the message is the entire value of the
    check to an operator.
    """
    module = load()
    now = time.time()
    failed, text = capture(
        module,
        [healthy("CLEM-gpu", now - 3.0, t0=now),
         healthy("TACC-gpu", now - 4.0, t0=now)],
        clocks={"slot1@CLEM-gpu": 0.0})            # TACC's clocks are absent
    assert failed is True, text
    assert "UNVERIFIABLE CLOCKS ['slot1@TACC-gpu']" in text, text
    assert "coherent: all" not in text, (
        f"reported coherence with an unverifiable clock: {text}")
    print("  only the unmeasurable node is named")


def test_checker_does_not_fire_on_a_healthy_recent_pool():
    """The clock check must not break the ordinary healthy case."""
    module = load()
    now = time.time()
    failed, text = capture(module, [
        healthy("CLEM-gpu", now - 2.0, t0=now),
        healthy("TACC-gpu", now - 6.0, t0=now),
    ])
    assert failed is False, text
    assert "IN THE FUTURE" not in text, text
    assert "coherent: all" in text, text
    print("  a healthy, recently-sampled pool still passes")


def test_a_slow_clock_fails_the_gate_exactly_like_a_fast_one():
    """A lagging clock displaces the advertised carbon just as far.

    The asymmetry this replaces was real: skew inferred from ad AGE fails a
    fast clock (a sample from the future) but lets a slow one through with a
    staleness warning, even though both put the site's replay position the
    same distance from its peers. Measuring the offset from the ad's two
    clocks removes the asymmetry.
    """
    module = load()
    t0 = 1000000.0
    for label, offset in (("slow", -600.0), ("fast", 600.0)):
        failed, text = capture(
            module,
            [healthy("CLEM-gpu", t0), healthy("TACC-gpu", t0 + 1.0)],
            clocks={"slot1@CLEM-gpu": 0.0, "slot1@TACC-gpu": offset})
        assert failed is True, f"{label} clock passed the gate: {text}"
        assert "CLOCK IS" in text, (label, text)
        assert ("BEHIND" if offset < 0 else "AHEAD OF") in text, (label, text)
        assert "50.0 replayed hours" in text, (label, text)
    print("  a slow clock fails the gate exactly like a fast one")


def test_dashboard_fails_a_slow_clock_too():
    """The badge must not disagree with the gate on the same pool."""
    d = load_dashboard()
    for label, offset in (("slow", -600.0), ("fast", 600.0)):
        verdict = d.replay_coherence(
            [skewed_clock("CLEM-gpu", 0.0),
             skewed_clock("TACC-gpu", offset)], NOW)
        assert verdict["coherent"] is False, (label, verdict)
        assert verdict["state"] == "bad", (label, verdict)
        assert ("behind" if offset < 0 else "ahead of") in verdict["detail"], (
            label, verdict)
        assert "50.0 replayed hours" in verdict["detail"], (label, verdict)
    print("  the badge fails a slow clock too")


def test_a_stale_ad_on_a_good_clock_is_not_called_a_clock_fault():
    """The other half of the disambiguation: age alone is NOT evidence.

    This pool's healthy ads run 208-268s old with UPDATE_INTERVAL=60, so
    treating age as skew would fail a perfectly synchronised pool — the exact
    cry-wolf failure this file exists to prevent.
    """
    module = load()
    now = time.time()
    failed, text = capture(
        module,
        [healthy("CLEM-gpu", now - 240.0, t0=now),
         healthy("TACC-gpu", now - 250.0, t0=now)],
        clocks={"slot1@CLEM-gpu": 0.0, "slot1@TACC-gpu": 0.0})
    assert failed is False, f"a stale-but-synchronised pool was failed: {text}"
    assert "CLOCK IS" not in text, text
    assert "WARNING" in text, f"staleness should still be reported: {text}"
    print("  a stale ad on a good clock is staleness, not a clock fault")


def test_dashboard_corrects_ad_age_for_a_measured_offset():
    """A slow clock must not masquerade as a stale ad in the freshness line.

    With the offset known, a node sampled 600s ago by its own slow clock was
    really sampled just now, and the freshness verdict must say so.
    """
    d = load_dashboard()
    # Offset -600: the node's clock is 600s behind, so its own sample time
    # reads 600s in the past while the sample is in fact current. Under the
    # allowance for the clock check? No — so use a small offset that is, and
    # verify the age correction rather than the fault path.
    small = d.CLOCK_SKEW_ALLOWANCE_S / 2.0
    verdict = d.replay_coherence(
        [skewed_clock("CLEM-gpu", 0.0, sampled_ago=2.0),
         skewed_clock("TACC-gpu", -small, sampled_ago=2.0)], NOW)
    assert verdict["coherent"] is True, verdict
    assert verdict["fresh"] is True, verdict
    print("  ad age is corrected by the measured clock offset")


def test_dashboard_fails_a_node_whose_clock_cannot_be_measured():
    """No MyCurrentTime/LastHeardFrom pair means no verdict, not a pass.

    Inferring skew from ad age instead only ever sees a clock running AHEAD,
    so a slow clock would sail through — the hole this closes. Same rule the
    check already applies to a site publishing CarbonIntensity without
    CarbonSpan: unverifiable is not fine.
    """
    d = load_dashboard()
    verdict = d.replay_coherence(
        [skewed_clock("CLEM-gpu", 0.0),
         skewed_clock("TACC-gpu", 0.0, clocks=False)], NOW)
    assert verdict["coherent"] is False, verdict
    assert verdict["state"] == "bad", verdict
    assert "TACC-gpu" in verdict["detail"], verdict
    assert "unverifiable" in verdict["detail"].lower(), verdict
    print("  a node with no measurable clock is not approved")


def test_a_slow_clock_with_no_clock_fields_does_not_slip_through():
    """The exact reported gap, stated as its own case.

    A node 600s SLOW that publishes neither clock: age-based inference sees
    only a stale ad, and every other check sees a self-consistent node.
    """
    d = load_dashboard()
    verdict = d.replay_coherence(
        [skewed_clock("CLEM-gpu", 0.0),
         skewed_clock("TACC-gpu", -600.0, clocks=False)], NOW)
    assert verdict["coherent"] is False, verdict
    assert verdict["state"] == "bad", verdict
    module = load()
    t0 = 1000000.0
    failed, text = capture(
        module, [healthy("CLEM-gpu", t0), healthy("TACC-gpu", t0 + 1.0)],
        clocks={"slot1@CLEM-gpu": 0.0})          # TACC publishes no clocks
    assert failed is True, f"gate passed an unmeasurable clock: {text}"
    assert "UNVERIFIABLE CLOCKS" in text, text
    print("  a slow clock with no clock fields is caught on both sides")


def test_healthy_nodes_really_do_publish_different_timestamps():
    """Pin the fact the docs kept getting wrong.

    The notebook twice claimed `CarbonTimestamp` is "identical everywhere",
    which reads as a thing you could verify with `condor_status -af
    CarbonTimestamp`. It is not: carbon_classad computes
    `global_start + ((now * ACCEL) % span)`, which is continuous in each
    node's own sample time, so unsynchronised 60s crons put healthy nodes up
    to 60 x ACCEL apart. Anyone comparing the raw values would "find" a fault
    on a healthy pool — the exact false positive this file exists to prevent.
    """
    a, b = dsite("CLEM-gpu", 1000000.0), dsite("TACC-gpu", 1000030.0)
    assert a["carbon_ts"] != b["carbon_ts"], (
        "fixture no longer models the real publisher")
    apart = abs(a["carbon_ts"] - b["carbon_ts"])
    assert apart == 30.0 * ACCEL, apart
    # ...and that difference is emphatically not small.
    assert apart / 3600.0 > 2.0, f"only {apart / 3600.0:.1f} replayed h apart"

    # Yet the check must still call this pool coherent.
    verdict = load_dashboard().replay_coherence([a, b], NOW)
    assert verdict["coherent"] is True, verdict
    print(f"  healthy nodes differ by {apart / 3600.0:.1f} replayed h "
          f"and are still coherent")


def test_the_badge_names_the_latest_sample_across_the_wrap():
    """`max(carbon_ts)` is not "newest" — position wraps, sample time doesn't.

    At the wrap boundary the most recently sampled node holds the SMALLEST
    position, so picking the maximum names the OLDEST node's position, off by
    nearly a whole span (157.9h here). Order by sample time instead.
    """
    d = load_dashboard()
    # The replay wraps every span/accel = 1896s of sample time, and pos0 sits
    # OUTSIDE the modulo — so the straddle has to come from `t0`. Anchor it
    # 1899s back: the older sample then lands 1895s in (position SPAN-300),
    # the newer one 1897s in, which has rolled over to position 300.
    older = dsite("CLEM-gpu", NOW - 4.0, t0=NOW - 1899.0)
    newer = dsite("TACC-gpu", NOW - 2.0, t0=NOW - 1899.0)
    assert newer["carbon_ts"] < older["carbon_ts"], (
        "fixture does not straddle the wrap")
    assert older["carbon_ts"] - newer["carbon_ts"] > SPAN - 3600, (
        "the two positions should be nearly a whole span apart numerically")

    verdict = d.replay_coherence([older, newer], NOW)
    assert verdict["coherent"] is True, verdict          # still a healthy pool
    shown = time.strftime("%Y-%m-%d %H:%M",
                          time.gmtime(newer["carbon_ts"]))
    stale = time.strftime("%Y-%m-%d %H:%M",
                          time.gmtime(older["carbon_ts"]))
    assert shown in verdict["headline"], (
        f"expected the latest sample's position {shown}: {verdict}")
    assert stale not in verdict["headline"], (
        f"named the OLDEST position {stale} as the latest: {verdict}")
    print("  the badge names the latest sample, not the largest position")


def test_dashboard_and_checker_share_one_allowance():
    """The two copies must not drift apart on the tolerance itself."""
    assert (load_dashboard().CLOCK_SKEW_ALLOWANCE_S
            == load().CLOCK_SKEW_ALLOWANCE_S), (
        "dashboard.py and check_pool_match.py disagree on "
        "CLOCK_SKEW_ALLOWANCE_S, so the dashboard and the pre-submit gate "
        "would give different verdicts on the same pool")
    print("  both copies share one clock-skew allowance")


if __name__ == "__main__":
    for fn in (test_healthy_pool_is_coherent,
               test_out_of_phase_crons_are_not_a_fault,
               test_wrap_boundary_is_not_a_fault,
               test_mismatched_span_is_caught,
               test_mismatched_accel_is_caught,
               test_a_genuinely_skewed_clock_is_caught,
               test_undefined_values_do_not_become_numbers,
               test_a_pool_on_the_old_publisher_is_not_approved,
               test_a_ranking_node_that_cannot_be_verified_fails,
               test_a_single_site_is_not_claimed_comparable,
               test_no_carbon_anywhere_is_left_to_the_matchmaking_check,
               test_dashboard_does_not_cry_wolf_on_out_of_phase_crons,
               test_dashboard_survives_the_wrap_boundary,
               test_dashboard_catches_a_genuinely_skewed_clock,
               test_dashboard_catches_mismatched_span_and_accel,
               test_dashboard_does_not_approve_an_unverifiable_ranking_site,
               test_dashboard_does_not_claim_a_single_site_is_comparable,
               test_dashboard_reports_freshness_separately_from_coherence,
               test_a_self_consistent_but_skewed_clock_is_not_green_lit,
               test_a_single_site_with_a_bad_clock_is_still_caught,
               test_normal_ntp_jitter_is_not_a_clock_fault,
               test_checker_names_only_the_nodes_whose_clocks_it_cannot_measure,
               test_checker_does_not_fire_on_a_healthy_recent_pool,
               test_a_slow_clock_fails_the_gate_exactly_like_a_fast_one,
               test_dashboard_fails_a_slow_clock_too,
               test_a_stale_ad_on_a_good_clock_is_not_called_a_clock_fault,
               test_dashboard_corrects_ad_age_for_a_measured_offset,
               test_dashboard_fails_a_node_whose_clock_cannot_be_measured,
               test_a_slow_clock_with_no_clock_fields_does_not_slip_through,
               test_healthy_nodes_really_do_publish_different_timestamps,
               test_the_badge_names_the_latest_sample_across_the_wrap,
               test_dashboard_and_checker_share_one_allowance):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
