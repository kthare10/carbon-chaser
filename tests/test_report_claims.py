"""The report's prose may only claim what its input files actually support.

The report and the dashboard are the artifacts a reader trusts, and both
narrate a story ("migrated to a cleaner grid", "resumed from a checkpoint")
that is far easier to assert than to verify. Three assertions in particular
are not derivable from the files these surfaces read:

1. WHY a segment moved. The progress files record which host trained each
   segment, never the carbon intensity the match was ranked on — that lives
   in the job ads. A host change can equally be a drain or a preemption.
2. That nothing moved BECAUSE one grid stayed cleanest. A single-site run is
   indistinguishable here from a single-site pool.
3. That each segment RESUMED from its predecessor. The evidence is the step
   counter climbing across every boundary, not the AUC curve looking
   well-behaved — a run that silently restarted still draws a plausible
   curve, and would be reported as a successful hand-off.

(3) is the dangerous one: it turns the demo's headline claim — carbon-aware
migration is free — into an unfalsifiable one. So the reset case is checked
on both surfaces, and the dashboard is fed its progress in the STEP order
the poller really sends, because that sort is what would reorder a restarted
counter into looking contiguous.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "carbon_chaser", "workload"))

from report_higgs import (chain_is_checkable, kpis,  # noqa: E402
                          migration_count, observed_restart,
                          steps_carry_over)

DASHBOARD = os.path.join(HERE, "..", "pegasus", "dashboard.py")

# Phrases that assert a cause or an outcome the input files cannot show.
UNSUPPORTED = (
    "because its grid had gone cleaner",
    "ever ranked cleaner",
    "stayed cleanest",
    "migrating to a cleaner grid mid-training",
)


def seg(index, auc, step, host):
    return {"index": index, "auc": auc, "step": step, "host": host,
            "gpu_wh": 2.2}


HEALTHY = [seg(1, 0.808, 12000, "STAR-gpu"), seg(2, 0.840, 24000, "TACC-gpu"),
           seg(3, 0.856, 36000, "TACC-gpu"), seg(4, 0.863, 48000, "UTAH-gpu")]
# Segment 3 starts over: the failure the prose must not paper over.
RESTARTED = [seg(1, 0.808, 12000, "STAR-gpu"), seg(2, 0.840, 24000, "STAR-gpu"),
             seg(3, 0.700, 9000, "UTAH-gpu"), seg(4, 0.850, 21000, "UTAH-gpu")]
SINGLE_SITE = [seg(i, 0.80 + i / 100, 12000 * i, "STAR-gpu")
               for i in range(1, 5)]
# Segments 3-4 never reported. Steps still climb across the hole, so a naive
# monotonicity check calls this continuous — but nothing was observed at the
# 2->5 boundary, so continuity is unknown, not verified.
GAPPED = [seg(1, 0.808, 12000, "STAR-gpu"), seg(2, 0.840, 24000, "STAR-gpu"),
          seg(5, 0.866, 60000, "UTAH-gpu"), seg(6, 0.871, 72000, "UTAH-gpu")]
# Segments 1-2 never reported. Internally consecutive and monotonic, so every
# per-pair check passes — but the two hand-offs before segment 3 were never
# observed, so "every segment resumed" is still unearned.
LATE = [seg(3, 0.856, 36000, "STAR-gpu"), seg(4, 0.863, 48000, "UTAH-gpu"),
        seg(5, 0.866, 60000, "UTAH-gpu")]
# Incomplete AND carrying a restart at a boundary both of whose sides
# reported (1->2, 3->4). Incompleteness must not demote proof to "unknown".
GAPPED_RESTART = [seg(1, 0.808, 12000, "STAR-gpu"),
                  seg(2, 0.700, 9000, "STAR-gpu"),
                  seg(5, 0.866, 60000, "UTAH-gpu"),
                  seg(6, 0.871, 72000, "UTAH-gpu")]
LATE_RESTART = [seg(3, 0.856, 36000, "STAR-gpu"),
                seg(4, 0.700, 9000, "UTAH-gpu"),
                seg(5, 0.850, 21000, "UTAH-gpu")]
RESULT = {"val_auc": 0.8709, "trained_steps": 48000, "val_events": 200000,
          "sig_eff_at_bkg_rej_0.99": 0.2319}


def text(html):
    return re.sub(r"<[^>]+>", " ", html)


def test_step_counter_is_what_detects_a_broken_chain():
    assert steps_carry_over(HEALTHY)
    assert steps_carry_over(SINGLE_SITE)
    assert not steps_carry_over(RESTARTED)
    # A dip in AUC is NOT a broken chain, and must not be read as one.
    dipped = [seg(1, 0.84, 1000, "A"), seg(2, 0.79, 2000, "A")]
    assert steps_carry_over(dipped)
    print("  a restarted counter is detected; an AUC dip is not mistaken "
          "for one")


def test_absent_evidence_is_not_verified_continuity():
    """all()/every() over an empty sequence is True. That is the trap."""
    assert not chain_is_checkable([]), "no segments cannot be 'checked fine'"
    assert not chain_is_checkable([seg(1, 0.8, 1000, "A")])
    assert not chain_is_checkable(GAPPED), "a hole is not an observed boundary"
    assert not chain_is_checkable(LATE), \
        "a chain starting at segment 3 says nothing about hand-offs 1->2->3"
    assert chain_is_checkable(HEALTHY)
    # steps_carry_over alone is fooled by BOTH — which is exactly why the
    # callers must pair it with chain_is_checkable.
    assert steps_carry_over(GAPPED) and steps_carry_over(LATE)
    print("  empty, single, gapped and late-starting chains are all treated "
          "as unchecked")


def test_incomplete_chain_makes_no_handoff_claim():
    for name, segments in (("gapped", GAPPED), ("late-starting", LATE)):
        body = text(kpis(segments, RESULT))
        assert "Never restarts" not in body, f"{name}: {body}"
        assert "one continuous training run" not in body, f"{name}: {body}"
        assert "cannot be checked" in body, f"{name}: {body}"
        assert "across the whole chain" not in body, f"{name}: {body}"
    print("  a chain with a hole, or one that starts late, claims nothing "
          "about its hand-offs")


def test_incompleteness_never_hides_a_proven_restart():
    """The worst failure mode: a hole demoting proof of failure to "unknown".

    A regression at a boundary whose both sides reported is observed fact. A
    gap somewhere else in the chain does not un-observe it — so the failure
    must survive incompleteness, even though the SUCCESS claim cannot.
    """
    assert observed_restart(GAPPED_RESTART) == [2], GAPPED_RESTART
    assert observed_restart(LATE_RESTART) == [4], LATE_RESTART
    # ...and it is judged without reference to completeness.
    assert not chain_is_checkable(GAPPED_RESTART)
    assert not chain_is_checkable(LATE_RESTART)
    # A clean chain, and clean-but-incomplete chains, prove no restart.
    for clean in (HEALTHY, SINGLE_SITE, GAPPED, LATE, []):
        assert observed_restart(clean) == [], clean

    for name, segments in (("gapped", GAPPED_RESTART),
                           ("late-starting", LATE_RESTART)):
        body = text(kpis(segments, RESULT))
        assert "Not one continuous run" in body, f"{name}: {body}"
        assert "cannot be checked" not in body, f"{name}: {body}"
        assert "Never restarts" not in body, f"{name}: {body}"
        # And it says WHICH segment, so the claim is checkable by hand.
        assert "started over" in body, f"{name}: {body}"
    print("  a restart proven at an observed boundary survives an "
          "incomplete chain")


def test_migration_count_is_a_floor_when_the_chain_is_incomplete():
    """Segments 2 and 5 are adjacent in the file list, not in the run."""
    exact = text(kpis(HEALTHY, RESULT))
    assert "migration(s) over" in exact, exact
    assert "≥" not in exact, exact

    floor = text(kpis(GAPPED, RESULT))
    assert "≥" in floor, floor
    assert "among the 4 segments reported" in floor, floor
    print("  an incomplete chain reports migrations as a floor, not a count")


def test_migration_count_follows_the_hosts():
    assert migration_count(HEALTHY) == 2
    assert migration_count(SINGLE_SITE) == 0
    assert migration_count([seg(1, 0.8, 1000, "A")]) == 0
    print("  migrations are counted from the recorded hosts")


def test_restarted_chain_is_not_reported_as_one_continuous_run():
    broken = text(kpis(RESTARTED, RESULT))
    assert "Never restarts" not in broken, broken
    assert "one continuous training run" not in broken, broken
    assert "Not one continuous run" in broken, broken
    # The step total is the final checkpoint's own count, not the chain's.
    assert "across the whole chain" not in broken, broken

    intact = text(kpis(HEALTHY, RESULT))
    assert "Never restarts" in intact, intact
    assert "across the whole chain" in intact, intact
    print("  a restarted chain is flagged; an intact one still reads as "
          "continuous")


def test_kpis_never_assert_a_cause_they_cannot_see():
    for name, segments in (("healthy", HEALTHY), ("restarted", RESTARTED),
                           ("single-site", SINGLE_SITE)):
        body = text(kpis(segments, RESULT)).lower()
        for phrase in UNSUPPORTED:
            assert phrase not in body, f"{name}: {phrase}"
    # A single-site run must not be explained, only reported.
    quiet = text(kpis(SINGLE_SITE, RESULT))
    assert "cannot say why" in quiet, quiet
    print("  no tile explains WHY the pool placed a segment where it did")


def test_dashboard_derives_the_chain_in_segment_order():
    """The client re-sorts by segment before judging contiguity.

    dashboard.py's poller sorts progress by step. Judging the chain in that
    order would find a restarted counter contiguous every time — the bug
    this assertion exists to keep out.
    """
    src = open(DASHBOARD).read()
    render = src[src.index("function renderScience"):]
    render = render[:render.index("const STATUS")]
    assert "sort((a, b) => a.segment - b.segment)" in render, \
        "renderScience must order the chain by segment, not by step"
    # Length must be checked BEFORE .every(), which is true on an empty
    # array — otherwise a run with no progress reads as verified continuous.
    assert "const checkable = chain.length > 1 && !gaps" in render, render
    # The failure must NOT be gated on completeness; the success must be.
    assert "const restarted = restartAt.length > 0;" in render, \
        "a proven restart must not depend on the chain being complete"
    assert "const unbroken = checkable && !restarted" in render, render
    # A chain that starts at segment 3 must count as gapped, same as a hole.
    assert "segs[0] !== 1" in render, \
        "renderScience must treat a late-starting chain as unverifiable"
    # All three branches must exist: the claim, its refusal, and "unknown".
    assert "counter never restarts" in render
    assert "not one continuous run" in render
    assert "cannot be\n           checked here" in render, \
        "renderScience needs an explicit unknown-continuity branch"
    for phrase in UNSUPPORTED:
        assert phrase not in render.lower(), phrase
    print("  the dashboard judges the chain by segment order, and has a "
          "both-ways branch")


if __name__ == "__main__":
    for fn in (test_step_counter_is_what_detects_a_broken_chain,
               test_absent_evidence_is_not_verified_continuity,
               test_incomplete_chain_makes_no_handoff_claim,
               test_incompleteness_never_hides_a_proven_restart,
               test_migration_count_is_a_floor_when_the_chain_is_incomplete,
               test_migration_count_follows_the_hosts,
               test_restarted_chain_is_not_reported_as_one_continuous_run,
               test_kpis_never_assert_a_cause_they_cannot_see,
               test_dashboard_derives_the_chain_in_segment_order):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
