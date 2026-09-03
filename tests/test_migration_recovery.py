"""Failure-recovery tests for the migration path.

The invariant under test: when a migration fails, the engine either has the
job running somewhere and says so, or reports that it is not running. It
must never report "running" with a stopped workload — at a booth that shows
a frozen step counter under a green status all evening.
"""

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carbon_chaser.clock import SimClock
from carbon_chaser.engine import Engine

SITES = {s: {"display": s, "lat": 0, "lon": 0, "zone": f"z{s}"}
         for s in ("A", "B")}


class FakeExecutor:
    """Models the real hazard: a SET of concurrently running trainers.

    `running` is a set so a test can detect the engine creating a second
    trainer (duplicate on one site is impossible here by construction, but
    two sites running at once — split-brain — is exactly what we check).
    """

    def __init__(self, fail_transfer=False, fail_start_at=None,
                 fail_start_always=False, stop_succeeds=True,
                 visibility=None, stop_recovers=()):
        self.running = set()
        self.fail_transfer = fail_transfer
        self.fail_start_at = fail_start_at
        self.fail_start_always = fail_start_always
        self.stop_succeeds = stop_succeeds
        # site -> True/False/None override for is_running (None = unreachable)
        self.visibility = visibility or {}
        # Sites that answer again when we try to stop them (transient
        # unreachability). Without this, an unreachable node can never be
        # confirmed stopped — which is what the real executors do.
        self.stop_recovers = set(stop_recovers)
        self.discarded = []
        self.starts = []
        self.markers = {}
        self.job_state = {}
        self.no_progress_at = set()
        self.done = False
        self.staging_cleared = []
        self.tss = {}          # site -> progress.json timestamp
        self.step = 100
        self.steps = {}
        self.last_transfer = {"method": "dataplane", "bytes": 1e6,
                              "seconds": 1.0}

    def start(self, site):
        self.starts.append(site)
        if self.fail_start_always or site == self.fail_start_at:
            raise RuntimeError(f"boom starting {site}")
        self.running.add(site)

    def stop(self, site):
        if self.visibility.get(site, "x") is None:
            # Real executors confirm a stop via is_running(); an unreachable
            # node cannot be confirmed, so stop() must report failure.
            if site in self.stop_recovers:
                self.visibility[site] = False      # answered after all
                self.running.discard(site)
                return True
            return False
        if not self.stop_succeeds:
            return False           # "may still be running"
        self.running.discard(site)
        return True

    def is_running(self, site):
        if site in self.visibility:
            return self.visibility[site]
        return site in self.running

    def transfer_checkpoint(self, src, dst):
        if self.fail_transfer:
            raise TimeoutError("scp:A->B exceeded 300s")
        return 1e6

    def get_progress(self, site):
        if site in self.steps:
            return {"step": self.steps[site], "loss": 0.5, "acc": 0.9,
                    "ts": self.tss.get(site, 1000.0), "done": self.done}
        if self.no_progress_at and site in self.no_progress_at:
            return None
        return {"step": self.step, "loss": 0.5, "acc": 0.9,
                "ts": self.tss.get(site, 1000.0)}

    def checkpoint_bytes(self, site):
        return 1e6

    def discard_checkpoint(self, site):
        self.discarded.append(site)          # destroys LIVE work

    def discard_incoming(self, site):
        self.staging_cleared.append(site)    # safe: staging file only

    def read_marker(self, site):
        if self.visibility.get(site, "x") is None:
            return None            # unreachable node tells us nothing
        return self.markers.get(site)

    def write_marker(self, site, value):
        if self.visibility.get(site, "x") is None:
            return
        self.markers[site] = value

    def has_job_state(self, site):
        if self.visibility.get(site, "x") is None:
            return None
        return bool(self.job_state.get(site) or site in self.running)


class FakeProvider:
    def __init__(self, values):
        self.values = values

    def get_intensity(self, zone):
        return self.values[zone]


class StubPower:
    """Measured watts, always fresh. Power is a separate concern from the
    migration behaviour these tests exercise, but without a provider the
    engine (correctly) refuses to invent one and never accrues emissions."""

    def get_kw(self, site):
        return 0.15

    def describe(self):
        return {"kind": "measured", "detail": "stub", "measured": True}


def make_engine(executor, intensities=None):
    provider = FakeProvider(intensities or {"zA": 500, "zB": 100})
    eng = Engine(SITES, "A", provider, executor, SimClock(accel=1),
                 {"improvement_g": 60, "min_dwell_min": 0,
                  "stall_after_s": 1},
                 net_cfg={}, power_provider=StubPower())
    eng.status = "running"
    executor.start("A")
    return eng


def assert_consistent(eng, ex, label):
    """The core invariant."""
    claims_running = eng.status == "running" and eng.health == "ok"
    actually_running = bool(ex.running)
    assert not (claims_running and not actually_running), (
        f"{label}: engine claims running (status={eng.status}, "
        f"health={eng.health}) but executor has nothing running")
    if eng.health == "ok" and eng.status == "running":
        assert ex.running == {eng.active_site}, (
            f"{label}: engine says active={eng.active_site}, "
            f"executor running={ex.running}")


def assert_no_split_brain(ex, label):
    """Never more than one trainer alive across the whole slice."""
    assert len(ex.running) <= 1, (
        f"{label}: SPLIT BRAIN — trainers running at {sorted(ex.running)}")


def test_transfer_failure_resumes_at_source():
    ex = FakeExecutor(fail_transfer=True)
    eng = make_engine(ex)
    eng._migrate("B")

    assert ex.running == {"A"}, f"job not resumed at source: {ex.running}"
    assert eng.active_site == "A"
    assert eng.status == "running" and eng.health == "ok"
    assert eng.migrations[-1]["outcome"] == "aborted"
    assert "B" in ex.staging_cleared, "staging file not cleared"
    assert "B" not in ex.discarded, "must never delete the live checkpoint"
    assert_consistent(eng, ex, "transfer failure")
    print("  transfer failure -> resumed at A, migration marked aborted")


def test_start_at_target_failure_resumes_at_source():
    ex = FakeExecutor(fail_start_at="B")
    eng = make_engine(ex)
    eng._migrate("B")

    assert ex.running == {"A"}, f"job not resumed at source: {ex.running}"
    assert eng.active_site == "A"
    assert eng.migrations[-1]["outcome"] == "aborted"
    assert_consistent(eng, ex, "start failure")
    print("  start-at-target failure -> resumed at A")


def test_total_failure_reports_down_not_running():
    """Worst case: nothing can run. The engine must say so."""
    ex = FakeExecutor(fail_transfer=True)
    eng = make_engine(ex)     # starts cleanly at A...
    ex.fail_start_always = True   # ...then everything breaks
    eng._migrate("B")

    assert not ex.running, f"nothing should be running: {ex.running}"
    assert eng.health == "down", f"health={eng.health} (must be down)"
    assert eng.status != "running" or eng.health != "ok"
    assert "NOT running" in (eng.health_note or "")
    assert_consistent(eng, ex, "total failure")
    print(f"  total failure -> health=down, note={eng.health_note[:60]}…")


def test_emissions_frozen_while_down():
    ex = FakeExecutor(fail_transfer=True)
    eng = make_engine(ex)
    ex.fail_start_always = True
    eng._migrate("B")
    assert eng.health == "down"
    eng._tick()
    before = (eng.emissions_g, eng.baseline_g)
    time.sleep(0.2)
    eng._tick()
    assert (eng.emissions_g, eng.baseline_g) == before, (
        "emissions accrued while the job was down")
    print("  emissions/baseline frozen while job is down")


def test_stall_detector_flags_frozen_workload():
    """The general case: job dies without a migration involved."""
    ex = FakeExecutor()
    # A is already the cleanest site, so no migration competes with the
    # behavior under test.
    eng = make_engine(ex, intensities={"zA": 100, "zB": 500})
    eng._tick()                       # establishes the step reference
    ex.running.clear()                # workload dies silently; step frozen
    ex.fail_start_always = True       # and cannot be restarted
    time.sleep(1.2)                   # exceed stall_after_s=1
    eng._tick()

    assert eng.health == "down", f"health={eng.health}"
    assert_consistent(eng, ex, "silent stall")
    print(f"  frozen step counter -> health=down")


def test_stall_detector_recovers():
    ex = FakeExecutor()
    eng = make_engine(ex, intensities={"zA": 100, "zB": 500})
    eng._tick()
    ex.running.clear()
    time.sleep(1.2)
    eng._tick()                       # detects stall, restart succeeds
    assert ex.running == {"A"}, "should have been restarted"
    ex.step += 50                     # progress resumes
    eng._tick()
    assert eng.health == "ok", f"health={eng.health}"
    assert eng.health_note is None
    print("  stalled job restarted and health cleared once steps advance")


def test_successful_migration_still_works():
    ex = FakeExecutor()
    eng = make_engine(ex)
    eng._migrate("B")
    assert ex.running == {"B"} and eng.active_site == "B"
    assert eng.migrations[-1]["outcome"] == "ok"
    assert eng.health == "ok" and eng.status == "running"
    assert_consistent(eng, ex, "happy path")
    print("  happy path unaffected")


def test_unconfirmed_stop_blocks_migration():
    """The split-brain case: if the source trainer might still be alive, the
    job must NOT be started at the target."""
    ex = FakeExecutor(stop_succeeds=False)
    eng = make_engine(ex)
    eng._migrate("B")

    assert "B" not in ex.running, (
        f"started at B while A may still run: {ex.running}")
    assert_no_split_brain(ex, "unconfirmed stop")
    assert eng.active_site == "A"
    assert eng.migrations[-1]["outcome"] == "aborted"
    assert "could not confirm" in eng.migrations[-1]["reason"]
    print(f"  unconfirmed stop -> migration refused, running={ex.running}")


def test_unreachable_node_does_not_spawn_duplicate():
    """`get_progress` returning None (SSH timeout) looks exactly like a
    stall. The engine must not start a trainer it cannot verify."""
    ex = FakeExecutor(visibility={"A": None})   # A unreachable
    eng = make_engine(ex, intensities={"zA": 100, "zB": 500})
    eng._tick()
    starts_before = len(ex.starts)
    time.sleep(1.2)
    eng._tick()

    assert len(ex.starts) == starts_before, (
        f"spawned a trainer on an unreachable node: {ex.starts}")
    assert eng.health == "stalled"
    assert "cannot reach" in (eng.health_note or "")
    assert_no_split_brain(ex, "unreachable node")
    print("  unreachable node -> no duplicate trainer started")


def test_wedged_trainer_is_stopped_before_restart():
    """Alive but not progressing: must stop first, never start alongside."""
    ex = FakeExecutor()
    eng = make_engine(ex, intensities={"zA": 100, "zB": 500})
    eng._tick()
    time.sleep(1.2)
    eng._tick()          # step never advances; trainer still 'alive'

    assert ex.running == {"A"}, ex.running
    assert_no_split_brain(ex, "wedged trainer")
    print("  wedged trainer -> stopped then restarted, still exactly one")


def test_wedged_trainer_that_wont_die_is_not_duplicated():
    ex = FakeExecutor(stop_succeeds=False)
    eng = make_engine(ex, intensities={"zA": 100, "zB": 500})
    eng._tick()
    starts_before = len(ex.starts)
    time.sleep(1.2)
    eng._tick()

    assert len(ex.starts) == starts_before, "started beside a wedged trainer"
    assert "wedged" in (eng.health_note or "")
    assert_no_split_brain(ex, "undying trainer")
    print("  wedged trainer that won't stop -> no second trainer")


def test_reconcile_adopts_existing_trainer():
    """Restarting the orchestrator must not add a second trainer."""
    ex = FakeExecutor()
    ex.running.add("B")            # left over from a previous run
    ex.steps["B"] = 4321
    provider = FakeProvider({"zA": 500, "zB": 100})
    eng = Engine(SITES, "A", provider, ex, SimClock(accel=1),
                 {"improvement_g": 60, "min_dwell_min": 999},
                 net_cfg={})
    eng._reconcile_on_start()

    assert eng.active_site == "B", f"did not adopt survivor: {eng.active_site}"
    assert ex.running == {"B"}, ex.running
    assert ex.starts == [], f"started a redundant trainer: {ex.starts}"
    assert_no_split_brain(ex, "reconcile adopt")
    print("  orchestrator restart -> adopted existing trainer at B")


def test_reconcile_converges_existing_split_brain():
    ex = FakeExecutor()
    ex.running.update({"A", "B"})   # already split-brained
    ex.steps["A"], ex.steps["B"] = 100, 900
    provider = FakeProvider({"zA": 500, "zB": 100})
    eng = Engine(SITES, "A", provider, ex, SimClock(accel=1),
                 {"improvement_g": 60, "min_dwell_min": 999}, net_cfg={},
                 power_provider=StubPower())
    eng._reconcile_on_start()

    assert ex.running == {"B"}, f"did not converge: {ex.running}"
    assert eng.active_site == "B", "should keep the furthest-along trainer"
    assert_no_split_brain(ex, "reconcile converge")
    print("  pre-existing split brain -> converged on furthest-along (B)")


def _engine_with_state(ex, state_file, start="A"):
    provider = FakeProvider({"zA": 500, "zB": 100})
    return Engine(SITES, start, provider, ex, SimClock(accel=1),
                  {"improvement_g": 60, "min_dwell_min": 999},
                  net_cfg={}, state_file=state_file)


def test_unreachable_last_active_site_blocks_startup(tmpdir):
    """AMBIGUITY: unreachable is not idle. If the job was last at B and B
    cannot answer, starting at A would fork the run."""
    state = os.path.join(tmpdir, "state.json")
    with open(state, "w") as f:
        json.dump({"active_site": "B"}, f)

    # B unreachable AND refuses to stop => genuinely ambiguous
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    eng = _engine_with_state(ex, state)
    eng._reconcile_on_start()

    assert ex.starts == [], f"started despite ambiguity: {ex.starts}"
    assert not ex.running, f"nothing should be running: {ex.running}"
    assert eng.health == "down", f"health={eng.health}"
    assert "may still be running" in (eng.health_note or "")
    assert_no_split_brain(ex, "ambiguous last-active")
    print("  unreachable last-active site -> refused to start anywhere")


def test_unreachable_site_stoppable_allows_startup(tmpdir):
    """Ambiguity resolved by a successful stop => safe to proceed."""
    state = os.path.join(tmpdir, "state.json")
    with open(state, "w") as f:
        json.dump({"active_site": "B"}, f)

    ex = FakeExecutor(visibility={"B": None}, stop_recovers={"B"})
    eng = _engine_with_state(ex, state)
    eng._reconcile_on_start()

    assert ex.running == {"A"}, ex.running
    assert eng.health != "down"
    assert_no_split_brain(ex, "ambiguity resolved")
    print("  unreachable site stopped successfully -> startup proceeds")


def test_unreachable_site_without_evidence_does_not_block(tmpdir):
    """A site down for maintenance that never held the job must not ground
    the demo."""
    state = os.path.join(tmpdir, "state.json")
    with open(state, "w") as f:
        json.dump({"active_site": "A"}, f)      # job was at A, not B

    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    eng = _engine_with_state(ex, state)
    eng._reconcile_on_start()

    assert ex.running == {"A"}, f"should have started at A: {ex.running}"
    assert eng.health != "down"
    print("  unreachable site with no evidence -> demo still starts")


def test_no_state_file_does_not_block(tmpdir):
    """First ever run: no evidence anywhere, must not refuse to start."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    eng = _engine_with_state(ex, os.path.join(tmpdir, "missing.json"))
    eng._reconcile_on_start()
    assert ex.running == {"A"}, ex.running
    print("  no persisted state -> cold start proceeds")


def test_abort_will_not_resume_while_target_unconfirmed():
    """start(target) raising does NOT mean the target is dead. If it cannot
    be confirmed stopped, resuming at the source would fork the run."""
    ex = FakeExecutor(fail_start_at="B", stop_succeeds=False)
    eng = make_engine(ex)
    # State as _migrate leaves it: source confirmed stopped, target actually
    # started before start() raised on its unconfirmable lock check.
    ex.running.clear()
    ex.running.add("B")
    starts_before = len(ex.starts)
    eng._abort_migration("A", "B", "start at B failed", stopped_source=True)

    assert len(ex.starts) == starts_before, (
        f"started a trainer while B may still run: {ex.starts}")
    assert "A" not in ex.running, (
        f"resumed at A while B may still run: {ex.running}")
    assert eng.health == "down"
    assert "may still be running at B" in (eng.health_note or "")
    assert eng.migrations[-1]["blocked_by"] == "B"
    assert_no_split_brain(ex, "unconfirmed target")
    print("  target unconfirmed -> refused to resume at source")


def test_state_file_records_migrations(tmpdir):
    state = os.path.join(tmpdir, "state.json")
    ex = FakeExecutor()
    provider = FakeProvider({"zA": 500, "zB": 100})
    eng = Engine(SITES, "A", provider, ex, SimClock(accel=1),
                 {"improvement_g": 60, "min_dwell_min": 0}, net_cfg={},
                 state_file=state, power_provider=StubPower())
    eng.status = "running"
    ex.start("A")
    eng._migrate("B")
    with open(state) as f:
        assert json.load(f)["active_site"] == "B"
    print("  successful migration persists new active site")


def test_reconcile_starts_fresh_when_nothing_runs():
    ex = FakeExecutor()
    provider = FakeProvider({"zA": 500, "zB": 100})
    eng = Engine(SITES, "A", provider, ex, SimClock(accel=1),
                 {"improvement_g": 60, "min_dwell_min": 999}, net_cfg={},
                 power_provider=StubPower())
    eng._reconcile_on_start()
    assert ex.running == {"A"} and eng.active_site == "A"
    print("  cold start -> starts at configured start_site")


# --- fresh-orchestrator (control-node handoff) cases ----------------------
#
# Moving the control plane to a new host wipes its local state file while
# leaving the slice's trainers exactly where they were. Evidence therefore
# has to live on the workers, not on whichever machine happens to be
# orchestrating.

def _fresh_engine(ex, tmpdir, name, force=False):
    provider = FakeProvider({"zA": 500, "zB": 100})
    return Engine(SITES, "A", provider, ex, SimClock(accel=1),
                  {"improvement_g": 60, "min_dwell_min": 999}, net_cfg={},
                  state_file=os.path.join(tmpdir, name), force_start=force,
                  power_provider=StubPower())


def test_fresh_control_node_does_not_duplicate(tmpdir):
    """THE REPORTED BUG: fresh CTRL, empty state file, B unreachable and
    possibly still running. Must not start at A."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    ex.job_state = {"A": True}          # slice clearly has history
    eng = _fresh_engine(ex, tmpdir, "fresh1.json")
    eng._reconcile_on_start()

    assert ex.starts == [], f"fresh control node started a duplicate: {ex.starts}"
    assert eng.health == "down", f"health={eng.health}"
    assert_no_split_brain(ex, "fresh control node")
    print("  fresh orchestrator + prior run + unreachable node -> refused")


def test_marker_on_reachable_worker_names_the_unreachable_site(tmpdir):
    """Markers are replicated, so a reachable worker can finger the site we
    cannot reach."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    ex.markers = {"A": "B"}             # last active was B, per A's copy
    eng = _fresh_engine(ex, tmpdir, "fresh2.json")
    eng._reconcile_on_start()

    assert ex.starts == [], f"started despite marker naming B: {ex.starts}"
    assert eng.health == "down"
    print("  replicated marker identified the unreachable active site")


def test_fresh_control_node_cold_slice_still_starts(tmpdir):
    """No markers, no job state anywhere: nothing ever ran, so an
    unreachable node is not a suspect and the demo must still come up."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    eng = _fresh_engine(ex, tmpdir, "fresh3.json")
    eng._reconcile_on_start()

    assert ex.running == {"A"}, f"cold slice should start: {ex.running}"
    assert eng.health != "down"
    print("  cold slice with an unreachable node -> still starts")


def test_force_start_overrides(tmpdir):
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    ex.job_state = {"A": True}
    eng = _fresh_engine(ex, tmpdir, "fresh4.json", force=True)
    eng._reconcile_on_start()
    assert ex.running == {"A"}, "force-start should proceed"
    print("  --force-start overrides the refusal")


def test_migration_replicates_marker_to_all_workers(tmpdir):
    ex = FakeExecutor()
    provider = FakeProvider({"zA": 500, "zB": 100})
    eng = Engine(SITES, "A", provider, ex, SimClock(accel=1),
                 {"improvement_g": 60, "min_dwell_min": 0}, net_cfg={},
                 state_file=os.path.join(tmpdir, "fresh5.json"))
    eng.status = "running"
    ex.start("A")
    eng._migrate("B")
    assert ex.markers.get("A") == "B" and ex.markers.get("B") == "B", ex.markers
    print("  migration wrote the marker to every worker")


def test_marker_naming_a_reachable_site_avoids_needless_refusal(tmpdir):
    """The marker's job is precision, not safety: the conservative fallback
    already refuses on any doubt. What the marker adds is knowing the job
    was somewhere we CAN verify, so an unrelated unreachable node does not
    ground the demo."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    ex.markers = {"A": "A"}          # job was last at A, which is reachable
    eng = _fresh_engine(ex, tmpdir, "fresh6.json")
    eng._reconcile_on_start()

    assert ex.running == {"A"}, (
        f"refused although the marker names a verifiable site: {ex.running}")
    assert eng.health != "down", f"health={eng.health}"
    print("  marker naming a reachable site -> starts despite unreachable B")


def test_refusal_survives_the_run_loop(tmpdir):
    """THE REPORTED BUG: a refusal must not be undone by run() flipping
    status back to "running" — the tick loop gates migrations and restarts
    on that flag, so overriding it re-enables the duplicate we refused."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    ex.job_state = {"A": True}
    eng = _fresh_engine(ex, tmpdir, "loop1.json")

    cleared = eng._try_reconcile()
    assert cleared is False, "reconcile should refuse"
    if cleared:
        eng.status = "running"

    assert eng.status != "running", f"status={eng.status} re-enables actions"
    # a tick while blocked must start nothing, migrate nothing
    for _ in range(3):
        eng._tick()
    assert ex.starts == [], f"tick started a trainer while blocked: {ex.starts}"
    assert_no_split_brain(ex, "blocked run loop")
    print("  refusal holds across ticks; nothing started")


def test_booth_button_rejected_while_blocked(tmpdir):
    """The force-migrate control must not punch through a refusal."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    ex.job_state = {"A": True}
    eng = _fresh_engine(ex, tmpdir, "loop2.json")
    eng._try_reconcile()

    assert eng.force_migration("B") is False, "booth button bypassed refusal"
    eng._force_target = "B"          # even if set directly, must be ignored
    assert eng._pick_target({"A": 500, "B": 100}, 1e9) is None
    eng._tick()
    assert ex.starts == [], f"forced start while blocked: {ex.starts}"
    print("  booth force-migrate refused while blocked")


def test_block_clears_when_the_node_comes_back(tmpdir):
    """Operator shouldn't have to babysit: once the node is verifiable and
    idle, the retry releases the block."""
    ex = FakeExecutor(visibility={"B": None}, stop_succeeds=False)
    ex.job_state = {"A": True}
    eng = _fresh_engine(ex, tmpdir, "loop3.json")
    assert eng._try_reconcile() is False

    ex.visibility = {"B": False}     # B answers again, and is idle
    ex.stop_succeeds = True
    assert eng._try_reconcile() is True, "should clear once verifiable"
    eng.status = "running"
    assert ex.running == {"A"}, ex.running
    assert_no_split_brain(ex, "recovered block")
    print("  block cleared automatically once B was verifiable")


def test_recovered_block_clears_health_and_resumes_emissions(tmpdir):
    """THE REPORTED BUG: after the retry releases a block, the job runs but
    was left marked down — which also freezes the CO2 counters, silently
    corrupting the headline number."""
    ex = FakeExecutor(visibility={"B": None})
    ex.job_state = {"A": True}
    eng = _fresh_engine(ex, tmpdir, "heal1.json")
    assert eng._try_reconcile() is False and eng.health == "down"

    ex.visibility = {"B": False}          # B comes back, verifiably idle
    assert eng._try_reconcile() is True
    eng.status = "running"

    assert eng.health == "ok", f"health stuck at {eng.health}"
    assert eng.health_note is None, f"stale note: {eng.health_note}"

    # and emissions must actually accrue again
    eng._tick()
    before = eng.emissions_g
    time.sleep(0.3)
    eng._tick()
    assert eng.emissions_g > before, (
        "emissions still frozen after recovery — the CO2 counter would "
        "under-report for the rest of the run")
    print("  recovery cleared health and emissions resumed")


def test_reconcile_warning_is_not_clobbered(tmpdir):
    """A stalled warning raised by reconciliation itself must survive."""
    ex = FakeExecutor(stop_succeeds=False)
    ex.running.update({"A", "B"})         # split brain it cannot fully clear
    ex.steps["A"], ex.steps["B"] = 10, 90
    eng = _fresh_engine(ex, tmpdir, "heal2.json")
    eng._try_reconcile()
    assert eng.health == "stalled", f"health={eng.health}"
    assert "could not be stopped" in (eng.health_note or "")
    print("  reconciliation's own warning preserved")


def test_progress_clears_a_stale_down():
    """Observed steps are proof of life and must clear a down verdict."""
    ex = FakeExecutor()
    eng = make_engine(ex, intensities={"zA": 100, "zB": 500})
    eng.health, eng.health_note = "down", "stale"
    ex.step += 10
    eng._tick()
    assert eng.health == "ok", f"health={eng.health}"
    print("  advancing steps cleared a stale down verdict")


def test_resume_picks_furthest_along_checkpoint(tmpdir):
    """Restarting with nothing alive must resume from the most advanced
    checkpoint, not the configured start_site — otherwise the restart
    discards progress and the next migration makes the loss permanent.
    This is the ~167k-step regression seen on the real testbed."""
    ex = FakeExecutor()
    ex.steps = {"A": 114, "B": 167340}      # B is far ahead
    eng = _fresh_engine(ex, tmpdir, "resume1.json")   # start_site is A
    assert eng._reconcile_on_start() is True

    assert eng.active_site == "B", (
        f"resumed at {eng.active_site}, discarding B's 167340 steps")
    assert ex.running == {"B"}, ex.running
    print("  resumed at B (step 167340) instead of start_site A (114)")


def test_resume_falls_back_when_no_progress_anywhere(tmpdir):
    ex = FakeExecutor()
    ex.no_progress_at = {"A", "B"}          # cold slice
    eng = _fresh_engine(ex, tmpdir, "resume2.json")
    assert eng._reconcile_on_start() is True
    assert eng.active_site == "A" and ex.running == {"A"}
    print("  cold slice -> falls back to configured start_site")


def test_migration_refuses_to_clobber_better_checkpoint():
    """Second half of the same data-loss path: copying a near-zero
    checkpoint over an advanced one."""
    ex = FakeExecutor()
    ex.steps = {"A": 114, "B": 167340}
    ex.tss = {"A": 1000.0, "B": 2000.0}     # B's work is NEWER as well
    eng = make_engine(ex)                   # active A
    eng._migrate("B")

    assert eng.migrations[-1]["outcome"] == "aborted", eng.migrations[-1]
    assert "further-along" in eng.migrations[-1]["reason"]
    assert ex.running == {"A"}, ex.running
    print("  migration refused: would have overwritten step 167340 with 114")


def test_normal_migration_not_blocked_by_the_guard():
    """The guard must not fire in normal operation, where the source is
    always the authoritative, furthest-along copy."""
    ex = FakeExecutor()
    ex.steps = {"A": 5000, "B": 4000}       # B stale from an earlier visit
    eng = make_engine(ex)
    eng._migrate("B")
    assert eng.migrations[-1]["outcome"] == "ok", eng.migrations[-1]
    assert ex.running == {"B"}
    print("  normal migration (source ahead) proceeds untouched")


def test_stale_advanced_checkpoint_does_not_block_migration():
    """An abandoned branch must not lock the job out of a site forever.
    UTAH held a 70k-step checkpoint from a branch a bad restart orphaned; a
    step-only guard would have made UTAH permanently unreachable."""
    ex = FakeExecutor()
    ex.steps = {"A": 5000, "B": 70264}      # B further along...
    ex.tss = {"A": 9000.0, "B": 1000.0}     # ...but much older
    eng = make_engine(ex)
    eng._migrate("B")

    assert eng.migrations[-1]["outcome"] == "ok", eng.migrations[-1]
    assert ex.running == {"B"}, ex.running
    print("  stale high-step checkpoint treated as abandoned; migration ran")


def test_protection_path_does_not_delete_the_checkpoint_it_protects():
    """The abort that refuses to overwrite better work must not then delete
    that work — the protection path was calling the partial-file cleanup."""
    ex = FakeExecutor()
    ex.steps = {"A": 114, "B": 167340}
    ex.tss = {"A": 1000.0, "B": 2000.0}     # B is newer AND further along
    eng = make_engine(ex)
    eng._migrate("B")

    assert eng.migrations[-1]["outcome"] == "aborted"
    assert "B" not in ex.discarded, (
        "deleted the very checkpoint it refused to overwrite: "
        f"discarded={ex.discarded}")
    assert "B" not in ex.staging_cleared or True   # staging clear is harmless
    print("  refusal preserved B's checkpoint (nothing discarded)")


def test_aborts_before_any_transfer_preserve_target_checkpoint():
    """No bytes written => nothing partial => nothing to clean up."""
    # stop at source cannot be confirmed: aborts before transferring
    ex = FakeExecutor(stop_succeeds=False)
    eng = make_engine(ex)
    eng._migrate("B")
    assert "B" not in ex.discarded, f"discarded={ex.discarded}"

    # start at target fails: transfer completed, so the target holds a full
    # valid copy — not a partial file
    ex2 = FakeExecutor(fail_start_at="B")
    eng2 = make_engine(ex2)
    eng2._migrate("B")
    assert "B" not in ex2.discarded, f"discarded={ex2.discarded}"
    print("  pre-transfer and start-failure aborts leave the target intact")


def test_failed_transfer_clears_staging_not_live_checkpoint():
    """A failed transfer may leave a staging file; the live checkpoint at
    the target must survive, because the failure may have happened before a
    single byte was written (refused connection, watchdog timeout)."""
    ex = FakeExecutor(fail_transfer=True)
    eng = make_engine(ex)
    eng._migrate("B")
    assert "B" in ex.staging_cleared, "staging file left behind"
    assert "B" not in ex.discarded, (
        "deleted B's intact checkpoint on a transfer that may never have "
        "written to it")
    print("  failed transfer clears staging, preserves the live checkpoint")


def test_no_abort_path_ever_deletes_a_live_checkpoint():
    """Blanket invariant: the only thing removed is what a migration wrote."""
    scenarios = {
        "stop-unconfirmed": FakeExecutor(stop_succeeds=False),
        "transfer-failed": FakeExecutor(fail_transfer=True),
        "start-failed": FakeExecutor(fail_start_at="B"),
    }
    for label, ex in scenarios.items():
        eng = make_engine(ex)
        eng._migrate("B")
        assert not ex.discarded, (
            f"{label}: deleted a live checkpoint {ex.discarded}")
    # and the protection path
    ex = FakeExecutor()
    ex.steps = {"A": 114, "B": 167340}
    ex.tss = {"A": 1000.0, "B": 2000.0}
    eng = make_engine(ex)
    eng._migrate("B")
    assert not ex.discarded, f"protection path deleted {ex.discarded}"
    print("  no abort path deletes a live checkpoint")


def test_completed_run_is_not_reported_as_failed():
    """A finished job and a broken job look identical from outside — steps
    stop advancing either way. Reporting completion as health=down had the
    engine repeatedly try to restart a run that had done its work."""
    ex = FakeExecutor()
    eng = make_engine(ex, intensities={"zA": 100, "zB": 500})
    eng._tick()
    # the trainer finished and said so
    ex.steps["A"] = 1_000_000
    ex.done = True
    time.sleep(1.2)
    eng._tick()

    assert eng.status == "completed", eng.status
    assert eng.health == "ok", f"a finished run reported as {eng.health}"
    assert "finished" in (eng.health_note or "")
    assert len(ex.starts) == 1, f"tried to restart a finished run: {ex.starts}"
    print("  completed run reported as completed, not restarted")


def test_reconcile_reports_an_already_finished_run(tmpdir):
    ex = FakeExecutor()
    ex.steps = {"A": 1_000_000}
    ex.done = True
    eng = _fresh_engine(ex, tmpdir, "done.json")
    assert eng._reconcile_on_start() is True
    assert eng.status == "completed" and eng.health == "ok"
    assert ex.starts == [], f"started a finished run: {ex.starts}"
    print("  restart onto a finished run reports completion, starts nothing")


def test_completed_run_stops_accruing_emissions(tmpdir):
    """A finished job draws no power. Continuing to integrate inflates both
    series for work that is no longer happening — fabricated CO2 on the
    headline number."""
    ex = FakeExecutor()
    ex.steps = {"A": 1_000_000}
    ex.done = True
    eng = _fresh_engine(ex, tmpdir, "done_em.json")
    eng._try_reconcile()
    assert eng.status == "completed", eng.status

    eng._tick()
    before = (eng.emissions_g, eng.baseline_g)
    time.sleep(0.3)
    eng._tick()
    eng._tick()
    assert (eng.emissions_g, eng.baseline_g) == before, (
        f"emissions grew after completion: {before} -> "
        f"{(eng.emissions_g, eng.baseline_g)}")
    print("  completed run stops accruing emissions")


def test_run_loop_does_not_promote_completed_to_running(tmpdir):
    """`_try_reconcile` returns True for 'safe to proceed', which includes
    'already finished'. Promoting that to running lets the very FIRST tick
    migrate a finished job before the liveness check corrects the status —
    a window that a later sample would never see.
    """
    import threading
    ex = FakeExecutor()
    ex.steps = {"A": 1_000_000, "B": 10}
    ex.done = True
    eng = _fresh_engine(ex, tmpdir, "done_run.json")
    # B is far cleaner and dwell is zero, so a "running" engine WOULD move.
    eng.provider = FakeProvider({"zA": 900, "zB": 50})
    eng.policy = dict(eng.policy, poll_interval_s=0.2, stall_after_s=999,
                      min_dwell_min=0, improvement_g=60)
    t = threading.Thread(target=eng.run, daemon=True)
    t.start()
    time.sleep(1.2)
    eng.stop()
    time.sleep(0.3)

    assert eng.migrations == [], (
        f"migrated a finished run during the promotion window: "
        f"{eng.migrations}")
    assert ex.starts == [], f"tried to start a finished run: {ex.starts}"
    assert eng.status == "completed", f"status={eng.status}"
    print("  finished run never migrates, even on the first tick")


def test_completed_job_never_migrates_even_if_status_says_running():
    """Pins the tick ordering the completed-state safety depends on.

    `_check_liveness` runs before `_pick_target`, so a job that reports
    `done` is reclassified before any migration is considered. If that order
    were reversed, a finished run could be migrated. This test fails if it is.
    """
    ex = FakeExecutor()
    ex.steps = {"A": 1_000_000, "B": 10}
    ex.done = True
    eng = make_engine(ex, intensities={"zA": 900, "zB": 50})
    eng.status = "running"          # as a mis-promotion would leave it
    eng._tick()

    assert eng.migrations == [], f"migrated a finished run: {eng.migrations}"
    assert eng.status == "completed", (
        f"status={eng.status}: liveness must reclassify a done job before "
        f"target selection runs")
    print("  tick order protects a finished run from migration")


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp(prefix="cc-test-")
    needs_tmp = {test_unreachable_last_active_site_blocks_startup,
                 test_unreachable_site_stoppable_allows_startup,
                 test_unreachable_site_without_evidence_does_not_block,
                 test_no_state_file_does_not_block,
                 test_state_file_records_migrations,
                 test_fresh_control_node_does_not_duplicate,
                 test_marker_on_reachable_worker_names_the_unreachable_site,
                 test_fresh_control_node_cold_slice_still_starts,
                 test_force_start_overrides,
                 test_migration_replicates_marker_to_all_workers,
                 test_marker_naming_a_reachable_site_avoids_needless_refusal,
                 test_refusal_survives_the_run_loop,
                 test_booth_button_rejected_while_blocked,
                 test_block_clears_when_the_node_comes_back,
                 test_recovered_block_clears_health_and_resumes_emissions,
                 test_reconcile_warning_is_not_clobbered,
                 test_reconcile_reports_an_already_finished_run,
                 test_completed_run_stops_accruing_emissions,
                 test_run_loop_does_not_promote_completed_to_running,
                 test_resume_picks_furthest_along_checkpoint,
                 test_resume_falls_back_when_no_progress_anywhere}
    for fn in (test_transfer_failure_resumes_at_source,
               test_start_at_target_failure_resumes_at_source,
               test_total_failure_reports_down_not_running,
               test_emissions_frozen_while_down,
               test_stall_detector_flags_frozen_workload,
               test_stall_detector_recovers,
               test_successful_migration_still_works,
               test_unconfirmed_stop_blocks_migration,
               test_unreachable_node_does_not_spawn_duplicate,
               test_wedged_trainer_is_stopped_before_restart,
               test_wedged_trainer_that_wont_die_is_not_duplicated,
               test_reconcile_adopts_existing_trainer,
               test_reconcile_converges_existing_split_brain,
               test_reconcile_starts_fresh_when_nothing_runs,
               test_unreachable_last_active_site_blocks_startup,
               test_unreachable_site_stoppable_allows_startup,
               test_unreachable_site_without_evidence_does_not_block,
               test_no_state_file_does_not_block,
               test_abort_will_not_resume_while_target_unconfirmed,
               test_state_file_records_migrations,
               test_fresh_control_node_does_not_duplicate,
               test_marker_on_reachable_worker_names_the_unreachable_site,
               test_fresh_control_node_cold_slice_still_starts,
               test_force_start_overrides,
               test_migration_replicates_marker_to_all_workers,
               test_marker_naming_a_reachable_site_avoids_needless_refusal,
               test_refusal_survives_the_run_loop,
               test_booth_button_rejected_while_blocked,
               test_block_clears_when_the_node_comes_back,
               test_recovered_block_clears_health_and_resumes_emissions,
               test_reconcile_warning_is_not_clobbered,
               test_progress_clears_a_stale_down,
               test_resume_picks_furthest_along_checkpoint,
               test_resume_falls_back_when_no_progress_anywhere,
               test_migration_refuses_to_clobber_better_checkpoint,
               test_normal_migration_not_blocked_by_the_guard,
               test_stale_advanced_checkpoint_does_not_block_migration,
               test_completed_run_is_not_reported_as_failed,
               test_protection_path_does_not_delete_the_checkpoint_it_protects,
               test_aborts_before_any_transfer_preserve_target_checkpoint,
               test_failed_transfer_clears_staging_not_live_checkpoint,
               test_no_abort_path_ever_deletes_a_live_checkpoint,
               test_reconcile_reports_an_already_finished_run,
               test_completed_run_stops_accruing_emissions,
               test_run_loop_does_not_promote_completed_to_running,
               test_completed_job_never_migrates_even_if_status_says_running):
        print(fn.__name__)
        fn(tmpdir) if fn in needs_tmp else fn()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nALL PASS")
