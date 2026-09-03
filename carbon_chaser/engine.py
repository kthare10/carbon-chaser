"""Orchestration engine: watch carbon intensity, migrate the job.

Policy: every tick, read all site intensities. If the cleanest site beats
the active one by `improvement_g` gCO2/kWh AND we've dwelled at least
`min_dwell_min` sim-minutes since the last move, migrate (stop -> transfer
checkpoint -> start). Hysteresis + dwell prevent flapping.

The engine also keeps the score that makes the demo legible: cumulative
emissions of the chasing job vs. a baseline that never leaves the start
site, integrated from the same intensity series.
"""

import json
import os
import threading
import time
from collections import deque
from typing import Dict, Optional

from .carbon import CarbonProvider
from .clock import Clock
from .executor import Executor


class Engine:
    def __init__(self, sites: Dict[str, dict], start_site: str,
                 provider: CarbonProvider, executor: Executor, clock: Clock,
                 policy: dict, managed_sites: Optional[list] = None,
                 net_model=None, net_cfg: Optional[dict] = None,
                 state_file: Optional[str] = None,
                 force_start: bool = False,
                 power_provider=None):
        self.sites = sites
        self.provider = provider
        self.executor = executor
        self.clock = clock
        self.policy = policy
        # Sites the executor can actually run on (fabric mode may manage a
        # subset of the map).
        self.managed = managed_sites or list(sites)
        self.state_file = state_file        # persists where the job was
        self.force_start = force_start      # operator override, booth escape
        # Power is the second factor in the CO2 arithmetic; like carbon it
        # carries provenance so a modelled watt never reads as a measured one.
        self.power = power_provider
        self.net_model = net_model          # FabricNetworkModel or None
        self.net_cfg = net_cfg or {}
        self._net_est: Dict[str, dict] = {} # candidate -> transfer estimate
        self._net_note: Optional[str] = None
        # EWMA of measured dataplane throughput (bits/s); calibrates the
        # headroom-derived estimate. Only real dataplane transfers feed it.
        # Downtime is modelled in three measured parts, because they scale
        # differently and conflating them mispredicts the sizes that matter:
        #   downtime = orchestration + setup + bytes/bandwidth
        # orchestration: stop + start cost, independent of checkpoint size
        # setup:         per-transfer connection cost, size-independent
        # bandwidth:     the only size-dependent term
        self._measured_rate_bps: Optional[float] = None
        self._measured_setup_s: Optional[float] = None
        self._measured_orchestration_s: Optional[float] = None

        self.active_site = start_site
        self.baseline_site = start_site
        self.status = "starting"          # starting | running | migrating | stopped
        self.health = "ok"                # ok | stalled | down  (THE JOB)
        self.health_note: Optional[str] = None
        # Data quality is a DIFFERENT question from whether the job runs.
        # Overloading `health` for both meant advancing training steps —
        # which prove only that the job is alive — silently cleared a
        # carbon-feed outage, and the run carried on reporting itself
        # healthy while integrating emissions from stale readings.
        self.carbon_status = "ok"         # ok | stale | unavailable
        self.carbon_note: Optional[str] = None
        self._intensity_seen: Dict[str, float] = {}   # site -> wall clock
        self._fresh_sites: set = set()
        # (step, wall-clock seen at) for the stall detector
        self._stall_ref: Optional[tuple] = None
        self.migrations = []              # [{from,to,sim_t,bytes,downtime_s}]
        self.last_move_sim_t = clock.now()

        self.emissions_g = 0.0            # chasing job
        self.baseline_g = 0.0             # never-moves job
        self.history = deque(maxlen=2000) # per-tick snapshots for charts

        self._last_tick_sim_t: Optional[float] = None
        self._intensities: Dict[str, float] = {}
        self._progress: Optional[dict] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._force_target: Optional[str] = None

    # -- public API (dashboard) -------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return {
                "sim_time_s": self.clock.now(),
                "sim_hour": round(self.clock.hour_of_day(), 2),
                "accel": getattr(self.clock, "accel", 1.0),
                "status": self.status,
                "health": self.health,
                "health_note": self.health_note,
                "active_site": self.active_site,
                "managed_sites": self.managed,
                # Unknown is null, never 0. Reporting a missing reading as
                # zero made an outage render as the CLEANEST possible grid
                # (0 gCO2/kWh, lightest colour on the map) — the most
                # flattering possible lie about data we do not have.
                "sites": {
                    name: {
                        **{k: cfg[k] for k in ("display", "lat", "lon", "zone")},
                        "intensity": (round(self._intensities[name], 1)
                                      if name in self._intensities else None),
                        "data": ("fresh" if name in self._fresh_sites
                                 else "stale" if name in self._intensities
                                 else "missing"),
                        "managed": name in self.managed,
                    }
                    for name, cfg in self.sites.items()
                },
                "emissions_g": round(self.emissions_g, 1),
                "baseline_g": round(self.baseline_g, 1),
                "saved_g": round(self.baseline_g - self.emissions_g, 1),
                "migrations": self.migrations[-20:],
                "progress": self._progress,
                "history": list(self.history),
                "carbon_source": self._describe_provider(),
                "power_source": self._describe_power(),
                "carbon_status": self.carbon_status,
                "carbon_note": self.carbon_note,
                "net_estimates": self._net_est,
                "net_note": self._net_note,
                "measured_rate_gbps": (round(self._measured_rate_bps / 1e9, 2)
                                       if self._measured_rate_bps else None),
                "measured_setup_s": (round(self._measured_setup_s, 2)
                                     if self._measured_setup_s else None),
                "measured_orchestration_s": (
                    round(self._measured_orchestration_s, 2)
                    if self._measured_orchestration_s else None),
            }

    def _power_kw(self, site: str) -> Optional[float]:
        """Measured watts, or None. There is no assumed value to fall back
        on: an invented power figure is exactly the kind of number this
        system refuses to produce."""
        if self.power is None:
            return None
        return self.power.get_kw(site)

    def _describe_power(self) -> dict:
        if self.power is None:
            return {"kind": "unavailable",
                    "detail": "no power provider configured",
                    "measured": False}
        try:
            return self.power.describe()
        except Exception:
            return {"kind": "unknown", "detail": "", "measured": False}

    def _describe_provider(self) -> dict:
        """Provenance of the carbon numbers, passed straight through so the
        UI can never imply measurement the data doesn't support."""
        try:
            return self.provider.describe()
        except Exception:
            return {"kind": "unknown", "detail": "", "injected_events": 0}

    def force_migration(self, site: str) -> bool:
        if self.status != "running":
            return False            # blocked or mid-migration: not now
        if site not in self.managed or site == self.active_site:
            return False
        self._force_target = site
        return True

    # -- engine loop -------------------------------------------------------

    def run(self):
        # Only go to "running" if reconciliation actually cleared us to run.
        # Forcing it here would silently undo a refusal: the tick loop gates
        # migrations and restarts on status, so flipping it back on would
        # let the very duplicate we just refused get started a tick later.
        if self._try_reconcile() and self.status != "completed":
            # Reconciliation returns True for "safe to proceed", which
            # includes "the run was already finished". Defence in depth: with
            # the current tick order (_check_liveness before _pick_target) a
            # promoted status self-corrects before anything acts on it, so
            # this guard is redundant TODAY — it exists so a future reorder
            # cannot resurrect the done-vs-dead confusion. The behavioural
            # invariant is pinned by test_completed_job_never_migrates.
            self.status = "running"

        poll = self.policy.get("poll_interval_s", 2)
        retry_every = self.policy.get("reconcile_retry_s", 30)
        last_retry = time.time()

        while not self._stop.is_set():
            try:
                # Safe while blocked: _tick starts nothing unless status is
                # "running" — it just keeps intensities and the dashboard live.
                self._tick()
                if self.status != "running" and self.health == "down":
                    if time.time() - last_retry >= retry_every:
                        last_retry = time.time()
                        # Cheap to retry, and it lets a node that comes back
                        # release the block without operator intervention.
                        if self._try_reconcile():
                            self.status = "running"
                            print("[engine] reconciliation now clear; "
                                  "resuming", flush=True)
            except Exception as e:  # keep the demo alive on transient errors
                print(f"[engine] tick error: {e}", flush=True)
            self._stop.wait(poll)

    def _try_reconcile(self) -> bool:
        """True only when it is safe to run. Never raises.

        Clearing a stale "down" here is not cosmetic: emissions accounting
        freezes while health is down, so a recovered job that stays marked
        down would run with its CO2 counters stopped — the headline number
        would silently drift wrong. A `stalled` warning raised by
        reconciliation itself (e.g. a second trainer it could not stop) is
        left alone.
        """
        try:
            ok = self._reconcile_on_start()
            if ok and self.health == "down":
                with self._lock:
                    self.health, self.health_note = "ok", None
            return ok
        except Exception as e:
            self.status = "stopped"
            self.health = "down"
            self.health_note = f"could not start job at {self.active_site}: {e}"
            print(f"[engine] {self.health_note}", flush=True)
            return False

    def stop(self):
        self._stop.set()

    def _reconcile_on_start(self) -> bool:
        """Adopt whatever is already running before starting anything.

        Returns True only when it is safe to run; False means the caller
        must NOT flip status to "running" — the tick loop gates migrations
        and restarts on that flag, so overriding it re-enables exactly what
        this refusal exists to prevent.

        Restarting the orchestrator mid-demo must not add a trainer beside
        one left over from the previous run — that is split-brain across
        sites, with both advancing the same checkpoint lineage.

        The subtle case is a site that answers neither yes nor no. An
        unreachable node is NOT an idle node: if the previous run left its
        trainer there, starting elsewhere forks the run. So unknown state is
        resolved, not assumed — and it only blocks startup when there is
        actual evidence a trainer may be there (the persisted last-active
        site), so a site down for maintenance doesn't ground the demo.
        """
        alive, unknown = {}, []
        for site in self.managed:
            state = self.executor.is_running(site)
            if state is None:
                unknown.append(site)
            elif state:
                alive[site] = ((self.executor.get_progress(site) or {})
                               .get("step") or 0)

        last_active, prior_run = self._discover_history(unknown)
        if last_active in unknown:
            risky = [last_active]
        elif last_active is None and prior_run and unknown:
            # Something ran here before and we cannot tell where. Absence of
            # evidence is not evidence of absence — every silent node is a
            # suspect. (This is the case a fresh control node lands in: its
            # own state file is empty, but the slice has history.)
            risky = list(unknown)
            print(f"[engine] prior run detected but its location is unknown; "
                  f"treating all unreachable sites as suspect: "
                  f"{sorted(risky)}", flush=True)
        else:
            risky = []
        for site in list(risky):
            print(f"[engine] {site} is unreachable and was running the job "
                  f"last — trying to stop it before starting anywhere",
                  flush=True)
            try:
                if self.executor.stop(site):
                    risky.remove(site)
            except Exception as e:
                print(f"[engine] stop:{site} errored: {e}", flush=True)
        if unknown:
            print(f"[engine] unreachable at startup: {sorted(unknown)} "
                  f"(last active: {last_active})", flush=True)

        if risky and self.force_start:
            print(f"[engine] --force-start: proceeding despite unverified "
                  f"{sorted(risky)}; a duplicate trainer there would corrupt "
                  f"the run", flush=True)
            risky = []

        if risky:
            # Cannot prove exclusivity; starting anything risks a fork.
            self.status = "stopped"
            self.health = "down"
            self.health_note = (
                f"not starting: {', '.join(risky)} cannot be reached and may "
                f"still be running the job. Kill the trainer there (or bring "
                f"the node back), then restart. Override with --force-start "
                f"only if you know those nodes are idle.")
            print(f"[engine] {self.health_note}", flush=True)
            return False

        if not alive:
            # Nothing is running, so resume from the FURTHEST-ALONG
            # checkpoint rather than the configured start_site. Starting at
            # start_site discards however far the job had got — and the next
            # migration then copies that near-zero checkpoint over a good
            # one, so the loss is permanent. (Observed on the testbed: a
            # restart threw away ~167k steps this way.)
            resume = self._best_resume_site()
            done_at = (self.executor.get_progress(resume) or {}).get("done")
            if done_at:
                with self._lock:
                    self.active_site = resume
                    self.status = "completed"
                    self.health = "ok"
                    self.health_note = (
                        f"training run already finished at {resume}; nothing "
                        f"to restart")
                print(f"[engine] {self.health_note}", flush=True)
                return True
            with self._lock:
                self.active_site = resume
            self.executor.start(resume)
            self._persist_last_active()
            self._replicate_marker()
            return True

        keep = max(alive, key=alive.get)
        if len(alive) > 1:
            print(f"[engine] found trainers at {sorted(alive)}; keeping "
                  f"{keep} (step {alive[keep]}) and stopping the rest",
                  flush=True)
            for site in alive:
                if site == keep:
                    continue
                if not self.executor.stop(site):
                    self.health = "stalled"
                    self.health_note = (
                        f"a second trainer at {site} could not be stopped; "
                        f"results may be inconsistent until it is killed")
        else:
            print(f"[engine] adopting running trainer at {keep} "
                  f"(step {alive[keep]})", flush=True)

        with self._lock:
            self.active_site = keep
            self.last_move_sim_t = self.clock.now()
        self._persist_last_active()
        self._replicate_marker()
        return True

    # -- persisted evidence of where the job was ---------------------------

    def _overwrite_would_lose_progress(self, src: str,
                                       target: str) -> Optional[str]:
        """Reason string if migrating would clobber genuinely better work.

        "Further along" alone is not enough. A site the job visited earlier
        can hold a high-step checkpoint from an abandoned branch — refusing
        on step count alone would lock the job out of that site forever
        (seen on the testbed: UTAH kept a 70k-step checkpoint from a branch
        that a bad restart orphaned). Only a checkpoint that is both further
        along AND more recent than the source's represents work we would
        actually be destroying.
        """
        try:
            src_p = self.executor.get_progress(src) or {}
            dst_p = self.executor.get_progress(target) or {}
        except Exception:
            return None                     # cannot tell; don't block
        src_step, dst_step = src_p.get("step"), dst_p.get("step")
        if src_step is None or dst_step is None or dst_step <= src_step:
            return None

        src_ts, dst_ts = src_p.get("ts"), dst_p.get("ts")
        if src_ts is not None and dst_ts is not None and dst_ts <= src_ts:
            print(f"[engine] {target} has a higher step ({dst_step}) but an "
                  f"older checkpoint than {src} ({src_step}); treating it as "
                  f"an abandoned branch and overwriting", flush=True)
            return None

        return (f"{target} holds newer, further-along work "
                f"(step {dst_step} vs {src_step} at {src}); refusing to "
                f"overwrite it")

    def _best_resume_site(self) -> str:
        """Where to restart so the least work is lost.

        The most advanced checkpoint wins. Falls back to the configured
        start site only when no reachable worker has any progress to
        resume from.
        """
        best, best_step = None, -1
        for site in self.managed:
            try:
                step = (self.executor.get_progress(site) or {}).get("step")
            except Exception:
                step = None
            if step is not None and step > best_step:
                best, best_step = site, step
        if best is None:
            return self.active_site
        if best != self.active_site:
            print(f"[engine] resuming at {best} (step {best_step}) rather "
                  f"than {self.active_site}: furthest-along checkpoint",
                  flush=True)
        return best

    def _discover_history(self, unknown: list) -> tuple:
        """Learn what ran here before, from evidence that outlives this host.

        Returns (last_active_site, prior_run_seen). The local state file is
        only a fast path: it is empty on a fresh control node even when the
        slice has plenty of history, so the authoritative evidence lives on
        the workers — a marker naming the active site, and any leftover job
        state proving *someone* ran here.
        """
        last_active = self._load_last_active()
        prior_run = last_active is not None
        newest = None

        for site in self.managed:
            if site in unknown:
                continue                      # cannot ask an unreachable node
            marker = None
            try:
                marker = self.executor.read_marker(site)
            except Exception:
                pass
            if marker:
                prior_run = True
                # Markers are replicated on every migration, so any reachable
                # worker can name the active site — including when that site
                # is the one we cannot reach.
                newest = marker
            try:
                if self.executor.has_job_state(site):
                    prior_run = True
            except Exception:
                pass

        if newest:
            last_active = newest
        if last_active:
            print(f"[engine] prior active site per slice evidence: "
                  f"{last_active}", flush=True)
        return last_active, prior_run

    def _replicate_marker(self):
        """Record the active site on every reachable worker, so the next
        orchestrator — on any host — can find it."""
        for site in self.managed:
            try:
                self.executor.write_marker(site, self.active_site)
            except Exception:
                pass

    def _load_last_active(self) -> Optional[str]:
        """Which site this orchestrator last had the job on, if known.

        Absence means no prior run wrote it, so an unreachable node carries
        no evidence of a trainer and must not block startup.
        """
        if not self.state_file:
            return None
        try:
            with open(self.state_file) as f:
                return json.load(f).get("active_site")
        except (OSError, json.JSONDecodeError):
            return None

    def _persist_last_active(self):
        if not self.state_file:
            return
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"active_site": self.active_site,
                           "ts": time.time()}, f)
            os.replace(tmp, self.state_file)
        except OSError as e:
            print(f"[engine] could not persist state: {e}", flush=True)

    def _tick(self):
        intensities = self._read_intensities()
        progress = self.executor.get_progress(self.active_site)
        sim_t = self.clock.now()
        self._update_net_estimates()

        self._check_liveness(progress)

        with self._lock:
            self._intensities = intensities
            if progress:
                self._progress = progress
            self._integrate_emissions(sim_t, intensities)
            self.history.append({
                "sim_t": round(sim_t, 1),
                "hour": round(self.clock.hour_of_day(), 2),
                "intensities": {k: round(v, 1) for k, v in intensities.items()},
                # Which of those were carried over rather than measured this
                # tick. Without it the chart draws a stale plateau as a solid
                # measured line, which is the flat-out wrong claim.
                "stale": sorted(k for k in intensities
                                if k not in self._fresh_sites),
                "active": self.active_site,
                "step": (progress or {}).get("step"),
                "loss": (progress or {}).get("loss"),
                "saved_g": round(self.baseline_g - self.emissions_g, 1),
            })

        target = self._pick_target(intensities, sim_t)
        if target:
            self._migrate(target)

    def _read_intensities(self) -> Dict[str, float]:
        """Read every site, tolerating a failure on any one of them.

        A per-zone read keeps the rest of the map live when one zone fails.
        Stale values are reused only briefly and only while labelled stale:
        an unbounded fallback means a dead feed silently freezes the whole
        map at its last values while emissions keep accruing against them,
        which manufactures the headline number out of nothing.
        """
        max_stale = float(self.net_cfg.get("max_stale_s")
                          or self.policy.get("carbon_max_stale_s", 120))
        now = time.time()
        out: Dict[str, float] = {}
        fresh, stale, lost = set(), [], []

        for name, cfg in self.sites.items():
            try:
                out[name] = self.provider.get_intensity(cfg["zone"])
                self._intensity_seen[name] = now
                fresh.add(name)
            except Exception as e:
                previous = self._intensities.get(name)
                age = now - self._intensity_seen.get(name, 0.0)
                if previous is not None and age <= max_stale:
                    out[name] = previous          # briefly, and labelled
                    stale.append(f"{name} ({int(age)}s old)")
                else:
                    lost.append(f"{name} ({e})")

        if fresh and not stale and not lost:
            status, note = "ok", None
        elif not fresh:
            status = "unavailable"
            note = ("carbon feed unavailable for every site — emissions "
                    "accounting paused" + (f"; {lost[0]}" if lost else ""))
        else:
            status = "stale"
            note = "carbon data stale/missing: " + ", ".join(
                (stale + lost)[:4])

        with self._lock:
            self.carbon_status = status
            self.carbon_note = note
            self._fresh_sites = fresh
        if note:
            print(f"[engine] {note}", flush=True)
        return out

    def _check_liveness(self, progress: Optional[dict]):
        """Catch the general 'says running but isn't' case.

        Migration failures are handled where they happen; this covers the
        rest (VM reboot, OOM-killed trainer, wedged node). If the step count
        stops advancing while we claim to be running, say so and try one
        restart rather than displaying a frozen counter all evening.
        """
        if self.status != "running":
            return
        # A finished run is not a broken one. Without this the stall detector
        # flags a completed job, tries to restart it, fails (the trainer exits
        # immediately), and reports health=down for a job that did its work.
        if (progress or {}).get("done"):
            with self._lock:
                self.status = "completed"
                self.health = "ok"
                self.health_note = (
                    f"training run finished at step {progress.get('step')}")
            print(f"[engine] {self.health_note}", flush=True)
            return
        stall_after = self.policy.get("stall_after_s", 90)
        step = (progress or {}).get("step")
        now = time.time()

        if step is None:
            if self._stall_ref is None:
                self._stall_ref = (None, now)
        elif self._stall_ref is None or step != self._stall_ref[0]:
            self._stall_ref = (step, now)       # progressing
            # Advancing steps are positive proof the job is running, so they
            # clear a "down" verdict as well as a "stalled" one — otherwise a
            # recovered job keeps a red badge and frozen emissions.
            if self.health in ("stalled", "down"):
                with self._lock:
                    self.health, self.health_note = "ok", None
            return

        if now - self._stall_ref[1] < stall_after:
            return

        site = self.active_site
        stalled_for = int(now - self._stall_ref[1])
        note = (f"no training progress at {site} for {stalled_for}s "
                f"(step {self._stall_ref[0]})")
        print(f"[engine] {note}", flush=True)
        with self._lock:
            self.health, self.health_note = "stalled", note

        # "No progress" has three causes and only one of them is fixed by
        # starting a trainer. Ask the node which it is before acting: a
        # blind restart on an unreachable node is how duplicates happen.
        alive = self.executor.is_running(site)

        if alive is None:
            with self._lock:
                self.health_note = (
                    f"{note}; cannot reach {site} to check — not starting "
                    f"another trainer (a duplicate would corrupt the run)")
            self._stall_ref = (self._stall_ref[0], now)
            return

        try:
            if alive:
                # Running but wedged: it holds the lock, so it must go first.
                if not self.executor.stop(site):
                    with self._lock:
                        self.health_note = (
                            f"{note}; trainer at {site} is wedged and would "
                            f"not stop — not starting a second one")
                    self._stall_ref = (self._stall_ref[0], now)
                    return
            self.executor.start(site)
            with self._lock:
                self.health_note = f"{note}; restarted"
            self._stall_ref = (self._stall_ref[0], now)  # give it another window
        except Exception as e:
            with self._lock:
                self.health = "down"
                self.health_note = (f"job is NOT running at {site}: "
                                    f"restart failed ({e})")

    def _integrate_emissions(self, sim_t: float, intensities: Dict[str, float]):
        # A stopped job burns nothing, and crediting it "savings" for downtime
        # would flatter the comparison — freeze both series while it is down.
        if self.health == "down" or self.status == "completed":
            # A finished run draws no power. Continuing to integrate would
            # keep inflating BOTH series for a job that has stopped working,
            # which is fabricated CO2 on the headline number.
            self._last_tick_sim_t = sim_t
            return
        # Only fresh readings may move the counters. Integrating a stale
        # intensity invents CO2 that was never measured, and the headline
        # number is exactly what a viewer trusts most.
        if (self.active_site not in self._fresh_sites
                or self.baseline_site not in self._fresh_sites):
            self._last_tick_sim_t = sim_t
            return
        active = intensities.get(self.active_site)
        baseline = intensities.get(self.baseline_site)
        if active is None or baseline is None:
            # Defensive: this runs inside the tick's lock and before the
            # history append, so raising here would blank the dashboard as
            # well as the counters.
            self._last_tick_sim_t = sim_t
            return
        if self._last_tick_sim_t is not None:
            dt_h = (sim_t - self._last_tick_sim_t) / 3600.0
            kw_active = self._power_kw(self.active_site)
            kw_base = self._power_kw(self.baseline_site)
            if kw_active is None or kw_base is None:
                self._last_tick_sim_t = sim_t
                return
            self.emissions_g += kw_active * dt_h * active
            self.baseline_g += kw_base * dt_h * baseline
        self._last_tick_sim_t = sim_t

    def _update_net_estimates(self):
        """Refresh per-candidate transfer estimates from FABRIC telemetry.

        Cheap after the first call in each TTL window — the model caches the
        instant query; this is label math on the cached link graph.
        """
        if self.net_model is None:
            return
        nbytes = (self.executor.checkpoint_bytes(self.active_site)
                  or self.net_cfg.get("default_checkpoint_mb", 1) * 1e6)
        est = {}
        for site in self.managed:
            if site == self.active_site:
                continue
            e = self.net_model.estimate(self.active_site, site, nbytes,
                                        rate_hint_bps=self._measured_rate_bps)
            if e.get("ok"):
                e["est_downtime_s"] = round(self._predict_downtime(e), 1)
                e["orchestration_s"] = (
                    round(self._measured_orchestration_s, 2)
                    if self._measured_orchestration_s is not None else None)
                e["setup_s"] = (round(self._measured_setup_s, 2)
                                if self._measured_setup_s is not None else None)
            est[site] = e
        with self._lock:
            self._net_est = est

    def _downtime_ok(self, target: str) -> bool:
        """Veto gate from live dataplane telemetry; fail-open on no data."""
        est = self._net_est.get(target)
        if not est or not est.get("ok"):
            return True
        budget = self.net_cfg.get("max_downtime_s", 120)
        if est["est_downtime_s"] <= budget:
            return True
        self._net_note = (f"vetoed {self.active_site}->{target}: predicted "
                          f"downtime {est['est_downtime_s']}s > {budget}s "
                          f"(path headroom {est['headroom_gbps']} Gbps)")
        print(f"[engine] {self._net_note}", flush=True)
        return False

    def _pick_target(self, intensities: Dict[str, float],
                     sim_t: float) -> Optional[str]:
        if self.status != "running":
            self._force_target = None   # drop stale booth requests
            return None
        if self._force_target:  # booth button bypasses policy AND veto
            target, self._force_target = self._force_target, None
            return target
        dwell_s = self.policy.get("min_dwell_min", 15) * 60.0
        if sim_t - self.last_move_sim_t < dwell_s:
            return None
        if self.active_site not in self._fresh_sites:
            return None      # stale reading: no basis for an improvement
        # Cleanest-first among candidates that clear the carbon gain AND the
        # predicted-downtime gate, so a congested path falls through to the
        # next-cleanest site instead of blocking migration entirely.
        current = intensities[self.active_site]
        min_gain = self.policy.get("improvement_g", 60)
        # A move is only justified by a current reading at both ends.
        candidates = [s for s in self.managed
                      if s in intensities and s in self._fresh_sites]
        for site in sorted(candidates, key=lambda s: intensities[s]):
            if site == self.active_site:
                return None  # nothing cleaner than where we already are
            if current - intensities[site] < min_gain:
                return None  # sorted, so no later candidate clears it either
            if self._downtime_ok(site):
                self._net_note = None
                return site
        return None

    def _migrate(self, target: str):
        """Move the job, or leave it demonstrably running somewhere.

        The invariant: when this returns, either the job is running at
        `active_site`, or `health` says it is not. Never "running" with a
        stopped workload — the transfer is a copy, so the source checkpoint
        survives a failed move and resuming there is always safe.
        """
        src = self.active_site
        predicted = (self._net_est.get(target) or {}).get("est_downtime_s")
        print(f"[engine] migrating {src} -> {target}", flush=True)
        self.status = "migrating"
        t0 = time.time()

        # Never start the job elsewhere while it might still run here: two
        # trainers advancing the same checkpoint would silently diverge, and
        # both would burn power we account for once.
        try:
            stopped = self.executor.stop(src)
        except Exception as e:
            self._abort_migration(src, target, f"stop at {src} failed: {e}",
                                  stopped_source=False)
            return
        if not stopped:
            self._abort_migration(
                src, target,
                f"could not confirm the trainer stopped at {src}",
                stopped_source=False)
            return

        # Transferring overwrites the target's checkpoint. In normal
        # operation the source is authoritative and further along, so this
        # never fires; if it does, something upstream went wrong and copying
        # would permanently destroy the better checkpoint.
        regression = self._overwrite_would_lose_progress(src, target)
        if regression:
            self._abort_migration(src, target, regression, stopped_source=True)
            return

        try:
            nbytes = self.executor.transfer_checkpoint(src, target)
        except Exception as e:
            self._abort_migration(src, target, f"transfer failed: {e}")
            return
        try:
            self.executor.start(target)
        except Exception as e:
            # Checkpoint reached the target but it won't run; the copy at
            # the source is still valid, so fall back there.
            self._abort_migration(src, target, f"start at {target} failed: {e}")
            return

        downtime = time.time() - t0
        xfer = getattr(self.executor, "last_transfer", None)
        self._calibrate(xfer, downtime)
        with self._lock:
            self.active_site = target
            self.last_move_sim_t = self.clock.now()
            self.status = "running"
            self.health = "ok"
            self.health_note = None
            self._stall_ref = None
            self.migrations.append({
                "from": src, "to": target,
                "sim_t": round(self.clock.now(), 1),
                "bytes": nbytes,
                "downtime_s": round(downtime, 1),
                "predicted_downtime_s": predicted,
                "transfer_method": (xfer or {}).get("method"),
                "transfer_s": (xfer or {}).get("seconds"),
                "outcome": "ok",
            })
        self._persist_last_active()
        self._replicate_marker()

    def _abort_migration(self, src: str, target: str, reason: str,
                         stopped_source: bool = True):
        """Failed move: restart at the source and tell the truth either way.

        `stopped_source=False` means we could not confirm the source trainer
        died — the job may already be running there, so `start` must be the
        idempotent kind (it is) and we must not assume anything.

        Cleanup only ever removes the staging file, which is safe on every
        path: the target's live checkpoint is replaced by an atomic commit
        after its size is verified, so it is never left partial and never
        needs deleting.
        """
        print(f"[engine] {src}->{target} aborted ({reason}); "
              f"resuming at {src}", flush=True)
        # The target may have started before the failure surfaced (start()
        # raises when it cannot CONFIRM the lock, which is not the same as
        # "did not start"). Resuming at the source without confirming the
        # target is dead is precisely how a fork happens.
        try:
            target_clear = self.executor.stop(target)
        except Exception as e:
            print(f"[engine] stop:{target} errored: {e}", flush=True)
            target_clear = False

        if not target_clear:
            with self._lock:
                self.active_site = src
                self.last_move_sim_t = self.clock.now()
                self.status = "stopped"
                self.health = "down"
                self.health_note = (
                    f"migration to {target} failed ({reason}) and {target} "
                    f"could not be confirmed stopped — NOT restarting at "
                    f"{src}, because a trainer may still be running at "
                    f"{target}. Clear {target}, then restart.")
                self.migrations.append({
                    "from": src, "to": target,
                    "sim_t": round(self.clock.now(), 1),
                    "outcome": "aborted", "reason": reason,
                    "resumed_at": None,
                    "blocked_by": target,
                })
            print(f"[engine] {self.health_note}", flush=True)
            return

        self._discard_incoming(target)
        resumed = False
        try:
            self.executor.start(src)
            resumed = True
        except Exception as e:
            print(f"[engine] CRITICAL: could not resume at {src}: {e}",
                  flush=True)
        with self._lock:
            self.active_site = src          # never moved
            self.last_move_sim_t = self.clock.now()   # back off before retrying
            self.status = "running" if resumed else "stopped"
            self.health = "ok" if resumed else "down"
            self.health_note = (
                f"migration to {target} aborted ({reason}); job resumed at {src}"
                if resumed else
                f"job is NOT running: {src} would not restart after "
                f"failed migration to {target} ({reason})")
            self._stall_ref = None
            self.migrations.append({
                "from": src, "to": target,
                "sim_t": round(self.clock.now(), 1),
                "outcome": "aborted", "reason": reason,
                "resumed_at": src if resumed else None,
            })
        if resumed:
            self._persist_last_active()
        self._replicate_marker()

    def _discard_incoming(self, site: str):
        """Remove only the staging file a transfer may have written.

        Safe on every abort path by construction: transfers commit by moving
        a size-verified staging file into place, so the live checkpoint is
        never partial and never deleted here. The previous version removed
        the live checkpoint, which destroyed intact work whenever the
        transfer failed before writing anything.
        """
        discard = getattr(self.executor, "discard_incoming", None)
        if discard is None:
            return
        try:
            discard(site)
        except Exception as e:
            print(f"[engine] could not clear staging file at "
                  f"{site}: {e}", flush=True)

    # A transfer smaller than this is dominated by connection setup, not
    # bandwidth. Dividing its size by its duration yields a "rate" that is
    # really a latency measurement — on FABRIC, a 12 KB checkpoint in 1.35 s
    # reads as 0.00007 Gbps, which would then predict ~10 days for 8 GB.
    MIN_RATE_CALIBRATION_BYTES = 50e6

    @staticmethod
    def _ewma(prev: Optional[float], sample: float, weight: float = 0.4):
        return sample if prev is None else (1 - weight) * prev + weight * sample

    def _calibrate(self, xfer: Optional[dict], downtime_s: Optional[float]):
        """Feed a completed migration back into the downtime model.

        Only dataplane transfers count: relay transfers cross the management
        network and local copies never hit the wire, so calibrating on them
        would teach the model a rate for a path it isn't pricing.

        The measured downtime — not the transfer duration — is what the veto
        is about, so orchestration cost is taken from
        `downtime - transfer`. Measuring only the transfer and calling it
        "overhead" halves the prediction: on FABRIC a 1.5s transfer sat
        inside a 3.6s downtime, and the missing 2.1s was the stop/start it
        never accounted for.
        """
        if not xfer or xfer.get("method") != "dataplane":
            return
        nbytes = xfer.get("bytes") or 0
        transfer_s = xfer.get("seconds") or 0
        if nbytes <= 0 or transfer_s <= 0:
            return

        with self._lock:
            if downtime_s is not None and downtime_s >= transfer_s:
                self._measured_orchestration_s = self._ewma(
                    self._measured_orchestration_s, downtime_s - transfer_s)

            if nbytes >= self.MIN_RATE_CALIBRATION_BYTES:
                # Wire time is what remains after per-transfer setup.
                setup = self._measured_setup_s or 0.0
                wire_s = max(1e-3, transfer_s - setup)
                rate = nbytes * 8 / wire_s
                self._measured_rate_bps = self._ewma(
                    self._measured_rate_bps, rate)
                note = (f"{rate/1e9:.2f} Gbps over {nbytes/1e6:.0f} MB "
                        f"(EWMA {self._measured_rate_bps/1e9:.2f})")
            else:
                self._measured_setup_s = self._ewma(
                    self._measured_setup_s, transfer_s)
                note = (f"{nbytes/1e3:.1f} KB in {transfer_s:.2f}s is "
                        f"setup-dominated -> setup EWMA "
                        f"{self._measured_setup_s:.2f}s (not bandwidth)")
            orch = self._measured_orchestration_s

        print(f"[engine] calibrated: {note}"
              + (f"; orchestration {orch:.2f}s" if orch is not None else ""),
              flush=True)

    def _predict_downtime(self, est: dict) -> float:
        """orchestration + setup + wire time, from measurements when we have
        them and the configured constant until then."""
        configured = self.net_cfg.get("overhead_s", 20)
        fixed = 0.0
        measured_any = False
        if self._measured_orchestration_s is not None:
            fixed += self._measured_orchestration_s
            measured_any = True
        if self._measured_setup_s is not None:
            fixed += self._measured_setup_s
            measured_any = True
        if not measured_any:
            fixed = configured          # conservative until measured
        return fixed + (est.get("est_transfer_s") or 0)
