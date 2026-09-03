#!/usr/bin/env python3
"""Live dashboard + demo controls for the Pegasus/HTCondor carbon-aware run.

One auto-refreshing page showing the whole story while a workflow runs:

* **Pool** — each worker's live `CarbonIntensity` (EIA replay) and measured
  `GPUWatts` from its machine ClassAd: exactly the numbers matchmaking ranks
  on. The cleanest site is where the NEXT segment lands.
* **Jobs** — every workflow job from `condor_q` and `condor_history`: which
  site won the match, the carbon and GPU power *recorded in the job ad at
  match time* (`job_machine_attrs`), wall time, and the bytes HTCondor
  actually moved in/out of the sandbox.
* **Energy is measured, not derived** — the trainer's NVML sampler integrates
  watts over its segment and stages the result out as `gpu_wh`; the dashboard
  joins it back to the job. gCO2 = measured Wh x intensity-at-match / 1000.
  A segment with no NVML reading shows unknown, and the totals say how many
  segments the sum actually covers.

## Demo controls, and their honesty contract

* **Inject carbon** (per site) writes `/opt/carbon/override` on that worker
  ("<gCO2/kWh> <expiry-epoch>"). Within one STARTD_CRON period the site
  advertises the injected intensity **plus `CarbonInjected = true`**, so the
  card is badged, the real trace value stays published alongside
  (`CarbonTraceIntensity`), and the override expires by itself. The next
  negotiation reacts to it: a REAL migration with real measured
  transfer/energy cost, triggered by a disclosed synthetic price. Power is
  deliberately NOT injectable — it is the measured half of the story, and
  carbon alone is what matchmaking ranks on.
* **Start workflow** runs the same generate -> plan -> submit sequence as the
  notebook, on the submit node, with the chosen segments x minutes.

## Where it runs

Made to run ON THE SUBMIT NODE (`--mode local`, auto-detected when
condor_status is on PATH) — stdlib only, since fablib is not installed there.
View it from a laptop over an SSH tunnel:
`ssh -L 8799:localhost:8799 ubuntu@<submit-mgmt-ip>`. It binds 127.0.0.1 on
purpose: the control endpoints execute commands, so the tunnel is the entire
exposure. Laptop mode (`--mode fablib`) also works; worker commands route
through the submit node either way, over the pool's own SSH keys.

Usage:
    python3 dashboard.py [--mode auto|local|fablib] [--port 8799]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# One round-trip per poll: several documents separated by ===MARKER===
# lines. Each marker is preceded by a bare `echo`, because a section whose
# content lacks a trailing newline glues the NEXT marker to its last line
# ("}===REPORT==="), which parse_sections cannot recognise — that section
# then silently vanishes. json.dump(indent=...) writes no trailing newline,
# and this bug ate the final result and report status on a live pool.
#
# Everything is SCOPED TO THE CURRENT RUN by WORKFLOW IDENTITY, not time.
# Every Pegasus job ad carries `pegasus_wf_uuid` and the newest run
# directory's braindump.yml records which uuid is current; a job belongs to
# exactly one workflow, so a uuid cannot leak the way a time filter does
# (a straggler outliving condor_rm's grace completes after the new epoch
# and would pass). With no current run the sentinel matches nothing, so the
# tables go honestly empty instead of showing a previous run's jobs.
#
# The PROGRESS FILES come from the same braindump, for the same reason:
# "newest epoch directory" is a DIFFERENT identity from "newest run dir"
# (epochs are stamped by whichever host initiated the submission, so clock
# skew can order them against run-dir order and join a previous run's
# measured energy onto this run's segments). No output-dir recorded -> a
# path that cannot exist, so progress is empty rather than wrong.
POLL_SCRIPT = r"""
RD=$(ls -d /home/ubuntu/wf-runs/*/pegasus/*/run* 2>/dev/null | tail -1)
UUID=$(grep -m1 '^wf_uuid:' "$RD/braindump.yml" 2>/dev/null \
       | awk '{print $2}' | tr -d '"')
UUID=${UUID:-no-current-run}
echo; echo '===UUID==='
echo "$UUID"
OUTDIR=$(sed -n 's/.*--output-dir[ =]\([^ "]*\).*/\1/p' "$RD/braindump.yml" \
         2>/dev/null | head -1)
echo; echo '===MACHINES==='
condor_status -json -attributes \
  Name,Machine,FabricSite,CarbonIntensity,CarbonInjected,CarbonTraceIntensity,GPUWatts,CarbonTimestamp,CarbonSampledAt,CarbonSpan,CarbonAccel,MyCurrentTime,LastHeardFrom,State,Activity \
  2>/dev/null
echo; echo '===QUEUE==='
condor_q -json -constraint "pegasus_wf_uuid == \"$UUID\"" -attributes \
  ClusterId,DAGNodeName,JobStatus,RemoteHost,JobCurrentStartDate,QDate,MachineAttrFabricSite0,MachineAttrCarbonIntensity0,MachineAttrGPUWatts0,MachineAttrCarbonInjected0 \
  2>/dev/null
echo; echo '===HISTORY==='
condor_history -limit 120 -json -constraint "pegasus_wf_uuid == \"$UUID\"" -attributes \
  ClusterId,DAGNodeName,LastRemoteHost,RemoteWallClockTime,BytesRecvd,BytesSent,TransferInputSizeMB,MachineAttrCarbonIntensity0,MachineAttrGPUWatts0,MachineAttrFabricSite0,MachineAttrCarbonInjected0,ExitCode,CompletionDate \
  2>/dev/null
echo; echo '===OUTDIR==='
echo "$OUTDIR"
echo; echo '===PROGRESS==='
for f in $(ls -t "${OUTDIR:-/nonexistent}"/progress_*.json 2>/dev/null | head -24); do
  echo "@@$f"; cat "$f" 2>/dev/null; echo
done
echo; echo '===RESULT==='
cat "${OUTDIR:-/nonexistent}/higgs_result.json" 2>/dev/null
echo; echo '===REPORT==='
# The report declares which run it describes (report_higgs.py writes
# "workflow run: <uuid>"). Existence is not provenance: a report left in a
# shared output directory by another run would otherwise be served as this
# run's deliverable. Emit the declared id and let the poller verify it.
if [ -s "${OUTDIR:-/nonexistent}/higgs_report.html" ]; then
  sed -n 's/.*workflow run: \([0-9a-f-]\{36\}\).*/\1/p' \
      "${OUTDIR}/higgs_report.html" | head -1
  echo PRESENT
else
  echo ABSENT
fi
"""

# Same wipe -> generate -> plan -> submit sequence as the notebook cell,
# parameterized. Runs on the submit node; output-dir is stamped so results
# stay attributable to THIS submission.
START_SCRIPT = """
cd /home/ubuntu || exit 9
for OLD in $(ls -d wf-runs/*/pegasus/*/run* 2>/dev/null); do
    pegasus-remove $OLD > /dev/null 2>&1 || true
done
condor_rm -all > /dev/null 2>&1 || true
for _ in 1 2 3 4 5 6 7 8 9 10; do
    test -z "$(condor_q -af ClusterId 2>/dev/null)" && break
    sleep 3
done
# Deliberately NOT 'rm -rf wf-runs wf-output': output dirs are
# epoch-namespaced and run_result.py attributes via each run's own
# braindump, so a wipe buys no isolation — it only destroys prior
# runs' evidence. That cost a real artifact: the first run ever
# observed migrating had its report deleted by the next submission.
python3 workflow_generator.py --segments {segments} --minutes {minutes} \
    --out higgs-wf.yml > gen.log 2>&1 || {{ echo GEN_FAILED; tail -5 gen.log; exit 1; }}
pegasus-plan --submit -s condorpool --output-site local \
    --conf pegasus.properties \
    --output-dir /home/ubuntu/wf-output/{epoch} \
    --dir wf-runs higgs-wf.yml > plan.log 2>&1 || {{ echo PLAN_FAILED; tail -8 plan.log; exit 1; }}
grep -q "submitted to cluster" plan.log && echo SUBMIT_OK || {{ echo NOT_SUBMITTED; tail -8 plan.log; exit 1; }}
"""

# Fetching the report and the run identity in ONE shell invocation is
# deliberate. Verifying `poller.report_status` (computed at the last poll,
# up to `interval` seconds ago) and then reading the file separately is a
# TOCTOU: between the two, the run can change or a concurrent run can
# overwrite the file in a shared output directory, and the dashboard would
# serve bytes it never checked. Here the declared id is extracted from the
# SAME bytes that are returned, so the check cannot drift from the content.
REPORT_SCRIPT = r"""
RD=$(ls -d /home/ubuntu/wf-runs/*/pegasus/*/run* 2>/dev/null | tail -1)
UUID=$(grep -m1 '^wf_uuid:' "$RD/braindump.yml" 2>/dev/null \
       | awk '{print $2}' | tr -d '"')
OUTDIR=$(sed -n 's/.*--output-dir[ =]\([^ "]*\).*/\1/p' "$RD/braindump.yml" \
         2>/dev/null | head -1)
echo; echo '===UUID==='
echo "${UUID:-no-current-run}"
echo; echo '===HTML==='
cat "${OUTDIR:-/nonexistent}/higgs_report.html" 2>/dev/null
"""

# The report declares its run as: workflow run: <uuid>
REPORT_RUN_ID = re.compile(r"workflow run:\s*([0-9a-f-]{36})")


JOB_STATUS = {1: "idle", 2: "running", 3: "removed", 4: "done", 5: "held",
              6: "transferring", 7: "suspended"}


class LocalBackend:
    """Runs directly on the submit node; workers reached over the pool's
    own SSH keys (exchanged at provisioning, hostnames in /etc/hosts)."""

    name = "local"

    def submit_cmd(self, cmd, timeout=120):
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout)
        return proc.stdout, proc.stderr

    def worker_cmd(self, host, cmd, timeout=60):
        return self.submit_cmd(
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"{host} {json.dumps(cmd)}", timeout)


class FablibBackend:
    """Laptop mode: everything is routed through the submit node over
    fablib, including worker commands (submit -> worker over pool SSH)."""

    name = "fablib"

    def __init__(self, slice_name):
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), ".."))
        from carbon_chaser.executor import call_with_timeout
        from fabrictestbed_extensions.fablib.fablib import FablibManager
        self._call = call_with_timeout
        self._submit = (FablibManager().get_slice(name=slice_name)
                        .get_node(name="submit"))

    def submit_cmd(self, cmd, timeout=120):
        out, err = self._call(
            lambda: self._submit.execute(cmd, quiet=True, retry=1),
            timeout, "dashboard cmd")
        return (out or ""), (err or "")

    def worker_cmd(self, host, cmd, timeout=60):
        return self.submit_cmd(
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"{host} {json.dumps(cmd)}", timeout)


def parse_sections(raw):
    parts = {}
    current = None
    for line in raw.splitlines():
        if line.startswith("===") and line.endswith("==="):
            current = line.strip("=")
            parts[current] = []
        elif current:
            parts[current].append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}


def parse_json_array(text):
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except ValueError:
        return []


def job_row(ad, queued=False):
    """One display row per workflow job, from a queue or history ad.

    Carbon and watts here are MATCH-TIME SNAPSHOTS from the job ad
    (job_machine_attrs) — kept as context, never multiplied by wall-clock:
    a snapshot times a duration is an estimate, and this project does not
    dress estimates as measurements. The measured energy (`measured_wh`)
    is joined in later from the segment's progress file, where the trainer
    integrates NVML samples over the segment itself.
    """
    name = ad.get("DAGNodeName") or f"cluster {ad.get('ClusterId')}"
    host = (ad.get("LastRemoteHost") or ad.get("RemoteHost") or "")
    host = host.split("@")[-1].split(".")[0]
    status = (JOB_STATUS.get(ad.get("JobStatus"), "idle") if queued
              else "done")
    # History ads carry RemoteWallClockTime; a queued ad does not (it is
    # only accumulated at job exit), so derive elapsed time from the start
    # date — but only while the current attempt still occupies its slot
    # (running, transferring output, suspended: wall-clock is genuinely
    # elapsing for all three). Idle and held jobs keep 0: their
    # JobCurrentStartDate is a stale leftover from an attempt that already
    # ended, and ticking a clock for them misreports the run.
    wall = float(ad.get("RemoteWallClockTime") or 0)
    if (status in ("running", "transferring", "suspended") and not wall
            and ad.get("JobCurrentStartDate")):
        wall = max(0.0, time.time() - float(ad["JobCurrentStartDate"]))
    return {
        "job": name,
        "status": status,
        "site": ad.get("MachineAttrFabricSite0") or host or "-",
        "wall_s": round(wall),
        "carbon_at_match": ad.get("MachineAttrCarbonIntensity0"),
        # True when the match was made against a demo-injected price; the
        # label travels with the number everywhere it is shown or summed.
        "injected_at_match": bool(ad.get("MachineAttrCarbonInjected0")),
        "gpu_watts_at_match": ad.get("MachineAttrGPUWatts0"),
        "measured_wh": None,
        "gco2": None,
        "mb_in": round(float(ad.get("BytesRecvd") or 0) / 1e6, 1),
        "mb_out": round(float(ad.get("BytesSent") or 0) / 1e6, 1),
        "exit": ad.get("ExitCode"),
    }


# Deliberately in step with check_pool_match.check_replay_coherence: same
# attributes, same allowance, same circular comparison. dashboard.py is
# uploaded to the submit node on its own and must stay stdlib-only, so the
# logic is duplicated rather than imported — change one, change both.
CLOCK_SKEW_ALLOWANCE_S = 5.0
COHERENCE_KEYS = ("carbon_accel", "carbon_span", "carbon_sampled_at",
                  "carbon_ts")


def circular_delta(a, b, span):
    """Distance between two positions on a wrapping timeline.

    `abs(a - b)` is wrong: the replay wraps, so a position just after the
    wrap and one just before it are ADJACENT in replayed time while being a
    whole span apart numerically.
    """
    gap = (a - b) % span
    return min(gap, span - gap)


def clock_offset(site):
    """How far a worker's clock sits from the collector's, or None.

    `MyCurrentTime` is stamped by the startd's own clock and `LastHeardFrom`
    by the collector when the ad arrives, so their difference measures the
    offset directly — independently of how old the ad is, which is what lets
    a slow clock be told apart from slow updates.

    None means UNMEASURABLE, and callers must treat that as a failure rather
    than as zero. There is deliberately no fallback: comparing the sample
    time against our own clock only ever sees a clock running AHEAD, because
    a lagging clock is indistinguishable from a stale ad that way — so it
    would silently pass exactly the case this function exists to catch.
    """
    node, heard = site.get("ad_node_time"), site.get("ad_heard_at")
    if node is None or heard is None:
        return None
    return float(node) - float(heard)


def replay_coherence(sites, now):
    """Are the sites reading the same moment of the carbon trace?

    Cross-node equality of `CarbonTimestamp` is NOT the test, and reporting
    it as one is worse than reporting nothing. STARTD_CRON samples each node
    every 60s, unsynchronised, so healthy nodes differ by 60s x ACCEL of
    replayed time; and two nodes either side of the wrap differ by a whole
    span. Each node is therefore checked against its OWN `CarbonSampledAt`,
    on a circle, after agreeing on span and accel — the comparison
    check_pool_match.py settled on for exactly these false positives.

    That per-node comparison verifies the replay FORMULA, and it cannot see a
    skewed clock: carbon_classad derives the sample time and the position
    from the same `time.time()`, so a node whose clock is off is internally
    consistent and scores a zero error. Clock skew is therefore measured
    separately and DIRECTLY, from the two clocks HTCondor puts in every ad
    (`MyCurrentTime` from the startd, `LastHeardFrom` from the collector) —
    which catches a slow clock as readily as a fast one. Inferring it from ad
    age instead cannot: on this pool healthy ads run 208-268s old, so age
    alone confuses a lagging clock with slow updates.

    A node that ranks on carbon but publishes nothing to check with is
    reported UNVERIFIABLE, never as fine: it competes for jobs on a price
    whose replay position nothing here can vouch for.

    Internal consistency is also not freshness — the negotiator ranks the
    collector's copy of an ad, so an ad older than one trace row
    (3600/accel seconds) means "cleanest advertised" has drifted from
    "cleanest", and that is surfaced rather than folded into a pass.

    Those are two independent facts, returned as separate keys because
    conflating them is how a stale pool passes for a healthy one:
    `coherent` (same replayed moment?) and `fresh` (are the ads the
    negotiator ranks younger than one trace row?). `state` is only the
    display severity derived from both.
    """
    ranking = [s for s in sites if s.get("carbon") is not None]
    if not ranking:
        return {"state": "none", "coherent": None, "fresh": None,
                "headline": "no site publishes carbon",
                "detail": "nothing is ranking on carbon intensity right now"}

    def checkable(site):
        return all(site.get(k) is not None for k in COHERENCE_KEYS)

    # Sorted by name so the reference node — and therefore the reported
    # worst-offender — is stable between polls rather than following the
    # carbon ranking.
    rows = sorted((s for s in ranking if checkable(s)),
                  key=lambda s: s["site"])
    blind = sorted(s["site"] for s in ranking if not checkable(s))

    problems = []
    if blind:
        problems.append(
            f"{', '.join(blind)} rank(s) on carbon but publish no checkable "
            f"replay position (CarbonSpan/CarbonSampledAt/CarbonAccel) — "
            f"probably an older carbon_classad.py; coherence there is "
            f"UNKNOWN, which is not the same as fine")
    spans = {s["carbon_span"] for s in rows}
    accels = {s["carbon_accel"] for s in rows}
    if len(spans) > 1:
        problems.append(
            f"{len(spans)} different CarbonSpan values — the nodes wrap the "
            f"trace over different periods, which is the per-zone replay bug")
    if len(accels) > 1:
        problems.append(
            f"{len(accels)} different CarbonAccel values — the nodes advance "
            f"through the trace at different speeds, so their intensities are "
            f"not comparable even if they agree right now")

    worst = allowance = 0.0
    headline = "sites not comparable"
    if not problems and rows:
        span, accel = next(iter(spans)), next(iter(accels))
        allowance = accel * CLOCK_SKEW_ALLOWANCE_S
        ref = rows[0]
        for site in rows[1:]:
            expected = (site["carbon_sampled_at"]
                        - ref["carbon_sampled_at"]) * accel
            worst = max(worst, circular_delta(
                site["carbon_ts"] - ref["carbon_ts"], expected, span))
        if worst > allowance:
            problems.append(
                f"worst site is {worst:.0f}s of replayed time out of step "
                f"with {ref['site']} after accounting for their sampling gap "
                f"(allowance {allowance:.0f}s) — its replay position is not "
                f"the function of time the others are evaluating")

        # The loop above is BLIND TO CLOCK SKEW by construction: carbon_classad
        # derives `sampled_at` AND `position` from the same node clock, so a
        # node whose clock is off is *perfectly self-consistent* while sitting
        # offset x accel of replayed time from its peers, and scores a zero
        # error above. Measured: 600s of offset at accel 300 is 50 replayed
        # hours, in EITHER direction.
        #
        # HTCondor hands us both clocks in the same ad — `MyCurrentTime` is
        # stamped by the startd's clock, `LastHeardFrom` by the collector's
        # when the ad arrives — so the offset is measurable directly instead
        # of being inferred from ad age. That distinction matters: age alone
        # cannot tell a slow clock from slow updates (this pool's own ads run
        # 208-268s old when perfectly healthy), which is why an earlier
        # version of this function failed a fast clock and let a slow one
        # through with a staleness warning. Both displace the advertised
        # carbon by the same amount, so both fail here.
        # No measurable offset is a FAILURE, on the same principle as a site
        # that publishes CarbonIntensity without CarbonSpan: a clock nothing
        # can vouch for, on a node competing for jobs on carbon. Passing it
        # would restore the exact hole this check closes, because a skewed
        # clock is invisible to every other test here.
        blind_clock = [s["site"] for s in rows if clock_offset(s) is None]
        if blind_clock:
            headline = f"{len(blind_clock)} site(s): clock unverifiable"
            problems.append(
                f"{', '.join(blind_clock)} publish no "
                f"MyCurrentTime/LastHeardFrom pair, so the clock offset "
                f"cannot be measured — and a skewed clock is invisible to "
                f"every other check here, since the position and the sample "
                f"time come from that same clock. Unverifiable is not the "
                f"same as fine; check `condor_status -l <node> | grep -E "
                f"'MyCurrentTime|LastHeardFrom'`")

        for site in sorted(rows,
                           key=lambda s: -abs(clock_offset(s) or 0.0)):
            offset = clock_offset(site)
            if offset is None or abs(offset) <= CLOCK_SKEW_ALLOWANCE_S:
                continue
            direction = "ahead of" if offset > 0 else "behind"
            headline = (f"{site['site']} clock {abs(offset):.0f}s "
                        f"{'fast' if offset > 0 else 'slow'}")
            problems.append(
                f"{site['site']}'s clock is {abs(offset):.0f}s {direction} "
                f"the collector's, so it advertises a replay position "
                f"{abs(offset) * accel / 3600.0:.1f} replayed hours from its "
                f"peers while looking perfectly self-consistent — its "
                f"position and its sample time come from that same clock. "
                f"Fix time sync on that node")
            break

    if problems:
        return {"state": "bad", "coherent": False, "fresh": None,
                "headline": headline, "detail": "; ".join(problems)}

    accel = next(iter(accels))
    # Age is measured on ONE clock: a node's sample time is in its own clock,
    # so the measured offset is what converts it to the collector's. Without
    # that correction a healthy node with a slow clock reads as a stale ad.
    # Age is measured on ONE clock: a node's sample time is in its own, so
    # the measured offset is what converts it to the collector's. Without
    # that correction a healthy node with a slow clock reads as a stale ad.
    measured = [(s, clock_offset(s)) for s in rows]
    if any(offset is None for _, offset in measured):
        # Unreachable while the unverifiable-clock check above runs first.
        # Kept as a hard stop rather than an `or 0.0`, so reordering these
        # checks can never make an unmeasurable clock read as offset zero —
        # which is exactly how a slow clock used to pass.
        return {"state": "bad", "coherent": False, "fresh": None,
                "headline": "clock unverifiable",
                "detail": "a ranking site's clock offset could not be "
                          "measured, so its replay position cannot be "
                          "vouched for"}
    ages = [now - (s["carbon_sampled_at"] - offset) for s, offset in measured]
    worst_age = max(ages)
    # One trace row is an hour, so an ad older than 3600/accel seconds ranks
    # on a value the trace has already moved past.
    budget = 3600.0 / accel if accel else float("inf")
    fresh = (f"worst ad {worst_age:.0f}s old = "
             f"{worst_age * accel / 3600.0:.1f} replayed h "
             f"(one trace row lasts {budget:.0f}s at accel {accel:.0f})")

    if len(rows) < 2:
        return {"state": "thin", "coherent": None,
                "fresh": worst_age <= budget,
                "headline": "1 checkable site — coherence not demonstrated",
                "detail": f"only {rows[0]['site']} publishes a checkable "
                          f"replay position, so there is nothing to compare "
                          f"it with: trivially satisfied, not verified. "
                          f"{fresh}"}

    # Which site's position to show: the most RECENTLY SAMPLED one. Order by
    # sample time (real wall-clock, on one clock via the measured offset),
    # never by position — position wraps, so at the wrap boundary the newest
    # sample holds the SMALLEST value and `max(carbon_ts)` would name the
    # oldest one, off by nearly a whole span.
    newest = max(measured,
                 key=lambda pair: pair[0]["carbon_sampled_at"] - pair[1])[0]
    when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(newest["carbon_ts"]))
    if worst_age > budget:
        return {"state": "warn", "coherent": True, "fresh": False,
                "headline": "coherent, but ads are stale",
                "detail": f"all {len(rows)} sites match their own sample "
                          f"times to within {worst:.0f}s of replayed time "
                          f"(allowance {allowance:.0f}s) — but the negotiator "
                          f"ranks the collector's copy, and the {fresh}. "
                          f"That is either slow updates or a clock running "
                          f"behind; both look the same from here. Placement "
                          f"stays honest; 'cleanest' becomes 'cleanest "
                          f"recently advertised'."}
    return {"state": "ok", "coherent": True, "fresh": True,
            # "newest" matters: the nodes' positions legitimately differ by
            # up to the cron period x accel, so naming one instant without
            # that word would imply an equality the replay does not have.
            "headline": f"trace ~{when}Z (latest sample) · coherent "
                        f"across {len(rows)} sites",
            "detail": f"each site matches its own sample time to within "
                      f"{worst:.0f}s of replayed time (allowance "
                      f"{allowance:.0f}s), on one span and one accel, so the "
                      f"intensities are comparable. {fresh}"}


def attach_measured_energy(jobs, progress):
    """Join each train job's measured gpu_wh (from its progress file) and
    derive gCO2 = measured Wh x intensity-at-match. Jobs without a
    measurement keep None — shown as unknown, never estimated."""
    wh_by_segment = {p["segment"]: p.get("gpu_wh") for p in progress
                     if p.get("segment") is not None}
    for job in jobs:
        match = re.search(r"train_(\d+)$", job["job"])
        if not match:
            continue
        wh = wh_by_segment.get(int(match.group(1)) + 1)
        if wh is None:
            continue
        job["measured_wh"] = wh
        carbon = job["carbon_at_match"]
        if carbon is not None:
            job["gco2"] = round(wh * float(carbon) / 1000.0, 2)


class Poller(threading.Thread):
    daemon = True

    def __init__(self, backend, interval):
        super().__init__(name="condor-poller")
        self.backend = backend
        self.interval = interval
        self.state = {"ok": False, "error": "first poll pending"}
        self.outdir = ""
        self.report_status = "pending"
        self.carbon_series = {}      # site -> [(ts, intensity), ...]
        self.wake = threading.Event()

    def poll_once(self):
        out, err = self.backend.submit_cmd(POLL_SCRIPT)
        sections = parse_sections(out or "")
        machines = parse_json_array(sections.get("MACHINES", ""))
        queue = parse_json_array(sections.get("QUEUE", ""))
        history = parse_json_array(sections.get("HISTORY", ""))

        now = time.time()
        sites = []
        for ad in machines:
            site = ad.get("FabricSite") or ad.get("Machine", "?").split(".")[0]
            intensity = ad.get("CarbonIntensity")
            # The chart is titled "replayed EIA trace", so it plots the
            # TRACE value even while an injection is active (the injected
            # effective price lives on the badged card, not in a series
            # that claims to be measured history). No trace value while
            # injected -> the line simply breaks.
            plot_value = (ad.get("CarbonTraceIntensity")
                          if ad.get("CarbonInjected") else intensity)
            if plot_value is not None:
                series = self.carbon_series.setdefault(site, [])
                series.append((now, float(plot_value)))
                del series[:-360]
            sites.append({
                "site": site,
                "machine": ad.get("Machine"),
                "carbon": intensity,
                "injected": bool(ad.get("CarbonInjected")),
                "trace_carbon": ad.get("CarbonTraceIntensity"),
                "gpu_watts": ad.get("GPUWatts"),
                # Replay position plus the three attributes needed to check
                # it: comparing intensities across sites is only meaningful
                # if they are reading the same moment of the trace, and
                # replay_coherence() is what decides whether they are.
                "carbon_ts": ad.get("CarbonTimestamp"),
                "carbon_sampled_at": ad.get("CarbonSampledAt"),
                "carbon_span": ad.get("CarbonSpan"),
                "carbon_accel": ad.get("CarbonAccel"),
                # The two clocks HTCondor stamps on every ad: the startd's
                # own (MyCurrentTime) and the collector's (LastHeardFrom).
                # Their difference IS the node's clock offset, which is the
                # only way to tell a slow clock from a slow update.
                "ad_node_time": ad.get("MyCurrentTime"),
                "ad_heard_at": ad.get("LastHeardFrom"),
                "state": f"{ad.get('State', '?')}/{ad.get('Activity', '?')}",
            })
        sites.sort(key=lambda s: (s["carbon"] is None,
                                  s["carbon"] if s["carbon"] is not None
                                  else 0))

        wf_names = ("train_", "predict_", "evaluate", "report", "stage_")
        jobs = [job_row(ad, queued=True) for ad in queue
                if (ad.get("DAGNodeName") or "").startswith(wf_names)]
        jobs += [job_row(ad) for ad in history
                 if (ad.get("DAGNodeName") or "").startswith(wf_names)]
        jobs.sort(key=lambda j: j["job"])

        progress, current_file = [], None
        for line in sections.get("PROGRESS", "").splitlines():
            line = line.strip()
            if line.startswith("@@"):
                current_file = line[2:]
                continue
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            digits = "".join(c for c in os.path.basename(current_file or "")
                             if c.isdigit())
            entry["segment"] = int(digits) if digits else None
            progress.append(entry)
        progress.sort(key=lambda p: p.get("step", 0))

        # Path scoping is not proof of provenance: every run writes
        # progress_001.json, so two runs staging into one output directory
        # produce indistinguishable, mutually overwriting files. The
        # trainer stamps the workflow uuid into each progress file, and
        # only files bearing THIS run's uuid may contribute energy. A file
        # without one (a trainer predating the stamp) is unverifiable, so
        # it is dropped and counted rather than trusted — the count is
        # surfaced so silently-missing energy is explained, not mysterious.
        current_uuid = sections.get("UUID", "").strip()
        verified, unverified, foreign = [], 0, 0
        for entry in progress:
            stamped = entry.get("wf_uuid")
            if stamped and current_uuid and stamped == current_uuid:
                verified.append(entry)
            elif not stamped:
                unverified += 1
            else:
                foreign += 1
        progress = verified

        attach_measured_energy(jobs, progress)
        train = [j for j in jobs if j["job"].startswith("train_")
                 and j["status"] == "done"]
        measured = [j for j in train if j["measured_wh"] is not None]
        totals = {
            # Sums of measured values only. If NVML was absent somewhere,
            # the count below says how many segments are actually in the
            # sum — a partial measured total beats a complete estimate.
            "energy_wh": round(sum(j["measured_wh"] for j in measured), 1),
            "gco2": round(sum(j["gco2"] for j in measured
                              if j["gco2"] is not None), 2),
            "measured_segments": len(measured),
            "unverified_progress": unverified,
            "foreign_progress": foreign,
            # Segments whose match price was demo-injected: their gCO2 is
            # measured-Wh x a synthetic intensity, and the total must say
            # so rather than blend it in silently.
            "injected_segments": sum(1 for j in train
                                     if j["injected_at_match"]),
            "gb_transferred": round(sum(j["mb_in"] + j["mb_out"]
                                        for j in jobs) / 1000, 2),
            "segments_done": len(train),
            "migrations": sum(
                1 for a, b in zip(train, train[1:])
                if a["site"] != b["site"] and a["site"] != "-"),
        }

        # The final result carries the same run stamp as the progress
        # files, so a result read out of a shared output directory is
        # verified rather than assumed to be this run's.
        result, result_status = None, "pending"
        raw_result = sections.get("RESULT", "").strip()
        if raw_result:
            try:
                candidate = json.loads(raw_result)
            except ValueError:
                result_status = "unreadable"
            else:
                stamped = candidate.get("wf_uuid")
                if stamped and current_uuid and stamped == current_uuid:
                    result, result_status = candidate, "verified"
                elif not stamped:
                    result_status = "unverified"
                else:
                    result_status = "foreign"
        report_lines = sections.get("REPORT", "").split()
        if "PRESENT" not in report_lines:
            report_status = "pending"
        else:
            declared = next((x for x in report_lines if x != "PRESENT"), "")
            if declared and current_uuid and declared == current_uuid:
                report_status = "verified"
            elif not declared:
                report_status = "unverified"
            else:
                report_status = "foreign"
        self.report_status = report_status
        self.outdir = sections.get("OUTDIR", "").strip()

        self.state = {
            "ok": True, "ts": now, "mode": self.backend.name,
            "result": result, "result_status": result_status,
            # Only a report that names THIS run is offered as this run's.
            "report_ready": report_status == "verified",
            "report_status": report_status,
            "sites": sites, "jobs": jobs, "progress": progress,
            "totals": totals, "coherence": replay_coherence(sites, now),
            "carbon_series": {s: v[-120:]
                              for s, v in self.carbon_series.items()},
        }

    def run(self):
        while True:
            try:
                self.poll_once()
            except Exception as exc:                     # keep serving
                self.state = {**self.state, "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}
            self.wake.wait(self.interval)
            self.wake.clear()


class Actions:
    """The two demo controls. Both validate their inputs hard: these run
    shell commands, and the site name comes over HTTP."""

    def __init__(self, backend, poller):
        self.backend = backend
        self.poller = poller
        self.start_status = {"state": "idle"}
        self._start_lock = threading.Lock()

    def _machine_for(self, site):
        for entry in self.poller.state.get("sites", []):
            if entry["site"] == site and entry.get("machine"):
                machine = entry["machine"].split(".")[0]
                if re.fullmatch(r"[A-Za-z0-9-]+", machine):
                    return machine
        return None

    def inject(self, site, gco2, minutes):
        machine = self._machine_for(site)
        if machine is None:
            return {"ok": False, "error": f"unknown site {site!r}"}
        try:
            gco2 = float(gco2)
            minutes = float(minutes)
            if not (0 <= gco2 <= 5000 and 0 < minutes <= 120):
                raise ValueError
        except (TypeError, ValueError):
            return {"ok": False, "error": "gco2 must be 0..5000, "
                                          "minutes 0..120"}
        expiry = int(time.time() + minutes * 60)
        out, err = self.backend.worker_cmd(
            machine,
            f"echo '{gco2:.1f} {expiry}' | sudo tee /opt/carbon/override "
            f">/dev/null && echo INJECT_OK")
        if "INJECT_OK" not in out:
            return {"ok": False, "error": (err or out)[-200:]}
        self.poller.wake.set()
        return {"ok": True, "note": f"{site} advertises {gco2:.0f} within "
                                    f"one cron period (60s), labeled "
                                    f"INJECTED, expires in {minutes:g} min"}

    def clear(self, site):
        machine = self._machine_for(site)
        if machine is None:
            return {"ok": False, "error": f"unknown site {site!r}"}
        out, err = self.backend.worker_cmd(
            machine, "sudo rm -f /opt/carbon/override && echo CLEAR_OK")
        if "CLEAR_OK" not in out:
            return {"ok": False, "error": (err or out)[-200:]}
        self.poller.wake.set()
        return {"ok": True, "note": f"{site} back to the measured trace "
                                    f"within one cron period"}

    def start(self, segments, minutes):
        try:
            segments = int(segments)
            minutes = int(minutes)
            if not (1 <= segments <= 48 and 1 <= minutes <= 120):
                raise ValueError
        except (TypeError, ValueError):
            return {"ok": False, "error": "segments must be 1..48, "
                                          "minutes 1..120"}
        if not self._start_lock.acquire(blocking=False):
            return {"ok": False, "error": "a submission is already running"}

        def run():
            try:
                self.start_status = {"state": "submitting",
                                     "detail": f"{segments}x{minutes}min"}
                script = START_SCRIPT.format(segments=segments,
                                             minutes=minutes,
                                             epoch=int(time.time()))
                out, err = self.backend.submit_cmd(script, timeout=600)
                tail = (out or err).strip().splitlines()[-3:]
                if "SUBMIT_OK" in (out or ""):
                    self.start_status = {"state": "submitted",
                                         "detail": " ".join(tail)}
                else:
                    self.start_status = {"state": "failed",
                                         "detail": " ".join(tail)[-400:]}
            except Exception as exc:
                self.start_status = {"state": "failed",
                                     "detail": f"{type(exc).__name__}: {exc}"}
            finally:
                self._start_lock.release()
                self.poller.wake.set()

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "note": "submitting (wipes the previous run)"}


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>carbon-chaser &mdash; Pegasus run</title>
<style>
:root{
  color-scheme:light;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,.10); --wash:rgba(11,11,11,.035);
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  color-scheme:dark;
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,.10); --wash:rgba(255,255,255,.05);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,.10); --wash:rgba(255,255,255,.05);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
}

*{box-sizing:border-box}
body{margin:0; padding:0 1.25rem 3rem; background:var(--plane); color:var(--ink);
     font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1180px; margin:0 auto}
h1{font-size:1.15rem; font-weight:650; margin:0; letter-spacing:-.01em}
h2{font-size:.8rem; font-weight:650; margin:0; letter-spacing:.06em;
   text-transform:uppercase; color:var(--ink2)}
.sub{color:var(--muted); font-size:.78rem; font-weight:400;
     letter-spacing:0; text-transform:none}

header{display:flex; align-items:center; gap:1rem; flex-wrap:wrap;
       padding:1.1rem 0 .9rem}
header .grow{flex:1}
.pill{display:inline-flex; align-items:center; gap:.4rem; font-size:.75rem;
      color:var(--ink2); background:var(--surface); border:1px solid var(--border);
      border-radius:99px; padding:.22rem .6rem; max-width:100%}
.pill b{font-weight:600; color:var(--ink)}
button{font:inherit; font-size:.8rem; cursor:pointer; color:var(--ink);
       background:var(--surface); border:1px solid var(--border);
       border-radius:6px; padding:.28rem .6rem}
button:hover{background:var(--wash)}
button.go{background:var(--s1); border-color:transparent; color:#fff;
          font-weight:600}
button.go:hover{filter:brightness(1.08)}
input[type=number]{font:inherit; font-size:.8rem; width:4.2rem; color:var(--ink);
  background:var(--surface); border:1px solid var(--border); border-radius:6px;
  padding:.24rem .4rem; font-variant-numeric:tabular-nums}

section{margin-top:1.5rem}
.shead{display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap;
       margin-bottom:.55rem}
.card{background:var(--surface); border:1px solid var(--border);
      border-radius:10px; padding:.85rem 1rem}
.grid{display:grid; gap:.75rem}
.tiles{grid-template-columns:repeat(auto-fit,minmax(min(160px,100%),1fr))}
/* Few tiles should not stretch to fill the row — cap them and pack left. */
.tiles.cap{grid-template-columns:repeat(auto-fit,minmax(min(230px,100%),285px));
           justify-content:start}
/* min() so a column never demands more width than the viewport has —
   without it the whole page scrolls sideways on a phone. */
.sites{grid-template-columns:repeat(auto-fit,minmax(min(215px,100%),1fr))}
.charts{grid-template-columns:repeat(auto-fit,minmax(min(380px,100%),1fr))}

.label{font-size:.74rem; color:var(--muted); line-height:1.35}
.value{font-size:1.55rem; font-weight:650; letter-spacing:-.02em}
/* Prose that teaches the panel. Capped at a readable measure so it does not
   run the full 1180px and become a wall. */
.explain{font-size:.8rem; color:var(--ink2); line-height:1.55;
         margin:0 0 .75rem; max-width:78ch}
.explain b{color:var(--ink); font-weight:650}
/* Second line on a tile: how to READ the number above it (scale, reference
   point, caveat) — a bare 0.8761 tells a reader nothing on its own. */
.hint{font-size:.7rem; color:var(--muted); line-height:1.35; opacity:.85;
      margin-top:.4rem; padding-top:.4rem; border-top:1px solid var(--grid)}
.hero{display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap}
.hero .fig{font-size:3.1rem; font-weight:650; line-height:1;
           letter-spacing:-.03em}
.hero .unit{font-size:1rem; color:var(--ink2); font-weight:500}
.dot{width:9px; height:9px; border-radius:99px; flex:none; display:inline-block}
.chip{display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem;
      color:var(--ink2); white-space:nowrap}
.tag{font-size:.66rem; font-weight:700; letter-spacing:.04em; border-radius:4px;
     padding:.1rem .34rem; white-space:nowrap}
.tag.clean{background:var(--good); color:#fff}
.tag.inj{background:var(--warn); color:#3a2a00}
.card.lead{box-shadow:inset 0 0 0 1px var(--good)}
.ctl{display:flex; align-items:center; gap:.35rem; margin-top:.6rem;
     padding-top:.6rem; border-top:1px solid var(--border)}
.msg{font-size:.8rem; min-height:1.2em; margin:.35rem 0 0; display:flex;
     gap:.6rem}
.msg .err{color:var(--crit)} .msg .ok{color:var(--good)}

.plot{position:relative; margin-top:.35rem}
svg{display:block; width:100%; height:auto; overflow:visible}
.legend{display:flex; gap:.85rem; flex-wrap:wrap; margin-top:.15rem}
.empty{font-size:.78rem; color:var(--muted); padding:1.4rem 0}

.tblwrap{overflow-x:auto; -webkit-overflow-scrolling:touch}
table{border-collapse:collapse; width:100%; font-size:.78rem;
      font-variant-numeric:tabular-nums}
th{text-align:right; font-weight:600; color:var(--muted); font-size:.72rem;
   white-space:nowrap; padding:.3rem .55rem;
   border-bottom:1px solid var(--axis)}
td{text-align:right; padding:.3rem .55rem; border-bottom:1px solid var(--grid);
   white-space:nowrap}
td:first-child,th:first-child{text-align:left; font-variant-numeric:normal}
td:nth-child(2),th:nth-child(2),td:nth-child(3),th:nth-child(3){text-align:left;
   font-variant-numeric:normal}
tr.live td{background:var(--wash)}
a{color:var(--s1)}

#tip{position:fixed; z-index:9; display:none; pointer-events:none;
     background:var(--surface); border:1px solid var(--border);
     border-radius:8px; padding:.4rem .55rem; font-size:.75rem;
     box-shadow:0 6px 20px rgba(0,0,0,.14); max-width:16rem}
#tip .th{font-weight:650; margin-bottom:.2rem}
#tip .row{display:flex; align-items:center; gap:.35rem;
          font-variant-numeric:tabular-nums}
#tip .row .n{margin-left:auto; font-weight:600}
</style></head><body>
<div class="wrap">

<header>
  <div>
    <h1>carbon-chaser</h1>
    <div class="sub">a Pegasus workflow placed on the cleanest grid, measured</div>
  </div>
  <div class="grow"></div>
  <span class="pill" id="clock">connecting&hellip;</span>
  <span class="pill" id="replay" hidden></span>
  <button id="theme" title="switch light / dark">&#9681; theme</button>
</header>

<div class="card" style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap">
  <b style="font-size:.8rem">Start workflow</b>
  <input id="segs" type="number" value="6" min="1" max="48"> <span class="label">segments &times;</span>
  <input id="mins" type="number" value="2" min="1" max="120"> <span class="label">min each</span>
  <button class="go" id="start">&#9654;&nbsp; submit</button>
  <span class="label" id="startstat"></span>
</div>
<div class="msg"><span class="err" id="err"></span>
  <span class="ok" id="note"></span></div>

<section>
  <div class="shead"><h2>This run</h2>
    <span class="sub">every number below is measured, or shown as unknown</span></div>
  <div class="grid tiles">
    <div class="card">
      <div class="label">CO&#8322; emitted so far</div>
      <div class="hero"><span class="fig" id="hero">&ndash;</span>
        <span class="unit">gCO&#8322;</span></div>
      <div class="label" id="heronote" style="margin-top:.35rem">
        NVML energy &times; grid intensity at match</div>
    </div>
    <div class="card"><div class="value" id="t_seg">&ndash;</div>
      <div class="label">segments finished</div></div>
    <div class="card"><div class="value" id="t_mig">&ndash;</div>
      <div class="label">migrations &mdash; a segment resuming at another site</div></div>
    <div class="card"><div class="value" id="t_wh">&ndash;</div>
      <div class="label" id="t_wh_note">Wh of GPU energy (NVML integral)</div></div>
    <div class="card"><div class="value" id="t_gb">&ndash;</div>
      <div class="label">GB moved by HTCondor file transfer</div></div>
  </div>
</section>

<section>
  <div class="shead"><h2>Pool</h2>
    <span class="sub">matchmaking inputs &mdash; HTCondor ranks on
      <code>RANK = -CarbonIntensity</code>, so the cleanest site wins the next segment</span></div>
  <div class="grid sites" id="sites"><div class="empty">waiting for
    <code>condor_status</code>&hellip;</div></div>
</section>

<section class="grid charts">
  <div class="card">
    <div class="shead"><h2>Grid carbon intensity</h2>
      <span class="sub">replayed EIA trace, gCO&#8322;/kWh</span></div>
    <div class="legend" id="tracekey"></div>
    <div class="plot"><svg id="trace"></svg>
      <div class="empty" id="traceempty" hidden>no trace samples yet</div></div>
  </div>
  <div class="card">
    <div class="shead"><h2>Validation AUC per segment</h2>
      <span class="sub">colour = site that trained it; a colour change is a migration</span></div>
    <div class="legend" id="auckey"></div>
    <div class="plot"><svg id="auc"></svg>
      <div class="empty" id="aucempty" hidden>the first segment publishes AUC
        as soon as it checkpoints</div></div>
  </div>
</section>

<section>
  <div class="shead"><h2>Science output</h2>
    <span class="sub">did the physics survive being moved around?</span></div>
  <p class="explain">The workload is a real classification task, not a
    benchmark loop. Each segment trains one neural net on the UCI
    <b>HIGGS</b> set &mdash; 11&nbsp;million simulated LHC collisions,
    roughly half from a process that produces Higgs bosons (<b>signal</b>)
    and roughly half from one that looks almost identical but does not
    (<b>background</b>). The net sees 21 raw
    detector quantities per collision and outputs P(signal); the tiles below
    score it on held-out events it never trained on. That is the point of the
    demo: chasing clean power can move the job between sites mid-training,
    and these numbers are how you check what that cost. For
    reference, Baldi&nbsp;et&nbsp;al. (<i>Nature Communications</i> 5:4308,
    2014) reach <b>AUC &asymp; 0.88</b> from the same 21 features.</p>
  <div class="grid tiles cap" id="science"></div>
  <div class="label" id="reportlink" style="margin-top:.6rem"></div>
</section>

<section>
  <div class="shead"><h2>Jobs</h2>
    <span class="sub">site, carbon &amp; power recorded in the job ad at match
      time, and the bytes HTCondor actually moved</span></div>
  <div class="card"><div class="tblwrap"><table id="jobs"></table></div></div>
</section>

</div>
<div id="tip"></div>
<script>
/* ---------- helpers ------------------------------------------------- */
const ENT = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};
function esc(v){
  return v === null || v === undefined || v === ""
    ? "–" : String(v).replace(/[&<>"']/g, c => ENT[c]);
}
function num(v, d){
  return v === null || v === undefined || Number.isNaN(Number(v))
    ? "–" : Number(v).toFixed(d === undefined ? 0 : d);
}
const $ = id => document.getElementById(id);

/* Colour follows the SITE, never its current rank: slots are handed out
   once, in sorted-name order, and never reassigned — so filtering or a
   re-sorted pool cannot repaint a site the reader has already learned. */
const SLOT = new Map();
function slot(site){
  if (!SLOT.has(site)) SLOT.set(site, SLOT.size);
  return SLOT.get(site);
}
function palette(){
  const cs = getComputedStyle(document.documentElement);
  return [1,2,3,4,5].map(i => cs.getPropertyValue("--s" + i).trim());
}
function ink(name){
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
}
function colour(site){ const p = palette(); return p[slot(site) % p.length]; }
function key(names, into){
  into.innerHTML = names.length < 2 ? "" : names.map(n =>
    `<span class="chip"><span class="dot" style="background:${colour(n)}"></span>
     ${esc(n)}</span>`).join("");
}

/* ---------- theme --------------------------------------------------- */
const root = document.documentElement;
if (localStorage.ccTheme) root.dataset.theme = localStorage.ccTheme;
$("theme").onclick = () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const now = root.dataset.theme || (dark ? "dark" : "light");
  root.dataset.theme = localStorage.ccTheme = now === "dark" ? "light" : "dark";
  if (LAST) render(LAST);
};

/* ---------- tooltip ------------------------------------------------- */
const tip = $("tip");
function showTip(evt, html){
  tip.innerHTML = html;
  tip.style.display = "block";
  const box = tip.getBoundingClientRect();
  tip.style.left = Math.max(6, Math.min(innerWidth - box.width - 6,
                                        evt.clientX + 14)) + "px";
  tip.style.top = (evt.clientY - box.height - 14 < 4
    ? evt.clientY + 18 : evt.clientY - box.height - 14) + "px";
}
function hideTip(){ tip.style.display = "none"; }

/* ---------- controls ------------------------------------------------ */
async function api(path, body){
  let r;
  try {
    r = await (await fetch(path, {method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body)})).json();
  } catch (e) { $("err").textContent = String(e); return; }
  $("note").textContent = r.ok ? (r.note || "ok") : "";
  $("err").textContent = r.ok ? "" : esc(r.error);
  setTimeout(tick, 800);
}
$("start").onclick = () => api("/api/start",
  {segments: $("segs").value, minutes: $("mins").value});
$("sites").addEventListener("click", ev => {
  const b = ev.target.closest("button[data-site]");
  if (!b) return;
  const site = b.dataset.site;
  if (b.dataset.act === "clear") return api("/api/clear", {site});
  const field = $("sites").querySelector(`input[data-site="${site}"]`);
  api("/api/inject", {site, gco2: field.value, minutes: 10});
});

/* ---------- render -------------------------------------------------- */
let LAST = null;
async function tick(){
  let s;
  try { s = await (await fetch("/state")).json(); }
  catch (e) { $("err").textContent = String(e); return; }
  if (!s.ok && s.error) $("err").textContent = esc(s.error);
  if (!s.sites) return;
  LAST = s;
  render(s);
}
addEventListener("resize", () => { if (LAST) render(LAST); });

function render(s){
  $("clock").innerHTML = "updated <b>" +
    new Date((s.ts || 0) * 1000).toLocaleTimeString() + "</b> · " +
    esc(s.mode);
  const st = s.start_status || {};
  $("startstat").textContent = !st.state || st.state === "idle" ? ""
    : st.state + (st.detail ? ": " + st.detail : "");

  [...new Set((s.sites || []).map(x => x.site))].sort().forEach(slot);
  renderReplay(s.coherence);
  renderSites(s.sites || []);
  renderTotals(s.totals || {});
  renderScience(s);
  renderJobs(s.jobs || []);
  drawTrace(s.carbon_series || {});
  drawAuc((s.progress || []).filter(p => p.auc != null));
}

/* One replay clock shared by every site is what makes the sites comparable
   at all, so say plainly whether they agree. The verdict is computed on the
   server by replay_coherence() — the same per-node, on-a-circle test
   check_pool_match.py uses, NOT cross-node equality of CarbonTimestamp
   (healthy unsynchronised crons differ, and so do nodes either side of the
   replay wrap). Hover for the full reasoning. */
const COH_DOT = {ok: "--good", warn: "--warn", bad: "--crit",
                 thin: "--muted", none: "--muted"};
function renderReplay(c){
  const el = $("replay");
  el.hidden = !c;
  if (!c) return;
  el.title = c.detail || "";
  el.innerHTML = `<span class="dot" style="background:var(${
    COH_DOT[c.state] || "--muted"})"></span>${esc(c.headline)}`;
}

function renderSites(sites){
  if (!sites.length){
    $("sites").innerHTML = '<div class="empty">no workers advertising</div>';
    return;
  }
  $("sites").innerHTML = sites.map((x, i) => {
    const lead = i === 0 && x.carbon != null;
    const tags = (lead ? '<span class="tag clean">CLEANEST</span>' : "") +
      (x.injected ? `<span class="tag inj" title="carbon overridden for the
        demo; the trace says ${num(x.trace_carbon)}">INJECTED</span>` : "");
    const ctl = x.carbon == null ? "" : `<div class="ctl">
      <input type="number" data-site="${esc(x.site)}" value="900"
             min="0" max="5000" title="gCO2/kWh to advertise for 10 min">
      <button data-site="${esc(x.site)}" data-act="inject">inject</button>
      <button data-site="${esc(x.site)}" data-act="clear">clear</button></div>`;
    return `<div class="card ${lead ? "lead" : ""}">
      <div class="chip" style="justify-content:flex-start">
        <span class="dot" style="background:${colour(x.site)}"></span>
        <b style="color:var(--ink);font-size:.9rem">${esc(x.site)}</b>${tags}</div>
      <div class="value" style="margin-top:.35rem">${num(x.carbon)}
        <span class="label" style="font-size:.72rem">gCO&#8322;/kWh</span></div>
      <div class="label">${num(x.gpu_watts)} W GPU · ${esc(x.state)}</div>
      ${ctl}</div>`;
  }).join("");
}

function renderTotals(t){
  // Nothing measured yet is not "0 gCO2": an unmeasured total is unknown.
  $("hero").textContent = t.measured_segments ? num(t.gco2, 1) : "–";
  const notes = [];
  if (t.injected_segments) notes.push(
    `${t.injected_segments} segment(s) priced by an INJECTED intensity`);
  if (t.unverified_progress) notes.push(
    `${t.unverified_progress} progress file(s) carry no run id — not counted`);
  if (t.foreign_progress) notes.push(
    `${t.foreign_progress} from another run — ignored`);
  $("heronote").innerHTML = "NVML energy &times; grid intensity at match" +
    (notes.length ? "<br>" + notes.map(esc).join("<br>") : "");
  $("t_seg").textContent = num(t.segments_done);
  $("t_mig").textContent = num(t.migrations);
  $("t_wh").textContent = num(t.energy_wh, 1);
  $("t_wh_note").textContent = "Wh of GPU energy (NVML integral, " +
    `${t.measured_segments ?? 0}/${t.segments_done ?? 0} segments measured)`;
  $("t_gb").textContent = num(t.gb_transferred, 2);
}

/* Each tile is [value, what it measures, how to read it]. The third slot is
   not decoration: AUC and "signal efficiency" mean nothing without a scale
   or a reference point, and this panel is the one a physicist and a
   sysadmin read side by side. */
function renderScience(s){
  const prog = (s.progress || []).filter(p => p.auc != null);
  const r = s.result, cards = [];
  const pct = v => v === null || v === undefined || Number.isNaN(Number(v))
    ? "–" : (Number(v) * 100).toFixed(1) + "%";

  /* Whether this run has actually moved, and whether its chain held, are
     facts about the run rather than the demo's intent — the tiles must not
     narrate migrations that never happened or a resume that never worked.
     Re-sorted by SEGMENT because the poller hands progress over sorted by
     step, and a segment that restarted its counter is exactly the case that
     sort would reorder into looking contiguous. */
  const chain = (s.progress || [])
    .filter(p => p.segment != null && p.step != null)
    .slice().sort((a, b) => a.segment - b.segment);
  const hosts = chain.map(p => (p.host || "").split("-")[0]).filter(Boolean);
  const moves = hosts.reduce((n, h, i) => n + (i && h !== hosts[i - 1]), 0);

  /* Continuity is only checkable across ADJACENT reported segments, and this
     panel does not always see them all: the poller tails the last 24
     progress files and drops any that fail the run-id check, so the chain
     can start late or skip. A chain that is empty, holds one entry, or has a
     hole cannot support the claim — and note that .every() on an empty array
     is TRUE, which is precisely how "no evidence" turns into "verified
     continuous" if the length is not checked first. */
  const segs = chain.map(p => p.segment);
  const gaps = segs.filter((v, i) => i && v !== segs[i - 1] + 1).length
             + (segs.length && segs[0] !== 1 ? 1 : 0);
  const checkable = chain.length > 1 && !gaps;
  /* A regression at a boundary whose BOTH sides reported is proven, and a
     hole elsewhere cannot un-prove it. So completeness gates only the
     positive claim — never the failure, which would let an incomplete chain
     hide a restart behind "cannot be checked". */
  const restartAt = chain.filter((p, i) => i
      && p.segment === chain[i - 1].segment + 1
      && p.step <= chain[i - 1].step).map(p => p.segment);
  const restarted = restartAt.length > 0;
  const unbroken = checkable && !restarted
    && chain.every((p, i) => !i || p.step > chain[i - 1].step);

  // Mid-run this is the only score there is; once the evaluator has spoken
  // it would just repeat the final AUC in a weaker form, so it steps aside.
  if (prog.length && !r){
    const last = prog[prog.length - 1];
    cards.push([last.auc.toFixed(4),
      `AUC so far &mdash; how well the net separates signal from background
       after segment ${last.segment ?? "?"}
       (${(last.step || 0).toLocaleString()} training steps)`,
      `the trainer publishes this every time it checkpoints, so it updates
       while the run is live &mdash; the chart to the left is the same number
       per segment, coloured by the site that earned it`]);
  }
  if (r){
    const gap = r.val_auc === null || r.val_auc === undefined
      ? null : Number(r.val_auc) - 0.88;
    cards.push([num(r.val_auc, 4),
      `final AUC, scored on ${(r.val_events || 0).toLocaleString()} held-out
       collisions the net never saw`,
      `1.00 = perfect separation, 0.50 = a coin flip` + (gap === null ? "" :
        ` &middot; ${gap >= 0 ? "+" : "&minus;"}${Math.abs(gap).toFixed(3)}
          vs the paper's 0.88`)]);

    const eff = r["sig_eff_at_bkg_rej_0.99"];
    if (eff !== null && eff !== undefined){
      // The other two operating points ride along as context rather than as
      // three near-identical tiles.
      const others = [["0.9", "90%"], ["0.999", "99.9%"]]
        .filter(([k]) => r["sig_eff_at_bkg_rej_" + k] !== undefined
                      && r["sig_eff_at_bkg_rej_" + k] !== null)
        .map(([k, at]) => `${pct(r["sig_eff_at_bkg_rej_" + k])} at ${at}`);
      cards.push([pct(eff),
        `of the real Higgs events are still kept when the cut is tightened
         until it discards 99% of background &mdash; the number a physics
         analysis actually quotes`,
        others.length
          ? `the same model at looser and tighter cuts: ${others.join(", ")}`
          : `a tighter cut buys purity by throwing away real signal too`]);
    }
    // Only the verified-contiguous case may call this the chain's total.
    const sites = moves
      ? `, carried across ${moves} site change${moves > 1 ? "s" : ""}`
      : `, and so far it has stayed on a single site`;
    cards.push([(r.trained_steps || 0).toLocaleString(),
      unbroken ? `gradient steps, summed across every segment of the chain`
               : `gradient steps behind the final checkpoint`,
      unbroken
        ? `every segment resumed from the previous one's checkpoint, so this
           counter never restarts &mdash; the model behind the AUC above is
           one continuous training run` + sites
        : restarted
        ? `<b>not one continuous run</b> &mdash; the step counter falls back
           at segment ${restartAt.join(", ")}, so that segment started over
           instead of resuming. Earlier training is not in this number` +
          (checkable ? `` : `, and the other boundaries are not covered here,
           so this may not be the only one`)
        : `whether every segment resumed from its predecessor cannot be
           checked here: ` + (chain.length < 2
            ? (chain.length
                ? `only 1 segment has published progress to this panel so far`
                : `no segment has published progress to this panel yet`)
            : `the ${chain.length} segments visible here do not run unbroken
               from segment 1`) +
          `, so this is the final checkpoint's own step count and nothing is
           claimed about the hand-offs`]);
  } else if (s.result_status && s.result_status !== "pending"){
    cards.push(["–", `final result ${esc(s.result_status)} &mdash; not shown`,
      `the evaluator refuses to print a score for a checkpoint it cannot
       verify belongs to this run`]);
  }
  $("science").innerHTML = cards.length
    ? cards.map(([v, l, h]) => `<div class="card"><div class="value">${v}</div>
        <div class="label">${l}</div>
        ${h ? `<div class="hint">${h}</div>` : ""}</div>`).join("")
    : `<div class="empty">no verified science output yet &mdash; the first
        segment publishes an AUC as soon as it checkpoints</div>`;
  $("reportlink").innerHTML = s.report_ready
    ? '<a href="/report" target="_blank">open the report the workflow itself ' +
      'rendered &mdash; ROC, score distributions, AUC by site &#8599;</a>'
    : s.report_status === "foreign"
      ? "a report sits in this run's output directory but names a different " +
        "workflow run — not shown"
      : s.report_status === "unverified"
      ? "a report exists but declares no run id — not shown"
      : "the report is the workflow's last job; the link appears here once " +
        "it exists and names this run";
}

const STATUS = {
  done:   ["--good", "done"],      running: ["--warn", "running"],
  transferring: ["--warn", "transferring"], held: ["--crit", "held"],
  removed: ["--crit", "removed"],  idle: ["--muted", "idle"],
  suspended: ["--muted", "suspended"],
};
function renderJobs(jobs){
  const head = ["job", "status", "site", "wall", "gCO₂/kWh @match",
    "W @match", "Wh measured", "gCO₂", "MB in", "MB out", "exit"];
  const rows = jobs.map(j => {
    let [tok, text] = STATUS[j.status] || ["--muted", j.status];
    if (j.status === "done" && j.exit) { tok = "--crit"; text = "failed"; }
    const carbon = num(j.carbon_at_match) + (j.injected_at_match
      ? ' <span class="tag inj" title="matched against a demo-injected price">INJ</span>'
      : "");
    const live = ["running", "transferring"].includes(j.status) ? "live" : "";
    return `<tr class="${live}"><td>${esc(j.job)}</td>
      <td><span class="chip"><span class="dot" style="background:var(${tok})">
        </span>${esc(text)}</span></td>
      <td><span class="chip"><span class="dot" style="background:${
        j.site && j.site !== "-" ? colour(j.site) : "var(--axis)"
        }"></span>${esc(j.site)}</span></td>
      <td>${j.wall_s}s</td><td>${carbon}</td>
      <td>${num(j.gpu_watts_at_match)}</td><td>${num(j.measured_wh, 1)}</td>
      <td>${num(j.gco2, 2)}</td><td>${num(j.mb_in, 1)}</td>
      <td>${num(j.mb_out, 1)}</td><td>${esc(j.exit)}</td></tr>`;
  }).join("");
  $("jobs").innerHTML = "<tr>" + head.map(h => `<th>${h}</th>`).join("") +
    "</tr>" + (rows || `<tr><td colspan="${head.length}"
      style="color:var(--muted)">no jobs for the current run yet</td></tr>`);
}

/* ---------- charts -------------------------------------------------- */
/* Shared chrome: hairline solid grid, recessive axes, labels in text
   tokens (never in a series colour), 2px lines, 8px markers ringed in
   the surface colour so overlaps stay legible. */
function frame(svg, h, pad){
  const w = Math.max(320, svg.parentNode.clientWidth);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("height", h);
  return {w, h, pad};
}
function axes(g, ticks, xlabels){
  const {w, h, pad} = g;
  let out = "";
  ticks.forEach(([y, text]) => {
    out += `<line x1="${pad.l}" y1="${y}" x2="${w - pad.r}" y2="${y}"
            stroke="var(--grid)" stroke-width="1"/>
            <text x="${pad.l - 8}" y="${y + 4}" text-anchor="end"
            font-size="10.5" fill="var(--muted)">${text}</text>`;
  });
  out += `<line x1="${pad.l}" y1="${h - pad.b}" x2="${w - pad.r}"
          y2="${h - pad.b}" stroke="var(--axis)" stroke-width="1"/>`;
  xlabels.forEach(([x, text, anchor]) => {
    out += `<text x="${x}" y="${h - pad.b + 15}" font-size="10.5"
            text-anchor="${anchor || "middle"}" fill="var(--muted)">${text}</text>`;
  });
  return out;
}
function niceTicks(lo, hi, n){
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag)
    .find(v => v >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    out.push(v);
  return out;
}

function drawTrace(series){
  const svg = $("trace");
  const names = Object.keys(series).filter(n => series[n].length).sort();
  const pts = names.flatMap(n => series[n]);
  $("traceempty").hidden = pts.length > 0;
  key(names, $("tracekey"));
  if (!pts.length){ svg.innerHTML = ""; svg.removeAttribute("height"); return; }

  const g = frame(svg, 232, {t:14, r:66, b:26, l:46});
  const {w, h, pad} = g;
  const vals = pts.map(p => p[1]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const span = (hi - lo) || Math.max(hi * 0.1, 1);
  lo -= span * 0.15; hi += span * 0.15;
  const t0 = Math.min(...pts.map(p => p[0])), t1 = Math.max(...pts.map(p => p[0]));
  const X = t => pad.l + (t - t0) / Math.max(t1 - t0, 1) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - (v - lo) / (hi - lo) * (h - pad.t - pad.b);

  let out = axes(g, niceTicks(lo, hi, 4).map(v => [Y(v), num(v)]),
    [[pad.l, new Date(t0 * 1000).toLocaleTimeString(), "start"],
     [w - pad.r, new Date(t1 * 1000).toLocaleTimeString(), "end"]]);
  names.forEach(n => {
    const c = colour(n), pl = series[n];
    out += `<polyline points="${pl.map(([t, v]) => `${X(t)},${Y(v)}`).join(" ")}"
            fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"
            stroke-linecap="round"/>`;
    const [lt, lv] = pl[pl.length - 1];
    out += `<circle cx="${X(lt)}" cy="${Y(lv)}" r="4" fill="${c}"
            stroke="var(--surface)" stroke-width="2"/>`;
    if (names.length <= 5)
      out += `<text x="${X(lt) + 9}" y="${Y(lv) + 4}" font-size="10.5"
              fill="var(--ink2)">${esc(n)} ${num(lv)}</text>`;
  });
  out += `<line id="tcross" x1="0" y1="${pad.t - 6}" x2="0" y2="${h - pad.b}"
          stroke="var(--axis)" stroke-width="1" visibility="hidden"/>
          <rect x="${pad.l}" y="0" width="${w - pad.l - pad.r}" height="${h}"
          fill="transparent" id="thit"/>`;
  svg.innerHTML = out;

  const hit = svg.querySelector("#thit"), cross = svg.querySelector("#tcross");
  hit.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); hideTip(); };
  hit.onmousemove = ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) / box.width * w;
    const t = t0 + (px - pad.l) / Math.max(w - pad.l - pad.r, 1) * (t1 - t0);
    cross.setAttribute("x1", X(t)); cross.setAttribute("x2", X(t));
    cross.setAttribute("visibility", "visible");
    const rows = names.map(n => {
      const near = series[n].reduce((a, b) =>
        Math.abs(b[0] - t) < Math.abs(a[0] - t) ? b : a);
      return `<div class="row"><span class="dot" style="background:${
        colour(n)}"></span>${esc(n)}<span class="n">${num(near[1])}</span></div>`;
    }).join("");
    showTip(ev, `<div class="th">${new Date(t * 1000).toLocaleTimeString()}
      · gCO₂/kWh</div>${rows}`);
  };
}

function drawAuc(prog){
  const svg = $("auc");
  $("aucempty").hidden = prog.length > 0;
  const siteOf = p => (p.host || "?").split("-")[0];
  const names = [...new Set(prog.map(siteOf))];
  key(names, $("auckey"));
  if (!prog.length){ svg.innerHTML = ""; svg.removeAttribute("height"); return; }

  const g = frame(svg, 232, {t:14, r:56, b:26, l:52});
  const {w, h, pad} = g;
  const aucs = prog.map(p => p.auc);
  let lo = Math.min(...aucs), hi = Math.max(...aucs);
  const span = (hi - lo) || 0.01;
  lo -= span * 0.25; hi += span * 0.25;
  const X = i => pad.l + (prog.length < 2 ? (w - pad.l - pad.r) / 2
    : i / (prog.length - 1) * (w - pad.l - pad.r));
  const Y = v => h - pad.b - (v - lo) / (hi - lo) * (h - pad.t - pad.b);

  const step = Math.ceil(prog.length / 8);
  let out = axes(g, niceTicks(lo, hi, 4).map(v => [Y(v), v.toFixed(3)]),
    prog.filter((p, i) => i % step === 0)
        .map((p, k) => [X(k * step), esc(p.segment ?? "?")]));
  // One polyline per adjacent pair, in the colour of the site that trained
  // the later segment: a colour change on the line IS the migration.
  for (let i = 1; i < prog.length; i++)
    out += `<polyline points="${X(i-1)},${Y(aucs[i-1])} ${X(i)},${Y(aucs[i])}"
            fill="none" stroke="${colour(siteOf(prog[i]))}" stroke-width="2"
            stroke-linecap="round"/>`;
  prog.forEach((p, i) => {
    out += `<circle cx="${X(i)}" cy="${Y(p.auc)}" r="4"
            fill="${colour(siteOf(p))}" stroke="var(--surface)"
            stroke-width="2"/>`;
  });
  const last = prog[prog.length - 1];
  out += `<text x="${X(prog.length - 1) + 9}" y="${Y(last.auc) + 4}"
          font-size="10.5" fill="var(--ink2)">${last.auc.toFixed(4)}</text>`;
  // A pinpoint 8px dot is a bad hover target, so each point gets a
  // transparent 24px hit circle on top of it.
  out += prog.map((p, i) => `<circle cx="${X(i)}" cy="${Y(p.auc)}" r="12"
          fill="transparent" data-i="${i}" class="hit"/>`).join("");
  svg.innerHTML = out;

  svg.querySelectorAll("circle.hit").forEach(node => {
    const p = prog[Number(node.dataset.i)];
    node.onmousemove = ev => showTip(ev,
      `<div class="th">segment ${esc(p.segment)}</div>
       <div class="row"><span class="dot" style="background:${
         colour(siteOf(p))}"></span>${esc(p.host)}
         <span class="n">${p.auc.toFixed(4)} AUC</span></div>
       <div class="row">${(p.step || 0).toLocaleString()} steps
         <span class="n">${num(p.gpu_wh, 1)} Wh</span></div>`);
    node.onmouseleave = hideTip;
  });
}

tick(); setInterval(tick, 5000);
</script></body></html>
"""


def serve(poller, actions, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload, ctype="application/json"):
            body = (payload if isinstance(payload, bytes)
                    else json.dumps(payload).encode())
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                self._send({**poller.state,
                            "start_status": actions.start_status})
            elif self.path == "/report":
                # Verify the bytes being served, not a cached verdict:
                # one fetch returns the current run id AND the report, and
                # the report must name that run. Nothing here trusts the
                # poller's last-seen status or a client-supplied path.
                out, _ = poller.backend.submit_cmd(REPORT_SCRIPT, timeout=120)
                fetched = parse_sections(out or "")
                current = fetched.get("UUID", "").strip()
                html = fetched.get("HTML", "")
                declared = REPORT_RUN_ID.search(html)
                if not html.strip():
                    reason = ("not produced yet — the report is the "
                              "workflow's last job")
                elif not declared:
                    reason = ("present but declaring no run id (produced "
                              "before reports were stamped)")
                elif not current or current == "no-current-run":
                    reason = "there is no current run to attribute it to"
                elif declared.group(1) != current:
                    reason = (f"it names workflow {declared.group(1)}, not "
                              f"the current run {current}")
                else:
                    self._send(html.encode(), "text/html; charset=utf-8")
                    return
                self._send(
                    (f"<p>no verified report for the current run: {reason}. "
                     f"A report is served only when the bytes themselves "
                     f"name this run — existence in the output directory is "
                     f"not provenance.</p>").encode(),
                    "text/html; charset=utf-8")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = {}
            if self.path == "/api/inject":
                result = actions.inject(body.get("site"), body.get("gco2"),
                                        body.get("minutes", 10))
            elif self.path == "/api/clear":
                result = actions.clear(body.get("site"))
            elif self.path == "/api/start":
                result = actions.start(body.get("segments"),
                                       body.get("minutes"))
            else:
                result = {"ok": False, "error": "unknown endpoint"}
            self._send(result)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "local", "fablib"],
                    default="auto",
                    help="local = running ON the submit node (stdlib only); "
                         "fablib = from a laptop via the FABRIC API")
    ap.add_argument("--slice-name", default="carbon-chaser-pegasus")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--interval", type=int, default=10,
                    help="seconds between polls")
    args = ap.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "local" if shutil.which("condor_status") else "fablib"
    backend = (LocalBackend() if mode == "local"
               else FablibBackend(args.slice_name))
    poller = Poller(backend, args.interval)
    poller.start()
    actions = Actions(backend, poller)
    print(f"dashboard on http://localhost:{args.port} (mode={mode}, "
          f"poll every {args.interval}s). Controls: per-site carbon "
          f"injection (disclosed, expiring) + start-workflow.", flush=True)
    serve(poller, actions, args.port)


if __name__ == "__main__":
    main()
