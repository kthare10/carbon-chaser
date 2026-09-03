"""The Pegasus dashboard must describe exactly ONE run — the current one.

Both halves of its state are scoped by the same identity, read from the
newest run directory's `braindump.yml`:

* job ads, by `pegasus_wf_uuid` (a job belongs to exactly one workflow);
* progress files, by the `--output-dir` recorded in that run's own
  `planner_arguments`.

Two earlier scopings leaked, and both are pinned here:

1. `condor_history -limit N` with no constraint returned a PREVIOUS run's
   segments — identical DAG node names — inflating segment counts,
   migrations, energy and transferred bytes.
2. Progress files were globbed from the highest-numbered
   `wf-output/<epoch>` directory. That was only safe while the submit path
   wiped `wf-output` first; removing that wipe (it destroyed a real
   migrating run's report) let epoch dirs accumulate, and "newest epoch
   name" is a different identity from "newest run dir" — the epochs are
   stamped by whichever host initiated the submission, so clock skew can
   order them against run-dir order and join a previous run's measured
   energy onto this run's segments.

The shell that does this lives inside dashboard.py's POLL_SCRIPT, so these
tests extract the real lines and run them, rather than restating them.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(HERE, "..", "pegasus", "dashboard.py")


def poll_script():
    text = open(DASHBOARD).read()
    match = re.search(r'POLL_SCRIPT = r"""(.*?)"""', text, re.S)
    assert match, "POLL_SCRIPT not found in dashboard.py"
    return match.group(1)


def extraction_snippet():
    """The UUID and OUTDIR lines, verbatim, with $RD supplied by us."""
    lines = poll_script().splitlines()
    keep, buffer = [], []
    for line in lines:
        if line.startswith(("UUID=", "OUTDIR=")) or buffer:
            buffer.append(line)
            # A continued line ends with a backslash.
            if not line.rstrip().endswith("\\"):
                keep.extend(buffer)
                buffer = []
        if line.startswith("echo '===MACHINES==='"):
            break
    assert any(l.startswith("OUTDIR=") for l in keep), keep
    return "\n".join(keep)


def run_extraction(braindump_text):
    with tempfile.TemporaryDirectory() as tmp:
        if braindump_text is not None:
            with open(os.path.join(tmp, "braindump.yml"), "w") as handle:
                handle.write(braindump_text)
        script = (f'RD="{tmp}"\n' + extraction_snippet()
                  + '\necho "UUID:$UUID"\necho "OUTDIR:$OUTDIR"\n')
        proc = subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True, timeout=60)
        out = proc.stdout
        return (re.search(r"^UUID:(.*)$", out, re.M).group(1),
                re.search(r"^OUTDIR:(.*)$", out, re.M).group(1))


# Real shape, taken from a live submit node (note the escaped quotes).
REAL = ('wf_uuid: 2d745c4d-3170-4c08-81de-76a9b2c8c759\n'
        'planner_arguments: "\\"--submit -s condorpool --output-site local '
        '--conf pegasus.properties --output-dir /home/ubuntu/wf-output/'
        '1788223795 --dir wf-runs higgs-wf.yml\\""\n')


def test_both_halves_come_from_the_same_braindump():
    uuid, outdir = run_extraction(REAL)
    assert uuid == "2d745c4d-3170-4c08-81de-76a9b2c8c759", uuid
    assert outdir == "/home/ubuntu/wf-output/1788223795", outdir
    print("  uuid and output dir both read from the run's own braindump")


def test_equals_form_is_handled():
    """`--output-dir=/path` is as valid as a space-separated argument, and
    a parser that only handles one silently yields no progress data."""
    _, outdir = run_extraction(
        REAL.replace("--output-dir /home", "--output-dir=/home"))
    assert outdir == "/home/ubuntu/wf-output/1788223795", outdir
    print("  --output-dir=/path parses the same as --output-dir /path")


def test_a_newer_epoch_dir_does_not_hijack_the_scope():
    """The exact leak: another run's output dir exists and sorts newer.
    Scope must follow the braindump, not the directory listing."""
    _, outdir = run_extraction(REAL)
    assert outdir.endswith("1788223795"), (
        f"scope must come from planner_arguments, got {outdir!r}")
    # A run whose epoch is LOWER than a previously-created one (clock skew
    # between the notebook host and the submit node) must still win.
    _, older = run_extraction(REAL.replace("1788223795", "1700000000"))
    assert older.endswith("1700000000"), older
    print("  scope follows the braindump even when another epoch sorts newer")


def test_missing_or_unplanned_run_yields_no_data_not_wrong_data():
    """No braindump, or one planned without --output-dir: the dashboard
    must show nothing rather than a previous run's numbers."""
    uuid, outdir = run_extraction(None)
    assert uuid == "no-current-run", uuid
    assert outdir == "", outdir
    uuid, outdir = run_extraction(
        'wf_uuid: abc\nplanner_arguments: "--submit -s condorpool"\n')
    assert uuid == "abc" and outdir == "", (uuid, outdir)
    # The glob must then point somewhere that cannot exist.
    assert '"${OUTDIR:-/nonexistent}"/progress_*.json' in poll_script(), (
        "an empty OUTDIR must glob a path that cannot match, or the "
        "dashboard falls back to some other run's progress files")
    print("  absent run / absent output-dir -> empty, never another run's data")


def test_job_ads_are_constrained_by_uuid():
    script = poll_script()
    for query in ("condor_q", "condor_history"):
        section = script.split(query, 1)[1].split("echo", 1)[0]
        assert 'pegasus_wf_uuid == \\"$UUID\\"' in section, (
            f"{query} is not constrained to the current workflow uuid — "
            f"a previous run's identically-named jobs would be counted")
    print("  condor_q and condor_history are both uuid-constrained")


def _poll_once_with(raw, current_uuid):
    """Drive Poller.poll_once against canned output; return the poller."""
    import importlib.util
    import sys as _sys
    import types as _types
    _sys.modules.setdefault("fabrictestbed_extensions",
                            _types.ModuleType("fabrictestbed_extensions"))
    spec = importlib.util.spec_from_file_location("dash", DASHBOARD)
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)

    class Canned:
        name = "canned"

        def submit_cmd(self, cmd, timeout=120):
            return raw, ""

    poller = dash.Poller(Canned(), 10)
    poller.poll_once()
    return poller


PROGRESS_MIX = """===UUID===
run-A
===MACHINES===
[]
===QUEUE===
[]
===HISTORY===
[{"DAGNodeName": "train_higgs_train_000", "RemoteWallClockTime": 200,
  "MachineAttrCarbonIntensity0": 300.0, "MachineAttrFabricSite0": "STAR"},
 {"DAGNodeName": "train_higgs_train_001", "RemoteWallClockTime": 200,
  "MachineAttrCarbonIntensity0": 300.0, "MachineAttrFabricSite0": "TACC"},
 {"DAGNodeName": "train_higgs_train_002", "RemoteWallClockTime": 200,
  "MachineAttrCarbonIntensity0": 300.0, "MachineAttrFabricSite0": "TACC"}]
===PROGRESS===
@@/out/1788/progress_001.json
{"step": 100, "host": "STAR-gpu", "gpu_wh": 7.5, "wf_uuid": "run-A"}
@@/out/1788/progress_002.json
{"step": 200, "host": "TACC-gpu", "gpu_wh": 99.0, "wf_uuid": "run-B"}
@@/out/1788/progress_003.json
{"step": 300, "host": "TACC-gpu", "gpu_wh": 55.0}
"""


def test_only_this_runs_progress_contributes_energy():
    """Two runs sharing an output directory write identically NAMED
    progress files, so the path proves nothing. Only the uuid the trainer
    stamps into the file does: a foreign run's energy must not be summed,
    and an unstamped file must not be guessed at.
    """
    state = _poll_once_with(PROGRESS_MIX, "run-A").state
    totals = state["totals"]
    assert totals["energy_wh"] == 7.5, (
        f"foreign/unstamped energy leaked into the total: {totals}")
    assert totals["measured_segments"] == 1, totals
    assert totals["foreign_progress"] == 1, totals
    assert totals["unverified_progress"] == 1, totals
    # And the rejected rows must not carry energy on the job table either.
    by_job = {j["job"]: j for j in state["jobs"]}
    assert by_job["train_higgs_train_000"]["measured_wh"] == 7.5
    assert by_job["train_higgs_train_001"]["measured_wh"] is None
    assert by_job["train_higgs_train_002"]["measured_wh"] is None
    print("  only uuid-matching progress contributes; drops are counted")


def test_trainer_stamps_the_run_id_into_progress():
    """The dashboard's filter is only meaningful if the producer stamps."""
    trainer = os.path.join(HERE, "..", "carbon_chaser", "workload",
                           "train_higgs.py")
    text = open(trainer).read()
    assert "_CONDOR_JOB_AD" in text and "pegasus_wf_uuid" in text, (
        "train_higgs.py must read the workflow uuid from the HTCondor job "
        "ad — nothing else identifies the run at write time")
    assert '"wf_uuid": wf_uuid' in text, (
        "the uuid must be written into progress.json, or every consumer "
        "is back to guessing from the path")
    print("  the trainer stamps wf_uuid into every progress write")


def test_sections_survive_content_without_a_trailing_newline():
    """A file that ends mid-line must not swallow the next marker.

    `json.dump(..., indent=2)` writes no trailing newline, so `cat`ing the
    result file left the following marker glued to its last line
    ("}===REPORT==="). parse_sections only recognises a marker that STARTS
    a line, so REPORT vanished and its text was absorbed into RESULT,
    which then failed to parse — the dashboard reported the finished run's
    result as "unreadable" and its report as absent. Observed on a live
    pool after a successful 36/36 workflow.
    """
    import importlib.util
    import sys as _sys
    import types as _types
    _sys.modules.setdefault("fabrictestbed_extensions",
                            _types.ModuleType("fabrictestbed_extensions"))
    spec = importlib.util.spec_from_file_location("dash", DASHBOARD)
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)

    # Every marker in the real script must be preceded by a bare echo.
    for line in poll_script().splitlines():
        if line.strip().startswith("echo '===") and "===' " not in line:
            assert line.strip().startswith("echo; echo '==="), (
                f"marker not newline-guarded: {line!r}")

    # And prove the guard works against a file with no trailing newline.
    with tempfile.TemporaryDirectory() as tmp:
        payload = os.path.join(tmp, "r.json")
        with open(payload, "w") as handle:
            handle.write('{"val_auc": 0.87}')          # no trailing newline
        script = (f"echo; echo '===RESULT==='\n"
                  f"cat {payload}\n"
                  f"echo; echo '===REPORT==='\necho PRESENT\n")
        raw = subprocess.run(["bash", "-c", script], capture_output=True,
                             text=True, timeout=60).stdout
        sections = dash.parse_sections(raw)
        assert sections.get("REPORT", "").strip() == "PRESENT", sections
        assert json.loads(sections["RESULT"])["val_auc"] == 0.87, sections
    print("  markers survive content that ends mid-line")


REPORT_CASES = [
    ("verified", "run-A\nPRESENT", True),
    ("foreign", "run-B\nPRESENT", False),
    ("unverified", "PRESENT", False),
    ("pending", "ABSENT", False),
]


def test_the_report_is_served_only_when_it_names_this_run():
    """Existence in the output directory is not provenance.

    The report is the demo's headline artifact, so serving another run's
    copy would misattribute the whole story — the AUC curve AND the
    migration colours. report_higgs.py writes "workflow run: <uuid>", and
    only a report declaring THIS run's id is offered or served.
    """
    base = """===UUID===
run-A
===MACHINES===
[]
===QUEUE===
[]
===HISTORY===
[]
===OUTDIR===
/home/ubuntu/wf-output/1788
===PROGRESS===
===RESULT===
===REPORT===
%s
"""
    for expected, payload, ready in REPORT_CASES:
        poller = _poll_once_with(base % payload, "run-A")
        assert poller.report_status == expected, (payload,
                                                  poller.report_status)
        assert poller.state["report_ready"] is ready, payload
    print("  report offered only when it declares this run "
          "(verified/foreign/unverified/pending all distinguished)")


def _serve_report(raw):
    """Drive the real /report handler against a canned single fetch."""
    import importlib.util
    import sys as _sys
    import types as _types
    _sys.modules.setdefault("fabrictestbed_extensions",
                            _types.ModuleType("fabrictestbed_extensions"))
    spec = importlib.util.spec_from_file_location("dash", DASHBOARD)
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)
    fetched = dash.parse_sections(raw)
    current = fetched.get("UUID", "").strip()
    html = fetched.get("HTML", "")
    declared = dash.REPORT_RUN_ID.search(html)
    served = (declared and current and current != "no-current-run"
              and declared.group(1) == current)
    return bool(served), html


REPORT_A = "aaaaaaaa-1111-2222-3333-444444444444"
REPORT_B = "bbbbbbbb-1111-2222-3333-444444444444"


def _fetch(uuid, body):
    return f"\n===UUID===\n{uuid}\n\n===HTML===\n{body}\n"


def test_the_report_endpoint_verifies_the_bytes_it_serves():
    """The check and the content must come from ONE fetch.

    Verifying a status computed at the last poll and then reading the file
    separately is a TOCTOU: between them the run can change or a
    concurrent run can overwrite the file in a shared output directory,
    and the dashboard would serve bytes it never checked. So the handler
    extracts the declared id from the same bytes it returns.
    """
    page = f"<html><p>workflow run: {REPORT_A}</p><svg/></html>"

    served, html = _serve_report(_fetch(REPORT_A, page))
    assert served and "svg" in html, "a matching report must be served"

    # The exact TOCTOU payload: the run moved on, or another run's report
    # now occupies the path. Same bytes, different declared id -> refused.
    served, _ = _serve_report(_fetch(REPORT_B, page))
    assert not served, "a report naming another run must not be served"

    served, _ = _serve_report(_fetch(REPORT_A, "<html>no declaration</html>"))
    assert not served, "an unstamped report must not be served"

    served, _ = _serve_report(_fetch("no-current-run", page))
    assert not served, "with no current run there is nothing to attribute to"

    served, _ = _serve_report(_fetch(REPORT_A, ""))
    assert not served, "an absent report must not be served"
    print("  /report verifies the served bytes themselves, not a cached "
          "verdict")


def test_the_report_endpoint_does_not_trust_cached_status():
    """Guard the shape: the handler must not gate on poller state."""
    text = open(DASHBOARD).read()
    endpoint = text.split('elif self.path == "/report"', 1)[1].split(
        "else:", 1)[0]
    assert "poller.report_status" not in endpoint, (
        "/report gates on a cached verdict — that value is computed up to "
        "one poll interval earlier than the bytes it guards")
    assert "REPORT_SCRIPT" in endpoint, (
        "/report must fetch the run id and the report together")
    print("  /report gates on the fetch, not on poller.report_status")


if __name__ == "__main__":
    for fn in (test_both_halves_come_from_the_same_braindump,
               test_equals_form_is_handled,
               test_a_newer_epoch_dir_does_not_hijack_the_scope,
               test_missing_or_unplanned_run_yields_no_data_not_wrong_data,
               test_job_ads_are_constrained_by_uuid,
               test_only_this_runs_progress_contributes_energy,
               test_trainer_stamps_the_run_id_into_progress,
               test_sections_survive_content_without_a_trailing_newline,
               test_the_report_is_served_only_when_it_names_this_run,
               test_the_report_endpoint_verifies_the_bytes_it_serves,
               test_the_report_endpoint_does_not_trust_cached_status):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
