"""Downtime-model tests, anchored to measurements from the real testbed.

The veto compares predicted downtime against a budget, so a model that
underestimates lets through migrations that should have been refused — and
would understate migration cost in the paper. These numbers come from the
2026-08-24 production run: a 12,262-byte checkpoint transferring in ~1.5s
inside a 3.6s total downtime.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carbon_chaser.clock import SimClock  # noqa: E402
from carbon_chaser.engine import Engine  # noqa: E402

REAL_BYTES = 12262
REAL_TRANSFER_S = 1.52
REAL_DOWNTIME_S = 3.6

SITES = {s: {"display": s, "lat": 0, "lon": 0, "zone": f"z{s}"}
         for s in ("A", "B")}


class Stub:
    def checkpoint_bytes(self, site):
        return REAL_BYTES


class Prov:
    def get_intensity(self, zone):
        return 300


def make_engine():
    return Engine(SITES, "A", Prov(), Stub(), SimClock(),
                  {}, net_cfg={"overhead_s": 20})


def dataplane(nbytes, seconds):
    return {"method": "dataplane", "bytes": nbytes, "seconds": seconds}


def test_small_transfer_predicts_real_downtime():
    eng = make_engine()
    eng._calibrate(dataplane(REAL_BYTES, REAL_TRANSFER_S), REAL_DOWNTIME_S)
    # wire time for 12KB on a 100G path is ~0
    pred = eng._predict_downtime({"est_transfer_s": 0.0})
    assert abs(pred - REAL_DOWNTIME_S) < 0.3, (
        f"predicted {pred:.2f}s vs measured {REAL_DOWNTIME_S}s — the "
        f"stop/start cost is missing from the model")
    print(f"  predicts {pred:.2f}s vs {REAL_DOWNTIME_S}s measured")


def test_transfer_duration_alone_is_not_the_overhead():
    """Guards the specific regression: using transfer time as the whole
    fixed cost halves the prediction."""
    eng = make_engine()
    eng._calibrate(dataplane(REAL_BYTES, REAL_TRANSFER_S), REAL_DOWNTIME_S)
    pred = eng._predict_downtime({"est_transfer_s": 0.0})
    assert pred > REAL_TRANSFER_S * 1.5, (
        f"prediction {pred:.2f}s is barely above the transfer time "
        f"{REAL_TRANSFER_S}s — orchestration cost was dropped")
    assert eng._measured_orchestration_s > 1.5
    print(f"  orchestration measured separately: "
          f"{eng._measured_orchestration_s:.2f}s")


def test_uncalibrated_falls_back_to_conservative_constant():
    eng = make_engine()
    pred = eng._predict_downtime({"est_transfer_s": 0.0})
    assert pred == 20, f"expected configured 20s fallback, got {pred}"
    print("  no measurements yet -> conservative configured overhead")


def test_large_transfer_calibrates_bandwidth_not_setup():
    eng = make_engine()
    eng._calibrate(dataplane(REAL_BYTES, REAL_TRANSFER_S), REAL_DOWNTIME_S)
    setup = eng._measured_setup_s
    eng._calibrate(dataplane(2e9, 22.0), 25.0)
    assert eng._measured_rate_bps is not None
    # wire time excludes setup, so the rate must exceed size/transfer_s
    naive = 2e9 * 8 / 22.0
    assert eng._measured_rate_bps > naive, (
        "setup time was not excluded from the bandwidth estimate")
    assert abs(eng._measured_setup_s - setup) < 1e-9, (
        "a large transfer must not move the setup estimate")
    print(f"  2GB -> {eng._measured_rate_bps/1e9:.2f} Gbps, setup unchanged")


def test_prediction_scales_with_size():
    eng = make_engine()
    eng._calibrate(dataplane(REAL_BYTES, REAL_TRANSFER_S), REAL_DOWNTIME_S)
    eng._calibrate(dataplane(2e9, 22.0), 25.0)
    small = eng._predict_downtime({"est_transfer_s": 0.0})
    big = eng._predict_downtime(
        {"est_transfer_s": 8e9 * 8 / eng._measured_rate_bps})
    assert big > small * 5, f"big={big:.1f}s small={small:.1f}s"
    print(f"  12KB -> {small:.1f}s, 8GB -> {big:.0f}s")


def test_relay_and_local_never_calibrate():
    eng = make_engine()
    eng._calibrate({"method": "relay", "bytes": 1e9, "seconds": 5}, 9.0)
    eng._calibrate({"method": "local", "bytes": 1e9, "seconds": 0.01}, 0.1)
    assert eng._measured_rate_bps is None
    assert eng._measured_setup_s is None
    assert eng._measured_orchestration_s is None
    print("  relay/local transfers ignored by all three estimators")


def test_absurd_downtime_does_not_corrupt_orchestration():
    """downtime < transfer would imply negative stop/start cost."""
    eng = make_engine()
    eng._calibrate(dataplane(REAL_BYTES, 5.0), 1.0)   # nonsense pairing
    assert eng._measured_orchestration_s is None
    print("  downtime < transfer rejected rather than going negative")


# --- transfer integrity (LocalExecutor, real files) ----------------------

def _mk(tmp):
    import os
    from carbon_chaser.executor import LocalExecutor
    ex = LocalExecutor(run_root=tmp)
    for site in ("A", "B"):
        os.makedirs(os.path.join(tmp, site), exist_ok=True)
    open(os.path.join(tmp, "A", "checkpoint.npz"), "wb").write(b"SOURCE" * 200)
    open(os.path.join(tmp, "B", "checkpoint.npz"), "wb").write(b"GOODWORK" * 150)
    return ex


def test_same_length_corruption_is_rejected(tmp):
    """Equal size is not equal content — a size-only check would commit
    corrupt bytes and the trainer would load them as weights."""
    import os, shutil
    ex = _mk(tmp)
    real = shutil.copyfile
    shutil.copyfile = lambda a, b: (
        open(b, "wb").write(b"\x00" * os.path.getsize(a)), b)[1]
    try:
        raised = False
        try:
            ex.transfer_checkpoint("A", "B")
        except Exception:
            raised = True
    finally:
        shutil.copyfile = real
    assert raised, "committed a same-length corrupt checkpoint"
    live = open(os.path.join(tmp, "B", "checkpoint.npz"), "rb").read()
    assert live[:8] == b"GOODWORK", "target's good checkpoint was destroyed"
    print("  same-length corruption rejected; target preserved")


def test_failed_commit_is_not_reported_as_success(tmp):
    """An unchecked commit reports success while the target keeps its old
    checkpoint — the job then resumes from stale weights."""
    import os
    ex = _mk(tmp)
    real = os.replace
    os.replace = lambda a, b: (_ for _ in ()).throw(OSError("disk full"))
    try:
        raised = False
        try:
            ex.transfer_checkpoint("A", "B")
        except Exception:
            raised = True
    finally:
        os.replace = real
    assert raised, "a failed commit was reported as a successful transfer"
    live = open(os.path.join(tmp, "B", "checkpoint.npz"), "rb").read()
    assert live[:8] == b"GOODWORK"
    print("  failed commit surfaces as an error, target untouched")


def test_healthy_transfer_commits_exact_bytes(tmp):
    import os
    ex = _mk(tmp)
    n = ex.transfer_checkpoint("A", "B")
    live = open(os.path.join(tmp, "B", "checkpoint.npz"), "rb").read()
    assert live == b"SOURCE" * 200 and n == 1200, (n, len(live))
    assert not os.path.exists(os.path.join(tmp, "B", "checkpoint.npz.incoming"))
    print("  healthy transfer commits exact bytes, leaves no staging file")


if __name__ == "__main__":
    for fn in (test_small_transfer_predicts_real_downtime,
               test_transfer_duration_alone_is_not_the_overhead,
               test_uncalibrated_falls_back_to_conservative_constant,
               test_large_transfer_calibrates_bandwidth_not_setup,
               test_prediction_scales_with_size,
               test_relay_and_local_never_calibrate,
               test_absurd_downtime_does_not_corrupt_orchestration):
        print(fn.__name__)
        fn()

    import shutil as _sh, tempfile as _tf
    for fn in (test_same_length_corruption_is_rejected,
               test_failed_commit_is_not_reported_as_success,
               test_healthy_transfer_commits_exact_bytes):
        print(fn.__name__)
        _d = _tf.mkdtemp(prefix="cc-xfer-")
        try:
            fn(_d)
        finally:
            _sh.rmtree(_d, ignore_errors=True)
    print("\nALL PASS")
