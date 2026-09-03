"""Hang-safety tests for FabricExecutor.

A stalled FABNet path must not wedge migration: the dataplane attempt has to
time out and the relay fallback has to run. These use fake nodes that block
forever, so a regression here fails the test instead of freezing the booth.
"""

import hashlib
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carbon_chaser.executor import (FabricExecutor, OperationTimeout,
                                    call_with_timeout)


class HangingNode:
    """Blocks forever on any command containing `hang_on`."""

    def __init__(self, name, hang_on="scp", hang_transfers=False):
        self.name = name
        self.hang_on = hang_on
        self.hang_transfers = hang_transfers
        self.calls = []

    PAYLOAD = b"x" * 4096          # what download_file writes

    def execute(self, command, **kw):
        self.calls.append(command)
        if self.hang_on and self.hang_on in command:
            threading.Event().wait()  # never returns
        # Emulate the real fingerprint command: size on one line, md5 on the
        # next. A fake that answers only the size would make _fingerprint
        # raise and hide whatever the test meant to exercise.
        if "md5sum" in command:
            digest = hashlib.md5(self.PAYLOAD).hexdigest()
            return (f"{len(self.PAYLOAD)}\n{digest}\n", "")
        if "mv -f" in command:         # verified commit prints the size
            return (f"{len(self.PAYLOAD)}\n", "")
        if "stat -c" in command:
            return (f"{len(self.PAYLOAD)}\n", "")
        if "progress.json" in command:
            return ('{"step": 7, "loss": 0.5, "acc": 0.9}', "")
        return ("SCP_OK\n" if "SCP_OK" in command else "ok", "")

    def get_username(self):
        return "ubuntu"

    def download_file(self, local, remote):
        if self.hang_transfers:
            threading.Event().wait()
        with open(local, "wb") as f:
            f.write(self.PAYLOAD)

    def upload_file(self, local, remote):
        if self.hang_transfers:
            threading.Event().wait()


def make_executor(nodes, tmp, transfer_timeout_s=2.0):
    ex = FabricExecutor.__new__(FabricExecutor)   # skip fablib/slice lookup
    ex.nodes = nodes
    ex.run_root = tmp
    ex.dataplane_ips = {n: f"10.0.0.{i+1}" for i, n in enumerate(nodes)}
    ex.last_transfer = None
    ex.transfer_timeout_s = transfer_timeout_s
    os.makedirs(tmp, exist_ok=True)
    return ex


def test_watchdog_fires():
    t0 = time.time()
    try:
        call_with_timeout(lambda: threading.Event().wait(), 1.0, "blocker")
    except OperationTimeout:
        elapsed = time.time() - t0
        assert elapsed < 3, f"watchdog took {elapsed:.1f}s"
        print(f"  watchdog fired after {elapsed:.1f}s")
        return
    raise AssertionError("watchdog did not fire")


def test_hanging_scp_falls_back_to_relay(tmp):
    """The reported bug: a stalled dataplane path must not block migration."""
    nodes = {"A": HangingNode("A", hang_on="scp"),
             "B": HangingNode("B", hang_on=None)}
    ex = make_executor(nodes, tmp, transfer_timeout_s=2.0)
    # Pin the budgets this timing assertion depends on. Left at the production
    # defaults (CMD_TIMEOUT_S=30 plus WATCHDOG_MARGIN_S=30) the elapsed time is
    # tens of seconds, so the bound below measures the host rather than the
    # behaviour — and on a loaded machine it fails for reasons unrelated to the
    # code under test.
    ex.CMD_TIMEOUT_S = 2
    ex.WATCHDOG_MARGIN_S = 1

    t0 = time.time()
    nbytes = ex.transfer_checkpoint("A", "B")
    elapsed = time.time() - t0

    assert ex.last_transfer["method"] == "relay", ex.last_transfer
    assert nbytes == 4096, nbytes
    # With the budgets pinned, falling back takes a few seconds, so this bound
    # is both tight and deterministic.
    assert elapsed < 15, f"took {elapsed:.1f}s"
    print(f"  scp hang -> relay fallback in {elapsed:.1f}s, "
          f"method={ex.last_transfer['method']}")


def test_hanging_relay_raises_not_hangs(tmp):
    """If BOTH paths are dead, the call must still return control."""
    nodes = {"A": HangingNode("A", hang_on="scp", hang_transfers=True),
             "B": HangingNode("B", hang_on=None, hang_transfers=True)}
    ex = make_executor(nodes, tmp, transfer_timeout_s=1.0)
    # Same reason as above, and this is the test that actually went red: it
    # asserted `elapsed < 60` while leaving the defaults in place, and the
    # full dead-scp-then-dead-relay path took ~136s on this machine. The bound
    # was racing the host. Pinned, the path completes in ~3s.
    ex.CMD_TIMEOUT_S = 2
    ex.WATCHDOG_MARGIN_S = 1
    t0 = time.time()
    try:
        ex.transfer_checkpoint("A", "B")
    except OperationTimeout as e:
        elapsed = time.time() - t0
        assert elapsed < 15, f"took {elapsed:.1f}s"
        print(f"  both paths dead -> raised in {elapsed:.1f}s ({e})")
        return
    raise AssertionError("expected OperationTimeout")


def test_progress_never_blocks_tick_loop(tmp):
    nodes = {"A": HangingNode("A", hang_on="progress.json")}
    ex = make_executor(nodes, tmp)
    ex.CMD_TIMEOUT_S = 1
    ex.WATCHDOG_MARGIN_S = 1
    t0 = time.time()
    result = ex.get_progress("A")
    elapsed = time.time() - t0
    assert result is None, result
    assert elapsed < 10, f"blocked {elapsed:.1f}s"
    print(f"  hung progress -> None in {elapsed:.1f}s (tick loop survives)")


def test_scp_command_is_noninteractive_and_bounded(tmp):
    """Guard the actual flags — this is what prevents the hang upstream."""
    nodes = {"A": HangingNode("A", hang_on=None),
             "B": HangingNode("B", hang_on=None)}
    ex = make_executor(nodes, tmp)
    ex._transfer_dataplane("A", "B")
    scp_cmd = next(c for c in nodes["A"].calls if "scp" in c)
    for required in ("BatchMode=yes", "ConnectTimeout=", "ServerAliveInterval=",
                     "ServerAliveCountMax=", "timeout -k"):
        assert required in scp_cmd, f"missing {required} in: {scp_cmd}"
    assert "sudo" not in scp_cmd, "must not run scp under sudo"
    print("  scp command is non-interactive and bounded")


if __name__ == "__main__":
    tmp = os.path.join(os.path.dirname(__file__), "..", "runs", "_test")
    print("test_watchdog_fires");                    test_watchdog_fires()
    print("test_hanging_scp_falls_back_to_relay");   test_hanging_scp_falls_back_to_relay(tmp)
    print("test_hanging_relay_raises_not_hangs");    test_hanging_relay_raises_not_hangs(tmp)
    print("test_progress_never_blocks_tick_loop");   test_progress_never_blocks_tick_loop(tmp)
    print("test_scp_command_is_noninteractive_and_bounded")
    test_scp_command_is_noninteractive_and_bounded(tmp)
    print("\nALL PASS")
