"""Workload executors.

LocalExecutor runs the trainer as a subprocess in a per-site sandbox dir —
the whole demo works on one laptop (and is the venue fallback). FabricExecutor
drives one VM per site on a pre-created FABRIC slice via fablib; checkpoint
transfer is relayed through the orchestrator host (download from the old
site, upload to the new one), which keeps it simple and works regardless of
dataplane reachability between VMs.
"""

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Optional

TRAIN_PY = os.path.join(os.path.dirname(__file__), "workload", "train.py")


def _md5_file(path: str) -> str:
    """Content digest. Size equality is not integrity: a same-length
    corruption would otherwise be committed and loaded as model weights."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OperationTimeout(Exception):
    """A node operation exceeded its wall-clock budget."""


def call_with_timeout(fn: Callable, timeout_s: float, what: str):
    """Run fn() with a hard wall-clock bound.

    Outermost guard against a wedged SSH/SFTP call: fablib's execute() loops
    on select() until the remote command reports an exit status, and retries
    on connection errors, so neither a black-holed dataplane path nor a dead
    management channel is guaranteed to return on its own. The worker is a
    daemon thread, so a truly stuck call is abandoned rather than joined.
    """
    box: Dict[str, object] = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # propagate to the caller's thread
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True,
                              name=f"node-op:{what}")
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise OperationTimeout(f"{what} exceeded {timeout_s}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class Executor:
    def start(self, site: str) -> None:
        """Start the workload at `site`. Must be idempotent: if a trainer is
        already running there, do not start a second one."""
        raise NotImplementedError

    def stop(self, site: str) -> bool:
        """Stop the workload at `site`.

        Returns True only when the workload is *confirmed* stopped. A False
        return means "may still be running" — the caller must not start the
        job elsewhere, or two trainers will diverge from the same
        checkpoint.
        """
        raise NotImplementedError

    def is_running(self, site: str) -> Optional[bool]:
        """True/False if known, None if it could not be determined."""
        return None

    def transfer_checkpoint(self, src_site: str, dst_site: str) -> float:
        """Move the latest checkpoint; returns bytes transferred.

        Implementations also set `self.last_transfer` to
        {"method": "dataplane"|"relay"|"local", "bytes", "seconds"}. Only
        "dataplane" transfers traverse the inter-rack links the network
        model prices, so only those may calibrate it.
        """
        raise NotImplementedError

    def get_progress(self, site: str) -> Optional[dict]:
        raise NotImplementedError

    def checkpoint_bytes(self, site: str) -> Optional[float]:
        """Size of the site's latest checkpoint, if known."""
        return None

    def get_power(self, site: str) -> Optional[dict]:
        """Latest NVML power sample the workload wrote, if any.

        Returning None means "not measured here" — the engine then reports
        the power figure as assumed rather than silently substituting one.
        """
        return None

    def discard_incoming(self, site: str) -> None:
        """Remove only the staging file this migration may have written.

        Transfers land on a temp path and are moved into place only after
        the full size is verified, so the live checkpoint is never partial
        and cleanup can never destroy real work. Guessing instead — "the
        transfer failed, so the target file must be broken" — deletes an
        intact checkpoint whenever the failure happened before any byte was
        written (a refused connection, a watchdog timeout).
        """

    def discard_checkpoint(self, site: str) -> None:
        """Remove a site's live checkpoint. Destructive; not part of the
        migration abort path."""

    # -- durable evidence, stored on the workers ---------------------------
    #
    # A new orchestrator (different host, fresh disk) must be able to learn
    # what the previous one was doing. Keeping that only on the orchestrator
    # means moving the control plane silently erases the very evidence that
    # prevents a duplicate trainer.

    def read_marker(self, site: str) -> Optional[str]:
        """Site name this worker last recorded as active, if readable."""
        return None

    def write_marker(self, site: str, value: str) -> None:
        """Record the active site on this worker (best effort)."""

    def has_job_state(self, site: str) -> Optional[bool]:
        """True if this worker shows any sign of having run the job.
        None if it cannot be determined."""
        return None


class LocalExecutor(Executor):
    def __init__(self, run_root: str = "runs"):
        self.run_root = run_root
        self._procs: Dict[str, subprocess.Popen] = {}
        self.last_transfer: Optional[dict] = None

    def _workdir(self, site: str) -> str:
        d = os.path.join(self.run_root, site)
        os.makedirs(d, exist_ok=True)
        return d

    def start(self, site: str) -> None:
        if site in self._procs and self._procs[site].poll() is None:
            return
        wd = self._workdir(site)
        log = open(os.path.join(wd, "train.log"), "a")
        self._procs[site] = subprocess.Popen(
            [sys.executable, TRAIN_PY, "--workdir", wd],
            stdout=log, stderr=subprocess.STDOUT,
        )

    def stop(self, site: str) -> bool:
        proc = self._procs.pop(site, None)
        if proc is None or proc.poll() is not None:
            # Untracked trainer (orphan from a previous run) may still hold
            # the workdir; report honestly rather than assuming it is gone.
            return not self._lock_held(site)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return proc.poll() is not None

    def is_running(self, site: str) -> Optional[bool]:
        """Ask the lock, not just our own process table.

        A trainer orphaned by a previous orchestrator run is not in
        `_procs`, but it still holds the workdir — and starting beside it
        would corrupt the checkpoint just as surely as on a remote node.
        """
        proc = self._procs.get(site)
        if proc is not None and proc.poll() is None:
            return True
        return self._lock_held(site)

    def _lock_held(self, site: str) -> bool:
        path = os.path.join(self._workdir(site), "train.lock")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return True          # someone else holds it
                fcntl.flock(handle, fcntl.LOCK_UN)
                return False
        except OSError:
            return False

    def transfer_checkpoint(self, src_site: str, dst_site: str) -> float:
        src = os.path.join(self._workdir(src_site), "checkpoint.npz")
        dst = os.path.join(self._workdir(dst_site), "checkpoint.npz")
        if not os.path.exists(src):
            self.last_transfer = None
            return 0.0
        t0 = time.time()
        # Stage then atomically replace, so an interrupted copy cannot leave
        # the destination's live checkpoint truncated.
        staging = dst + ".incoming"
        shutil.copyfile(src, staging)
        expected_size, expected_digest = os.path.getsize(src), _md5_file(src)
        if (os.path.getsize(staging) != expected_size
                or _md5_file(staging) != expected_digest):
            raise RuntimeError("staged checkpoint does not match source")
        os.replace(staging, dst)        # raises on failure
        if os.path.getsize(dst) != expected_size:
            raise RuntimeError("commit left an unexpected size")
        nbytes = float(expected_size)
        # A local file copy — never a network measurement.
        self.last_transfer = {"method": "local", "bytes": nbytes,
                              "seconds": round(time.time() - t0, 3)}
        return nbytes

    def get_progress(self, site: str) -> Optional[dict]:
        path = os.path.join(self._workdir(site), "progress.json")
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def checkpoint_bytes(self, site: str) -> Optional[float]:
        path = os.path.join(self._workdir(site), "checkpoint.npz")
        return float(os.path.getsize(path)) if os.path.exists(path) else None

    def get_power(self, site: str) -> Optional[dict]:
        try:
            with open(os.path.join(self._workdir(site), "power.json")) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def discard_incoming(self, site: str) -> None:
        path = os.path.join(self._workdir(site), "checkpoint.npz.incoming")
        if os.path.exists(path):
            os.remove(path)

    def discard_checkpoint(self, site: str) -> None:
        path = os.path.join(self._workdir(site), "checkpoint.npz")
        if os.path.exists(path):
            os.remove(path)

    def read_marker(self, site: str) -> Optional[str]:
        try:
            with open(os.path.join(self._workdir(site), "active_site")) as f:
                return f.read().strip() or None
        except OSError:
            return None

    def write_marker(self, site: str, value: str) -> None:
        try:
            with open(os.path.join(self._workdir(site), "active_site"),
                      "w") as f:
                f.write(value)
        except OSError:
            pass

    def has_job_state(self, site: str) -> Optional[bool]:
        wd = self._workdir(site)
        return any(os.path.exists(os.path.join(wd, f))
                   for f in ("progress.json", "checkpoint.npz"))

    def shutdown(self):
        for site in list(self._procs):
            self.stop(site)


class SshExecutor(Executor):
    """Drives worker nodes over FABNetv4 from *inside* the slice.

    Used when the orchestrator itself runs on a FABRIC control node: it
    reaches workers by their dataplane addresses with the shared relay key,
    so no fablib, no FABRIC token, and no bastion hop are involved — and the
    control plane rides FABRIC's network rather than the operator's laptop
    link. Checkpoints still move worker-to-worker; they never transit the
    control node.

    Same command discipline as FabricExecutor: bounded twice, non-interactive
    SSH, subshell launch, lock as the source of truth.
    """

    REMOTE_WD = "job"
    LOCK = "job/train.lock"
    KEY = "~/.ssh/ccrelay"
    USER = "ubuntu"
    # Anchored so it matches only the trainer, never the shell running pkill.
    TRAINER_PAT = "'^python3 .*train.*\\.py'"
    SSH_OPTS = ("-o BatchMode=yes -o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 "
                "-o ServerAliveInterval=5 -o ServerAliveCountMax=3")
    CMD_TIMEOUT_S = 30
    WATCHDOG_MARGIN_S = 30

    def __init__(self, sites: list, dataplane_ips: Dict[str, str],
                 run_root: str = "runs", transfer_timeout_s: float = 300.0,
                 workload: str = "train_gpu.py",
                 checkpoint_file: str = "checkpoint.pt"):
        self.sites = sites
        self.dataplane_ips = dataplane_ips
        self.run_root = run_root
        os.makedirs(run_root, exist_ok=True)
        self.transfer_timeout_s = transfer_timeout_s
        # The workload script and its checkpoint name travel together: a
        # mismatch means migration transfers a file the trainer never wrote.
        self.workload = workload
        self.checkpoint_file = checkpoint_file
        self.last_transfer: Optional[dict] = None
        self._last_ckpt_bytes: Optional[float] = None

    # -- bounded remote exec ------------------------------------------------

    def _ssh(self, site: str, command: str,
             timeout_s: Optional[float] = None, what: str = "exec"):
        budget = timeout_s or self.CMD_TIMEOUT_S
        ip = self.dataplane_ips[site]
        remote = f"timeout -k 10 {int(budget)} bash -lc {shlex.quote(command)}"
        argv = ["ssh", *shlex.split(self.SSH_OPTS),
                "-i", os.path.expanduser(self.KEY),
                f"{self.USER}@{ip}", remote]

        def run():
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=budget + 10)
            return proc.stdout, proc.stderr

        return call_with_timeout(run, budget + self.WATCHDOG_MARGIN_S, what)

    def _ssh_quiet(self, site: str, command: str, what: str):
        try:
            return self._ssh(site, command, what=what)
        except Exception as e:
            print(f"[executor] {what} failed: {e}", flush=True)
            return None

    # -- Executor interface -------------------------------------------------

    def start(self, site: str) -> None:
        self._ssh(site,
                  f"mkdir -p {self.REMOTE_WD} && "
                  f"( setsid python3 ~/{self.workload} --workdir {self.REMOTE_WD} "
                  f"< /dev/null >> {self.REMOTE_WD}/train.log 2>&1 & ) ; "
                  f"echo started",
                  what=f"start:{site}")
        for _ in range(5):
            time.sleep(1)
            if self.is_running(site):
                return
        raise RuntimeError(f"trainer did not start at {site}")

    def is_running(self, site: str) -> Optional[bool]:
        res = self._ssh_quiet(
            site,
            f"mkdir -p {self.REMOTE_WD} && "
            f"(flock -n 9 && echo FREE || echo HELD) 9>{self.LOCK}",
            f"is_running:{site}")
        if res is None:
            return None
        out = res[0] or ""
        return True if "HELD" in out else (False if "FREE" in out else None)

    def stop(self, site: str) -> bool:
        self._ssh_quiet(site, f"pkill -f {self.TRAINER_PAT} || true",
                        f"stop:{site}")
        for attempt in range(10):
            state = self.is_running(site)
            if state is False:
                return True
            if state is None and attempt >= 2:
                return False
            time.sleep(1)
        self._ssh_quiet(site, f"pkill -9 -f {self.TRAINER_PAT} || true",
                        f"stop-kill:{site}")
        time.sleep(2)
        return self.is_running(site) is False

    def transfer_checkpoint(self, src_site: str, dst_site: str) -> float:
        """scp worker-to-worker; the control node never sees the bytes."""
        remote = f"{self.REMOTE_WD}/{self.checkpoint_file}"
        staging = f"{remote}.incoming"
        dst_ip = self.dataplane_ips[dst_site]
        self._ssh(dst_site, f"mkdir -p {self.REMOTE_WD}",
                  what=f"mkdir:{dst_site}")
        expected_size, expected_digest = self._fingerprint(
            src_site, remote, f"src:{src_site}")
        t0 = time.time()
        # Staged, then committed only on a verified size (see the executor
        # interface note on discard_incoming).
        out, err = self._ssh(
            src_site,
            f"scp {self.SSH_OPTS} -i {self.KEY} {remote} "
            f"{self.USER}@{dst_ip}:{staging} && echo SCP_OK",
            timeout_s=self.transfer_timeout_s,
            what=f"scp:{src_site}->{dst_site}")
        if "SCP_OK" not in (out or ""):
            raise RuntimeError(f"scp failed: {(err or out or '').strip()[:200]}")
        seconds = time.time() - t0
        got_size, got_digest = self._fingerprint(dst_site, staging,
                                                 f"staged:{dst_site}")
        if got_size != expected_size or got_digest != expected_digest:
            raise RuntimeError(
                f"corrupt transfer to {dst_site}: {got_size}B/{got_digest} "
                f"!= {expected_size}B/{expected_digest}")
        nbytes = float(expected_size)
        self._commit(dst_site, staging, remote, expected_size)
        self._last_ckpt_bytes = nbytes
        self.last_transfer = {"method": "dataplane", "bytes": nbytes,
                              "seconds": round(seconds, 2)}
        return nbytes

    def get_progress(self, site: str) -> Optional[dict]:
        res = self._ssh_quiet(
            site, f"cat {self.REMOTE_WD}/progress.json 2>/dev/null || true",
            f"progress:{site}")
        if res is None:
            return None
        try:
            return json.loads((res[0] or "").strip())
        except (json.JSONDecodeError, AttributeError):
            return None

    def checkpoint_bytes(self, site: str) -> Optional[float]:
        return self._last_ckpt_bytes

    def get_power(self, site: str) -> Optional[dict]:
        res = self._ssh_quiet(
            site, f"cat {self.REMOTE_WD}/power.json 2>/dev/null || true",
            f"power:{site}")
        if res is None:
            return None
        try:
            return json.loads((res[0] or "").strip())
        except (json.JSONDecodeError, AttributeError):
            return None

    def _fingerprint(self, site: str, path: str, what: str) -> tuple:
        out, err = self._ssh(
            site, f"stat -c %s {path} && md5sum {path} | cut -d\' \' -f1",
            what=f"fingerprint:{what}")
        lines = [l.strip() for l in (out or "").strip().splitlines() if l.strip()]
        if len(lines) < 2 or not lines[0].isdigit():
            raise RuntimeError(
                f"could not fingerprint {path} ({what}): {(err or out or '')[:120]}")
        return int(lines[0]), lines[1]

    def _commit(self, site: str, staging: str, live: str,
                expected_size: int) -> None:
        """Move into place and prove it — an unchecked mv fails silently and
        leaves the target resuming from its old checkpoint."""
        out, err = self._ssh(site, f"mv -f {staging} {live} && stat -c %s {live}",
                             what=f"commit:{site}")
        text = (out or "").strip().splitlines()
        if not text or not text[-1].strip().isdigit():
            raise RuntimeError(
                f"commit failed at {site}: {(err or out or '').strip()[:160]}")
        if int(text[-1].strip()) != expected_size:
            raise RuntimeError(
                f"commit at {site} left {text[-1].strip()}B, "
                f"expected {expected_size}B")

    def discard_incoming(self, site: str) -> None:
        self._ssh_quiet(
            site, f"rm -f {self.REMOTE_WD}/{self.checkpoint_file}.incoming",
            f"discard-incoming:{site}")

    def discard_checkpoint(self, site: str) -> None:
        self._ssh_quiet(site,
                        f"rm -f {self.REMOTE_WD}/{self.checkpoint_file}",
                        f"discard:{site}")

    def read_marker(self, site: str) -> Optional[str]:
        res = self._ssh_quiet(
            site, f"cat {self.REMOTE_WD}/active_site 2>/dev/null || true",
            f"marker:{site}")
        return (res[0] or "").strip() or None if res else None

    def write_marker(self, site: str, value: str) -> None:
        self._ssh_quiet(
            site, f"mkdir -p {self.REMOTE_WD} && "
                  f"printf %s {shlex.quote(value)} > {self.REMOTE_WD}/active_site",
            f"mark:{site}")

    def has_job_state(self, site: str) -> Optional[bool]:
        res = self._ssh_quiet(
            site,
            f"ls {self.REMOTE_WD}/progress.json "
            f"{self.REMOTE_WD}/{self.checkpoint_file} "
            f"2>/dev/null | head -1", f"jobstate:{site}")
        return None if res is None else bool((res[0] or "").strip())

    def shutdown(self):
        pass


class FabricExecutor(Executor):
    """Drives VMs on an existing FABRIC slice (see fabric/create_slice.py).

    Node naming convention: one node per site, named exactly like the site
    (CLEM, TACC, ...). train.py is uploaded to ~/train.py at slice-creation
    time; runtime state lives in ~/job/ on each node.

    Checkpoints move node-to-node over FABNetv4 (scp between dataplane IPs
    recorded by create_slice.py in fabric/dataplane_ips.yaml, using the
    inter-node key it installs) — i.e. over the same inter-rack links the
    network model prices. Relaying through the orchestrator host is only a
    fallback, and is labeled as such in the migration record so a relay
    transfer is never mistaken for a dataplane measurement.
    """

    REMOTE_WD = "job"
    LOCK = "job/train.lock"
    RELAY_KEY = "~/.ssh/ccrelay"
    # Anchored so it matches only the trainer itself — a bare 'train.py'
    # also matches the `bash -lc "pkill -f train.py"` wrapper running the
    # kill, which would signal its own shell mid-command.
    TRAINER_PAT = "'^python3 .*train\\.py'"

    # Non-interactive and bounded at every layer: BatchMode never prompts
    # (a prompt on a stdin-less channel hangs forever), ConnectTimeout caps
    # the handshake, and ServerAlive probes kill a transfer that stalls
    # mid-stream on a black-holed path.
    SSH_OPTS = ("-o BatchMode=yes -o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 "
                "-o ServerAliveInterval=5 -o ServerAliveCountMax=3")
    CMD_TIMEOUT_S = 30          # short control commands
    WATCHDOG_MARGIN_S = 30      # watchdog allowance over the shell timeout

    def __init__(self, slice_name: str, sites: list, run_root: str = "runs",
                 dataplane_ips_file: Optional[str] = None,
                 transfer_timeout_s: float = 300.0):
        from fabrictestbed_extensions.fablib.fablib import FablibManager
        self.fablib = FablibManager()
        self.slice = self.fablib.get_slice(name=slice_name)
        self.nodes = {s: self.slice.get_node(name=s) for s in sites}
        self.run_root = run_root  # local staging area for relay fallback
        os.makedirs(run_root, exist_ok=True)
        self.dataplane_ips = {}
        if dataplane_ips_file and os.path.exists(dataplane_ips_file):
            import yaml
            with open(dataplane_ips_file) as f:
                self.dataplane_ips = yaml.safe_load(f) or {}
        self.last_transfer: Optional[dict] = None
        self.transfer_timeout_s = transfer_timeout_s

    # -- bounded node operations -------------------------------------------

    def _exec(self, node, command: str, timeout_s: Optional[float] = None,
              what: str = "exec"):
        """Run a command on a node, bounded twice.

        `timeout` (coreutils, not fablib's `sudo timeout` wrapper — which
        would run the command as root and resolve ~ against root's home)
        bounds the remote side so the SSH channel closes normally; the
        watchdog bounds the local side if the channel itself wedges.
        """
        budget = timeout_s or self.CMD_TIMEOUT_S
        wrapped = f"timeout -k 10 {int(budget)} bash -lc {shlex.quote(command)}"
        return call_with_timeout(
            lambda: node.execute(wrapped, quiet=True, retry=1),
            budget + self.WATCHDOG_MARGIN_S, what)

    def _exec_quiet(self, node, command: str, what: str) -> Optional[tuple]:
        """_exec that swallows timeouts/errors — for best-effort calls that
        must never wedge the engine tick loop."""
        try:
            return self._exec(node, command, what=what)
        except Exception as e:
            print(f"[executor] {what} failed: {e}", flush=True)
            return None

    def start(self, site: str) -> None:
        """Launch the trainer under `flock -n`, which makes this idempotent.

        The lock fd is inherited by the trainer and held for its lifetime,
        so a duplicate start is refused by the kernel rather than by a
        check-then-launch race. Without this, a restart triggered by a
        merely-unreachable node would put two trainers on the same
        checkpoint.
        """
        node = self.nodes[site]
        # train.py takes the lock itself (see acquire_lock), so a duplicate
        # launch exits immediately instead of racing. That keeps the
        # guarantee with the workload — one process, one mechanism, and it
        # holds whether the trainer was started by this executor or by hand.
        # The `( … & )` subshell is load-bearing, not style. Backgrounding
        # directly (`setsid nohup … &`) leaves the SSH channel open until
        # the trainer exits, so the call blocks for its full watchdog even
        # though the trainer started fine — measured on FABRIC: 40s+ vs
        # 0.2s. The subshell exits immediately, the trainer is reparented,
        # and all three of its fds point away from the channel.
        self._exec(node,
                   f"mkdir -p {self.REMOTE_WD} && "
                   f"( setsid python3 ~/{self.workload} --workdir {self.REMOTE_WD} "
                   f"< /dev/null >> {self.REMOTE_WD}/train.log 2>&1 & ) ; "
                   f"echo started",
                   what=f"start:{site}")
        # Confirm it took the lock; otherwise report the truth to the engine.
        for _ in range(5):
            time.sleep(1)
            if self.is_running(site):
                return
        raise RuntimeError(f"trainer did not start at {site} "
                           f"(lock {self.LOCK} not held)")

    def is_running(self, site: str) -> Optional[bool]:
        """Ask the lock, not the process table.

        `pgrep -f train.py` would also match the `flock` wrapper's own
        command line, so the lock is the only unambiguous answer.
        """
        res = self._exec_quiet(
            self.nodes[site],
            f"mkdir -p {self.REMOTE_WD} && "
            f"(flock -n 9 && echo FREE || echo HELD) 9>{self.LOCK}",
            f"is_running:{site}")
        if res is None:
            return None                      # unreachable: unknown, not "no"
        out = (res[0] or "")
        if "HELD" in out:
            return True
        if "FREE" in out:
            return False
        return None

    def stop(self, site: str) -> bool:
        node = self.nodes[site]
        # SIGTERM so the trainer writes a final checkpoint, then confirm exit.
        self._exec_quiet(node, f"pkill -f {self.TRAINER_PAT} || true",
                         f"stop:{site}")
        for attempt in range(10):
            state = self.is_running(site)
            if state is False:
                return True
            if state is None and attempt >= 2:
                # Cannot see the node; must assume the trainer may live.
                print(f"[executor] stop:{site} unconfirmed (node "
                      f"unreachable)", flush=True)
                return False
            time.sleep(1)
        # Still holding the lock after SIGTERM: escalate once, then re-check.
        self._exec_quiet(node, f"pkill -9 -f {self.TRAINER_PAT} || true",
                         f"stop-kill:{site}")
        time.sleep(2)
        confirmed = self.is_running(site) is False
        if not confirmed:
            print(f"[executor] stop:{site} UNCONFIRMED after SIGKILL",
                  flush=True)
        return confirmed

    def transfer_checkpoint(self, src_site: str, dst_site: str) -> float:
        if self.dataplane_ips.get(dst_site):
            try:
                return self._transfer_dataplane(src_site, dst_site)
            except Exception as e:
                print(f"[executor] dataplane transfer failed ({e}); "
                      f"falling back to relay", flush=True)
        return self._transfer_relay(src_site, dst_site)

    def _transfer_dataplane(self, src_site: str, dst_site: str) -> float:
        """scp src->dst over FABNetv4 — the path the network model prices.

        Raises on timeout/failure so transfer_checkpoint() can fall back to
        the relay; the bound must therefore actually fire.
        """
        src, dst = self.nodes[src_site], self.nodes[dst_site]
        remote = f"{self.REMOTE_WD}/checkpoint.npz"
        staging = f"{remote}.incoming"
        dst_ip = self.dataplane_ips[dst_site]
        self._exec(dst, f"mkdir -p {self.REMOTE_WD}", what=f"mkdir:{dst_site}")

        # Digest, not size: equal length is not equal content, and a
        # corrupt-but-same-length checkpoint would be loaded as weights.
        expected_size, expected_digest = self._fingerprint(
            src, remote, f"src:{src_site}")

        t0 = time.time()
        # Land on a staging path; the destination's live checkpoint is
        # replaced only after the transfer is verified, so a failed or
        # partial transfer never damages it.
        out, err = self._exec(
            src,
            f"scp {self.SSH_OPTS} -i {self.RELAY_KEY} "
            f"{remote} {dst.get_username()}@{dst_ip}:{staging} && echo SCP_OK",
            timeout_s=self.transfer_timeout_s,
            what=f"scp:{src_site}->{dst_site}",
        )
        if "SCP_OK" not in (out or ""):
            raise RuntimeError(f"scp failed: {(err or out or '').strip()[:200]}")
        seconds = time.time() - t0

        got_size, got_digest = self._fingerprint(dst, staging,
                                                 f"staged:{dst_site}")
        if got_size != expected_size or got_digest != expected_digest:
            raise RuntimeError(
                f"corrupt transfer to {dst_site}: {got_size}B/{got_digest} "
                f"!= {expected_size}B/{expected_digest}")
        nbytes = float(expected_size)
        self._commit(dst, staging, remote, expected_size, dst_site)
        self._last_ckpt_bytes = nbytes
        self.last_transfer = {"method": "dataplane", "bytes": nbytes,
                              "seconds": round(seconds, 2)}
        return nbytes

    def _transfer_relay(self, src_site: str, dst_site: str) -> float:
        """Fallback: relay via the orchestrator over the management network.
        NOT a dataplane-path transfer; labeled so it is never used to
        calibrate or validate the network model."""
        local = os.path.join(self.run_root, "checkpoint-relay.npz")
        remote = f"{self.REMOTE_WD}/checkpoint.npz"
        staging = f"{remote}.incoming"
        src, dst = self.nodes[src_site], self.nodes[dst_site]
        t0 = time.time()
        # SFTP has no internal deadline either — bound both legs. Upload to
        # staging and commit, so a broken upload leaves the destination's
        # live checkpoint intact.
        call_with_timeout(lambda: src.download_file(local, remote),
                          self.transfer_timeout_s, f"download:{src_site}")
        self._exec(dst, f"mkdir -p {self.REMOTE_WD}", what=f"mkdir:{dst_site}")
        call_with_timeout(lambda: dst.upload_file(local, staging),
                          self.transfer_timeout_s, f"upload:{dst_site}")
        if not os.path.exists(local):
            raise RuntimeError("relay download produced no file")
        nbytes = float(os.path.getsize(local))
        local_digest = _md5_file(local)
        got_size, got_digest = self._fingerprint(dst, staging,
                                                 f"staged:{dst_site}")
        if got_size != int(nbytes) or got_digest != local_digest:
            raise RuntimeError(
                f"corrupt relay upload to {dst_site}: {got_size}B/{got_digest} "
                f"!= {int(nbytes)}B/{local_digest}")
        self._commit(dst, staging, remote, int(nbytes), dst_site)
        self._last_ckpt_bytes = nbytes
        self.last_transfer = {"method": "relay", "bytes": nbytes,
                              "seconds": round(time.time() - t0, 2)}
        return nbytes

    def checkpoint_bytes(self, site: str) -> Optional[float]:
        return getattr(self, "_last_ckpt_bytes", None)

    def _fingerprint(self, node, path: str, what: str) -> tuple:
        """(size, md5) of a remote file; raises if it cannot be read."""
        out, err = self._exec(
            node, f"stat -c %s {path} && md5sum {path} | cut -d\' \' -f1",
            what=f"fingerprint:{what}")
        lines = [l.strip() for l in (out or "").strip().splitlines() if l.strip()]
        if len(lines) < 2 or not lines[0].isdigit():
            raise RuntimeError(
                f"could not fingerprint {path} ({what}): {(err or out or '')[:120]}")
        return int(lines[0]), lines[1]

    def _commit(self, node, staging: str, live: str, expected_size: int,
                site: str) -> None:
        """Move staging into place and PROVE it happened.

        `_exec` does not raise on a non-zero exit, so an unchecked `mv`
        fails silently — the caller would report a successful transfer while
        the target still holds its old checkpoint, and the job would resume
        from stale weights.
        """
        out, err = self._exec(
            node, f"mv -f {staging} {live} && stat -c %s {live}",
            what=f"commit:{site}")
        text = (out or "").strip().splitlines()
        if not text or not text[-1].strip().isdigit():
            raise RuntimeError(
                f"commit failed at {site}: {(err or out or '').strip()[:160]}")
        if int(text[-1].strip()) != expected_size:
            raise RuntimeError(
                f"commit at {site} left {text[-1].strip()}B, "
                f"expected {expected_size}B")

    def discard_incoming(self, site: str) -> None:
        self._exec_quiet(self.nodes[site],
                         f"rm -f {self.REMOTE_WD}/checkpoint.npz.incoming",
                         f"discard-incoming:{site}")

    def discard_checkpoint(self, site: str) -> None:
        self._exec_quiet(self.nodes[site],
                         f"rm -f {self.REMOTE_WD}/checkpoint.npz",
                         f"discard:{site}")

    def get_power(self, site: str) -> Optional[dict]:
        res = self._exec_quiet(
            self.nodes[site],
            f"cat {self.REMOTE_WD}/power.json 2>/dev/null || true",
            f"power:{site}")
        if res is None:
            return None
        try:
            return json.loads((res[0] or "").strip())
        except (json.JSONDecodeError, AttributeError):
            return None

    def read_marker(self, site: str) -> Optional[str]:
        res = self._exec_quiet(
            self.nodes[site],
            f"cat {self.REMOTE_WD}/active_site 2>/dev/null || true",
            f"marker:{site}")
        return (res[0] or "").strip() or None if res else None

    def write_marker(self, site: str, value: str) -> None:
        self._exec_quiet(
            self.nodes[site],
            f"mkdir -p {self.REMOTE_WD} && "
            f"printf %s {shlex.quote(value)} > {self.REMOTE_WD}/active_site",
            f"mark:{site}")

    def has_job_state(self, site: str) -> Optional[bool]:
        res = self._exec_quiet(
            self.nodes[site],
            f"ls {self.REMOTE_WD}/progress.json "
            f"{self.REMOTE_WD}/{self.checkpoint_file} "
            f"2>/dev/null | head -1", f"jobstate:{site}")
        return None if res is None else bool((res[0] or "").strip())

    def get_progress(self, site: str) -> Optional[dict]:
        # Runs on the engine tick loop, so it must never block it: bounded
        # short, and a timeout just yields None (engine keeps the last value).
        res = self._exec_quiet(
            self.nodes[site],
            f"cat {self.REMOTE_WD}/progress.json 2>/dev/null || true",
            f"progress:{site}")
        if res is None:
            return None
        try:
            return json.loads((res[0] or "").strip())
        except (json.JSONDecodeError, AttributeError):
            return None

    def shutdown(self):
        pass  # leave the slice running; teardown is explicit via fabric/ scripts
