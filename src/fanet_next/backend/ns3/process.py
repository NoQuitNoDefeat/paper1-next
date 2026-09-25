"""POSIX supervision with bounded pipe I/O and exclusive per-run shared resources.

Ported from paper1-baseline (ns3_adapter/process.py, commit 8451021); the project
root, lock files, contrib sources and worker module are this project's.

The supervisor imports no native extension. Worker and ns-3 are its direct children;
it can kill and reap both even when either is stuck in an upstream semaphore wait.
"""

import hashlib
import math
import os
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from .wire import ProtocolError, decode_value

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NS3_ROOT = PROJECT_ROOT / "simulator" / "ns-3-dev"


MAX_EXCHANGE_SECONDS = 1800.0  # an INIT carries a whole episode (tens to hundreds of MB)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransportConfig:
    ns3_root: Path = DEFAULT_NS3_ROOT
    profile: str = "default"
    shm_bytes: int = 8 * 1024 * 1024
    tx_capacity: int = 2 * 1024 * 1024
    rx_capacity: int = 2 * 1024 * 1024
    timeout_seconds: float = 15.0

    def validate(self) -> None:
        if os.name != "posix":
            raise ValueError("this supervisor supports POSIX process groups only")
        for value in (self.shm_bytes, self.tx_capacity, self.rx_capacity):
            if type(value) is not int:
                raise ValueError("capacities must be integer bytes")
        if not (16384 <= self.shm_bytes <= 256 * 1024 * 1024
                and self.tx_capacity >= 4096 and self.rx_capacity >= 4096
                and self.tx_capacity + self.rx_capacity + 8192 <= self.shm_bytes):
            raise ValueError("shared-memory size/capacity preflight failed")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 60):
            raise ValueError("timeout must be finite and within (0, 60] seconds")
        if not self.profile or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                                   for c in self.profile):
            raise ValueError("invalid build profile")


def build_identity(root: Path) -> tuple[str, str, str]:
    """Require the selected tree's locked versions and current project source digest."""
    project = PROJECT_ROOT
    commits = []
    for lock, tree in (("ns3.lock", root), ("ns3-ai.lock", root / "contrib/ai")):
        values = dict(line.split("=", 1) for line in (project / "dependencies" / lock)
                      .read_text().splitlines() if line and not line.startswith("#"))
        actual = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5,
                                check=True).stdout.strip()
        if actual != values["commit"]:
            raise ValueError(f"{lock}: configured tree does not match lock")
        commits.append(actual)
    source = project / "simulator/ns-3-external-contrib/fanet-scheduler"
    files = sorted(p for p in source.rglob("*") if p.is_file()
                   and (p.suffix in {".cc", ".h"} or p.name == "CMakeLists.txt"))
    manifest = "".join(f"{p.relative_to(source).as_posix()}:"
                       f"{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in files)
    return hashlib.sha256(manifest.encode()).hexdigest(), *commits


class Supervisor:
    """One run, two directly owned children and one atomically reserved shared segment."""

    def __init__(self, config: TransportConfig, run: str):
        config.validate()
        self.config, self.run = config, run
        self.name = "p1ai_" + run[:20]
        self.root = config.ns3_root.expanduser().resolve()
        self.identity = build_identity(self.root)
        version = (self.root / "VERSION").read_text().strip()
        self.binary = self.root / "build/contrib/fanet-scheduler/examples" / (
            f"ns{version}-fanet-bridge-{config.profile}"
        )
        self.binding_dir = self.root / "build/bindings/fanet-scheduler"
        if not self.binary.is_file() or not any(
            (self.binding_dir / f"fanet_bridge_native{s}").is_file() for s in EXTENSION_SUFFIXES
        ):
            raise FileNotFoundError("build project fanet-bridge and fanet_bridge_native first")
        self.worker = self.peer = None
        self._owned = False
        self._closed = False
        self._last_diagnostics = ""
        self._log = tempfile.TemporaryFile(mode="w+b")
        try:
            self._control("--reserve", str(config.shm_bytes))
            self._owned = True
            self.worker = subprocess.Popen(
                [sys.executable, "-I", "-B", "-m", "fanet_next.backend.ns3.worker",
                 "--binding-dir", str(self.binding_dir), "--name", self.name, "--run", run,
                 "--shm", str(config.shm_bytes), "--tx", str(config.tx_capacity),
                 "--rx", str(config.rx_capacity)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, start_new_session=True, bufsize=0,
            )
            os.set_blocking(self.worker.stdin.fileno(), False)
            os.set_blocking(self.worker.stdout.fileno(), False)
            ready = decode_value(self._read_frame(time.monotonic() + config.timeout_seconds))
            source, ns3, ai = self.identity
            if (set(ready) != {"ready", "build"} or ready["ready"] is not True
                    or len(ready["build"]) != 4 or ready["build"][0] != source
                    or ready["build"][1] != ns3 or ready["build"][3] != ai):
                raise ProtocolError("native binding provenance mismatch; rebuild project targets")
            self.peer = subprocess.Popen(
                [str(self.binary), "--serve", self.name, run, str(config.shm_bytes),
                 str(config.tx_capacity), str(config.rx_capacity)],
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log,
                start_new_session=True,
            )
        except BaseException as error:
            self._last_diagnostics = self.diagnostics()
            self.abort()
            if isinstance(error, Exception):
                raise ProtocolError(f"{error}\n{self._last_diagnostics}".strip()) from error
            raise

    def _control(self, mode, *args):
        result = subprocess.run([str(self.binary), mode, self.name, self.run, *args],
                                capture_output=True, text=True, timeout=5, check=False)
        if result.returncode:
            raise ProtocolError(result.stderr.strip() or "shared resource helper failed")
        return result.stdout.strip()

    def _io(self, size, deadline, outgoing=None):
        done = bytearray()
        sent = 0
        view = memoryview(outgoing) if outgoing is not None else None  # no per-write copies
        while (len(done) if outgoing is None else sent) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("ns3-ai exchange exceeded wall-clock timeout")
            stream = self.worker.stdout if outgoing is None else self.worker.stdin
            fd = stream.fileno()
            r, w, _ = select.select([fd] if outgoing is None else [],
                                     [] if outgoing is None else [fd], [], min(remaining, 0.05))
            if r:
                data = os.read(fd, size - len(done))
                if not data:
                    raise ProtocolError("native worker exited before a complete response")
                done.extend(data)
            elif w:
                count = os.write(fd, view[sent:])
                if count == 0:
                    raise ProtocolError("worker input pipe closed")
                sent += count
            elif self.worker.poll() is not None or (
                self.peer is not None and self.peer.poll() is not None
            ):
                raise ProtocolError("peer exited before completing the exchange")
        return bytes(done)

    def _read_frame(self, deadline):
        size = struct.unpack("<I", self._io(4, deadline))[0]
        if size == 0 or size > self.config.rx_capacity:
            raise ProtocolError("worker response exceeds capacity")
        return self._io(size, deadline)

    def exchange(self, data: bytes, timeout: float | None = None) -> bytes:
        """One request/response; ``timeout`` (default: the transport's) bounds the whole exchange."""
        if self._closed:
            raise ProtocolError("transport is closed")
        if not 0 < len(data) <= self.config.tx_capacity:
            raise ValueError("outgoing frame exceeds capacity")
        timeout = self.config.timeout_seconds if timeout is None else float(timeout)
        if not 0 < timeout <= MAX_EXCHANGE_SECONDS:
            raise ValueError(f"exchange timeout must be within (0, {MAX_EXCHANGE_SECONDS}] s")
        deadline = time.monotonic() + timeout
        try:
            self._io(4, deadline, struct.pack("<I", len(data)))
            self._io(len(data), deadline, data)
            return self._read_frame(deadline)
        except BaseException as error:
            self._last_diagnostics = self.diagnostics()
            self.abort()
            if isinstance(error, Exception):
                raise ProtocolError(f"{error}\n{self._last_diagnostics}".strip()) from error
            raise

    def finish(self) -> None:
        """Called only after validated FINAL/CLOSED; require both children to exit cleanly."""
        try:
            self._io(4, time.monotonic() + self.config.timeout_seconds, b"\0\0\0\0")
            for process in (self.peer, self.worker):
                if process.wait(timeout=self.config.timeout_seconds) != 0:
                    raise ProtocolError("child failed during final close")
        finally:
            self.abort()

    def diagnostics(self) -> str:
        if self._log.closed:
            return self._last_diagnostics
        self._log.seek(0, os.SEEK_END)
        self._log.seek(max(0, self._log.tell() - 8192))
        return self._log.read().decode("utf-8", errors="replace")

    def abort(self) -> None:
        """Idempotently reap owned children, then remove only our token-matching segment."""
        if self._closed:
            return
        for process in (self.peer, self.worker):
            if process is not None:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=5)
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
        try:
            if self._owned:
                self._control("--cleanup")
                if self._control("--probe") != "absent":
                    raise ProtocolError("owned segment remains after cleanup")
                self._owned = False
            self._closed = True
        finally:
            self._log.close()
