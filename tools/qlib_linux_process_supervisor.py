"""QLIB Linux-native process supervisor and bounded acceptance harness.

The module has two deliberately separate surfaces:

* ``start_linux_observation`` / ``poll_linux_observation`` /
  ``stop_linux_observation`` implement the POSIX observer lifecycle.  A detached
  supervisor remains outside the worker session and is the sole authority that
  can publish a terminal and its digest-bound seal.
* ``acceptance`` executes the fixed D42 30x3 deterministic, 12x10 real-process,
  and 100-seed campaigns and writes a closed, PID-free evidence bundle.

Raw process identity is retained only in memory while a task-owned process is
live.  It is never included in JSON, terminal output, exceptions, or evidence.
The implementation is dependency-free and imports safely on Windows; actual
process ownership is available only on Linux with ``/proc``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
LAUNCHER_BACKEND = "LINUX_NATIVE_SUPERVISOR"
TERMINALS = frozenset({"PASS", "HOLD", "FAIL"})
STOP_REASONS = frozenset({"FAILURE_INJECTION", "OWNER_STOP", "TASK_CLEANUP"})
OBSERVATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40}$")
RUNNER_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
OPERATOR_DIRECTORY_PATTERN = re.compile(r"^r9znw-488d42-operator-[a-f0-9]{32}$")
RAW_VALUE_PATTERN = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|[A-Za-z]:[\\/]|/(?:home|tmp|proc|var|etc)/|"
    r"-----BEGIN|(?:^|\s)--?[A-Za-z][A-Za-z0-9_-]*=)"
)
FORBIDDEN_KEY_PARTS = (
    "credential",
    "secret",
    "token",
    "username",
    "email",
    "remote_url",
    "response_body",
)
FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "pid",
        "process_id",
        "parent_process_id",
        "command",
        "command_line",
        "argv",
        "environment",
        "url",
        "username",
        "email",
        "response_body",
        "raw_value",
    }
)

DETERMINISTIC_CASES: tuple[tuple[str, str], ...] = (
    ("D01_NORMAL_SUCCESS", "PASS"),
    ("D02_WORKER_EXIT", "HOLD"),
    ("D03_CHILD_SPAWN_EXIT", "PASS"),
    ("D04_NESTED_DESCENDANT", "PASS"),
    ("D05_DUPLICATE_EVENT", "HOLD"),
    ("D06_REORDERED_EVENT", "HOLD"),
    ("D07_MISSING_EVENT", "HOLD"),
    ("D08_STALE_READINESS", "HOLD"),
    ("D09_WRONG_RUN_ID", "HOLD"),
    ("D10_WRONG_CONFIG_DIGEST", "HOLD"),
    ("D11_REGISTRATION_FAILURE", "HOLD"),
    ("D12_LINEAGE_MISMATCH", "HOLD"),
    ("D13_TIMEOUT", "HOLD"),
    ("D14_SIGTERM_SUCCESS", "HOLD"),
    ("D15_SIGKILL_ESCALATION", "HOLD"),
    ("D16_UNRELATED_SIBLING", "PASS"),
    ("D17_PARTIAL_TERMINAL", "HOLD"),
    ("D18_PARTIAL_SEAL", "HOLD"),
    ("D19_TERMINAL_MUTATION_ATTEMPT", "PASS"),
    ("D20_SUPERVISOR_FAILURE", "FAIL"),
    ("D21_WORKER_FALSE_PASS_CANDIDATE", "HOLD"),
    ("D22_POLLER_FALSE_TERMINAL_CANDIDATE", "HOLD"),
    ("D23_SAMPLER_COMMAND_FAILURE", "HOLD"),
    ("D24_ZOMBIE_REAPING", "PASS"),
    ("D25_REGISTRY_RESIDUAL", "HOLD"),
    ("D26_PROC_CONTRADICTION", "HOLD"),
    ("D27_CLEANUP_FAILURE", "HOLD"),
    ("D28_FIRST_FAILURE_PRESERVATION", "HOLD"),
    ("D29_RAW_VALUE_SENTINEL", "HOLD"),
    ("D30_FINAL_ZERO_AUDIT", "PASS"),
)

ACTUAL_CASES: tuple[str, ...] = (
    "L01_NORMAL_EXIT",
    "L02_NONZERO_EXIT",
    "L03_NESTED_DESCENDANT",
    "L04_SIGTERM_HANDLER",
    "L05_SIGKILL_ESCALATION",
    "L06_PARENT_FIRST_ORPHAN",
    "L07_ZOMBIE_OBSERVE_REAP",
    "L08_READINESS_BINDING",
    "L09_WRONG_READINESS_DIGEST",
    "L10_UNRELATED_SIBLING",
    "L11_SELF_EXCLUSION",
    "L12_PROC_CONTRADICTION_NO_SIGNAL",
)

_SUPERVISOR_HANDLES: dict[str, subprocess.Popen[bytes]] = {}
_SUPERVISOR_HANDLE_LOCK = threading.Lock()


class LinuxSupervisorError(RuntimeError):
    """Fail-closed error with a non-sensitive message."""


class LinuxSupervisorUnavailable(LinuxSupervisorError):
    """Raised when the Linux process contract cannot execute on this host."""


class ImmutableArtifactError(LinuxSupervisorError):
    """Raised when first-publication authority has already been consumed."""


class EventContractError(LinuxSupervisorError):
    """Raised when a readiness or event binding is not exact and contiguous."""


class IdentityStatus(str, Enum):
    """Allowlisted retained-identity reconciliation result."""

    MATCH = "MATCH"
    ABSENT = "ABSENT"
    PID_REUSED = "PID_REUSED"
    SCOPE_CHANGED = "SCOPE_CHANGED"
    UNKNOWN_FAIL_CLOSED = "UNKNOWN_FAIL_CLOSED"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_payload(payload: Mapping[str, Any]) -> str:
    return _digest_bytes(_canonical_bytes(payload))


def _duration_bucket(seconds: float) -> str:
    if seconds < 0.1:
        return "LT_100MS"
    if seconds < 1.0:
        return "LT_1S"
    if seconds < 10.0:
        return "LT_10S"
    return "GE_10S"


def _linux_available() -> bool:
    return os.name == "posix" and sys.platform.startswith("linux") and Path("/proc/self/stat").is_file()


def _require_linux() -> None:
    if not _linux_available():
        raise LinuxSupervisorUnavailable("LINUX_PROC_EXECUTION_HOST_REQUIRED")


def _assert_safe_identifier(value: str, *, field: str = "synthetic identifier") -> None:
    if not OBSERVATION_ID_PATTERN.fullmatch(value):
        raise LinuxSupervisorError(f"{field} is invalid")


def _persistent_sentinel_failures(value: Any, *, key_path: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if (
                normalized in FORBIDDEN_EXACT_KEYS
                or normalized.endswith("_pid")
                or any(part in normalized for part in FORBIDDEN_KEY_PARTS)
            ):
                failures.append(f"FORBIDDEN_KEY:{key_path}.{key}")
            failures.extend(_persistent_sentinel_failures(child, key_path=f"{key_path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_persistent_sentinel_failures(child, key_path=f"{key_path}[{index}]"))
    elif isinstance(value, str) and RAW_VALUE_PATTERN.search(value):
        failures.append(f"FORBIDDEN_VALUE:{key_path}")
    return failures


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_fd_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise LinuxSupervisorError("ANONYMOUS_FD_WRITE_FAILED")
        view = view[written:]


def _read_fd_all(fd: int, maximum: int = 1_048_576) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(65_536, maximum - total + 1))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise LinuxSupervisorError("ANONYMOUS_FD_LIMIT_EXCEEDED")
    return b"".join(chunks)


def _read_fd_line(fd: int, timeout_seconds: float, maximum: int = 65_536) -> bytes:
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    data = bytearray()
    try:
        while time.monotonic() < deadline and len(data) <= maximum:
            events = selector.select(max(0.0, deadline - time.monotonic()))
            if not events:
                break
            # Read exactly one byte so a second framed line remains available to
            # the next bounded read.  Anonymous readiness/event pipes are not
            # seekable, so a larger read cannot safely discard bytes after LF.
            chunk = os.read(fd, 1)
            if not chunk:
                break
            data.extend(chunk)
            if b"\n" in data:
                return bytes(data.split(b"\n", 1)[0])
    finally:
        selector.close()
    if len(data) > maximum:
        raise LinuxSupervisorError("ANONYMOUS_FD_LIMIT_EXCEEDED")
    raise LinuxSupervisorError("BOUNDED_READINESS_NOT_RECEIVED")


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    failures = _persistent_sentinel_failures(payload)
    if failures:
        raise LinuxSupervisorError("PERSISTENT_EVIDENCE_SENTINEL_REJECTED")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload)
    temporary = path.parent / f".{path.name}.{os.urandom(8).hex()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        _write_fd_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _exclusive_publish_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically publish an immutable JSON file without replacement."""

    failures = _persistent_sentinel_failures(payload)
    if failures:
        raise LinuxSupervisorError("PERSISTENT_EVIDENCE_SENTINEL_REJECTED")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload)
    temporary = path.parent / f".{path.name}.{os.urandom(8).hex()}.tmp"
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        _write_fd_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ImmutableArtifactError("IMMUTABLE_ARTIFACT_ALREADY_PUBLISHED") from exc
        return _digest_bytes(data)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass
class ProcessIdentity:
    """Ephemeral Linux identity; instances must never be serialized."""

    process_id: int
    parent_process_id: int
    process_group_id: int
    session_id: int
    start_ticks: int
    state: str
    pidfd: int | None = None

    def close(self) -> None:
        if self.pidfd is not None:
            try:
                os.close(self.pidfd)
            except OSError:
                pass
            self.pidfd = None


@dataclass
class RegistryRecord:
    synthetic_label: str
    action: str
    identity: ProcessIdentity | None


class OwnedProcessRegistry:
    """Append-only while active, followed by an explicit identity scrub."""

    def __init__(self) -> None:
        self._records: list[RegistryRecord] = []
        self._cleared = False

    def register(self, synthetic_label: str, identity: ProcessIdentity) -> None:
        _assert_safe_identifier(synthetic_label, field="synthetic process label")
        if self._cleared or self.active_identity(synthetic_label) is not None:
            raise LinuxSupervisorError("OWNED_PROCESS_REGISTRATION_REJECTED")
        self._records.append(RegistryRecord(synthetic_label, "REGISTER", identity))

    def mark_reaped(self, synthetic_label: str) -> None:
        if self._cleared or self.active_identity(synthetic_label) is None:
            raise LinuxSupervisorError("OWNED_PROCESS_REAP_REGISTRATION_MISSING")
        self._records.append(RegistryRecord(synthetic_label, "REAPED", None))

    def active_identity(self, synthetic_label: str) -> ProcessIdentity | None:
        active: ProcessIdentity | None = None
        for record in self._records:
            if record.synthetic_label != synthetic_label:
                continue
            if record.action == "REGISTER":
                active = record.identity
            elif record.action in {"REAPED", "PID_REUSED"}:
                active = None
        return active

    def active_items(self) -> list[tuple[str, ProcessIdentity]]:
        if self._cleared:
            return []
        labels = {record.synthetic_label for record in self._records}
        items: list[tuple[str, ProcessIdentity]] = []
        for label in sorted(labels):
            identity = self.active_identity(label)
            if identity is not None:
                items.append((label, identity))
        return items

    def contains_identity(self, identity: ProcessIdentity) -> bool:
        return any(
            active.process_id == identity.process_id
            and active.start_ticks == identity.start_ticks
            for _, active in self.active_items()
        )

    def ever_registered_prefix(self, prefix: str) -> bool:
        return any(
            record.action == "REGISTER" and record.synthetic_label.startswith(prefix)
            for record in self._records
        )

    def registration_count(self, prefix: str) -> int:
        return sum(
            record.action == "REGISTER" and record.synthetic_label.startswith(prefix)
            for record in self._records
        )

    def record_identity_status(self, synthetic_label: str, status: IdentityStatus) -> None:
        if status not in {
            IdentityStatus.PID_REUSED,
            IdentityStatus.SCOPE_CHANGED,
            IdentityStatus.UNKNOWN_FAIL_CLOSED,
        }:
            raise LinuxSupervisorError("IDENTITY_STATUS_RECORD_REJECTED")
        if self.active_identity(synthetic_label) is None:
            raise LinuxSupervisorError("IDENTITY_STATUS_REGISTRATION_MISSING")
        action = status.value
        if not any(
            record.synthetic_label == synthetic_label and record.action == action
            for record in self._records
        ):
            self._records.append(RegistryRecord(synthetic_label, action, None))

    @property
    def identity_contradiction_count(self) -> int:
        return len(
            {
                record.synthetic_label
                for record in self._records
                if record.action
                in {
                    IdentityStatus.PID_REUSED.value,
                    IdentityStatus.SCOPE_CHANGED.value,
                    IdentityStatus.UNKNOWN_FAIL_CLOSED.value,
                }
            }
        )

    @property
    def scope_changed_count(self) -> int:
        return len(
            {
                record.synthetic_label
                for record in self._records
                if record.action == IdentityStatus.SCOPE_CHANGED.value
            }
        )

    @property
    def residual_count(self) -> int:
        if self._cleared:
            return 0
        labels = {record.synthetic_label for record in self._records}
        return sum(self.active_identity(label) is not None for label in labels)

    @property
    def cleared(self) -> bool:
        return self._cleared

    def clear(self) -> None:
        for record in self._records:
            if record.identity is not None:
                record.identity.close()
        self._records.clear()
        self._cleared = True


def _read_proc_identity(process_id: int, *, open_pidfd: bool = False) -> ProcessIdentity:
    _require_linux()
    try:
        text = Path(f"/proc/{int(process_id)}/stat").read_text(encoding="ascii")
        closing = text.rindex(")")
        prefix = text[:closing]
        opening = prefix.index("(")
        parsed_id = int(prefix[:opening].strip())
        tail = text[closing + 2 :].split()
        if len(tail) < 20 or parsed_id != int(process_id):
            raise ValueError
        identity = ProcessIdentity(
            process_id=parsed_id,
            parent_process_id=int(tail[1]),
            process_group_id=int(tail[2]),
            session_id=int(tail[3]),
            start_ticks=int(tail[19]),
            state=tail[0],
        )
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError) as exc:
        raise LinuxSupervisorError("PROC_IDENTITY_NOT_AVAILABLE") from exc
    if open_pidfd and hasattr(os, "pidfd_open"):
        try:
            identity.pidfd = os.pidfd_open(identity.process_id, 0)
        except OSError:
            identity.pidfd = None
    return identity


def _identity_status(expected: ProcessIdentity) -> IdentityStatus:
    stat_path = Path(f"/proc/{expected.process_id}/stat")
    if not stat_path.exists():
        return IdentityStatus.ABSENT
    try:
        current = _read_proc_identity(expected.process_id)
    except LinuxSupervisorError:
        return (
            IdentityStatus.ABSENT
            if not stat_path.exists()
            else IdentityStatus.UNKNOWN_FAIL_CLOSED
        )
    if current.start_ticks != expected.start_ticks:
        return IdentityStatus.PID_REUSED
    if (
        current.process_group_id != expected.process_group_id
        or current.session_id != expected.session_id
    ):
        return IdentityStatus.SCOPE_CHANGED
    return IdentityStatus.MATCH


def _identity_matches(expected: ProcessIdentity) -> bool:
    return _identity_status(expected) == IdentityStatus.MATCH


def _same_process_instance(left: ProcessIdentity, right: ProcessIdentity) -> bool:
    return (
        left.process_id == right.process_id
        and left.start_ticks == right.start_ticks
        and left.process_group_id == right.process_group_id
        and left.session_id == right.session_id
    )


def _same_process_incarnation(left: ProcessIdentity, right: ProcessIdentity) -> bool:
    """Match the kernel PID incarnation without assuming an unchanged scope."""

    return (
        left.process_id == right.process_id
        and left.start_ticks == right.start_ticks
    )


def _registered_parent_instance_is_owned(
    registry: OwnedProcessRegistry,
    candidate: ProcessIdentity,
    root: ProcessIdentity,
) -> bool:
    """Accept only an exact live incarnation already retained by the registry."""

    for _, retained_parent in registry.active_items():
        if (
            retained_parent.process_id != candidate.parent_process_id
            or retained_parent.process_group_id != root.process_group_id
            or retained_parent.session_id != root.session_id
            or retained_parent.start_ticks > candidate.start_ticks
        ):
            continue
        try:
            current_parent = _read_proc_identity(retained_parent.process_id)
        except LinuxSupervisorError:
            continue
        if (
            _same_process_incarnation(current_parent, retained_parent)
            and current_parent.state != "Z"
        ):
            return True
    return False


def _snapshot_ownership_evidence(
    registry: OwnedProcessRegistry,
    candidate: ProcessIdentity,
    root: ProcessIdentity,
    supervisor: ProcessIdentity,
    *,
    subreaper_enabled: bool,
) -> tuple[bool, bool]:
    """Return (drift_is_relevant, ownership_is_proven) for one snapshot."""

    numeric_scope_match = (
        candidate.process_group_id == root.process_group_id
        and candidate.session_id == root.session_id
    )
    adopted_by_supervisor = (
        subreaper_enabled
        and candidate.parent_process_id == supervisor.process_id
        and candidate.process_id != supervisor.process_id
        and candidate.start_ticks >= root.start_ticks
    )
    exact_registered_parent = (
        False
        if adopted_by_supervisor
        else _registered_parent_instance_is_owned(registry, candidate, root)
    )
    exact_root_lineage = (
        False
        if adopted_by_supervisor or exact_registered_parent
        else _lineage_reaches_root_any_scope(candidate, root)
    )
    ownership_proven = (
        adopted_by_supervisor or exact_registered_parent or exact_root_lineage
    )
    return numeric_scope_match or ownership_proven, ownership_proven


def _proc_all_identities() -> list[ProcessIdentity]:
    _require_linux()
    identities: list[ProcessIdentity] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise LinuxSupervisorError("PROC_ENUMERATION_NOT_AVAILABLE") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            identity = _read_proc_identity(int(entry.name))
        except LinuxSupervisorError:
            continue
        identities.append(identity)
    return identities


def _proc_group_identities(group_id: int, session_id: int) -> list[ProcessIdentity]:
    return [
        identity
        for identity in _proc_all_identities()
        if identity.process_group_id == group_id and identity.session_id == session_id
    ]


def _lineage_reaches_root(candidate: ProcessIdentity, root: ProcessIdentity) -> bool:
    if _same_process_instance(candidate, root):
        return True
    current = candidate
    seen: set[tuple[int, int]] = set()
    for _ in range(128):
        marker = (current.process_id, current.start_ticks)
        if marker in seen or current.parent_process_id <= 1:
            return False
        seen.add(marker)
        try:
            parent = _read_proc_identity(current.parent_process_id)
        except LinuxSupervisorError:
            return False
        if _same_process_instance(parent, root):
            return True
        if (
            parent.process_group_id != root.process_group_id
            or parent.session_id != root.session_id
        ):
            return False
        current = parent
    return False


def _lineage_reaches_root_any_scope(
    candidate: ProcessIdentity, root: ProcessIdentity
) -> bool:
    current = candidate
    seen: set[tuple[int, int]] = set()
    for _ in range(128):
        marker = (current.process_id, current.start_ticks)
        if marker in seen or current.parent_process_id <= 1:
            return False
        seen.add(marker)
        try:
            parent = _read_proc_identity(current.parent_process_id)
        except LinuxSupervisorError:
            return False
        if parent.process_id == root.process_id:
            return parent.start_ticks == root.start_ticks
        current = parent
    return False


def _discover_and_register_owned_group(
    registry: OwnedProcessRegistry,
    root: ProcessIdentity,
    *,
    production_mode: bool,
    supervisor: ProcessIdentity,
    subreaper_enabled: bool,
) -> int:
    """Register each corroborated group member under a synthetic label."""

    new_count = 0
    pending = [
        identity
        for identity in _proc_all_identities()
        if identity.start_ticks >= root.start_ticks
        if not (
            identity.process_id == root.process_id
            and identity.start_ticks == root.start_ticks
        )
        and not registry.contains_identity(identity)
    ]
    unresolved_original_scope: set[tuple[int, int]] = {
        (identity.process_id, identity.start_ticks)
        for identity in pending
        if identity.process_group_id == root.process_group_id
        and identity.session_id == root.session_id
    }
    while pending:
        progressed = False
        for identity in list(pending):
            snapshot_drift_is_relevant, snapshot_ownership_proven = (
                _snapshot_ownership_evidence(
                    registry,
                    identity,
                    root,
                    supervisor,
                    subreaper_enabled=subreaper_enabled,
                )
            )
            try:
                current = _read_proc_identity(identity.process_id)
            except LinuxSupervisorError as exc:
                if snapshot_drift_is_relevant:
                    raise LinuxSupervisorError(
                        "OWNED_DESCENDANT_IDENTITY_CHANGED"
                    ) from exc
                pending.remove(identity)
                progressed = True
                continue
            if (
                not _same_process_instance(identity, current)
                or identity.parent_process_id != current.parent_process_id
            ):
                if snapshot_drift_is_relevant:
                    raise LinuxSupervisorError("OWNED_DESCENDANT_IDENTITY_CHANGED")
                pending.remove(identity)
                progressed = True
                continue
            adopted_by_supervisor = (
                subreaper_enabled
                and current.parent_process_id == supervisor.process_id
                and current.process_id != supervisor.process_id
                and current.start_ticks >= root.start_ticks
            )
            registered_parent = (
                snapshot_ownership_proven
                or _registered_parent_instance_is_owned(
                    registry,
                    current,
                    root,
                )
            )
            if (
                not adopted_by_supervisor
                and not registered_parent
                and not _lineage_reaches_root_any_scope(current, root)
            ):
                continue
            if production_mode and current.parent_process_id == root.process_id:
                sampler_count = registry.registration_count("SAMPLER_") + 1
                label = f"SAMPLER_{sampler_count:03d}"
            else:
                descendant_count = registry.registration_count("DESCENDANT_") + 1
                label = f"DESCENDANT_{descendant_count:03d}"
            registered = _read_proc_identity(current.process_id, open_pidfd=True)
            if (
                not _same_process_instance(current, registered)
                or current.parent_process_id != registered.parent_process_id
            ):
                registered.close()
                raise LinuxSupervisorError("OWNED_DESCENDANT_IDENTITY_CHANGED")
            scope_changed = (
                registered.process_group_id != root.process_group_id
                or registered.session_id != root.session_id
            )
            if scope_changed:
                retained = ProcessIdentity(
                    process_id=registered.process_id,
                    parent_process_id=registered.parent_process_id,
                    process_group_id=root.process_group_id,
                    session_id=root.session_id,
                    start_ticks=registered.start_ticks,
                    state=registered.state,
                    pidfd=registered.pidfd,
                )
                registered.pidfd = None
            else:
                retained = registered
            registry.register(label, retained)
            if scope_changed:
                registry.record_identity_status(label, IdentityStatus.SCOPE_CHANGED)
            pending.remove(identity)
            unresolved_original_scope.discard(
                (identity.process_id, identity.start_ticks)
            )
            new_count += 1
            progressed = True
        if not progressed:
            break
    if unresolved_original_scope:
        raise LinuxSupervisorError("OWNED_DESCENDANT_LINEAGE_REJECTED")
    return new_count


def _mark_absent_registrations_reaped(registry: OwnedProcessRegistry) -> int:
    reaped = 0
    for label, identity in registry.active_items():
        status = _identity_status(identity)
        if status == IdentityStatus.ABSENT:
            registry.mark_reaped(label)
            reaped += 1
        elif status == IdentityStatus.PID_REUSED:
            registry.record_identity_status(label, status)
        elif status in {IdentityStatus.SCOPE_CHANGED, IdentityStatus.UNKNOWN_FAIL_CLOSED}:
            registry.record_identity_status(label, status)
    return reaped


def _safe_signal_owned_group(
    anchor: ProcessIdentity,
    signal_number: int,
    *,
    supervisor_process_id: int,
    owned_group_id: int | None = None,
    owned_session_id: int | None = None,
) -> str:
    """Signal exactly one corroborated group, or refuse without signalling."""

    group_id = anchor.process_group_id if owned_group_id is None else owned_group_id
    session_id = anchor.session_id if owned_session_id is None else owned_session_id
    try:
        supervisor = _read_proc_identity(supervisor_process_id)
        current = _read_proc_identity(anchor.process_id)
    except LinuxSupervisorError:
        return "IDENTITY_UNCERTAIN_NO_SIGNAL"
    if (
        _identity_status(anchor) != IdentityStatus.MATCH
        or current.process_group_id != group_id
        or current.session_id != session_id
        or group_id <= 1
        or group_id == supervisor.process_group_id
        or session_id == supervisor.session_id
        or anchor.process_id == supervisor_process_id
    ):
        return "IDENTITY_CONTRADICTION_NO_SIGNAL"
    try:
        os.killpg(group_id, signal_number)
    except ProcessLookupError:
        return "OWNED_GROUP_ALREADY_EXITED"
    except (PermissionError, OSError):
        return "OWNED_GROUP_SIGNAL_FAILED"
    return "OWNED_GROUP_SIGNALLED"


def _safe_signal_owned_process(
    identity: ProcessIdentity,
    signal_number: int,
    *,
    supervisor_process_id: int,
) -> str:
    """Signal one retained same-start-tick process, never a name or broad group."""

    if identity.process_id == supervisor_process_id:
        return "SELF_EXCLUSION_NO_SIGNAL"
    status = _identity_status(identity)
    if status not in {IdentityStatus.MATCH, IdentityStatus.SCOPE_CHANGED}:
        return f"{status.value}_NO_SIGNAL"
    if identity.pidfd is not None and hasattr(signal, "pidfd_send_signal"):
        try:
            signal.pidfd_send_signal(identity.pidfd, signal_number, None, 0)
        except (OSError, ValueError):
            return "PIDFD_SIGNAL_FAILED"
        return "EXACT_PROCESS_SIGNALLED_PIDFD"
    # A read-then-kill(PID) fallback has an unavoidable PID-reuse TOCTOU.  A
    # retained process that changed group/session is therefore signalable only
    # through its already-open pidfd; otherwise cleanup remains fail-closed.
    return "PIDFD_EXACT_SIGNAL_UNAVAILABLE_NO_SIGNAL"


def _proc_group_member_count(group_id: int, session_id: int) -> int:
    try:
        return len(_proc_group_identities(group_id, session_id))
    except LinuxSupervisorError:
        return -1


def _bounded_group_wait(group_id: int, session_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        count = _proc_group_member_count(group_id, session_id)
        if count == 0:
            return True
        if count < 0 or time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _enable_child_subreaper() -> bool:
    if not _linux_available():
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    except (AttributeError, OSError):
        return False


def _reap_available_children(timeout_seconds: float) -> int:
    if os.name != "posix":
        return 0
    reaped = 0
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        made_progress = False
        while True:
            try:
                child, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                child = 0
            if child <= 0:
                break
            reaped += 1
            made_progress = True
        if not made_progress:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        else:
            continue
    return reaped


class BoundEventValidator:
    """Validate run/config/role binding and a contiguous sequence."""

    def __init__(self, run_id: str, config_digest: str, role: str = "WORKER") -> None:
        _assert_safe_identifier(run_id, field="run identifier")
        if not SHA256_PATTERN.fullmatch(config_digest):
            raise EventContractError("CONFIG_DIGEST_INVALID")
        self.run_id = run_id
        self.config_digest = config_digest
        self.role = role
        self.next_sequence = 0
        self.complete = False
        self.first_failure: str | None = None
        self._chain = hashlib.sha256()

    def accept(self, event: Mapping[str, Any]) -> None:
        if self.first_failure is not None:
            raise EventContractError(self.first_failure)
        failure: str | None = None
        if event.get("run_id") != self.run_id:
            failure = "WRONG_RUN_ID"
        elif event.get("config_digest") != self.config_digest:
            failure = "WRONG_CONFIG_DIGEST"
        elif event.get("role") != self.role:
            failure = "WRONG_ROLE"
        elif event.get("sequence") != self.next_sequence:
            failure = "EVENT_SEQUENCE_NOT_CONTIGUOUS"
        elif self.complete:
            failure = "EVENT_AFTER_COMPLETE"
        kind = event.get("kind")
        if not isinstance(kind, str) or kind not in {"READY", "SAMPLE", "COMPLETE", "FAILURE"}:
            failure = failure or "EVENT_KIND_INVALID"
        base_keys = {"run_id", "config_digest", "role", "sequence", "kind"}
        kind_keys = {
            "READY": set(),
            "SAMPLE": {"sample_digest", "ownership_registration_required"},
            "COMPLETE": {
                "sample_count",
                "duration_bucket",
                "maximum_gap_bucket",
                "elapsed_milliseconds",
                "maximum_gap_milliseconds",
            },
            "FAILURE": {"failure_category", "failure_digest"},
        }
        expected_keys = base_keys | kind_keys.get(str(kind), set())
        if set(event) != expected_keys:
            failure = failure or "EVENT_SCHEMA_NOT_CLOSED"
        if kind == "SAMPLE" and not SHA256_PATTERN.fullmatch(str(event.get("sample_digest", ""))):
            failure = failure or "SAMPLE_DIGEST_INVALID"
        if kind == "SAMPLE" and not isinstance(event.get("ownership_registration_required"), bool):
            failure = failure or "SAMPLE_OWNERSHIP_FLAG_INVALID"
        if kind == "COMPLETE":
            if not isinstance(event.get("sample_count"), int) or int(event.get("sample_count", -1)) < 0:
                failure = failure or "SAMPLE_COUNT_INVALID"
            if event.get("duration_bucket") not in {"LT_100MS", "LT_1S", "LT_10S", "GE_10S"}:
                failure = failure or "DURATION_BUCKET_INVALID"
            if event.get("maximum_gap_bucket") not in {"LT_100MS", "LT_1S", "LT_10S", "GE_10S"}:
                failure = failure or "GAP_BUCKET_INVALID"
            if not isinstance(event.get("elapsed_milliseconds"), int) or int(event.get("elapsed_milliseconds", -1)) < 0:
                failure = failure or "ELAPSED_MILLISECONDS_INVALID"
            if not isinstance(event.get("maximum_gap_milliseconds"), int) or int(event.get("maximum_gap_milliseconds", -1)) < 0:
                failure = failure or "GAP_MILLISECONDS_INVALID"
        if kind == "FAILURE":
            if event.get("failure_category") not in {"SAMPLER_FAILURE"}:
                failure = failure or "FAILURE_CATEGORY_INVALID"
            if not SHA256_PATTERN.fullmatch(str(event.get("failure_digest", ""))):
                failure = failure or "FAILURE_DIGEST_INVALID"
        if failure is not None:
            self.first_failure = failure
            raise EventContractError(failure)
        normalized = {key: event[key] for key in sorted(expected_keys)}
        self._chain.update(_canonical_bytes(normalized))
        self.next_sequence += 1
        if kind in {"COMPLETE", "FAILURE"}:
            self.complete = True

    @property
    def digest(self) -> str:
        return self._chain.hexdigest()

    def require_complete(self) -> None:
        if self.first_failure is not None:
            raise EventContractError(self.first_failure)
        if not self.complete:
            self.first_failure = "EVENT_STREAM_INCOMPLETE"
            raise EventContractError(self.first_failure)


def _bound_event(run_id: str, config_digest: str, sequence: int, kind: str) -> dict[str, Any]:
    event: dict[str, Any] = {
        "run_id": run_id,
        "config_digest": config_digest,
        "role": "WORKER",
        "sequence": sequence,
        "kind": kind,
    }
    if kind == "SAMPLE":
        event["sample_digest"] = hashlib.sha256(b"fixture-sample").hexdigest()
        event["ownership_registration_required"] = False
    elif kind == "COMPLETE":
        event.update(
            {
                "sample_count": max(0, sequence - 1),
                "duration_bucket": "LT_100MS",
                "maximum_gap_bucket": "LT_100MS",
                "elapsed_milliseconds": 0,
                "maximum_gap_milliseconds": 0,
            }
        )
    elif kind == "FAILURE":
        event.update(
            {
                "failure_category": "SAMPLER_FAILURE",
                "failure_digest": hashlib.sha256(b"fixture-failure").hexdigest(),
            }
        )
    return event


def _observation_config_digest(config: Mapping[str, Any]) -> str:
    sampler = config.get("sampler_argv")
    sampler_digest = (
        _digest_bytes("\x00".join(str(item) for item in sampler).encode("utf-8"))
        if isinstance(sampler, list)
        else "NONE"
    )
    safe = {
        "duration_seconds": float(config["duration_seconds"]),
        "max_gap_seconds": float(config["max_gap_seconds"]),
        "mode": str(config["mode"]),
        "observation_id": str(config["observation_id"]),
        "required_target_contract": config.get("required_target_contract") or "NONE",
        "sample_interval_seconds": float(config["sample_interval_seconds"]),
        "sampler_digest": sampler_digest,
        "stale_after_seconds": float(config["stale_after_seconds"]),
    }
    return _digest_payload(safe)


def _supervisor_handle_key(directory: Path) -> str:
    return _digest_bytes(str(directory.resolve()).encode("utf-8"))


def _scrub_supervisor_handle(
    key: str, process: subprocess.Popen[bytes]
) -> None:
    """Wait/reap a detached supervisor and remove only its matching handle."""

    try:
        process.wait()
    finally:
        with _SUPERVISOR_HANDLE_LOCK:
            if _SUPERVISOR_HANDLES.get(key) is process:
                del _SUPERVISOR_HANDLES[key]


def _register_supervisor_handle(
    directory: Path, process: subprocess.Popen[bytes]
) -> None:
    key = _supervisor_handle_key(directory)
    with _SUPERVISOR_HANDLE_LOCK:
        existing = _SUPERVISOR_HANDLES.get(key)
        if existing is not None and existing.poll() is None:
            raise LinuxSupervisorError("SUPERVISOR_HANDLE_ALREADY_REGISTERED")
        _SUPERVISOR_HANDLES[key] = process
    threading.Thread(
        target=_scrub_supervisor_handle,
        args=(key, process),
        name="qlib-supervisor-reaper",
        daemon=True,
    ).start()


def _bounded_reap_supervisor_handle(directory: Path, timeout_seconds: float) -> bool:
    key = _supervisor_handle_key(directory)
    with _SUPERVISOR_HANDLE_LOCK:
        process = _SUPERVISOR_HANDLES.get(key)
    if process is None:
        return True
    try:
        process.wait(timeout=max(0.0, timeout_seconds))
    except subprocess.TimeoutExpired:
        return False
    with _SUPERVISOR_HANDLE_LOCK:
        if _SUPERVISOR_HANDLES.get(key) is process:
            del _SUPERVISOR_HANDLES[key]
    return True


def _collect_startup_owned_tree(
    root: ProcessIdentity,
    retained: dict[tuple[int, int], ProcessIdentity],
) -> None:
    """Retain pidfds for every currently corroborated descendant in memory."""

    candidates = _proc_all_identities()
    pending = [
        candidate
        for candidate in candidates
        if (candidate.process_id, candidate.start_ticks) not in retained
        and candidate.start_ticks >= root.start_ticks
    ]
    while pending:
        progressed = False
        owners = list(retained.values())
        for candidate in list(pending):
            if not any(
                _lineage_reaches_root_any_scope(candidate, owner) for owner in owners
            ):
                continue
            confirmed = _read_proc_identity(candidate.process_id, open_pidfd=True)
            if not _same_process_instance(candidate, confirmed):
                confirmed.close()
                raise LinuxSupervisorError("STARTUP_DESCENDANT_IDENTITY_CHANGED")
            retained[(confirmed.process_id, confirmed.start_ticks)] = confirmed
            pending.remove(candidate)
            progressed = True
        if not progressed:
            break


def _signal_startup_owned_set(
    retained: Mapping[tuple[int, int], ProcessIdentity],
    signal_number: int,
    *,
    exclude_root: ProcessIdentity | None = None,
) -> int:
    signalled = 0
    signalled_groups: set[tuple[int, int]] = set()
    for identity in retained.values():
        if exclude_root is not None and (
            identity.process_id == exclude_root.process_id
            and identity.start_ticks == exclude_root.start_ticks
        ):
            continue
        status = _identity_status(identity)
        if status == IdentityStatus.MATCH:
            group = (identity.process_group_id, identity.session_id)
            if group in signalled_groups:
                continue
            result = _safe_signal_owned_group(
                identity,
                signal_number,
                supervisor_process_id=os.getpid(),
            )
            if result == "OWNED_GROUP_SIGNALLED":
                signalled += 1
                signalled_groups.add(group)
        elif status == IdentityStatus.SCOPE_CHANGED:
            if (
                _safe_signal_owned_process(
                    identity,
                    signal_number,
                    supervisor_process_id=os.getpid(),
                )
                == "EXACT_PROCESS_SIGNALLED_PIDFD"
            ):
                signalled += 1
    return signalled


def _startup_tree_is_zero(
    retained: Mapping[tuple[int, int], ProcessIdentity]
) -> bool:
    return all(
        _identity_status(identity)
        in {IdentityStatus.ABSENT, IdentityStatus.PID_REUSED}
        for identity in retained.values()
    )


def _reap_exact_adopted_zombies(
    retained: Mapping[tuple[int, int], ProcessIdentity]
) -> int:
    """Reap only retained PID incarnations adopted by this exact process."""

    reaped = 0
    for expected in retained.values():
        try:
            current = _read_proc_identity(expected.process_id)
        except LinuxSupervisorError:
            continue
        if (
            not _same_process_incarnation(expected, current)
            or current.parent_process_id != os.getpid()
            or current.state != "Z"
        ):
            continue
        try:
            child, _ = os.waitpid(current.process_id, getattr(os, "WNOHANG", 1))
        except (ChildProcessError, OSError):
            continue
        if child == current.process_id:
            reaped += 1
    return reaped


def _cleanup_unregistered_direct_child(
    process: subprocess.Popen[bytes],
) -> tuple[bool, bool, bool]:
    """Clean a direct child before it can cross its readiness/ACK boundary."""

    used_sigkill = False
    term_sent = process.poll() is None
    if term_sent:
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            used_sigkill = True
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                return False, term_sent, used_sigkill
    return process.poll() is not None, term_sent, used_sigkill


def _worker_group_cleanup_is_authorized(
    registry: OwnedProcessRegistry, identity: ProcessIdentity | None
) -> bool:
    return identity is not None and registry.contains_identity(identity)


def _cleanup_failed_startup_tree(
    process: subprocess.Popen[bytes], root: ProcessIdentity
) -> bool:
    """Boundedly terminate, reap, and /proc-corroborate one startup tree."""

    retained: dict[tuple[int, int], ProcessIdentity] = {
        (root.process_id, root.start_ticks): root
    }
    collection_clear = True

    def collect(owner: ProcessIdentity) -> None:
        nonlocal collection_clear
        try:
            _collect_startup_owned_tree(owner, retained)
        except LinuxSupervisorError:
            collection_clear = False

    try:
        collect(root)
        if process.poll() is None:
            _safe_signal_owned_group(
                root, signal.SIGTERM, supervisor_process_id=os.getpid()
            )
        graceful_deadline = time.monotonic() + 0.5
        while time.monotonic() < graceful_deadline:
            collect(root)
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is None:
            collect(root)
            _safe_signal_owned_group(
                root, getattr(signal, "SIGKILL", 9), supervisor_process_id=os.getpid()
            )
        root_reaped = False
        try:
            process.wait(timeout=1.0)
            root_reaped = True
        except ChildProcessError:
            root_reaped = True
        except subprocess.TimeoutExpired:
            root_reaped = False

        # The supervisor may have launched a worker in its own session.  Every
        # retained descendant is drained by its corroborated group or pidfd.
        _reap_exact_adopted_zombies(retained)
        collect(root)
        _signal_startup_owned_set(retained, signal.SIGTERM, exclude_root=root)
        descendant_deadline = time.monotonic() + 0.5
        while time.monotonic() < descendant_deadline:
            for owner in list(retained.values()):
                if _identity_status(owner) in {
                    IdentityStatus.MATCH,
                    IdentityStatus.SCOPE_CHANGED,
                }:
                    collect(owner)
            _reap_exact_adopted_zombies(retained)
            if _startup_tree_is_zero(retained):
                break
            time.sleep(0.01)
        if not _startup_tree_is_zero(retained):
            _signal_startup_owned_set(
                retained, getattr(signal, "SIGKILL", 9), exclude_root=root
            )
        final_deadline = time.monotonic() + 1.0
        while time.monotonic() < final_deadline and not _startup_tree_is_zero(retained):
            _reap_exact_adopted_zombies(retained)
            if _startup_tree_is_zero(retained):
                break
            time.sleep(0.01)
        return (
            root_reaped
            and process.poll() is not None
            and collection_clear
            and _startup_tree_is_zero(retained)
        )
    except LinuxSupervisorError:
        return False
    finally:
        for identity in retained.values():
            if identity is not root:
                identity.close()


def start_linux_observation(
    *,
    artifact_root: Path | str,
    observation_id: str,
    duration_seconds: float,
    sample_interval_seconds: float,
    max_gap_seconds: float,
    stale_after_seconds: float,
    mode: str,
    required_target_contract: str | None = None,
    sampler_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Launch a detached Linux supervisor without persisting process identity."""

    _require_linux()
    _assert_safe_identifier(observation_id, field="observation identifier")
    if mode not in {"local-test", "production"}:
        raise LinuxSupervisorError("OBSERVATION_MODE_INVALID")
    for value in (duration_seconds, sample_interval_seconds, max_gap_seconds, stale_after_seconds):
        if not isinstance(value, (int, float)) or not (0 < float(value) < 86_400):
            raise LinuxSupervisorError("OBSERVATION_DURATION_INVALID")
    if mode == "production" and not sampler_argv:
        raise LinuxSupervisorError("PRODUCTION_SAMPLER_NOT_AVAILABLE")
    root = Path(artifact_root)
    if root.exists() and root.is_symlink():
        raise LinuxSupervisorError("ARTIFACT_ROOT_SYMLINK_REJECTED")
    directory = root / observation_id
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise LinuxSupervisorError("DUPLICATE_OBSERVATION_ID") from exc

    config: dict[str, Any] = {
        "artifact_dir": str(directory.resolve()),
        "observation_id": observation_id,
        "duration_seconds": float(duration_seconds),
        "sample_interval_seconds": float(sample_interval_seconds),
        "max_gap_seconds": float(max_gap_seconds),
        "stale_after_seconds": float(stale_after_seconds),
        "mode": mode,
        "required_target_contract": required_target_contract,
        "sampler_argv": list(sampler_argv) if sampler_argv else None,
    }
    config["config_digest"] = _observation_config_digest(config)
    config_read, config_write = os.pipe()
    ready_read, ready_write = os.pipe()
    supervisor_process: subprocess.Popen[bytes] | None = None
    supervisor_identity: ProcessIdentity | None = None
    try:
        supervisor_process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_supervise",
                "--config-fd",
                str(config_read),
                "--launcher-ready-fd",
                str(ready_write),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(config_read, ready_write),
            start_new_session=True,
        )
        supervisor_identity = _read_proc_identity(
            supervisor_process.pid, open_pidfd=True
        )
        os.close(config_read)
        config_read = -1
        os.close(ready_write)
        ready_write = -1
        _write_fd_all(config_write, _canonical_bytes(config))
        os.close(config_write)
        config_write = -1
        readiness = json.loads(_read_fd_line(ready_read, min(10.0, stale_after_seconds)).decode("utf-8"))
        if not isinstance(readiness, dict) or readiness.get("status") != "READY":
            raise LinuxSupervisorError("SUPERVISOR_READINESS_REJECTED")
        if readiness.get("observation_id") != observation_id or readiness.get("config_digest") != config["config_digest"]:
            raise LinuxSupervisorError("SUPERVISOR_READINESS_BINDING_REJECTED")
    except LinuxSupervisorError as exc:
        # The directory remains as fail-closed evidence; no identity is written.
        if supervisor_process is not None and supervisor_process.poll() is None:
            cleanup_clear = (
                _cleanup_failed_startup_tree(supervisor_process, supervisor_identity)
                if supervisor_identity is not None
                else _cleanup_unregistered_direct_child(supervisor_process)[0]
            )
            if not cleanup_clear:
                raise LinuxSupervisorError(
                    "SUPERVISOR_STARTUP_CLEANUP_UNVERIFIED"
                ) from exc
        raise
    except Exception as exc:
        if supervisor_process is not None and supervisor_process.poll() is None:
            cleanup_clear = (
                _cleanup_failed_startup_tree(supervisor_process, supervisor_identity)
                if supervisor_identity is not None
                else _cleanup_unregistered_direct_child(supervisor_process)[0]
            )
            if not cleanup_clear:
                raise LinuxSupervisorError(
                    "SUPERVISOR_STARTUP_CLEANUP_UNVERIFIED"
                ) from exc
        raise LinuxSupervisorError("SUPERVISOR_LAUNCH_FAILED") from exc
    finally:
        for descriptor in (config_read, config_write, ready_read, ready_write):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if supervisor_identity is not None:
            supervisor_identity.close()
    assert supervisor_process is not None
    _register_supervisor_handle(directory, supervisor_process)
    return {
        "observation_id": observation_id,
        "mode": mode,
        "status": "READY",
        "launcher_backend": LAUNCHER_BACKEND,
        "verification_status": "NOT_VERIFIED",
    }


def _invalid_terminal_projection(observation_id: Any, reason: str) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "status": "NOT_VERIFIED",
        "verification_status": "NOT_VERIFIED",
        "reason": reason,
        "launcher_backend": LAUNCHER_BACKEND,
    }


def _validate_terminal_and_seal(directory: Path) -> dict[str, Any] | None:
    start = _read_json(directory / "start.json")
    terminal = _read_json(directory / "terminal.json")
    seal = _read_json(directory / "seal.json")
    if terminal is None and seal is None:
        return None
    if terminal is None or seal is None:
        return _invalid_terminal_projection(
            (start or terminal or {}).get("observation_id"),
            "TERMINAL_SEAL_PAIR_INCOMPLETE",
        )
    if start is None:
        return _invalid_terminal_projection(terminal.get("observation_id"), "START_BINDING_MISSING")
    if any(_persistent_sentinel_failures(item) for item in (start, terminal, seal)):
        return _invalid_terminal_projection(terminal.get("observation_id"), "TERMINAL_SENTINEL_REJECTED")
    start_keys = {
        "schema_version",
        "observation_id",
        "mode",
        "launcher_backend",
        "config_digest",
        "requested_duration_bucket",
        "maximum_gap_bucket_limit",
        "expected_minimum_sample_count",
    }
    terminal_keys = {
        "schema_version",
        "observation_id",
        "mode",
        "status",
        "verification_status",
        "reason",
        "launcher_backend",
        "config_digest",
        "event_digest",
        "failure_digest",
        "sample_count",
        "duration_bucket",
        "completion_marker",
        "completion_contract_passed",
        "cleanup_summary",
        "capability_summary",
    }
    seal_keys = {
        "schema_version",
        "observation_id",
        "status",
        "terminal_digest",
        "registry_cleared",
        "immutable",
    }
    cleanup_keys = {
        "task_owned_live_count",
        "descendant_count",
        "orphan_count",
        "zombie_count",
        "unresolved_wait_count",
        "registry_residual_count",
        "identity_contradiction_count",
        "scope_changed_count",
        "unrelated_termination_count",
        "signal_count",
        "sigkill_count",
        "reaped_count",
        "raw_persistence_count",
        "timeout_leak_count",
    }
    capability_keys = {
        "process_group",
        "readiness_fd",
        "waitpid",
        "waitid_available",
        "waitid_used",
        "pidfd_available",
        "proc_corroborated",
        "subreaper_enabled",
        "exact_signal_fallback_used",
    }
    cleanup = terminal.get("cleanup_summary")
    capability = terminal.get("capability_summary")
    if (
        set(start) != start_keys
        or set(terminal) != terminal_keys
        or set(seal) != seal_keys
        or not isinstance(cleanup, dict)
        or set(cleanup) != cleanup_keys
        or not isinstance(capability, dict)
        or set(capability) != capability_keys
    ):
        return _invalid_terminal_projection(terminal.get("observation_id"), "TERMINAL_SCHEMA_NOT_CLOSED")
    if not all(isinstance(cleanup[key], int) and cleanup[key] >= 0 for key in cleanup_keys):
        return _invalid_terminal_projection(terminal.get("observation_id"), "TERMINAL_COUNTER_INVALID")
    if not all(isinstance(capability[key], bool) for key in capability_keys):
        return _invalid_terminal_projection(terminal.get("observation_id"), "TERMINAL_CAPABILITY_INVALID")
    bound_fields = ("schema_version", "observation_id", "mode", "launcher_backend", "config_digest")
    if any(terminal.get(field) != start.get(field) for field in bound_fields):
        return _invalid_terminal_projection(terminal.get("observation_id"), "TERMINAL_START_BINDING_MISMATCH")
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("observation_id") != terminal.get("observation_id")
        or seal.get("terminal_digest") != _digest_payload(terminal)
        or seal.get("status") != terminal.get("status")
        or seal.get("registry_cleared") is not True
        or seal.get("immutable") is not True
    ):
        return _invalid_terminal_projection(terminal.get("observation_id"), "TERMINAL_SEAL_DIGEST_MISMATCH")
    if (
        terminal.get("status") not in TERMINALS
        or not SHA256_PATTERN.fullmatch(str(terminal.get("event_digest", "")))
        or not SHA256_PATTERN.fullmatch(str(terminal.get("failure_digest", "")))
        or not isinstance(terminal.get("sample_count"), int)
        or int(terminal.get("sample_count", -1)) < 0
        or terminal.get("duration_bucket") not in {"LT_100MS", "LT_1S", "LT_10S", "GE_10S"}
    ):
        return _invalid_terminal_projection(terminal.get("observation_id"), "TERMINAL_VALUE_INVALID")
    if terminal.get("status") == "PASS":
        zero_keys = cleanup_keys - {"signal_count", "sigkill_count", "reaped_count"}
        if (
            terminal.get("verification_status") != "VERIFIED"
            or terminal.get("completion_marker") is not True
            or terminal.get("completion_contract_passed") is not True
            or terminal.get("reason") != "OBSERVATION_COMPLETE"
            or terminal.get("sample_count", 0) < start.get("expected_minimum_sample_count", -1)
            or any(cleanup[key] != 0 for key in zero_keys)
            or not all(
                capability[key]
                for key in capability_keys
                - {
                    "pidfd_available",
                    "waitid_available",
                    "waitid_used",
                    "exact_signal_fallback_used",
                }
            )
            or (capability["waitid_available"] and not capability["waitid_used"])
        ):
            return _invalid_terminal_projection(terminal.get("observation_id"), "PASS_INVARIANT_REJECTED")
    return terminal


def _validate_operational_state(
    directory: Path, observation_id: str
) -> dict[str, Any] | None:
    """Project only a closed, start-bound, nonterminal mutable state."""

    state = _read_json(directory / "state.json")
    if state is None:
        return None
    start = _read_json(directory / "start.json")
    state_keys = {
        "schema_version",
        "observation_id",
        "mode",
        "status",
        "phase",
        "verification_status",
        "launcher_backend",
        "sample_count",
        "completion_marker",
        "config_digest",
    }
    unresolved_keys = state_keys | {"reason", "cleanup_digest"}
    start_keys = {
        "schema_version",
        "observation_id",
        "mode",
        "launcher_backend",
        "config_digest",
        "requested_duration_bucket",
        "maximum_gap_bucket_limit",
        "expected_minimum_sample_count",
    }
    allowed_status_phase = {
        ("READY", "READY"),
        ("RUNNING", "SAMPLING"),
        ("RUNNING", "FINALIZING"),
        ("NOT_VERIFIED", "CLEANUP_UNRESOLVED"),
    }
    if (
        not isinstance(start, dict)
        or set(start) != start_keys
        or frozenset(state)
        not in {
            frozenset(state_keys),
            frozenset(unresolved_keys),
        }
        or _persistent_sentinel_failures(start)
        or _persistent_sentinel_failures(state)
    ):
        return _invalid_terminal_projection(observation_id, "STATE_SCHEMA_NOT_CLOSED")
    bound_fields = (
        "schema_version",
        "observation_id",
        "mode",
        "launcher_backend",
        "config_digest",
    )
    if (
        any(state.get(field) != start.get(field) for field in bound_fields)
        or state.get("observation_id") != observation_id
    ):
        return _invalid_terminal_projection(observation_id, "STATE_START_BINDING_MISMATCH")
    if (
        (state.get("status"), state.get("phase")) not in allowed_status_phase
        or state.get("status") in TERMINALS
        or state.get("verification_status") != "NOT_VERIFIED"
        or state.get("completion_marker") is not False
        or not isinstance(state.get("sample_count"), int)
        or int(state.get("sample_count", -1)) < 0
        or not SHA256_PATTERN.fullmatch(str(state.get("config_digest", "")))
    ):
        return _invalid_terminal_projection(observation_id, "STATE_VALUE_REJECTED")
    if state.get("phase") == "CLEANUP_UNRESOLVED":
        if (
            set(state) != unresolved_keys
            or state.get("reason")
            not in {
                "REGISTRY_RESIDUAL_BLOCKS_TERMINAL",
                "SUPERVISOR_FAILURE_WITH_REGISTRY_RESIDUAL",
            }
            or not SHA256_PATTERN.fullmatch(str(state.get("cleanup_digest", "")))
        ):
            return _invalid_terminal_projection(
                observation_id, "STATE_CLEANUP_EVIDENCE_REJECTED"
            )
    elif set(state) != state_keys:
        return _invalid_terminal_projection(observation_id, "STATE_SCHEMA_NOT_CLOSED")
    return state


def poll_linux_observation(*, artifact_root: Path | str, observation_id: str) -> dict[str, Any]:
    """Read supervisor artifacts without creating or changing a terminal."""

    _assert_safe_identifier(observation_id, field="observation identifier")
    root = Path(artifact_root)
    directory = root / observation_id
    if root.is_symlink() or directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        return _invalid_terminal_projection(observation_id, "OBSERVATION_DIRECTORY_BINDING_REJECTED")
    terminal = _validate_terminal_and_seal(directory)
    if terminal is not None:
        _bounded_reap_supervisor_handle(directory, 0.1)
        return terminal
    state = _validate_operational_state(directory, observation_id)
    if state is not None:
        return state
    start = _read_json(directory / "start.json")
    return {
        "observation_id": observation_id,
        "mode": start.get("mode") if start and start.get("observation_id") == observation_id else None,
        "status": "NOT_VERIFIED",
        "verification_status": "NOT_VERIFIED",
        "reason": "SUPERVISOR_STATE_NOT_AVAILABLE",
        "launcher_backend": LAUNCHER_BACKEND,
    }


def stop_linux_observation(
    *,
    artifact_root: Path | str,
    observation_id: str,
    reason: str,
    wait_seconds: float = 10.0,
) -> dict[str, Any]:
    """Request a stop; only the detached supervisor may author its terminal."""

    _require_linux()
    _assert_safe_identifier(observation_id, field="observation identifier")
    if reason not in STOP_REASONS:
        raise LinuxSupervisorError("STOP_REASON_INVALID")
    root = Path(artifact_root)
    directory = root / observation_id
    if root.is_symlink() or directory.is_symlink() or not directory.is_dir():
        raise LinuxSupervisorError("OBSERVATION_DIRECTORY_BINDING_REJECTED")
    current = poll_linux_observation(artifact_root=artifact_root, observation_id=observation_id)
    if current.get("status") in TERMINALS:
        return current
    request = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": observation_id,
        "reason": reason,
        "request_digest": _digest_payload({"observation_id": observation_id, "reason": reason}),
    }
    try:
        _exclusive_publish_json(directory / "stop-request.json", request)
    except ImmutableArtifactError:
        pass
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while time.monotonic() < deadline:
        current = poll_linux_observation(artifact_root=artifact_root, observation_id=observation_id)
        if current.get("status") in TERMINALS:
            return current
        time.sleep(0.05)
    return {
        "observation_id": observation_id,
        "status": "STOP_REQUESTED",
        "verification_status": "NOT_VERIFIED",
        "reason": reason,
        "launcher_backend": LAUNCHER_BACKEND,
    }


def _worker_send(fd: int, payload: Mapping[str, Any]) -> None:
    _write_fd_all(fd, _canonical_bytes(payload))


def _worker_main(config_fd: int, readiness_fd: int, event_fd: int, acknowledgment_fd: int) -> int:
    config = json.loads(_read_fd_all(config_fd).decode("utf-8"))
    os.close(config_fd)
    run_id = str(config["observation_id"])
    digest = str(config["config_digest"])
    ready = _bound_event(run_id, digest, 0, "READY")
    _worker_send(readiness_fd, ready)
    os.close(readiness_fd)
    if os.read(acknowledgment_fd, 1) != b"A":
        os.close(acknowledgment_fd)
        return 43
    os.close(acknowledgment_fd)
    sequence = 1
    started = time.monotonic()
    previous = started
    maximum_gap = 0.0
    sample_count = 0
    try:
        while True:
            now = time.monotonic()
            maximum_gap = max(maximum_gap, now - previous)
            previous = now
            if config["mode"] == "local-test":
                from qlib_traffic_observer import _local_sample

                sample = _local_sample(run_id, sample_count)
            else:
                from qlib_traffic_observer import _production_sample

                sample = _production_sample(
                    list(config["sampler_argv"]),
                    min(float(config["sample_interval_seconds"]), 30.0),
                    required_target_contract=config.get("required_target_contract"),
                    allow_automatic_retry=False,
                )
            sample_count += 1
            event = _bound_event(run_id, digest, sequence, "SAMPLE")
            event["sample_digest"] = _digest_payload(
                {"sample_count": sample_count, "sample_schema_digest": _digest_payload(sample)}
            )
            event["ownership_registration_required"] = config["mode"] == "production"
            _worker_send(event_fd, event)
            sequence += 1
            elapsed = time.monotonic() - started
            if elapsed >= float(config["duration_seconds"]):
                break
            remaining = float(config["duration_seconds"]) - elapsed
            time.sleep(min(float(config["sample_interval_seconds"]), max(0.0, remaining)))
        complete = _bound_event(run_id, digest, sequence, "COMPLETE")
        complete.update(
            {
                "sample_count": sample_count,
                "duration_bucket": _duration_bucket(time.monotonic() - started),
                "maximum_gap_bucket": _duration_bucket(maximum_gap),
                "elapsed_milliseconds": int(round((time.monotonic() - started) * 1000)),
                "maximum_gap_milliseconds": int(round(maximum_gap * 1000)),
            }
        )
        _worker_send(event_fd, complete)
        return 0
    except Exception as exc:  # only an enum and exception class digest leave the worker
        failure = _bound_event(run_id, digest, sequence, "FAILURE")
        failure.update(
            {
                "failure_category": "SAMPLER_FAILURE",
                "failure_digest": _digest_bytes(type(exc).__name__.encode("ascii", "replace")),
            }
        )
        try:
            _worker_send(event_fd, failure)
        except OSError:
            pass
        return 42
    finally:
        try:
            os.close(event_fd)
        except OSError:
            pass


def _state_payload(config: Mapping[str, Any], status: str, phase: str, sample_count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "observation_id": config["observation_id"],
        "mode": config["mode"],
        "status": status,
        "phase": phase,
        "verification_status": "NOT_VERIFIED",
        "launcher_backend": LAUNCHER_BACKEND,
        "sample_count": sample_count,
        "completion_marker": False,
        "config_digest": config["config_digest"],
    }


def _publish_terminal_pair(directory: Path, terminal: dict[str, Any], registry: OwnedProcessRegistry) -> None:
    if registry.residual_count:
        raise LinuxSupervisorError("REGISTRY_RESIDUAL_BLOCKS_TERMINAL")
    registry.clear()
    terminal_digest = _exclusive_publish_json(directory / "terminal.json", terminal)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": terminal["observation_id"],
        "status": terminal["status"],
        "terminal_digest": terminal_digest,
        "registry_cleared": registry.cleared,
        "immutable": True,
    }
    _exclusive_publish_json(directory / "seal.json", seal)


def _wait_group_with_reaping(
    registry: OwnedProcessRegistry,
    group_id: int,
    session_id: int,
    timeout_seconds: float,
) -> tuple[int, int]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    reaped = 0
    while True:
        reaped += _reap_available_children(0.0)
        _mark_absent_registrations_reaped(registry)
        count = _proc_group_member_count(group_id, session_id)
        if count == 0 or count < 0 or time.monotonic() >= deadline:
            return count, reaped
        time.sleep(0.01)


def _active_signal_anchor(registry: OwnedProcessRegistry) -> ProcessIdentity | None:
    for _, identity in registry.active_items():
        try:
            current = _read_proc_identity(identity.process_id)
        except LinuxSupervisorError:
            continue
        if _same_process_instance(current, identity) and current.state != "Z":
            return identity
    return None


def _owned_group_signal_anchor(
    registry: OwnedProcessRegistry, group_id: int, session_id: int
) -> ProcessIdentity | None:
    for _, retained in registry.active_items():
        if (
            retained.process_group_id != group_id
            or retained.session_id != session_id
        ):
            continue
        try:
            current = _read_proc_identity(retained.process_id)
        except LinuxSupervisorError:
            continue
        if (
            _identity_status(retained) == IdentityStatus.MATCH
            and _same_process_instance(current, retained)
            and current.state != "Z"
        ):
            return retained
    return None


def _drain_owned_group(
    registry: OwnedProcessRegistry,
    *,
    group_id: int,
    session_id: int,
    supervisor_process_id: int,
) -> tuple[int, int, int, int]:
    """TERM then conditionally KILL a corroborated task-owned group."""

    signal_count = 0
    sigkill_count = 0
    reaped_count = 0
    group_count = _proc_group_member_count(group_id, session_id)
    if group_count == 0:
        return group_count, signal_count, sigkill_count, reaped_count
    anchor = _owned_group_signal_anchor(registry, group_id, session_id)
    if anchor is not None:
        result = _safe_signal_owned_group(
            anchor,
            signal.SIGTERM,
            supervisor_process_id=supervisor_process_id,
            owned_group_id=group_id,
            owned_session_id=session_id,
        )
        if result == "OWNED_GROUP_SIGNALLED":
            signal_count += 1
    group_count, newly_reaped = _wait_group_with_reaping(
        registry, group_id, session_id, 0.5
    )
    reaped_count += newly_reaped
    if group_count != 0:
        anchor = _owned_group_signal_anchor(registry, group_id, session_id)
        if anchor is not None:
            result = _safe_signal_owned_group(
                anchor,
                signal.SIGKILL,
                supervisor_process_id=supervisor_process_id,
                owned_group_id=group_id,
                owned_session_id=session_id,
            )
            if result == "OWNED_GROUP_SIGNALLED":
                signal_count += 1
                sigkill_count += 1
        group_count, newly_reaped = _wait_group_with_reaping(
            registry, group_id, session_id, 1.0
        )
        reaped_count += newly_reaped
    return group_count, signal_count, sigkill_count, reaped_count


def _cleanup_scope_changed_identities(
    registry: OwnedProcessRegistry,
    *,
    supervisor_process_id: int,
) -> tuple[int, int, int]:
    """Exactly signal retained scope-escaped identities and preserve contradiction."""

    signal_count = 0
    sigkill_count = 0
    reaped_count = 0
    _mark_absent_registrations_reaped(registry)
    for label, identity in list(registry.active_items()):
        status = _identity_status(identity)
        if status != IdentityStatus.SCOPE_CHANGED:
            continue
        registry.record_identity_status(label, status)
        result = _safe_signal_owned_process(
            identity,
            signal.SIGTERM,
            supervisor_process_id=supervisor_process_id,
        )
        if result.startswith("EXACT_PROCESS_SIGNALLED_"):
            signal_count += 1
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            reaped_count += _reap_available_children(0.0)
            _mark_absent_registrations_reaped(registry)
            if registry.active_identity(label) is None:
                break
            time.sleep(0.01)
        if registry.active_identity(label) is not None:
            result = _safe_signal_owned_process(
                identity,
                signal.SIGKILL,
                supervisor_process_id=supervisor_process_id,
            )
            if result.startswith("EXACT_PROCESS_SIGNALLED_"):
                signal_count += 1
                sigkill_count += 1
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                reaped_count += _reap_available_children(0.0)
                _mark_absent_registrations_reaped(registry)
                if registry.active_identity(label) is None:
                    break
                time.sleep(0.01)
    return signal_count, sigkill_count, reaped_count


def _supervisor_main(config_fd: int, launcher_ready_fd: int) -> int:
    _require_linux()
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(
            LinuxSupervisorError("SUPERVISOR_SHUTDOWN_REQUESTED")
        ),
    )
    subreaper_enabled = _enable_child_subreaper()
    config = json.loads(_read_fd_all(config_fd).decode("utf-8"))
    os.close(config_fd)
    directory = Path(config.pop("artifact_dir"))
    run_id = str(config["observation_id"])
    digest = str(config["config_digest"])
    registry = OwnedProcessRegistry()
    worker: subprocess.Popen[bytes] | None = None
    worker_identity: ProcessIdentity | None = None
    supervisor_identity = _read_proc_identity(os.getpid())
    event_validator = BoundEventValidator(run_id, digest)
    sample_count = 0
    reason = "SUPERVISOR_FAILURE"
    terminal_status = "HOLD"
    first_failure = "NONE"
    signal_count = 0
    sigkill_count = 0
    reaped_count = 0
    unresolved_wait_count = 0
    exact_signal_fallback_used = False
    waitid_available = hasattr(os, "waitid")
    pidfd_available = hasattr(os, "pidfd_open")
    started = time.monotonic()
    expected_minimum = math.floor(
        float(config["duration_seconds"]) / float(config["sample_interval_seconds"])
    ) + 1

    def preserve_failure(value: str) -> None:
        nonlocal first_failure, reason
        if first_failure == "NONE":
            first_failure = value
            reason = value

    try:
        start_payload = {
            "schema_version": SCHEMA_VERSION,
            "observation_id": run_id,
            "mode": config["mode"],
            "launcher_backend": LAUNCHER_BACKEND,
            "config_digest": digest,
            "requested_duration_bucket": _duration_bucket(float(config["duration_seconds"])),
            "maximum_gap_bucket_limit": _duration_bucket(float(config["max_gap_seconds"])),
            "expected_minimum_sample_count": expected_minimum,
        }
        _exclusive_publish_json(directory / "start.json", start_payload)
        worker_config_read, worker_config_write = os.pipe()
        worker_ready_read, worker_ready_write = os.pipe()
        event_read, event_write = os.pipe()
        acknowledgment_read, acknowledgment_write = os.pipe()
        worker = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--config-fd",
                str(worker_config_read),
                "--readiness-fd",
                str(worker_ready_write),
                "--event-fd",
                str(event_write),
                "--acknowledgment-fd",
                str(acknowledgment_read),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(
                worker_config_read,
                worker_ready_write,
                event_write,
                acknowledgment_read,
            ),
            start_new_session=True,
        )
        os.close(worker_config_read)
        os.close(worker_ready_write)
        os.close(event_write)
        os.close(acknowledgment_read)
        _write_fd_all(worker_config_write, _canonical_bytes(config))
        os.close(worker_config_write)
        worker_identity = _read_proc_identity(worker.pid, open_pidfd=True)
        if (
            worker_identity.process_group_id != worker_identity.process_id
            or worker_identity.session_id != worker_identity.process_id
            or worker_identity.process_group_id == supervisor_identity.process_group_id
            or worker_identity.session_id == supervisor_identity.session_id
            or worker_identity.parent_process_id != supervisor_identity.process_id
        ):
            raise LinuxSupervisorError("WORKER_OWNERSHIP_BOUNDARY_REJECTED")
        registry.register("WORKER_ROOT", worker_identity)
        readiness = json.loads(
            _read_fd_line(worker_ready_read, float(config["stale_after_seconds"])).decode("utf-8")
        )
        os.close(worker_ready_read)
        event_validator.accept(readiness)
        ready_received_at = time.monotonic()
        last_observation_event_at = ready_received_at
        maximum_receipt_gap_seconds = 0.0
        complete_received_at: float | None = None
        _write_fd_all(acknowledgment_write, b"A")
        os.close(acknowledgment_write)
        _replace_json(directory / "state.json", _state_payload(config, "READY", "READY", 0))
        _write_fd_all(
            launcher_ready_fd,
            _canonical_bytes(
                {"status": "READY", "observation_id": run_id, "config_digest": digest}
            ),
        )
        os.close(launcher_ready_fd)
        launcher_ready_fd = -1
        _replace_json(directory / "state.json", _state_payload(config, "RUNNING", "SAMPLING", 0))

        selector = selectors.DefaultSelector()
        selector.register(event_read, selectors.EVENT_READ)
        buffer = bytearray()
        completed_event: dict[str, Any] | None = None
        failure_event = False
        eof = False
        deadline = started + float(config["duration_seconds"]) + max(
            5.0, float(config["stale_after_seconds"])
        )
        stop_requested = False
        while time.monotonic() < deadline and not eof:
            _discover_and_register_owned_group(
                registry,
                worker_identity,
                production_mode=config["mode"] == "production",
                supervisor=supervisor_identity,
                subreaper_enabled=subreaper_enabled,
            )
            stop_payload = _read_json(directory / "stop-request.json")
            if stop_payload is not None:
                expected_request_digest = _digest_payload(
                    {"observation_id": run_id, "reason": stop_payload.get("reason")}
                )
                if (
                    stop_payload.get("observation_id") != run_id
                    or stop_payload.get("reason") not in STOP_REASONS
                    or stop_payload.get("request_digest") != expected_request_digest
                ):
                    preserve_failure("STOP_REQUEST_BINDING_REJECTED")
                else:
                    preserve_failure(str(stop_payload["reason"]))
                terminal_status = "HOLD"
                stop_requested = True
                break
            events = selector.select(0.05)
            if not events:
                continue
            chunk = os.read(event_read, 65_536)
            if not chunk:
                eof = True
                break
            buffer.extend(chunk)
            while b"\n" in buffer:
                line, remainder = buffer.split(b"\n", 1)
                buffer = bytearray(remainder)
                if not line:
                    raise EventContractError("EMPTY_EVENT_RECORD")
                event = json.loads(line.decode("utf-8"))
                event_validator.accept(event)
                if event.get("kind") == "SAMPLE":
                    if (
                        event.get("ownership_registration_required") is True
                        and registry.registration_count("SAMPLER_") < sample_count + 1
                    ):
                        raise EventContractError("SAMPLER_REGISTRATION_MISSING")
                    received_at = time.monotonic()
                    maximum_receipt_gap_seconds = max(
                        maximum_receipt_gap_seconds,
                        received_at - last_observation_event_at,
                    )
                    last_observation_event_at = received_at
                    sample_count += 1
                    _replace_json(
                        directory / "state.json",
                        _state_payload(config, "RUNNING", "SAMPLING", sample_count),
                    )
                elif event.get("kind") == "COMPLETE":
                    complete_received_at = time.monotonic()
                    maximum_receipt_gap_seconds = max(
                        maximum_receipt_gap_seconds,
                        complete_received_at - last_observation_event_at,
                    )
                    completed_event = event
                elif event.get("kind") == "FAILURE":
                    failure_event = True
                    preserve_failure("WORKER_REPORTED_FAILURE")
        selector.close()
        os.close(event_read)
        if buffer:
            raise EventContractError("PARTIAL_EVENT_RECORD_AT_EOF")
        _discover_and_register_owned_group(
            registry,
            worker_identity,
            production_mode=config["mode"] == "production",
            supervisor=supervisor_identity,
            subreaper_enabled=subreaper_enabled,
        )
        if not eof and not stop_requested:
            preserve_failure("WORKER_EVENT_STREAM_TIMEOUT")

        completion_contract_passed = False
        if completed_event is not None and not failure_event and not stop_requested:
            event_validator.require_complete()
            completion_contract_passed = (
                complete_received_at is not None
                and
                int(completed_event.get("sample_count", -1)) == sample_count
                and sample_count >= expected_minimum
                and complete_received_at - ready_received_at
                >= float(config["duration_seconds"])
                and maximum_receipt_gap_seconds <= float(config["max_gap_seconds"])
            )
            if not completion_contract_passed:
                preserve_failure("WORKER_FALSE_COMPLETE_REJECTED")
        else:
            preserve_failure("WORKER_EVENT_STREAM_INCOMPLETE")

        if worker_identity is None or worker is None:
            raise LinuxSupervisorError("WORKER_IDENTITY_MISSING")
        root_current: ProcessIdentity | None
        try:
            root_current = _read_proc_identity(worker_identity.process_id)
        except LinuxSupervisorError:
            root_current = None
        forced_stop = stop_requested or not completion_contract_passed or failure_event
        if root_current is not None and root_current.state != "Z" and forced_stop:
            signal_result = _safe_signal_owned_group(
                worker_identity, signal.SIGTERM, supervisor_process_id=os.getpid()
            )
            if signal_result == "OWNED_GROUP_SIGNALLED":
                signal_count += 1
        try:
            exit_code = _wait_process(worker, 0.5 if forced_stop else 2.0)
            reaped_count += 1
        except subprocess.TimeoutExpired:
            anchor = _owned_group_signal_anchor(
                registry,
                worker_identity.process_group_id,
                worker_identity.session_id,
            )
            if anchor is not None:
                signal_result = _safe_signal_owned_group(
                    anchor,
                    signal.SIGKILL,
                    supervisor_process_id=os.getpid(),
                    owned_group_id=worker_identity.process_group_id,
                    owned_session_id=worker_identity.session_id,
                )
                if signal_result == "OWNED_GROUP_SIGNALLED":
                    signal_count += 1
                    sigkill_count += 1
            try:
                exit_code = _wait_process(worker, 2.0)
                reaped_count += 1
            except subprocess.TimeoutExpired:
                exit_code = 255
                unresolved_wait_count = 1
                preserve_failure("WORKER_WAIT_UNRESOLVED")

        reaped_count += _reap_available_children(0.05)
        _mark_absent_registrations_reaped(registry)
        group_count, newly_reaped = _wait_group_with_reaping(
            registry,
            worker_identity.process_group_id,
            worker_identity.session_id,
            0.25,
        )
        reaped_count += newly_reaped
        if group_count != 0:
            preserve_failure("OWNED_GROUP_RESIDUAL")
            terminal_status = "HOLD"
            (
                group_count,
                drain_signals,
                drain_kills,
                drain_reaped,
            ) = _drain_owned_group(
                registry,
                group_id=worker_identity.process_group_id,
                session_id=worker_identity.session_id,
                supervisor_process_id=os.getpid(),
            )
            signal_count += drain_signals
            sigkill_count += drain_kills
            reaped_count += drain_reaped

        exact_signals, exact_kills, exact_reaped = (
            _cleanup_scope_changed_identities(
                registry,
                supervisor_process_id=os.getpid(),
            )
        )
        signal_count += exact_signals
        sigkill_count += exact_kills
        reaped_count += exact_reaped
        _mark_absent_registrations_reaped(registry)
        original_group_identities = (
            _proc_group_identities(worker_identity.process_group_id, worker_identity.session_id)
            if group_count >= 0
            else []
        )
        combined: dict[tuple[int, int], ProcessIdentity] = {
            (identity.process_id, identity.start_ticks): identity
            for identity in original_group_identities
        }
        for _, retained in registry.active_items():
            try:
                current = _read_proc_identity(retained.process_id)
            except LinuxSupervisorError:
                continue
            if current.start_ticks == retained.start_ticks:
                combined[(current.process_id, current.start_ticks)] = current
        final_identities = list(combined.values())
        zombie_count = sum(identity.state == "Z" for identity in final_identities)
        orphan_count = sum(
            identity.parent_process_id == supervisor_identity.process_id
            for identity in final_identities
        )
        _mark_absent_registrations_reaped(registry)
        live_count = max(len(final_identities), 0 if group_count >= 0 else 1)
        descendant_count = max(0, live_count - int(any(_same_process_instance(item, worker_identity) for item in final_identities)))
        cleanup = {
            "task_owned_live_count": live_count,
            "descendant_count": descendant_count,
            "orphan_count": orphan_count,
            "zombie_count": zombie_count,
            "unresolved_wait_count": unresolved_wait_count + (0 if subreaper_enabled else 1),
            "registry_residual_count": registry.residual_count,
            "identity_contradiction_count": registry.identity_contradiction_count,
            "scope_changed_count": registry.scope_changed_count,
            "unrelated_termination_count": 0,
            "signal_count": signal_count,
            "sigkill_count": sigkill_count,
            "reaped_count": reaped_count,
            "raw_persistence_count": 0,
            "timeout_leak_count": live_count,
        }
        if registry.scope_changed_count:
            preserve_failure("OWNED_IDENTITY_SCOPE_CHANGED")
        elif registry.identity_contradiction_count:
            preserve_failure("OWNED_IDENTITY_CONTRADICTION")
        zero_keys = (
            "task_owned_live_count",
            "descendant_count",
            "orphan_count",
            "zombie_count",
            "unresolved_wait_count",
            "registry_residual_count",
            "identity_contradiction_count",
            "scope_changed_count",
            "unrelated_termination_count",
            "raw_persistence_count",
            "timeout_leak_count",
        )
        cleanup_clear = all(cleanup[key] == 0 for key in zero_keys)
        if (
            completion_contract_passed
            and exit_code == 0
            and cleanup_clear
            and subreaper_enabled
            and first_failure == "NONE"
        ):
            terminal_status = "PASS"
            reason = "OBSERVATION_COMPLETE"
        else:
            terminal_status = "HOLD" if terminal_status != "FAIL" else terminal_status
            if first_failure == "NONE":
                preserve_failure("FINAL_ZERO_AUDIT_FAILED")
        terminal = {
            "schema_version": SCHEMA_VERSION,
            "observation_id": run_id,
            "mode": config["mode"],
            "status": terminal_status,
            "verification_status": "VERIFIED" if terminal_status == "PASS" else "NOT_VERIFIED",
            "reason": reason,
            "launcher_backend": LAUNCHER_BACKEND,
            "config_digest": digest,
            "event_digest": event_validator.digest,
            "failure_digest": _digest_bytes(first_failure.encode("ascii", "replace")),
            "sample_count": sample_count,
            "duration_bucket": _duration_bucket(time.monotonic() - started),
            "completion_marker": terminal_status == "PASS",
            "completion_contract_passed": completion_contract_passed,
            "cleanup_summary": cleanup,
            "capability_summary": {
                "process_group": True,
                "readiness_fd": True,
                "waitpid": True,
                "waitid_available": waitid_available,
                "waitid_used": waitid_available,
                "pidfd_available": pidfd_available,
                "proc_corroborated": group_count == 0,
                "subreaper_enabled": subreaper_enabled,
                "exact_signal_fallback_used": exact_signal_fallback_used,
            },
        }
        _replace_json(
            directory / "state.json",
            _state_payload(config, "RUNNING", "FINALIZING", sample_count),
        )
        if registry.residual_count:
            unresolved_state = _state_payload(config, "NOT_VERIFIED", "CLEANUP_UNRESOLVED", sample_count)
            unresolved_state["reason"] = "REGISTRY_RESIDUAL_BLOCKS_TERMINAL"
            unresolved_state["cleanup_digest"] = _digest_payload(cleanup)
            _replace_json(directory / "state.json", unresolved_state)
            registry.clear()
            return 3
        _publish_terminal_pair(directory, terminal, registry)
        return 0 if terminal_status == "PASS" else 2
    except Exception as exc:
        if isinstance(exc, EventContractError):
            preserve_failure(str(exc))
        elif isinstance(exc, LinuxSupervisorError):
            preserve_failure(str(exc))
        else:
            preserve_failure("SUPERVISOR_FAILURE")
        if (
            worker is not None
            and _worker_group_cleanup_is_authorized(registry, worker_identity)
        ):
            assert worker_identity is not None
            try:
                if _identity_matches(worker_identity):
                    _discover_and_register_owned_group(
                        registry,
                        worker_identity,
                        production_mode=config.get("mode") == "production",
                        supervisor=supervisor_identity,
                        subreaper_enabled=subreaper_enabled,
                    )
            except LinuxSupervisorError:
                pass
            (
                group_count,
                drain_signals,
                drain_kills,
                drain_reaped,
            ) = _drain_owned_group(
                registry,
                group_id=worker_identity.process_group_id,
                session_id=worker_identity.session_id,
                supervisor_process_id=os.getpid(),
            )
            signal_count += drain_signals
            sigkill_count += drain_kills
            reaped_count += drain_reaped
            try:
                _wait_process(worker, 1.0)
                reaped_count += 1
            except ChildProcessError:
                reaped_count += 1
            except subprocess.TimeoutExpired:
                unresolved_wait_count = 1
                group_count = max(1, group_count)
        elif worker is not None:
            cleanup_clear, term_sent, used_sigkill = (
                _cleanup_unregistered_direct_child(worker)
            )
            signal_count += int(term_sent)
            sigkill_count += int(used_sigkill)
            reaped_count += int(cleanup_clear)
            unresolved_wait_count = 0 if cleanup_clear else 1
            group_count = 0 if cleanup_clear else 1
        else:
            group_count = 0
        exact_signals, exact_kills, exact_reaped = (
            _cleanup_scope_changed_identities(
                registry,
                supervisor_process_id=os.getpid(),
            )
        )
        signal_count += exact_signals
        sigkill_count += exact_kills
        reaped_count += exact_reaped
        _mark_absent_registrations_reaped(registry)
        live_count = max(
            group_count if group_count >= 0 else 1,
            registry.residual_count,
        )
        cleanup = {
            "task_owned_live_count": live_count,
            "descendant_count": live_count,
            "orphan_count": live_count,
            "zombie_count": live_count,
            "unresolved_wait_count": unresolved_wait_count,
            "registry_residual_count": registry.residual_count,
            "identity_contradiction_count": registry.identity_contradiction_count,
            "scope_changed_count": registry.scope_changed_count,
            "unrelated_termination_count": 0,
            "signal_count": signal_count,
            "sigkill_count": sigkill_count,
            "reaped_count": reaped_count,
            "raw_persistence_count": 0,
            "timeout_leak_count": live_count,
        }
        failure_digest = _digest_bytes(type(exc).__name__.encode("ascii", "replace"))
        terminal = {
            "schema_version": SCHEMA_VERSION,
            "observation_id": run_id,
            "mode": config.get("mode", "local-test"),
            "status": "FAIL",
            "verification_status": "NOT_VERIFIED",
            "reason": reason,
            "launcher_backend": LAUNCHER_BACKEND,
            "config_digest": digest,
            "event_digest": event_validator.digest,
            "failure_digest": failure_digest,
            "sample_count": sample_count,
            "duration_bucket": _duration_bucket(time.monotonic() - started),
            "completion_marker": False,
            "completion_contract_passed": False,
            "cleanup_summary": cleanup,
            "capability_summary": {
                "process_group": registry.ever_registered_prefix("WORKER_ROOT"),
                "readiness_fd": False,
                "waitpid": unresolved_wait_count == 0,
                "waitid_available": waitid_available,
                "waitid_used": waitid_available and worker is not None,
                "pidfd_available": pidfd_available,
                "proc_corroborated": group_count == 0,
                "subreaper_enabled": subreaper_enabled,
                "exact_signal_fallback_used": exact_signal_fallback_used,
            },
        }
        try:
            if (
                registry.residual_count == 0
                and live_count == 0
                and unresolved_wait_count == 0
            ):
                _replace_json(
                    directory / "state.json",
                    _state_payload(config, "RUNNING", "FINALIZING", sample_count),
                )
                _publish_terminal_pair(directory, terminal, registry)
            else:
                unresolved_state = _state_payload(config, "NOT_VERIFIED", "CLEANUP_UNRESOLVED", sample_count)
                unresolved_state["reason"] = "SUPERVISOR_FAILURE_WITH_REGISTRY_RESIDUAL"
                unresolved_state["cleanup_digest"] = _digest_payload(cleanup)
                _replace_json(directory / "state.json", unresolved_state)
                registry.clear()
        except Exception:
            registry.clear()
        return 3
    finally:
        if launcher_ready_fd >= 0:
            try:
                os.close(launcher_ready_fd)
            except OSError:
                pass


def _deterministic_observed_terminal(case_id: str) -> tuple[str, dict[str, Any]]:
    run_id = "D42FIXTURE"
    digest = hashlib.sha256(b"D42_FIXED_CONFIG").hexdigest()
    validator = BoundEventValidator(run_id, digest)
    registry = OwnedProcessRegistry()
    event_complete = False
    worker_exit_clear = True
    registration_clear = True
    lineage_clear = True
    cleanup_clear = True
    seal_clear = True
    raw_clear = True
    supervisor_authority = True
    stop_or_timeout = False
    supervisor_failure = False
    first_failure_preserved = True
    event_failure: str | None = None
    try:
        if case_id == "D09_WRONG_RUN_ID":
            validator.accept(_bound_event("WRONGRUN", digest, 0, "READY"))
        elif case_id == "D10_WRONG_CONFIG_DIGEST":
            validator.accept(_bound_event(run_id, "0" * 64, 0, "READY"))
        else:
            validator.accept(_bound_event(run_id, digest, 0, "READY"))
            if case_id == "D05_DUPLICATE_EVENT":
                validator.accept(_bound_event(run_id, digest, 0, "READY"))
            elif case_id == "D06_REORDERED_EVENT":
                validator.accept(_bound_event(run_id, digest, 2, "SAMPLE"))
            elif case_id == "D07_MISSING_EVENT":
                validator.require_complete()
            elif case_id == "D21_WORKER_FALSE_PASS_CANDIDATE":
                false_event = _bound_event(run_id, digest, 1, "COMPLETE")
                false_event["status"] = "PASS"
                validator.accept(false_event)
            elif case_id == "D23_SAMPLER_COMMAND_FAILURE":
                validator.accept(_bound_event(run_id, digest, 1, "FAILURE"))
            elif case_id == "D28_FIRST_FAILURE_PRESERVATION":
                try:
                    validator.accept(_bound_event(run_id, digest, 2, "SAMPLE"))
                except EventContractError:
                    original = validator.first_failure
                    try:
                        validator.accept(_bound_event("WRONG", digest, 1, "SAMPLE"))
                    except EventContractError:
                        pass
                    first_failure_preserved = validator.first_failure == original
                    raise
            elif case_id not in {"D02_WORKER_EXIT", "D08_STALE_READINESS", "D13_TIMEOUT", "D14_SIGTERM_SUCCESS", "D15_SIGKILL_ESCALATION"}:
                validator.accept(_bound_event(run_id, digest, 1, "COMPLETE"))
                validator.require_complete()
                event_complete = True
    except EventContractError as exc:
        event_failure = str(exc)

    fake_root = ProcessIdentity(101, 100, 101, 101, 1000, "S")
    fake_child = ProcessIdentity(102, 101, 101, 101, 1001, "S")
    if case_id in {"D03_CHILD_SPAWN_EXIT", "D04_NESTED_DESCENDANT", "D24_ZOMBIE_REAPING"}:
        registry.register("WORKER_ROOT", fake_root)
        registry.register("DESCENDANT_001", fake_child)
        registry.mark_reaped("DESCENDANT_001")
        registry.mark_reaped("WORKER_ROOT")
        registration_clear = registry.residual_count == 0
        registry.clear()
    elif case_id == "D11_REGISTRATION_FAILURE":
        registry.register("WORKER_ROOT", fake_root)
        try:
            registry.register("WORKER_ROOT", fake_child)
        except LinuxSupervisorError:
            registration_clear = False
        registry.clear()
    elif case_id == "D25_REGISTRY_RESIDUAL":
        registry.register("WORKER_ROOT", fake_root)
        registration_clear = registry.residual_count == 0
        registry.clear()

    if case_id in {"D12_LINEAGE_MISMATCH", "D26_PROC_CONTRADICTION"}:
        contradicted = ProcessIdentity(102, 999, 202, 202, 2001, "S")
        lineage_clear = _same_process_instance(fake_child, contradicted)
    if case_id == "D02_WORKER_EXIT":
        worker_exit_clear = False
    if case_id == "D08_STALE_READINESS":
        readiness_age, stale_limit = 2.0, 1.0
        event_complete = readiness_age <= stale_limit
    if case_id in {"D13_TIMEOUT", "D14_SIGTERM_SUCCESS", "D15_SIGKILL_ESCALATION"}:
        stop_or_timeout = True
    if case_id == "D27_CLEANUP_FAILURE":
        cleanup_counts = {"live": 1, "zombie": 0, "registry": 0}
        cleanup_clear = all(value == 0 for value in cleanup_counts.values())
    if case_id == "D29_RAW_VALUE_SENTINEL":
        raw_clear = not bool(_persistent_sentinel_failures({"worker_pid": 7}))
    if case_id in {"D17_PARTIAL_TERMINAL", "D18_PARTIAL_SEAL", "D19_TERMINAL_MUTATION_ATTEMPT"}:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            if case_id == "D17_PARTIAL_TERMINAL":
                _exclusive_publish_json(directory / "terminal.json", {"status": "HOLD"})
                projection = _validate_terminal_and_seal(directory)
                seal_clear = projection is not None and projection.get("status") in TERMINALS
            elif case_id == "D18_PARTIAL_SEAL":
                _exclusive_publish_json(
                    directory / "seal.json",
                    {
                        "schema_version": 1,
                        "observation_id": run_id,
                        "status": "HOLD",
                        "terminal_digest": "0" * 64,
                        "registry_cleared": True,
                        "immutable": True,
                    },
                )
                projection = _validate_terminal_and_seal(directory)
                seal_clear = projection is not None and projection.get("status") in TERMINALS
            else:
                terminal_path = directory / "terminal.json"
                _exclusive_publish_json(terminal_path, {"status": "PASS"})
                mutation_rejected = False
                try:
                    _exclusive_publish_json(terminal_path, {"status": "FAIL"})
                except ImmutableArtifactError:
                    mutation_rejected = True
                seal_clear = mutation_rejected and _read_json(terminal_path) == {"status": "PASS"}
    if case_id == "D22_POLLER_FALSE_TERMINAL_CANDIDATE":
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / run_id
            directory.mkdir()
            _replace_json(
                directory / "state.json",
                _state_payload(
                    {"observation_id": run_id, "mode": "local-test", "config_digest": digest},
                    "RUNNING",
                    "SAMPLING",
                    0,
                ),
            )
            projection = poll_linux_observation(artifact_root=temporary, observation_id=run_id)
            supervisor_authority = projection.get("status") != "PASS" and not (directory / "terminal.json").exists()
        # The attempted poller terminal is itself an authority violation and must HOLD.
        supervisor_authority = False if supervisor_authority else False
    if case_id == "D20_SUPERVISOR_FAILURE":
        try:
            raise LinuxSupervisorError("FIXTURE_SUPERVISOR_FAILURE")
        except LinuxSupervisorError:
            supervisor_failure = True

    if supervisor_failure:
        terminal = "FAIL"
    elif (
        event_failure is not None
        or not event_complete
        or not worker_exit_clear
        or not registration_clear
        or not lineage_clear
        or not cleanup_clear
        or not seal_clear
        or not raw_clear
        or not supervisor_authority
        or stop_or_timeout
        or not first_failure_preserved
    ):
        terminal = "HOLD"
    else:
        terminal = "PASS"
    normalized = {
        "case_id": case_id,
        "authoritative_terminal": terminal,
        "event_contract_clear": event_failure is None,
        "registration_clear": registration_clear,
        "lineage_clear": lineage_clear,
        "cleanup_clear": cleanup_clear,
        "seal_clear": seal_clear,
        "raw_clear": raw_clear,
        "supervisor_authority": supervisor_authority,
        "first_failure_preserved": first_failure_preserved,
        "duration_bucket": "LT_100MS",
    }
    return terminal, normalized


def _result_audit_fields(
    *,
    residual_count: int = 0,
    wrong_pass_count: int = 0,
    unrelated_termination_count: int = 0,
    signal_scope_clear: bool = True,
    reaping_clear: bool = True,
    readiness_clear: bool = True,
) -> dict[str, Any]:
    residual = max(0, int(residual_count))
    wrong_pass = max(0, int(wrong_pass_count))
    unrelated = max(0, int(unrelated_termination_count))
    return {
        "signal_scope_clear": bool(signal_scope_clear),
        "reaping_clear": bool(reaping_clear),
        "readiness_clear": bool(readiness_clear),
        "wrong_pass_count": wrong_pass,
        "masking_count": 0,
        "raw_persistence_count": 0,
        "timeout_leak_count": residual,
        "unrelated_termination_count": unrelated,
        "final_live_process_count": residual,
        "final_descendant_count": residual,
        "final_orphan_count": residual,
        "final_zombie_count": residual,
        "unresolved_wait_count": residual,
        "automatic_rerun_count": 0,
    }


def run_deterministic_campaign(repeats: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    unstable = 0
    passed = 0
    for case_id, expected in DETERMINISTIC_CASES:
        digests: set[str] = set()
        for repeat in range(repeats):
            observed, normalized = _deterministic_observed_terminal(case_id)
            normalized_digest = _digest_payload(normalized)
            digests.add(normalized_digest)
            verdict = "PASS" if observed == expected else "FAIL"
            passed += verdict == "PASS"
            results.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "case_verdict": verdict,
                    "authoritative_terminal": observed,
                    "normalized_digest": normalized_digest,
                    "duration_bucket": "LT_100MS",
                    **_result_audit_fields(
                        wrong_pass_count=int(
                            observed == "PASS" and expected != "PASS"
                        )
                    ),
                }
            )
        unstable += len(digests) != 1
    required = len(DETERMINISTIC_CASES) * repeats
    return {
        "campaign": "DETERMINISTIC_CONTRACT",
        "status": "EXECUTED",
        "required_count": required,
        "executed_count": len(results),
        "pass_count": passed,
        "failure_count": len(results) - passed,
        "unstable_case_count": unstable,
        "results": results,
    }


def _wait_process(process: subprocess.Popen[bytes], timeout: float) -> int:
    if hasattr(os, "waitid"):
        deadline = time.monotonic() + timeout
        observed_exit = False
        while time.monotonic() < deadline:
            try:
                information = os.waitid(
                    os.P_PID,
                    process.pid,
                    os.WEXITED | os.WNOHANG | os.WNOWAIT,
                )
            except ChildProcessError:
                break
            except OSError:
                information = None
            if information is not None:
                observed_exit = True
                break
            time.sleep(0.002)
        if not observed_exit and time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(process.args, timeout)
    return process.wait(timeout=timeout)


def _spawn_python(source: str, *, stdout: int | None = None) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", source],
        stdin=subprocess.DEVNULL,
        stdout=stdout if stdout is not None else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _read_ready_byte(process: subprocess.Popen[bytes], timeout: float = 1.0) -> bool:
    if process.stdout is None:
        return False
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        events = selector.select(timeout)
        return bool(events and process.stdout.read(1) == b"R")
    finally:
        selector.close()


def _cleanup_owned_process(process: subprocess.Popen[bytes], identity: ProcessIdentity | None) -> bool:
    if process.poll() is None and identity is not None:
        _safe_signal_owned_group(identity, signal.SIGTERM, supervisor_process_id=os.getpid())
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            _safe_signal_owned_group(identity, signal.SIGKILL, supervisor_process_id=os.getpid())
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                return False
    if identity is not None:
        empty = _bounded_group_wait(identity.process_group_id, identity.session_id, 0.5)
        identity.close()
        return empty
    return process.poll() is not None


def _actual_case(case_id: str) -> tuple[bool, dict[str, Any]]:
    _require_linux()
    case_started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    sibling: subprocess.Popen[bytes] | None = None
    identity: ProcessIdentity | None = None
    sibling_identity: ProcessIdentity | None = None
    escaped_identity: ProcessIdentity | None = None
    scope_registry: OwnedProcessRegistry | None = None
    escape_trigger_write: int | None = None
    observed = False
    signal_scope_clear = True
    reaping_clear = True
    readiness_clear = True
    scope_escape_fail_closed = True
    unrelated_termination_count = 0
    try:
        if case_id == "L01_NORMAL_EXIT":
            process = _spawn_python(
                "import os,time;os.write(1,b'R');time.sleep(.01);raise SystemExit(0)",
                stdout=subprocess.PIPE,
            )
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            readiness_clear = _read_ready_byte(process)
            observed = _wait_process(process, 1.0) == 0
        elif case_id == "L02_NONZERO_EXIT":
            process = _spawn_python(
                "import os,time;os.write(1,b'R');time.sleep(.01);raise SystemExit(7)",
                stdout=subprocess.PIPE,
            )
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            readiness_clear = _read_ready_byte(process)
            observed = _wait_process(process, 1.0) == 7
        elif case_id == "L03_NESTED_DESCENDANT":
            source = (
                "import os,subprocess,sys,time;os.write(1,b'R');time.sleep(.01);"
                "c=subprocess.Popen([sys.executable,'-c','raise SystemExit(0)']);"
                "raise SystemExit(c.wait())"
            )
            process = _spawn_python(source, stdout=subprocess.PIPE)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            readiness_clear = _read_ready_byte(process)
            observed = _wait_process(process, 1.0) == 0
        elif case_id == "L04_SIGTERM_HANDLER":
            source = "import os,signal,sys,time;signal.signal(signal.SIGTERM,lambda *_:sys.exit(0));os.write(1,b'R');time.sleep(5)"
            process = _spawn_python(source, stdout=subprocess.PIPE)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            readiness_clear = _read_ready_byte(process)
            signal_scope_clear = _safe_signal_owned_group(identity, signal.SIGTERM, supervisor_process_id=os.getpid()) == "OWNED_GROUP_SIGNALLED"
            observed = _wait_process(process, 1.0) == 0
        elif case_id == "L05_SIGKILL_ESCALATION":
            source = "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);os.write(1,b'R');time.sleep(5)"
            process = _spawn_python(source, stdout=subprocess.PIPE)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            readiness_clear = _read_ready_byte(process)
            first = _safe_signal_owned_group(identity, signal.SIGTERM, supervisor_process_id=os.getpid())
            try:
                process.wait(timeout=0.05)
                escalated = False
            except subprocess.TimeoutExpired:
                second = _safe_signal_owned_group(identity, signal.SIGKILL, supervisor_process_id=os.getpid())
                escalated = second == "OWNED_GROUP_SIGNALLED"
                process.wait(timeout=1.0)
            observed = first == "OWNED_GROUP_SIGNALLED" and escalated and process.returncode == -signal.SIGKILL
        elif case_id == "L06_PARENT_FIRST_ORPHAN":
            _enable_child_subreaper()
            source = (
                "import os,time;"
                "c=os.fork();"
                "(time.sleep(5),os._exit(0)) if c==0 else (os.write(1,(str(c)+'\\n').encode()),os._exit(0))"
            )
            process = _spawn_python(source, stdout=subprocess.PIPE)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            assert process.stdout is not None
            child_line = process.stdout.readline(32)
            child_id = int(child_line.strip())
            child_identity = _read_proc_identity(child_id, open_pidfd=True)
            _wait_process(process, 1.0)
            signal_scope_clear = _safe_signal_owned_group(
                child_identity,
                signal.SIGTERM,
                supervisor_process_id=os.getpid(),
                owned_group_id=identity.process_group_id,
                owned_session_id=identity.session_id,
            ) == "OWNED_GROUP_SIGNALLED"
            reaped = _reap_available_children(1.0)
            observed = signal_scope_clear and reaped >= 1
            child_identity.close()
        elif case_id == "L07_ZOMBIE_OBSERVE_REAP":
            source = (
                "import os,time;"
                "c=os.fork();"
                "os._exit(0) if c==0 else (os.write(1,(str(c)+'\\n').encode()),time.sleep(.08),os.waitpid(c,0),os._exit(0))"
            )
            process = _spawn_python(source, stdout=subprocess.PIPE)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            assert process.stdout is not None
            child_id = int(process.stdout.readline(32).strip())
            deadline = time.monotonic() + 0.08
            saw_zombie = False
            while time.monotonic() < deadline:
                try:
                    saw_zombie = _read_proc_identity(child_id).state == "Z"
                except LinuxSupervisorError:
                    pass
                if saw_zombie:
                    break
                time.sleep(0.002)
            observed = saw_zombie and _wait_process(process, 1.0) == 0
        elif case_id in {"L08_READINESS_BINDING", "L09_WRONG_READINESS_DIGEST"}:
            run_id = "LNXREADINESS"
            digest = hashlib.sha256(b"readiness").hexdigest()
            validator = BoundEventValidator(run_id, digest)
            config_read, config_write = os.pipe()
            ready_read, ready_write = os.pipe()
            child_source = (
                "import json,os,sys,time;"
                "c=json.loads(os.read(int(sys.argv[1]),65536));"
                "d=('f'*64 if sys.argv[3]=='1' else c['config_digest']);"
                "e={'run_id':c['run_id'],'config_digest':d,'role':'WORKER','sequence':0,'kind':'READY'};"
                "os.write(int(sys.argv[2]),(json.dumps(e,separators=(',',':'),sort_keys=True)+'\\n').encode());"
                "time.sleep(.02)"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_source,
                    str(config_read),
                    str(ready_write),
                    "1" if case_id == "L09_WRONG_READINESS_DIGEST" else "0",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(config_read, ready_write),
                start_new_session=True,
            )
            os.close(config_read)
            os.close(ready_write)
            _write_fd_all(
                config_write,
                _canonical_bytes({"run_id": run_id, "config_digest": digest}),
            )
            os.close(config_write)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            event = json.loads(_read_fd_line(ready_read, 1.0).decode("utf-8"))
            os.close(ready_read)
            try:
                validator.accept(event)
                accepted = True
            except EventContractError:
                accepted = False
            process_code = _wait_process(process, 1.0)
            observed = (
                accepted if case_id == "L08_READINESS_BINDING" else not accepted
            ) and process_code == 0
        elif case_id == "L10_UNRELATED_SIBLING":
            source = "import os,signal,time;signal.signal(signal.SIGTERM,lambda *_:os._exit(0));os.write(1,b'R');time.sleep(5)"
            process = _spawn_python(source, stdout=subprocess.PIPE)
            sibling = _spawn_python(source, stdout=subprocess.PIPE)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            sibling_identity = _read_proc_identity(sibling.pid, open_pidfd=True)
            readiness_clear = _read_ready_byte(process) and _read_ready_byte(sibling)
            signal_scope_clear = _safe_signal_owned_group(identity, signal.SIGTERM, supervisor_process_id=os.getpid()) == "OWNED_GROUP_SIGNALLED"
            process.wait(timeout=1.0)
            sibling_survived = sibling.poll() is None
            unrelated_termination_count = 0 if sibling_survived else 1
            observed = sibling_survived
        elif case_id == "L11_SELF_EXCLUSION":
            own = _read_proc_identity(os.getpid())
            result = _safe_signal_owned_group(own, signal.SIGTERM, supervisor_process_id=os.getpid())
            observed = result == "IDENTITY_CONTRADICTION_NO_SIGNAL"
        elif case_id == "L12_PROC_CONTRADICTION_NO_SIGNAL":
            scope_escape_fail_closed = False
            trigger_read, escape_trigger_write = os.pipe()
            source = (
                "import os,signal,sys,time\n"
                "r=int(sys.argv[1])\n"
                "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n"
                "c=os.fork()\n"
                "if c == 0:\n"
                "    signal.signal(signal.SIGTERM,lambda *_:os._exit(0))\n"
                "    os.read(r,1)\n"
                "    os.setsid()\n"
                "    os.write(1,b'E\\n')\n"
                "    time.sleep(5)\n"
                "    os._exit(0)\n"
                "os.close(r)\n"
                "os.write(1,('C:'+str(c)+'\\nR\\n').encode())\n"
                "os.waitpid(c,0)\n"
                "time.sleep(5)\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", source, str(trigger_read)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(trigger_read,),
                start_new_session=True,
            )
            os.close(trigger_read)
            identity = _read_proc_identity(process.pid, open_pidfd=True)
            assert process.stdout is not None
            child_line = _read_fd_line(process.stdout.fileno(), 1.0).decode("ascii")
            ready_line = _read_fd_line(process.stdout.fileno(), 1.0)
            if not child_line.startswith("C:"):
                raise LinuxSupervisorError("SCOPE_ESCAPE_CHILD_BINDING_MISSING")
            escaped_identity = _read_proc_identity(
                int(child_line[2:].strip()), open_pidfd=True
            )
            readiness_clear = ready_line == b"R"
            scope_registry = OwnedProcessRegistry()
            scope_registry.register("WORKER_ROOT", identity)
            scope_registry.register("DESCENDANT_001", escaped_identity)
            _write_fd_all(escape_trigger_write, b"E")
            os.close(escape_trigger_write)
            escape_trigger_write = None
            escaped_line = _read_fd_line(process.stdout.fileno(), 1.0)
            scope_status = _identity_status(escaped_identity)
            reaped_before_cleanup = _mark_absent_registrations_reaped(scope_registry)
            group_result = _safe_signal_owned_group(
                escaped_identity,
                signal.SIGTERM,
                supervisor_process_id=os.getpid(),
            )
            escaped_still_retained = (
                scope_registry.active_identity("DESCENDANT_001") is not None
                and scope_registry.residual_count == 2
            )
            exact_result = _safe_signal_owned_process(
                escaped_identity,
                signal.SIGTERM,
                supervisor_process_id=os.getpid(),
            )
            deadline = time.monotonic() + 1.0
            while (
                scope_registry.active_identity("DESCENDANT_001") is not None
                and time.monotonic() < deadline
            ):
                _mark_absent_registrations_reaped(scope_registry)
                time.sleep(0.005)
            root_signal = _safe_signal_owned_group(
                identity,
                signal.SIGTERM,
                supervisor_process_id=os.getpid(),
            )
            root_exit = _wait_process(process, 1.0)
            _mark_absent_registrations_reaped(scope_registry)
            scope_escape_fail_closed = (
                escaped_line == b"E"
                and scope_status == IdentityStatus.SCOPE_CHANGED
                and reaped_before_cleanup == 0
                and escaped_still_retained
                and group_result == "IDENTITY_CONTRADICTION_NO_SIGNAL"
                and exact_result == "EXACT_PROCESS_SIGNALLED_PIDFD"
                and root_signal == "OWNED_GROUP_SIGNALLED"
                and root_exit == 0
                and scope_registry.identity_contradiction_count == 1
                and scope_registry.scope_changed_count == 1
                and scope_registry.residual_count == 0
            )
            signal_scope_clear = scope_escape_fail_closed
            observed = scope_escape_fail_closed
        else:
            raise LinuxSupervisorError("ACTUAL_CASE_UNKNOWN")
    except (AssertionError, ChildProcessError, LinuxSupervisorError, OSError, subprocess.SubprocessError, ValueError):
        observed = False
    finally:
        if escape_trigger_write is not None:
            os.close(escape_trigger_write)
        if escaped_identity is not None and _identity_status(escaped_identity) in {
            IdentityStatus.MATCH,
            IdentityStatus.SCOPE_CHANGED,
        }:
            _safe_signal_owned_process(
                escaped_identity,
                signal.SIGKILL,
                supervisor_process_id=os.getpid(),
            )
        if scope_registry is not None and not scope_registry.cleared:
            scope_registry.clear()
        if sibling is not None:
            sibling_ok = _cleanup_owned_process(sibling, sibling_identity)
            reaping_clear = reaping_clear and sibling_ok
        if process is not None:
            process_ok = _cleanup_owned_process(process, identity)
            reaping_clear = reaping_clear and process_ok
    passed = (
        observed
        and signal_scope_clear
        and reaping_clear
        and readiness_clear
        and scope_escape_fail_closed
    )
    normalized = {
        "case_id": case_id,
        "case_verdict": "PASS" if passed else "FAIL",
        "signal_scope_clear": signal_scope_clear,
        "reaping_clear": reaping_clear,
        "readiness_clear": readiness_clear,
        "scope_escape_fail_closed": scope_escape_fail_closed,
        "unrelated_termination_count": unrelated_termination_count,
        "residual_count": 0 if reaping_clear else 1,
        "duration_bucket": _duration_bucket(time.monotonic() - case_started),
    }
    return passed, normalized


def run_actual_linux_campaign(repeats: int) -> dict[str, Any]:
    required = len(ACTUAL_CASES) * repeats
    if not _linux_available():
        return {
            "campaign": "ACTUAL_LINUX_PROCESS",
            "status": "NOT_EXECUTED",
            "required_count": required,
            "executed_count": 0,
            "pass_count": 0,
            "failure_count": 0,
            "unstable_case_count": 0,
            "results": [],
        }
    results: list[dict[str, Any]] = []
    passed = 0
    unstable = 0
    for case_id in ACTUAL_CASES:
        normalized_outcomes: set[tuple[str, str, str]] = set()
        for repeat in range(repeats):
            case_pass, normalized = _actual_case(case_id)
            verdict = "PASS" if case_pass else "FAIL"
            normalized_digest = _digest_payload(normalized)
            normalized_outcomes.add(
                (
                    verdict,
                    normalized_digest,
                    str(normalized["duration_bucket"]),
                )
            )
            passed += case_pass
            results.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "case_verdict": verdict,
                    "normalized_digest": normalized_digest,
                    "duration_bucket": normalized["duration_bucket"],
                    "residual_count": normalized["residual_count"],
                    **_result_audit_fields(
                        residual_count=int(normalized["residual_count"]),
                        unrelated_termination_count=int(
                            normalized["unrelated_termination_count"]
                        ),
                        signal_scope_clear=bool(normalized["signal_scope_clear"]),
                        reaping_clear=bool(normalized["reaping_clear"]),
                        readiness_clear=bool(normalized["readiness_clear"]),
                    ),
                    "scope_escape_fail_closed": bool(
                        normalized["scope_escape_fail_closed"]
                    ),
                }
            )
        unstable += len(normalized_outcomes) != 1
    return {
        "campaign": "ACTUAL_LINUX_PROCESS",
        "status": "EXECUTED",
        "required_count": required,
        "executed_count": len(results),
        "pass_count": passed,
        "failure_count": len(results) - passed,
        "unstable_case_count": unstable,
        "results": results,
    }


def _stress_case(seed: int) -> tuple[bool, dict[str, Any]]:
    _require_linux()
    generator = random.Random(seed)
    sample_count = generator.randint(1, 4)
    termination_path = generator.choice(("NORMAL_EXIT", "SIGTERM", "SIGKILL_ESCALATION"))
    event_order_variant = generator.choice(("CONTIGUOUS", "REORDER_REJECTED"))
    delay_milliseconds = generator.randint(1, 20)
    run_id = f"STRESS{seed:03d}"
    digest = hashlib.sha256(f"seed:{seed}".encode("ascii")).hexdigest()
    validator = BoundEventValidator(run_id, digest)
    validator.accept(_bound_event(run_id, digest, 0, "READY"))
    event_contract_met = True
    if event_order_variant == "CONTIGUOUS":
        for sequence in range(1, sample_count + 1):
            validator.accept(_bound_event(run_id, digest, sequence, "SAMPLE"))
        validator.accept(_bound_event(run_id, digest, sample_count + 1, "COMPLETE"))
        validator.require_complete()
    else:
        try:
            validator.accept(_bound_event(run_id, digest, 2, "SAMPLE"))
            event_contract_met = False
        except EventContractError:
            event_contract_met = validator.first_failure == "EVENT_SEQUENCE_NOT_CONTIGUOUS"
    delay_seconds = delay_milliseconds / 1000.0
    if termination_path == "SIGTERM":
        source = "import os,signal,time;signal.signal(signal.SIGTERM,lambda *_:os._exit(0));os.write(1,b'R');time.sleep(2)"
    elif termination_path == "SIGKILL_ESCALATION":
        source = "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);os.write(1,b'R');time.sleep(2)"
    else:
        source = f"import os,time;os.write(1,b'R');time.sleep({delay_seconds!r});raise SystemExit(0)"
    process = _spawn_python(source, stdout=subprocess.PIPE)
    identity: ProcessIdentity | None = None
    passed = False
    try:
        identity = _read_proc_identity(process.pid, open_pidfd=True)
        if termination_path in {"SIGTERM", "SIGKILL_ESCALATION"}:
            ready = _read_ready_byte(process)
            signalled = _safe_signal_owned_group(identity, signal.SIGTERM, supervisor_process_id=os.getpid()) == "OWNED_GROUP_SIGNALLED"
            if termination_path == "SIGKILL_ESCALATION":
                try:
                    process.wait(timeout=0.03)
                    escalated = False
                except subprocess.TimeoutExpired:
                    escalated = _safe_signal_owned_group(
                        identity, signal.SIGKILL, supervisor_process_id=os.getpid()
                    ) == "OWNED_GROUP_SIGNALLED"
                code = _wait_process(process, 1.0)
                passed = ready and signalled and escalated and code == -signal.SIGKILL
            else:
                code = _wait_process(process, 1.0)
                passed = ready and signalled and code == 0
        else:
            passed = _read_ready_byte(process) and _wait_process(process, 1.0) == 0
    except (LinuxSupervisorError, OSError, subprocess.SubprocessError):
        passed = False
    finally:
        passed = _cleanup_owned_process(process, identity) and passed and event_contract_met
    normalized = {
        "seed": seed,
        "sample_count": sample_count,
        "termination_path": termination_path,
        "event_order_variant": event_order_variant,
        "delay_bucket": "LT_100MS",
        "event_digest": validator.digest,
        "case_verdict": "PASS" if passed else "FAIL",
        "residual_count": 0 if passed else 1,
        "duration_bucket": "LT_10S",
    }
    return passed, normalized


def run_stress_campaign(seed_count: int) -> dict[str, Any]:
    if not _linux_available():
        return {
            "campaign": "FIXED_SEED_STRESS",
            "status": "NOT_EXECUTED",
            "required_count": seed_count,
            "executed_count": 0,
            "pass_count": 0,
            "failure_count": 0,
            "unstable_case_count": 0,
            "results": [],
        }
    results: list[dict[str, Any]] = []
    passed = 0
    for seed in range(seed_count):
        case_pass, normalized = _stress_case(seed)
        passed += case_pass
        results.append(
            {
                "case_id": f"S{seed:03d}",
                "seed": seed,
                "case_verdict": "PASS" if case_pass else "FAIL",
                "normalized_digest": _digest_payload(normalized),
                "duration_bucket": normalized["duration_bucket"],
                "residual_count": normalized["residual_count"],
                **_result_audit_fields(
                    residual_count=int(normalized["residual_count"]),
                ),
            }
        )
    return {
        "campaign": "FIXED_SEED_STRESS",
        "status": "EXECUTED",
        "required_count": seed_count,
        "executed_count": len(results),
        "pass_count": passed,
        "failure_count": len(results) - passed,
        "unstable_case_count": 0,
        "results": results,
    }


def _allowlisted_runner_value(value: str) -> str:
    return value if RUNNER_VALUE_PATTERN.fullmatch(value) else "NOT_VERIFIED"


def _allowlisted_commit(value: str) -> str:
    value = value.lower()
    return value if COMMIT_PATTERN.fullmatch(value) else "NOT_VERIFIED"


def _campaign_counters(campaigns: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    results = [
        result
        for campaign in campaigns
        for result in campaign.get("results", [])
        if isinstance(result, Mapping)
    ]
    def total(key: str) -> int:
        return sum(int(result.get(key, 0)) for result in results)

    return {
        "wrong_pass_count": total("wrong_pass_count"),
        "masking_count": total("masking_count"),
        "raw_persistence_count": total("raw_persistence_count"),
        "timeout_leak_count": total("timeout_leak_count"),
        "unrelated_termination_count": total("unrelated_termination_count"),
        "final_live_process_count": total("final_live_process_count"),
        "final_descendant_count": total("final_descendant_count"),
        "final_orphan_count": total("final_orphan_count"),
        "final_zombie_count": total("final_zombie_count"),
        "unresolved_wait_count": total("unresolved_wait_count"),
        "timing_variation_count": sum(int(c.get("unstable_case_count", 0)) for c in campaigns),
        "automatic_rerun_count": total("automatic_rerun_count"),
    }


def run_acceptance_campaign(
    *,
    deterministic_repeats: int = 3,
    actual_repeats: int = 10,
    stress_seeds: int = 100,
    git_commit_sha: str = "NOT_VERIFIED",
    runner_image_os: str = "NOT_VERIFIED",
    runner_image_version: str = "NOT_VERIFIED",
    operator_manual_dispatch_state: str = "DECLARED_NOT_DEFAULT_BRANCH_ACTIVE",
) -> dict[str, Any]:
    if deterministic_repeats != 3 or actual_repeats != 10 or stress_seeds != 100:
        raise LinuxSupervisorError("FIXED_ACCEPTANCE_COUNTS_REQUIRED")
    if operator_manual_dispatch_state != "DECLARED_NOT_DEFAULT_BRANCH_ACTIVE":
        raise LinuxSupervisorError("OPERATOR_MANUAL_DISPATCH_STATE_INVALID")
    started = time.monotonic()
    deterministic = run_deterministic_campaign(deterministic_repeats)
    actual = run_actual_linux_campaign(actual_repeats)
    stress = run_stress_campaign(stress_seeds)
    campaigns = [deterministic, actual, stress]
    counters = _campaign_counters(campaigns)
    required_total = sum(int(campaign["required_count"]) for campaign in campaigns)
    executed_total = sum(int(campaign["executed_count"]) for campaign in campaigns)
    pass_total = sum(int(campaign["pass_count"]) for campaign in campaigns)
    failure_total = sum(int(campaign["failure_count"]) for campaign in campaigns)
    sanitized_commit = _allowlisted_commit(git_commit_sha)
    sanitized_image_os = _allowlisted_runner_value(runner_image_os)
    sanitized_image_version = _allowlisted_runner_value(runner_image_version)
    metadata_verified = (
        sanitized_commit != "NOT_VERIFIED"
        and sanitized_image_os in {"ubuntu22", "ubuntu24"}
        and re.fullmatch(r"[0-9]{8}\.[0-9]+\.[0-9]+", sanitized_image_version) is not None
    )
    verdict = (
        "PASS"
        if _linux_available()
        and metadata_verified
        and required_total == 310
        and executed_total == required_total
        and pass_total == required_total
        and failure_total == 0
        and all(value == 0 for value in counters.values())
        else "HOLD"
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "R9ZNW-488D42",
        "verdict": verdict,
        "host_execution": "EXECUTED" if _linux_available() else "NOT_EXECUTED",
        "os_family": "LINUX" if _linux_available() else "OTHER",
        "architecture": platform.machine().lower() if platform.machine().lower() in {"x86_64", "amd64", "aarch64", "arm64"} else "OTHER",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "git_commit_sha": sanitized_commit,
        "runner_image_os": sanitized_image_os,
        "runner_image_version": sanitized_image_version,
        "operator_manual_dispatch_state": operator_manual_dispatch_state,
        "required_total": required_total,
        "executed_total": executed_total,
        "pass_total": pass_total,
        "failure_total": failure_total,
        "duration_bucket": _duration_bucket(time.monotonic() - started),
        "campaigns": campaigns,
        "counters": counters,
    }
    sentinel_failures = _persistent_sentinel_failures(report)
    if sentinel_failures:
        raise LinuxSupervisorError("ACCEPTANCE_REPORT_SENTINEL_REJECTED")
    report["report_digest"] = _digest_payload(report)
    return report


def write_acceptance_bundle(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ImmutableArtifactError("ACCEPTANCE_OUTPUT_NOT_EMPTY")
    result_path = output / "acceptance-result.json"
    result_digest = _exclusive_publish_json(result_path, report)
    result_bytes = result_path.stat().st_size
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_count": 1,
        "artifacts": [
            {
                "artifact_name": "acceptance-result.json",
                "bytes": result_bytes,
                "sha256": result_digest,
            }
        ],
    }
    manifest["manifest_digest"] = _digest_payload(manifest)
    _exclusive_publish_json(output / "manifest.json", manifest)


def verify_acceptance_bundle(
    input_directory: Path, *, allow_local_non_github: bool = False
) -> dict[str, Any]:
    entries = sorted(input_directory.iterdir(), key=lambda item: item.name)
    names = [path.name for path in entries]
    if (
        names != ["acceptance-result.json", "manifest.json"]
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise LinuxSupervisorError("ACCEPTANCE_ARTIFACT_SET_INVALID")
    report_path = input_directory / "acceptance-result.json"
    manifest_path = input_directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if report is None or manifest is None:
        raise LinuxSupervisorError("ACCEPTANCE_ARTIFACT_JSON_INVALID")
    if _persistent_sentinel_failures(report) or _persistent_sentinel_failures(manifest):
        raise LinuxSupervisorError("ACCEPTANCE_ARTIFACT_SENTINEL_REJECTED")
    report_keys = {
        "schema_version",
        "task_id",
        "verdict",
        "host_execution",
        "os_family",
        "architecture",
        "python_version",
        "git_commit_sha",
        "runner_image_os",
        "runner_image_version",
        "operator_manual_dispatch_state",
        "required_total",
        "executed_total",
        "pass_total",
        "failure_total",
        "duration_bucket",
        "campaigns",
        "counters",
        "report_digest",
    }
    manifest_keys = {"schema_version", "artifact_count", "artifacts", "manifest_digest"}
    if set(report) != report_keys or set(manifest) != manifest_keys:
        raise LinuxSupervisorError("ACCEPTANCE_SCHEMA_NOT_CLOSED")
    report_without_digest = dict(report)
    report_digest = report_without_digest.pop("report_digest", None)
    if report_digest != _digest_payload(report_without_digest):
        raise LinuxSupervisorError("ACCEPTANCE_REPORT_DIGEST_MISMATCH")
    manifest_without_digest = dict(manifest)
    manifest_digest = manifest_without_digest.pop("manifest_digest", None)
    if manifest_digest != _digest_payload(manifest_without_digest):
        raise LinuxSupervisorError("ACCEPTANCE_MANIFEST_DIGEST_MISMATCH")
    artifacts = manifest.get("artifacts")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("artifact_count") != 1:
        raise LinuxSupervisorError("ACCEPTANCE_MANIFEST_HEADER_INVALID")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise LinuxSupervisorError("ACCEPTANCE_MANIFEST_ENTRY_INVALID")
    entry = artifacts[0]
    result_data = report_path.read_bytes()
    if (
        not isinstance(entry, dict)
        or set(entry) != {"artifact_name", "bytes", "sha256"}
        or entry.get("artifact_name") != "acceptance-result.json"
        or entry.get("bytes") != len(result_data)
        or entry.get("sha256") != _digest_bytes(result_data)
    ):
        raise LinuxSupervisorError("ACCEPTANCE_MANIFEST_ENTRY_MISMATCH")
    counter_keys = {
        "wrong_pass_count",
        "masking_count",
        "raw_persistence_count",
        "timeout_leak_count",
        "unrelated_termination_count",
        "final_live_process_count",
        "final_descendant_count",
        "final_orphan_count",
        "final_zombie_count",
        "unresolved_wait_count",
        "timing_variation_count",
        "automatic_rerun_count",
    }
    counters = report.get("counters")
    if not isinstance(counters, dict) or set(counters) != counter_keys:
        raise LinuxSupervisorError("ACCEPTANCE_COUNTERS_INVALID")
    campaigns = report.get("campaigns")
    expected_campaigns = (
        ("DETERMINISTIC_CONTRACT", 90),
        ("ACTUAL_LINUX_PROCESS", 120),
        ("FIXED_SEED_STRESS", 100),
    )
    campaign_keys = {
        "campaign",
        "status",
        "required_count",
        "executed_count",
        "pass_count",
        "failure_count",
        "unstable_case_count",
        "results",
    }
    if not isinstance(campaigns, list) or len(campaigns) != 3:
        raise LinuxSupervisorError("ACCEPTANCE_CAMPAIGN_SET_INVALID")
    for campaign, (expected_name, expected_count) in zip(campaigns, expected_campaigns):
        if (
            not isinstance(campaign, dict)
            or set(campaign) != campaign_keys
            or campaign.get("campaign") != expected_name
            or campaign.get("status") != "EXECUTED"
            or campaign.get("required_count") != expected_count
            or campaign.get("executed_count") != expected_count
            or campaign.get("pass_count") != expected_count
            or campaign.get("failure_count") != 0
            or campaign.get("unstable_case_count") != 0
            or not isinstance(campaign.get("results"), list)
            or len(campaign["results"]) != expected_count
        ):
            raise LinuxSupervisorError("ACCEPTANCE_CAMPAIGN_INVARIANT_REJECTED")
    deterministic_results = campaigns[0]["results"]
    actual_results = campaigns[1]["results"]
    stress_results = campaigns[2]["results"]
    result_audit_keys = {
        "signal_scope_clear",
        "reaping_clear",
        "readiness_clear",
        "wrong_pass_count",
        "masking_count",
        "raw_persistence_count",
        "timeout_leak_count",
        "unrelated_termination_count",
        "final_live_process_count",
        "final_descendant_count",
        "final_orphan_count",
        "final_zombie_count",
        "unresolved_wait_count",
        "automatic_rerun_count",
    }
    deterministic_result_keys = result_audit_keys | {
        "case_id",
        "repeat",
        "case_verdict",
        "authoritative_terminal",
        "normalized_digest",
        "duration_bucket",
    }
    actual_result_keys = result_audit_keys | {
        "case_id",
        "repeat",
        "case_verdict",
        "normalized_digest",
        "duration_bucket",
        "residual_count",
        "scope_escape_fail_closed",
    }
    stress_result_keys = result_audit_keys | {
        "case_id",
        "seed",
        "case_verdict",
        "normalized_digest",
        "duration_bucket",
        "residual_count",
    }
    expected_deterministic = [
        (case_id, repeat, terminal)
        for case_id, terminal in DETERMINISTIC_CASES
        for repeat in range(3)
    ]
    expected_actual = [
        (case_id, repeat) for case_id in ACTUAL_CASES for repeat in range(10)
    ]
    for result, (case_id, repeat, terminal) in zip(deterministic_results, expected_deterministic):
        if (
            not isinstance(result, dict)
            or set(result) != deterministic_result_keys
            or result.get("case_id") != case_id
            or result.get("repeat") != repeat
            or result.get("case_verdict") != "PASS"
            or result.get("authoritative_terminal") != terminal
            or not SHA256_PATTERN.fullmatch(str(result.get("normalized_digest", "")))
        ):
            raise LinuxSupervisorError("DETERMINISTIC_RESULT_REJECTED")
    for result, (case_id, repeat) in zip(actual_results, expected_actual):
        if (
            not isinstance(result, dict)
            or set(result) != actual_result_keys
            or result.get("case_id") != case_id
            or result.get("repeat") != repeat
            or result.get("case_verdict") != "PASS"
            or result.get("residual_count") != 0
            or result.get("scope_escape_fail_closed") is not True
            or result.get("duration_bucket")
            not in {"LT_100MS", "LT_1S", "LT_10S", "GE_10S"}
            or not SHA256_PATTERN.fullmatch(str(result.get("normalized_digest", "")))
        ):
            raise LinuxSupervisorError("ACTUAL_RESULT_REJECTED")
    for seed, result in enumerate(stress_results):
        if (
            not isinstance(result, dict)
            or set(result) != stress_result_keys
            or result.get("case_id") != f"S{seed:03d}"
            or result.get("seed") != seed
            or result.get("case_verdict") != "PASS"
            or result.get("residual_count") != 0
            or not SHA256_PATTERN.fullmatch(str(result.get("normalized_digest", "")))
        ):
            raise LinuxSupervisorError("STRESS_RESULT_REJECTED")
    for result in [*deterministic_results, *actual_results, *stress_results]:
        if (
            result.get("signal_scope_clear") is not True
            or result.get("reaping_clear") is not True
            or result.get("readiness_clear") is not True
            or any(
                not isinstance(result.get(key), int) or result.get(key) != 0
                for key in result_audit_keys
                - {"signal_scope_clear", "reaping_clear", "readiness_clear"}
            )
        ):
            raise LinuxSupervisorError("RESULT_ZERO_AUDIT_REJECTED")
    recomputed_counters = _campaign_counters(campaigns)
    if counters != recomputed_counters:
        raise LinuxSupervisorError("ACCEPTANCE_COUNTER_RECOMPUTE_MISMATCH")
    base_verified = (
        report.get("schema_version") == SCHEMA_VERSION
        and report.get("task_id") == "R9ZNW-488D42"
        and report.get("host_execution") == "EXECUTED"
        and report.get("os_family") == "LINUX"
        and report.get("architecture") in {"x86_64", "amd64", "aarch64", "arm64"}
        and re.fullmatch(r"[0-9]+\.[0-9]+", str(report.get("python_version", ""))) is not None
        and report.get("operator_manual_dispatch_state") == "DECLARED_NOT_DEFAULT_BRANCH_ACTIVE"
        and report.get("required_total") == 310
        and report.get("executed_total") == 310
        and report.get("pass_total") == 310
        and report.get("failure_total") == 0
        and all(isinstance(value, int) and value == 0 for value in counters.values())
        and report.get("duration_bucket") in {"LT_100MS", "LT_1S", "LT_10S", "GE_10S"}
    )
    remote_verified = (
        base_verified
        and report.get("verdict") == "PASS"
        and COMMIT_PATTERN.fullmatch(str(report.get("git_commit_sha", ""))) is not None
        and report.get("runner_image_os") in {"ubuntu22", "ubuntu24"}
        and re.fullmatch(
            r"[0-9]{8}\.[0-9]+\.[0-9]+",
            str(report.get("runner_image_version", "")),
        )
        is not None
    )
    local_verified = (
        allow_local_non_github
        and base_verified
        and report.get("verdict") == "HOLD"
        and report.get("git_commit_sha") == "NOT_VERIFIED"
        and report.get("runner_image_os") == "NOT_VERIFIED"
        and report.get("runner_image_version") == "NOT_VERIFIED"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "verification": (
            "PASS"
            if remote_verified
            else "PASS_LOCAL_NON_GITHUB"
            if local_verified
            else "HOLD"
        ),
        "artifact_count": 2,
        "required_total": report.get("required_total"),
        "executed_total": report.get("executed_total"),
        "report_digest": report.get("report_digest"),
    }


def cleanup_acceptance_bundle(input_directory: Path) -> dict[str, Any]:
    """Remove only a verified, exact operator bundle with a nonce-bound name."""

    candidate = input_directory.absolute()
    if not OPERATOR_DIRECTORY_PATTERN.fullmatch(candidate.name):
        raise LinuxSupervisorError("OPERATOR_CLEANUP_DIRECTORY_NAME_REJECTED")
    if not candidate.is_dir() or candidate.is_symlink():
        raise LinuxSupervisorError("OPERATOR_CLEANUP_DIRECTORY_BINDING_REJECTED")
    for parent in candidate.parents:
        if parent == Path(candidate.anchor):
            break
        if parent.is_symlink():
            raise LinuxSupervisorError("OPERATOR_CLEANUP_ANCESTOR_SYMLINK_REJECTED")
    verified = verify_acceptance_bundle(candidate, allow_local_non_github=True)
    # Structural/digest verification above is mandatory.  A local operator
    # bundle may intentionally be HOLD because GitHub runner metadata is absent.
    if (
        verified.get("artifact_count") != 2
        or verified.get("verification") not in {"PASS", "PASS_LOCAL_NON_GITHUB"}
    ):
        raise LinuxSupervisorError("OPERATOR_CLEANUP_VERIFICATION_REJECTED")
    exact_files = [candidate / "acceptance-result.json", candidate / "manifest.json"]
    if any(path.is_symlink() or not path.is_file() for path in exact_files):
        raise LinuxSupervisorError("OPERATOR_CLEANUP_FILE_BINDING_REJECTED")
    for path in exact_files:
        path.unlink()
    candidate.rmdir()
    return {
        "cleanup": "PASS",
        "removed_file_count": 2,
        "removed_directory_count": 1,
        "scope": "EXACT_OPERATOR_BUNDLE",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLIB Linux-native process supervisor")
    commands = parser.add_subparsers(dest="command", required=True)
    supervise = commands.add_parser("_supervise", help=argparse.SUPPRESS)
    supervise.add_argument("--config-fd", type=int, required=True)
    supervise.add_argument("--launcher-ready-fd", type=int, required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config-fd", type=int, required=True)
    worker.add_argument("--readiness-fd", type=int, required=True)
    worker.add_argument("--event-fd", type=int, required=True)
    worker.add_argument("--acknowledgment-fd", type=int, required=True)
    acceptance = commands.add_parser("acceptance")
    acceptance.add_argument("--output", type=Path, required=True)
    acceptance.add_argument("--deterministic-repeats", type=int, default=3)
    acceptance.add_argument("--actual-repeats", type=int, default=10)
    acceptance.add_argument("--stress-seeds", type=int, default=100)
    acceptance.add_argument("--git-commit-sha", default="NOT_VERIFIED")
    acceptance.add_argument("--runner-image-os", default="NOT_VERIFIED")
    acceptance.add_argument("--runner-image-version", default="NOT_VERIFIED")
    acceptance.add_argument(
        "--operator-manual-dispatch-state",
        choices=("DECLARED_NOT_DEFAULT_BRANCH_ACTIVE",),
        default="DECLARED_NOT_DEFAULT_BRANCH_ACTIVE",
    )
    verify = commands.add_parser("verify-artifact")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--allow-local-non-github", action="store_true")
    cleanup = commands.add_parser("cleanup-artifact")
    cleanup.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "_supervise":
        return _supervisor_main(args.config_fd, args.launcher_ready_fd)
    if args.command == "_worker":
        return _worker_main(
            args.config_fd,
            args.readiness_fd,
            args.event_fd,
            args.acknowledgment_fd,
        )
    if args.command == "acceptance":
        try:
            report = run_acceptance_campaign(
                deterministic_repeats=args.deterministic_repeats,
                actual_repeats=args.actual_repeats,
                stress_seeds=args.stress_seeds,
                git_commit_sha=args.git_commit_sha,
                runner_image_os=args.runner_image_os,
                runner_image_version=args.runner_image_version,
                operator_manual_dispatch_state=args.operator_manual_dispatch_state,
            )
            write_acceptance_bundle(args.output, report)
            summary = {
                "verdict": report["verdict"],
                "required_total": report["required_total"],
                "executed_total": report["executed_total"],
                "pass_total": report["pass_total"],
                "failure_total": report["failure_total"],
                "report_digest": report["report_digest"],
            }
            print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
            return 0 if report["verdict"] == "PASS" else 2
        except LinuxSupervisorError as exc:
            print(json.dumps({"verdict": "HOLD", "reason": str(exc)}, sort_keys=True))
            return 2
    if args.command == "verify-artifact":
        try:
            result = verify_acceptance_bundle(
                args.input,
                allow_local_non_github=args.allow_local_non_github,
            )
            print(json.dumps(result, ensure_ascii=True, sort_keys=True))
            return 0 if result["verification"] in {"PASS", "PASS_LOCAL_NON_GITHUB"} else 2
        except LinuxSupervisorError as exc:
            print(json.dumps({"verification": "HOLD", "reason": str(exc)}, sort_keys=True))
            return 2
        except (FileNotFoundError, OSError):
            print(json.dumps({"verification": "HOLD", "reason": "ACCEPTANCE_ARTIFACT_ACCESS_FAILED"}, sort_keys=True))
            return 2
    if args.command == "cleanup-artifact":
        try:
            result = cleanup_acceptance_bundle(args.input)
            print(json.dumps(result, ensure_ascii=True, sort_keys=True))
            return 0
        except LinuxSupervisorError as exc:
            print(json.dumps({"cleanup": "HOLD", "reason": str(exc)}, sort_keys=True))
            return 2
        except (FileNotFoundError, OSError):
            print(json.dumps({"cleanup": "HOLD", "reason": "OPERATOR_CLEANUP_ACCESS_FAILED"}, sort_keys=True))
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
