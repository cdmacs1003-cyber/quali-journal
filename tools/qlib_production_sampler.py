"""Canonical bounded production sampler for the QLIB Skill-up observer.

The sampler keeps endpoint and identity material in memory only.  It emits a
small sanitized JSON contract to stdout and a replace-only progress marker for
the parent observer.  The progress marker intentionally contains no argv,
location, response body, identity material, question, or answer.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import qlib_traffic_observer as observer


SERVICE = "qlib-skillup-runtime"
REGION = "asia-northeast1"
TOTAL_DEADLINE_SECONDS = 30.0
PROGRESS_FILE_ENV = "QLIB_SAMPLER_PROGRESS_FILE"
DEADLINE_ENV = "QLIB_SAMPLER_TOTAL_DEADLINE_SECONDS"
TARGET_CONTRACT = "REVISION_FUNCTIONAL"
PHASES = (
    "AUTHENTICATION",
    "TARGET_RESOLUTION",
    "REQUEST_PREPARATION",
    "HTTP_REQUEST",
    "NORMALIZATION",
    "EVIDENCE_TRACE_SAFE_EXTRACTION",
    "SERIALIZATION",
    "PARENT_CHILD_IPC",
    "CHILD_CLEANUP",
    "OBSERVER_COLLECTION",
)
ALLOWED_EVIDENCE = {
    "ev-flux-safe-summary-v1",
    "ev-solder-basic-and-types-safe-summary-v1",
    "ev-soldering-safe-summary-v1",
    "ev-wetting-safe-summary-v1",
}
IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,61}[a-z0-9]$")
ASSET_PATTERN = re.compile(r"(?:src|href)=[\"'](assets/[^\"'?]+)")
SENSITIVE_PATTERN = re.compile(
    r"(?i)(bearer\s+[a-z0-9._-]{20,}|private[_ -]?key|credential[_ -]?value)"
)


class SafeFailure(RuntimeError):
    """Sanitized failure category safe for observer artifacts."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class TotalDeadlineExceeded(SafeFailure):
    def __init__(self) -> None:
        super().__init__("TIMEOUT")


@dataclass(frozen=True)
class RequestContract:
    """Validated in-memory routing and audience values."""

    audience: str
    stable_health: str
    tagged_health: str
    tagged_origin: str


class Deadline:
    def __init__(
        self,
        seconds: float = TOTAL_DEADLINE_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if seconds <= 0 or seconds > TOTAL_DEADLINE_SECONDS:
            raise SafeFailure("ARGUMENT_OR_SERIALIZATION_DEFECT")
        self.seconds = float(seconds)
        self._clock = clock
        self._started = clock()
        self._ends = self._started + self.seconds

    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    def remaining_seconds(self) -> float:
        remaining = self._ends - self._clock()
        if remaining <= 0:
            raise TotalDeadlineExceeded()
        return remaining


class CommandCounters:
    def __init__(self) -> None:
        self.read_only = 0
        self.mutation = 0

    def record_read_only(self) -> None:
        self.read_only += 1

    def record_mutation(self) -> None:
        self.mutation += 1
        raise SafeFailure("ARGUMENT_OR_SERIALIZATION_DEFECT")


class PhaseRecorder:
    def __init__(
        self,
        *,
        counters: CommandCounters,
        progress_path: Path | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.counters = counters
        self.progress_path = progress_path
        self._clock = clock
        self.current_phase = "NOT_STARTED"
        self.last_completed_phase = "NONE"
        self.phase_timings: list[dict[str, Any]] = []
        self._phase_started = clock()

    def snapshot(
        self,
        *,
        timeout_reason: str = "NONE",
        valid_sample_count: int = 0,
        child_exit_state: str = "RUNNING",
        child_cancellation_state: str = "NOT_REQUIRED",
    ) -> dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "last_completed_phase": self.last_completed_phase,
            "phase_timings": list(self.phase_timings),
            "timeout_reason": timeout_reason,
            "read_only_command_count": int(self.counters.read_only),
            "mutation_command_count": int(self.counters.mutation),
            "valid_sample_count": int(valid_sample_count),
            "child_exit_state": child_exit_state,
            "child_cancellation_state": child_cancellation_state,
            "orphan_child_count": 0,
        }

    def _write_progress(self, **overrides: Any) -> None:
        if self.progress_path is None:
            return
        payload = self.snapshot(**overrides)
        temporary = self.progress_path.with_name(
            f".{self.progress_path.name}.{os.getpid()}.tmp"
        )
        try:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.progress_path)
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass

    def begin(self, phase: str) -> None:
        if phase not in PHASES:
            raise SafeFailure("ARGUMENT_OR_SERIALIZATION_DEFECT")
        self.current_phase = phase
        self._phase_started = self._clock()
        self._write_progress()

    def complete(self, phase: str) -> None:
        if self.current_phase != phase:
            raise SafeFailure("ARGUMENT_OR_SERIALIZATION_DEFECT")
        elapsed_ms = max(0.0, (self._clock() - self._phase_started) * 1000.0)
        self.phase_timings.append(
            {"phase": phase, "elapsed_ms": round(elapsed_ms, 3), "status": "PASS"}
        )
        self.last_completed_phase = phase
        self.current_phase = "BETWEEN_PHASES"
        self._write_progress()

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        self.begin(phase)
        try:
            yield
        except Exception as exc:
            elapsed_ms = min(
                TOTAL_DEADLINE_SECONDS * 1000.0,
                max(0.0, (self._clock() - self._phase_started) * 1000.0),
            )
            self.phase_timings.append(
                {
                    "phase": phase,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "status": (
                        "TIMEOUT" if isinstance(exc, TotalDeadlineExceeded) else "FAIL"
                    ),
                }
            )
            self._write_progress(
                timeout_reason=(
                    "TOTAL_DEADLINE_EXCEEDED"
                    if isinstance(exc, TotalDeadlineExceeded)
                    else "NONE"
                )
            )
            raise
        else:
            self.complete(phase)


GcloudRunner = Callable[[list[str], Deadline, CommandCounters], str]
HttpRunner = Callable[
    [str, Deadline, CommandCounters, str, str, bytes | None],
    tuple[int, bytes, float],
]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validated_run_app_origin(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SafeFailure("TARGET_ROUTING_CONTRACT_DEFECT")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise SafeFailure("TARGET_ROUTING_CONTRACT_DEFECT") from exc
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".run.app")
        or hostname == "run.app"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.netloc != hostname
        or value != f"https://{hostname}"
    ):
        raise SafeFailure("TARGET_ROUTING_CONTRACT_DEFECT")
    return value


def _request_contract(service: dict[str, Any], routes: list[dict[str, Any]]) -> RequestContract:
    if len(routes) != 1 or not routes[0].get("url"):
        raise SafeFailure("TARGET_ROUTING_CONTRACT_DEFECT")
    stable_origin = _validated_run_app_origin(
        (service.get("status") or {}).get("url")
    )
    tagged_origin = _validated_run_app_origin(routes[0].get("url"))
    return RequestContract(
        audience=stable_origin,
        stable_health=stable_origin + "/health",
        tagged_health=tagged_origin + "/health",
        tagged_origin=tagged_origin,
    )


def _identity_token_arguments(audience: str) -> list[str]:
    arguments = [
        "auth",
        "print-identity-token",
        f"--audiences={audience}",
        "--quiet",
    ]
    _validate_identity_token_arguments(arguments, audience)
    return arguments


def _validate_identity_token_arguments(arguments: list[str], audience: str) -> None:
    audience_arguments = [item for item in arguments if item.startswith("--audiences=")]
    if (
        arguments
        != [
            "auth",
            "print-identity-token",
            f"--audiences={audience}",
            "--quiet",
        ]
        or audience_arguments != [f"--audiences={audience}"]
    ):
        raise SafeFailure("AUTH_FAILURE")


def _token_audience_matches(identity: str, audience: str) -> bool:
    try:
        sections = identity.split(".")
        if len(sections) != 3 or not sections[1]:
            raise ValueError("invalid JWT shape")
        payload_segment = sections[1]
        padding = "=" * (-len(payload_segment) % 4)
        payload = base64.urlsafe_b64decode((payload_segment + padding).encode("ascii"))
        claims = json.loads(payload.decode("utf-8"))
    except (
        UnicodeError,
        ValueError,
        binascii.Error,
        json.JSONDecodeError,
    ) as exc:
        raise SafeFailure("AUTH_FAILURE") from exc
    if not isinstance(claims, dict) or not isinstance(claims.get("aud"), str):
        raise SafeFailure("AUTH_FAILURE")
    return claims["aud"] == audience


def _deadline_from_environment(environment: dict[str, str] | None = None) -> float:
    raw = (environment or os.environ).get(DEADLINE_ENV, str(TOTAL_DEADLINE_SECONDS))
    try:
        value = float(raw)
    except ValueError as exc:
        raise SafeFailure("ARGUMENT_OR_SERIALIZATION_DEFECT") from exc
    if value <= 0 or value > TOTAL_DEADLINE_SECONDS:
        raise SafeFailure("ARGUMENT_OR_SERIALIZATION_DEFECT")
    return value


def _progress_path(environment: dict[str, str] | None = None) -> Path | None:
    raw = (environment or os.environ).get(PROGRESS_FILE_ENV, "")
    return Path(raw) if raw else None


def _gcloud_text(
    arguments: list[str], deadline: Deadline, counters: CommandCounters
) -> str:
    counters.record_read_only()
    timeout = max(0.05, deadline.remaining_seconds())
    try:
        completed = subprocess.run(
            ["gcloud.cmd", *arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TotalDeadlineExceeded() from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafeFailure("DEPENDENCY_OR_SUBPROCESS_DEFECT") from exc
    if completed.returncode != 0:
        raise SafeFailure("DEPENDENCY_OR_SUBPROCESS_DEFECT")
    return completed.stdout.strip()


def _http_request(
    location: str,
    deadline: Deadline,
    counters: CommandCounters,
    identity: str = "",
    method: str = "GET",
    body: bytes | None = None,
) -> tuple[int, bytes, float]:
    counters.record_read_only()
    headers = {"Content-Type": "application/json"}
    if identity:
        headers["Authorization"] = "Bearer " + identity
    request = urllib.request.Request(
        location, data=body, headers=headers, method=method
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
            request, timeout=max(0.05, deadline.remaining_seconds())
        ) as response:
            data = response.read()
            return (
                int(response.status),
                data,
                round((time.monotonic() - started) * 1000.0, 3),
            )
    except urllib.error.HTTPError as exc:
        data = exc.read()
        return (
            int(exc.code),
            data,
            round((time.monotonic() - started) * 1000.0, 3),
        )
    except (TimeoutError, urllib.error.URLError) as exc:
        if deadline.elapsed_seconds() >= deadline.seconds:
            raise TotalDeadlineExceeded() from exc
        raise SafeFailure("DEPENDENCY_OR_SUBPROCESS_DEFECT") from exc


def _parse_object(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeFailure("JSON_PARSE_FAILURE") from exc
    if not isinstance(value, dict):
        raise SafeFailure("JSON_PARSE_FAILURE")
    return value


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _strings(nested)]
    return []


def _fixed_payloads() -> tuple[bytes, bytes]:
    answered_question = "".join(chr(value) for value in (0xC194, 0xB354, 0xB9C1, 0xC774, 0xB780, 0x3F))
    held_question = "".join(
        chr(value)
        for value in (
            0xC2B9,
            0xC778,
            0xB41C,
            0x20,
            0xADFC,
            0xAC70,
            0xAC00,
            0x20,
            0xC5C6,
            0xB294,
            0x20,
            0xD604,
            0xC7A5,
            0x20,
            0xC9C8,
            0xBB38,
        )
    )
    base = {"requester_module": "Skillup", "ui_mode": "beta_minimal"}
    answered = dict(base, request_payload={"question": answered_question})
    held = dict(base, request_payload={"question": held_question})
    return (
        json.dumps(answered, ensure_ascii=False).encode("utf-8"),
        json.dumps(held, ensure_ascii=False).encode("utf-8"),
    )


def _failure_counts(category: str) -> dict[str, int]:
    counts = {
        "http_403_count": 0,
        "http_404_count": 0,
        "unexpected_5xx_count": 0,
        "timeout_count": 0,
        "json_parse_failure_count": 0,
        "auth_failure_count": 0,
        "dependency_failure_count": 0,
        "target_routing_failure_count": 0,
    }
    key = {
        "HTTP_403": "http_403_count",
        "HTTP_404": "http_404_count",
        "HTTP_5XX": "unexpected_5xx_count",
        "TIMEOUT": "timeout_count",
        "JSON_PARSE_FAILURE": "json_parse_failure_count",
        "AUTH_FAILURE": "auth_failure_count",
        "TARGET_ROUTING_CONTRACT_DEFECT": "target_routing_failure_count",
    }.get(category, "dependency_failure_count")
    counts[key] = 1
    return counts


def _base_summary(
    *,
    recorder: PhaseRecorder,
    target_contract: str,
    sample_status: str,
    valid_sample_count: int,
    failure_category: str = "NONE",
) -> dict[str, Any]:
    diagnostics = recorder.snapshot(
        timeout_reason=(
            "TOTAL_DEADLINE_EXCEEDED" if failure_category == "TIMEOUT" else "NONE"
        ),
        valid_sample_count=valid_sample_count,
        child_exit_state=("EXITING_ZERO" if sample_status == "PASS" else "EXITING_NONZERO"),
    )
    return {
        "sample_status": sample_status,
        "failure_category": failure_category,
        "target_contract": target_contract,
        "target_selector_hash": _sha256_text("tagged-functional-only"),
        "command_contract_hash": _sha256_text(
            "expected-candidate,expected-stable,target-contract"
        ),
        "argument_name_set": [
            "expected_candidate",
            "expected_stable",
            "target_contract",
        ],
        "sanitized_environment_key_set": ["comspec", "systemroot", "temp"],
        "import_status": "PASS",
        "dependency_status": "PASS" if sample_status == "PASS" else "FAIL",
        "auth_handoff_status": "PASS" if sample_status == "PASS" else "FAIL",
        "target_construction_status": "PASS" if sample_status == "PASS" else "FAIL",
        "health_sample_status": "PASS" if sample_status == "PASS" else "FAIL",
        "readiness_status": "PASS" if sample_status == "PASS" else "FAIL",
        "latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "evidence_trace_safe_summary": (
            "PASS" if sample_status == "PASS" else "NOT_VERIFIED"
        ),
        **_failure_counts(failure_category),
        **diagnostics,
        "raw_response_persisted": False,
        "identity_material_persisted": False,
        "production_write_count": 0,
    }


def collect(
    *,
    target_contract: str,
    candidate: str,
    tag: str,
    expected_candidate: int,
    expected_stable: int,
    deadline_seconds: float = TOTAL_DEADLINE_SECONDS,
    gcloud_runner: GcloudRunner = _gcloud_text,
    http_runner: HttpRunner = _http_request,
    clock: Callable[[], float] = time.monotonic,
    progress_path: Path | None = None,
    counters: CommandCounters | None = None,
    deadline: Deadline | None = None,
    recorder: PhaseRecorder | None = None,
) -> dict[str, Any]:
    if (
        target_contract != TARGET_CONTRACT
        or not IDENTIFIER_PATTERN.fullmatch(candidate)
        or not IDENTIFIER_PATTERN.fullmatch(tag)
        or expected_candidate != 0
        or expected_stable != 100
    ):
        raise SafeFailure("ARGUMENT_OR_SERIALIZATION_DEFECT")

    counters = counters or CommandCounters()
    deadline = deadline or Deadline(deadline_seconds, clock=clock)
    recorder = recorder or PhaseRecorder(
        counters=counters, progress_path=progress_path, clock=clock
    )
    identity = ""
    active_project = ""
    identity_arguments: list[str] = []
    request_contract: RequestContract | None = None
    responses: dict[str, tuple[int, bytes, float]] = {}
    normalized: dict[str, Any] = {}

    try:
        with recorder.phase("TARGET_RESOLUTION"):
            active_project = gcloud_runner(
                ["config", "get-value", "project", "--quiet"], deadline, counters
            )
            service_text = gcloud_runner(
                [
                    "run",
                    "services",
                    "describe",
                    SERVICE,
                    "--region",
                    REGION,
                    "--project",
                    active_project,
                    "--format=json",
                    "--quiet",
                ],
                deadline,
                counters,
            )
            try:
                service = json.loads(service_text)
            except json.JSONDecodeError as exc:
                raise SafeFailure("JSON_PARSE_FAILURE") from exc
            routes = [
                item
                for item in service.get("status", {}).get("traffic", [])
                if item.get("tag") == tag
            ]
            if (
                len(routes) != 1
                or routes[0].get("revisionName") != candidate
            ):
                raise SafeFailure("TARGET_ROUTING_CONTRACT_DEFECT")
            request_contract = _request_contract(service, routes)
            deadline.remaining_seconds()

        with recorder.phase("AUTHENTICATION"):
            identity_arguments = _identity_token_arguments(request_contract.audience)
            identity = gcloud_runner(identity_arguments, deadline, counters)
            deadline.remaining_seconds()
            if not identity or not _token_audience_matches(
                identity, request_contract.audience
            ):
                raise SafeFailure("AUTH_FAILURE")

        with recorder.phase("REQUEST_PREPARATION"):
            answered_payload, held_payload = _fixed_payloads()
            deadline.remaining_seconds()

        with recorder.phase("HTTP_REQUEST"):
            responses["unauth_health"] = http_runner(
                request_contract.tagged_health, deadline, counters, "", "GET", None
            )
            responses["auth_health"] = http_runner(
                request_contract.tagged_health, deadline, counters, identity, "GET", None
            )
            responses["ui"] = http_runner(
                request_contract.tagged_origin + "/", deadline, counters, identity, "GET", None
            )
            ui_text = responses["ui"][1].decode("utf-8", errors="replace")
            assets = sorted(set(ASSET_PATTERN.findall(ui_text)))
            for index, asset in enumerate(assets):
                responses[f"asset_{index}"] = http_runner(
                    request_contract.tagged_origin + "/" + asset,
                    deadline,
                    counters,
                    identity,
                    "GET",
                    None,
                )
            endpoint = request_contract.tagged_origin + "/api/f13/bridge/skillup/bridge-answer"
            responses["answered"] = http_runner(
                endpoint, deadline, counters, identity, "POST", answered_payload
            )
            responses["held"] = http_runner(
                endpoint, deadline, counters, identity, "POST", held_payload
            )
            identity = ""
            answered_payload = held_payload = b""
            deadline.remaining_seconds()

        with recorder.phase("NORMALIZATION"):
            unauth_status = responses["unauth_health"][0]
            auth_status = responses["auth_health"][0]
            ui_status = responses["ui"][0]
            health = _parse_object(responses["auth_health"][1]) if auth_status == 200 else {}
            answered = _parse_object(responses["answered"][1]) if responses["answered"][0] == 200 else {}
            held = _parse_object(responses["held"][1]) if responses["held"][0] == 200 else {}
            latencies = [item[2] for item in responses.values()]
            statuses = [item[0] for item in responses.values()]
            normalized = {
                "unauth_status": unauth_status,
                "auth_status": auth_status,
                "ui_status": ui_status,
                "health": health.get("status"),
                "health_service": health.get("service"),
                "answered": answered,
                "held": held,
                "latencies": latencies,
                "statuses": statuses,
                "asset_count": len(assets),
                "ui_marker": "beta-minimal-form" in ui_text,
            }
            ui_text = ""
            responses = {}
            deadline.remaining_seconds()

        with recorder.phase("EVIDENCE_TRACE_SAFE_EXTRACTION"):
            answered = normalized["answered"]
            held = normalized["held"]
            evidence_ids = sorted(
                {
                    str(item.get("evidence_id"))
                    for item in answered.get("evidence") or []
                    if isinstance(item, dict) and item.get("evidence_id")
                }
            )
            evidence_ok = bool(evidence_ids) and all(
                value in ALLOWED_EVIDENCE for value in evidence_ids
            )
            trace_ok = bool(answered.get("trace_id"))
            classifications = [
                observer.privacy_safe_internal_path_classification(
                    value, response_surface_category=category
                )
                for value, category in (
                    (answered, "SUPPORTED_JSON"),
                    (held, "HOLD_JSON"),
                )
            ]
            exposure_count = sum(
                item["classification"] == "TRUE_INTERNAL_PATH"
                for result in classifications
                for item in result["matches"]
            ) + sum(
                len(SENSITIVE_PATTERN.findall(value))
                for value in _strings([answered, held])
            )
            retention_count = int(answered.get("raw_query_answer_retention_count") or 0) + int(
                held.get("raw_query_answer_retention_count") or 0
            )
            production_write_count = int(answered.get("production_write_count") or 0) + int(
                held.get("production_write_count") or 0
            )
            statuses = normalized["statuses"]
            failures = sum(
                (
                    normalized["unauth_status"] != 403,
                    normalized["auth_status"] != 200,
                    type(normalized["health"]) is not str,
                    normalized["health"] != "ok",
                    type(normalized["health_service"]) is not str,
                    normalized["health_service"] != SERVICE,
                    normalized["ui_status"] != 200,
                    not normalized["ui_marker"],
                    normalized["asset_count"] != 5,
                    any(value != 200 for key, value in enumerate(statuses) if key >= 3),
                    normalized["answered"].get("answer_status") != "ANSWERED",
                    normalized["held"].get("answer_status") != "HOLD",
                    not evidence_ok,
                    not trace_ok,
                    exposure_count != 0,
                    retention_count != 0,
                    production_write_count != 0,
                    max(normalized["latencies"] or [0.0]) > 3000.0,
                )
            )
            deadline.remaining_seconds()

        if failures:
            raise SafeFailure("FUNCTIONAL_HTTP_FAILURE")

        with recorder.phase("SERIALIZATION"):
            summary = _base_summary(
                recorder=recorder,
                target_contract=target_contract,
                sample_status="PASS",
                valid_sample_count=1,
            )
            summary.update(
                {
                    "unauth_http": int(normalized["unauth_status"]),
                    "auth_http": int(normalized["auth_status"]),
                    "normalized_health": str(normalized["health"]),
                    "audience_match": True,
                    "audience_sha256": _sha256_text(request_contract.audience),
                    "health_schema_match": True,
                    "latency_ms": round(float(normalized["latencies"][1]), 3),
                    "max_latency_ms": round(max(normalized["latencies"] or [0.0]), 3),
                    "evidence_missing_count": 0,
                    "trace_missing_count": 0,
                    "production_write_count": 0,
                }
            )
            json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
            deadline.remaining_seconds()

        recorder.begin("PARENT_CHILD_IPC")
        summary.update(recorder.snapshot(valid_sample_count=1, child_exit_state="EXITING_ZERO"))
        recorder._write_progress(valid_sample_count=1, child_exit_state="EXITING_ZERO")
        return summary
    finally:
        identity = ""
        active_project = ""
        identity_arguments = []
        request_contract = None
        responses = {}


def execute(
    args: argparse.Namespace,
    *,
    environment: dict[str, str] | None = None,
    gcloud_runner: GcloudRunner = _gcloud_text,
    http_runner: HttpRunner = _http_request,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], int]:
    counters = CommandCounters()
    recorder = PhaseRecorder(
        counters=counters,
        progress_path=_progress_path(environment),
        clock=clock,
    )
    try:
        deadline_seconds = _deadline_from_environment(environment)
        deadline = Deadline(deadline_seconds, clock=clock)
        result = collect(
            target_contract=args.target_contract,
            candidate=args.candidate,
            tag=args.tag,
            expected_candidate=args.expected_candidate,
            expected_stable=args.expected_stable,
            deadline_seconds=deadline_seconds,
            gcloud_runner=gcloud_runner,
            http_runner=http_runner,
            clock=clock,
            progress_path=_progress_path(environment),
            counters=counters,
            deadline=deadline,
            recorder=recorder,
        )
        return result, 0
    except SafeFailure as exc:
        failure = _base_summary(
            recorder=recorder,
            target_contract=str(args.target_contract),
            sample_status="FAIL",
            valid_sample_count=0,
            failure_category=exc.category,
        )
        recorder._write_progress(
            timeout_reason=(
                "TOTAL_DEADLINE_EXCEEDED" if exc.category == "TIMEOUT" else "NONE"
            ),
            valid_sample_count=0,
            child_exit_state="EXITING_NONZERO",
        )
        return failure, 42
    except Exception:
        failure = _base_summary(
            recorder=recorder,
            target_contract=str(args.target_contract),
            sample_status="FAIL",
            valid_sample_count=0,
            failure_category="DEPENDENCY_OR_SUBPROCESS_DEFECT",
        )
        recorder._write_progress(
            valid_sample_count=0, child_exit_state="EXITING_NONZERO"
        )
        return failure, 42


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-contract", choices=(TARGET_CONTRACT,), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-candidate", type=int, required=True)
    parser.add_argument("--expected-stable", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result, exit_code = execute(args)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
