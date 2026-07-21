"""Run the exact D42C platform tiers without hiding legacy failures.

Native Win32-only cases remain executable diagnostics.  Dual-platform tests
whose Windows branch asserts the retired native launcher stay required on
Linux and are separated only from the Windows portable tier.  This runner
persists counts, enums, and digests; unittest tracebacks and raw runtime values
remain in memory and are never written to the evidence bundle.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "R9ZNW-488D42C_PLATFORM_TIER_EVIDENCE_V1"
MANIFEST_SCHEMA_VERSION = "R9ZNW-488D42C_TEST_TIERS_V1"
LINUX_TIER = "LINUX_SHARED_REQUIRED"
WINDOWS_TIER = "WINDOWS_PORTABLE_SHARED"
WINDOWS_NATIVE_VERDICT = "WINDOWS_NATIVE_OBSERVER_NOT_APPROVED"
FAILURE_SCHEMA_VERSION = "R9ZNW-488D42C_PLATFORM_TIER_FAILURE_V2"
FAILURE_FILENAME = "platform-tier-failure.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXACT_MODULE_ORDER = (
    "tools.test_qlib_linux_process_supervisor",
    "tools.test_qlib_traffic_observer",
    "tools.test_qlib_observer_evidence_contract",
    "tools.test_qlib_production_sampler",
)
ROOT_KEYS = frozenset(
    {
        "schema_version",
        "module_order",
        "expected_discovered_test_count",
        "expected_discovered_test_ids_digest",
        "required_shared_test_ids",
        "legacy_native_win32_test_ids",
        "windows_native_public_legacy_expectation_test_ids",
        "windows_portable_linux_only_exclusions",
        "tiers",
        "windows_native_support_verdict",
    }
)
SKIP_OR_XFAIL_DECORATORS = frozenset(
    {"skip", "skipIf", "skipUnless", "expectedFailure"}
)
FAILURE_OUTCOMES = frozenset(
    {"FAILURE", "ERROR", "SKIPPED", "EXPECTED_FAILURE", "UNEXPECTED_SUCCESS"}
)
FAILURE_DETAIL_CLASSES = frozenset(
    {
        "ASSERTION_ERROR",
        "ATTRIBUTE_ERROR",
        "BROKEN_PIPE_ERROR",
        "CHILD_PROCESS_ERROR",
        "EOF_ERROR",
        "FILE_NOT_FOUND_ERROR",
        "INDEX_ERROR",
        "JSON_DECODE_ERROR",
        "KEY_ERROR",
        "LINUX_SUPERVISOR_ERROR",
        "NAME_ERROR",
        "NOT_IMPLEMENTED_ERROR",
        "OBSERVER_ERROR",
        "OS_ERROR",
        "PERMISSION_ERROR",
        "PROCESS_LOOKUP_ERROR",
        "RECURSION_ERROR",
        "RUNTIME_ERROR",
        "SAMPLER_FAILURE",
        "STOP_ITERATION",
        "SUBPROCESS_TIMEOUT_EXPIRED",
        "TIMEOUT_ERROR",
        "TYPE_ERROR",
        "UNCLASSIFIED_NONPASS",
        "UNICODE_ERROR",
        "VALUE_ERROR",
    }
)
FAILURE_DETAIL_REASONS = frozenset(
    {
        "BOUNDED_READINESS_NOT_RECEIVED",
        "CLEANUP_POLL_FAILED",
        "CLEANUP_STOP_FAILED",
        "EMERGENCY_STOP_PUBLICATION_FAILED",
        "EXACT_CHILD_REMAINS",
        "EXACT_FALLBACK_SIGNAL_REJECTED",
        "EXACT_IDENTITY_UNCERTAIN",
        "MARKER_BINDING_REJECTED",
        "MARKER_IDENTITY_UNAVAILABLE",
        "MARKER_IDENTITY_MISMATCH",
        "MARKER_SCHEMA_REJECTED",
        "MARKER_WITHOUT_EXACT_IDENTITY",
        "AMBIGUOUS",
        "NONE",
        "PROC_IDENTITY_NOT_AVAILABLE",
        "PROC_ENUMERATION_NOT_AVAILABLE",
        "DEPENDENCY_OR_SUBPROCESS_DEFECT",
        "DUPLICATE_OBSERVATION_ID",
        "SUPERVISOR_HANDLE_ALREADY_REGISTERED",
        "SUPERVISOR_HANDLE_REAP_FAILED",
        "SUPERVISOR_LAUNCH_FAILED",
        "SUPERVISOR_READINESS_BINDING_REJECTED",
        "SUPERVISOR_READINESS_REJECTED",
        "SUPERVISOR_STARTUP_CLEANUP_UNVERIFIED",
        "STARTUP_DESCENDANT_IDENTITY_CHANGED",
        "TIMEOUT",
    }
)
FAILURE_DETAIL_STAGES = frozenset(
    {
        "SUPERVISOR_HANDLE_REAP",
        "SUPERVISOR_STARTUP",
        "TIMEOUT_FIXTURE_CLEANUP",
        "TIMEOUT_FIXTURE_IDENTITY_CAPTURE",
        "TIMEOUT_TEST_BODY",
        "UNKNOWN",
    }
)
FAILURE_CAUSES = frozenset(
    {
        "SELECTED_TEST_NONPASS",
        "MANIFEST_SCHEMA_INVALID",
        "MANIFEST_CARDINALITY_INVALID",
        "MANIFEST_ORDER_INVALID",
        "MANIFEST_READ_INVALID",
        "MODULE_SET_INVALID",
        "WINDOWS_SUPPORT_VERDICT_INVALID",
        "DISCOVERY_BINDING_INVALID",
        "MANIFEST_GROUP_OVERLAP",
        "TIER_CONTRACT_INVALID",
        "DISCOVERY_OBJECT_INVALID",
        "TEST_MODULE_IMPORT_FAILED",
        "TEST_DISCOVERY_FAILED",
        "DUPLICATE_TEST_ID",
        "LEGACY_TEST_ID_INVALID",
        "LEGACY_TEST_SOURCE_MISSING",
        "LEGACY_TEST_SOURCE_INVALID",
        "LEGACY_TEST_CLASS_MISSING",
        "LEGACY_TEST_METHOD_MISSING",
        "LEGACY_TEST_DECORATED",
        "DISCOVERY_SET_MISMATCH",
        "MANIFEST_TEST_ID_MISSING",
        "UNSUPPORTED_PLATFORM",
        "REQUIRED_SHARED_TEST_NOT_SELECTED",
        "NONPASS_EVIDENCE_COUNT_MISMATCH",
        "UNEXPECTED_FAIL_CLOSED",
    }
)


class TierGateError(RuntimeError):
    """Fail-closed platform-tier validation error with an allowlisted cause."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _id_digest(test_ids: Iterable[str]) -> str:
    return _digest(sorted(test_ids))


def _require_string_list(
    manifest: Mapping[str, object], key: str, expected_count: int
) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TierGateError("MANIFEST_SCHEMA_INVALID")
    result = tuple(value)
    if len(result) != expected_count or len(set(result)) != expected_count:
        raise TierGateError("MANIFEST_CARDINALITY_INVALID")
    if result != tuple(sorted(result)):
        raise TierGateError("MANIFEST_ORDER_INVALID")
    return result


def _load_manifest(
    path: Path,
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    int,
    str,
]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TierGateError("MANIFEST_READ_INVALID") from exc
    if not isinstance(manifest, dict) or frozenset(manifest) != ROOT_KEYS:
        raise TierGateError("MANIFEST_SCHEMA_INVALID")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TierGateError("MANIFEST_SCHEMA_INVALID")
    if tuple(manifest.get("module_order", ())) != EXACT_MODULE_ORDER:
        raise TierGateError("MODULE_SET_INVALID")
    if manifest.get("windows_native_support_verdict") != WINDOWS_NATIVE_VERDICT:
        raise TierGateError("WINDOWS_SUPPORT_VERDICT_INVALID")

    expected_discovered_count = manifest.get("expected_discovered_test_count")
    expected_discovered_digest = manifest.get("expected_discovered_test_ids_digest")
    if (
        not isinstance(expected_discovered_count, int)
        or expected_discovered_count <= 0
        or not isinstance(expected_discovered_digest, str)
        or len(expected_discovered_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_discovered_digest)
    ):
        raise TierGateError("DISCOVERY_BINDING_INVALID")
    required_shared_ids = _require_string_list(
        manifest, "required_shared_test_ids", expected_count=16
    )

    legacy_ids = _require_string_list(
        manifest, "legacy_native_win32_test_ids", expected_count=23
    )
    windows_public_legacy_ids = _require_string_list(
        manifest,
        "windows_native_public_legacy_expectation_test_ids",
        expected_count=5,
    )
    linux_only_ids = _require_string_list(
        manifest, "windows_portable_linux_only_exclusions", expected_count=9
    )
    groups = (
        set(required_shared_ids),
        set(legacy_ids),
        set(windows_public_legacy_ids),
        set(linux_only_ids),
    )
    if any(
        groups[left].intersection(groups[right])
        for left in range(4)
        for right in range(left + 1, 4)
    ):
        raise TierGateError("MANIFEST_GROUP_OVERLAP")

    tiers = manifest.get("tiers")
    expected_tiers = {
        LINUX_TIER: {
            "os_name": "posix",
            "sys_platform": "linux",
            "excluded_groups": ["legacy_native_win32_test_ids"],
        },
        WINDOWS_TIER: {
            "os_name": "nt",
            "sys_platform": "win32",
            "excluded_groups": [
                "legacy_native_win32_test_ids",
                "windows_native_public_legacy_expectation_test_ids",
                "windows_portable_linux_only_exclusions",
            ],
        },
    }
    if tiers != expected_tiers:
        raise TierGateError("TIER_CONTRACT_INVALID")
    return (
        manifest,
        required_shared_ids,
        legacy_ids,
        windows_public_legacy_ids,
        linux_only_ids,
        expected_discovered_count,
        expected_discovered_digest,
    )


def _flatten(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        elif isinstance(item, unittest.TestCase):
            yield item
        else:
            raise TierGateError("DISCOVERY_OBJECT_INVALID")


def _load_exact_modules(
) -> tuple[tuple[ModuleType, ...], tuple[unittest.TestCase, ...]]:
    repository_root = str(REPOSITORY_ROOT)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    modules: list[ModuleType] = []
    tests: list[unittest.TestCase] = []
    loader = unittest.TestLoader()
    for module_name in EXACT_MODULE_ORDER:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise TierGateError("TEST_MODULE_IMPORT_FAILED") from exc
        modules.append(module)
        tests.extend(_flatten(loader.loadTestsFromModule(module)))
    if loader.errors:
        raise TierGateError("TEST_DISCOVERY_FAILED")
    test_ids = [test.id() for test in tests]
    if len(test_ids) != len(set(test_ids)):
        raise TierGateError("DUPLICATE_TEST_ID")
    return tuple(modules), tuple(tests)


def _decorator_name(node: ast.expr) -> str:
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return "UNKNOWN"


def _prove_excluded_tests_are_executable(
    modules: Sequence[ModuleType], excluded_ids: Sequence[str]
) -> tuple[int, int]:
    modules_by_name = {module.__name__: module for module in modules}
    trees: dict[str, ast.Module] = {}
    static_locations: set[str] = set()
    runtime_locations: set[str] = set()
    for test_id in excluded_ids:
        try:
            module_name, class_name, method_name = test_id.rsplit(".", 2)
        except ValueError as exc:
            raise TierGateError("LEGACY_TEST_ID_INVALID") from exc
        module = modules_by_name.get(module_name)
        if module is None or not module.__file__:
            raise TierGateError("LEGACY_TEST_SOURCE_MISSING")
        if module_name not in trees:
            try:
                trees[module_name] = ast.parse(
                    Path(module.__file__).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise TierGateError("LEGACY_TEST_SOURCE_INVALID") from exc
        class_node = next(
            (
                node
                for node in trees[module_name].body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if class_node is None:
            raise TierGateError("LEGACY_TEST_CLASS_MISSING")
        method_node = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method_name
            ),
            None,
        )
        if method_node is None:
            raise TierGateError("LEGACY_TEST_METHOD_MISSING")
        for owner, decorators in (
            (f"{module_name}.{class_name}", class_node.decorator_list),
            (test_id, method_node.decorator_list),
        ):
            for decorator in decorators:
                if _decorator_name(decorator) in SKIP_OR_XFAIL_DECORATORS:
                    static_locations.add(owner)

        test_class = getattr(module, class_name, None)
        if not isinstance(test_class, type):
            raise TierGateError("LEGACY_TEST_CLASS_MISSING")
        method = getattr(test_class, method_name, None)
        if method is None:
            raise TierGateError("LEGACY_TEST_METHOD_MISSING")
        for owner, target in (
            (f"{module_name}.{class_name}", test_class),
            (test_id, method),
        ):
            if bool(getattr(target, "__unittest_skip__", False)) or bool(
                getattr(target, "__unittest_expecting_failure__", False)
            ):
                runtime_locations.add(owner)

    decorator_count = len(static_locations)
    runtime_flag_count = len(runtime_locations)
    if decorator_count or runtime_flag_count:
        raise TierGateError("LEGACY_TEST_DECORATED")
    return decorator_count, runtime_flag_count


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["report_digest"] = _digest(payload)
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _classify_nonpass_detail(raw_detail: str) -> tuple[str, str, str]:
    """Project raw unittest detail into closed class, reason, and stage enums."""

    class_markers = (
        ("LinuxSupervisorUnavailable", "LINUX_SUPERVISOR_ERROR"),
        ("ImmutableArtifactError", "LINUX_SUPERVISOR_ERROR"),
        ("EventContractError", "LINUX_SUPERVISOR_ERROR"),
        ("LinuxSupervisorError", "LINUX_SUPERVISOR_ERROR"),
        ("subprocess.TimeoutExpired", "SUBPROCESS_TIMEOUT_EXPIRED"),
        ("TimeoutExpired", "SUBPROCESS_TIMEOUT_EXPIRED"),
        ("json.decoder.JSONDecodeError", "JSON_DECODE_ERROR"),
        ("JSONDecodeError", "JSON_DECODE_ERROR"),
        ("ChildProcessError", "CHILD_PROCESS_ERROR"),
        ("ProcessLookupError", "PROCESS_LOOKUP_ERROR"),
        ("FileNotFoundError", "FILE_NOT_FOUND_ERROR"),
        ("PermissionError", "PERMISSION_ERROR"),
        ("BrokenPipeError", "BROKEN_PIPE_ERROR"),
        ("TimeoutError", "TIMEOUT_ERROR"),
        ("AssertionError", "ASSERTION_ERROR"),
        ("AttributeError", "ATTRIBUTE_ERROR"),
        ("NotImplementedError", "NOT_IMPLEMENTED_ERROR"),
        ("UnboundLocalError", "NAME_ERROR"),
        ("NameError", "NAME_ERROR"),
        ("IndexError", "INDEX_ERROR"),
        ("KeyError", "KEY_ERROR"),
        ("RecursionError", "RECURSION_ERROR"),
        ("StopIteration", "STOP_ITERATION"),
        ("EOFError", "EOF_ERROR"),
        ("UnicodeError", "UNICODE_ERROR"),
        ("SamplerFailure", "SAMPLER_FAILURE"),
        ("DuplicateObservationError", "OBSERVER_ERROR"),
        ("ObserverError", "OBSERVER_ERROR"),
        ("TypeError", "TYPE_ERROR"),
        ("ValueError", "VALUE_ERROR"),
        ("RuntimeError", "RUNTIME_ERROR"),
        ("OSError", "OS_ERROR"),
    )
    terminal_class = "UNCLASSIFIED_NONPASS"
    terminal_message = ""
    lines = raw_detail.rstrip().splitlines()
    terminal_index: int | None = None
    traceback_start = next(
        (
            index
            for index in range(len(lines) - 1, -1, -1)
            if lines[index] == "Traceback (most recent call last):"
        ),
        -1,
    )
    for index in range(traceback_start + 1, len(lines)):
        line = lines[index]
        for marker, enum in class_markers:
            match = re.fullmatch(
                rf"(?:[A-Za-z_][A-Za-z0-9_.]*\.)?{re.escape(marker)}(?:: ?(.*))?",
                line,
            )
            if match is not None:
                terminal_class = enum
                terminal_message = match.group(1) or ""
                terminal_index = index
                break
        if terminal_class != "UNCLASSIFIED_NONPASS":
            break
    if terminal_class == "UNCLASSIFIED_NONPASS":
        return terminal_class, "NONE", "UNKNOWN"
    allowed_reasons = FAILURE_DETAIL_REASONS - {"NONE", "AMBIGUOUS"}
    reason = "NONE"
    if terminal_class in {
        "LINUX_SUPERVISOR_ERROR",
        "OBSERVER_ERROR",
        "SAMPLER_FAILURE",
    }:
        reason = terminal_message if terminal_message in allowed_reasons else "NONE"
    elif terminal_class == "ASSERTION_ERROR" and terminal_message.startswith(
        "LINUX_TIMEOUT_FIXTURE_CLEANUP_FAILED:"
    ):
        raw_reasons = terminal_message.partition(":")[2].split(",")
        if raw_reasons and all(item in allowed_reasons for item in raw_reasons):
            reason = raw_reasons[0] if len(raw_reasons) == 1 else "AMBIGUOUS"
        else:
            reason = "AMBIGUOUS"
    stage = "UNKNOWN"
    assert terminal_index is not None
    stage_functions = {
        "_bounded_reap_supervisor_handle": "SUPERVISOR_HANDLE_REAP",
        "_capture_linux_timeout_fixture_identity": "TIMEOUT_FIXTURE_IDENTITY_CAPTURE",
        "_cleanup_linux_timeout_fixture": "TIMEOUT_FIXTURE_CLEANUP",
        "_assert_linux_supervised_timeout_contract": "TIMEOUT_TEST_BODY",
        "test_timeout_preserves_phase_counter_and_cancels_process_tree": "TIMEOUT_TEST_BODY",
        "start_linux_observation": "SUPERVISOR_STARTUP",
    }
    for line in reversed(lines[max(0, traceback_start + 1) : terminal_index]):
        frame = re.fullmatch(
            r'  File ".+", line [0-9]+, in ([A-Za-z_][A-Za-z0-9_]*)', line
        )
        if frame is not None and frame.group(1) in stage_functions:
            stage = stage_functions[frame.group(1)]
            break
    return terminal_class, reason, stage


def _nonpass_records(result: unittest.TestResult) -> list[dict[str, str]]:
    """Digest raw unittest detail immediately; never return identifiers or traces."""

    records: list[dict[str, str]] = []
    channels: tuple[tuple[str, Iterable[Any]], ...] = (
        ("FAILURE", result.failures),
        ("ERROR", result.errors),
        ("SKIPPED", result.skipped),
        ("EXPECTED_FAILURE", result.expectedFailures),
        (
            "UNEXPECTED_SUCCESS",
            ((test, "UNEXPECTED_SUCCESS") for test in result.unexpectedSuccesses),
        ),
    )
    for outcome, entries in channels:
        for test, raw_detail in entries:
            test_id = test.id()
            if not isinstance(test_id, str) or not isinstance(raw_detail, str):
                raise TierGateError("NONPASS_EVIDENCE_COUNT_MISMATCH")
            detail_class, detail_reason, detail_stage = _classify_nonpass_detail(
                raw_detail
            )
            records.append(
                {
                    "outcome_enum": outcome,
                    "failed_test_id_sha256": _sha256_text(test_id),
                    "detail_sha256": _sha256_text(raw_detail),
                    "detail_class_enum": detail_class,
                    "detail_reason_enum": detail_reason,
                    "detail_stage_enum": detail_stage,
                }
            )
    return sorted(
        records,
        key=lambda record: (
            record["outcome_enum"],
            record["failed_test_id_sha256"],
            record["detail_sha256"],
        ),
    )


def _failure_payload(
    *,
    phase: str,
    cause: str,
    platform: str,
    tier: str,
    counts: Mapping[str, int],
    records: Sequence[Mapping[str, str]],
    gate_detail: str,
) -> dict[str, object]:
    safe_cause = cause if cause in FAILURE_CAUSES else "UNEXPECTED_FAIL_CLOSED"
    payload: dict[str, object] = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "verdict": "FAIL",
        "phase_enum": phase,
        "cause_enum": safe_cause,
        "platform_enum": platform,
        "tier_enum": tier,
        "failure_count": int(counts.get("failure_count", 0)),
        "error_count": int(counts.get("error_count", 0)),
        "skipped_count": int(counts.get("skipped_count", 0)),
        "expected_failure_count": int(counts.get("expected_failure_count", 0)),
        "unexpected_success_count": int(counts.get("unexpected_success_count", 0)),
        "diagnostic_record_count": len(records),
        "diagnostic_records": [dict(record) for record in records],
        "gate_detail_sha256": _sha256_text(gate_detail),
        "raw_detail_persisted": False,
    }
    return _sealed(payload)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_failure_evidence_directory(directory: Path) -> None:
    """Verify the exact sanitized failure envelope without rendering its contents."""

    if not directory.is_dir() or directory.is_symlink():
        raise TierGateError("FAILURE_EVIDENCE_DIRECTORY_INVALID")
    entries = tuple(directory.iterdir())
    names = {path.name for path in entries if path.is_file()}
    allowed_sets = (
        {FAILURE_FILENAME},
        {
            FAILURE_FILENAME,
            "linux-shared-regression.json",
            "windows-native-support.json",
        },
    )
    if (
        names not in allowed_sets
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise TierGateError("FAILURE_EVIDENCE_FILE_SET_INVALID")
    try:
        payload = json.loads((directory / FAILURE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TierGateError("FAILURE_EVIDENCE_READ_INVALID") from exc
    expected_keys = {
        "schema_version",
        "verdict",
        "phase_enum",
        "cause_enum",
        "platform_enum",
        "tier_enum",
        "failure_count",
        "error_count",
        "skipped_count",
        "expected_failure_count",
        "unexpected_success_count",
        "diagnostic_record_count",
        "diagnostic_records",
        "gate_detail_sha256",
        "raw_detail_persisted",
        "report_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise TierGateError("FAILURE_EVIDENCE_SCHEMA_INVALID")
    if (
        payload.get("schema_version") != FAILURE_SCHEMA_VERSION
        or payload.get("verdict") != "FAIL"
        or payload.get("phase_enum") not in {"PRE_TEST_GATE", "TEST_EXECUTION"}
        or payload.get("cause_enum") not in FAILURE_CAUSES
        or payload.get("platform_enum") not in {"LINUX", "WINDOWS_NT", "UNKNOWN"}
        or payload.get("tier_enum") not in {LINUX_TIER, WINDOWS_TIER, "UNKNOWN"}
        or payload.get("raw_detail_persisted") is not False
        or not _is_sha256(payload.get("gate_detail_sha256"))
        or not _is_sha256(payload.get("report_digest"))
    ):
        raise TierGateError("FAILURE_EVIDENCE_SCHEMA_INVALID")
    count_keys = (
        "failure_count",
        "error_count",
        "skipped_count",
        "expected_failure_count",
        "unexpected_success_count",
        "diagnostic_record_count",
    )
    if any(type(payload.get(key)) is not int or payload[key] < 0 for key in count_keys):
        raise TierGateError("FAILURE_EVIDENCE_COUNT_INVALID")
    records = payload.get("diagnostic_records")
    if not isinstance(records, list) or len(records) != payload["diagnostic_record_count"]:
        raise TierGateError("FAILURE_EVIDENCE_COUNT_INVALID")
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "outcome_enum",
                "failed_test_id_sha256",
                "detail_sha256",
                "detail_class_enum",
                "detail_reason_enum",
                "detail_stage_enum",
            }
            or record.get("outcome_enum") not in FAILURE_OUTCOMES
            or not _is_sha256(record.get("failed_test_id_sha256"))
            or not _is_sha256(record.get("detail_sha256"))
            or record.get("detail_class_enum") not in FAILURE_DETAIL_CLASSES
            or record.get("detail_reason_enum") not in FAILURE_DETAIL_REASONS
            or record.get("detail_stage_enum") not in FAILURE_DETAIL_STAGES
        ):
            raise TierGateError("FAILURE_EVIDENCE_RECORD_INVALID")
        outcome = record["outcome_enum"]
        detail_class = record["detail_class_enum"]
        detail_reason = record["detail_reason_enum"]
        detail_stage = record["detail_stage_enum"]
        if (
            (
                detail_class == "UNCLASSIFIED_NONPASS"
                and (detail_reason, detail_stage) != ("NONE", "UNKNOWN")
            )
            or (outcome == "FAILURE" and detail_class != "ASSERTION_ERROR")
            or (
                outcome == "ERROR"
                and detail_class == "ASSERTION_ERROR"
            )
            or (
                outcome in {"SKIPPED", "UNEXPECTED_SUCCESS"}
                and (detail_class, detail_reason, detail_stage)
                != ("UNCLASSIFIED_NONPASS", "NONE", "UNKNOWN")
            )
            or (
                outcome == "EXPECTED_FAILURE"
                and detail_class == "UNCLASSIFIED_NONPASS"
            )
        ):
            raise TierGateError("FAILURE_EVIDENCE_RECORD_INVALID")
    expected_record_count = sum(int(payload[key]) for key in count_keys[:-1])
    if payload["phase_enum"] == "TEST_EXECUTION":
        platform_tier_pair = (
            payload["platform_enum"],
            payload["tier_enum"],
        )
        if (
            payload["cause_enum"] != "SELECTED_TEST_NONPASS"
            or platform_tier_pair
            not in {("LINUX", LINUX_TIER), ("WINDOWS_NT", WINDOWS_TIER)}
            or payload["diagnostic_record_count"] <= 0
            or expected_record_count != payload["diagnostic_record_count"]
        ):
            raise TierGateError("FAILURE_EVIDENCE_COUNT_INVALID")
        expected_outcome_counts = {
            "FAILURE": payload["failure_count"],
            "ERROR": payload["error_count"],
            "SKIPPED": payload["skipped_count"],
            "EXPECTED_FAILURE": payload["expected_failure_count"],
            "UNEXPECTED_SUCCESS": payload["unexpected_success_count"],
        }
        actual_outcome_counts = {
            outcome: sum(record["outcome_enum"] == outcome for record in records)
            for outcome in FAILURE_OUTCOMES
        }
        if actual_outcome_counts != expected_outcome_counts:
            raise TierGateError("FAILURE_EVIDENCE_COUNT_INVALID")
    elif (
        payload["cause_enum"] == "SELECTED_TEST_NONPASS"
        or payload["platform_enum"] != "UNKNOWN"
        or payload["tier_enum"] != "UNKNOWN"
        or expected_record_count != 0
        or payload["diagnostic_record_count"] != 0
    ):
        raise TierGateError("FAILURE_EVIDENCE_COUNT_INVALID")
    unsealed = dict(payload)
    report_digest = unsealed.pop("report_digest")
    if _digest(unsealed) != report_digest:
        raise TierGateError("FAILURE_EVIDENCE_DIGEST_MISMATCH")


def failure_evidence_verifier_main(directory: Path) -> int:
    """Return a fixed verdict without rendering rejected evidence or exceptions."""

    try:
        verify_failure_evidence_directory(directory)
    except BaseException:
        print("failure_evidence_verdict=REJECTED")
        return 2
    print("failure_evidence_verdict=ACCEPTED")
    return 0


def _publish_gate_failure(output: Path, cause: str, gate_detail: str) -> None:
    payload = _failure_payload(
        phase="PRE_TEST_GATE",
        cause=cause,
        platform="UNKNOWN",
        tier="UNKNOWN",
        counts={},
        records=(),
        gate_detail=gate_detail,
    )
    _atomic_write(output / FAILURE_FILENAME, payload)


def _run(manifest_path: Path, output: Path) -> int:
    (
        manifest,
        required_shared_ids,
        legacy_ids,
        windows_public_legacy_ids,
        linux_only_ids,
        expected_discovered_count,
        expected_discovered_digest,
    ) = _load_manifest(manifest_path)
    modules, discovered_tests = _load_exact_modules()
    discovered_by_id = {test.id(): test for test in discovered_tests}
    discovered_digest = _id_digest(discovered_by_id)
    if (
        len(discovered_by_id) != expected_discovered_count
        or discovered_digest != expected_discovered_digest
    ):
        raise TierGateError("DISCOVERY_SET_MISMATCH")
    required_manifest_ids = (
        set(required_shared_ids)
        .union(legacy_ids)
        .union(windows_public_legacy_ids)
        .union(linux_only_ids)
    )
    if not required_manifest_ids.issubset(discovered_by_id):
        raise TierGateError("MANIFEST_TEST_ID_MISSING")

    preserved_windows_ids = set(legacy_ids).union(windows_public_legacy_ids)
    decorator_count, runtime_flag_count = _prove_excluded_tests_are_executable(
        modules, tuple(sorted(preserved_windows_ids))
    )
    if os.name == "posix" and sys.platform == "linux":
        platform_enum = "LINUX"
        tier = LINUX_TIER
        excluded_ids = set(legacy_ids)
    elif os.name == "nt" and sys.platform == "win32":
        platform_enum = "WINDOWS_NT"
        tier = WINDOWS_TIER
        excluded_ids = (
            set(legacy_ids)
            .union(windows_public_legacy_ids)
            .union(linux_only_ids)
        )
    else:
        raise TierGateError("UNSUPPORTED_PLATFORM")

    selected_tests = tuple(
        discovered_by_id[test_id]
        for test_id in sorted(discovered_by_id)
        if test_id not in excluded_ids
    )
    selected_test_ids = {test.id() for test in selected_tests}
    if not set(required_shared_ids).issubset(selected_test_ids):
        raise TierGateError("REQUIRED_SHARED_TEST_NOT_SELECTED")
    result = unittest.TestResult()
    unittest.TestSuite(selected_tests).run(result)

    failure_count = len(result.failures)
    error_count = len(result.errors)
    skipped_count = len(result.skipped)
    expected_failure_count = len(result.expectedFailures)
    unexpected_success_count = len(result.unexpectedSuccesses)
    tests_run_count = int(result.testsRun)
    selected_count = len(selected_tests)
    nonpass_records = _nonpass_records(result)
    nonpass_count = (
        failure_count
        + error_count
        + skipped_count
        + expected_failure_count
        + unexpected_success_count
    )
    if len(nonpass_records) != nonpass_count:
        raise TierGateError("NONPASS_EVIDENCE_COUNT_MISMATCH")
    passed = (
        tests_run_count == selected_count
        and failure_count == 0
        and error_count == 0
        and skipped_count == 0
        and expected_failure_count == 0
        and unexpected_success_count == 0
    )

    regression = _sealed(
        {
            "schema_version": SCHEMA_VERSION,
            "tier": tier,
            "platform": platform_enum,
            "verdict": "PASS" if passed else "FAIL",
            "module_count": len(modules),
            "discovered_test_count": len(discovered_tests),
            "selected_test_count": selected_count,
            "excluded_test_count": len(excluded_ids),
            "tests_run_count": tests_run_count,
            "failure_count": failure_count,
            "error_count": error_count,
            "skipped_count": skipped_count,
            "expected_failure_count": expected_failure_count,
            "unexpected_success_count": unexpected_success_count,
            "manifest_digest": _digest(manifest),
            "discovered_test_ids_digest": discovered_digest,
            "required_shared_test_count": len(required_shared_ids),
            "required_shared_test_ids_digest": _id_digest(required_shared_ids),
            "selected_test_ids_digest": _id_digest(
                test.id() for test in selected_tests
            ),
            "excluded_test_ids_digest": _id_digest(excluded_ids),
        }
    )
    windows_native = _sealed(
        {
            "schema_version": SCHEMA_VERSION,
            "support_verdict": WINDOWS_NATIVE_VERDICT,
            "gate_membership": "SEPARATE_LEGACY_DIAGNOSTIC",
            "automatic_selection_state": "DISABLED",
            "required_linux_gate_state": "EXCLUDED_BY_EXACT_ID",
            "diagnostic_execution_state": "NOT_EXECUTED_IN_LINUX_ACCEPTANCE",
            "legacy_test_count": len(legacy_ids),
            "legacy_test_ids_digest": _id_digest(legacy_ids),
            "windows_native_public_legacy_expectation_test_count": len(
                windows_public_legacy_ids
            ),
            "windows_native_public_legacy_expectation_test_ids_digest": _id_digest(
                windows_public_legacy_ids
            ),
            "preserved_windows_native_test_count": len(legacy_ids)
            + len(windows_public_legacy_ids),
            "public_windows_guard_state": "FAIL_CLOSED_TEST_SELECTED_BY_EXACT_ID",
            "skip_or_xfail_decorator_count": decorator_count,
            "runtime_skip_or_xfail_flag_count": runtime_flag_count,
        }
    )
    _atomic_write(output / "linux-shared-regression.json", regression)
    _atomic_write(output / "windows-native-support.json", windows_native)
    if not passed:
        failure = _failure_payload(
            phase="TEST_EXECUTION",
            cause="SELECTED_TEST_NONPASS",
            platform=platform_enum,
            tier=tier,
            counts={
                "failure_count": failure_count,
                "error_count": error_count,
                "skipped_count": skipped_count,
                "expected_failure_count": expected_failure_count,
                "unexpected_success_count": unexpected_success_count,
            },
            records=nonpass_records,
            gate_detail="SELECTED_TEST_NONPASS",
        )
        _atomic_write(output / FAILURE_FILENAME, failure)
    print(f"platform_tier_status={'PASS' if passed else 'FAIL'}")
    print(f"platform_tier={tier}")
    print(f"tests_run_count={tests_run_count}")
    if not passed:
        print(f"nonpass_case_count={len(nonpass_records)}")
        print(
            "nonpass_case_sha256="
            + ",".join(
                record["failed_test_id_sha256"] for record in nonpass_records
            )
        )
    return 0 if passed else 1


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact D42C portable test tier"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("qlib_linux_acceptance_test_tiers.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return _run(args.manifest.resolve(), args.output.resolve())
    except TierGateError as exc:
        try:
            _publish_gate_failure(args.output.resolve(), exc.cause, exc.cause)
        except Exception:
            pass
        print("platform_tier_status=FAIL")
        print(f"cause={exc.cause if exc.cause in FAILURE_CAUSES else 'UNEXPECTED_FAIL_CLOSED'}")
        return 2
    except Exception as exc:
        try:
            _publish_gate_failure(
                args.output.resolve(),
                "UNEXPECTED_FAIL_CLOSED",
                type(exc).__name__,
            )
        except Exception:
            pass
        print("platform_tier_status=FAIL")
        print("cause=UNEXPECTED_FAIL_CLOSED")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
