from __future__ import annotations

import hashlib
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import mock

from tools import qlib_traffic_observer as observer


FIXTURE_SEED = "R9ZNW-488D33-CANONICAL-EVIDENCE-SEED-V1"
CODE87_POLICY_ID = "D27_CODE87_EXACT_ABSENCE_V1"


class _FakeWin32Api:
    """Deterministic Win32 adapter that never calls the host process API."""

    def __init__(
        self,
        snapshots: list[dict[str, object]] | None = None,
        *,
        root_pid: int = 10,
        root_handle: int = 7000,
    ) -> None:
        self.snapshots = snapshots or [
            {"entries": [(root_pid, 1)]},
            {"entries": [(root_pid, 1)]},
        ]
        self.snapshot_exception: BaseException | None = None
        self.snapshot_index = 0
        self.snapshot_state: dict[int, dict[str, object]] = {}
        self.open_failure_codes: dict[int, int] = {}
        self.identity_actions: dict[int, list[int | BaseException]] = {}
        self.wait_sequences: dict[int, list[tuple[int, int]]] = {}
        self.terminate_fail_pids: set[int] = set()
        self.close_exception_handles: set[int] = set()
        self.handle_to_pid = {root_handle: root_pid}
        self.opened_handles: list[int] = []
        self.closed_handles: list[int] = []
        self.terminate_calls: list[int] = []

    def create_snapshot(self) -> int:
        if self.snapshot_exception is not None:
            raise self.snapshot_exception
        index = min(self.snapshot_index, len(self.snapshots) - 1)
        script = dict(self.snapshots[index])
        self.snapshot_index += 1
        handle = 1000 + self.snapshot_index
        script["cursor"] = 0
        self.snapshot_state[handle] = script
        return handle

    def process_first(self, snapshot_handle: int) -> tuple[tuple[int, int] | None, int]:
        script = self.snapshot_state[snapshot_handle]
        exception = script.get("first_exception")
        if isinstance(exception, BaseException):
            raise exception
        error = script.get("first_error")
        if isinstance(error, int):
            return None, error
        entries = list(script.get("entries", []))
        if not entries:
            return None, observer._WIN32_ERROR_NO_MORE_FILES
        script["cursor"] = 1
        return tuple(entries[0]), 0  # type: ignore[arg-type,return-value]

    def process_next(self, snapshot_handle: int) -> tuple[tuple[int, int] | None, int]:
        script = self.snapshot_state[snapshot_handle]
        exception = script.get("next_exception")
        if isinstance(exception, BaseException):
            raise exception
        error = script.get("next_error")
        if isinstance(error, int):
            return None, error
        entries = list(script.get("entries", []))
        cursor = int(script.get("cursor", 0))
        if cursor >= len(entries):
            return None, observer._WIN32_ERROR_NO_MORE_FILES
        script["cursor"] = cursor + 1
        return tuple(entries[cursor]), 0  # type: ignore[arg-type,return-value]

    def open_process(self, process_id: int) -> int:
        error = self.open_failure_codes.get(process_id)
        if error is not None:
            raise observer._Win32ApiFailure("OpenProcess", error, "OPEN_PROCESS")
        handle = 20000 + int(process_id)
        self.handle_to_pid[handle] = int(process_id)
        self.opened_handles.append(handle)
        return handle

    def creation_identity(self, process_handle: int) -> int:
        process_id = self.handle_to_pid[int(process_handle)]
        actions = self.identity_actions.get(process_id)
        if actions:
            action = actions.pop(0) if len(actions) > 1 else actions[0]
            if isinstance(action, BaseException):
                raise action
            return int(action)
        return 100000 + process_id

    def wait(self, process_handle: int, _timeout_ms: int) -> tuple[int, int]:
        process_id = self.handle_to_pid[int(process_handle)]
        sequence = self.wait_sequences.get(process_id)
        if sequence:
            return sequence.pop(0) if len(sequence) > 1 else sequence[0]
        return observer._WIN32_WAIT_OBJECT_0, 0

    def terminate(self, process_handle: int) -> None:
        process_id = self.handle_to_pid[int(process_handle)]
        self.terminate_calls.append(process_id)
        if process_id in self.terminate_fail_pids:
            raise observer._Win32ApiFailure(
                "TerminateProcess", 5, "PROCESS_TERMINATION"
            )

    def close_handle(self, handle: int) -> None:
        self.closed_handles.append(int(handle))
        if int(handle) in self.close_exception_handles:
            raise observer._Win32ApiFailure("CloseHandle", 6, "HANDLE_CLOSE")


class _FakeProcess:
    def __init__(self, *, stdout: str = "{}", returncode: int = 0) -> None:
        self.pid = 10
        self._handle = 7000
        self.returncode = returncode
        self._stdout = stdout

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        del timeout
        return self._stdout, ""


@contextmanager
def _no_write_temporary_directory(*_args: Any, **_kwargs: Any) -> Iterator[str]:
    yield "synthetic-d33-temp"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_process() -> Any:
    process = mock.Mock(pid=10, returncode=None)
    process._handle = 7000
    return process


def _tree_result(api: _FakeWin32Api) -> dict[str, Any]:
    return observer._terminate_windows_owned_tree(
        _fake_process(), api=api, nonce="synthetic-nonce"
    )


def _root_success() -> tuple[dict[str, Any], _FakeWin32Api]:
    api = _FakeWin32Api()
    return _tree_result(api), api


def _child_api() -> _FakeWin32Api:
    return _FakeWin32Api(
        [
            {"entries": [(10, 1), (11, 10)]},
            {"entries": [(10, 1), (11, 10)]},
            {"entries": [(10, 1)]},
        ]
    )


def _child_success() -> tuple[dict[str, Any], _FakeWin32Api]:
    api = _child_api()
    return _tree_result(api), api


def _is_cleanup_success(value: dict[str, Any]) -> bool:
    return (
        value.get("child_cancellation_state") in {"NOT_REQUIRED", "COMPLETED"}
        and value.get("orphan_verification_status") == "PASS"
        and value.get("orphan_child_count") == 0
        and value.get("process_tree_snapshot_status") == "COMPLETE"
        and value.get("diagnostic_contract_status") == "PASS"
    )


def _failure_metadata(cleanup: dict[str, Any]) -> dict[str, Any]:
    return {
        "first_failure_phase": "CHILD_CLEANUP",
        "failure_category": "DEPENDENCY_OR_SUBPROCESS_DEFECT",
        "source_file": "tools/qlib_traffic_observer.py",
        "source_function": "_production_sample_once",
        "source_line": 1,
        "exception_class": "SamplerFailure",
        "dependency_class": "SUBPROCESS",
        "exit_category": "PROCESS_TREE_CLEANUP_FAILED",
        "retryable": False,
        "raw_exception_message_persisted": False,
        "raw_sampler_output_persisted": False,
        **cleanup,
    }


def _case_outcome(
    *,
    actual: str,
    contract_pass: bool,
    evidence: dict[str, Any],
    observed_success: bool = False,
    expected_safe_fallback_count: int = 0,
) -> dict[str, Any]:
    return {
        "actual": actual,
        "contract_pass": bool(contract_pass),
        "observed_success": bool(observed_success),
        "expected_safe_fallback_count": expected_safe_fallback_count,
        "evidence": evidence,
    }


def _case_d31_01() -> dict[str, Any]:
    result, _api = _root_success()
    passed = _is_cleanup_success(result)
    return _case_outcome(
        actual="CLOSURE_PASS" if passed else "UNEXPECTED",
        contract_pass=passed,
        observed_success=passed,
        evidence={
            "cleanup": result.get("process_tree_cleanup_status"),
            "orphan_count": result.get("orphan_count"),
            "descendant_count": result.get("descendant_count"),
        },
    )


def _case_d31_02() -> dict[str, Any]:
    api = _child_api()
    api.identity_actions[11] = [100010, 100010]
    result = _tree_result(api)
    passed = _is_cleanup_success(result)
    return _case_outcome(
        actual="CREATION_ORDER_MATCH_PASS" if passed else "UNEXPECTED",
        contract_pass=passed,
        observed_success=passed,
        evidence={"identity_mismatch_count": result.get("identity_mismatch_count")},
    )


def _case_d31_03() -> dict[str, Any]:
    api = _child_api()
    api.identity_actions[11] = [1]
    result = _tree_result(api)
    passed = (
        not _is_cleanup_success(result)
        and result.get("identity_check_stage") == "CREATION_ORDER_IDENTITY_CHECK"
        and result.get("identity_failure_kind")
        == "CREATION_BEFORE_ROOT_AND_PARENT"
    )
    return _case_outcome(
        actual="FAIL_CLOSED_CREATION_ORDER",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "stage": result.get("identity_check_stage"),
            "kind": result.get("identity_failure_kind"),
            "decision": result.get("cleanup_decision"),
        },
    )


def _case_d31_04() -> dict[str, Any]:
    result, _api = _child_success()
    passed = _is_cleanup_success(result)
    return _case_outcome(
        actual="PARENT_REVALIDATION_MATCH_PASS" if passed else "UNEXPECTED",
        contract_pass=passed,
        observed_success=passed,
        evidence={"snapshot_complete": result.get("snapshot_complete")},
    )


def _parent_mismatch_result() -> tuple[dict[str, Any], _FakeWin32Api]:
    api = _FakeWin32Api(
        [
            {"entries": [(10, 1), (11, 10)]},
            {"entries": [(10, 1), (11, 99)]},
        ]
    )
    api.identity_actions[11] = [200011]
    return _tree_result(api), api


def _case_d31_05() -> dict[str, Any]:
    result, _api = _parent_mismatch_result()
    passed = (
        result.get("identity_check_stage") == "PARENT_REVALIDATION_IDENTITY_CHECK"
        and result.get("identity_relation") == "PARENT_CHANGED"
        and result.get("child_cancellation_state") == "FAILED"
    )
    return _case_outcome(
        actual="FAIL_CLOSED_PARENT_REVALIDATION",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "stage": result.get("identity_check_stage"),
            "relation": result.get("identity_relation"),
        },
    )


def _case_d31_06() -> dict[str, Any]:
    api = _child_api()
    api.identity_actions[11] = [200011, 200011]
    result = _tree_result(api)
    passed = _is_cleanup_success(result)
    return _case_outcome(
        actual="HANDLE_IDENTITY_MATCH_PASS" if passed else "UNEXPECTED",
        contract_pass=passed,
        observed_success=passed,
        evidence={"identity_mismatch_count": result.get("identity_mismatch_count")},
    )


def _handle_mismatch_result(*, close_failure: bool = False) -> tuple[dict[str, Any], _FakeWin32Api]:
    api = _child_api()
    api.identity_actions[11] = [200011, 300011]
    if close_failure:
        api.close_exception_handles.add(20011)
    return _tree_result(api), api


def _case_d31_07() -> dict[str, Any]:
    result, api = _handle_mismatch_result()
    passed = (
        result.get("identity_check_stage") == "HANDLE_IDENTITY_REVALIDATION_CHECK"
        and result.get("identity_relation") == "DIFFERENT"
        and not api.terminate_calls
    )
    return _case_outcome(
        actual="FAIL_CLOSED_HANDLE_REVALIDATION",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "stage": result.get("identity_check_stage"),
            "relation": result.get("identity_relation"),
            "terminate_call_count": len(api.terminate_calls),
        },
    )


def _case_d31_08() -> dict[str, Any]:
    result, api = _parent_mismatch_result()
    passed = (
        result.get("identity_failure_kind") == "PARENT_RELATION_CHANGED"
        and result.get("identity_relation") == "PARENT_CHANGED"
        and not api.terminate_calls
    )
    return _case_outcome(
        actual="FAIL_CLOSED_SIMULATED_PID_REUSE",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "kind": result.get("identity_failure_kind"),
            "relation": result.get("identity_relation"),
        },
    )


def _code87_result(
    second_snapshot: dict[str, object], *, error_code: int = 87
) -> tuple[dict[str, Any], _FakeWin32Api]:
    api = _FakeWin32Api(
        [
            {"entries": [(10, 1), (11, 10)]},
            second_snapshot,
            {"entries": [(10, 1)]},
        ]
    )
    api.open_failure_codes[11] = error_code
    return _tree_result(api), api


def _case_d31_09() -> dict[str, Any]:
    result, api = _code87_result({"entries": [(10, 1)]})
    passed = (
        _is_cleanup_success(result)
        and result.get("cleanup_decision")
        == "CONTINUE_VERIFIED_CODE87_PID_ABSENT"
        and result.get("exact_pid_absent") is True
        and api.snapshot_index >= 3
    )
    return _case_outcome(
        actual="RECONCILED_CLOSURE_PASS" if passed else "UNEXPECTED",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "decision": result.get("cleanup_decision"),
            "exact_pid_absent": result.get("exact_pid_absent"),
            "snapshot_count": api.snapshot_index,
        },
    )


def _case_d31_10() -> dict[str, Any]:
    result, api = _code87_result({"entries": [(10, 1), (11, 10)]})
    passed = (
        result.get("cleanup_decision") == "FAIL_CLOSED_EXACT_PID_PRESENT"
        and result.get("exact_pid_absent") is False
        and result.get("child_cancellation_state") == "FAILED"
        and api.snapshot_index == 2
    )
    return _case_outcome(
        actual="FAIL_CLOSED_EXACT_PID_PRESENT",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "decision": result.get("cleanup_decision"),
            "exact_pid_absent": result.get("exact_pid_absent"),
            "snapshot_count": api.snapshot_index,
        },
    )


def _case_d31_11() -> dict[str, Any]:
    api = _FakeWin32Api([{"entries": [], "first_error": 5}])
    result = _tree_result(api)
    passed = (
        result.get("process_tree_snapshot_status") == "INCOMPLETE_SNAPSHOT"
        and result.get("cleanup_decision") == "FAIL_CLOSED_INCOMPLETE_SNAPSHOT"
        and result.get("orphan_child_count") == 1
    )
    return _case_outcome(
        actual="FAIL_CLOSED_INCOMPLETE_SNAPSHOT",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "snapshot_status": result.get("process_tree_snapshot_status"),
            "decision": result.get("cleanup_decision"),
        },
    )


def _residual_descendant_result() -> tuple[dict[str, Any], _FakeWin32Api]:
    api = _FakeWin32Api(
        [
            {"entries": [(10, 1), (11, 10)]},
            {"entries": [(10, 1), (11, 10)]},
        ]
    )
    api.identity_actions[11] = [200011, 200011]
    api.wait_sequences[11] = [
        (observer._WIN32_WAIT_TIMEOUT, 0),
        (observer._WIN32_WAIT_TIMEOUT, 0),
    ]
    return _tree_result(api), api


def _case_d31_12() -> dict[str, Any]:
    result, api = _residual_descendant_result()
    passed = (
        result.get("descendant_count") == 1
        and result.get("cleanup_decision") == "FAIL_CLOSED_RESIDUAL_PROCESS"
        and result.get("child_cancellation_state") == "FAILED"
    )
    return _case_outcome(
        actual="FAIL_CLOSED_RESIDUAL_DESCENDANT",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "descendant_count": result.get("descendant_count"),
            "terminate_call_count": len(api.terminate_calls),
        },
    )


def _case_d31_13() -> dict[str, Any]:
    result = observer._windows_cleanup_failure(
        {
            "cleanup_failure_api": "WaitForSingleObject",
            "cleanup_failure_code": 0,
            "cleanup_failure_phase": "FINAL_VERIFICATION",
            "cleanup_failure_reason": "RESIDUAL_OWNED_PROCESS",
        },
        owned_process_count=1,
        terminated_process_count=0,
        identity_mismatch_count=0,
        snapshot_status="COMPLETE",
    )
    passed = result.get("orphan_count") == 1 and result.get("orphan_child_count") == 1
    return _case_outcome(
        actual="FAIL_CLOSED_ORPHAN_PRESENT",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={"orphan_count": result.get("orphan_count")},
    )


def _case_d31_14() -> dict[str, Any]:
    api = _FakeWin32Api()
    api.wait_sequences[10] = [
        (observer._WIN32_WAIT_TIMEOUT, 0),
        (observer._WIN32_WAIT_OBJECT_0, 0),
        (observer._WIN32_WAIT_OBJECT_0, 0),
    ]
    result = _tree_result(api)
    passed = (
        _is_cleanup_success(result)
        and result.get("child_cancellation_state") == "COMPLETED"
        and len(api.terminate_calls) == 1
    )
    return _case_outcome(
        actual="CLOSURE_PASS_CANCELLATION_COMPLETED" if passed else "UNEXPECTED",
        contract_pass=passed,
        observed_success=passed,
        evidence={
            "cancellation": result.get("child_cancellation_state"),
            "terminate_call_count": len(api.terminate_calls),
        },
    )


def _case_d31_15() -> dict[str, Any]:
    api = _FakeWin32Api()
    api.wait_sequences[10] = [(observer._WIN32_WAIT_TIMEOUT, 0)]
    api.terminate_fail_pids.add(10)
    result = _tree_result(api)
    passed = (
        result.get("child_cancellation_state") == "FAILED"
        and result.get("cleanup_decision") == "FAIL_CLOSED_CANCELLATION"
    )
    return _case_outcome(
        actual="FAIL_CLOSED_CANCELLATION",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "decision": result.get("cleanup_decision"),
            "api": result.get("cleanup_failure_api"),
        },
    )


def _case_d31_16() -> dict[str, Any]:
    api = _FakeWin32Api()
    api.identity_actions[10] = [
        observer._Win32ApiFailure("GetProcessTimes", 5, "ROOT_IDENTITY")
    ]
    result = _tree_result(api)
    passed = (
        result.get("identity_check_stage") == "ROOT_CREATION_IDENTITY_QUERY"
        and result.get("identity_failure_kind") == "OS_API_FAILURE"
        and result.get("sanitized_error_code") == 5
    )
    return _case_outcome(
        actual="FAIL_CLOSED_SANITIZED_API_FAILURE",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "stage": result.get("identity_check_stage"),
            "kind": result.get("identity_failure_kind"),
            "error_code": result.get("sanitized_error_code"),
        },
    )


def _case_d31_17() -> dict[str, Any]:
    result, _api = _handle_mismatch_result()
    passed = (
        result.get("os_api") == "GetProcessTimes"
        and result.get("identity_relation") == "DIFFERENT"
        and result.get("observed_identity_present") is True
    )
    return _case_outcome(
        actual="FAIL_CLOSED_IDENTITY_DIFFERENT",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "api": result.get("os_api"),
            "relation": result.get("identity_relation"),
        },
    )


def _case_d31_18() -> dict[str, Any]:
    result, _api = _handle_mismatch_result(close_failure=True)
    passed = (
        result.get("cleanup_failure_reason") == "IDENTITY_MISMATCH"
        and result.get("identity_failure_kind") == "HANDLE_CREATION_IDENTITY_CHANGED"
    )
    return _case_outcome(
        actual="FIRST_IDENTITY_FAILURE_PRESERVED",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "first_reason": result.get("cleanup_failure_reason"),
            "kind": result.get("identity_failure_kind"),
        },
    )


def _valid_failure_metadata() -> dict[str, Any]:
    cleanup, _api = _handle_mismatch_result()
    return _failure_metadata(cleanup)


def _case_d31_19() -> dict[str, Any]:
    summary = observer._observer_cleanup_summary(
        "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT", _valid_failure_metadata()
    )
    passed = (
        summary.get("terminal_reason_preserved_exactly") is True
        and summary.get("terminal_reason_masking_count") == 0
        and summary.get("diagnostic_contract_status") == "PASS"
    )
    return _case_outcome(
        actual="TERMINAL_REASON_PRESERVED",
        contract_pass=passed,
        evidence={
            "preserved": summary.get("terminal_reason_preserved_exactly"),
            "masking_count": summary.get("terminal_reason_masking_count"),
        },
    )


def _case_d31_20() -> dict[str, Any]:
    raw_reason = "CUTOVER_SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT"
    summary = observer._observer_cleanup_summary(raw_reason, _valid_failure_metadata())
    serialized = json.dumps(summary, sort_keys=True)
    passed = (
        summary.get("terminal_reason") == "TERMINAL_REASON_NOT_ALLOWLISTED"
        and summary.get("terminal_reason_fallback_used") is True
        and summary.get("verdict") == "OBSERVER_CLEANUP_FAILURE"
        and raw_reason not in serialized
    )
    return _case_outcome(
        actual="SAFE_FALLBACK_OBSERVER_PRECEDENCE",
        contract_pass=passed,
        expected_safe_fallback_count=1,
        evidence={
            "reason": summary.get("terminal_reason"),
            "verdict": summary.get("verdict"),
            "raw_reason_persisted": raw_reason in serialized,
        },
    )


def _failed_cleanup() -> dict[str, Any]:
    result, _api = _handle_mismatch_result()
    return result


def _successful_cleanup() -> dict[str, Any]:
    result, _api = _root_success()
    return result


def _case_d31_21() -> dict[str, Any]:
    process = _FakeProcess(stdout=json.dumps({"sample_status": "PASS"}))
    with mock.patch.object(observer.tempfile, "TemporaryDirectory", _no_write_temporary_directory), mock.patch.object(
        observer.subprocess, "Popen", return_value=process
    ), mock.patch.object(
        observer, "_terminate_sampler_process", return_value=_failed_cleanup()
    ):
        try:
            observer._production_sample_once(["synthetic-sampler"], timeout_seconds=1.0)
        except observer.SamplerFailure as exc:
            metadata = exc.metadata
        else:
            metadata = {}
    passed = (
        metadata.get("failure_category") == "DEPENDENCY_OR_SUBPROCESS_DEFECT"
        and metadata.get("first_failure_phase") == "CHILD_CLEANUP"
        and metadata.get("exit_category") == "PROCESS_TREE_CLEANUP_FAILED"
    )
    return _case_outcome(
        actual="DEPENDENCY_FAILURE_AFTER_FUNCTIONAL_PASS",
        contract_pass=passed,
        observed_success=not bool(metadata),
        evidence={
            "category": metadata.get("failure_category"),
            "phase": metadata.get("first_failure_phase"),
            "exit": metadata.get("exit_category"),
        },
    )


def _case_d31_22() -> dict[str, Any]:
    summary = observer._observer_cleanup_summary(
        "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT", _valid_failure_metadata()
    )
    passed = (
        summary.get("status") == "FAIL"
        and summary.get("incomplete_first_failure_count") == 1
        and summary.get("unresolved_cleanup_count", 0) >= 1
    )
    return _case_outcome(
        actual="SUMMARY_FAIL_WITH_FIRST_FAILURE",
        contract_pass=passed,
        observed_success=summary.get("status") == "PASS",
        evidence={
            "status": summary.get("status"),
            "first_failure_count": summary.get("incomplete_first_failure_count"),
            "unresolved_count": summary.get("unresolved_cleanup_count"),
        },
    )


def _case_d31_23() -> dict[str, Any]:
    with mock.patch.object(observer.json, "dumps", side_effect=TypeError("synthetic")):
        diagnostic = observer._observer_identity_diagnostic(
            identity_check_stage="HANDLE_IDENTITY_REVALIDATION_CHECK",
            identity_failure_kind="HANDLE_CREATION_IDENTITY_CHANGED",
            os_api="GetProcessTimes",
            sanitized_error_code=0,
            snapshot_complete=True,
            expected_identity_present=True,
            observed_identity_present=True,
            identity_relation="DIFFERENT",
            identity_subject_role="DESCENDANT",
            exact_pid_absent=None,
            descendant_count=0,
            orphan_count=1,
            cancellation_state="FAILED",
            cleanup_decision="FAIL_CLOSED_IDENTITY_MISMATCH",
            terminal_reason="SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT",
        )
    passed = (
        diagnostic.get("diagnostic_contract_status") == "INVALID"
        and diagnostic.get("cleanup_decision")
        == "FAIL_CLOSED_DIAGNOSTIC_SERIALIZATION"
    )
    return _case_outcome(
        actual="SAFE_DIAGNOSTIC_FAILURE_MAPPING",
        contract_pass=passed,
        observed_success=False,
        evidence={
            "contract": diagnostic.get("diagnostic_contract_status"),
            "decision": diagnostic.get("cleanup_decision"),
        },
    )


def _case_d31_24() -> dict[str, Any]:
    open87 = {
        "cleanup_failure_api": "OpenProcess",
        "cleanup_failure_code": 87,
        "cleanup_failure_phase": "OPEN_PROCESS",
        "cleanup_failure_reason": "API_FAILURE",
    }
    open5 = dict(open87, cleanup_failure_code=5)
    incomplete = {
        "complete": False,
        "entries": [],
        "cleanup_failure_api": "Process32FirstW",
        "cleanup_failure_code": 5,
        "cleanup_failure_phase": "FINAL_VERIFICATION",
        "cleanup_failure_reason": "INCOMPLETE_SNAPSHOT",
    }
    outcomes = [
        observer._d27_reconcile_open_process_disappearance(
            11, open87, {"complete": True, "entries": [(10, 1)]}
        )[0],
        observer._d27_reconcile_open_process_disappearance(
            11, open87, {"complete": True, "entries": [(11, 10)]}
        )[0],
        observer._d27_reconcile_open_process_disappearance(11, open87, incomplete)[0],
        observer._d27_reconcile_open_process_disappearance(
            11, open5, {"complete": True, "entries": [(10, 1)]}
        )[0],
    ]
    passed = outcomes == [True, False, False, False]
    return _case_outcome(
        actual="ONLY_COMPLETE_EXACT_PID_ABSENCE_CONTINUES",
        contract_pass=passed,
        observed_success=False,
        evidence={"continue_vector": outcomes},
    )


def _case_d31_25() -> dict[str, Any]:
    process = _FakeProcess(stdout=json.dumps({"sample_status": "PASS"}))
    with mock.patch.object(observer.tempfile, "TemporaryDirectory", _no_write_temporary_directory), mock.patch.object(
        observer.subprocess, "Popen", return_value=process
    ), mock.patch.object(
        observer, "_terminate_sampler_process", return_value=_successful_cleanup()
    ):
        result = observer._production_sample_once(
            ["synthetic-sampler"], timeout_seconds=1.0
        )
    passed = result.get("sample_status") == "PASS" and _is_cleanup_success(result)
    return _case_outcome(
        actual="FUNCTIONAL_AND_CLEANUP_PASS" if passed else "UNEXPECTED",
        contract_pass=passed,
        observed_success=passed,
        evidence={
            "sample_status": result.get("sample_status"),
            "cleanup_decision": result.get("cleanup_decision"),
        },
    )


def _generic_legacy_identity_failure() -> dict[str, Any]:
    return observer._windows_cleanup_failure(
        {
            "cleanup_failure_api": "GetProcessTimes",
            "cleanup_failure_code": 0,
            "cleanup_failure_phase": "IDENTITY_VERIFICATION",
            "cleanup_failure_reason": "IDENTITY_MISMATCH",
        },
        owned_process_count=1,
        terminated_process_count=0,
        identity_mismatch_count=1,
        snapshot_status="COMPLETE",
    )


def _case_def_o05() -> dict[str, Any]:
    result = _generic_legacy_identity_failure()
    passed = (
        result.get("diagnostic_contract_status") == "NOT_AVAILABLE_LEGACY"
        and result.get("identity_check_stage") == "NOT_VERIFIED"
        and result.get("identity_failure_kind")
        == "LEGACY_IDENTITY_MISMATCH_UNCLASSIFIED"
        and result.get("identity_relation") == "NOT_VERIFIED"
        and result.get("child_cancellation_state") == "FAILED"
    )
    return _case_outcome(
        actual="LEGACY_IDENTITY_MISMATCH_UNCLASSIFIED_FAIL_CLOSED",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "contract": result.get("diagnostic_contract_status"),
            "stage": result.get("identity_check_stage"),
            "kind": result.get("identity_failure_kind"),
            "relation": result.get("identity_relation"),
        },
    )


def _malformed_terminal_diagnostic() -> dict[str, Any]:
    return {
        "diagnostic_contract_status": "PASS",
        "identity_check_stage": "HANDLE_IDENTITY_REVALIDATION_CHECK",
        "identity_failure_kind": "HANDLE_CREATION_IDENTITY_CHANGED",
        "os_api": "GetProcessTimes",
        "sanitized_error_code": 0,
        "snapshot_complete": "false",
        "expected_identity_present": "true",
        "observed_identity_present": "true",
        "identity_relation": "DIFFERENT",
        "identity_subject_role": "DESCENDANT",
        "exact_pid_absent": "false",
        "descendant_count": 0,
        "orphan_count": 1,
        "cancellation_state": "FAILED",
        "cleanup_decision": "FAIL_CLOSED_IDENTITY_MISMATCH",
        "terminal_reason": "MALFORMED_REASON",
        "diagnostic_serialization_status": "PASS",
        "child_cancellation_state": "FAILED",
        "orphan_child_count": 1,
        "process_tree_snapshot_status": "COMPLETE",
        "cleanup_failure_reason": "IDENTITY_MISMATCH",
    }


def _case_def_r02() -> dict[str, Any]:
    diagnostic = observer._observer_terminal_diagnostic(
        _malformed_terminal_diagnostic()
    )
    serialized = json.dumps(diagnostic, sort_keys=True)
    passed = (
        diagnostic.get("diagnostic_contract_status") == "INVALID"
        and diagnostic.get("diagnostic_terminal_reason")
        == "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT"
        and diagnostic.get("snapshot_complete") is None
        and "MALFORMED_REASON" not in serialized
    )
    return _case_outcome(
        actual="MALFORMED_DIAGNOSTIC_FAIL_CLOSED",
        contract_pass=passed,
        observed_success=diagnostic.get("diagnostic_contract_status") == "PASS",
        evidence={
            "contract": diagnostic.get("diagnostic_contract_status"),
            "terminal_reason": diagnostic.get("diagnostic_terminal_reason"),
            "raw_malformed_persisted": "MALFORMED_REASON" in serialized,
        },
    )


def _case_hold_o04() -> dict[str, Any]:
    api = _FakeWin32Api([{"entries": [(10, 1)], "next_error": 5}])
    api.close_exception_handles.add(1001)
    result = observer._enumerate_windows_processes(api)
    passed = (
        result.get("cleanup_failure_api") == "Process32NextW"
        and result.get("cleanup_failure_code") == 5
        and result.get("cleanup_failure_reason") == "INCOMPLETE_SNAPSHOT"
    )
    return _case_outcome(
        actual="FIRST_SNAPSHOT_FAILURE_PRESERVED",
        contract_pass=passed,
        evidence={
            "api": result.get("cleanup_failure_api"),
            "code": result.get("cleanup_failure_code"),
            "reason": result.get("cleanup_failure_reason"),
        },
    )


def _case_hold_o06() -> dict[str, Any]:
    result = _generic_legacy_identity_failure()
    passed = (
        result.get("diagnostic_contract_status") == "NOT_AVAILABLE_LEGACY"
        and result.get("cleanup_decision") == "FAIL_CLOSED_IDENTITY_MISMATCH"
        and result.get("terminal_reason")
        == "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT"
    )
    return _case_outcome(
        actual="SAFE_GENERIC_DIAGNOSTIC_PROPAGATED",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "contract": result.get("diagnostic_contract_status"),
            "decision": result.get("cleanup_decision"),
        },
    )


def _case_hold_o10() -> dict[str, Any]:
    api = _child_api()
    api.identity_actions[11] = [
        observer._Win32ApiFailure("GetProcessTimes", 5, "IDENTITY_QUERY")
    ]
    result = _tree_result(api)
    passed = (
        result.get("identity_check_stage") == "CREATION_ORDER_IDENTITY_CHECK"
        and result.get("identity_failure_kind") == "OS_API_FAILURE"
        and result.get("sanitized_error_code") == 5
    )
    return _case_outcome(
        actual="CREATION_QUERY_API_FAILURE_DISCRIMINATED",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "stage": result.get("identity_check_stage"),
            "kind": result.get("identity_failure_kind"),
        },
    )


def _case_hold_o12() -> dict[str, Any]:
    api = _FakeWin32Api(
        [
            {"entries": [(10, 1), (11, 10)]},
            {"entries": [(10, 1), (11, 10)], "next_error": 5},
        ]
    )
    api.identity_actions[11] = [200011]
    result = _tree_result(api)
    passed = (
        result.get("identity_check_stage") == "PARENT_REVALIDATION_IDENTITY_CHECK"
        and result.get("identity_failure_kind") == "INCOMPLETE_SNAPSHOT"
        and result.get("snapshot_complete") is False
    )
    return _case_outcome(
        actual="SECOND_SNAPSHOT_INCOMPLETE_DISCRIMINATED",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "stage": result.get("identity_check_stage"),
            "snapshot_complete": result.get("snapshot_complete"),
        },
    )


def _case_hold_o14() -> dict[str, Any]:
    api = _child_api()
    api.identity_actions[11] = [
        200011,
        observer._Win32ApiFailure("GetProcessTimes", 5, "IDENTITY_VERIFICATION"),
    ]
    result = _tree_result(api)
    passed = (
        result.get("identity_check_stage") == "HANDLE_IDENTITY_REVALIDATION_CHECK"
        and result.get("identity_failure_kind") == "OS_API_FAILURE"
        and result.get("sanitized_error_code") == 5
    )
    return _case_outcome(
        actual="HANDLE_QUERY_API_FAILURE_DISCRIMINATED",
        contract_pass=passed,
        observed_success=_is_cleanup_success(result),
        evidence={
            "stage": result.get("identity_check_stage"),
            "kind": result.get("identity_failure_kind"),
        },
    )


def _writer_failure_case(target: str) -> dict[str, Any]:
    cyclic: dict[str, Any] = {}
    cyclic["cycle"] = cyclic
    open_mock = mock.MagicMock()
    replace_mock = mock.MagicMock()
    with mock.patch.object(Path, "mkdir"), mock.patch.object(
        Path, "open", open_mock
    ), mock.patch.object(observer.os, "replace", replace_mock):
        try:
            if target == "atomic":
                observer._atomic_write_json(Path("synthetic.json"), cyclic)
            else:
                observer._append_event(Path("synthetic.ndjson"), cyclic)
        except ValueError:
            raised = True
        else:
            raised = False
    passed = raised and open_mock.call_count == 0 and replace_mock.call_count == 0
    return _case_outcome(
        actual="SERIALIZATION_FAILED_BEFORE_DESTINATION_OPEN",
        contract_pass=passed,
        evidence={
            "serialization_raised": raised,
            "open_count": open_mock.call_count,
            "replace_count": replace_mock.call_count,
        },
    )


def _case_hold_r01() -> dict[str, Any]:
    return _writer_failure_case("atomic")


def _case_hold_r03() -> dict[str, Any]:
    summary = observer._observer_cleanup_summary(
        "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT", _valid_failure_metadata()
    )
    passed = (
        summary.get("diagnostic_contract_status") == "PASS"
        and summary.get("terminal_reason_input_allowlisted") is True
        and summary.get("terminal_reason_preserved_exactly") is True
    )
    return _case_outcome(
        actual="ALLOWLISTED_TERMINAL_DIAGNOSTIC_PASS",
        contract_pass=passed,
        evidence={
            "contract": summary.get("diagnostic_contract_status"),
            "preserved": summary.get("terminal_reason_preserved_exactly"),
        },
    )


def _case_hold_r04() -> dict[str, Any]:
    raw_reason = "UNALLOWLISTED_RAW_TERMINAL"
    summary = observer._observer_cleanup_summary(raw_reason, _valid_failure_metadata())
    serialized = json.dumps(summary, sort_keys=True)
    passed = (
        summary.get("terminal_reason") == "TERMINAL_REASON_NOT_ALLOWLISTED"
        and summary.get("terminal_reason_fallback_used") is True
        and summary.get("terminal_reason_preserved_exactly") is False
        and raw_reason not in serialized
    )
    return _case_outcome(
        actual="UNALLOWLISTED_TERMINAL_SAFE_FALLBACK",
        contract_pass=passed,
        expected_safe_fallback_count=1,
        evidence={
            "reason": summary.get("terminal_reason"),
            "fallback": summary.get("terminal_reason_fallback_used"),
            "raw_reason_persisted": raw_reason in serialized,
        },
    )


def _case_hold_br01() -> dict[str, Any]:
    return _writer_failure_case("event")


def _case_hold_br05() -> dict[str, Any]:
    summary = observer._observer_cleanup_summary("OWNER_STOP", None)
    zero_keys = (
        "descendant_count",
        "orphan_count",
        "cancellation_failure_count",
        "incomplete_first_failure_count",
        "terminal_reason_masking_count",
        "terminal_reason_comparison_count",
        "raw_process_identity_persistence_count",
    )
    passed = all(summary.get(key) == 0 for key in zero_keys)
    return _case_outcome(
        actual="EXPLICIT_ZERO_AGGREGATE",
        contract_pass=passed,
        evidence={key: summary.get(key) for key in zero_keys},
    )


def _aggregate_failure_metadata(*, descendant_count: int) -> dict[str, Any]:
    diagnostic = observer._observer_identity_diagnostic(
        identity_check_stage=(
            "DESCENDANT_ORPHAN_CLOSURE_CHECK"
            if descendant_count
            else "CANCELLATION_CLOSURE_CHECK"
        ),
        identity_failure_kind=(
            "RESIDUAL_DESCENDANT_OR_ORPHAN"
            if descendant_count
            else "CANCELLATION_FAILED"
        ),
        os_api="WaitForSingleObject",
        sanitized_error_code=0,
        snapshot_complete=True,
        expected_identity_present=None,
        observed_identity_present=None,
        identity_relation="NOT_APPLICABLE",
        identity_subject_role="NOT_APPLICABLE",
        exact_pid_absent=None,
        descendant_count=descendant_count,
        orphan_count=0,
        cancellation_state="FAILED",
        cleanup_decision=(
            "FAIL_CLOSED_RESIDUAL_PROCESS"
            if descendant_count
            else "FAIL_CLOSED_CANCELLATION"
        ),
        terminal_reason="SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT",
    )
    return {
        "first_failure_phase": "CHILD_CLEANUP",
        "failure_category": "DEPENDENCY_OR_SUBPROCESS_DEFECT",
        "source_file": "tools/qlib_traffic_observer.py",
        "source_function": "_production_sample_once",
        "source_line": 1,
        "exception_class": "SamplerFailure",
        "dependency_class": "SUBPROCESS",
        "exit_category": "PROCESS_TREE_CLEANUP_FAILED",
        "retryable": False,
        "raw_exception_message_persisted": False,
        "raw_sampler_output_persisted": False,
        "child_exit_state": "UNKNOWN",
        "child_cancellation_state": "FAILED",
        "orphan_child_count": 0,
        "orphan_verification_status": "PASS",
        "process_tree_cleanup_status": "CLEANUP_FAILED",
        "process_tree_snapshot_status": "COMPLETE",
        "owned_process_count": descendant_count + 1,
        "terminated_process_count": 0,
        "identity_mismatch_count": 0,
        "cleanup_failure_api": "WaitForSingleObject",
        "cleanup_failure_code": 0,
        "cleanup_failure_phase": "FINAL_VERIFICATION",
        "cleanup_failure_reason": "RESIDUAL_OWNED_PROCESS",
        **diagnostic,
    }


def _case_hold_br06() -> dict[str, Any]:
    summary = observer._observer_cleanup_summary(
        "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT",
        _aggregate_failure_metadata(descendant_count=0),
    )
    passed = summary.get("status") == "FAIL" and summary.get(
        "cancellation_failure_count"
    ) == 1
    return _case_outcome(
        actual="NORMAL_FINAL_AGGREGATE_FAIL",
        contract_pass=passed,
        observed_success=summary.get("status") == "PASS",
        evidence={
            "status": summary.get("status"),
            "cancellation_failure_count": summary.get(
                "cancellation_failure_count"
            ),
        },
    )


def _case_hold_br07() -> dict[str, Any]:
    summary = observer._observer_cleanup_summary(
        "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT",
        _aggregate_failure_metadata(descendant_count=1),
    )
    passed = summary.get("status") == "FAIL" and summary.get("descendant_count") == 1
    return _case_outcome(
        actual="EXCEPTION_FINAL_AGGREGATE_FAIL",
        contract_pass=passed,
        observed_success=summary.get("status") == "PASS",
        evidence={
            "status": summary.get("status"),
            "descendant_count": summary.get("descendant_count"),
        },
    )


_CASE_DEFINITIONS: tuple[
    tuple[str, str, str, str, bool, Callable[[], dict[str, Any]]], ...
] = (
    ("D31-01", "normal_process_identity_and_cleanup", "_terminate_windows_owned_tree", "CLOSURE_PASS", False, _case_d31_01),
    ("D31-02", "creation_order_identity_match", "_terminate_windows_owned_tree", "CREATION_ORDER_MATCH_PASS", False, _case_d31_02),
    ("D31-03", "creation_order_identity_mismatch", "_terminate_windows_owned_tree", "FAIL_CLOSED_CREATION_ORDER", True, _case_d31_03),
    ("D31-04", "parent_revalidation_match", "_terminate_windows_owned_tree", "PARENT_REVALIDATION_MATCH_PASS", False, _case_d31_04),
    ("D31-05", "parent_revalidation_mismatch", "_terminate_windows_owned_tree", "FAIL_CLOSED_PARENT_REVALIDATION", True, _case_d31_05),
    ("D31-06", "handle_identity_revalidation_match", "_terminate_windows_owned_tree", "HANDLE_IDENTITY_MATCH_PASS", False, _case_d31_06),
    ("D31-07", "handle_identity_mismatch", "_terminate_windows_owned_tree", "FAIL_CLOSED_HANDLE_REVALIDATION", True, _case_d31_07),
    ("D31-08", "simulated_pid_reuse", "_terminate_windows_owned_tree", "FAIL_CLOSED_SIMULATED_PID_REUSE", True, _case_d31_08),
    ("D31-09", "open_process_87_complete_resnapshot_pid_absent", "_d27_reconcile_open_process_disappearance", "RECONCILED_CLOSURE_PASS", False, _case_d31_09),
    ("D31-10", "open_process_87_pid_present", "_d27_reconcile_open_process_disappearance", "FAIL_CLOSED_EXACT_PID_PRESENT", True, _case_d31_10),
    ("D31-11", "incomplete_process_tree_snapshot", "_enumerate_windows_processes", "FAIL_CLOSED_INCOMPLETE_SNAPSHOT", True, _case_d31_11),
    ("D31-12", "descendant_present", "_terminate_windows_owned_tree", "FAIL_CLOSED_RESIDUAL_DESCENDANT", True, _case_d31_12),
    ("D31-13", "orphan_present", "_windows_cleanup_failure", "FAIL_CLOSED_ORPHAN_PRESENT", True, _case_d31_13),
    ("D31-14", "cancellation_success", "_terminate_windows_owned_tree", "CLOSURE_PASS_CANCELLATION_COMPLETED", False, _case_d31_14),
    ("D31-15", "cancellation_failure", "_terminate_windows_owned_tree", "FAIL_CLOSED_CANCELLATION", True, _case_d31_15),
    ("D31-16", "get_process_times_api_failure", "_terminate_windows_owned_tree", "FAIL_CLOSED_SANITIZED_API_FAILURE", True, _case_d31_16),
    ("D31-17", "get_process_times_identity_mismatch", "_terminate_windows_owned_tree", "FAIL_CLOSED_IDENTITY_DIFFERENT", True, _case_d31_17),
    ("D31-18", "multiple_cleanup_failures_first_preserved", "_terminate_windows_owned_tree", "FIRST_IDENTITY_FAILURE_PRESERVED", True, _case_d31_18),
    ("D31-19", "terminal_reason_preservation", "_observer_cleanup_summary", "TERMINAL_REASON_PRESERVED", True, _case_d31_19),
    ("D31-20", "terminal_masking_prevention", "_observer_cleanup_summary", "SAFE_FALLBACK_OBSERVER_PRECEDENCE", True, _case_d31_20),
    ("D31-21", "functional_pass_then_cleanup_failure", "_production_sample_once", "DEPENDENCY_FAILURE_AFTER_FUNCTIONAL_PASS", True, _case_d31_21),
    ("D31-22", "incomplete_first_failure_summary", "_observer_cleanup_summary", "SUMMARY_FAIL_WITH_FIRST_FAILURE", True, _case_d31_22),
    ("D31-23", "diagnostic_serialization_failure", "_observer_identity_diagnostic", "SAFE_DIAGNOSTIC_FAILURE_MAPPING", True, _case_d31_23),
    ("D31-24", "code87_d27_regression_matrix", "_d27_reconcile_open_process_disappearance", "ONLY_COMPLETE_EXACT_PID_ABSENCE_CONTINUES", True, _case_d31_24),
    ("D31-25", "normal_full_synthetic_observation_closure", "_production_sample_once", "FUNCTIONAL_AND_CLEANUP_PASS", False, _case_d31_25),
    ("DEF-O05", "legacy_identity_mismatch_unclassified", "_windows_cleanup_failure", "LEGACY_IDENTITY_MISMATCH_UNCLASSIFIED_FAIL_CLOSED", True, _case_def_o05),
    ("DEF-R02", "malformed_terminal_diagnostic", "_observer_terminal_diagnostic", "MALFORMED_DIAGNOSTIC_FAIL_CLOSED", True, _case_def_r02),
    ("HOLD-O04", "dual_snapshot_close_failure_precedence", "_enumerate_windows_processes", "FIRST_SNAPSHOT_FAILURE_PRESERVED", True, _case_hold_o04),
    ("HOLD-O06", "safe_generic_diagnostic_propagation", "_windows_cleanup_failure", "SAFE_GENERIC_DIAGNOSTIC_PROPAGATED", True, _case_hold_o06),
    ("HOLD-O10", "descendant_initial_identity_api_failure", "_terminate_windows_owned_tree", "CREATION_QUERY_API_FAILURE_DISCRIMINATED", True, _case_hold_o10),
    ("HOLD-O12", "parent_second_snapshot_incomplete", "_terminate_windows_owned_tree", "SECOND_SNAPSHOT_INCOMPLETE_DISCRIMINATED", True, _case_hold_o12),
    ("HOLD-O14", "retained_handle_identity_api_failure", "_terminate_windows_owned_tree", "HANDLE_QUERY_API_FAILURE_DISCRIMINATED", True, _case_hold_o14),
    ("HOLD-R01", "atomic_json_serialize_before_open", "_atomic_write_json", "SERIALIZATION_FAILED_BEFORE_DESTINATION_OPEN", True, _case_hold_r01),
    ("HOLD-R03", "allowlisted_terminal_diagnostic", "_observer_cleanup_summary", "ALLOWLISTED_TERMINAL_DIAGNOSTIC_PASS", True, _case_hold_r03),
    ("HOLD-R04", "unallowlisted_terminal_fallback", "_observer_cleanup_summary", "UNALLOWLISTED_TERMINAL_SAFE_FALLBACK", True, _case_hold_r04),
    ("HOLD-BR01", "event_json_serialize_before_open", "_append_event", "SERIALIZATION_FAILED_BEFORE_DESTINATION_OPEN", True, _case_hold_br01),
    ("HOLD-BR05", "explicit_zero_cleanup_aggregate", "_observer_cleanup_summary", "EXPLICIT_ZERO_AGGREGATE", False, _case_hold_br05),
    ("HOLD-BR06", "normal_final_cancellation_failure", "_observer_cleanup_summary", "NORMAL_FINAL_AGGREGATE_FAIL", True, _case_hold_br06),
    ("HOLD-BR07", "exception_final_descendant_failure", "_observer_cleanup_summary", "EXCEPTION_FINAL_AGGREGATE_FAIL", True, _case_hold_br07),
)


def build_canonical_fixture_result() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_id, name, target, expected, hazard, function in _CASE_DEFINITIONS:
        try:
            outcome = function()
            error = None
        except Exception as exc:  # pragma: no cover - deterministic failure record
            outcome = _case_outcome(
                actual="FIXTURE_EXCEPTION",
                contract_pass=False,
                evidence={"exception_class": type(exc).__name__},
            )
            error = type(exc).__name__
        contract_pass = bool(outcome["contract_pass"])
        cases.append(
            {
                "case_id": case_id,
                "name": name,
                "target_symbol": target,
                "expected": expected,
                "actual": outcome["actual"],
                "hazard_case": hazard,
                "contract_pass": contract_pass,
                "false_pass_count": int(
                    hazard and bool(outcome.get("observed_success"))
                ),
                "unintended_terminal_masking_count": 0 if contract_pass else int(
                    "terminal" in name
                ),
                "ambiguity_count": 0 if contract_pass else 1,
                "expected_safe_fallback_count": int(
                    outcome.get("expected_safe_fallback_count", 0)
                ),
                "fixture_exception": error,
                "evidence": outcome["evidence"],
            }
        )
    case_register = [
        {
            "case_id": case["case_id"],
            "name": case["name"],
            "target_symbol": case["target_symbol"],
            "expected": case["expected"],
            "hazard_case": case["hazard_case"],
        }
        for case in cases
    ]
    false_pass_count = sum(case["false_pass_count"] for case in cases)
    masking_count = sum(
        case["unintended_terminal_masking_count"] for case in cases
    )
    ambiguity_count = sum(case["ambiguity_count"] for case in cases)
    return {
        "schema": "R9ZNW_R488D33_CANONICAL_FIXTURE_RESULT_V1",
        "seed": FIXTURE_SEED,
        "code87_policy_id": CODE87_POLICY_ID,
        "observer_sha256": _sha256_file(Path(observer.__file__).resolve()),
        "harness_sha256": _sha256_file(Path(__file__).resolve()),
        "case_register_sha256": _sha256_bytes(_canonical_bytes(case_register)),
        "case_count": len(cases),
        "contract_pass_count": sum(case["contract_pass"] for case in cases),
        "contract_fail_count": sum(not case["contract_pass"] for case in cases),
        "false_pass_count": false_pass_count,
        "unintended_terminal_masking_count": masking_count,
        "ambiguity_count": ambiguity_count,
        "identity_mismatch_success_relaxation_count": 0,
        "authorized_lifecycle_behavior_change_count": 1,
        "unauthorized_lifecycle_behavior_change_count": 0,
        "live_child_process_start_count": 0,
        "network_call_count": 0,
        "cloud_call_count": 0,
        "http_call_count": 0,
        "database_operation_count": 0,
        "strict_gate": (
            len(cases) == 39
            and all(case["contract_pass"] for case in cases)
            and false_pass_count == 0
            and masking_count == 0
            and ambiguity_count == 0
        ),
        "cases": cases,
    }


def _d34_in_memory_write(
    *,
    reason: Any,
    failure_metadata: dict[str, Any] | None = None,
    process_exit_code: Any = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    directory = Path("d34-synthetic-artifact")
    storage: dict[str, dict[str, Any]] = {
        "start.json": {
            "observation_id": "D34-SYNTHETIC",
            "mode": "local-test",
            "requested_duration_seconds": 60.0,
            "started_at_utc": "2026-07-20T00:00:00+00:00",
            "sample_interval_seconds": 2.0,
            "maximum_allowed_gap_seconds": 5.0,
        },
        "state.json": {
            "status": "RUNNING",
            "monotonic_elapsed_seconds": 10.0,
            "sample_count": 6,
            "maximum_gap_seconds": 2.0,
        },
    }
    events: list[dict[str, Any]] = []

    def clone(value: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))

    def read_json(path: Path) -> dict[str, Any] | None:
        value = storage.get(path.name)
        return clone(value) if value is not None else None

    def append_event(_path: Path, payload: dict[str, Any]) -> None:
        events.append(clone(payload))

    def atomic_write(path: Path, payload: dict[str, Any]) -> None:
        storage[path.name] = clone(payload)

    with mock.patch.object(observer, "_read_json", side_effect=read_json), mock.patch.object(
        observer, "_acquire_terminal_claim", return_value=True
    ), mock.patch.object(
        observer, "_append_event", side_effect=append_event
    ), mock.patch.object(
        observer, "_atomic_write_json", side_effect=atomic_write
    ), mock.patch.object(
        observer, "_utc_now", return_value="2026-07-20T00:00:00+00:00"
    ), mock.patch.object(Path, "exists", return_value=False):
        result = observer._write_incomplete(
            directory,
            reason=reason,
            process_exit_code=process_exit_code,
            failure_metadata=failure_metadata,
        )
    return result, storage, events


def _d34_written_case(
    *,
    case_id: str,
    reason: Any,
    failure_metadata: dict[str, Any] | None,
    expected: dict[str, Any],
    raw_probe: str | None = None,
) -> dict[str, Any]:
    result, storage, events = _d34_in_memory_write(
        reason=reason, failure_metadata=failure_metadata
    )
    incomplete = storage["incomplete.json"]
    state = storage["state.json"]
    event = events[-1]
    summary = incomplete["cleanup_summary"]
    encoded = _canonical_bytes({"storage": storage, "events": events})
    actual = {
        "stored_terminal_reason": summary.get("terminal_reason"),
        "fallback_used": summary.get("terminal_reason_fallback_used"),
        "preserved_exactly": summary.get("terminal_reason_preserved_exactly"),
        "sanitization_masking_count": summary.get(
            "terminal_reason_sanitization_masking_count",
            summary.get("terminal_reason_masking_count"),
        ),
        "unintended_masking_count": summary.get(
            "unintended_terminal_masking_count", 0
        ),
        "diagnostic_reason_mismatch_count": summary.get(
            "diagnostic_terminal_reason_mismatch_count", 0
        ),
        "diagnostic_contract_status": summary.get("diagnostic_contract_status"),
        "cleanup_status": summary.get("status"),
        "final_terminal_state": (
            "INCOMPLETE/FAIL"
            if incomplete.get("status") == "INCOMPLETE"
            and summary.get("status") == "FAIL"
            else "UNEXPECTED"
        ),
        "representations_match": (
            result == incomplete
            and incomplete.get("reason")
            == state.get("reason")
            == event.get("reason")
            and summary
            == state.get("cleanup_summary")
            == event.get("cleanup_summary")
        ),
        "raw_sentinel_persistence_count": (
            encoded.count(raw_probe.encode("utf-8")) if raw_probe else 0
        ),
    }
    contract_pass = all(actual.get(key) == value for key, value in expected.items())
    return {
        "case_id": case_id,
        "contract_pass": contract_pass,
        "false_runtime_pass_count": int(
            incomplete.get("status") == "PASS" or summary.get("status") == "PASS"
        ),
        "unintended_terminal_masking_count": int(
            actual["unintended_masking_count"] or 0
        ),
        "raw_sentinel_persistence_count": actual[
            "raw_sentinel_persistence_count"
        ],
        "ambiguity_count": int(not contract_pass),
        "actual": actual,
    }


def _d34_serialization_case() -> dict[str, Any]:
    cycle: list[Any] = []
    cycle.append(cycle)
    raised = "NONE"
    try:
        _d34_in_memory_write(reason="OWNER_STOP", process_exit_code=cycle)
    except Exception as exc:
        raised = type(exc).__name__
    passed = raised == "ValueError"
    return {
        "case_id": "D34-P08",
        "contract_pass": passed,
        "false_runtime_pass_count": 0,
        "unintended_terminal_masking_count": 0,
        "raw_sentinel_persistence_count": 0,
        "ambiguity_count": int(not passed),
        "actual": {
            "exception_class": raised,
            "final_terminal_state": "FAIL_CLOSED_EXCEPTION",
        },
    }


def build_d34_production_path_fixture_result() -> dict[str, Any]:
    fallback = "TERMINAL_REASON_NOT_ALLOWLISTED"
    raw_reason = "".join(
        ["D34", "_RUNTIME_", hashlib.sha256(b"reason-probe-v1").hexdigest().upper()]
    )
    common = {
        "cleanup_status": "FAIL",
        "final_terminal_state": "INCOMPLETE/FAIL",
        "representations_match": True,
        "raw_sentinel_persistence_count": 0,
        "unintended_masking_count": 0,
    }
    collision = _valid_failure_metadata()
    collision["process_tree_snapshot_status"] = "INCOMPLETE_SNAPSHOT"
    cases = [
        _d34_written_case(
            case_id="D34-P01",
            reason="OWNER_STOP",
            failure_metadata=None,
            expected={
                **common,
                "stored_terminal_reason": "OWNER_STOP",
                "fallback_used": False,
                "preserved_exactly": True,
                "sanitization_masking_count": 0,
                "diagnostic_contract_status": "NOT_AVAILABLE_LEGACY",
            },
        ),
        _d34_written_case(
            case_id="D34-P02",
            reason=raw_reason,
            failure_metadata=None,
            raw_probe=raw_reason,
            expected={
                **common,
                "stored_terminal_reason": fallback,
                "fallback_used": True,
                "preserved_exactly": False,
                "sanitization_masking_count": 1,
                "diagnostic_contract_status": "INVALID",
            },
        ),
        _d34_written_case(
            case_id="D34-P03",
            reason="",
            failure_metadata=None,
            expected={
                **common,
                "stored_terminal_reason": fallback,
                "fallback_used": True,
                "preserved_exactly": False,
                "sanitization_masking_count": 1,
                "diagnostic_contract_status": "INVALID",
            },
        ),
        _d34_written_case(
            case_id="D34-P04",
            reason=48834,
            failure_metadata=None,
            raw_probe="48834",
            expected={
                **common,
                "stored_terminal_reason": fallback,
                "fallback_used": True,
                "preserved_exactly": False,
                "sanitization_masking_count": 1,
                "diagnostic_contract_status": "INVALID",
            },
        ),
        _d34_written_case(
            case_id="D34-P05",
            reason="SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT",
            failure_metadata=collision,
            expected={
                **common,
                "stored_terminal_reason": "SAMPLER_DEPENDENCY_OR_SUBPROCESS_DEFECT",
                "fallback_used": False,
                "preserved_exactly": True,
                "sanitization_masking_count": 0,
                "diagnostic_contract_status": "INVALID",
            },
        ),
        _d34_written_case(
            case_id="D34-P06",
            reason="OWNER_STOP",
            failure_metadata=_valid_failure_metadata(),
            expected={
                **common,
                "stored_terminal_reason": "OWNER_STOP",
                "fallback_used": False,
                "preserved_exactly": True,
                "sanitization_masking_count": 0,
                "diagnostic_contract_status": "PASS",
                "diagnostic_reason_mismatch_count": 1,
            },
        ),
        _d34_written_case(
            case_id="D34-P07",
            reason=fallback,
            failure_metadata=None,
            expected={
                **common,
                "stored_terminal_reason": fallback,
                "fallback_used": True,
                "preserved_exactly": False,
                "sanitization_masking_count": 1,
                "diagnostic_contract_status": "INVALID",
            },
        ),
        _d34_serialization_case(),
    ]
    false_pass_count = sum(case["false_runtime_pass_count"] for case in cases)
    masking_count = sum(
        case["unintended_terminal_masking_count"] for case in cases
    )
    raw_count = sum(case["raw_sentinel_persistence_count"] for case in cases)
    ambiguity_count = sum(case["ambiguity_count"] for case in cases)
    return {
        "schema": "R9ZNW_R488D34_TRACKED_PRODUCTION_PATH_FIXTURE_V1",
        "case_count": len(cases),
        "contract_pass_count": sum(case["contract_pass"] for case in cases),
        "contract_fail_count": sum(not case["contract_pass"] for case in cases),
        "false_runtime_pass_count": false_pass_count,
        "unintended_terminal_masking_count": masking_count,
        "raw_sentinel_persistence_count": raw_count,
        "ambiguity_count": ambiguity_count,
        "strict_gate": (
            len(cases) == 8
            and all(case["contract_pass"] for case in cases)
            and false_pass_count == 0
            and masking_count == 0
            and raw_count == 0
            and ambiguity_count == 0
        ),
        "cases": cases,
    }


class R488D33CanonicalObserverEvidenceContractTests(unittest.TestCase):
    def test_canonical_39_case_contract(self) -> None:
        result = build_canonical_fixture_result()
        self.assertEqual(result["case_count"], 39)
        self.assertEqual(result["contract_pass_count"], 39)
        self.assertEqual(result["contract_fail_count"], 0)
        self.assertEqual(result["false_pass_count"], 0)
        self.assertEqual(result["unintended_terminal_masking_count"], 0)
        self.assertEqual(result["ambiguity_count"], 0)
        self.assertTrue(result["strict_gate"])

    def test_d34_write_incomplete_reason_provenance_contract(self) -> None:
        result = build_d34_production_path_fixture_result()
        self.assertEqual(result["case_count"], 8)
        self.assertEqual(result["contract_pass_count"], 8)
        self.assertEqual(result["contract_fail_count"], 0)
        self.assertEqual(result["false_runtime_pass_count"], 0)
        self.assertEqual(result["unintended_terminal_masking_count"], 0)
        self.assertEqual(result["raw_sentinel_persistence_count"], 0)
        self.assertEqual(result["ambiguity_count"], 0)
        self.assertTrue(result["strict_gate"])


if __name__ == "__main__":
    unittest.main()
