from __future__ import annotations

import inspect
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tools import qlib_traffic_observer as observer


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = REPO_ROOT / "deploy" / "qlib-skillup-runtime" / "operations-runbook.md"


class _ScriptedWin32Api:
    """Deterministic Win32 fault injector; it never touches the host OS."""

    def __init__(
        self,
        snapshots: list[dict[str, object]] | None = None,
        *,
        root_pid: int = 10,
        root_handle: int = 7000,
    ) -> None:
        self.snapshots = snapshots or [{"entries": []}, {"entries": []}]
        self.snapshot_exception: BaseException | None = None
        self.open_fail_pids: set[int] = set()
        self.terminate_fail_pids: set[int] = set()
        self.identity_fail_pids: set[int] = set()
        self.identity_sequences: dict[int, list[int]] = {}
        self.wait_sequences: dict[int, list[tuple[int, int]]] = {}
        self.handle_to_pid = {root_handle: root_pid}
        self.snapshot_state: dict[int, dict[str, object]] = {}
        self.snapshot_index = 0
        self.opened_handles: list[int] = []
        self.closed_handles: list[int] = []
        self.terminate_calls: list[int] = []
        self.close_exception_handles: set[int] = set()

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
        first_error = script.get("first_error")
        if isinstance(first_error, int):
            return None, first_error
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
        next_error = script.get("next_error")
        if isinstance(next_error, int):
            return None, next_error
        entries = list(script.get("entries", []))
        cursor = int(script.get("cursor", 0))
        if cursor >= len(entries):
            return None, observer._WIN32_ERROR_NO_MORE_FILES
        script["cursor"] = cursor + 1
        return tuple(entries[cursor]), 0  # type: ignore[arg-type,return-value]

    def open_process(self, process_id: int) -> int:
        if process_id in self.open_fail_pids:
            raise observer._Win32ApiFailure("OpenProcess", 5, "OPEN_PROCESS")
        handle = 20000 + int(process_id)
        self.handle_to_pid[handle] = int(process_id)
        self.opened_handles.append(handle)
        return handle

    def creation_identity(self, process_handle: int) -> int:
        process_id = self.handle_to_pid[int(process_handle)]
        if process_id in self.identity_fail_pids:
            raise observer._Win32ApiFailure(
                "GetProcessTimes", 5, "IDENTITY_QUERY"
            )
        sequence = self.identity_sequences.get(process_id)
        if sequence:
            if len(sequence) > 1:
                return sequence.pop(0)
            return sequence[0]
        return 100000 + process_id

    def wait(self, process_handle: int, _timeout_ms: int) -> tuple[int, int]:
        process_id = self.handle_to_pid[int(process_handle)]
        sequence = self.wait_sequences.get(process_id)
        if sequence:
            if len(sequence) > 1:
                return sequence.pop(0)
            return sequence[0]
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


def _utc_offset(seconds: float = 0.0) -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return value.isoformat().replace("+00:00", "Z")


class QlibTrafficObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_parent = Path(os.environ.get("R477_TEST_TMP", tempfile.gettempdir()))
        temp_parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="r477-observer-test-", dir=temp_parent
        )
        self.root = Path(self._temporary.name)
        self.observation_ids: list[str] = []
        self.addCleanup(self._temporary.cleanup)
        self.addCleanup(self._stop_running_observations)

    def _new_id(self, prefix: str) -> str:
        observation_id = f"{prefix}-{uuid.uuid4().hex[:10]}"
        self.observation_ids.append(observation_id)
        return observation_id

    def _stop_running_observations(self) -> None:
        for observation_id in self.observation_ids:
            directory = self.root / observation_id
            if not directory.exists():
                continue
            result = observer.poll_observation(
                artifact_root=self.root, observation_id=observation_id
            )
            if result.get("status") == "RUNNING":
                observer.stop_observation(
                    artifact_root=self.root,
                    observation_id=observation_id,
                    reason="TASK_CLEANUP",
                    wait_seconds=3.0,
                )

    def _start_short(
        self, prefix: str, *, duration: float = 0.45, interval: float = 0.1
    ) -> dict[str, object]:
        observation_id = self._new_id(prefix)
        return observer.start_observation(
            artifact_root=self.root,
            observation_id=observation_id,
            duration_seconds=duration,
            sample_interval_seconds=interval,
            max_gap_seconds=2.0,
            stale_after_seconds=5.0,
            mode="local-test",
        )

    def _wait_terminal(self, observation_id: str, timeout: float = 6.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        latest: dict[str, object] = {}
        while time.monotonic() < deadline:
            latest = observer.poll_observation(
                artifact_root=self.root, observation_id=observation_id
            )
            if latest.get("status") != "RUNNING":
                return latest
            time.sleep(0.05)
        self.fail(f"observation did not reach a terminal state: {latest}")

    def _fixture_directory(
        self,
        prefix: str,
        *,
        duration: float = 10.0,
        interval: float = 0.5,
        max_gap: float = 1.0,
        stale_after: float = 2.0,
    ) -> tuple[str, Path]:
        observation_id = self._new_id(prefix)
        directory = self.root / observation_id
        directory.mkdir()
        observer._atomic_write_json(
            directory / "start.json",
            {
                "schema_version": observer.SCHEMA_VERSION,
                "observation_id": observation_id,
                "mode": "local-test",
                "launcher_started_at_utc": _utc_offset(),
                "started_at_utc": _utc_offset(),
                "requested_duration_seconds": duration,
                "sample_interval_seconds": interval,
                "maximum_allowed_gap_seconds": max_gap,
                "stale_after_seconds": stale_after,
            },
        )
        (directory / "events.ndjson").write_text("", encoding="utf-8")
        return observation_id, directory

    def test_detached_start_poll_finalize_and_incremental_artifacts(self) -> None:
        launched_at = time.monotonic()
        launch = self._start_short("normal")
        launcher_elapsed = time.monotonic() - launched_at
        observation_id = str(launch["observation_id"])
        directory = Path(str(launch["artifact_path"]))

        self.assertLess(launcher_elapsed, 2.0)
        self.assertEqual(launch["status"], "STARTED")
        self.assertTrue((directory / "start.json").exists())
        self.assertTrue((directory / "events.ndjson").exists())

        initial = observer.poll_observation(
            artifact_root=self.root, observation_id=observation_id
        )
        self.assertIn(initial["status"], {"RUNNING", "PASS"})
        final = self._wait_terminal(observation_id)

        self.assertEqual(final["status"], "PASS")
        self.assertTrue(final["completion_marker"])
        self.assertEqual(final["mode"], "local-test")
        self.assertGreaterEqual(
            float(final["monotonic_elapsed_seconds"]),
            float(final["requested_duration_seconds"]),
        )
        self.assertGreaterEqual(int(final["sample_count"]), 2)
        self.assertLessEqual(
            float(final["maximum_gap_seconds"]),
            float(final["maximum_allowed_gap_seconds"]),
        )
        self.assertEqual(final["process_exit_code"], 0)
        self.assertTrue((directory / "heartbeat.json").exists())
        self.assertTrue((directory / "state.json").exists())
        self.assertTrue((directory / "final.json").exists())
        events = (directory / "events.ndjson").read_text(encoding="utf-8")
        self.assertIn('"event": "SAMPLE"', events)
        self.assertIn('"event": "COMPLETED"', events)

    def test_atomic_write_retries_transient_replace_contention(self) -> None:
        target = self.root / "atomic-retry.json"
        real_replace = os.replace
        attempts = 0

        def replace_with_contention(source: object, destination: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError("simulated transient sharing violation")
            real_replace(source, destination)

        with mock.patch.object(observer.os, "replace", side_effect=replace_with_contention):
            observer._atomic_write_json(target, {"status": "PASS"})

        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["status"], "PASS")
        self.assertEqual(attempts, 3)
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_requested_duration_short_final_is_never_pass(self) -> None:
        observation_id, directory = self._fixture_directory("short-final", duration=10.0)
        observer._atomic_write_json(
            directory / "final.json",
            {
                "schema_version": observer.SCHEMA_VERSION,
                "observation_id": observation_id,
                "mode": "local-test",
                "status": "PASS",
                "verification_status": "PASS",
                "completion_marker": True,
                "requested_duration_seconds": 10.0,
                "monotonic_elapsed_seconds": 9.99,
                "sample_count": 3,
                "maximum_gap_seconds": 0.5,
                "process_exit_code": 0,
            },
        )
        result = observer.poll_observation(
            artifact_root=self.root, observation_id=observation_id
        )
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertIn("REQUESTED_DURATION_NOT_MET", result["validation_failures"])

    def test_missing_final_marker_and_child_loss_are_incomplete(self) -> None:
        observation_id, directory = self._fixture_directory("child-loss")
        observer._atomic_write_json(
            directory / "state.json",
            {
                "status": "RUNNING",
                "sample_count": 1,
                "monotonic_elapsed_seconds": 0.5,
                "maximum_gap_seconds": 0.0,
            },
        )
        observer._atomic_write_json(
            directory / "heartbeat.json",
            {"heartbeat_at_utc": _utc_offset(), "sequence": 1},
        )
        observer._atomic_write_json(directory / "process.json", {"pid": 2147483647})
        observer._atomic_write_json(
            directory / "process-loss.json",
            {
                "first_detected_at_utc": _utc_offset(-2.0),
                "heartbeat_sequence": 1,
                "consecutive_detection_count": 1,
            },
        )
        result = observer.poll_observation(
            artifact_root=self.root, observation_id=observation_id
        )
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["verification_status"], "NOT_VERIFIED")
        self.assertEqual(result["reason"], "CHILD_PROCESS_LOSS")
        self.assertFalse((directory / "final.json").exists())
        self.assertTrue((directory / "incomplete.json").exists())

    def test_stale_heartbeat_and_sample_gap_are_detected(self) -> None:
        cases = (
            ("stale", 0.1, _utc_offset(-10.0), "STALE_HEARTBEAT"),
            ("gap", 2.0, _utc_offset(), "SAMPLE_GAP_EXCEEDED"),
        )
        for prefix, maximum_gap, heartbeat_at, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                observation_id, directory = self._fixture_directory(prefix)
                observer._atomic_write_json(
                    directory / "state.json",
                    {
                        "status": "RUNNING",
                        "sample_count": 2,
                        "monotonic_elapsed_seconds": 1.0,
                        "maximum_gap_seconds": maximum_gap,
                    },
                )
                observer._atomic_write_json(
                    directory / "heartbeat.json",
                    {"heartbeat_at_utc": heartbeat_at},
                )
                observer._atomic_write_json(
                    directory / "process.json", {"pid": os.getpid()}
                )
                result = observer.poll_observation(
                    artifact_root=self.root, observation_id=observation_id
                )
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(result["reason"], expected_reason)
                self.assertFalse((directory / "final.json").exists())

    def test_duplicate_observation_id_is_rejected(self) -> None:
        observation_id = self._new_id("duplicate")
        (self.root / observation_id).mkdir()
        with self.assertRaises(observer.DuplicateObservationError):
            observer.start_observation(
                artifact_root=self.root,
                observation_id=observation_id,
                duration_seconds=1.0,
                sample_interval_seconds=0.1,
                max_gap_seconds=0.5,
                stale_after_seconds=1.0,
                mode="local-test",
            )

    def test_repeated_detached_launches_do_not_race_process_metadata(self) -> None:
        observation_ids: list[str] = []
        for index in range(5):
            launch = self._start_short(f"race-{index}", duration=0.2, interval=0.2)
            observation_ids.append(str(launch["observation_id"]))
        for observation_id in observation_ids:
            result = self._wait_terminal(observation_id)
            self.assertEqual(result["status"], "PASS")
            directory = self.root / observation_id
            self.assertFalse(list(directory.glob(".*.tmp")))

    def test_task_owned_early_stop_preserves_unrelated_sibling(self) -> None:
        sibling = self.root / "unrelated-sentinel"
        sibling.mkdir()
        sentinel = sibling / "keep.txt"
        sentinel.write_bytes(b"unchanged")
        before = sentinel.read_bytes()

        launch = self._start_short("failure-injection", duration=30.0)
        observation_id = str(launch["observation_id"])
        pid = int(launch["pid"])
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            current = observer.poll_observation(
                artifact_root=self.root, observation_id=observation_id
            )
            if current.get("status") == "RUNNING":
                break
            time.sleep(0.05)
        result = observer.stop_observation(
            artifact_root=self.root,
            observation_id=observation_id,
            reason="FAILURE_INJECTION",
            wait_seconds=5.0,
        )

        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["verification_status"], "NOT_VERIFIED")
        self.assertEqual(result["reason"], "FAILURE_INJECTION")
        self.assertFalse((self.root / observation_id / "final.json").exists())
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertTrue(observer._pid_exists(os.getpid()))
        exit_deadline = time.monotonic() + 3.0
        while observer._pid_exists(pid) and time.monotonic() < exit_deadline:
            time.sleep(0.05)
        self.assertFalse(observer._pid_exists(pid))

    def test_artifacts_are_redacted_and_local_mode_ignores_production_sampler(self) -> None:
        marker = self.root / "unexpected-sampler-execution.txt"
        previous_sampler = os.environ.get(observer.PRODUCTION_SAMPLER_ENV)
        previous_identity = os.environ.get(observer.IDENTITY_MATERIAL_ENV)
        os.environ[observer.PRODUCTION_SAMPLER_ENV] = json.dumps(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('bad')"]
        )
        os.environ[observer.IDENTITY_MATERIAL_ENV] = "synthetic-memory-only-value"
        try:
            launch = self._start_short("redaction")
            observation_id = str(launch["observation_id"])
            result = self._wait_terminal(observation_id)
        finally:
            if previous_sampler is None:
                os.environ.pop(observer.PRODUCTION_SAMPLER_ENV, None)
            else:
                os.environ[observer.PRODUCTION_SAMPLER_ENV] = previous_sampler
            if previous_identity is None:
                os.environ.pop(observer.IDENTITY_MATERIAL_ENV, None)
            else:
                os.environ[observer.IDENTITY_MATERIAL_ENV] = previous_identity

        self.assertEqual(result["status"], "PASS")
        self.assertFalse(marker.exists())
        directory = self.root / observation_id
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(directory.iterdir())
            if path.is_file()
        )
        self.assertIsNone(re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", combined))
        self.assertIsNone(re.search(r"[^\s@]+@[^\s@]+", combined))
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", combined))
        for forbidden in ("Authorization", "raw_query", "raw_answer", "internal_path"):
            self.assertNotIn(forbidden, combined)

    def test_production_sample_redaction_rejects_sensitive_fields(self) -> None:
        safe = observer._sanitize_sample_value(
            {"status": "PASS", "latency_ms": 12.5, "unexpected_5xx_count": 0}
        )
        self.assertEqual(safe["status"], "PASS")
        for field in (
            "raw_url",
            "email",
            "query",
            "answer",
            "authorization",
            "identity_token",
            "internal_path",
        ):
            with self.subTest(field=field):
                with self.assertRaises(observer.ObserverError):
                    observer._sanitize_sample_value({field: "value"})

    def test_local_sampler_is_deterministic_and_has_no_network_surface(self) -> None:
        self.assertEqual(observer._local_sample("local-check", 4), observer._local_sample("local-check", 4))
        source = inspect.getsource(observer._local_sample)
        for forbidden in ("socket", "urllib", "http", "requests", "subprocess"):
            self.assertNotIn(forbidden, source.lower())

    def test_runbook_uses_canonical_reconnectable_observer(self) -> None:
        text = RUNBOOK_PATH.read_text(encoding="utf-8")
        self.assertIn("tools/qlib_traffic_observer.py start", text)
        self.assertIn("tools/qlib_traffic_observer.py poll", text)
        self.assertIn("--mode production", text)
        self.assertIn("completion_marker", text)
        self.assertIn("monotonic_elapsed_seconds", text)
        self.assertIn("00010-zuj", text)
        self.assertIn("must not be reused", text)


class R479DecisionContractTests(unittest.TestCase):
    def test_health_normalization_and_functional_zero_tolerance(self) -> None:
        for value in ("OK", "ok", "  Ok  "):
            with self.subTest(value=value):
                result = observer.evaluate_functional_contract(
                    unauthenticated_status=403,
                    authenticated_status=200,
                    health_value=value,
                )
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["normalized_health"], "ok")

        self.assertEqual(
            observer.evaluate_functional_contract(
                unauthenticated_status=403,
                authenticated_status=200,
                health_value="healthy",
            )["status"],
            "STOP",
        )
        self.assertEqual(
            observer.evaluate_functional_contract(
                unauthenticated_status=200,
                authenticated_status=200,
                health_value="ok",
                unexpected_5xx_count=1,
            )["status"],
            "STOP",
        )

    def test_latency_and_error_limits_require_two_consecutive_windows(self) -> None:
        self.assertEqual(observer.latency_stop_limit_ms(1200), 3000)
        self.assertEqual(observer.latency_stop_limit_ms(2000), 4000)
        self.assertEqual(observer.error_rate_stop_limit_percent(0.2), 1.0)
        self.assertEqual(observer.error_rate_stop_limit_percent(0.8), 1.3)

        latency_stop = observer.evaluate_aggregate_windows(
            [
                {"request_count": 20, "p95_latency_ms": 3001, "five_xx_rate_percent": 0},
                {"request_count": 20, "p95_latency_ms": 3002, "five_xx_rate_percent": 0},
            ],
            stable_baseline_p95_ms=100,
            stable_baseline_5xx_rate_percent=0,
        )
        self.assertEqual(latency_stop["status"], "STOP")
        self.assertTrue(latency_stop["latency_stop"])

        error_stop = observer.evaluate_aggregate_windows(
            [
                {"request_count": 20, "p95_latency_ms": 100, "five_xx_rate_percent": 1.1},
                {"request_count": 20, "p95_latency_ms": 100, "five_xx_rate_percent": 1.2},
            ],
            stable_baseline_p95_ms=100,
            stable_baseline_5xx_rate_percent=0,
        )
        self.assertEqual(error_stop["status"], "STOP")
        self.assertTrue(error_stop["error_rate_stop"])

        nonconsecutive = observer.evaluate_aggregate_windows(
            [
                {"request_count": 20, "p95_latency_ms": 3001, "five_xx_rate_percent": 1.1},
                {"request_count": 20, "p95_latency_ms": 100, "five_xx_rate_percent": 0},
                {"request_count": 20, "p95_latency_ms": 3001, "five_xx_rate_percent": 1.1},
            ],
            stable_baseline_p95_ms=100,
            stable_baseline_5xx_rate_percent=0,
        )
        self.assertEqual(nonconsecutive["status"], "PASS")

    def test_low_volume_is_insufficient_and_requires_exact_fallback(self) -> None:
        aggregate = observer.evaluate_aggregate_windows(
            [{"request_count": 19, "p95_latency_ms": 100, "five_xx_rate_percent": 0}],
            stable_baseline_p95_ms=100,
            stable_baseline_5xx_rate_percent=0,
        )
        self.assertEqual(aggregate["aggregate_metric_status"], "INSUFFICIENT_DATA")
        self.assertTrue(aggregate["fallback_required"])

        fallback = observer.evaluate_low_volume_fallback(
            synthetic_health_failure_count=0,
            synthetic_auth_failure_count=0,
            unexpected_synthetic_5xx_count=0,
            synthetic_p95_latency_ms=250,
            latency_limit_ms=3000,
            timeout_count=0,
            evidence_trace_missing_count=0,
            capacity_status="PASS",
            cost_proxy_status="PASS_WITH_LIMITS",
            observer_final_status="PASS",
        )
        self.assertEqual(fallback["status"], "PASS_WITH_LOW_VOLUME_LIMITS")
        failed = observer.evaluate_low_volume_fallback(
            synthetic_health_failure_count=0,
            synthetic_auth_failure_count=0,
            unexpected_synthetic_5xx_count=0,
            synthetic_p95_latency_ms=3001,
            latency_limit_ms=3000,
            timeout_count=0,
            evidence_trace_missing_count=0,
            capacity_status="PASS",
            cost_proxy_status="PASS_WITH_LIMITS",
            observer_final_status="PASS",
        )
        self.assertEqual(failed["status"], "STOP")

    def test_candidate_capacity_is_separate_from_historical_stable_maxscale(self) -> None:
        contract = dict(
            effective_min=0,
            effective_max=2,
            immutable_maxscale=2,
            active_instances=2,
            failed_startup_count=0,
            request_drop_or_throttle_count=0,
            pending_not_ready_seconds=120,
            concurrency=80,
            cpu=1,
            memory="512Mi",
            timeout_seconds=300,
        )
        self.assertEqual(observer.evaluate_capacity_contract(**contract)["status"], "PASS")
        contract["immutable_maxscale"] = 20
        self.assertEqual(observer.evaluate_capacity_contract(**contract)["status"], "STOP")

    def test_cost_proxy_is_bounded_and_never_claims_unverified_amount(self) -> None:
        contract = dict(
            candidate_min_instances=0,
            candidate_max_instances=2,
            candidate_revision_creation_count=1,
            total_observation_seconds=3000,
            unexpected_billable_resource_delta_count=0,
            image_push_count=0,
            cloud_sql_binding_count=0,
            additional_service_or_scheduler_count=0,
            active_candidate_instance_count=2,
        )
        result = observer.evaluate_cost_proxy_contract(**contract)
        self.assertEqual(result["status"], "PASS_WITH_LIMITS")
        self.assertEqual(result["real_time_billing_amount_status"], "NOT_VERIFIED")
        contract["total_observation_seconds"] = 3001
        self.assertEqual(observer.evaluate_cost_proxy_contract(**contract)["status"], "STOP")

    def test_registry_intermediate_output_and_short_final_are_rejected(self) -> None:
        registry = {
            "manifest_digest_match": True,
            "config_digest_match": True,
            "layer_count": 16,
            "required_labels_match": True,
            "source_commit_match": True,
            "runtime_user_match": True,
            "private_access": True,
            "latest_used": False,
            "registry_mutation_audit_count": 0,
            "image_push_republication_count": 0,
            "deletion_count": 0,
        }
        self.assertIn("completion_marker", observer.validate_registry_final_artifact(registry))
        registry["completion_marker"] = True
        self.assertEqual(observer.validate_registry_final_artifact(registry), [])

        start = {
            "observation_id": "sample-minimum",
            "mode": "production",
            "requested_duration_seconds": 600,
            "sample_interval_seconds": 50,
            "maximum_allowed_gap_seconds": 58,
        }
        final = {
            "observation_id": "sample-minimum",
            "mode": "production",
            "status": "PASS",
            "verification_status": "PASS",
            "completion_marker": True,
            "monotonic_elapsed_seconds": 600,
            "maximum_gap_seconds": 50,
            "sample_count": 12,
            "process_exit_code": 0,
        }
        self.assertIn(
            "SAMPLE_COUNT_BELOW_EXPECTED_MINIMUM",
            observer._validate_final(start, final),
        )

    def test_runbook_and_observer_constants_match(self) -> None:
        text = RUNBOOK_PATH.read_text(encoding="utf-8")
        expected_fragments = (
            "historical rollback revision's immutable maxScale is 20",
            "LATENCY_STOP_LIMIT_MS=max(3000, stable_baseline_p95_ms * 2.0)",
            "ERROR_RATE_STOP_LIMIT_PERCENT=max(1.0, stable_baseline_5xx_rate_percent + 0.5)",
            "fewer than 20 requests is `INSUFFICIENT_DATA`",
            "active instances<=2",
            "total authorized staged observation seconds<=3000",
            "case-insensitive, surrounding-whitespace-trimmed normalization",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        self.assertEqual(observer.UNAUTHENTICATED_EXPECTED_STATUS, 403)
        self.assertEqual(observer.AUTHENTICATED_EXPECTED_STATUS, 200)
        self.assertEqual(observer.ABSOLUTE_P95_LATENCY_LIMIT_MS, 3000)
        self.assertEqual(observer.RELATIVE_P95_MULTIPLIER_FROM_STABLE_BASELINE, 2.0)
        self.assertEqual(observer.MINIMUM_AGGREGATE_WINDOW_REQUEST_COUNT, 20)
        self.assertEqual(observer.NEW_CANDIDATE_IMMUTABLE_MAXSCALE, 2)
        self.assertEqual(observer.TOTAL_AUTHORIZED_OBSERVATION_SECONDS, 3000)


class R481PretrafficEvidenceContractTests(unittest.TestCase):
    def _audit_result(
        self,
        *,
        start: str,
        end: str,
        count: int = 0,
        event_hashes: tuple[str, ...] = (),
        completion: bool = True,
        partial: bool = False,
    ) -> dict[str, object]:
        return {
            "query_started_at_utc": start,
            "query_completed_at_utc": end,
            "filter_contract_sha256": "contract",
            "exact_filter_sha256": "filter",
            "completion_marker": completion,
            "partial_result": partial,
            "authoritative_unique_event_count": count,
            "event_identifier_hash_set": list(event_hashes),
            "status": "PASS",
        }

    def test_authoritative_iam_query_is_exact_closed_and_resource_scoped(self) -> None:
        query = observer.build_authoritative_set_iam_policy_query(
            project_id="project-under-test",
            service="qlib-skillup-runtime",
            region="asia-northeast1",
            window_start_utc="2026-07-15T05:08:53.2859932Z",
            window_end_utc="2026-07-15T05:29:58.3341407Z",
        )
        text = query["query_filter"]
        self.assertIn('timestamp>="2026-07-15T05:08:53.2859932Z"', text)
        self.assertIn('timestamp<="2026-07-15T05:29:58.3341407Z"', text)
        self.assertIn('resource.labels.service_name="qlib-skillup-runtime"', text)
        self.assertIn('resource.labels.location="asia-northeast1"', text)
        self.assertIn(
            'protoPayload.methodName="google.cloud.run.v1.Services.SetIamPolicy"',
            text,
        )
        self.assertNotIn("=~", text)
        self.assertEqual(query["safe_contract"]["project"], "MASKED")

    def test_authoritative_iam_entries_exclude_broad_events_and_deduplicate(self) -> None:
        resource_name = "namespaces/project-under-test/services/qlib-skillup-runtime"

        def entry(
            insert_id: str,
            *,
            method: str = observer.EXACT_SET_IAM_POLICY_METHOD,
            resource: str = resource_name,
            service_name: str = "qlib-skillup-runtime",
            location: str = "asia-northeast1",
            resource_type: str = "cloud_run_revision",
        ) -> dict[str, object]:
            return {
                "insertId": insert_id,
                "timestamp": "2026-07-15T05:20:00Z",
                "resource": {
                    "type": resource_type,
                    "labels": {"service_name": service_name, "location": location},
                },
                "protoPayload": {
                    "methodName": method,
                    "resourceName": resource,
                    "authenticationInfo": {"principalEmail": "masked@example.invalid"},
                },
            }

        result = observer.sanitize_authoritative_set_iam_policy_entries(
            [
                entry("one"),
                entry("one"),
                entry("traffic", method="google.cloud.run.v1.Services.UpdateTraffic"),
                entry("registry", method="artifactregistry.tags.create"),
                entry("other", service_name="other-service"),
            ],
            project_id="project-under-test",
            service="qlib-skillup-runtime",
            region="asia-northeast1",
            before_iam_sha256="same",
            after_iam_sha256="same",
        )
        self.assertEqual(result["authoritative_unique_event_count"], 1)
        self.assertEqual(result["excluded_broad_or_wrong_method_count"], 3)
        serialized = json.dumps(result)
        self.assertNotIn("masked@example.invalid", serialized)
        self.assertIn("USER_OR_WORKFORCE_IDENTITY", serialized)

    def test_authoritative_repeats_reject_partial_incomplete_and_010(self) -> None:
        zero_results = [
            self._audit_result(
                start="2026-07-15T06:00:00Z",
                end="2026-07-15T06:00:01Z",
            ),
            self._audit_result(
                start="2026-07-15T06:00:31Z",
                end="2026-07-15T06:00:32Z",
            ),
            self._audit_result(
                start="2026-07-15T06:01:02Z",
                end="2026-07-15T06:01:03Z",
            ),
        ]
        self.assertEqual(
            observer.evaluate_authoritative_iam_repeats(zero_results)["status"],
            "PASS",
        )

        partial = [dict(result) for result in zero_results]
        partial[1]["partial_result"] = True
        self.assertEqual(
            observer.evaluate_authoritative_iam_repeats(partial)["status"],
            "NOT_VERIFIED",
        )
        incomplete = [dict(result) for result in zero_results]
        incomplete[2]["completion_marker"] = False
        self.assertEqual(
            observer.evaluate_authoritative_iam_repeats(incomplete)["status"],
            "NOT_VERIFIED",
        )
        non_monotonic = [dict(result) for result in zero_results]
        non_monotonic[1]["authoritative_unique_event_count"] = 1
        non_monotonic[1]["event_identifier_hash_set"] = ["event"]
        decision = observer.evaluate_authoritative_iam_repeats(non_monotonic)
        self.assertEqual(decision["status"], "NOT_VERIFIED")
        self.assertTrue(decision["non_monotonic_result"])

    def test_internal_path_true_fixtures_and_safe_values(self) -> None:
        true_values = (
            r"C:\internal\file",
            r"H:\private\file",
            r"\\server\share\file",
            "/workspace/internal",
            "/root/internal",
            "/home/internal",
            "/tmp/internal",
            "/var/internal",
        )
        safe_values = (
            "/health",
            "/assets/app.js",
            "/app/route",
            "https://example.invalid/route",
            "/items/0",
            "schema.selector",
            "ev-soldering-safe-summary-v1",
            "field_name",
            "ok",
            r"C:\\",
            r"H:\\",
        )
        for value in true_values:
            with self.subTest(value=value):
                self.assertEqual(
                    observer.classify_internal_path_value(value)["classification"],
                    "TRUE_INTERNAL_PATH",
                )
        for value in safe_values:
            with self.subTest(value=value):
                self.assertNotEqual(
                    observer.classify_internal_path_value(value)["classification"],
                    "TRUE_INTERNAL_PATH",
                )

    def test_internal_path_scans_values_not_field_names_and_redacts_matches(self) -> None:
        payload = {
            r"C:\field\name": "ok",
            "route": "/assets/app.js",
            "nested": {"location": r"H:\private\file"},
        }
        result = observer.privacy_safe_internal_path_classification(
            payload,
            response_surface_category="FIXTURE_JSON",
        )
        self.assertEqual(result["true_internal_path_count"], 1)
        serialized = json.dumps(result)
        self.assertNotIn(r"C:\field\name", serialized)
        self.assertNotIn(r"H:\private\file", serialized)
        self.assertIn("matched_value_sha256", serialized)
        self.assertFalse(result["raw_fragment_persisted"])

    def test_runbook_contains_r481_evidence_contract(self) -> None:
        text = RUNBOOK_PATH.read_text(encoding="utf-8")
        for fragment in (
            "google.cloud.run.v1.Services.SetIamPolicy",
            "insertId",
            "closed start/end window",
            "parsed response values only",
            "escaped drive-root schema literals",
            "raw response fragments must remain memory-only",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment.lower(), text.lower())


class R483SamplerAndRollbackReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cleanup_patch_active = True
        self._cleanup_patcher = mock.patch.object(
            observer,
            "_terminate_sampler_process",
            return_value={
                "child_exit_state": "EXITED",
                "child_cancellation_state": "NOT_REQUIRED",
                "orphan_child_count": 0,
                "orphan_verification_status": "PASS",
                "process_tree_cleanup_status": "NOT_REQUIRED",
                "process_tree_snapshot_status": "COMPLETE",
                "owned_process_count": 1,
                "terminated_process_count": 0,
                "identity_mismatch_count": 0,
                "cleanup_failure_api": "NONE",
                "cleanup_failure_code": 0,
                "cleanup_failure_phase": "NONE",
                "cleanup_failure_reason": "NONE",
            },
        )
        self._cleanup_patcher.start()
        self.addCleanup(self._stop_cleanup_patch)

    def _stop_cleanup_patch(self) -> None:
        if self._cleanup_patch_active:
            self._cleanup_patcher.stop()
            self._cleanup_patch_active = False

    @staticmethod
    def _completed(payload: dict[str, object], returncode: int = 42) -> mock.Mock:
        return mock.Mock(
            returncode=returncode,
            stdout=json.dumps(payload),
            stderr="raw dependency detail must not persist",
        )

    @staticmethod
    def _popen_result(payload: dict[str, object] | str, returncode: int = 42) -> mock.Mock:
        process = mock.Mock(pid=424242, returncode=returncode)
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        process.communicate.return_value = (
            stdout,
            "raw dependency detail must not persist",
        )
        process.poll.return_value = returncode
        return process

    @staticmethod
    def _readiness_sample(target_contract: str) -> dict[str, object]:
        return {
            "sample_status": "PASS",
            "target_contract": target_contract,
            "target_selector_hash": "a" * 64,
            "command_contract_hash": "b" * 64,
            "argument_name_set": ["target", "expected_candidate", "expected_stable"],
            "sanitized_environment_key_set": ["PYTHONDONTWRITEBYTECODE"],
            "import_status": "PASS",
            "dependency_status": "PASS",
            "auth_handoff_status": "PASS",
            "target_construction_status": "PASS",
            "health_sample_status": "PASS",
            "readiness_status": "PASS",
            "evidence_trace_safe_summary": "PASS",
            "read_only_command_count": 1,
            "mutation_command_count": 0,
            "valid_sample_count": 1,
            "production_write_count": 0,
            "raw_response_persisted": False,
            "identity_material_persisted": False,
        }

    def test_r482_stage_fixture_is_structured_not_generic_sampler_failure(self) -> None:
        r482_failure = {
            "sample_status": "FAIL",
            "timeout_count": 0,
            "unexpected_5xx_count": 0,
            "auth_failure_count": 0,
            "ui_failure_count": 1,
            "flow_failure_count": 1,
            "evidence_missing_count": 1,
            "trace_missing_count": 1,
            "raw_response_persisted": False,
            "identity_material_persisted": False,
        }
        with mock.patch.object(
            observer.subprocess, "Popen", return_value=self._popen_result(r482_failure)
        ):
            with self.assertRaises(observer.SamplerFailure) as caught:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(caught.exception.category, "FUNCTIONAL_HTTP_FAILURE")
        self.assertNotEqual(caught.exception.category, "SAMPLER_FAILURE")
        self.assertEqual(
            caught.exception.metadata["first_failure_phase"],
            "SAMPLER_FUNCTIONAL_RESULT",
        )

    def test_http_auth_timeout_parse_and_dependency_categories_are_distinct(self) -> None:
        for category in ("HTTP_403", "HTTP_404", "HTTP_5XX", "AUTH_FAILURE"):
            with self.subTest(category=category):
                payload = {"sample_status": "FAIL", "failure_category": category}
                with mock.patch.object(
                    observer.subprocess, "Popen", return_value=self._popen_result(payload)
                ):
                    with self.assertRaises(observer.SamplerFailure) as caught:
                        observer._production_sample(["sampler"], 10)
                self.assertEqual(caught.exception.category, category)

        timeout_process = mock.Mock(pid=424242, returncode=None)
        timeout_process.communicate.side_effect = [
            observer.subprocess.TimeoutExpired("sampler", 10),
            ("", ""),
        ]
        with mock.patch.object(observer.subprocess, "Popen", return_value=timeout_process), mock.patch.object(
            observer,
            "_terminate_sampler_process",
            return_value={
                "child_exit_state": "TERMINATED",
                "child_cancellation_state": "COMPLETED",
                "orphan_child_count": 0,
            },
        ):
            with self.assertRaises(observer.SamplerFailure) as timeout:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(timeout.exception.category, "TIMEOUT")

        invalid = self._popen_result("not-json", returncode=0)
        with mock.patch.object(observer.subprocess, "Popen", return_value=invalid):
            with self.assertRaises(observer.SamplerFailure) as parse:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(parse.exception.category, "JSON_PARSE_FAILURE")

        with mock.patch.object(observer.subprocess, "Popen", side_effect=OSError("raw")):
            with self.assertRaises(observer.SamplerFailure) as dependency:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(
            dependency.exception.category, "DEPENDENCY_OR_SUBPROCESS_DEFECT"
        )

    def test_partial_output_and_sample_zero_fail_closed(self) -> None:
        partial = self._popen_result('{"sample_status":', returncode=0)
        with mock.patch.object(observer.subprocess, "Popen", return_value=partial):
            with self.assertRaises(observer.SamplerFailure) as caught_partial:
                observer._production_sample(
                    ["sampler"],
                    10,
                    required_target_contract="REVISION_FUNCTIONAL",
                )
        self.assertEqual(caught_partial.exception.category, "JSON_PARSE_FAILURE")

        sample_zero = self._readiness_sample("REVISION_FUNCTIONAL")
        sample_zero["valid_sample_count"] = 0
        completed = self._popen_result(sample_zero, returncode=0)
        with mock.patch.object(observer.subprocess, "Popen", return_value=completed):
            with self.assertRaises(observer.SamplerFailure) as caught_zero:
                observer._production_sample(
                    ["sampler"],
                    10,
                    required_target_contract="REVISION_FUNCTIONAL",
                )
        self.assertEqual(
            caught_zero.exception.category,
            "ARGUMENT_OR_SERIALIZATION_DEFECT",
        )
        self.assertEqual(
            caught_zero.exception.metadata["exit_category"],
            "INVALID_VALID_SAMPLE_CONTRACT",
        )

    def test_stable_candidate_only_surface_404_is_non_retryable_stop(self) -> None:
        completed = self._popen_result(
            {"sample_status": "FAIL", "failure_category": "HTTP_404"}
        )
        with mock.patch.object(
            observer.subprocess, "Popen", return_value=completed
        ) as popen:
            with self.assertRaises(observer.SamplerFailure) as caught:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(caught.exception.category, "HTTP_404")
        self.assertFalse(caught.exception.metadata["retryable"])
        self.assertEqual(popen.call_count, 1)

    def test_incomplete_artifact_has_structured_failure_without_raw_message(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r483-incomplete-") as temporary:
            directory = Path(temporary)
            observer._atomic_write_json(
                directory / "start.json",
                {
                    "observation_id": "r483-fixture",
                    "mode": "production",
                    "requested_duration_seconds": 600,
                    "sample_interval_seconds": 50,
                    "maximum_allowed_gap_seconds": 58,
                },
            )
            (directory / "events.ndjson").write_text("", encoding="utf-8")
            failure = observer.SamplerFailure(
                "HTTP_404",
                phase="SAMPLER_FUNCTIONAL_RESULT",
                dependency_class="SAMPLER_CONTRACT",
                exit_category="SAMPLER_EXIT_NONZERO",
                source_line=123,
            )
            result = observer._write_incomplete(
                directory,
                reason="SAMPLER_HTTP_404",
                process_exit_code=42,
                failure_metadata=failure.metadata,
            )
            serialized = json.dumps(result)
            self.assertEqual(result["first_failure"]["failure_category"], "HTTP_404")
            self.assertEqual(result["first_failure"]["source_line"], 123)
            self.assertFalse(result["first_failure"]["retryable"])
            self.assertNotIn("raw dependency detail", serialized)
            self.assertNotIn("Authorization", serialized)
            self.assertNotRegex(serialized, r"[A-Za-z][A-Za-z0-9+.-]*://")

    def test_detached_stage_failure_fixture_writes_http_category(self) -> None:
        payload = json.dumps(
            {"sample_status": "FAIL", "failure_category": "HTTP_404"}
        )
        sampler = [
            sys.executable,
            "-c",
            f"import sys; print({payload!r}); sys.exit(42)",
        ]
        previous = os.environ.get(observer.PRODUCTION_SAMPLER_ENV)
        with tempfile.TemporaryDirectory(prefix="r483-stage-fixture-") as temporary:
            try:
                os.environ[observer.PRODUCTION_SAMPLER_ENV] = json.dumps(sampler)
                launch = observer.start_observation(
                    artifact_root=temporary,
                    observation_id="stage-005-fixture",
                    duration_seconds=1,
                    sample_interval_seconds=1,
                    max_gap_seconds=2,
                    stale_after_seconds=3,
                    mode="production",
                )
                deadline = time.monotonic() + 8
                result: dict[str, object] = {}
                while time.monotonic() < deadline:
                    result = observer.poll_observation(
                        artifact_root=temporary,
                        observation_id=str(launch["observation_id"]),
                    )
                    if result.get("status") != "RUNNING":
                        break
                    time.sleep(0.05)
            finally:
                if previous is None:
                    os.environ.pop(observer.PRODUCTION_SAMPLER_ENV, None)
                else:
                    os.environ[observer.PRODUCTION_SAMPLER_ENV] = previous
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["reason"], "SAMPLER_HTTP_404")
            self.assertEqual(result["first_failure"]["failure_category"], "HTTP_404")
            self.assertFalse(result["completion_marker"])

    def test_service_and_revision_targets_are_separate_and_readiness_fails_closed(self) -> None:
        revision = self._readiness_sample("REVISION_FUNCTIONAL")
        split = self._readiness_sample("SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL")
        observer.validate_production_sample_contract(
            revision, required_target_contract="REVISION_FUNCTIONAL"
        )
        observer.validate_production_sample_contract(
            split,
            required_target_contract="SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL",
        )
        with self.assertRaises(observer.SamplerFailure) as mismatch:
            observer.validate_production_sample_contract(
                revision,
                required_target_contract="SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL",
            )
        self.assertEqual(mismatch.exception.category, "TARGET_ROUTING_CONTRACT_DEFECT")
        failed = dict(split)
        failed["auth_handoff_status"] = "STOP"
        with self.assertRaises(observer.SamplerFailure) as auth:
            observer.validate_production_sample_contract(
                failed,
                required_target_contract="SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL",
            )
        self.assertEqual(auth.exception.category, "AUTH_FAILURE")

    def test_detached_child_contract_uses_absolute_module_and_scrubbed_environment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r483-launch-") as temporary:
            fake_child = mock.Mock(pid=424242, returncode=None)
            sampler = [sys.executable, "-c", "print('{}')"]
            previous_sampler = os.environ.get(observer.PRODUCTION_SAMPLER_ENV)
            previous_identity = os.environ.get(observer.IDENTITY_MATERIAL_ENV)
            os.environ[observer.PRODUCTION_SAMPLER_ENV] = json.dumps(sampler)
            os.environ[observer.IDENTITY_MATERIAL_ENV] = "memory-only-fixture"
            try:
                with mock.patch.object(
                    observer.subprocess, "Popen", return_value=fake_child
                ) as popen:
                    result = observer.start_observation(
                        artifact_root=temporary,
                        observation_id="r483-launch",
                        duration_seconds=1,
                        sample_interval_seconds=1,
                        max_gap_seconds=2,
                        stale_after_seconds=3,
                        mode="production",
                        required_target_contract="REVISION_FUNCTIONAL",
                    )
            finally:
                if previous_sampler is None:
                    os.environ.pop(observer.PRODUCTION_SAMPLER_ENV, None)
                else:
                    os.environ[observer.PRODUCTION_SAMPLER_ENV] = previous_sampler
                if previous_identity is None:
                    os.environ.pop(observer.IDENTITY_MATERIAL_ENV, None)
                else:
                    os.environ[observer.IDENTITY_MATERIAL_ENV] = previous_identity
            command = popen.call_args.args[0]
            options = popen.call_args.kwargs
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(Path(command[2]).resolve(), Path(observer.__file__).resolve())
            self.assertNotIn("cwd", options)
            self.assertNotIn(observer.PRODUCTION_SAMPLER_ENV, options["env"])
            self.assertNotIn(observer.IDENTITY_MATERIAL_ENV, options["env"])
            self.assertEqual(result["status"], "STARTED")

    def test_verified_external_transient_retry_is_bounded(self) -> None:
        transient = observer.SamplerFailure(
            "VERIFIED_EXTERNAL_TRANSIENT",
            phase="SAMPLER_FUNCTIONAL_RESULT",
            dependency_class="SAMPLER_CONTRACT",
            exit_category="SAMPLER_EXIT_NONZERO",
            retryable=True,
        )
        passed = {"sample_status": "PASS"}
        with mock.patch.object(
            observer, "_production_sample_once", side_effect=[transient, passed]
        ) as run_once:
            result = observer._production_sample(["sampler"], 10)
        self.assertEqual(result["sample_status"], "PASS")
        self.assertEqual(run_once.call_count, 2)
        self.assertEqual(observer.MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRIES, 1)
        self.assertLessEqual(observer.MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRY_SECONDS, 10)

    def test_retry_uses_one_monotonic_total_deadline(self) -> None:
        transient = observer.SamplerFailure(
            "VERIFIED_EXTERNAL_TRANSIENT",
            phase="SAMPLER_FUNCTIONAL_RESULT",
            dependency_class="SAMPLER_CONTRACT",
            exit_category="SAMPLER_EXIT_NONZERO",
            retryable=True,
        )
        observed_budgets: list[float] = []

        def run_once(_argv: list[str], timeout_seconds: float) -> dict[str, object]:
            observed_budgets.append(timeout_seconds)
            if len(observed_budgets) == 1:
                raise transient
            return {"sample_status": "PASS"}

        with mock.patch.object(
            observer.time, "monotonic", side_effect=[0.0, 0.0, 5.0, 5.0]
        ), mock.patch.object(observer, "_production_sample_once", side_effect=run_once):
            result = observer._production_sample(["sampler"], 30)
        self.assertEqual(result["sample_status"], "PASS")
        self.assertEqual(observed_budgets, [30.0, 25.0])

    def test_timeout_preserves_phase_counter_and_cancels_process_tree(self) -> None:
        self._stop_cleanup_patch()
        with tempfile.TemporaryDirectory(prefix="r487a-cancel-") as temporary:
            marker = Path(temporary) / "grandchild.txt"
            script = "\n".join(
                (
                    "import json, os, pathlib, subprocess, sys, time",
                    "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                    "pathlib.Path(os.environ['QLIB_TEST_GRANDCHILD_PID_FILE']).write_text(str(grandchild.pid), encoding='utf-8')",
                    "progress = {'current_phase':'HTTP_REQUEST','last_completed_phase':'REQUEST_PREPARATION','phase_timings':[{'phase':'AUTHENTICATION','elapsed_ms':1.0,'status':'PASS'}],'timeout_reason':'NONE','read_only_command_count':2,'mutation_command_count':0,'valid_sample_count':0,'child_exit_state':'RUNNING','child_cancellation_state':'NOT_REQUIRED','orphan_child_count':0}",
                    "pathlib.Path(os.environ['QLIB_SAMPLER_PROGRESS_FILE']).write_text(json.dumps(progress), encoding='utf-8')",
                    "print('{\"sample_status\":', flush=True)",
                    "time.sleep(60)",
                )
            )
            with mock.patch.dict(
                os.environ,
                {"QLIB_TEST_GRANDCHILD_PID_FILE": str(marker)},
                clear=False,
            ):
                with self.assertRaises(observer.SamplerFailure) as caught:
                    observer._production_sample_once(
                        [sys.executable, "-c", script], timeout_seconds=0.75
                    )
            self.assertEqual(caught.exception.category, "TIMEOUT")
            metadata = caught.exception.metadata
            self.assertEqual(metadata["current_phase"], "HTTP_REQUEST")
            self.assertEqual(metadata["last_completed_phase"], "REQUEST_PREPARATION")
            self.assertEqual(metadata["read_only_command_count"], 2)
            self.assertEqual(metadata["mutation_command_count"], 0)
            self.assertEqual(metadata["valid_sample_count"], 0)
            self.assertEqual(metadata["child_cancellation_state"], "COMPLETED")
            self.assertEqual(metadata["orphan_child_count"], 0)
            grandchild_pid = int(marker.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 3.0
            while observer._pid_exists(grandchild_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(observer._pid_exists(grandchild_pid))

    def test_rollback_command_and_post_verification_are_deterministic(self) -> None:
        command = observer.build_stable_rollback_command(
            service="qlib-skillup-runtime",
            project="project-under-test",
            region="asia-northeast1",
            stable_revision="qlib-skillup-runtime-00002-d9g",
        )
        self.assertEqual(command[:4], ["gcloud", "run", "services", "update-traffic"])
        self.assertIn("qlib-skillup-runtime-00002-d9g=100", command)
        serialized = " ".join(command)
        for forbidden in (
            "set-iam-policy",
            "allow-unauthenticated",
            "images delete",
            "quali-admin-domap",
        ):
            self.assertNotIn(forbidden, serialized)
        snapshot = {
            "historical_stable_traffic_percent": 100,
            "other_positive_traffic_count": 0,
            "historical_stable_ready": True,
            "unauthenticated_health_http": 403,
            "authenticated_health_http": 200,
            "normalized_health": " OK ",
            "iam_hash_match": True,
            "public_member_count": 0,
            "authoritative_set_iam_policy_count": 0,
        }
        self.assertEqual(
            observer.evaluate_stable_rollback_verification(snapshot)["status"],
            "PASS",
        )
        snapshot["other_positive_traffic_count"] = 1
        self.assertEqual(
            observer.evaluate_stable_rollback_verification(snapshot)["status"],
            "STOP",
        )

    def test_runbook_records_compatibility_hold_and_reliability_contract(self) -> None:
        text = RUNBOOK_PATH.read_text(encoding="utf-8")
        for fragment in (
            "PRODUCT_SPLIT_TRAFFIC_COMPATIBILITY_DEFECT",
            "SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL",
            "SAMPLER_HTTP_404",
            "MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRIES=1",
            "mutation command is executed and recorded separately",
            "R484 deployment remains NOT_GRANTED",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


class R488C2Win32ProcessTreeFailClosedTests(unittest.TestCase):
    @staticmethod
    def _fake_process(pid: int = 10, handle: int = 7000) -> mock.Mock:
        process = mock.Mock(pid=pid, returncode=None)
        process._handle = handle
        return process

    @staticmethod
    def _sampler_flags() -> int:
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )

    @staticmethod
    def _terminate_exact_handle(
        api: object, handle: int, creation_identity: int
    ) -> None:
        try:
            if int(api.creation_identity(handle)) != creation_identity:  # type: ignore[attr-defined]
                return
            state = observer._wait_windows_process(api, handle, 0)
            if state["state"] == "RUNNING":
                api.terminate(handle)  # type: ignore[attr-defined]
                observer._wait_windows_process(
                    api, handle, observer._WIN32_PROCESS_TREE_WAIT_MS
                )
        except Exception:
            return

    @staticmethod
    def _wait_for_marker(path: Path, timeout_seconds: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not path.exists():
            raise AssertionError("synthetic process marker was not created")

    def test_root_exited_grandchild_alive_with_inherited_pipe(self) -> None:
        nonce = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="r488c2-root-exited-") as temporary:
            marker = Path(temporary) / f"{nonce}.pid"
            child_source = "import time; time.sleep(60)"
            root_source = "\n".join(
                (
                    "import os, pathlib, subprocess, sys",
                    f"child = subprocess.Popen([sys.executable, '-c', {child_source!r}])",
                    "pathlib.Path(os.environ['R488C2_MARKER']).write_text(str(child.pid), encoding='utf-8')",
                )
            )
            environment = dict(os.environ)
            environment["R488C2_MARKER"] = str(marker)
            root = subprocess.Popen(
                [sys.executable, "-c", root_source],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                shell=False,
                creationflags=self._sampler_flags(),
            )
            api = observer._Win32ProcessApi()
            child_handle: int | None = None
            child_identity = 0
            root_identity = int(api.creation_identity(int(root._handle)))
            try:
                self._wait_for_marker(marker)
                child_pid = int(marker.read_text(encoding="utf-8"))
                child_handle = int(api.open_process(child_pid))
                child_identity = int(api.creation_identity(child_handle))
                deadline = time.monotonic() + 3.0
                while root.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertIsNotNone(root.poll())
                with self.assertRaises(subprocess.TimeoutExpired):
                    root.communicate(timeout=0.2)

                result = observer._terminate_sampler_process(
                    root, api=api, nonce=nonce
                )
                root.communicate(timeout=2.0)
                self.assertEqual(result["child_cancellation_state"], "COMPLETED")
                self.assertEqual(result["orphan_verification_status"], "PASS")
                self.assertEqual(result["orphan_child_count"], 0)
                self.assertEqual(
                    observer._wait_windows_process(api, child_handle, 0)["state"],
                    "EXITED",
                )
            finally:
                self._terminate_exact_handle(api, int(root._handle), root_identity)
                if child_handle is not None:
                    self._terminate_exact_handle(api, child_handle, child_identity)
                    api.close_handle(child_handle)
                try:
                    root.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

    def test_detached_fixture_inside_current_job_object_without_breakaway(self) -> None:
        kernel32 = observer.ctypes.WinDLL(  # type: ignore[attr-defined]
            "kernel32", use_last_error=True
        )
        kernel32.IsProcessInJob.argtypes = (
            observer.ctypes.c_void_p,
            observer.ctypes.c_void_p,
            observer.ctypes.POINTER(observer.ctypes.c_int),
        )
        kernel32.IsProcessInJob.restype = observer.ctypes.c_int
        in_job = observer.ctypes.c_int()
        self.assertTrue(
            kernel32.IsProcessInJob(
                observer.ctypes.c_void_p(-1), None, observer.ctypes.byref(in_job)
            )
        )
        self.assertEqual(in_job.value, 1)

        flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )
        self.assertEqual(
            flags & getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000), 0
        )
        child = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            shell=False,
            creationflags=flags,
        )
        self.assertEqual(child.wait(timeout=5.0), 0)
        source = inspect.getsource(observer.start_observation)
        self.assertNotIn("CREATE_BREAKAWAY_FROM_JOB", source)
        self.assertNotIn("_launch_via_wmi", source)

    def test_unrelated_sibling_survives_owned_tree_cleanup(self) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        root = subprocess.Popen(command, creationflags=self._sampler_flags())
        sibling = subprocess.Popen(command, creationflags=self._sampler_flags())
        api = observer._Win32ProcessApi()
        root_identity = int(api.creation_identity(int(root._handle)))
        sibling_identity = int(api.creation_identity(int(sibling._handle)))
        try:
            result = observer._terminate_sampler_process(
                root, api=api, nonce=uuid.uuid4().hex
            )
            self.assertEqual(result["orphan_child_count"], 0)
            self.assertEqual(result["orphan_verification_status"], "PASS")
            self.assertEqual(
                observer._wait_windows_process(api, int(sibling._handle), 0)["state"],
                "RUNNING",
            )
            self.assertEqual(
                int(api.creation_identity(int(sibling._handle))), sibling_identity
            )
        finally:
            self._terminate_exact_handle(api, int(root._handle), root_identity)
            self._terminate_exact_handle(api, int(sibling._handle), sibling_identity)
            for process in (root, sibling):
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

    def test_pid_creation_identity_mismatch_is_not_terminated(self) -> None:
        api = _ScriptedWin32Api(
            [
                {"entries": [(10, 1), (11, 10)]},
                {"entries": [(10, 1), (11, 10)]},
            ]
        )
        api.identity_sequences[11] = [100011, 200011]
        result = observer._terminate_sampler_process(
            self._fake_process(), api=api, nonce="fixture-nonce"
        )
        self.assertEqual(result["child_cancellation_state"], "FAILED")
        self.assertEqual(result["orphan_verification_status"], "NOT_VERIFIED")
        self.assertEqual(result["identity_mismatch_count"], 1)
        self.assertEqual(api.terminate_calls, [])

        query_failure = _ScriptedWin32Api(
            [{"entries": [(10, 1), (11, 10)]}, {"entries": []}]
        )
        query_failure.identity_fail_pids.add(11)
        failed = observer._terminate_sampler_process(
            self._fake_process(), api=query_failure, nonce="fixture-nonce"
        )
        self.assertEqual(failed["cleanup_failure_api"], "GetProcessTimes")
        self.assertEqual(failed["cleanup_failure_phase"], "IDENTITY_QUERY")
        self.assertEqual(failed["orphan_verification_status"], "NOT_VERIFIED")
        self.assertEqual(query_failure.terminate_calls, [])

        root_query_failure = _ScriptedWin32Api()
        root_query_failure.identity_fail_pids.add(10)
        root_failed = observer._terminate_sampler_process(
            self._fake_process(), api=root_query_failure, nonce="fixture-nonce"
        )
        self.assertEqual(root_failed["cleanup_failure_api"], "GetProcessTimes")
        self.assertEqual(root_failed["cleanup_failure_phase"], "ROOT_IDENTITY")
        self.assertEqual(root_failed["process_tree_snapshot_status"], "NOT_STARTED")

        reused_pid = _ScriptedWin32Api(
            [
                {"entries": [(10, 1), (11, 10)]},
                {"entries": [(10, 1), (11, 99)]},
            ]
        )
        reused_pid.identity_sequences[11] = [200011]
        reused = observer._terminate_sampler_process(
            self._fake_process(), api=reused_pid, nonce="fixture-nonce"
        )
        self.assertEqual(reused["cleanup_failure_reason"], "IDENTITY_MISMATCH")
        self.assertEqual(reused["cleanup_failure_phase"], "IDENTITY_VERIFICATION")
        self.assertEqual(reused["orphan_verification_status"], "NOT_VERIFIED")
        self.assertEqual(reused_pid.terminate_calls, [])

    def test_create_toolhelp_snapshot_failure_is_incomplete(self) -> None:
        api = _ScriptedWin32Api()
        api.snapshot_exception = observer._Win32ApiFailure(
            "CreateToolhelp32Snapshot", 5, "SNAPSHOT_CREATE"
        )
        result = observer._enumerate_windows_processes(api)
        self.assertFalse(result["complete"])
        self.assertEqual(result["cleanup_failure_api"], "CreateToolhelp32Snapshot")
        self.assertEqual(result["cleanup_failure_reason"], "INCOMPLETE_SNAPSHOT")
        self.assertEqual(api.closed_handles, [])

    def test_process_first_no_more_files_is_complete_empty(self) -> None:
        api = _ScriptedWin32Api(
            [{"entries": [], "first_error": observer._WIN32_ERROR_NO_MORE_FILES}]
        )
        result = observer._enumerate_windows_processes(api)
        self.assertTrue(result["complete"])
        self.assertEqual(result["entries"], [])
        self.assertEqual(len(api.closed_handles), 1)

    def test_process_first_unexpected_error_is_incomplete(self) -> None:
        api = _ScriptedWin32Api([{"entries": [], "first_error": 5}])
        result = observer._enumerate_windows_processes(api)
        self.assertFalse(result["complete"])
        self.assertEqual(result["cleanup_failure_api"], "Process32FirstW")
        self.assertEqual(result["cleanup_failure_code"], 5)

        raised = _ScriptedWin32Api(
            [{"entries": [], "first_exception": RuntimeError("raw first detail") }]
        )
        raised_result = observer._enumerate_windows_processes(
            raised, phase="FINAL_VERIFICATION"
        )
        self.assertFalse(raised_result["complete"])
        self.assertEqual(raised_result["cleanup_failure_api"], "Process32FirstW")
        self.assertEqual(
            raised_result["cleanup_failure_phase"], "FINAL_VERIFICATION"
        )
        self.assertNotIn("raw first detail", json.dumps(raised_result))

    def test_process_next_no_more_files_is_complete(self) -> None:
        api = _ScriptedWin32Api([{"entries": [(10, 1)]}])
        result = observer._enumerate_windows_processes(api)
        self.assertTrue(result["complete"])
        self.assertEqual(result["entries"], [(10, 1)])
        self.assertEqual(len(api.closed_handles), 1)

        close_failure = _ScriptedWin32Api([{"entries": [(10, 1)]}])
        close_failure.close_exception_handles.add(1001)
        failed = observer._enumerate_windows_processes(close_failure)
        self.assertFalse(failed["complete"])
        self.assertEqual(failed["cleanup_failure_api"], "CloseHandle")
        self.assertEqual(failed["cleanup_failure_reason"], "HANDLE_CLOSE_FAILED")

        owned_handle_close_failure = _ScriptedWin32Api(
            [
                {"entries": [(10, 1), (11, 10)]},
                {"entries": [(10, 1), (11, 10)]},
                {"entries": [(10, 1)]},
            ]
        )
        owned_handle_close_failure.close_exception_handles.add(20011)
        cleanup_failed = observer._terminate_sampler_process(
            self._fake_process(),
            api=owned_handle_close_failure,
            nonce="fixture-nonce",
        )
        self.assertEqual(cleanup_failed["cleanup_failure_api"], "CloseHandle")
        self.assertEqual(
            cleanup_failed["cleanup_failure_reason"], "HANDLE_CLOSE_FAILED"
        )
        self.assertEqual(
            cleanup_failed["orphan_verification_status"], "NOT_VERIFIED"
        )

    def test_process_next_unexpected_error_is_incomplete(self) -> None:
        api = _ScriptedWin32Api([{"entries": [(10, 1)], "next_error": 5}])
        result = observer._enumerate_windows_processes(api)
        self.assertFalse(result["complete"])
        self.assertEqual(result["cleanup_failure_api"], "Process32NextW")
        self.assertEqual(result["cleanup_failure_reason"], "INCOMPLETE_SNAPSHOT")

        raised = _ScriptedWin32Api(
            [{"entries": [(10, 1)], "next_exception": RuntimeError("raw next detail")}]
        )
        raised_result = observer._enumerate_windows_processes(
            raised, phase="FINAL_VERIFICATION"
        )
        self.assertFalse(raised_result["complete"])
        self.assertEqual(raised_result["cleanup_failure_api"], "Process32NextW")
        self.assertEqual(
            raised_result["cleanup_failure_phase"], "FINAL_VERIFICATION"
        )
        self.assertNotIn("raw next detail", json.dumps(raised_result))

    def test_wait_object_zero_is_exited(self) -> None:
        api = _ScriptedWin32Api()
        api.wait_sequences[10] = [(observer._WIN32_WAIT_OBJECT_0, 0)]
        self.assertEqual(
            observer._wait_windows_process(api, 7000, 0)["state"], "EXITED"
        )

    def test_wait_timeout_is_running(self) -> None:
        api = _ScriptedWin32Api()
        api.wait_sequences[10] = [(observer._WIN32_WAIT_TIMEOUT, 0)]
        self.assertEqual(
            observer._wait_windows_process(api, 7000, 0)["state"], "RUNNING"
        )

    def test_wait_failed_is_cleanup_failed(self) -> None:
        api = _ScriptedWin32Api()
        api.wait_sequences[10] = [(observer._WIN32_WAIT_FAILED, 6)]
        result = observer._wait_windows_process(api, 7000, 0)
        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(result["cleanup_failure_reason"], "WAIT_FAILED")
        self.assertEqual(result["cleanup_failure_code"], 6)

    def test_unexpected_wait_result_is_cleanup_failed(self) -> None:
        api = _ScriptedWin32Api()
        api.wait_sequences[10] = [(17, 0)]
        result = observer._wait_windows_process(api, 7000, 0)
        self.assertEqual(result["state"], "FAILED_UNEXPECTED_WAIT_RESULT")
        self.assertEqual(result["cleanup_failure_reason"], "UNEXPECTED_WAIT_RESULT")

    def test_open_process_failure_is_not_verified(self) -> None:
        api = _ScriptedWin32Api(
            [{"entries": [(10, 1), (11, 10)]}, {"entries": []}]
        )
        api.open_fail_pids.add(11)
        result = observer._terminate_sampler_process(
            self._fake_process(), api=api, nonce="fixture-nonce"
        )
        self.assertEqual(result["child_cancellation_state"], "FAILED")
        self.assertEqual(result["cleanup_failure_api"], "OpenProcess")
        self.assertEqual(result["orphan_child_count"], 1)
        self.assertEqual(api.terminate_calls, [])

    def test_terminate_process_failure_is_not_verified(self) -> None:
        api = _ScriptedWin32Api(
            [{"entries": [(10, 1)]}, {"entries": [(10, 1)]}]
        )
        api.wait_sequences[10] = [(observer._WIN32_WAIT_TIMEOUT, 0)]
        api.terminate_fail_pids.add(10)
        result = observer._terminate_sampler_process(
            self._fake_process(), api=api, nonce="fixture-nonce"
        )
        self.assertEqual(result["child_cancellation_state"], "FAILED")
        self.assertEqual(result["cleanup_failure_api"], "TerminateProcess")
        self.assertEqual(result["orphan_verification_status"], "NOT_VERIFIED")

    def test_win32_exception_is_sanitized(self) -> None:
        api = _ScriptedWin32Api()
        api.snapshot_exception = OSError(
            5, "raw https://forbidden.example Authorization token C:\\private"
        )
        result = observer._enumerate_windows_processes(api)
        serialized = json.dumps(result, sort_keys=True)
        self.assertFalse(result["complete"])
        self.assertNotIn("forbidden.example", serialized)
        self.assertNotIn("Authorization", serialized)
        self.assertNotIn("private", serialized)
        self.assertEqual(result["cleanup_failure_reason"], "INCOMPLETE_SNAPSHOT")

    def test_incomplete_enumeration_cannot_produce_orphan_zero(self) -> None:
        api = _ScriptedWin32Api([{"entries": [], "first_error": 5}])
        result = observer._terminate_sampler_process(
            self._fake_process(), api=api, nonce="fixture-nonce"
        )
        self.assertEqual(result["process_tree_snapshot_status"], "INCOMPLETE_SNAPSHOT")
        self.assertEqual(result["orphan_verification_status"], "NOT_VERIFIED")
        self.assertNotEqual(result["orphan_child_count"], 0)

        final_incomplete = _ScriptedWin32Api(
            [
                {"entries": [(10, 1)]},
                {"entries": [], "first_error": 5},
            ]
        )
        late_failure = observer._terminate_sampler_process(
            self._fake_process(), api=final_incomplete, nonce="fixture-nonce"
        )
        self.assertEqual(
            late_failure["process_tree_snapshot_status"], "INCOMPLETE_SNAPSHOT"
        )
        self.assertEqual(
            late_failure["cleanup_failure_phase"], "FINAL_VERIFICATION"
        )
        self.assertEqual(late_failure["orphan_child_count"], 1)

    def test_cleanup_failure_cannot_be_promoted_to_observer_pass(self) -> None:
        process = mock.Mock(pid=10, returncode=None)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("sampler", 0.1),
            ("", ""),
        ]
        failed_cleanup = {
            "child_exit_state": "UNKNOWN",
            "child_cancellation_state": "FAILED",
            "orphan_child_count": 1,
            "orphan_verification_status": "NOT_VERIFIED",
            "process_tree_cleanup_status": "CLEANUP_FAILED",
            "process_tree_snapshot_status": "INCOMPLETE_SNAPSHOT",
            "owned_process_count": 1,
            "terminated_process_count": 0,
            "identity_mismatch_count": 0,
            "cleanup_failure_api": "Process32NextW",
            "cleanup_failure_code": 5,
            "cleanup_failure_phase": "SNAPSHOT_NEXT",
            "cleanup_failure_reason": "INCOMPLETE_SNAPSHOT",
        }
        with mock.patch.object(observer.subprocess, "Popen", return_value=process), mock.patch.object(
            observer, "_terminate_sampler_process", return_value=failed_cleanup
        ):
            with self.assertRaises(observer.SamplerFailure) as caught:
                observer._production_sample_once(["sampler"], timeout_seconds=0.1)
        self.assertEqual(caught.exception.metadata["child_cancellation_state"], "FAILED")
        self.assertEqual(caught.exception.metadata["orphan_child_count"], 1)
        self.assertEqual(
            caught.exception.metadata["orphan_verification_status"], "NOT_VERIFIED"
        )

        valid_sample = R483SamplerAndRollbackReliabilityTests._readiness_sample(
            "REVISION_FUNCTIONAL"
        )
        successful_process = mock.Mock(pid=10, returncode=0)
        successful_process.communicate.return_value = (
            json.dumps(valid_sample),
            "",
        )
        with mock.patch.object(
            observer.subprocess, "Popen", return_value=successful_process
        ), mock.patch.object(
            observer, "_terminate_sampler_process", return_value=failed_cleanup
        ):
            with self.assertRaises(observer.SamplerFailure) as cleanup_only_failure:
                observer._production_sample_once(["sampler"], timeout_seconds=1.0)
        self.assertEqual(
            cleanup_only_failure.exception.metadata["exit_category"],
            "PROCESS_TREE_CLEANUP_FAILED",
        )
        self.assertEqual(
            cleanup_only_failure.exception.metadata["orphan_verification_status"],
            "NOT_VERIFIED",
        )

        io_process = mock.Mock(pid=10, returncode=None)
        io_process.communicate.side_effect = OSError(
            5, "raw post-spawn https://forbidden.example token"
        )
        completed_cleanup = dict(failed_cleanup)
        completed_cleanup.update(
            {
                "child_exit_state": "TERMINATED",
                "child_cancellation_state": "COMPLETED",
                "orphan_child_count": 0,
                "orphan_verification_status": "PASS",
                "process_tree_cleanup_status": "CLEANUP_COMPLETED",
                "process_tree_snapshot_status": "COMPLETE",
                "terminated_process_count": 1,
                "cleanup_failure_api": "NONE",
                "cleanup_failure_code": 0,
                "cleanup_failure_phase": "NONE",
                "cleanup_failure_reason": "NONE",
            }
        )
        with mock.patch.object(
            observer.subprocess, "Popen", return_value=io_process
        ), mock.patch.object(
            observer,
            "_terminate_sampler_process",
            return_value=completed_cleanup,
        ) as cleanup_call:
            with self.assertRaises(observer.SamplerFailure) as io_failure:
                observer._production_sample_once(["sampler"], timeout_seconds=1.0)
        self.assertEqual(cleanup_call.call_count, 1)
        self.assertEqual(
            io_failure.exception.metadata["exit_category"],
            "SUBPROCESS_POST_SPAWN_IO_FAILURE",
        )
        self.assertNotIn("forbidden.example", json.dumps(io_failure.exception.metadata))

    def test_marker_owned_residual_process_count_is_zero(self) -> None:
        nonce = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="r488c2-marker-") as temporary:
            marker = Path(temporary) / f"{nonce}.pid"
            child_source = "import time; time.sleep(60)"
            root_source = "\n".join(
                (
                    "import os, pathlib, subprocess, sys, time",
                    f"child = subprocess.Popen([sys.executable, '-c', {child_source!r}])",
                    "pathlib.Path(os.environ['R488C2_MARKER']).write_text(str(child.pid), encoding='utf-8')",
                    "time.sleep(60)",
                )
            )
            environment = dict(os.environ)
            environment["R488C2_MARKER"] = str(marker)
            root = subprocess.Popen(
                [sys.executable, "-c", root_source],
                env=environment,
                creationflags=self._sampler_flags(),
            )
            api = observer._Win32ProcessApi()
            child_handle: int | None = None
            child_identity = 0
            root_identity = int(api.creation_identity(int(root._handle)))
            try:
                self._wait_for_marker(marker)
                child_pid = int(marker.read_text(encoding="utf-8"))
                child_handle = int(api.open_process(child_pid))
                child_identity = int(api.creation_identity(child_handle))
                result = observer._terminate_sampler_process(
                    root, api=api, nonce=nonce
                )
                self.assertEqual(result["orphan_child_count"], 0)
                self.assertEqual(result["orphan_verification_status"], "PASS")
                self.assertEqual(
                    observer._wait_windows_process(api, int(root._handle), 0)["state"],
                    "EXITED",
                )
                self.assertEqual(
                    observer._wait_windows_process(api, child_handle, 0)["state"],
                    "EXITED",
                )
            finally:
                self._terminate_exact_handle(api, int(root._handle), root_identity)
                if child_handle is not None:
                    self._terminate_exact_handle(api, child_handle, child_identity)
                    api.close_handle(child_handle)
                try:
                    root.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        test_ids: list[str] = []

        def collect(item: unittest.TestSuite | unittest.TestCase) -> None:
            if isinstance(item, unittest.TestCase):
                test_ids.append(item.id())
                return
            for child in item:
                collect(child)

        collect(suite)
        legacy_ids = sorted(
            test_id
            for test_id in test_ids
            if ".R488C2Win32ProcessTreeFailClosedTests." not in test_id
        )
        new_ids = [
            test_id
            for test_id in test_ids
            if ".R488C2Win32ProcessTreeFailClosedTests." in test_id
        ]
        self.assertEqual(len(legacy_ids), 38)
        self.assertEqual(
            hashlib.sha256(("\n".join(legacy_ids) + "\n").encode("utf-8")).hexdigest(),
            "cf17b582f731652e97f8a94493d56e764b3a6e60e0acf7f9f9cf47427c78fb2c",
        )
        self.assertEqual(len(new_ids), 19)


if __name__ == "__main__":
    unittest.main()
