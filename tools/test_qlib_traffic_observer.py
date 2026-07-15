from __future__ import annotations

import inspect
import json
import os
import re
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


if __name__ == "__main__":
    unittest.main()
