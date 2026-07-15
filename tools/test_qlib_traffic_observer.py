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
    @staticmethod
    def _completed(payload: dict[str, object], returncode: int = 42) -> mock.Mock:
        return mock.Mock(
            returncode=returncode,
            stdout=json.dumps(payload),
            stderr="raw dependency detail must not persist",
        )

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
            observer.subprocess, "run", return_value=self._completed(r482_failure)
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
                    observer.subprocess, "run", return_value=self._completed(payload)
                ):
                    with self.assertRaises(observer.SamplerFailure) as caught:
                        observer._production_sample(["sampler"], 10)
                self.assertEqual(caught.exception.category, category)

        with mock.patch.object(
            observer.subprocess,
            "run",
            side_effect=observer.subprocess.TimeoutExpired("sampler", 10),
        ):
            with self.assertRaises(observer.SamplerFailure) as timeout:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(timeout.exception.category, "TIMEOUT")

        invalid = mock.Mock(returncode=0, stdout="not-json", stderr="private detail")
        with mock.patch.object(observer.subprocess, "run", return_value=invalid):
            with self.assertRaises(observer.SamplerFailure) as parse:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(parse.exception.category, "JSON_PARSE_FAILURE")

        with mock.patch.object(observer.subprocess, "run", side_effect=OSError("raw")):
            with self.assertRaises(observer.SamplerFailure) as dependency:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(
            dependency.exception.category, "DEPENDENCY_OR_SUBPROCESS_DEFECT"
        )

    def test_stable_candidate_only_surface_404_is_non_retryable_stop(self) -> None:
        completed = self._completed(
            {"sample_status": "FAIL", "failure_category": "HTTP_404"}
        )
        with mock.patch.object(
            observer.subprocess, "run", return_value=completed
        ) as run:
            with self.assertRaises(observer.SamplerFailure) as caught:
                observer._production_sample(["sampler"], 10)
        self.assertEqual(caught.exception.category, "HTTP_404")
        self.assertFalse(caught.exception.metadata["retryable"])
        self.assertEqual(run.call_count, 1)

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
        transient = self._completed(
            {
                "sample_status": "FAIL",
                "failure_category": "VERIFIED_EXTERNAL_TRANSIENT",
            }
        )
        passed = self._completed({"sample_status": "PASS"}, returncode=0)
        with mock.patch.object(
            observer.subprocess, "run", side_effect=[transient, passed]
        ) as run:
            result = observer._production_sample(["sampler"], 10)
        self.assertEqual(result["sample_status"], "PASS")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(observer.MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRIES, 1)
        self.assertLessEqual(observer.MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRY_SECONDS, 10)

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


if __name__ == "__main__":
    unittest.main()
