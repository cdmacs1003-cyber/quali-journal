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

    def _start_short(self, prefix: str, *, duration: float = 0.45) -> dict[str, object]:
        observation_id = self._new_id(prefix)
        return observer.start_observation(
            artifact_root=self.root,
            observation_id=observation_id,
            duration_seconds=duration,
            sample_interval_seconds=0.05,
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
            launch = self._start_short(f"race-{index}", duration=0.2)
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


if __name__ == "__main__":
    unittest.main()
