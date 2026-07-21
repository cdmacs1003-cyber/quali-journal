"""Focused tests for the Linux-native supervisor safety contract."""

from __future__ import annotations

import json
import io
import inspect
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tools import qlib_linux_acceptance_test_tiers as platform_tiers
from tools import qlib_linux_process_supervisor as linux_supervisor


class EventContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = "SYNTHETIC_RUN"
        self.digest = "a" * 64

    def test_contiguous_bound_stream_completes(self) -> None:
        validator = linux_supervisor.BoundEventValidator(self.run_id, self.digest)
        validator.accept(linux_supervisor._bound_event(self.run_id, self.digest, 0, "READY"))
        validator.accept(linux_supervisor._bound_event(self.run_id, self.digest, 1, "SAMPLE"))
        validator.accept(linux_supervisor._bound_event(self.run_id, self.digest, 2, "COMPLETE"))
        validator.require_complete()
        self.assertRegex(validator.digest, r"^[a-f0-9]{64}$")

    def test_duplicate_event_fails_closed(self) -> None:
        validator = linux_supervisor.BoundEventValidator(self.run_id, self.digest)
        event = linux_supervisor._bound_event(self.run_id, self.digest, 0, "READY")
        validator.accept(event)
        with self.assertRaises(linux_supervisor.EventContractError):
            validator.accept(event)

    def test_reordered_and_missing_events_fail_closed(self) -> None:
        validator = linux_supervisor.BoundEventValidator(self.run_id, self.digest)
        with self.assertRaises(linux_supervisor.EventContractError):
            validator.accept(linux_supervisor._bound_event(self.run_id, self.digest, 1, "SAMPLE"))
        second = linux_supervisor.BoundEventValidator(self.run_id, self.digest)
        second.accept(linux_supervisor._bound_event(self.run_id, self.digest, 0, "READY"))
        with self.assertRaises(linux_supervisor.EventContractError):
            second.require_complete()

    def test_wrong_run_config_and_role_fail_closed(self) -> None:
        variants = (
            linux_supervisor._bound_event("WRONG", self.digest, 0, "READY"),
            linux_supervisor._bound_event(self.run_id, "b" * 64, 0, "READY"),
            {**linux_supervisor._bound_event(self.run_id, self.digest, 0, "READY"), "role": "POLLER"},
        )
        for event in variants:
            with self.subTest(event=event["role"]):
                validator = linux_supervisor.BoundEventValidator(self.run_id, self.digest)
                with self.assertRaises(linux_supervisor.EventContractError):
                    validator.accept(event)


class RegistryAndPublicationTests(unittest.TestCase):
    def _identity(self) -> linux_supervisor.ProcessIdentity:
        return linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")

    def test_registry_is_append_only_then_scrubbed(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        identity = self._identity()
        registry.register("WORKER_ROOT", identity)
        self.assertEqual(registry.residual_count, 1)
        with self.assertRaises(linux_supervisor.LinuxSupervisorError):
            registry.register("WORKER_ROOT", self._identity())
        registry.mark_reaped("WORKER_ROOT")
        self.assertEqual(registry.residual_count, 0)
        registry.clear()
        self.assertTrue(registry.cleared)

    def test_scope_change_remains_residual_until_exact_identity_is_absent(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        registry.register("DESCENDANT_001", self._identity())
        with mock.patch.object(
            linux_supervisor,
            "_identity_status",
            return_value=linux_supervisor.IdentityStatus.SCOPE_CHANGED,
        ):
            self.assertEqual(
                linux_supervisor._mark_absent_registrations_reaped(registry), 0
            )
        self.assertEqual(registry.residual_count, 1)
        self.assertEqual(registry.identity_contradiction_count, 1)
        self.assertEqual(registry.scope_changed_count, 1)
        with mock.patch.object(
            linux_supervisor,
            "_identity_status",
            return_value=linux_supervisor.IdentityStatus.ABSENT,
        ):
            self.assertEqual(
                linux_supervisor._mark_absent_registrations_reaped(registry), 1
            )
        self.assertEqual(registry.residual_count, 0)
        self.assertEqual(registry.identity_contradiction_count, 1)
        registry.clear()

    def test_pid_reuse_is_not_counted_as_reaped(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        registry.register("DESCENDANT_001", self._identity())
        with mock.patch.object(
            linux_supervisor,
            "_identity_status",
            return_value=linux_supervisor.IdentityStatus.PID_REUSED,
        ):
            self.assertEqual(
                linux_supervisor._mark_absent_registrations_reaped(registry), 0
            )
        self.assertEqual(registry.residual_count, 0)
        self.assertEqual(registry.identity_contradiction_count, 1)
        registry.clear()

    def test_scope_changed_identity_without_pidfd_is_never_raw_pid_signalled(self) -> None:
        identity = self._identity()
        self.assertIsNone(identity.pidfd)
        with mock.patch.object(
            linux_supervisor,
            "_identity_status",
            return_value=linux_supervisor.IdentityStatus.SCOPE_CHANGED,
        ), mock.patch.object(linux_supervisor.os, "kill") as raw_kill:
            result = linux_supervisor._safe_signal_owned_process(
                identity, 15, supervisor_process_id=999
            )
        self.assertEqual(result, "PIDFD_EXACT_SIGNAL_UNAVAILABLE_NO_SIGNAL")
        raw_kill.assert_not_called()

    def test_unregistered_unresponsive_direct_child_escalates_and_reaps(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.done = False
                self.terminated = 0
                self.killed = 0

            def poll(self):
                return 0 if self.done else None

            def terminate(self) -> None:
                self.terminated += 1

            def kill(self) -> None:
                self.killed += 1
                self.done = True

            def wait(self, timeout=None):
                del timeout
                if not self.done:
                    raise subprocess.TimeoutExpired("synthetic", 0.5)
                return 0

        process = FakeProcess()
        clear, term_sent, kill_sent = (
            linux_supervisor._cleanup_unregistered_direct_child(process)
        )
        self.assertTrue(clear)
        self.assertTrue(term_sent)
        self.assertTrue(kill_sent)
        self.assertEqual(process.terminated, 1)
        self.assertEqual(process.killed, 1)

    def test_failed_startup_still_drains_descendant_when_root_wait_unresolved(self) -> None:
        root = linux_supervisor.ProcessIdentity(201, 200, 201, 201, 300, "S")
        child = linux_supervisor.ProcessIdentity(202, 201, 202, 202, 301, "S")

        class FakeProcess:
            done = False

            def poll(self):
                return 0 if self.done else None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("synthetic", timeout)

        process = FakeProcess()
        descendant_zero = False
        clock = 0.0

        def monotonic() -> float:
            nonlocal clock
            clock += 0.25
            return clock

        def collect(owner, retained) -> None:
            del owner
            retained[(child.process_id, child.start_ticks)] = child

        def group_signal(identity, signal_number, **kwargs):
            del identity, kwargs
            if signal_number == getattr(linux_supervisor.signal, "SIGKILL", 9):
                process.done = True
            return "OWNED_GROUP_SIGNALLED"

        def set_signal(retained, signal_number, **kwargs):
            nonlocal descendant_zero
            del retained, kwargs
            if signal_number == getattr(linux_supervisor.signal, "SIGKILL", 9):
                descendant_zero = True
            return 1

        def status(identity):
            if identity is root:
                return (
                    linux_supervisor.IdentityStatus.ABSENT
                    if process.done
                    else linux_supervisor.IdentityStatus.MATCH
                )
            return (
                linux_supervisor.IdentityStatus.ABSENT
                if descendant_zero
                else linux_supervisor.IdentityStatus.MATCH
            )

        with mock.patch.object(
            linux_supervisor, "_collect_startup_owned_tree", side_effect=collect
        ), mock.patch.object(
            linux_supervisor, "_safe_signal_owned_group", side_effect=group_signal
        ) as group_mock, mock.patch.object(
            linux_supervisor, "_signal_startup_owned_set", side_effect=set_signal
        ), mock.patch.object(
            linux_supervisor, "_identity_status", side_effect=status
        ), mock.patch.object(
            linux_supervisor.time, "monotonic", side_effect=monotonic
        ), mock.patch.object(linux_supervisor.time, "sleep"):
            self.assertFalse(
                linux_supervisor._cleanup_failed_startup_tree(process, root)
            )
        self.assertEqual(
            [call.args[1] for call in group_mock.call_args_list],
            [
                linux_supervisor.signal.SIGTERM,
                getattr(linux_supervisor.signal, "SIGKILL", 9),
            ],
        )

    def test_identity_present_but_unregistered_cannot_authorize_group_cleanup(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        identity = self._identity()
        self.assertFalse(
            linux_supervisor._worker_group_cleanup_is_authorized(registry, identity)
        )
        registry.register("WORKER_ROOT", identity)
        self.assertTrue(
            linux_supervisor._worker_group_cleanup_is_authorized(registry, identity)
        )
        registry.clear()

    def test_registered_supervisor_handle_is_waited_and_scrubbed(self) -> None:
        class CompletedProcess:
            def poll(self):
                return 0

            def wait(self, timeout=None):
                del timeout
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "OBS"
            process = CompletedProcess()
            linux_supervisor._register_supervisor_handle(directory, process)
            key = linux_supervisor._supervisor_handle_key(directory)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with linux_supervisor._SUPERVISOR_HANDLE_LOCK:
                    present = key in linux_supervisor._SUPERVISOR_HANDLES
                if not present:
                    break
                time.sleep(0.005)
            self.assertFalse(present)

    def test_discovery_retains_original_scope_for_escaped_descendant(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        root = linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")
        supervisor = linux_supervisor.ProcessIdentity(100, 1, 100, 100, 100, "S")
        escaped = linux_supervisor.ProcessIdentity(102, 101, 102, 102, 201, "S")
        registry.register("WORKER_ROOT", root)

        def read_identity(process_id: int, *, open_pidfd: bool = False):
            del open_pidfd
            if process_id == root.process_id:
                return root
            if process_id == escaped.process_id:
                return linux_supervisor.ProcessIdentity(
                    escaped.process_id,
                    escaped.parent_process_id,
                    escaped.process_group_id,
                    escaped.session_id,
                    escaped.start_ticks,
                    escaped.state,
                )
            raise linux_supervisor.LinuxSupervisorError("PROC_IDENTITY_NOT_AVAILABLE")

        with mock.patch.object(
            linux_supervisor, "_proc_all_identities", return_value=[root, escaped]
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", side_effect=read_identity
        ):
            self.assertEqual(
                linux_supervisor._discover_and_register_owned_group(
                    registry,
                    root,
                    production_mode=False,
                    supervisor=supervisor,
                    subreaper_enabled=False,
                ),
                1,
            )
        retained = registry.active_identity("DESCENDANT_001")
        self.assertIsNotNone(retained)
        assert retained is not None
        self.assertEqual(retained.process_group_id, root.process_group_id)
        self.assertEqual(retained.session_id, root.session_id)
        self.assertEqual(registry.scope_changed_count, 1)
        registry.clear()

    def test_reused_registered_parent_cannot_authorize_unrelated_child(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        root = linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")
        retained_parent = linux_supervisor.ProcessIdentity(
            202, 101, 101, 101, 201, "S"
        )
        reused_parent = linux_supervisor.ProcessIdentity(
            202, 1, 202, 202, 999, "S"
        )
        unrelated_child = linux_supervisor.ProcessIdentity(
            303, 202, 303, 303, 1000, "S"
        )
        supervisor = linux_supervisor.ProcessIdentity(100, 1, 100, 100, 100, "S")
        registry.register("WORKER_ROOT", root)
        registry.register("DESCENDANT_001", retained_parent)

        def read_identity(process_id: int, *, open_pidfd: bool = False):
            del open_pidfd
            if process_id == reused_parent.process_id:
                return reused_parent
            if process_id == unrelated_child.process_id:
                return unrelated_child
            raise linux_supervisor.LinuxSupervisorError(
                "PROC_IDENTITY_NOT_AVAILABLE"
            )

        with mock.patch.object(
            linux_supervisor,
            "_proc_all_identities",
            return_value=[unrelated_child],
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", side_effect=read_identity
        ):
            registered = linux_supervisor._discover_and_register_owned_group(
                registry,
                root,
                production_mode=False,
                supervisor=supervisor,
                subreaper_enabled=False,
            )

        self.assertEqual(registered, 0)
        self.assertIsNone(registry.active_identity("DESCENDANT_002"))
        self.assertEqual(registry.registration_count("DESCENDANT_"), 1)
        registry.clear()

    def test_current_registered_parent_authorizes_its_current_child(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        root = linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")
        retained_parent = linux_supervisor.ProcessIdentity(
            202, 101, 101, 101, 201, "S"
        )
        child = linux_supervisor.ProcessIdentity(303, 202, 303, 303, 202, "S")
        supervisor = linux_supervisor.ProcessIdentity(100, 1, 100, 100, 100, "S")
        registry.register("WORKER_ROOT", root)
        registry.register("DESCENDANT_001", retained_parent)

        def read_identity(process_id: int, *, open_pidfd: bool = False):
            del open_pidfd
            if process_id == retained_parent.process_id:
                return retained_parent
            if process_id == child.process_id:
                return child
            raise linux_supervisor.LinuxSupervisorError(
                "PROC_IDENTITY_NOT_AVAILABLE"
            )

        with mock.patch.object(
            linux_supervisor, "_proc_all_identities", return_value=[child]
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", side_effect=read_identity
        ):
            registered = linux_supervisor._discover_and_register_owned_group(
                registry,
                root,
                production_mode=False,
                supervisor=supervisor,
                subreaper_enabled=False,
            )

        self.assertEqual(registered, 1)
        self.assertEqual(registry.registration_count("DESCENDANT_"), 2)
        retained_child = registry.active_identity("DESCENDANT_002")
        self.assertIsNotNone(retained_child)
        assert retained_child is not None
        self.assertEqual(retained_child.process_id, child.process_id)
        registry.clear()

    def test_candidate_parent_change_between_snapshot_and_read_fails_closed(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        root = linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")
        snapshot = linux_supervisor.ProcessIdentity(303, 101, 101, 101, 202, "S")
        reparented = linux_supervisor.ProcessIdentity(303, 1, 101, 101, 202, "S")
        supervisor = linux_supervisor.ProcessIdentity(100, 1, 100, 100, 100, "S")
        registry.register("WORKER_ROOT", root)

        with mock.patch.object(
            linux_supervisor, "_proc_all_identities", return_value=[snapshot]
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", return_value=reparented
        ):
            with self.assertRaisesRegex(
                linux_supervisor.LinuxSupervisorError,
                "OWNED_DESCENDANT_IDENTITY_CHANGED",
            ):
                linux_supervisor._discover_and_register_owned_group(
                    registry,
                    root,
                    production_mode=False,
                    supervisor=supervisor,
                    subreaper_enabled=False,
                )
        self.assertEqual(registry.registration_count("DESCENDANT_"), 0)
        registry.clear()

    def test_unrelated_disappearing_process_is_ignored_without_registration(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        root = linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")
        unrelated = linux_supervisor.ProcessIdentity(303, 1, 303, 303, 202, "S")
        supervisor = linux_supervisor.ProcessIdentity(100, 1, 100, 100, 100, "S")
        registry.register("WORKER_ROOT", root)

        with mock.patch.object(
            linux_supervisor, "_proc_all_identities", return_value=[unrelated]
        ), mock.patch.object(
            linux_supervisor,
            "_read_proc_identity",
            side_effect=linux_supervisor.LinuxSupervisorError(
                "PROC_IDENTITY_NOT_AVAILABLE"
            ),
        ):
            registered = linux_supervisor._discover_and_register_owned_group(
                registry,
                root,
                production_mode=False,
                supervisor=supervisor,
                subreaper_enabled=False,
            )

        self.assertEqual(registered, 0)
        self.assertEqual(registry.registration_count("DESCENDANT_"), 0)
        self.assertEqual(registry.residual_count, 1)
        registry.clear()

    def test_parent_exit_after_snapshot_proof_cannot_drop_owned_child(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        root = linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")
        parent = linux_supervisor.ProcessIdentity(202, 101, 101, 101, 201, "S")
        child = linux_supervisor.ProcessIdentity(303, 202, 303, 303, 202, "S")
        supervisor = linux_supervisor.ProcessIdentity(100, 1, 100, 100, 100, "S")
        registry.register("WORKER_ROOT", root)
        registry.register("DESCENDANT_001", parent)
        parent_reads = 0

        def read_identity(process_id: int, *, open_pidfd: bool = False):
            nonlocal parent_reads
            del open_pidfd
            if process_id == parent.process_id:
                parent_reads += 1
                if parent_reads == 1:
                    return parent
                raise linux_supervisor.LinuxSupervisorError(
                    "PROC_IDENTITY_NOT_AVAILABLE"
                )
            if process_id == child.process_id:
                return child
            raise linux_supervisor.LinuxSupervisorError(
                "PROC_IDENTITY_NOT_AVAILABLE"
            )

        with mock.patch.object(
            linux_supervisor, "_proc_all_identities", return_value=[child]
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", side_effect=read_identity
        ):
            registered = linux_supervisor._discover_and_register_owned_group(
                registry,
                root,
                production_mode=False,
                supervisor=supervisor,
                subreaper_enabled=False,
            )

        self.assertEqual(registered, 1)
        self.assertEqual(parent_reads, 1)
        self.assertIsNotNone(registry.active_identity("DESCENDANT_002"))
        registry.clear()

    def test_reused_numeric_group_without_lineage_fails_closed_unregistered(
        self,
    ) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        root = linux_supervisor.ProcessIdentity(101, 100, 101, 101, 200, "S")
        unrelated = linux_supervisor.ProcessIdentity(404, 1, 101, 101, 201, "S")
        supervisor = linux_supervisor.ProcessIdentity(100, 1, 100, 100, 100, "S")
        registry.register("WORKER_ROOT", root)

        with mock.patch.object(
            linux_supervisor, "_proc_all_identities", return_value=[unrelated]
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", return_value=unrelated
        ), mock.patch.object(
            linux_supervisor, "_safe_signal_owned_group"
        ) as signal_group:
            with self.assertRaisesRegex(
                linux_supervisor.LinuxSupervisorError,
                "OWNED_DESCENDANT_LINEAGE_REJECTED",
            ):
                linux_supervisor._discover_and_register_owned_group(
                    registry,
                    root,
                    production_mode=False,
                    supervisor=supervisor,
                    subreaper_enabled=False,
                )

        self.assertEqual(registry.registration_count("DESCENDANT_"), 0)
        signal_group.assert_not_called()
        registry.clear()

    def test_group_anchor_never_promotes_unregistered_numeric_member(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        retained = self._identity()
        reused = linux_supervisor.ProcessIdentity(
            retained.process_id,
            1,
            retained.process_group_id,
            retained.session_id,
            retained.start_ticks + 1,
            "S",
        )
        unrelated = linux_supervisor.ProcessIdentity(
            404,
            1,
            retained.process_group_id,
            retained.session_id,
            retained.start_ticks + 2,
            "S",
        )
        registry.register("WORKER_ROOT", retained)
        with mock.patch.object(
            linux_supervisor, "_read_proc_identity", return_value=reused
        ), mock.patch.object(
            linux_supervisor,
            "_proc_group_identities",
            return_value=[unrelated],
        ) as proc_members:
            anchor = linux_supervisor._owned_group_signal_anchor(
                registry,
                retained.process_group_id,
                retained.session_id,
            )
        self.assertIsNone(anchor)
        proc_members.assert_not_called()
        registry.clear()

    def test_group_drain_without_live_retained_anchor_never_signals(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        retained = self._identity()
        reused = linux_supervisor.ProcessIdentity(
            retained.process_id,
            1,
            retained.process_group_id,
            retained.session_id,
            retained.start_ticks + 1,
            "S",
        )
        registry.register("WORKER_ROOT", retained)
        with mock.patch.object(
            linux_supervisor, "_proc_group_member_count", return_value=1
        ), mock.patch.object(
            linux_supervisor,
            "_wait_group_with_reaping",
            return_value=(1, 0),
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", return_value=reused
        ), mock.patch.object(
            linux_supervisor, "_safe_signal_owned_group"
        ) as signal_group:
            group_count, signals, kills, reaped = (
                linux_supervisor._drain_owned_group(
                    registry,
                    group_id=retained.process_group_id,
                    session_id=retained.session_id,
                    supervisor_process_id=999,
                )
            )
        self.assertEqual((group_count, signals, kills, reaped), (1, 0, 0, 0))
        signal_group.assert_not_called()
        registry.clear()

    def test_live_exact_retained_member_is_group_signal_anchor(self) -> None:
        registry = linux_supervisor.OwnedProcessRegistry()
        retained = self._identity()
        registry.register("WORKER_ROOT", retained)
        with mock.patch.object(
            linux_supervisor,
            "_identity_status",
            return_value=linux_supervisor.IdentityStatus.MATCH,
        ), mock.patch.object(
            linux_supervisor, "_read_proc_identity", return_value=retained
        ):
            anchor = linux_supervisor._owned_group_signal_anchor(
                registry,
                retained.process_group_id,
                retained.session_id,
            )
        self.assertIs(anchor, retained)
        registry.clear()

    def test_exact_adopted_zombie_reap_never_uses_broad_waitpid(self) -> None:
        adopted = linux_supervisor.ProcessIdentity(101, 50, 101, 101, 200, "S")
        unrelated_parent = linux_supervisor.ProcessIdentity(
            102, 50, 102, 102, 201, "S"
        )
        reused = linux_supervisor.ProcessIdentity(103, 50, 103, 103, 202, "S")
        current = {
            101: linux_supervisor.ProcessIdentity(
                101, os.getpid(), 101, 101, 200, "Z"
            ),
            102: linux_supervisor.ProcessIdentity(102, 777, 102, 102, 201, "Z"),
            103: linux_supervisor.ProcessIdentity(
                103, os.getpid(), 103, 103, 999, "Z"
            ),
        }
        retained = {
            (identity.process_id, identity.start_ticks): identity
            for identity in (adopted, unrelated_parent, reused)
        }
        with mock.patch.object(
            linux_supervisor,
            "_read_proc_identity",
            side_effect=lambda process_id: current[process_id],
        ), mock.patch.object(
            linux_supervisor.os, "waitpid", return_value=(101, 0)
        ) as waitpid:
            reaped = linux_supervisor._reap_exact_adopted_zombies(retained)
        self.assertEqual(reaped, 1)
        waitpid.assert_called_once_with(101, getattr(os, "WNOHANG", 1))
        self.assertNotIn(
            "waitpid(-1",
            inspect.getsource(linux_supervisor._reap_exact_adopted_zombies),
        )

    def test_exclusive_atomic_publication_is_first_writer_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            first = {"status": "PASS", "synthetic_id": "CASE"}
            linux_supervisor._exclusive_publish_json(path, first)
            with self.assertRaises(linux_supervisor.ImmutableArtifactError):
                linux_supervisor._exclusive_publish_json(path, {"status": "FAIL"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), first)

    def test_raw_sentinel_rejects_keys_and_values(self) -> None:
        self.assertTrue(linux_supervisor._persistent_sentinel_failures({"worker_pid": 7}))
        self.assertTrue(linux_supervisor._persistent_sentinel_failures({"value": "https://invalid.example"}))
        self.assertFalse(
            linux_supervisor._persistent_sentinel_failures(
                {"case_id": "D01", "digest": "a" * 64, "count": 0, "ok": True}
            )
        )


class AuthorityAndProjectionTests(unittest.TestCase):
    def test_poller_does_not_author_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "OBS"
            directory.mkdir()
            digest = "a" * 64
            linux_supervisor._exclusive_publish_json(
                directory / "start.json",
                {
                    "schema_version": 1,
                    "observation_id": "OBS",
                    "mode": "local-test",
                    "launcher_backend": linux_supervisor.LAUNCHER_BACKEND,
                    "config_digest": digest,
                    "requested_duration_bucket": "LT_1S",
                    "maximum_gap_bucket_limit": "LT_1S",
                    "expected_minimum_sample_count": 1,
                },
            )
            linux_supervisor._replace_json(
                directory / "state.json",
                {
                    "schema_version": 1,
                    "observation_id": "OBS",
                    "mode": "local-test",
                    "status": "RUNNING",
                    "phase": "SAMPLING",
                    "verification_status": "NOT_VERIFIED",
                    "launcher_backend": linux_supervisor.LAUNCHER_BACKEND,
                    "sample_count": 0,
                    "completion_marker": False,
                    "config_digest": digest,
                },
            )
            result = linux_supervisor.poll_linux_observation(artifact_root=root, observation_id="OBS")
            self.assertEqual(result["status"], "RUNNING")
            self.assertFalse((directory / "terminal.json").exists())
            self.assertFalse((directory / "seal.json").exists())

    def test_terminal_requires_matching_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "OBS"
            directory.mkdir()
            linux_supervisor._exclusive_publish_json(
                directory / "terminal.json",
                {
                    "schema_version": 1,
                    "observation_id": "OBS",
                    "status": "PASS",
                    "verification_status": "VERIFIED",
                },
            )
            result = linux_supervisor.poll_linux_observation(artifact_root=root, observation_id="OBS")
            self.assertEqual(result["status"], "NOT_VERIFIED")
            self.assertEqual(result["reason"], "TERMINAL_SEAL_PAIR_INCOMPLETE")

    def test_mutable_state_cannot_spoof_any_terminal_or_short_circuit_stop(self) -> None:
        for forged_status in ("PASS", "HOLD", "FAIL"):
            with self.subTest(status=forged_status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = root / "OBS"
                directory.mkdir()
                digest = "a" * 64
                linux_supervisor._exclusive_publish_json(
                    directory / "start.json",
                    {
                        "schema_version": 1,
                        "observation_id": "OBS",
                        "mode": "local-test",
                        "launcher_backend": linux_supervisor.LAUNCHER_BACKEND,
                        "config_digest": digest,
                        "requested_duration_bucket": "LT_1S",
                        "maximum_gap_bucket_limit": "LT_1S",
                        "expected_minimum_sample_count": 1,
                    },
                )
                linux_supervisor._replace_json(
                    directory / "state.json",
                    {
                        "schema_version": 1,
                        "observation_id": "OBS",
                        "mode": "local-test",
                        "status": forged_status,
                        "phase": "FINALIZING",
                        "verification_status": "VERIFIED",
                        "launcher_backend": linux_supervisor.LAUNCHER_BACKEND,
                        "sample_count": 1,
                        "completion_marker": True,
                        "config_digest": digest,
                    },
                )
                projected = linux_supervisor.poll_linux_observation(
                    artifact_root=root, observation_id="OBS"
                )
                self.assertEqual(projected["status"], "NOT_VERIFIED")
                with mock.patch.object(linux_supervisor, "_require_linux"):
                    stopped = linux_supervisor.stop_linux_observation(
                        artifact_root=root,
                        observation_id="OBS",
                        reason="TASK_CLEANUP",
                        wait_seconds=0.0,
                    )
                self.assertEqual(stopped["status"], "STOP_REQUESTED")
                self.assertTrue((directory / "stop-request.json").is_file())

    def test_start_response_schema_contains_no_raw_identity(self) -> None:
        response = {
            "observation_id": "OBS",
            "mode": "local-test",
            "status": "READY",
            "launcher_backend": linux_supervisor.LAUNCHER_BACKEND,
            "verification_status": "NOT_VERIFIED",
        }
        self.assertFalse(linux_supervisor._persistent_sentinel_failures(response))
        self.assertNotIn("pid", json.dumps(response).lower())


class DeterministicCampaignTests(unittest.TestCase):
    def test_fixed_manifest_is_exact(self) -> None:
        self.assertEqual(len(linux_supervisor.DETERMINISTIC_CASES), 30)
        self.assertEqual(len(linux_supervisor.ACTUAL_CASES), 12)
        self.assertEqual(
            [case for case, _ in linux_supervisor.DETERMINISTIC_CASES],
            [f"D{index:02d}_{suffix}" for index, suffix in enumerate(
                (
                    "NORMAL_SUCCESS", "WORKER_EXIT", "CHILD_SPAWN_EXIT", "NESTED_DESCENDANT",
                    "DUPLICATE_EVENT", "REORDERED_EVENT", "MISSING_EVENT", "STALE_READINESS",
                    "WRONG_RUN_ID", "WRONG_CONFIG_DIGEST", "REGISTRATION_FAILURE", "LINEAGE_MISMATCH",
                    "TIMEOUT", "SIGTERM_SUCCESS", "SIGKILL_ESCALATION", "UNRELATED_SIBLING",
                    "PARTIAL_TERMINAL", "PARTIAL_SEAL", "TERMINAL_MUTATION_ATTEMPT", "SUPERVISOR_FAILURE",
                    "WORKER_FALSE_PASS_CANDIDATE", "POLLER_FALSE_TERMINAL_CANDIDATE", "SAMPLER_COMMAND_FAILURE",
                    "ZOMBIE_REAPING", "REGISTRY_RESIDUAL", "PROC_CONTRADICTION", "CLEANUP_FAILURE",
                    "FIRST_FAILURE_PRESERVATION", "RAW_VALUE_SENTINEL", "FINAL_ZERO_AUDIT",
                ), start=1)
            ],
        )

    def test_deterministic_90_is_stable_and_passes(self) -> None:
        campaign = linux_supervisor.run_deterministic_campaign(3)
        self.assertEqual(campaign["required_count"], 90)
        self.assertEqual(campaign["executed_count"], 90)
        self.assertEqual(campaign["pass_count"], 90)
        self.assertEqual(campaign["failure_count"], 0)
        self.assertEqual(campaign["unstable_case_count"], 0)

    def test_deterministic_campaign_counts_wrong_pass_candidates(self) -> None:
        with mock.patch.object(
            linux_supervisor,
            "_deterministic_observed_terminal",
            return_value=("PASS", {"synthetic": True}),
        ):
            campaign = linux_supervisor.run_deterministic_campaign(1)

        expected_wrong_passes = sum(
            expected != "PASS"
            for _, expected in linux_supervisor.DETERMINISTIC_CASES
        )
        self.assertEqual(campaign["failure_count"], expected_wrong_passes)
        self.assertEqual(
            sum(
                result["wrong_pass_count"]
                for result in campaign["results"]
            ),
            expected_wrong_passes,
        )

    def test_fixed_counts_cannot_be_weakened(self) -> None:
        with self.assertRaises(linux_supervisor.LinuxSupervisorError):
            linux_supervisor.run_acceptance_campaign(
                deterministic_repeats=2, actual_repeats=10, stress_seeds=100
            )

    def test_actual_campaign_detects_allowlisted_duration_bucket_variation(self) -> None:
        calls = 0

        def synthetic_actual(case_id: str):
            nonlocal calls
            bucket = "LT_1S" if calls == 1 else "LT_100MS"
            calls += 1
            return True, {
                "case_id": case_id,
                "case_verdict": "PASS",
                "signal_scope_clear": True,
                "reaping_clear": True,
                "readiness_clear": True,
                "scope_escape_fail_closed": True,
                "unrelated_termination_count": 0,
                "residual_count": 0,
                "duration_bucket": bucket,
            }

        with mock.patch.object(
            linux_supervisor, "_linux_available", return_value=True
        ), mock.patch.object(
            linux_supervisor, "_actual_case", side_effect=synthetic_actual
        ):
            campaign = linux_supervisor.run_actual_linux_campaign(10)
        self.assertEqual(campaign["executed_count"], 120)
        self.assertEqual(campaign["unstable_case_count"], 1)

    def test_actual_campaign_propagates_measured_unrelated_termination_count(
        self,
    ) -> None:
        def synthetic_actual(case_id: str):
            unrelated_count = 1 if case_id == "L10_UNRELATED_SIBLING" else 0
            return unrelated_count == 0, {
                "case_id": case_id,
                "case_verdict": "FAIL" if unrelated_count else "PASS",
                "signal_scope_clear": unrelated_count == 0,
                "reaping_clear": True,
                "readiness_clear": True,
                "scope_escape_fail_closed": True,
                "unrelated_termination_count": unrelated_count,
                "residual_count": 0,
                "duration_bucket": "LT_100MS",
            }

        with mock.patch.object(
            linux_supervisor, "_linux_available", return_value=True
        ), mock.patch.object(
            linux_supervisor, "_actual_case", side_effect=synthetic_actual
        ):
            campaign = linux_supervisor.run_actual_linux_campaign(10)

        sibling_results = [
            result
            for result in campaign["results"]
            if result["case_id"] == "L10_UNRELATED_SIBLING"
        ]
        self.assertEqual(campaign["failure_count"], 10)
        self.assertEqual(len(sibling_results), 10)
        self.assertEqual(
            sum(result["unrelated_termination_count"] for result in sibling_results),
            10,
        )


class PlatformTierFailureEvidenceTests(unittest.TestCase):
    def test_nonpass_diagnostics_are_digest_only_and_workflow_preserves_failure(self) -> None:
        def synthetic_test(test_id: str) -> mock.Mock:
            test = mock.Mock()
            test.id.return_value = test_id
            return test

        result = unittest.TestResult()
        raw_values = (
            "synthetic.failure.case raw-token raw-path",
            "synthetic.error.case raw-command raw-identity",
            "synthetic.skip.case raw-url raw-pid",
            "synthetic.expected.case raw-environment",
            "synthetic.unexpected.case raw-response-body",
        )
        tests = tuple(synthetic_test(value) for value in raw_values)
        result.failures = [(tests[0], "AssertionError: raw failure traceback token")]
        result.errors = [
            (
                tests[1],
                "Traceback (most recent call last):\n"
                '  File "raw-path", line 1, in _cleanup_linux_timeout_fixture\n'
                "    raw-command raw-identity\n"
                "AttributeError: raw error traceback command "
                "SUPERVISOR_HANDLE_REAP_FAILED\n",
            )
        ]
        result.skipped = [(tests[2], "raw skip reason url")]
        result.expectedFailures = [
            (tests[3], "AssertionError: raw expected failure environment")
        ]
        result.unexpectedSuccesses = [tests[4]]

        records = platform_tiers._nonpass_records(result)
        self.assertEqual(len(records), 5)
        self.assertEqual(
            [record["outcome_enum"] for record in records],
            sorted(platform_tiers.FAILURE_OUTCOMES),
        )
        serialized_records = json.dumps(records, sort_keys=True)
        for raw_value in (
            *raw_values,
            "raw failure traceback token",
            "raw-path",
            "raw-command",
            "raw-identity",
            "raw error traceback command",
        ):
            self.assertNotIn(raw_value, serialized_records)
        for record in records:
            self.assertRegex(record["failed_test_id_sha256"], r"^[a-f0-9]{64}$")
            self.assertRegex(record["detail_sha256"], r"^[a-f0-9]{64}$")
            self.assertIn(
                record["detail_class_enum"],
                platform_tiers.FAILURE_DETAIL_CLASSES,
            )
            self.assertIn(
                record["detail_reason_enum"],
                platform_tiers.FAILURE_DETAIL_REASONS,
            )
            self.assertIn(
                record["detail_stage_enum"],
                platform_tiers.FAILURE_DETAIL_STAGES,
            )
        error_record = next(
            record for record in records if record["outcome_enum"] == "ERROR"
        )
        self.assertEqual(error_record["detail_class_enum"], "ATTRIBUTE_ERROR")
        self.assertEqual(
            error_record["detail_reason_enum"], "NONE"
        )
        self.assertEqual(
            error_record["detail_stage_enum"], "TIMEOUT_FIXTURE_CLEANUP"
        )
        source_collision = platform_tiers._classify_nonpass_detail(
            "Traceback (most recent call last):\n"
            "  raise LinuxSupervisorError('PROC_IDENTITY_NOT_AVAILABLE')\n"
            "AssertionError: terminal assertion only\n"
        )
        self.assertEqual(
            source_collision, ("ASSERTION_ERROR", "NONE", "UNKNOWN")
        )
        chained = platform_tiers._classify_nonpass_detail(
            "ValueError: CLEANUP_POLL_FAILED\n\n"
            "The above exception was the direct cause of the following exception:\n\n"
            "Traceback (most recent call last):\n"
            '  File "safe", line 2, in _assert_linux_supervised_timeout_contract\n'
            "RuntimeError: terminal runtime error\n"
        )
        self.assertEqual(
            chained, ("RUNTIME_ERROR", "NONE", "TIMEOUT_TEST_BODY")
        )
        ambiguous = platform_tiers._classify_nonpass_detail(
            "AssertionError: LINUX_TIMEOUT_FIXTURE_CLEANUP_FAILED:"
            "CLEANUP_POLL_FAILED,EXACT_CHILD_REMAINS\n"
        )
        self.assertEqual(
            ambiguous, ("ASSERTION_ERROR", "AMBIGUOUS", "UNKNOWN")
        )
        exact_cleanup_reason = platform_tiers._classify_nonpass_detail(
            "Traceback (most recent call last):\n"
            '  File "safe", line 3, in _cleanup_linux_timeout_fixture\n'
            "AssertionError: LINUX_TIMEOUT_FIXTURE_CLEANUP_FAILED:"
            "EXACT_CHILD_REMAINS\n"
        )
        self.assertEqual(
            exact_cleanup_reason,
            ("ASSERTION_ERROR", "EXACT_CHILD_REMAINS", "TIMEOUT_FIXTURE_CLEANUP"),
        )
        multiline_decoy = platform_tiers._classify_nonpass_detail(
            "Traceback (most recent call last):\n"
            '  File "safe", line 4, in _assert_linux_supervised_timeout_contract\n'
            "AttributeError: terminal message begins\n"
            "RuntimeError: decoy continuation\n"
        )
        self.assertEqual(
            multiline_decoy, ("ATTRIBUTE_ERROR", "NONE", "TIMEOUT_TEST_BODY")
        )
        self.assertEqual(
            platform_tiers._classify_nonpass_detail("opaque nonpass detail"),
            ("UNCLASSIFIED_NONPASS", "NONE", "UNKNOWN"),
        )

        payload = platform_tiers._failure_payload(
            phase="TEST_EXECUTION",
            cause="SELECTED_TEST_NONPASS",
            platform="LINUX",
            tier=platform_tiers.LINUX_TIER,
            counts={
                "failure_count": 1,
                "error_count": 1,
                "skipped_count": 1,
                "expected_failure_count": 1,
                "unexpected_success_count": 1,
            },
            records=records,
            gate_detail="raw gate detail must be hashed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / platform_tiers.FAILURE_FILENAME
            platform_tiers._atomic_write(path, payload)
            platform_tiers.verify_failure_evidence_directory(root)
            persisted = path.read_text(encoding="utf-8")
            for raw_value in (
                *raw_values,
                "raw gate detail must be hashed",
                "raw-path",
                "raw-command",
                "raw-identity",
                "raw error traceback command",
            ):
                self.assertNotIn(raw_value, persisted)
            opaque_error = json.loads(persisted)
            opaque_error.pop("report_digest")
            for record in opaque_error["diagnostic_records"]:
                if record["outcome_enum"] == "ERROR":
                    record["detail_class_enum"] = "UNCLASSIFIED_NONPASS"
                    record["detail_reason_enum"] = "NONE"
                    record["detail_stage_enum"] = "UNKNOWN"
            platform_tiers._atomic_write(
                path, platform_tiers._sealed(opaque_error)
            )
            platform_tiers.verify_failure_evidence_directory(root)
            contradictory = dict(payload)
            contradictory.pop("report_digest")
            contradictory["phase_enum"] = "PRE_TEST_GATE"
            contradictory["platform_enum"] = "UNKNOWN"
            contradictory["tier_enum"] = "UNKNOWN"
            platform_tiers._atomic_write(
                path, platform_tiers._sealed(contradictory)
            )
            with self.assertRaisesRegex(
                platform_tiers.TierGateError,
                "FAILURE_EVIDENCE_COUNT_INVALID",
            ):
                platform_tiers.verify_failure_evidence_directory(root)
            outcome_mismatch = json.loads(persisted)
            outcome_mismatch.pop("report_digest")
            outcome_mismatch["failure_count"] = 0
            outcome_mismatch["error_count"] = 2
            platform_tiers._atomic_write(
                path, platform_tiers._sealed(outcome_mismatch)
            )
            with self.assertRaisesRegex(
                platform_tiers.TierGateError,
                "FAILURE_EVIDENCE_COUNT_INVALID",
            ):
                platform_tiers.verify_failure_evidence_directory(root)
            semantic_mismatch = json.loads(persisted)
            semantic_mismatch.pop("report_digest")
            semantic_mismatch["diagnostic_records"][0][
                "detail_class_enum"
            ] = "UNCLASSIFIED_NONPASS"
            semantic_mismatch["diagnostic_records"][0][
                "detail_reason_enum"
            ] = "SUPERVISOR_LAUNCH_FAILED"
            platform_tiers._atomic_write(
                path, platform_tiers._sealed(semantic_mismatch)
            )
            with self.assertRaisesRegex(
                platform_tiers.TierGateError,
                "FAILURE_EVIDENCE_RECORD_INVALID",
            ):
                platform_tiers.verify_failure_evidence_directory(root)
            tampered = json.loads(persisted)
            tampered["report_digest"] = "0" * 64
            platform_tiers._atomic_write(path, tampered)
            with self.assertRaisesRegex(
                platform_tiers.TierGateError,
                "FAILURE_EVIDENCE_DIGEST_MISMATCH",
            ):
                platform_tiers.verify_failure_evidence_directory(root)

        with tempfile.TemporaryDirectory() as temporary:
            rejected = Path(temporary) / "raw-token-path-must-not-render"
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout, mock.patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr:
                return_code = platform_tiers.failure_evidence_verifier_main(rejected)
            self.assertEqual(return_code, 2)
            self.assertEqual(stdout.getvalue(), "failure_evidence_verdict=REJECTED\n")
            self.assertEqual(stderr.getvalue(), "")
            self.assertNotIn(str(rejected), stdout.getvalue())

        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "qlib-linux-observer-acceptance.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("continue-on-error", workflow)
        self.assertNotIn("|| true", workflow)
        self.assertIn("steps.platform_tier.outcome == 'failure'", workflow)
        self.assertIn(
            "steps.platform_failure_verifier.outcome == 'success'", workflow
        )
        self.assertIn("except BaseException:", workflow)
        self.assertEqual(workflow.count("platform-tier-failure.json"), 1)
        self.assertIn('test "${file_count}" -eq 2', workflow)


class BundleTests(unittest.TestCase):
    def _passing_report(self) -> dict[str, object]:
        deterministic_results = [
            {
                "case_id": case_id,
                "repeat": repeat,
                "case_verdict": "PASS",
                "authoritative_terminal": terminal,
                "normalized_digest": "a" * 64,
                "duration_bucket": "LT_100MS",
                **linux_supervisor._result_audit_fields(),
            }
            for case_id, terminal in linux_supervisor.DETERMINISTIC_CASES
            for repeat in range(3)
        ]
        actual_results = [
            {
                "case_id": case_id,
                "repeat": repeat,
                "case_verdict": "PASS",
                "normalized_digest": "b" * 64,
                "duration_bucket": "LT_10S",
                "residual_count": 0,
                "scope_escape_fail_closed": True,
                **linux_supervisor._result_audit_fields(),
            }
            for case_id in linux_supervisor.ACTUAL_CASES
            for repeat in range(10)
        ]
        stress_results = [
            {
                "case_id": f"S{seed:03d}",
                "seed": seed,
                "case_verdict": "PASS",
                "normalized_digest": "c" * 64,
                "duration_bucket": "LT_10S",
                "residual_count": 0,
                **linux_supervisor._result_audit_fields(),
            }
            for seed in range(100)
        ]
        campaigns = [
            {
                "campaign": "DETERMINISTIC_CONTRACT",
                "status": "EXECUTED",
                "required_count": 90,
                "executed_count": 90,
                "pass_count": 90,
                "failure_count": 0,
                "unstable_case_count": 0,
                "results": deterministic_results,
            },
            {
                "campaign": "ACTUAL_LINUX_PROCESS",
                "status": "EXECUTED",
                "required_count": 120,
                "executed_count": 120,
                "pass_count": 120,
                "failure_count": 0,
                "unstable_case_count": 0,
                "results": actual_results,
            },
            {
                "campaign": "FIXED_SEED_STRESS",
                "status": "EXECUTED",
                "required_count": 100,
                "executed_count": 100,
                "pass_count": 100,
                "failure_count": 0,
                "unstable_case_count": 0,
                "results": stress_results,
            },
        ]
        report: dict[str, object] = {
            "schema_version": 1,
            "task_id": "R9ZNW-488D42",
            "verdict": "PASS",
            "host_execution": "EXECUTED",
            "os_family": "LINUX",
            "architecture": "x86_64",
            "python_version": "3.12",
            "git_commit_sha": "d" * 40,
            "runner_image_os": "ubuntu24",
            "runner_image_version": "20260714.240.1",
            "operator_manual_dispatch_state": "DECLARED_NOT_DEFAULT_BRANCH_ACTIVE",
            "required_total": 310,
            "executed_total": 310,
            "pass_total": 310,
            "failure_total": 0,
            "duration_bucket": "LT_10S",
            "campaigns": campaigns,
            "counters": {
                "wrong_pass_count": 0,
                "masking_count": 0,
                "raw_persistence_count": 0,
                "timeout_leak_count": 0,
                "unrelated_termination_count": 0,
                "final_live_process_count": 0,
                "final_descendant_count": 0,
                "final_orphan_count": 0,
                "final_zombie_count": 0,
                "unresolved_wait_count": 0,
                "timing_variation_count": 0,
                "automatic_rerun_count": 0,
            },
        }
        report["report_digest"] = linux_supervisor._digest_payload(report)
        return report

    def test_bundle_has_exact_closed_file_set_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            linux_supervisor.write_acceptance_bundle(output, self._passing_report())
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["acceptance-result.json", "manifest.json"],
            )
            verified = linux_supervisor.verify_acceptance_bundle(output)
            self.assertEqual(verified["verification"], "PASS")

    def test_bundle_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            linux_supervisor.write_acceptance_bundle(output, self._passing_report())
            (output / "extra.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(linux_supervisor.LinuxSupervisorError):
                linux_supervisor.verify_acceptance_bundle(output)

    def test_bundle_rejects_metadata_and_campaign_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            report = self._passing_report()
            report["runner_image_os"] = "NOT_VERIFIED"
            report["report_digest"] = linux_supervisor._digest_payload(
                {key: value for key, value in report.items() if key != "report_digest"}
            )
            linux_supervisor.write_acceptance_bundle(output, report)
            verified = linux_supervisor.verify_acceptance_bundle(output)
            self.assertEqual(verified["verification"], "HOLD")

    def test_local_non_github_mode_keeps_remote_default_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            report = self._passing_report()
            report.update(
                {
                    "verdict": "HOLD",
                    "git_commit_sha": "NOT_VERIFIED",
                    "runner_image_os": "NOT_VERIFIED",
                    "runner_image_version": "NOT_VERIFIED",
                }
            )
            report["report_digest"] = linux_supervisor._digest_payload(
                {key: value for key, value in report.items() if key != "report_digest"}
            )
            linux_supervisor.write_acceptance_bundle(output, report)
            self.assertEqual(
                linux_supervisor.verify_acceptance_bundle(output)["verification"],
                "HOLD",
            )
            self.assertEqual(
                linux_supervisor.verify_acceptance_bundle(
                    output, allow_local_non_github=True
                )["verification"],
                "PASS_LOCAL_NON_GITHUB",
            )

    def test_cleanup_is_exact_nonce_bound_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / ("r9znw-488d42-operator-" + "a" * 32)
            linux_supervisor.write_acceptance_bundle(output, self._passing_report())
            cleaned = linux_supervisor.cleanup_acceptance_bundle(output)
            self.assertEqual(cleaned["cleanup"], "PASS")
            self.assertFalse(output.exists())

    def test_cleanup_rejects_broad_or_unbound_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            linux_supervisor.write_acceptance_bundle(output, self._passing_report())
            with self.assertRaises(linux_supervisor.LinuxSupervisorError):
                linux_supervisor.cleanup_acceptance_bundle(output)

    def test_cleanup_preserves_semantically_rejected_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / ("r9znw-488d42-operator-" + "b" * 32)
            report = self._passing_report()
            report["runner_image_os"] = "NOT_VERIFIED"
            report["report_digest"] = linux_supervisor._digest_payload(
                {key: value for key, value in report.items() if key != "report_digest"}
            )
            linux_supervisor.write_acceptance_bundle(output, report)
            with self.assertRaises(linux_supervisor.LinuxSupervisorError):
                linux_supervisor.cleanup_acceptance_bundle(output)
            self.assertTrue((output / "acceptance-result.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())


@unittest.skipUnless(linux_supervisor._linux_available(), "requires Linux /proc")
class LinuxProcessContractTests(unittest.TestCase):
    def test_self_exclusion_refuses_signal(self) -> None:
        identity = linux_supervisor._read_proc_identity(os.getpid())
        result = linux_supervisor._safe_signal_owned_group(
            identity, 0, supervisor_process_id=os.getpid()
        )
        self.assertEqual(result, "IDENTITY_CONTRADICTION_NO_SIGNAL")

    def test_proc_contradiction_does_not_signal_child(self) -> None:
        process = linux_supervisor._spawn_python("import time;time.sleep(2)")
        identity = linux_supervisor._read_proc_identity(process.pid, open_pidfd=True)
        contradicted = linux_supervisor.ProcessIdentity(
            identity.process_id,
            identity.parent_process_id,
            identity.process_group_id,
            identity.session_id,
            identity.start_ticks + 1,
            identity.state,
        )
        try:
            result = linux_supervisor._safe_signal_owned_group(
                contradicted, 15, supervisor_process_id=os.getpid()
            )
            self.assertEqual(result, "IDENTITY_CONTRADICTION_NO_SIGNAL")
            self.assertIsNone(process.poll())
        finally:
            self.assertTrue(linux_supervisor._cleanup_owned_process(process, identity))

    def test_scope_escape_is_retained_fail_closed_and_exactly_cleaned(self) -> None:
        passed, normalized = linux_supervisor._actual_case(
            "L12_PROC_CONTRADICTION_NO_SIGNAL"
        )
        self.assertTrue(passed)
        self.assertTrue(normalized["scope_escape_fail_closed"])
        self.assertEqual(normalized["residual_count"], 0)

    def test_real_failed_startup_drains_unresponsive_separate_session_child(self) -> None:
        source = (
            "import os,signal,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "c=os.fork()\n"
            "if c == 0:\n"
            "    os.setsid()\n"
            "    signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "    os.write(1,b'R')\n"
            "    time.sleep(10)\n"
            "    os._exit(0)\n"
            "time.sleep(10)\n"
        )
        process = linux_supervisor._spawn_python(source, stdout=subprocess.PIPE)
        identity = linux_supervisor._read_proc_identity(
            process.pid, open_pidfd=True
        )
        try:
            self.assertTrue(linux_supervisor._read_ready_byte(process))
            self.assertTrue(
                linux_supervisor._cleanup_failed_startup_tree(process, identity)
            )
            self.assertEqual(
                linux_supervisor._proc_group_member_count(
                    identity.process_group_id, identity.session_id
                ),
                0,
            )
        finally:
            identity.close()

    def test_group_drain_kills_term_ignoring_descendant_after_root_exit(self) -> None:
        linux_supervisor._enable_child_subreaper()
        source = (
            "import os,signal,sys,time\n"
            "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n"
            "c=os.fork()\n"
            "if c == 0:\n"
            "    signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "    time.sleep(10)\n"
            "    os._exit(0)\n"
            "os.write(1,('C:'+str(c)+'\\n').encode())\n"
            "time.sleep(10)\n"
        )
        process = linux_supervisor._spawn_python(source, stdout=subprocess.PIPE)
        root = linux_supervisor._read_proc_identity(process.pid, open_pidfd=True)
        assert process.stdout is not None
        child_line = linux_supervisor._read_fd_line(process.stdout.fileno(), 1.0)
        child = linux_supervisor._read_proc_identity(
            int(child_line[2:]), open_pidfd=True
        )
        registry = linux_supervisor.OwnedProcessRegistry()
        registry.register("WORKER_ROOT", root)
        registry.register("DESCENDANT_001", child)
        try:
            group_count, signals, kills, _ = linux_supervisor._drain_owned_group(
                registry,
                group_id=root.process_group_id,
                session_id=root.session_id,
                supervisor_process_id=os.getpid(),
            )
            linux_supervisor._mark_absent_registrations_reaped(registry)
            self.assertEqual(group_count, 0)
            self.assertGreaterEqual(signals, 2)
            self.assertEqual(kills, 1)
            self.assertEqual(registry.residual_count, 0)
        finally:
            registry.clear()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1.0)

    def test_repeated_lifecycle_reaps_private_supervisor_handles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for index in range(2):
                observation_id = f"LINUX_REPEAT_{index}"
                linux_supervisor.start_linux_observation(
                    artifact_root=temporary,
                    observation_id=observation_id,
                    duration_seconds=0.02,
                    sample_interval_seconds=0.01,
                    max_gap_seconds=1.0,
                    stale_after_seconds=2.0,
                    mode="local-test",
                )
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    result = linux_supervisor.poll_linux_observation(
                        artifact_root=temporary, observation_id=observation_id
                    )
                    if result.get("status") in linux_supervisor.TERMINALS:
                        break
                    time.sleep(0.02)
                self.assertIn(result.get("status"), linux_supervisor.TERMINALS)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with linux_supervisor._SUPERVISOR_HANDLE_LOCK:
                    count = len(linux_supervisor._SUPERVISOR_HANDLES)
                if count == 0:
                    break
                time.sleep(0.01)
            self.assertEqual(count, 0)

    def test_operational_local_observation_has_sealed_pass_and_no_raw_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            started = linux_supervisor.start_linux_observation(
                artifact_root=temporary,
                observation_id="LINUX_SMOKE",
                duration_seconds=0.03,
                sample_interval_seconds=0.01,
                max_gap_seconds=1.0,
                stale_after_seconds=2.0,
                mode="local-test",
            )
            self.assertEqual(started["status"], "READY")
            deadline = time.monotonic() + 5.0
            result: dict[str, object] = started
            while time.monotonic() < deadline:
                result = linux_supervisor.poll_linux_observation(
                    artifact_root=temporary, observation_id="LINUX_SMOKE"
                )
                if result.get("status") in linux_supervisor.TERMINALS:
                    break
                time.sleep(0.02)
            self.assertEqual(result.get("status"), "PASS")
            for path in (Path(temporary) / "LINUX_SMOKE").iterdir():
                if path.suffix == ".json":
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self.assertFalse(linux_supervisor._persistent_sentinel_failures(payload))


if __name__ == "__main__":
    unittest.main()
