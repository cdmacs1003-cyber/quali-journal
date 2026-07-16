from __future__ import annotations

import argparse
import ast
import json
import tempfile
import unittest
from pathlib import Path

from tools import qlib_production_sampler as sampler
from tools import qlib_traffic_observer as observer


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SamplerFixture:
    def __init__(self, *, malformed_answer: bool = False, missing_evidence: bool = False) -> None:
        self.http_calls = 0
        self.malformed_answer = malformed_answer
        self.missing_evidence = missing_evidence

    @staticmethod
    def gcloud(arguments: list[str], deadline: sampler.Deadline, counters: sampler.CommandCounters) -> str:
        counters.record_read_only()
        deadline.remaining_seconds()
        if arguments[:2] == ["auth", "print-identity-token"]:
            return "synthetic-identity"
        if arguments[:3] == ["config", "get-value", "project"]:
            return "synthetic-project"
        if arguments[:3] == ["run", "services", "describe"]:
            return json.dumps(
                {
                    "status": {
                        "url": "memory-service",
                        "traffic": [
                            {
                                "tag": "r487a-test",
                                "revisionName": "qlib-skillup-runtime-00099-abc",
                                "url": "memory-tag",
                            }
                        ],
                    }
                }
            )
        raise AssertionError(f"unexpected sanitized command shape: {arguments[:3]}")

    def http(
        self,
        location: str,
        deadline: sampler.Deadline,
        counters: sampler.CommandCounters,
        identity: str = "",
        method: str = "GET",
        body: bytes | None = None,
    ) -> tuple[int, bytes, float]:
        counters.record_read_only()
        deadline.remaining_seconds()
        self.http_calls += 1
        if location.endswith("/health") and not identity:
            return 403, b"{}", 1.0
        if location.endswith("/health"):
            return 200, b'{"status":"ok"}', 2.0
        if location == "memory-tag/":
            assets = "".join(f'<script src="assets/a{index}.js"></script>' for index in range(5))
            return 200, ("beta-minimal-form" + assets).encode("utf-8"), 3.0
        if "assets/" in location:
            return 200, b"asset", 1.0
        if method == "POST" and self.http_calls == 9:
            if self.malformed_answer:
                return 200, b"{", 2.0
            payload = {
                "answer_status": "ANSWERED",
                "evidence": (
                    []
                    if self.missing_evidence
                    else [{"evidence_id": "ev-flux-safe-summary-v1"}]
                ),
                "trace_id": "" if self.missing_evidence else "trace-safe",
                "raw_text_included": False,
                "raw_query_answer_retention_count": 0,
                "production_write_count": 0,
            }
            return 200, json.dumps(payload).encode("utf-8"), 2.0
        if method == "POST" and self.http_calls == 10:
            payload = {
                "answer_status": "HOLD",
                "raw_text_included": False,
                "raw_query_answer_retention_count": 0,
                "production_write_count": 0,
            }
            return 200, json.dumps(payload).encode("utf-8"), 2.0
        raise AssertionError(f"unexpected in-memory request number: {self.http_calls}")


class QlibProductionSamplerTests(unittest.TestCase):
    @staticmethod
    def args() -> argparse.Namespace:
        return argparse.Namespace(
            target_contract="REVISION_FUNCTIONAL",
            candidate="qlib-skillup-runtime-00099-abc",
            tag="r487a-test",
            expected_candidate=0,
            expected_stable=100,
        )

    def test_python_ast_and_machine_readable_contract(self) -> None:
        source_path = Path(sampler.__file__)
        ast.parse(source_path.read_text(encoding="utf-8"))
        contract = json.loads(
            source_path.with_name("qlib_production_sampler_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(contract["canonical_source"], "tools/qlib_production_sampler.py")
        self.assertEqual(contract["target"]["class"], "REVISION_FUNCTIONAL")
        self.assertEqual(contract["deadline"]["total_seconds"], 30)
        self.assertTrue(contract["deadline"]["increase_forbidden"])
        self.assertEqual(contract["timeout"]["orphan_child_limit"], 0)
        self.assertEqual(set(contract["phases"]), set(sampler.PHASES))
        self.assertTrue(
            set(contract["output"]["required_fields"])
            <= set(self._fast_result()[0])
        )

    def _fast_result(self) -> tuple[dict[str, object], Path]:
        fixture = SamplerFixture()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        progress = Path(temporary.name) / "progress.json"
        result, exit_code = sampler.execute(
            self.args(),
            environment={
                sampler.PROGRESS_FILE_ENV: str(progress),
                sampler.DEADLINE_ENV: "30",
            },
            gcloud_runner=fixture.gcloud,
            http_runner=fixture.http,
            clock=FakeClock(),
        )
        self.assertEqual(exit_code, 0)
        return result, progress

    def test_fast_sample_path_phase_timing_evidence_and_counters(self) -> None:
        result, progress = self._fast_result()
        self.assertEqual(result["sample_status"], "PASS")
        self.assertEqual(result["valid_sample_count"], 1)
        self.assertEqual(result["evidence_trace_safe_summary"], "PASS")
        self.assertEqual(result["read_only_command_count"], 13)
        self.assertEqual(result["mutation_command_count"], 0)
        self.assertEqual(result["current_phase"], "PARENT_CHILD_IPC")
        self.assertEqual(result["last_completed_phase"], "SERIALIZATION")
        self.assertGreaterEqual(len(result["phase_timings"]), 7)
        observer._sanitize_sample_value(result)
        persisted = json.loads(progress.read_text(encoding="utf-8"))
        self.assertEqual(persisted["read_only_command_count"], 13)
        encoded = json.dumps([result, persisted], ensure_ascii=True)
        for forbidden in ("synthetic-identity", "memory-tag", "Authorization", "Bearer "):
            self.assertNotIn(forbidden, encoded)

    def test_exact_total_deadline_and_timeout_phase_marker(self) -> None:
        clock = FakeClock()
        fixture = SamplerFixture()

        def timeout_http(
            location: str,
            deadline: sampler.Deadline,
            counters: sampler.CommandCounters,
            identity: str = "",
            method: str = "GET",
            body: bytes | None = None,
        ) -> tuple[int, bytes, float]:
            del location, identity, method, body
            counters.record_read_only()
            clock.advance(30.001)
            deadline.remaining_seconds()
            raise AssertionError("deadline did not stop the request")

        with tempfile.TemporaryDirectory() as temporary:
            progress = Path(temporary) / "progress.json"
            result, exit_code = sampler.execute(
                self.args(),
                environment={
                    sampler.PROGRESS_FILE_ENV: str(progress),
                    sampler.DEADLINE_ENV: "30",
                },
                gcloud_runner=fixture.gcloud,
                http_runner=timeout_http,
                clock=clock,
            )
        self.assertEqual(exit_code, 42)
        self.assertEqual(result["failure_category"], "TIMEOUT")
        self.assertEqual(result["timeout_reason"], "TOTAL_DEADLINE_EXCEEDED")
        self.assertEqual(result["current_phase"], "HTTP_REQUEST")
        self.assertEqual(result["last_completed_phase"], "REQUEST_PREPARATION")
        self.assertEqual(result["valid_sample_count"], 0)
        self.assertEqual(result["read_only_command_count"], 4)
        self.assertEqual(result["phase_timings"][-1]["phase"], "HTTP_REQUEST")
        self.assertEqual(result["phase_timings"][-1]["status"], "TIMEOUT")
        self.assertLessEqual(result["phase_timings"][-1]["elapsed_ms"], 30000.0)

    def test_malformed_and_evidence_missing_results_fail_closed(self) -> None:
        for fixture, category in (
            (SamplerFixture(malformed_answer=True), "JSON_PARSE_FAILURE"),
            (SamplerFixture(missing_evidence=True), "FUNCTIONAL_HTTP_FAILURE"),
        ):
            with self.subTest(category=category):
                result, exit_code = sampler.execute(
                    self.args(),
                    environment={sampler.DEADLINE_ENV: "30"},
                    gcloud_runner=fixture.gcloud,
                    http_runner=fixture.http,
                    clock=FakeClock(),
                )
                self.assertEqual(exit_code, 42)
                self.assertEqual(result["sample_status"], "FAIL")
                self.assertEqual(result["failure_category"], category)
                self.assertEqual(result["valid_sample_count"], 0)
                self.assertEqual(result["evidence_trace_safe_summary"], "NOT_VERIFIED")
                self.assertEqual(result["mutation_command_count"], 0)

    def test_invalid_deadline_cannot_raise_the_thirty_second_contract(self) -> None:
        result, exit_code = sampler.execute(
            self.args(), environment={sampler.DEADLINE_ENV: "30.001"}
        )
        self.assertEqual(exit_code, 42)
        self.assertEqual(result["sample_status"], "FAIL")
        self.assertEqual(result["valid_sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
