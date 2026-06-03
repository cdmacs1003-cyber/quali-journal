from typing import Any

from admin.f13_proofpack_manifest import build_f13_proofpack_manifest


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _safe_complete_payload() -> dict[str, Any]:
    return {
        "project": "QLIB Track A",
        "scope": "F13 ProofPack Manifest",
        "source_snapshot_ref": "commit:21e43db",
        "release_board_ref": "beta-board:diagnostic",
        "required_artifacts": [
            "selected_evidence_snapshot",
            "release_board",
            "gate_results",
        ],
        "artifacts": [
            {
                "name": "selected_evidence_snapshot",
                "repo_relative_path": "reports/track_a/QLIB_TA1_07SOJ_TRACK_A_P0_SELECTED_EVIDENCE_SNAPSHOT_20260603.md",
                "sha256": _HASH_A,
                "evidence_type": "snapshot",
                "status": "RECORDED",
            },
            {
                "name": "release_board",
                "repo_relative_path": "reports/track_a/beta_release_board_contract.md",
                "sha256": _HASH_B,
                "evidence_type": "release_board",
                "status": "RECORDED",
            },
            {
                "name": "gate_results",
                "repo_relative_path": "reports/track_a/selected_gate_results.md",
                "sha256": _HASH_C,
                "evidence_type": "gate_results",
                "status": "RECORDED",
            },
        ],
        "gate_results": {
            "Bridge Boundary": "VERIFIED",
            "Evidence Required": "VERIFIED",
            "Raw Leak": "VERIFIED",
            "Feedback Loop": "VERIFIED",
        },
        "not_verified_items": [],
    }


def _walk(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.append(str(key))
            out.extend(_walk(child))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_walk(child))
        return out
    return [str(value)]


def _assert_no_raw_internal_or_secret_surface(manifest: dict[str, Any]) -> None:
    rendered = "\n".join(_walk(manifest)).lower()
    assert "raw_text" not in rendered
    assert "raw_prompt" not in rendered
    assert "raw_query" not in rendered
    assert "full_source_text" not in rendered
    assert "internal_path" not in rendered
    assert "h:\\" not in rendered
    assert "c:\\" not in rendered
    assert "file://" not in rendered
    assert "secret" not in rendered
    assert "api_key" not in rendered
    assert "credential" not in rendered
    assert "dsn" not in rendered
    assert "customer confidential" not in rendered
    assert "paid standard" not in rendered


def _assert_no_pass_escalation(manifest: dict[str, Any]) -> None:
    assert manifest["not_granted_claims"] == {
        "f13_pass": "NOT_GRANTED",
        "track_a_pass": "NOT_GRANTED",
        "beta_pass": "NOT_GRANTED",
    }


def test_proofpack_manifest_requires_hashes_and_gate_results():
    manifest = build_f13_proofpack_manifest(
        {
            "project": "QLIB Track A",
            "scope": "F13 ProofPack Manifest",
            "source_snapshot_ref": "commit:21e43db",
            "required_artifacts": ["selected_evidence_snapshot", "release_board"],
            "artifacts": [
                {
                    "name": "selected_evidence_snapshot",
                    "repo_relative_path": "reports/track_a/snapshot.md",
                    "sha256": "",
                    "evidence_type": "snapshot",
                    "status": "RECORDED",
                }
            ],
            "gate_results": {},
        }
    )

    assert manifest["final_recommendation"] == "REVIEW_REQUIRED"
    assert "MISSING_ARTIFACT_HASH:selected_evidence_snapshot" in manifest["open_items"]
    assert "MISSING_ARTIFACT_HASH:release_board" in manifest["open_items"]
    assert "GATE_RESULTS_REQUIRED" in manifest["open_items"]
    assert "RELEASE_BOARD_REF_REQUIRED" in manifest["open_items"]
    _assert_no_pass_escalation(manifest)


def test_proofpack_manifest_blocks_pass_when_not_verified_remains():
    payload = _safe_complete_payload()
    payload["gate_results"]["DB behavior"] = "NOT_VERIFIED"
    payload["gate_results"]["Full regression"] = "NOT_EXECUTED"
    payload["not_verified_items"] = [
        "DB_BEHAVIOR",
        "PRODUCTION_RAW_LEAK_SAFETY",
        "FULL_REGRESSION_SAFETY",
        "PROOFPACK_COMPLETENESS",
    ]

    manifest = build_f13_proofpack_manifest(payload)

    assert manifest["final_recommendation"] != "APPROVE"
    assert manifest["final_recommendation"] == "REVIEW_REQUIRED"
    assert "DB_BEHAVIOR" in manifest["not_verified_items"]
    assert "PRODUCTION_RAW_LEAK_SAFETY" in manifest["not_verified_items"]
    assert "FULL_REGRESSION_SAFETY" in manifest["not_verified_items"]
    assert "PROOFPACK_COMPLETENESS" in manifest["not_verified_items"]
    assert "GATE_NOT_VERIFIED:DB_BEHAVIOR" in manifest["open_items"]
    assert "GATE_NOT_VERIFIED:FULL_REGRESSION" in manifest["open_items"]
    _assert_no_pass_escalation(manifest)


def test_proofpack_manifest_rejects_or_reviews_failed_safety_gate():
    payload = _safe_complete_payload()
    payload["gate_results"]["Raw Leak"] = "FAILED"

    manifest = build_f13_proofpack_manifest(payload)

    assert manifest["final_recommendation"] in {"REJECT", "REVIEW_REQUIRED"}
    assert manifest["final_recommendation"] == "REJECT"
    _assert_no_pass_escalation(manifest)


def test_proofpack_manifest_sanitizes_raw_internal_secret_fields():
    payload = _safe_complete_payload()
    payload["artifacts"].append(
        {
            "name": "unsafe_payload",
            "repo_relative_path": "H:\\internal\\source.txt",
            "sha256": "d" * 64,
            "evidence_type": "snapshot",
            "status": "RECORDED",
            "raw_text": "do not echo",
            "internal_path": "H:\\internal\\source.txt",
            "api_key": "do-not-echo",
            "dsn": "do-not-echo",
            "summary": "customer confidential paid standard raw text",
        }
    )

    manifest = build_f13_proofpack_manifest(payload)

    assert manifest["artifact_hashes"]["unsafe_payload"] == "d" * 64
    unsafe_descriptor = [
        item for item in manifest["artifact_descriptors"] if item["name"] == "unsafe_payload"
    ][0]
    assert unsafe_descriptor["repo_relative_path"] == ""
    assert set(unsafe_descriptor) == {
        "name",
        "repo_relative_path",
        "sha256",
        "evidence_type",
        "status",
    }
    _assert_no_raw_internal_or_secret_surface(manifest)
    _assert_no_pass_escalation(manifest)


def test_proofpack_manifest_id_is_stable_for_same_safe_inputs():
    first = build_f13_proofpack_manifest(_safe_complete_payload())
    second = build_f13_proofpack_manifest(_safe_complete_payload())

    assert first["proofpack_id"] == second["proofpack_id"]
    assert first["manifest_version"] == second["manifest_version"]
    assert first["artifact_hashes"] == second["artifact_hashes"]
