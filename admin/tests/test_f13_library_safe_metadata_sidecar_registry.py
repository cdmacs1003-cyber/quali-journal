from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from admin.f13_library_db_evidence_retrieval import retrieve_bridge_evidence_from_sqlite
from admin.f13_library_safe_metadata_materializer import materialize_safe_metadata_sidecar
from admin.f13_library_safe_metadata_sidecar_registry import (
    ARTIFACT_STATE_PROOFPACKED,
    SAFE_SIDECAR_APPROVED_SUMMARY,
    create_safe_sidecar_manifest,
    refresh_governance_contract,
    resolve_safe_sidecar_manifest,
    validate_refresh_proposal,
    write_safe_sidecar_manifest,
)
from admin.f13_skillup_answer_hold_adapter import adapt_skillup_answer_hold_response
from admin.f13_skillup_bridge import skillup_answer_from_bridge_response


def _walk(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.append(str(key))
            out.extend(_walk(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_walk(child))
        return out
    return [str(value)]


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "evidence_id": "ev:registry-safe-1",
        "bridge_trace_id": "btrace:registry:safe-1",
        "safe_summary": "Registry-approved synthetic safe summary.",
        "pointer_uri": "qlib://library/evidence_seeds/registry/ev-registry-safe-1",
        "raw_text_policy": "SUMMARY_ONLY",
        "rights_status": "INTERNAL",
        "summary_source": SAFE_SIDECAR_APPROVED_SUMMARY,
        "semantic_summary_verified": False,
        "raw_text_exposed": False,
        "production_path_exposed": False,
    }
    record.update(overrides)
    return record


def _materialize(tmp_path: Path, *, name: str = "safe") -> tuple[Path, Path, dict[str, Any]]:
    sqlite_path = tmp_path / f"{name}_sidecar.sqlite"
    json_path = tmp_path / f"{name}_sidecar.json"
    result = materialize_safe_metadata_sidecar(
        [_record(evidence_id=f"ev:registry-{name}-1", bridge_trace_id=f"btrace:registry:{name}-1")],
        sqlite_path=sqlite_path,
        json_path=json_path,
    )
    assert result["result_status"] == "OK"
    return sqlite_path, json_path, result


def _manifest(tmp_path: Path, *, name: str = "safe") -> tuple[Path, dict[str, Any]]:
    sqlite_path, json_path, result = _materialize(tmp_path, name=name)
    manifest = create_safe_sidecar_manifest(
        sidecar_id=f"sidecar:registry:{name}",
        created_by_task="R9ZNW-343",
        source_task_id="R9ZNW-342",
        source_proofpack_refs=["task-owned-synthetic-safe-sidecar"],
        sidecar_sqlite_path=sqlite_path,
        sidecar_json_path=json_path,
        record_count=result["accepted_count"] + result["hold_only_count"] + result["rejected_count"],
        accepted_record_count=result["accepted_count"],
        hold_only_record_count=result["hold_only_count"],
        rejected_record_count=result["rejected_count"],
        artifact_state=ARTIFACT_STATE_PROOFPACKED,
    )
    manifest_path = write_safe_sidecar_manifest(manifest, tmp_path / f"{name}_manifest.json")
    return manifest_path, manifest


def test_resolver_validates_manifest_hash_and_retrieval_helper_consumes_sidecar(tmp_path: Path) -> None:
    manifest_path, _ = _manifest(tmp_path)

    resolved = resolve_safe_sidecar_manifest(manifest_path)

    assert resolved["result_status"] == "OK"
    assert resolved["public_pointer_exposure_allowed"] is False
    assert resolved["skillup_direct_db_access_allowed"] is False
    assert resolved["production_db_write_allowed"] is False
    assert resolved["production_raw_text_read_allowed"] is False

    bridge_response = retrieve_bridge_evidence_from_sqlite(
        resolved["sidecar_sqlite_path"],
        table_name=resolved["table_name"],
    )
    assert bridge_response["result_status"] == "OK"
    assert len(bridge_response["evidence_items"]) == 1

    bridge_context = {
        **bridge_response,
        "role": "student",
        "evidence_depth": "student_safe",
        "course_id": "course:r343-registry",
        "module_id": "module:r343-registry",
        "binding_id": "binding:r343-registry",
        "tenant_id": "tenant:r343-registry",
        "organization_id": "org:r343-registry",
        "cohort_id": "cohort:r343-registry",
    }
    helper_response = skillup_answer_from_bridge_response(bridge_context)
    adapted = adapt_skillup_answer_hold_response(
        helper_response,
        request_context={"requester_module": "Skillup"},
        bridge_payload=bridge_context,
    )

    assert adapted["result_status"] == "OK"
    assert adapted["answer_status"] == "ANSWERED"
    rendered = "\n".join(_walk(adapted)).lower()
    for forbidden in (
        "qlib://",
        "manifest",
        "sidecar",
        ".sqlite",
        ".json",
        "h:\\",
        "c:\\",
        "secret",
        "token",
        "credential",
        "raw text",
    ):
        assert forbidden not in rendered


def test_resolver_rejects_tampered_hash(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    manifest["sidecar_sha256"]["sqlite"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    resolved = resolve_safe_sidecar_manifest(manifest_path)

    assert resolved["result_status"] == "HOLD"
    assert "hash mismatch" in resolved["hold_reason"]


def test_resolver_rejects_missing_sidecar_path(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    manifest["sidecar_sqlite_path"] = str(tmp_path / "missing.sqlite")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    resolved = resolve_safe_sidecar_manifest(manifest_path)

    assert resolved["result_status"] == "HOLD"
    assert "missing" in resolved["hold_reason"]


def test_resolver_rejects_forbidden_permissions(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    manifest["public_pointer_exposure_allowed"] = True
    manifest["production_db_write_allowed"] = True
    manifest["production_raw_text_read_allowed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    resolved = resolve_safe_sidecar_manifest(manifest_path)

    assert resolved["result_status"] == "HOLD"
    joined_errors = " ".join(resolved["errors"])
    assert "public_pointer_exposure_allowed" in joined_errors
    assert "production_db_write_allowed" in joined_errors
    assert "production_raw_text_read_allowed" in joined_errors


def test_resolver_rejects_unapproved_artifact_state(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    manifest["artifact_state"] = "DRAFT"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    resolved = resolve_safe_sidecar_manifest(manifest_path)

    assert resolved["result_status"] == "HOLD"
    assert "artifact_state" in " ".join(resolved["errors"])


def test_refresh_governance_requires_new_sidecar_and_explicit_review(tmp_path: Path) -> None:
    _, current = _manifest(tmp_path, name="current")
    _, proposed = _manifest(tmp_path, name="proposed")

    decision = validate_refresh_proposal(current, proposed)

    assert decision["result_status"] == "OK"
    assert decision["automatic_refresh_allowed"] is False
    assert decision["production_db_write_allowed"] is False
    assert decision["production_raw_text_read_allowed"] is False
    assert decision["failure_keeps_prior_sidecar_valid"] is True

    blocked = validate_refresh_proposal(current, current)
    assert blocked["result_status"] == "HOLD"
    assert any("new sidecar_id" in error for error in blocked["errors"])
    assert any("new sidecar hash" in error for error in blocked["errors"])


def test_refresh_governance_contract_is_non_automatic_and_non_mutating() -> None:
    contract = refresh_governance_contract()

    assert contract["refresh_mode"] == "EXPLICIT_REVIEW_REQUIRED"
    assert contract["automatic_refresh_allowed"] is False
    assert contract["approved_safe_metadata_sources_only"] is True
    assert contract["production_db_write_allowed"] is False
    assert contract["production_raw_text_read_allowed"] is False
    assert contract["public_pointer_exposure_allowed"] is False
