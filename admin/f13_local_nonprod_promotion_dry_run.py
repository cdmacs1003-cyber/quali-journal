"""Non-mutating local promotion dry-run and pointer-only evidence materialization."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from admin.f13_analytics_event_contract import validate_analytics_event
from admin.f13_analytics_improvement_candidate_contract import validate_analytics_improvement_candidate
from admin.f13_local_nonprod_warehouse_review_lifecycle import validate_review_event


SCHEMA_VERSION = 1
CONTRACT_VERSION = "promotion.local_nonproduction.pointer_only.v1"
STATUS_DRY_RUN_READY = "dry_run_ready"
RAW_TEXT_POLICY_POINTER_ONLY = "POINTER_ONLY"
TARGET_DOC_KIND = "REFERENCE"
TARGET_CANONICAL_LANG = "EN"
ALLOWED_RIGHTS = {"owned", "licensed", "permission_granted", "public_reference", "internal_only"}
ALLOWED_SENSITIVITY = {"public", "internal", "restricted"}
ALLOWED_REVIEWER_ROLES = {"CURATOR", "OWNER"}
ALLOWED_FILES = (
    "promotion_plan.json",
    "evidence_pointer.json",
    "provenance.json",
    "materialization_manifest.json",
    "README_POINTER_ONLY.md",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._@-]{0,199}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDEMPOTENCY_RE = re.compile(r"^idem:promotion:[0-9a-f]{64}$")
_TARGET_RE = re.compile(r"^QLIB:ANALYTICS_REF:[0-9a-f]{24}@v1$")
_PROHIBITED_FIELDS = {
    "raw_query", "query", "raw_body", "body", "prompt", "raw_prompt", "raw_text",
    "answer", "answer_text", "safe_short_answer", "hold_source", "evidence", "evidence_text",
    "standard_text", "paid_standard_text", "user_id", "user_id_hash", "personal_name", "email",
    "phone", "address", "secret", "api_key", "access_token", "refresh_token", "password",
    "credential", "cookie", "authorization", "internal_path", "local_path", "file_path", "db_path",
    "dsn", "connection_string", "library_write_target", "promotion_commit",
}
_PLAN_FIELDS = {
    "schema_version", "contract_version", "promotion_plan_id", "warehouse_item_id",
    "warehouse_item_revision", "approval_event_id", "tenant_id", "organization_id", "cohort_id",
    "source_candidate_id", "source_event_id", "source_request_id", "source_trace_id", "query_hash",
    "target_library_node_id", "target_doc_kind", "target_canonical_lang", "evidence_pointer",
    "provenance", "rights_status", "raw_text_policy", "materialization_manifest", "plan_hash",
    "idempotency_key", "status", "human_approval_required", "actual_promotion_performed",
    "library_write_performed", "created_at",
}
_POINTER_FIELDS = {
    "evidence_id", "source_doc_id", "source_doc_kind", "source_hash", "source_span_id",
    "pointer_uri", "rights_status", "raw_text_policy", "evidence_summary_code", "created_at",
    "created_by", "validation_shape_ids",
}


class PromotionDryRunError(RuntimeError):
    """Controlled fail-closed dry-run/materialization error."""


def _safe_identifier(value: Any, reason: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise PromotionDryRunError(reason)
    return value


def _safe_scope(tenant_id: Any, organization_id: Any) -> tuple[str, str]:
    return _safe_identifier(tenant_id, "TENANT_SCOPE_REQUIRED"), _safe_identifier(
        organization_id, "ORGANIZATION_SCOPE_REQUIRED"
    )


def _safe_timestamp(value: Any, reason: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PromotionDryRunError(reason) from exc
    else:
        raise PromotionDryRunError(reason)
    if parsed.tzinfo is None:
        raise PromotionDryRunError(reason)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now(clock: Callable[[], datetime] | None) -> str:
    value = clock() if clock else datetime.now(timezone.utc)
    return _safe_timestamp(value, "TIMEZONE_AWARE_CLOCK_REQUIRED")


def canonical_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    serialized = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return serialized, "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def build_promotion_idempotency_key(*parts: str) -> str:
    if not parts or any(not isinstance(part, str) or _IDENTIFIER_RE.fullmatch(part) is None for part in parts):
        raise PromotionDryRunError("INVALID_IDEMPOTENCY_INPUT")
    return "idem:promotion:" + hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _prohibited_fields(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key).lower()
            path = f"{prefix}.{name}" if prefix else name
            if name in _PROHIBITED_FIELDS:
                found.append(path)
            found.extend(_prohibited_fields(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_prohibited_fields(nested, f"{prefix}[{index}]"))
    return sorted(set(found))


def validate_evidence_pointer(pointer: Any) -> dict[str, Any]:
    invalid: list[str] = []
    if not isinstance(pointer, Mapping):
        return {"valid": False, "reason_code": "POINTER_NOT_MAPPING", "invalid_fields": []}
    if set(pointer) != _POINTER_FIELDS:
        invalid.append("FIELD_SET")
    for field in ("evidence_id", "source_doc_id", "source_span_id", "created_by"):
        if not isinstance(pointer.get(field), str) or _IDENTIFIER_RE.fullmatch(pointer[field]) is None:
            invalid.append(field)
    if pointer.get("source_doc_kind") != "WAREHOUSE_ITEM":
        invalid.append("source_doc_kind")
    if not isinstance(pointer.get("source_hash"), str) or _HASH_RE.fullmatch(pointer["source_hash"]) is None:
        invalid.append("source_hash")
    if not isinstance(pointer.get("pointer_uri"), str) or not pointer["pointer_uri"].startswith("pointer://warehouse/"):
        invalid.append("pointer_uri")
    if pointer.get("raw_text_policy") != RAW_TEXT_POLICY_POINTER_ONLY:
        invalid.append("raw_text_policy")
    if pointer.get("rights_status") not in ALLOWED_RIGHTS:
        invalid.append("rights_status")
    if not isinstance(pointer.get("evidence_summary_code"), str) or _IDENTIFIER_RE.fullmatch(pointer["evidence_summary_code"]) is None:
        invalid.append("evidence_summary_code")
    shapes = pointer.get("validation_shape_ids")
    if shapes != ["SH-F13-EVIDENCE-001"]:
        invalid.append("validation_shape_ids")
    try:
        _safe_timestamp(pointer.get("created_at"), "INVALID_CREATED_AT")
    except PromotionDryRunError:
        invalid.append("created_at")
    if _prohibited_fields(pointer):
        invalid.append("PROHIBITED_FIELD")
    return {"valid": not invalid, "reason_code": "POINTER_VALID" if not invalid else "POINTER_INVALID", "invalid_fields": sorted(set(invalid))}


def validate_promotion_plan(plan: Any) -> dict[str, Any]:
    invalid: list[str] = []
    if not isinstance(plan, Mapping):
        return {"valid": False, "reason_code": "PLAN_NOT_MAPPING", "invalid_fields": []}
    if set(plan) != _PLAN_FIELDS:
        invalid.append("FIELD_SET")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "target_doc_kind": TARGET_DOC_KIND,
        "target_canonical_lang": TARGET_CANONICAL_LANG,
        "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY,
        "status": STATUS_DRY_RUN_READY,
        "human_approval_required": True,
        "actual_promotion_performed": False,
        "library_write_performed": False,
    }
    for field, expected in fixed.items():
        if plan.get(field) != expected:
            invalid.append(field)
    for field in (
        "promotion_plan_id", "warehouse_item_id", "approval_event_id", "tenant_id", "organization_id",
        "cohort_id", "source_candidate_id", "source_event_id", "source_request_id", "source_trace_id",
    ):
        if not isinstance(plan.get(field), str) or _IDENTIFIER_RE.fullmatch(plan[field]) is None:
            invalid.append(field)
    if not isinstance(plan.get("warehouse_item_revision"), int) or plan["warehouse_item_revision"] < 1:
        invalid.append("warehouse_item_revision")
    for field in ("query_hash", "plan_hash"):
        if not isinstance(plan.get(field), str) or _HASH_RE.fullmatch(plan[field]) is None:
            invalid.append(field)
    if not isinstance(plan.get("target_library_node_id"), str) or _TARGET_RE.fullmatch(plan["target_library_node_id"]) is None:
        invalid.append("target_library_node_id")
    if not isinstance(plan.get("idempotency_key"), str) or _IDEMPOTENCY_RE.fullmatch(plan["idempotency_key"]) is None:
        invalid.append("idempotency_key")
    if not isinstance(plan.get("provenance"), Mapping) or set(plan["provenance"]) != {
        "provider_type", "provider_ref", "source_event_id", "source_trace_id", "collection_reason"
    }:
        invalid.append("provenance")
    if plan.get("rights_status") not in ALLOWED_RIGHTS:
        invalid.append("rights_status")
    manifest = plan.get("materialization_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("files") != list(ALLOWED_FILES) or manifest.get("atomic_finalization") is not True:
        invalid.append("materialization_manifest")
    if not validate_evidence_pointer(plan.get("evidence_pointer"))["valid"]:
        invalid.append("evidence_pointer")
    try:
        _safe_timestamp(plan.get("created_at"), "INVALID_CREATED_AT")
    except PromotionDryRunError:
        invalid.append("created_at")
    if _prohibited_fields(plan):
        invalid.append("PROHIBITED_FIELD")
    if not invalid and isinstance(plan.get("plan_hash"), str):
        basis = dict(plan)
        expected = basis.pop("plan_hash")
        if canonical_payload(basis)[1] != expected:
            invalid.append("plan_hash")
    return {"valid": not invalid, "reason_code": "PLAN_VALID" if not invalid else "PLAN_INVALID", "invalid_fields": sorted(set(invalid))}


class PromotionDryRunPlanner:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock
        self._idempotency: dict[str, tuple[str, dict[str, Any]]] = {}

    @staticmethod
    def _read_sources(source_repository: Any, item: Mapping[str, Any], tenant: str, organization: str) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_result = source_repository.read_record(
            item["source_candidate_record_id"], tenant_id=tenant, organization_id=organization
        )
        candidate = candidate_result.get("domain_object") if isinstance(candidate_result, Mapping) and candidate_result.get("found") is True else None
        if not isinstance(candidate, Mapping) or not validate_analytics_improvement_candidate(candidate).get("valid"):
            raise PromotionDryRunError("SOURCE_CANDIDATE_NOT_ACTIVE_OR_VALID")
        records = source_repository.list_records(tenant_id=tenant, organization_id=organization).get("records", [])
        source_record_id = next((row.get("record_id") for row in records if row.get("record_type") == "ANALYTICS_EVENT" and row.get("domain_object_id") == item.get("source_event_id")), None)
        if not source_record_id:
            raise PromotionDryRunError("SOURCE_EVENT_NOT_ACTIVE_OR_VALID")
        event_result = source_repository.read_record(source_record_id, tenant_id=tenant, organization_id=organization)
        event = event_result.get("domain_object") if isinstance(event_result, Mapping) and event_result.get("found") is True else None
        if not isinstance(event, Mapping) or not validate_analytics_event(event).get("valid"):
            raise PromotionDryRunError("SOURCE_EVENT_NOT_ACTIVE_OR_VALID")
        return dict(candidate), dict(event)

    def create_plan(
        self,
        warehouse_repository: Any,
        source_repository: Any,
        warehouse_item_id: Any,
        *,
        tenant_id: Any,
        organization_id: Any,
        expected_revision: Any,
        approval_event_id: Any,
        idempotency_key: Any,
        created_by: Any,
        created_at: Any | None = None,
        existing_target_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        tenant, organization = _safe_scope(tenant_id, organization_id)
        item_id = _safe_identifier(warehouse_item_id, "INVALID_WAREHOUSE_ITEM_ID")
        approval_id = _safe_identifier(approval_event_id, "APPROVAL_EVENT_REQUIRED")
        creator = _safe_identifier(created_by, "CREATED_BY_REQUIRED")
        if not isinstance(idempotency_key, str) or _IDEMPOTENCY_RE.fullmatch(idempotency_key) is None:
            raise PromotionDryRunError("INVALID_IDEMPOTENCY_KEY")
        request = {
            "warehouse_item_id": item_id, "tenant_id": tenant, "organization_id": organization,
            "expected_revision": expected_revision, "approval_event_id": approval_id,
            "created_by": creator,
        }
        request_hash = canonical_payload(request)[1]
        prior = self._idempotency.get(idempotency_key)
        if prior is not None:
            if prior[0] == request_hash:
                return {"plan_created": False, "reason_code": "IDEMPOTENT_REPLAY", "plan": prior[1]}
            return {"plan_created": False, "reason_code": "IDEMPOTENCY_CONFLICT", "plan": None}
        item_result = warehouse_repository.read_item(item_id, tenant_id=tenant, organization_id=organization)
        item = item_result.get("item") if isinstance(item_result, Mapping) and item_result.get("found") is True else None
        if not isinstance(item, Mapping):
            raise PromotionDryRunError("WAREHOUSE_ITEM_NOT_FOUND_OR_NOT_VISIBLE")
        if item.get("current_status") != "approved_for_warehouse":
            raise PromotionDryRunError("WAREHOUSE_ITEM_NOT_APPROVED")
        if not isinstance(expected_revision, int) or item.get("revision") != expected_revision:
            raise PromotionDryRunError("STALE_WAREHOUSE_REVISION")
        if item.get("approved_for_library") is not False or item.get("auto_promote") is not False:
            raise PromotionDryRunError("PROMOTION_MARKER_FORBIDDEN")
        if item.get("rights_status") not in ALLOWED_RIGHTS:
            raise PromotionDryRunError("RIGHTS_NOT_POINTER_ELIGIBLE")
        if item.get("sensitivity") not in ALLOWED_SENSITIVITY:
            raise PromotionDryRunError("SENSITIVITY_NOT_DRY_RUN_ELIGIBLE")
        if _prohibited_fields(item):
            raise PromotionDryRunError("WAREHOUSE_ITEM_PROHIBITED_FIELDS")
        approval_result = warehouse_repository.read_review_event_by_approval(
            item_id, approval_id, tenant_id=tenant, organization_id=organization
        )
        approval = approval_result.get("review_event") if isinstance(approval_result, Mapping) and approval_result.get("found") is True else None
        if not isinstance(approval, Mapping):
            raise PromotionDryRunError("APPROVAL_EVENT_NOT_FOUND")
        if approval.get("approval_event_id") != approval_id or approval.get("decision") != "APPROVE_WAREHOUSE":
            raise PromotionDryRunError("APPROVAL_EVENT_MISMATCH")
        if approval.get("reviewer_role") not in ALLOWED_REVIEWER_ROLES:
            raise PromotionDryRunError("APPROVAL_REVIEWER_ROLE_INVALID")
        if not validate_review_event(approval).get("valid"):
            raise PromotionDryRunError("APPROVAL_EVENT_INVALID")
        if approval.get("new_revision") != expected_revision - 1 or approval.get("next_status") != "reviewed":
            raise PromotionDryRunError("APPROVAL_REVISION_MISMATCH")
        candidate, event = self._read_sources(source_repository, item, tenant, organization)
        continuity = {
            "tenant_id": tenant,
            "organization_id": organization,
            "cohort_id": event.get("cohort_id"),
            "source_candidate_id": candidate.get("candidate_id"),
            "source_event_id": event.get("event_id"),
            "source_request_id": event.get("request_id"),
            "source_trace_id": event.get("trace_id"),
            "query_hash": event.get("query_hash"),
        }
        if any(item.get(field) != value for field, value in continuity.items()):
            raise PromotionDryRunError("SOURCE_CONTINUITY_INVALID")
        if any(candidate.get(field) != item.get(field) for field in ("source_event_id", "source_request_id", "source_trace_id", "query_hash")):
            raise PromotionDryRunError("CANDIDATE_CONTINUITY_INVALID")
        created = _safe_timestamp(created_at if created_at is not None else _now(self._clock), "INVALID_CREATED_AT")
        identity_seed = canonical_payload({
            "warehouse_item_id": item_id, "source_candidate_id": item["source_candidate_id"],
            "tenant_id": tenant, "organization_id": organization,
        })[1].split(":", 1)[1]
        target_id = f"QLIB:ANALYTICS_REF:{identity_seed[:24]}@v1"
        if target_id in set(existing_target_ids):
            raise PromotionDryRunError("TARGET_ID_CONFLICT")
        plan_id = f"promotion:plan:{identity_seed[:32]}"
        candidate_hash = canonical_payload(candidate)[1]
        evidence_id = f"evidence:pointer:{identity_seed[:32]}"
        pointer = {
            "evidence_id": evidence_id,
            "source_doc_id": item["source_candidate_id"],
            "source_doc_kind": "WAREHOUSE_ITEM",
            "source_hash": candidate_hash,
            "source_span_id": "span:pointer-only:0",
            "pointer_uri": f"pointer://warehouse/{item_id}#candidate={item['source_candidate_id']}",
            "rights_status": item["rights_status"],
            "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY,
            "evidence_summary_code": candidate["summary_code"],
            "created_at": created,
            "created_by": creator,
            "validation_shape_ids": ["SH-F13-EVIDENCE-001"],
        }
        plan_basis = {
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "promotion_plan_id": plan_id,
            "warehouse_item_id": item_id,
            "warehouse_item_revision": expected_revision,
            "approval_event_id": approval_id,
            "tenant_id": tenant,
            "organization_id": organization,
            "cohort_id": item["cohort_id"],
            "source_candidate_id": item["source_candidate_id"],
            "source_event_id": item["source_event_id"],
            "source_request_id": item["source_request_id"],
            "source_trace_id": item["source_trace_id"],
            "query_hash": item["query_hash"],
            "target_library_node_id": target_id,
            "target_doc_kind": TARGET_DOC_KIND,
            "target_canonical_lang": TARGET_CANONICAL_LANG,
            "evidence_pointer": pointer,
            "provenance": dict(item["provenance"]),
            "rights_status": item["rights_status"],
            "raw_text_policy": RAW_TEXT_POLICY_POINTER_ONLY,
            "materialization_manifest": {"files": list(ALLOWED_FILES), "atomic_finalization": True},
            "idempotency_key": idempotency_key,
            "status": STATUS_DRY_RUN_READY,
            "human_approval_required": True,
            "actual_promotion_performed": False,
            "library_write_performed": False,
            "created_at": created,
        }
        plan_hash = canonical_payload(plan_basis)[1]
        plan = dict(plan_basis)
        plan["plan_hash"] = plan_hash
        if not validate_promotion_plan(plan)["valid"]:
            raise PromotionDryRunError("PROMOTION_PLAN_INVALID")
        self._idempotency[idempotency_key] = (request_hash, plan)
        return {"plan_created": True, "reason_code": "DRY_RUN_PLAN_READY", "plan": plan}


def _safe_materialization_path(output_directory: str | Path, approved_runtime_root: str | Path) -> tuple[Path, Path]:
    output = Path(output_directory)
    root = Path(approved_runtime_root)
    if not output.is_absolute() or not root.is_absolute():
        raise PromotionDryRunError("ABSOLUTE_MATERIALIZATION_PATH_REQUIRED")
    resolved_root = root.resolve()
    resolved_output = output.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise PromotionDryRunError("MATERIALIZATION_PATH_OUTSIDE_APPROVED_ROOT") from exc
    lowered = str(resolved_output).lower()
    if any(marker in lowered for marker in ("production", "\\library\\", "/library/")):
        raise PromotionDryRunError("PRODUCTION_LIBRARY_PATH_FORBIDDEN")
    return resolved_output, resolved_root


def _manifest_payload(plan: Mapping[str, Any], file_bytes: Mapping[str, bytes]) -> dict[str, Any]:
    entries = [{"path": name, "sha256": _sha256(file_bytes[name]), "hash_basis": "file_bytes"} for name in ALLOWED_FILES if name != "materialization_manifest.json"]
    self_entry = {"path": "materialization_manifest.json", "sha256": "", "hash_basis": "manifest_without_self_sha256"}
    entries.append(self_entry)
    manifest = {
        "schema_version": 1,
        "promotion_plan_id": plan["promotion_plan_id"],
        "plan_hash": plan["plan_hash"],
        "files": entries,
        "actual_promotion_performed": False,
        "library_write_performed": False,
    }
    self_entry["sha256"] = canonical_payload(manifest)[1]
    return manifest


def verify_materialization(output_directory: str | Path) -> dict[str, Any]:
    output = Path(output_directory)
    if not output.is_dir():
        raise PromotionDryRunError("MATERIALIZATION_NOT_FOUND")
    names = sorted(path.name for path in output.iterdir() if path.is_file())
    if names != sorted(ALLOWED_FILES):
        raise PromotionDryRunError("MATERIALIZATION_FILE_SET_INVALID")
    manifest = json.loads((output / "materialization_manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, list) or sorted(entry.get("path") for entry in entries if isinstance(entry, Mapping)) != sorted(ALLOWED_FILES):
        raise PromotionDryRunError("MATERIALIZATION_MANIFEST_INVALID")
    for entry in entries:
        name = entry["path"]
        if name == "materialization_manifest.json":
            check = json.loads(json.dumps(manifest))
            self_entry = next(item for item in check["files"] if item["path"] == name)
            expected = self_entry["sha256"]
            self_entry["sha256"] = ""
            actual = canonical_payload(check)[1]
        else:
            expected = entry["sha256"]
            actual = _sha256((output / name).read_bytes())
        if actual != expected:
            raise PromotionDryRunError("MATERIALIZATION_HASH_MISMATCH")
    plan = json.loads((output / "promotion_plan.json").read_text(encoding="utf-8"))
    if not validate_promotion_plan(plan)["valid"] or plan.get("plan_hash") != manifest.get("plan_hash"):
        raise PromotionDryRunError("MATERIALIZED_PLAN_INVALID")
    hashes = {name: _sha256((output / name).read_bytes()) for name in ALLOWED_FILES}
    return {"verified": True, "reason_code": "MATERIALIZATION_VERIFIED", "plan": plan, "file_hashes": hashes}


def materialize_pointer_only_evidence(
    plan: Any,
    output_directory: str | Path,
    *,
    approved_runtime_root: str | Path,
    inject_failure: str | None = None,
) -> dict[str, Any]:
    validation = validate_promotion_plan(plan)
    if not validation["valid"]:
        raise PromotionDryRunError("PROMOTION_PLAN_INVALID")
    output, root = _safe_materialization_path(output_directory, approved_runtime_root)
    if inject_failure == "before_directory_creation":
        raise PromotionDryRunError("INJECTED_BEFORE_DIRECTORY_CREATION")
    if output.exists():
        try:
            verified = verify_materialization(output)
        except (PromotionDryRunError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise PromotionDryRunError("MATERIALIZATION_CONFLICT") from exc
        if verified["plan"]["plan_hash"] != plan["plan_hash"]:
            raise PromotionDryRunError("MATERIALIZATION_CONFLICT")
        return {**verified, "materialized": False, "reason_code": "IDEMPOTENT_REPLAY"}
    root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / f".{output.name}.tmp-{plan['promotion_plan_id'].split(':')[-1][:12]}"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=False)
    try:
        content: dict[str, bytes] = {
            "promotion_plan.json": _json_bytes(plan),
            "evidence_pointer.json": _json_bytes(plan["evidence_pointer"]),
            "provenance.json": _json_bytes(plan["provenance"]),
            "README_POINTER_ONLY.md": (
                "# Pointer-only task evidence\n\n"
                f"Promotion plan: `{plan['promotion_plan_id']}`\n\n"
                "This task-local set contains metadata pointers only. It is not a Library write or promotion.\n"
            ).encode("utf-8"),
        }
        for index, name in enumerate(("promotion_plan.json", "evidence_pointer.json", "provenance.json", "README_POINTER_ONLY.md")):
            (temp / name).write_bytes(content[name])
            if index == 0 and inject_failure == "after_one_temporary_file":
                raise PromotionDryRunError("INJECTED_AFTER_TEMPORARY_FILE")
        if inject_failure == "before_manifest_finalization":
            raise PromotionDryRunError("INJECTED_BEFORE_MANIFEST_FINALIZATION")
        manifest = _manifest_payload(plan, content)
        content["materialization_manifest.json"] = _json_bytes(manifest)
        (temp / "materialization_manifest.json").write_bytes(content["materialization_manifest.json"])
        if inject_failure == "hash_mismatch_readback":
            (temp / "provenance.json").write_bytes(b"{}\n")
        verify_materialization(temp)
        if inject_failure == "during_final_atomic_rename":
            raise PromotionDryRunError("INJECTED_DURING_ATOMIC_RENAME")
        temp.replace(output)
        verified = verify_materialization(output)
        return {**verified, "materialized": True, "reason_code": "MATERIALIZATION_CREATED"}
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def cleanup_materialization(output_directory: str | Path, *, approved_runtime_root: str | Path) -> dict[str, Any]:
    output, _root = _safe_materialization_path(output_directory, approved_runtime_root)
    if output.exists():
        shutil.rmtree(output)
        return {"deleted_count": 1, "reason_code": "TASK_MATERIALIZATION_REMOVED"}
    return {"deleted_count": 0, "reason_code": "TASK_MATERIALIZATION_ABSENT"}


__all__ = [
    "ALLOWED_FILES", "ALLOWED_RIGHTS", "ALLOWED_SENSITIVITY", "CONTRACT_VERSION",
    "PromotionDryRunError", "PromotionDryRunPlanner", "RAW_TEXT_POLICY_POINTER_ONLY",
    "SCHEMA_VERSION", "STATUS_DRY_RUN_READY", "TARGET_CANONICAL_LANG", "TARGET_DOC_KIND",
    "build_promotion_idempotency_key", "canonical_payload", "cleanup_materialization",
    "materialize_pointer_only_evidence", "validate_evidence_pointer", "validate_promotion_plan",
    "verify_materialization",
]
