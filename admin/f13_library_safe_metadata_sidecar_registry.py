"""Manifest and resolver governance for safe Library metadata sidecars.

The registry accepts only explicit manifest paths. It validates a sidecar hash
and returns an internal reference that Bridge retrieval helpers can consume.
It does not discover files through config, environment variables, DSNs, network
locations, or production Library roots.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from admin.f13_runtime_guard import (
    RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY,
    RAW_TEXT_POLICY_SUMMARY_ONLY,
    RESULT_HOLD,
    RESULT_OK,
    RIGHTS_INTERNAL,
    RIGHTS_LICENSED,
    RIGHTS_PUBLIC,
    normalize_raw_text_policy,
    normalize_rights_status,
)


MANIFEST_VERSION = "R9ZNW-343.v1"
SIDECAR_KIND_SQLITE_JSON = "SAFE_METADATA_SQLITE_JSON_SIDECAR"

ARTIFACT_STATE_APPROVED_SOURCE = "APPROVED_SOURCE"
ARTIFACT_STATE_PROOFPACKED = "PROOFPACKED"
ARTIFACT_STATE_CANONICAL_CANDIDATE = "CANONICAL_CANDIDATE_FOR_PLACEMENT"

SAFE_SIDECAR_APPROVED_SUMMARY = "SAFE_SIDECAR_APPROVED_SUMMARY"
APPROVED_SAFE_SUMMARY = "APPROVED_SAFE_SUMMARY"
APPROVED_SAFE_SHORT_ANSWER = "APPROVED_SAFE_SHORT_ANSWER"
CURATED_SAFE_SUMMARY = "CURATED_SAFE_SUMMARY"
SYNTHETIC_SAFE_SUMMARY = "SYNTHETIC_SAFE_SUMMARY"
VERIFIED_SEMANTIC_SUMMARY = "VERIFIED_SEMANTIC_SUMMARY"

REFRESH_MODE_EXPLICIT_REVIEW = "EXPLICIT_REVIEW_REQUIRED"

_REQUIRED_MANIFEST_FIELDS = (
    "manifest_version",
    "sidecar_id",
    "sidecar_kind",
    "created_at_utc",
    "created_by_task",
    "source_task_id",
    "source_proofpack_refs",
    "sidecar_sqlite_path",
    "sidecar_json_path",
    "sidecar_sha256",
    "record_count",
    "accepted_record_count",
    "hold_only_record_count",
    "rejected_record_count",
    "allowed_rights_statuses",
    "raw_text_policy_allowed_values",
    "summary_source_allowed_values",
    "semantic_summary_required",
    "raw_text_exposed_required_false",
    "production_path_exposed_required_false",
    "public_pointer_exposure_allowed",
    "skillup_direct_db_access_allowed",
    "production_db_write_allowed",
    "production_raw_text_read_allowed",
    "refresh_policy",
    "rollback_policy",
    "expiry_or_review_required_at",
    "approver_or_review_status",
    "artifact_state",
)

_ALLOWED_ARTIFACT_STATES = {
    ARTIFACT_STATE_APPROVED_SOURCE,
    ARTIFACT_STATE_PROOFPACKED,
    ARTIFACT_STATE_CANONICAL_CANDIDATE,
}
_ALLOWED_RIGHTS = {RIGHTS_PUBLIC, RIGHTS_INTERNAL, RIGHTS_LICENSED}
_ALLOWED_RAW_POLICIES = {
    RAW_TEXT_POLICY_SUMMARY_ONLY,
    RAW_TEXT_POLICY_REDACTED_SUMMARY_ONLY,
}
_ALLOWED_SUMMARY_SOURCES = {
    APPROVED_SAFE_SUMMARY,
    APPROVED_SAFE_SHORT_ANSWER,
    CURATED_SAFE_SUMMARY,
    SAFE_SIDECAR_APPROVED_SUMMARY,
    SYNTHETIC_SAFE_SUMMARY,
    VERIFIED_SEMANTIC_SUMMARY,
}
_SECRET_LIKE_FILENAME_MARKERS = (
    ".env",
    ".pem",
    ".key",
    "secret",
    "credential",
    "token",
    "service-account",
)
_PRODUCTION_DB_NAMES = {
    "chat.db",
    "ripple_index.sqlite",
    "warehouse.sqlite",
    "warehouse.db",
}


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_token(value: object) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _safe_text(value: object, max_length: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > max_length:
        return None
    if any(ord(char) < 32 for char in text):
        return None
    return text


def _positive(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _safe_token(value) in {"TRUE", "YES", "Y", "1"}


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _manifest_path(path_value: str | Path) -> Path:
    return Path(path_value).expanduser().resolve()


def _path_from_manifest(value: object, *, manifest_dir: Path) -> Path | None:
    text = _safe_text(value, 1024)
    if text is None:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.expanduser().resolve()


def _reject_secret_like_filename(path: Path) -> bool:
    lowered = path.name.lower()
    return any(marker in lowered for marker in _SECRET_LIKE_FILENAME_MARKERS)


def _reject_production_db_name(path: Path) -> bool:
    return path.name.lower() in _PRODUCTION_DB_NAMES


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest for an explicit sidecar artifact path."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    value = manifest.get("sidecar_sha256")
    if isinstance(value, Mapping):
        return {
            "sqlite": str(value.get("sqlite") or "").strip().lower(),
            "json": str(value.get("json") or "").strip().lower(),
        }
    return {"sqlite": str(value or "").strip().lower(), "json": ""}


def _validate_sidecar_path(
    path: Path | None,
    *,
    required: bool,
    expected_hash: str,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if path is None:
        if required:
            errors.append(f"{label} path is missing")
        return errors
    if _reject_secret_like_filename(path):
        errors.append(f"{label} filename is secret-like")
        return errors
    if _reject_production_db_name(path):
        errors.append(f"{label} path targets a production DB name")
        return errors
    if not path.is_file():
        errors.append(f"{label} sidecar file is missing")
        return errors
    if expected_hash:
        actual_hash = sha256_file(path)
        if actual_hash.lower() != expected_hash.lower():
            errors.append(f"{label} sidecar hash mismatch")
    elif required:
        errors.append(f"{label} sidecar hash is missing")
    return errors


def _validate_counts(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    for field in (
        "record_count",
        "accepted_record_count",
        "hold_only_record_count",
        "rejected_record_count",
    ):
        value = manifest.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{field} must be a non-negative integer")
            continue
        counts[field] = value
    if len(counts) == 4:
        expected = (
            counts["accepted_record_count"]
            + counts["hold_only_record_count"]
            + counts["rejected_record_count"]
        )
        if counts["record_count"] != expected:
            errors.append("record_count must equal accepted plus hold-only plus rejected counts")
    return errors


def _validate_policy_lists(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rights = {normalize_rights_status(value) for value in _as_list(manifest.get("allowed_rights_statuses"))}
    if not rights:
        errors.append("allowed_rights_statuses is empty")
    if rights - _ALLOWED_RIGHTS:
        errors.append("allowed_rights_statuses contains non-answer rights")

    raw_policies = {
        normalize_raw_text_policy(value)
        for value in _as_list(manifest.get("raw_text_policy_allowed_values"))
    }
    if not raw_policies:
        errors.append("raw_text_policy_allowed_values is empty")
    if raw_policies - _ALLOWED_RAW_POLICIES:
        errors.append("raw_text_policy_allowed_values contains unsafe policy")

    summary_sources = {_safe_token(value) for value in _as_list(manifest.get("summary_source_allowed_values"))}
    if not summary_sources:
        errors.append("summary_source_allowed_values is empty")
    if summary_sources - _ALLOWED_SUMMARY_SOURCES:
        errors.append("summary_source_allowed_values contains unapproved source")
    return errors


def validate_safe_sidecar_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a safe sidecar manifest without returning public fields."""

    errors: list[str] = []
    directory = Path(manifest_dir or ".").resolve()

    for field in _REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            errors.append(f"{field} is missing")

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append("manifest_version is unsupported")
    if manifest.get("sidecar_kind") != SIDECAR_KIND_SQLITE_JSON:
        errors.append("sidecar_kind is unsupported")
    for field in ("sidecar_id", "created_by_task", "source_task_id", "approver_or_review_status"):
        if _safe_text(manifest.get(field), 160) is None:
            errors.append(f"{field} must be a safe non-empty label")
    if not _as_list(manifest.get("source_proofpack_refs")):
        errors.append("source_proofpack_refs must be a non-empty list")

    artifact_state = str(manifest.get("artifact_state") or "")
    if artifact_state not in _ALLOWED_ARTIFACT_STATES:
        errors.append("artifact_state is not approved for resolver use")

    for field in (
        "public_pointer_exposure_allowed",
        "skillup_direct_db_access_allowed",
        "production_db_write_allowed",
        "production_raw_text_read_allowed",
    ):
        if _positive(manifest.get(field)):
            errors.append(f"{field} must be false")
        elif manifest.get(field) is not False:
            errors.append(f"{field} must be explicit false")

    for field in ("raw_text_exposed_required_false", "production_path_exposed_required_false"):
        if manifest.get(field) is not True:
            errors.append(f"{field} must be explicit true")
    if not isinstance(manifest.get("semantic_summary_required"), bool):
        errors.append("semantic_summary_required must be explicit true/false")

    errors.extend(_validate_counts(manifest))
    errors.extend(_validate_policy_lists(manifest))

    expected_hashes = _expected_hashes(manifest)
    sqlite_path = _path_from_manifest(manifest.get("sidecar_sqlite_path"), manifest_dir=directory)
    json_path = _path_from_manifest(manifest.get("sidecar_json_path"), manifest_dir=directory)
    errors.extend(
        _validate_sidecar_path(
            sqlite_path,
            required=True,
            expected_hash=expected_hashes.get("sqlite", ""),
            label="sqlite",
        )
    )
    errors.extend(
        _validate_sidecar_path(
            json_path,
            required=True,
            expected_hash=expected_hashes.get("json", ""),
            label="json",
        )
    )

    result_status = RESULT_OK if not errors else RESULT_HOLD
    return {
        "result_status": result_status,
        "ok": result_status == RESULT_OK,
        "errors": errors,
        "sidecar_id": _safe_text(manifest.get("sidecar_id"), 160),
        "artifact_state": artifact_state,
        "sidecar_sqlite_path": str(sqlite_path) if sqlite_path is not None else None,
        "sidecar_json_path": str(json_path) if json_path is not None else None,
        "record_count": manifest.get("record_count"),
        "accepted_record_count": manifest.get("accepted_record_count"),
        "hold_only_record_count": manifest.get("hold_only_record_count"),
        "rejected_record_count": manifest.get("rejected_record_count"),
    }


def resolve_safe_sidecar_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Resolve an explicit safe sidecar manifest to an internal Bridge reference."""

    path = _manifest_path(manifest_path)
    if _reject_secret_like_filename(path):
        return {
            "result_status": RESULT_HOLD,
            "ok": False,
            "hold_reason": "manifest filename is secret-like",
            "errors": ["manifest filename is secret-like"],
        }
    try:
        with path.open("r", encoding="utf-8-sig") as input_file:
            manifest = json.load(input_file)
    except (OSError, json.JSONDecodeError):
        return {
            "result_status": RESULT_HOLD,
            "ok": False,
            "hold_reason": "safe sidecar manifest could not be read",
            "errors": ["safe sidecar manifest could not be read"],
        }
    if not isinstance(manifest, Mapping):
        return {
            "result_status": RESULT_HOLD,
            "ok": False,
            "hold_reason": "safe sidecar manifest is not an object",
            "errors": ["safe sidecar manifest is not an object"],
        }

    validation = validate_safe_sidecar_manifest(manifest, manifest_dir=path.parent)
    if validation["result_status"] != RESULT_OK:
        errors = [str(item) for item in validation.get("errors", [])]
        return {
            "result_status": RESULT_HOLD,
            "ok": False,
            "hold_reason": errors[0] if errors else "safe sidecar manifest requires review",
            "errors": errors,
        }

    return {
        "result_status": RESULT_OK,
        "ok": True,
        "manifest_path": str(path),
        "sidecar_id": validation["sidecar_id"],
        "sidecar_kind": manifest["sidecar_kind"],
        "sidecar_sqlite_path": validation["sidecar_sqlite_path"],
        "sidecar_json_path": validation["sidecar_json_path"],
        "record_count": validation["record_count"],
        "accepted_record_count": validation["accepted_record_count"],
        "hold_only_record_count": validation["hold_only_record_count"],
        "rejected_record_count": validation["rejected_record_count"],
        "artifact_state": validation["artifact_state"],
        "table_name": manifest.get("table_name") or "bridge_evidence",
        "public_pointer_exposure_allowed": False,
        "skillup_direct_db_access_allowed": False,
        "production_db_write_allowed": False,
        "production_raw_text_read_allowed": False,
    }


def create_safe_sidecar_manifest(
    *,
    sidecar_id: str,
    created_by_task: str,
    source_task_id: str,
    source_proofpack_refs: Sequence[str],
    sidecar_sqlite_path: str | Path,
    sidecar_json_path: str | Path,
    record_count: int,
    accepted_record_count: int,
    hold_only_record_count: int,
    rejected_record_count: int,
    artifact_state: str = ARTIFACT_STATE_PROOFPACKED,
    approver_or_review_status: str = "PROOFPACKED_TASK_EVIDENCE_ONLY",
    expiry_or_review_required_at: str = "NEXT_PLACEMENT_GATE_REVIEW_REQUIRED",
    allowed_rights_statuses: Sequence[str] = (RIGHTS_INTERNAL, RIGHTS_PUBLIC, RIGHTS_LICENSED),
    raw_text_policy_allowed_values: Sequence[str] = (RAW_TEXT_POLICY_SUMMARY_ONLY,),
    summary_source_allowed_values: Sequence[str] = (SAFE_SIDECAR_APPROVED_SUMMARY,),
    semantic_summary_required: bool = False,
    table_name: str = "bridge_evidence",
) -> dict[str, Any]:
    """Create a manifest dict for explicit task-owned sidecar artifacts."""

    sqlite_path = Path(sidecar_sqlite_path).expanduser().resolve()
    json_path = Path(sidecar_json_path).expanduser().resolve()
    return {
        "manifest_version": MANIFEST_VERSION,
        "sidecar_id": sidecar_id,
        "sidecar_kind": SIDECAR_KIND_SQLITE_JSON,
        "created_at_utc": _created_at(),
        "created_by_task": created_by_task,
        "source_task_id": source_task_id,
        "source_proofpack_refs": list(source_proofpack_refs),
        "sidecar_sqlite_path": str(sqlite_path),
        "sidecar_json_path": str(json_path),
        "sidecar_sha256": {
            "sqlite": sha256_file(sqlite_path),
            "json": sha256_file(json_path),
        },
        "table_name": table_name,
        "record_count": int(record_count),
        "accepted_record_count": int(accepted_record_count),
        "hold_only_record_count": int(hold_only_record_count),
        "rejected_record_count": int(rejected_record_count),
        "allowed_rights_statuses": list(allowed_rights_statuses),
        "raw_text_policy_allowed_values": list(raw_text_policy_allowed_values),
        "summary_source_allowed_values": list(summary_source_allowed_values),
        "semantic_summary_required": bool(semantic_summary_required),
        "raw_text_exposed_required_false": True,
        "production_path_exposed_required_false": True,
        "public_pointer_exposure_allowed": False,
        "skillup_direct_db_access_allowed": False,
        "production_db_write_allowed": False,
        "production_raw_text_read_allowed": False,
        "refresh_policy": {
            "mode": REFRESH_MODE_EXPLICIT_REVIEW,
            "automatic_refresh_allowed": False,
            "approved_safe_metadata_sources_only": True,
            "new_sidecar_id_required": True,
            "new_hash_required": True,
            "preserve_prior_sidecar_until_reviewed_replacement": True,
            "emit_accept_reject_hold_counts": True,
        },
        "rollback_policy": {
            "preserve_prior_sidecar": True,
            "rollback_by_manifest_repoint_after_review": True,
            "production_db_mutation_required": False,
        },
        "expiry_or_review_required_at": expiry_or_review_required_at,
        "approver_or_review_status": approver_or_review_status,
        "artifact_state": artifact_state,
    }


def write_safe_sidecar_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write a manifest JSON file to an explicit caller-provided path."""

    target = Path(output_path).expanduser().resolve()
    if _reject_secret_like_filename(target):
        raise ValueError("manifest output filename is secret-like")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def refresh_governance_contract() -> dict[str, Any]:
    """Return the explicit refresh governance contract used by the resolver."""

    return {
        "refresh_mode": REFRESH_MODE_EXPLICIT_REVIEW,
        "automatic_refresh_allowed": False,
        "approved_safe_metadata_sources_only": True,
        "new_sidecar_id_required": True,
        "new_hash_required": True,
        "prior_sidecar_preserved_until_review": True,
        "production_db_write_allowed": False,
        "production_raw_text_read_allowed": False,
        "public_pointer_exposure_allowed": False,
        "accept_reject_hold_counts_required": True,
        "review_status_required_before_placement": True,
        "failure_keeps_prior_sidecar_valid": True,
    }


def validate_refresh_proposal(
    current_manifest: Mapping[str, Any],
    proposed_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a sidecar refresh is reviewable and non-mutating."""

    errors: list[str] = []
    current = _as_mapping(current_manifest)
    proposed = _as_mapping(proposed_manifest)
    if not current or not proposed:
        errors.append("current and proposed manifests are required")
    if current.get("sidecar_id") == proposed.get("sidecar_id"):
        errors.append("refresh must produce a new sidecar_id")
    if current.get("sidecar_sha256") == proposed.get("sidecar_sha256"):
        errors.append("refresh must produce a new sidecar hash")

    validation = validate_safe_sidecar_manifest(proposed)
    if validation.get("result_status") != RESULT_OK:
        errors.extend(str(item) for item in validation.get("errors", []))

    if proposed.get("refresh_policy") != current.get("refresh_policy"):
        proposed_policy = _as_mapping(proposed.get("refresh_policy"))
        if _positive(proposed_policy.get("automatic_refresh_allowed")):
            errors.append("refresh must remain explicit")
    if not _safe_text(proposed.get("approver_or_review_status"), 160):
        errors.append("review status is required before placement")

    status = RESULT_OK if not errors else RESULT_HOLD
    return {
        "result_status": status,
        "ok": status == RESULT_OK,
        "errors": errors,
        **refresh_governance_contract(),
    }


__all__ = [
    "APPROVED_SAFE_SHORT_ANSWER",
    "APPROVED_SAFE_SUMMARY",
    "ARTIFACT_STATE_APPROVED_SOURCE",
    "ARTIFACT_STATE_CANONICAL_CANDIDATE",
    "ARTIFACT_STATE_PROOFPACKED",
    "CURATED_SAFE_SUMMARY",
    "MANIFEST_VERSION",
    "REFRESH_MODE_EXPLICIT_REVIEW",
    "SAFE_SIDECAR_APPROVED_SUMMARY",
    "SIDECAR_KIND_SQLITE_JSON",
    "SYNTHETIC_SAFE_SUMMARY",
    "VERIFIED_SEMANTIC_SUMMARY",
    "create_safe_sidecar_manifest",
    "refresh_governance_contract",
    "resolve_safe_sidecar_manifest",
    "sha256_file",
    "validate_refresh_proposal",
    "validate_safe_sidecar_manifest",
    "write_safe_sidecar_manifest",
]
