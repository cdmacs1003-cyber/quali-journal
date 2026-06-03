from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


CREATED_AT = "1970-01-01T00:00:00Z"
GAP_MAP_VERSION = "0.1"
NOT_GRANTED = "NOT_GRANTED"
NOT_VERIFIED = "NOT_VERIFIED"
NOT_EXECUTED = "NOT_EXECUTED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"

REQUIRED_F13_CONTRACTS = (
    "F13_SPEC_MATERIALIZED",
    "JSON_SCHEMAS_MATERIALIZED",
    "GAP_MAP_MATERIALIZED",
    "BRIDGE_BOUNDARY_ENFORCED",
    "EVIDENCE_REQUIRED_ENFORCED",
    "RAW_LEAK_ENFORCED",
    "FEEDBACK_LOOP_ENFORCED",
    "PROOFPACK_MANIFEST_PRESENT",
    "RELEASE_BOARD_PRESENT",
    "GATE_RESULTS_PRESENT",
    "FINAL_APPROVAL_RECORDED",
)

_LIMITED_CLOSED_STATUSES = {
    "PASS",
    "PASSED",
    "OK",
    "EXECUTED",
    "RECORDED",
    "SELECTED_TESTED",
    "OBSERVED_LIMITED",
    "LIMITED_ENDPOINT_RESULT_ONLY",
    "LIMITED_ROUTE_RUNTIME_RESULT",
    "VERIFIED",
    "VERIFIED_LIMITED",
    "CLOSED_LIMITED",
}
_UNSAFE_FIELD_MARKERS = (
    "raw_text",
    "raw_prompt",
    "raw_query",
    "full_source_text",
    "internal_path",
    "local_path",
    "secret",
    "token",
    "credential",
    "api_key",
    "access_key",
    "secret_key",
    "dsn",
    "customer_confidential",
    "paid_standard",
)
_UNSAFE_VALUE_MARKERS = (
    "raw text",
    "raw prompt",
    "raw query",
    "full source text",
    "internal path",
    "h:\\",
    "c:\\",
    "file://",
    "secret",
    "token",
    "credential",
    "api key",
    "dsn",
    "customer confidential",
    "paid standard",
)
_OPEN_STATUS_ITEMS = {
    "db_behavior": "DB_BEHAVIOR_NOT_VERIFIED",
    "production_raw_leak_safety": "PRODUCTION_RAW_LEAK_SAFETY_NOT_VERIFIED",
    "full_regression_safety": "FULL_REGRESSION_SAFETY_NOT_VERIFIED",
    "actual_proofpack_generation": "ACTUAL_PROOFPACK_GENERATION_NOT_EXECUTED",
    "final_approval": "FINAL_APPROVAL_NOT_RECORDED",
}


def _unsafe_key(key: Any) -> bool:
    lowered = str(key or "").lower()
    return any(marker in lowered for marker in _UNSAFE_FIELD_MARKERS)


def _unsafe_value(value: Any) -> bool:
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in _UNSAFE_VALUE_MARKERS)


def _safe_text(value: Any, fallback: str, max_length: int = 180) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    if _unsafe_value(text):
        return "redacted_safety_summary"
    return text[:max_length]


def _safe_token(value: Any, fallback: str, max_length: int = 96) -> str:
    text = _safe_text(value, fallback, max_length=max_length)
    token = "".join(ch for ch in text if ch.isalnum() or ch in ":._-/")
    return token.strip("/") or fallback


def _normal_label(value: Any, fallback: str = "UNKNOWN") -> str:
    text = _safe_text(value, fallback, max_length=96)
    label = "".join(ch if ch.isalnum() else "_" for ch in text.upper())
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or fallback


def _safe_repo_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    lowered = text.lower()
    if (
        not text
        or _unsafe_value(text)
        or text.startswith("/")
        or ".." in text.split("/")
        or "://" in text
        or len(text) > 180
        or len(text) > 1 and text[1] == ":"
        or "secret" in lowered
        or "credential" in lowered
        or "token" in lowered
    ):
        return ""
    return "".join(ch for ch in text if ch.isalnum() or ch in "._-/")


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _required_contracts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(REQUIRED_F13_CONTRACTS)
    contracts = [_normal_label(item, "") for item in value if _normal_label(item, "")]
    return contracts or list(REQUIRED_F13_CONTRACTS)


def _limited_status(value: Any) -> str:
    status = _normal_label(value, "UNKNOWN")
    if status in {NOT_EXECUTED, NOT_VERIFIED, "UNKNOWN", ""}:
        return status
    if status in {"FAIL", "FAILED", "REJECT", "REJECTED"}:
        return REVIEW_REQUIRED
    if status in _LIMITED_CLOSED_STATUSES or "PASS" in status or "VERIFIED" in status:
        return "VERIFIED_LIMITED"
    return "CLOSED_LIMITED"


def _observed_surfaces(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    surfaces: list[dict[str, Any]] = []
    for index, item in enumerate(value[:80], start=1):
        if not isinstance(item, Mapping):
            continue

        unsafe_payload = any(_unsafe_key(key) for key in item) or any(
            _unsafe_value(child) for child in item.values()
        )
        surface_id = _normal_label(
            item.get("surface_id") or item.get("contract") or item.get("name") or f"surface:{index}",
            f"SURFACE_{index}",
        )
        contract = _normal_label(item.get("contract") or item.get("required_contract") or surface_id, surface_id)
        safe_surface = {
            "surface_id": surface_id,
            "contract": contract,
            "status": _limited_status(item.get("status") or item.get("result")),
            "evidence_ref": _safe_token(item.get("evidence_ref") or item.get("gate"), f"evidence:{index}"),
            "commit_id": _safe_token(item.get("commit_id") or item.get("commit"), "commit:not_recorded"),
            "test_count": int(item.get("test_count") or item.get("tests") or 0),
            "summary": _safe_text(item.get("summary"), "bounded evidence recorded"),
            "repo_relative_path": _safe_repo_path(item.get("repo_relative_path")),
        }
        if unsafe_payload:
            safe_surface["summary"] = "redacted_safety_summary"
            safe_surface["repo_relative_path"] = ""
        surfaces.append(safe_surface)
    return sorted(surfaces, key=lambda surface: surface["surface_id"])


def _status_items(source: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    open_items: list[str] = []
    not_executed_items: list[str] = []
    not_verified_items: list[str] = []

    for key, label in _OPEN_STATUS_ITEMS.items():
        status = _normal_label(source.get(key), "")
        if status in {"", NOT_EXECUTED, NOT_VERIFIED, "UNKNOWN"}:
            open_items.append(label)
        if status in {"", NOT_EXECUTED} and "NOT_EXECUTED" in label:
            not_executed_items.append(label)
        if status in {"", NOT_VERIFIED, NOT_EXECUTED, "UNKNOWN"} and "NOT_VERIFIED" in label:
            not_verified_items.append(label)
        if key == "final_approval" and status not in {"APPROVED", "RECORDED"}:
            open_items.append(label)

    extra_not_executed = source.get("not_executed_items")
    if isinstance(extra_not_executed, list):
        not_executed_items.extend(_normal_label(item) for item in extra_not_executed)

    extra_not_verified = source.get("not_verified_items")
    if isinstance(extra_not_verified, list):
        not_verified_items.extend(_normal_label(item) for item in extra_not_verified)

    return sorted(set(open_items)), sorted(set(not_executed_items)), sorted(set(not_verified_items))


def build_f13_gap_map(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, Mapping) else {}
    project = _safe_text(source.get("project"), "QLIB Track A")
    scope = _safe_text(source.get("scope"), "F13 Gap Map")
    source_snapshot_ref = _safe_token(source.get("source_snapshot_ref"), "snapshot:not_recorded")
    required_contracts = _required_contracts(source.get("required_f13_contracts"))
    observed_surfaces = _observed_surfaces(source.get("observed_surfaces"))
    observed_by_contract = {surface["contract"]: surface for surface in observed_surfaces}

    closed_items: list[dict[str, str]] = []
    gap_items: list[dict[str, str]] = []
    open_items, not_executed_items, not_verified_items = _status_items(source)

    for contract in required_contracts:
        surface = observed_by_contract.get(contract)
        if surface and surface["status"] in {"CLOSED_LIMITED", "VERIFIED_LIMITED"}:
            closed_items.append(
                {
                    "contract": contract,
                    "status": surface["status"],
                    "evidence_ref": surface["evidence_ref"],
                }
            )
            continue

        if surface:
            reason = f"CONTRACT_NOT_FULLY_VERIFIED:{contract}"
            if surface["status"] == NOT_EXECUTED:
                not_executed_items.append(contract)
            if surface["status"] in {NOT_EXECUTED, NOT_VERIFIED, "UNKNOWN", REVIEW_REQUIRED}:
                not_verified_items.append(contract)
        else:
            reason = f"MISSING_CONTRACT:{contract}"
            not_verified_items.append(contract)

        gap_items.append({"contract": contract, "status": REVIEW_REQUIRED, "reason": reason})
        open_items.append(reason)

    final_recommendation = REVIEW_REQUIRED
    recommended_next_actions = [
        f"RESOLVE:{item}" for item in sorted(set(open_items))[:20]
    ] or ["CONTINUE_SELECTED_REVIEW"]
    signature = "|".join(
        [
            project,
            scope,
            source_snapshot_ref,
            ",".join(required_contracts),
            ",".join(
                f"{surface['surface_id']}:{surface['contract']}:{surface['status']}:{surface['evidence_ref']}"
                for surface in observed_surfaces
            ),
            ",".join(sorted(set(open_items))),
            ",".join(sorted(set(not_executed_items))),
            ",".join(sorted(set(not_verified_items))),
        ]
    )

    return {
        "gap_map_id": f"f13-gap-map:{_stable_digest(signature)}",
        "gap_map_version": GAP_MAP_VERSION,
        "project": project,
        "scope": scope,
        "created_at": CREATED_AT,
        "source_snapshot_ref": source_snapshot_ref,
        "observed_surfaces": observed_surfaces,
        "required_f13_contracts": required_contracts,
        "gap_items": sorted(gap_items, key=lambda item: item["contract"]),
        "closed_items": sorted(closed_items, key=lambda item: item["contract"]),
        "open_items": sorted(set(open_items)),
        "not_executed_items": sorted(set(not_executed_items)),
        "not_verified_items": sorted(set(not_verified_items)),
        "not_granted_claims": {
            "f13_pass": NOT_GRANTED,
            "track_a_pass": NOT_GRANTED,
            "beta_pass": NOT_GRANTED,
        },
        "recommended_next_actions": recommended_next_actions,
        "final_recommendation": final_recommendation,
    }


__all__ = [
    "CREATED_AT",
    "GAP_MAP_VERSION",
    "NOT_EXECUTED",
    "NOT_GRANTED",
    "NOT_VERIFIED",
    "REQUIRED_F13_CONTRACTS",
    "REVIEW_REQUIRED",
    "build_f13_gap_map",
]
