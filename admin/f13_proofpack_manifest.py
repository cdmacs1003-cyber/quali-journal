from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


CREATED_AT = "1970-01-01T00:00:00Z"
MANIFEST_VERSION = "0.1"
NOT_GRANTED = "NOT_GRANTED"
NOT_VERIFIED = "NOT_VERIFIED"
NOT_EXECUTED = "NOT_EXECUTED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
REJECT = "REJECT"
APPROVE = "APPROVE"

_DEFAULT_REQUIRED_ARTIFACTS = (
    "selected_evidence_snapshot",
    "release_board",
    "gate_results",
)
_SAFETY_CRITICAL_GATES = (
    "BRIDGE_BOUNDARY",
    "EVIDENCE_REQUIRED",
    "RAW_LEAK",
    "FEEDBACK_LOOP",
)
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


def _safe_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text):
        return text
    return ""


def _normal_label(value: Any, fallback: str = "UNKNOWN") -> str:
    text = _safe_text(value, fallback, max_length=96)
    label = "".join(ch if ch.isalnum() else "_" for ch in text.upper())
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or fallback


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _artifact_descriptors(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    descriptors: list[dict[str, str]] = []
    for index, item in enumerate(value[:50], start=1):
        if not isinstance(item, Mapping):
            continue
        descriptors.append(
            {
                "name": _safe_token(item.get("name"), f"artifact:{index}"),
                "repo_relative_path": _safe_repo_path(item.get("repo_relative_path")),
                "sha256": _safe_sha256(item.get("sha256")),
                "evidence_type": _safe_token(item.get("evidence_type"), "evidence"),
                "status": _normal_label(item.get("status"), "RECORDED"),
            }
        )
    return descriptors


def _required_artifacts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(_DEFAULT_REQUIRED_ARTIFACTS)
    required = [_safe_token(item, "") for item in value if _safe_token(item, "")]
    return required or list(_DEFAULT_REQUIRED_ARTIFACTS)


def _gate_results(value: Any) -> dict[str, str]:
    results: dict[str, str] = {}
    if isinstance(value, Mapping):
        for name, status in value.items():
            safe_name = _normal_label(name)
            if isinstance(status, Mapping):
                results[safe_name] = _normal_label(status.get("status"), "UNKNOWN")
            else:
                results[safe_name] = _normal_label(status, "UNKNOWN")
        return dict(sorted(results.items()))

    if isinstance(value, list):
        for index, item in enumerate(value, start=1):
            if not isinstance(item, Mapping):
                continue
            safe_name = _normal_label(item.get("name") or item.get("gate") or f"gate:{index}")
            results[safe_name] = _normal_label(item.get("status"), "UNKNOWN")
    return dict(sorted(results.items()))


def _input_not_verified(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = [_normal_label(item) for item in value]
    return sorted({label for label in labels if label})


def _artifact_hashes(descriptors: list[dict[str, str]]) -> dict[str, str]:
    return {
        descriptor["name"]: descriptor["sha256"]
        for descriptor in descriptors
        if descriptor.get("name") and descriptor.get("sha256")
    }


def _open_items(
    *,
    required_artifacts: list[str],
    artifact_hashes: Mapping[str, str],
    gate_results: Mapping[str, str],
    release_board_ref: str,
    not_verified_items: list[str],
) -> list[str]:
    items: list[str] = []
    if not release_board_ref:
        items.append("RELEASE_BOARD_REF_REQUIRED")
    if not gate_results:
        items.append("GATE_RESULTS_REQUIRED")

    for artifact_name in required_artifacts:
        if not artifact_hashes.get(artifact_name):
            items.append(f"MISSING_ARTIFACT_HASH:{artifact_name}")

    for gate, status in gate_results.items():
        if status in {NOT_EXECUTED, NOT_VERIFIED, "UNKNOWN", ""}:
            items.append(f"GATE_NOT_VERIFIED:{gate}")
        elif "NOT_EXECUTED" in status or "NOT_VERIFIED" in status:
            items.append(f"GATE_NOT_VERIFIED:{gate}")

    for item in not_verified_items:
        items.append(f"NOT_VERIFIED:{item}")

    return sorted(set(items))


def _failed_safety_gate(gate_results: Mapping[str, str]) -> bool:
    for gate, status in gate_results.items():
        if status not in {"FAIL", "FAILED", "REJECT", "REJECTED"}:
            continue
        if any(critical in gate for critical in _SAFETY_CRITICAL_GATES):
            return True
    return False


def build_f13_proofpack_manifest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, Mapping) else {}
    project = _safe_text(source.get("project"), "QLIB Track A")
    scope = _safe_text(source.get("scope"), "F13 ProofPack Manifest")
    source_snapshot_ref = _safe_token(source.get("source_snapshot_ref"), "snapshot:not_recorded")
    release_board_ref = _safe_token(source.get("release_board_ref"), "")
    required_artifacts = _required_artifacts(source.get("required_artifacts"))
    descriptors = _artifact_descriptors(source.get("artifacts"))
    artifact_hashes = _artifact_hashes(descriptors)
    gate_results = _gate_results(source.get("gate_results"))
    not_verified_items = _input_not_verified(source.get("not_verified_items"))
    open_items = _open_items(
        required_artifacts=required_artifacts,
        artifact_hashes=artifact_hashes,
        gate_results=gate_results,
        release_board_ref=release_board_ref,
        not_verified_items=not_verified_items,
    )

    if _failed_safety_gate(gate_results):
        final_recommendation = REJECT
    elif open_items:
        final_recommendation = REVIEW_REQUIRED
    else:
        final_recommendation = APPROVE

    signature = "|".join(
        [
            project,
            scope,
            source_snapshot_ref,
            release_board_ref,
            ",".join(required_artifacts),
            ",".join(f"{name}:{artifact_hashes.get(name, '')}" for name in sorted(required_artifacts)),
            ",".join(f"{name}:{status}" for name, status in sorted(gate_results.items())),
            ",".join(open_items),
        ]
    )

    return {
        "proofpack_id": f"proofpack-manifest:{_stable_digest(signature)}",
        "manifest_version": MANIFEST_VERSION,
        "project": project,
        "scope": scope,
        "created_at": CREATED_AT,
        "source_snapshot_ref": source_snapshot_ref,
        "required_artifacts": required_artifacts,
        "artifact_hashes": artifact_hashes,
        "artifact_descriptors": descriptors,
        "gate_results": dict(gate_results),
        "release_board_ref": release_board_ref,
        "open_items": open_items,
        "not_granted_claims": {
            "f13_pass": NOT_GRANTED,
            "track_a_pass": NOT_GRANTED,
            "beta_pass": NOT_GRANTED,
        },
        "not_verified_items": not_verified_items,
        "final_recommendation": final_recommendation,
    }


__all__ = [
    "APPROVE",
    "CREATED_AT",
    "MANIFEST_VERSION",
    "NOT_EXECUTED",
    "NOT_GRANTED",
    "NOT_VERIFIED",
    "REJECT",
    "REVIEW_REQUIRED",
    "build_f13_proofpack_manifest",
]
