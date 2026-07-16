"""Detached, reconnectable traffic-observation controller.

The controller owns only its observation directory and child process.  Local-test
mode is deterministic and performs no network or HTTP work.  Production mode
delegates one sample at a time to an approved sampler whose argv is supplied in
the process-only ``QLIB_OBSERVER_PRODUCTION_SAMPLER`` environment variable.  The
argv and the sampler's raw output are never written to artifacts.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PRODUCTION_SAMPLER_ENV = "QLIB_OBSERVER_PRODUCTION_SAMPLER"
IDENTITY_MATERIAL_ENV = "QLIB_OBSERVER_IDENTITY_MATERIAL"
OBSERVATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_SAMPLE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SAFE_CONTRACT_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
SHA256_HEX_PATTERN = re.compile(r"^[a-f0-9]{64}$")
SENSITIVE_KEY_PARTS = (
    "answer",
    "authorization",
    "credential",
    "email",
    "path",
    "project",
    "query",
    "revision",
    "secret",
    "service",
    "token",
    "url",
)
RAW_LOCATION_PATTERN = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|[A-Za-z]:[\\/]|(?:^|\s)[\\/]{2}|[^\s@]+@[^\s@]+)"
)
STOP_REASONS = ("FAILURE_INJECTION", "OWNER_STOP", "TASK_CLEANUP")
PRODUCTION_TARGET_CONTRACTS = (
    "HEALTH_ONLY_SERVICE",
    "REVISION_FUNCTIONAL",
    "SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL",
)
SAMPLER_FAILURE_CATEGORIES = (
    "HTTP_403",
    "HTTP_404",
    "HTTP_5XX",
    "TIMEOUT",
    "JSON_PARSE_FAILURE",
    "AUTH_FAILURE",
    "DEPENDENCY_OR_SUBPROCESS_DEFECT",
    "ARGUMENT_OR_SERIALIZATION_DEFECT",
    "DETACHED_ENVIRONMENT_OR_IMPORT_DEFECT",
    "TARGET_ROUTING_CONTRACT_DEFECT",
    "FUNCTIONAL_HTTP_FAILURE",
    "VERIFIED_EXTERNAL_TRANSIENT",
)
MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRIES = 1
MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRY_SECONDS = 10.0
SAMPLER_TOTAL_DEADLINE_SECONDS = 30.0
SAMPLER_PROGRESS_FILE_ENV = "QLIB_SAMPLER_PROGRESS_FILE"
SAMPLER_DEADLINE_ENV = "QLIB_SAMPLER_TOTAL_DEADLINE_SECONDS"
SAMPLER_PHASES = (
    "AUTHENTICATION",
    "TARGET_RESOLUTION",
    "REQUEST_PREPARATION",
    "HTTP_REQUEST",
    "NORMALIZATION",
    "EVIDENCE_TRACE_SAFE_EXTRACTION",
    "SERIALIZATION",
    "PARENT_CHILD_IPC",
    "CHILD_CLEANUP",
    "OBSERVER_COLLECTION",
)

# R479 staged-traffic decision contract.  These values are deliberately kept
# in the observer module so the runbook and focused tests share one executable
# source of truth without coupling the detached lifecycle to cloud APIs.
UNAUTHENTICATED_EXPECTED_STATUS = 403
AUTHENTICATED_EXPECTED_STATUS = 200
HEALTH_NORMALIZED_VALUE = "ok"
PROBE_TIMEOUT_SECONDS = 10.0
SYNTHETIC_HEALTH_FAILURE_ALLOWED = 0
SYNTHETIC_AUTH_FAILURE_ALLOWED = 0
UNEXPECTED_5XX_ALLOWED = 0
EVIDENCE_TRACE_MISSING_ALLOWED = 0
RAW_SECRET_INTERNAL_PATH_EXPOSURE_ALLOWED = 0
PRODUCTION_WRITE_ALLOWED = 0

ABSOLUTE_P95_LATENCY_LIMIT_MS = 3000.0
RELATIVE_P95_MULTIPLIER_FROM_STABLE_BASELINE = 2.0
AGGREGATE_5XX_ABSOLUTE_LIMIT_PERCENT = 1.0
AGGREGATE_5XX_BASELINE_DELTA_LIMIT_PERCENTAGE_POINT = 0.5
MINIMUM_AGGREGATE_WINDOW_REQUEST_COUNT = 20
CONSECUTIVE_BREACH_WINDOW_LIMIT = 2

NEW_CANDIDATE_EFFECTIVE_MIN = 0
NEW_CANDIDATE_EFFECTIVE_MAX = 2
NEW_CANDIDATE_IMMUTABLE_MAXSCALE = 2
NEW_CANDIDATE_ACTIVE_INSTANCE_LIMIT = 2
FAILED_STARTUP_ALLOWED = 0
REQUEST_DROP_OR_THROTTLE_ALLOWED = 0
PENDING_NOT_READY_LIMIT_SECONDS = 120
CONCURRENCY_EXPECTED = 80
CPU_EXPECTED = 1.0
MEMORY_EXPECTED = "512Mi"
TIMEOUT_EXPECTED_SECONDS = 300

CANDIDATE_REVISION_CREATION_LIMIT = 1
TOTAL_AUTHORIZED_OBSERVATION_SECONDS = 3000
UNEXPECTED_BILLABLE_RESOURCE_DELTA_ALLOWED = 0
IMAGE_PUSH_ALLOWED = 0
CLOUD_SQL_BINDING_ALLOWED = 0
ADDITIONAL_SERVICE_OR_SCHEDULER_ALLOWED = 0


def normalize_health_value(value: Any) -> str:
    """Normalize the bounded health value without accepting other values."""

    return value.strip().lower() if isinstance(value, str) else ""


def latency_stop_limit_ms(stable_baseline_p95_ms: float) -> float:
    baseline = float(stable_baseline_p95_ms)
    if not math.isfinite(baseline) or baseline < 0:
        raise ObserverError("stable baseline p95 must be a finite non-negative value")
    return max(
        ABSOLUTE_P95_LATENCY_LIMIT_MS,
        baseline * RELATIVE_P95_MULTIPLIER_FROM_STABLE_BASELINE,
    )


def error_rate_stop_limit_percent(stable_baseline_5xx_rate_percent: float) -> float:
    baseline = float(stable_baseline_5xx_rate_percent)
    if not math.isfinite(baseline) or baseline < 0:
        raise ObserverError("stable baseline 5xx rate must be a finite non-negative value")
    return max(
        AGGREGATE_5XX_ABSOLUTE_LIMIT_PERCENT,
        baseline + AGGREGATE_5XX_BASELINE_DELTA_LIMIT_PERCENTAGE_POINT,
    )


def evaluate_functional_contract(
    *,
    unauthenticated_status: int,
    authenticated_status: int,
    health_value: Any,
    timeout_count: int = 0,
    unexpected_5xx_count: int = 0,
    evidence_trace_missing_count: int = 0,
    exposure_count: int = 0,
    production_write_count: int = 0,
) -> dict[str, Any]:
    normalized = normalize_health_value(health_value)
    health_failures = int(normalized != HEALTH_NORMALIZED_VALUE)
    auth_failures = int(
        unauthenticated_status != UNAUTHENTICATED_EXPECTED_STATUS
        or authenticated_status != AUTHENTICATED_EXPECTED_STATUS
    )
    failures = {
        "synthetic_health_failure_count": health_failures,
        "synthetic_auth_failure_count": auth_failures,
        "timeout_count": int(timeout_count),
        "unexpected_5xx_count": int(unexpected_5xx_count),
        "evidence_trace_missing_count": int(evidence_trace_missing_count),
        "exposure_count": int(exposure_count),
        "production_write_count": int(production_write_count),
    }
    passed = (
        failures["synthetic_health_failure_count"]
        <= SYNTHETIC_HEALTH_FAILURE_ALLOWED
        and failures["synthetic_auth_failure_count"] <= SYNTHETIC_AUTH_FAILURE_ALLOWED
        and failures["timeout_count"] == 0
        and failures["unexpected_5xx_count"] <= UNEXPECTED_5XX_ALLOWED
        and failures["evidence_trace_missing_count"]
        <= EVIDENCE_TRACE_MISSING_ALLOWED
        and failures["exposure_count"]
        <= RAW_SECRET_INTERNAL_PATH_EXPOSURE_ALLOWED
        and failures["production_write_count"] <= PRODUCTION_WRITE_ALLOWED
    )
    return {
        "status": "PASS" if passed else "STOP",
        "normalized_health": normalized,
        **failures,
    }


def evaluate_aggregate_windows(
    windows: list[dict[str, Any]],
    *,
    stable_baseline_p95_ms: float,
    stable_baseline_5xx_rate_percent: float,
) -> dict[str, Any]:
    latency_limit = latency_stop_limit_ms(stable_baseline_p95_ms)
    error_limit = error_rate_stop_limit_percent(stable_baseline_5xx_rate_percent)
    latency_consecutive = 0
    error_consecutive = 0
    latency_stop = False
    error_stop = False
    insufficient_count = 0
    window_results: list[dict[str, Any]] = []

    for index, window in enumerate(windows):
        request_count = int(window.get("request_count", 0))
        if request_count < MINIMUM_AGGREGATE_WINDOW_REQUEST_COUNT:
            insufficient_count += 1
            latency_consecutive = 0
            error_consecutive = 0
            window_results.append(
                {
                    "index": index,
                    "request_count": request_count,
                    "status": "INSUFFICIENT_DATA",
                }
            )
            continue

        p95_latency_ms = float(window["p95_latency_ms"])
        rate_percent = float(window["five_xx_rate_percent"])
        latency_breach = p95_latency_ms > latency_limit
        error_breach = rate_percent > error_limit
        latency_consecutive = latency_consecutive + 1 if latency_breach else 0
        error_consecutive = error_consecutive + 1 if error_breach else 0
        latency_stop = latency_stop or (
            latency_consecutive >= CONSECUTIVE_BREACH_WINDOW_LIMIT
        )
        error_stop = error_stop or (
            error_consecutive >= CONSECUTIVE_BREACH_WINDOW_LIMIT
        )
        window_results.append(
            {
                "index": index,
                "request_count": request_count,
                "status": "PASS",
                "latency_breach": latency_breach,
                "error_rate_breach": error_breach,
            }
        )

    stop = latency_stop or error_stop
    status = (
        "STOP"
        if stop
        else "INSUFFICIENT_DATA"
        if insufficient_count
        else "PASS"
    )
    return {
        "status": status,
        "aggregate_metric_status": status,
        "fallback_required": insufficient_count > 0,
        "latency_stop_limit_ms": latency_limit,
        "error_rate_stop_limit_percent": error_limit,
        "latency_stop": latency_stop,
        "error_rate_stop": error_stop,
        "window_results": window_results,
    }


def evaluate_low_volume_fallback(
    *,
    synthetic_health_failure_count: int,
    synthetic_auth_failure_count: int,
    unexpected_synthetic_5xx_count: int,
    synthetic_p95_latency_ms: float,
    latency_limit_ms: float,
    timeout_count: int,
    evidence_trace_missing_count: int,
    capacity_status: str,
    cost_proxy_status: str,
    observer_final_status: str,
) -> dict[str, Any]:
    passed = (
        synthetic_health_failure_count == 0
        and synthetic_auth_failure_count == 0
        and unexpected_synthetic_5xx_count == 0
        and float(synthetic_p95_latency_ms) <= float(latency_limit_ms)
        and timeout_count == 0
        and evidence_trace_missing_count == 0
        and capacity_status == "PASS"
        and cost_proxy_status == "PASS_WITH_LIMITS"
        and observer_final_status == "PASS"
    )
    return {
        "status": "PASS_WITH_LOW_VOLUME_LIMITS" if passed else "STOP",
        "aggregate_metric_status": "INSUFFICIENT_DATA",
        "fallback_required": True,
    }


def evaluate_capacity_contract(
    *,
    effective_min: int,
    effective_max: int,
    immutable_maxscale: int,
    active_instances: int,
    failed_startup_count: int,
    request_drop_or_throttle_count: int,
    pending_not_ready_seconds: float,
    concurrency: int,
    cpu: float,
    memory: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    passed = (
        effective_min == NEW_CANDIDATE_EFFECTIVE_MIN
        and effective_max == NEW_CANDIDATE_EFFECTIVE_MAX
        and immutable_maxscale == NEW_CANDIDATE_IMMUTABLE_MAXSCALE
        and active_instances <= NEW_CANDIDATE_ACTIVE_INSTANCE_LIMIT
        and failed_startup_count <= FAILED_STARTUP_ALLOWED
        and request_drop_or_throttle_count <= REQUEST_DROP_OR_THROTTLE_ALLOWED
        and float(pending_not_ready_seconds) <= PENDING_NOT_READY_LIMIT_SECONDS
        and concurrency == CONCURRENCY_EXPECTED
        and float(cpu) == CPU_EXPECTED
        and memory == MEMORY_EXPECTED
        and timeout_seconds == TIMEOUT_EXPECTED_SECONDS
    )
    return {"status": "PASS" if passed else "STOP"}


def evaluate_cost_proxy_contract(
    *,
    candidate_min_instances: int,
    candidate_max_instances: int,
    candidate_revision_creation_count: int,
    total_observation_seconds: int,
    unexpected_billable_resource_delta_count: int,
    image_push_count: int,
    cloud_sql_binding_count: int,
    additional_service_or_scheduler_count: int,
    active_candidate_instance_count: int,
    authoritative_billing_available: bool = False,
) -> dict[str, Any]:
    passed = (
        candidate_min_instances == NEW_CANDIDATE_EFFECTIVE_MIN
        and candidate_max_instances == NEW_CANDIDATE_EFFECTIVE_MAX
        and candidate_revision_creation_count <= CANDIDATE_REVISION_CREATION_LIMIT
        and total_observation_seconds <= TOTAL_AUTHORIZED_OBSERVATION_SECONDS
        and unexpected_billable_resource_delta_count
        <= UNEXPECTED_BILLABLE_RESOURCE_DELTA_ALLOWED
        and image_push_count <= IMAGE_PUSH_ALLOWED
        and cloud_sql_binding_count <= CLOUD_SQL_BINDING_ALLOWED
        and additional_service_or_scheduler_count
        <= ADDITIONAL_SERVICE_OR_SCHEDULER_ALLOWED
        and active_candidate_instance_count <= NEW_CANDIDATE_ACTIVE_INSTANCE_LIMIT
    )
    return {
        "status": "PASS_WITH_LIMITS" if passed else "STOP",
        "real_time_billing_amount_status": (
            "VERIFIED" if authoritative_billing_available else "NOT_VERIFIED"
        ),
    }


_CLOUD_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,62}$")


def build_stable_rollback_command(
    *, service: str, project: str, region: str, stable_revision: str
) -> list[str]:
    """Construct the one deterministic mutation command; execution is external."""

    values = (service, project, region, stable_revision)
    if any(not _CLOUD_IDENTIFIER_PATTERN.fullmatch(value) for value in values):
        raise ObserverError("rollback identifiers must use bounded cloud identifiers")
    return [
        "gcloud",
        "run",
        "services",
        "update-traffic",
        service,
        "--project",
        project,
        "--region",
        region,
        "--to-revisions",
        f"{stable_revision}=100",
        "--quiet",
    ]


def evaluate_stable_rollback_verification(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate read-only rollback evidence separately from mutation execution."""

    failures = []
    expected = {
        "historical_stable_traffic_percent": 100,
        "other_positive_traffic_count": 0,
        "unauthenticated_health_http": UNAUTHENTICATED_EXPECTED_STATUS,
        "authenticated_health_http": AUTHENTICATED_EXPECTED_STATUS,
        "public_member_count": 0,
        "authoritative_set_iam_policy_count": 0,
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            failures.append(key)
    if snapshot.get("historical_stable_ready") is not True:
        failures.append("historical_stable_ready")
    if normalize_health_value(snapshot.get("normalized_health")) != HEALTH_NORMALIZED_VALUE:
        failures.append("normalized_health")
    if snapshot.get("iam_hash_match") is not True:
        failures.append("iam_hash_match")
    return {
        "status": "PASS" if not failures else "STOP",
        "verification_failures": failures,
        "mutation_result_is_separate": True,
    }


def validate_registry_final_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return failures; an intermediate registry response can never pass."""

    required_true = (
        "completion_marker",
        "manifest_digest_match",
        "config_digest_match",
        "required_labels_match",
        "source_commit_match",
        "runtime_user_match",
        "private_access",
    )
    failures = [name for name in required_true if artifact.get(name) is not True]
    if int(artifact.get("layer_count", 0)) <= 0:
        failures.append("layer_count")
    if artifact.get("latest_used") is not False:
        failures.append("latest_used")
    for name in (
        "registry_mutation_audit_count",
        "image_push_republication_count",
        "deletion_count",
    ):
        if int(artifact.get(name, -1)) != 0:
            failures.append(name)
    return failures


EXACT_SET_IAM_POLICY_METHOD = "google.cloud.run.v1.Services.SetIamPolicy"
AUTHORITATIVE_IAM_REPEAT_COUNT = 3
AUTHORITATIVE_IAM_MINIMUM_REPEAT_GAP_SECONDS = 30.0


def _contract_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_authoritative_set_iam_policy_query(
    *,
    project_id: str,
    service: str,
    region: str,
    window_start_utc: str,
    window_end_utc: str,
) -> dict[str, Any]:
    """Build the exact closed-window query; callers must not persist its raw filter."""

    start = datetime.fromisoformat(window_start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(window_end_utc.replace("Z", "+00:00"))
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ObserverError("authoritative IAM window must be closed and increasing")
    resource_name = f"namespaces/{project_id}/services/{service}"
    query_filter = (
        f'timestamp>="{window_start_utc}" AND timestamp<="{window_end_utc}" '
        f'AND resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{service}" '
        f'AND resource.labels.location="{region}" '
        f'AND protoPayload.resourceName="{resource_name}" '
        f'AND protoPayload.methodName="{EXACT_SET_IAM_POLICY_METHOD}"'
    )
    safe_contract = {
        "project": "MASKED",
        "resource_type": "cloud_run_revision",
        "service": service,
        "region": region,
        "resource_identity_hash": _contract_sha256(resource_name),
        "exact_method": EXACT_SET_IAM_POLICY_METHOD,
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
        "deduplication": "INSERT_ID_OR_CANONICAL_EVENT_IDENTIFIER_SHA256",
    }
    contract_json = json.dumps(
        safe_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "query_filter": query_filter,
        "safe_contract": safe_contract,
        "filter_contract_sha256": _contract_sha256(contract_json),
        "exact_filter_sha256": _contract_sha256(query_filter),
        "resource_name": resource_name,
    }


def _principal_category(entry: dict[str, Any]) -> str:
    payload = entry.get("protoPayload") or {}
    authentication = payload.get("authenticationInfo") or {}
    principal = str(authentication.get("principalEmail") or "")
    if not principal:
        return "NONE_OR_SYSTEM"
    if principal.endswith("gserviceaccount.com"):
        return "SERVICE_ACCOUNT"
    if "@" in principal:
        return "USER_OR_WORKFORCE_IDENTITY"
    return "OTHER_REDACTED_IDENTITY"


def sanitize_authoritative_set_iam_policy_entries(
    entries: list[dict[str, Any]],
    *,
    project_id: str,
    service: str,
    region: str,
    before_iam_sha256: str,
    after_iam_sha256: str,
) -> dict[str, Any]:
    """Exclude broad events, deduplicate exact events and retain hash-only identity."""

    resource_name = f"namespaces/{project_id}/services/{service}"
    unique: dict[str, dict[str, Any]] = {}
    excluded_count = 0
    for entry in entries:
        payload = entry.get("protoPayload") or {}
        resource = entry.get("resource") or {}
        labels = resource.get("labels") or {}
        exact = (
            payload.get("methodName") == EXACT_SET_IAM_POLICY_METHOD
            and payload.get("resourceName") == resource_name
            and resource.get("type") == "cloud_run_revision"
            and labels.get("service_name") == service
            and labels.get("location") == region
        )
        if not exact:
            excluded_count += 1
            continue
        canonical_id = str(entry.get("insertId") or "") or "|".join(
            (
                str(entry.get("timestamp") or ""),
                str(payload.get("methodName") or ""),
                str(payload.get("resourceName") or ""),
            )
        )
        event_id_hash = _contract_sha256(canonical_id)
        unique.setdefault(event_id_hash, entry)

    sanitized = []
    for event_id_hash in sorted(unique):
        entry = unique[event_id_hash]
        payload = entry.get("protoPayload") or {}
        sanitized.append(
            {
                "timestamp_utc": str(entry.get("timestamp") or ""),
                "method_name": EXACT_SET_IAM_POLICY_METHOD,
                "resource_identity_hash": _contract_sha256(
                    str(payload.get("resourceName") or "")
                ),
                "event_identifier_hash": event_id_hash,
                "principal_category": _principal_category(entry),
                "policy_delta": before_iam_sha256 != after_iam_sha256,
                "before_iam_normalized_sha256": before_iam_sha256,
                "after_iam_normalized_sha256": after_iam_sha256,
                "raw_principal_persisted": False,
                "raw_policy_persisted": False,
            }
        )
    return {
        "authoritative_unique_event_count": len(sanitized),
        "event_identifier_hash_set": [
            event["event_identifier_hash"] for event in sanitized
        ],
        "events": sanitized,
        "excluded_broad_or_wrong_method_count": excluded_count,
    }


def evaluate_authoritative_iam_repeats(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require three complete, identical, sufficiently separated closed-window reads."""

    if len(results) != AUTHORITATIVE_IAM_REPEAT_COUNT:
        return {"status": "NOT_VERIFIED", "reason": "REPEAT_COUNT_MISMATCH"}
    complete = all(
        result.get("completion_marker") is True
        and result.get("partial_result") is False
        and result.get("status") == "PASS"
        for result in results
    )
    contract_hashes = {str(result.get("filter_contract_sha256")) for result in results}
    filter_hashes = {str(result.get("exact_filter_sha256")) for result in results}
    counts = [int(result.get("authoritative_unique_event_count", -1)) for result in results]
    event_sets = [tuple(result.get("event_identifier_hash_set") or ()) for result in results]
    gaps = []
    try:
        for previous, current in zip(results, results[1:]):
            previous_end = datetime.fromisoformat(
                str(previous["query_completed_at_utc"]).replace("Z", "+00:00")
            )
            current_start = datetime.fromisoformat(
                str(current["query_started_at_utc"]).replace("Z", "+00:00")
            )
            gaps.append((current_start - previous_end).total_seconds())
    except (KeyError, TypeError, ValueError):
        return {"status": "NOT_VERIFIED", "reason": "INVALID_QUERY_TIMESTAMPS"}
    non_monotonic = len(set(counts)) != 1 or len(set(event_sets)) != 1
    if (
        not complete
        or len(contract_hashes) != 1
        or len(filter_hashes) != 1
        or any(gap < AUTHORITATIVE_IAM_MINIMUM_REPEAT_GAP_SECONDS for gap in gaps)
        or non_monotonic
    ):
        return {
            "status": "NOT_VERIFIED",
            "reason": "INCOMPLETE_OR_NON_MONOTONIC_REPEATS",
            "repeat_counts": counts,
            "minimum_repeat_gap_seconds": min(gaps) if gaps else None,
            "non_monotonic_result": non_monotonic,
        }
    count = counts[0]
    return {
        "status": "PASS" if count == 0 else "STOP",
        "authoritative_set_iam_policy_count": count,
        "repeat_counts": counts,
        "minimum_repeat_gap_seconds": min(gaps),
        "non_monotonic_result": False,
    }


_WINDOWS_ESCAPED_DRIVE_ROOT = re.compile(r"^[A-Za-z]:\\\\+$")
_WINDOWS_ABSOLUTE_WITH_SEGMENT = re.compile(
    r"^[A-Za-z]:\\(?!\\)(?:[^\\/\s]+)(?:\\[^\\/\s]+)*$"
)
_UNC_ABSOLUTE_WITH_SHARE = re.compile(r"^\\\\[^\\/\s]+\\[^\\/\s]+")
_UNIX_SENSITIVE_WITH_SEGMENT = re.compile(
    r"^/(?:workspace|root|home|tmp|var)/(?:[^/\s]+)(?:/[^/\s]+)*$"
)


def classify_internal_path_value(value: Any) -> dict[str, str]:
    """Classify one parsed value; field names and serialized bodies are out of scope."""

    if not isinstance(value, str):
        return {"rule_id": "NON_STRING_VALUE", "classification": "NO_MATCH"}
    if _WINDOWS_ESCAPED_DRIVE_ROOT.fullmatch(value):
        return {
            "rule_id": "ESCAPED_DRIVE_ROOT_SCHEMA_LITERAL",
            "classification": "SAFE_ROUTE_OR_SCHEMA_VALUE",
        }
    if _WINDOWS_ABSOLUTE_WITH_SEGMENT.match(value):
        return {
            "rule_id": "WINDOWS_ABSOLUTE_DRIVE_PATH",
            "classification": "TRUE_INTERNAL_PATH",
        }
    if _UNC_ABSOLUTE_WITH_SHARE.match(value):
        return {
            "rule_id": "UNC_ABSOLUTE_PATH",
            "classification": "TRUE_INTERNAL_PATH",
        }
    if _UNIX_SENSITIVE_WITH_SEGMENT.match(value):
        return {
            "rule_id": "UNIX_SENSITIVE_ABSOLUTE_PATH",
            "classification": "TRUE_INTERNAL_PATH",
        }
    if (
        value == "/health"
        or value.startswith("/assets/")
        or value.startswith("/app/")
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value)
        or value.startswith("ev-")
        or value.strip().lower() == "ok"
    ):
        return {
            "rule_id": "SAFE_ROUTE_SCHEMA_OR_IDENTIFIER_VALUE",
            "classification": "SAFE_ROUTE_OR_SCHEMA_VALUE",
        }
    return {"rule_id": "NO_INTERNAL_PATH_RULE", "classification": "NO_MATCH"}


def privacy_safe_internal_path_classification(
    payload: Any,
    *,
    response_surface_category: str,
) -> dict[str, Any]:
    """Traverse parsed values only and return hash-only match metadata."""

    classified = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                visit(child, f"{pointer}/{escaped}")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}")
            return
        decision = classify_internal_path_value(value)
        if decision["classification"] == "NO_MATCH":
            return
        text = str(value)
        classified.append(
            {
                "detector_rule_id": decision["rule_id"],
                "response_surface_category": response_surface_category,
                "selector_sha256": _contract_sha256(
                    f"{response_surface_category}|{pointer or '/'}"
                ),
                "matched_value_sha256": _contract_sha256(text),
                "value_length": len(text),
                "match_type": "PARSED_VALUE_CLASSIFICATION",
                "classification": decision["classification"],
                "raw_fragment_persisted": False,
            }
        )

    visit(payload, "")
    true_count = sum(
        item["classification"] == "TRUE_INTERNAL_PATH" for item in classified
    )
    return {
        "true_internal_path_count": true_count,
        "safe_route_or_schema_value_count": sum(
            item["classification"] == "SAFE_ROUTE_OR_SCHEMA_VALUE"
            for item in classified
        ),
        "matches": classified,
        "raw_fragment_persisted": False,
    }


class ObserverError(RuntimeError):
    """Base error whose message is safe to show to an operator."""


class DuplicateObservationError(ObserverError):
    """Raised when an observation id already owns an artifact directory."""


class SamplerFailure(ObserverError):
    """A sanitized sampler failure that is safe to persist in artifacts."""

    def __init__(
        self,
        category: str,
        *,
        phase: str,
        dependency_class: str,
        exit_category: str,
        retryable: bool = False,
        source_function: str = "_production_sample",
        source_line: int = 0,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if category not in SAMPLER_FAILURE_CATEGORIES:
            category = "ARGUMENT_OR_SERIALIZATION_DEFECT"
        self.category = category
        self.metadata = {
            "first_failure_phase": phase,
            "failure_category": category,
            "source_file": "tools/qlib_traffic_observer.py",
            "source_function": source_function,
            "source_line": int(source_line),
            "exception_class": "SamplerFailure",
            "dependency_class": dependency_class,
            "exit_category": exit_category,
            "retryable": bool(retryable),
            "raw_exception_message_persisted": False,
            "raw_sampler_output_persisted": False,
        }
        if diagnostics:
            self.metadata.update(_sanitize_sampler_diagnostics(diagnostics))
        super().__init__(f"production sampler failed: {category}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_observation_id(observation_id: str) -> str:
    if not OBSERVATION_ID_PATTERN.fullmatch(observation_id):
        raise ObserverError("observation_id must use 1-64 safe identifier characters")
    return observation_id


def _artifact_dir(artifact_root: Path | str, observation_id: str) -> Path:
    safe_id = _validate_observation_id(observation_id)
    return Path(artifact_root).resolve() / safe_id


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)
        except OSError as exc:
            if (
                os.name != "nt"
                or getattr(exc, "winerror", None) not in {5, 32, 33}
                or attempt == 39
            ):
                raise
            time.sleep(0.025)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _pid_exists(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _acquire_terminal_claim(directory: Path, outcome: str) -> bool:
    claim_path = directory / "terminal-claim.json"
    for _attempt in range(2):
        try:
            descriptor = os.open(
                claim_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            if (directory / "final.json").exists() or (
                directory / "incomplete.json"
            ).exists():
                return False
            claim = _read_json(claim_path) or {}
            owner_pid = claim.get("pid")
            if isinstance(owner_pid, int) and not _pid_exists(owner_pid):
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            return False
        payload = {
            "schema_version": SCHEMA_VERSION,
            "outcome": outcome,
            "pid": os.getpid(),
            "claimed_at_utc": _utc_now(),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True
    return False


def _launch_via_wmi(command: list[str]) -> int:
    command_line = subprocess.list2cmdline(command)
    encoded_command_line = base64.b64encode(command_line.encode("utf-8")).decode("ascii")
    powershell_source = (
        "$c=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"
        + encoded_command_line
        + "'));"
        + "$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        + "-Arguments @{CommandLine=$c};"
        + "if($null-eq$r-or[int]$r.ReturnValue-ne0){exit 41};"
        + "[Console]::Out.Write([string]$r.ProcessId)"
    )
    encoded_powershell = base64.b64encode(
        powershell_source.encode("utf-16le")
    ).decode("ascii")
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-EncodedCommand",
                encoded_powershell,
            ],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
        )
    except OSError as exc:
        raise ObserverError("WMI observer launch is unavailable") from exc
    process_ids = re.findall(r"\b\d+\b", completed.stdout)
    if completed.returncode != 0 or not process_ids:
        raise ObserverError("WMI observer launch is unavailable")
    process_id = int(process_ids[-1])
    if process_id <= 0:
        raise ObserverError("WMI observer launch is unavailable")
    return process_id


def _validate_configuration(
    *,
    duration_seconds: float,
    sample_interval_seconds: float,
    max_gap_seconds: float,
    stale_after_seconds: float,
    mode: str,
) -> None:
    if mode not in {"local-test", "production"}:
        raise ObserverError("mode must be local-test or production")
    if duration_seconds <= 0:
        raise ObserverError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ObserverError("sample_interval_seconds must be positive")
    if max_gap_seconds < sample_interval_seconds:
        raise ObserverError("max_gap_seconds must be at least the sample interval")
    if stale_after_seconds <= max_gap_seconds:
        raise ObserverError("stale_after_seconds must exceed max_gap_seconds")


def _load_production_sampler(environment: dict[str, str] | None = None) -> list[str]:
    source = (environment or os.environ).get(PRODUCTION_SAMPLER_ENV, "")
    try:
        argv = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ObserverError("production sampler configuration is missing or invalid") from exc
    return _validate_sampler_argv(argv)


def _validate_sampler_argv(argv: Any) -> list[str]:
    if (
        not isinstance(argv, list)
        or not argv
        or len(argv) > 32
        or any(not isinstance(item, str) or not item or len(item) > 1024 for item in argv)
    ):
        raise ObserverError("production sampler configuration is missing or invalid")
    return list(argv)


def _decode_sampler_argv(encoded: str) -> list[str]:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        argv = json.loads(raw)
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObserverError("production sampler configuration is missing or invalid") from exc
    return _validate_sampler_argv(argv)


def _sanitize_sample_value(value: Any, *, key: str = "sample") -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ObserverError("sampler returned a non-finite number")
        return value
    if isinstance(value, str):
        if len(value) > 128 or RAW_LOCATION_PATTERN.search(value) or "\\" in value or "/" in value:
            raise ObserverError("sampler returned a disallowed string value")
        return value
    if isinstance(value, list):
        if len(value) > 64:
            raise ObserverError("sampler returned an oversized list")
        return [_sanitize_sample_value(item, key=key) for item in value]
    if isinstance(value, dict):
        if len(value) > 64:
            raise ObserverError("sampler returned an oversized object")
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not SAFE_SAMPLE_KEY_PATTERN.fullmatch(raw_key):
                raise ObserverError("sampler returned an invalid field name")
            if any(part in raw_key.lower() for part in SENSITIVE_KEY_PARTS):
                raise ObserverError("sampler returned a disallowed field")
            sanitized[raw_key] = _sanitize_sample_value(raw_value, key=raw_key)
        return sanitized
    raise ObserverError(f"sampler returned an unsupported value for {key}")


def _local_sample(observation_id: str, sequence: int) -> dict[str, Any]:
    seed = f"{observation_id}:{sequence}".encode("utf-8")
    return {
        "sample_status": "OK",
        "deterministic_value": hashlib.sha256(seed).hexdigest()[:16],
    }


def _classify_sampler_failure_payload(payload: dict[str, Any]) -> str:
    declared = str(payload.get("failure_category") or "")
    if declared in SAMPLER_FAILURE_CATEGORIES:
        return declared
    ordered_counts = (
        ("http_403_count", "HTTP_403"),
        ("http_404_count", "HTTP_404"),
        ("unexpected_5xx_count", "HTTP_5XX"),
        ("timeout_count", "TIMEOUT"),
        ("json_parse_failure_count", "JSON_PARSE_FAILURE"),
        ("auth_failure_count", "AUTH_FAILURE"),
        ("dependency_failure_count", "DEPENDENCY_OR_SUBPROCESS_DEFECT"),
        ("target_routing_failure_count", "TARGET_ROUTING_CONTRACT_DEFECT"),
    )
    for key, category in ordered_counts:
        try:
            if int(payload.get(key, 0)) > 0:
                return category
        except (TypeError, ValueError):
            return "ARGUMENT_OR_SERIALIZATION_DEFECT"
    for key in (
        "ui_failure_count",
        "flow_failure_count",
        "evidence_missing_count",
        "trace_missing_count",
    ):
        try:
            if int(payload.get(key, 0)) > 0:
                return "FUNCTIONAL_HTTP_FAILURE"
        except (TypeError, ValueError):
            return "ARGUMENT_OR_SERIALIZATION_DEFECT"
    return "DEPENDENCY_OR_SUBPROCESS_DEFECT"


def validate_production_sample_contract(
    sample: dict[str, Any], *, required_target_contract: str | None
) -> None:
    """Fail closed on the pretraffic/stage sampler readiness contract."""

    if required_target_contract is None:
        return
    if required_target_contract not in PRODUCTION_TARGET_CONTRACTS:
        raise SamplerFailure(
            "ARGUMENT_OR_SERIALIZATION_DEFECT",
            phase="TARGET_CONTRACT_VALIDATION",
            dependency_class="SAMPLER_CONFIGURATION",
            exit_category="INVALID_REQUIRED_TARGET_CONTRACT",
            source_function="validate_production_sample_contract",
            source_line=sys._getframe().f_lineno,
        )
    if sample.get("target_contract") != required_target_contract:
        raise SamplerFailure(
            "TARGET_ROUTING_CONTRACT_DEFECT",
            phase="TARGET_CONTRACT_VALIDATION",
            dependency_class="TARGET_CONSTRUCTION",
            exit_category="TARGET_CONTRACT_MISMATCH",
            source_function="validate_production_sample_contract",
            source_line=sys._getframe().f_lineno,
        )
    for key in ("target_selector_hash", "command_contract_hash"):
        if not SHA256_HEX_PATTERN.fullmatch(str(sample.get(key) or "")):
            raise SamplerFailure(
                "ARGUMENT_OR_SERIALIZATION_DEFECT",
                phase="SAMPLER_READINESS_GATE",
                dependency_class="SANITIZED_COMMAND_CONTRACT",
                exit_category="MISSING_OR_INVALID_CONTRACT_HASH",
                source_function="validate_production_sample_contract",
                source_line=sys._getframe().f_lineno,
            )
    for key in ("argument_name_set", "sanitized_environment_key_set"):
        names = sample.get(key)
        if (
            not isinstance(names, list)
            or not names
            or any(
                not isinstance(name, str)
                or not SAFE_CONTRACT_NAME_PATTERN.fullmatch(name)
                or any(part in name.lower() for part in SENSITIVE_KEY_PARTS)
                for name in names
            )
        ):
            raise SamplerFailure(
                "ARGUMENT_OR_SERIALIZATION_DEFECT",
                phase="SAMPLER_READINESS_GATE",
                dependency_class="SANITIZED_COMMAND_CONTRACT",
                exit_category="INVALID_SANITIZED_NAME_SET",
                source_function="validate_production_sample_contract",
                source_line=sys._getframe().f_lineno,
            )
    status_categories = {
        "import_status": "DETACHED_ENVIRONMENT_OR_IMPORT_DEFECT",
        "dependency_status": "DEPENDENCY_OR_SUBPROCESS_DEFECT",
        "auth_handoff_status": "AUTH_FAILURE",
        "target_construction_status": "TARGET_ROUTING_CONTRACT_DEFECT",
        "health_sample_status": "FUNCTIONAL_HTTP_FAILURE",
        "readiness_status": "FUNCTIONAL_HTTP_FAILURE",
    }
    for key, category in status_categories.items():
        if sample.get(key) != "PASS":
            raise SamplerFailure(
                category,
                phase="SAMPLER_READINESS_GATE",
                dependency_class=key.removesuffix("_status").upper(),
                exit_category="READINESS_CONTRACT_STOP",
                source_function="validate_production_sample_contract",
                source_line=sys._getframe().f_lineno,
            )
    try:
        valid_sample_count = int(sample.get("valid_sample_count", 0))
        read_only_command_count = int(sample.get("read_only_command_count", 0))
        mutation_command_count = int(sample.get("mutation_command_count", -1))
        production_write_count = int(sample.get("production_write_count", -1))
    except (TypeError, ValueError) as exc:
        raise SamplerFailure(
            "ARGUMENT_OR_SERIALIZATION_DEFECT",
            phase="SAMPLER_READINESS_GATE",
            dependency_class="VALID_SAMPLE_CONTRACT",
            exit_category="INVALID_COUNTER_CONTRACT",
            source_function="validate_production_sample_contract",
            source_line=sys._getframe().f_lineno,
        ) from exc
    if (
        sample.get("sample_status") != "PASS"
        or valid_sample_count != 1
        or read_only_command_count < 1
        or mutation_command_count != 0
        or production_write_count != 0
        or sample.get("evidence_trace_safe_summary") != "PASS"
    ):
        raise SamplerFailure(
            "ARGUMENT_OR_SERIALIZATION_DEFECT",
            phase="SAMPLER_READINESS_GATE",
            dependency_class="VALID_SAMPLE_CONTRACT",
            exit_category="INVALID_VALID_SAMPLE_CONTRACT",
            source_function="validate_production_sample_contract",
            source_line=sys._getframe().f_lineno,
        )
    if sample.get("raw_response_persisted") is not False or sample.get(
        "identity_material_persisted"
    ) is not False:
        raise SamplerFailure(
            "ARGUMENT_OR_SERIALIZATION_DEFECT",
            phase="SAMPLER_READINESS_GATE",
            dependency_class="PRIVACY_BOUNDARY",
            exit_category="PERSISTENCE_CONTRACT_STOP",
            source_function="validate_production_sample_contract",
            source_line=sys._getframe().f_lineno,
        )


def _sanitize_sampler_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    allowed_current = set(SAMPLER_PHASES) | {"NOT_STARTED", "BETWEEN_PHASES"}
    allowed_last = set(SAMPLER_PHASES) | {"NONE"}
    current = str(value.get("current_phase") or "NOT_STARTED")
    last = str(value.get("last_completed_phase") or "NONE")
    timings: list[dict[str, Any]] = []
    raw_timings = value.get("phase_timings")
    if isinstance(raw_timings, list):
        for item in raw_timings[: len(SAMPLER_PHASES)]:
            if not isinstance(item, dict):
                continue
            phase = str(item.get("phase") or "")
            status = str(item.get("status") or "")
            try:
                elapsed_ms = float(item.get("elapsed_ms", 0.0))
            except (TypeError, ValueError):
                continue
            if (
                phase in SAMPLER_PHASES
                and status in {"PASS", "FAIL", "TIMEOUT"}
                and math.isfinite(elapsed_ms)
                and 0.0 <= elapsed_ms <= SAMPLER_TOTAL_DEADLINE_SECONDS * 1000.0
            ):
                timings.append(
                    {
                        "phase": phase,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "status": status,
                    }
                )

    def bounded_count(key: str, maximum: int = 1000) -> int:
        try:
            result = int(value.get(key, 0))
        except (TypeError, ValueError):
            return 0
        return result if 0 <= result <= maximum else 0

    timeout_reason = str(value.get("timeout_reason") or "NONE")
    if timeout_reason not in {"NONE", "TOTAL_DEADLINE_EXCEEDED"}:
        timeout_reason = "NONE"
    exit_state = str(value.get("child_exit_state") or "UNKNOWN")
    if exit_state not in {
        "UNKNOWN",
        "RUNNING",
        "EXITED",
        "EXITING_ZERO",
        "EXITING_NONZERO",
        "TERMINATED",
    }:
        exit_state = "UNKNOWN"
    cancellation_state = str(value.get("child_cancellation_state") or "NOT_REQUIRED")
    if cancellation_state not in {"NOT_REQUIRED", "COMPLETED", "FAILED"}:
        cancellation_state = "FAILED"
    return {
        "current_phase": current if current in allowed_current else "NOT_STARTED",
        "last_completed_phase": last if last in allowed_last else "NONE",
        "phase_timings": timings,
        "timeout_reason": timeout_reason,
        "read_only_command_count": bounded_count("read_only_command_count"),
        "mutation_command_count": bounded_count("mutation_command_count"),
        "valid_sample_count": min(1, bounded_count("valid_sample_count", 1)),
        "child_exit_state": exit_state,
        "child_cancellation_state": cancellation_state,
        "orphan_child_count": min(1, bounded_count("orphan_child_count", 1)),
    }


def _read_sampler_progress(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeError):
        return _sanitize_sampler_diagnostics({})
    return _sanitize_sampler_diagnostics(value if isinstance(value, dict) else {})


def _terminate_sampler_process(process: subprocess.Popen[str]) -> dict[str, Any]:
    if process.poll() is not None:
        return {
            "child_exit_state": "EXITED",
            "child_cancellation_state": "NOT_REQUIRED",
            "orphan_child_count": 0,
        }

    cancellation_completed = False
    if os.name == "nt":
        try:
            killed = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
            )
            cancellation_completed = killed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            cancellation_completed = False
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            cancellation_completed = True
        except (OSError, ProcessLookupError):
            cancellation_completed = process.poll() is not None

    if not cancellation_completed and process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            pass
    exited = process.poll() is not None
    return {
        "child_exit_state": "TERMINATED" if exited else "RUNNING",
        "child_cancellation_state": "COMPLETED" if exited else "FAILED",
        "orphan_child_count": 0 if exited and cancellation_completed else 1,
    }


def _bounded_sampler_elapsed_ms(started: float) -> float:
    return round(
        min(
            SAMPLER_TOTAL_DEADLINE_SECONDS * 1000.0,
            max(0.0, (time.monotonic() - started) * 1000.0),
        ),
        3,
    )


def _production_sample_once(argv: list[str], timeout_seconds: float) -> dict[str, Any]:
    timeout_seconds = min(float(timeout_seconds), SAMPLER_TOTAL_DEADLINE_SECONDS)
    if timeout_seconds <= 0:
        raise SamplerFailure(
            "TIMEOUT",
            phase="SAMPLER_SUBPROCESS",
            dependency_class="SUBPROCESS",
            exit_category="TOTAL_DEADLINE_EXCEEDED",
            source_line=sys._getframe().f_lineno,
            diagnostics={"timeout_reason": "TOTAL_DEADLINE_EXCEEDED"},
        )
    with tempfile.TemporaryDirectory(prefix="qlib-sampler-") as temporary:
        progress_path = Path(temporary) / "progress.json"
        environment = dict(os.environ)
        environment[SAMPLER_PROGRESS_FILE_ENV] = str(progress_path)
        environment[SAMPLER_DEADLINE_ENV] = str(timeout_seconds)
        popen_options: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "shell": False,
            "env": environment,
        }
        if os.name == "nt":
            popen_options["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            )
        else:
            popen_options["start_new_session"] = True
        ipc_started = time.monotonic()
        try:
            process = subprocess.Popen(argv, **popen_options)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SamplerFailure(
                "DEPENDENCY_OR_SUBPROCESS_DEFECT",
                phase="SAMPLER_SUBPROCESS",
                dependency_class="SUBPROCESS",
                exit_category="SUBPROCESS_LAUNCH_OR_IO_FAILURE",
                source_line=sys._getframe().f_lineno,
            ) from exc
        try:
            stdout, _stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            ipc_elapsed_ms = _bounded_sampler_elapsed_ms(ipc_started)
            cleanup_started = time.monotonic()
            cancellation = _terminate_sampler_process(process)
            try:
                process.communicate(timeout=2.0)
            except (OSError, subprocess.SubprocessError):
                pass
            cleanup_elapsed_ms = _bounded_sampler_elapsed_ms(cleanup_started)
            collection_started = time.monotonic()
            diagnostics = _read_sampler_progress(progress_path)
            parent_timings = [
                {
                    "phase": "PARENT_CHILD_IPC",
                    "elapsed_ms": ipc_elapsed_ms,
                    "status": "TIMEOUT",
                },
                {
                    "phase": "CHILD_CLEANUP",
                    "elapsed_ms": cleanup_elapsed_ms,
                    "status": (
                        "PASS"
                        if cancellation["child_cancellation_state"] == "COMPLETED"
                        else "FAIL"
                    ),
                },
                {
                    "phase": "OBSERVER_COLLECTION",
                    "elapsed_ms": _bounded_sampler_elapsed_ms(collection_started),
                    "status": "PASS",
                },
            ]
            diagnostics["phase_timings"] = [
                *diagnostics.get("phase_timings", []),
                *parent_timings,
            ]
            diagnostics.update(cancellation)
            diagnostics["timeout_reason"] = "TOTAL_DEADLINE_EXCEEDED"
            diagnostics["valid_sample_count"] = 0
            raise SamplerFailure(
                "TIMEOUT",
                phase="SAMPLER_SUBPROCESS",
                dependency_class="SUBPROCESS",
                exit_category="SUBPROCESS_TIMEOUT",
                source_line=sys._getframe().f_lineno,
                diagnostics=diagnostics,
            ) from exc

        ipc_elapsed_ms = _bounded_sampler_elapsed_ms(ipc_started)
        collection_started = time.monotonic()
        diagnostics = _read_sampler_progress(progress_path)
        diagnostics.update(
            {
                "child_exit_state": "EXITED",
                "child_cancellation_state": "NOT_REQUIRED",
                "orphan_child_count": 0,
            }
        )
        try:
            raw_sample = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise SamplerFailure(
                "JSON_PARSE_FAILURE",
                phase="SAMPLER_OUTPUT_PARSE",
                dependency_class="SAMPLER_SERIALIZATION",
                exit_category=(
                    "SAMPLER_EXIT_NONZERO" if process.returncode else "SAMPLER_EXIT_ZERO"
                ),
                source_line=sys._getframe().f_lineno,
                diagnostics=diagnostics,
            ) from exc
        try:
            sanitized = _sanitize_sample_value(raw_sample)
        except ObserverError as exc:
            raise SamplerFailure(
                "ARGUMENT_OR_SERIALIZATION_DEFECT",
                phase="SAMPLER_OUTPUT_SANITIZATION",
                dependency_class="SAMPLER_SERIALIZATION",
                exit_category=(
                    "SAMPLER_EXIT_NONZERO" if process.returncode else "SAMPLER_EXIT_ZERO"
                ),
                source_line=sys._getframe().f_lineno,
                diagnostics=diagnostics,
            ) from exc
        if not isinstance(sanitized, dict):
            raise SamplerFailure(
                "ARGUMENT_OR_SERIALIZATION_DEFECT",
                phase="SAMPLER_OUTPUT_SANITIZATION",
                dependency_class="SAMPLER_SERIALIZATION",
                exit_category="NON_OBJECT_OUTPUT",
                source_line=sys._getframe().f_lineno,
                diagnostics=diagnostics,
            )
        sample_diagnostics = _sanitize_sampler_diagnostics(sanitized)
        sample_diagnostics["phase_timings"] = [
            *sample_diagnostics.get("phase_timings", []),
            {
                "phase": "PARENT_CHILD_IPC",
                "elapsed_ms": ipc_elapsed_ms,
                "status": "PASS",
            },
            {
                "phase": "CHILD_CLEANUP",
                "elapsed_ms": 0.0,
                "status": "PASS",
            },
            {
                "phase": "OBSERVER_COLLECTION",
                "elapsed_ms": _bounded_sampler_elapsed_ms(collection_started),
                "status": "PASS",
            },
        ]
        sample_diagnostics.update(
            {
                "current_phase": "BETWEEN_PHASES",
                "last_completed_phase": "OBSERVER_COLLECTION",
                "child_exit_state": "EXITED",
                "child_cancellation_state": "NOT_REQUIRED",
                "orphan_child_count": 0,
            }
        )
        if process.returncode != 0 or sanitized.get("sample_status") == "FAIL":
            category = _classify_sampler_failure_payload(sanitized)
            raise SamplerFailure(
                category,
                phase="SAMPLER_FUNCTIONAL_RESULT",
                dependency_class="SAMPLER_CONTRACT",
                exit_category=(
                    "SAMPLER_EXIT_NONZERO"
                    if process.returncode
                    else "SAMPLER_EXIT_ZERO_FAIL"
                ),
                retryable=category == "VERIFIED_EXTERNAL_TRANSIENT",
                source_line=sys._getframe().f_lineno,
                diagnostics=sample_diagnostics,
            )
        sanitized.update(sample_diagnostics)
        return sanitized


def _production_sample(
    argv: list[str],
    timeout_seconds: float,
    *,
    required_target_contract: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    total_deadline_seconds = min(
        float(timeout_seconds), SAMPLER_TOTAL_DEADLINE_SECONDS
    )
    retries = 0
    while True:
        elapsed = time.monotonic() - started
        remaining = total_deadline_seconds - elapsed
        if remaining <= 0:
            raise SamplerFailure(
                "TIMEOUT",
                phase="SAMPLER_SUBPROCESS",
                dependency_class="SUBPROCESS",
                exit_category="TOTAL_DEADLINE_EXCEEDED",
                source_line=sys._getframe().f_lineno,
                diagnostics={
                    "timeout_reason": "TOTAL_DEADLINE_EXCEEDED",
                    "valid_sample_count": 0,
                },
            )
        try:
            sample = _production_sample_once(argv, remaining)
            validate_production_sample_contract(
                sample, required_target_contract=required_target_contract
            )
            return sample
        except SamplerFailure as exc:
            elapsed = time.monotonic() - started
            may_retry = (
                exc.category == "VERIFIED_EXTERNAL_TRANSIENT"
                and retries < MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRIES
                and elapsed < MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRY_SECONDS
                and elapsed < total_deadline_seconds
            )
            if not may_retry:
                raise
            retries += 1


def _write_incomplete(
    directory: Path,
    *,
    reason: str,
    process_exit_code: int,
    failure_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_final = _read_json(directory / "final.json")
    if existing_final is not None:
        return existing_final
    existing = _read_json(directory / "incomplete.json")
    if existing is not None:
        return existing

    start = _read_json(directory / "start.json") or {}
    state = _read_json(directory / "state.json") or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": start.get("observation_id"),
        "mode": start.get("mode"),
        "status": "INCOMPLETE",
        "verification_status": "NOT_VERIFIED",
        "reason": reason,
        "requested_duration_seconds": start.get("requested_duration_seconds"),
        "started_at_utc": start.get("started_at_utc"),
        "ended_at_utc": _utc_now(),
        "monotonic_elapsed_seconds": state.get("monotonic_elapsed_seconds", 0.0),
        "sample_count": state.get("sample_count", 0),
        "sample_interval_seconds": start.get("sample_interval_seconds"),
        "maximum_gap_seconds": state.get("maximum_gap_seconds", 0.0),
        "maximum_allowed_gap_seconds": start.get("maximum_allowed_gap_seconds"),
        "process_exit_code": process_exit_code,
        "completion_marker": False,
    }
    if failure_metadata is not None:
        payload["first_failure"] = dict(failure_metadata)
    if not _acquire_terminal_claim(directory, "INCOMPLETE"):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            existing_final = _read_json(directory / "final.json")
            if existing_final is not None:
                return existing_final
            existing = _read_json(directory / "incomplete.json")
            if existing is not None:
                return existing
            time.sleep(0.02)
        payload["reason"] = "TERMINAL_WRITE_PENDING"
        return payload

    if (directory / "final.json").exists():
        return _read_json(directory / "final.json") or payload
    _append_event(
        directory / "events.ndjson",
        {
            "event": "INCOMPLETE",
            "at_utc": payload["ended_at_utc"],
            "reason": reason,
            "process_exit_code": process_exit_code,
            "failure_category": (
                failure_metadata.get("failure_category")
                if failure_metadata is not None
                else None
            ),
        },
    )
    state.update(
        {
            "status": "INCOMPLETE",
            "verification_status": "NOT_VERIFIED",
            "reason": reason,
            "ended_at_utc": payload["ended_at_utc"],
        }
    )
    if failure_metadata is not None:
        state["first_failure"] = dict(failure_metadata)
    _atomic_write_json(directory / "state.json", state)
    _atomic_write_json(directory / "incomplete.json", payload)
    return _read_json(directory / "incomplete.json") or payload


def start_observation(
    *,
    artifact_root: Path | str,
    observation_id: str,
    duration_seconds: float,
    sample_interval_seconds: float,
    max_gap_seconds: float,
    stale_after_seconds: float,
    mode: str,
    required_target_contract: str | None = None,
) -> dict[str, Any]:
    _validate_configuration(
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        max_gap_seconds=max_gap_seconds,
        stale_after_seconds=stale_after_seconds,
        mode=mode,
    )
    if required_target_contract is not None:
        if mode != "production" or required_target_contract not in PRODUCTION_TARGET_CONTRACTS:
            raise ObserverError("required target contract is invalid for this mode")
    production_sampler = _load_production_sampler() if mode == "production" else None

    directory = _artifact_dir(artifact_root, observation_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as exc:
        raise DuplicateObservationError("duplicate observation_id detected") from exc

    launched_at = _utc_now()
    start = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": observation_id,
        "mode": mode,
        "launcher_started_at_utc": launched_at,
        "requested_duration_seconds": duration_seconds,
        "sample_interval_seconds": sample_interval_seconds,
        "maximum_allowed_gap_seconds": max_gap_seconds,
        "stale_after_seconds": stale_after_seconds,
        "required_target_contract": required_target_contract,
    }
    _atomic_write_json(directory / "start.json", start)
    _atomic_write_json(
        directory / "state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "observation_id": observation_id,
            "mode": mode,
            "status": "STARTING",
            "sample_count": 0,
            "maximum_gap_seconds": 0.0,
            "monotonic_elapsed_seconds": 0.0,
        },
    )
    _append_event(
        directory / "events.ndjson",
        {"event": "LAUNCHED", "at_utc": launched_at, "mode": mode},
    )

    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "_run",
        "--artifact-dir",
        str(directory),
    ]
    if production_sampler is not None:
        encoded_sampler = base64.urlsafe_b64encode(
            json.dumps(production_sampler, ensure_ascii=True).encode("utf-8")
        ).decode("ascii")
        command.extend(["--sampler-config-b64", encoded_sampler])
    child_environment = os.environ.copy()
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    child_environment.pop(PRODUCTION_SAMPLER_ENV, None)
    child_environment.pop(IDENTITY_MATERIAL_ENV, None)
    popen_options: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "env": child_environment,
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        )
    else:
        popen_options["start_new_session"] = True

    try:
        child = subprocess.Popen(command, **popen_options)
    except PermissionError as exc:
        if os.name != "nt" or getattr(exc, "winerror", None) != 5:
            _write_incomplete(directory, reason="LAUNCH_FAILED", process_exit_code=70)
            raise ObserverError("detached observer launch failed") from exc
        try:
            child_pid = _launch_via_wmi(command)
        except ObserverError:
            _write_incomplete(directory, reason="LAUNCH_FAILED", process_exit_code=70)
            raise
        _atomic_write_json(
            directory / "process.json",
            {
                "schema_version": SCHEMA_VERSION,
                "observation_id": observation_id,
                "pid": child_pid,
                "launched_at_utc": launched_at,
                "launcher_backend": "WINDOWS_WMI",
            },
        )
        return {
            "observation_id": observation_id,
            "mode": mode,
            "status": "STARTED",
            "pid": child_pid,
            "launcher_backend": "WINDOWS_WMI",
            "artifact_path": str(directory),
        }
    except OSError as exc:
        _write_incomplete(directory, reason="LAUNCH_FAILED", process_exit_code=70)
        raise ObserverError("detached observer launch failed") from exc

    child_pid = child.pid
    # The detached worker is intentionally not waitable by this short-lived
    # launcher.  Mark only the local Popen wrapper as released so its destructor
    # does not report the still-running, independently owned worker as a leak.
    child.returncode = 0
    _atomic_write_json(
        directory / "process.json",
        {
            "schema_version": SCHEMA_VERSION,
            "observation_id": observation_id,
            "pid": child_pid,
            "launched_at_utc": launched_at,
        },
    )
    return {
        "observation_id": observation_id,
        "mode": mode,
        "status": "STARTED",
        "pid": child_pid,
        "launcher_backend": "DIRECT_PROCESS",
        "artifact_path": str(directory),
    }


def _run_worker(
    directory: Path,
    *,
    sampler_config_b64: str | None = None,
) -> int:
    start = _read_json(directory / "start.json")
    if start is None:
        return 70
    observation_id = str(start.get("observation_id", ""))
    mode = str(start.get("mode", ""))
    try:
        _validate_observation_id(observation_id)
        duration_seconds = float(start["requested_duration_seconds"])
        interval_seconds = float(start["sample_interval_seconds"])
        max_gap_seconds = float(start["maximum_allowed_gap_seconds"])
        stale_after_seconds = float(start["stale_after_seconds"])
        _validate_configuration(
            duration_seconds=duration_seconds,
            sample_interval_seconds=interval_seconds,
            max_gap_seconds=max_gap_seconds,
            stale_after_seconds=stale_after_seconds,
            mode=mode,
        )
        sampler_argv = (
            _decode_sampler_argv(sampler_config_b64)
            if mode == "production" and sampler_config_b64
            else _load_production_sampler()
            if mode == "production"
            else None
        )
        required_target_contract = start.get("required_target_contract")
        if required_target_contract is not None and (
            mode != "production"
            or required_target_contract not in PRODUCTION_TARGET_CONTRACTS
        ):
            raise ObserverError("required target contract is invalid for this mode")
    except (KeyError, TypeError, ValueError, ObserverError):
        _write_incomplete(directory, reason="INVALID_START_METADATA", process_exit_code=70)
        return 70

    started_at_utc = _utc_now()
    start["started_at_utc"] = started_at_utc
    _atomic_write_json(directory / "start.json", start)
    _append_event(
        directory / "events.ndjson",
        {"event": "WORKER_STARTED", "at_utc": started_at_utc, "mode": mode},
    )

    started_monotonic = time.monotonic()
    previous_sample_monotonic: float | None = None
    maximum_gap_seconds = 0.0
    sample_count = 0
    next_sample_monotonic = started_monotonic

    while True:
        if (directory / "incomplete.json").exists():
            return 42
        stop_request = _read_json(directory / "stop-request.json")
        if stop_request is not None:
            reason = str(stop_request.get("reason", "TASK_CLEANUP"))
            if reason not in STOP_REASONS:
                reason = "TASK_CLEANUP"
            _write_incomplete(directory, reason=reason, process_exit_code=42)
            return 42

        now_monotonic = time.monotonic()
        if now_monotonic < next_sample_monotonic:
            time.sleep(min(0.2, next_sample_monotonic - now_monotonic))
            continue

        elapsed = now_monotonic - started_monotonic
        if previous_sample_monotonic is not None:
            maximum_gap_seconds = max(
                maximum_gap_seconds, now_monotonic - previous_sample_monotonic
            )
        previous_sample_monotonic = now_monotonic

        try:
            if mode == "local-test":
                sample = _local_sample(observation_id, sample_count)
            else:
                assert sampler_argv is not None
                sample = _production_sample(
                    sampler_argv,
                    timeout_seconds=max(1.0, min(interval_seconds, 30.0)),
                    required_target_contract=(
                        str(required_target_contract)
                        if required_target_contract is not None
                        else None
                    ),
                )
        except SamplerFailure as exc:
            _write_incomplete(
                directory,
                reason=f"SAMPLER_{exc.category}",
                process_exit_code=42,
                failure_metadata=exc.metadata,
            )
            return 42
        except ObserverError:
            metadata = SamplerFailure(
                "ARGUMENT_OR_SERIALIZATION_DEFECT",
                phase="SAMPLER_CONTROLLER",
                dependency_class="OBSERVER_CONTROLLER",
                exit_category="OBSERVER_ERROR",
                source_function="_run_worker",
                source_line=sys._getframe().f_lineno,
            ).metadata
            _write_incomplete(
                directory,
                reason="SAMPLER_ARGUMENT_OR_SERIALIZATION_DEFECT",
                process_exit_code=42,
                failure_metadata=metadata,
            )
            return 42

        sampled_at_utc = _utc_now()
        sample_count += 1
        rounded_elapsed = round(time.monotonic() - started_monotonic, 6)
        rounded_gap = round(maximum_gap_seconds, 6)
        sample_event = {
            "event": "SAMPLE",
            "at_utc": sampled_at_utc,
            "sequence": sample_count,
            "monotonic_elapsed_seconds": rounded_elapsed,
            "mode": mode,
            "sample": sample,
        }
        _append_event(directory / "events.ndjson", sample_event)
        _atomic_write_json(
            directory / "heartbeat.json",
            {
                "schema_version": SCHEMA_VERSION,
                "observation_id": observation_id,
                "mode": mode,
                "heartbeat_at_utc": sampled_at_utc,
                "sequence": sample_count,
                "monotonic_elapsed_seconds": rounded_elapsed,
                "maximum_gap_seconds": rounded_gap,
            },
        )
        state = {
            "schema_version": SCHEMA_VERSION,
            "observation_id": observation_id,
            "mode": mode,
            "status": "RUNNING",
            "verification_status": "NOT_VERIFIED",
            "pid": os.getpid(),
            "started_at_utc": started_at_utc,
            "last_heartbeat_at_utc": sampled_at_utc,
            "sample_count": sample_count,
            "monotonic_elapsed_seconds": rounded_elapsed,
            "maximum_gap_seconds": rounded_gap,
        }
        _atomic_write_json(directory / "state.json", state)

        if maximum_gap_seconds > max_gap_seconds:
            _write_incomplete(directory, reason="SAMPLE_GAP_EXCEEDED", process_exit_code=42)
            return 42
        if rounded_elapsed >= duration_seconds:
            ended_at_utc = _utc_now()
            final = {
                "schema_version": SCHEMA_VERSION,
                "observation_id": observation_id,
                "mode": mode,
                "status": "PASS",
                "verification_status": "PASS",
                "completion_marker": True,
                "requested_duration_seconds": duration_seconds,
                "started_at_utc": started_at_utc,
                "ended_at_utc": ended_at_utc,
                "monotonic_elapsed_seconds": rounded_elapsed,
                "sample_count": sample_count,
                "expected_minimum_sample_count": math.floor(
                    duration_seconds / interval_seconds
                )
                + 1,
                "sample_interval_seconds": interval_seconds,
                "maximum_gap_seconds": rounded_gap,
                "maximum_allowed_gap_seconds": max_gap_seconds,
                "process_exit_code": 0,
            }
            if (directory / "incomplete.json").exists():
                return 42
            if not _acquire_terminal_claim(directory, "PASS"):
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if (directory / "final.json").exists():
                        return 0
                    if (directory / "incomplete.json").exists():
                        return 42
                    time.sleep(0.02)
                return 42
            _append_event(
                directory / "events.ndjson",
                {
                    "event": "COMPLETED",
                    "at_utc": ended_at_utc,
                    "monotonic_elapsed_seconds": rounded_elapsed,
                    "sample_count": sample_count,
                },
            )
            state.update(
                {
                    "status": "PASS",
                    "verification_status": "PASS",
                    "ended_at_utc": ended_at_utc,
                    "process_exit_code": 0,
                }
            )
            _atomic_write_json(directory / "state.json", state)
            # The completion marker is the final write.  A reconnecting poller
            # can therefore treat its presence as proof that all incremental
            # state has already been flushed.
            _atomic_write_json(directory / "final.json", final)
            return 0

        next_sample_monotonic = started_monotonic + sample_count * interval_seconds


def _validate_final(start: dict[str, Any], final: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if final.get("completion_marker") is not True:
        failures.append("MISSING_COMPLETION_MARKER")
    if final.get("status") != "PASS" or final.get("verification_status") != "PASS":
        failures.append("FINAL_STATUS_NOT_PASS")
    if final.get("observation_id") != start.get("observation_id"):
        failures.append("OBSERVATION_ID_MISMATCH")
    if final.get("mode") != start.get("mode"):
        failures.append("MODE_MISMATCH")
    try:
        requested = float(start["requested_duration_seconds"])
        elapsed = float(final["monotonic_elapsed_seconds"])
        maximum_gap = float(final["maximum_gap_seconds"])
        allowed_gap = float(start["maximum_allowed_gap_seconds"])
        sample_count = int(final["sample_count"])
        sample_interval = float(start["sample_interval_seconds"])
    except (KeyError, TypeError, ValueError):
        failures.append("FINAL_NUMERIC_FIELDS_INVALID")
    else:
        if elapsed < requested:
            failures.append("REQUESTED_DURATION_NOT_MET")
        if maximum_gap > allowed_gap:
            failures.append("SAMPLE_GAP_EXCEEDED")
        if sample_interval <= 0:
            failures.append("FINAL_NUMERIC_FIELDS_INVALID")
        else:
            expected_minimum = math.floor(requested / sample_interval) + 1
            if sample_count < expected_minimum:
                failures.append("SAMPLE_COUNT_BELOW_EXPECTED_MINIMUM")
    if final.get("process_exit_code") != 0:
        failures.append("PROCESS_EXIT_CODE_NOT_ZERO")
    return failures


def poll_observation(
    *, artifact_root: Path | str, observation_id: str
) -> dict[str, Any]:
    directory = _artifact_dir(artifact_root, observation_id)
    start = _read_json(directory / "start.json")
    if start is None:
        return {
            "observation_id": observation_id,
            "status": "NOT_VERIFIED",
            "verification_status": "NOT_VERIFIED",
            "reason": "START_METADATA_MISSING",
        }

    final = _read_json(directory / "final.json")
    incomplete = _read_json(directory / "incomplete.json")
    if final is not None and incomplete is not None:
        return {
            "observation_id": observation_id,
            "mode": start.get("mode"),
            "status": "NOT_VERIFIED",
            "verification_status": "NOT_VERIFIED",
            "reason": "TERMINAL_ARTIFACT_CONFLICT",
        }
    if incomplete is not None:
        return incomplete
    if final is not None:
        failures = _validate_final(start, final)
        if failures:
            return {
                "observation_id": observation_id,
                "mode": start.get("mode"),
                "status": "NOT_VERIFIED",
                "verification_status": "NOT_VERIFIED",
                "reason": "INVALID_FINAL_SUMMARY",
                "validation_failures": failures,
            }
        return final

    state = _read_json(directory / "state.json") or {}
    heartbeat = _read_json(directory / "heartbeat.json") or {}
    process = _read_json(directory / "process.json") or {}
    stop_request = _read_json(directory / "stop-request.json")
    if stop_request is not None:
        requested_reason = str(stop_request.get("reason", "TASK_CLEANUP"))
        if requested_reason not in STOP_REASONS:
            requested_reason = "TASK_CLEANUP"
        stop_pid = process.get("pid")
        if _pid_exists(stop_pid if isinstance(stop_pid, int) else None):
            return {
                "schema_version": SCHEMA_VERSION,
                "observation_id": observation_id,
                "mode": start.get("mode"),
                "status": "RUNNING",
                "phase": "STOPPING",
                "verification_status": "NOT_VERIFIED",
                "pid": stop_pid,
                "sample_count": state.get("sample_count", 0),
                "monotonic_elapsed_seconds": state.get(
                    "monotonic_elapsed_seconds", 0.0
                ),
                "completion_marker": False,
            }
        return _write_incomplete(
            directory, reason=requested_reason, process_exit_code=42
        )
    try:
        maximum_gap = float(state.get("maximum_gap_seconds", 0.0))
        allowed_gap = float(start["maximum_allowed_gap_seconds"])
    except (KeyError, TypeError, ValueError):
        maximum_gap, allowed_gap = 0.0, 0.0
    if maximum_gap > allowed_gap:
        return _write_incomplete(
            directory, reason="SAMPLE_GAP_EXCEEDED", process_exit_code=42
        )

    heartbeat_at = (
        heartbeat.get("heartbeat_at_utc")
        or start.get("started_at_utc")
        or start.get("launcher_started_at_utc")
    )
    heartbeat_age: float | None = None
    if isinstance(heartbeat_at, str):
        try:
            heartbeat_age = max(
                0.0,
                (datetime.now(timezone.utc) - _parse_utc(heartbeat_at)).total_seconds(),
            )
        except ValueError:
            heartbeat_age = None
    try:
        stale_after = float(start["stale_after_seconds"])
    except (KeyError, TypeError, ValueError):
        stale_after = 0.0
    if heartbeat_age is None or heartbeat_age > stale_after:
        return _write_incomplete(
            directory, reason="STALE_HEARTBEAT", process_exit_code=42
        )

    pid = process.get("pid")
    if pid is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_id": observation_id,
            "mode": start.get("mode"),
            "status": "RUNNING",
            "phase": "STARTING",
            "verification_status": "NOT_VERIFIED",
            "pid": None,
            "monotonic_elapsed_seconds": state.get("monotonic_elapsed_seconds", 0.0),
            "sample_count": state.get("sample_count", 0),
            "maximum_gap_seconds": maximum_gap,
            "maximum_allowed_gap_seconds": allowed_gap,
            "heartbeat_age_seconds": round(heartbeat_age, 6),
            "completion_marker": False,
        }
    if not _pid_exists(pid if isinstance(pid, int) else None):
        loss_path = directory / "process-loss.json"
        loss = _read_json(loss_path)
        heartbeat_sequence = heartbeat.get("sequence")
        if loss is None or loss.get("heartbeat_sequence") != heartbeat_sequence:
            loss = {
                "schema_version": SCHEMA_VERSION,
                "observation_id": observation_id,
                "first_detected_at_utc": _utc_now(),
                "heartbeat_sequence": heartbeat_sequence,
                "consecutive_detection_count": 1,
            }
            _atomic_write_json(loss_path, loss)
        else:
            loss["consecutive_detection_count"] = int(
                loss.get("consecutive_detection_count", 1)
            ) + 1
            loss["last_detected_at_utc"] = _utc_now()
            _atomic_write_json(loss_path, loss)
        try:
            loss_age = (
                datetime.now(timezone.utc)
                - _parse_utc(str(loss["first_detected_at_utc"]))
            ).total_seconds()
        except (KeyError, ValueError):
            loss_age = 0.0
        if loss_age >= 1.0 and int(loss.get("consecutive_detection_count", 0)) >= 2:
            return _write_incomplete(
                directory, reason="CHILD_PROCESS_LOSS", process_exit_code=42
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_id": observation_id,
            "mode": start.get("mode"),
            "status": "RUNNING",
            "phase": "CHILD_PROCESS_LOSS_SUSPECTED",
            "verification_status": "NOT_VERIFIED",
            "pid": pid,
            "monotonic_elapsed_seconds": state.get("monotonic_elapsed_seconds", 0.0),
            "sample_count": state.get("sample_count", 0),
            "completion_marker": False,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "observation_id": observation_id,
        "mode": start.get("mode"),
        "status": "RUNNING",
        "verification_status": "NOT_VERIFIED",
        "pid": pid,
        "monotonic_elapsed_seconds": state.get("monotonic_elapsed_seconds", 0.0),
        "sample_count": state.get("sample_count", 0),
        "maximum_gap_seconds": maximum_gap,
        "maximum_allowed_gap_seconds": allowed_gap,
        "heartbeat_age_seconds": round(heartbeat_age, 6),
        "completion_marker": False,
    }


def stop_observation(
    *,
    artifact_root: Path | str,
    observation_id: str,
    reason: str,
    wait_seconds: float = 10.0,
) -> dict[str, Any]:
    if reason not in STOP_REASONS:
        raise ObserverError("unsupported stop reason")
    directory = _artifact_dir(artifact_root, observation_id)
    current = poll_observation(artifact_root=artifact_root, observation_id=observation_id)
    if current.get("status") != "RUNNING":
        return current
    _atomic_write_json(
        directory / "stop-request.json",
        {
            "schema_version": SCHEMA_VERSION,
            "observation_id": observation_id,
            "reason": reason,
            "requested_at_utc": _utc_now(),
        },
    )
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = poll_observation(
            artifact_root=artifact_root, observation_id=observation_id
        )
        if current.get("status") != "RUNNING":
            return current
    return {
        "observation_id": observation_id,
        "status": "STOP_REQUESTED",
        "verification_status": "NOT_VERIFIED",
        "reason": reason,
    }


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detached QLIB traffic observer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--artifact-root", required=True)
    start_parser.add_argument("--observation-id", required=True)
    start_parser.add_argument("--duration-seconds", type=float, required=True)
    start_parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    start_parser.add_argument("--max-gap-seconds", type=float, default=8.0)
    start_parser.add_argument("--stale-after-seconds", type=float, default=20.0)
    start_parser.add_argument("--mode", choices=("local-test", "production"), required=True)
    start_parser.add_argument(
        "--required-target-contract", choices=PRODUCTION_TARGET_CONTRACTS
    )

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument("--artifact-root", required=True)
    poll_parser.add_argument("--observation-id", required=True)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--artifact-root", required=True)
    stop_parser.add_argument("--observation-id", required=True)
    stop_parser.add_argument("--reason", choices=STOP_REASONS, required=True)

    worker_parser = subparsers.add_parser("_run", help=argparse.SUPPRESS)
    worker_parser.add_argument("--artifact-dir", required=True)
    worker_parser.add_argument("--sampler-config-b64")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "start":
            result = start_observation(
                artifact_root=args.artifact_root,
                observation_id=args.observation_id,
                duration_seconds=args.duration_seconds,
                sample_interval_seconds=args.sample_interval_seconds,
                max_gap_seconds=args.max_gap_seconds,
                stale_after_seconds=args.stale_after_seconds,
                mode=args.mode,
                required_target_contract=args.required_target_contract,
            )
        elif args.command == "poll":
            result = poll_observation(
                artifact_root=args.artifact_root,
                observation_id=args.observation_id,
            )
        elif args.command == "stop":
            result = stop_observation(
                artifact_root=args.artifact_root,
                observation_id=args.observation_id,
                reason=args.reason,
            )
        else:
            return _run_worker(
                Path(args.artifact_dir).resolve(),
                sampler_config_b64=args.sampler_config_b64,
            )
    except DuplicateObservationError as exc:
        _print_json(
            {
                "status": "DUPLICATE_OBSERVATION_ID",
                "verification_status": "NOT_VERIFIED",
                "reason": str(exc),
            }
        )
        return 3
    except ObserverError as exc:
        _print_json(
            {
                "status": "NOT_VERIFIED",
                "verification_status": "NOT_VERIFIED",
                "reason": str(exc),
            }
        )
        return 2

    _print_json(result)
    return 0 if result.get("status") in {"STARTED", "RUNNING", "PASS"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
