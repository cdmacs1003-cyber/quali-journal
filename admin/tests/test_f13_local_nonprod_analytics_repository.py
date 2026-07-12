import copy
import hashlib
import inspect
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import admin.f13_local_nonprod_analytics_repository as repository_module
from admin.f13_analytics_improvement_candidate_contract import (
    build_improvement_idempotency_key,
    map_consent_analytics_to_warehouse_candidate,
)
from admin.f13_local_nonprod_analytics_repository import (
    ENVIRONMENT_LOCAL_NONPRODUCTION,
    LocalNonprodAnalyticsRepository,
    LocalNonprodRepositoryError,
    RECORD_TYPE_ANALYTICS,
    RECORD_TYPE_CANDIDATE,
    REQUIRED_ENVELOPE_FIELDS,
    validate_local_repository_path,
    validate_repository_record_envelope,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "f13_local_nonprod_analytics_repository_record.schema.json"
BASE_TIME = datetime(2026, 7, 12, tzinfo=timezone.utc)
RETENTION = "2026-07-13T00:00:00Z"
TENANT = "TEN-R454-SYNTHETIC"
ORG = "ORGUNIT-R454-SYNTHETIC"
COHORT = "COH-R454-SYNTHETIC"
USER_HASH = "sha256:" + "a" * 64
QUERY_HASH = "sha256:" + "b" * 64


def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _event(event_type="hold_created", suffix="1", **overrides):
    event = {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "event_id": f"EVT-R454-SYNTHETIC-{suffix}",
        "tenant_id": TENANT,
        "organization_id": ORG,
        "cohort_id": COHORT,
        "user_id_hash": USER_HASH,
        "event_type": event_type,
        "request_id": f"REQ-R454-SYNTHETIC-{suffix}",
        "trace_id": f"btrace:R454:SYNTHETIC:{suffix}",
        "query_hash": QUERY_HASH,
        "query_summary": "hold_event" if event_type == "hold_created" else "answer_event",
        "raw_query_stored": False,
        "risk_flags": ["evidence_missing"] if event_type == "hold_created" else [],
        "occurred_at": "2026-07-12T00:00:00Z",
    }
    event.update(overrides)
    return event


def _allowed(event_type="hold_created", suffix="1", **overrides):
    return {
        "policy_status": "ALLOW",
        "reason_code": "CONSENT_ALLOWED",
        "analytics_event_present": True,
        "analytics_event": _event(event_type, suffix, **overrides),
    }


def _candidate_result(event_type="hold_created", trigger="evidence_hold", suffix="1", **overrides):
    return map_consent_analytics_to_warehouse_candidate(
        _allowed(event_type, suffix, **overrides),
        improvement_trigger=trigger,
        created_at="2026-07-12T00:00:00Z",
    )


def _repository(tmp_path, clock=None, name="repository.sqlite"):
    root = tmp_path / "approved"
    return LocalNonprodAnalyticsRepository(
        root / name,
        approved_local_root=root,
        environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
        explicit_allow_local_durable=True,
        clock=clock or (lambda: BASE_TIME),
    )


def _store_event(repository, result=None, retention=RETENTION):
    return repository.store_analytics_event(
        result or _allowed(),
        retention_policy_id="RET-R454-SYNTHETIC",
        retention_until=retention,
    )


def _store_candidate(repository, result=None, retention=RETENTION):
    return repository.store_improvement_candidate(
        result or _candidate_result(),
        retention_policy_id="RET-R454-SYNTHETIC",
        retention_until=retention,
    )


def test_explicit_local_nonproduction_activation_initializes_empty_database(tmp_path):
    repository = _repository(tmp_path)
    assert repository.database_path.exists()
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 0
    assert repository.integrity_check() == {"integrity_ok": True, "reason_code": "INTEGRITY_OK"}
    repository.close()


@pytest.mark.parametrize(
    ("environment", "allow", "reason"),
    [
        (ENVIRONMENT_LOCAL_NONPRODUCTION, False, "EXPLICIT_LOCAL_DURABLE_ALLOW_REQUIRED"),
        ("production", True, "LOCAL_NONPRODUCTION_ENVIRONMENT_REQUIRED"),
        ("prod", True, "LOCAL_NONPRODUCTION_ENVIRONMENT_REQUIRED"),
        ("staging", True, "LOCAL_NONPRODUCTION_ENVIRONMENT_REQUIRED"),
    ],
)
def test_activation_fails_closed(environment, allow, reason, tmp_path):
    root = tmp_path / "approved"
    with pytest.raises(LocalNonprodRepositoryError, match=reason):
        LocalNonprodAnalyticsRepository(
            root / "repository.sqlite",
            approved_local_root=root,
            environment=environment,
            explicit_allow_local_durable=allow,
        )


@pytest.mark.parametrize("database_path,root", [("relative.db", "relative-root"), ("", "")])
def test_implicit_or_relative_paths_are_rejected(database_path, root):
    with pytest.raises(LocalNonprodRepositoryError):
        validate_local_repository_path(
            database_path,
            approved_local_root=root,
            environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
            explicit_allow_local_durable=True,
        )


def test_path_outside_approved_root_is_rejected(tmp_path):
    with pytest.raises(LocalNonprodRepositoryError, match="DATABASE_PATH_OUTSIDE_APPROVED_ROOT"):
        validate_local_repository_path(
            tmp_path / "outside" / "repository.sqlite",
            approved_local_root=tmp_path / "approved",
            environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
            explicit_allow_local_durable=True,
        )


def test_unc_path_is_rejected(tmp_path):
    with pytest.raises(LocalNonprodRepositoryError, match="NETWORK_PATH_FORBIDDEN"):
        validate_local_repository_path(
            r"\\server\share\repository.sqlite",
            approved_local_root=tmp_path,
            environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
            explicit_allow_local_durable=True,
        )


def test_production_like_path_is_rejected():
    with pytest.raises(LocalNonprodRepositoryError, match="PRODUCTION_LIKE_PATH_FORBIDDEN"):
        validate_local_repository_path(
            r"H:\tmp\production\repository.sqlite",
            approved_local_root=r"H:\tmp",
            environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
            explicit_allow_local_durable=True,
        )


def test_parent_traversal_is_rejected(tmp_path):
    raw = str(tmp_path / "approved" / ".." / "repository.sqlite")
    with pytest.raises(LocalNonprodRepositoryError, match="PARENT_TRAVERSAL_FORBIDDEN"):
        validate_local_repository_path(
            raw,
            approved_local_root=tmp_path / "approved",
            environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
            explicit_allow_local_durable=True,
        )


def test_schema_version_and_tables_are_recorded(tmp_path):
    repository = _repository(tmp_path)
    version = repository.connection.execute(
        "SELECT metadata_value FROM local_repository_metadata WHERE metadata_key = 'repository_schema_version'"
    ).fetchone()[0]
    tables = {row[0] for row in repository.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert version == "1"
    assert {"local_repository_metadata", "local_repository_records"} <= tables
    repository.close()


def test_envelope_schema_self_check_and_alignment():
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(REQUIRED_ENVELOPE_FIELDS)
    assert schema["properties"]["repository_schema_version"]["const"] == 1


@pytest.mark.parametrize("event_type", ["answer_rendered", "hold_created"])
def test_consent_allowed_analytics_event_is_durable_and_hash_verified(event_type, tmp_path):
    repository = _repository(tmp_path)
    result = _store_event(repository, _allowed(event_type))
    read = repository.read_record(result["record_id"], tenant_id=TENANT, organization_id=ORG)
    row = repository.connection.execute("SELECT * FROM local_repository_records").fetchone()
    envelope = dict(row)

    assert result["write_performed"] is True
    assert read["found"] is True
    assert read["domain_object"]["raw_query_stored"] is False
    assert validate_repository_record_envelope(envelope)["valid"] is True
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(envelope)
    repository.close()


def test_close_reopen_preserves_committed_event(tmp_path):
    repository = _repository(tmp_path)
    stored = _store_event(repository)
    path, root = repository.database_path, repository.approved_local_root
    repository.close()
    reopened = LocalNonprodAnalyticsRepository(
        path,
        approved_local_root=root,
        environment=ENVIRONMENT_LOCAL_NONPRODUCTION,
        explicit_allow_local_durable=True,
        clock=lambda: BASE_TIME,
    )
    assert reopened.read_record(stored["record_id"], tenant_id=TENANT, organization_id=ORG)["found"] is True
    assert reopened.integrity_check()["integrity_ok"] is True
    reopened.close()


@pytest.mark.parametrize(
    "reason",
    ["CONSENT_REQUIRED", "CONSENT_RECORD_INVALID", "CONSENT_REVOKED", "ANALYTICS_EXCLUDED", "CONSENT_USER_MISMATCH", "CONSENT_SCOPE_DENIED"],
)
def test_denied_consent_never_writes(reason, tmp_path):
    repository = _repository(tmp_path)
    result = repository.store_analytics_event(
        {"policy_status": "EXCLUDE", "reason_code": reason, "analytics_event_present": False, "analytics_event": None},
        retention_policy_id="RET-R454-SYNTHETIC",
        retention_until=RETENTION,
    )
    assert result == {"write_performed": False, "record_id": None, "reason_code": "CONSENT_NOT_ALLOWED"}
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 0
    repository.close()


def test_invalid_or_prohibited_analytics_event_never_writes(tmp_path):
    repository = _repository(tmp_path)
    result = _store_event(repository, _allowed(raw_query="R454_REJECTED_MARKER"))
    assert result["reason_code"] == "ANALYTICS_EVENT_INVALID"
    assert "R454_REJECTED_MARKER" not in repr(result)
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 0
    repository.close()


@pytest.mark.parametrize("policy,until", [("", RETENTION), ("RET-R454", None), ("RET-R454", "2026-07-11T00:00:00Z")])
def test_retention_metadata_is_required_and_future(policy, until, tmp_path):
    repository = _repository(tmp_path)
    with pytest.raises(LocalNonprodRepositoryError):
        repository.store_analytics_event(_allowed(), retention_policy_id=policy, retention_until=until)
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 0
    repository.close()


@pytest.mark.parametrize(
    ("event_type", "trigger"),
    [("hold_created", "evidence_hold"), ("hold_created", "content_gap"), ("answer_rendered", "review_needed")],
)
def test_valid_candidate_is_durable_and_linked(event_type, trigger, tmp_path):
    repository = _repository(tmp_path)
    event_result = _allowed(event_type)
    _store_event(repository, event_result)
    candidate_result = map_consent_analytics_to_warehouse_candidate(event_result, improvement_trigger=trigger, created_at="2026-07-12T00:00:00Z")
    stored = _store_candidate(repository, candidate_result)
    read = repository.read_record(stored["record_id"], tenant_id=TENANT, organization_id=ORG)
    row = repository.connection.execute("SELECT subject_hash, source_event_id FROM local_repository_records WHERE record_type = ?", (RECORD_TYPE_CANDIDATE,)).fetchone()
    candidate = read["domain_object"]

    assert stored["write_performed"] is True
    assert row["subject_hash"] is None
    assert row["source_event_id"] == event_result["analytics_event"]["event_id"]
    assert candidate["review_required"] is True
    assert candidate["auto_promote"] is False
    assert candidate["approved_for_library"] is False
    assert "user_id_hash" not in candidate
    repository.close()


def test_candidate_missing_source_is_rejected(tmp_path):
    repository = _repository(tmp_path)
    result = _store_candidate(repository)
    assert result["reason_code"] == "SOURCE_EVENT_NOT_ACTIVE"
    repository.close()


def test_candidate_deleted_source_is_rejected(tmp_path):
    repository = _repository(tmp_path)
    stored_event = _store_event(repository)
    repository.delete_record(stored_event["record_id"], tenant_id=TENANT, organization_id=ORG)
    assert _store_candidate(repository)["reason_code"] == "SOURCE_EVENT_NOT_ACTIVE"
    repository.close()


def test_candidate_expired_source_is_rejected(tmp_path):
    current = [BASE_TIME]
    repository = _repository(tmp_path, clock=lambda: current[0])
    _store_event(repository, retention="2026-07-12T01:00:00Z")
    current[0] = BASE_TIME + timedelta(hours=2)
    assert _store_candidate(repository, retention="2026-07-13T00:00:00Z")["reason_code"] == "SOURCE_EVENT_EXPIRED"
    repository.close()


def _retarget_candidate(result, *, tenant=TENANT, organization=ORG):
    changed = copy.deepcopy(result)
    candidate = changed["candidate"]
    candidate["tenant_context"]["tenant_id"] = tenant
    candidate["tenant_context"]["organization_id"] = organization
    candidate["idempotency_key"] = build_improvement_idempotency_key(
        tenant_id=tenant,
        organization_id=organization,
        source_event_id=candidate["source_event_id"],
        source_trace_id=candidate["source_trace_id"],
        improvement_trigger=candidate["improvement_trigger"],
    )
    return changed


def test_cross_tenant_candidate_source_link_is_denied(tmp_path):
    repository = _repository(tmp_path)
    source = _allowed(tenant_id="TEN-R454-OTHER")
    _store_event(repository, source)
    candidate = map_consent_analytics_to_warehouse_candidate(source, improvement_trigger="evidence_hold", created_at="2026-07-12T00:00:00Z")
    candidate = _retarget_candidate(candidate, tenant=TENANT)
    assert _store_candidate(repository, candidate)["reason_code"] == "CROSS_TENANT_LINK_DENIED"
    repository.close()


def test_cross_organization_candidate_source_link_is_denied(tmp_path):
    repository = _repository(tmp_path)
    source = _allowed(organization_id="ORGUNIT-R454-OTHER")
    _store_event(repository, source)
    candidate = map_consent_analytics_to_warehouse_candidate(source, improvement_trigger="evidence_hold", created_at="2026-07-12T00:00:00Z")
    candidate = _retarget_candidate(candidate, organization=ORG)
    assert _store_candidate(repository, candidate)["reason_code"] == "CROSS_ORGANIZATION_LINK_DENIED"
    repository.close()


def test_invalid_or_promotion_candidate_is_rejected(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    candidate = _candidate_result()
    candidate["candidate"]["auto_promote"] = True
    result = _store_candidate(repository, candidate)
    assert result["reason_code"] == "CANDIDATE_INVALID"
    repository.close()


def test_candidate_with_user_hash_is_rejected_without_echo(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    candidate = _candidate_result()
    candidate["candidate"]["user_id_hash"] = "sha256:" + "c" * 64
    result = _store_candidate(repository, candidate)
    assert result["reason_code"] == "CANDIDATE_INVALID"
    assert "c" * 64 not in repr(result)
    repository.close()


def test_same_idempotency_and_payload_is_replay_without_count_change(tmp_path):
    repository = _repository(tmp_path)
    first = _store_event(repository)
    second = _store_event(repository)
    assert first["write_performed"] is True
    assert second["reason_code"] == "IDEMPOTENT_REPLAY"
    assert second["record_id"] == first["record_id"]
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 1
    repository.close()


def test_same_idempotency_different_payload_conflicts_without_mutation(tmp_path):
    repository = _repository(tmp_path)
    first = _store_event(repository)
    changed = _allowed(query_summary=None)
    conflict = _store_event(repository, changed)
    read = repository.read_record(first["record_id"], tenant_id=TENANT, organization_id=ORG)
    assert conflict["reason_code"] == "IDEMPOTENCY_CONFLICT"
    assert read["domain_object"]["query_summary"] == "hold_event"
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 1
    repository.close()


def test_same_domain_candidate_different_payload_conflicts(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    first_result = _candidate_result()
    _store_candidate(repository, first_result)
    changed = _candidate_result(trigger="content_gap")
    changed["candidate"]["candidate_id"] = first_result["candidate"]["candidate_id"]
    conflict = _store_candidate(repository, changed)
    assert conflict["reason_code"] == "DOMAIN_OBJECT_CONFLICT"
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 2
    repository.close()


def test_same_tenant_read_and_list_are_scoped(tmp_path):
    repository = _repository(tmp_path)
    stored = _store_event(repository)
    assert repository.read_record(stored["record_id"], tenant_id=TENANT, organization_id=ORG)["found"] is True
    listed = repository.list_records(tenant_id=TENANT, organization_id=ORG)
    assert listed["count"] == 1
    assert listed["records"][0]["record_id"] == stored["record_id"]
    repository.close()


@pytest.mark.parametrize(
    ("tenant", "organization"),
    [("TEN-R454-OTHER", ORG), (TENANT, "ORGUNIT-R454-OTHER")],
)
def test_cross_scope_read_and_delete_are_hidden(tenant, organization, tmp_path):
    repository = _repository(tmp_path)
    stored = _store_event(repository)
    read = repository.read_record(stored["record_id"], tenant_id=tenant, organization_id=organization)
    deletion = repository.delete_record(stored["record_id"], tenant_id=tenant, organization_id=organization)
    assert read == {"found": False, "reason_code": "RECORD_NOT_FOUND_OR_NOT_VISIBLE", "domain_object": None}
    assert deletion["reason_code"] == "RECORD_NOT_FOUND_OR_NOT_VISIBLE"
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 1
    repository.close()


def test_cross_tenant_list_and_purge_cannot_affect_other_scope(tmp_path):
    repository = _repository(tmp_path)
    other = _allowed(tenant_id="TEN-R454-OTHER")
    _store_event(repository, other, retention="2026-07-12T01:00:00Z")
    listed = repository.list_records(tenant_id=TENANT, organization_id=ORG)
    purge = repository.purge_expired(
        tenant_id=TENANT,
        organization_id=ORG,
        now="2026-07-13T00:00:00Z",
    )
    assert listed["count"] == 0
    assert purge["deleted_event_count"] == 0
    assert repository.count_records(tenant_id="TEN-R454-OTHER", organization_id=ORG) == 1
    repository.close()


@pytest.mark.parametrize("method", ["read", "list", "delete"])
def test_unscoped_operations_are_forbidden(method, tmp_path):
    repository = _repository(tmp_path)
    with pytest.raises(LocalNonprodRepositoryError):
        if method == "read":
            repository.read_record("record:event:test", tenant_id="", organization_id=ORG)
        elif method == "list":
            repository.list_records(tenant_id=TENANT, organization_id="")
        else:
            repository.delete_record("record:event:test", tenant_id="*", organization_id=ORG)
    repository.close()


def test_expired_record_is_not_active_and_purge_removes_linked_candidate(tmp_path):
    current = [BASE_TIME]
    repository = _repository(tmp_path, clock=lambda: current[0])
    event = _store_event(repository, retention="2026-07-12T01:00:00Z")
    candidate = _store_candidate(repository, retention="2026-07-14T00:00:00Z")
    current[0] = BASE_TIME + timedelta(hours=2)
    assert repository.read_record(event["record_id"], tenant_id=TENANT, organization_id=ORG)["reason_code"] == "RECORD_EXPIRED"
    purge = repository.purge_expired(tenant_id=TENANT, organization_id=ORG)
    assert purge["deleted_event_count"] == 1
    assert purge["deleted_candidate_count"] == 1
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 0
    assert candidate["record_id"] is not None
    repository.close()


def test_retention_purge_preserves_unexpired_record(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    purge = repository.purge_expired(tenant_id=TENANT, organization_id=ORG, now="2026-07-12T12:00:00Z")
    assert purge["deleted_event_count"] == 0
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 1
    repository.close()


def test_single_candidate_delete_and_repeated_delete_are_idempotent(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    candidate = _store_candidate(repository)
    first = repository.delete_record(candidate["record_id"], tenant_id=TENANT, organization_id=ORG)
    second = repository.delete_record(candidate["record_id"], tenant_id=TENANT, organization_id=ORG)
    assert first["deleted_candidate_count"] == 1
    assert second["already_absent_count"] == 1
    assert set(first) == {"deleted_event_count", "deleted_candidate_count", "already_absent_count", "scope", "reason_code"}
    repository.close()


def test_subject_hash_delete_cascades_linked_candidate_and_returns_counts_only(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    _store_candidate(repository)
    result = repository.delete_by_subject_hash(tenant_id=TENANT, organization_id=ORG, subject_hash=USER_HASH)
    assert result["deleted_event_count"] == 1
    assert result["deleted_candidate_count"] == 1
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 0
    assert "payload" not in repr(result).lower()
    repository.close()


def test_event_write_injected_failure_rolls_back(tmp_path):
    repository = _repository(tmp_path)
    with pytest.raises(LocalNonprodRepositoryError, match="INJECTED_WRITE_FAILURE"):
        repository.store_analytics_event(
            _allowed(), retention_policy_id="RET-R454", retention_until=RETENTION, inject_failure=True
        )
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 0
    repository.close()


def test_candidate_write_injected_failure_rolls_back(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    with pytest.raises(LocalNonprodRepositoryError, match="INJECTED_WRITE_FAILURE"):
        repository.store_improvement_candidate(
            _candidate_result(), retention_policy_id="RET-R454", retention_until=RETENTION, inject_failure=True
        )
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 1
    repository.close()


def test_cascade_delete_injected_failure_rolls_back(tmp_path):
    repository = _repository(tmp_path)
    event = _store_event(repository)
    _store_candidate(repository)
    with pytest.raises(LocalNonprodRepositoryError, match="INJECTED_CASCADE_FAILURE"):
        repository.delete_record(event["record_id"], tenant_id=TENANT, organization_id=ORG, inject_failure=True)
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 2
    repository.close()


def test_purge_injected_failure_rolls_back(tmp_path):
    current = [BASE_TIME]
    repository = _repository(tmp_path, clock=lambda: current[0])
    _store_event(repository, retention="2026-07-12T01:00:00Z")
    _store_candidate(repository, retention="2026-07-14T00:00:00Z")
    current[0] = BASE_TIME + timedelta(hours=2)
    with pytest.raises(LocalNonprodRepositoryError, match="INJECTED_PURGE_FAILURE"):
        repository.purge_expired(tenant_id=TENANT, organization_id=ORG, inject_failure=True)
    assert repository.count_records(tenant_id=TENANT, organization_id=ORG) == 2
    repository.close()


def test_wrong_schema_version_fails_closed_without_migration(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    path = root / "wrong.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE local_repository_metadata (metadata_key TEXT PRIMARY KEY, metadata_value TEXT NOT NULL)")
    connection.execute("INSERT INTO local_repository_metadata VALUES ('repository_schema_version', '999')")
    connection.commit()
    connection.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(LocalNonprodRepositoryError, match="REPOSITORY_SCHEMA_VERSION_UNSUPPORTED"):
        LocalNonprodAnalyticsRepository(path, approved_local_root=root, environment=ENVIRONMENT_LOCAL_NONPRODUCTION, explicit_allow_local_durable=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_missing_schema_fingerprint_fails_closed_without_repair(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    path = root / "malformed.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE local_repository_metadata (metadata_key TEXT PRIMARY KEY, metadata_value TEXT NOT NULL)")
    connection.execute("CREATE TABLE local_repository_records (record_id TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO local_repository_metadata VALUES ('repository_schema_version', '1')")
    connection.commit()
    connection.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(LocalNonprodRepositoryError, match="REPOSITORY_SCHEMA_FINGERPRINT_INVALID"):
        LocalNonprodAnalyticsRepository(path, approved_local_root=root, environment=ENVIRONMENT_LOCAL_NONPRODUCTION, explicit_allow_local_durable=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("content", [b"NOT_A_DATABASE", b""])
def test_corrupt_or_non_database_file_fails_closed_without_overwrite(content, tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    path = root / "corrupt.sqlite"
    path.write_bytes(content)
    before = path.read_bytes()
    with pytest.raises(LocalNonprodRepositoryError, match="CORRUPT_OR_NON_SQLITE_STORAGE"):
        LocalNonprodAnalyticsRepository(path, approved_local_root=root, environment=ENVIRONMENT_LOCAL_NONPRODUCTION, explicit_allow_local_durable=True)
    assert path.read_bytes() == before


def test_canonical_serialization_and_payload_hash_are_deterministic(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    first = repository.connection.execute("SELECT payload_json, payload_hash FROM local_repository_records").fetchone()
    payload = json.loads(first["payload_json"])
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    assert first["payload_json"] == canonical
    assert first["payload_hash"] == digest
    repository.close()


def test_privacy_and_external_side_effect_boundaries_are_static():
    source = inspect.getsource(repository_module).lower()
    forbidden_imports = ("import requests", "import httpx", "import socket", "import boto", "import psycopg", "import mysql")
    assert all(token not in source for token in forbidden_imports)
    assert "warehouse_core" not in source
    assert "library_write" not in source
    assert "auto_promote = true" not in source


def test_persisted_objects_contain_no_raw_answer_or_evidence_values(tmp_path):
    repository = _repository(tmp_path)
    _store_event(repository)
    _store_candidate(repository)
    payloads = [json.loads(row[0]) for row in repository.connection.execute("SELECT payload_json FROM local_repository_records")]
    prohibited = {"raw_query", "raw_body", "raw_answer", "raw_evidence_text", "answer_text", "evidence_text", "personal_name"}
    assert all(not (prohibited & set(payload)) for payload in payloads)
    assert payloads[0]["raw_query_stored"] is False
    candidate = next(payload for payload in payloads if payload.get("candidate_type"))
    assert all(
        candidate[field] is False
        for field in ("raw_query_stored", "raw_body_stored", "raw_answer_stored", "raw_evidence_text_stored")
    )
    repository.close()
