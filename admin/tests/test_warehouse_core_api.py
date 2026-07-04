import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server_quali import app, authorize


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("QUALI_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("QUALI_WAREHOUSE_ROOT", str(tmp_path / "data" / "warehouse"))
    monkeypatch.setenv("QUALI_LIBRARY_ROOT", str(tmp_path / "data" / "library"))
    monkeypatch.setenv("QUALI_WAREHOUSE_BACKUP_ROOT", str(tmp_path / "backup" / "warehouse"))
    monkeypatch.setenv("QUALI_WAREHOUSE_PROOFPACK_ROOT", str(tmp_path / "reports" / "proofpacks" / "warehouse"))
    monkeypatch.setenv("QUALI_WAREHOUSE_RELEASE_ROOT", str(tmp_path / "releases" / "warehouse"))

    async def _ok():
        return True

    app.dependency_overrides[authorize] = _ok
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(authorize, None)


def _create_good_item(client: TestClient) -> str:
    resp = client.post(
        "/api/warehouse/items",
        json={
            "item_type": "expert_knowhow",
            "title": "IPC solder joint inspection field note",
            "summary": "A reviewer-approved field note for warehouse simulation.",
            "raw_text": "Field note summary only. No paid standard raw text is included.",
            "raw_mime_type": "text/plain",
            "provenance": {
                "source_type": "expert",
                "source_title": "Internal field note",
                "source_author": "reviewer-a",
                "source_org": "Quali",
                "source_date": "2026-05-14",
                "captured_by": "capturer-a",
                "source_locator": "internal://field-note/001",
            },
            "rights_status": "owned",
            "sensitivity": "internal",
            "visibility": "library_candidate",
            "tags": ["ipc", "soldering"],
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["ok"] is True
    item_id = body["item"]["warehouse_item_id"]
    assert body["item"]["raw_hash"].startswith("sha256:")
    return item_id


def _move_to_pending_review(client: TestClient, item_id: str) -> None:
    for status in ["untriaged", "triaged", "pending_review"]:
        resp = client.patch(
            f"/api/warehouse/items/{item_id}/status",
            json={"status": status, "actor_id": "tester", "reason": "simulation"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["item"]["status"] == status


def _review_and_approve(client: TestClient, item_id: str) -> None:
    _move_to_pending_review(client, item_id)
    review = client.post(
        f"/api/warehouse/items/{item_id}/reviews",
        json={
            "reviewer_id": "subject-reviewer-1",
            "reviewer_role": "Subject Reviewer",
            "review_decision": "approved_for_library",
            "review_note": "Source, rights, and training value reviewed.",
            "quality_score": 88,
            "confidence_score": 0.92,
            "rights_status_confirmed": True,
            "sensitivity_confirmed": True,
            "promotion_recommendation": "library_reference_card",
        },
    )
    assert review.status_code == 200, review.text
    approval = client.post(
        f"/api/warehouse/items/{item_id}/approve",
        json={"approver_id": "approver-1", "approval_note": "Approved for dry-run promotion."},
    )
    assert approval.status_code == 200, approval.text


def _extract_ripple_package(tmp_path: Path) -> Path:
    zip_path = Path(r"H:\장기기억\장기기억_등록.zip")
    if not zip_path.exists():
        pytest.skip("real qualilibrary_ripple package zip is not available")

    target = tmp_path / "qualilibrary_ripple_src"
    marker = "qualilibrary_ripple_integrated_extensible_v0.3.2/"
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if marker in name]
        if not names:
            pytest.skip("qualilibrary_ripple package was not found in zip")
        for name in names:
            rel = name.split(marker, 1)[1]
            if not rel or rel.endswith("/"):
                continue
            parts = set(Path(rel).parts)
            if ".venv" in parts or "__pycache__" in parts or rel.endswith(".pyc"):
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    return target


def test_warehouse_full_simulation_flow(client: TestClient):
    manifest = client.get("/api/warehouse/manifest")
    assert manifest.status_code == 200
    assert manifest.json()["manifest"]["module_id"] == "QLIB-WAREHOUSE"

    item_id = _create_good_item(client)
    _move_to_pending_review(client, item_id)

    review = client.post(
        f"/api/warehouse/items/{item_id}/reviews",
        json={
            "reviewer_id": "subject-reviewer-1",
            "reviewer_role": "Subject Reviewer",
            "review_decision": "approved_for_library",
            "review_note": "Source, rights, and training value reviewed.",
            "quality_score": 88,
            "confidence_score": 0.92,
            "rights_status_confirmed": True,
            "sensitivity_confirmed": True,
            "promotion_recommendation": "library_reference_card",
        },
    )
    assert review.status_code == 200, review.text
    assert review.json()["item"]["quality_score"] == 88

    approval = client.post(
        f"/api/warehouse/items/{item_id}/approve",
        json={"approver_id": "approver-1", "approval_note": "Approved for dry-run promotion."},
    )
    assert approval.status_code == 200, approval.text
    assert approval.json()["item"]["status"] == "approved_for_library"

    dry_run = client.post(
        f"/api/warehouse/items/{item_id}/promotion-dry-run",
        json={"promotion_target": "library_reference_card", "created_by": "librarian-1"},
    )
    assert dry_run.status_code == 200, dry_run.text
    dry_body = dry_run.json()
    assert dry_body["ok"] is True
    assert dry_body["dry_run"]["decision"] == "PASS"

    promoted = client.post(
        f"/api/warehouse/items/{item_id}/promote",
        json={"promotion_target": "library_reference_card", "created_by": "librarian-1"},
    )
    assert promoted.status_code == 200, promoted.text
    promoted_body = promoted.json()
    assert promoted_body["item"]["status"] == "promoted"
    trace_id = promoted_body["trace"]["promotion_trace_id"]

    trace = client.get(f"/api/warehouse/traces/{trace_id}")
    assert trace.status_code == 200
    assert trace.json()["trace"]["warehouse_item_id"] == item_id

    backup = client.post("/api/warehouse/backup/run", json={"created_by": "operator-1"})
    assert backup.status_code == 200, backup.text
    backup_id = backup.json()["backup"]["backup_id"]

    restore = client.post(f"/api/warehouse/backup/restore-dry-run/{backup_id}")
    assert restore.status_code == 200, restore.text
    assert restore.json()["backup"]["restore_dry_run_pass"] is True

    release = client.post(
        f"/api/warehouse/release-board/update?item_id={item_id}",
        json={
            "scope": "warehouse simulation",
            "changed_files": ["admin/warehouse_core.py"],
            "test_results": [{"name": "test_warehouse_full_simulation_flow", "decision": "PASS"}],
            "rollback_plan": "restore latest warehouse backup",
            "approver": "operator-1",
            "handover_path": "HANDOVER_REPORT.md",
        },
    )
    assert release.status_code == 200, release.text
    assert release.json()["release_board"]["decision"] == "PASS"

    validation = client.post(f"/api/warehouse/validate?item_id={item_id}")
    assert validation.status_code == 200, validation.text
    assert validation.json()["ok"] is True

    status = client.get("/api/warehouse/status")
    assert status.status_code == 200
    assert status.json()["state_counts"]["promoted"] == 1


def test_warehouse_promote_calls_real_qualilibrary_ripple(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ripple_src = _extract_ripple_package(tmp_path)
    ltm_root = tmp_path / "ltm_root"
    monkeypatch.setenv("QUALI_LIBRARY_RIPPLE_ENABLED", "1")
    monkeypatch.setenv("QUALI_LIBRARY_RIPPLE_PYTHONPATH", str(ripple_src))
    monkeypatch.setenv("QUALI_LIBRARY_LTM_ROOT", str(ltm_root))
    monkeypatch.setenv("LTM_ROOT", str(ltm_root))

    item_id = _create_good_item(client)
    _review_and_approve(client, item_id)
    binding = {
        "org": "QLIB",
        "doc_code": "warehouse-integration",
        "rev": "v1",
        "year": 2026,
        "source_lang": "EN",
        "doc_kind": "REFERENCE",
        "title_en": "Warehouse Integration Solder Joint Sample",
        "search_query": "solder joint",
    }

    dry_run = client.post(
        f"/api/warehouse/items/{item_id}/promotion-dry-run",
        json={"promotion_target": "library_reference_card", "created_by": "librarian-1", "library_binding": binding},
    )
    assert dry_run.status_code == 200, dry_run.text
    dry_body = dry_run.json()["dry_run"]
    assert dry_body["decision"] == "PASS"
    assert dry_body["library_engine"]["enabled"] is True
    assert dry_body["library_engine"]["dry_run_result"]["ok"] is True

    promoted = client.post(
        f"/api/warehouse/items/{item_id}/promote",
        json={"promotion_target": "library_reference_card", "created_by": "librarian-1", "library_binding": binding},
    )
    assert promoted.status_code == 200, promoted.text
    trace = promoted.json()["trace"]
    engine = trace["library_engine"]
    assert engine["enabled"] is True
    assert engine["decision"] == "PASS"
    assert engine["add_result"]["ok"] is True
    assert engine["verify_result"]["ok"] is True
    assert engine["ripple_rebuild_result"]["ok"] is True
    assert engine["ripple_search_result"]["ok"] is True
    assert engine["ripple_search_hits"]
    assert engine["node_id"] == "QLIB:warehouse-integration@v1"

    artifacts = engine["expected_artifacts"]
    for key in [
        "brain_db_path",
        "graph_db_path",
        "library_raw_path",
        "library_template_path",
        "library_card_path",
        "ripple_index_path",
    ]:
        assert Path(artifacts[key]).exists(), key

    backup = client.post("/api/warehouse/backup/run", json={"created_by": "operator-1"})
    assert backup.status_code == 200, backup.text
    backup_id = backup.json()["backup"]["backup_id"]
    restore = client.post(f"/api/warehouse/backup/restore-dry-run/{backup_id}")
    assert restore.status_code == 200, restore.text
    release = client.post(
        f"/api/warehouse/release-board/update?item_id={item_id}",
        json={
            "scope": "warehouse real qualilibrary ripple integration",
            "changed_files": ["admin/warehouse_core.py", "admin/tests/test_warehouse_core_api.py"],
            "test_results": [{"name": "test_warehouse_promote_calls_real_qualilibrary_ripple", "decision": "PASS"}],
            "rollback_plan": "restore latest warehouse backup",
            "approver": "operator-1",
            "handover_path": "HANDOVER_REPORT.md",
        },
    )
    assert release.status_code == 200, release.text

    validation = client.post(f"/api/warehouse/validate?item_id={item_id}")
    assert validation.status_code == 200, validation.text
    assert validation.json()["ok"] is True


def test_warehouse_blocks_unknown_rights_approval(client: TestClient):
    item_id = _create_good_item(client)
    item = client.get(f"/api/warehouse/items/{item_id}").json()["item"]
    item["rights_status"] = "unknown"

    # Create a separate unknown-rights item through the public API to exercise
    # the hard gate rather than mutating the store.
    resp = client.post(
        "/api/warehouse/items",
        json={
            "item_type": "report",
            "title": "Unknown rights report",
            "summary": "Should not pass approval.",
            "raw_text": "Pointer-safe summary.",
            "provenance": {
                "source_type": "public_source",
                "source_title": "Unknown source",
                "captured_by": "capturer-a",
                "source_locator": "https://example.test/unknown",
            },
            "rights_status": "unknown",
            "sensitivity": "internal",
            "visibility": "library_candidate",
        },
    )
    assert resp.status_code == 201, resp.text
    unknown_id = resp.json()["item"]["warehouse_item_id"]
    _move_to_pending_review(client, unknown_id)
    review = client.post(
        f"/api/warehouse/items/{unknown_id}/reviews",
        json={
            "reviewer_id": "reviewer-1",
            "review_decision": "approved_for_library",
            "review_note": "Content looks useful but rights are unknown.",
            "quality_score": 90,
            "confidence_score": 0.9,
            "rights_status_confirmed": False,
            "sensitivity_confirmed": True,
        },
    )
    assert review.status_code == 200
    approval = client.post(
        f"/api/warehouse/items/{unknown_id}/approve",
        json={"approver_id": "approver-1", "approval_note": "Attempt should fail."},
    )
    assert approval.status_code == 409
    assert approval.json()["detail"]["code"] == "WH-APPROVAL-GATE"


def test_warehouse_blocks_invalid_state_transition(client: TestClient):
    item_id = _create_good_item(client)
    resp = client.patch(
        f"/api/warehouse/items/{item_id}/status",
        json={"status": "hold", "actor_id": "tester", "reason": "invalid jump"},
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "WH-STATE-INVALID"
