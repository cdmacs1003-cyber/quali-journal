# admin/tests/test_tools_quick.py

import pytest
from fastapi.testclient import TestClient

from server_quali import app, authorize, _run_py, _sync_after_save  # _run_py, _sync_after_save는 실제 모듈 이름에 맞게


client = TestClient(app)


@pytest.fixture(autouse=True)
def override_auth():
    """Quick Tools 테스트용: 인증 무조건 통과."""
    def _fake_authorize():
        return True

    app.dependency_overrides[authorize] = _fake_authorize
    yield
    app.dependency_overrides.pop(authorize, None)


@pytest.fixture
def patch_run_and_sync(monkeypatch):
    """
    _run_py, _sync_after_save 결과를 테스트마다 바꿔 끼우기 위한 헬퍼.
    사용 예:
        patch_run_and_sync(rc=0, err="", sync_ok=True)
    """
    def _factory(rc: int, stderr: str, sync_ok: bool):
        def fake_run_py(script_name, args):
            # stdout은 테스트 중요도가 낮으므로 빈 문자열로 둔다.
            return rc, "", stderr

        def fake_sync_after_save():
            return {"ok": sync_ok, "stdout": "fallback merge ok" if sync_ok else "failed"}

        monkeypatch.setattr("server_quali._run_py", fake_run_py)
        monkeypatch.setattr("server_quali._sync_after_save", fake_sync_after_save)

    return _factory


# T1: approve_top – rc=0 이면 ok=True
def test_approve_top_rc0_ok(monkeypatch, patch_run_and_sync):
    patch_run_and_sync(rc=0, stderr="", sync_ok=True)

    resp = client.post("/api/tools/approve_top?n=20")
    data = resp.json()

    assert resp.status_code == 200
    assert data["ok"] is True
    assert data["rc"] == 0
    assert data["top"] == 20


# T2: approve_top – rc=127 + sync.ok=True + "not found" → ok=True (B안)
def test_approve_top_rc127_fallback_ok(monkeypatch, patch_run_and_sync):
    patch_run_and_sync(
        rc=127,
        stderr="force_approve_top20.py not found",
        sync_ok=True,
    )

    resp = client.post("/api/tools/approve_top?n=20")
    data = resp.json()

    assert resp.status_code == 200
    assert data["rc"] == 127
    assert data["ok"] is True
    assert data["synced"] is True
    assert data["sync_log"]["ok"] is True


# T3: approve_top – rc=127 + sync.ok=False → ok=False
def test_approve_top_rc127_sync_fail(monkeypatch, patch_run_and_sync):
    patch_run_and_sync(
        rc=127,
        stderr="force_approve_top20.py not found",
        sync_ok=False,
    )

    resp = client.post("/api/tools/approve_top?n=20")
    data = resp.json()

    assert resp.status_code == 200  # API 자체는 200, 내용만 실패
    assert data["rc"] == 127
    assert data["ok"] is False
    assert data["synced"] is False or data["sync_log"]["ok"] is False


# T4: repair – rc=0 → ok=True
def test_repair_rc0_ok(monkeypatch, patch_run_and_sync):
    patch_run_and_sync(rc=0, stderr="", sync_ok=True)

    resp = client.post("/api/tools/repair")
    data = resp.json()

    assert resp.status_code == 200
    assert data["ok"] is True
    assert data["rc"] == 0


# T5: repair – rc=127 + sync.ok=True + "not found" → ok=True
def test_repair_rc127_fallback_ok(monkeypatch, patch_run_and_sync):
    patch_run_and_sync(
        rc=127,
        stderr="repair_selection_files.py not found",
        sync_ok=True,
    )

    resp = client.post("/api/tools/repair")
    data = resp.json()

    assert resp.status_code == 200
    assert data["rc"] == 127
    assert data["ok"] is True
    assert data["synced"] is True
    assert data["sync_log"]["ok"] is True


# T6: repair – rc=127 + sync.ok=False → ok=False
def test_repair_rc127_sync_fail(monkeypatch, patch_run_and_sync):
    patch_run_and_sync(
        rc=127,
        stderr="repair_selection_files.py not found",
        sync_ok=False,
    )

    resp = client.post("/api/tools/repair")
    data = resp.json()

    assert resp.status_code == 200
    assert data["rc"] == 127
    assert data["ok"] is False
