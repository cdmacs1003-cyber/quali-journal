import os

import pytest
from fastapi.testclient import TestClient

from server_quali import (
    app,
    StatusData,
    ItemsData,
    GateConfigData,
    ReportData,
    TasksFlowData,
)

"""
Admin API 응답 스키마 DoD 테스트

- /api/status
- /api/items?state=ready
- /api/config/gate_required (GET/PATCH)
- /api/report
- (/api/tasks/flow 는 환경 의존성이 커서 기본은 skip)

기존 test_report_archive_smoke.py 와 동일하게
ADMIN_TOKEN 기반 Authorization 헤더를 사용한다.
"""

# 서버와 동일한 ADMIN_TOKEN 값을 사용하도록 환경변수에서 읽어온다.
# (로컬에서 .env 또는 PowerShell $env:ADMIN_TOKEN 로 설정해 두었다는 전제)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "local-test-token")

client = TestClient(app)


def _auth_headers() -> dict:
    """테스트용 Authorization 헤더 생성 헬퍼"""
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def _validate_model(model_cls, payload: dict):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def test_status_schema_basic():
    """
    /api/status 응답 스키마 검증

    - HTTP 200
    - StatusData 로 파싱 가능해야 함
    - gate_required / selection_total / selection_approved / state_counts / gate_pass 필드 타입 검증
    """
    resp = client.get("/api/status", headers=_auth_headers())
    assert resp.status_code == 200

    body = resp.json()
    # Pydantic 모델로 1차 구조 검증 (필수 필드와 타입)
    _validate_model(StatusData, body)

    assert isinstance(body["gate_required"], int)
    assert isinstance(body["selection_total"], int)
    assert isinstance(body["selection_approved"], int)
    assert isinstance(body["state_counts"], dict)
    assert isinstance(body["gate_pass"], bool)


def test_items_ready_schema_basic():
    """
    /api/items?state=ready 응답 스키마 검증

    - HTTP 200
    - ItemsData 로 파싱 가능해야 함
    - items 는 list 이어야 함
    """
    resp = client.get("/api/items?state=ready", headers=_auth_headers())
    assert resp.status_code == 200

    body = resp.json()
    _validate_model(ItemsData, body)

    assert "items" in body
    assert isinstance(body["items"], list)


def test_gate_required_get_and_patch_schema():
    """
    /api/config/gate_required GET/PATCH 스키마 검증

    - GET: GateConfigData 로 파싱 가능
    - PATCH: {"ok": True, "gate_required": int} 구조 확인
    - 테스트는 gate_required 값을 그대로 다시 써서 '상태를 바꾸지 않는' 형태로 운용
    """
    # 1) GET으로 현재 값 조회 (/api/ready/... 가 getter 역할)
    resp_get = client.get("/api/ready/config/gate_required", headers=_auth_headers())
    assert resp_get.status_code == 200
    data_get = resp_get.json()

    _validate_model(GateConfigData, data_get)
    original = int(data_get["gate_required"])

    # 2) PATCH로 같은 값 다시 세팅 (상태 변화 없음)
    resp_patch = client.patch(
        "/api/config/gate_required",
        headers=_auth_headers(),
        json={"gate_required": original},
    )
    assert resp_patch.status_code == 200
    data_patch = resp_patch.json()

    if "ok" in data_patch:
        assert data_patch["ok"] is True
    assert int(data_patch["gate_required"]) == original


def test_report_schema_basic():
    """
    /api/report 응답 스키마 검증

    - HTTP 200
    - body["ok"] is True
    - ReportData 로 파싱 가능(op/path/count/duration_ms)
    """
    resp = client.post("/api/report", headers=_auth_headers())
    assert resp.status_code == 200

    body = resp.json()
    assert isinstance(body, dict)
    assert body.get("ok") is True

    # ReportData 는 op/path/count/duration_ms 중심으로 스키마를 고정
    _validate_model(ReportData, body)

    assert isinstance(body["path"], str)
    assert body["path"]  # 빈 문자열이 아니어야 함
    assert isinstance(body["count"], int)


@pytest.mark.skip(reason="orchestrator 실행 등 환경 의존성이 커서 기본은 skip; 준비되면 해제")
def test_tasks_flow_daily_schema():
    """
    /api/tasks/flow(kind=daily) 응답 스키마 검증 (선택 테스트)

    - HTTP 200
    - TasksFlowData 로 파싱 가능(job_id/status/kind/args)
    - 실제로는 orchestrator.py 를 실행하므로
      기본 설정에서는 skip하고, 환경 준비 후 수동으로 해제하는 것을 권장
    """
    resp = client.post(
        "/api/tasks/flow",
        headers=_auth_headers(),
        json={"kind": "daily"},
    )
    assert resp.status_code == 200

    body = resp.json()
    _validate_model(TasksFlowData, body)

    assert isinstance(body["job_id"], str)
    assert isinstance(body["status"], str)
    assert isinstance(body["kind"], str)
