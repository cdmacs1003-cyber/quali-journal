import os

from fastapi.testclient import TestClient

from server_quali import app


"""
ADMIN_TOKEN 기반 인증 헤더 설정

- /api/report, /api/archive/* 엔드포인트는
  Cloud Scheduler에서도 `Authorization: Bearer <ADMIN_TOKEN>` 으로 호출하도록
  SSOT에 정의되어 있음.
- 동일한 방식으로 테스트에서도 헤더를 붙여 401(Unauthorized)을 피하고,
  실제 운영 계약과 동일한 조건에서 스모크 테스트를 수행한다.
"""

# 서버와 동일한 ADMIN_TOKEN 값을 사용하도록 환경변수에서 읽어온다.
# (로컬에서 .env 또는 PowerShell $env:ADMIN_TOKEN 로 이미 설정되어 있다는 전제)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "local-test-token")

client = TestClient(app)


def _auth_headers() -> dict:
    """테스트용 Authorization 헤더 생성 헬퍼"""
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}



def test_report_smoke():
    """
    /api/report 스모크 테스트

    - 200 응답 여부
    - JSON 구조에 ok/op/path/count 필드 존재 여부
    - ok=True, op="report" 기본 계약 확인
    """
    resp = client.post("/api/report", headers=_auth_headers())
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, dict)

    # 기본 필드 존재 여부
    assert "ok" in data
    assert "op" in data
    assert "path" in data
    assert "count" in data

    # 최소 계약(invariant)
    assert data["ok"] is True
    assert data["op"] == "report"
    assert isinstance(data["path"], str)
    assert data["path"]  # 빈 문자열이 아니어야 함
    assert isinstance(data["count"], int)


def test_archive_smoke_from_report():
    """
    /api/report 결과에서 path를 받아 /api/archive/{path}를 스모크 테스트

    - /api/report 200 + 유효한 path 문자열
    - /api/archive/{path} 200 응답
    - Content-Type 헤더가 text/* 계열인지 확인
      (향후 SSOT에서 더 구체적인 MIME 타입으로 좁힐 수 있음)
    """
    # 먼저 /api/report 를 한 번 호출해서 path 얻기
    report_resp = client.post("/api/report", headers=_auth_headers())
    assert report_resp.status_code == 200
    report_data = report_resp.json()

    path = report_data.get("path")
    assert isinstance(path, str)
    assert path

    # 해당 path로 /api/archive/{path} 호출
    archive_resp = client.get(f"/api/archive/{path}", headers=_auth_headers())
    assert archive_resp.status_code == 200

    content_type = archive_resp.headers.get("content-type", "")
    # 우선은 text/* 계열인지 정도만 확인해두고,
    # 나중에 SSOT에서 'text/markdown' 등으로 더 구체화할 수 있음.
    assert content_type.startswith("text/")
