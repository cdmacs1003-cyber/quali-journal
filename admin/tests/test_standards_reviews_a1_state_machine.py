# admin/tests/test_standards_reviews_a1_state_machine.py

import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient

# tests\*.py 기준으로 admin 루트를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 테스트용 ADMIN_TOKEN 값을 ENV에 먼저 심는다.
# server_quali.import 시점에 authorize()가 이 값을 읽어간다고 가정. 
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")

from server_quali import app

client = TestClient(app)

# 헤더에 사용할 토큰은 ENV와 동일 값으로 사용
ADMIN_TOKEN = os.environ["ADMIN_TOKEN"]
STD_ID = "TEST-STD-1"


def _headers():
    # 헌법/Runbook 기준: X-Admin-Token 헤더 필수 :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
    return {"X-Admin-Token": ADMIN_TOKEN}


def test_a1_state_machine_flow():
    """
    L3 표준 리뷰 A1 Gate – 상태머신 상수화 테스트.

    Flow:
      1) test/init (reset=True)
      2) HOLD 리스트 확인
      3) approve(r1) → 여전히 HOLD
      4) approve(r2) → REVIEWED
      5) publish → PUBLISHED + PASS
    헌법 A/C와 인수인계 런북에 정의된 SSOT를 그대로 검증한다.
    """

    # 1) 테스트 카드 시드 생성 (reset=true)
    resp = client.post(
        "/api/standards/reviews/test/init",
        json={"standard_id": STD_ID, "reset": True},
        headers=_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    # 응답 스키마는 {ok, data} 또는 바로 data 일 수 있으니 둘 다 허용
    data = body.get("data", body)
    task = data.get("review_task", data)

    assert task["standard_id"] == STD_ID
    assert task["status"] == "HOLD"
    assert task.get("required_reviewers", 2) == 2
    assert task.get("approved_by", []) == []

    # 2) HOLD 리스트에서 TEST-STD-1 확인 :contentReference[oaicite:5]{index=5}
    resp = client.get(
        "/api/standards/reviews?status=HOLD",
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json().get("data", resp.json())
    items = data.get("items", data.get("reviews", []))

    assert any(
        it.get("standard_id") == STD_ID
        and it.get("status") == "HOLD"
        and it.get("approved_by", []) == []
        for it in items
    )

    # 3) 1차 승인(r1) – 여전히 HOLD 유지 (required_reviewers=2) :contentReference[oaicite:6]{index=6}
    resp = client.post(
        f"/api/standards/reviews/{STD_ID}/approve",
        json={"reviewer_id": "r1"},
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json().get("data", resp.json())
    task = data.get("review_task", data)
    assert "r1" in task["approved_by"]
    assert task["status"] == "HOLD"

    # 4) 2차 승인(r2) – REVIEWED 전환
    resp = client.post(
        f"/api/standards/reviews/{STD_ID}/approve",
        json={"reviewer_id": "r2"},
        headers=_headers(),
    )
    assert resp.status_code == 200

    # REVIEWED 리스트에서 확인 :contentReference[oaicite:7]{index=7}
    resp = client.get(
        "/api/standards/reviews?status=REVIEWED",
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json().get("data", resp.json())
    items = data.get("items", data.get("reviews", []))

    reviewed = [
        it for it in items
        if it.get("standard_id") == STD_ID
    ]
    assert reviewed, "REVIEWED 리스트에 TEST-STD-1이 있어야 합니다."
    task = reviewed[0]
    assert task["status"] == "REVIEWED"
    assert set(task.get("approved_by", [])) >= {"r1", "r2"}
    assert task.get("required_reviewers", 2) == 2

    # 5) publish – PUBLISHED + PASS 승격 :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    resp = client.post(
        f"/api/standards/reviews/{STD_ID}/publish",
        headers={**_headers(), "Content-Length": "0"},
    )
    assert resp.status_code == 200

    resp = client.get(
        "/api/standards/reviews?status=PUBLISHED",
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json().get("data", resp.json())
    items = data.get("items", data.get("reviews", []))

    published = [
        it for it in items
        if it.get("standard_id") == STD_ID
    ]
    assert published, "PUBLISHED 리스트에 TEST-STD-1이 있어야 합니다."
    task = published[0]
    assert task["status"] == "PUBLISHED"
    assert set(task.get("approved_by", [])) >= {"r1", "r2"}
    assert task.get("decision") == "PASS"

def test_a1_state_machine_error_cases():
    """
    표준 리뷰 A1 Gate – 에러 케이스(B/C Gate) 상수화 테스트.

    케이스:
      1) 존재하지 않는 ID에 대한 approve → 404 + "review task not found"
      2) HOLD 상태에서 바로 publish → 409 + "review task not reviewed"
    C 헌법 Runbook 에 정의된 에러 규칙을 그대로 검증한다.
    """

    # 먼저 정상 테스트 카드가 있다는 것을 보장한다.
    # reset=true 로 큐를 초기화하고 TEST-STD-1 HOLD 카드 1개만 남긴다.
    resp = client.post(
        "/api/standards/reviews/test/init",
        json={"standard_id": STD_ID, "reset": True},
        headers=_headers(),
    )
    assert resp.status_code == 200

    # 1) 존재하지 않는 ID에 대한 approve → 404 + detail 메시지 확인
    resp = client.post(
        "/api/standards/reviews/NO-SUCH-ID/approve",
        json={"reviewer_id": "rX"},
        headers=_headers(),
    )
    assert resp.status_code == 404
    body = resp.json()
    # 공통 에러 스키마 {ok:false,error_code,...,detail} 를 감안해서 detail 필드를 중심으로 본다.
    detail = body.get("detail") or body.get("error") or ""
    assert "review task not found" in str(detail)

    # 2) HOLD 상태에서 바로 publish → 409 + detail 메시지 확인
    # 다시 한 번 reset=true 로 HOLD 상태를 보장한다.
    resp = client.post(
        "/api/standards/reviews/test/init",
        json={"standard_id": STD_ID, "reset": True},
        headers=_headers(),
    )
    assert resp.status_code == 200

    # 승인 없이 바로 publish 호출
    resp = client.post(
        f"/api/standards/reviews/{STD_ID}/publish",
        headers={**_headers(), "Content-Length": "0"},
    )
    assert resp.status_code == 409
    body = resp.json()
    detail = body.get("detail") or body.get("error") or ""
    assert "review task not reviewed" in str(detail)

