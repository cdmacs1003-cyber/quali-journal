"""
/api/standards/reviews 리뷰 API 기본 동작 체크용 테스트 세트.

R1: 리스트 엔드포인트 스키마 확인
R2: 큐가 비어 있을 때 동작 (count == 0 가 되도록 초기화 후 확인)
R3: HOLD 상태 리뷰에 대해 2인 승인 플로우 검증
R4: REVIEWED 상태 리뷰를 PUBLISHED 로 올리는 발행 플로우 검증

※ 주의
- 이 테스트는 server_quali.py 안에 다음 요소들이 존재한다고 가정한다.
  - app (FastAPI 인스턴스)
  - _reviews_file()  (standard_reviews.json 파일 경로를 돌려주는 헬퍼)
- 아직 _reviews_file 이 없다면, 먼저 서버 쪽 패치(리뷰 API + 헬퍼)를 적용한 뒤 사용하는 것을 권장.
"""
import os
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server_quali import app

# _reviews_file 헬퍼가 서버에 존재한다는 가정 (우리가 설계한 버전 기준)
try:
    from server_quali import _reviews_file  # type: ignore[attr-defined]
except ImportError:
    # 만약 아직 서버에 헬퍼가 없다면, 임시로 logs/standard_reviews.json 을 사용
    def _reviews_file() -> Path:  # type: ignore[override]
        return Path("logs") / "standard_reviews.json"


# ---- 공통 client / headers fixture ----

@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def admin_headers() -> dict:
    """
    테스트용 관리자 헤더.
    - 실제 ADMIN_TOKEN / API_TOKEN 환경변수와 맞춰주기 위해,
      우선 env에서 토큰 값을 읽어와 사용한다.
    - 둘 다 없으면 테스트 전용 기본값("TEST_ADMIN_TOKEN")을 쓴다.
    """
    token = (
        os.environ.get("ADMIN_TOKEN")
        or os.environ.get("API_TOKEN")
        or "TEST_ADMIN_TOKEN"
    )
    return {
        "X-Admin-Token": token,
    }


# ---- 리뷰 파일 헬퍼 & 시드 ----

def _ensure_reviews_dir() -> Path:
    path = _reviews_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_reviews_json(items: list[dict]) -> Path:
    """
    standard_reviews.json 을 SSOT 구조에 맞게 저장:
    {
      "items": [...],
      "updated_at": "ISO8601"
    }
    """
    path = _ensure_reviews_dir()
    payload = {
        "items": items,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def clear_reviews_queue():
    """
    R2: 큐 비움용 fixture.
    - 테스트 시작 전에 standard_reviews.json 을 '빈 큐' 상태로 맞춘다.
    """
    path = _ensure_reviews_dir()
    _write_reviews_json([])
    yield
    # 테스트 후에도 깨끗한 상태 유지하고 싶으면 여기서 다시 비워도 됨
    _write_reviews_json([])


@pytest.fixture
def seed_hold_review_task():
    """
    R3: HOLD 상태 ReviewTask 1개를 시드.
    - standard_id: "dummy-hold-001"
    - decision: HOLD
    - status: HOLD
    - required_reviewers: 2
    - approved_by: []
    """
    items = [
        {
            "standard_id": "dummy-hold-001",
            "decision": "HOLD",
            "status": "HOLD",
            "required_reviewers": 2,
            "approved_by": [],
            "reason_short": "벤더 신뢰도 재검토 필요",
            "log": [
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "created",
                    "source": "pytest_seed",
                }
            ],
        }
    ]
    _write_reviews_json(items)
    return "dummy-hold-001"


@pytest.fixture
def seed_reviewed_task():
    """
    R4: REVIEWED 상태 ReviewTask 1개를 시드.
    - standard_id: "dummy-reviewed-001"
    - decision: HOLD (publish 시 PASS 승격 A안 가정)
    - status: REVIEWED
    - approved_by: ["editor_a", "editor_b"]
    """
    items = [
        {
            "standard_id": "dummy-reviewed-001",
            "decision": "HOLD",
            "status": "REVIEWED",
            "required_reviewers": 2,
            "approved_by": ["editor_a", "editor_b"],
            "reason_short": "시범 적용 후 발행 대기",
            "log": [
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "created",
                    "source": "pytest_seed",
                },
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "approve",
                    "reviewer_id": "editor_a",
                },
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "approve",
                    "reviewer_id": "editor_b",
                },
            ],
        }
    ]
    _write_reviews_json(items)
    return "dummy-reviewed-001"


# ---- R1 ~ R4 테스트 ----

def test_reviews_list_basic_schema(client: TestClient, admin_headers: dict):
    """
    R1: 리뷰 큐가 비어 있든 아니든, 기본 스키마와 HTTP 200을 보장하는 테스트.
    - GET /api/standards/reviews
    - status_code == 200
    - body: { ok: true, name: "...", data: { count: int, items: list } }
    """
    resp = client.get("/api/standards/reviews", headers=admin_headers)
    assert resp.status_code == 200

    body = resp.json()
    assert isinstance(body, dict)
    assert body.get("ok") is True
    assert "data" in body

    data = body["data"]
    assert isinstance(data.get("count"), int)
    assert isinstance(data.get("items"), list)


@pytest.mark.usefixtures("clear_reviews_queue")
def test_reviews_list_empty_queue(client: TestClient, admin_headers: dict):
    """
    R2: 리뷰 큐가 비어 있을 때 동작 확인.
    - 사전 조건: standard_reviews.json 비움 (clear_reviews_queue fixture)
    - 기대:
      - GET /api/standards/reviews
      - count == 0, items == []
    """
    resp = client.get("/api/standards/reviews", headers=admin_headers)
    assert resp.status_code == 200

    data = resp.json()["data"]
    assert data["count"] == 0
    assert data["items"] == []


def test_reviews_approve_flow_two_reviewers(
    client: TestClient,
    admin_headers: dict,
    seed_hold_review_task,
):
    """
    R3: HOLD 상태 ReviewTask에 대해 2인 승인 플로우 검증.
    - 1차 승인: status == HOLD, approved_by 길이 1
    - 2차 승인: status == REVIEWED, approved_by 길이 >= 2
    """
    standard_id = seed_hold_review_task

    # 1차 승인
    resp1 = client.post(
        f"/api/standards/reviews/{standard_id}/approve",
        headers=admin_headers,
        json={"reviewer_id": "editor_a"},
    )
    assert resp1.status_code == 200
    task1 = resp1.json()["data"]["review_task"]
    assert task1["standard_id"] == standard_id
    assert "editor_a" in task1["approved_by"]
    # 첫 승인 이후 status 는 HOLD 또는 REVIEWED 둘 다 허용 (정책 A/B 차이 대비)
    assert task1["status"] in ("HOLD", "REVIEWED")

    # 2차 승인
    resp2 = client.post(
        f"/api/standards/reviews/{standard_id}/approve",
        headers=admin_headers,
        json={"reviewer_id": "editor_b"},
    )
    assert resp2.status_code == 200
    task2 = resp2.json()["data"]["review_task"]
    assert task2["standard_id"] == standard_id
    assert "editor_a" in task2["approved_by"]
    assert "editor_b" in task2["approved_by"]
    assert task2["status"] == "REVIEWED"


def test_reviews_publish_flow(
    client: TestClient,
    admin_headers: dict,
    seed_reviewed_task,
):
    """
    R4: REVIEWED → PUBLISHED 발행 플로우 검증.
    - 사전 조건: REVIEWED 상태 리뷰 하나 (seed_reviewed_task)
    - 기대:
      - status == PUBLISHED
      - (정책 A안) decision == PASS 로 승격
    """
    standard_id = seed_reviewed_task

    resp = client.post(
        f"/api/standards/reviews/{standard_id}/publish",
        headers=admin_headers,
    )
    assert resp.status_code == 200

    task = resp.json()["data"]["review_task"]
    assert task["standard_id"] == standard_id
    assert task["status"] == "PUBLISHED"

    # 정책 A안: publish 시 decision 을 PASS 로 승격하는 경우
    # 서버 구현이 이 정책을 따른다면 아래 assert 를 유지,
    # 아니라면 주석 처리하거나 정책에 맞게 수정.
    assert task.get("decision") == "PASS"
