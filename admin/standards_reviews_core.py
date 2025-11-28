"""
admin/standards_reviews_core.py

- standard_reviews.json(표준 검수 큐) 전용 헬퍼 모듈
- server_quali.py 의 파일 구조/상태머신(2인 검수)과 동일한 규칙을 사용한다.
- 로컬/Cloud Run 모두에서 같은 경로 규칙을 따르도록 구현한다.

주요 기능
- get_reviews(status_filter, decision_filter): 목록 조회
- approve_review(standard_id, reviewer_id): HOLD/REVIEWED/PUBLISHED 승인 상태 전이
- publish_review(standard_id): REVIEWED -> PUBLISHED 발행 전이 + decision PASS 승격
- ensure_test_review_task(standard_id): 테스트용 리뷰 카드 1개 생성/보장
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status


# ---------------------------------------------------------------------------
# 공통 경로 헬퍼 (server_quali.py 와 동일한 기준)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent       # admin/
ROOT_DIR = BASE_DIR.parent                       # 프로젝트 루트 (server_quali.ROOT 와 일치해야 함)


def _is_cloud() -> bool:
    """Cloud Run 여부: K_SERVICE 환경변수 기준 (server_quali.py 와 동일)."""
    try:
        return bool(os.getenv("K_SERVICE"))
    except Exception:
        return False


def _logs_dir() -> Path:
    """
    standard_reviews.json 을 두는 logs 디렉터리.

    - Cloud Run: /tmp/logs
    - Local   : ROOT_DIR / "logs"
    """
    if _is_cloud():
        base = Path("/tmp/logs")
    else:
        base = ROOT_DIR / "logs"

    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        # 읽기 전용 환경에서도 import 자체는 실패하지 않도록 방어
        pass
    return base


def _reviews_file() -> Path:
    """standard_reviews.json 실제 파일 경로."""
    return _logs_dir() / "standard_reviews.json"


def _now_iso() -> str:
    """ISO 8601 UTC 문자열 (초 단위)."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ---------------------------------------------------------------------------
# 내부 I/O 유틸
# ---------------------------------------------------------------------------

def _load_reviews_raw() -> List[Dict[str, Any]]:
    """
    standard_reviews.json 을 그대로 읽어서 리스트 형태로 반환.

    허용 구조:
    - [ {...}, {...} ]
    - {"items": [ ... ], ...}

    파싱 실패 / 형식 불일치 시에는 빈 리스트를 반환한다.
    """
    path = _reviews_file()
    if not path.exists():
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    if not text.strip():
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 운영상 여기서 바로 죽지 않도록, 빈 리스트로 처리
        return []

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        raw = data.get("items")
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]

    # 그 외의 경우는 운영 로그에 남기고 빈 리스트로 처리
    return []


def _save_reviews_raw(items: List[Dict[str, Any]]) -> None:
    """
    standard_reviews.json 에 items 를 저장.

    - {"items": [...], "updated_at": "..."} 형태로 저장 (server_quali 와 최대한 일치)
    """
    path = _reviews_file()
    payload = {
        "items": items,
        "updated_at": _now_iso(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # 파일 쓰기 실패는 상위에서 별도 로깅하는 것을 권장.
        # 여기서는 조용히 무시(테스트/로컬 용도 고려).
        pass


def _match_standard_id(task: Dict[str, Any], standard_id: str) -> bool:
    """
    ReviewTask 에서 standard_id 를 찾는 헬퍼.

    - task["standard_id"]
    - task["item"]["id"]
    둘 중 하나라도 일치하면 동일한 것으로 본다.
    """
    if task.get("standard_id") == standard_id:
        return True
    item = task.get("item") or {}
    if item.get("id") == standard_id:
        return True
    return False


# ---------------------------------------------------------------------------
# 퍼블릭 API
# ---------------------------------------------------------------------------

def get_reviews(
    status_filter: Optional[str] = None,
    decision_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    standard_reviews.json 목록 조회.

    - status_filter / decision_filter 가 주어지면 단순 필터링
    - server_quali 의 /api/standards/reviews 와 호환되는 구조를 반환
      (여기서는 단순 리스트만 반환; HTTP 포맷은 FastAPI 레이어에서 감싼다)
    """
    items = _load_reviews_raw()
    s = (status_filter or "").strip().upper()
    d = (decision_filter or "").strip().upper()

    result: List[Dict[str, Any]] = []
    for task in items:
        if s and str(task.get("status") or "").upper() != s:
            continue
        # decision 은 task["decision"] 또는 task["item"]["decision"] 둘 다 허용
        decision_val = (task.get("decision") or ((task.get("item") or {}).get("decision") or "")).upper()
        if d and decision_val != d:
            continue
        result.append(task)
    return result


def approve_review(standard_id: str, reviewer_id: str) -> Dict[str, Any]:
    """
    HOLD/REVIEWED/PUBLISHED 상태에서의 승인 처리.

    - HOLD 인 경우:
      - approved_by 에 reviewer_id 추가
      - approved_by 길이가 required_reviewers(기본 2)에 도달하면 status=REVIEWED
    - REVIEWED/PUBLISHED 인 경우:
      - approved_by 에만 멱등하게 추가 (상태는 그대로 유지)
    - 존재하지 않는 standard_id 이면 HTTP 404
    """
    items = _load_reviews_raw()
    found = False
    standard_id = str(standard_id).strip()
    reviewer_id = str(reviewer_id).strip()
    if not reviewer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="reviewer_id required",
        )

    for task in items:
        if not _match_standard_id(task, standard_id):
            continue

        found = True

        # 기본값/HOLD 처리
        status_val = (task.get("status") or "HOLD").upper()
        required = int(task.get("required_reviewers") or 2)

        approved_by = task.get("approved_by") or []
        if not isinstance(approved_by, list):
            approved_by = [approved_by]
        if reviewer_id not in approved_by:
            approved_by.append(reviewer_id)
        task["approved_by"] = approved_by

        # 상태머신: HOLD -> REVIEWED
        if status_val == "HOLD" and len(approved_by) >= required:
            task["status"] = "REVIEWED"
        # REVIEWED/PUBLISHED 인 경우에는 상태 유지 (멱등)

        # 로그
        log = task.get("log") or {}
        history = log.get("history") or []
        if not isinstance(history, list):
            history = [history]
        now = _now_iso()
        history.append({"ts": now, "event": "approve", "by": reviewer_id})
        log["history"] = history
        log["updated_at"] = now
        task["log"] = log
        break

    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="review task not found",
        )

    _save_reviews_raw(items)
    return task


def publish_review(standard_id: str) -> Dict[str, Any]:
    """
    REVIEWED → PUBLISHED 발행 플로우.

    - status=REVIEWED 인 태스크만 발행 허용
    - 이미 PUBLISHED 인 경우 멱등하게 그대로 반환
    - 그 외 상태(HOLD 등)에서는 HTTP 409 (server_quali 와 맞춤)
    """
    items = _load_reviews_raw()
    found = False
    standard_id = str(standard_id).strip()

    for task in items:
        if not _match_standard_id(task, standard_id):
            continue

        found = True
        status_val = (task.get("status") or "HOLD").upper()

        if status_val not in ("REVIEWED", "PUBLISHED"):
            # 아직 리뷰 미완료(HOLD 등) 상태에서 발행 시도
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="review task not reviewed",
            )

        # 이미 PUBLISHED 면 멱등하게 그대로 반환
        if status_val == "PUBLISHED":
            return task

        # REVIEWED -> PUBLISHED 전이 + decision PASS 승격(A안)
        task["status"] = "PUBLISHED"

        # 정책 A안: 발행된 표준은 PASS 로 간주
        # task["decision"] 또는 task["item"]["decision"] 둘 중 하나를 유지
        task["decision"] = "PASS"
        item = task.get("item") or {}
        if item.get("decision") != "PASS":
            item["decision"] = "PASS"
        task["item"] = item

        # 로그
        log = task.get("log") or {}
        history = log.get("history") or []
        if not isinstance(history, list):
            history = [history]
        now = _now_iso()
        history.append({"ts": now, "event": "publish", "by": "system"})
        log["history"] = history
        log["updated_at"] = now
        task["log"] = log
        break

    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="review task not found",
        )

    _save_reviews_raw(items)
    return task


# ---------------------------------------------------------------------------
# 테스트용 리뷰 카드 1개 생성 헬퍼
# ---------------------------------------------------------------------------

STANDARD_KEY_SSOT = {
    "name": "QualiJournal-Admin-Standards-API",
    "rev": "v1",
    "date": "2025-11-19",
}


def _make_test_item(standard_id: str) -> Dict[str, Any]:
    """SSOT 헌법의 StandardItem 스키마를 따르는 테스트용 item."""
    now = _now_iso()
    return {
        "id": standard_id,
        "title": "[TEST] HOLD→REVIEWED→PUBLISHED 상태머신 검증용 표준",
        "url": "https://example.com/qualijournal/test-standard",
        "standard_key": STANDARD_KEY_SSOT,
        # 4축 스코어 예시 (PASS 기준 14점 미만으로 설정해 HOLD 유지)
        "score_regular": 3,
        "score_applic": 3,
        "score_evid": 3,
        "score_trust": 3,
        "score_total": 12,
        "decision": "HOLD",
        "reason_short": "[TEST] 자동 생성된 상태머신 검증용 카드",
        "meta": {
            "publisher": "QualiJournal-Test",
            "published_at": None,
            "language": "en",
            "tags": ["TEST", "STATE_MACHINE"],
        },
        "log": {
            "created_at": now,
            "updated_at": now,
            "created_by": "system",
            "updated_by": "system",
        },
    }


def ensure_test_review_task(standard_id: str = "TEST-STD-1") -> Dict[str, Any]:
    """
    테스트용 리뷰 카드 1개를 보장한다.

    - 이미 해당 ID의 태스크가 있으면 그대로 반환
    - 없으면 HOLD 상태의 새 태스크를 생성하여 standard_reviews.json 에 저장 후 반환
    """
    standard_id = str(standard_id).strip()
    items = _load_reviews_raw()

    for task in items:
        if _match_standard_id(task, standard_id):
            return task

    now = _now_iso()
    item = _make_test_item(standard_id)
    new_task: Dict[str, Any] = {
        "standard_id": standard_id,
        "status": "HOLD",
        "decision": "HOLD",
        "required_reviewers": 2,
        "approved_by": [],
        "item": item,
        "log": {
            "created_at": now,
            "updated_at": now,
            "history": [
                {"ts": now, "event": "init_test_task", "by": "system"},
            ],
        },
    }

    items.append(new_task)
    _save_reviews_raw(items)
    return new_task


# ---------------------------------------------------------------------------
# CLI 사용 (로컬에서 python -m admin.standards_reviews_core init-test 등)
# ---------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage:")
        print("  python -m admin.standards_reviews_core init-test [STANDARD_ID]")
        return 0

    cmd = argv[0]
    if cmd == "init-test":
        std_id = argv[1] if len(argv) > 1 else "TEST-STD-1"
        task = ensure_test_review_task(std_id)
        path = _reviews_file()
        print(f"[OK] 테스트 리뷰 태스크 생성/보장 standard_id={std_id}")
        print(f" - 파일 경로: {path}")
        print(f" - status: {task.get('status')}, approved_by={task.get('approved_by')}")
        return 0

    print(f"Unknown command: {cmd}")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI 진입점
    raise SystemExit(_main())
