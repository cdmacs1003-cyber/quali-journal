# -*- coding: utf-8 -*-
"""
QualiJournal Admin API + Lite Black UI (stable, cleaned)
- Community / Keyword / Daily flows (sync & async)
- Async Task Manager (/api/tasks/*) + SSE stream
- Gate config API (GET/PATCH /api/config/gate_required)
- Report & Export (/api/report, /api/export/{md|csv})
- Log viewing (/api/logs/*)
- UTF-8 safe on Windows; non-breaking fallback if optional modules are missing.
"""

from __future__ import annotations

import os
import sys
import json
import csv
import io
import subprocess
import datetime as _dt
import hashlib
import asyncio
import threading
import secrets
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# FastAPI / Pydantic
from fastapi import FastAPI, Query, Response, HTTPException, Depends, Body, Request, APIRouter, status as http_status 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from admin.f13_bridge_api import router as f13_bridge_router

# === DB KPI helpers (optional; safe fallback) ================================
try:
    # SQLAlchemy ORM (패치셋에서 제공)
    from app.qj_db import get_session, get_or_create_edition, kpi_for_edition, approve_top_n  # noqa: E401
    _DB_READY = True
except Exception:
    _DB_READY = False
# ============================================================================ 

# === Optional GCS client (보고서 백업용; 없어도 동작) =========================
try:
    from google.cloud import storage  # type: ignore
    _GCS_READY = True
except Exception:  # pragma: no cover
    # 라이브러리가 없거나 자격증명이 없으면 GCS 업로드는 비활성화
    storage = None  # type: ignore
    _GCS_READY = False
# ============================================================================ 


# SSOT: 서브프로세스도 이 파이썬으로 실행
PYEXE = os.getenv("PYTHON_EXE") or sys.executable or "python"

# ---------------------------------------------------------------------------
# .env (optional) - load first unless explicitly skipped; define MODE fallback regardless of availability
# ---------------------------------------------------------------------------
if os.environ.get("QUALIJOURNAL_SKIP_DOTENV", "").strip().lower() not in {"1", "true", "yes", "on"}:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        # dotenv is optional; ignore if not present
        pass

MODE = (os.getenv("QUALI_DB_MODE") or "local").lower().strip()

# admin.db (optional). If unavailable, db endpoints gracefully degrade.
try:
    from admin.db import make_engine  # type: ignore
    _engine = None
except Exception:  # pragma: no cover
    make_engine = None  # type: ignore
    _engine = None

# === FastAPI app (define FIRST) ==============================================
app = FastAPI(title="QualiJournal Admin API")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# 공통 응답·데이터 스키마 (테스트/문서용)
# 운영 응답 JSON은 그대로 두고, pytest에서 스키마 검증에만 사용한다.

class AdminResponseOK(BaseModel):
    """
    ok=True + data=dict 구조를 표현하기 위한 공통 응답 스키마 (테스트/문서용).
    현재 /api/status, /api/items 등은 data 래퍼 없이 바로 dict를 내려주지만,
    DoD 상 'ok+data' 패턴을 설명할 때 참고용으로 사용한다.
    """
    ok: bool = True
    data: Dict[str, Any]


class AdminResponseError(BaseModel):
    """
    ok=False + error/error_code 구조를 표현하기 위한 공통 에러 스키마 (테스트/문서용).
    /api/report 의 _err(...) 패턴과 개념적으로 동일하다.
    """
    ok: bool = False
    error: str
    error_code: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None

class StatusData(BaseModel):
    """
    /api/status 응답 구조.
    현재 server_quali.get_status 가 내려주는 JSON을 그대로 모델로 표현하되,
    헌법에서 쓰는 total/ready_count/ready_rate 필드를 함께 포함한다.
    """
    selected: int
    approved: int
    published: int
    gate_required: int
    ts: int
    selection_total: int
    selection_approved: int
    state_counts: Dict[str, int]
    community_total: int
    keyword_total: int
    gate_pass: bool
    date: Optional[str] = None
    keyword: str = ""
    # Ready 게이트 연동용 필드 (없으면 None)
    total: int | None = None
    ready_count: int | None = None
    ready_rate: float | None = None

class ItemsData(BaseModel):
    """
    /api/items 응답 래퍼 구조.
    items 내부는 기사 dict 구조가 다양할 수 있어서 일단 Dict[str, Any] 로 둔다.
    """
    date: Optional[str] = None
    keyword: Optional[str] = None
    state: str
    items: List[Dict[str, Any]]


class GateConfigData(BaseModel):
    """
    /api/config/gate_required GET 응답 구조.
    PATCH 응답은 {"ok": True, "gate_required": int} 이라서
    gate_required 필드를 중심으로만 검증한다.
    """
    gate_required: int


class ReportData(BaseModel):
    """
    /api/report 성공 시 핵심 필드 구조.
    ok/op/ts 는 AdminResponseOK + 메타 정보로 보고,
    여기서는 op/path/count/duration_ms 만 스키마로 고정한다.
    """
    op: str
    path: str
    count: int
    duration_ms: int


class TasksFlowData(BaseModel):
    """
    /api/tasks/flow 성공 시 구조.
    실제 호출은 환경 의존성이 커서 테스트에서는 선택적으로 사용한다.
    """
    job_id: str
    status: str
    kind: str
    args: List[Any]


class StandardScoreRequest(BaseModel):
    """
    표준·기술 자료 1건에 대해
    4축 스코어를 계산하기 위한 입력 스키마.
    """
    id: str | None = None
    title: str
    url: str
    source_tier: str = Field("official", description="official/association/vendor 중 하나")

    standard_name: str | None = None
    standard_rev: str | None = None
    standard_date: str | None = None

    meta_publisher: str | None = None
    meta_published_at: str | None = None
    meta_language: str | None = None

    tags: List[str] | None = None
    target_keywords: List[str] | None = None


class ReviewApproveReq(BaseModel):
    """
    2인 검수 승인 요청용 스키마.
    reviewer_id: 검수자 식별자(이름/이메일/ID 등)
    """
    reviewer_id: str = Field(..., description="검수자 ID (이름/이메일/계정)")

class ReviewTestInitReq(BaseModel):
    """
    표준 리뷰 테스트 카드 시드를 위한 요청 스키마.

    - standard_id: 테스트용 표준 ID (기본값 TEST-STD-1)
    - reset: True 이면 기존 리뷰 큐를 비우고 이 카드 1개만 남긴다.
    """
    standard_id: str = Field(
        default="TEST-STD-1",
        description="테스트용 standard_id (기본값 TEST-STD-1)",
    )
    reset: bool = Field(
        default=False,
        description="기존 리뷰 큐를 지우고 이 테스트 카드 1개만 남길지 여부",
    )

class EnrichReq(BaseModel):
    date: Optional[str] = None
    keyword: Optional[str] = None
    mode: Optional[str] = "keyword"  # "keyword" or "selection"
    items: Optional[List[Dict[str, Any]]] = None

class GatePatch(BaseModel):
    gate_required: int

class PublishOneReq(BaseModel):
    approve: bool = Field(default=True, description="승인 여부")
    editor_note: Optional[str] = Field(default=None, description="편집장 한마디(선택)")

class TaskItem(BaseModel):
    id: str
    size: int

class TasksRecent(BaseModel):
    items: List[TaskItem]

class ReportReq(BaseModel):
    """요청 본문: date(선택), keyword(선택)"""
    date: str | None = None
    keyword: str | None = None

class ReportResult(BaseModel):
    """
    /api/report 응답 공통 스키마
    - 성공: ok=True, path/count/duration_ms 세팅
    - 실패: ok=False, error / error_code 세팅 (path는 비워둘 수 있음)
    """
    ok: bool = True
    op: str = "report"
    path: Optional[str] = None
    count: int = 0
    ts: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None


class ErrorResponse(BaseModel):
    """
    에러 응답 공통 형태(참고용).
    지금은 /api/report 등에서 _err(...)가 같은 구조로 JSON을 내려줌.
    """
    ok: bool = False
    op: str = "report"
    error: str
    ts: int = 0
    error_code: Optional[str] = None
    duration_ms: Optional[int] = None


class FlowReq(BaseModel):
    kind: str            # daily|community|keyword
    keyword: str | None = None
    # Flag to indicate whether external RSS sources should be used for keyword collection
    use_external_rss: bool = False

class PublishReq(BaseModel):
    keyword: str

class FlowKwReq(BaseModel):
    keyword: str
    use_external_rss: bool = False

# ---------------------------------------------------------------------------
# Optional JWT utils (safe fallbacks if module missing)
# ---------------------------------------------------------------------------
try:
    from auth_utils import verify_jwt_token  # type: ignore
except Exception:  # pragma: no cover
    async def verify_jwt_token(*args, **kwargs):  # type: ignore
        return {}

# Simple Bearer Token Authorization (Cloud Run OIDC + App Token 동시 지원)
security = HTTPBearer(auto_error=False)

async def authorize(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> bool:
    """
    허용 규칙
    - ADMIN_TOKEN 또는 API_TOKEN 둘 중 하나라도 설정되어 있지 않으면 open mode(통과)
    - 설정되어 있으면 다음 중 '하나'라도 맞으면 통과
      1) 헤더 X-Admin-Token: <ADMIN_TOKEN or API_TOKEN>
      2) Authorization: Bearer <ADMIN_TOKEN or API_TOKEN>   (레거시 호환)
    - Cloud Run 비공개 서비스에서 Authorization은 보통 'ID 토큰'이므로,
      이 경우 X-Admin-Token 으로 앱 토큰을 따로 실어야 통과됨.
    """
    expected = [
        (os.environ.get("ADMIN_TOKEN") or "").strip(),
        (os.environ.get("API_TOKEN") or "").strip(),
    ]
    expected = [t for t in expected if t]
    if not expected:
        return True  # open mode

    # 앱 전용 토큰(권장): X-Admin-Token
    x_admin = (request.headers.get("X-Admin-Token") or "").strip()

    # 레거시/일반: Authorization: Bearer <token>
    supplied = credentials.credentials if credentials else ""
    if not supplied:
        # security가 못 뽑았을 때 대비
        auth = (request.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()

    if (x_admin and x_admin in expected) or (supplied and supplied in expected):
        return True

    raise HTTPException(status_code=401, detail="invalid or missing token")


def _is_local_only_non_secret_f13_bridge_answer_override_request(request: Request) -> bool:
    if os.environ.get("QJ_LOCAL_ONLY_NON_SECRET_AUTH_OVERRIDE") != "1":
        return False
    if request.method != "POST":
        return False
    if request.url.path != "/api/f13/bridge/skillup/bridge-answer":
        return False
    if request.query_params:
        return False
    for header_name in ("authorization", "x-admin-token", "x-api-token", "x-api-key", "cookie"):
        if header_name in request.headers:
            return False

    client_host = (request.client.host if request.client else "").lower()
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        return False

    host = (request.headers.get("host") or "").lower()
    allowed_hosts = {
        "127.0.0.1",
        "localhost",
        "[::1]",
    }
    if host not in allowed_hosts and not (
        host.startswith("127.0.0.1:")
        or host.startswith("localhost:")
        or host.startswith("[::1]:")
    ):
        return False

    return True


async def authorize_f13_bridge_with_local_override(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    if _is_local_only_non_secret_f13_bridge_answer_override_request(request):
        return True
    return await authorize(request, credentials)

def _auth_header_or_qs_ok(request: Request) -> bool:
    expected = [(os.environ.get("ADMIN_TOKEN") or "").strip(),
                (os.environ.get("API_TOKEN") or "").strip()]
    expected = [x for x in expected if x]
    if not expected:
        return True
    supplied = None
    hdr = request.headers.get("authorization") or request.headers.get("Authorization")
    if hdr and hdr.lower().startswith("bearer "):
        supplied = hdr.split(" ", 1)[1].strip()
    qs = request.query_params.get("token")
    if (supplied and supplied in expected) or (qs and qs in expected):
        return True
    raise HTTPException(status_code=401, detail="invalid or missing token")

# === READY/SSOT PATCH START (no conflicts version) ===========================
# - Prefix '/api/ready'로 충돌 제거
# - 인가: Depends(authorize) 사용(서버 전역과 통일)
# - 모델 중복 방지: ReadyGatePatch 사용
READY_DATA_DIR = os.environ.get("QUALI_DATA_DIR", "data")
READY_CONFIG   = os.environ.get("QUALI_CONFIG", "config.json")
READY_FILES = [
    os.path.join(READY_DATA_DIR, "selected_articles.json"),
    os.path.join(READY_DATA_DIR, "selected_keyword_articles.json"),
]

def _ready_load_json(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _ready_all_items() -> List[Dict]:
    """지원 형태: list 또는 {"articles":[…]} / {"items":[…]}"""
    acc: List[Dict] = []
    for p in READY_FILES:
        obj = _ready_load_json(p)
        if isinstance(obj, list):
            acc.extend([x for x in obj if isinstance(x, dict)])
        elif isinstance(obj, dict):
            arr = obj.get("articles") or obj.get("items") or []
            if isinstance(arr, list):
                acc.extend([x for x in arr if isinstance(x, dict)])
    return acc

def _ready_load_cfg() -> Dict:
    cfg = {"gate_required": 70}
    if os.path.exists(READY_CONFIG):
        try:
            with open(READY_CONFIG, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                if isinstance(file_cfg, dict):
                    cfg.update(file_cfg)
        except Exception:
            pass
    return cfg

ready_router = APIRouter(prefix="/api/ready", tags=["ready-ssot"])

@ready_router.get("/items")
def ready_items(
    state: str = Query("ready"),
    date: str | None = None,
    keyword: str | None = None,
    authorized: bool = Depends(authorize)
):

    """
    SSOT 기반 Ready 전용 아이템 뷰:
    - 기본 state=ready → ready==true 항목만 반환
    - date/keyword 지정 시 해당 필드가 같은 항목만 반환(필드 없으면 필터 미적용)
    반환: List[dict]
    """
    def _match(it: dict) -> bool:
        if date and str(it.get("date","")) != str(date):
            return False
        if keyword:
            if str(it.get("keyword","")).strip() != str(keyword).strip():
                return False
        return True

    items = _ready_all_items()
    s = (state or "").lower().strip()
    if s == "ready":
        items = [i for i in items if i.get("ready") is True]
    else:
        # 다른 state가 들어오더라도, SSOT 원칙상 ready 재판정은 금지 → 그대로 통과 + date/keyword만 필터
        pass
    return [i for i in items if _match(i)]

@ready_router.get("/status")
def ready_status(authorized: bool = Depends(authorize)):
    """
    SSOT 기반 상태: ready_count/ready_rate는 파일에 저장된 파생값(ready) 기준
    """
    items = _ready_all_items()
    total = len(items)
    ready_true = sum(1 for i in items if i.get("ready") is True)
    cfg = _ready_load_cfg()
    return {
        "total": total,
        "ready_count": ready_true,
        "ready_rate": (ready_true / total) if total else 0.0,
        "gate_required": int(cfg.get("gate_required", 70)),
    }

@ready_router.get("/config/gate_required")
def ready_gate_get(authorized: bool = Depends(authorize)):
    return {"gate_required": int(_ready_load_cfg().get("gate_required", 70))}

class ReadyGatePatch(BaseModel):
    gate_required: int

@ready_router.patch("/config/gate_required")
def ready_gate_patch(p: ReadyGatePatch, authorized: bool = Depends(authorize)):
    cfg = _ready_load_cfg()
    cfg["gate_required"] = int(p.gate_required)
    with open(READY_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return {"ok": True, "gate_required": cfg["gate_required"]}

app.include_router(ready_router)
app.include_router(f13_bridge_router, dependencies=[Depends(authorize_f13_bridge_with_local_override)])
# === READY/SSOT PATCH END ===================================================
# ---------------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent  # admin/

def _is_cloud() -> bool:
    """Cloud Run 여부: K_SERVICE 환경변수 기준."""
    try:
        return bool(os.getenv("K_SERVICE"))
    except Exception:
        return False


def _detect_root() -> Path:
    """
    Orchestrator root auto-detection:
    prefer admin/.. → admin/ → admin/../..
    """
    cands = [BASE.parent, BASE, BASE.parent.parent]
    for r in cands:
        if (r / "orchestrator.py").exists():
            return r
    # orchestrator.py 가 하나도 없을 때:
    # - Cloud Run(컨테이너): BASE(/app)를 ROOT로 사용
    # - 로컬: BASE.parent(프로젝트 루트)를 ROOT로 사용
    if _is_cloud():
        return BASE
    return BASE.parent



ROOT = _detect_root()
# 로컬에서만 쓰는 기본 archive 경로(Cloud Run에서는 아래에서 /tmp로 치환)
ARCHIVE_LOCAL = ROOT / "archive"
TOOLS   = ROOT / "tools"
ORCH    = ROOT / "orchestrator.py"


SEL_COMM = ROOT / "selected_community.json"
SEL_WORK = ROOT / "data" / "selected_keyword_articles.json"
SEL_PUB  = ROOT / "selected_articles.json"

# Cloud Run 여부 플래그 (K_SERVICE 기준)
IS_CLOUD = _is_cloud()


ARCHIVE_CLOUD = Path("/tmp/archive")

# Cloud Run에서는 /tmp/archive, 로컬에서는 ROOT/archive 사용
ARCHIVE = ARCHIVE_CLOUD if IS_CLOUD else ARCHIVE_LOCAL

# /archive 하위 폴더들
ENRICHED_DIR = ARCHIVE / "enriched"
REPORT_DIR   = ARCHIVE / "reports"
ENRICH_DIR   = ENRICHED_DIR

# 커뮤니티 스냅샷 후보: 루트 selected_community.json → archive 안 복사본 순서
CAND_COMM = [SEL_COMM, ARCHIVE / "selected_community.json"]

# 키워드/작업 스냅샷 후보:
# - ROOT/data/selected_keyword_articles.json (현재 SSOT)
# - ROOT/selected_articles.json, ROOT/data_selected_articles.json (옛 형식 호환)
CAND_WORK = [
    SEL_WORK,                              # ROOT / "data/selected_keyword_articles.json"
    ROOT / "selected_articles.json",       # ROOT / "selected_articles.json"
    ROOT / "data_selected_articles.json",  # ROOT / "data_selected_articles.json"
]


INDEX_HTML = BASE / "index.html"
INDEX_LITE = BASE / "index_lite_black.html"
CONFIG_FILE = ROOT / "config.json"



# runtime small stores
GATE = {"gate_required": 15}
KPI = {}

# 빌드 태그(Cloud Run /health 응답 확인용)
# - 기본값: admin-20251126-A1
# - 필요하면 Cloud Run 환경변수 QUALI_BUILD_TAG 로 덮어쓸 수 있음
BUILD_TAG = os.getenv("QUALI_BUILD_TAG", "admin-20251127-R1")

# === SAFE LOGGING (Cloud Run & local) =====================================
# 여기서는 import를 새로 하지 않고, 파일 상단의
# os / sys / logging / Path / IS_CLOUD 을 그대로 재사용한다.

LOGS_DIR = None
try:
    # Cloud Run이면 /tmp/logs, 로컬이면 프로젝트 루트/logs 사용
    LOGS_DIR = (Path("/tmp/logs") if IS_CLOUD else (Path(__file__).resolve().parent.parent / "logs"))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # 읽기 전용 컨테이너 등에서는 파일 로그 없이 stdout만 사용
    LOGS_DIR = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("server")
logger.info("server_quali logger initialized (file logging=%s)", bool(LOGS_DIR))

def _upload_report_to_gcs(local_path: Path, rel_path: str) -> bool:
    """
    /api/report 결과물을 GCS 버킷으로 best-effort 업로드.
    - 환경변수 GCS_BUCKET 이 비어 있거나 클라이언트가 없으면 False 반환.
    - 업로드 실패해도 예외를 넘기지 않고 경고 로그만 남긴다.
    - rel_path 는 'archive/reports/...' 같은 웹 경로를 기대하며,
      GCS object name 으로 그대로 사용한다.
    """
    bucket_name = (os.getenv("GCS_BUCKET") or "").strip()
    if not bucket_name or not _GCS_READY:
        # GCS 사용 불가 상태 → 그냥 로컬 파일만 두고 종료
        return False

    # 앞쪽에 '/'가 붙어 있으면 제거하고 object 이름으로 사용
    blob_name = rel_path.lstrip("/") or local_path.name

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))

        try:
            logger.info(
                "REPORT_GCS_UPLOAD_OK bucket=%s blob=%s size=%s",
                bucket_name,
                blob_name,
                local_path.stat().st_size,
            )
        except Exception:
            # 로깅 실패는 조용히 무시
            pass
        return True
    except Exception as e:
        # 업로드 실패해도 /api/report 응답은 그대로 200 유지
        try:
            logger.warning("REPORT_GCS_UPLOAD_FAILED: %s", e)
        except Exception:
            pass
        return False

# =========================================================================
# FastAPI app
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in (os.getenv("ALLOWED_ORIGINS","").split(",")) if o.strip()] or ["https://admin.standardai.co.kr"],
    allow_methods=["GET","POST","PATCH","OPTIONS"],
    allow_headers=["Authorization","Content-Type","X-Admin-Token"],
    allow_credentials=True,
)

# UTF-8 강제 헤더(정적 .md도 포함)
@app.middleware("http")
async def _force_utf8_markdown(request: Request, call_next):
    resp = await call_next(request)
    ctype = (resp.headers.get("content-type") or "").lower()
    if ctype.startswith("text/markdown") and "charset=" not in ctype:
        resp.headers["content-type"] = "text/markdown; charset=utf-8"
    return resp

def _detect_media_type(full: Path) -> str:
    """
    /api/archive용 간단 MIME 탐지:
    - .md  → text/markdown; charset=utf-8
    - .txt → text/plain; charset=utf-8
    - 그 외 → application/octet-stream
    """
    name = full.name.lower()
    if name.endswith(".md"):
        return "text/markdown; charset=utf-8"
    if name.endswith(".txt"):
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


# Static mounts
try:
    _paths = [getattr(r, "path", None) for r in getattr(app, "routes", [])]
    if "/archive/reports" not in _paths:
        app.mount("/archive/reports", StaticFiles(directory=str(REPORT_DIR)), name="archive-reports")
    if "/archive/enriched" not in _paths:
        app.mount("/archive/enriched", StaticFiles(directory=str(ENRICH_DIR)), name="archive-enriched")
    if "/archive" not in _paths:
        root_for_browse = (ARCHIVE_CLOUD if IS_CLOUD else BASE / "archive")
        root_for_browse.mkdir(parents=True, exist_ok=True)
        app.mount("/archive", StaticFiles(directory=str(root_for_browse)), name="archive-root")
except Exception:
    pass
# === UI Static & Cache Headers (dist 우선) ====================================

# admin/ 기준
_UI_BASE = BASE              # admin/
_UI_DIST = BASE / "dist"     # admin/dist
# dist/index.html이 존재하면 dist를, 아니면 개발중 원본(admin/)을 사용
_UI_ACTIVE = _UI_DIST if (_UI_DIST / "index.html").exists() else _UI_BASE
_ASSETS_DIR = _UI_ACTIVE / "assets"

# /assets 정적 서빙 (dist/assets 또는 admin/assets)
try:
    if _ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")
except Exception:
    pass

# 해시 자산: 1년 + immutable / 루트 HTML: no-cache
@app.middleware("http")
async def _cache_headers_ui(request: Request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if re.match(r"^/assets/.+\.[0-9a-f]{8,}\.(js|css|png|jpg|svg|woff2?)$", p, re.I):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif p in ("/", "/index.html"):
        # HTML은 재검증 허용
        resp.headers["Cache-Control"] = "no-cache"
    return resp

@app.get("/api/archive/{path:path}")
def download_archive(path: str, request: Request):
    """
    보호된 다운로드 엔드포인트.
    - Cloud Run: ARCHIVE_CLOUD(/tmp/archive) 기준
    - path 인자는 ARCHIVE 기준 상대경로를 기대하지만,
      관용적으로 'archive/...' 접두어도 허용한다.
    """
    # 1) 인증 확인 (헤더 우선, ?token 허용)
    _auth_header_or_qs_ok(request)

    # 2) path 정규화
    #    - 선행 '/' 제거
    #    - 'archive/' 로 시작하면 관용적으로 한 번 잘라냄
    raw = path.lstrip("/")
    if raw.startswith("archive/"):
        raw = raw[len("archive/"):]  # 'archive/' 접두어 제거

    # Path 객체로 변환 (여기까지는 아직 상대경로)
    safe_rel = Path(raw)

    # 3) ARCHIVE 기준 절대 경로 계산
    base = ARCHIVE.resolve()
    full = (base / safe_rel).resolve()

    # 4) 경로 이탈 방지(../ 등 차단) + 실제 파일 여부 확인
    if not str(full).startswith(str(base)) or not full.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # 5) 파일 응답 (확장자별 media_type 지정)
    media_type = _detect_media_type(full)
    return FileResponse(str(full), filename=full.name, media_type=media_type)

# ---------------------------------------------------------------------------
# Health & Misc
# ---------------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
async def health():
    # Cloud Run / 로컬 모두 공통 사용:
    # - ok: 헬스 자체 OK 여부
    # - status: 이전 버전과의 호환용 플래그
    return {"ok": True, "status": True, "build": BUILD_TAG}

@app.get("/readyz", include_in_schema=False)
async def readyz():
    return {"ready": True}

@app.get("/api/db/mode")
def db_mode():
    return {"mode": MODE}

# --- Approve UI opener -------------------------------------------------------
@app.post("/api/approve-ui/start")
def approve_ui_start(request: Request, authorized: bool = Depends(authorize)):
    """
    승인 UI가 열릴 때 UI URL과 현재 스냅샷(날짜/키워드/게이트)을 안내.
    - UI는 반환된 ui_url을 새 창으로 open (index.html의 startApprove 사용).
    """
    snap = _get_work_snapshot()  # {"date","keyword","articles":[...]}
    ui_url_env = (os.getenv("APPROVE_UI_URL") or "").strip()
    ui_url = ui_url_env if ui_url_env else None
    return _ok(
        "approve_ui_start",
        ui_url=ui_url,
        date=snap.get("date"),
        keyword=snap.get("keyword", ""),
        gate_required=int(GATE.get("gate_required", 15)),
    )

@app.get("/api/db/ping")
def db_ping():
    """Graceful DB availability check. Falls back to not-connected if missing."""
    global _engine
    if MODE != "cloud" or make_engine is None:
        return {"ok": True, "mode": MODE, "db": "not-connected"}
    if _engine is None:
        _engine = make_engine()
    with _engine.connect() as c:
        c.exec_driver_sql("SELECT 1;")
    return {"ok": True, "mode": "cloud", "db": "postgresql"}

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def devtools_config():
    return Response(content="{}", media_type="application/json", status_code=200)

@app.get("/favicon.ico")
def favicon_blank():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
def index():
    """
    dist/index.html이 있으면 dist를, 없으면 admin/index.html(또는 라이트 버전)로 폴백.
    HTML은 no-cache(재검증), 자산(.hash.*)은 미들웨어에서 1년+immutable.
    """
    # admin/ 기준 경로
    base_dir = Path(__file__).resolve().parent            # admin
    dist_dir = base_dir / "dist"                          # admin/dist
    index_dist = dist_dir / "index.html"
    index_src  = base_dir / "index.html"
    index_lite = base_dir / "index_lite_black.html" if (base_dir / "index_lite_black.html").exists() else None

    # dist 우선
    target = index_dist if index_dist.exists() else (index_src if index_src.exists() else index_lite)
    if target and target.exists():
        return HTMLResponse(target.read_text(encoding="utf-8"),
                            headers={"Cache-Control": "no-cache"})
    return HTMLResponse("<h1>QualiJournal Admin</h1><p>index.html이 없습니다.</p>",
                        headers={"Cache-Control": "no-store"})


# 디버그: Cloud Run 런타임/리비전/커밋 표시
@app.get("/api/debug/runtime")
def runtime_info(authorized: bool = Depends(authorize)):
    kst = _dt.datetime.utcnow().astimezone(_dt.timezone(_dt.timedelta(hours=9)))
    return JSONResponse({
        "service": os.getenv("K_SERVICE", ""),
        "revision": os.getenv("K_REVISION", ""),
        "commit": os.getenv("COMMIT_SHA", os.getenv("BUILD_ID","")),
        "time_kst": kst.strftime("%Y-%m-%d %H:%M")
    })


_LAST_BACKUP = {"ts":0,"ok":False,"size_md":0,"size_csv":0}

class BackupNotify(BaseModel):
    ok: bool
    ts: int = Field(ge=0)
    size_md: int = Field(default=0, ge=0)
    size_csv: int = Field(default=0, ge=0)


@app.post("/api/backup/notify")
def backup_notify(req: BackupNotify, authorized: bool = Depends(authorize)):
    global _LAST_BACKUP
    # Pydantic v2(model_dump) / v1(dict) 모두 호환
    data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    # 타입 안전화
    data = {
        "ok": bool(data.get("ok")),
        "ts": int(data.get("ts") or 0),
        "size_md": int(data.get("size_md") or 0),
        "size_csv": int(data.get("size_csv") or 0),
    }
    _LAST_BACKUP = data
    return {"ok": True}


@app.get("/api/backup/status")
def backup_status():
    return _LAST_BACKUP

# 디버그: Cloud Run에서 실제로 어떤 HTML 파일을 서빙하는지 확인
@app.get("/api/debug/html_info")
def html_info(authorized: bool = Depends(authorize)):
    from hashlib import md5
    info = {}
    p = INDEX_HTML if INDEX_HTML.exists() else (INDEX_LITE if INDEX_LITE.exists() else None)
    if p and p.exists():
        data = p.read_bytes()
        info = {
            "path": str(p),
            "exists": True,
            "size": len(data),
            "md5": md5(data).hexdigest(),
            "snippet": data[:120].decode("utf-8", errors="ignore")
        }
    else:
        info = {"path": None, "exists": False}
    return JSONResponse({
        "base": str(BASE),
        "k_service": os.getenv("K_SERVICE", ""),
        "html": info
    })

# ---------------------------------------------------------------------------
# Helpers

# === UI 연동용 응답 유틸(멱등) ===
def _now_ms() -> int:
    return int(time.time() * 1000)

def _ok(op: str, **kw):
    return JSONResponse(
        {"ok": True, "op": op, "ts": int(time.time()), **kw},
        headers={"Cache-Control": "no-store"},
    )

def _err(op: str, msg: str, **kw):
    return JSONResponse(
        {"ok": False, "op": op, "error": str(msg), "ts": int(time.time()), **kw},
        headers={"Cache-Control": "no-store"},
    )

# ---------------------------------------------------------------------------
def _task_log_dir() -> Path:
    d = (LOGS_DIR / "tasks")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        try:
            return json.loads(p.read_bytes().decode("utf-8", errors="ignore").lstrip("\ufeff"))
        except Exception:
            return {}

def _write_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def _slug_kw(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (s or "")).strip("-").upper()

def _ensure_id(item: dict) -> str:
    """Ensure a stable unique id in article dict."""
    if not isinstance(item, dict):
        return ""
    if item.get("id"):
        return str(item["id"])
    base = (item.get("url") or item.get("link") or item.get("title") or "").strip()
    if not base:
        base = json.dumps(item, ensure_ascii=False, sort_keys=True)
    h = hashlib.md5(base.encode("utf-8", "ignore")).hexdigest()
    item["id"] = h
    return h

def _load_cfg() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_cfg(obj: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _generate_summary_md(date: str, keyword: str, articles: List[dict], *, selected: bool = False) -> Path:
    """Create a Markdown summary file under ENRICHED_DIR and return its path."""
    slug = _slug_kw(keyword or "").strip()
    suffix = "selected" if selected else "all"
    fname = f"{date}_{slug}_{suffix}.md" if slug else f"{date}_{suffix}.md"
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ENRICHED_DIR / fname

    header_kw = slug.replace("_", " ") if slug else ""
    header_suffix = "선정본" if selected else "전체"
    title_parts = ["QualiNews", date]
    if header_kw:
        title_parts.append(header_kw)
    title_parts.append(f"({header_suffix})")
    lines: List[str] = [" — ".join(title_parts), ""]

    for i, art in enumerate(articles, 1):
        title = art.get("title") or art.get("headline") or "(no title)"
        url   = art.get("url") or art.get("link") or ""
        summary = art.get("summary") or art.get("ko_summary") or art.get("desc") or ""
        note  = art.get("editor_note") or ""
        lines.append(f"### {i}. {title}")
        if url:
            lines.append(f"- 원문: {url}")
        if summary:
            lines.append(f"- 요약: {summary}")
        if note:
            lines.append(f"- 편집자 코멘트: {note}")
        lines.append("")
    md = "\n".join(lines)
    out_path.write_text(md, encoding="utf-8")
    return out_path

def _run_orch(*args: str) -> dict:
    """Run orchestrator.py with UTF-8 safety; return {'ok', 'stdout', 'stderr', 'cmd'}."""
    py  = PYEXE
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # find script
    script = None
    if ORCH.exists():
        script = ORCH
    elif (TOOLS / "orchestrator.py").exists():
        script = TOOLS / "orchestrator.py"
    elif (ROOT / "orchestrator.py").exists():
        script = ROOT / "orchestrator.py"

    if not script:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "orchestrator.py not found in image. (빌드 컨텍스트를 repo 루트로 잡았는지 확인)",
            "cmd": f"{py} orchestrator.py {' '.join(args)}"
        }

    cp = subprocess.run(
        [py, str(script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    try:
        logger.info("run orch: %s rc=%s", " ".join([str(script), *args]), cp.returncode)
        if cp.stderr:
            logger.warning("orch stderr: %s", cp.stderr.strip().replace("\n", " ")[:200])
    except Exception:
        pass
    return {"ok": cp.returncode == 0, "stdout": cp.stdout, "stderr": cp.stderr,
            "cmd": " ".join([str(script), *args])}

def _run_py(script_name: str, args: List[str] | None = None):
    """Run tools/*.py or project-root scripts with UTF-8 safety."""
    candidates = [TOOLS / script_name, ROOT / script_name]
    target = next((p for p in candidates if p.exists()), None)
    if not target:
        return 127, "", f"{script_name} not found"
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cp = subprocess.run(
        [PYEXE, str(target), *(args or [])],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    return cp.returncode, cp.stdout, cp.stderr

def _get_community_snapshot() -> dict:
    obj = {}
    arts: list[dict] = []
    for p in CAND_COMM:
        if p.exists():
            obj = _read_json(p)
            arts = obj.get("articles", [])
            if arts:
                break
    if not arts:
        obj = _read_json(SEL_WORK)
        arts = [a for a in obj.get("articles", []) if a.get("type") == "community"] or obj.get("articles", [])
    for a in arts:
        _ensure_id(a); a.setdefault("approved", False); a.setdefault("editor_note", "")
        a.setdefault("score", a.get("score", 0)); a.setdefault("source", a.get("source", ""))
    date = obj.get("date") or _dt.date.today().isoformat()
    keyword = obj.get("keyword", "")
    return {"date": date, "keyword": keyword, "articles": arts}

def _get_work_snapshot() -> dict:
    """Load selected_keyword_articles.json as work snapshot."""
    # (변경) 여러 후보 경로에서 최초로 발견되는 것을 사용
    obj = {}
    for p in CAND_WORK:
        if p.exists():
            obj = _read_json(p) or {}
            # 허용 구조: {"articles":[...]}, {"items":[...]}, [ ... ]
            if (isinstance(obj, dict) and (obj.get("articles") or obj.get("items"))) or isinstance(obj, list):
                break

    # articles 추출 (dict/items/list 호환)
    if isinstance(obj, dict):
        arts = obj.get("articles") or obj.get("items") or []
        date = obj.get("date") or _dt.date.today().isoformat()
        keyword = obj.get("keyword", "")
    elif isinstance(obj, list):
        arts = obj
        date = _dt.date.today().isoformat()
        keyword = ""
    else:
        arts = []
        date = _dt.date.today().isoformat()
        keyword = ""

    for a in arts:
        _ensure_id(a)
        a.setdefault("approved", False)
        a.setdefault("editor_note", "")
        a.setdefault("state", (a.get("state") or "").lower() or "candidate")

    return {"date": date, "keyword": keyword, "articles": arts}


def _sync_after_save() -> dict:
    """Sync selected -> publish (tools/sync_selected_for_publish.py if exists; else safe merge)."""
    py  = PYEXE
    script = TOOLS / "sync_selected_for_publish.py"
    if script.exists():
        try:
            env = os.environ.copy(); env.setdefault("PYTHONIOENCODING", "utf-8")
            cp = subprocess.run([py, str(script)], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, env=env)
            return {"ok": cp.returncode == 0, "stdout": cp.stdout, "stderr": cp.stderr}
        except Exception as e:
            return {"ok": False, "stderr": str(e)}
    # fallback merge (work -> publish only approved)
    work = {}
    for p in CAND_WORK:
        if p.exists():
            work = _read_json(p) or {}
            if work: break

    pub  = _read_json(SEL_PUB)  or {}
    merged: Dict[str, dict] = {}
    for art in pub.get("articles", []) or []:
        try: _ensure_id(art); merged[art["id"]] = art
        except Exception: continue
    for art in work.get("articles", []) or []:
        if not art.get("approved"): continue
        try: _ensure_id(art); aid = art["id"]
        except Exception: continue
        if aid in merged:
            ex = merged[aid]
            ex["approved"] = art.get("approved", ex.get("approved"))
            ex["editor_note"] = art.get("editor_note", ex.get("editor_note", ""))
            if art.get("pinned") is not None: ex["pinned"] = art.get("pinned")
            if art.get("pin_ts"): ex["pin_ts"] = art.get("pin_ts")
            if art.get("selected") is not None: ex["selected"] = art.get("selected")
        else:
            merged[aid] = art
    date_val = work.get("date") or pub.get("date") or _dt.date.today().isoformat()
    out = {"date": date_val, "articles": list(merged.values())}
    _write_json(SEL_PUB, out)
    return {"ok": True, "stdout": "fallback merge ok"}

def _rollover_archive_if_needed(keyword: str) -> Optional[List[str]]:
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    created = []; ARCHIVE.mkdir(parents=True, exist_ok=True)
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE / f"{base}{ext}"
        if p.exists():
            ts = _dt.datetime.now().strftime("%H%M")
            newp = ARCHIVE / f"{base}_{ts}{ext}"
            p.rename(newp); created.append(str(newp))
    return created or None

def _latest_published_paths(keyword: str) -> List[str]:
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    out = []
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE / f"{base}{ext}"
        if p.exists(): out.append(str(p))
    return out

def _read_any_items(root: Path):
    """
    Priority:
      1) data/selected_keyword_articles.json  {date, keyword, items:[...]}
      2) selected_articles.json               [ ... ]
      3) data_selected_articles.json          [ ... ]
    """
    cand = [
        root / "data" / "selected_keyword_articles.json",
        root / "selected_articles.json",
        root / "data_selected_articles.json",
    ]
    for p in cand:
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and "items" in obj:
                    return obj.get("items") or [], obj.get("date"), obj.get("keyword")
                if isinstance(obj, list):
                    return obj, None, None
            except Exception:
                pass
    return [], None, None


# ---------------------------------------------------------------------------
# Standards Reviews storage helpers (standard_reviews.json)
# ---------------------------------------------------------------------------

def _reviews_file() -> Path:
    """Return path to standard_reviews.json under logs directory (SSOT).

    Cloud Run:
        - LOGS_DIR 가 설정되어 있으면: LOGS_DIR (보통 /tmp/logs)
        - LOGS_DIR 가 없으면: /tmp/logs 로 강제
    Local:
        - LOGS_DIR 가 설정되어 있으면: LOGS_DIR
        - LOGS_DIR 가 없으면: ROOT / "logs"
    """
    # 1) LOGS_DIR 우선 사용
    if LOGS_DIR is not None:
        base = LOGS_DIR
    else:
        # 2) Cloud Run 이면 무조건 /tmp/logs
        if IS_CLOUD:
            base = Path("/tmp/logs")
        else:
            # 3) 로컬에서는 프로젝트 루트/logs
            base = ROOT / "logs"

    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        # 디렉터리 생성 실패가 나더라도 앱이 부팅은 되도록 방어
        pass

    return base / "standard_reviews.json"



def _load_reviews() -> list[dict]:
    """Load review tasks list from standard_reviews.json.

    허용 구조:
    - [ {...}, {...} ]
    - {"items": [ ... ], ...}
    - {"review_tasks": [ ... ], ...}
    그 외 형식/파싱 실패 시 빈 리스트 반환.
    """
    path = _reviews_file()
    obj = _read_json(path)
    items: list[dict] = []
    if isinstance(obj, list):
        items = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        raw = obj.get("items") or obj.get("review_tasks") or []
        if isinstance(raw, list):
            items = [x for x in raw if isinstance(x, dict)]
    return items


def _save_reviews(items: list[dict]) -> None:
    """Save review tasks list to standard_reviews.json with metadata."""
    path = _reviews_file()
    payload = {
        "items": items,
        "updated_at": int(time.time()),
    }
    _write_json(path, payload)


def _normalize_review_task(task: dict, standard_id: str | None = None) -> dict:
    """Ensure required fields for a review task with safe defaults."""
    t: dict[str, Any] = dict(task or {})
    if standard_id:
        t.setdefault("standard_id", standard_id)
    # decision / status
    t["decision"] = (t.get("decision") or "HOLD").upper()
    t["status"] = (t.get("status") or "HOLD").upper()
    # required_reviewers
    try:
        req = int(t.get("required_reviewers", 2))
        if req <= 0:
            req = 2
    except Exception:
        req = 2
    t["required_reviewers"] = req
    # approved_by list
    ab = t.get("approved_by")
    if isinstance(ab, list):
        lst = [str(x) for x in ab]
    elif ab is None:
        lst = []
    else:
        lst = [str(ab)]
    t["approved_by"] = lst
    # reason_short
    t["reason_short"] = t.get("reason_short") or ""
    # log list
    log = t.get("log") or []
    if isinstance(log, list):
        t["log"] = log
    else:
        t["log"] = [log]
    return t


def _find_review_index(items: list[dict], standard_id: str) -> int | None:
    """Find index of review task by standard_id in items list."""
    for idx, it in enumerate(items):
        if str(it.get("standard_id")) == str(standard_id):
            return idx
    return None


# ---------------------------------------------------------------------------
# In-memory Async Task Manager (+SSE)
# ---------------------------------------------------------------------------
class Task:
    def __init__(self, kind: str, args: list[str]):
        self.id = secrets.token_hex(8)
        self.kind = kind
        self.args = args
        self.status = "pending"          # pending|running|done|error|canceled
        self.created_at = time.time()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.exit_code: int | None = None
        self.logs: list[str] = []
        self._cancel = False
        self._lock = threading.Lock()
        # persistent log file path
        # Use LOGS_DIR/tasks when available
        self.log_file: Path | None = None
        try:
            if LOGS_DIR:
                task_dir = LOGS_DIR / "tasks"
                task_dir.mkdir(parents=True, exist_ok=True)
                p = task_dir / f"{self.id}.log"
                p.write_text("", encoding="utf-8")
                self.log_file = p
        except Exception:
            self.log_file = None


    def append(self, line: str):
        with self._lock:
            ts = _dt.datetime.now().strftime("%H:%M:%S")
            msg = f"[{ts}] {line}"
            self.logs.append(msg)
            # persist to disk
            try:
                if self.log_file:
                    with self.log_file.open("a", encoding="utf-8") as fp:
                        write_line = msg + "\n"
                        fp.write(write_line)
            except Exception:
                pass

class TaskManager:
    def __init__(self, keep=50):
        self.keep = keep
        self.jobs: dict[str, Task] = {}
        self._lock = threading.Lock()

    def add(self, t: Task):
        with self._lock:
            self.jobs[t.id] = t
            if len(self.jobs) > self.keep:
                for jid in sorted(self.jobs.keys())[:-self.keep]:
                    self.jobs.pop(jid, None)

    def get(self, jid: str) -> Task | None:
        return self.jobs.get(jid)

TM = TaskManager()

def _run_task(task: Task):
    task.status = "running"; task.started_at = time.time()

    def run_cmd(cmd: list[str]) -> int:
        """
        Run a subprocess command and stream its output into the task log.
        If task._cancel is set, kill the process and return a non-zero
        exit code. This helper ensures long‑running commands can be
        interrupted cleanly.
        """
        task.append(f"$ {' '.join(cmd)}")
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        while True:
            # Check for cancel during execution
            if task._cancel:
                # Kill the subprocess and mark cancellation
                try:
                    p.kill()
                except Exception:
                    pass
                task.append("! canceled")
                # return a non-zero code to indicate failure so that later
                # logic can detect cancellation
                return 1
            # Read line by line until the process finishes
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            if line:
                task.append(line.rstrip())
        # Return the subprocess's exit code, defaulting to 0 if None
        return p.returncode if p.returncode is not None else 0

    try:
        py = PYEXE
        rc = 0
        # Run commands based on the task kind. After each command,
        # check whether the task has been canceled; if so, skip
        # remaining steps.
        if task.kind == "daily":
            # 1. collect community
            rc = run_cmd([py, str(ORCH), "--collect-community"])
            # If canceled, skip remaining steps
            if not task._cancel:
                # 2. publish community
                rc2 = run_cmd([py, str(ORCH), "--publish-community", "--format", "all"])
                rc = max(rc, rc2)
            if not task._cancel:
                # 3. publish all
                rc3 = run_cmd([py, str(ORCH), "--publish", "--format", "all"])
                rc = max(rc, rc3)
        elif task.kind == "community":
            # 1. collect community
            rc = run_cmd([py, str(ORCH), "--collect-community"])
            if not task._cancel:
                # 2. publish community
                rc2 = run_cmd([py, str(ORCH), "--publish-community", "--format", "all"])
                rc = max(rc, rc2)
        elif task.kind == "keyword":
            kw = task.args[0] if task.args else ""
            ext = task.args[1] if len(task.args) > 1 else ""
            if not kw:
                raise RuntimeError("keyword required")
            # collect keyword (with optional external rss flag)
            if str(ext).strip() == "--use-external-rss":
                rc = run_cmd([py, str(ORCH), "--collect-keyword", kw, "--use-external-rss"])
            else:
                rc = run_cmd([py, str(ORCH), "--collect-keyword", kw])
            # approve keyword top 15
            if not task._cancel:
                rc2 = run_cmd([py, str(ORCH), "--approve-keyword", kw, "--approve-keyword-top", "15"])
                rc = max(rc, rc2)
            # publish keyword
            if not task._cancel:
                rc3 = run_cmd([py, str(ORCH), "--publish-keyword", kw])
                rc = max(rc, rc3)
        else:
            raise RuntimeError(f"unknown kind: {task.kind}")

        # Set exit code and status. If the task was canceled, reflect that
        # in the status so the SSE stream can signal cancellation.
        task.exit_code = rc
        if task._cancel:
            # Mark canceled; set a non-zero exit code if none
            task.status = "canceled"
            # Optionally set exit_code for cancellation to a distinct value
            if task.exit_code in (0, None):
                task.exit_code = 130
        else:
            task.status = "done" if rc == 0 else "error"
            if rc != 0:
                task.append(f"! exit={rc}")
    except Exception as e:
        task.status = "error"
        task.append(f"! error: {e}")
    finally:
        task.ended_at = time.time()

# ---------------------------------------------------------------------------
# Tasks API (protected)
# ---------------------------------------------------------------------------
@app.post("/api/tasks/flow")
def create_flow(req: FlowReq, authorized: bool = Depends(authorize)):
    kind = (req.kind or "").lower().strip()
    # Build argument list for Task; for keyword flows, include the keyword and optional external RSS flag
    if kind == "keyword":
        # Ensure keyword is non-null; add external RSS flag when requested
        if req.use_external_rss:
            args = [req.keyword or "", "--use-external-rss"]
        else:
            args = [req.keyword or ""]
    else:
        args = []
    t = Task(kind, args)
    TM.add(t)
    th = threading.Thread(target=_run_task, args=(t,), daemon=True)
    th.start()
    return {"job_id": t.id, "status": t.status, "kind": t.kind, "args": t.args}

# Register recent BEFORE parameterized route to avoid conflicts
@app.get("/api/tasks/recent", response_model=TasksRecent)
def tasks_recent(limit: int = Query(10, ge=1, le=50), authorized: bool = Depends(authorize)) -> TasksRecent:
    try:
        d = _task_log_dir()
        files = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = [TaskItem(id=p.stem, size=p.stat().st_size) for p in files[:limit]]
        return TasksRecent(items=items)
    except Exception:
        return TasksRecent(items=[])

@app.get("/api/tasks/{job_id}")
def get_task(job_id: str, authorized: bool = Depends(authorize)):
    t = TM.get(job_id)
    if not t:
        raise HTTPException(404, "job not found")
    return {
        "id": t.id,
        "kind": t.kind,
        "status": t.status,
        "created_at": t.created_at,
        "started_at": t.started_at,
        "ended_at": t.ended_at,
        "exit_code": t.exit_code,
        "lines": len(t.logs),
        "log_file": str(getattr(t, "log_file", "")) if getattr(t, "log_file", None) else None,
    }

@app.post("/api/tasks/{job_id}/cancel")
def cancel_task(job_id: str, authorized: bool = Depends(authorize)):
    t = TM.get(job_id)
    if not t:
        raise HTTPException(404, "job not found")
    # Mark the task as canceled and log the request. SSE will pick up
    # the cancellation when status changes.
    t._cancel = True
    try:
        t.append("! cancel requested by user")
    except Exception:
        pass
    return {"ok": True}

@app.get("/api/tasks/{job_id}/stream")
async def stream_task(job_id: str, request: Request):
    _auth_header_or_qs_ok(request)  # 헤더 또는 ?token= 허용

    t = TM.get(job_id)
    if not t:
        raise HTTPException(404, "job not found")

    async def _gen():
        idx = 0
        while True:
            if idx < len(t.logs):
                chunk = "\n".join(t.logs[idx:])
                idx = len(t.logs)
                yield f"data: {chunk}\n\n"
            if t.status in ("done", "error", "canceled"):
                yield f"event: end\ndata: {t.status}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Report / Enrich / Export — protected
# ---------------------------------------------------------------------------
@app.get("/api/report")
def get_report(date: str | None = None, authorized: bool = Depends(authorize)):
    day = date or _dt.date.today().isoformat()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    items = []
    for p in ARCHIVE.glob(f"{day}*.*"):
        items.append({"name": p.name, "size": p.stat().st_size})
    for p in ARCHIVE.glob(f"community_{day}.*"):
        items.append({"name": p.name, "size": p.stat().st_size})
    for p in ARCHIVE.glob(f"daily_{day}.*"):
        items.append({"name": p.name, "size": p.stat().st_size})
    return {"date": day, "files": items}


@app.patch("/api/config/gate_required")
async def set_gate_required(p: GatePatch, authorized: bool = Depends(authorize)):
    v = max(1, min(100, int(p.gate_required)))  # 1~100 clamp
    if _DB_READY:
        try:
            snap = _get_work_snapshot() or {}
            edate = snap.get("date") or _dt.date.today().isoformat()
            ekw   = (snap.get("keyword") or "").strip()
            sess = get_session()
            ed = get_or_create_edition(sess, etype="keyword",
                                       edate=_dt.date.fromisoformat(str(edate)),
                                       keyword=ekw or None)
            ed.gate_required = v
            sess.commit()
        except Exception:
            GATE["gate_required"] = v
    else:
        GATE["gate_required"] = v
    return {"ok": True, "gate_required": v}

@app.post("/api/report")
def post_report(payload: dict | None = Body(default=None), authorized: bool = Depends(authorize)):
    """
    UI 스피너/토스트 연동용 리포트 생성 엔드포인트.
    - 정상(S1/S2/S3): ok=True, HTTP 200
      · S1: 기사 N건
      · S2: JSON은 있으나 items=[] (기사 0건)
      · S3: 입력 JSON 파일 자체가 없음 (NO_SOURCE)
    - 에러(E1/E2): ok=False, HTTP 200 + error_code 포함
      · E1: 소스 JSON 파싱 실패 REPORT_SOURCE_INVALID
      · E2: 파일 쓰기 등 내부 오류 REPORT_WRITE_FAILED
    """
    t0 = _now_ms()
    try:
        reports_dir = REPORT_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)

        date = (payload or {}).get("date") or _dt.date.today().isoformat()

        candidates = [
            BASE / "data" / "selected_keyword_articles.json",  # items array
            ROOT / "selected_articles.json",                   # list
            BASE / "data_selected_articles.json",              # list
        ]
        items: list[dict] = []
        keyword = "report"
        source_found = False   # 후보 JSON이 하나라도 있었는지
        source_error = False   # JSON 파싱 에러가 났는지

        def _slug(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "")).strip("_") or "report"

        # S1/S2/E1 판정용: 후보 JSON 탐색
        for p in candidates:
            if p.exists():
                source_found = True
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(obj, dict) and "items" in obj:
                        items = obj.get("items") or []
                        keyword = _slug(obj.get("keyword") or keyword)
                    elif isinstance(obj, list):
                        items = obj
                    break
                except Exception as e:
                    # 소스 JSON은 있는데 파싱 실패(E1)
                    source_error = True
                    try:
                        logger.warning("REPORT_SOURCE_INVALID: %s (%s)", p, e)
                    except Exception:
                        pass
                    continue

        # S3: 후보 JSON이 전혀 없는 경우 → 빈 리포트지만 정상 처리 (ok=True)
        if not source_found:
            try:
                logger.warning(
                    "REPORT_NO_SOURCE_JSON: %s",
                    ", ".join(str(p) for p in candidates),
                )
            except Exception:
                pass
        # E1: JSON은 있었지만 모두 파싱 실패 → 즉시 에러 응답(ok=False)
        elif source_error and not items:
            return _err(
                "report",
                "REPORT_SOURCE_INVALID",
                error_code="REPORT_SOURCE_INVALID",
                duration_ms=_now_ms() - t0,
            )

        def _esc(s: Any) -> str:
            return re.sub(r"[\r\n]+", " ", str(s or "")).strip()

        lines = [f"# {date} · {keyword.upper()} · Daily Report", ""]
        if items:
            for i, it in enumerate(items, 1):
                t = _esc(it.get("title") or it.get("headline") or "(제목 없음)")
                u = _esc(it.get("url") or it.get("link") or "")
                se = _esc(it.get("summary_en") or it.get("summary") or "")
                sk = _esc(it.get("summary_ko") or it.get("summary_kr") or "")
                note = _esc(it.get("editor_note") or "")
                lines.append(f"## {i}. {t}")
                if u:
                    lines.append(f"- 원문: {u}")
                if se:
                    lines.append(f"- 요약(EN): {se}")
                if sk:
                    lines.append(f"- 요약(KO): {sk}")
                if note:
                    lines.append(f"- 코멘트: {note}")
                lines.append("")
        else:
            # S2/S3: 기사 0건 또는 소스 JSON 없음 → 빈 리포트 문구만 출력
            lines += ["(수집된 기사 없음)", ""]

        out = reports_dir / f"{date}_{keyword}_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")

        # 웹에서 접근할 상대 경로 (로컬 /archive 정적 서빙 + GCS object path 공통 사용)
        rel = f"archive/reports/{out.name}"

        # GCS_BUCKET 이 설정되어 있으면 best-effort 로 GCS에도 업로드
        try:
            _upload_report_to_gcs(out, rel)
        except Exception:
            # 업로드 중 문제는 /api/report 응답에 영향을 주지 않도록 삼킨다.
            pass

        return _ok("report", path=rel, count=len(items), duration_ms=_now_ms() - t0)

    except Exception as e:
        # E2: 파일 쓰기 등 내부 오류 → ok=False + error_code
        return _err(
            "report",
            str(e),
            error_code="REPORT_WRITE_FAILED",
            duration_ms=_now_ms() - t0,
        )


@app.post("/api/enrich/keyword")
def enrich_keyword(req: EnrichReq | None = Body(default=None), authorized: bool = Depends(authorize)):
    t0 = _now_ms()
    req = req or EnrichReq()   # ← 본문 없을 때 기본값
    try:
        items, date, kw_in = _read_any_items(BASE)
        date = req.date or date or _dt.date.today().isoformat()
        kw   = (req.keyword or kw_in or "report")
        arts = items or []
        out_path = _generate_summary_md(date, kw, arts, selected=False)
        web_path = f"archive/enriched/{out_path.name}"
        return _ok("enrich_keyword", path=web_path, count=len(arts), duration_ms=_now_ms()-t0)
    except Exception as e:
        return _err("enrich_keyword", str(e), duration_ms=_now_ms()-t0)


@app.post("/api/enrich/selection")
def enrich_selection(req: EnrichReq | None = Body(default=None), authorized: bool = Depends(authorize)):
    t0 = _now_ms()
    req = req or EnrichReq()
    try:
        items, date, kw_in = _read_any_items(BASE)
        date = req.date or date or _dt.date.today().isoformat()
        kw   = (req.keyword or kw_in or "report")

        def _is_selected(a: dict) -> bool:
            s = (a.get("state") or "").lower()
            return bool(a.get("selected") or a.get("approved") or s in ("published", "ready"))

        arts = [a for a in (items or []) if _is_selected(a)]
        out_path = _generate_summary_md(date, kw, arts, selected=True)
        web_path = f"archive/enriched/{out_path.name}"
        return _ok("enrich_selection", path=web_path, count=len(arts), duration_ms=_now_ms()-t0)
    except Exception as e:
        return _err("enrich_selection", str(e), duration_ms=_now_ms()-t0)


@app.get("/api/export/{fmt}")
def export_fmt(fmt: str, date: str | None = None, preview: bool = Query(False), authorized: bool = Depends(authorize)):
    """
    Export final selection or community articles.
    If fmt is 'md' or 'csv', export selected_articles.json as Markdown or CSV.
    Otherwise, fallback to community export based on archive/community_date.json.
    """
    day = date or _dt.date.today().isoformat()
    fmt_lower = fmt.lower()

    # final selection export
    if fmt_lower in ("md", "csv"):
        data = _read_json(SEL_PUB)
        articles = data.get("articles", []) or []
        kw = data.get("keyword", "") or ""
        if fmt_lower == "md":
            lines = [f"# QualiNews — {day} — {kw}", ""]
            for i, a in enumerate(articles, 1):
                title = a.get("title", "(no title)")
                url   = a.get("url") or a.get("link") or ""
                summ  = a.get("summary") or a.get("ko_summary") or a.get("desc") or ""
                note  = a.get("editor_note") or ""
                lines.append(f"### {i}. {title}")
                if url:  lines.append(f"- 원문: {url}")
                if summ: lines.append(f"- 요약: {summ}")
                if note: lines.append(f"- 편집자 코멘트: {note}")
                lines.append("")
            md = "\n".join(lines)
            fname = f"quali_{day}_{_slug_kw(kw)}.md"
            headers = {} if preview else {"Content-Disposition": f'attachment; filename="{fname}"'}
            return Response(content=md, media_type="text/markdown; charset=utf-8", headers=headers)
        elif fmt_lower == "csv":
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["title","url","source","date","score","approved","editor_note","summary"])
            for a in articles:
                w.writerow([
                    a.get("title",""), a.get("url") or a.get("link") or "",
                    a.get("source",""), a.get("date",""),
                    a.get("score",""), a.get("approved",""),
                    a.get("editor_note",""), a.get("summary") or a.get("ko_summary") or a.get("desc","")
                ])
            csv_text = buf.getvalue()
            data_out  = "\ufeff" + csv_text  # add BOM for Excel
            fname = f"quali_{day}_{_slug_kw(kw)}.csv"
            headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
            return Response(content=data_out, media_type="text/csv; charset=utf-8", headers=headers)

    # fallback community export
    cj = ARCHIVE / f"community_{day}.json"
    if not cj.exists():
        raise HTTPException(404, "community json not found")
    obj = _read_json(cj); arts = obj.get("articles", [])
    if fmt_lower == "md":
        lines = [f"# Community — {day}", ""]
        for a in arts:
            title = a.get("title") or "(no title)"
            url = a.get("url") or "#"
            meta = f"👍{a.get('upvotes',0)} · 💬{a.get('comments',0)} · 👀{a.get('views','-')}"
            lines.append(f"- [{title}]({url})  \n  {meta} · {a.get('source','')}")
        md = "\n".join(lines) + "\n"
        return Response(content=md, media_type="text/markdown; charset=utf-8")
    if fmt_lower == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["title","url","source","upvotes","comments","views"])
        for a in arts:
            w.writerow([a.get("title",""), a.get("url",""), a.get("source",""),
                        a.get("upvotes",0), a.get("comments",0), a.get("views","")])
        data = "\ufeff" + buf.getvalue()  # UTF-8 BOM
        return Response(content=data, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="community_{day}.csv"'})
    raise HTTPException(400, "unsupported format")

# --- explicit aliases for stability ---
@app.get("/api/export/md")
def export_md(preview: bool = Query(False), authorized: bool = Depends(authorize)):
    return export_fmt(fmt="md", preview=preview, authorized=authorized)

# (선택) 레거시/짧은 경로도 허용
@app.get("/export/md")
def export_md_alias(preview: bool = Query(False), authorized: bool = Depends(authorize)):
    return export_fmt(fmt="md", preview=preview, authorized=authorized)

@app.get("/export/csv")
def export_csv_alias(authorized: bool = Depends(authorize)):
    return export_fmt(fmt="csv", authorized=authorized)

# ---------------------------------------------------------------------------
# Tools runner (repair / approve_top) — protected
# ---------------------------------------------------------------------------

def _tools_ok(rc: int, sync_log: dict | None, stderr: str | None) -> bool:
    """
    Quick Tools B안 성공 판정 규칙:
    - 기본: rc == 0 → ok = True
    - B안: rc == 127 이면서 sync_log.ok == True 이고 stderr에 'not found' 포함 → ok = True
    - 그 외 → ok = False
    """
    sync_ok = bool((sync_log or {}).get("ok"))
    stderr_text = (stderr or "").lower()

    if rc == 0:
        return True

    if rc == 127 and sync_ok and "not found" in stderr_text:
        return True

    return False


@app.post("/api/tools/repair")
def api_tools_repair(authorized: bool = Depends(authorize)):
    """
    repair_selection_files.py 실행:
    - 작업본/발행본 JSON 구조 교정 + 승인본 재작성
    - 실행 로그와 성공 여부를 반환
    """
    rc, out, err = _run_py("repair_selection_files.py", [])
    # 실행 후 발행본 싱크 보강(있으면 내부 스크립트가 수행하지만, 안전하게 한 번 더)
    sync = _sync_after_save()
    ok = _tools_ok(rc, sync, err)

    return {
        "ok": ok,
        "rc": rc,
        "stdout": out.strip(),
        "stderr": err.strip(),
        "synced": bool(sync.get("ok")),
        "sync_log": sync,
    }

@app.post("/api/tools/approve_top")
def api_tools_approve_top(
    n: int = Query(20, ge=1, le=100),
    authorized: bool = Depends(authorize),
):
    """
    force_approve_top20.py 실행:
    - 상위 n개 승인(approved=True) 처리
    - 처리 후 발행본을 동기화
    """
    # (선택) DB 기반 TopN 승인  ← 들여쓰기 바로잡음
    if _DB_READY:
        try:
            snap = _get_work_snapshot() or {}
            edate = snap.get("date") or _dt.date.today().isoformat()
            ekw = (snap.get("keyword") or "").strip()

            sess = get_session()
            ed = get_or_create_edition(
                sess,
                etype="keyword",
                edate=_dt.date.fromisoformat(str(edate)),
                keyword=ekw or None,
            )
            approve_top_n(sess, ed, n=n)
            sess.commit()
        except Exception:
            # DB 경로 에러는 무시하고, 아래 스크립트 경로로 폴백
            pass

    rc, out, err = _run_py("force_approve_top20.py", ["--top", str(n)])

    # 스크립트 내부에서 싱크하더라도, 최종 한번 더 보정
    sync = _sync_after_save()
    ok = _tools_ok(rc, sync, err)


    return {
        "ok": ok,
        "rc": rc,
        "stdout": out.strip(),
        "stderr": err.strip(),
        "synced": bool(sync.get("ok")),
        "sync_log": sync,
        "top": n,
    }


# ---------------------------------------------------------------------------
# Standards Reviews API — protected
# ---------------------------------------------------------------------------

@app.get("/api/standards/reviews")
def api_standards_reviews_list(
    status: str | None = Query(None),
    decision: str | None = Query(None),
    authorized: bool = Depends(authorize),
):
    """리뷰 큐 목록 조회.

    - standard_reviews.json 에 저장된 리뷰 태스크 목록을 반환
    - 선택적으로 status/decision 으로 필터 가능
    - 응답은 SSOT에서 요구하는 ok+data(count/items) 구조를 따른다.
    """
    items = [_normalize_review_task(it) for it in _load_reviews()]

    s = (status or "").strip().upper()
    d = (decision or "").strip().upper()

    def _match(it: dict) -> bool:
        if s and str(it.get("status","")).upper() != s:
            return False
        if d and str(it.get("decision","")).upper() != d:
            return False
        return True

    items_filtered = [it for it in items if _match(it)]
    body = {
        "ok": True,
        "name": "standards_reviews_list",
        "data": {
            "count": len(items_filtered),
            "items": items_filtered,
        },
    }
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


@app.post("/api/standards/reviews/{standard_id}/approve")
def api_standards_reviews_approve(
    standard_id: str,
    req: ReviewApproveReq,
    authorized: bool = Depends(authorize),
):
    """2인 검수 승인 플로우.

    - status=HOLD 인 태스크에 대해 reviewer_id 를 approved_by 에 추가
    - approved_by 길이가 required_reviewers(기본 2)에 도달하면 status=REVIEWED
    - REVIEWED/PUBLISHED 인 상태에서 재호출 시에는 상태를 바꾸지 않고 그대로 반환
    """
    if not req.reviewer_id:
        raise HTTPException(status_code=400, detail="reviewer_id required")

    items = [_normalize_review_task(it) for it in _load_reviews()]
    idx = _find_review_index(items, standard_id)
    if idx is None:
        raise HTTPException(status_code=404, detail="review task not found")

    task = items[idx]
    # 멱등성을 위해 동일 reviewer_id 중복 추가는 허용하지 않음
    reviewer = str(req.reviewer_id).strip()
    if reviewer and reviewer not in task["approved_by"]:
        task["approved_by"].append(reviewer)

    # 상태 전이: HOLD -> REVIEWED (필요 시)
    status_now = str(task.get("status", "HOLD")).upper()
    if status_now == "HOLD":
        if len(task["approved_by"]) >= int(task.get("required_reviewers", 2)):
            task["status"] = "REVIEWED"

    # PUBLISHED 인 경우에는 상태를 바꾸지 않고 그대로 반환 (idempotent)
    # REVIEWED 상태에서 추가 승인 요청이 와도 그대로 유지

    # 로그 기록
    task_log = task.get("log") or []
    if not isinstance(task_log, list):
        task_log = [task_log]
    task_log.append({
        "ts": int(time.time()),
        "event": "approve",
        "reviewer_id": reviewer,
    })
    task["log"] = task_log

    # 저장
    items[idx] = task
    _save_reviews(items)

    body = {
        "ok": True,
        "name": "standards_reviews_approve",
        "data": {
            "review_task": task,
        },
    }
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


@app.post("/api/standards/reviews/{standard_id}/publish")
def api_standards_reviews_publish(
    standard_id: str,
    authorized: bool = Depends(authorize),
):
    """REVIEWED → PUBLISHED 발행 플로우.

    - status=REVIEWED 인 태스크만 발행 허용
    - 발행 시 status=PUBLISHED 로 변경
    - 정책 A안: decision 을 PASS 로 승격 (SSOT 초안 기준)
    """
    items = [_normalize_review_task(it) for it in _load_reviews()]
    idx = _find_review_index(items, standard_id)
    if idx is None:
        raise HTTPException(status_code=404, detail="review task not found")

    task = items[idx]
    status_now = str(task.get("status", "HOLD")).upper()
    if status_now not in ("REVIEWED", "PUBLISHED"):
        # 아직 리뷰 미완료(HOLD 등) 상태에서 발행 시도
        raise HTTPException(status_code=409, detail="review task not reviewed")

    # 이미 PUBLISHED 면 멱등하게 그대로 반환
    if status_now == "PUBLISHED":
        body = {
            "ok": True,
            "name": "standards_reviews_publish",
            "data": {
                "review_task": task,
            },
        }
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    # REVIEWED -> PUBLISHED 전이 + decision 승격(A안)
    task["status"] = "PUBLISHED"
    # 정책 A안: 발행된 표준은 PASS 로 간주
    task["decision"] = "PASS"

    # 로그 기록
    task_log = task.get("log") or []
    if not isinstance(task_log, list):
        task_log = [task_log]
    task_log.append({
        "ts": int(time.time()),
        "event": "publish",
    })
    task["log"] = task_log

    items[idx] = task
    _save_reviews(items)

    body = {
        "ok": True,
        "name": "standards_reviews_publish",
        "data": {
            "review_task": task,
        },
    }
    return JSONResponse(body, headers={"Cache-Control": "no-store"})

@app.post("/api/standards/reviews/test/init")
def api_standards_reviews_test_init(
    req: ReviewTestInitReq,
    authorized: bool = Depends(authorize),
):
    """
    테스트용 리뷰 카드 1개를 보장해 주는 Admin 전용 엔드포인트.

    - reset=True  이면: 기존 큐를 모두 비우고 테스트 카드 1개만 남긴다.
    - reset=False 이면: 같은 standard_id 카드가 있으면 그대로 normalize 해서 반환,
                        없으면 새로 생성해서 큐에 추가한다.
    """
    std_id = (req.standard_id or "TEST-STD-1").strip() or "TEST-STD-1"

    # 현재 큐 로드 + normalize
    items = [_normalize_review_task(it) for it in _load_reviews()]

    # reset 옵션 처리: 큐를 깨끗이 비우고 새로 시작
    if req.reset:
        items = []

    # 이미 같은 standard_id 가 있는지 먼저 확인
    for it in items:
        sid = str(it.get("standard_id") or (it.get("item") or {}).get("id") or "")
        if sid == std_id:
            task = _normalize_review_task(it, standard_id=std_id)
            _save_reviews(items)
            return {
                "ok": True,
                "name": "standards_reviews_test_init",
                "data": {
                    "review_task": task,
                    "created": False,
                    "reset": req.reset,
                },
            }

    # 여기까지 왔다는 것은: 해당 ID 카드가 없다는 뜻 → 새로 만든다.
    now = int(time.time())

    standard_key = {
        "name": "QualiJournal-Admin-Standards-API",
        "rev": "v1",
        "date": "2025-11-19",
    }

    item = {
        "id": std_id,
        "title": "[TEST] HOLD→REVIEWED→PUBLISHED 상태머신 검증용 표준",
        "url": "https://example.com/qualijournal/test-standard",
        "standard_key": standard_key,
        # 4축 스코어 예시: PASS 임계점보다 살짝 낮게 잡아서 기본 HOLD 유지
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

    task = {
        "standard_id": std_id,
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

    # normalize 로 필수 필드/기본값 정리
    task = _normalize_review_task(task, standard_id=std_id)
    items.append(task)
    _save_reviews(items)

    return {
        "ok": True,
        "name": "standards_reviews_test_init",
        "data": {
            "review_task": task,
            "created": True,
            "reset": req.reset,
        },
    }

# ---------------------------------------------------------------------------
# Selection Approvals — protected (work JSON <-> publish JSON)
# ---------------------------------------------------------------------------
_SEL_LOCK = threading.Lock()

class SelectionItemPatch(BaseModel):
    idx: Optional[int] = Field(default=None, description="작업본 articles[] 인덱스(권장)")
    id: Optional[str] = Field(default=None, description="기사 고유 id(_ensure_id 결과)")
    approved: Optional[bool] = None
    editor_note: Optional[str] = None

class SelectionPatchRequest(BaseModel):
    updates: List[SelectionItemPatch]
    autosync: bool = True  # 저장 후 selected_articles.json 자동 동기화

@app.get("/api/selection")
def api_selection_list(
    authorized: bool = Depends(authorize),
    keyword: Optional[str] = None,
    date: Optional[str] = None
):
    """
    기사 승인 표용 데이터.
    1) 스냅샷(_get_work_snapshot) 시도
    2) 폴백: 작업본(SEL_WORK) 직접 읽기
    ※ keyword/date 필터가 있어도 'idx'는 항상 '원본 파일 인덱스'를 보존.
    """
    def _match_kw(a: Dict[str, Any], kw: Optional[str]) -> bool:
        # 키워드가 없으면 필터 미적용
        if not kw: 
            return True
        kw = kw.strip().lower()

        # 제목/출처/URL/keyword 필드 모두에서 찾아본다
        bag = " ".join([
            str(a.get("keyword") or a.get("kw") or ""),
            str(a.get("title") or ""),
            str(a.get("source") or a.get("publisher") or ""),
            str(a.get("url") or a.get("link") or "")
        ]).lower()

        # 매칭 재료가 전혀 없으면 '걸러내지 말고 통과' (수집형식 다양성 보호)
        if bag.strip() == "":
            return True
        return (kw in bag)

    def _match_date(a: Dict[str, Any], d: Optional[str]) -> bool:
        if not d: 
            return True
        return d in str(a.get("date") or "")

    # 1) 스냅샷 시도
    try:
        snap = _get_work_snapshot() or {}
    except Exception:
        snap = {}

    arts_full = (snap.get("articles") or [])
    snap_date = snap.get("date")
    snap_kw   = snap.get("keyword") or ""

    # 2) 폴백: 스냅샷이 비면 작업본 JSON 직접 읽기
    if not arts_full:
        work = {}
        for p in CAND_WORK:
            if p.exists():
                work = _read_json(p) or {}
                if work: break

        if isinstance(work, dict):
            arts_full = work.get("articles") or work.get("items") or []
            snap_date = snap_date or work.get("date")
            snap_kw   = snap_kw   or work.get("keyword") or work.get("kw") or ""
        elif isinstance(work, list):
            arts_full = work

    # 원본 인덱스를 보존한 채로 필터링
    items: List[Dict[str, Any]] = []
    approved = 0
    for i, a in enumerate(arts_full or []):
        if not _match_date(a, date): 
            continue
        if not _match_kw(a, keyword):
            continue
        try:
            aid = _ensure_id(a)
        except Exception:
            aid = ""
        ap = bool(a.get("approved") or a.get("selected") or (str(a.get("state","")).lower() in ("ready","published")))
        if ap: 
            approved += 1
        items.append({
            "idx": i,  # ← 원본 파일 인덱스(필터 전 인덱스) 유지
            "id": aid,
            "title": a.get("title"),
            "source": a.get("source") or a.get("publisher"),
            "date": a.get("date"),
            "score": a.get("score"),
            "url": a.get("url") or a.get("link"),
            "approved": ap,
            "editor_note": a.get("editor_note",""),
            "keyword": a.get("keyword") or a.get("kw") or ""
        })

    gate_required = int(GATE.get("gate_required", 15))
    return {
        "items": items,
        "summary": {
            "total": len(items),
            "approved": approved,
            "gate_required": gate_required,
            "gate_pass": bool(approved >= gate_required),
            "date": snap_date,
            "keyword": snap_kw,
        }
    }


@app.patch("/api/selection")
def api_selection_patch(req: SelectionPatchRequest, authorized: bool = Depends(authorize)):
    """
    승인/메모 배치 저장. req.updates = [{idx|id, approved?, editor_note?}, ...]
    - 작업본 저장 후 autosync=True면 발행본(selected_articles.json) 동기화.
    """
    if not req.updates:
        return {"updated": 0, "synced": False}

    with _SEL_LOCK:
        # 읽기는 후보 경로 → 저장은 SEL_WORK(SSOT)
        work = {}
        for p in CAND_WORK:
            if p.exists():
                work = _read_json(p) or {}
                if work: break

        arts = work.get("articles", []) or []
        if not arts:
            raise HTTPException(status_code=400, detail="no articles in work file")

        # id -> index 매핑
        id_to_idx: Dict[str, int] = {}
        for i, a in enumerate(arts):
            try:
                id_to_idx[_ensure_id(a)] = i
            except Exception:
                continue

        changed = 0
        for p in req.updates:
            # 우선순위: idx > id
            target_idx = p.idx
            if target_idx is None and p.id:
                target_idx = id_to_idx.get(p.id)
            if target_idx is None or target_idx < 0 or target_idx >= len(arts):
                continue
            row = arts[target_idx]
            before_approved = bool(row.get("approved"))
            before_note = row.get("editor_note","")

            if p.approved is not None:
                row["approved"] = bool(p.approved)
                if p.approved:
                    # 편리성: 승인 시 state도 ready로 끌어올림(발행 파이프라인과 정합)
                    row["state"] = (row.get("state") or "candidate")
                    if row["state"].lower() == "candidate":
                        row["state"] = "ready"
            if p.editor_note is not None:
                row["editor_note"] = p.editor_note

            if bool(row.get("approved")) != before_approved or row.get("editor_note","") != before_note:
                changed += 1

        if changed:
            work["articles"] = arts
            _write_json(SEL_WORK, work)

        synced = False
        if req.autosync:
            sync = _sync_after_save()
            synced = bool(sync.get("ok"))

        # 현황 집계 반환
        approved_cnt = sum(1 for a in arts if a.get("approved"))
        gate_required = int(GATE.get("gate_required", 15))
        return {
            "updated": changed,
            "synced": synced,
            "approved": approved_cnt,
            "gate_required": gate_required,
            "gate_pass": bool(approved_cnt >= gate_required),
        }

# ---------------------------------------------------------------------------
# Community / Items / Publish — protected
# ---------------------------------------------------------------------------
class SaveItem(BaseModel):
    id: str
    approved: bool
    editor_note: str = ""

class SavePayload(BaseModel):
    changes: List[SaveItem]

@app.get("/api/community")
def api_get_community(only_pending: bool = Query(True), authorized: bool = Depends(authorize)):
    snap = _get_community_snapshot()
    arts = snap["articles"]
    if only_pending:
        arts = [a for a in arts if not a.get("approved")]
    total = len(snap["articles"])
    approved = sum(1 for a in snap["articles"] if a.get("approved"))
    return {"date": snap["date"], "keyword": snap.get("keyword",""), "total": total,
            "approved": approved, "pending": total-approved, "articles": arts}

@app.post("/api/community/save")
def api_save(payload: SavePayload, authorized: bool = Depends(authorize)):
    target = SEL_COMM if SEL_COMM.exists() else SEL_WORK
    obj = _read_json(target); items = obj.get("articles", []); idx = {_ensure_id(a): a for a in items}
    changed = 0
    for c in payload.changes:
        row = idx.get(c.id)
        if not row:
            continue
        if row.get("approved") != c.approved or (row.get("editor_note","") != c.editor_note):
            row["approved"] = c.approved
            row["editor_note"] = c.editor_note
            changed += 1
    if changed:
        _write_json(target, obj)
    sync = _sync_after_save()
    return {"saved": changed, "synced": sync.get("ok", False), "sync_log": sync}

@app.get("/api/items")
def api_items(
    state: str = Query("ready"),
    date: str | None = None,
    keyword: str | None = None,
    authorized: bool = Depends(authorize),
):
    """
    state=candidate|ready|rejected|published
    - ready: selected_articles.json(items/articles) 또는 work 스냅샷을 자동 탐색
    - published: selected_articles.json(articles) 사용
    - candidate/rejected: work 스냅샷(state 필터)
    - date/keyword가 주어지면 해당 필드가 있는 항목만 추가 필터
    """

    # 공통 필터 함수
    def _match(it: dict) -> bool:
        # state는 상단에서 선별하므로 여기선 date/keyword만 보조 확인
        if date and str(it.get("date","")) != str(date):
            return False
        if keyword:  # 대부분 선정본은 keyword가 없으니, 주면 일치하는 것만 남김
            if str(it.get("keyword","")).strip() != str(keyword).strip():
                return False
        return True

    s = (state or "").lower().strip()

    # 1) published는 최종본에서만
    if s == "published":
        pub = _read_json(SEL_PUB) or {}
        items = [a for a in (pub.get("articles", []) or []) if _match(a)]
        return {
            "date": date or pub.get("date"),
            "keyword": keyword or "",
            "state": s,
            "items": items,
        }

    # 2) ready는 범용 로더 → 없으면 work 스냅샷에서 state=ready
    if s == "ready":
        items_any, d_auto, kw_auto = _read_any_items(BASE)  # items/date/keyword 자동 탐색
        src_items = items_any or []
        if not src_items:
            snap = _get_work_snapshot()
            src_items = [a for a in (snap.get("articles", []) or []) if (a.get("state","").lower()=="ready")]
            d_auto = d_auto or snap.get("date")
            kw_auto = kw_auto or snap.get("keyword","")
        items = [a for a in src_items if _match(a)]
        return {
            "date": date or d_auto,
            "keyword": keyword or kw_auto or "",
            "state": s,
            "items": items,
        }

    # 3) 그 외(candidate/rejected/all)는 work 스냅샷 기준
    snap = _get_work_snapshot()
    arts = snap.get("articles", []) or []
    if s in ("candidate","rejected"):
        src_items = [a for a in arts if (a.get("state","").lower()==s)]
    else:
        src_items = arts  # all 또는 빈 state

    items = [a for a in src_items if _match(a)]
    return {
        "date": date or snap.get("date"),
        "keyword": keyword or snap.get("keyword",""),
        "state": s or "all",
        "items": items,
    }


@app.post("/api/items/{item_id}/publish")
def api_items_publish(item_id: str, req: PublishOneReq, authorized: bool = Depends(authorize)):
    """
    Publish single item: update approval and notes in work file then sync to publish file.
    """
    work = _read_json(SEL_WORK) or {}
    arts = work.get("articles", []) or []
    idx = None
    for i, a in enumerate(arts):
        try:
            if _ensure_id(a) == item_id:
                idx = i
                break
        except Exception:
            continue
    if idx is None:
        raise HTTPException(status_code=404, detail="item not found")

    a = arts[idx]
    a["approved"] = bool(req.approve)
    a["selected"] = True
    a["editor_note"] = req.editor_note or a.get("editor_note","")
    a["state"] = "published"

    work["articles"] = arts
    _write_json(SEL_WORK, work)

    sync = _sync_after_save()
    return {"ok": True, "synced": sync.get("ok", False), "item_id": item_id}

@app.post("/api/publish-keyword")
def api_publish(req: PublishReq, authorized: bool = Depends(authorize)):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    rollover = _rollover_archive_if_needed(req.keyword)
    out = _run_orch("--publish-keyword", req.keyword)
    outputs = _latest_published_paths(req.keyword)
    return {**out, "rolled_over": rollover or [], "created": outputs}

# Legacy sync flows (kept for compatibility)
@app.post("/api/flow/community")
def api_flow_comm(authorized: bool = Depends(authorize)):
    return _run_orch("--collect-community")

@app.post("/api/flow/daily")
def api_flow_daily(authorized: bool = Depends(authorize)):
    steps = [
        _run_orch("--collect-community"),
        _run_orch("--publish-community", "--format", "all"),
        _run_orch("--publish", "--format", "all"),
    ]
    ok = all(s.get("ok", True) for s in steps)
    return {"ok": ok, "steps": steps}

@app.post("/api/flow/keyword")
def api_flow_keyword(req: FlowKwReq, authorized: bool = Depends(authorize)):
    steps = []
    if req.use_external_rss:
        steps.append(_run_orch("--collect-keyword", req.keyword, "--use-external-rss"))
    else:
        steps.append(_run_orch("--collect-keyword", req.keyword))
    steps.append(_run_orch("--approve-keyword-top", "20", "--approve-keyword", req.keyword))
    steps.append(_sync_after_save())
    steps.append(_run_orch("--publish-keyword", req.keyword))
    ok = all(s.get("ok", True) for s in steps)
    return {"ok": ok, "steps": steps}

# ---------------------------------------------------------------------------
# Logs (protected + optional JWT)
# ---------------------------------------------------------------------------
@app.get("/api/logs")
def list_logs(authorized: bool = Depends(authorize), user: dict = Depends(verify_jwt_token)):
    logs_dir = (LOGS_DIR or (ROOT / 'logs'))
    items = []
    if logs_dir.exists() and logs_dir.is_dir():
        for p in logs_dir.iterdir():
            if p.is_file() and p.suffix == ".log":
                try:
                    stat = p.stat()
                    items.append({"name": p.name, "size": stat.st_size,
                                  "modified": _dt.datetime.utcfromtimestamp(stat.st_mtime).isoformat()+"Z"})
                except Exception:
                    continue
    return {"items": items}

@app.get("/api/logs/{log_name}")
def get_log(log_name: str, lines: int = 200, authorized: bool = Depends(authorize), user: dict = Depends(verify_jwt_token)):
    path = ((LOGS_DIR or (ROOT / 'logs')) / log_name)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "log not found")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines_data = f.readlines()
        content = "".join(lines_data[-int(lines):]) if (lines and lines > 0) else "".join(lines_data)
    except Exception as e:
        raise HTTPException(500, str(e))
    return PlainTextResponse(content, media_type="text/plain")

@app.get("/api/logs/{log_name}/download")
def download_log(log_name: str, authorized: bool = Depends(authorize), user: dict = Depends(verify_jwt_token)):
    path = ((LOGS_DIR or (ROOT / 'logs')) / log_name)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "log not found")
    return FileResponse(str(path), filename=log_name, media_type="text/plain")

# ---------------------------------------------------------------------------
# KPI status — protected (fixes 'always authenticated' UI issue)
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status(
    date: str | None = None,
    keyword: str | None = None,
    authorized: bool = Depends(authorize),
):
    """
    관리자 KPI/상태 조회 (DB 우선, 실패 시 기존 스냅샷 폴백)
    - DB 사용 가능(_DB_READY)하면 Edition별 집계(kpi_for_edition)
    - 아니면 현 스냅샷 로직(_get_work_snapshot 등) 사용
    """
    # 1) DB 경로(권장)
    if _DB_READY:
        try:
            # date/keyword가 없으면 스냅샷 값으로 폴백
            snap = _get_work_snapshot() or {}
            edate = date or snap.get("date") or _dt.date.today().isoformat()
            ekw   = (keyword or snap.get("keyword") or "").strip()
            sess = get_session()
            ed = get_or_create_edition(sess, etype="keyword", edate=_dt.date.fromisoformat(str(edate)), keyword=ekw or None)
            k = kpi_for_edition(sess, ed)  # {'total','approved','ready','gate_required'}
            return {
                "selected": 0,
                "approved": 0,
                "published": 0,  # 누적 KPI는 추후 확장
                "gate_required": int(k.get("gate_required", 15)),
                "ts": int(time.time()),
                "selection_total": int(k.get("total", 0)),
                "selection_approved": int(k.get("approved", 0)),
                # /api/status 최소 필드 (DB 기반)
                "total": int(k.get("total", 0)),
                "ready_count": int(k.get("ready", 0)),
                "ready_rate": (
                    int(k.get("ready", 0)) / int(k.get("total", 0))
                ) if int(k.get("total", 0)) else 0.0,
                "state_counts": {
                    "candidate": int(k.get("total", 0)) - int(k.get("ready", 0)),
                    "ready": int(k.get("ready", 0)),
                    "rejected": 0,
                },
                "community_total": 0,            # 필요 시 커뮤니티도 DB화해서 합산
                "keyword_total": int(k.get("total", 0)),
                "gate_pass": bool(int(k.get("approved", 0)) >= int(k.get("gate_required", 15))),
                "date": str(edate),
                "keyword": ekw,
            }

        except Exception:
            # 아래 폴백으로 진행
            pass

    # 2) 스냅샷 폴백(현행 로직 유지)
    try:
        work = _get_work_snapshot()            # {"date","keyword","articles":[...]}
        arts = work.get("articles", []) or []

        # 상태별 집계
        state_counts = {"candidate": 0, "ready": 0, "rejected": 0}
        selection_total = len(arts)
        selection_approved = 0
        for a in arts:
            st = (a.get("state") or "candidate").lower()
            if st in state_counts:
                state_counts[st] += 1
            if a.get("approved") or a.get("selected") or st in ("ready", "published"):
                selection_approved += 1

        # 커뮤니티 집계
        comm = _get_community_snapshot()       # {"date","keyword","articles":[...]}
        community_total = len(comm.get("articles", []) or [])

        # 게이트
        gate_required = int(GATE.get("gate_required", 15))
        gate_pass = bool(selection_approved >= gate_required)

        return {
            "selected": KPI.get("selected", 0) if isinstance(KPI, dict) else 0,
            "approved": KPI.get("approved", 0) if isinstance(KPI, dict) else 0,
            "published": KPI.get("published", 0) if isinstance(KPI, dict) else 0,
            "gate_required": gate_required,
            "ts": int(time.time()),
            "selection_total": selection_total,
            "selection_approved": selection_approved,
            # /api/status 최소 필드 (스냅샷 기반)
            "total": selection_total,
            "ready_count": int(state_counts.get("ready", 0)),
            "ready_rate": (
                int(state_counts.get("ready", 0)) / selection_total
            ) if selection_total else 0.0,
            "state_counts": state_counts,
            "community_total": community_total,
            "keyword_total": selection_total,
            "gate_pass": gate_pass,
            "date": date or work.get("date"),
            "keyword": (keyword or work.get("keyword", "")).strip(),
        }

    except Exception:
        # 안전 폴백(절대 500 안 냄)
        return {
            "selected": 0,
            "approved": 0,
            "published": 0,
            "gate_required": 15,
            "ts": int(time.time()),
            "selection_total": 0,
            "selection_approved": 0,
            "total": 0,
            "ready_count": 0,
            "ready_rate": 0.0,
            "state_counts": {"candidate": 0, "ready": 0, "rejected": 0},
            "community_total": 0,
            "keyword_total": 0,
            "gate_pass": False,
            "date": date,
            "keyword": keyword or "",
        }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    _port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=_port, log_level="info")
