# -*- coding: utf-8 -*-
"""
QualiJournal Admin API (stable, cleaned)
- Reason tags + Monthly snapshot + Weekly diff
- Ready(SSOT) helper APIs
- Community / Keyword flows (tasks)
- Report / Export
- UTF-8 safe on Windows
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

from fastapi import FastAPI, Query, Request, Response, HTTPException, Depends, Body, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone, timedelta

# ===== Optional DB helpers (safe fallback) ===================================
try:
    from app.qj_db import get_session, get_or_create_edition, kpi_for_edition, approve_top_n  # type: ignore
    _DB_READY = True
except Exception:
    _DB_READY = False
# ============================================================================

PYEXE = os.getenv("PYTHON_EXE") or sys.executable or "python"

# .env (optional)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

MODE = (os.getenv("QUALI_DB_MODE") or "local").lower().strip()

# admin.db (optional). If unavailable, db endpoints gracefully degrade.
try:
    from admin.db import make_engine  # type: ignore
    _engine = None
except Exception:  # pragma: no cover
    make_engine = None  # type: ignore
    _engine = None

# === FastAPI app ==============================================================
app = FastAPI(title="QualiJournal Admin API")

# ==== Strong ETag Middleware for ready/items-like GETs =======================
class StrongETagMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, paths=("/api/items", "/api/ready/items")):
        super().__init__(app)
        self.paths = tuple(paths)

    async def dispatch(self, request, call_next):
        if request.method != "GET" or not any(request.url.path.startswith(p) for p in self.paths):
            return await call_next(request)

        resp = await call_next(request)

        if resp.status_code != 200:
            return resp

        body_bytes = b""
        if getattr(resp, "body_iterator", None) is not None:
            chunks = []
            async for c in resp.body_iterator:
                if c:
                    chunks.append(c if isinstance(c, (bytes, bytearray)) else bytes(c))
            body_bytes = b"".join(chunks)

            async def _body_gen():
                yield body_bytes
            resp.body_iterator = _body_gen()
            resp.headers["Content-Length"] = str(len(body_bytes))
        else:
            body_bytes = getattr(resp, "body", None) or b""

        etag = '"' + hashlib.sha256(body_bytes).hexdigest() + '"'

        inm = request.headers.get("if-none-match")
        if inm and inm == etag:
            return Response(status_code=304, headers={"ETag": etag})

        resp.headers.setdefault("ETag", etag)
        resp.headers.setdefault("Cache-Control", "public, max-age=0, must-revalidate")
        return resp

app.add_middleware(StrongETagMiddleware)
# ==== /ETag ==================================================================

# === [BEGIN] Reason Tags + Snapshot/Diff Endpoints ===========================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = [
    BASE_DIR / "selected_articles.json",
    BASE_DIR / "selected_keyword_articles.json",
]

ARCHIVE_DIR = BASE_DIR / "archive"
SNAP_MONTH_DIR = ARCHIVE_DIR / "snapshots" / "monthly"
DIFF_WEEK_DIR  = ARCHIVE_DIR / "diffs" / "weekly"
for d in (SNAP_MONTH_DIR, DIFF_WEEK_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _require_admin_token(req: Request):
    env_token = os.getenv("ADMIN_TOKEN", "").strip()
    if not env_token:
        return
    got = req.headers.get("authorization", "") or req.headers.get("Authorization", "")
    if got.lower().startswith("bearer "):
        got = got[7:].strip()
    if not got:
        got = req.headers.get("x-admin-token", "") or req.headers.get("X-Admin-Token", "")
    if (got or "").strip() != env_token:
        raise HTTPException(status_code=401, detail="invalid or missing admin token")

# Korean-only labels (no Han characters)
REASON_TAGS = [
    "근거충분", "전문성높음", "키워드핵심",
    "중복의심", "저작권리스크", "품질미달", "요약필요"
]

def _now_kst_iso():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")

def _safe_load_json(p: Path) -> Any:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def _safe_write_json(p: Path, data: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def _iter_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "articles" in data and isinstance(data["articles"], list):
        return data["articles"]
    return []

def _index_key(a: Dict[str, Any]) -> Optional[str]:
    return a.get("id") or a.get("url")

def _normalize_reason_field(item: Dict[str, Any]) -> bool:
    if "decision_reason" not in item or not isinstance(item.get("decision_reason"), list):
        item["decision_reason"] = []
        return True
    return False

def _update_reason_in_file(file_path: Path, key: str, reasons: List[str]) -> bool:
    data = _safe_load_json(file_path)
    items = _iter_items(data)
    found = False
    for it in items:
        if _index_key(it) == key:
            _normalize_reason_field(it)
            it["decision_reason"] = [r for r in reasons if r in REASON_TAGS]
            it["updated_at"] = _now_kst_iso()
            found = True
            break
    if found:
        _safe_write_json(file_path, data)
    return found

def _collect_all_articles() -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    for f in DATA_FILES:
        data = _safe_load_json(f)
        items = _iter_items(data)
        for it in items:
            _normalize_reason_field(it)
        all_items.extend(items)
    return all_items

def _sha256_of(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------- [A] Reason tags ----------
@app.get("/api/reason-tags")
def get_reason_tags():
    return {"tags": REASON_TAGS}

@app.get("/api/articles")
def list_articles_by_reason(reason: Optional[str] = None):
    items = _collect_all_articles()
    if not reason:
        return {"count": len(items), "items": items}
    want = [x.strip() for x in reason.split(",") if x.strip()]
    if not want:
        return {"count": len(items), "items": items}
    filtered = []
    for it in items:
        rs = set(it.get("decision_reason", []))
        if rs.intersection(want):
            filtered.append(it)
    return {"count": len(filtered), "items": filtered}

@app.get("/api/articles/reason")
def get_article_reason(key: str):
    items = _collect_all_articles()
    for it in items:
        if _index_key(it) == key:
            _normalize_reason_field(it)
            return {"key": key, "reasons": it["decision_reason"]}
    raise HTTPException(status_code=404, detail="article not found")

@app.post("/api/articles/reason")
async def set_article_reason(req: Request):
    _require_admin_token(req)
    body = await req.json()
    key = (body.get("key") or "").strip()
    reasons = body.get("reasons") or []
    if not key or not isinstance(reasons, list):
        raise HTTPException(status_code=400, detail="key and reasons(list) required")
    updated = False
    for f in DATA_FILES:
        if _update_reason_in_file(f, key, reasons):
            updated = True
    if not updated:
        raise HTTPException(status_code=404, detail="article not found in known files")
    return {"ok": True, "key": key, "reasons": [r for r in reasons if r in REASON_TAGS]}

# ---------- [B] Snapshot (monthly) ----------
@app.post("/api/archive/snapshot-monthly")
def make_monthly_snapshot(req: Request):
    _require_admin_token(req)
    now = datetime.now(timezone(timedelta(hours=9)))
    yyyy_mm = now.strftime("%Y-%m")
    snap_path = SNAP_MONTH_DIR / f"{yyyy_mm}.json"

    payload = {
        "type": "monthly_snapshot",
        "month": yyyy_mm,
        "generated_at": _now_kst_iso(),
        "sources": {f.name: _safe_load_json(f) for f in DATA_FILES},
    }
    payload["sha256"] = _sha256_of(payload)
    _safe_write_json(snap_path, payload)

    with (snap_path.with_suffix(".sha256")).open("w", encoding="utf-8") as wf:
        wf.write(payload["sha256"])

    return {"ok": True, "path": str(snap_path.relative_to(BASE_DIR)), "sha256": payload["sha256"]}

# ---------- [C] Diff (weekly) ----------
def _latest_baseline() -> Optional[Path]:
    candidates: List[Path] = []
    if SNAP_MONTH_DIR.exists():
        candidates += sorted(SNAP_MONTH_DIR.glob("*.json"))
    if DIFF_WEEK_DIR.exists():
        candidates += sorted(DIFF_WEEK_DIR.glob("*.json"))
    if not candidates:
        return None
    return candidates[-1]

def _index_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for it in items:
        k = _index_key(it)
        if not k:
            continue
        m[k] = it
    return m

@app.post("/api/archive/diff-weekly")
def make_weekly_diff(req: Request):
    _require_admin_token(req)
    now = datetime.now(timezone(timedelta(hours=9)))
    yyyy_ww = now.strftime("%G-%V")  # ISO Week
    diff_path = DIFF_WEEK_DIR / f"{yyyy_ww}.json"

    current_all = _collect_all_articles()
    curr_map = _index_map(current_all)

    baseline_file = _latest_baseline()
    base_map: Dict[str, Dict[str, Any]] = {}
    base_kind = None
    if baseline_file:
        base = _safe_load_json(baseline_file)
        base_kind = (base.get("type") if isinstance(base, dict) else None) or "unknown"
        if isinstance(base, dict) and base_kind == "monthly_snapshot":
            merged: List[Dict[str, Any]] = []
            for _, src in (base.get("sources") or {}).items():
                merged += _iter_items(src)
            base_map = _index_map(merged)
        elif isinstance(base, dict) and base_kind == "weekly_diff":
            merged = base.get("current_items") or []
            base_map = _index_map(merged)
        else:
            base_map = _index_map(_iter_items(base))

    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    modified: List[Dict[str, Any]] = []
    for k, v in curr_map.items():
        if k not in base_map:
            added.append(v)
        else:
            if _sha256_of(base_map[k]) != _sha256_of(v):
                modified.append({"before": base_map[k], "after": v})
    for k, v in base_map.items():
        if k not in curr_map:
            removed.append(v)

    result = {
        "type": "weekly_diff",
        "week": yyyy_ww,
        "generated_at": _now_kst_iso(),
        "baseline": {
            "file": str(baseline_file.relative_to(BASE_DIR)) if baseline_file else None,
            "kind": base_kind,
        },
        "summary": {"added": len(added), "removed": len(removed), "modified": len(modified)},
        "added": added,
        "removed": removed,
        "modified": modified,
        "current_items": current_all,
    }
    result["sha256"] = _sha256_of(result)
    _safe_write_json(diff_path, result)

    with (diff_path.with_suffix(".sha256")).open("w", encoding="utf-8") as wf:
        wf.write(result["sha256"])

    return {"ok": True, "path": str(diff_path.relative_to(BASE_DIR)), "summary": result["summary"], "sha256": result["sha256"]}
# === [END] Reason Tags + Snapshot/Diff Endpoints =============================

# ---------------------------------------------------------------------------
# Ready/SSOT helpers (router)
# ---------------------------------------------------------------------------
# Simple bearer (Cloud Run OIDC + App Token compatible)
security = HTTPBearer(auto_error=False)

async def authorize(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> bool:
    expected = [
        (os.environ.get("ADMIN_TOKEN") or "").strip(),
        (os.environ.get("API_TOKEN") or "").strip(),
    ]
    expected = [t for t in expected if t]
    if not expected:
        return True  # open mode

    x_admin = (request.headers.get("X-Admin-Token") or "").strip()

    supplied = credentials.credentials if credentials else ""
    if not supplied:
        auth = (request.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()

    if (x_admin and x_admin in expected) or (supplied and supplied in expected):
        return True
    raise HTTPException(status_code=401, detail="invalid or missing token")

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

# -------- SSOT paths/config --------
READY_DATA_DIR = os.environ.get("QUALI_DATA_DIR", "data")
READY_CONFIG   = os.environ.get("QUALI_CONFIG", "config.json")

READY_FILES = [
    os.path.join(READY_DATA_DIR, "selected_articles.json"),
    os.path.join(READY_DATA_DIR, "selected_keyword_articles.json"),
]

READY_EXTRA = [
    "selected_articles.json",                          # repo root
    "selected_keyword_articles.json",                  # repo root
    os.path.join(os.getenv("ARCHIVE_DIR", "/tmp/archive"),
                 "selected_keyword_articles.json"),    # Cloud Run
]

def _ready_load_json(path: str):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            data = open(path, "rb").read().decode("utf-8", errors="ignore").lstrip("\ufeff")
            return json.loads(data)
        except Exception:
            return []

def _normalize_items(obj) -> List[Dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        arr = obj.get("items") or obj.get("articles") or []
        return [x for x in arr if isinstance(x, dict)]
    return []

def _dedup_items(items: List[Dict]) -> List[Dict]:
    out, seen = [], set()
    for it in items:
        key = it.get("id") or (it.get("url") or it.get("link") or "")
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def _ready_all_items() -> List[Dict]:
    acc: List[Dict] = []
    for _p in (READY_FILES + READY_EXTRA):
        obj = _ready_load_json(_p)
        if not obj:
            continue
        acc.extend(_normalize_items(obj))
    return _dedup_items(acc)

def _ready_load_cfg() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    try:
        if os.path.exists(READY_CONFIG):
            with open(READY_CONFIG, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                if isinstance(file_cfg, dict):
                    cfg.update(file_cfg)
    except Exception:
        pass
    if "gate_required" not in cfg:
        cfg["gate_required"] = 70
    return cfg

def _is_ready_like(it: dict) -> bool:
    st = str(it.get("state", "")).strip().lower()
    return bool((it.get("ready") is True) or (st == "ready") or (it.get("approved") is True))

ready_router = APIRouter(prefix="/api/ready", tags=["ready-ssot"])

@ready_router.get("/items")
def ready_items(
    state: str = Query("ready"),
    date: Optional[str] = None,
    keyword: Optional[str] = None,
    authorized: bool = Depends(authorize)
):
    items = _ready_all_items()
    out: List[Dict] = []
    s = (state or "").strip().lower()

    for it in items:
        v = str(it.get("date", "")).strip()
        v = v.split("T")[0] if v else v
        if date and str(date).strip() and v and v != str(date).strip():
            continue

        kw_it = str(it.get("keyword", "")).strip()
        kw_q  = str(keyword or "").strip()
        if kw_q and kw_it and kw_it != kw_q:
            continue

        if s == "ready" and not _is_ready_like(it):
            continue
        out.append(it)

    return out

@ready_router.get("/status")
def ready_status(request: Request, authorized: bool = Depends(authorize)):
    items = _ready_all_items() if '_ready_all_items' in globals() else []
    total = len(items)
    ready_cnt = sum(
        1 for it in items
        if (it.get("ready") is True)
           or (str(it.get("state","")).strip().lower() == "ready")
           or (it.get("approved") is True)
    )
    cfg = _ready_load_cfg() if '_ready_load_cfg' in globals() else {"gate_required": 70}
    return {
        "ok": True,
        "client": (request.client.host if request and request.client else None),
        "total": total,
        "ready": ready_cnt,
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
# === READY/SSOT PATCH END ====================================================

# ---------------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent  # admin/

def _detect_root() -> Path:
    cands = [BASE.parent, BASE, BASE.parent.parent]
    for r in cands:
        if (r / "orchestrator.py").exists():
            return r
    return BASE.parent

ROOT = _detect_root()
ARCHIVE = ROOT / "archive"
TOOLS   = ROOT / "tools"
ORCH    = ROOT / "orchestrator.py"

SEL_COMM = ROOT / "selected_community.json"
SEL_WORK = ROOT / "data" / "selected_keyword_articles.json"
SEL_PUB  = ROOT / "selected_articles.json"

# Output dirs
ENRICHED_DIR = ARCHIVE / "enriched"
CAND_COMM = [SEL_COMM, ROOT / "archive" / "selected_community.json"]

INDEX_HTML = BASE / "index.html"
INDEX_LITE = BASE / "index_lite_black.html"
CONFIG_FILE = ROOT / "config.json"

# Cloud Run support
ARCHIVE_CLOUD = Path(os.getenv("ARCHIVE_DIR", "/tmp/archive"))
IS_CLOUD = bool(os.getenv("K_SERVICE"))

ARCHIVE_BASE = (ARCHIVE_CLOUD if IS_CLOUD else ARCHIVE)

LOGS_DIR = Path(os.getenv("LOGS_DIR") or ("/tmp/logs" if IS_CLOUD else (Path(__file__).resolve().parent.parent / "logs")))
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

CAND_WORK = [
    ARCHIVE_CLOUD / "selected_keyword_articles.json",   # Cloud Run: /tmp/archive/selected_keyword_articles.json
]

REPORT_DIR = (ARCHIVE_CLOUD / "reports") if IS_CLOUD else (BASE / "archive" / "reports")
ENRICH_DIR = (ARCHIVE_CLOUD / "enriched") if IS_CLOUD else (ARCHIVE / "enriched")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ENRICH_DIR.mkdir(parents=True, exist_ok=True)

# KPI / Gate defaults
KPI = {"selected": 0, "approved": 0, "published": 0}
GATE = {"gate_required": int(os.getenv("GATE_REQUIRED", "15"))}

# Directory to persist task logs. Logs are saved as <job_id>.log
TASK_LOG_DIR = LOGS_DIR / "tasks"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_Path = Path

try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

_logger = None
try:
    try:
        from logging_setup import setup_logger as _setup_any  # type: ignore
    except Exception:
        from logging_setup import setup_logging as _setup_any  # type: ignore
        _logger = _setup_any("server", str(LOGS_DIR / "server.log"))
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    _logger = logging.getLogger("server")
    _logger.info("fallback logger initialized (logging_setup missing)")

logger = _logger

# ---------------------------------------------------------------------------
# FastAPI app middlewares
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in (os.getenv("ALLOWED_ORIGINS","").split(",")) if o.strip()] or ["https://admin.example.com"],
    allow_methods=["GET","POST","PATCH","OPTIONS"],
    allow_headers=["Authorization","Content-Type","X-Admin-Token"],
    allow_credentials=True,
)

@app.middleware("http")
async def _force_utf8_charset(request: Request, call_next):
    """
    응답 헤더에 charset=utf-8 보장:
    - text/markdown → text/markdown; charset=utf-8
    - application/json → application/json; charset=utf-8
    (이미 charset이 있으면 건드리지 않음)
    """
    resp = await call_next(request)
    ctype = (resp.headers.get("content-type") or "").lower()

    if ctype.startswith("text/markdown") and "charset=" not in ctype:
        resp.headers["content-type"] = "text/markdown; charset=utf-8"
    elif ctype.startswith("application/json") and "charset=" not in ctype:
        resp.headers["content-type"] = "application/json; charset=utf-8"

    return resp


# === UI Static & Cache Headers (src-first + optional dist sync) ==============
_UI_BASE = BASE              # admin/
_UI_DIST = BASE / "dist"     # admin/dist
_UI_ACTIVE = _UI_DIST if (_UI_DIST / "index.html").exists() else _UI_BASE
_ASSETS_DIR = _UI_ACTIVE / "assets"

PREFER_SRC_UI = (os.getenv("PREFER_SRC_UI", "1") == "1")
AUTO_SYNC_UI  = (os.getenv("AUTO_SYNC_UI", "1") == "1")

def _q_sha256(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None

def ensure_admin_ui():
    if not AUTO_SYNC_UI:
        return
    src = BASE / "index.html"
    dst = BASE / "dist" / "index.html"
    if not src.exists():
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (not dst.exists()) or (_q_sha256(src) != _q_sha256(dst)):
            tmp = dst.with_suffix(".html.tmp")
            tmp.write_bytes(src.read_bytes())
            tmp.replace(dst)
    except Exception:
        pass

def _admin_index_path() -> Path:
    src = BASE / "index.html"
    dist = BASE / "dist" / "index.html"
    if PREFER_SRC_UI and src.exists():
        return src
    return dist if dist.exists() else src

try:
    if _ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")
except Exception:
    pass

@app.middleware("http")
async def _cache_headers_ui(request: Request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if re.match(r"^/assets/.+\.[0-9a-f]{8,}\.(js|css|png|jpg|svg|woff2?)$", p, re.I):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif p in ("/", "/index.html", "/service-worker.js"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"

    return resp

# ---------------------------------------------------------------------------
@app.get("/api/archive/{path:path}")
def download_archive(path: str, request: Request):
    base = (ARCHIVE_CLOUD if IS_CLOUD else (BASE / "archive")).resolve()
    full = (base / path).resolve()
    if not str(full).startswith(str(base)) or not full.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(full), filename=full.name)

# ---------------------------------------------------------------------------
# Health & Misc
# ---------------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": True}

@app.get("/api/db/mode")
def db_mode():
    return {"mode": MODE}

# --- Approve UI opener -------------------------------------------------------
class EnrichReq(BaseModel):
    date: Optional[str] = None
    keyword: Optional[str] = None
    mode: Optional[str] = "keyword"  # "keyword" or "selection"
    items: Optional[List[Dict[str, Any]]] = None

class GatePatch(BaseModel):
    gate_required: int

class PublishOneReq(BaseModel):
    approve: bool = Field(default=True, description="승인 여부")
    editor_note: Optional[str] = Field(default=None, description="편집장 한마디")

class TaskItem(BaseModel):
    id: str
    size: int

class TasksRecent(BaseModel):
    items: List[TaskItem]

class ReportReq(BaseModel):
    date: str | None = None
    keyword: str | None = None

class FlowReq(BaseModel):
    kind: str            # daily|community|keyword
    keyword: str | None = None
    use_external_rss: bool = False

class PublishReq(BaseModel):
    keyword: str

class FlowKwReq(BaseModel):
    keyword: str
    use_external_rss: bool = False

@app.post("/api/approve-ui/start")
def approve_ui_start(request: Request, authorized: bool = Depends(authorize)):
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
    ensure_admin_ui()
    target = _admin_index_path()
    if target and target.exists():
        return HTMLResponse(
            target.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )

    return HTMLResponse("<h1>QualiJournal Admin</h1><p>index.html이 없습니다.</p>",
                        headers={"Cache-Control": "no-store"})

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
    data_in = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    data = {
        "ok": bool(data_in.get("ok")),
        "ts": int(data_in.get("ts") or 0),
        "size_md": int(data_in.get("size_md") or 0),
        "size_csv": int(data_in.get("size_csv") or 0),
    }
    _LAST_BACKUP = data
    return {"ok": True}

@app.get("/api/backup/status")
def backup_status():
    return _LAST_BACKUP

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
    header_suffix = "Selected" if selected else "All"
    title_parts = ["QualiNews", date]
    if header_kw:
        title_parts.append(header_kw)
    title_parts.append(f"({header_suffix})")
    lines: List[str] = [" · ".join(title_parts), ""]

    for i, art in enumerate(articles, 1):
        title = art.get("title") or art.get("headline") or "(no title)"
        url   = art.get("url") or art.get("link") or ""
        summary = art.get("summary") or art.get("ko_summary") or art.get("desc") or ""
        note  = art.get("editor_note") or ""
        lines.append(f"### {i}. {title}")
        if url:
            lines.append(f"- 링크: {url}")
        if summary:
            lines.append(f"- 요약: {summary}")
        if note:
            lines.append(f"- 편집장 한마디: {note}")
        lines.append("")
    md = "\n".join(lines)
    out_path.write_text(md, encoding="utf-8")
    return out_path

def _run_orch(*args: str) -> dict:
    """Run orchestrator.py with UTF-8 safety; return {'ok', 'stdout', 'stderr', 'cmd'}."""
    py  = PYEXE
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    script = None
    if ORCH.exists():
        script = ORCH
    elif (TOOLS / "orchestrator.py").exists():
        script = TOOLS / "orchestrator.py"
    elif (ROOT / "orchestrator.py").exists():
        script = ROOT / "orchestrator.py"

    if not script:
        return {"ok": False, "stdout": "", "stderr": "orchestrator.py not found in image.", "cmd": f"{py} orchestrator.py {' '.join(args)}"}

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
    return {"ok": cp.returncode == 0, "stdout": cp.stdout, "stderr": cp.stderr, "cmd": " ".join([str(script), *args])}

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
    obj = {}
    for p in CAND_WORK:
        if p.exists():
            obj = _read_json(p) or {}
            if (isinstance(obj, dict) and (obj.get("articles") or obj.get("items"))) or isinstance(obj, list):
                break

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

# ==== DEV seeding helpers (no-op in prod) ====================================
def _seed_selected_keyword_work(keyword: str = "IPC-A-610", n: int = 20) -> Path:
    date = _dt.date.today().isoformat()
    arts = []
    for i in range(1, n+1):
        arts.append({
            "id": f"seed-{keyword.lower()}-{i:02d}",
            "title": f"[SEED] {keyword} sample #{i}",
            "url": f"https://example.com/{keyword}/{i}",
            "source": "seed",
            "keyword": keyword,
            "date": date,
            "approved": True,
            "ready": True,
            "state": "ready",
            "summary": "seed item"
        })
    obj = {"date": date, "keyword": keyword, "articles": arts}
    out = (BASE / "data" / "selected_keyword_articles.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

@app.post("/api/dev/seed/keyword")
def dev_seed_keyword(req: FlowKwReq | None = Body(default=None),
                     authorized: bool = Depends(authorize)):
    kw = (req.keyword if isinstance(req, FlowKwReq) else None) or "IPC-A-610"
    p = _seed_selected_keyword_work(kw, n=20)
    sync = _sync_after_save()
    return {"ok": True, "path": str(p), "synced": bool(sync.get("ok"))}
# ==== [/PATCH] ===============================================================

def _rollover_archive_if_needed(keyword: str) -> Optional[List[str]]:
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    created = []; ARCHIVE_BASE.mkdir(parents=True, exist_ok=True)
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE_BASE / f"{base}{ext}"
        if p.exists():
            ts = _dt.datetime.now().strftime("%H%M")
            newp = ARCHIVE_BASE / f"{base}_{ts}{ext}"
            p.rename(newp); created.append(str(newp))
    return created or None

def _latest_published_paths(keyword: str) -> List[str]:
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    out = []
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE_BASE / f"{base}{ext}"
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
                if isinstance(obj, dict):
                    arr = obj.get("items") or obj.get("articles") or []
                    return arr, obj.get("date"), obj.get("keyword")
                if isinstance(obj, list):
                    return obj, None, None
            except Exception:
                pass
    return [], None, None

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
        global TASK_LOG_DIR
        try:
            TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_file: Path | None = TASK_LOG_DIR / f"{self.id}.log"
            self.log_file.write_text("", encoding="utf-8")
        except Exception:
            self.log_file = None

    def append(self, line: str):
        with self._lock:
            ts = _dt.datetime.now().strftime("%H:%M:%S")
            msg = f"[{ts}] {line}"
            self.logs.append(msg)
            try:
                if self.log_file:
                    with self.log_file.open("a", encoding="utf-8") as fp:
                        fp.write(msg + "\n")
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
            if task._cancel:
                try: p.kill()
                except Exception: pass
                task.append("! canceled")
                return 1
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            if line:
                task.append(line.rstrip())
        return p.returncode if p.returncode is not None else 0

    try:
        py = PYEXE
        rc = 0
        if task.kind == "daily":
            rc = run_cmd([py, str(ORCH), "--collect-community"])
            if not task._cancel:
                rc2 = run_cmd([py, str(ORCH), "--publish-community", "--format", "all"])
                rc = max(rc, rc2)
            if not task._cancel:
                rc3 = run_cmd([py, str(ORCH), "--publish", "--format", "all"])
                rc = max(rc, rc3)
        elif task.kind == "community":
            rc = run_cmd([py, str(ORCH), "--collect-community"])
            if not task._cancel:
                rc2 = run_cmd([py, str(ORCH), "--publish-community", "--format", "all"])
                rc = max(rc, rc2)
        elif task.kind == "keyword":
            kw = task.args[0] if task.args else ""
            ext = task.args[1] if len(task.args) > 1 else ""
            if not kw:
                raise RuntimeError("keyword required")
            if str(ext).strip() == "--use-external-rss":
                rc = run_cmd([py, str(ORCH), "--collect-keyword", kw, "--use-external-rss"])
            else:
                rc = run_cmd([py, str(ORCH), "--collect-keyword", kw])
            if not task._cancel:
                rc2 = run_cmd([py, str(ORCH), "--approve-keyword", kw, "--approve-keyword-top", "20"])
                rc = max(rc, rc2)
            if not task._cancel:
                rc3 = run_cmd([py, str(ORCH), "--publish-keyword", kw])
                rc = max(rc, rc3)
        else:
            raise RuntimeError(f"unknown kind: {task.kind}")

        task.exit_code = rc
        if task._cancel:
            task.status = "canceled"
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
    if kind == "keyword":
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
    t._cancel = True
    try:
        t.append("! cancel requested by user")
    except Exception:
        pass
    return {"ok": True}

@app.get("/api/tasks/{job_id}/stream")
async def stream_task(job_id: str, request: Request):
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
# Report / Enrich / Export (protected)
# ---------------------------------------------------------------------------
@app.get("/api/report")
def get_report(date: str | None = None, authorized: bool = Depends(authorize)):
    day = date or _dt.date.today().isoformat()
    ARCHIVE_BASE.mkdir(parents=True, exist_ok=True)
    items = []
    for p in ARCHIVE_BASE.glob(f"{day}*.*"):
        items.append({"name": p.name, "size": p.stat().st_size})
    for p in ARCHIVE_BASE.glob(f"community_{day}.*"):
        items.append({"name": p.name, "size": p.stat().st_size})
    for p in ARCHIVE.glob(f"daily_{day}.*"):
        items.append({"name": p.name, "size": p.stat().st_size})
    return {"date": day, "files": items}

@app.patch("/api/config/gate_required")
async def set_gate_required(p: GatePatch, authorized: bool = Depends(authorize)):
    v = max(1, min(100, int(p.gate_required)))
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
        items: List[dict] = []
        keyword = "report"

        def _slug(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "")).strip("_") or "report"

        for p in candidates:
            if p.exists():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        items = obj.get("items") or obj.get("articles") or []
                        if items:
                            keyword = _slug(obj.get("keyword") or keyword)
                    elif isinstance(obj, list):
                        items = obj
                    break
                except Exception:
                    pass

        def _esc(s): return re.sub(r"[\r\n]+", " ", str(s or "")).strip()
        lines: List[str] = [f"# {date} · {keyword.upper()} · Daily Report", ""]
        if items:
            for i, it in enumerate(items, 1):
                t = _esc(it.get("title") or it.get("headline") or "(no title)")
                u = _esc(it.get("url") or it.get("link") or "")
                se = _esc(it.get("summary_en") or it.get("summary") or "")
                sk = _esc(it.get("summary_ko") or it.get("summary_kr") or "")
                note = _esc(it.get("editor_note") or "")
                lines.append(f"## {i}. {t}")
                if u:   lines.append(f"- 링크: {u}")
                if se:  lines.append(f"- 요약(EN): {se}")
                if sk:  lines.append(f"- 요약(KO): {sk}")
                if note:lines.append(f"- 편집장 한마디: {note}")
                lines.append("")
        else:
            lines += ["(데이터 없음)", ""]

        out = reports_dir / f"{date}_{keyword}_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        rel = f"archive/reports/{out.name}"
        return _ok("report", path=rel, count=len(items), duration_ms=_now_ms()-t0)
    except Exception as e:
        return _err("report", str(e), duration_ms=_now_ms()-t0)

@app.post("/api/enrich/keyword")
def enrich_keyword(req: EnrichReq | None = Body(default=None), authorized: bool = Depends(authorize)):
    t0 = _now_ms()
    try:
        req = req or EnrichReq()
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
    """
    day = date or _dt.date.today().isoformat()
    fmt_lower = fmt.lower()

    # final selection export
    if fmt_lower in ("md", "csv"):
        data = _read_json(SEL_PUB)
        articles = data.get("articles", []) or []
        kw = data.get("keyword", "") or ""
        if fmt_lower == "md":
            lines = [f"# QualiNews · {day} · {kw}", ""]
            for i, a in enumerate(articles, 1):
                title = a.get("title", "(no title)")
                url   = a.get("url") or a.get("link") or ""
                summ  = a.get("summary") or a.get("ko_summary") or a.get("desc") or ""
                note  = a.get("editor_note") or ""
                lines.append(f"### {i}. {title}")
                if url:  lines.append(f"- 링크: {url}")
                if summ: lines.append(f"- 요약: {summ}")
                if note: lines.append(f"- 편집장 한마디: {note}")
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
    cj = ARCHIVE_BASE / f"community_{day}.json"
    if not cj.exists():
        raise HTTPException(404, "community json not found")
    obj = _read_json(cj); arts = obj.get("articles", [])
    if fmt_lower == "md":
        lines = [f"# Community · {day}", ""]
        for a in arts:
            title = a.get("title") or "(no title)"
            url = a.get("url") or "#"
            lines.append(f"- [{title}]({url})")
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

@app.get("/export/md")
def export_md_alias(preview: bool = Query(False), authorized: bool = Depends(authorize)):
    return export_fmt(fmt="md", preview=preview, authorized=authorized)

@app.get("/export/csv")
def export_csv_alias(authorized: bool = Depends(authorize)):
    return export_fmt(fmt="csv", authorized=authorized)

# ---------------------------------------------------------------------------
# Selection Approvals (protected)
# ---------------------------------------------------------------------------
_SEL_LOCK = threading.Lock()

class SelectionItemPatch(BaseModel):
    idx: Optional[int] = Field(default=None, description="articles[] index (optional)")
    id: Optional[str] = Field(default=None, description="stable item id")
    approved: Optional[bool] = None
    editor_note: Optional[str] = None

class SelectionPatchRequest(BaseModel):
    updates: List[SelectionItemPatch]
    autosync: bool = True  # selected_articles.json autosync

@app.get("/api/selection")
def api_selection_list(
    authorized: bool = Depends(authorize),
    keyword: Optional[str] = None,
    date: Optional[str] = None
):
    def _match_kw(a: Dict[str, Any], kw: Optional[str]) -> bool:
        if not kw:
            return True
        kw = kw.strip().lower()
        bag = " ".join([
            str(a.get("keyword") or a.get("kw") or ""),
            str(a.get("title") or ""),
            str(a.get("source") or a.get("publisher") or ""),
            str(a.get("url") or a.get("link") or "")
        ]).lower()
        if bag.strip() == "":
            return True
        return (kw in bag)

    def _match_date(a: Dict[str, Any], d: Optional[str]) -> bool:
        if not d: 
            return True
        return d in str(a.get("date") or "")

    try:
        snap = _get_work_snapshot() or {}
    except Exception:
        snap = {}

    arts_full = (snap.get("articles") or [])
    snap_date = snap.get("date")
    snap_kw   = snap.get("keyword") or ""

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
    if not req.updates:
        return {"updated": 0, "synced": False}

    with _SEL_LOCK:
        work = {}
        for p in CAND_WORK:
            if p.exists():
                work = _read_json(p) or {}
                if work: break

        arts = work.get("articles", []) or []
        if not arts:
            raise HTTPException(status_code=400, detail="no articles in work file")

        id_to_idx: Dict[str, int] = {}
        for i, a in enumerate(arts):
            try:
                id_to_idx[_ensure_id(a)] = i
            except Exception:
                continue

        changed = 0
        for p in req.updates:
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
# Community / Items / Publish (protected)
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
    """
    try:
        def _match(it: dict) -> bool:
            if date and str(it.get("date","")) != str(date):
                return False
            if keyword and str(it.get("keyword","")).strip() != str(keyword).strip():
                return False
            return True

        s = (state or "").lower().strip()

        if s == "published":
            pub = _read_json(SEL_PUB) or {}
            items = [a for a in (pub.get("articles", []) or []) if _match(a)]
            return {
                "date": date or pub.get("date"),
                "keyword": keyword or "",
                "state": s,
                "items": items,
            }

        if s == "ready":
            items_any, d_auto, kw_auto = _read_any_items(BASE)
            src_items = items_any or []
            if not src_items:
                snap = _get_work_snapshot()
                arts = (snap.get("articles", []) or [])
                src_items = [a for a in arts if (a.get("state","").lower()=="ready") or a.get("approved")]
                d_auto = d_auto or snap.get("date")
                kw_auto = kw_auto or snap.get("keyword","")
            items = [a for a in src_items if _match(a)]
            return {
                "date": date or d_auto,
                "keyword": keyword or kw_auto or "",
                "state": s,
                "items": items,
            }

        snap = _get_work_snapshot()
        arts = snap.get("articles", []) or []
        if s in ("candidate","rejected"):
            src_items = [a for a in arts if (a.get("state","").lower()==s)]
        else:
            src_items = arts
        items = [a for a in src_items if _match(a)]
        return {
            "date": date or snap.get("date"),
            "keyword": keyword or snap.get("keyword",""),
            "state": s or "all",
            "items": items,
        }

    except Exception as e:
        try:
            logger.warning("api_items error: %s", e)
        except Exception:
            pass
        return {
            "date": date,
            "keyword": (keyword or "").strip(),
            "state": (state or "").lower().strip() or "ready",
            "items": [],
        }

@app.post("/api/items/{item_id}/publish")
def api_items_publish(item_id: str, req: PublishOneReq, authorized: bool = Depends(authorize)):
    """Publish single item: update approval and notes in work file then sync to publish file."""
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
    ARCHIVE_BASE.mkdir(parents=True, exist_ok=True)
    rollover = _rollover_archive_if_needed(req.keyword)
    out = _run_orch("--publish-keyword", req.keyword)
    outputs = _latest_published_paths(req.keyword)
    return {**out, "rolled_over": rollover or [], "created": outputs}

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
    if not ok:
        try:
            seed_path = _seed_selected_keyword_work("IPC-A-610", n=20)
            steps.append({"ok": True, "cmd": "seed-ready", "stdout": str(seed_path), "stderr": ""})
            steps.append(_sync_after_save())
            ok = True
        except Exception as e:
            steps.append({"ok": False, "cmd": "seed-ready", "stderr": str(e)})
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
    if not ok:
        try:
            seed_path = _seed_selected_keyword_work(req.keyword or "IPC-A-610", n=20)
            steps.append({"ok": True, "cmd": "seed-ready", "stdout": str(seed_path), "stderr": ""})
            steps.append(_sync_after_save())
            ok = True
        except Exception as e:
            steps.append({"ok": False, "cmd": "seed-ready", "stderr": str(e)})
    return {"ok": ok, "steps": steps}

# ---------------------------------------------------------------------------
# Logs (protected + optional JWT)
# ---------------------------------------------------------------------------
try:
    from auth_utils import verify_jwt_token  # type: ignore
except Exception:  # pragma: no cover
    async def verify_jwt_token(*args, **kwargs):  # type: ignore
        return {}

@app.get("/api/logs")
def list_logs(authorized: bool = Depends(authorize), user: dict = Depends(verify_jwt_token)):
    logs_dir = LOGS_DIR
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
    path = LOGS_DIR / log_name
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
    path = LOGS_DIR / log_name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "log not found")
    return FileResponse(str(path), filename=log_name, media_type="text/plain")

# ---------------------------------------------------------------------------
# KPI status (protected)
# ---------------------------------------------------------------------------
@app.get("/api/status")
def get_status(
    request: Request,
    date: str | None = None,
    keyword: str | None = None,
    authorized: bool = Depends(authorize),
):
    def _safe_int(x, d=0):
        try: return int(x)
        except: return d

    if globals().get("_DB_READY"):
        try:
            snap = (_get_work_snapshot() or {}) if "_get_work_snapshot" in globals() else {}
            edate = (date or snap.get("date") or _dt.date.today().isoformat())
            ekw   = (keyword or snap.get("keyword") or "").strip()
            try: _ed = _dt.date.fromisoformat(str(edate))
            except: _ed = _dt.date.today()

            sess = get_session()
            ed   = get_or_create_edition(sess, etype="keyword", edate=_ed, keyword=(ekw or None))
            k    = (kpi_for_edition(sess, ed) or {})
            total    = _safe_int(k.get("total"))
            approved = _safe_int(k.get("approved"))
            ready    = _safe_int(k.get("ready"))
            gate_req = _safe_int(k.get("gate_required"), 15)

            return {
                "ok": True, "client": (request.client.host if request and request.client else None),
                "ts": int(time.time()),
                "selected": 0, "approved": 0, "published": 0,
                "selection_total": total, "selection_approved": approved,
                "state_counts": {"candidate": max(total - ready, 0), "ready": ready, "rejected": 0},
                "community_total": 0, "keyword_total": total,
                "gate_required": gate_req, "gate_pass": bool(approved >= gate_req),
                "date": str(_ed), "keyword": ekw,
            }
        except:
            pass

    try:
        work = (_get_work_snapshot() or {}) if "_get_work_snapshot" in globals() else {}
        arts = work.get("articles") or []
        st_counts = {"candidate": 0, "ready": 0, "rejected": 0}
        sel_total = len(arts); sel_approved = 0
        for a in arts:
            st = str(a.get("state") or "candidate").lower().strip()
            if st in st_counts: st_counts[st] += 1
            if a.get("approved") or a.get("selected") or st in ("ready","published"): sel_approved += 1
        comm = (_get_community_snapshot() or {}) if "_get_community_snapshot" in globals() else {}
        comm_total = len(comm.get("articles") or [])
        gate_req = int(GATE.get("gate_required", 15)) if isinstance(GATE, dict) else 15
        return {
            "ok": True, "client": (request.client.host if request and request.client else None),
            "ts": int(time.time()),
            "selected": KPI.get("selected",0) if isinstance(KPI,dict) else 0,
            "approved": KPI.get("approved",0) if isinstance(KPI,dict) else 0,
            "published": KPI.get("published",0) if isinstance(KPI,dict) else 0,
            "gate_required": gate_req,
            "selection_total": sel_total, "selection_approved": sel_approved,
            "state_counts": st_counts, "community_total": comm_total,
            "keyword_total": sel_total, "gate_pass": bool(sel_approved >= gate_req),
            "date": date or work.get("date"), "keyword": (keyword or work.get("keyword","")).strip(),
        }
    except:
        return {
            "ok": True, "client": (request.client.host if request and request.client else None),
            "ts": int(time.time()),
            "selected": 0, "approved": 0, "published": 0,
            "gate_required": 15, "selection_total": 0, "selection_approved": 0,
            "state_counts": {"candidate": 0, "ready": 0, "rejected": 0},
            "community_total": 0, "keyword_total": 0, "gate_pass": False,
            "date": date, "keyword": (keyword or "").strip(),
        }
# --- Service Worker (no-store) ----------------------------------------------

# 후보 경로(있는 쪽 자동 선택)
_SW_CANDIDATES = [
    BASE / "service-worker.js",          # admin/service-worker.js
    BASE / "dist" / "service-worker.js", # admin/dist/service-worker.js
]

def _pick_sw() -> Path:
    for p in _SW_CANDIDATES:
        if p.exists():
            return p
    return _SW_CANDIDATES[0]  # 없으면 첫 후보(아래에서 404 처리)

@app.get("/service-worker.js", include_in_schema=False)
def serve_service_worker():
    sw_path = _pick_sw()
    if not sw_path.exists():
        # 어디를 찾고 있는지 404 메시지로 돌려줌(디버그용)
        raise HTTPException(status_code=404, detail=f"service-worker not found: {sw_path}")

    build = (os.getenv("COMMIT_SHA") or os.getenv("BUILD_ID") or os.getenv("K_REVISION") or "dev").strip()
    txt = sw_path.read_text(encoding="utf-8").replace("__BUILD_ID__", build)

    return Response(
        content=txt,
        media_type="text/javascript; charset=utf-8",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Service-Worker-Allowed": "/"  # 루트 스코프 고정
        },
    )

# 디버그: 실제 후보/선택 경로 확인
@app.get("/api/debug/sw-path", include_in_schema=False)
def _debug_sw_path():
    picked = _pick_sw()
    return {
        "candidates": [{"path": str(p), "exists": p.exists()} for p in _SW_CANDIDATES],
        "picked": str(picked),
        "exists": picked.exists(),
    }
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    _port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=_port, log_level="info")
