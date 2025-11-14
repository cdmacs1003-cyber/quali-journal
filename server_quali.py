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
from pathlib import Path
from typing import List, Dict, Any, Optional

# FastAPI / Pydantic
from fastapi import FastAPI, Query, Response, HTTPException, Depends, Body, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

# SSOT: 서브프로세스도 이 파이썬으로 실행
PYEXE = os.getenv("PYTHON_EXE") or sys.executable or "python"

# ---------------------------------------------------------------------------
# .env (optional) - load first; define MODE fallback regardless of availability
# ---------------------------------------------------------------------------
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

class FlowReq(BaseModel):
    kind: str            # daily|community|keyword
    keyword: str | None = None

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

# ---------------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent  # admin/

def _detect_root() -> Path:
    """
    Orchestrator root auto-detection:
    prefer admin/.. → admin/ → admin/../..
    """
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

# Enriched summaries output dir
ENRICHED_DIR = ARCHIVE / "enriched"
CAND_COMM = [SEL_COMM, ROOT / "archive" / "selected_community.json"]

INDEX_HTML = BASE / "index.html"
INDEX_LITE = BASE / "index_lite_black.html"
CONFIG_FILE = ROOT / "config.json"

# Cloud Run support
ARCHIVE_CLOUD = Path(os.getenv("ARCHIVE_DIR", "/tmp/archive"))
IS_CLOUD = bool(os.getenv("K_SERVICE"))

REPORT_DIR = (ARCHIVE_CLOUD / "reports") if IS_CLOUD else (BASE / "archive" / "reports")
ENRICH_DIR = (ARCHIVE_CLOUD / "enriched") if IS_CLOUD else (ARCHIVE / "enriched")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ENRICH_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_ADMIN = BASE / "archive"   # admin\archive (report .md 저장 위치)

# KPI / Gate defaults
KPI = {"selected": 0, "approved": 0, "published": 0}
GATE = {"gate_required": int(os.getenv("GATE_REQUIRED", "15"))}

# Directory to persist task logs. Logs are saved as <job_id>.log
TASK_LOG_DIR = ROOT / "logs" / "tasks"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from logging_setup import setup_logger  # type: ignore
    logger = setup_logger("server", str(ROOT / "logs" / "server.log"))
except Exception:  # pragma: no cover
    import logging
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(ROOT / "logs" / "server.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )
    logger = logging.getLogger("server")
    logger.info("fallback logger initialized")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in (os.getenv("ALLOWED_ORIGINS","").split(",")) if o.strip()] or ["https://admin.example.com"],
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

# === 보호 다운로드 엔드포인트 (/api/archive/...) ===
@app.get("/api/archive/{path:path}")
def download_archive(path: str, request: Request):
    """
    보호된 다운로드(헤더 Bearer 또는 ?token= 허용).
    공개 정적 브라우징(/archive)은 유지하되, 실사용 링크는 이 경로 권장.
    """
    _auth_header_or_qs_ok(request)  # 헤더 우선, ?token 허용

    base = (ARCHIVE_CLOUD if IS_CLOUD else (BASE / "archive")).resolve()
    full = (base / path).resolve()

    # 경로 이탈 방지(../../ 차단)
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
@app.post("/api/approve-ui/start")
def approve_ui_start(request: Request, authorized: bool = Depends(authorize)):
    """
    승인 UI가 열릴 때 UI URL과 현재 스냅샷(날짜/키워드/게이트)을 안내.
    - UI는 반환된 ui_url을 새 창으로 open (index.html의 startApprove 사용).
    """
    snap = _get_work_snapshot()  # {"date","keyword","articles":[...]}
    ui_url = (os.getenv("APPROVE_UI_URL") or str(request.base_url).rstrip("/"))
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

@app.get("/")
def index():
    p = INDEX_HTML if INDEX_HTML.exists() else (INDEX_LITE if INDEX_LITE.exists() else None)
    if p:
        # 캐시 무시 헤더로 구판 캐시 완전 차단
        return HTMLResponse(
            p.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )
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
    d = (ROOT / "logs" / "tasks")
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
    obj = _read_json(SEL_WORK) or {}
    arts = obj.get("articles", []) or []
    for a in arts:
        _ensure_id(a)
        a.setdefault("approved", False)
        a.setdefault("editor_note", "")
        a.setdefault("state", (a.get("state") or "").lower() or "candidate")
    date = obj.get("date") or _dt.date.today().isoformat()
    keyword = obj.get("keyword", "")
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
    work = _read_json(SEL_WORK) or {}
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
            # persist to disk
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
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8"
        )
        while True:
            if task._cancel:
                p.kill(); task.append("! canceled"); return 1
            line = p.stdout.readline()
            if not line and p.poll() is not None: break
            if line: task.append(line.rstrip())
        return p.returncode if p.returncode is not None else 0

    try:
        py = PYEXE
        if task.kind == "daily":
            rc1 = run_cmd([py, str(ORCH), "--collect-community"])
            rc2 = run_cmd([py, str(ORCH), "--publish-community", "--format", "all"])
            rc3 = run_cmd([py, str(ORCH), "--publish", "--format", "all"])
            rc = max(rc1, rc2, rc3)
        elif task.kind == "community":
            rc1 = run_cmd([py, str(ORCH), "--collect-community"])
            rc2 = run_cmd([py, str(ORCH), "--publish-community", "--format", "all"])
            rc = max(rc1, rc2)
        elif task.kind == "keyword":
            kw = task.args[0] if task.args else ""
            if not kw:
                raise RuntimeError("keyword required")
            rc1 = run_cmd([py, str(ORCH), "--collect-keyword", kw])
            rc2 = run_cmd([py, str(ORCH), "--approve-keyword", kw, "--approve-keyword-top", "15"])
            rc3 = run_cmd([py, str(ORCH), "--publish-keyword", kw])
            rc = max(rc1, rc2, rc3)
        else:
            raise RuntimeError(f"unknown kind: {task.kind}")
        task.exit_code = rc
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
    args = [req.keyword] if kind == "keyword" else []
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
    t._cancel = True
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
# Gate config (GET/PATCH) — protected
# ---------------------------------------------------------------------------
@app.get("/api/config/gate_required")
async def get_gate_required(authorized: bool = Depends(authorize)):
    return {"gate_required": int(GATE.get("gate_required", 15))}

@app.patch("/api/config/gate_required")
async def set_gate_required(p: GatePatch, authorized: bool = Depends(authorize)):
    v = max(1, min(100, int(p.gate_required)))  # 1~100 clamp
    GATE["gate_required"] = v
    return {"ok": True, "gate_required": v}

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

@app.post("/api/report")
def post_report(payload: dict | None = Body(default=None), authorized: bool = Depends(authorize)):
    """
    UI 스피너/토스트 연동을 위해 항상 {"ok", "op", "path", "count", "duration_ms"} 형태로 응답.
    실패 시에도 200 JSON으로 {"ok":False,"error":...} 반환(토스트용).
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
        items = []
        keyword = "report"

        def _slug(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "")).strip("_") or "report"

        for p in candidates:
            if p.exists():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(obj, dict) and "items" in obj:
                        items = obj.get("items") or []
                        keyword = _slug(obj.get("keyword") or keyword)
                    elif isinstance(obj, list):
                        items = obj
                    break
                except Exception:
                    pass

        def _esc(s): return re.sub(r"[\r\n]+", " ", str(s or "")).strip()
        lines = [f"# {date} · {keyword.upper()} · Daily Report", ""]
        if items:
            for i, it in enumerate(items, 1):
                t = _esc(it.get("title") or it.get("headline") or "(제목 없음)")
                u = _esc(it.get("url") or it.get("link") or "")
                se = _esc(it.get("summary_en") or it.get("summary") or "")
                sk = _esc(it.get("summary_ko") or it.get("summary_kr") or "")
                note = _esc(it.get("editor_note") or "")
                lines.append(f"## {i}. {t}")
                if u:   lines.append(f"- 원문: {u}")
                if se:  lines.append(f"- 요약(EN): {se}")
                if sk:  lines.append(f"- 요약(KO): {sk}")
                if note:lines.append(f"- 코멘트: {note}")
                lines.append("")
        else:
            lines += ["(수집된 기사 없음)", ""]

        out = reports_dir / f"{date}_{keyword}_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        rel = f"archive/reports/{out.name}"
        return _ok("report", path=rel, count=len(items), duration_ms=_now_ms()-t0)
    except Exception as e:
        return _err("report", str(e), duration_ms=_now_ms()-t0)


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

@app.post("/api/tools/repair")
def api_tools_repair(authorized: bool = Depends(authorize)):
    """
    repair_selection_files.py 실행:
    - 작업본/발행본 JSON 구조 교정 + 승인본 재작성
    - 실행 로그와 성공 여부를 반환
    """
    rc, out, err = _run_py("repair_selection_files.py", [])
    ok = (rc == 0)
    # 실행 후 발행본 싱크 보강(있으면 내부 스크립트가 수행하지만, 안전하게 한 번 더)
    sync = _sync_after_save()
    return {
        "ok": ok,
        "rc": rc,
        "stdout": out.strip(),
        "stderr": err.strip(),
        "synced": bool(sync.get("ok")),
        "sync_log": sync,
    }

@app.post("/api/tools/approve_top")
def api_tools_approve_top(n: int = Query(20, ge=1, le=100), authorized: bool = Depends(authorize)):
    """
    force_approve_top20.py 실행:
    - 상위 n개 승인(approved=True) 처리
    - 처리 후 발행본을 동기화
    """
    rc, out, err = _run_py("force_approve_top20.py", ["--top", str(n)])
    ok = (rc == 0)
    sync = _sync_after_save()  # 스크립트가 싱크하더라도 안전하게 최종 보정
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
    logs_dir = ROOT / "logs"
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
    path = ROOT / "logs" / log_name
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
    path = ROOT / "logs" / log_name
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
    관리자 KPI/상태 조회 (토큰 보호)
    - selection/community 집계, 게이트 통과 여부, 타임스탬프 포함
    - date/keyword 쿼리 파라미터는 선택(지정 없으면 스냅샷 값 사용)
    """
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
            # 누적 KPI(없으면 0)
            "selected": KPI.get("selected", 0) if isinstance(KPI, dict) else 0,
            "approved": KPI.get("approved", 0) if isinstance(KPI, dict) else 0,
            "published": KPI.get("published", 0) if isinstance(KPI, dict) else 0,

            # 현재 스냅샷 집계
            "gate_required": gate_required,
            "ts": int(time.time()),

            "selection_total": selection_total,
            "selection_approved": selection_approved,
            "state_counts": state_counts,

            "community_total": community_total,
            "keyword_total": selection_total,   # 선택본 전체를 키워드 총량으로 간주

            "gate_pass": gate_pass,

            # 표시용 메타(요청값 우선 → 스냅샷 값 폴백)
            "date": date or work.get("date"),
            "keyword": keyword or work.get("keyword", ""),
        }
    except Exception:
        # 안전 폴백(절대 500 안 내고 최소 정보 제공)
        return {
            "selected": KPI.get("selected", 0) if isinstance(KPI, dict) else 0,
            "approved": KPI.get("approved", 0) if isinstance(KPI, dict) else 0,
            "published": KPI.get("published", 0) if isinstance(KPI, dict) else 0,
            "gate_required": int(GATE.get("gate_required", 15)) if isinstance(GATE, dict) else 15,
            "ts": int(time.time()),
            "selection_total": 0,
            "selection_approved": 0,
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
