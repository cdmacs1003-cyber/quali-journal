# -*- coding: utf-8 -*-
"""
QualiJournal Admin API + Lite Black UI (stable)
- Community / Keyword / Daily flows (sync & async)
- Async Task Manager (/api/tasks/*) + SSE stream
- Gate config API (GET/PATCH /api/config/gate_required)
- Report & Export (/api/report, /api/export/{md|csv})
- Log viewing (/api/logs/*)
- UTF-8 safe on Windows; non-breaking fallback if optional modules are missing.
"""

from __future__ import annotations
import os, sys, json, csv, io, shutil, subprocess, datetime as _dt, hashlib, asyncio, threading, secrets, time, re
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import Header
try:
    from dotenv import load_dotenv
    load_dotenv()
    MODE = (os.getenv("QUALI_DB_MODE") or "local").lower().strip()
    # 예: local | cloud | test
    from admin.db import make_engine
    _engine = None

except Exception:
    # dotenv is optional at runtime; skip if missing
    pass
from fastapi import FastAPI, Query, Response, HTTPException, Depends, Body, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ==== INSERT START: EnrichReq model ====
class EnrichReq(BaseModel):
    date: Optional[str] = None
    keyword: Optional[str] = None
    mode: Optional[str] = "keyword"  # "keyword" or "selection"
    items: Optional[List[Dict[str, Any]]] = None
# ==== INSERT END: EnrichReq model ====

# [ADD] ─────────────────────────────────────────────────────────────────
class GatePatch(BaseModel):
    gate_required: int
# 전역 기본값(없으면 생성)
try:
    GATE
except NameError:
    
    GATE = {"gate_required": int(os.getenv("GATE_REQUIRED", "3"))}

# KPI 집계가 아직 없으면 0으로 응답
KPI = {"selected": 0, "approved": 0, "published": 0}

# --- [ADD] ForwardRef 안전장치 & 요청모델 정의 ---------------------------------
class PublishOneReq(BaseModel):
    approve: bool = Field(default=True, description="승인 여부")
    editor_note: Optional[str] = Field(default=None, description="편집장 한마디(선택)")
# ------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Auth (optional) — if auth_utils is missing, fall back to open access
# ---------------------------------------------------------------------------
try:
    from auth_utils import verify_jwt_token, require_two_factor  # type: ignore
except Exception:  # pragma: no cover
    async def verify_jwt_token(*args, **kwargs):  # type: ignore
        return {}
    async def require_two_factor(*args, **kwargs):  # type: ignore
        return {}

# ---------------------------------------------------------------------------
# Authorization (simple token based) — protects selected endpoints when API_TOKEN is set
# ---------------------------------------------------------------------------
# Use HTTPBearer to extract Authorization header in the form "Bearer <token>". If API_TOKEN
# environment variable is set, incoming requests must provide a matching token. Otherwise
# authorization is skipped. The dependency can be injected via Depends(authorize).
security = HTTPBearer(auto_error=False)

def _expected_tokens() -> set[str]:
    """환경변수(API_TOKEN, ADMIN_TOKEN)에 설정된 어느 것이든 허용"""
    vals = [os.getenv("API_TOKEN"), os.getenv("ADMIN_TOKEN")]
    return {v.strip() for v in vals if v}

async def authorize(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # ADMIN_TOKEN 또는 API_TOKEN 중 하나라도 일치하면 통과
    expected_admin = (os.environ.get("ADMIN_TOKEN") or "").strip()
    expected_api   = (os.environ.get("API_TOKEN") or "").strip()
    if not expected_admin and not expected_api:
        return True  # 둘 다 없으면 인증 생략
    supplied = credentials.credentials if credentials else None
    if not supplied or supplied not in (expected_admin, expected_api):
        raise HTTPException(status_code=401, detail="invalid or missing token")
    return True

# ---------------------------------------------------------------------------
# SSE / Bearer 인증 헬퍼
# ---------------------------------------------------------------------------
def _auth_header_or_qs_ok(request: Request) -> bool:
    """
    헤더를 보낼 수 없는 SSE 연결을 위해 ?token= 쿼리로도 인증을 허용합니다.
    - ADMIN_TOKEN 또는 API_TOKEN이 설정되어 있지 않으면 누구나 접근 가능(open mode)
    - 값이 있으면 Authorization Bearer 또는 ?token= 중 하나라도 일치해야 통과
    """
    expected = [ (os.environ.get("ADMIN_TOKEN") or "").strip(), (os.environ.get("API_TOKEN") or "").strip() ]
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
    컨테이너 빌드 컨텍스트가 admin/이든 repo/이든 상관없이
    orchestrator.py가 실제로 존재하는 상위 폴더를 자동 탐지한다.
    우선순위: admin/..  → admin/  → admin/../..
    """
    cands = [BASE.parent, BASE, BASE.parent.parent]
    for r in cands:
        if (r / "orchestrator.py").exists():
            return r
    return BASE.parent  # 최후의 기본값

ROOT = _detect_root()                   # repo root로 보정
ARCHIVE = ROOT / "archive"
TOOLS   = ROOT / "tools"
ORCH    = ROOT / "orchestrator.py"

SEL_COMM = ROOT / "selected_community.json"

SEL_WORK = ROOT / "data" / "selected_keyword_articles.json"
SEL_PUB  = ROOT / "selected_articles.json"

# Directory for summary markdown outputs.  Summarization endpoints will write
# enriched summary files here.  It lives under ``archive/enriched`` relative
# to the project root.  If the directory does not exist, it will be created
# automatically when a summary is generated.
ENRICHED_DIR = ARCHIVE / "enriched"
CAND_COMM = [SEL_COMM, ROOT / "archive" / "selected_community.json"]

INDEX_HTML = BASE / "index.html"
INDEX_LITE = BASE / "index_lite_black.html"

CONFIG_FILE = ROOT / "config.json"

# --- Cloud Run(/tmp) 지원 ---
ARCHIVE_CLOUD = Path(os.getenv("ARCHIVE_DIR", "/tmp/archive"))  # Cloud Run 기본 쓰기 위치
IS_CLOUD = bool(os.getenv("K_SERVICE"))                          # Cloud Run 여부

# 공통 사용 디렉터리
REPORT_DIR = (ARCHIVE_CLOUD / "reports") if IS_CLOUD else (BASE / "archive" / "reports")
ENRICH_DIR = (ARCHIVE_CLOUD / "enriched") if IS_CLOUD else (ARCHIVE / "enriched")

# 보장
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ENRICH_DIR.mkdir(parents=True, exist_ok=True)


ARCHIVE_ADMIN = BASE / "archive"   # admin\archive (보고서 .md 저장되는 곳)

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
# App ------------------------------------------------------------
app = FastAPI(title="QualiJournal Admin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# --- Static mounts (브라우저에서 /archive/* 바로 열기) ---
try:
    _paths = [getattr(r, "path", None) for r in getattr(app, "routes", [])]

    # /archive/reports → REPORT_DIR(로컬=admin/archive/reports, Cloud Run=/tmp/archive/reports)
    if "/archive/reports" not in _paths:
        app.mount("/archive/reports", StaticFiles(directory=str(REPORT_DIR)), name="archive-reports")

    # /archive/enriched → ENRICH_DIR(로컬=root/archive/enriched, Cloud Run=/tmp/archive/enriched)
    if "/archive/enriched" not in _paths:
        app.mount("/archive/enriched", StaticFiles(directory=str(ENRICH_DIR)), name="archive-enriched")

    # 루트 /archive 브라우징(선택)
    if "/archive" not in _paths:
        root_for_browse = (ARCHIVE_CLOUD if IS_CLOUD else BASE / "archive")
        root_for_browse.mkdir(parents=True, exist_ok=True)
        app.mount("/archive", StaticFiles(directory=str(root_for_browse)), name="archive-root")
except Exception:
    pass


# ✅ Cloud Run 헬스체크
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": True}

@app.get("/api/db/mode")
def db_mode():
    return {"mode": MODE}

# ---------------------------------------------------------------------------
# Task response models and helpers
# ---------------------------------------------------------------------------
# When returning recent tasks, we use Pydantic models to validate response
# structures. The ``id`` is a string (filename stem) and ``size`` is an
# integer representing the file size in bytes. The ``TasksRecent`` model
# wraps a list of ``TaskItem`` instances. A helper ``_task_log_dir``
# guarantees that the ``logs/tasks`` directory exists.
class TaskItem(BaseModel):
    id: str
    size: int

class TasksRecent(BaseModel):
    items: List[TaskItem]

def _task_log_dir() -> Path:
    """Return the directory where task logs are stored and ensure it exists."""
    d = (ROOT / "logs" / "tasks")
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# JSON helpers (BOM tolerant / safe write)
# ---------------------------------------------------------------------------
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
    """
    기사 객체에 안정적인 고유 ID를 보장한다.
    - 이미 id가 있으면 그대로 반환
    - 없으면 url > link > title 을 기반으로 MD5 해시를 만들어 부여
    """
    if not isinstance(item, dict):
        return ""
    if item.get("id"):
        return str(item["id"])
    base = (item.get("url") or item.get("link") or item.get("title") or "").strip()
    if not base:
        # 최후 수단: 정렬된 JSON 문자열 전체를 해싱(안정성)
        base = json.dumps(item, ensure_ascii=False, sort_keys=True)
    h = hashlib.md5(base.encode("utf-8", "ignore")).hexdigest()
    item["id"] = h
    return h

# ---------------------------------------------------------------------------
# Summary generation helpers
# ---------------------------------------------------------------------------
def _generate_summary_md(date: str, keyword: str, articles: List[dict], *, selected: bool = False) -> Path:
    """Generate a Markdown summary file from a list of articles.

    Parameters
    ----------
    date: str
        The date used in the filename (YYYY-MM-DD).
    keyword: str
        Keyword slug for naming.  If empty, the slug will be omitted.
    articles: List[dict]
        List of article objects containing at least title, url/link, summary, ko_summary, desc,
        and editor_note fields.
    selected: bool, optional
        When True, indicates the summary is for the selected articles.  The filename
        will include ``_selected``; otherwise ``_all``.  Defaults to False.

    Returns
    -------
    Path
        Absolute Path to the generated Markdown file.
    """
    # Determine slug and suffix for the filename.  Use the helper to slugify the keyword.
    slug = _slug_kw(keyword or "").strip()
    suffix = "selected" if selected else "all"
    # Build filename: date_keyword_suffix.md.  Omit keyword section if blank.
    if slug:
        fname = f"{date}_{slug}_{suffix}.md"
    else:
        fname = f"{date}_{suffix}.md"
    # Ensure output directory exists
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ENRICHED_DIR / fname
    # Compose Markdown content
    header_kw = slug.replace("_", " ") if slug else ""
    header_suffix = "선정본" if selected else "전체"
    title_parts = ["QualiNews", date]
    if header_kw:
        title_parts.append(header_kw)
    title_parts.append(f"({header_suffix})")
    lines: List[str] = [" — ".join(title_parts), ""]
    # Append each article with details
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

# Orchestrator runner (UTF-8 safe)
def _run_orch(*args: str) -> dict:
    """
    orchestrator.py 실행기(UTF-8 안전)
    - 컨테이너 내 경로 자동 탐지 결과(ORCH)가 없으면 tools/ 또는 ROOT에서 재탐색
    - 그래도 없으면 친절한 에러 반환(이미지 포함 누락 안내)
    """
    py  = sys.executable or "python"
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # 1) 스크립트 후보 찾기
    script = None
    if (ORCH).exists():
        script = ORCH
    elif (TOOLS / "orchestrator.py").exists():
        script = TOOLS / "orchestrator.py"
    elif (ROOT / "orchestrator.py").exists():
        script = ROOT / "orchestrator.py"

    # 2) 없으면 에러(이미지 빌드 범위 점검 힌트 포함)
    if not script:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "orchestrator.py not found in image. (빌드 컨텍스트를 repo 루트로 잡았는지 확인하세요)",
            "cmd": f"{py} orchestrator.py {' '.join(args)}"
        }

    # 3) 실행
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


# ---------------------------------------------------------------------------
# Tools runner (UTF-8 safe)
# ---------------------------------------------------------------------------
def _run_py(script_name: str, args: List[str] | None = None):
    """
    UTF-8 safe subprocess runner for tools/*.py or project-root scripts.
    Looks for script in TOOLS or project root and executes it with optional args.
    Returns (rc, stdout, stderr).
    """
    candidates = [TOOLS / script_name, ROOT / script_name]
    target = next((p for p in candidates if p.exists()), None)
    if not target:
        return 127, "", f"{script_name} not found"
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cp = subprocess.run(
        [sys.executable or "python", str(target), * (args or [])],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    return cp.returncode, cp.stdout, cp.stderr

# ---------------------------------------------------------------------------
# Community helpers
# ---------------------------------------------------------------------------
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
    py = sys.executable or "python"
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
        # persistent log file path. Use a dedicated tasks log directory.  
        # Store logs to disk so they survive server restarts.  
        global TASK_LOG_DIR  # defined below
        try:
            TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_file: Path | None = TASK_LOG_DIR / f"{self.id}.log"
            # ensure file exists
            self.log_file.write_text("", encoding="utf-8")
        except Exception:
            # if file cannot be created, fallback to None
            self.log_file = None

    def append(self, line: str):
        with self._lock:
            ts = _dt.datetime.now().strftime("%H:%M:%S")
            msg = f"[{ts}] {line}"
            self.logs.append(msg)
            # also persist to disk
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
        # use current Python interpreter or fallback
        py = sys.executable or "python3"
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

# request model
class FlowReq(BaseModel):
    kind: str            # daily|community|keyword
    keyword: str | None = None

@app.post("/api/tasks/flow")
def create_flow(req: FlowReq):
    kind = (req.kind or "").lower().strip()
    args = [req.keyword] if kind == "keyword" else []
    t = Task(kind, args)
    TM.add(t)
    th = threading.Thread(target=_run_task, args=(t,), daemon=True)
    th.start()
    return {"job_id": t.id, "status": t.status, "kind": t.kind, "args": t.args}

# ---------------------------------------------------------------------------
# Recent tasks API – lists persisted log files in logs/tasks directory
# Register this route before the parameterized `/api/tasks/{job_id}` to avoid
# path conflicts (e.g. ``recent`` being interpreted as a job_id).
# ---------------------------------------------------------------------------
@app.get("/api/tasks/recent", response_model=TasksRecent)
def tasks_recent(limit: int = Query(10, ge=1, le=50)) -> TasksRecent:
    """
    Return a list of recent task logs. Each item represents a log file saved
    in ``logs/tasks`` directory. The ``id`` field is the filename stem and
    ``size`` is the file size in bytes. If an exception occurs, an empty
    list is returned. Pydantic response model ensures proper typing and
    prevents runtime validation errors.
    """
    try:
        d = _task_log_dir()
        files = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = [TaskItem(id=p.stem, size=p.stat().st_size) for p in files[:limit]]
        return TasksRecent(items=items)
    except Exception:
        return TasksRecent(items=[])

@app.get("/api/db/ping")
def db_ping():
    global _engine
    if MODE != "cloud":
        return {"ok": True, "mode": MODE, "db": "not-connected"}
    if _engine is None:
        _engine = make_engine()
    with _engine.connect() as c:
        c.exec_driver_sql("SELECT 1;")
    return {"ok": True, "mode": "cloud", "db": "postgresql"}



@app.get("/api/tasks/{job_id}")
def get_task(job_id: str):
    t = TM.get(job_id)
    if not t: raise HTTPException(404, "job not found")
    return {
        "id": t.id,
        "kind": t.kind,
        "status": t.status,
        "created_at": t.created_at,
        "started_at": t.started_at,
        "ended_at": t.ended_at,
        "exit_code": t.exit_code,
        "lines": len(t.logs),
        # include path to the persistent log file if available
        "log_file": str(getattr(t, "log_file", "")) if getattr(t, "log_file", None) else None,
    }

@app.post("/api/tasks/{job_id}/cancel")
def cancel_task(job_id: str):
    t = TM.get(job_id)
    if not t: raise HTTPException(404, "job not found")
    t._cancel = True; return {"ok": True}

# ---------------------------------------------------------------------------
# Recent tasks API – lists persisted log files in logs/tasks directory
# Uses Pydantic models for response validation. See `_task_log_dir()` helper below.
# ---------------------------------------------------------------------------

## NOTE: tasks_recent is defined above create_flow to ensure it is registered before
##       the parameterized route /api/tasks/{job_id}. See insertion further above.

@app.get("/api/tasks/{job_id}/stream")
async def stream_task(job_id: str, request: Request):
    """
    SSE 스트림 엔드포인트. 헤더를 전송할 수 없는 EventSource를 위해 ?token= 허용.
    인증은 `_auth_header_or_qs_ok`에서 처리됩니다.
    """
    _auth_header_or_qs_ok(request)
    t = TM.get(job_id)
    if not t:
        raise HTTPException(404, "job not found")

    async def _gen():
        idx = 0
        while True:
            if idx < len(t.logs):
                chunk = "\n".join(t.logs[idx:]); idx = len(t.logs)
                yield f"data: {chunk}\n\n"
            if t.status in ("done", "error", "canceled"):
                yield f"event: end\ndata: {t.status}\n\n"; break
            await asyncio.sleep(0.5)

    return StreamingResponse(_gen(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Gate config (GET/PATCH) — features.gate_required (keep backward compatibility)
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_cfg(obj: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Report / Export
# ---------------------------------------------------------------------------
@app.get("/api/report")
def get_report(date: str | None = None):
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

# ----------------------------------------------------------------------------
# 추가: 보고서 작성 및 카드 요약/번역 엔드포인트

class ReportReq(BaseModel):
    """요청 본문: date(선택), keyword(선택)"""
    date: str | None = None
    keyword: str | None = None

@app.post("/api/report")
def post_report(payload: dict | None = Body(default=None)):
    """
    안전모드: 본문이 없어도 동작하며, 외부 스크립트를 기다리지 않고
    로컬 JSON을 읽어 즉시 Markdown 보고서를 생성합니다.
    """
    root = Path(__file__).resolve().parent
    reports_dir = REPORT_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 날짜/키워드 추론(본문이 없으면 오늘 날짜, 키워드는 JSON에서 추출)
    date = (payload or {}).get("date") or _dt.date.today().isoformat()

    # 1) 데이터 소스 우선순위(있는 것만 사용)
    candidates = [
        root / "data" / "selected_keyword_articles.json",  # items 배열
        root / "selected_articles.json",                   # 리스트
        root / "data_selected_articles.json",              # 리스트
    ]
    items = []
    keyword = "report"

    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "")).strip("_") or "report"

    for p in candidates:
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                # 포맷 자동 인식
                if isinstance(obj, dict) and "items" in obj:
                    items = obj.get("items") or []
                    keyword = _slug(obj.get("keyword") or keyword)
                elif isinstance(obj, list):
                    items = obj
                break
            except Exception:
                pass

    # 2) Markdown 생성
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

    # 3) 파일 저장
    out = reports_dir / f"{date}_{keyword}_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    rel = f"archive/reports/{out.name}"
    return {"ok": True, "path": rel, "count": len(items)}

# --- Enrich (키워드/선정본 요약) for Admin UI ---
@app.post("/api/enrich/keyword")
def enrich_keyword(req: EnrichReq, authorized: bool = Depends(authorize)):
    root = Path(__file__).resolve().parent
    items, date, kw_in = _read_any_items(root)
    date = req.date or date or _dt.date.today().isoformat()
    kw   = (req.keyword or kw_in or "report")
    arts = items or []
    out_path = _generate_summary_md(date, kw, arts, selected=False)
    web_path = f"archive/enriched/{out_path.name}"
    return {"ok": True, "path": web_path, "count": len(arts)}

@app.post("/api/enrich/selection")
def enrich_selection(req: EnrichReq, authorized: bool = Depends(authorize)):
    root = Path(__file__).resolve().parent
    items, date, kw_in = _read_any_items(root)
    date = req.date or date or _dt.date.today().isoformat()
    kw   = (req.keyword or kw_in or "report")

    def _is_selected(a: dict) -> bool:
        s = (a.get("state") or "").lower()
        return bool(a.get("selected") or a.get("approved") or s in ("published","ready"))

    arts = [a for a in (items or []) if _is_selected(a)]
    out_path = _generate_summary_md(date, kw, arts, selected=True)
    web_path = f"archive/enriched/{out_path.name}"
    return {"ok": True, "path": web_path, "count": len(arts)}



    


@app.get("/api/export/{fmt}")
def export_fmt(fmt: str, date: str | None = None, authorized: bool = Depends(authorize)):
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
            headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
            return Response(content=md, media_type="text/markdown; charset=utf-8", headers=headers)
        elif fmt_lower == "csv":
            buf = io.StringIO(); w = csv.writer(buf)
            w.writerow(["title","url","source","date","score","approved","editor_note","summary"])
            for a in articles:
                w.writerow([
                    a.get("title",""), a.get("url") or a.get("link") or "",
                    a.get("source",""), a.get("date",""),
                    a.get("score",""), a.get("approved",""),
                    a.get("editor_note",""), a.get("summary") or a.get("ko_summary") or a.get("desc","")
                ])
            csv_text = buf.getvalue()
            data_out  = "\ufeff" + csv_text  # add BOM
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
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["title","url","source","upvotes","comments","views"])
        for a in arts:
            w.writerow([a.get("title",""), a.get("url",""), a.get("source",""),
                        a.get("upvotes",0), a.get("comments",0), a.get("views","")])
        data = "\ufeff" + buf.getvalue()  # UTF-8 BOM
        return Response(content=data, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="community_{day}.csv"'})
    raise HTTPException(400, "unsupported format")

# ---------------------------------------------------------------------------
# Community / Save / Status / Publish endpoints (sync compatible)
# ---------------------------------------------------------------------------
class SaveItem(BaseModel):
    id: str
    approved: bool
    editor_note: str = ""

class SavePayload(BaseModel):
    changes: List[SaveItem]

@app.get("/api/community")
def api_get_community(only_pending: bool = Query(True)):
    snap = _get_community_snapshot()
    arts = snap["articles"]
    if only_pending: arts = [a for a in arts if not a.get("approved")]
    total = len(snap["articles"]); approved = sum(1 for a in snap["articles"] if a.get("approved"))
    return {"date": snap["date"], "keyword": snap.get("keyword",""), "total": total,
            "approved": approved, "pending": total-approved, "articles": arts}

@app.post("/api/community/save")
def api_save(payload: SavePayload):
    target = SEL_COMM if SEL_COMM.exists() else SEL_WORK
    obj = _read_json(target); items = obj.get("articles", []); idx = {_ensure_id(a): a for a in items}
    changed = 0
    for c in payload.changes:
        row = idx.get(c.id)
        if not row: continue
        if row.get("approved") != c.approved or (row.get("editor_note","") != c.editor_note):
            row["approved"] = c.approved; row["editor_note"] = c.editor_note; changed += 1
    if changed: _write_json(target, obj)
    sync = _sync_after_save()
    return {"saved": changed, "synced": sync.get("ok", False), "sync_log": sync}

@app.get("/api/items")
def api_items(state: str = Query("ready"), date: str | None = None, keyword: str | None = None):
    """
    state=candidate|ready|rejected|published
    date/keyword는 필터 힌트(현재 파일 구조에선 주로 상태 필터 사용)
    """
    snap = _get_work_snapshot()
    s = (state or "").lower().strip()
    arts = snap["articles"]
    if s:
        if s == "published":
            # 발행본은 selected_articles.json에서 조회
            pub = _read_json(SEL_PUB) or {}
            items = pub.get("articles", []) or []
            return {"date": pub.get("date", snap["date"]), "keyword": snap.get("keyword",""), "state": s, "items": items}
        items = [a for a in arts if (a.get("state","").lower() == s)]
    else:
        items = arts
    return {"date": snap["date"], "keyword": snap.get("keyword",""), "state": s or "all", "items": items}

@app.post("/api/items/{item_id}/publish")
def api_items_publish(item_id: str, req: PublishOneReq):
    """
    단일 아이템 발행:
    - work 파일에서 해당 id 기사 찾기 → approved/selected/editor_note 업데이트
    - publish 파일로 동기화(_sync_after_save)
    """
    work = _read_json(SEL_WORK) or {}
    arts = work.get("articles", []) or []
    idx = None
    for i, a in enumerate(arts):
        try:
            if _ensure_id(a) == item_id:
                idx = i; break
        except Exception:
            continue
    if idx is None:
        raise HTTPException(status_code=404, detail="item not found")

    a = arts[idx]
    a["approved"] = bool(req.approve)
    a["selected"] = True
    a["editor_note"] = req.editor_note or a.get("editor_note","")
    # 'published' 상태 표시는 선택적(리스트 필터 용이). 실제 발행 파일 반영은 sync에서 처리.
    a["state"] = "published"

    work["articles"] = arts
    _write_json(SEL_WORK, work)

    sync = _sync_after_save()
    return {"ok": True, "synced": sync.get("ok", False), "item_id": item_id}

class PublishReq(BaseModel):
    keyword: str

@app.post("/api/publish-keyword")
def api_publish(req: PublishReq):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    rollover = _rollover_archive_if_needed(req.keyword)
    out = _run_orch("--publish-keyword", req.keyword)
    outputs = _latest_published_paths(req.keyword)
    return {**out, "rolled_over": rollover or [], "created": outputs}

# Legacy sync flows (kept for compatibility)
@app.post("/api/flow/community")
def api_flow_comm():
    return _run_orch("--collect-community")

@app.post("/api/flow/daily")
def api_flow_daily():
    steps = [_run_orch("--collect-community"), _run_orch("--publish-community","--format","all"), _run_orch("--publish","--format","all")]
    ok = all(s.get("ok", True) for s in steps)
    return {"ok": ok, "steps": steps}

class FlowKwReq(BaseModel):
    keyword: str
    use_external_rss: bool = False

@app.post("/api/flow/keyword")
def api_flow_keyword(req: FlowKwReq):
    steps = []
    if req.use_external_rss:
        steps.append(_run_orch("--collect-keyword", req.keyword, "--use-external-rss"))
    else:
        steps.append(_run_orch("--collect-keyword", req.keyword))
    steps.append(_run_orch("--approve-keyword-top","20","--approve-keyword",req.keyword))
    steps.append(_sync_after_save())
    steps.append(_run_orch("--publish-keyword", req.keyword))
    ok = all(s.get("ok", True) for s in steps)
    return {"ok": ok, "steps": steps}

# ---------------------------------------------------------------------------
# Log viewing / download
# ---------------------------------------------------------------------------
@app.get("/api/logs")
def list_logs(user: dict = Depends(verify_jwt_token)):
    logs_dir = ROOT / "logs"; items = []
    if logs_dir.exists() and logs_dir.is_dir():
        for p in logs_dir.iterdir():
            if p.is_file() and p.suffix == ".log":
                try:
                    stat = p.stat()
                    items.append({"name": p.name, "size": stat.st_size,
                                  "modified": _dt.datetime.utcfromtimestamp(stat.st_mtime).isoformat()+"Z"})
                except Exception: continue
    return {"items": items}

@app.get("/api/logs/{log_name}")
def get_log(log_name: str, lines: int = 200, user: dict = Depends(verify_jwt_token)):
    path = ROOT / "logs" / log_name
    if not path.exists() or not path.is_file(): raise HTTPException(404, "log not found")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines_data = f.readlines()
        content = "".join(lines_data[-int(lines):]) if (lines and lines > 0) else "".join(lines_data)
    except Exception as e:
        raise HTTPException(500, str(e))
    return PlainTextResponse(content, media_type="text/plain")

@app.get("/api/logs/{log_name}/download")
def download_log(log_name: str, user: dict = Depends(verify_jwt_token)):
    path = ROOT / "logs" / log_name
    if not path.exists() or not path.is_file(): raise HTTPException(404, "log not found")
    return FileResponse(str(path), filename=log_name, media_type="text/plain")

@app.get("/api/config/gate_required")
async def get_gate_required():
    return {"gate_required": int(GATE.get("gate_required", 3))}

@app.patch("/api/config/gate_required")
async def set_gate_required(p: GatePatch):
    v = max(1, min(100, int(p.gate_required)))  # 1~100로 보호
    GATE["gate_required"] = v
    return {"ok": True, "gate_required": v}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def devtools_config():
    return Response(content="{}", media_type="application/json", status_code=200)

@app.get("/favicon.ico")
def favicon_blank():
    return Response(status_code=204)

@app.get("/")
def index():
    p = INDEX_HTML if INDEX_HTML.exists() else (INDEX_LITE if INDEX_LITE.exists() else None)
    if p: return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>QualiJournal Admin</h1><p>index.html이 없습니다.</p>")

# ---------------------------------------------------------------------------
# Approve UI launcher (protected)
# ---------------------------------------------------------------------------
@app.post("/api/approve-ui/start")
def approve_ui_start(request: Request, authorized: bool = Depends(authorize)):
    """
    관리자 승인 UI를 열기 위한 엔드포인트.
    - 기본: 현재 호스트의 루트에 panel=approve 쿼리로 이동
    - 환경변수 APPROVE_UI_URL 이 있으면 해당 URL을 반환
    """
    base = str(request.base_url).rstrip("/")
    ui_url = os.getenv("APPROVE_UI_URL")
    if not ui_url:
        panel = os.getenv("APPROVE_UI_PANEL", "approve")
        ui_url = f"{base}/?panel={panel}&ts={int(time.time())}"
    return {"ok": True, "ui_url": ui_url}

# [ADD] KPI 현황 조회 (UI가 30초마다 호출)
@app.get("/api/status")
def get_status(date: str | None = None, keyword: str | None = None, authorized: bool = Depends(authorize)):
    """
    KPI 및 현황을 확장하여 반환합니다.

    반환 필드:
      - selected/approved/published: 기본 KPI 수치 (legacy)
      - gate_required: 게이트 임계값
      - ts: 현재 타임스탬프
      - selection_total: 작업 기사 총수
      - selection_approved: 승인(또는 발행/ready) 상태 기사 수
      - state_counts: 상태별 기사 수 (candidate/ready/rejected)
      - community_total: 커뮤니티 기사 수
      - keyword_total: 작업 기사 총수(동일)
      - gate_pass: selection_approved >= gate_required 여부
      - date/keyword: 현재 스냅샷 날짜, 키워드
    """
    work = _get_work_snapshot()
    arts = work.get("articles", []) or []
    selection_total = len(arts)
    state_counts = {"candidate": 0, "ready": 0, "rejected": 0}
    selection_approved = 0
    for a in arts:
        st = (a.get("state") or "candidate").lower()
        if st in state_counts:
            state_counts[st] += 1
        # approved/selected or ready/published counts towards approved
        if a.get("approved") or a.get("selected") or st in ("ready", "published"):
            selection_approved += 1
    comm = _get_community_snapshot()
    community_total = len(comm.get("articles", []) or [])
    keyword_total = selection_total
    gate_required = int(GATE.get("gate_required", 3))
    gate_pass = bool(selection_approved >= gate_required)
    return {
        "selected": KPI.get("selected", 0),
        "approved": KPI.get("approved", 0),
        "published": KPI.get("published", 0),
        "gate_required": gate_required,
        "ts": int(time.time()),
        "selection_total": selection_total,
        "selection_approved": selection_approved,
        "state_counts": state_counts,
        "community_total": community_total,
        "keyword_total": keyword_total,
        "gate_pass": gate_pass,
        "date": date or work.get("date"),
        "keyword": keyword or work.get("keyword", ""),
    }
def _read_any_items(root: Path):
    """
    우선순위:
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

@app.get("/api/export/csv")
def export_csv(authorized: bool = Depends(authorize)):
    """Backward-compat endpoint → 표준 라우트(/api/export/{fmt})로 위임."""
    # 위임: 구형 export() 대신 export_fmt 호출
    return export_fmt(fmt="csv", authorized=authorized)


    # CSV 만들기 (엑셀 호환 BOM 포함)
    buf = io.StringIO(newline="")
    fieldnames = ["title","url","summary_en","summary_ko","editor_note","source","date","score","approved","state"]
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for it in items:
        w.writerow({
            "title": _get(it,"title","headline"),
            "url": _get(it,"url","link"),
            "summary_en": _get(it,"summary_en","summary"),
            "summary_ko": _get(it,"summary_ko","summary_kr"),
            "editor_note": _get(it,"editor_note"),
            "source": _get(it,"source","origin"),
            "date": _get(it,"date","published_at"),
            "score": _get(it,"score"),
            "approved": _get(it,"approved"),
            "state": _get(it,"state"),
        })
    csv_text = buf.getvalue()
    data = ("\ufeff" + csv_text).encode("utf-8")  # BOM
    today = (date or _dt.date.today().isoformat())
    fn = f"QualiNews_{today}_{(kw or 'report')}.csv"
    return Response(content=data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{fn}"'})
# ==== INSERT END: /api/export/csv (safe-export) ====

if __name__ == "__main__":
    # Local run helper (for dev/testing): respects Cloud Run-style PORT
    import uvicorn, os as _os
    _port = int(_os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=_port, log_level="info")
