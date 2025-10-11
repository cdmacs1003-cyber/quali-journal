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
import os, sys, json, csv, io, shutil, subprocess, datetime as _dt, hashlib, asyncio, threading, secrets, time
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
from fastapi import FastAPI, Query, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

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

async def authorize(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected = os.environ.get("API_TOKEN")
    # If no token configured, open access
    if not expected:
        return True
    # When API_TOKEN is set, require a Bearer token
    token = credentials.credentials if credentials else None
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="invalid or missing token")
    return True

# ---------------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent          # admin/
ROOT = BASE.parent                               # repo root
ARCHIVE = ROOT / "archive"
TOOLS = ROOT / "tools"
ORCH = ROOT / "orchestrator.py"
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
    if item.get("id"):
        return item["id"]

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
    h = hashlib.md5((item.get("url") or item.get("title","")).encode("utf-8")).hexdigest()
    item["id"] = h
    return h

# ---------------------------------------------------------------------------
# Orchestrator runner (UTF-8 safe)
# ---------------------------------------------------------------------------
def _run_orch(*args: str) -> dict:
    py = sys.executable or "python"
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cp = subprocess.run(
        [py, str(ORCH), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    try:
        logger.info("run orch: %s rc=%s", " ".join([str(ORCH), *args]), cp.returncode)
        if cp.stderr:
            logger.warning("orch stderr: %s", cp.stderr.strip().replace("\n", " ")[:200])
    except Exception:
        pass
    return {"ok": cp.returncode == 0, "stdout": cp.stdout, "stderr": cp.stderr, "cmd": " ".join([str(ORCH), *args])}

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
async def stream_task(job_id: str):
    t = TM.get(job_id)
    if not t: raise HTTPException(404, "job not found")

    async def _gen():
        idx = 0
        while True:
            if idx < len(t.logs):
                chunk = "\n".join(t.logs[idx:]); idx = len(t.logs)
                yield f"data: {chunk}\n\n"
            if t.status in ("done","error","canceled"):
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
def post_report(req: ReportReq, authorized: bool = Depends(authorize)):
    """
    Create a daily report by invoking make_daily_report.py.
    Returns path to the generated Markdown file.
    """
    day = req.date or _dt.date.today().isoformat()
    kw = (req.keyword or "").strip()
    # build args for script
    args: List[str] = []
    if req.date:
        args += ["--date", day]
    if kw:
        args += ["--keyword", kw]
    # run the report script
    rc, out, err = _run_py("make_daily_report.py", args)
    # determine report path: script prints the path on stdout
    if rc == 0:
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        report_path: Optional[str] = lines[-1] if lines else None
        # if script did not output path, build default candidate
        if not report_path:
            filename = f"{day}_{kw}".strip("_") + "_report.md"
            report_path = str((ARCHIVE / "reports" / filename))
        return JSONResponse({"ok": True, "path": report_path, "stdout": out, "stderr": err})
    else:
        return JSONResponse({"ok": False, "stderr": err}, status_code=500)

class EnrichReq(BaseModel):
    """요약/번역 요청: date, keyword(선택)"""
    date: str | None = None
    keyword: str | None = None

@app.post("/api/enrich/keyword")
def enrich_keyword(req: EnrichReq, authorized: bool = Depends(authorize)):
    """
    Enrich all collected articles for a given date/keyword.
    Invokes enrich_cards.py with --date and --keyword.
    """
    day = req.date or _dt.date.today().isoformat()
    kw  = (req.keyword or "").strip()
    args: List[str] = []
    if req.date:
        args += ["--date", day]
    if kw:
        args += ["--keyword", kw]
    rc, out, err = _run_py("enrich_cards.py", args)
    # After enrichment script runs, attempt to build a human-friendly summary
    summary_path: Optional[Path] = None
    try:
        # Determine which JSON file holds the enriched articles.
        # Prefer archive/{date}_{keyword}.json if it exists; otherwise fall back to
        # selected_keyword_articles.json to summarise all collected items.
        json_path: Optional[Path] = None
        if kw:
            # replace spaces with hyphens for filename
            fname = f"{day}_{kw.replace(' ', '-')}.json"
            cand = ARCHIVE / fname
            if cand.exists():
                json_path = cand
        # Fallback to the working keyword file.  Try multiple candidate locations
        if not json_path:
            # Prefer data/selected_keyword_articles.json
            if SEL_WORK.exists():
                json_path = SEL_WORK
            else:
                # Also consider top-level selected_keyword_articles.json if present
                root_kw = ROOT / "selected_keyword_articles.json"
                if root_kw.exists():
                    json_path = root_kw
        data = _read_json(json_path) if json_path else {}
        # Extract list of articles; if missing, default to empty list
        articles = []
        if isinstance(data, dict):
            articles = data.get("articles") or []
            # Some JSON structures might nest articles under sections
            if not articles and isinstance(data.get("sections"), dict):
                for v in data["sections"].values():
                    if isinstance(v, list):
                        articles = v
                        break
        # Determine keyword to use for filename; if not provided, fallback to data's keyword
        kw_for_file = kw or data.get("keyword") or ""
        summary_path = _generate_summary_md(date=day, keyword=kw_for_file, articles=articles, selected=False)
    except Exception as e:
        # Log error but proceed
        logger.error("enrich_keyword: summary generation failed: %s", e, exc_info=True)
        summary_path = None
    # Respond with path to summary markdown along with stdout/stderr
    return JSONResponse(
        {
            "ok": rc == 0,
            "path": str(summary_path) if summary_path else None,
            "stdout": out,
            "stderr": err,
        },
        # Always return a 200 status so that callers receive the summary file path even
        # if the underlying enrichment script fails.  The "ok" field reflects the
        # success or failure of the script itself.
        status_code=200,
    )

@app.post("/api/enrich/selection")
def enrich_selection(req: EnrichReq, authorized: bool = Depends(authorize)):
    """
    Enrich only the selected/published articles.
    Invokes enrich_cards.py with --selection pointing to selected_articles.json.
    """
    args = ["--selection", str(SEL_PUB)]
    rc, out, err = _run_py("enrich_cards.py", args)
    # Build a summary of only the selected/published articles
    summary_path: Optional[Path] = None
    try:
        # selected_articles.json holds the approved list.  Read and summarise it.
        # Load the selected articles JSON.  If ``SEL_PUB`` does not exist or is empty,
        # attempt to locate a fallback file (data/selected_articles.json or top-level).
        data = _read_json(SEL_PUB)
        if not data:
            # Try standard locations for selected_articles.json
            for pth in [ROOT / "selected_articles.json", ROOT / "data" / "selected_articles.json"]:
                if pth.exists():
                    data = _read_json(pth)
                    if data:
                        break
        articles = []
        if isinstance(data, dict):
            articles = data.get("articles") or []
            if not articles and isinstance(data.get("sections"), dict):
                for v in data["sections"].values():
                    if isinstance(v, list):
                        articles = v
                        break
        # Determine date and keyword for file naming.  Use request.date if provided,
        # otherwise attempt to get from data; fallback to today.
        sel_day = req.date or data.get("date") or _dt.date.today().isoformat()
        # Determine keyword: use request.keyword if provided or fallback to data's keyword.
        sel_kw = (req.keyword or "").strip() or data.get("keyword") or ""
        summary_path = _generate_summary_md(date=sel_day, keyword=sel_kw, articles=articles, selected=True)
    except Exception as e:
        logger.error("enrich_selection: summary generation failed: %s", e, exc_info=True)
        summary_path = None
    return JSONResponse(
        {
            "ok": rc == 0,
            "path": str(summary_path) if summary_path else None,
            "stdout": out,
            "stderr": err,
        },
        # Always return a 200 status so that clients can access the summary file even
        # when the enrichment script has a non-zero exit code.  Use the "ok" field
        # to indicate script success or failure.
        status_code=200,
    )

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

# [ADD] KPI 현황 조회 (UI가 30초마다 호출)
@app.get("/api/status")
async def get_status():
    return {
        "selected": KPI.get("selected", 0),
        "approved": KPI.get("approved", 0),
        "published": KPI.get("published", 0),
        "gate_required": GATE["gate_required"],
        "ts": int(time.time())
    }

if __name__ == "__main__":
    # Local run helper (for dev/testing): respects Cloud Run-style PORT
    import uvicorn, os as _os
    _port = int(_os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=_port, log_level="info")
