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
except Exception:
    # dotenv is optional at runtime; skip if missing
    pass
from fastapi import FastAPI, Query, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

class TaskItem(BaseModel):
    id: str           # 파일명 stem 등 문자열 ID
    size: int         # 바이트 단위 파일 크기

class TasksRecent(BaseModel):
    items: List[TaskItem]

def _task_log_dir() -> Path:
    d = (ROOT / "logs" / "tasks")
    d.mkdir(parents=True, exist_ok=True)
    return d
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
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="QualiJournal Admin API")

# --- 디버그: 오케스트레이터/설정/피드 파일 존재 여부 확인 ---
@app.get("/api/debug/files")
def debug_files() -> Dict[str, str | bool]:
    feeds_dir = ROOT / "feeds"
    community_json = feeds_dir / "community_sources.json"
    return {
        "cwd": os.getcwd(),
        "ROOT": str(ROOT),
        "ORCH": str(ORCH),
        "ORCH_exists": ORCH.exists(),
        "CONFIG_FILE": str(CONFIG_FILE),
        "CONFIG_exists": CONFIG_FILE.exists(),
        "feeds_dir": str(feeds_dir),
        "feeds_dir_exists": feeds_dir.exists(),
        "community_sources.json": str(community_json),
        "community_sources_exists": community_json.exists(),
    }

# --- ensure task log dir exists (for /api/tasks/*) ---
try:
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)  # logs/tasks
except Exception:
    pass

# --- safe recent tasks endpoint (never 404, return empty list) ---

@app.get("/api/tasks/recent", response_model=TasksRecent)
def tasks_recent(limit: int = Query(10, ge=1, le=50)) -> TasksRecent:
    """
    최근 작업 로그 파일 목록을 반환합니다.
    - id: 로그 파일명 stem (문자열)
    - size: 파일 크기 (정수)
    - 예외가 나도 500 대신 빈 배열을 반환합니다.
    """
    try:
        d = _task_log_dir()
        files = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = [TaskItem(id=p.stem, size=p.stat().st_size) for p in files[:limit]]
        return TasksRecent(items=items)
    except Exception:
        return TasksRecent(items=[])


# ---------------------------------------------------------------------------
# Health and root endpoints (Cloud Run friendly)
# ---------------------------------------------------------------------------
# NOTE:
# - Ensure there is NO other `@app.get("/")` in this file to avoid duplicate route errors.
# - `/health` is used by probes; it must be fast and side-effect free.

# CORS (add once, keeps things simple for admin UI / local tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/health", tags=["system"])
async def health() -> Dict[str, bool]:  # pragma: no cover
    """Liveness check for Cloud Run / health probe."""
    return {"status": True}

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
    h = hashlib.md5((item.get("url") or item.get("title","")).encode("utf-8")).hexdigest()
    item["id"] = h
    return h

# ---------------------------------------------------------------------------
# Orchestrator runner (UTF-8 safe)
# ---------------------------------------------------------------------------
def _run_orch(*args: str) -> dict:
    py = sys.executable or "python"
    # 존재 확인(없으면 바로 원인 리턴)
    if not ORCH.exists():
        msg = f"[ORCH] not found: {ORCH}"
        logger.error(msg)
        return {"ok": False, "stdout": "", "stderr": msg, "cmd": f"{py} {ORCH} {' '.join(args)}"}

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        cp = subprocess.run(
            [py, "-u", str(ORCH), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=env,
        )
        # 표준출력/오류를 한 줄 요약으로 로깅
        so = (cp.stdout or "").strip()
        se = (cp.stderr or "").strip()
        logger.info("run orch cmd=%s rc=%s", " ".join([str(ORCH), *args]), cp.returncode)
        if se:
            logger.warning("orch stderr: %s", se[:500].replace("\n", " "))
        return {"ok": cp.returncode == 0, "stdout": so, "stderr": se, "cmd": " ".join([str(ORCH), *args])}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": f"{type(e).__name__}: {e}", "cmd": " ".join([str(ORCH), *args])}


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
        if task.kind == "daily":
            rc1 = run_cmd(["py", str(ORCH), "--collect-community"])
            rc2 = run_cmd(["py", str(ORCH), "--publish-community", "--format", "all"])
            rc3 = run_cmd(["py", str(ORCH), "--publish", "--format", "all"])
            rc = max(rc1, rc2, rc3)
        elif task.kind == "community":
            rc1 = run_cmd(["py", str(ORCH), "--collect-community"])
            rc2 = run_cmd(["py", str(ORCH), "--publish-community", "--format", "all"])
            rc = max(rc1, rc2)
        elif task.kind == "keyword":
            kw = task.args[0] if task.args else ""
            if not kw: raise RuntimeError("keyword required")
            rc1 = run_cmd(["py", str(ORCH), "--collect-keyword", kw])
            rc2 = run_cmd(["py", str(ORCH), "--approve-keyword", kw, "--approve-keyword-top", "15"])
            rc3 = run_cmd(["py", str(ORCH), "--publish-keyword", kw])
            rc = max(rc1, rc2, rc3)
        else:
            raise RuntimeError(f"unknown kind: {task.kind}")
        task.exit_code = rc; task.status = "done" if rc == 0 else "error"
        if rc != 0: task.append(f"! exit={rc}")
    except Exception as e:
        task.status = "error"; task.append(f"! error: {e}")
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
class GateReq(BaseModel):
    value: int

def _load_cfg() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_cfg(obj: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

@app.get("/api/config/gate_required")
def get_gate_required():
    cfg = _load_cfg()
    val = (cfg.get("features") or {}).get("gate_required")
    if val is None:
        val = cfg.get("gate_required", 15)  # backward compat
    return {"gate_required": int(val)}

@app.patch("/api/config/gate_required")
def patch_gate_required(req: GateReq):
    cfg = _load_cfg()
    feats = cfg.setdefault("features", {})
    feats["gate_required"] = int(req.value)
    # preserve flag if present
    feats["require_editor_approval"] = bool(feats.get("require_editor_approval", True))
    _save_cfg(cfg)
    return {"ok": True, "gate_required": feats["gate_required"]}

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
    """지정된 날짜와 키워드로 보고서를 생성하여 archive/reports/에 저장합니다."""
    day = req.date or _dt.date.today().isoformat()
    kw = req.keyword or ""
    name = f"{day}_{kw}".strip("_") + ".md"
    reports_dir = ARCHIVE / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / name
    try:
        content = f"# Report for {day} {kw}\n\n이 리포트는 자동 생성되었습니다.\n"
        path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(path)}
    except Exception as e:
        raise HTTPException(500, f"report error: {e}")

class EnrichReq(BaseModel):
    """요약/번역 요청: keyword(선택)"""
    keyword: str | None = None

@app.post("/api/enrich/keyword")
def enrich_keyword(req: EnrichReq, authorized: bool = Depends(authorize)):
    kw = (req.keyword or "").strip()
    name = f"enrich_keyword_{kw or _dt.date.today().isoformat()}.md"
    path = ARCHIVE / name
    try:
        content = f"# Summary for keyword {kw}\n\n요약과 번역 내용이 여기 표시됩니다.\n"
        path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(path)}
    except Exception as e:
        raise HTTPException(500, f"enrich keyword error: {e}")

@app.post("/api/enrich/selection")
def enrich_selection(req: EnrichReq, authorized: bool = Depends(authorize)):
    kw = (req.keyword or "").strip()
    name = f"enrich_selection_{kw or _dt.date.today().isoformat()}.md"
    path = ARCHIVE / name
    try:
        content = f"# Summary for selection {kw}\n\n선정본 요약과 번역 내용이 여기 표시됩니다.\n"
        path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(path)}
    except Exception as e:
        # If an error occurs when creating the selection summary, bubble up
        raise HTTPException(500, f"enrich selection error: {e}")

@app.get("/api/export/{fmt}")
def export_fmt(fmt: str, date: str | None = None, authorized: bool = Depends(authorize)):
    day = date or _dt.date.today().isoformat()
    cj = ARCHIVE / f"community_{day}.json"
    if not cj.exists():
        raise HTTPException(404, "community json not found")
    obj = _read_json(cj); arts = obj.get("articles", [])

    if fmt.lower() == "md":
        lines = [f"# Community — {day}", ""]
        for a in arts:
            title = a.get("title") or "(no title)"
            url = a.get("url") or "#"
            meta = f"👍{a.get('upvotes',0)} · 💬{a.get('comments',0)} · 👀{a.get('views','-')}"
            lines.append(f"- [{title}]({url})  \n  {meta} · {a.get('source','')}")
        md = "\n".join(lines) + "\n"
        return Response(content=md, media_type="text/markdown; charset=utf-8")

    if fmt.lower() == "csv":
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

@app.get("/api/status")
def api_status():
    work = _read_json(SEL_WORK); sel_total = len(work.get("articles", []))
    sel_approved = sum(1 for a in work.get("articles", []) if a.get("approved"))
    snap = _get_community_snapshot(); comm_total = len(snap.get("articles", []))
    cfg = _read_json(CONFIG_FILE)
    gate_required = int((cfg.get("features") or {}).get("gate_required") or cfg.get("gate_required") or 15)
    require_editor_approval = bool((cfg.get("features") or {}).get("require_editor_approval", True))
    return {"date": work.get("date", _dt.date.today().isoformat()),
            "keyword": work.get("keyword",""),
            "selection_total": sel_total, "selection_approved": sel_approved,
            "community_total": comm_total, "keyword_total": sel_total,
            "gate_required": gate_required, "require_editor_approval": require_editor_approval,
            "gate_pass": sel_approved >= gate_required,
            "paths": {"work": str(SEL_WORK), "publish": str(SEL_PUB), "community": "root_or_archive"}}

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


if __name__ == "__main__":
    # Local run helper (for dev/testing): respects Cloud Run-style PORT
    import uvicorn, os as _os
    _port = int(_os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=_port, log_level="info")
