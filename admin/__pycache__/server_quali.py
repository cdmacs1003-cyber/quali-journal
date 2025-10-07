
# -*- coding: utf-8 -*-
"""
server_quali.py — QualiJournal Admin API + Unified UI (patched)
변경점
- (NEW) POST /api/enrich/selection  : tools/enrich_cards.py --selection
- (NEW) POST /api/enrich/keyword    : tools/enrich_cards.py --date --keyword
- (NEW) POST /api/report            : tools/make_daily_report.py --date [--keyword]
- (NEW) POST /api/export/md         : make_daily_report로 MD 리포트 생성 후 경로 반환
- (NEW) POST /api/export/csv        : selected_articles.json → CSV 내보내기
- (CHG) POST /api/flow/keyword      : 요청 바디의 date를 수용(현재 orchestrator는 내부 today 사용)
- GET "/"                           : admin_quali/index.html(본 파일과 같은 폴더) 우선 서빙
"""
from __future__ import annotations

import os, sys, json, csv, subprocess, datetime as _dt, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------- 경로 유틸 ----------
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
ARCHIVE = ROOT / "archive"
TOOLS = ROOT / "tools"
ORCH = ROOT / "orchestrator.py"
SEL_COMM = ROOT / "selected_community.json"
SEL_WORK = ROOT / "data" / "selected_keyword_articles.json"
SEL_PUB  = ROOT / "selected_articles.json"
CAND_COMM = [SEL_COMM, ROOT / "archive" / "selected_community.json"]

INDEX_HTML = (BASE / "index.html")  # 통합본

def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    raw = p.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    return json.loads(raw)

def _write_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def _slug_kw(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in s).strip("-").upper()

def _ensure_id(item: dict) -> str:
    if item.get("id"):
        return item["id"]
    h = hashlib.md5((item.get("url") or item.get("title","")).encode("utf-8")).hexdigest()
    item["id"] = h
    return h

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
        _ensure_id(a)
        a.setdefault("approved", False)
        a.setdefault("editor_note", "")
        a.setdefault("score", a.get("score", 0))
        a.setdefault("source", a.get("source", ""))
    date = obj.get("date") or _dt.date.today().isoformat()
    keyword = obj.get("keyword", "")
    return {"date": date, "keyword": keyword, "articles": arts}

def _rollover_archive_if_needed(keyword: str) -> Optional[List[str]]:
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    created = []
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE / f"{base}{ext}"
        if p.exists():
            ts = _dt.datetime.now().strftime("%H%M")
            newp = ARCHIVE / f"{base}_{ts}{ext}"
            p.rename(newp)
            created.append(str(newp))
    return created

def _latest_published_paths(keyword: str) -> List[str]:
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    out = []
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE / f"{base}{ext}"
        if p.exists():
            out.append(str(p))
    return out

def _run(cmd: list[str]) -> dict:
    py = sys.executable or "python"
    cp = subprocess.run([py] + cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {"ok": cp.returncode == 0, "stdout": cp.stdout, "stderr": cp.stderr, "cmd": " ".join(cmd)}

def _run_orch(*args: str) -> dict:
    return _run([str(ORCH), *args])

def _sync_after_save() -> dict:
    # tools/sync_selected_for_publish.py 사용, 없으면 폴백
    script = TOOLS / "sync_selected_for_publish.py"
    if script.exists():
        return _run([str(script)])
    work = _read_json(SEL_WORK)
    items = [a for a in work.get("articles", []) if a.get("approved")]
    out = {"date": work.get("date", _dt.date.today().isoformat()), "articles": items}
    _write_json(SEL_PUB, out)
    return {"ok": True, "stdout": "fallback sync done"}

# ---------- FastAPI ----------
app = FastAPI(title="QualiJournal Admin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ----- Schemas -----
class SaveItem(BaseModel):
    id: str
    approved: bool
    editor_note: str = ""

class SavePayload(BaseModel):
    changes: List[SaveItem]

class PublishReq(BaseModel):
    keyword: str

class FlowKwReq(BaseModel):
    date: Optional[str] = None
    keyword: str
    use_external_rss: bool = True

class EnrichSelReq(BaseModel):
    selection: str = "selected_articles.json"
    max: int = 20

class EnrichKwReq(BaseModel):
    date: Optional[str] = None
    keyword: str
    max: int = 20

class ReportReq(BaseModel):
    date: str
    keyword: Optional[str] = None

class ExportDateReq(BaseModel):
    date: str

# ----- Endpoints -----
@app.get("/api/community")
def api_get_community(only_pending: bool = Query(True, description="미승인만 보기")):
    snap = _get_community_snapshot()
    arts = snap["articles"]
    if only_pending:
        arts = [a for a in arts if not a.get("approved")]
    total = len(snap["articles"])
    approved = sum(1 for a in snap["articles"] if a.get("approved"))
    return {
        "date": snap["date"],
        "keyword": snap.get("keyword", ""),
        "total": total,
        "approved": approved,
        "pending": total - approved,
        "articles": arts,
    }

@app.post("/api/community/save")
def api_save(payload: SavePayload):
    target = SEL_COMM if SEL_COMM.exists() else SEL_WORK
    obj = _read_json(target)
    items = obj.get("articles", [])
    idx = { (i.get("id") or _ensure_id(i)): i for i in items }
    changed = 0
    for c in payload.changes:
        row = idx.get(c.id)
        if not row:
            continue
        if row.get("approved") != c.approved or row.get("editor_note","") != c.editor_note:
            row["approved"] = c.approved
            row["editor_note"] = c.editor_note
            changed += 1
    if changed:
        _write_json(target, obj)
    sync = _sync_after_save()
    return {"saved": changed, "synced": sync.get("ok", False), "sync_log": sync}

@app.get("/api/status")
def api_status():
    work = _read_json(SEL_WORK)
    sel_total = len(work.get("articles", []))
    sel_approved = sum(1 for a in work.get("articles", []) if a.get("approved"))
    snap = _get_community_snapshot()
    comm_total = len(snap.get("articles", []))
    return {
        "date": work.get("date", _dt.date.today().isoformat()),
        "keyword": work.get("keyword", ""),
        "selection_total": sel_total,
        "selection_approved": sel_approved,
        "community_total": comm_total,
        "keyword_total": sel_total,
        "gate_required": 15,
        "gate_pass": sel_approved >= 15,
        "paths": {
            "work": str(SEL_WORK),
            "publish": str(SEL_PUB),
            "community": "root_or_archive"
        }
    }

@app.post("/api/publish-keyword")
def api_publish(req: PublishReq):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    _rollover_archive_if_needed(req.keyword)
    out = _run_orch("--publish-keyword", req.keyword)
    outputs = _latest_published_paths(req.keyword)
    return {**out, "created": outputs}

# ----- 원클릭 흐름 -----
@app.post("/api/flow/daily")
def api_flow_daily():
    steps = []
    steps.append(_run_orch("--collect-community"))
    steps.append(_run_orch("--approve-top", "20"))
    steps.append(_sync_after_save())
    ok = all(s.get("ok", True) for s in steps)
    return {"ok": ok, "steps": steps}

@app.post("/api/flow/community")
def api_flow_comm():
    return _run_orch("--collect-community")

@app.post("/api/flow/keyword")
def api_flow_keyword(req: FlowKwReq):
    steps = []
    if req.use_external_rss:
        steps.append(_run_orch("--collect-keyword", req.keyword, "--use-external-rss"))
    else:
        steps.append(_run_orch("--collect-keyword", req.keyword))
    steps.append(_run_orch("--approve-keyword-top", "20", "--approve-keyword", req.keyword))
    steps.append(_sync_after_save())
    steps.append(_run_orch("--publish-keyword", req.keyword))
    ok = all(s.get("ok", True) for s in steps)
    return {"ok": ok, "steps": steps, "note": "현재 orchestrator는 날짜를 내부 today로 사용합니다."}

# ----- 도구 통합 (NEW) -----
@app.post("/api/enrich/selection")
def api_enrich_selection(req: EnrichSelReq):
    script = TOOLS / "enrich_cards.py"
    if not script.exists():
        return {"ok": False, "error": f"missing {script}"}
    out = _run([str(script), "--selection", req.selection, "--max", str(req.max)])
    return out

@app.post("/api/enrich/keyword")
def api_enrich_keyword(req: EnrichKwReq):
    script = TOOLS / "enrich_cards.py"
    if not script.exists():
        return {"ok": False, "error": f"missing {script}"}
    args = [str(script), "--max", str(req.max)]
    if req.date: args += ["--date", req.date]
    args += ["--keyword", req.keyword]
    out = _run(args)
    return out

@app.post("/api/report")
def api_report(req: ReportReq):
    script = TOOLS / "make_daily_report.py"
    if not script.exists():
        return {"ok": False, "error": f"missing {script}"}
    args = [str(script), "--date", req.date]
    if req.keyword: args += ["--keyword", req.keyword]
    out = _run(args)
    # make_daily_report.py는 최종 경로를 stdout으로 출력
    return out

@app.post("/api/export/md")
def api_export_md(req: ExportDateReq):
    # make_daily_report.py 를 재사용해 MD 생성
    script = TOOLS / "make_daily_report.py"
    if not script.exists():
        return {"ok": False, "error": f"missing {script}"}
    out = _run([str(script), "--date", req.date])
    return out

@app.post("/api/export/csv")
def api_export_csv(req: ExportDateReq):
    # selected_articles.json을 CSV로 내보내기
    j = _read_json(SEL_PUB)
    rows = j.get("articles", [])
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    out_dir = ARCHIVE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{req.date}_selected.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title","url","source","section","approved","pinned","score"])
        for a in rows:
            w.writerow([
                a.get("title",""), a.get("url",""), a.get("source",""),
                a.get("section",""), a.get("approved",False), a.get("pinned",False),
                a.get("total_score") or a.get("score") or ""
            ])
    return {"ok": True, "created": str(out_path)}

# ----- index 서빙 -----
@app.get("/")
def index():
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>QualiJournal Admin</h1><p>index.html이 없습니다.</p>")
