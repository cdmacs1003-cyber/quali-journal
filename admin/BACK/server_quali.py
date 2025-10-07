from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
"""
server_quali.py ??QualiJournal Admin API + Lite Black UI

?듭떖
- GET  /api/community                 : 而ㅻ??덊떚(?먮뒗 ?묒뾽蹂? 紐⑸줉 議고쉶(archive ?꾨낫源뚯? ?먯깋)
- POST /api/community/save            : ?뱀씤/肄붾찘?????+ 諛쒗뻾蹂??숆린??- GET  /api/status                    : KPI(?뱀씤 ??寃뚯씠???듦낵 ??
- POST /api/publish-keyword           : ?ㅼ썙??諛쒗뻾 (以묐났 ???댁쟾蹂?HHMM ??꾩뒪?ы봽 諛깆뾽)
- POST /api/flow/daily                : (?먰겢由? 而ㅻ??섏쭛?뭈opN ?뱀씤?믩룞湲고솕
- POST /api/flow/community            : 而ㅻ??덊떚 ?섏쭛
- POST /api/flow/keyword              : ?ㅼ썙???섏쭛?뭈opN ?뱀씤?믩룞湲고솕?믩컻??- POST /api/approve-ui/start          : ?ㅼ??ㅽ듃?덉씠???뱀씤 UI ?ㅽ뻾(127.0.0.1:8765)
- GET  /                               : index.html ?덉쑝硫?洹멸구, ?놁쑝硫?index_lite_black.html ?쒕튃
?ъ슜:
    uvicorn data.admin_quali.server_quali:app --port 8010 --reload   # (猷⑦듃?먯꽌)
"""
from __future__ import annotations

import os, sys, json, shutil, subprocess, datetime as _dt, hashlib, time, asyncio, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Query, Response, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------- 寃쎈줈 ?좏떥 ----------
BASE = Path(__file__).resolve().parent      # admin_quali ?대뜑
ROOT = BASE.parent                           # ???꾨줈?앺듃 猷⑦듃(怨좎젙)
ARCHIVE = ROOT / "archive"
TOOLS = ROOT / "tools"
ORCH = ROOT / "orchestrator.py"
SEL_COMM = ROOT / "selected_community.json"
SEL_WORK = ROOT / "data" / "selected_keyword_articles.json"
SEL_PUB  = ROOT / "selected_articles.json"
CAND_COMM = [SEL_COMM, ROOT / "archive" / "selected_community.json"]  # ??archive ?꾨낫 ?ы븿

INDEX_HTML = (BASE / "index.html")
INDEX_LITE = (BASE / "index_lite_black.html")  # Lite Black 愿由ъ옄 ?섏씠吏

def _read_json(p: Path) -> dict:
    """Generic JSON reader that tolerates UTF?? BOMs and malformed data.

    Many editors on Windows save JSON files with a UTF?? BOM (Byte Order Mark).
    When such a file is read using the plain ``utf-8`` codec the BOM
    appears as a ``\ufeff`` character at the start of the string, causing
    ``json.loads`` to throw ``JSONDecodeError: Unexpected UTF-8 BOM``??39548676784551?쟊6-L8??  To
    support these files we first attempt to decode using the ``utf-8-sig``
    codec which consumes the BOM automatically??39548676784551?쟊32-L37??  If that fails we
    fall back to decoding the bytes ourselves and stripping any leading BOM
    before parsing.  Any error results in returning an empty dict rather than
    propagating the exception.

    Args:
        p: Path to a JSON file.

    Returns:
        Parsed JSON object or empty dict on failure/missing file.
    """
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8-sig")
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            data = p.read_bytes()
            text = data.decode("utf-8", errors="ignore").lstrip("\ufeff")
            return json.loads(text)
        except Exception:
            return {}

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
    """而ㅻ??덊떚 ?꾨낫 ?ㅻ깄?? 猷⑦듃/?꾩뭅?대툕 selected_community.json ???놁쑝硫??묒뾽蹂몄뿉??community留?異붿텧."""
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

def _sync_after_save() -> dict:
    """諛쒗뻾蹂?selected_articles.json) ?숆린??

    ?곗꽑 ``tools/sync_selected_for_publish.py`` ?ㅽ겕由쏀듃媛 議댁옱?섎㈃ 洹멸쾬???몄텧?쒕떎. ???ㅽ겕由쏀듃媛 ?놁쑝硫?    ?댁옣 ?대갚 濡쒖쭅???ъ슜??``selected_articles.json``???낅뜲?댄듃?쒕떎.

    湲곕낯 ?대갚? ?⑥닚??``data/selected_keyword_articles.json``???뱀씤????ぉ留뚯쓣
    ``selected_articles.json``????뼱?곕뒗 諛⑹떇?댁뿀?쇰굹, 洹몃젃寃??섎㈃ 湲곗〈 怨듭떇/而ㅻ??덊떚 ?좎젙蹂몄씠
    ?щ씪??蹂묓빀 諛쒗뻾 ??鍮??뱀뀡??諛쒖깮?????덈떎. ??援ы쁽?먯꽌??湲곗〈 諛쒗뻾蹂멸낵 ?묒뾽蹂몄쓣
    ``id`` 湲곗??쇰줈 蹂묓빀?섏뿬 蹂댁〈?섎㈃?? ?뱀씤 ?곹깭? ?몄쭛??肄붾찘?몃쭔 媛깆떊?쒕떎.

    Returns:
        dict: ``ok`` ?뚮옒洹몄? ``stdout``/``stderr`` 硫붿떆吏
    """
    py = sys.executable or "python"
    script = TOOLS / "sync_selected_for_publish.py"
    if script.exists():
        try:
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            cp = subprocess.run(
                [py, str(script)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=env,
            )
            return {
                "ok": cp.returncode == 0,
                "stdout": cp.stdout,
                "stderr": cp.stderr,
            }
        except Exception as e:
            return {"ok": False, "stderr": str(e)}
    # ?대갚: ?묒뾽蹂멸낵 湲곗〈 諛쒗뻾蹂몄쓣 蹂묓빀
    work = _read_json(SEL_WORK) or {}
    pub = _read_json(SEL_PUB) or {}
    merged: Dict[str, dict] = {}
    # 1) 湲곗〈 諛쒗뻾蹂몄쓽 湲곗궗?ㅼ쓣 留듭뿉 ?ｌ뼱 ?좎??쒕떎
    for art in pub.get("articles", []) or []:
        try:
            _ensure_id(art)
            merged[art["id"]] = art
        except Exception:
            continue
    # 2) ?묒뾽蹂몄뿉???뱀씤??湲곗궗?ㅼ쓣 蹂묓빀?섍굅??異붽??쒕떎
    for art in work.get("articles", []) or []:
        if not art.get("approved"):
            continue
        try:
            _ensure_id(art)
            aid = art["id"]
        except Exception:
            continue
        if aid in merged:
            # 湲곗〈 ??ぉ 媛깆떊: ?뱀씤 ?곹깭? ?몄쭛??肄붾찘?? ? ?щ?, ?좏깮 ?щ?瑜?理쒖떊?쇰줈
            existing = merged[aid]
            existing["approved"] = art.get("approved", existing.get("approved"))
            existing["editor_note"] = art.get("editor_note", existing.get("editor_note", ""))
            # pinned/pin_ts/select ?뺣낫???묒뾽蹂몄쓣 ?곗꽑?쇰줈 ?곸슜?섎릺, 媛믪씠 ?놁쑝硫?湲곗〈 媛??좎?
            if art.get("pinned") is not None:
                existing["pinned"] = art.get("pinned")
            if art.get("pin_ts"):
                existing["pin_ts"] = art.get("pin_ts")
            if art.get("selected") is not None:
                existing["selected"] = art.get("selected")
        else:
            # ????ぉ? 洹몃?濡?異붽??쒕떎
            merged[aid] = art
    # ?좎쭨???묒뾽蹂몄쓽 ?좎쭨瑜??곗꽑 ?ъ슜?섍퀬, ?놁쑝硫?湲곗〈 諛쒗뻾蹂몄쓽 ?좎쭨, 洹몃쭏????놁쑝硫??ㅻ뒛 ?좎쭨
    date_val = work.get("date") or pub.get("date") or _dt.date.today().isoformat()
    out = {"date": date_val, "articles": list(merged.values())}
    _write_json(SEL_PUB, out)
    return {"ok": True, "stdout": "fallback sync done (merge)"}

def _rollover_archive_if_needed(keyword: str) -> Optional[List[str]]:
    """媛숈? ?좎쭨쨌?ㅼ썙??諛쒗뻾蹂몄씠 ?덉쑝硫?HHMM ?ㅽ꺃??遺숈뿬 諛깆뾽."""
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    created = []
    changed = False
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE / f"{base}{ext}"
        if p.exists():
            ts = _dt.datetime.now().strftime("%H%M")
            newp = ARCHIVE / f"{base}_{ts}{ext}"
            p.rename(newp)
            created.append(str(newp))
            changed = True
    return created if changed else None

def _latest_published_paths(keyword: str) -> List[str]:
    date = _dt.date.today().isoformat()
    base = f"{date}_{_slug_kw(keyword)}"
    out = []
    for ext in (".html", ".md", ".json"):
        p = ARCHIVE / f"{base}{ext}"
        if p.exists():
            out.append(str(p))
    return out

def _run_orch(*args: str) -> dict:
    py = sys.executable or "python"
    # Set PYTHONIOENCODING to UTF-8 so that any Unicode characters printed by
    # orchestrator or its sub?몋ools do not cause encoding errors on Windows consoles.
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Run the orchestrator synchronously. We explicitly specify encoding and error handling so
    # that the output is decoded as UTF?? regardless of the underlying console encoding. Without
    # this, Python defaults to the system locale (e.g., cp949 on Windows) when text=True,
    # causing UnicodeDecodeError when control characters or non?멇SCII bytes are present.
    cp = subprocess.run(
        [py, str(ORCH), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return {
        "ok": cp.returncode == 0,
        "stdout": cp.stdout,
        "stderr": cp.stderr,
        "cmd": " ".join([str(ORCH), *args]),
    }

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

class ApproveTopReq(BaseModel):
    n: int = 20

class FlowKwReq(BaseModel):
    keyword: str
    use_external_rss: bool = True

# Additional schema for gate config patch
class GatePatch(BaseModel):
    gate_required: int

# ----- Endpoints -----
@app.get("/api/community")
def api_get_community(only_pending: bool = Query(True, description="誘몄듅?몃쭔 蹂닿린")):
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
    # ?곗꽑?쒖쐞: selected_community.json -> selected_keyword_articles.json
    target = SEL_COMM if SEL_COMM.exists() else SEL_WORK
    obj = _read_json(target)
    items = obj.get("articles", [])
    idx = { _ensure_id(a): a for a in items }
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
    # 寃뚯씠???꾧퀎媛믪쓣 config.json?먯꽌 ?쎈뒗??(external_rss.gate_required ?먮뒗 猷⑦듃 gate_required)
    cfg = _read_json(ROOT / "config.json")
    gate_req = cfg.get("gate_required") or cfg.get("external_rss", {}).get("gate_required", 15)
    return {
        "date": work.get("date", _dt.date.today().isoformat()),
        "keyword": work.get("keyword", ""),
        "selection_total": sel_total,
        "selection_approved": sel_approved,
        "community_total": comm_total,
        "keyword_total": sel_total,
        "gate_required": gate_req,
        "gate_pass": sel_approved >= gate_req,
        "paths": {
            "work": str(SEL_WORK),
            "publish": str(SEL_PUB),
            "community": "root_or_archive"
        }
    }

@app.post("/api/publish-keyword")
def api_publish(req: PublishReq):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    rollover = _rollover_archive_if_needed(req.keyword)
    out = _run_orch("--publish-keyword", req.keyword)
    outputs = _latest_published_paths(req.keyword)
    return {**out, "rolled_over": rollover or [], "created": outputs}

# ----- ?먰겢由??먮쫫(?듭뀡) -----
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
    return {"ok": ok, "steps": steps}

# ----- 蹂묓빀 諛쒗뻾 ?뚮줈??-----
@app.post("/api/flow/merged")
def api_flow_merged():
    """
    ?먰겢由??듯빀 諛쒗뻾 ?뚮줈??

    怨듭떇(smt/supply) ?섏쭛 ??而ㅻ??덊떚 ?섏쭛 ??而ㅻ??덊떚 ?곸쐞 N ?먮룞 ?뱀씤 ??蹂묓빀 諛쒗뻾??    ?숆린?곸쑝濡??ㅽ뻾?쒕떎. 蹂묓빀 諛쒗뻾? ``orchestrator.py --publish``瑜??몄텧?섏뿬
    ``archive/?꾨━?댁뒪_YYYY-MM-DD.*`` ?뚯씪?ㅼ쓣 ?앹꽦?쒕떎.
    Returns:
        dict: ``ok`` ?뚮옒洹?諛??④퀎蹂?寃곌낵
    """
    steps: List[dict] = []
    # ??怨듭떇 ?섏쭛: smt/supply ?깆뿉??泥?湲곗궗留?異붿텧?섏뿬 selected_articles.json???앹꽦
    steps.append(_run_orch("--collect"))
    # ??而ㅻ??덊떚 ?섏쭛: reddit/forum ?깆뿉???꾨낫瑜??섏쭛?섏뿬 selected_community.json?????    steps.append(_run_orch("--collect-community"))
    # ??而ㅻ??덊떚 ?먮룞 ?뱀씤 Top-N: 湲곕낯 20嫄댁쓣 ?뱀씤?섏뿬 ?꾩냽 諛쒗뻾???ы븿(蹂댁“)
    steps.append(_run_orch("--approve-top", "20"))
    # ??蹂묓빀 諛쒗뻾: official + community瑜??뱀뀡 ?쒖꽌?濡?蹂묓빀?섏뿬 ?꾨━?댁뒪 諛쒗뻾臾??앹꽦
    steps.append(_run_orch("--publish"))
    ok = all(s.get("ok", True) for s in steps)
    return {"ok": ok, "steps": steps}


@app.post("/api/approve-ui/start")
def api_open_ui():
    py = sys.executable or "python"
    subprocess.Popen([py, str(ORCH), "--approve-ui"], cwd=str(ROOT))
    return {"ok": True, "url": "http://127.0.0.1:8765"}

# ----- 鍮꾨룞湲??묒뾽 API -----
# ?묒뾽 ?곹깭 ??μ냼 (硫붾え由?援ы쁽)
TASKS: Dict[str, Dict[str, Any]] = {}

def _now_ms() -> int:
    """?꾩옱 ?쒓컙??諛由ъ큹 ?⑥쐞濡?諛섑솚"""
    return int(time.time() * 1000)

async def _run_async_job(job_id: str, kind: str, payload: Dict[str, Any]):
    """二쇱뼱吏?kind濡?orchestrator ?묒뾽??鍮꾨룞湲곗쟻?쇰줈 ?ㅽ뻾?섍퀬 ?곹깭瑜??낅뜲?댄듃?쒕떎.

    Args:
        job_id: ?묒뾽 ?앸퀎??        kind: ?뚮줈??醫낅쪟 (daily/community/keyword/merged ??
        payload: 異붽? ?뚮씪誘명꽣 (?ㅼ썙????
    """
    TASKS[job_id] = {
        "status": "running",
        "kind": kind,
        "name": f"flow-{kind}",
        "started_at": _now_ms(),
        "ended_at": None,
        "steps": []
    }
    # 紐낅졊 ?몄닔 援ъ꽦
    args: List[str] = []
    if kind == "daily":
        args = ["--collect-community", "--approve-top", "20"]
    elif kind == "community":
        args = ["--collect-community"]
    elif kind == "keyword":
        # payload?먯꽌 keyword? use_external_rss ?쎄린
        kw = payload.get("keyword", "")
        use_ext = payload.get("use_external_rss", True)
        args = ["--collect-keyword", kw]
        if use_ext:
            args.append("--use-external-rss")
        # approve-keyword-top, approve-keyword, publish-keyword
        args.extend(["--approve-keyword-top", "20", "--approve-keyword", kw, "--publish-keyword", kw])
    elif kind == "merged":
        args = ["--collect", "--collect-community", "--approve-top", "20", "--publish"]
    else:
        # 湲고? ?ㅼ썙?쒕뒗 吏곸젒 ?꾨떖
        args = [f"--{kind}"]
    # orchestration ?ㅽ뻾 ?④퀎蹂?異붿쟻???꾪빐 ??紐낅졊???ㅽ뻾
    # ?⑥씪 而ㅻ㎤??由ъ뒪?몃? ?④퀎蹂꾨줈 ?섎늻???ㅽ뻾?쒕떎
    steps: List[dict] = []
    # ?좏떥 ?⑥닔濡???紐낅졊 ?ㅽ뻾
    async def run_cmd(cmd_args: List[str]):
        py = sys.executable or "python"
        # ?섍꼍 ?ㅼ젙: UTF-8 媛뺤젣
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        t_start_ms = _now_ms()
        proc = await asyncio.create_subprocess_exec(
            py, str(ORCH), *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT), env=env
        )
        stdout_lines = []
        # stdout 由ъ뼹????쎄린
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                decoded = line.decode("utf-8", errors="replace")
            except Exception:
                decoded = line.decode("utf-8", "replace")
            stdout_lines.append(decoded.rstrip())
        # stderr ?꾨? ?쎄린
        stderr_bytes = await proc.stderr.read()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        rc = await proc.wait()
        t_end_ms = _now_ms()
        step = {
            "cmd": " ".join([str(ORCH)] + cmd_args),
            "ok": rc == 0,
            "stdout": "\n".join(stdout_lines),
            "stderr": stderr_text,
            "t_start": t_start_ms,
            "t_end": t_end_ms,
            "ms": t_end_ms - t_start_ms
        }
        return step
    # ?④퀎蹂??ㅽ뻾
    # 吏곸쟾 ?④퀎 ?섑뻾 寃곌낵瑜?steps??異붽?
    try:
        i = 0
        while i < len(args):
            # 紐낅졊 援щ텇: -- ?듭뀡??寃쎌슦 ?ㅼ쓬 ?몄옄源뚯? ?ы븿
            cmd_segment: List[str] = []
            # 泥??몄옄
            cmd_segment.append(args[i])
            i += 1
            # ?듭뀡 以?蹂듯빀 ?뚮옒洹?(--approve-top 20 ??瑜??꾪빐 ?몄옄 由ъ뒪??異붿쟻
            while i < len(args) and not args[i].startswith("--"):
                cmd_segment.append(args[i])
                i += 1
            # ?⑥씪 紐낅졊 ?ㅽ뻾
            step_result = await run_cmd(cmd_segment)
            steps.append(step_result)
            TASKS[job_id]["steps"] = steps
            # 以묎컙 ?곹깭 ?낅뜲?댄듃
            TASKS[job_id]["status"] = "running"
        TASKS[job_id]["status"] = "done" if all(s["ok"] for s in steps) else "error"
    except Exception as exc:
        TASKS[job_id]["status"] = "error"
        steps.append({"cmd": "exception", "ok": False, "stdout": "", "stderr": str(exc), "t_start": _now_ms(), "t_end": _now_ms(), "ms": 0})
        TASKS[job_id]["steps"] = steps
    finally:
        TASKS[job_id]["ended_at"] = _now_ms()

@app.post("/api/tasks/flow/{kind}")
async def api_tasks_flow(kind: str, req: Request):
    """鍮꾨룞湲??묒뾽 ?쒖옉 ?붾뱶?ъ씤?? kind???곕씪 ?묒뾽 ?먮쫫???ㅽ뻾?쒕떎.

    Returns job_id for tracking.
    """
    try:
        payload = await req.json()
    except Exception:
        payload = {}
    # ?좏슚 kind: daily/community/keyword/merged ??    job_id = uuid.uuid4().hex[:12]
    # background task
    asyncio.create_task(_run_async_job(job_id, kind, payload))
    return {"job_id": job_id}

@app.get("/api/tasks/{job_id}")
def api_tasks_get(job_id: str):
    st = TASKS.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    return st

@app.get("/api/tasks/{job_id}/stream")
async def api_tasks_stream(job_id: str):
    """Server-Sent Events(SSE)濡??묒뾽 ?곹깭瑜?吏?띿쟻?쇰줈 ?꾩넚"""
    async def event_gen():
        # 珥덇린 吏??        while True:
            st = TASKS.get(job_id)
            if not st:
                break
            # ?꾩넚
            yield f"data: {json.dumps(st, ensure_ascii=False)}\n\n"
            # ?꾨즺/?먮윭/痍⑥냼 ??醫낅즺
            if st.get("status") in {"done", "error", "canceled"}:
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/api/tasks/recent")
def api_tasks_recent(limit: int = 20):
    """理쒓렐 ?묒뾽 紐⑸줉 議고쉶"""
    ids = list(TASKS.keys())[-limit:]
    return [{"job_id": i, **TASKS[i]} for i in ids]

@app.post("/api/tasks/{job_id}/cancel")
async def api_tasks_cancel(job_id: str):
    # ?⑥닚 ?곹깭?쒖떆留?蹂寃? ?ㅼ젣 ?쒕툕?꾨줈?몄뒪 醫낅즺??援ы쁽?섏? ?딆쓬
    st = TASKS.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    st["status"] = "canceled"
    st["ended_at"] = _now_ms()
    return {"ok": True}

@app.patch("/api/config/gate_required")
def api_config_gate_required(body: GatePatch):
    """?뱀씤 寃뚯씠???꾧퀎媛믪쓣 ?섏젙?섍퀬 諛섑솚?쒕떎."""
    cfg_path = ROOT / "config.json"
    cfg = _read_json(cfg_path)
    val = int(body.gate_required)
    cfg.setdefault("external_rss", {})
    cfg["external_rss"]["gate_required"] = val
    # 蹂꾨룄 猷⑦듃 ?ㅻ줈?????(?섏쐞 踰꾩쟾 ?명솚)
    cfg["gate_required"] = val
    _write_json(cfg_path, cfg)
    return {"gate_required": val}


# ----- 湲고? ?명솚 ?붾뱶?ъ씤??-----
#
# Chrome devtools will automatically request the file
# ``/.well-known/appspecific/com.chrome.devtools.json`` on every page load.
# We return a minimal JSON response with a 200 status to avoid Starlette runtime
# errors.  A 204 response must not include a body, so returning content with
# status 204 previously caused the server to raise
# "Response content longer than Content-Length".  Using a 200 status with an
# empty JSON object keeps the console quiet and satisfies Chrome devtools.
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def devtools_config():
    return Response(content="{}", media_type="application/json", status_code=200)

# Also suppress 404 for favicon.ico requests.
@app.get("/favicon.ico")
def favicon_blank():
    return Response(status_code=204)

# ----- index ?쒕튃 -----
@app.get("/")
def index():
    # index.html ?곗꽑, ?놁쑝硫?Lite Black ?ъ슜
    p = INDEX_HTML if INDEX_HTML.exists() else (INDEX_LITE if INDEX_LITE.exists() else None)
    if p:
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>QualiJournal Admin</h1><p>index.html???놁뒿?덈떎.</p>")

