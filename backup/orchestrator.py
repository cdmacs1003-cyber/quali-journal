from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
"""
orchestrator.py ??QualiNews ?듯빀 ?뚯씠?꾨씪??
?붽뎄 諛섏쁺:
- ?ㅼ썙???꾨낫 ?좊퀎: (?좏겙??) OR (?뺢퇋???덊듃) ???듦낵
- ?좊ː ?꾨찓??fallback: ?좊ː ?꾨찓?몄씪 ??(?좏겙?? or ?レ옄 ?뚰듃)濡?蹂댁“ ?듦낵(?レ옄 ?⑤룆 X)
- 而ㅻ??덊떚 ?꾪꽣??config.json??community.filters濡???뼱?곌린(?λ㉧吏) 媛??- ?몃? RSS??--use-external-rss + config.external_rss.enabled 紐⑤몢 True???뚮쭔 ?ъ슜
"""

from __future__ import annotations

import os, re, sys, json, time, math, argparse
from pathlib import Path
import re as _re
from typing import List, Dict, Optional
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin, quote_plus

import requests
from bs4 import BeautifulSoup
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Logging configuration
#
# The orchestrator is executed both via the admin API and as a standalone
# command?멿ine tool.  To aid in debugging and post?몀ortem analysis it writes
# informational messages to a dedicated log file.  We rely on the shared
# `logging_setup` module to create a daily rotating file handler.  The log
# file is stored under the project?멿evel ``logs`` directory.  When the
# orchestrator is invoked, a one?멿ine summary of the command line arguments
# is emitted.  Additional logging calls throughout this module can be added
# as needed to capture important events without cluttering standard output.

from logging_setup import setup_logger  # type: ignore

# Compute the path to the ``logs`` directory.  The server module determines
# the project root as the parent directory of the admin module.  To keep
# consistency across modules we ascend one level above this file to the
# parent directory (``.../``) and then append ``logs/orchestrator.log``.  If
# the ``logs`` directory does not exist it will be created by
# ``setup_logger``.  See logging_setup.py for details.
_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "orchestrator.log"

# Initialize the orchestrator logger.  We use a module?멿evel variable named
# ``logger`` (prefixed with underscore to avoid polluting public API) for
# consistency with other modules.  The default log level is INFO and log
# rotation occurs at midnight with seven backups.
logger = setup_logger("orchestrator", str(_LOG_PATH))

# Import shared utility functions to avoid duplicating logic across modules.
# canonical_url/deep_merge: normalization and configuration merging helpers.
# split_keyword_tokens/safe_compile_regex_list/norm01_log/value_score: scoring utilities.
from common_utils import (
    canonical_url as util_canonical_url,
    deep_merge as util_deep_merge,
    split_keyword_tokens,
    safe_compile_regex_list,
    norm01_log,
    value_score,
)


# ----------------------------- 湲곕낯 ?곸닔/?ㅼ젙 -----------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36 QualiNewsBot/2025-10"
)
REQUEST_TO = (8, 16)  # (connect, read)

DEFAULT_CFG = {
    "paths": {
        "archive": "archive",
        "reports": "archive/reports",
        # 怨듭떇 ?좎젙 ?뚯씪(?좎?)
        "selection_file": "selected_articles.json",
        # 而ㅻ??덊떚 ?좎젙 ?뚯씪(?좎?)
        "community_selection_file": "archive/selected_community.json",
        # ?뚯뒪 ?뚯씪 ?먯깋
        "community_sources_candidates": [
            "feeds/community_sources.json",
            "community_sources.json"
        ],
        "keywords_txt": "而ㅻ??덊떚_?ㅼ썙??txt",
        # (?좏깮) 怨듭떇 ?뚯뒪 ?뚯씪 寃쎈줈
        "smt_sources_file": "smt_sources.json",
        "supply_sources_file": "supply_sources.json",
    },
    "features": {
        "require_editor_approval": True,
        "output_formats": ["html", "md", "json"],
        # (?좏깮) ?좊ː ?꾨찓??異붽? 而ㅼ뒪?곕쭏?댁쭠
        "trusted_domains": []
    },
    "community": {
        "enabled": True,
        "fresh_hours": 336,          # 2二?        "min_upvotes": 0,
        "min_comments": 0,
        "score_threshold": 0.0,
        "max_total": 300,
        "reddit_pages": 5,
        "score_weights": {"keyword": 3, "upvotes": 5, "views": 2},
        "norms": {"kw_base": 2, "upvotes_max": 200, "views_max": 100000},
        # config.json?먯꽌 community.filters 濡??몃? ?쒕떇 媛??        # ?? {"kw_min_tokens":2, "kw_regex":[...], "allow_domains":[...], "require_keyword":true,...}
        "filters": {}
    },
    "external_rss": {
        "enabled": False,
        "max_total": 50  # build_keyword_selection --use-external-rss ?????곹븳
    }
}

# 蹂묓빀 諛쒗뻾 ?뱀뀡 ?쒖꽌
SECTION_ORDER = ["?쒖? ?댁뒪", "湲濡쒕쾶 ?꾩옄?앹궛", "AI ?댁뒪", "而ㅻ??덊떚"]

# ----------------------------- ?좏떥 ?⑥닔 -----------------------------
def _std_headers() -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache"
    }

def _os_rel(*p):
    return os.path.join(os.getcwd(), *p)

def _load_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _find_first(cands: List[str]) -> Optional[str]:
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

def _read_lines(path: Optional[str]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out

def canonical_url(u: str) -> str:
    try:
        u = (u or "").strip()
        u = re.sub(r"#.*$", "", u)
        u = re.sub(r"/+$", "", u)
        return u
    except Exception:
        return u or ""

def _split_kw_tokens(kw: str) -> list[str]:
    """?ㅼ썙?쒕? ?좏겙?쇰줈 履쇨컻?? -, /, +, . ???좎?(?쒖? 踰덊샇 ?뺥깭瑜??대━湲??꾪븿)."""
    if not kw:
        return []
    toks = re.split(r"[^\w\-/+\.]+", kw)
    # 湲몄씠 1 ?뚰뙆踰??좏겙? ?쒓굅 (?? IPC-A-610??'a')
    return [t.lower() for t in toks if t and (t.isdigit() or len(t) >= 2)]

def _safe_regex_list(patterns: list[str]) -> list[re.Pattern]:
    """臾몃쾿 ?ㅻ쪟?섎뒗 ?뺢퇋?앹? 嫄대꼫?곌퀬, ??뚮Ц??臾댁떆濡?而댄뙆??"""
    out = []
    for p in patterns or []:
        try:
            out.append(re.compile(p, re.I))
        except re.error:
            pass
    return out

def _domain_of(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def _now_utc():
    return datetime.now(timezone.utc)

def _hours_ago(ts_iso: str | None) -> float:
    if not ts_iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (_now_utc() - dt).total_seconds() / 3600.0
    except Exception:
        return 0.0

def _article_id(url: str) -> str:
    import hashlib
    return hashlib.md5(canonical_url(url).encode("utf-8")).hexdigest()

def _norm01_log(x: int | float, maxv: int | float) -> float:
    if maxv <= 0:
        return 0.0
    return min(1.0, math.log1p(max(0.0, float(x))) / math.log1p(float(maxv)))

def _value_score(kw_raw: int, upvotes: int, views: int, cfg: dict) -> float:
    c = (cfg.get("community") or {})
    W = (c.get("score_weights") or {"keyword":3,"upvotes":5,"views":2})
    N = (c.get("norms") or {"kw_base":2,"upvotes_max":200,"views_max":100000})
    kw = min(1.0, (kw_raw or 0) / max(1, int(N.get("kw_base", 2))))
    up = _norm01_log(upvotes or 0, int(N.get("upvotes_max", 200)))
    vw = _norm01_log(views or 0, int(N.get("views_max", 100000)))
    return float(W.get("keyword",3))*kw + float(W.get("upvotes",5))*up + float(W.get("views",2))*vw


# -----------------------------------------------------------------------------
# Override helper functions with implementations from common_utils
#
# The functions ``canonical_url``, ``deep_merge``, ``_split_kw_tokens``,
# ``_safe_regex_list``, ``_norm01_log`` and ``_value_score`` were originally
# defined in this module. To reduce duplication across modules, we import
# alternative implementations from ``common_utils`` and reassign the local
# names to these shared helpers. This ensures the rest of the code uses
# consistent logic without modifying call sites.
canonical_url = util_canonical_url  # type: ignore
deep_merge = util_deep_merge  # type: ignore
_split_kw_tokens = split_keyword_tokens  # type: ignore
_safe_regex_list = safe_compile_regex_list  # type: ignore
_norm01_log = norm01_log  # type: ignore
_value_score = value_score  # type: ignore


# ----------------------------- ?ㅼ썙??濡쒕뵫 & 留ㅼ묶 (遺遺?臾몄옄??諛⑹떇) -----------------------------
def _load_keyword_list(path: Optional[str]) -> List[str]:
    kws = _read_lines(path)
    return [k.strip().lower() for k in kws if k and len(k.strip())>0]

def _count_kw_hits_in_text(text: str, kw_list: List[str]) -> int:
    """遺遺?臾몄옄??留ㅼ묶 諛⑹떇: 媛??ㅼ썙?쒓? ?띿뒪?몄뿉 議댁옱?섎㈃ 1濡?移댁슫??(以묐났 ?ㅼ썙?쒕뒗 紐⑤몢 ?뷀븿)."""
    if not text or not kw_list:
        return 0
    t = text.lower()
    hits = 0
    for kw in kw_list:
        if kw in t:
            hits += 1
    return hits


# ----------------------------- ?쒕ぉ 釉붾줈而?(?뺢퇋??湲곕컲) -----------------------------
def _compile_title_blockers(filters: dict) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    for k in ("block_title_patterns", "deny_title_regex"):
        for p in (filters.get(k) or []):
            try:
                pats.append(re.compile(p, re.I))
            except re.error:
                pass
    return pats

def _blocked_by_title(title: str, blockers: List[re.Pattern], min_len: int = 0) -> bool:
    t = (title or "").strip()
    if min_len and len(t) < min_len:
        return True
    return any(rx.search(t) for rx in blockers)


# ======================================================================
#                          怨듭떇 ?뚯뒪(?좏깮)
# ======================================================================
def _sel_official_path(cfg: dict) -> str:
    """怨듭떇 ?좎젙 ?뚯씪 寃쎈줈(selected_articles.json)"""
    p = ((cfg.get("paths") or {}).get("selection_file")) or "selected_articles.json"
    return _os_rel(p)

def read_selection_official(cfg: dict) -> dict:
    path = _sel_official_path(cfg)
    data = _load_json(path, {}) or {}
    data.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
    data.setdefault("articles", [])
    return data

def write_selection_official(articles: List[Dict], cfg: dict) -> str:
    """湲곗〈 approved/selected/pinned/editor_note瑜?蹂댁〈 蹂묓빀 ???硫깅벑)."""
    path = _sel_official_path(cfg)
    prev = read_selection_official(cfg)
    prev_map = {a.get("id"): a for a in prev.get("articles", [])}
    out = {"date": datetime.now().strftime("%Y-%m-%d"), "articles": []}
    for art in articles:
        old = prev_map.get(art.get("id"), {})
        out["articles"].append({
            **art,
            "approved": bool(old.get("approved", False)),
            "selected": bool(old.get("selected", False)),
            "pinned": bool(old.get("pinned", False)),
            "editor_note": (old.get("editor_note") or "")
        })
    _save_json(path, out)
    print(f"[怨듭떇 ?좎젙 ?뚯씪 ??? {path} (珥?{len(out['articles'])}嫄?")
    return path

def _pick_first_article_url(base_url: str, html: str) -> Optional[str]:
    """由ъ뒪???쒕뵫?먯꽌 泥?湲곗궗 URL 異붿젙(?대━?ㅽ떛)."""
    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.select('a[href]'):
        href = a.get("href") or ""
        if "/article/" in href or "/news/" in href:
            return canonical_url(urljoin(base_url, href))
    a = soup.find("a", href=True)
    return canonical_url(urljoin(base_url, a["href"])) if a else None

def _fetch_title_desc(url: str) -> tuple[str, str]:
    """?⑥씪 湲곗궗 ?섏씠吏?먯꽌 ?쒕ぉ+?붿빟 ?꾨낫 異붿텧(?ㅽ뙣 ??URL 諛섑솚)."""
    try:
        r = requests.get(url, headers=_std_headers(), timeout=REQUEST_TO)
        if r.status_code != 200:
            return (url, "")
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.find("h1") or soup.find("title") or soup.find("h2"))
        title_txt = (title.get_text(" ", strip=True) if title else url)
        desc = ""
        md = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if md and md.get("content"):
            desc = (md["content"] or "").strip()
        return (title_txt or url, desc)
    except Exception:
        return (url, "")

def _classify_section(label: str, url: str) -> str:
    d = _domain_of(url)
    lbl = (label or "").lower()
    if any(k in lbl for k in ["ecss","ipc","nasa","mil"]) or "ecss.nl" in d or "ipc.org" in d:
        return "?쒖? ?댁뒪"
    if any(k in lbl for k in ["ai","openai","x.ai","gemini","meta"]):
        return "AI ?댁뒪"
    return "湲濡쒕쾶 ?꾩옄?앹궛"

def collect_official(cfg: dict) -> List[Dict]:
    """smt/supply ?뚯뒪?먯꽌 泥?湲곗궗 留곹겕瑜?異붿텧??移대뱶??"""
    paths = (cfg.get("paths") or {})
    smt_path = _os_rel(paths.get("smt_sources_file") or "smt_sources.json")
    sup_path = _os_rel(paths.get("supply_sources_file") or "supply_sources.json")
    smt = _load_json(smt_path, {}) or {}
    sup = _load_json(sup_path, {}) or {}

    items: List[Dict] = []

    def _from_dict(d: dict):
        for label, url in (d.items() if isinstance(d, dict) else []):
            if not url:
                continue
            base = canonical_url(url)
            try:
                r = requests.get(base, headers=_std_headers(), timeout=REQUEST_TO)
                if r.status_code != 200:
                    continue
                first = _pick_first_article_url(base, r.text) or base
                title, desc = _fetch_title_desc(first)
                section = _classify_section(label, first)
                art = {
                    "id": _article_id(first),
                    "title": title,
                    "url": first,
                    "source": label,
                    "section": section,
                    "qg_status": None, "qg_score": 0,
                    "fc_status": None, "fc_score": 0,
                    "kw_score": 0, "kw_hits": [],
                    "domain_score": 0, "total_score": 0,
                    "selected": False, "approved": False,
                    "pinned": False, "pin_ts": 0,
                    "summary_ko_text": (desc or None)
                }
                items.append(art)
            except Exception:
                continue

    _from_dict(smt)   # SMT/PCB/MilAero ??    _from_dict(sup)   # 怨듦툒留?EMS ?숉뼢

    # URL 以묐났 ?쒓굅
    seen = set(); uniq = []
    for a in items:
        u = a.get("url")
        if u in seen:
            continue
        uniq.append(a); seen.add(u)
    print(f"[怨듭떇 ?섏쭛] {len(uniq)}嫄?)
    return uniq


# ======================================================================
#                          而ㅻ??덊떚 ?섏쭛(?좎?)
# ======================================================================
def _reddit_old_new(sub: str, pages: int = 1) -> List[Dict]:
    """
    old.reddit.com/r/<sub>/new/ HTML ?뚯떛
    諛섑솚 ??ぉ: dict keys: title,url,upvotes,comments,ts,selftext
    selftext??.json ?붿껌?쇰줈 ?쒕룄?댁꽌 ?살쓬(?놁쑝硫?鍮?臾몄옄??
    """
    out: List[Dict] = []
    base = f"https://old.reddit.com/r/{sub}/new/"
    url = base
    headers = _std_headers()
    for _ in range(max(1, int(pages))):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TO)
        except Exception:
            break
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup.select("div.thing"):
            a = t.select_one("a.title")
            if not a or not a.get("href"):
                continue
            title = a.get_text(" ", strip=True)
            href = urljoin("https://old.reddit.com", a["href"])
            sc = t.select_one(".score")
            up = 0
            if sc:
                if sc.has_attr("title"):
                    up = int(re.sub(r"\D", "", sc["title"] or "0") or 0)
                else:
                    up = int(re.sub(r"\D", "", sc.get_text(" ", strip=True) or "0") or 0)
            cm_a = t.select_one("a.comments")
            cm = int(re.sub(r"\D", "", (cm_a.get_text(" ", strip=True) if cm_a else "") or "0") or 0)
            tm = t.find("time")
            ts = (tm.get("datetime") if tm and tm.has_attr("datetime") else None)
            post = {
                "title": title,
                "url": canonical_url(href),
                "upvotes": up,
                "comments": cm,
                "ts": ts,
                "selftext": ""
            }
            # try to fetch post.json to extract selftext
            try:
                jurl = post["url"] + ".json"
                jr = requests.get(jurl, headers=headers, timeout=REQUEST_TO)
                if jr.status_code == 200:
                    jd = jr.json()
                    if isinstance(jd, list) and jd:
                        data = jd[0].get("data", {}).get("children", [])
                        if data and isinstance(data, list) and data[0].get("data"):
                            d0 = data[0].get("data", {})
                            st = d0.get("selftext") or ""
                            post["selftext"] = st
            except Exception:
                pass
            out.append(post)
        nx = soup.select_one("span.next-button a")
        url = (nx.get("href") if nx else None)
        if not url:
            break
        time.sleep(0.4)
    return out

def _parse_forum_html(base_url: str, html: str, limit: int = 20) -> List[Dict]:
    soup = BeautifulSoup(html or "", "html.parser")
    rows: List[Dict] = []
    for a in soup.find_all("a", href=True):
        title = (a.get_text(" ", strip=True) or "").strip()
        href = urljoin(base_url, (a["href"] or "").strip())
        if not title or not href:
            continue
        path = urlparse(href).path.lower()
        if any(k in path for k in ("/thread", "/topic", "/showthread", "/discussion", "/t/", "/threads/")) and len(title) >= 10:
            row = a.find_parent(["tr","li","div"])
            txt = (row.get_text(" ", strip=True) if row else "")
            def _pick(rx):
                m = re.search(rx, txt, flags=re.I)
                return int(re.sub(r"[^\d]", "", (m.group(1) if m else "") or "0") or 0)
            vw = _pick(r"views?\s*[:\-\s]*([0-9,\.]+)") or _pick(r"議고쉶\s*[:\-\s]*([0-9,\.]+)")
            rp = _pick(r"(repl(y|ies)|comments?)\s*[:\-\s]*([0-9,\.]+)")
            rows.append({"title": title, "url": canonical_url(href), "upvotes": rp, "comments": rp, "views": vw, "ts": None})
            if len(rows) >= limit:
                break
    return rows

def collect_community(cfg: dict) -> List[Dict]:
    cpaths = (cfg.get("paths") or {})
    comm_cfg = (cfg.get("community") or {})
    fresh_h = int(comm_cfg.get("fresh_hours", 72))
    min_up  = int(comm_cfg.get("min_upvotes", 5))
    min_cm  = int(comm_cfg.get("min_comments", 0))
    thr     = float(comm_cfg.get("score_threshold", 3.5))
    pages   = int(comm_cfg.get("reddit_pages", 1))

    cand = cpaths.get("community_sources_candidates") or ["feeds/community_sources.json","community_sources.json"]
    src_path = _find_first([_os_rel(p) for p in cand]) or ""
    if not src_path:
        print("[WARNING] community_sources.json not found in candidates:", cand)
    sources = _load_json(src_path, {}) if src_path else {}
    reddit_cfg = (sources.get("reddit") or {})
    forums_cfg = (sources.get("forums") or [])
    filters = (sources.get("filters") or {})

    # [NEW] config.json ??community.filters ?λ㉧吏(?댁쁺 ?쒕떇)
    filters = deep_merge(filters, ((cfg.get("community") or {}).get("filters") or {}))

    allow_domains = set(filters.get("allow_domains") or [])
    min_title_len = int(filters.get("min_title_len") or 0)
    title_blockers = _compile_title_blockers(filters)

    # keywords
    kw_list = _load_keyword_list(cpaths.get("keywords_txt"))
    require_kw = bool(filters.get("require_keyword") or comm_cfg.get("require_keyword") or False)

    items: List[Dict] = []

    # -- Reddit --
    subs = reddit_cfg.get("subs") or []
    for sub in subs:
        try:
            rows = _reddit_old_new(sub, pages=pages)
        except Exception:
            rows = []
        for r in rows:
            url = r.get("url") or ""
            dom = _domain_of(url)
            if allow_domains and dom not in allow_domains:
                continue
            title = r.get("title") or ""
            if _blocked_by_title(title, title_blockers, min_len=min_title_len):
                continue
            hrs = _hours_ago(r.get("ts"))
            if fresh_h and hrs > fresh_h:
                continue
            if int(r.get("upvotes",0)) < min_up or int(r.get("comments",0)) < min_cm:
                continue
            combined_text = (title or "") + " " + (r.get("selftext") or "")
            kw_hits = _count_kw_hits_in_text(combined_text, kw_list)
            if require_kw and kw_hits <= 0:
                continue
            score = _value_score(kw_hits, int(r.get("upvotes",0)), 0, cfg)
            if score < thr:
                continue
            items.append({
                "id": _article_id(url),
                "title": title,
                "url": url,
                "source": f"[COMM][Reddit/{sub}]",
                "upvotes": int(r.get("upvotes",0)),
                "comments": int(r.get("comments",0)),
                "views": 0,
                "ts": r.get("ts"),
                "kw_hits": kw_hits,
                "total_score": round(float(score), 3),
                "section": "而ㅻ??덊떚",
                "approved": False,
                "selected": False
            })

    # -- Forums --
    for f_url in forums_cfg:
        try:
            r = requests.get(f_url, headers=_std_headers(), timeout=REQUEST_TO)
            r.raise_for_status()
            rows = _parse_forum_html(f_url, r.text, limit=10)
        except Exception:
            rows = []
        for e in rows:
            url = e.get("url") or ""
            dom = _domain_of(url)
            if allow_domains and dom not in allow_domains:
                continue
            title = e.get("title") or ""
            if _blocked_by_title(title, title_blockers, min_len=min_title_len):
                continue
            if int(e.get("upvotes",0)) < min_up or int(e.get("comments",0)) < min_cm:
                continue
            combined_text = (title or "")
            kw_hits = _count_kw_hits_in_text(combined_text, kw_list)
            if require_kw and kw_hits <= 0:
                continue
            score = _value_score(kw_hits, int(e.get("upvotes",0)), int(e.get("views",0)), cfg)
            if score < thr:
                continue
            items.append({
                "id": _article_id(url),
                "title": title,
                "url": url,
                "source": "[COMM][Forum]",
                "upvotes": int(e.get("upvotes",0)),
                "comments": int(e.get("comments",0)),
                "views": int(e.get("views",0)),
                "ts": None,
                "kw_hits": kw_hits,
                "total_score": round(float(score), 3),
                "section": "而ㅻ??덊떚",
                "approved": False,
                "selected": False
            })

    # dedupe & sort
    seen = set(); uniq = []
    for a in sorted(items, key=lambda x: x.get("total_score", 0), reverse=True):
        if a.get("url") in seen:
            continue
        uniq.append(a); seen.add(a.get("url"))
    max_total = int(comm_cfg.get("max_total", 50))
    return uniq[:max_total]


# ----------------------------- 而ㅻ??덊떚 ?좎젙 ?뚯씪 I/O -----------------------------
def _sel_path(cfg: dict) -> str:
    p = ((cfg.get("paths") or {}).get("community_selection_file")) or "archive/selected_community.json"
    return _os_rel(p)

def read_selection(cfg: dict) -> dict:
    path = _sel_path(cfg)
    data = _load_json(path, {}) or {}
    if "articles" not in data and "items" in data:
        data["articles"] = data.get("items") or []
    data.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
    data.setdefault("articles", [])
    return data

def write_selection(articles: List[Dict], cfg: dict) -> str:
    path = _sel_path(cfg)
    prev = read_selection(cfg)
    prev_map = {a.get("id"): a for a in prev.get("articles", [])}
    out = {"date": datetime.now().strftime("%Y-%m-%d"), "articles": []}
    for art in articles:
        old = prev_map.get(art.get("id"), {})
        out["articles"].append({
            **art,
            "approved": bool(old.get("approved", False)),
            "selected": bool(old.get("selected", False)),
            "pinned": bool(old.get("pinned", False)),
            "editor_note": (old.get("editor_note") or "")
        })
    _save_json(path, out)
    print(f"[?좎젙 ?뚯씪 ??? {path} (珥?{len(out['articles'])}嫄?")
    return path


# ----------------------------- ?뱀씤 UI(而ㅻ??덊떚) -----------------------------
HTML_TMPL = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>而ㅻ??덊떚 ?뱀씤</title><style>
body{font-family:system-ui,Segoe UI,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px;}
.card{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0;}
.h{display:flex;align-items:center;gap:8px;}
.meta{color:#666;font-size:12px;margin-top:6px}
.btn{padding:8px 12px;border:1px solid #555;border-radius:6px;background:#fff;cursor:pointer;}
.btn:hover{background:#f5f5f5}
</style></head><body>
<h1>而ㅻ??덊떚 ?뱀씤(UI)</h1>
<div id="list"></div>
<div style="margin-top:12px">
<button class="btn" onclick="save()">???/button>
</div>
<script>
async function fetchData(){
  const r=await fetch('/data'); const j=await r.json();
  const root=document.getElementById('list'); root.innerHTML='';
  j.articles.forEach(a=>{
    const d=document.createElement('div'); d.className='card';
    d.innerHTML = `
      <div class="h">
        <input type="checkbox" ${a.approved?'checked':''} data-id="${a.id}">
        <a href="${a.url}" target="_blank">${a.title}</a>
      </div>
      <div class="meta">?먯닔 ${a.total_score||0} 쨌 ${a.source||''} 쨌 ??{a.upvotes||0} 쨌 ?뮠${a.comments||0} 쨌 ?몓${a.views||'-'}</div>
    `;
    root.appendChild(d);
  });
}
async function save(){
  const ids=[...document.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.getAttribute('data-id'));
  const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({approved_ids:ids})});
  if(r.ok){alert('????꾨즺');}else{alert('????ㅽ뙣');}
}
fetchData();
</script></body></html>"""

class ApproveHandler(BaseHTTPRequestHandler):
    cfg: dict = {}

    def _write(self, code=200, ctype="text/html; charset=utf-8", body=""):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            return self._write(200, "text/html; charset=utf-8", HTML_TMPL)
        elif self.path == "/data":
            data = read_selection(self.cfg)
            return self._write(200, "application/json; charset=utf-8", json.dumps(data, ensure_ascii=False))
        return self._write(404, "text/plain; charset=utf-8", "Not Found")

    def do_POST(self):
        if self.path == "/save":
            ln = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(ln).decode("utf-8")
            try:
                req = json.loads(raw or "{}")
                ids = set(req.get("approved_ids") or [])
                data = read_selection(self.cfg)
                for a in data["articles"]:
                    a["approved"] = (a.get("id") in ids)
                _save_json(_sel_path(self.cfg), data)
                return self._write(200, "application/json; charset=utf-8", json.dumps({"ok": True}))
            except Exception as e:
                return self._write(500, "application/json; charset=utf-8", json.dumps({"ok": False, "error": str(e)}))
        return self._write(404, "text/plain; charset=utf-8", "Not Found")

def run_approve_ui(cfg: dict, host="127.0.0.1", port=8765):
    ApproveHandler.cfg = cfg
    httpd = HTTPServer((host, port), ApproveHandler)
    print(f"[?뱀씤 UI] http://{host}:{port} (Ctrl+C 醫낅즺)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ----------------------------- 諛쒗뻾(而ㅻ??덊떚 ?⑤룆) -----------------------------
def _today():
    return datetime.now().strftime("%Y-%m-%d")

def export_json_comm(cfg: dict):
    data = read_selection(cfg)
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))
    arts = [a for a in data.get("articles", []) if (a.get("approved") if require_appr else a.get("selected"))]
    arts.sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)
    out = {"date": _today(), "articles": arts}
    path = _os_rel(os.path.join("archive", f"而ㅻ??덊떚_{_today()}.json"))
    _save_json(path, out)
    print(f"[諛쒗뻾:JSON] {path}")

def export_md_comm(cfg: dict):
    data = read_selection(cfg)
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))
    arts = [a for a in data.get("articles", []) if (a.get("approved") if require_appr else a.get("selected"))]
    arts.sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)
    lines = [f"# ?꾨━ 而ㅻ??덊떚 ??{_today()}", ""]
    for a in arts:
        meta = f"??a.get('upvotes',0)} 쨌 ?뮠{a.get('comments',0)} 쨌 ?몓{a.get('views','-')}"
        lines.append(f"- [{a.get('title')}]({a.get('url')})  \n  {meta} 쨌 {a.get('source')}")
    path = _os_rel(os.path.join("archive", f"而ㅻ??덊떚_{_today()}.md"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[諛쒗뻾:MD] {path}")

def export_html_comm(cfg: dict):
    data = read_selection(cfg)
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))
    arts = [a for a in data.get("articles", []) if (a.get("approved") if require_appr else a.get("selected"))]
    arts.sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)

    cards = []
    for a in arts:
        title = a.get("title") or "(?쒕ぉ ?놁쓬)"
        url = a.get("url") or "#"
        meta = f"??a.get('upvotes',0)} 쨌 ?뮠{a.get('comments',0)} 쨌 ?몓{a.get('views','-')} 쨌 {a.get('source','')}"
        cards.append(
            f"<div class='card'><h2><a href='{url}' target='_blank' rel='noopener'>{title}</a></h2>"
            f"<div class='meta'>{meta}</div></div>"
        )
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>?꾨━ 而ㅻ??덊떚 ??{_today()}</title>
<style>
body{{font-family:system-ui,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}}
.meta{{color:#666;font-size:12px;margin-top:6px}}
</style><body>
<h1>?꾨━ 而ㅻ??덊떚 ??{_today()}</h1>
{''.join(cards)}
</body></html>"""
    path = _os_rel(os.path.join("archive", f"而ㅻ??덊떚_{_today()}.html"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[諛쒗뻾:HTML] {path}")

def publish_community(cfg: dict, fmt: str = "all"):
    fmt = (fmt or "all").lower()
    if fmt in ("all", "json"): export_json_comm(cfg)
    if fmt in ("all", "md"):   export_md_comm(cfg)
    if fmt in ("all", "html"): export_html_comm(cfg)


# --- ?ㅼ썙???⑦꽩 ?꾩슦誘??쒖?踰덊샇 蹂???덉슜) ---
def _kw_patterns(kw: str):
    r"""
    'IPC-A-610' 媛숈? ?ㅼ썙?쒖쓽 蹂??IPC A 610 / IPCA610 ??源뚯? 留ㅼ묶?섎뒗 ?뺢퇋??紐⑸줉.
    """
    k = (kw or "").strip()
    if not k:
        return []

    # 1) ?뺥솗??    exact = _re.escape(k)

    # 2) ?먯뒯?? ?섏씠??怨듬갚 媛蹂 ?덉슜  (?? IPC[-\s]?A[-\s]?610)
    parts = _re.split(r"[-\s]+", k.strip())
    parts = [_re.escape(p) for p in parts if p]
    loose = r"\b" + r"[-\s]?".join(parts) + r"\b" if parts else exact

    # 3) ?뺤텞?? ?뱀닔臾몄옄 ?쒓굅  (?? IPCA610)
    core = _re.sub(r"[^A-Za-z0-9]+", "", k)
    compact = _re.escape(core) if core else exact

    # 以묐났 ?쒓굅 ??而댄뙆??    uniq = {exact, loose, compact}
    return [_re.compile(p, _re.I) for p in uniq if p]


# ----------------------------------------------------------------------
# ?몃? RSS/?댁뒪 寃???섏쭛
# ----------------------------------------------------------------------
EXTERNAL_RSS_TEMPLATES: List[str] = [
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
    "https://www.bing.com/news/search?q={query}&format=rss&setlang=en-US",
]

def _parse_rss_feed(content: bytes) -> List[Dict[str, str]]:
    """RSS/Atom ?쇰뱶?먯꽌 (title, link, description, pubDate) 異붿텧."""
    items: List[Dict[str, str]] = []
    try:
        soup = BeautifulSoup(content, "xml")  # RSS/Atom??XML濡??뚯떛
    except Exception:
        return items
    for it in soup.find_all("item"):
        try:
            title = it.find("title").get_text(strip=True) if it.find("title") else ""
            link = it.find("link").get_text(strip=True) if it.find("link") else ""
            description = it.find("description").get_text(strip=True) if it.find("description") else ""
            pub = ""
            for tag in ("pubDate", "published", "dc:date"):
                if it.find(tag):
                    pub = it.find(tag).get_text(strip=True)
                    break
            items.append({"title": title, "link": link, "description": description, "published": pub})
        except Exception:
            continue
    return items

# ?ㅼ썙???숈쓽???좏깮?곸쑝濡??뺤옣)
KEYWORD_SYNONYMS: Dict[str, List[str]] = {
    "ipc-a-610": [
        "ipc a 610","ipc-a610","ipca610","ipc610","ipc 610","ipc-610",
        "j-std-001","jstd001","j std 001","ipc j-std-001","ipc-jstd-001",
        "whma-a-620","whma a 620","ipc/whma-a-620"
    ],
    "j-std-001": [
        "jstd001","j std 001","jstd-001","j-std001","ipc j-std-001","ipc-jstd-001",
        "whma-a-620","whma a 620","whma620"
    ],
    "whma-a-620": [
        "whma a 620","whma-a620","whma620","ipc/whma-a-620","ipc whma a 620","j-std-001","jstd001"
    ],
    "smt": [
        "surface mount technology","surface-mount technology","?쒕㈃ ?ㅼ옣","?쒕㈃?ㅼ옣","smd assembly"
    ],
    "reflow": [
        "reflow soldering","reflow oven","由ы뵆濡쒖슦","由ы뵆濡쒖슦 ?붾뜑留?
    ]
}

# ?좊ː ?꾨찓???쒖?/?꾩옄?앹궛/?곌뎄쨌?숈닠/?꾨Ц留ㅼ껜)
TRUST_DOMAINS: tuple = (
    # ?쒖?/湲곌?
    "ipc.org","iec.ch","jedec.org","ieee.org","nist.gov","smta.org",
    # ?꾨Ц SMT/?꾩옄?앹궛 留ㅼ껜
    "iconnect007.com","smt007.com","pcb007.com",
    "smttoday.com","smttoday.net","globalsmt.net","globalsmt.com","myglobalsmt.com","myglobalsmt.net",
    # 湲곗닠/AI/?곌뎄
    "technologyreview.com","nvidia.com","blogs.nvidia.com","developer.nvidia.com",
    "deepmind.com","openai.com","anthropic.com","meta.com","ai.googleblog.com",
    "arxiv.org","ieeexplore.ieee.org","semanticscholar.org",
    # 而ㅻ??덊떚(?곗꽑 ?щ읆)
    "eeworld.com.cn","eeworld.com"
)

def _match_external_article(entry: Dict[str, str], keyword: str, patterns: List[_re.Pattern], tokens: List[str]) -> bool:
    """?몃? RSS ??ぉ???ㅼ썙??洹쒖튃?쇰줈 ?꾪꽣留?"""
    text = " ".join([
        entry.get("title") or "",
        entry.get("description") or "",
        entry.get("link") or "",
    ])
    t = (text or "").lower()
    kw_l = (keyword or "").lower()
    if not kw_l:
        return False
    if kw_l in t:
        return True
    if patterns and any(p.search(text) for p in patterns):
        return True
    if tokens:
        for tok in tokens:
            if tok and tok in t:
                return True
    digits = "".join(ch for ch in keyword if ch.isdigit())
    if digits and (digits in t or digits in (entry.get("link") or "").lower()):
        return True
    try:
        dom = _domain_of(entry.get("link") or "")
        if dom and any(tr_d in dom for tr_d in TRUST_DOMAINS):
            return True
    except Exception:
        pass
    return False

def fetch_external_articles(keyword: str, patterns: List[_re.Pattern], tokens: List[str], max_total: int = 50) -> List[dict]:
    """?몃? RSS?먯꽌 湲곗궗 ?섏쭛 ???꾪꽣."""
    from urllib.parse import quote
    collected: List[dict] = []
    for tmpl in EXTERNAL_RSS_TEMPLATES:
        if len(collected) >= max_total:
            break
        try:
            url = tmpl.format(query=quote(keyword))
        except Exception:
            continue
        try:
            resp = requests.get(url, headers=_std_headers(), timeout=REQUEST_TO)
            if resp.status_code != 200 or not resp.content:
                continue
            entries = _parse_rss_feed(resp.content)
        except Exception:
            continue
        for ent in entries:
            if len(collected) >= max_total:
                break
            if not _match_external_article(ent, keyword, patterns, tokens):
                continue
            link = ent.get("link")
            if not link:
                continue
            cid = _article_id(link)
            if any(e.get("id") == cid for e in collected):
                continue
            desc = ent.get("description") or ent.get("title") or ""
            domain = _domain_of(link)
            trust = 0.0
            try:
                rules = _load_json_safe(_os_rel("editor_rules.json"), {})
                weights = ((rules.get("weights") or {}).get("domain") or {})
                trust = float(weights.get(domain, 0))
            except Exception:
                trust = 0.0
            hits = 0
            for tok in tokens:
                if tok and ((tok.lower() in (ent.get("title") or "").lower()) or (tok.lower() in (ent.get("description") or "").lower())):
                    hits += 1
            item = {
                "id": cid,
                "type": "news",
                "title": ent.get("title") or "",
                "source": domain,
                "url": link,
                "published_date": ent.get("published") or "",
                "summary": desc,
                "score": trust,
                "kw_hits": hits,
                "trust_score": trust,
                "approved": False,
                "editor_note": ""
            }
            collected.append(item)
    return collected


# ====== [?ㅼ썙???댁뒪 ?좏떥] ======
def _load_json_safe(p: str, default: dict) -> dict:
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as rf:
                d = json.load(rf)
                if isinstance(d, dict):
                    return d
    except Exception:
        pass
    return default

def _norm_text(s: str) -> str:
    return (s or "").strip().lower()

DATA_DIR = _os_rel("data")
os.makedirs(DATA_DIR, exist_ok=True)
KW_FILE = _os_rel("data", "selected_keyword_articles.json")


# ======================= [?듭떖] ?ㅼ썙???꾨낫 ?좊퀎 (援먯껜蹂? =======================
def build_keyword_selection(keyword: str, cfg: dict | None = None, use_external_rss: bool = False) -> str:
    """
    ?ㅼ썙???꾩슜 ?꾨낫瑜?留뚮뱾??data/selected_keyword_articles.json ?????

    ?좊퀎 洹쒖튃:
      - ?듦낵(?듭떖): ???ㅼ썙??'?좏겙' 2媛??댁긽 ?쇱튂  or  ???ㅼ썙???뺢퇋???쇱튂
      - 蹂댁“(?덉쇅 ?덉슜): '?좊ː ?꾨찓????寃쎌슦 (?좏겙?? or ?レ옄 ?뚰듃) ???듦낵(fallback)
      - ?レ옄 ?뚰듃??蹂댁“ ?좏샇濡쒕쭔 ?ъ슜(?⑤룆 ?듦낵 X)

    ?몃? RSS??(use_external_rss=True) AND (config.external_rss.enabled=True) ???뚮쭔 ?ъ슜.
    """
    kw = (keyword or "").strip()
    if not kw:
        raise ValueError("?ㅼ썙?쒓? 鍮꾩뼱?덉뒿?덈떎.")
    cfg = cfg or load_config()

    # ???寃쎈줈
    out_path = KW_FILE

    # --- ?좏겙/?뺢퇋???レ옄 以鍮?---
    tokens = _split_kw_tokens(kw)  # ?? IPC-A-610 ??["ipc","610"]
    kw_loose = re.escape(kw).replace(r"\ ", r"[\s\-]?").replace(r"\-", r"[\s\-]?")
    rx_auto = _safe_regex_list([rf"\b{kw_loose}\b"])
    num_rx = re.compile(r"\b\d{3,6}\b")

    # ?좊ː ?꾨찓???꾩뿭 + config.features.trusted_domains)
    trust_set = set()
    try:
        trust_set.update(list(TRUST_DOMAINS))
    except Exception:
        pass
    trust_set.update([d.lower() for d in (cfg.get("features", {}).get("trusted_domains") or [])])

    # ?ㅼ썙??蹂???뺢퇋??+ ?숈쓽??    pats = _kw_patterns(kw)
    syns = KEYWORD_SYNONYMS.get(kw.lower(), [])
    for s in syns:
        pats.extend(_kw_patterns(s))
    for s in syns:  # ?좏겙??蹂닿컯
        for t in _re.split(r"[^A-Za-z0-9]+", s):
            t = (t or "").lower()
            if t and (t.isdigit() or len(t) >= 2) and t not in tokens:
                tokens.append(t)

    # --- ?섏쭛蹂?濡쒕뱶: 怨듭떇/而ㅻ??덊떚 + (?듭뀡)?몃? RSS ---
    official = _load_json_safe(_os_rel("selected_articles.json"), {"articles": []})
    community = _load_json_safe(_os_rel("archive", "selected_community.json"), {"articles": []})
    candidates: list[dict] = []
    candidates.extend(official.get("articles", []))
    candidates.extend(community.get("articles", []))

    if use_external_rss and bool((cfg.get("external_rss") or {}).get("enabled", False)):
        try:
            ext_max = int((cfg.get("external_rss") or {}).get("max_total", 50))
        except Exception:
            ext_max = 50
        try:
            ext = fetch_external_articles(kw, pats, tokens, max_total=ext_max)
        except Exception as e:
            print(f"[WARN] external fetch error: {e}")
            ext = []
        candidates.extend(ext)

    # --- 以묐났 ?쒓굅(by url/id) ---
    seen = set(); uniq = []
    for a in candidates:
        aid = a.get("id") or _article_id(a.get("url") or a.get("title") or "")
        if aid and aid not in seen:
            a["id"] = aid
            uniq.append(a); seen.add(aid)
    candidates = uniq

    # --- ?좊퀎 洹쒖튃 ?곸슜 ---
    selected: list[dict] = []
    for a in candidates:
        title = a.get("title") or ""
        summ  = a.get("summary_ko_text") or a.get("summary") or " ".join(a.get("summary_ko") or [])
        blob  = f"{title}\n{summ}\n{a.get('url') or ''}\n{a.get('source') or ''}".lower()

        # 1) ?좏겙 ?덊듃(?쒕줈 ?ㅻⅨ ?좏겙 媛쒖닔)
        tok_hits = 0
        found = set()
        for t in tokens:
            if t and t not in found and re.search(rf"(?i)\b{re.escape(t)}\b", blob):
                found.add(t); tok_hits += 1

        # 2) ?뺢퇋???쇱튂(?ㅼ썙??蹂???먯뒯 ?⑦꽩)
        rx_hit = any(p.search(blob) for p in pats) or any(r.search(blob) for r in rx_auto)

        # 3) ?レ옄 ?뚰듃(?⑤룆 ?듦낵 湲덉?)
        numeric_hint = bool(num_rx.search(blob))

        # 4) ?좊ː ?꾨찓???щ?
        dom = _domain_of(a.get("url") or "")
        is_trusted = dom in trust_set

        ok_core = (tok_hits >= 2) or rx_hit
        ok_fallback = (not ok_core) and is_trusted and (tok_hits >= 1 or numeric_hint)

        if not (ok_core or ok_fallback):
            continue

        score = float(a.get("total_score") or a.get("qg_score") or a.get("score") or 0.0)
        summary_txt = summ if isinstance(summ, str) else ((" ".join(summ)).strip() if isinstance(summ, list) else "")

        selected.append({
            "id": a["id"],
            "type": (a.get("type") or ("community" if str(a.get("source") or "").startswith("[COMM]") else "news")),
            "title": title,
            "source": a.get("source") or "",
            "url": a.get("url") or "",
            "published_date": a.get("published_date") or "",
            "summary": summary_txt,
            "score": round(score, 3),
            "kw_hits": tok_hits,
            "trust_score": 1 if is_trusted else 0,
            "approved": bool(a.get("approved") or a.get("selected") or False),
            "editor_note": a.get("editor_note") or "",
            "kw_match": "rx" if rx_hit else f"tok{tok_hits}",
            "fallback": bool(ok_fallback)
        })

    # ?뺣젹: fallback ?꾨떂 ???먯닔 ???좏겙 ??    selected.sort(key=lambda x: (not x.get("fallback", False), float(x.get("score") or 0.0), int(x.get("kw_hits") or 0)), reverse=True)

    out = {"keyword": kw, "date": _today(), "articles": selected}
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, ensure_ascii=False, indent=2)
    print(f"[?ㅼ썙???묒뾽蹂???? {out_path} (珥?{len(selected)}嫄?")
    return out_path


def publish_keyword_page(keyword: str) -> str:
    """
    ?ㅼ썙???묒뾽蹂?JSON)???쎌뼱 ?곕?湲??좏삎 ?뱀뀡(?댁뒪/?쇰Ц/?쒖?/而ㅻ??덊떚)?쇰줈
    HTML/MD/JSON??archive??諛쒗뻾.
    - ?뱀씤 15嫄?誘몃쭔?대㈃ 寃쎄퀬留?異쒕젰(諛쒗뻾? 吏꾪뻾)
    - Windows ?뚯씪紐??덉쟾???ㅼ썙?쒖뿉 湲덉?臾몄옄 ?덉쓣 ???泥?
    """
    kw = (keyword or "").strip()
    if not kw:
        raise ValueError("?ㅼ썙?쒓? 鍮꾩뼱?덉뒿?덈떎.")

    # ?덉쟾 ?뚯씪紐?    import re as __re
    safe_kw = __re.sub(r'[\\/:*?"<>|]+', "-", kw).strip() or "keyword"

    # ?묒뾽蹂?濡쒕뱶
    base = _os_rel("archive")
    os.makedirs(base, exist_ok=True)
    data = _load_json_safe(KW_FILE, {"keyword": kw, "date": _today(), "articles": []})
    arts = data.get("articles", [])

    # ?뱀씤 ??寃쎄퀬
    approved_cnt = sum(1 for a in arts if a.get("approved"))
    if approved_cnt < 15:
        print(f"[WARN] ?뱀씤 {approved_cnt}嫄?(<15). ?몄쭛???뱀씤 ???꾩슂?⑸땲??")

    # ?곕? ?뺣젹
    arts_sorted = sorted(arts, key=lambda a: a.get("published_date") or "")

    # ?좏삎 ?뱀뀡 援ъ꽦
    sections = {"news": [], "paper": [], "standard": [], "community": []}
    for a in arts_sorted:
        tp = (a.get("type") or "").strip().lower()
        if tp not in sections:
            tp = "news"
        sections[tp].append(a)

    today = _today()

    # JSON
    jpath = _os_rel("archive", f"{today}_{safe_kw}.json")
    with open(jpath, "w", encoding="utf-8") as jf:
        json.dump({"keyword": kw, "date": today, "sections": sections}, jf, ensure_ascii=False, indent=2)
    print(f"[諛쒗뻾:JSON] {jpath}")

    # MD
    mpath = _os_rel("archive", f"{today}_{safe_kw}.md")
    sec_title = {"news":"?댁뒪", "paper":"?쇰Ц/蹂닿퀬??, "standard":"?쒖?/?뺤콉", "community":"而ㅻ??덊떚"}
    md = [f"# ?ㅼ썙????{kw} ({today})", ""]
    for sec in ("news","paper","standard","community"):
        items = sections.get(sec) or []
        if not items:
            continue
        md.append(f"## {sec_title[sec]}")
        for a in items:
            title = a.get("title") or "(?쒕ぉ ?놁쓬)"
            url = a.get("url") or "#"
            note = (a.get("editor_note") or "").strip()
            ns = f" ???몄쭛?? {note}" if note else ""
            md.append(f"- [{title}]({url}){ns}")
        md.append("")
    with open(mpath, "w", encoding="utf-8") as mf:
        mf.write("\n".join(md))
    print(f"[諛쒗뻾:MD] {mpath}")

    # HTML
    cards = []
    for sec in ("news","paper","standard","community"):
        items = sections.get(sec) or []
        if not items:
            continue
        cards.append(f"<h2>{sec_title[sec]}</h2>")
        for a in items:
            title = a.get("title") or "(?쒕ぉ ?놁쓬)"
            url = a.get("url") or "#"
            score = a.get("score") or a.get("total_score") or 0
            note = (a.get("editor_note") or "").strip()
            meta = f"score {score}"
            note_s = f"<div class='note'>?몄쭛????{note}</div>" if note else ""
            cards.append(
                f"<div class='card'><h3><a href='{url}' target='_blank' rel='noopener'>{title}</a></h3>"
                f"<div class='meta'>{meta}</div>{note_s}</div>"
            )

    hpath = _os_rel("archive", f"{today}_{safe_kw}.html")
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>{kw} ??{today}</title>
<style>
body{{font-family:system-ui,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px;background:#fff;color:#111}}
h1{{font-size:22px;margin-bottom:6px}}
h2{{margin-top:22px;border-top:1px solid #eee;padding-top:12px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}}
.meta{{color:#666;font-size:12px;margin-top:6px}}
.note{{margin-top:6px;font-size:13px}}
</style><body>
<h1>?ㅼ썙????{kw} ({today})</h1>
{''.join(cards)}
</body></html>"""
    with open(hpath, "w", encoding="utf-8") as hf:
        hf.write(html)
    print(f"[諛쒗뻾:HTML] {hpath}")

    return hpath


# ======================================================================
#                      蹂묓빀 諛쒗뻾(怨듭떇 + 而ㅻ??덊떚)
# ======================================================================
def _approved_items(items: List[Dict], require_appr: bool) -> List[Dict]:
    return [a for a in items if (a.get("approved") if require_appr else a.get("selected"))]

def export_all(cfg: dict, fmt: str = "all"):
    """怨듭떇 + 而ㅻ??덊떚瑜??뱀뀡 ?쒖꽌?濡?蹂묓빀 諛쒗뻾."""
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))

    off = read_selection_official(cfg).get("articles", [])
    com = read_selection(cfg).get("articles", [])

    off_a = _approved_items(off, require_appr)
    com_a = _approved_items(com, require_appr)

    by_sec = {s: [] for s in SECTION_ORDER}
    for a in off_a + com_a:
        sec = a.get("section") or "湲濡쒕쾶 ?꾩옄?앹궛"
        by_sec.setdefault(sec, []).append(a)

    for s in by_sec:
        by_sec[s].sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)

    today = _today()

    # JSON
    if fmt in ("all", "json"):
        out = {"date": today, "sections": {s: by_sec[s] for s in SECTION_ORDER}}
        p = _os_rel(os.path.join("archive", f"?꾨━?댁뒪_{today}.json"))
        _save_json(p, out); print(f"[諛쒗뻾:JSON] {p}")

    # MD
    if fmt in ("all", "md"):
        lines = [f"# ?꾨━?댁뒪 ??{today}", ""]
        for s in SECTION_ORDER:
            if not by_sec[s]:
                continue
            lines.append(f"## {s}")
            for a in by_sec[s]:
                note = (a.get("editor_note") or "").strip()
                note_s = (f" ???몄쭛?? {note}" if note else "")
                lines.append(f"- [{a.get('title')}]({a.get('url')}){note_s}")
            lines.append("")
        p = _os_rel(os.path.join("archive", f"?꾨━?댁뒪_{today}.md"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[諛쒗뻾:MD] {p}")

    # HTML
    if fmt in ("all", "html"):
        cards=[]
        for s in SECTION_ORDER:
            if not by_sec[s]:
                continue
            cards.append(f"<h2>{s}</h2>")
            for a in by_sec[s]:
                note = (a.get("editor_note") or "").strip()
                meta = []
                if a.get("source"): meta.append(a["source"])
                if a.get("total_score"): meta.append(f"score {a['total_score']}")
                meta_s = " 쨌 ".join(meta)
                note_s = (f"<div class='note'>?몄쭛???쒕쭏????{note}</div>" if note else "")
                cards.append(
                    f"<div class='card'><h3><a href='{a.get('url','#')}' target='_blank' rel='noopener'>{a.get('title') or '(?쒕ぉ ?놁쓬)'}</a></h3>"
                    f"<div class='meta'>{meta_s}</div>{note_s}</div>"
                )
        html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>?꾨━?댁뒪 ??{today}</title>
<style>
body{{font-family:system-ui,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px;background:#fff;color:#111}}
h1{{font-size:22px;margin-bottom:6px}}
h2{{margin-top:22px;border-top:1px solid #eee;padding-top:12px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}}
.meta{{color:#666;font-size:12px;margin-top:6px}}
.note{{margin-top:6px;font-size:13px}}
</style><body>
<h1>?꾨━?댁뒪 ??{today}</h1>
{''.join(cards)}
</body></html>"""
        p = _os_rel(os.path.join("archive", f"?꾨━?댁뒪_{today}.html"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[諛쒗뻾:HTML] {p}")


# ----------------------------- ?ㅼ젙 濡쒕뱶 諛?main -----------------------------
# deep_merge function removed ??use util_deep_merge from common_utils instead.

def load_config() -> dict:
    """
    config.json???쎌뼱 DEFAULT_CFG? 蹂묓빀?섏뿬 諛섑솚.
    - ?뚯씪???녾굅???뚯떛 ?ㅽ뙣?대룄 ??긽 dict瑜?諛섑솚(?덈? None 諛섑솚?섏? ?딆쓬)
    """
    cfg_path = qj_rel("config.json")
    user_cfg: dict = {}

    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as rf:
                data = json.load(rf)
                if isinstance(data, dict):
                    user_cfg = data
    except Exception as e:
        print(f"[WARN] config.json 濡쒕뱶 ?ㅻ쪟: {e} ??湲곕낯媛믪쑝濡?吏꾪뻾")

    return deep_merge(DEFAULT_CFG, user_cfg)

def _auto_approve_keyword(kw: str, top_n: int = 15, min_score: float = 0.0):
    """
    selected_articles.json + archive/selected_community.json?먯꽌
    ?ㅼ썙??蹂???ы븿) ?쇱튂 & ?먯닔 議곌굔 留뚯”?섎뒗 ??ぉ???곸쐞 N媛?'approved=True'濡??쒖떆(?щ엺 蹂댁“).
    """
    if not kw:
        raise ValueError("?ㅼ썙?쒓? 鍮꾩뿀?듬땲??")
    pats = _kw_patterns(kw)

    def _load(p, d):  # ?덉쟾 濡쒕뜑
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as rf:
                    j = json.load(rf)
                    return j if isinstance(j, dict) else d
        except Exception:
            pass
        return d

    off_p = _os_rel("selected_articles.json")
    com_p = _os_rel("archive", "selected_community.json")

    off = _load(off_p, {"articles": []})
    com = _load(com_p, {"articles": []})

    def hits(items):
        rows = []
        for a in items:
            text = " ".join([
                a.get("title") or "", a.get("summary") or "",
                a.get("summary_ko_text") or "", " ".join(a.get("summary_ko") or [])
            ])
            t = text.lower()
            score = float(a.get("total_score") or a.get("qg_score") or 0)
            if kw.lower() in t or any(p.search(text) for p in pats):
                if score >= min_score:
                    rows.append((score, a))
        rows.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in rows]

    picked = (hits(off.get("articles", [])) + hits(com.get("articles", [])))[:max(1, int(top_n))]
    ids = {a.get("id") for a in picked if a.get("id")}
    for a in off.get("articles", []):
        if a.get("id") in ids:
            a["approved"] = True
    for a in com.get("articles", []):
        if a.get("id") in ids:
            a["approved"] = True

    with open(off_p, "w", encoding="utf-8") as wf:
        json.dump(off, wf, ensure_ascii=False, indent=2)
    with open(com_p, "w", encoding="utf-8") as wf:
        json.dump(com, wf, ensure_ascii=False, indent=2)

    print(f"[AUTO] '{(kw or '').strip()}' ?쇱튂 ?곸쐞 {len(ids)}嫄??먮룞 ?뱀씤 ?꾨즺(?먯닔 ??{min_score}).")


def _slug_kw_hist(s: str) -> str:
    """
    ?ㅼ썙??臾몄옄?댁쓣 ?꾩뭅?대툕 ?뚯씪紐낆뿉 ?ъ슜?섎뒗 ?щ윭洹몃줈 蹂?섑븳??

    ?곷Ц/?レ옄/?섏씠???몃뜑諛?留덉묠?쒕쭔 ?좎??섎ŉ, ?ㅻⅨ 臾몄옄???섏씠?덉쑝濡?移섑솚?쒕떎.
    ?臾몄옄濡?蹂?섑븯??IPC?멭XX ???쒖? 踰덊샇 ?뺥깭瑜?蹂댁〈?쒕떎.

    Args:
        s: ?먮낯 ?ㅼ썙??    Returns:
        ?꾩뭅?대툕 ?뚯씪?먯꽌 ?ъ슜?섎뒗 ?щ윭洹??臾몄옄)
    """
    if not s:
        return ""
    slug = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in s).strip("-")
    return slug.upper()


def collect_keyword_history(keyword: str, cfg: dict) -> dict:
    """
    二쇱뼱吏??ㅼ썙?쒖뿉 ???怨쇨굅 諛쒗뻾 ?덉뒪?좊━瑜?遺꾩꽍?쒕떎.

    ?꾩뭅?대툕 ?붾젆?곕━?먯꽌 ``YYYY-MM-DD_{SLUG}.json`` ?⑦꽩???뚯씪??李얠븘
    ?좎쭨蹂?湲곗궗 媛쒖닔瑜?吏묎퀎?섍퀬, 媛??理쒓렐 諛쒗뻾?됯낵 怨쇨굅 ?됯퇏??鍮꾧탳?섏뿬
    ?곸듅/媛먯냼 異붿꽭瑜?怨꾩궛?쒕떎. 寃곌낵??JSON ?뺥깭濡?由ы룷???붾젆?곕━????λ릺硫?    ?쒖? 異쒕젰?쇰줈??諛섑솚?쒕떎.

    Args:
        keyword: 遺꾩꽍 ????ㅼ썙??(?먮낯 臾몄옄??
        cfg: ?ㅼ젙 ?뺤뀛?덈━ (paths.archive, paths.reports ?ъ슜)

    Returns:
        dict: ``keyword``, ``slug``, ``date``, ``trend_ratio``, ``description``, ``timeline`` ?ы븿
    """
    slug = _slug_kw_hist(keyword)
    # ?꾩뭅?대툕 寃쎈줈 諛?由ы룷??寃쎈줈 寃곗젙
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    archive_dir = paths.get("archive") or "archive"
    reports_dir = paths.get("reports") or os.path.join(archive_dir, "reports")
    # ?좎쭨蹂?湲곗궗 ??吏묎퀎
    counts: dict[str, int] = {}
    try:
        for fname in os.listdir(archive_dir):
            # JSON ?뚯씪留???곸쑝濡??쒕떎.
            if not fname.lower().endswith(".json"):
                continue
            # ?뚯씪紐낆씠 ?좎쭨 ?⑦꽩怨??щ윭洹몃? 紐⑤몢 ?ы븿?댁빞 ?쒕떎.
            if slug not in fname.upper():
                continue
            # ?욎쓽 10?먮━(YYYY-MM-DD) 異붿텧
            date_part = fname[:10]
            if not re.match(r"\d{4}-\d{2}-\d{2}", date_part):
                continue
            # ?щ윭洹??쇱튂 ?щ? ?뺤씤
            parts = fname[11:].split("_")
            if not parts:
                continue
            slug_part = parts[0]
            if slug_part.upper() != slug:
                continue
            # 湲곗궗 媛쒖닔 ?쎄린
            fpath = os.path.join(archive_dir, fname)
            count = 0
            try:
                with open(fpath, "r", encoding="utf-8") as rf:
                    obj = json.load(rf)
                    arts = obj.get("articles", [])
                    count = len(arts) if isinstance(arts, list) else 0
            except Exception:
                pass
            counts[date_part] = counts.get(date_part, 0) + count
    except FileNotFoundError:
        counts = {}
    # ??꾨씪???뺣젹
    timeline = [{"date": d, "count": counts[d]} for d in sorted(counts.keys())]
    # 異붿꽭 怨꾩궛
    trend_ratio = 0.0
    desc = "?곗씠?곌? ?놁뒿?덈떎."
    if timeline:
        last_date = timeline[-1]["date"]
        last_count = timeline[-1]["count"]
        prev_counts = [item["count"] for item in timeline[:-1]]
        avg_prev = sum(prev_counts) / len(prev_counts) if prev_counts else last_count
        denom = avg_prev if avg_prev != 0 else 1.0
        trend_ratio = (last_count - avg_prev) / denom
        trend_pct = round(trend_ratio * 100.0, 1)
        trend_word = "?곸듅" if last_count >= avg_prev else "媛먯냼"
        desc = f"{last_date} 諛쒗뻾 {last_count}嫄? ?댁쟾 ?됯퇏 {avg_prev:.1f}嫄???{trend_word} {trend_pct:+.1f}%"
    result = {
        "keyword": keyword,
        "slug": slug,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "trend_ratio": trend_ratio,
        "description": desc,
        "timeline": timeline,
    }
    # 由ы룷?????    try:
        os.makedirs(reports_dir, exist_ok=True)
        outfile = os.path.join(
            reports_dir,
            f"{result['date']}_{slug}_history.json",
        )
        with open(outfile, "w", encoding="utf-8") as wf:
            json.dump(result, wf, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # STDOUT 異쒕젰
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    # (?좏깮) 怨듭떇 ?뚮옒洹?    ap.add_argument("--collect", action="store_true", help="Collect official sources (smt/supply)")
    ap.add_argument("--publish", action="store_true", help="Publish merged (official + community)")
    # 而ㅻ??덊떚 ?뚮옒洹?    ap.add_argument("--collect-community", action="store_true", help="Collect community candidates")
    ap.add_argument("--approve-ui", action="store_true", help="Open approval UI (http://127.0.0.1:8765)")
    ap.add_argument("--approve-top", type=int, default=0, help="Automatically approve top N")
    ap.add_argument("--publish-community", action="store_true", help="Publish approved community items")
    ap.add_argument("--format", choices=["all","html","md","json"], default="all", help="Publish format")
    # ?ㅼ썙??紐⑤뱶
    ap.add_argument("--collect-keyword", type=str, default="", help="?ㅼ썙???묒뾽蹂??앹꽦")
    ap.add_argument("--publish-keyword", type=str, default="", help="?ㅼ썙???묒뾽蹂?諛쒗뻾")
    ap.add_argument("--approve-keyword-top", type=int, default=0, help="?ㅼ썙???쇱튂 ?곸쐞 N ?먮룞 ?뱀씤(蹂댁“)")
    ap.add_argument("--approve-keyword", type=str, default="", help="?먮룞 ?뱀씤???ъ슜???ㅼ썙??)
    # [NEW] ?몃? RSS ?ъ슜 ?щ?
    ap.add_argument("--use-external-rss", action="store_true", help="?ㅼ썙???섏쭛 ???몃? RSS(?댁뒪 寃?????④퍡 ?ъ슜")

    # [NEW] ?ㅼ썙????궗/?몃젋??遺꾩꽍 ?듭뀡
    ap.add_argument(
        "--collect-keyword-history",
        type=str,
        default="",
        help="吏?뺥븳 ?ㅼ썙?쒖쓽 怨쇨굅 諛쒗뻾?됱쓣 遺꾩꽍?섍퀬 ?곸듅/媛먯냼 異붿꽭瑜?怨꾩궛?⑸땲??,
    )

    # ?몃젋???곗씠???낅뜲?댄듃: trend JSON ?뚯씪??諛쏆븘 editor_rules.json怨?keyword_synonyms.json 媛깆떊
    ap.add_argument("--update-trend-data", type=str, default="", help="?몃젋???ㅼ썙???곗씠??JSON???쎌뼱 媛以묒튂? ?숈쓽?대? 媛깆떊?⑸땲??)

    args = ap.parse_args(argv)
    # Emit a summary of how the orchestrator was invoked.  This helps
    # troubleshoot issues when invoked from the admin API.  The arguments are
    # reconstructed from sys.argv if argv is None.  We intentionally log
    # user?몊upplied options only and avoid logging full configuration data.
    try:
        cmd_args = argv if argv is not None else sys.argv[1:]
        logger.info(f"orchestrator start args: {' '.join(str(a) for a in cmd_args)}")
    except Exception:
        # Fail silently if logging cannot be performed at this early stage.
        pass
    cfg = load_config()

    # --- ?몃젋???ㅼ썙???곗씠???낅뜲?댄듃 ---
    if args.update_trend_data:
        try:
            from tools.update_trend_data import update_trend_data
        except Exception:
            update_trend_data = None
        if update_trend_data:
            trend_file = args.update_trend_data
            editor_file = os.path.join(os.getcwd(), "editor_rules.json")
            synonym_file = os.path.join(os.getcwd(), "keyword_synonyms.json")
            update_trend_data(Path(trend_file), Path(editor_file), Path(synonym_file))
        else:
            print("update_trend_data module not available")
        return

    # --- ?ㅼ썙????궗/?몃젋??遺꾩꽍 ---
    if args.collect_keyword_history:
        # 吏?뺥븳 ?ㅼ썙?쒖쓽 怨쇨굅 諛쒗뻾?됱쓣 遺꾩꽍?⑸땲??
        collect_keyword_history(args.collect_keyword_history, cfg)
        return

    # --- ?ㅼ썙??紐⑤뱶(?쇱씠?? ---
    if args.collect_keyword:
        build_keyword_selection(args.collect_keyword, cfg, use_external_rss=args.use_external_rss)
        return

    if args.publish_keyword:
        publish_keyword_page(args.publish_keyword)
        return

    # --- ?ㅼ썙???먮룞 ?뱀씤(蹂댁“) ---
    if args.approve_keyword_top > 0 and args.approve_keyword:
        _auto_approve_keyword(args.approve_keyword, args.approve_keyword_top, min_score=0.0)
        return

    # --- 怨듭떇 ?섏쭛 ---
    if args.collect:
        arts = collect_official(cfg)
        write_selection_official(arts, cfg)
        return

    # --- 而ㅻ??덊떚 ?섏쭛/?뱀씤/諛쒗뻾 ---
    if args.collect_community:
        arts = collect_community(cfg)
        write_selection(arts, cfg)
        return

    if args.approve_ui:
        if not os.path.exists(_sel_path(cfg)):
            arts = collect_community(cfg)
            write_selection(arts, cfg)
        run_approve_ui(cfg)
        return

    if args.approve_top and args.approve_top > 0:
        data = read_selection(cfg)
        arts = sorted(data.get("articles", []), key=lambda x: x.get("total_score", 0), reverse=True)
        for a in arts[:args.approve_top]:
            a["approved"] = True
        _save_json(_sel_path(cfg), {"date": data.get("date") or _today(), "articles": arts})
        print(f"[?먮룞 ?뱀씤] ?곸쐞 {args.approve_top}嫄??뱀씤 ?꾨즺")
        return

    if args.publish_community:
        publish_community(cfg, fmt=args.format)
        return

    # --- 蹂묓빀 諛쒗뻾(怨듭떇+而ㅻ??덊떚) ---
    if args.publish:
        export_all(cfg, fmt=args.format)
        return

    ap.print_help()


if __name__ == "__main__":
    main()

