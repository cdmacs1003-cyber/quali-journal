# -*- coding: utf-8 -*-
"""
QualiJournal Orchestrator
- 공식(official)·커뮤니티(community) 소스 수집, 선별/승인, 발행(HTML/MD/JSON)
- 키워드 기반 선별/발행 및 키워드 히스토리 생성
- 외부 의존(common_utils/logging_setup/qj_paths)이 없어도 안전하게 동작
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import argparse
import hashlib
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin, quote_plus

import requests
from bs4 import BeautifulSoup
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------
# 로깅: logging_setup 모듈이 있으면 사용, 없으면 표준 logging 사용
# ---------------------------------------------------------------------
try:
    from logging_setup import setup_logger  # type: ignore
    _ROOT_DIR = Path(__file__).resolve().parent
    _LOG_PATH = _ROOT_DIR / "logs" / "orchestrator.log"
    logger = setup_logger("orchestrator", str(_LOG_PATH))
except Exception:
    import logging
    _ROOT_DIR = Path(__file__).resolve().parent
    _LOG_PATH = _ROOT_DIR / "logs" / "orchestrator.log"
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(_LOG_PATH),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )
    logger = logging.getLogger("orchestrator")
    logger.info("fallback logger initialized")

# ---------------------------------------------------------------------
# 환경/기본값
# ---------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0 Safari/537.36 QualiNewsBot/2025-10"
)
REQUEST_TO = (8, 16)  # (connect timeout, read timeout)

DEFAULT_CFG: Dict = {
    "paths": {
        "archive": "archive",
        "reports": "archive/reports",
        "selection_file": "selected_articles.json",                   # 공식 선택본
        "community_selection_file": "archive/selected_community.json",# 커뮤니티 선택본
        "community_sources_candidates": [                             # community_sources.json 탐색 후보
            "feeds/community_sources.json",
            "community_sources.json",
        ],
        "keywords_txt": "community_keywords.txt",                     # 커뮤니티 키워드 파일(옵션)
        "smt_sources_file": "smt_sources.json",
        "supply_sources_file": "supply_sources.json",
    },
    "features": {
        "require_editor_approval": True,      # 발행 시 편집자 승인 필요
        "output_formats": ["html", "md", "json"],
        "trusted_domains": [],                # 가산점 도메인 추가(옵션)
    },
    "community": {
        "enabled": True,
        "fresh_hours": 336,                   # 2주
        "min_upvotes": 0,
        "min_comments": 0,
        "score_threshold": 0.0,
        "max_total": 300,
        "reddit_pages": 5,
        "score_weights": {"keyword": 3, "upvotes": 5, "views": 2},
        "norms": {"kw_base": 2, "upvotes_max": 200, "views_max": 100000},
        "filters": {}                         # config.json의 community.filters와 머지
    },
    "external_rss": {
        "enabled": False,
        "max_total": 50
    },
}

SECTION_ORDER = ["정책/표준", "일반 뉴스", "AI 뉴스", "커뮤니티"]

# ---------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------
def _std_headers() -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
    }

def _os_rel(*p: str) -> str:
    """프로젝트 루트 기준 상대경로."""
    return os.path.join(str(_ROOT_DIR), *p)

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
    """프래그먼트·꼬리 슬래시 제거."""
    try:
        u = (u or "").strip()
        u = re.sub(r"#.*$", "", u)
        u = re.sub(r"/+$", "", u)
        return u
    except Exception:
        return u or ""

def _split_kw_tokens(kw: str) -> List[str]:
    """키워드를 토큰으로 분해(숫자·영문, 길이 2+ 유지)."""
    if not kw:
        return []
    toks = re.split(r"[^\w\-/+\.]+", kw)
    return [t.lower() for t in toks if t and (t.isdigit() or len(t) >= 2)]

def _safe_regex_list(patterns: List[str]):
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

def _hours_ago(ts_iso: Optional[str]) -> float:
    if not ts_iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (_now_utc() - dt).total_seconds() / 3600.0
    except Exception:
        return 0.0

def _article_id(url: str) -> str:
    return hashlib.md5(canonical_url(url).encode("utf-8")).hexdigest()

def _norm01_log(x: float, maxv: float) -> float:
    if maxv <= 0:
        return 0.0
    return min(1.0, math.log1p(max(0.0, float(x))) / math.log1p(float(maxv)))

def _value_score(kw_raw: int, upvotes: int, views: int, cfg: dict) -> float:
    c = (cfg.get("community") or {})
    W = (c.get("score_weights") or {"keyword": 3, "upvotes": 5, "views": 2})
    N = (c.get("norms") or {"kw_base": 2, "upvotes_max": 200, "views_max": 100000})
    kw = min(1.0, (kw_raw or 0) / max(1, int(N.get("kw_base", 2))))
    up = _norm01_log(upvotes or 0, int(N.get("upvotes_max", 200)))
    vw = _norm01_log(views or 0, int(N.get("views_max", 100000)))
    return float(W.get("keyword", 3)) * kw + float(W.get("upvotes", 5)) * up + float(W.get("views", 2)) * vw

# =============================== VALUE GATE ===============================
def _safe_load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _load_editor_rules() -> dict:
    # CWD 기준 editor_rules.json, 없으면 기본값
    rules = _safe_load_json(os.path.join(os.getcwd(), "editor_rules.json"))
    # 기본값 보강
    rules.setdefault("score_weights", {"domain_trust":0.30,"doc_type":0.25,"standards_coverage":0.25,"tech_depth":0.15,"citations":0.05})
    rules.setdefault("doc_type_weights", {"pdf":1.0,"doc":0.9,"docx":0.9,"ppt":0.75,"pptx":0.75,"xls":0.7,"xlsx":0.7,"html":0.6,"md":0.6,"text":0.35})
    rules.setdefault("domain_weights", {})
    rules.setdefault("thresholds", {"auto_approve":0.65,"needs_review_min":0.50})
    rules.setdefault("modifiers", {"doc_type_requires_standard":True,"doc_type_factor_if_no_standard":0.5})
    rules.setdefault("standards_bundles", {})
    return rules

def _guess_doc_type(url: str) -> str:
    u = (url or "").lower()
    if re.search(r"\.pdf(\?|$)", u): return "pdf"
    if re.search(r"\.(docx?|rtf)(\?|$)", u): return "docx" if ".docx" in u else "doc"
    if re.search(r"\.pptx?(\?|$)", u): return "pptx" if ".pptx" in u else "ppt"
    if re.search(r"\.xlsx?(\?|$)", u): return "xlsx" if ".xlsx" in u else "xls"
    if re.search(r"\.(md|markdown)(\?|$)", u): return "md"
    if re.search(r"^https?://", u): return "html"
    return "text"

def _doc_type_score(tp: str, rules: dict, has_standard: bool) -> float:
    w = (rules.get("doc_type_weights") or {})
    base = float(w.get(tp or "text", w.get("text", 0.35)))
    mods = (rules.get("modifiers") or {})
    if mods.get("doc_type_requires_standard", True) and not has_standard:
        base *= float(mods.get("doc_type_factor_if_no_standard", 0.5))
    return max(0.0, min(1.0, base))

def _domain_trust(url: str, rules: dict) -> float:
    host = _domain_of(url)
    wmap = (rules.get("domain_weights") or {})
    w = float(wmap.get(host, 0.0))
    return max(0.0, min(1.0, w))

def _standards_coverage(text: str, rules: dict, synonyms: dict) -> float:
    text = (text or "").lower()
    bundles = (rules.get("standards_bundles") or {})
    core_hit = 0; ctx_hit = 0
    # 1) bundles 우선
    for b in bundles.values():
        for c in (b.get("core") or []):
            if c and c.lower() in text:
                core_hit += 1; break
        for c in (b.get("context") or []):
            if c and c.lower() in text:
                ctx_hit += 1; break
    # 2) synonyms 보조(core로만 취급)
    for k, arr in (synonyms or {}).items():
        for c in arr or []:
            if c and c.lower() in text:
                core_hit += 1; break
    if core_hit==0 and ctx_hit==0: return 0.0
    if core_hit>=1 and ctx_hit>=1: return 1.0
    if core_hit>=1 and ctx_hit==0: return 0.6
    return 0.3

def _tech_depth(text: str) -> float:
    text = (text or "").lower()
    cues = ["procedure","inspection","class 3","rework","acceptability","acceptance criteria","defect","workmanship","requirements","figure","table"]
    hits = sum(1 for c in cues if c in text)
    return max(0.0, min(1.0, hits/6.0))

def _citations(text: str) -> float:
    # 간단: 링크 갯수로 근사
    n = len(re.findall(r"https?://", text or ""))
    return max(0.0, min(1.0, n/5.0))

def compute_value_score(a: dict, cfg: dict) -> tuple[float, dict]:
    rules = _load_editor_rules()
    w = (rules.get("score_weights") or {})
    syn = _safe_load_json(os.path.join(os.getcwd(), "keyword_synonyms.json"))
    title = a.get("title") or ""
    body  = a.get("summary_ko_text") or a.get("summary") or a.get("selftext") or ""
    url   = a.get("url") or ""
    blob  = f"{title}\n{body}\n{url}"
    has_std = _standards_coverage(blob, rules, syn)
    dtp = _guess_doc_type(url)
    sc_domain = _domain_trust(url, rules)
    sc_doc    = _doc_type_score(dtp, rules, has_std >= 0.6)
    sc_std    = has_std
    sc_depth  = _tech_depth(blob)
    sc_cite   = _citations(blob)
    V = (w.get("domain_trust",0.3)*sc_domain + w.get("doc_type",0.25)*sc_doc +
         w.get("standards_coverage",0.25)*sc_std + w.get("tech_depth",0.15)*sc_depth +
         w.get("citations",0.05)*sc_cite)
    brk = {"domain_trust":round(sc_domain,3),"doc_type":round(sc_doc,3),"standards":round(sc_std,3),
           "tech_depth":round(sc_depth,3),"citations":round(sc_cite,3),"doc_type_name":dtp}
    return round(float(V),3), brk

def _state_from_value(v: float) -> str:
    rules = _load_editor_rules()
    th = (rules.get("thresholds") or {})
    auto_ready = float(th.get("auto_approve", 0.65))
    need_min   = float(th.get("needs_review_min", 0.50))
    if v >= auto_ready: return "ready"
    if v >= need_min:   return "candidate"
    return "rejected"
# ============================ /VALUE GATE END ============================


def deep_merge(a: dict, b: dict) -> dict:
    """dict 재귀 병합(a <- b)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# ---------------------------------------------------------------------
# 제목 필터
# ---------------------------------------------------------------------
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

# =====================================================================
#                         공식(official)
# =====================================================================
def _sel_official_path(cfg: dict) -> str:
    p = ((cfg.get("paths") or {}).get("selection_file")) or "selected_articles.json"
    return _os_rel(p)

def read_selection_official(cfg: dict) -> dict:
    path = _sel_official_path(cfg)
    data = _load_json(path, {}) or {}
    data.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
    data.setdefault("articles", [])
    return data

def write_selection_official(articles: List[Dict], cfg: dict) -> str:
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
    return path

def _pick_first_article_url(base_url: str, html: str) -> Optional[str]:
    """리스트 페이지에서 첫 기사 URL을 추출."""
    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.select('a[href]'):
        href = a.get("href") or ""          # ← 필수: href 미정의 방지
        # 사이트마다 다르지만 대개 article/news 키워드를 포함
        if "/article/" in href or "/news/" in href:
            return canonical_url(urljoin(base_url, href))
    a = soup.find("a", href=True)
    return canonical_url(urljoin(base_url, a["href"])) if a else None

def _fetch_title_desc(url: str) -> tuple[str, str]:
    """기사 페이지에서 제목/설명 추출."""
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
    if any(k in lbl for k in ["ecss", "ipc", "nasa", "mil"]) or "ecss.nl" in d or "ipc.org" in d:
        return "정책/표준"
    if any(k in lbl for k in ["ai", "openai", "x.ai", "gemini", "meta"]):
        return "AI 뉴스"
    return "일반 뉴스"

def collect_official(cfg: dict) -> List[Dict]:
    """smt/supply 소스에서 첫 기사 링크 추출."""
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
                    "pinned": False, "pin_ts": 0,
                    "summary_ko_text": (desc or None)
                }
                # --- Value Gate 적용 ---
                vscore, brk = compute_value_score({"title": title, "summary_ko_text": desc or "", "url": first}, cfg)
                art["doc_type"] = brk.get("doc_type_name","text")
                art["value_score"] = vscore
                art["scores_breakdown"] = brk
                art["total_score"] = vscore
                art["state"] = _state_from_value(vscore)  # candidate/ready/rejected
                art["approved"] = False
                art["selected"] = False
                items.append(art)
            except Exception:
                continue

    _from_dict(smt)
    _from_dict(sup)

    # URL 중복 제거
    seen = set(); uniq = []
    for a in items:
        u = a.get("url")
        if u in seen:
            continue
        uniq.append(a); seen.add(u)
    return uniq

# =====================================================================
#                         커뮤니티(community)
# =====================================================================
def _reddit_old_new(sub: str, pages: int = 1) -> List[Dict]:
    """
    old.reddit.com/r/<sub>/new/ 목록 수집.
    반환: dict(title,url,upvotes,comments,ts,selftext)
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
                    up = int(re.sub(r"\D", "", sc.get("title") or "0") or 0)
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
            # selftext 보강(가능할 때만)
            try:
                jurl = post["url"] + ".json"
                jr = requests.get(jurl, headers=headers, timeout=REQUEST_TO)
                if jr.status_code == 200:
                    jd = jr.json()
                    if isinstance(jd, list) and jd:
                        data = jd[0].get("data", {}).get("children", [])
                        if data and isinstance(data, list) and data[0].get("data"):
                            d0 = data[0].get("data", {})
                            post["selftext"] = d0.get("selftext") or ""
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
            row = a.find_parent(["tr", "li", "div"])
            txt = (row.get_text(" ", strip=True) if row else "")
            def _pick(rx):
                m = re.search(rx, txt, flags=re.I)
                return int(re.sub(r"[^\d]", "", (m.group(1) if m else "") or "0") or 0)
            vw = _pick(r"views?\s*[:\-\s]*([0-9,\.]+)") or _pick(r"조회\s*[:\-\s]*([0-9,\.]+)")
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

    cand = cpaths.get("community_sources_candidates") or ["feeds/community_sources.json", "community_sources.json"]
    src_path = _find_first([_os_rel(p) for p in cand]) or ""
    if not src_path:
        logger.warning("community_sources.json not found in candidates: %s", cand)
    sources = _load_json(src_path, {}) if src_path else {}
    reddit_cfg = (sources.get("reddit") or {})
    forums_cfg = (sources.get("forums") or [])
    filters = (sources.get("filters") or {})

    # config.json의 community.filters와 병합
    filters = deep_merge(filters, ((cfg.get("community") or {}).get("filters") or {}))

    allow_domains = set(filters.get("allow_domains") or [])
    min_title_len = int(filters.get("min_title_len") or 0)
    title_blockers = _compile_title_blockers(filters)

    # 키워드
    kw_list = _read_lines(cpaths.get("keywords_txt"))
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
            if int(r.get("upvotes", 0)) < min_up or int(r.get("comments", 0)) < min_cm:
                continue
            combined_text = (title or "") + " " + (r.get("selftext") or "")
            kw_hits = 0
            t = combined_text.lower()
            for kw in kw_list:
                if kw.lower() in t:
                    kw_hits += 1
            if require_kw and kw_hits <= 0:
                continue
            # --- Value Gate로 재점수화 ---
            tmp_article = {
                "title": title, "summary_ko_text": r.get("selftext") or "",
                "url": url
            }
            vscore, brk = compute_value_score(tmp_article, cfg)
            state = _state_from_value(vscore)
            if state == "rejected":
                continue
            items.append({
                "id": _article_id(url),
                "title": title,
                "url": url,
                "source": f"[COMM][Reddit/{sub}]",
                "upvotes": int(r.get("upvotes", 0)),
                "comments": int(r.get("comments", 0)),
                "views": 0,
                "ts": r.get("ts"),
                "kw_hits": kw_hits,
                "doc_type": brk.get("doc_type_name","text"),
                "value_score": vscore,
                "scores_breakdown": brk,
                "total_score": vscore,            # 기존 정렬 호환
                "section": "커뮤니티",
                "state": state,                   # candidate / ready
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
            if int(e.get("upvotes", 0)) < min_up or int(e.get("comments", 0)) < min_cm:
                continue
            kw_hits = 0
            t = (title or "").lower()
            for kw in kw_list:
                if kw.lower() in t:
                    kw_hits += 1
            if require_kw and kw_hits <= 0:
                continue
            score = _value_score(kw_hits, int(e.get("upvotes", 0)), int(e.get("views", 0)), cfg)
            if score < thr:
                continue
            items.append({
                "id": _article_id(url),
                "title": title,
                "url": url,
                "source": "[COMM][Forum]",
                "upvotes": int(e.get("upvotes", 0)),
                "comments": int(e.get("comments", 0)),
                "views": int(e.get("views", 0)),
                "ts": None,
                "kw_hits": kw_hits,
                "total_score": round(float(score), 3),
                "section": "커뮤니티",
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

# ----------------------------- 커뮤니티 선택본 I/O -----------------------------
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
    return path

# ----------------------------- 커뮤니티 발행 -----------------------------
def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def export_json_comm(cfg: dict):
    data = read_selection(cfg)
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))
    arts = [a for a in data.get("articles", []) if (a.get("approved") if require_appr else a.get("selected"))]
    arts.sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)
    out = {"date": _today(), "articles": arts}
    path = _os_rel(os.path.join("archive", f"community_{_today()}.json"))
    _save_json(path, out)

def export_md_comm(cfg: dict):
    data = read_selection(cfg)
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))
    arts = [a for a in data.get("articles", []) if (a.get("approved") if require_appr else a.get("selected"))]
    arts.sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)

    lines = [f"# 커뮤니티 픽 ({_today()})", ""]
    for a in arts:
        meta = f"👍 {a.get('upvotes',0)} · 💬 {a.get('comments',0)} · 👀 {a.get('views','-')}"
        lines.append(f"- [{a.get('title')}]({a.get('url')})  \n  {meta} · {a.get('source')}")
    path = _os_rel(os.path.join("archive", f"community_{_today()}.md"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def export_html_comm(cfg: dict):
    data = read_selection(cfg)
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))
    arts = [a for a in data.get("articles", []) if (a.get("approved") if require_appr else a.get("selected"))]
    arts.sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)

    cards = []
    for a in arts:
        title = a.get("title") or "(제목 없음)"
        url = a.get("url") or "#"
        meta = f"👍 {a.get('upvotes',0)} · 💬 {a.get('comments',0)} · 👀 {a.get('views','-')} · {a.get('source','')}"
        cards.append(f"<div class='card'><h2><a href='{url}' target='_blank' rel='noopener'>{title}</a></h2>"
                     f"<div class='meta'>{meta}</div></div>")
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>커뮤니티 픽 {_today()}</title>
<style>
body{{font-family:system-ui,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}}
.meta{{color:#666;font-size:12px;margin-top:6px}}
</style><body>
<h1>커뮤니티 픽 {_today()}</h1>
{''.join(cards)}
</body></html>"""
    path = _os_rel(os.path.join("archive", f"community_{_today()}.html"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def publish_community(cfg: dict, fmt: str = "all"):
    fmt = (fmt or "all").lower()
    if fmt in ("all", "json"):
        export_json_comm(cfg)
    if fmt in ("all", "md"):
        export_md_comm(cfg)
    if fmt in ("all", "html"):
        export_html_comm(cfg)

# ----------------------------- 승인 UI -----------------------------
HTML_TMPL = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>커뮤니티 승인</title><style>
body{font-family:system-ui,Segoe UI,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px;}
.card{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0;}
.h{display:flex;align-items:center;gap:8px;}
.meta{color:#666;font-size:12px;margin-top:6px}
.btn{padding:8px 12px;border:1px solid #555;border-radius:6px;background:#fff;cursor:pointer;}
.btn:hover{background:#f5f5f5}
</style></head><body>
<h1>커뮤니티 승인(UI)</h1>
<div id="list"></div>
<div style="margin-top:12px">
<button class="btn" onclick="save()">저장</button>
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
      <div class="meta">score ${a.total_score||0} · ${a.source||''} · 👍${a.upvotes||0} · 💬${a.comments||0} · 👀${a.views||'-'}</div>
    `;
    root.appendChild(d);
  });
}
async function save(){
  const ids=[...document.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.getAttribute('data-id'));
  const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({approved_ids:ids})});
  if(r.ok){alert('저장 완료');}else{alert('저장 실패');}
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
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

# =====================================================================
#                      키워드 기반 선별/발행
# =====================================================================
DATA_DIR = _os_rel("data")
os.makedirs(DATA_DIR, exist_ok=True)
KW_FILE = _os_rel("data", "selected_keyword_articles.json")

EXTERNAL_RSS_TEMPLATES: List[str] = [
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
    "https://www.bing.com/news/search?q={query}&format=rss&setlang=en-US",
]

TRUST_DOMAINS: tuple = (
    "ipc.org","iec.ch","jedec.org","ieee.org","nist.gov","smta.org",
    "iconnect007.com","smt007.com","pcb007.com",
    "smttoday.com","smttoday.net","globalsmt.net","globalsmt.com","myglobalsmt.com","myglobalsmt.net",
    "technologyreview.com","nvidia.com","blogs.nvidia.com","developer.nvidia.com",
    "deepmind.com","openai.com","anthropic.com","meta.com","ai.googleblog.com",
    "arxiv.org","ieeexplore.ieee.org","semanticscholar.org",
    "eeworld.com.cn","eeworld.com"
)

def _parse_rss_feed(content: bytes) -> List[Dict[str, str]]:
    """RSS/Atom에서 (title, link, description, pubDate) 추출."""
    items: List[Dict[str, str]] = []
    try:
        soup = BeautifulSoup(content, "xml")
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

def _match_external_article(entry: Dict[str, str], keyword: str, patterns: List[re.Pattern], tokens: List[str]) -> bool:
    text = " ".join([entry.get("title") or "", entry.get("description") or "", entry.get("link") or ""])
    t = text.lower()
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

def fetch_external_articles(keyword: str, patterns: List[re.Pattern], tokens: List[str], max_total: int = 50) -> List[dict]:
    collected: List[dict] = []
    for tmpl in EXTERNAL_RSS_TEMPLATES:
        if len(collected) >= max_total:
            break
        try:
            url = tmpl.format(query=quote_plus(keyword))
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
            trust = 1.0 if domain in TRUST_DOMAINS else 0.0
            hits = sum(1 for tok in tokens if tok and ((tok.lower() in (ent.get("title") or "").lower()) or (tok.lower() in (ent.get("description") or "").lower())))
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

def build_keyword_selection(keyword: str, cfg: dict | None = None, use_external_rss: bool = False) -> str:
    """
    키워드 기반 선별 결과를 data/selected_keyword_articles.json 로 저장.
    """
    kw = (keyword or "").strip()
    if not kw:
        raise ValueError("키워드를 입력해 주세요.")
    cfg = cfg or load_config()

    out_path = KW_FILE

    tokens = _split_kw_tokens(kw)
    kw_loose = re.escape(kw).replace(r"\ ", r"[\s\-]?").replace(r"\-", r"[\s\-]?")
    rx_auto = _safe_regex_list([rf"\b{kw_loose}\b"])
    num_rx = re.compile(r"\b\d{3,6}\b")

    trust_set = set(TRUST_DOMAINS)
    trust_set.update([d.lower() for d in (cfg.get("features", {}).get("trusted_domains") or [])])

    pats: List[re.Pattern] = []
    # (필요 시 여기서 동의어/패턴을 더 추가)

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
        except Exception:
            ext = []
        candidates.extend(ext)

    # 중복 제거(by id/url)
    seen = set(); uniq = []
    for a in candidates:
        aid = a.get("id") or _article_id(a.get("url") or a.get("title") or "")
        if aid and aid not in seen:
            a["id"] = aid
            uniq.append(a); seen.add(aid)
    candidates = uniq

    selected: list[dict] = []
    for a in candidates:
        title = a.get("title") or ""
        summ  = a.get("summary_ko_text") or a.get("summary") or " ".join(a.get("summary_ko") or [])
        blob  = f"{title}\n{summ}\n{a.get('url') or ''}\n{a.get('source') or ''}".lower()

        # 1) 토큰 기반(최소 2개 일치)
        tok_hits = 0
        found = set()
        for t in tokens:
            if t and t not in found and re.search(rf"(?i)\b{re.escape(t)}\b", blob):
                found.add(t); tok_hits += 1

        # 2) 정규식 기반
        rx_hit = any(p.search(blob) for p in pats) or any(r.search(blob) for r in rx_auto)

        # 3) 숫자 힌트
        numeric_hint = bool(num_rx.search(blob))

        # 4) 신뢰 도메인
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
            "fallback": bool(ok_fallback)
        })

    # 정렬: fallback은 뒤로, 점수/키워드 일치 수 우선
    selected.sort(key=lambda x: (not x.get("fallback", False), float(x.get("score") or 0.0), int(x.get("kw_hits") or 0)), reverse=True)

    out = {"keyword": kw, "date": _today(), "articles": selected}
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, ensure_ascii=False, indent=2)
    return out_path

def publish_keyword_page(keyword: str) -> str:
    """선별 결과(JSON)를 읽어 HTML/MD/JSON으로 archive에 발행."""
    kw = (keyword or "").strip()
    if not kw:
        raise ValueError("키워드를 입력해 주세요.")

    from datetime import datetime as __dt
    safe_kw = re.sub(r'[\\/:*?"<>|]+', "-", kw).strip() or "keyword"

    arc_dir = Path(_os_rel("archive"))
    arc_dir.mkdir(parents=True, exist_ok=True)
    data = _load_json_safe(KW_FILE, {"keyword": kw, "date": _today(), "articles": []})
    arts = data.get("articles", [])

    approved_cnt = sum(1 for a in arts if a.get("approved"))
    if approved_cnt < 15:
        logger.warning("approved %d (<15). 발행은 계속 진행합니다.", approved_cnt)

    # 유형별 섹션
    sections = {"news": [], "paper": [], "standard": [], "community": []}
    for a in sorted(arts, key=lambda a: a.get("published_date") or ""):
        tp = (a.get("type") or "").strip().lower()
        if tp not in sections:
            tp = "news"
        sections[tp].append(a)

    today = _today()
    base = f"{today}_{safe_kw}"

    def _unique(p: Path) -> Path:
        if not p.exists():
            return p
        ts = __dt.now().strftime("%H%M%S")
        return p.with_name(f"{p.stem}_{ts}{p.suffix}")

    # JSON
    jpath = _unique(arc_dir / f"{base}.json")
    with jpath.open("w", encoding="utf-8") as jf:
        json.dump({"keyword": kw, "date": today, "sections": sections}, jf, ensure_ascii=False, indent=2)

    # MD
    sec_title = {"news": "뉴스", "paper": "논문/보고서", "standard": "정책/표준", "community": "커뮤니티"}
    md_lines = [f"# 키워드 뉴스 — {kw} ({today})", ""]
    for sec in ("news", "paper", "standard", "community"):
        items = sections.get(sec) or []
        if not items:
            continue
        md_lines.append(f"## {sec_title[sec]}")
        for a in items:
            title = a.get("title") or "(제목 없음)"
            url = a.get("url") or "#"
            note = (a.get("editor_note") or "").strip()
            ns = f" — 편집자메모: {note}" if note else ""
            md_lines.append(f"- [{title}]({url}){ns}")
        md_lines.append("")
    mpath = _unique(arc_dir / f"{base}.md")
    mpath.write_text("\n".join(md_lines), encoding="utf-8")

    # HTML
    cards = []
    for sec in ("news", "paper", "standard", "community"):
        items = sections.get(sec) or []
        if not items:
            continue
        cards.append(f"<h2>{sec_title[sec]}</h2>")
        for a in items:
            title = a.get("title") or "(제목 없음)"
            url = a.get("url") or "#"
            score = a.get("score") or a.get("total_score") or 0
            note = (a.get("editor_note") or "").strip()
            meta = f"score {score}"
            note_s = f"<div class='note'>편집자메모: {note}</div>" if note else ""
            cards.append(
                f"<div class='card'><h3><a href='{url}' target='_blank' rel='noopener'>{title}</a></h3>"
                f"<div class='meta'>{meta}</div>{note_s}</div>"
            )
    hpath = _unique(arc_dir / f"{base}.html")
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>{kw} — {today}</title>
<style>
body{{font-family:system-ui,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px;background:#fff;color:#111}}
h1{{font-size:22px;margin-bottom:6px}}
h2{{margin-top:22px;border-top:1px solid #eee;padding-top:12px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}}
.meta{{color:#666;font-size:12px;margin-top:6px}}
.note{{margin-top:6px;font-size:13px}}
</style><body>
<h1>키워드 뉴스 — {kw} ({today})</h1>
{''.join(cards)}
</body></html>"""
    hpath.write_text(html, encoding="utf-8")

    return str(hpath)

# =====================================================================
#                      통합 발행(공식 + 커뮤니티)
# =====================================================================
def _approved_items(items: List[Dict], require_appr: bool) -> List[Dict]:
    return [a for a in items if (a.get("approved") if require_appr else a.get("selected"))]

def export_all(cfg: dict, fmt: str = "all"):
    feats = (cfg.get("features") or {})
    require_appr = bool(feats.get("require_editor_approval", True))

    off = read_selection_official(cfg).get("articles", [])
    com = read_selection(cfg).get("articles", [])

    off_a = _approved_items(off, require_appr)
    com_a = _approved_items(com, require_appr)

    by_sec = {s: [] for s in SECTION_ORDER}
    for a in off_a + com_a:
        sec = a.get("section") or "일반 뉴스"
        by_sec.setdefault(sec, []).append(a)

    for s in by_sec:
        by_sec[s].sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)

    today = _today()

    if fmt in ("all", "json"):
        out = {"date": today, "sections": {s: by_sec.get(s, []) for s in SECTION_ORDER}}
        p = _os_rel(os.path.join("archive", f"daily_{today}.json"))
        _save_json(p, out)

    if fmt in ("all", "md"):
        lines = [f"# 데일리 뉴스 — {today}", ""]
        for s in SECTION_ORDER:
            if not by_sec.get(s):
                continue
            lines.append(f"## {s}")
            for a in by_sec[s]:
                note = (a.get("editor_note") or "").strip()
                note_s = (f" — 편집자메모: {note}" if note else "")
                lines.append(f"- [{a.get('title')}]({a.get('url')}){note_s}")
            lines.append("")
        p = _os_rel(os.path.join("archive", f"daily_{today}.md"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    if fmt in ("all", "html"):
        cards=[]
        for s in SECTION_ORDER:
            if not by_sec.get(s):
                continue
            cards.append(f"<h2>{s}</h2>")
            for a in by_sec[s]:
                note = (a.get("editor_note") or "").strip()
                meta = []
                if a.get("source"): meta.append(a["source"])
                if a.get("total_score"): meta.append(f"score {a['total_score']}")
                meta_s = " · ".join(meta)
                note_s = (f"<div class='note'>편집자메모: {note}</div>" if note else "")
                cards.append(
                    f"<div class='card'><h3><a href='{a.get('url','#')}' target='_blank' rel='noopener'>{a.get('title') or '(제목 없음)'}</a></h3>"
                    f"<div class='meta'>{meta_s}</div>{note_s}</div>"
                )
        html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>데일리 뉴스 — {today}</title>
<style>
body{{font-family:system-ui,Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:24px;background:#fff;color:#111}}
h1{{font-size:22px;margin-bottom:6px}}
h2{{margin-top:22px;border-top:1px solid #eee;padding-top:12px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}}
.meta{{color:#666;font-size:12px;margin-top:6px}}
.note{{margin-top:6px;font-size:13px}}
</style><body>
<h1>데일리 뉴스 — {today}</h1>
{''.join(cards)}
</body></html>"""
        p = _os_rel(os.path.join("archive", f"daily_{today}.html"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)

# ---------------------------------------------------------------------
# 설정 로드
# ---------------------------------------------------------------------
def load_config() -> dict:
    """
    config.json을 읽어 DEFAULT_CFG와 병합(없으면 기본값 사용).
    """
    cfg_path = _os_rel("config.json")
    user_cfg: dict = {}
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as rf:
                data = json.load(rf)
                if isinstance(data, dict):
                    user_cfg = data
    except Exception as e:
        logger.warning("config.json load warning: %s", e)
    return deep_merge(DEFAULT_CFG, user_cfg)

# ---------------------------------------------------------------------
# 키워드 자동 승인(상위 N)
# ---------------------------------------------------------------------
def _auto_approve_keyword(kw: str, top_n: int = 15, min_score: float = 0.0):
    if not kw:
        raise ValueError("키워드를 입력해 주세요.")
    pats = _safe_regex_list([re.escape(kw)])

    def _load(p, d):
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

# ---------------------------------------------------------------------
# 키워드 히스토리(날짜별 건수 집계)
# ---------------------------------------------------------------------
def _slug_kw_hist(s: str) -> str:
    if not s:
        return ""
    slug = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in s).strip("-")
    return slug.upper()

def collect_keyword_history(keyword: str, cfg: dict) -> dict:
    slug = _slug_kw_hist(keyword)
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    archive_dir = _os_rel(paths.get("archive") or "archive")
    reports_dir = _os_rel(paths.get("reports") or os.path.join("archive", "reports"))

    counts: dict[str, int] = {}
    try:
        for fname in os.listdir(archive_dir):
            if not fname.lower().endswith(".json"):
                continue
            if slug not in fname.upper():
                continue
            date_part = fname[:10]
            if not re.match(r"\d{4}-\d{2}-\d{2}", date_part):
                continue
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

    timeline = [{"date": d, "count": counts[d]} for d in sorted(counts.keys())]
    trend_ratio = 0.0
    desc = "데이터가 충분하지 않습니다."
    if timeline:
        last_date = timeline[-1]["date"]
        last_count = timeline[-1]["count"]
        prev_counts = [item["count"] for item in timeline[:-1]]
        avg_prev = sum(prev_counts) / len(prev_counts) if prev_counts else last_count
        denom = avg_prev if avg_prev != 0 else 1.0
        trend_ratio = (last_count - avg_prev) / denom
        trend_pct = round(trend_ratio * 100.0, 1)
        desc = f"{last_date} 기준 {last_count}건(평균 {avg_prev:.1f}건 대비 {trend_pct:+.1f}%)"

    result = {
        "keyword": keyword,
        "slug": slug,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "trend_ratio": trend_ratio,
        "description": desc,
        "timeline": timeline,
    }
    try:
        os.makedirs(reports_dir, exist_ok=True)
        outfile = os.path.join(reports_dir, f"{result['date']}_{slug}_history.json")
        with open(outfile, "w", encoding="utf-8") as wf:
            json.dump(result, wf, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    # 공식 수집/발행
    ap.add_argument("--collect", action="store_true", help="Collect official sources (smt/supply)")
    ap.add_argument("--publish", action="store_true", help="Publish merged (official + community)")
    # 커뮤니티
    ap.add_argument("--collect-community", action="store_true", help="Collect community candidates")
    ap.add_argument("--approve-ui", action="store_true", help="Open approval UI (http://127.0.0.1:8765)")
    ap.add_argument("--approve-top", type=int, default=0, help="Automatically approve top N (community)")
    ap.add_argument("--publish-community", action="store_true", help="Publish approved community items")
    ap.add_argument("--format", choices=["all", "html", "md", "json"], default="all", help="Publish format")
    # 키워드
    ap.add_argument("--collect-keyword", type=str, default="", help="Build keyword selection")
    ap.add_argument("--publish-keyword", type=str, default="", help="Publish keyword page")
    ap.add_argument("--approve-keyword-top", type=int, default=0, help="Auto-approve top N (keyword)")
    ap.add_argument("--approve-keyword", type=str, default="", help="Keyword to auto-approve")
    ap.add_argument("--use-external-rss", action="store_true", help="Use external RSS for keyword selection")
    # 키워드 히스토리
    ap.add_argument("--collect-keyword-history", type=str, default="", help="Collect keyword history (timeline)")
    # 트렌드 데이터 갱신(옵션)
    ap.add_argument("--update-trend-data", type=str, default="", help="Update trend json (external tool)")

    try:
        cmd_args = argv if argv is not None else sys.argv[1:]
        logger.info("orchestrator start args: %s", " ".join(str(a) for a in cmd_args))
    except Exception:
        pass

    args = ap.parse_args(argv)
    cfg = load_config()

    # 트렌드 데이터 외부 도구 사용(있을 때만)
    if args.update_trend_data:
        try:
            from tools.update_trend_data import update_trend_data  # type: ignore
            trend_file = args.update_trend_data
            editor_file = os.path.join(os.getcwd(), "editor_rules.json")
            synonym_file = os.path.join(os.getcwd(), "keyword_synonyms.json")
            update_trend_data(Path(trend_file), Path(editor_file), Path(synonym_file))
        except Exception:
            print("update_trend_data module not available")
        return

    if args.collect_keyword_history:
        collect_keyword_history(args.collect_keyword_history, cfg)
        return

    if args.collect_keyword:
        build_keyword_selection(args.collect_keyword, cfg, use_external_rss=args.use_external_rss)
        return

    if args.publish_keyword:
        publish_keyword_page(args.publish_keyword)
        return

    if args.approve_keyword_top > 0 and args.approve_keyword:
        _auto_approve_keyword(args.approve_keyword, args.approve_keyword_top, min_score=0.0)
        return

    # 공식
    if args.collect:
        arts = collect_official(cfg)
        write_selection_official(arts, cfg)
        return

    # 커뮤니티
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
        return

    if args.publish_community:
        publish_community(cfg, fmt=args.format)
        return

    # 통합 발행
    if args.publish:
        export_all(cfg, fmt=args.format)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
