from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
"""
퀄리뉴스 수집 + QG + FC + 선정/발행 파이프라인
- 기사(공식 소스)와 커뮤니티 분리 운영
- 커뮤니티: AI 토픽 제외, 업보트/댓글/최근성 기준
- 허브/목차(/news, /blog, IPC Newsletter index 등) 배제 + '첫 기사' 점프
- 발행 시 선택 기사에 번역(전문/요약) + 편집장 한마디 자동 보강
필수 패키지: requests, beautifulsoup4
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html as _html
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ====================== 안전 가드/기본값 ======================
logger = logging.getLogger("quali")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

USER_AGENT = os.getenv("QUALI_UA", "Mozilla/5.0 (QualiNewsBot)")
REQUEST_TO = int(os.getenv("QUALI_REQ_TO", "12"))

# Configuration defaults are defined in a dedicated module to ensure
# consistency across the various subsystems.  Import the unified
# configuration defaults and loader here rather than defining a separate
# DEFAULT_CFG in this file.  See ``config_schema.py`` for details.
from config_schema import DEFAULT_CFG as SCHEMA_DEFAULT_CFG, load_config as schema_load_config  # noqa: E402

# Adopt the shared default configuration as the canonical defaults for
# engine_core.  This dictionary includes sensible defaults for paths,
# features, community settings and external RSS ingestion.  Should
# environment variables be used to override values (e.g. translation
# model), do so when reading the configuration rather than hard‑coding
# them here.
DEFAULT_CFG = SCHEMA_DEFAULT_CFG

def load_config() -> dict:
    """
    Load the runtime configuration using the unified schema.  This
    function delegates to ``config_schema.load_config`` to parse
    ``config.json`` and apply default values.  The returned mapping
    is a plain dict suitable for consumption by the legacy code in
    engine_core.

    Environment variables such as ``OPENAI_TRANSLATE_MODEL`` may
    override certain settings after the schema has been applied.

    Returns:
        dict: merged configuration
    """
    cfg_model = schema_load_config()
    # Convert the Pydantic model to a plain dictionary.  If using
    # Pydantic v2 this would be ``model_dump()``, but v1 supports
    # ``dict()``.  Note: nested BaseModels are also converted.
    cfg = cfg_model.dict()
    # Override translate_model from environment if provided
    env_model = os.getenv("OPENAI_TRANSLATE_MODEL")
    if env_model:
        cfg.setdefault("features", {}).setdefault("translate_model", env_model)
    return cfg

# 런모드 (인자/플래그로도 제어)
RUN_MODE = os.getenv("JJIPPA_RUN_MODE", "collect")
COLLECT_PHASE = RUN_MODE == "collect"

# OpenAI
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_OPENAI_MODEL = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# 프록시
PROXY_PREFIX = "https://r.jina.ai/"
PROXY_FIRST_DOMAINS = {
    "openai.com",
    "x.ai",
    "ipc.org",
    "www.ipc.org",
    "ipcglobalinsight.org",
    "www.ipcglobalinsight.org",
}

# ======= 공통 유틸 =======
def setup_rotating_logger(log_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        fh = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=7, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        root = logging.getLogger()
        if not any(isinstance(h, TimedRotatingFileHandler) for h in root.handlers):
            root.addHandler(fh)
    except Exception:
        pass


def canonical_url(u: str) -> str:
    try:
        u = (u or "").strip()
        u = re.sub(r"#.*$", "", u)
        u = re.sub(r"/+$", "", u)
        return u
    except Exception:
        return u or ""


def _via_proxy(url: str) -> str:
    scheme = "https" if url.lower().startswith("https") else "http"
    prox = f"{PROXY_PREFIX}{scheme}://{url.split('://', 1)[1]}"
    rp = requests.get(prox, headers={"User-Agent": USER_AGENT}, timeout=(8, max(REQUEST_TO, 15)), allow_redirects=True)
    rp.raise_for_status()
    text = re.sub(r"\s+", " ", str(rp.text)).strip()[:8000]
    return f"<html><head><title>{url}</title></head><body>{text}</body></html>"


def _get(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if any(h == host or host.endswith(f".{h}") for h in PROXY_FIRST_DOMAINS):
            try:
                html = _via_proxy(url)
                if html and html.count("<a ") >= 1:
                    return html
            except Exception:
                pass
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        r = requests.get(url, headers=headers, timeout=(8, REQUEST_TO), allow_redirects=True)
        if r.status_code in (401, 403):
            headers["Referer"] = url
            r = requests.get(url, headers=headers, timeout=(8, REQUEST_TO), allow_redirects=True)
        if r.status_code == 404:
            alt = url.rstrip("/") if url.endswith("/") else url + "/"
            r = requests.get(alt, headers=headers, timeout=(8, REQUEST_TO), allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception:
        try:
            return _via_proxy(url)
        except Exception as e2:
            return f"__ERROR__:request:{e2}"


def _is_mostly_english(txt: str, thresh: float = 0.6) -> bool:
    if not txt:
        return False
    letters = sum(c.isascii() and (c.isalpha() or c.isdigit() or c.isspace()) for c in txt)
    return (letters / max(1, len(txt))) >= thresh


def _sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", t)
    return [p.strip(" \t\r\n\"'*") for p in parts if len(p.strip()) >= 3]


def _best_ko_summary(text: str, max_sent: int = 3, max_chars: int = 420) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    parts, buf = [], []
    enders = set(".!?。！？")
    for ch in t:
        buf.append(ch)
        if ch in enders:
            parts.append("".join(buf).strip())
            buf = []
    if buf:
        parts.append("".join(buf).strip())
    cleaned = []
    for s in parts:
        if not s:
            continue
        low = s.lower()
        if low.startswith(("copyright", "subscribe", "cookies")):
            continue
        if "http://" in low or "https://" in low:
            continue
        cleaned.append(s.strip())
    return " ".join(cleaned[:max_sent])[:max_chars].rstrip()


def _clean_for_translation(text: str) -> str:
    if not text:
        return ""
    t = text
    patterns = [
        r"(?im)^(accept|manage)\s+(cookies?|cookie preferences).*$",
        r"(?im)^subscribe(d|r)?\b.*$",
        r"(?im)^(log ?in|sign ?in|sign ?up|register)\b.*$",
        r"(?im)^(share this|follow us|related (stories|articles)).*$",
        r"(?im)©\s?\d{4}.*?(all rights reserved\.?)$",
        r"(?im)^\s*advertisement\s*$",
    ]
    for p in patterns:
        t = re.sub(p, " ", t)
    t = re.sub(
        r"(?im)^\s*by\s+[A-Z][\w.\- ]+\s*\|\s*[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}\s*$",
        " ",
        t,
    )
    return re.sub(r"\s+", " ", t).strip()


def _translate_en_to_ko(text: str) -> str | None:
    # 발행 모드에서만 동작
    collect_phase = bool(globals().get("COLLECT_PHASE", os.getenv("JJIPPA_RUN_MODE", "collect").lower() == "collect"))
    if collect_phase or not text or not isinstance(text, str):
        return None
    if os.getenv("QUALI_DISABLE_OPENAI", "0") == "1":
        return None
    if not _OPENAI_KEY:
        return None
    try:
        payload = {
            "model": _OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate English news/article text into natural Korean. Preserve lists and simple markdown. Do not add commentary.",
                },
                {"role": "user", "content": text[:12000]},
            ],
            "temperature": 0.2,
        }
        r = requests.post(
            _OPENAI_URL,
            headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=(8, 12),
        )
        if not r.ok:
            return None
        data = r.json()
        msg = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        return (msg or "").strip() or None
    except Exception:
        return None


# ====================== QG / FC ======================
TRUST_DOMAIN = {
    "iconnect007.com": 20,
    "ai.meta.com": 18,
    "blog.google": 18,
    "openai.com": 18,
    "x.ai": 16,
    "ipc.org": 19,
    "ipcglobalinsight.org": 16,
    "ecss.nl": 20,
    "teamsmt.com": 12,
}
BAN_PATH_PATTERNS = [
    r"/(tag|tags|category|categories)(/|$)",
    r"^/(newsletters|ipc-insights|overview|index)/?$",
    r"^/(blog|news)/?$",
    r"^/newsletters/global-insight/?$",  # IPC Newsletter Index 루트 컷
]
BAN_PHRASES = [
    "requested page not found",
    "page not found",
    "404",
    "not found",
    "cookie policy",
    "privacy policy",
    "terms of use",
    "login",
    "log in",
    "sign in",
    "register",
    "subscription",
    "subscribe",
]

STD_PATTERNS = [
    (r"\bJ-STD-\d{3,4}[A-Z]?\b", "IPC/J-STD"),
    (r"\bIPC[-\s]?[A-Z]?-?\d{3,4}[A-Z]?\b", "IPC"),
    (r"\bECSS[-\s]?(?:Q|E|M|U|S)[-\s]?ST[-\s]?\d+(?:-\d+)?[A-Z]?\b", "ECSS"),
    (r"\bISO\s?\d{3,5}[-:]?[A-Z0-9:]*\b", "ISO"),
    (r"\bIEC\s?\d{3,5}[-:]?[A-Z0-9:]*\b", "IEC"),
    (r"\bJESD[-\s]?\d+[A-Z0-9\-]*\b", "JEDEC"),
    (r"\bASTM\s?[A-Z]\d{1,4}[A-Z]?\b", "ASTM"),
    (r"\bSAE\s?[A-Z\-]?\d{1,6}[A-Z]?\b", "SAE"),
    (r"\bMIL[-\s]?STD[-\s]?\d+[A-Z\-]*\b", "MIL-STD"),
    (r"\bRFC\s?\d{3,5}\b", "IETF RFC"),
    (r"\bEN\s?\d{3,5}[-:]?\d*\b", "EN"),
]
TRUSTED_DOMAINS = {
    "ipc.org",
    "ecss.nl",
    "iec.ch",
    "iso.org",
    "iconnect007.com",
    "ai.meta.com",
    "blog.google",
    "openai.com",
    "x.ai",
    "teamsmt.com",
    "nist.gov",
    "eur-lex.europa.eu",
    "europa.eu",
    "jedec.org",
    "sae.org",
    "astm.org",
    "ietf.org",
    "rfc-editor.org",
}


def _extract_main_text_from_html(raw_html: str) -> tuple[str, int, int, int]:
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "svg", "figure"]):
            t.decompose()
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", id=re.compile("content|article|post", re.I))
            or soup
        )
        paras = [p.get_text(" ", strip=True) for p in main.find_all(["p", "li"])]
        paras = [x for x in paras if len(x) >= 40]
        text = " ".join(paras) if paras else soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        sent_n = len(_sentences(text))
        link_n = len(main.find_all("a", href=True))
        return text, len(text), sent_n, link_n
    except Exception:
        t = re.sub(r"\s+", " ", raw_html or "").strip()
        return t, len(t), len(_sentences(t)), 0


def _looks_like_url(s: str) -> bool:
    return bool(re.search(r"https?://", s)) or s.strip().startswith(("www.", "http", "https"))


def _title_is_generic_qg(title: str, url: str) -> bool:
    GENERIC = {
        "home",
        "index",
        "overview",
        "blog",
        "insights",
        "insight",
        "newsletter",
        "newsroom",
        "news",
        "press",
        "updates",
        "archive",
        "latest",
        "articles",
        "posts",
        "resources",
        "documentation",
        "profile",
        "about",
        "subscribe",
    }
    t = (title or "").strip()
    if not t:
        return True
    if _looks_like_url(t):
        return True
    low = t.lower()
    if any(w in low for w in GENERIC):
        return True
    try:
        p = urlparse(url)
        path = (p.path or "/").strip("/").lower()
        if path == "" or path in {"blog", "insights", "news", "newsletter", "overview", "index"}:
            return True
        if len(path) < 6 and "/" not in path:
            return True
    except Exception:
        pass
    return False


def quality_gate(url: str, html: str, title: str, source_label: str, cfg: dict | None) -> dict:
    feats_q = (((cfg or {}).get("features", {}) or {}).get("quality") or {})
    MIN_CHARS = int(feats_q.get("min_chars", 600))
    MIN_SENTS = int(feats_q.get("min_sentences", 5))
    MIN_LINKS = int(feats_q.get("min_links", 2))
    STRICT = bool(feats_q.get("strict", True))
    pu = urlparse(url)
    path = (pu.path or "").lower()
    if any(re.search(p, path) for p in BAN_PATH_PATTERNS):
        return {"status": "REJECT", "score": 0, "reason": "목록/허브 경로"}
    text, nchar, nsent, nlink = _extract_main_text_from_html(html or "")
    low = (text or "").lower()
    if any(b in low for b in BAN_PHRASES):
        return {"status": "REJECT", "score": 0, "reason": "로그인/구독/정책 안내"}
    if _title_is_generic_qg(title, url):
        return {"status": "REJECT" if STRICT else "HOLD", "score": 0, "reason": "일반/목록형 제목"}
    trust = TRUST_DOMAIN.get(pu.netloc.lower().lstrip("www."), 10)
    has_num = 1 if re.search(r"\b\d+([.,]\d+)?\b", text) else 0
    has_date = 1 if re.search(r"\b(19|20)\d{2}\b", text) else 0
    score = min(nchar, 3000) / 3000 * 40 + min(nsent, 30) / 30 * 20 + min(nlink, 10) / 10 * 10 + trust + has_num * 5 + has_date * 5
    score = int(round(score))
    if (nchar >= MIN_CHARS) and (nsent >= MIN_SENTS) and (nlink >= MIN_LINKS):
        return {"status": "PASS", "score": score, "reason": ""}
    if (nchar >= int(MIN_CHARS * 0.7)) and (nsent >= max(3, MIN_SENTS - 1)):
        return {"status": "HOLD", "score": score, "reason": "아슬아슬 통과"}
    return {"status": "REJECT", "score": score, "reason": "불충분"}


def _extract_standards(text: str) -> List[Tuple[str, str]]:
    found = []
    for pat, org in STD_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            token = m.group(0).strip()
            if token not in [x[0] for x in found]:
                found.append((token, org))
    return found


def _collect_external_links(base_url: str, raw_html: str) -> Tuple[int, int, List[str]]:
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        base = urlparse(base_url).netloc.lower().lstrip("www.")
        ext, pdf, hits = 0, 0, []
        for a in soup.find_all("a", href=True):
            href = (a["href"] or "").strip()
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            if href.startswith("//"):
                href = "https:" + href
            if href.startswith("/"):
                href = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}{href}"
            pu = urlparse(href)
            dom = pu.netloc.lower().lstrip("www.")
            if not dom:
                continue
            if dom != base and dom in TRUSTED_DOMAINS:
                ext += 1
                hits.append(href)
                if href.lower().endswith(".pdf"):
                    pdf += 1
        return ext, pdf, hits
    except Exception:
        return 0, 0, []


def fact_check(url: str, raw_html: str, title: str, ko_summaries: List[str], cfg: dict | None) -> dict:
    feats = (((cfg or {}).get("features", {}) or {}).get("factcheck") or {})
    if not bool(feats.get("enabled", True)):
        return {"status": "PASS", "score": 100, "evidence": [], "standards": [], "reason": "disabled"}
    text, nchar, nsent, nlink = _extract_main_text_from_html(raw_html or "")
    text_low = (text or "").lower()
    nums = len(re.findall(r"\b\d+([.,]\d+)?\b", text_low))
    dates = len(re.findall(r"\b(19|20)\d{2}\b", text_low))
    stds = _extract_standards(text)
    ext_links, pdf_links, _hits = _collect_external_links(url, raw_html)
    score = int(min(100, min(nums, 10) * 2 + min(dates, 6) * 2 + min(len(stds), 3) * 10 + min(ext_links, 3) * 10 + min(pdf_links, 2) * 15))
    ev = []
    if stds:
        ev.append(f"표준 {len(stds)}개")
    if nums:
        ev.append(f"숫자 {min(nums, 99)}개")
    if dates:
        ev.append(f"연도 {min(dates, 99)}개")
    if ext_links:
        ev.append(f"공식외부링크 {ext_links}개")
    if pdf_links:
        ev.append(f"PDF {pdf_links}개")
    min_ev = int(feats.get("min_evidence", 2))
    min_sc = int(feats.get("min_score", 52))
    strict = bool(feats.get("strict", True))
    if (len(ev) >= min_ev) and (score >= min_sc):
        return {"status": "PASS", "score": score, "evidence": ev, "standards": stds, "reason": ""}
    return {
        "status": ("HOLD" if not strict else "FAIL"),
        "score": score,
        "evidence": ev,
        "standards": stds,
        "reason": ("근거부족" if len(ev) < min_ev else f"점수 {score}<{min_sc}"),
    }


# ============ 가중치/키워드 로더 ============
def _load_editor_rules(path: str | None = None) -> dict:
    paths = [
        path,
        os.path.join(os.getcwd(), "feeds", "editor_rules.json"),
        os.path.join(os.getcwd(), "editor_rules.json"),
    ]
    for p in [p for p in paths if p]:
        try:
            return json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
    return {}


_STD_PATTERNS_DEFAULT = {
    "IPC-A-610": r"\bIPC[-\s]?A[-\s]?610([A-Z])?\b",
    "J-STD-001": r"\bJ[-\s]?STD[-\s]?0?01([A-Z])?\b",
    "IPC-A-620": r"\bIPC[-\s]?A[-\s]?620([A-Z])?\b",
    "ECSS-Q-ST-70-61C": r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?61[A-Z]?\b",
    "NASA-STD-8739": r"\bNASA[-\s]?STD[-\s]?8739(?:\.\d+)?[A-Z]?\b",
}


def _domain_score(url: str, rules: dict) -> int:
    try:
        dom = urlparse(url).netloc.lower().lstrip("www.")
        return int(((rules.get("weights", {}) or {}).get("domain", {}) or {}).get(dom, 0))
    except Exception:
        return 0


def _keyword_score(title: str, text: str, rules: dict) -> tuple[int, list[str]]:
    weights = ((rules.get("weights", {}) or {}).get("keywords", {}) or {})
    rx_map = ((rules.get("weights", {}) or {}).get("keywords_regex", {}) or {})
    syns = ((rules.get("weights", {}) or {}).get("synonyms", {}) or {})

    blob = f"{title or ''}\n{text or ''}"
    score, hits = 0, []

    rx_all = dict(_STD_PATTERNS_DEFAULT)
    rx_all.update(rx_map)
    for key, rx in rx_all.items():
        try:
            if re.search(rx, blob, flags=re.I):
                w = int(weights.get(key, 0))
                if w:
                    score += w
                    hits.append(key)
        except re.error:
            continue

    for key, w in weights.items():
        pat = re.escape(key).replace(r"\ ", r"[\s\-]?")
        try:
            if re.search(rf"(?i)\b{pat}\b", blob):
                score += int(w)
                hits.append(key)
        except re.error:
            if key.lower() in blob.lower():
                score += int(w)
                hits.append(key)

    for base, alts in syns.items():
        for a in (alts or []):
            pat = re.escape(a).replace(r"\ ", r"[\s\-]?")
            if re.search(rf"(?i)\b{pat}\b", blob):
                w = int(weights.get(base, 0))
                if w:
                    score += w // 2
                    hits.append(a)

    hits = sorted(set(hits))
    return score, hits


def _pick_top_articles_by_keywords(base_url: str, listing_html: str, rules: dict, top_k: int = 1) -> list[str]:
    try:
        soup = BeautifulSoup(listing_html or "", "html.parser")
        base = urlparse(base_url)
        base_net = base.netloc.lower().lstrip("www.")
        cands = []
        for a in soup.find_all("a", href=True):
            t = (a.get_text(" ", strip=True) or "").strip()
            href = (a["href"] or "").strip()
            if not t or not href:
                continue
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            href = urljoin(base_url, href)
            pu = urlparse(href)
            this_net = pu.netloc.lower().lstrip("www.")
            if this_net and this_net != base_net:  # 외부 링크 제외
                continue
            path = (pu.path or "").lower()
            if any(seg in path for seg in ("/login", "/subscribe", "/advertis", "/account", "/img/", "/tag/", "/topic/", "/category/")):
                continue
            if any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip", ".mp4", ".mov", ".webp")):
                continue
            kw, _ = _keyword_score(t, "", rules)
            if any(k in path for k in ("/article", "/news", "/insight", "/standard", "/view", "/post", "/press")):
                kw += 2
            if kw > 0:
                cands.append((kw, canonical_url(href)))
        cands.sort(key=lambda x: x[0], reverse=True)
        seen, out = set(), []
        for _, u in cands:
            if u not in seen:
                seen.add(u)
                out.append(u)
            if len(out) >= max(1, int(top_k)):
                break
        return out
    except Exception:
        return []


# ============ 타이틀/섹션/메타 ============
def _kr_one_liner(title: str, label: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip()) or "제목 없음"
    if len(t) > 80:
        t = t[:77] + "…"
    lab = (label or "").lower()
    if "openai" in lab:
        return f"OpenAI: '{t}' 발표"
    if "x.ai" in lab:
        return f"xAI: '{t}' 소식"
    if "meta" in lab:
        return f"메타 AI: '{t}' 업데이트"
    if "gemini" in lab or "google" in lab:
        return f"Google: '{t}' 소식"
    if "smt" in lab and "007" in lab:
        return f"iConnect007(SMT): '{t}' 기사"
    if "pcb" in lab and "007" in lab:
        return f"iConnect007(PCB): '{t}' 기사"
    if "teamsmt" in lab:
        return f"TeamSMT: '{t}' 블로그"
    if "ipc" in lab:
        return f"IPC: '{t}' 안내"
    if "ecss" in lab:
        return f"ECSS: '{t}' 표준/지침"
    return f"'{t}' 핵심 한 줄"


def _tags_for(source: str) -> str:
    s = (source or "").lower()
    if "ipc" in s:
        return "#퀄리뉴스 #표준 #IPC"
    if "ecss" in s:
        return "#퀄리뉴스 #표준 #ECSS #ESA"
    if "smt" in s:
        return "#퀄리뉴스 #SMT #글로벌전자생산"
    if "pcb" in s:
        return "#퀄리뉴스 #PCB #글로벌전자생산"
    if "teamsmt" in s:
        return "#퀄리뉴스 #EMS #공급망"
    if any(k in s for k in ("openai", "gemini", "meta", "xai")):
        return "#퀄리뉴스 #AI뉴스"
    return "#퀄리뉴스"


def _section_for(source: str) -> str:
    src = (source or "").lower()
    if any(t in src for t in ("ipc", "ecss", "[gov]", "mnd", "dapa", "nato", "mil-std", "mil ")):
        return "표준 뉴스"
    if "[comm]" in src:
        return "글로벌 전자생산"
    if any(t in src for t in ("smt", "pcb", "teamsmt", "assembly", "manufactur", "iconnect007", "eetimes", "ecnmag")):
        return "글로벌 전자생산"
    if any(t in src for t in ("openai", "gemini", "meta", "xai", " ai ", " ai.", " ml ")):
        return "AI 뉴스"
    return "퀄리뉴스"


# ============ 카드/선택파일 ============
def _article_id(title: str, url: str) -> str:
    return hashlib.md5(canonical_url(url).encode("utf-8")).hexdigest()


def _hash_url(url: str) -> str:
    return hashlib.md5(canonical_url(url).encode("utf-8")).hexdigest()


def _hash_title(title: str) -> str:
    return hashlib.md5((title or "").encode("utf-8")).hexdigest()


def make_card(article: Dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    tags = _tags_for(article["source"])
    section = _section_for(article["source"])
    qg_meta = f' QG:{article.get("qg_status","HOLD")} QS:{int(article.get("qg_score",0))} QR:{(article.get("qg_reason","") or "")[:120]}'
    stds = ";".join([s for s, _ in (article.get("fc_standards") or [])])
    evn = str(len(article.get("fc_evidence", []) or []))
    fc_meta = f' FC:{article.get("fc_status","HOLD")} FS:{int(article.get("fc_score",0))} FE:{evn} ST:{stds}'
    kwds_meta = f' KW:{int(article.get("kw_score",0))} DS:{int(article.get("domain_score",0))} TS:{int(article.get("total_score", article.get("qg_score",0)+article.get("fc_score",0)))}'
    meta_line = f"<!--ID:{article['id']} URL_HASH:{article['url_hash']} TITLE_HASH:{article['title_hash']}{qg_meta}{fc_meta}{kwds_meta}-->"
    se = (article.get("summary_en") or [])[:3]
    sk = (article.get("summary_ko") or [])[:3]
    while len(se) < 3:
        se.append("Pending editor review")
    while len(sk) < 3:
        sk.append("편집 검토/추가 예정")
    return f"""
{meta_line}
[퀄리뉴스] {section} — {today}
📰 제목: "{article['title']}"
출처: {article['source']} ({article['url']})

요약 3줄 (EN/KR 교차):
- EN: {se[0]}
- KO: {sk[0]}
- EN: {se[1]}
- KO: {sk[1]}
- EN: {se[2]}
- KO: {sk[2]}

🖊 편집장 한마디:
- {article.get('editor_note','') or ''}

#태그: {tags}
""".lstrip()


def _selection_path(cfg: dict) -> str:
    sel = (((cfg or {}).get("features", {}) or {}).get("selection_file")) or "selected_articles.json"
    return os.path.join(os.getcwd(), sel)


def _selection_path_comm(cfg: dict) -> str:
    sel = (((cfg or {}).get("community", {}) or {}).get("selection_file")) or "selected_community.json"
    return os.path.join(os.getcwd(), sel)


def write_selection_file(articles: List[Dict], cfg: dict) -> str:
    path = _selection_path(cfg)
    today = datetime.now().strftime("%Y-%m-%d")
    prev = {}
    if os.path.exists(path):
        try:
            prev = json.load(open(path, "r", encoding="utf-8"))
            if prev.get("date") != today:
                prev = {}
        except Exception:
            prev = {}
    selected_prev = {a.get("id"): a for a in (prev.get("articles") or [])}
    out = {"date": today, "articles": []}
    for art in articles:
        old = selected_prev.get(art["id"], {})
        ko_title = _kr_one_liner(art["title"], art["source"])
        out["articles"].append(
            {
                "id": art["id"],
                "title": art["title"],
                "ko_title": ko_title,
                "source": art["source"],
                "url": art["url"],
                "full_ko": art.get("full_ko"),
                "summary_ko_text": art.get("summary_ko_text"),
                "section": _section_for(art["source"]),
                "qg_status": art["qg_status"],
                "qg_score": art["qg_score"],
                "fc_status": art["fc_status"],
                "fc_score": art["fc_score"],
                "kw_score": int(art.get("kw_score", 0)),
                "kw_hits": art.get("kw_hits", []),
                "domain_score": int(art.get("domain_score", 0)),
                "total_score": int(art.get("total_score", art.get("qg_score", 0) + art.get("fc_score", 0))),
                "selected": bool(old.get("selected", False)),
                "editor_note": old.get("editor_note", ""),
                "pinned": bool(old.get("pinned", False)),
                "pin_ts": int(old.get("pin_ts") or 0),
            }
        )
    with open(path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, ensure_ascii=False, indent=2)
    print(f"선택 파일 저장: {path} (총 {len(out['articles'])}건)")
    return path


def write_selection_file_community(articles: List[Dict], cfg: dict) -> str:
    path = _selection_path_comm(cfg)
    today = datetime.now().strftime("%Y-%m-%d")
    prev = {}
    if os.path.exists(path):
        try:
            prev = json.load(open(path, "r", encoding="utf-8"))
            if prev.get("date") != today:
                prev = {}
        except Exception:
            prev = {}
    selected_prev = {a.get("id"): a for a in (prev.get("articles") or [])}
    out = {"date": today, "articles": []}
    for art in articles:
        old = selected_prev.get(art["id"], {})
        out["articles"].append(
            {
                "id": art["id"],
                "title": art["title"],
                "ko_title": _kr_one_liner(art["title"], art["source"]),
                "source": art["source"],
                "url": art["url"],
                "qg_status": art["qg_status"],
                "qg_score": art["qg_score"],
                "fc_status": art["fc_status"],
                "fc_score": art["fc_score"],
                "summary_en": art.get("summary_en", []),
                "summary_ko": art.get("summary_ko", []),
                "kw_score": int(art.get("kw_score", 0)),
                "kw_hits": art.get("kw_hits", []),
                "domain_score": int(art.get("domain_score", 0)),
                "total_score": int(art.get("total_score", 0)),
                "section": "커뮤니티",
                "selected": bool(old.get("selected", False)),
                "editor_note": old.get("editor_note", ""),
                "pinned": bool(old.get("pinned", False)),
                "pin_ts": int(old.get("pin_ts") or 0),
                "full_ko": old.get("full_ko", ""),
                "summary_ko_text": old.get("summary_ko_text", ""),
            }
        )
    with open(path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, ensure_ascii=False, indent=2)
    print(f"커뮤니티 선택 파일 저장: {path} (총 {len(out['articles'])}건)")
    return path


def load_selection(cfg: dict) -> Tuple[set, dict]:
    path = _selection_path(cfg)
    if not os.path.exists(path):
        return set(), {}
    data = json.load(open(path, "r", encoding="utf-8"))
    ids = {a["id"] for a in data.get("articles", []) if a.get("selected")}
    notes = {a["id"]: a.get("editor_note", "") for a in data.get("articles", [])}
    return ids, notes


def _load_selmap_extra(cfg: dict) -> dict:
    try:
        sel_path = _selection_path(cfg or {})
        if not os.path.exists(sel_path):
            return {}
        data = json.load(open(sel_path, "r", encoding="utf-8"))
        mp = {}
        for a in data.get("articles", []):
            mp[a.get("id")] = {
                "full_ko": a.get("full_ko"),
                "summary_ko_text": a.get("summary_ko_text"),
                "pinned": bool(a.get("pinned")),
                "pin_ts": int(a.get("pin_ts") or 0),
            }
        return mp
    except Exception:
        return {}


# ====================== 수집 ======================
def _fetch_title_desc(url: str) -> Tuple[str, str]:
    html = _get(url)
    if isinstance(html, str) and html.startswith("__ERROR__"):
        return f"[오류 페이지] {url}", html
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title").get_text(strip=True) if soup.find("title") else url
    desc_tag = (
        soup.find("meta", attrs={"name": "description"})
        or soup.find("meta", attrs={"property": "og:description"})
        or soup.find("meta", attrs={"name": "twitter:description"})
    )
    desc = (desc_tag.get("content", "").strip() if desc_tag else "")
    if not desc:
        p = soup.find("p")
        desc = p.get_text(" ", strip=True) if p else ""
    desc = re.sub(r"\s+", " ", desc).strip()[:600] if desc else "내용 요약을 찾지 못함"
    return title, desc


def _pick_first_article_url(base_url: str, listing_html: str, cfg: dict | None = None) -> str:
    try:
        base = urlparse(base_url)
        host = base.netloc.lower().lstrip("www.")
        soup = BeautifulSoup(listing_html or "", "html.parser")
        rules = (((cfg or {}).get("features", {}) or {}).get("landing_rules", {}) or {})
        rule = None
        for k in rules.keys():
            if k in host:
                rule = rules[k]
                break

        anchors = soup.find_all("a", href=True)
        links = []
        for a in anchors:
            href = (a.get("href") or "").strip()
            if not href or href.startswith(("#", "mailto:", "javascript:")):
                continue
            if href.startswith("//"):
                href = f"{base.scheme}:{href}"
            elif href.startswith("/"):
                href = f"{base.scheme}://{base.netloc}{href}"
            links.append(canonical_url(href))

        def _match_any(u: str, pats: list[str] | None) -> bool:
            return any(re.search(p, u, re.I) for p in (pats or []))

        if rule and rule.get("deny_patterns"):
            links = [u for u in links if not _match_any(u, rule["deny_patterns"])]
        if rule and rule.get("prefer_patterns"):
            for u in links:
                if _match_any(u, rule["prefer_patterns"]):
                    return u

        def _pick_by_selector(sel_list):
            for sel in (sel_list or []):
                a = soup.select_one(sel)
                if a and a.get("href"):
                    href = (a["href"] or "").strip()
                    if href.startswith("//"):
                        href = f"{base.scheme}:{href}"
                    elif href.startswith("/"):
                        href = f"{base.scheme}://{base.netloc}{href}"
                    return canonical_url(href)
            return None

        if host == "ipcglobalinsight.org":
            cand = _pick_by_selector(['a[href^="/newsletters/"][href*="/20"]', 'a[href^="/20"]', 'a[href^="/news/20"]'])
            if cand:
                return cand
        if host == "iconnect007.com":
            cand = _pick_by_selector(['a[href^="/article/"]', 'a[href*="/article/"][href$="/"]'])
            if cand:
                return cand
        if host == "ipc.org":
            cand = _pick_by_selector(
                ['a[href*="/news/"]:not([href$="/news"])', 'a[href*="/ipc-insights/"]:not([href$="/ipc-insights"])', 'a[href*="/press-"]']
            )
            if cand:
                return cand
        if host == "openai.com":
            cand = _pick_by_selector(['a[href^="/blog/"]:not([href="/blog/"])'])
            if cand:
                return cand
        if host == "x.ai":
            cand = _pick_by_selector(['a[href^="/blog/"]:not([href="/blog"])', 'a[href^="/news/"]:not([href="/news"])'])
            if cand:
                return cand

        candidates = []
        for u in links:
            pu = urlparse(u)
            if pu.netloc.lower().lstrip("www.") != host:
                continue
            path = (pu.path or "").lower()
            if any(seg in path for seg in ("/landing/", "/login", "/privacy", "/terms", "/account", "/subscribe", "/advertis", "/ads/", "/img/")):
                continue
            if any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip", ".mp4", ".mov", ".webp")):
                continue
            score = 0
            if any(k in path for k in ("/article", "/articles", "/news", "/blog", "/post", "/press", "/insight", "/standard", "/view")):
                score += 10
            if "-" in path:
                score += 1
            if path.count("/") >= 3:
                score += 1
            candidates.append((score, canonical_url(u)))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return canonical_url(base_url)
    except Exception:
        return canonical_url(base_url)


# ====================== 커뮤니티 수집(분리) ======================
COMM_AI_BLOCK = [
    r"(?i)\bAI\b",
    r"(?i)artificial\s+intelligence",
    r"(?i)deep\s+learning",
    r"(?i)machine\s+learning",
    r"(?i)\bLLM\b",
    r"(?i)\bGPT\b",
    r"(?i)ChatGPT",
    r"(?i)prompt(ing)?",
]
# ====== 커뮤니티 화이트리스트 표준/키워드(대표님 지정) ======
COMM_WHITELIST_RX = [
    # IPC / J-STD
    r"\bIPC\b",
    r"\bIPC[-\s]?A[-\s]?610\b",
    r"\bJ[-\s]?STD[-\s]?001\b",
    r"\bIPC[-\s]?WHMA[-\s]?A[-\s]?620\b",
    r"\bIPC[-\s]?7711(?:\/21)?\b",
    r"\bIPC[-\s]?6012\b",
    r"\bIPC[-\s]?2221\b",
    r"\bIPC[-\s]?7351\b",
    r"\bJ[-\s]?STD[-\s]?002\b",
    r"\bJ[-\s]?STD[-\s]?003\b",
    r"\bIPC[-\s]?9252\b",
    # ECSS / ESA
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?61\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?08\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?38\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?26\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?28\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?30\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?10\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?11\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70[-\s]?-?12\b",
    r"\bECSS[-\s]?Q[-\s]?-?ST[-\s]?-?70\b",
    # NASA
    r"\bNASA[-\s]?STD[-\s]?8739(?:\.\d+)?\b",
    r"\bNASA[-\s]?STD[-\s]?4009\b",
    r"\bNASA[-\s]?STD[-\s]?4003\b",
    r"\bNASA[-\s]?STD[-\s]?6016\b",
    # MIL
    r"\bMIL[-\s]?PRF[-\s]?31032\b",
    r"\bMIL[-\s]?STD[-\s]?883\b",
    r"\bMIL[-\s]?STD[-\s]?202\b",
    r"\bMIL[-\s]?STD[-\s]?1686\b",
    r"\bMIL[-\s]?STD[-\s]?461\b",
    r"\bMIL[-\s]?STD[-\s]?750\b",
    r"\bMIL[-\s]?DTL[-\s]?38999\b",
    r"\bMIL[-\s]?DTL[-\s]?83513\b",
    r"\bMIL[-\s]?PRF[-\s]?50884\b",
]


def _whitelist_hits(title: str) -> list[str]:
    t = title or ""
    hits = []
    for rx in COMM_WHITELIST_RX:
        try:
            if re.search(rx, t, re.I):
                hits.append(rx)
        except re.error:
            continue
    return hits


def _reddit_old_new(sub: str, pages: int = 1) -> list[dict]:
    """old.reddit.com에서 /new/ 글 목록 파싱(업보트/댓글/시간)."""
    out = []
    base = f"https://old.reddit.com/r/{sub}/new/"
    headers = {"User-Agent": USER_AGENT}
    url = base
    for _ in range(max(1, pages)):
        r = requests.get(url, headers=headers, timeout=(8, REQUEST_TO))
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all("div", class_=re.compile(r"\bthing\b")):
            a = t.find("a", class_=re.compile(r"\btitle\b"))
            if not a or not a.get("href"):
                continue
            title = a.get_text(" ", strip=True)
            href = urljoin("https://old.reddit.com", a["href"])
            sc = t.find("div", class_=re.compile(r"\bscore\b"))
            up = 0
            if sc:
                if sc.has_attr("title"):
                    up = int(re.sub(r"\D", "", sc["title"] or "0") or 0)
                else:
                    up = int(re.sub(r"\D", "", sc.get_text(" ", strip=True) or "0") or 0)
            cc = t.find("a", class_=re.compile(r"\bcomments\b"))
            cm = int(re.sub(r"\D", "", (cc.get_text(" ", strip=True) if cc else "") or "0") or 0)
            tm = t.find("time")
            ts = tm.get("datetime") if tm and tm.has_attr("datetime") else ""
            out.append({"sub": sub, "title": title, "url": canonical_url(href), "upvotes": up, "comments": cm, "ts": ts})
        nxt = soup.find("span", class_="next-button")
        url = nxt.find("a")["href"] if nxt and nxt.find("a") else None
        if not url:
            break
    return out


def fetch_community(cfg: dict | None = None) -> List[Dict]:
    """커뮤니티 전용: 키워드 동일, AI 토픽 제외, 업보트/댓글/최근성 기준."""
    feeds_dir = os.path.join(os.getcwd(), "feeds")
    comm_cfg_path = os.path.join(feeds_dir, "community_sources.json")
    comm_cfg = json.load(open(comm_cfg_path, "r", encoding="utf-8")) if os.path.exists(comm_cfg_path) else {}
    RULES = _load_editor_rules()

    subs = (comm_cfg.get("reddit", {}) or {}).get("subs", []) or []
    limit_per = int((comm_cfg.get("reddit", {}) or {}).get("limit_per_query", 3))
    fresh_hours = int(((cfg or {}).get("community", {}) or {}).get("fresh_hours", 168))
    min_up = int(((cfg or {}).get("community", {}) or {}).get("min_upvotes", 10))
    min_cm = int(((cfg or {}).get("community", {}) or {}).get("min_comments", 3))
    excl_ai = bool(((cfg or {}).get("community", {}) or {}).get("exclude_ai", True))

    def _drop_ai(title: str) -> bool:
        if not excl_ai:
            return False
        return any(re.search(p, title or "", re.I) for p in COMM_AI_BLOCK)

    items = []
    for sub in subs:
        try:
            rows = _reddit_old_new(sub, pages=1)
        except Exception:
            rows = []
        tmp = []
        for r in rows:
            t = r["title"]
            if _drop_ai(t):
                continue

            # 1) 화이트리스트 키워드 필수
            wl_hits = _whitelist_hits(t)
            if not wl_hits:
                continue  # 지정 키워드 외 전부 제외

            # 2) 50/30/20 가중치 점수 계산
            # - 키워드(50): 매칭 개수에 따라 1개=70, 2개=90, 3개 이상=100
            kw_raw = len(wl_hits)
            kw_norm = 70 if kw_raw == 1 else (90 if kw_raw == 2 else 100)

            # - 추천수(30): 업보트 0~100 정규화
            up = max(0, int(r.get("upvotes") or 0))
            up_norm = min(100, up)

            # - 조회수(20): 댓글 수를 근사치로 사용
            cm = max(0, int(r.get("comments") or 0))
            view_norm = min(100, cm)

            # 합성 점수
            final_score = int(round(0.5 * kw_norm + 0.3 * up_norm + 0.2 * view_norm))

            # 3) 최신성/최소 기준
            ok_recent = True
            if r.get("ts"):
                try:
                    dt = datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).astimezone(timezone.utc)
                    hrs = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
                    ok_recent = hrs <= fresh_hours
                except Exception:
                    ok_recent = True

            if ok_recent and up >= min_up and cm >= min_cm:
                tmp.append(
                    {
                        "id": _article_id(t, r["url"]),
                        "title": t,
                        "source": f"[COMM][Reddit/{sub}]",
                        "url": r["url"],
                        # QG/FC는 커뮤니티 카드에 최소 정보만 PASS 처리
                        "qg_status": "PASS",
                        "qg_score": final_score,
                        "qg_reason": "",
                        "fc_status": "PASS",
                        "fc_score": 0,
                        "fc_evidence": [],
                        "fc_standards": [],
                        # EN/KR 요약(메타로 추천/댓글 표시)
                        "summary_en": [f"▲{up} · 💬{cm} · r/{sub}", "", ""],
                        "summary_ko": [f"추천 {up} · 댓글 {cm} · r/{sub}", "", ""],
                        "editor_note": "",
                        "full_ko": "",
                        "summary_ko_text": "",
                        # 메타 점수(어드민 정렬용)
                        "kw_score": kw_norm,
                        "kw_hits": wl_hits,
                        "domain_score": 0,
                        "total_score": final_score,
                    }
                )

        tmp.sort(key=lambda x: x["qg_score"], reverse=True)
        # 서브당 상위 limit_per
        seen, kept = set(), []
        for it in tmp:
            if it["url"] in seen:
                continue
            kept.append(it)
            seen.add(it["url"])
            if len(kept) >= max(1, limit_per):
                break
        items.extend(kept)
    return items


# ====================== 기사 수집(공식 소스) ======================
def fetch_articles(cfg: dict | None = None) -> List[Dict]:
    feeds_dir = os.path.join(os.getcwd(), "feeds")
    RULES = _load_editor_rules()
    TOPK = int((RULES.get("top_k") or 1))

    MODEL_HINT = ((cfg or {}).get("features", {}) or {}).get("translate_model")
    feats_fc = (((cfg or {}).get("features", {}) or {}).get("factcheck") or {})
    strict_fc = bool(feats_fc.get("strict", True))

    articles: List[Dict] = []

    # 기본 공식 소스
    sources: Dict[str, str] = {
        "ECSS Active Standards": "https://ecss.nl/standards/active-standards/",
        "IPC Global Insight": "https://ipcglobalinsight.org/newsletters/global-insight",
        "IPC News": "https://www.ipc.org/news",
        "PCB007": "https://iconnect007.com/page/pcb007/12/",
        "SMT007": "https://iconnect007.com/page/smt007/11/",
        "Meta AI Blog": "https://ai.meta.com/blog/",
    }
    off_path = os.path.join(feeds_dir, "official_sources.json")
    if os.path.exists(off_path):
        try:
            src_map = json.load(open(off_path, "r", encoding="utf-8"))
            if isinstance(src_map, dict):
                sources = {k: v for k, v in src_map.items() if v}
        except Exception:
            pass

    for label, url in sources.items():
        if not url:
            continue
        try:
            url_listing = canonical_url(url)
            listing_html = _get(url_listing)
            cand_urls = _pick_top_articles_by_keywords(url_listing, listing_html, RULES, top_k=TOPK)
            if not cand_urls:
                first_article = _pick_first_article_url(url_listing, listing_html, cfg) if (listing_html and not listing_html.startswith("__ERROR__")) else url_listing
                cand_urls = [canonical_url(first_article)]

            for art_url in cand_urls:
                title, desc = _fetch_title_desc(art_url)
                logging.info(f"[QG START] {label} | {art_url}")
                try:
                    raw_html = _get(art_url)
                except Exception as e:
                    logging.warning(f"GET fail: {art_url} | {e}")
                    raw_html = ""

                eng_src_text, _, _, _ = _extract_main_text_from_html(raw_html or "")
                eng_src = _clean_for_translation(eng_src_text or desc or title or "")
                summary_ko_text = _best_ko_summary(eng_src)

                qg = quality_gate(art_url, raw_html, title, label, cfg)
                three_en = (_sentences(desc) or _sentences(eng_src) or ["Pending editor review"])[:3]
                while len(three_en) < 3:
                    three_en.append("Pending editor review")

                # 발행 단계에서만 1문장 번역
                if (not COLLECT_PHASE) and _OPENAI_KEY:
                    three_ko = []
                    for ln in three_en:
                        payload = {
                            "model": MODEL_HINT or _OPENAI_MODEL,
                            "messages": [
                                {"role": "system", "content": "Translate to concise Korean. ONE sentence only."},
                                {"role": "user", "content": (ln or "")},
                            ],
                            "temperature": 0.2,
                        }
                        try:
                            r = requests.post(
                                _OPENAI_URL,
                                headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
                                data=json.dumps(payload),
                                timeout=(8, 12),
                            )
                            if r.ok:
                                data = r.json()
                                ko = ((data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or ln)
                            else:
                                ko = ln
                        except Exception:
                            ko = ln
                        three_ko.append(ko)
                else:
                    three_ko = three_en

                fc = fact_check(art_url, raw_html, title, three_ko, cfg)
                fc_status, fc_score = fc["status"], int(fc["score"])
                agg_status, agg_reason = qg["status"], qg.get("reason", "")
                if fc_status == "FAIL" and strict_fc:
                    agg_status, agg_reason = "REJECT", f"FC실패:{fc.get('reason','')}"
                elif fc_status == "HOLD" and qg["status"] == "PASS":
                    agg_status, agg_reason = "HOLD", "FC대기"

                kw_s, kw_hits = _keyword_score(title, eng_src, RULES)
                dom_s = _domain_score(art_url, RULES)
                total = int(qg.get("score", 0)) + int(fc.get("score", 0)) + kw_s + dom_s

                art = {
                    "id": _article_id(title, art_url),
                    "title_hash": _hash_title(title),
                    "url_hash": _hash_url(art_url),
                    "title": title,
                    "source": label,
                    "url": art_url,
                    "summary_en": three_en,
                    "summary_ko": three_ko,
                    "editor_note": f"{label} 자동 요약",
                    "qg_status": agg_status,
                    "qg_score": int(qg.get("score", 0)),
                    "qg_reason": agg_reason,
                    "fc_status": fc_status,
                    "fc_score": fc_score,
                    "fc_evidence": fc.get("evidence", []),
                    "fc_standards": fc.get("standards", []),
                    "full_ko": None,
                    "summary_ko_text": summary_ko_text,
                    "kw_score": int(kw_s),
                    "kw_hits": kw_hits,
                    "domain_score": int(dom_s),
                    "total_score": int(total),
                }
                articles.append(art)
        except Exception as e:
            logger.warning("fetch_articles item fail: %s | %s", url, e)
            continue

    return articles


# ====================== 카드 저장/업서트 ======================
def _archive_md_path(archive_dir: str, name: str = "퀄리뉴스") -> str:
    today = datetime.now()
    p = os.path.join(archive_dir, today.strftime(f"{name}_%Y-%m-%d.md"))
    return p


def _upsert_card(md_path: str, art: dict, block_text: str) -> None:
    import io

    card_id = art.get("id") or ""
    try:
        with io.open(md_path, "r", encoding="utf-8") as rf:
            doc = rf.read()
    except FileNotFoundError:
        doc = ""
    pat = re.compile(rf"(?s)<!--ID:{re.escape(card_id)}\b.*?-->(?:.*?)(?=(?:\s*<!--ID:)|\Z)")
    m = pat.search(doc)
    if m:
        if m.group(0).strip() != block_text.strip():
            new_doc = doc[: m.start()] + block_text + doc[m.end() :]
            os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
            with open(md_path, "w", encoding="utf-8", newline="") as wf:
                wf.write(new_doc)
            logger.info("UPsert: 카드 갱신(ID=%s)", card_id)
    else:
        if doc and not doc.endswith("\n\n"):
            doc += "\n\n"
        doc += block_text
        if not doc.endswith("\n"):
            doc += "\n"
        os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
        with open(md_path, "w", encoding="utf-8", newline="") as wf:
            wf.write(doc)
        logger.info("UPsert: 신규 추가(ID=%s)", card_id)


def save_cards(articles: List[Dict]) -> None:
    out_dir = os.path.join(os.getcwd(), "archive")
    os.makedirs(out_dir, exist_ok=True)
    out_path = _archive_md_path(out_dir, "퀄리뉴스")
    written = 0
    for art in articles:
        block = make_card(art)
        had = False
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as rf:
                had = f"<!--ID:{art['id']}" in rf.read()
        _upsert_card(out_path, art, block)
        if not had:
            written += 1
    print(f"[{datetime.now().strftime('%Y%m%d')}] 카드 저장 완료({written}건) → {out_path}")


# ============== HTML 출력(기사) — 핀 우선 ==============
def export_html(md_path: str, cfg: dict | None = None) -> None:
    html_out = md_path.replace(".md", ".html")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    feats = (cfg or {}).get("features", {}) if cfg else {}
    DUAL_TITLE = bool(feats.get("dual_title", True))
    SECTION_ORDER = list(feats.get("section_order", ["표준 뉴스", "글로벌 전자생산", "AI 뉴스", "퀄리뉴스"]))

    fq = feats.get("quality") or {}
    MAX_TOTAL = int(fq.get("max_total", 6))
    MAX_PER_SEC = int(fq.get("max_per_section", 3))

    ff = feats.get("factcheck") or {}
    FC_STRICT = bool(ff.get("strict", True))

    def _domain(u: str) -> str:
        try:
            d = urlparse(u).netloc.lower()
            return d[4:] if d.startswith("www.") else d
        except Exception:
            return ""

    def _clean(t: str) -> str:
        if not t:
            return ""
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(r"`{1,3}[^`]+`{1,3}", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def _is_placeholder_ko(s: str) -> bool:
        if not s or not s.strip():
            return True
        n = re.sub(r"[^\w\s:.\-()%가-힣]", " ", s).strip().lower()
        return ("편집 검토/추가 예정" in n) or (n in {"기사"}) or n.startswith(("title:", "url source:", "published time:", "warning:"))

    def _ko_title(en: str, src: str) -> str:
        if not DUAL_TITLE or not en:
            return ""
        try:
            ko = ""
            if _OPENAI_KEY:
                payload = {
                    "model": feats.get("translate_model") or _OPENAI_MODEL,
                    "messages": [{"role": "system", "content": "Translate to concise Korean. ONE sentence only."}, {"role": "user", "content": en}],
                    "temperature": 0.2,
                }
                r = requests.post(
                    _OPENAI_URL,
                    headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=(8, 12),
                )
                if r.ok:
                    data = r.json()
                    ko = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                if ko:
                    return ko
        except Exception:
            pass
        return _kr_one_liner(en, src)

    def _attr(s: str) -> str:
        return _html.escape(s or "").replace('"', "&quot;").replace("'", "&#39;")

    # 선택 로드
    sel_ids, sel_notes = load_selection(cfg or {})
    sel_extra = _load_selmap_extra(cfg or {})

    blocks = [b for b in re.split(r"(?=<!--ID:)", md_text) if b.strip().startswith("<!--ID:")]
    per_section: dict[str, list[tuple[int, str, str]]] = {}
    pinned_cards: list[tuple[int, str, int]] = []
    seen_ids = set()

    for block in blocks:
        first = block.splitlines()[0] if block.splitlines() else ""
        m1 = re.search(r"QG:([A-Z]+)", first)
        m2 = re.search(r"QS:(\d+)", first)
        qg_status = m1.group(1) if m1 else "HOLD"
        qg_score = int(m2.group(1)) if m2 else 0
        fc_s = re.search(r"FC:([A-Z]+)", first)
        fc_sc = re.search(r"FS:(\d+)", first)
        fc_status = fc_s.group(1) if fc_s else "HOLD"
        fc_score = int(fc_sc.group(1)) if fc_sc else 0
        ts_m = re.search(r"TS:(\d+)", first)
        kw_m = re.search(r"KW:(\d+)", first)
        ds_m = re.search(r"DS:(\d+)", first)
        ts = int(ts_m.group(1)) if ts_m else 0
        kw = int(kw_m.group(1)) if kw_m else 0
        ds = int(ds_m.group(1)) if ds_m else 0

        lines = [ln for ln in block.splitlines() if ln.strip()]
        title = url = source = section = id_hint = None
        sum_ko = []

        if lines and lines[0].startswith("<!--ID:"):
            m = re.search(r"<!--ID:([0-9a-fA-F]{32})", lines[0])
            id_hint = m.group(1) if m else None
        m_sec = re.search(r"^\[퀄리뉴스\]\s*(.*?)\s*—\s*\d{4}-\d{2}-\d{2}", block, flags=re.MULTILINE)
        section = (m_sec.group(1).strip() if m_sec else "퀄리뉴스")

        for ln in lines:
            if ln.startswith("📰 제목:"):
                m = re.search(r'📰 제목:\s*"?(.+?)"?\s*$', ln)
                title = m.group(1).strip() if m else None
            elif ln.startswith("출처:"):
                source = (ln.split("출처:", 1)[1] or "").strip()
                m = re.search(r"\((https?://[^)\s]+)\)", ln)
                url = m.group(1).strip() if m else None
                if "(" in source:
                    source = source.split("(", 1)[0].strip()
            elif ln.strip().startswith("- KO:"):
                sum_ko.append(_clean(ln.split(":", 1)[1]))

        ko_from_sel = (sel_extra.get(id_hint, {}) or {}).get("summary_ko_text") or ""
        kos = [k for k in sum_ko if not _is_placeholder_ko(k)]
        sum_text = (ko_from_sel if ko_from_sel.strip() else " ".join(kos))[:220]

        chosen = (id_hint in sel_ids) if sel_ids else ((qg_status == "PASS") and (not FC_STRICT or fc_status == "PASS"))
        if not chosen:
            continue

        ko_t = _ko_title(title or "", source or "")
        headline = ko_t or title or "(제목 없음)"
        h2 = (
            f"<h2 class='headline'><a href='{_html.escape(url)}' target='_blank' rel='noopener' title='{_attr(sum_text)}'>{_html.escape(headline)}</a></h2>"
            if url
            else f"<h2 class='headline'>{_html.escape(headline)}</h2>"
        )
        meta = f"<div class='meta'>{_html.escape(_domain(url))}</div>" if url else "<div class='meta'>&nbsp;</div>"

        ednote = (sel_notes.get(id_hint, "") or "").strip()
        if not ednote:
            fallback = ko_t or (kos[0] if kos else "")
            ednote = f"{fallback} — (자동 생성)" if fallback else "핵심만 요약해 전달합니다."
        ed_html = f"<div class='ednote'>🖊 {_html.escape(ednote)}</div>"

        meta_sel = sel_extra.get(id_hint or "", {})
        ko_full = meta_sel.get("full_ko") or ""
        ko_sum = meta_sel.get("summary_ko_text") or ""
        ko_body = (ko_full or "").strip() or (ko_sum or "").strip()
        tr_html = f"<details class='tr'><summary>번역</summary><div class='ko-inner'>{_html.escape(ko_body)}</div></details>" if ko_body else ""

        card_html = f"<div class='card' data-summary='{_attr(sum_text)}'>{h2}{meta}{ed_html}{tr_html}</div>"
        score = ts or (qg_score + fc_score + kw + ds)

        pin_meta = sel_extra.get(id_hint or "", {})
        if pin_meta.get("pinned"):
            pinned_cards.append((int(pin_meta.get("pin_ts") or 0), card_html, score))
            seen_ids.add(id_hint)

        per_section.setdefault(section, []).append((score, card_html, id_hint))

    ordered_sections = [s for s in SECTION_ORDER if s in per_section] + [s for s in per_section if s not in SECTION_ORDER]

    pinned_cards.sort(key=lambda x: (x[0], -x[2]))
    top_section_html = ""
    if pinned_cards:
        top_section_html = "<div class='section' id='sec-0'><h2 class='sec-title'>⭐ 중요 뉴스</h2>" + "".join(html for _, html, _ in pinned_cards) + "</div>"
        for sec in list(per_section.keys()):
            per_section[sec] = [t for t in per_section[sec] if (t[2] not in seen_ids)]

    section_html_parts, total = [], 0
    for idx_sec, sec in enumerate(ordered_sections, 1):
        cards = sorted(per_section[sec], key=lambda x: x[0], reverse=True)[:MAX_PER_SEC]
        kept = []
        for sc, html_card, _id in cards:
            if total >= MAX_TOTAL:
                break
            kept.append(html_card)
            total += 1
        if kept:
            section_html_parts.append(f"<div class='section' id='sec-{idx_sec}'><h2 class='sec-title'>{_html.escape(sec)}</h2>" + "".join(kept) + "</div>")

    toc = (
        '<div class="toc">'
        + " · ".join(f'<a href="#sec-{i}">{_html.escape(sec)}</a>' for i, sec in enumerate([*[s for s in ordered_sections][:len(section_html_parts)]], 1))
        + "</div>"
    ) if section_html_parts else ""

    html_doc = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"><title>퀄리뉴스</title>
<style>
:root {{ --ink:#111; --line:#e5e7eb; --muted:#6b7280; }}
body {{ font-family: system-ui,"Noto Sans KR","맑은 고딕",sans-serif; margin:40px; background:#fff; color:var(--ink); }}
h1 {{ text-align:center; font-size:24px; margin:0 0 8px; }}
.toc {{ text-align:center; color:var(--muted); font-size:12px; margin:0 0 16px; }}
.section h2.sec-title {{ font-size:14px; border-left:4px solid var(--ink); padding-left:8px; margin:28px 0 6px; }}
.card {{ position:relative; background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:16px 0; }}
.headline {{ margin:0 0 6px; font-size:18px; line-height:1.4; }}
.headline a {{ color:var(--ink); text-decoration:none; }} .headline a:hover {{ text-decoration:underline; }}
.meta {{ color:var(--muted); font-size:12px; }}
.ednote {{ margin-top:8px; font-size:13px; border-left:3px solid #bbb; padding-left:10px; }}
.tr {{ margin-top:8px; }} .tr summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
.ko-inner {{ font-size:14px; line-height:1.7; margin-top:6px; white-space:pre-wrap; }}
</style></head><body>
<h1>퀄리뉴스 ({datetime.now().strftime('%Y-%m-%d')})</h1>
{toc}
{top_section_html + "".join(section_html_parts)}
</body></html>"""
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"신문 HTML 저장: {html_out}")


# ============== HTML 출력(커뮤니티) ==============
def export_html_community(cfg: dict | None = None) -> None:
    sel_path = _selection_path_comm(cfg or {})
    if not os.path.exists(sel_path):
        print("커뮤니티 선택 파일이 없습니다.")
        return
    data = json.load(open(sel_path, "r", encoding="utf-8"))
    items = [a for a in data.get("articles", []) if a.get("selected")]
    items.sort(key=lambda x: (x.get("pinned", False), x.get("total_score", 0)), reverse=True)
    html_out = os.path.join(os.getcwd(), "archive", f"퀄리_커뮤니티_{datetime.now().strftime('%Y-%m-%d')}.html")
    os.makedirs(os.path.dirname(html_out), exist_ok=True)

    def _attr(s: str) -> str:
        return _html.escape(s or "").replace('"', "&quot;").replace("'", "&#39;")

    cards = []
    for a in items:
        title = a.get("ko_title") or a.get("title") or "(제목 없음)"
        url = a.get("url")
        meta = (a.get("summary_ko") or ["", "", ""])[0]
        ed = (a.get("editor_note") or "").strip() or _kr_one_liner(title, a.get("source", ""))
        h = f"""<div class='card'>
  <h2 class='headline'><a href='{_attr(url)}' target='_blank' rel='noopener'>{_html.escape(title)}</a></h2>
  <div class='meta'>{_html.escape(a.get("source",""))} · {_html.escape(meta)}</div>
  <div class='ednote'>🖊 {_html.escape(ed)}</div>
</div>"""
        cards.append(h)

    doc = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"><title>퀄리 커뮤니티</title>
<style>
body {{ font-family: system-ui, "Noto Sans KR", sans-serif; margin:40px; }}
h1 {{ text-align:center; font-size:24px; }}
.card {{ border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; margin:16px 0; }}
.headline {{ margin:0 0 6px; font-size:18px; }} .headline a {{ color:#111; text-decoration:none; }}
.headline a:hover {{ text-decoration:underline; }} .meta {{ color:#6b7280; font-size:12px; }}
.ednote {{ margin-top:8px; font-size:13px; border-left:3px solid #bbb; padding-left:10px; }}
</style></head><body>
<h1>퀄리 커뮤니티 ({datetime.now().strftime('%Y-%m-%d')})</h1>
{''.join(cards) if cards else '<p>선택된 항목이 없습니다.</p>'}
</body></html>"""
    with open(html_out, "w", encoding="utf-8") as wf:
        wf.write(doc)
    print(f"커뮤니티 HTML 저장: {html_out}")


# ====================== 발행 전 자동 보강 ======================
def _editor_note_from_text(title: str, text: str, ko_title: str = "", source: str = "") -> str:
    base_hint = (ko_title or title or "").strip()
    core = _clean_for_translation(text or "")
    if _OPENAI_KEY and core:
        try:
            payload = {
                "model": os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o"),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior editor for SMT/PCB manufacturing. Write ONE Korean sentence (30~60 chars) that captures the practical takeaway for process/quality engineers. No fluff, no emojis.",
                    },
                    {"role": "user", "content": f"제목: {base_hint}\n본문요약: {core[:1500]}"},
                ],
                "temperature": 0.2,
            }
            r = requests.post(
                _OPENAI_URL,
                headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=(8, 12),
            )
            if r.ok:
                data = r.json()
                note = ((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
                if note:
                    return " ".join(note.split())
        except Exception:
            pass
    if base_hint:
        return f"{base_hint} — 공정·품질 관점 핵심만 요약"
    return "핵심만 요약해 전달합니다."


def postprocess_selected_for_publish(cfg: dict) -> None:
    """선택된 기사만 전문 번역/한마디 자동 생성 → selection 파일에 주입."""
    sel_path = _selection_path(cfg)
    if not os.path.exists(sel_path):
        return
    data = json.load(open(sel_path, "r", encoding="utf-8"))
    changed = False
    new_articles = []
    for a in data.get("articles", []):
        if not a.get("selected"):
            new_articles.append(a)
            continue
        url = a.get("url", "")
        title = a.get("title", "")
        ko_title = a.get("ko_title", "")
        try:
            raw_html = _get(url)
        except Exception:
            raw_html = ""
        text, *_ = _extract_main_text_from_html(raw_html or "")
        text = _clean_for_translation(text or "")
        # 전문 번역 (영문 비중 높을 때만)
        if not a.get("full_ko"):
            if _is_mostly_english(text):
                ko = _translate_en_to_ko(text)
                if ko:
                    a["full_ko"] = ko
                    changed = True
            elif a.get("summary_ko_text"):
                pass
        # 편집장 한마디
        ed = (a.get("editor_note", "") or "").strip()
        if not ed:
            note = _editor_note_from_text(title, text, ko_title)
            if note:
                a["editor_note"] = note
                changed = True
        new_articles.append(a)
    if changed:
        data["articles"] = new_articles
        with open(sel_path, "w", encoding="utf-8") as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        logger.info("선택 파일 갱신(번역/한마디): %s", sel_path)


# ====================== 설정/작업 ======================
def validate_config(cfg: dict) -> Tuple[dict, List[str]]:
    out = json.loads(json.dumps(DEFAULT_CFG))

    def deep(base, over):
        for k, v in (over or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = deep(base.get(k, {}), v)
            else:
                base[k] = v
        return base

    merged = deep(out, cfg or {})
    warns: List[str] = []
    for key in ["archive", "reports", "log", "backup"]:
        if not merged["paths"].get(key):
            merged["paths"][key] = DEFAULT_CFG["paths"][key]
    return merged, warns


def load_config(path: str = "config.json") -> dict:
    try:
        raw = json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        logging.getLogger().warning("config.json 읽기 오류 → 기본값 사용(%s)", e)
        raw = {}
    cfg, _ = validate_config(raw)
    for p in ["archive", "reports", "backup"]:
        os.makedirs(cfg["paths"][p], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["log"]) or ".", exist_ok=True)
    return cfg


def job_collect() -> None:
    cfg = load_config("config.json")
    setup_rotating_logger(cfg["paths"]["log"])
    arts = fetch_articles(cfg)
    save_cards(arts)
    write_selection_file(arts, cfg)


def job_publish() -> None:
    # 발행 모드 전환 + 번역/한마디 보강
    global RUN_MODE, COLLECT_PHASE
    RUN_MODE = "publish"
    COLLECT_PHASE = False
    cfg = load_config("config.json")
    setup_rotating_logger(cfg["paths"]["log"])
    postprocess_selected_for_publish(cfg)
    md_path = _archive_md_path(cfg["paths"]["archive"], "퀄리뉴스")
    if md_path and os.path.exists(md_path):
        export_html(md_path, cfg)
    else:
        print("오늘 MD가 없습니다. 먼저 --collect 를 실행하세요.")
    reports_dir = cfg["paths"]["reports"]
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, f"news_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"), "w", encoding="utf-8") as wf:
        wf.write("퀄리뉴스 발행 완료\n")


def job_collect_community() -> None:
    cfg = load_config("config.json")
    setup_rotating_logger(cfg["paths"]["log"])
    items = fetch_community(cfg)
    write_selection_file_community(items, cfg)
    print("커뮤니티 수집 완료")


def job_publish_community() -> None:
    cfg = load_config("config.json")
    setup_rotating_logger(cfg["paths"]["log"])
    export_html_community(cfg)


# ====================== CLI ======================
def main() -> None:
    ap = argparse.ArgumentParser(description="퀄리뉴스 수집/발행 — 기사/커뮤니티 분리 운영")
    ap.add_argument("--collect", action="store_true", help="기사 수집/선택파일 생성")
    ap.add_argument("--publish", action="store_true", help="기사 HTML 발행(번역/한마디 자동 보강)")
    ap.add_argument("--collect-community", action="store_true", help="커뮤니티 수집/선택파일 생성")
    ap.add_argument("--publish-community", action="store_true", help="커뮤니티 HTML 발행")
    args = ap.parse_args()

    if args.publish:
        job_publish()
    elif args.collect_community:
        job_collect_community()
    elif args.publish_community:
        job_publish_community()
    else:
        job_collect()


if __name__ == "__main__":
    main()

