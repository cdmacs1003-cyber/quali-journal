# app/qgfc.py
"""
오케스트레이터에서 수집 직후 호출할 QG/FC/스코어링 유틸.
- apply_quality_gates(item, cfg) -> bool
- compute_score(item, weights, trust_lookup) -> float
- fc_confirm(item, universe, cfg) -> bool
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import math

def _norm(x, maxv):
    if maxv <= 0: return 0.0
    return max(0.0, min(1.0, float(x) / float(maxv)))

def _title_sim(a:str, b:str)->float:
    # 간단한 Jaccard 유사도(토큰 기반)
    ta = set(t for t in _tokenize(a))
    tb = set(t for t in _tokenize(b))
    if not ta or not tb: return 0.0
    inter = len(ta & tb); union = len(ta | tb)
    return inter/union if union else 0.0

def _tokenize(s:str)->List[str]:
    return [t.lower() for t in s.split() if t.isalnum() or t.replace("-","").isalnum()]

def apply_quality_gates(item:Dict[str,Any], cfg:Dict[str,Any])->bool:
    title = (item.get("title") or "").strip()
    summary = (item.get("summary") or item.get("desc") or "").strip()
    url = (item.get("url") or "").strip()

    if len(title) < cfg["qg"]["min_title_len"]: return False
    if len(summary) < cfg["qg"]["min_summary_len"]: return False
    if not cfg["qg"].get("allow_query_in_url", False) and "?" in url: return False

    dom = (item.get("source_domain") or "").lower()
    if dom and dom in set(d.lower() for d in cfg["qg"].get("banned_domains", [])): return False

    # 키워드 요구조건
    rq_all = cfg["qg"].get("required_keywords_all") or []
    rq_any = cfg["qg"].get("required_keywords_any") or []
    text = f"{title}\n{summary}".lower()
    if rq_all and not all(k.lower() in text for k in rq_all): return False
    if rq_any and not any(k.lower() in text for k in rq_any): return False

    return True

def fc_confirm(item:Dict[str,Any], universe:List[Dict[str,Any]], cfg:Dict[str,Any])->bool:
    thr = float(cfg["fc"]["title_similarity_threshold"])
    need = int(cfg["fc"]["min_confirming_sources"])
    hours = int(cfg["fc"]["time_window_hours"])

    title = (item.get("title") or "")
    t0 = _dt(item.get("published_at"))
    dom0 = (item.get("source_domain") or "").lower()

    confirms = 0
    for other in universe:
        if other is item: continue
        dom = (other.get("source_domain") or "").lower()
        if dom == dom0: # 같은 도메인은 독립 증거로 보지 않음
            continue
        sim = _title_sim(title, other.get("title") or "")
        if sim < thr: continue
        if hours and t0 and abs((_dt(other.get("published_at")) - t0).total_seconds()) > hours * 3600:
            continue
        confirms += 1
        if confirms >= need:
            return True
    return False

def _dt(s)->datetime:
    if not s: return None
    if isinstance(s, datetime): return s
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00"))
    except Exception:
        return None

def compute_score(item:Dict[str,Any], weights:Dict[str,float], trust_lookup:Dict[str,float])->float:
    w = weights
    up = int(item.get("upvotes") or 0)
    vw = int(item.get("views") or 0)
    hits = int(item.get("kw_hits") or 0)
    length = int(item.get("length") or 0)
    fc_pass = bool(item.get("fc_pass"))
    qg_pass = bool(item.get("qg_pass"))

    trust = trust_lookup.get((item.get("source_category") or "default"), trust_lookup.get("default", 0.5))
    # 단순 정규화(실전에서는 최대값 사전집계 권장)
    score = (
        w["kw_hits"] * _norm(hits, 10) +
        w["upvotes"] * _norm(up, 500) +
        w["views"]   * _norm(vw, 10000) +
        w["source_trust"] * float(trust) +
        w["length"]  * _norm(length, 8000)
    )
    # 최신성 보너스(최대 1.0)
    p = _dt(item.get("published_at"))
    if p:
        age_hours = max(1.0, (datetime.utcnow() - p).total_seconds()/3600.0)
        rec = 1.0 / (1.0 + math.log(age_hours, 2))
        score += w.get("recency", 0.0) * rec

    if fc_pass: score += w.get("fc_bonus", 0.0)
    if qg_pass: score += w.get("qg_bonus", 0.0)
    return round(score, 6)
