from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
"""
Minimal classify_and_rank.py
- tools.make_daily_report.py, tests.d3_regression_check.py ?먯꽌 ?붽뎄?섎뒗 理쒖냼 湲곕뒫 ?쒓났
- ?먯닔: total_score > score > trust 媛以묒튂
- 遺꾨쪟: ?쒖? ?댁뒪 / AI ?댁뒪 / 湲濡쒕쾶 ?꾩옄?앹궛 / 而ㅻ??덊떚(?뚯뒪 ?쒓렇 ?ы븿 ??
"""
from __future__ import annotations
import os, json, math, re
from typing import List, Dict
from urllib.parse import urlparse

def load_rules(base_dir: str = ".") -> dict:
    """editor_rules.json ???덉쑝硫?媛以묒튂/?꾨찓??洹쒖튃???쎄퀬, ?놁쑝硫?湲곕낯媛?"""
    path = os.path.join(base_dir, "editor_rules.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
                if isinstance(j, dict):
                    return j
        except Exception:
            pass
    # 湲곕낯 洹쒖튃
    return {
        "weights": {
            "domain": {
                "ipc.org": 1.0, "ecss.nl": 1.0, "ieee.org": 0.8,
                "iconnect007.com": 0.4, "smt007.com": 0.4, "pcb007.com": 0.4
            }
        },
        "keywords": {
            "standard": ["ipc", "ecss", "mil", "nasa", "iec", "jedec"],
            "ai": ["ai","openai","x.ai","gemini","deepmind","anthropic","meta"]
        }
    }

def _host(u: str) -> str:
    try:
        d = urlparse(u or "").netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def _score(row: dict, domain_weights: dict) -> float:
    s = float(row.get("total_score") or row.get("qg_score") or row.get("score") or 0.0)
    w = domain_weights.get(_host(row.get("url","")), 0.0)
    return s + float(w)

def rank_articles(items: List[dict], rules: dict) -> List[dict]:
    dw = (rules.get("weights") or {}).get("domain") or {}
    rows = []
    for it in items or []:
        r = dict(it)
        r["_host"] = _host(r.get("url",""))
        r["_rank_score"] = _score(r, dw)
        rows.append(r)
    rows.sort(key=lambda x: (x.get("_rank_score",0), x.get("approved",False), x.get("selected",False)), reverse=True)
    return rows

def classify_sections(items: List[dict], rules: dict) -> Dict[str, List[dict]]:
    kws = (rules.get("keywords") or {})
    std_k = set(kws.get("standard") or [])
    ai_k  = set(kws.get("ai") or [])
    out = {"?쒖? ?댁뒪": [], "湲濡쒕쾶 ?꾩옄?앹궛": [], "AI ?댁뒪": [], "而ㅻ??덊떚": []}
    for a in items or []:
        title = (a.get("title") or "").lower()
        url = (a.get("url") or "").lower()
        source = (a.get("source") or "").lower()
        host = _host(url)
        sec = None
        blob = " ".join([title, url, source])
        if source.startswith("[comm]") or "community" in (a.get("type","")).lower():
            sec = "而ㅻ??덊떚"
        elif any(k in blob for k in std_k) or any(k in host for k in ("ipc.org","ecss.nl","ieee.org")):
            sec = "?쒖? ?댁뒪"
        elif any(k in blob for k in ai_k):
            sec = "AI ?댁뒪"
        else:
            sec = "湲濡쒕쾶 ?꾩옄?앹궛"
        out[sec].append(a)
    return out

