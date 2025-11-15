# -*- coding: utf-8 -*-
"""
tools/enrich_cards.py
- Read working JSON (selected_keyword_articles.json / selected_articles.json)
- Fill `summary` (EN) and `summary_ko`/`ko_summary` (KO) if missing
- Save JSON back in-place (non-destructive: keep unknown fields)
- Print a small JSON result to STDOUT for server logging
  { "ok": true, "source": "...", "count": N, "updated": M }
"""
from __future__ import annotations
import os, sys, json, re
from pathlib import Path
from datetime import datetime as dt

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA  = ROOT / "data"
SRC_KEYWORD = DATA / "selected_keyword_articles.json"
SRC_SELECTED = ROOT / "selected_articles.json"

def _load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

def _save_json(path: Path, obj) -> str | None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return None
    except Exception as e:
        return str(e)

_re_hangul = re.compile(r"[가-힣]")

def _is_hangul(s: str) -> bool:
    return bool(_re_hangul.search(s or ""))

def _text(*xs) -> str:
    s = " ".join([str(x or "") for x in xs])
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _shorten(s: str, limit: int = 320) -> str:
    if len(s) <= limit:
        return s
    # cut at sentence boundary or word boundary
    cut = max(s.rfind(". ", 0, int(limit*0.9)), s.rfind(" ", 0, int(limit*0.95)))
    if cut < 80:
        cut = limit
    return s[:cut].rstrip() + "..."

def _ensure_summaries(a: dict) -> bool:
    """
    Idempotently fill summary fields. Return True if modified.
    Prefers existing values; only fills blanks.
    """
    changed = False
    title = a.get("title") or a.get("headline") or ""
    desc  = a.get("desc") or a.get("description") or a.get("snippet") or ""
    body  = a.get("content") or a.get("text") or ""
    current_en = (a.get("summary_en") or a.get("summary") or "").strip()
    current_ko = (a.get("summary_ko") or a.get("ko_summary") or a.get("summary_kr") or "").strip()

    if not current_en:
        base = _text(desc, body) or title
        # naive compression
        s = _shorten(base, 360)
        a["summary"] = s
        a.setdefault("summary_en", s)
        changed = True

    # refresh current_ko after potential change
    current_en = (a.get("summary_en") or a.get("summary") or "").strip()
    current_ko = (a.get("summary_ko") or a.get("ko_summary") or a.get("summary_kr") or "").strip()

    if not current_ko:
        # If the English summary already contains Hangul, treat as KO.
        if _is_hangul(current_en) or _is_hangul(title) or _is_hangul(desc):
            a["summary_ko"] = current_en
            a.setdefault("ko_summary", current_en)
        else:
            # Place a friendly Korean stub to avoid empty MD
            stub = "요약: " + _shorten(current_en or title, 360)
            a["summary_ko"] = stub
            a.setdefault("ko_summary", stub)
        changed = True

    return changed

def _iter_articles(obj):
    # supports both {"articles":[...]} and {"items":[...]} or a list
    if isinstance(obj, dict):
        for key in ("articles", "items"):
            if isinstance(obj.get(key), list):
                return obj.get(key), key
        return [], None
    if isinstance(obj, list):
        return obj, None
    return [], None

def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    # args: --mode keyword|selection (default keyword)
    mode = "keyword"
    for i, x in enumerate(argv):
        if x == "--mode" and i+1 < len(argv):
            mode = argv[i+1].strip().lower()

    # pick source path
    src = SRC_KEYWORD if mode != "selection" else SRC_SELECTED
    if not src.exists():
        # fallbacks
        src = SRC_SELECTED if SRC_SELECTED.exists() else SRC_KEYWORD

    obj, err = _load_json(src)
    if obj is None:
        print(json.dumps({"ok": False, "error": f"load failed: {src.name}: {err}"}), flush=True)
        return 1

    arr, key = _iter_articles(obj)
    if not isinstance(arr, list):
        print(json.dumps({"ok": False, "error": "no articles/items array"}), flush=True)
        return 2

    updated = 0
    for a in arr:
        if isinstance(a, dict):
            if _ensure_summaries(a):
                updated += 1

    # Also mirror back into the other key if present to keep UIs in sync
    if isinstance(obj, dict) and key == "items" and "articles" in obj and isinstance(obj["articles"], list):
        # naive sync by id/title match
        by_id = { (x.get("id") or x.get("url") or x.get("title")): x for x in arr }
        for b in obj["articles"]:
            k = (b.get("id") or b.get("url") or b.get("title"))
            if k in by_id:
                for fld in ("summary","summary_en","summary_ko","ko_summary","summary_kr"):
                    if fld in by_id[k]:
                        b[fld] = by_id[k][fld]

    err2 = _save_json(src, obj)
    if err2:
        print(json.dumps({"ok": False, "error": f"save failed: {err2}"}), flush=True)
        return 3

    # done
    print(json.dumps({
        "ok": True,
        "source": str(src.relative_to(ROOT)),
        "mode": mode,
        "count": len(arr),
        "updated": updated,
        "ts": int(dt.now().timestamp())
    }, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
