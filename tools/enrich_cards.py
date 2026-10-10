# -*- coding: utf-8 -*-
"""
tools/enrich_cards.py
- Read working JSON (selected_keyword_articles.json / selected_articles.json)
- Fill source summaries; generate Korean only when --translate is requested
- Save JSON back in-place (non-destructive: keep unknown fields)
- Print a small JSON result to STDOUT for server logging
  { "ok": true, "source": "...", "count": N, "updated": M }
"""
from __future__ import annotations
import argparse, os, json, re
from pathlib import Path
from datetime import datetime as dt

if __package__:
    from ._d4_utils import has_korean, summary_text, to_ko
else:  # Also support python tools/enrich_cards.py
    from _d4_utils import has_korean, summary_text, to_ko

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

def _text(*xs) -> str:
    s = " ".join(summary_text(x) for x in xs)
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

def _ensure_summaries(a: dict, translate: bool = False) -> bool:
    """
    Idempotently fill summary fields. Return True if modified.
    Preserve real Korean summaries and retain rejected legacy values separately.
    Without --translate this function never calls OpenAI, even if a key exists.
    """
    before = dict(a)
    title = a.get("title") or a.get("headline") or ""
    desc  = a.get("desc") or a.get("description") or a.get("snippet") or ""
    body  = a.get("content") or a.get("text") or ""
    current_en = summary_text(a.get("summary_en") or a.get("summary"))

    if not current_en:
        base = _text(desc, body) or summary_text(title)
        # naive compression
        s = _shorten(base, 360)
        a["summary"] = s
        a["summary_en"] = s

    current_en = summary_text(a.get("summary_en") or a.get("summary"))
    ko_fields = ("summary_ko", "ko_summary", "summary_kr")
    current_ko = next((summary_text(a.get(f)) for f in ko_fields
                       if has_korean(summary_text(a.get(f)))), "")
    rejected = dict(a.get("summary_ko_rejected") or {})
    for field in ko_fields:
        value = a.get(field)
        if value and not has_korean(summary_text(value)):
            rejected[field] = value
    if rejected:
        a["summary_ko_rejected"] = rejected

    if current_ko:
        status = a.get("summary_ko_status")
        if status not in ("translated", "source_korean", "existing"):
            status = "existing"
    elif has_korean(current_en):
        current_ko, status = current_en, "source_korean"
    elif not current_en:
        status = "no_source"
    elif not translate:
        status = "not_requested"
    elif os.getenv("QUALI_DISABLE_OPENAI", "0") == "1":
        status = "disabled"
    elif not (os.getenv("OPENAI_API_KEY") or "").strip():
        status = "missing_key"
    else:
        current_ko = to_ko(current_en)
        if has_korean(current_ko):
            status = "translated"
        else:
            current_ko, status = "", "unavailable"

    # A Korean title alone must not turn an English summary into Korean text.
    a["summary_ko"] = current_ko
    a["ko_summary"] = current_ko
    if "summary_kr" in a:
        a["summary_kr"] = current_ko
    a["summary_ko_status"] = status
    return a != before

def _iter_articles(obj):
    # supports both {"articles":[...]} and {"items":[...]} or a list
    if isinstance(obj, dict):
        for key in ("articles", "items"):
            if isinstance(obj.get(key), list):
                return obj.get(key), key
        return None, None
    if isinstance(obj, list):
        return obj, None
    return None, None

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("keyword", "selection"), default="keyword")
    parser.add_argument("--translate", action="store_true",
                        help="Translate missing Korean summaries using the existing OpenAI key (billable).")
    args = parser.parse_args(argv)
    mode = args.mode

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
            if _ensure_summaries(a, translate=args.translate):
                updated += 1

    # Also mirror back into the other key if present to keep UIs in sync
    if isinstance(obj, dict) and key == "items" and "articles" in obj and isinstance(obj["articles"], list):
        # naive sync by id/title match
        by_id = { (x.get("id") or x.get("url") or x.get("title")): x for x in arr }
        for b in obj["articles"]:
            k = (b.get("id") or b.get("url") or b.get("title"))
            if k in by_id:
                for fld in ("summary","summary_en","summary_ko","ko_summary","summary_kr",
                            "summary_ko_status", "summary_ko_rejected"):
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
        "korean_available": sum(bool(summary_text(a.get("summary_ko"))) for a in arr if isinstance(a, dict)),
        "translation_requested": args.translate,
        "ts": int(dt.now().timestamp())
    }, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
