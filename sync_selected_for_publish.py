# tools/sync_selected_for_publish.py
from __future__ import annotations
import json, os, sys, datetime
from typing import Any, Dict, Iterable, List

try:
    from qj_paths import rel as qj_rel
except ModuleNotFoundError:
    _THIS = os.path.abspath(__file__)
    _TOOLS = os.path.dirname(_THIS)
    _ROOT = os.path.dirname(_TOOLS)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    def qj_rel(*parts: str) -> str:
        return os.path.join(_ROOT, *parts)

KW_JSON_PATH  = qj_rel("data", "selected_keyword_articles.json")
OUT_JSON_PATH = qj_rel("selected_articles.json")         # 루트
ALT_JSON_PATH = qj_rel("data", "selected_articles.json") # data/

def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        sys.exit(f"[X] 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _iter_articles(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("articles"), list):
        return payload["articles"]
    if isinstance(payload, list):
        return payload
    return []

def _unique_key(a: Dict[str, Any]) -> str | None:
    for k in ("url","link","guid","id"):
        v = a.get(k)
        if v:
            s = str(v).strip()
            if s: return s
    return None

def _filter_approved(arts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []; seen=set()
    for a in arts:
        if not (isinstance(a, dict) and a.get("approved") is True):
            continue
        key=_unique_key(a)
        if key and key in seen: continue
        if key: seen.add(key)
        out.append(a)
    return out

def _ensure_parent(path:str)->None:
    p=os.path.dirname(path) or "."
    if p and not os.path.exists(p): os.makedirs(p, exist_ok=True)

def main()->int:
    payload  = _read_json(KW_JSON_PATH)
    articles = list(_iter_articles(payload))
    approved = _filter_approved(articles)

    # ★ ready 태그와 날짜 보강
    today = datetime.date.today().strftime("%Y-%m-%d")
    for a in approved:
        if isinstance(a, dict):
            a.setdefault("state", "ready")
            a.setdefault("date",  today)

    out = { "date": today, "items": approved, "articles": approved }

    _ensure_parent(OUT_JSON_PATH)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    _ensure_parent(ALT_JSON_PATH)
    with open(ALT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved -> {OUT_JSON_PATH} (approved_count={len(approved)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
