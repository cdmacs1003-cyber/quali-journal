# tools/repair_selection_files.py
from __future__ import annotations
import os, sys, json, datetime
from typing import Any, Dict, Iterable, List, Tuple

# --- 공통 경로 헬퍼 (tools/sync_selected_for_publish.py와 동일 컨셉) ---
try:
    from qj_paths import rel as qj_rel  # 있으면 사용
except ModuleNotFoundError:
    _THIS = os.path.abspath(__file__)
    _TOOLS = os.path.dirname(_THIS)
    _ROOT = os.path.dirname(_TOOLS)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    def qj_rel(*parts: str) -> str:
        return os.path.join(_ROOT, *parts)

# --- 경로 정의 ---
KW_JSON_CANDIDATES  = [qj_rel("data","selected_keyword_articles.json"), qj_rel("selected_keyword_articles.json")]
SEL_JSON_CANDIDATES = [qj_rel("data","selected_articles.json"),          qj_rel("selected_articles.json")]

# --- 유틸 ---
def _now_date() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")

def _load_first(paths: List[str]) -> Tuple[str, Any]:
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                try:
                    return p, json.load(f)
                except json.JSONDecodeError:
                    print(f"[!] JSON 깨짐: {p} -> 빈 배열로 치유")
                    return p, []
    # 파일이 없으면 템플릿 반환
    return paths[-1], {"date": _now_date(), "items": [], "articles": []}

def _unique_key(a: Dict[str, Any]) -> str|None:
    for k in ("url","link","guid","id"):
        v = a.get(k)
        if v:
            s = str(v).strip()
            if s:
                return s
    # 그래도 없으면 제목+소스로 대체 키
    t = str(a.get("title", "")).strip()
    s = str(a.get("source", "") or a.get("domain","")).strip()
    return f"{t}|{s}" if (t or s) else None

def _iter_articles(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("articles"), list): return payload["articles"]
        if isinstance(payload.get("items"), list):    return payload["items"]
    if isinstance(payload, list):
        return payload
    return []

def _normalize_payload(raw: Any) -> Dict[str, Any]:
    """표준 구조로 정규화: {date, items[], articles[]} 두 리스트는 같은 레퍼런스를 가짐."""
    arts = _iter_articles(raw)
    if not isinstance(arts, list):
        arts = []
    date = None
    if isinstance(raw, dict):
        date = raw.get("date")
    if not isinstance(date, str) or not date:
        date = _now_date()
    # items와 articles가 같은 리스트를 가리키도록
    payload = {"date": date, "items": arts, "articles": arts}
    return payload

def _dedupe_and_fix(arts: List[Dict[str, Any]], approved_only: bool|None=None) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    today = _now_date()
    for a in arts:
        if not isinstance(a, dict):
            continue
        # 기본 필드 치유
        a.setdefault("title", "")
        a.setdefault("source", a.get("domain",""))
        a.setdefault("date", today)
        # 승인값 정규화
        appr = a.get("approved")
        a["approved"] = bool(appr) if appr is not None else False
        # 상태값 정규화
        a.setdefault("state", "ready" if a["approved"] else "candidate")
        # 고유키 중복 제거
        k = _unique_key(a) or f"__row_{len(out)}"
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    # 선택 필터
    if approved_only is True:
        out = [a for a in out if a.get("approved") is True]
    return out

def _save(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main() -> int:
    # 1) 작업본(KW) 로드/치유
    kw_path, kw_raw = _load_first(KW_JSON_CANDIDATES)
    kw_payload = _normalize_payload(kw_raw)
    kw_payload["articles"] = kw_payload["items"] = _dedupe_and_fix(_iter_articles(kw_payload), approved_only=None)
    _save(kw_path, kw_payload)
    print(f"[OK] repaired: {kw_path} (count={len(kw_payload['articles'])})")

    # 2) 발행본(SELECTED) 로드/치유: 승인만 남김
    sel_path, sel_raw = _load_first(SEL_JSON_CANDIDATES)
    sel_payload = _normalize_payload(sel_raw)
    # 발행본은 승인(True)만 남기고 치유
    sel_payload["articles"] = sel_payload["items"] = _dedupe_and_fix(_iter_articles(kw_payload), approved_only=True)
    _save(sel_path, sel_payload)
    print(f"[OK] rebuilt : {sel_path} (approved_count={len(sel_payload['articles'])})")

    print("[HINT] 필요하면 이어서: python tools/sync_selected_for_publish.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
