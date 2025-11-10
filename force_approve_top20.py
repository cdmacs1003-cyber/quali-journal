# tools/force_approve_top20.py
from __future__ import annotations
import os, sys, json, datetime, argparse, subprocess
from typing import Any, Dict, List, Iterable, Tuple

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

KW_JSON_PATH = qj_rel("data","selected_keyword_articles.json")
KW_JSON_ALT  = qj_rel("selected_keyword_articles.json")  # 대체 위치

def _load_kw() -> Tuple[str, Any]:
    for p in (KW_JSON_PATH, KW_JSON_ALT):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return p, json.load(f)
    raise SystemExit("[X] selected_keyword_articles.json 파일을 찾지 못했습니다.")

def _iter_articles(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("articles"), list): return payload["articles"]
        if isinstance(payload.get("items"), list):    return payload["items"]
    if isinstance(payload, list):
        return payload
    return []

def _score(a: Dict[str, Any]) -> float:
    # 가능한 필드들을 조합해서 안전하게 점수 계산(없으면 0)
    def g(*keys, default=0.0):
        for k in keys:
            v = a.get(k)
            if isinstance(v, (int,float)): return float(v)
            try:
                return float(str(v))
            except Exception:
                pass
        return float(default)
    base   = g("score", default=0)
    up     = g("upvotes", "points", default=0)
    views  = g("views", default=0) / 10000.0
    kwhit  = g("kw_hits", default=0)
    trust  = g("trust_score", default=0)
    # 가중치 합산(단순/보수적)
    return base*5 + up*2 + views*1 + kwhit*2 + trust*1

def _unique_key(a: Dict[str, Any]) -> str:
    for k in ("url","link","guid","id"):
        v = a.get(k)
        if v:
            s = str(v).strip()
            if s: return s
    return str(a.get("title","")).strip()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20, help="자동 승인 개수 (기본 20)")
    parser.add_argument("--dry-run", action="store_true", help="파일 저장 없이 시뮬레이션만")
    parser.add_argument("--no-sync", action="store_true", help="승인 후 동기화 스크립트 호출 안 함")
    args = parser.parse_args()

    path, payload = _load_kw()
    arts = list(_iter_articles(payload))
    if not arts:
        print("[!] 기사 없음")
        return 0

    # 정렬 및 상위 N 선정
    ranked = sorted(arts, key=_score, reverse=True)
    topN   = ranked[: max(1, args.top)]

    # 승인 처리
    top_keys = set(_unique_key(a) for a in topN)
    approved_cnt = 0
    today = datetime.date.today().strftime("%Y-%m-%d")
    for a in arts:
        key = _unique_key(a)
        if key in top_keys:
            a["approved"] = True
            a.setdefault("state", "ready")
            a.setdefault("date",  today)
            approved_cnt += 1

    # 표준 형태로 저장
    out = {"date": today, "items": arts, "articles": arts}
    if args.dry_run:
        print(f"[DRY-RUN] approve top-{args.top}: {approved_cnt}건")
        return 0

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {approved_cnt}건 승인 완료 -> {path}")

    # 승인 후 발행본 재생성(동기화)
    if not args.no_sync:
        sync_py = qj_rel("tools", "sync_selected_for_publish.py")
        if os.path.exists(sync_py):
            print("[i] sync_selected_for_publish.py 실행…")
            subprocess.run([sys.executable, sync_py], check=False)
        else:
            print("[!] sync_selected_for_publish.py를 찾지 못했습니다. 수동 실행하세요.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
