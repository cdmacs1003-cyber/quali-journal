from qj_paths import rel as qj_rel
# tools/normalize_and_approve_top20.py
# ?쒖? 援ъ“(dict+articles)濡??뺢퇋??+ ?먯닔 ?곸쐞 20媛?approved=True
import json, os, sys, datetime

BASE = os.getcwd()
DATA_DIR = os.path.join(BASE, "data")
OUT_PATH = os.path.join(DATA_DIR, "selected_keyword_articles.json")
KEYWORD = "IPC-A-610"  # ?꾩슂??諛붽퓭???ъ슜

def _flatten_any(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "sections" in obj and isinstance(obj["sections"], dict):
            items = []
            for lst in obj["sections"].values():
                if isinstance(lst, list):
                    items.extend(lst)
            return items
        if "articles" in obj and isinstance(obj["articles"], list):
            return obj["articles"]
    raise SystemExit("[X] 吏?먰븯吏 ?딅뒗 JSON 援ъ“(list/sections/articles ?꾨떂)")

def _score(x):
    try:
        return float(x.get("score") or x.get("total_score") or 0)
    except Exception:
        return 0.0

def main():
    # ?낅젰 ?꾨낫(?ㅻ뒛 ?ㅼ썙???뚯씪 ?곗꽑, ?놁쑝硫?湲곗〈 ?묒뾽蹂?
    candidates = []
    for name in os.listdir(BASE):
        if name.endswith(".json") and KEYWORD in name:
            candidates.append(os.path.join(BASE, name))
    if os.path.exists(OUT_PATH):
        candidates.insert(0, OUT_PATH)

    if not candidates:
        raise SystemExit("[X] ?낅젰 ?꾨낫 JSON???놁뒿?덈떎.")

    src_path, items = None, None
    last_err = None
    for p in candidates:
        try:
            raw = json.load(open(p, "r", encoding="utf-8"))
            items = _flatten_any(raw)
            src_path = p
            break
        except Exception as e:
            last_err = e
    if items is None:
        raise SystemExit(f"[X] ?뚯떛 ?ㅽ뙣: {last_err}")

    # ?곸쐞 20 ?먮룞 ?뱀씤(approved=True)
    top = sorted(items, key=_score, reverse=True)[:20]
    top_keys = {(x.get("url") or x.get("link") or x.get("guid") or x.get("id") or "").strip() for x in top}
    for x in items:
        u = (x.get("url") or x.get("link") or x.get("guid") or x.get("id") or "").strip()
        if u in top_keys:
            x["approved"] = True

    os.makedirs(DATA_DIR, exist_ok=True)
    out = {
        "keyword": KEYWORD,
        "date": datetime.date.today().isoformat(),
        "articles": items,
    }
    json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ok = sum(1 for a in items if a.get("approved"))
    print(f"???낅젰: {src_path}")
    print(f"????? {OUT_PATH} (dict+articles, {len(items)}嫄?")
    print(f"???꾩옱 ?뱀씤 ??approved): {ok}嫄?(紐⑺몴??5)")

if __name__ == "__main__":
    main()

