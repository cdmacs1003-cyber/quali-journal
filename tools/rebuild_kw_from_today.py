from qj_paths import rel as qj_rel
# tools/rebuild_kw_from_today.py
# 紐⑹쟻:
#  - ?ㅻ뒛/理쒓렐 ?앹꽦???ㅼ썙??JSON(猷⑦듃 ?먮뒗 archive ?대뜑)?????
#    sections/articles/由ъ뒪???대뼡 援ъ“???됲깂????dict+articles ?쒖??쇰줈 ?щ퉴??
#  - approved ?섎? 理쒖냼 20媛쒕줈 蹂댁옣
#
# ?ъ슜踰?
#   python tools\rebuild_kw_from_today.py           # 湲곕낯 ?ㅼ썙??IPC-A-610
#   python tools\rebuild_kw_from_today.py "J-STD-001"   # ?ㅼ썙??吏??

import os, json, sys, datetime, re

BASE = os.getcwd()
ARCH = os.path.join(BASE, "archive")
OUT = os.path.join(BASE, "data", "selected_keyword_articles.json")
KEYWORD = sys.argv[1] if len(sys.argv) > 1 else "IPC-A-610"

def _flatten_any(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "sections" in obj and isinstance(obj["sections"], dict):
            arts = []
            for v in obj["sections"].values():
                if isinstance(v, list):
                    arts.extend(v)
            return arts
        if "articles" in obj and isinstance(obj["articles"], list):
            return obj["articles"]
    return []

def _score(a):
    try:
        return float(a.get("score") or a.get("total_score") or 0)
    except Exception:
        return 0.0

def _candidates():
    pats = [
        # 猷⑦듃
        (BASE, re.compile(rf"\d{{4}}-\d{{2}}-\d{{2}}.*{re.escape(KEYWORD)}.*\.json$", re.I)),
        # archive ?대뜑
        (ARCH, re.compile(rf"\d{{4}}-\d{{2}}-\d{{2}}.*{re.escape(KEYWORD)}.*\.json$", re.I)),
    ]
    cands = []
    for root, rx in pats:
        if not os.path.exists(root):
            continue
        for name in os.listdir(root):
            if rx.search(name):
                cands.append(os.path.join(root, name))
    # 理쒖떊 ?좎쭨/?대쫫 ?곗꽑
    cands.sort(reverse=True)
    return cands

def main():
    cands = _candidates()
    if not cands:
        sys.exit(f"[X] 猷⑦듃/?꾩뭅?대툕??'{KEYWORD}' ?ㅼ썙??JSON???놁뒿?덈떎.")

    src_path, items = None, None
    for p in cands:
        try:
            raw = json.load(open(p, "r", encoding="utf-8"))
            arts = _flatten_any(raw)
            if arts:
                src_path, items = p, arts
                break
        except Exception:
            continue
    if not items:
        sys.exit("[X] ?됲깂??媛?ν븳 湲곗궗 由ъ뒪?몃? 李얠? 紐삵뻽?듬땲??")

    # ?먯닔 湲곗??쇰줈 ?뺣젹 ???곸쐞 20 ?뱀씤 蹂댁옣
    items.sort(key=_score, reverse=True)
    for i, a in enumerate(items[:20]):
        a["approved"] = True

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {
        "keyword": KEYWORD,
        "date": datetime.date.today().isoformat(),
        "articles": items
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ok = sum(1 for a in items if a.get("approved") is True)
    print(f"???щ퉴???꾨즺 from: {src_path}")
    print(f"????? {OUT} (dict+articles, {len(items)}嫄?")
    print(f"???꾩옱 ?뱀씤 ??approved): {ok}嫄?(紐⑺몴??5)")

if __name__ == "__main__":
    main()

