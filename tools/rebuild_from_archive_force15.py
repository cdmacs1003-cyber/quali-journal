from qj_paths import rel as qj_rel
# tools/rebuild_from_archive_force15.py
# 紐⑹쟻:
# - archive\YYYY-MM-DD_IPC-A-610.json ?먯꽌 湲곗궗 紐⑸줉???쎌뼱
#   data\selected_keyword_articles.json ??dict+articles ?쒖? 援ъ“濡??ъ옉??
# - approved ?섎? 理쒖냼 20媛쒕줈 蹂댁옣(?곸쐞 ?먯닔 湲곗?)
# - selected ?뚮옒洹몃룄 ?④퍡 ?명똿(UX ?쇨???

import os, json, sys, datetime, re

BASE = os.getcwd()
ARCH = os.path.join(BASE, "archive")
OUT  = os.path.join(BASE, "data", "selected_keyword_articles.json")
KEY  = "IPC-A-610"  # ?꾩슂 ??諛붽퓭???ъ슜

def _flatten_any(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "articles" in obj and isinstance(obj["articles"], list):
            return obj["articles"]
        if "sections" in obj and isinstance(obj["sections"], dict):
            flat=[]
            for v in obj["sections"].values():
                if isinstance(v, list):
                    flat.extend(v)
            return flat
    return []

def _score(a):
    try:
        return float(a.get("score") or a.get("total_score") or 0)
    except Exception:
        return 0.0

def main():
    # ?ㅻ뒛??理쒖떊 ?ㅼ썙???뚯씪 李얘린(archive ?곗꽑)
    cands=[]
    if os.path.exists(ARCH):
        for name in os.listdir(ARCH):
            if name.lower().endswith(".json") and KEY.lower() in name.lower():
                cands.append(os.path.join(ARCH, name))
    for name in os.listdir(BASE):
        if name.lower().endswith(".json") and KEY.lower() in name.lower():
            cands.append(os.path.join(BASE, name))
    if not cands:
        sys.exit("[X] ?꾩뭅?대툕/猷⑦듃?먯꽌 ?ㅼ썙??JSON??李얠? 紐삵븿")

    cands.sort(reverse=True)  # 理쒖떊 ?곗꽑
    src, items = None, None
    for p in cands:
        try:
            raw = json.load(open(p, "r", encoding="utf-8"))
            arts = _flatten_any(raw)
            if arts:
                src, items = p, arts
                break
        except Exception:
            continue
    if items is None:
        sys.exit("[X] ?됲깂??媛?ν븳 湲곗궗 由ъ뒪?멸? ?놁쓬")

    # ?먯닔 ?대┝李⑥닚 ?뺣젹
    items.sort(key=_score, reverse=True)

    # approved/selected 蹂댁옣: ?곸쐞 20媛쒖뿉 True
    for i, a in enumerate(items[:20]):
        a["approved"] = True
        a["selected"] = True

    # ?쒖? 援ъ“濡????
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {
        "keyword": KEY,
        "date": datetime.date.today().isoformat(),
        "articles": items
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ok = sum(1 for a in items if a.get("approved") is True)
    print(f"???ш뎄???꾨즺 from: {src}")
    print(f"????? {OUT} (dict+articles, total={len(items)})")
    print(f"???꾩옱 ?뱀씤 ??approved): {ok}嫄?(紐⑺몴??5)")
if __name__ == "__main__":
    main()

