from qj_paths import rel as qj_rel
# tools/approve_top20.py
# 紐⑹쟻: data/selected_keyword_articles.json?먯꽌 ?먯닔 ?곸쐞 20媛??먮룞 ?뱀씤(selected=true)

import json, os, sys

BASE = os.getcwd()
KW_PATH = os.path.join(BASE, "data", "selected_keyword_articles.json")

def main():
    if not os.path.exists(KW_PATH):
        print(f"[X] ?뚯씪 ?놁쓬: {KW_PATH}")
        sys.exit(1)

    with open(KW_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
        if not isinstance(items, list):
            print("[X] JSON ?뺤떇??紐⑸줉(list)???꾨떂")
            sys.exit(1)

    # ?먯닔 ?곸쐞 20 異붿텧(?먯닔 ?놁쑝硫?0)
    items_sorted = sorted(items, key=lambda x: x.get("score", 0), reverse=True)[:20]
    # ?곸쐞 20媛?selected=true
    top_urls = { (x.get("url") or x.get("link") or "") for x in items_sorted }
    approved = 0
    for x in items:
        u = (x.get("url") or x.get("link") or "")
        if u in top_urls:
            x["selected"] = True
            approved += 1

    with open(KW_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # ?뱀씤 媛쒖닔 吏묎퀎
    ok = sum(1 for x in items if x.get("selected") is True)
    print(f"???먮룞 ?뱀씤 諛섏쁺 ?꾨즺 | ?꾩옱 ?뱀씤 ?? {ok}嫄?)

if __name__ == "__main__":
    main()

