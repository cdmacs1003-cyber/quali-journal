from qj_paths import rel as qj_rel
# tools/sync_selected_for_publish.py
# 紐⑹쟻:
# - data/selected_keyword_articles.json??approved=True 湲곗궗留?異붿텧
# - 猷⑦듃 selected_articles.json??dict+articles ?쒖? 援ъ“濡????
import json, os, datetime, sys

BASE = os.getcwd()
KW = os.path.join(BASE, "data", "selected_keyword_articles.json")
PUB_OFFICIAL = os.path.join(BASE, "selected_articles.json")

def _load_kw():
    if not os.path.exists(KW):
        sys.exit(f"[X] ?뚯씪 ?놁쓬: {KW}")
    with open(KW, "r", encoding="utf-8") as f:
        j = json.load(f)
    arts = []
    if isinstance(j, dict) and isinstance(j.get("articles"), list):
        arts = j["articles"]
    elif isinstance(j, list):
        arts = j
    else:
        arts = []
    return arts

def main():
    arts = _load_kw()
    # Extract approved articles and remove duplicates based on URL, GUID or ID.
    approved = []
    seen_keys = set()
    for a in arts:
        if not (isinstance(a, dict) and a.get("approved")):
            continue
        # Determine a unique key for this article: URL preferred, then guid/id.
        key = (
            a.get("url")
            or a.get("link")
            or a.get("guid")
            or a.get("id")
            or None
        )
        # Normalise key to string
        if key:
            key = str(key).strip()
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        approved.append(a)
    out = {
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "articles": approved,
    }
    with open(PUB_OFFICIAL, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # Use plain ASCII characters in the message to avoid encoding errors on consoles without Unicode support
    print(
        f"[OK] 諛쒗뻾???숆린???꾨즺 ??{PUB_OFFICIAL} (?뱀씤 {len(approved)}嫄? dict+articles)"
    )

if __name__ == "__main__":
    main()

