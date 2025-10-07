from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, sys

DATE = sys.argv[1] if len(sys.argv) > 1 else None
KEY  = sys.argv[2] if len(sys.argv) > 2 else None
if not (DATE and KEY):
    print("?ъ슜踰? python -m tools.promote_keyword_to_selection YYYY-MM-DD \"IPC-A-610\"")
    raise SystemExit(2)

arc = os.path.join("archive", f"{DATE}_{KEY.replace(' ','-')}.json")
sel = "selected_articles.json"

def _load(path, default):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return default

kw = _load(arc, {})
items = kw.get("sections",{}).get("news",[]) or kw.get("news",[]) or kw.get("articles",[]) or []
if not items:
    print("[WARN] ?ㅼ썙???곗텧??鍮꾩뼱 ?덉뒿?덈떎.")
    raise SystemExit(1)

# ?곸쐞 10媛쒕쭔, approved=True濡??щ┝
top = items[:10]
for it in top:
    it["approved"] = True

selj = _load(sel, {})
arr = selj.get("articles", [])
arr = top + arr  # ?욎そ??異붽?
selj["articles"] = arr
with open(sel, "w", encoding="utf-8") as f:
    json.dump(selj, f, ensure_ascii=False, indent=2)
print(f"[OK] {len(top)}嫄댁쓣 {sel}??諛섏쁺 ?꾨즺")

