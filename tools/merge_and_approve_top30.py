from qj_paths import rel as qj_rel
# tools/merge_and_approve_top30.py
# 怨듭떇/而ㅻ??덊떚 ?꾨낫瑜??묒뾽蹂몄뿉 蹂묓빀 + ?먯닔 ?곸쐞 30 approved=True
import json, os, sys, datetime

BASE = os.getcwd()
DATA_DIR = os.path.join(BASE, "data")
KW_PATH = os.path.join(DATA_DIR, "selected_keyword_articles.json")
OFFICIAL = os.path.join(BASE, "selected_articles.json")
COMM = os.path.join(BASE, "selected_community.json")

def _ensure_articles(obj):
    if isinstance(obj, dict) and isinstance(obj.get("articles"), list):
        return obj["articles"]
    if isinstance(obj, dict) and isinstance(obj.get("sections"), dict):
        out=[]
        for v in obj["sections"].values():
            if isinstance(v, list): out.extend(v)
        return out
    if isinstance(obj, list):
        return obj
    return []

def _load(p):
    return json.load(open(p,"r",encoding="utf-8")) if os.path.exists(p) else None

def _url(x):
    return (x.get("url") or x.get("link") or x.get("guid") or x.get("id") or "").strip()

def _score(x):
    try: return float(x.get("score") or x.get("total_score") or 0)
    except: return 0.0

def main():
    base = _load(KW_PATH)
    base_items = _ensure_articles(base) if base else []

    off = _ensure_articles(_load(OFFICIAL)) or []
    com = _ensure_articles(_load(COMM)) or []

    seen = set(_url(x) for x in base_items if _url(x))
    def _append(dst, src):
        added = 0
        for x in src:
            u = _url(x)
            if not u or u in seen: continue
            if "score" not in x: x["score"]=0
            dst.append(x); seen.add(u); added += 1
        return added

    add_off = _append(base_items, off)
    add_com = _append(base_items, com)

    # ?곸쐞 30 approved
    top = sorted(base_items, key=_score, reverse=True)[:30]
    keys = {_url(x) for x in top}
    for x in base_items:
        if _url(x) in keys:
            x["approved"] = True

    out = {"keyword": base.get("keyword") if isinstance(base, dict) else "IPC-A-610",
           "date": datetime.date.today().isoformat(),
           "articles": base_items}
    json.dump(out, open(KW_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    ok = sum(1 for a in base_items if a.get("approved"))
    print(f"??蹂묓빀 ?꾨즺: 怨듭떇+{add_off}, 而ㅻ??덊떚+{add_com}")
    print(f"???꾩옱 ?뱀씤 ??approved): {ok}嫄?/ 紐⑺몴??5")
    print(f"????? {KW_PATH} (珥?{len(base_items)}嫄?")

if __name__ == "__main__":
    main()

