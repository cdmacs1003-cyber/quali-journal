from qj_paths import rel as qj_rel
# tools/augment_from_official_pool.py
# 紐⑹쟻:
# - data/selected_keyword_articles.json(dict+articles) ?묒뾽蹂몄쓣 湲곗??쇰줈
#   selected_articles.json(dict+articles)?먯꽌 'IPC-A-610' 留ㅼ묶 湲곗궗瑜?蹂댁땐
# - 珥?20媛??댁긽 梨꾩슦怨? ?뱀씤(approved)??15媛??댁긽 蹂댁옣
# - 蹂댁땐 ??ぉ?먮뒗 approved=True, selected=True ?숈떆 ?명똿
import os, json, re, sys

BASE     = os.getcwd()
KWPATH   = os.path.join(BASE, "data", "selected_keyword_articles.json")
OFFICIAL = os.path.join(BASE, "selected_articles.json")

# IPC-A-610 蹂??留ㅼ묶 (怨듬갚/?섏씠???좎뿰 泥섎━)
RX = re.compile(r"\bIPC\s*[- ]?\s*A\s*[- ]?\s*610\b", re.I)

def _load_articles(p):
    if not os.path.exists(p):
        return []
    j = json.load(open(p, "r", encoding="utf-8"))
    if isinstance(j, dict) and isinstance(j.get("articles"), list):
        return j["articles"]
    if isinstance(j, list):
        return j
    return []

def _save_kw(arts):
    data = {"keyword": "IPC-A-610", "articles": arts}
    json.dump(data, open(KWPATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def _url(a):
    return (a.get("url") or a.get("link") or a.get("guid") or a.get("id") or "").strip()

def main():
    base = _load_articles(KWPATH)
    off  = _load_articles(OFFICIAL)

    base_urls = set(_url(a) for a in base if _url(a))

    # 怨듭떇 ??먯꽌 ?ㅼ썙??留ㅼ묶 + 以묐났 ?쒓굅
    pool = []
    for a in off:
        u = _url(a)
        if (not u) or (u in base_urls):
            continue
        text = " ".join([str(a.get("title","")), str(a.get("desc","")), str(a.get("summary",""))])
        if RX.search(text):
            pool.append(a)

    # 珥?20媛쒓퉴吏 蹂댁땐(?욎뿉?쒕???
    need  = max(0, 20 - len(base))
    added = 0
    for a in pool:
        if added >= need:
            break
        if "score" not in a: a["score"] = 0
        a["approved"] = True
        a["selected"] = True
        base.append(a)
        base_urls.add(_url(a))
        added += 1

    # ?뱀씤 ?섍? 15 誘몃쭔?대㈃ ?⑥? 誘몄듅?몄쓣 ?뱀씤?쇰줈 梨꾩썙 15 蹂댁옣
    approved = sum(1 for a in base if a.get("approved") is True)
    if approved < 15:
        for a in base:
            if approved >= 15:
                break
            if not a.get("approved"):
                a["approved"] = True
                a["selected"] = True
                approved += 1

    _save_kw(base)
    print("OK augment: added=%d, total=%d, approved=%d (target>=15)" % (added, len(base), approved))

if __name__ == "__main__":
    main()

