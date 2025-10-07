from qj_paths import rel as qj_rel
# tools/force_set_selected_and_approved_top20.py
# data/selected_keyword_articles.json ??dict+articles 援ъ“?쇨퀬 媛??
# ?곸쐞 20媛쒖뿉 approved=True, selected=True 瑜?紐⑤몢 ?명똿.

import json, os, sys

BASE = os.getcwd()
KW = os.path.join(BASE, "data", "selected_keyword_articles.json")

def score(a):
    try:
        return float(a.get("score") or a.get("total_score") or 0)
    except:
        return 0.0

def main():
    if not os.path.exists(KW):
        sys.exit(f"[X] ?뚯씪 ?놁쓬: {KW}")
    data = json.load(open(KW, "r", encoding="utf-8"))
    arts = data["articles"] if isinstance(data, dict) else data

    # ?먯닔 ?대┝李⑥닚 ?뺣젹 ???곸쐞 20媛쒖뿉 ??源껊컻 紐⑤몢 ?명똿
    arts_sorted = sorted(arts, key=score, reverse=True)
    for i, a in enumerate(arts_sorted):
        if i < 20:
            a["approved"] = True
            a["selected"] = True

    json.dump(data if isinstance(data, dict) else arts_sorted,
              open(KW, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    ap = sum(1 for x in arts_sorted if x.get("approved") is True)
    se = sum(1 for x in arts_sorted if x.get("selected") is True)
    print(f"??媛뺤젣 ?명똿 ?꾨즺: approved={ap}, selected={se}, total={len(arts_sorted)}")

if __name__ == "__main__":
    main()

