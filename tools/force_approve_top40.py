from qj_paths import rel as qj_rel
# tools/force_approve_top40.py
# 紐⑹쟻: data/selected_keyword_articles.json (dict+articles)?먯꽌
# ?먯닔? 臾닿??섍쾶 ?곸쐞 40媛쒕? approved=True濡?媛뺤젣 ?쒓린

import json, os, sys

BASE = os.getcwd()
KW = os.path.join(BASE, "data", "selected_keyword_articles.json")

def _load_articles():
    if not os.path.exists(KW):
        sys.exit(f"[X] ?뚯씪 ?놁쓬: {KW}")
    data = json.load(open(KW, "r", encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("articles"), list):
        return data, data["articles"]
    if isinstance(data, list):
        # 鍮꾪몴以??寃쎌슦???덉쟾?섍쾶 泥섎━
        return {"articles": data}, data
    return {"articles": []}, []

def main():
    data, arts = _load_articles()
    # ?곸쐞 40媛?媛뺤젣 ?뱀씤(?뺣젹 ?놁씠 ?욎뿉?쒕???40; ?덉쟾?섍쾶 len 怨좊젮)
    N = min(40, len(arts))
    for i, a in enumerate(arts):
        if i < N:
            a["approved"] = True
    # ???
    with open(KW, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    ok = sum(1 for a in arts if a.get("approved") is True)
    print(f"??媛뺤젣 ?뱀씤 ?꾨즺: approved={ok} / total={len(arts)} (紐⑺몴??5)")

if __name__ == "__main__":
    main()

