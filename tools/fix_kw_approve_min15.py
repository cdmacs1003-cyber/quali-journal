from qj_paths import rel as qj_rel
# tools/fix_kw_approve_min15.py
# 紐⑹쟻:
#  - data/selected_keyword_articles.json ??'dict + articles' ?쒖? 援ъ“濡?援먯젙
#  - selected=True ?ㅼ쓣 approved=True 濡??밴꺽
#  - approved ?섍? 15 誘몃쭔?대㈃ score ?곸쐞?먯꽌 異붽? ?밴꺽?섏뿬 >=15 蹂댁옣

import json, os, sys, datetime

BASE = os.getcwd()
KW_PATH = os.path.join(BASE, "data", "selected_keyword_articles.json")

def _load():
    if not os.path.exists(KW_PATH):
        sys.exit(f"[X] ?뚯씪 ?놁쓬: {KW_PATH}")
    with open(KW_PATH, "r", encoding="utf-8") as f:
        j = json.load(f)
    # ?쒖??? dict+articles
    if isinstance(j, list):
        arts = j
        data = {"keyword": "IPC-A-610", "date": datetime.date.today().isoformat(), "articles": arts}
    elif isinstance(j, dict):
        # sections?뺤씠硫??됲깂??
        if "articles" in j and isinstance(j["articles"], list):
            data = j
        elif "sections" in j and isinstance(j["sections"], dict):
            arts = []
            for v in j["sections"].values():
                if isinstance(v, list):
                    arts.extend(v)
            data = {"keyword": j.get("keyword") or "IPC-A-610", "date": j.get("date") or datetime.date.today().isoformat(), "articles": arts}
        else:
            data = {"keyword": j.get("keyword") or "IPC-A-610", "date": j.get("date") or datetime.date.today().isoformat(), "articles": []}
    else:
        data = {"keyword": "IPC-A-610", "date": datetime.date.today().isoformat(), "articles": []}
    return data

def _score(a):
    try:
        return float(a.get("score") or a.get("total_score") or 0.0)
    except Exception:
        return 0.0

def main():
    data = _load()
    arts = data.get("articles") or []
    # selected -> approved ?밴꺽
    for a in arts:
        if a.get("selected") is True and not a.get("approved"):
            a["approved"] = True

    # approved < 15 ?대㈃ ?먯닔?곸쐞?먯꽌 異붽? ?밴꺽
    approved_cnt = sum(1 for a in arts if a.get("approved"))
    if approved_cnt < 15:
        need = 15 - approved_cnt
        # ?꾩쭅 誘몄듅?몄씤 寃?以??먯닔 ?곸쐞
        remain = [a for a in arts if not a.get("approved")]
        remain.sort(key=_score, reverse=True)
        for a in remain[:max(0, need)]:
            a["approved"] = True
        approved_cnt = sum(1 for a in arts if a.get("approved"))

    # ???
    with open(KW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"???쒖? 援ъ“ ????꾨즺: dict+articles / ?뱀씤 ?? {approved_cnt}嫄?(紐⑺몴??5)")

if __name__ == "__main__":
    main()

