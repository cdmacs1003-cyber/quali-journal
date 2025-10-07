from qj_paths import rel as qj_rel

# -*- coding: utf-8 -*-
"""
D3 ?뚭? 泥댄겕 ??媛꾩씠 ?뚯뒪??- ?좎젙蹂?selected_articles.json)怨??ㅼ썙???곗텧臾?archive/*_KEYWORD.json) 湲곗큹 ?덉쭏 ?뺤씤
?ㅽ뻾:
    python tests/d3_regression_check.py --date 2025-10-04 --keyword "IPC-A-610"
"""
from __future__ import annotations
import os, json, argparse, sys
from tools.classify_and_rank import load_rules, rank_articles, classify_sections

def loadj(p):
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--keyword", required=False, default=None)
    ap.add_argument("--selection", default="selected_articles.json")
    args = ap.parse_args()

    rules = load_rules(".")
    sel = loadj(args.selection)
    items = sel.get("articles") or []
    ranked = rank_articles(items, rules) if items else []
    sections = classify_sections(ranked, rules) if ranked else {}

    ok = True
    # 1) 理쒖냼 媛쒖닔 寃쎈낫(?덈Т ??쑝硫??뚮┝)
    if len(items) < 10:
        print(f"[WARN] selected_articles 珥앸웾????뒿?덈떎: {len(items)} < 10")
        ok = False
    # 2) ?쒖? ?댁뒪 理쒖냼 1媛?湲곕?
    if sections and len(sections.get("?쒖? ?댁뒪",[])) == 0:
        print("[WARN] ?쒖? ?댁뒪 ?뱀뀡??鍮꾩뿀?듬땲??")
        ok = False

    # 3) ?ㅼ썙???뱀쭛 寃??    if args.keyword:
        kpath = os.path.join("archive", f"{args.date}_{args.keyword.replace(' ','-')}.json")
        kj = loadj(kpath)
        kitems = kj.get("sections",{}).get("news",[]) or kj.get("news",[]) or []
        if len(kitems) < 10:
            print(f"[WARN] ?ㅼ썙???뱀쭛 嫄댁닔媛 ??뒿?덈떎: {len(kitems)} < 10")
            ok = False

    if ok:
        print("[OK] 湲곕낯 ?뚭? 泥댄겕 ?듦낵")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

