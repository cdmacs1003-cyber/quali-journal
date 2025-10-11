from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
# (1) __future__??諛섎뱶??理쒖긽??
from __future__ import annotations

# (2) ?쇰컲 import
import os, sys, json, argparse, datetime, glob

# (3) ?곸쐞 ?대뜑瑜?sys.path??異붽?(?뚯씪 ?ㅽ뻾 ??'tools' ?몄떇)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# (4) ?대? 紐⑤뱢 import
from tools.classify_and_rank import load_rules, rank_articles, classify_sections

"""
make_daily_report.py ??D3: ?쇱씪 由ы룷???앹꽦湲??ы쁽???먭?)
- selected_articles.json / archive/*_KEYWORD.json ???쎌뼱
  ?먯닔/?뱀뀡 遺꾨쪟 ??Markdown 由ы룷?몃? ?앹꽦?⑸땲??

?ъ슜踰???:
    python tools/make_daily_report.py --date 2025-10-04 --keyword "IPC-A-610"
異쒕젰:
    archive/reports/2025-10-04_report.md
"""

def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))

def detect_archive_paths(date_str, keyword=None):
    arc = "archive"
    os.makedirs(os.path.join(arc, "reports"), exist_ok=True)
    kw_json = None
    if keyword:
        patt = os.path.join(arc, f"{date_str}_{keyword.replace(' ','-')}.json")
        if os.path.exists(patt):
            kw_json = patt
    return arc, kw_json

def to_md_table(items, max_n=10):
    out = []
    for i, it in enumerate(items[:max_n], 1):
        t = it.get("title","").replace("|","竊?)
        u = it.get("url","")
        host = it.get("_host","")
        sc = it.get("_rank_score",0)
        out.append(f"{i}. [{t}]({u}) ??{host} 쨌 score {sc}")
    return "\n".join(out) if out else "_(no items)_"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=False, default=datetime.date.today().isoformat())
    ap.add_argument("--keyword", required=False, default=None)
    ap.add_argument("--selection", required=False, default="selected_articles.json")
    args = ap.parse_args()

    rules = load_rules(".")

    # 1) 醫낇빀 selection 湲곗? 由ы룷??    try:
        sel = load_json(args.selection)
        base_items = sel.get("articles") or sel.get("sections",{}).get("?댁뒪",[]) or []
    except Exception:
        base_items = []

    ranked = rank_articles(base_items, rules) if base_items else []
    sections = classify_sections(ranked, rules) if ranked else {"?쒖? ?댁뒪":[], "湲濡쒕쾶 ?꾩옄?앹궛":[], "AI ?댁뒪":[], "而ㅻ??덊떚":[]}

    # 2) ?ㅼ썙??由ы룷??    arc_dir, kw_path = detect_archive_paths(args.date, args.keyword)
    kw_items = []
    if kw_path and os.path.exists(kw_path):
        try:
            kj = load_json(kw_path)
            kw_items = kj.get("sections",{}).get("news",[]) or kj.get("news",[]) or []
            kw_items = rank_articles(kw_items, rules)
        except Exception:
            kw_items = []

    # 3) Markdown 異쒕젰
    md = []
    md.append(f"# ?쇱씪 由ы룷????{args.date}")
    if args.keyword:
        md.append(f"## ?ㅼ썙???뱀쭛: {args.keyword}")
        md.append(to_md_table(kw_items, 15))
    md.append("## 醫낇빀 ???쒖? ?댁뒪")
    md.append(to_md_table(sections.get("?쒖? ?댁뒪",[]), 10))
    md.append("## 醫낇빀 ??湲濡쒕쾶 ?꾩옄?앹궛")
    md.append(to_md_table(sections.get("湲濡쒕쾶 ?꾩옄?앹궛",[]), 10))
    md.append("## 醫낇빀 ??AI ?댁뒪")
    md.append(to_md_table(sections.get("AI ?댁뒪",[]), 10))
    md.append("## 醫낇빀 ??而ㅻ??덊떚")
    md.append(to_md_table(sections.get("而ㅻ??덊떚",[]), 10))

    out_dir = os.path.join(arc_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.date}_report.md")
    open(out_path, "w", encoding="utf-8").write("\n\n".join(md))
    print(out_path)

if __name__ == "__main__":
    main()

