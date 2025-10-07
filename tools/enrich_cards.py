from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, argparse, datetime, time
from typing import Optional
from ._d4_utils import enrich_item, detect_article_list, set_article_list

def _load(p, default=None):
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return default if default is not None else {}

def _backup(path: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.{ts}.bak"
    try:
        import shutil
        shutil.copy2(path, bak)
    except Exception:
        pass
    return bak

def enrich_selection(selection_path: str, max_items: Optional[int] = None) -> str:
    j = _load(selection_path, {"articles":[]})
    keypath, arr = detect_article_list(j)
    out = []
    cnt = 0
    for it in arr:
        out.append(enrich_item(it))
        cnt += 1
        if max_items and cnt >= max_items:
            break
    set_article_list(j, keypath, out + arr[cnt:])  # keep the rest as-is
    _backup(selection_path)
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)
    return selection_path

def enrich_keyword(date: str, keyword: str, archive_dir: str = "archive", max_items: Optional[int] = None) -> str:
    fname = os.path.join(archive_dir, f"{date}_{keyword.replace(' ','-')}.json")
    j = _load(fname, {})
    keypath, arr = detect_article_list(j)
    out = []
    cnt = 0
    for it in arr:
        out.append(enrich_item(it))
        cnt += 1
        if max_items and cnt >= max_items:
            break
    set_article_list(j, keypath, out + arr[cnt:])
    _backup(fname)
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)
    return fname

def main():
    ap = argparse.ArgumentParser(description="D4 ???붿빟/踰덉뿭(?좏깮)?쇰줈 移대뱶 ?덉쭏 ?μ긽")
    ap.add_argument("--selection", default=None, help="selected_articles.json 寃쎈줈")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--keyword", default=None, help='?? "IPC-A-610"')
    ap.add_argument("--max", type=int, default=20, help="泥섎━??理쒕? 移대뱶 ??湲곕낯 20)")
    args = ap.parse_args()

    if args.selection:
        path = enrich_selection(args.selection, max_items=args.max)
        print(f"[OK] selection enriched: {path}")
    if args.date and args.keyword:
        path = enrich_keyword(args.date, args.keyword, max_items=args.max)
        print(f"[OK] keyword enriched: {path}")

    if not (args.selection or (args.date and args.keyword)):
        ap.error("?섎굹 ?댁긽 吏???꾩슂: --selection ?먮뒗 (--date + --keyword)")

if __name__ == "__main__":
    main()

