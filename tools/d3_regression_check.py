from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, json, argparse, importlib.util

# 1) ?⑦궎吏 寃쎈줈 ?곗꽑 異붽?(?뺤긽 耳?댁뒪)
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

def _load_tools_fallback():
    """
    tools.classify_and_rank ?꾪룷?멸? ?ㅽ뙣?섎㈃
    tools/classify_and_rank.py ?뚯씪??吏곸젒 濡쒕뱶(?먭? 濡쒕뵫)?쒕떎.
    """
    mod_name = "classify_and_rank_fallback"
    target = os.path.join(PARENT, "tools", "classify_and_rank.py")
    if not os.path.exists(target):
        raise ModuleNotFoundError("tools.classify_and_rank not found and no file at " + target)
    spec = importlib.util.spec_from_file_location(mod_name, target)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "invalid spec for classify_and_rank.py"
    spec.loader.exec_module(mod)  # type: ignore
    return mod

try:
    from tools.classify_and_rank import load_rules, rank_articles, classify_sections
except Exception:
    _mod = _load_tools_fallback()
    load_rules = _mod.load_rules
    rank_articles = _mod.rank_articles
    classify_sections = _mod.classify_sections

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser(
        description="D3 ?뚭? 泥댄겕 ???좎젙蹂??ㅼ썙???뱀쭛 理쒖냼 ?덉쭏 ?먭?"
    )
    ap.add_argument("--date", required=True, help="?? 2025-10-04")
    ap.add_argument("--keyword", default=None, help='?? "IPC-A-610"')
    ap.add_argument("--selection", default="selected_articles.json", help="醫낇빀 ?좎젙蹂?JSON) 寃쎈줈")
    ap.add_argument("--min-total", type=int, default=10, help="?좎젙蹂?理쒖냼 媛쒖닔 寃쎈낫 湲곗?(湲곕낯 10)")
    ap.add_argument("--min-keyword", type=int, default=10, help="?ㅼ썙???뱀쭛 理쒖냼 媛쒖닔 寃쎈낫 湲곗?(湲곕낯 10)")
    args = ap.parse_args()

    rules = load_rules(PARENT)

    # --- 1) 醫낇빀 ?좎젙蹂?寃??---
    sel = _load_json(args.selection)
    items = sel.get("articles") or []
    ranked = rank_articles(items, rules) if items else []
    sections = classify_sections(ranked, rules) if ranked else {}

    ok = True
    if len(items) < args.min_total:
        print(f"[WARN] ?좎젙蹂?珥앸웾 ??쓬: {len(items)} < {args.min_total}")
        ok = False
    if sections and len(sections.get("?쒖? ?댁뒪", [])) == 0:
        print("[WARN] ?쒖? ?댁뒪 ?뱀뀡??鍮꾩뿀?듬땲??")
        ok = False

    # --- 2) ?ㅼ썙???뱀쭛 寃??---
    if args.keyword:
        arc_path = os.path.join("archive", f"{args.date}_{args.keyword.replace(' ','-')}.json")
        kj = _load_json(arc_path)
        kitems = (
            kj.get("sections", {}).get("news", [])
            or kj.get("news", [])
            or kj.get("articles", [])
            or []
        )
        if len(kitems) < args.min_keyword:
            print(f"[WARN] ?ㅼ썙???뱀쭛 嫄댁닔 ??쓬: {len(kitems)} < {args.min_keyword}")
            ok = False

    # --- 寃곌낵 ---
    if ok:
        print("[OK] 湲곕낯 ?뚭? 泥댄겕 ?듦낵")
        raise SystemExit(0)
    else:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

