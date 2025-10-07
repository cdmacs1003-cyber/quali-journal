from qj_paths import rel as qj_rel

# -*- coding: utf-8 -*-
"""
cleanup_legacy.py ??援щ쾭??以묐났 愿由ъ옄 ?뚯씪 ?뺣━ ?꾩슦誘?(Dry-Run 湲곕낯)
?ъ슜:
  python tools/cleanup_legacy.py              # ??젣 ??紐⑸줉留?異쒕젰
  python tools/cleanup_legacy.py --force      # ?ㅼ젣 ??젣 ?ㅽ뻾
???
  - 猷⑦듃 server_admin.py / admin(誘몃땲) ?대뜑??index.html
  - 猷⑦듃 index.html(誘몃땲) / README.txt(誘몃땲)
  - archive/admin_min, archive/admin ??遺덊븘??愿由ъ옄 ?붿옱
?덉쟾?μ튂:
  - ?꾨줈?앺듃 ?듭떖 ?뚯씪(orchestrator.py, data/, admin_quali/, tools/)? 嫄대뱶由ъ? ?딆쓬
"""
from __future__ import annotations
import os, sys, shutil, argparse, pathlib

ROOT = pathlib.Path(os.getcwd())
TARGETS = [
    ROOT / "server_admin.py",
    ROOT / "index.html",           # mini admin index
    ROOT / "README.txt",           # mini admin readme
    ROOT / "admin",                # mini admin folder (if exists)
    ROOT / "admin_min",            # legacy
    ROOT / "archive" / "admin_min",
    ROOT / "archive" / "admin",
]

SAFE_KEEP = {
    "orchestrator.py", "selected_articles.json", ".env",
    "data", "archive", "tools", "admin_quali"
}

def exists(p): 
    try: return p.exists()
    except Exception: return False

def can_delete(p: pathlib.Path) -> bool:
    # 理쒖긽??以묒슂 ?대뜑??蹂댄샇
    if p.name in SAFE_KEEP: 
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="?ㅼ젣 ??젣 ?ㅽ뻾")
    args = ap.parse_args()
    to_remove = [p for p in TARGETS if exists(p) and can_delete(p)]
    print("?㏏ ?뺣━ ???")
    for p in to_remove:
        print(" -", p)
    if not args.force:
        print("\n(?쒕씪?대윴) ?ㅼ젣 ??젣?섎젮硫?--force ?듭뀡??異붽??섏꽭??")
        return
    for p in to_remove:
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
            print("  ????젣:", p)
        except Exception as e:
            print("  ???ㅽ뙣:", p, e)

if __name__ == "__main__":
    main()

