# -*- coding: utf-8 -*-
"""
enrich_cards.py
- 키워드/선정본 요약 파일을 안전하게 생성
- 입력(STDIN/ARGS) 유연 처리, 파일만 만들어 주면 OK
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from datetime import datetime as _dt

ROOT = Path(__file__).resolve().parents[1]    # /app 또는 프로젝트 루트
ENRICH = ROOT / "archive" / "enriched"
ENRICH.mkdir(parents=True, exist_ok=True)

def _read_stdin_json() -> dict:
    try:
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)
    except Exception:
        pass
    return {}

def _clean(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\r"," ").replace("\n"," ").replace("\t"," ")
    return " ".join(s.split())

def main():
    stdin = _read_stdin_json()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--date", dest="date")
    ap.add_argument("--keyword", dest="keyword")
    args, _ = ap.parse_known_args()

    date = stdin.get("date") or stdin.get("ymd") or args.date or _dt.now().strftime("%Y-%m-%d")
    kw   = stdin.get("keyword") or args.keyword or "UNKNOWN"

    date = _clean(date); kw = _clean(kw)

    all_path     = ENRICH / f"{date}_{kw}_all.md"
    selected_path= ENRICH / f"{date}_{kw}_selected.md"

    all_text = f"# Keyword Enrich — {date} — {kw}\n\n- (자동) 키워드 전체 요약 초안 파일입니다.\n"
    sel_text = f"# Selection Enrich — {date} — {kw}\n\n- (자동) 선정본 요약 초안 파일입니다.\n"

    all_path.write_text(all_text, encoding="utf-8")
    selected_path.write_text(sel_text, encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "paths": {
            "ALL": str(all_path.relative_to(ROOT)),
            "SELECTED": str(selected_path.relative_to(ROOT))
        }
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
