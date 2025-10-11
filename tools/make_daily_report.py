# -*- coding: utf-8 -*-
"""
make_daily_report.py
- 안전한 문자열 정리와 유연한 입력(STDIN/ARGS) 지원
- 없으면 빈 틀이라도 보고서 파일을 만들어 OK로 종료
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from datetime import datetime as _dt

ROOT = Path(__file__).resolve().parents[1]   # /app (컨테이너) 또는 프로젝트 루트
ARCHIVE = ROOT / "archive" / "reports"
ARCHIVE.mkdir(parents=True, exist_ok=True)

def _read_stdin_json() -> dict:
    try:
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)
    except Exception:
        pass
    return {}

def _clean(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = s.replace('"', '"')  # 그대로 두되 공백 정리
    return " ".join(s.split())

def main():
    # 1) 입력 받기: stdin JSON 우선  CLI 인자 보조
    stdin = _read_stdin_json()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--date", dest="date")
    ap.add_argument("--keyword", dest="keyword")
    args, _ = ap.parse_known_args()

    date = stdin.get("date") or stdin.get("ymd") or args.date or _dt.now().strftime("%Y-%m-%d")
    kw   = stdin.get("keyword") or args.keyword or "UNKNOWN"

    date = _clean(date)
    kw   = _clean(kw)

    # 2) 본문 구성(데이터가 없어도 최소 보고서 생성)
    title = f"[QualiJournal] Daily Report — {date} — {kw}"
    lines = [
        f"# {title}",
        "",
        f"- Date   : {date}",
        f"- Keyword: {kw}",
        "",
        "## Summary",
        "- (자동 생성) 오늘의 요약은 추후 엔진 결과로 대체됩니다.",
        "",
        "## Notes",
        "- 데이터가 없으면 빈 틀을 출력합니다.",
    ]
    out_path = ARCHIVE / f"{date}_{kw}_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    # 3) 성공 메시지(JSON 출력은 서버가 안 읽어도 무해)
    print(json.dumps({"ok": True, "path": str(out_path.relative_to(ROOT))}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
