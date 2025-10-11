# make_daily_report.py (safe ascii)
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime as dt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "archive" / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)

def read_stdin_json():
    try:
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)
    except Exception:
        pass
    return {}

def val(x: str) -> str:
    if x is None: return ""
    s = str(x).replace("\r"," ").replace("\n"," ").replace("\t"," ")
    return " ".join(s.split())

def main():
    body = read_stdin_json()
    date = val(body.get("date") or body.get("ymd") or dt.now().strftime("%Y-%m-%d"))
    kw   = val(body.get("keyword") or "UNKNOWN")

    title = "[QualiJournal] Daily Report - %s - %s" % (date, kw)
    lines = [
        "# " + title, "",
        "- Date: " + date,
        "- Keyword: " + kw, "",
        "## Summary", "- Auto draft.", "",
        "## Notes", "- This file is generated even if data is empty."
    ]
    out = OUTDIR / ("%s_%s_report.md" % (date, kw))
    out.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"ok": True, "path": str(out.relative_to(ROOT))}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())