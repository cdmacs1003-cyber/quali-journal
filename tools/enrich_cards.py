# enrich_cards.py (safe ascii)
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime as dt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "archive" / "enriched"
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

    p_all = OUTDIR / ("%s_%s_all.md" % (date, kw))
    p_sel = OUTDIR / ("%s_%s_selected.md" % (date, kw))

    p_all.write_text("# Keyword Enrich - %s - %s\n\n- Auto draft.\n" % (date, kw), encoding="utf-8")
    p_sel.write_text("# Selection Enrich - %s - %s\n\n- Auto draft.\n" % (date, kw), encoding="utf-8")

    print(json.dumps({"ok": True, "path_all": str(p_all.relative_to(ROOT)), "path_selected": str(p_sel.relative_to(ROOT))}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())