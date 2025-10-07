# tools/quick_publish_keyword.py — 임시 발행기(안전)
import json, sys, os, datetime as dt
root = os.path.dirname(os.path.abspath(__file__)) + os.sep + ".."
root = os.path.abspath(root)
os.chdir(root)

kw = sys.argv[1] if len(sys.argv) > 1 else "IPC-A-610"
src = "selected_articles.json"  # sync 이후 승인 기사만 있어야 함
if not os.path.exists(src):
    raise SystemExit(f"[publish] {src} not found")

with open(src,"r",encoding="utf-8") as f:
    data = json.load(f)

today = dt.date.today().isoformat()
os.makedirs("archive", exist_ok=True)
base = os.path.join("archive", f"{today}_{kw}")

# 최소 3종 출력
with open(base+".json","w",encoding="utf-8") as f:
    json.dump({"keyword":kw,"articles":data.get("articles",data)}, f, ensure_ascii=False, indent=2)

def to_md(arts):
    out = [f"# {kw} — {today}", ""]
    for i,a in enumerate(arts,1):
        t = a.get("title","(no title)")
        u = a.get("url","")
        out += [f"{i}. [{t}]({u})"]
    return "\n".join(out)

md = to_md(data.get("articles",data))
open(base+".md","w",encoding="utf-8").write(md)
open(base+".html","w",encoding="utf-8").write("<pre>"+md+"</pre>")

print(f"[publish] archive files -> {base}.(json|md|html)")
