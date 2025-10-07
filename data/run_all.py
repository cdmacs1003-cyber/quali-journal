import argparse, json, re, sys, os
from pathlib import Path
from collections import Counter

OK, WARN, FAIL = "PASS", "WARN", "FAIL"

def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, str(e)

def find_first(repo: Path, cands):
    for rel in cands:
        p = repo/rel
        if p.exists():
            return p
    return None

def check1(repo: Path):
    cfg_p = find_first(repo, ["config.json"])
    sch_p = find_first(repo, ["config_schema_definition.json","config_schema.json"])
    if not cfg_p:
        return FAIL, "config.json 없음"
    cfg, e = load_json(cfg_p)
    if e: return FAIL, f"config.json 읽기 오류: {e}"
    gr = cfg.get("gate_required")
    rea = (cfg.get("features") or {}).get("require_editor_approval")
    s = []
    st = OK
    if not isinstance(gr, int): st = FAIL; s.append("gate_required 정수 아님")
    if not isinstance(rea, bool): st = FAIL; s.append("features.require_editor_approval 불리언 아님")
    if sch_p and sch_p.exists():
        txt = sch_p.read_text(encoding="utf-8", errors="ignore")
        if "gate_required" not in txt:
            st = FAIL; s.append("스키마에 gate_required 누락")
    return st, "; ".join(s) or "OK"

def check2(repo: Path, do_net: bool=False):
    comm = find_first(repo, ["feeds/community_sources.json","community_sources.json"])
    off  = find_first(repo, ["feeds/official_sources.json","official_sources.json"])
    if not comm and not off: return FAIL, "feeds/* 파일 없음"
    msg=[]; st=OK
    if comm:
        c,_ = load_json(comm)
        if not c: st=FAIL; msg.append("community_sources.json 파싱 오류")
        else:
            text=comm.read_text(encoding="utf-8", errors="ignore")
            if not re.search(r"min_upvotes|min_title_len|score_threshold", text):
                st=FAIL; msg.append("커뮤니티 임계값 키 미발견")
    else: st=FAIL; msg.append("community_sources.json 없음")
    if off:
        try: _=off.read_text(encoding="utf-8")
        except Exception as e: st=FAIL; msg.append(f"official_sources.json 읽기 오류:{e}")
    else: st=FAIL; msg.append("official_sources.json 없음")
    return st, "; ".join(msg) or "OK"

def check3(repo: Path):
    orch = find_first(repo, ["orchestrator.py"])
    runpy = find_first(repo, ["run_quali_today.py"])
    if not orch: return FAIL,"orchestrator.py 없음"
    return OK if runpy else WARN, "run_quali_today.py 없음(PS 의존 가능)"

def check4(repo: Path):
    tdir = repo/"tools"
    if not tdir.exists(): return WARN,"tools 폴더 없음"
    names=[p.name for p in tdir.glob("*.py")]
    st=OK; m=[]
    if "repair_selection_files.py" in names and "sync_selected_for_publish.py" in names:
        st=WARN; m.append("sync+repair 중복 가능")
    return st, "; ".join(m) or "OK"

def check5(repo: Path):
    idx = find_first(repo, ["admin/index.html","admin/index.htm"])
    svr = find_first(repo, ["admin/server_quali.py","server_quali.py"])
    st=OK; m=[]
    if not idx: st=WARN; m.append("index.html 없음")
    else:
        h=idx.read_text(encoding="utf-8", errors="ignore").lower()
        if "charset" not in h: st=WARN; m.append("인코딩 메타 누락")
    if not svr: st=WARN; m.append("server_quali.py 없음")
    return st, "; ".join(m) or "OK"

def _articles(p):
    j,_=load_json(p)
    return (j or {}).get("articles") if isinstance(j,dict) else None

def check6(repo: Path):
    kw = find_first(repo, ["selected_keyword_articles.json","data/selected_keyword_articles.json"])
    sel= find_first(repo, ["selected_articles.json","data/selected_articles.json"])
    if not kw and not sel: return WARN,"selected_* 파일 없음"
    st=OK; m=[]
    if kw:
        arts=_articles(kw); 
        if not isinstance(arts,list): return FAIL,"selected_keyword_articles 구조 비표준"
        urls=[a.get("url") for a in arts if isinstance(a,dict)]
        if any(c>1 for c in Counter(urls).values()): st=WARN; m.append("작업본 중복 URL 발견")
    if sel:
        arts=_articles(sel)
        if not isinstance(arts,list): return FAIL,"selected_articles 구조 비표준"
    return st, "; ".join(m) or "OK"

def check7(repo: Path):
    arc=repo/"archive"
    if not arc.exists(): return WARN,"archive 없음"
    names=[p.name for p in arc.iterdir() if p.is_file()]
    # 타임스탬프 흔적이 하나라도 있으면 PASS
    if any(re.search(r"_\d{4,6}\.",n) for n in names): return OK,"timestamp 발견"
    # 아니면 베이스 3종(md/html/json) 존재만 확인
    return OK if any(n.endswith(".html") for n in names) else WARN, "재발행 대비 권장"

def check8(repo: Path):
    req=find_first(repo,["requirements.txt"])
    if not req: return FAIL,"requirements.txt 없음"
    t=req.read_text(encoding="utf-8", errors="ignore").lower()
    need=["fastapi","uvicorn","pydantic","praw","feedparser","beautifulsoup4","lxml","requests"]
    miss=[n for n in need if n not in t]
    return OK if not miss else WARN, "누락: "+", ".join(miss) if miss else "OK"

def check9(repo: Path):
    files=list(repo.rglob("*.py"))
    bad=0
    for f in files:
        t=f.read_text(encoding="utf-8", errors="ignore")
        if re.search(r'open\\(\\s*"(config\\.json|admin/index\\.html)"', t):
            bad+=1; break
    return OK if bad==0 else FAIL, "하드코딩 open(\"config.json\"|\"admin/index.html\") 발견" if bad else "OK"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--net", action="store_true")
    args=ap.parse_args()
    repo=Path(args.repo).resolve()
    checks=[
        ("1. config.json  schema", check1),
        ("2. community/official 소스", lambda r: check2(r, args.net)),
        ("3. orchestrator & run 스크립트", check3),
        ("4. tools 중복/역할", check4),
        ("5. admin/index.html  server_quali.py", check5),
        ("6. selected_* 무결성", check6),
        ("7. archive 파일명 규칙", check7),
        ("8. requirements 의존성", check8),
        ("9. 경로/모듈 호출 체계", check9),
    ]
    lines=["===== SUMMARY ====="]
    for title,fn in checks:
        try:
            st,msg=fn(repo)
        except Exception as e:
            st,msg=FAIL,f"예외: {e}"
        lines.append(f"[{st}] {title}")
    report="\\n".join(lines)+"\\nReport written to 'health_report.md' (in repo or current directory)."
    try:
        (repo/"health_report.md").write_text(report,encoding="utf-8")
    except Exception:
        Path("health_report.md").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()

