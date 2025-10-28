# app/import_from_json.py
"""
JSON 파일(selected_keyword_articles.json, selected_articles.json)을 DB로 적재.
- 환경변수 QJ_DB_URL 지정(없으면 sqlite:///./qj.sqlite3)
사용:
  python app/import_from_json.py  --work selected_keyword_articles.json --final selected_articles.json --etype keyword --keyword "ipc-a-610" --edate 2025-10-27
"""
from __future__ import annotations
import argparse, json, os
from datetime import date, datetime
from urllib.parse import urlparse

from qj_db import get_session, create_all, get_or_create_edition, get_or_create_source, Article, EditionArticle

def _load_json(path):
    if not path or not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _infer_items(data):
    if not data: return []
    # 다양성 대비: { items:[...]} 또는 { articles:[...]} 또는 [...]
    if isinstance(data, dict):
        for key in ("items", "articles", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    if isinstance(data, list):
        return data
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="selected_keyword_articles.json")
    ap.add_argument("--final", default="selected_articles.json")
    ap.add_argument("--etype", default="keyword", choices=["daily","keyword","community"])
    ap.add_argument("--keyword", default=None)
    ap.add_argument("--edate", default=str(date.today()))
    args = ap.parse_args()

    create_all()
    sess = get_session()
    ed = get_or_create_edition(sess, args.etype, date.fromisoformat(args.edate), args.keyword)

    work = _infer_items(_load_json(args.work))
    fin  = _infer_items(_load_json(args.final))
    final_urls = set((x.get("url") or "").strip() for x in fin if x.get("url"))

    count_new = 0
    for it in work:
        url = (it.get("url") or "").strip()
        if not url: continue
        dom = urlparse(url).netloc.lower()
        src = get_or_create_source(sess, dom, category=("community" if ("reddit" in dom or "forum" in dom) else "official"))
        art = sess.query(Article).filter_by(url=url).one_or_none()
        if not art:
            art = Article(
                url=url,
                title=(it.get("title") or ""),
                summary=(it.get("summary") or it.get("desc")),
                source_id=src.id,
                source_domain=dom,
                published_at=_safe_dt(it.get("published_at") or it.get("date")),
                keyword=args.keyword,
                upvotes=int(it.get("upvotes") or 0),
                views=int(it.get("views") or 0),
                kw_hits=int(it.get("kw_hits") or 0),
                length=int(it.get("length") or 0),
                qg_pass=bool(it.get("qg_pass") or False),
                fc_pass=bool(it.get("fc_pass") or False),
                score=float(it.get("score") or 0.0),
                created_at=datetime.utcnow(),           # ✅ 추가
                updated_at=datetime.utcnow(),           # ✅ 추가
            )

            sess.add(art); sess.flush()
            count_new += 1

        ea = sess.query(EditionArticle).filter_by(edition_id=ed.id, article_id=art.id).one_or_none()
        if not ea:
            ea = EditionArticle(
                edition_id=ed.id,
                article_id=art.id,
                created_at=datetime.utcnow(),           # ✅ 추가
                updated_at=datetime.utcnow(),           # ✅ 추가
            )

            sess.add(ea)

        if url in final_urls:
            ea.approved = True
            ea.state = "ready"

    sess.commit()
    print(f"Imported: work={len(work)} items, new_articles={count_new}, final={len(final_urls)} approved=ready.")

def _safe_dt(s):
    if not s: return None
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00"))
    except Exception:
        return None

if __name__ == "__main__":
    main()
