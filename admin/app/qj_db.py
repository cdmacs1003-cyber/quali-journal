# app/qj_db.py
# SQLAlchemy ORM + Session Factory for QualiJournal
from __future__ import annotations
import os
from datetime import datetime, date
from typing import Optional, Iterable, Dict, Any

from sqlalchemy import (
    create_engine, ForeignKey, func, Enum, text, select, String, Integer, Boolean, Date, DateTime, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid

DB_URL = os.getenv("QJ_DB_URL", "sqlite:///./qj.sqlite3")
ECHO = os.getenv("QJ_DB_ECHO", "").lower() in ("1","true","yes")

class Base(DeclarativeBase):
    pass

def _uuid_col():
    if DB_URL.startswith("postgresql"):
        return mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    return mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    domain: Mapped[str] = mapped_column(String, unique=True)
    category: Mapped[str] = mapped_column(String, default="official")
    trust_score: Mapped[float] = mapped_column(default=0.5)

    # ✅ Postgres=JSON(dict), SQLite=Text("{}" 문자열)
    if DB_URL.startswith("postgresql"):
        meta: Mapped[dict] = mapped_column(JSON, default=dict)
    else:
        meta: Mapped[str] = mapped_column(Text, default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Edition(Base):
    __tablename__ = "editions"
    id: Mapped[str] = _uuid_col()
    etype: Mapped[str] = mapped_column(String, default="daily") # daily/keyword/community
    edate: Mapped[date] = mapped_column(Date)
    keyword: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    gate_required: Mapped[int] = mapped_column(Integer, default=15)
    status: Mapped[str] = mapped_column(String, default="draft") # draft/ready/published
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Article(Base):
    __tablename__ = "articles"
    id: Mapped[str] = _uuid_col()
    url: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sources.id"))
    source_domain: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    lang: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    keyword: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    upvotes: Mapped[int] = mapped_column(Integer, default=0)
    views: Mapped[int] = mapped_column(Integer, default=0)
    kw_hits: Mapped[int] = mapped_column(Integer, default=0)
    length: Mapped[int] = mapped_column(Integer, default=0)

    qg_pass: Mapped[bool] = mapped_column(Boolean, default=False)
    fc_pass: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(default=0.0)

    # Postgres는 JSON, SQLite는 Text + 문자열 기본값
    if DB_URL.startswith("postgresql"):
        flags: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    else:
        flags: Mapped[Optional[str]]  = mapped_column(Text, default="{}")


        created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
        updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class EditionArticle(Base):
    __tablename__ = "edition_articles"
    edition_id: Mapped[str] = mapped_column(ForeignKey("editions.id"), primary_key=True)
    article_id: Mapped[str] = mapped_column(ForeignKey("articles.id"), primary_key=True)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    state: Mapped[str] = mapped_column(String, default="candidate")  # candidate/ready/published
    editor_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

engine = create_engine(DB_URL, echo=ECHO, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

def create_all():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def get_or_create_source(sess, domain:str, category:str="official", trust:float=0.5)->Source:
    s = sess.query(Source).filter_by(domain=domain).one_or_none()
    if s: return s
    s = Source(domain=domain, category=category, trust_score=trust)
    sess.add(s); sess.flush()
    return s

def get_or_create_edition(sess, etype:str, edate:date, keyword:Optional[str])->Edition:
    e = (sess.query(Edition).filter(Edition.etype==etype, Edition.edate==edate, Edition.keyword==(keyword or None)).one_or_none())
    if e: return e
    e = Edition(etype=etype, edate=edate, keyword=keyword)
    sess.add(e); sess.flush()
    return e

def kpi_for_edition(sess, edition:Edition)->dict:
    from sqlalchemy import case
    total = sess.query(EditionArticle).filter_by(edition_id=edition.id).count()
    approved = sess.query(EditionArticle).filter_by(edition_id=edition.id, approved=True).count()
    ready = sess.query(EditionArticle).filter_by(edition_id=edition.id, state="ready").count()
    return {
        "total": total,
        "approved": approved,
        "ready": ready,
        "gate_required": edition.gate_required
    }

def approve_top_n(sess, edition:Edition, n:int=20):
    q = (sess.query(Article.id, Article.score)
            .join(EditionArticle, EditionArticle.article_id==Article.id)
            .filter(EditionArticle.edition_id==edition.id, EditionArticle.state.in_(("candidate","ready")))
            .order_by(Article.score.desc())
            .limit(n)
        )
    ids = [row.id for row in q]
    if not ids: return 0
    sess.query(EditionArticle).filter(EditionArticle.edition_id==edition.id, EditionArticle.article_id.in_(ids)).update(
        {"approved": True, "state": "ready"}, synchronize_session=False
    )
    return len(ids)
