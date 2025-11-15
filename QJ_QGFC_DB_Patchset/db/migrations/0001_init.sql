
-- QualiJournal DB Schema (PostgreSQL)
-- ENUMs
DO $$ BEGIN
  CREATE TYPE edition_type AS ENUM ('daily','keyword','community');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE item_state AS ENUM ('candidate','ready','published');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE edition_status AS ENUM ('draft','ready','published');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- TABLES
CREATE TABLE IF NOT EXISTS sources (
  id            SERIAL PRIMARY KEY,
  domain        TEXT UNIQUE NOT NULL,
  category      TEXT NOT NULL DEFAULT 'official', -- 'official' | 'community'
  trust_score   NUMERIC NOT NULL DEFAULT 0.5,
  meta          JSONB NOT NULL DEFAULT '{{}}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS editions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  etype         edition_type NOT NULL,
  edate         DATE NOT NULL,
  keyword       TEXT,
  gate_required INT NOT NULL DEFAULT 15,
  status        edition_status NOT NULL DEFAULT 'draft',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (etype, edate, COALESCE(keyword,''))
);

CREATE TABLE IF NOT EXISTS articles (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  url           TEXT UNIQUE NOT NULL,
  title         TEXT NOT NULL,
  summary       TEXT,
  content       TEXT,
  source_id     INT REFERENCES sources(id) ON DELETE SET NULL,
  source_domain TEXT,
  lang          TEXT,
  published_at  TIMESTAMPTZ,
  keyword       TEXT,
  upvotes       INT NOT NULL DEFAULT 0,
  views         INT NOT NULL DEFAULT 0,
  kw_hits       INT NOT NULL DEFAULT 0,
  length        INT NOT NULL DEFAULT 0,
  qg_pass       BOOLEAN NOT NULL DEFAULT FALSE,
  fc_pass       BOOLEAN NOT NULL DEFAULT FALSE,
  score         NUMERIC NOT NULL DEFAULT 0,
  flags         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS edition_articles (
  edition_id    UUID REFERENCES editions(id) ON DELETE CASCADE,
  article_id    UUID REFERENCES articles(id) ON DELETE CASCADE,
  approved      BOOLEAN NOT NULL DEFAULT FALSE,
  state         item_state NOT NULL DEFAULT 'candidate',
  editor_comment TEXT,
  position      INT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (edition_id, article_id)
);
