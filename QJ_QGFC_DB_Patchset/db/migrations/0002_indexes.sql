
-- Indexes & performance tweaks
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_score ON articles(score DESC);
CREATE INDEX IF NOT EXISTS idx_editions_comp ON editions(etype, edate, keyword);
CREATE INDEX IF NOT EXISTS idx_ea_state ON edition_articles(edition_id, state);
CREATE INDEX IF NOT EXISTS idx_ea_approved ON edition_articles(edition_id, approved);
