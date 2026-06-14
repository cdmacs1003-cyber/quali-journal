-- R9ZMO_SKILLUP_FEEDBACK_QUEUE_SQLITE_FIXTURE_20260614
-- Test-scoped local disposable SQLite fixture migration design.
-- This artifact is not executed by R9ZMO.
-- Boundary:
-- - no production or shared DB target
-- - no external DSN, credential, token, key, or service-account input
-- - no network database client
-- - minimized durable queue record columns only
-- - no raw text, raw prompt, raw source, internal path, URI, hostname, secret, DSN, token, credential, key, or Bridge raw payload columns

CREATE TABLE IF NOT EXISTS skillup_feedback_queue_items (
    contract_version TEXT NOT NULL,
    persistence_mechanism TEXT NOT NULL CHECK (persistence_mechanism = 'DB_BACKED_QUEUE_DEFERRED'),
    feedback_id TEXT NOT NULL PRIMARY KEY,
    origin_event_id TEXT NOT NULL,
    current_status TEXT NOT NULL CHECK (
        current_status IN ('duplicate', 'queued', 'rejected', 'resolved', 'review_required')
    ),
    dedup_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    review_reason_code TEXT NOT NULL,
    safe_summary TEXT NOT NULL,
    trace_id TEXT,
    request_id TEXT,
    raw_text_included INTEGER NOT NULL DEFAULT 0 CHECK (raw_text_included = 0),
    internal_path_included INTEGER NOT NULL DEFAULT 0 CHECK (internal_path_included = 0),
    db_access_executed INTEGER NOT NULL DEFAULT 0 CHECK (db_access_executed = 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_skillup_feedback_queue_items_dedup_key
    ON skillup_feedback_queue_items (dedup_key);

-- Fixture rollback/cleanup expectation for future approved gates:
-- DROP INDEX IF EXISTS idx_skillup_feedback_queue_items_dedup_key;
-- DROP TABLE IF EXISTS skillup_feedback_queue_items;
