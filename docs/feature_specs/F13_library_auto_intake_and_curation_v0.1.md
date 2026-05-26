# F13 Library Auto Intake and Curation v0.1

## Scope

This specification restores the static F13 Bridge/F13 planning surface for the clean worktree. It defines a no-DB Bridge boundary for Skillup-facing evidence retrieval, policy checks, and trace explanation.

## Boundary

- Bridge/F13 routes must consume caller-provided evidence only.
- Bridge/F13 routes must not query databases, warehouses, library indexes, files, networks, or runtime services.
- Bridge functional 200 behavior remains NOT_VERIFIED until a separate runtime gate is approved.
- Runtime smoke, authenticated smoke, Track A approval, Beta approval, F13 approval, and release approval remain NOT_GRANTED.

## Required Route Surface

- `/api/f13/bridge/retrieve-evidence`
- `/api/f13/bridge/check-policy`
- `/api/f13/bridge/explain-trace`

## Evidence Contract

Evidence returned across the Bridge boundary must contain safe pointer metadata only:

- `evidence_id`
- `bridge_trace_id`
- `safe_summary`
- `pointer_uri`
- `raw_text_policy`
- `rights_status`
- optional `source_doc_kind`
- optional `validation_shape_ids`

Bridge responses must keep `raw_text_included=false` and `internal_path_included=false`.

## Guard Contract

The guard must fail closed for:

- raw text markers
- internal path markers
- secret or token markers
- DB or DSN markers
- missing evidence identifiers
- missing safe summaries
- restricted rights
- unverified rights or raw text policy
- schema/model mismatch

The guard must never return the offending value in an external response.

## Deferred Items

- `reports/f13/`
- `docs/f13/`
- runtime smoke
- authenticated functional smoke
- Bridge functional 200 verification
