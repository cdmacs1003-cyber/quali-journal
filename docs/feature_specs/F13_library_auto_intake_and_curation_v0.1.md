# F13 Library Auto Intake and Curation v0.1

## Scope

This specification restores the static F13 Bridge/F13 planning surface for the clean worktree. It defines a no-DB Bridge boundary for Skillup-facing evidence retrieval, policy checks, and trace explanation.

## Boundary

- Bridge/F13 routes must consume caller-provided evidence only.
- Bridge/F13 routes must not query databases, warehouses, library indexes, files, networks, or runtime services.
- R9ZNW-265 exception: Bridge may read canonical safe metadata-only Library Evidence seed JSON under
  `data/library/evidence_seeds/` only through the approved Bridge-side seed adapter. General file-backed
  retrieval remains forbidden. Skillup direct DB/file access remains forbidden.
- R9ZNW-271 exception: Bridge may read the canonical safe metadata-only solder-domain JSON concept map under
  `data/library/ontology/` and semantic term registry under `data/library/semantic_terms/` only to implement
  bounded Bridge.resolve_terms before seed retrieval. This is not GraphRAG, RDF, OWL, a triple store, embedding
  retrieval, or LLM extraction. Skillup direct DB/file access remains forbidden.
- Bridge functional 200 behavior remains NOT_VERIFIED until a separate runtime gate is approved.
- Runtime smoke, authenticated smoke, Track A approval, Beta approval, F13 approval, and release approval remain NOT_GRANTED.

## Required Route Surface

- `/api/f13/bridge/retrieve-evidence`
- `/api/f13/bridge/check-policy`
- `/api/f13/bridge/explain-trace`

## Trace Explanation Contract

The explain-trace response must expose a top-level `feedback_candidate_required`
boolean and a top-level `feedback_candidate` value. `feedback_candidate` is
`null` when the trace explanation is `OK`; otherwise it is a safe review object
with bounded reason text and no raw evidence, internal path, secret-like value,
or DB/DSN detail.

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

## Shape Contract

`shapes/f13_bridge_runtime_contract_shape.json` is the canonical
machine-readable Bridge/F13 runtime contract shape. The matching Markdown file
`shapes/f13_bridge_runtime_contract_shape.md` is a human-readable documentation
wrapper and must not replace the JSON shape.

## Evidence Schema Alias Policy

`schemas/f13_bridge_evidence_response.schema.json` is the canonical external
retrieve-evidence schema. `schemas/f13/bridge_evidence_response.schema.json`
is retained as a compatibility alias for recovered static tests; required fields
and properties should remain aligned with the canonical schema.

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
- test execution until a separate explicit gate
