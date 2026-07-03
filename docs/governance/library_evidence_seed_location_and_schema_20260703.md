# Library Evidence Seed Location and Schema Binding - 2026-07-03

## Task and Purpose

Task ID:
`R9ZNW-262_APPROVE_MINIMAL_LIBRARY_EVIDENCE_SEED_LOCATION_AND_SCHEMA_FOR_SOLDERING_SAFE_SUMMARY_NO_RUNTIME_NO_DB_NO_BROWSER_NO_HTTP_NO_TESTS`

Purpose:
approve one concrete tracked additive repository location and seed-record schema binding for minimal Library Evidence seed records. This enables a later materialization gate to add a soldering safe-summary seed without creating a local glossary fallback, hardcoded beta answer, or direct Skillup data access path.

This gate answers only:

1. Where minimal Library Evidence seed records live.
2. Which schema binding governs them.
3. Which fields each seed record must contain.
4. Which raw-leak, rights, Bridge, and approval gates block unsafe use.
5. Which later task may materialize the soldering seed.

## Decision

```text
DECISION=APPROVE_MINIMAL_LIBRARY_EVIDENCE_SEED_LOCATION_AND_SCHEMA_WITH_LIMITS
APPROVED_LOCATION=data/library/evidence_seeds/
APPROVED_RECORD_FORMAT=JSON
APPROVED_SCOPE=SAFE_METADATA_ONLY_LIBRARY_EVIDENCE_SEEDS
FIRST_USE_CASE=soldering_safe_summary
RAW_TEXT_POLICY=SAFE_SUMMARY_ONLY_OR_POINTER_ONLY
SKILLUP_DIRECT_ACCESS=FORBIDDEN
BRIDGE_FILE_BACKED_RETRIEVAL=NOT_APPROVED_IN_THIS_GATE
NEXT_GATE=R9ZNW-263_MATERIALIZE_SOLDERING_SAFE_SUMMARY_LIBRARY_EVIDENCE_SEED_NO_RUNTIME_NO_DB_NO_BROWSER_NO_HTTP_NO_DEPLOY
```

Approved tracked additive location:
`data/library/evidence_seeds/`

Future seed file pattern:
`data/library/evidence_seeds/<domain>/<evidence_id>.json`

Explicit first-use domain:
soldering / 솔더링 / 납땜 safe summary.

This document does not create a seed record. It does not approve file-backed Bridge retrieval. It does not approve Skillup direct file, DB, or directory access.

## Schema Binding

Minimal Library Evidence seed records under `data/library/evidence_seeds/` are governed by this document as the repository-level seed-location and seed-record binding.

The seed-record binding must remain compatible with the existing Bridge evidence contracts:

- `ImplementationCompletion/F13/schemas/evidence_pointer.schema.json`
- `ImplementationCompletion/F13/schemas/evidence_response.schema.json`
- `schemas/f13_bridge_evidence_response.schema.json`
- `schemas/f13/bridge_evidence_response.schema.json`
- `schemas/skillup_answer_hold_response.schema.json`

Compatibility means a later approved Bridge adapter may derive Bridge response metadata from a seed only through safe fields such as `evidence_id`, generated or derived `bridge_trace_id`, `safe_summary`, `pointer_uri`, `raw_text_policy`, `rights_status`, `source_doc_kind`, and `validation_shape_ids`.

If a future JSON Schema file is added for seed records, it must preserve the blocking rules in this document or be stricter.

## Required Seed Fields

Every seed record must be JSON and contain these fields:

- `schema_version`
- `evidence_id`
- `evidence_type`
- `created_for_query_domain`
- `safe_summary`
- `pointer_uri`
- `raw_text_policy`
- `rights_status`
- `source_doc_kind`
- `provenance_summary`
- `review_status`
- `approval_status`
- `approval_record_id`
- `proofpack_id`
- `bridge_trace_seed`
- `raw_text_excluded`
- `standard_raw_text_not_included`
- `created_at`
- `created_by`
- `validation_shape_ids`

## Allowed Values

`evidence_type`:

- `SAFE_SUMMARY_ONLY`

`raw_text_policy`:

- `SAFE_SUMMARY_ONLY`
- `POINTER_ONLY`

`rights_status`:

- `PUBLIC`
- `INTERNAL`
- `LICENSED`

`UNKNOWN` is forbidden for Skillup use.

`source_doc_kind`:

- `REFERENCE`
- `WAREHOUSE_ITEM`
- `FIELD_CASE`
- `QA_CASE`
- `STANDARD_POINTER_ONLY`

`review_status`:

- `DRAFT`
- `REVIEW_REQUIRED`
- `APPROVED_FOR_LIBRARY_EVIDENCE`
- `HOLD`

`approval_status`:

- `NOT_APPROVED`
- `APPROVED_WITH_LIMITS`
- `HOLD`

## Blocking Rules

These conditions block use of a seed for Skillup-facing safe answers:

- `evidence_id` missing => `HOLD`
- `safe_summary` missing => `HOLD`
- `pointer_uri` missing => `HOLD`
- `raw_text_policy` missing => `HOLD`
- `rights_status` is `UNKNOWN` => `DENIED` or `HOLD`
- `rights_status` outside `PUBLIC`, `INTERNAL`, or `LICENSED` => `DENIED` or `HOLD`
- `review_status` is not `APPROVED_FOR_LIBRARY_EVIDENCE` => `HOLD`
- `approval_status` is not `APPROVED_WITH_LIMITS` => `HOLD`
- `raw_text_excluded` is not `true` => `REJECT`
- `standard_raw_text_not_included` is not `true` => `REJECT`
- `bridge_trace_seed` missing => `HOLD` until a Bridge trace is generated
- `validation_shape_ids` missing or empty => `HOLD`

A seed record that fails any blocking rule must not produce `safe_short_answer`.

## Raw Leak Rules

Seed records must not include:

- paid standard raw text
- long standard quotation
- class-specific acceptance criteria copied from a standard
- raw evidence/source body
- full JSON intended for user display
- internal filesystem paths intended for user display
- tokens, keys, cookies, credentials, DSNs, auth headers, or secret-like values

User-facing answer output may use `safe_summary` only. The UI must not render raw response body, full JSON, raw source text, raw standard text, or internal paths.

## Rights and Approval Gates

`rights_status` must be known and limited to `PUBLIC`, `INTERNAL`, or `LICENSED`.

`approval_status=APPROVED_WITH_LIMITS` and `review_status=APPROVED_FOR_LIBRARY_EVIDENCE` are both required before a seed may be used for a Skillup-facing safe answer.

If provenance, rights, proofpack, or approval evidence is unclear, the seed must remain `HOLD` or `REVIEW_REQUIRED`.

## Bridge Boundary

Skillup must not read `data/library/evidence_seeds/` directly.

Bridge adapter consumption of this location is not approved in this gate. A later approved adapter gate must define any file-backed retrieval behavior, trace generation, filtering, validation, and output mapping.

Any later Bridge adapter must:

- require `evidence_id`
- generate or derive `bridge_trace_id` from `bridge_trace_seed`
- expose only `safe_summary` as answer material
- preserve `raw_text_included=false`
- preserve `internal_path_included=false`
- avoid DB, provider, browser, HTTP, and cloud dependencies unless separately approved
- return `HOLD` when a seed is missing, unapproved, rights-blocked, malformed, or raw-leak unsafe

## First Materialization Candidate

The first later seed may target:

`data/library/evidence_seeds/soldering/ev-soldering-safe-summary-v1.json`

It may support the query domain:
soldering / 솔더링 / 납땜.

That later seed must be safe metadata only and must not be a local glossary fallback or hardcoded beta answer. It must be an approved Library Evidence seed under this binding.

## Next Allowed Task

`R9ZNW-263_MATERIALIZE_SOLDERING_SAFE_SUMMARY_LIBRARY_EVIDENCE_SEED_NO_RUNTIME_NO_DB_NO_BROWSER_NO_HTTP_NO_DEPLOY`

## Explicitly Forbidden Next Actions

- local glossary fallback
- hardcoded answer
- direct Skillup DB query
- direct Skillup file query
- runtime smoke
- browser automation
- HTTP smoke
- broad Beta PASS claim
- Track A PASS claim
- F13 PASS claim
- deploy
- release
- production readiness claim
