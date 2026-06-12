# F13 Bridge Runtime Contract Shape Wrapper

## 0. Metadata

* Task ID: R9ZJI_STATIC_GAP_MAP_DOC_ONLY_CLOSURE_DRAFT_PACKET_NO_RUNTIME_NO_HTTP_NO_DB
* Document status: DRAFT_STATIC_DOC_WRAPPER
* Scope: Static documentation wrapper for existing F13 Bridge runtime contract shape surfaces only.
* Created from:
  * admin/f13_bridge_api.py
  * admin/f13_runtime_guard.py
  * schemas/f13_bridge_evidence_response.schema.json
  * schemas/f13_bridge_check_policy_response.schema.json
  * schemas/f13_bridge_explain_trace_response.schema.json
  * gap_maps/F13_current_gap_map.md
  * docs/feature_specs/F13_library_auto_intake_and_curation_v0.1.md
* Runtime execution: NOT_EXECUTED
* HTTP behavior: NOT_VERIFIED
* DB/network behavior: NOT_VERIFIED
* Release readiness: NOT_GRANTED

## 1. Purpose

This file is a static documentation wrapper for the existing F13 Bridge runtime
contract shape surfaces. It summarizes the current documented and schema-backed
Bridge/F13 response shapes for retrieve-evidence, check-policy, explain-trace,
safe evidence projection, and visible HOLD or fail-closed posture.

This file is not a schema replacement, not a runtime proof, not a release
proof, and not a test result. The JSON schemas remain the machine-readable
response contracts for the response surfaces listed below, and the feature spec
continues to identify the matching runtime contract shape as the canonical
machine-readable shape.

## 2. Source Surfaces

Existing static surfaces used as inputs:

* admin/f13_bridge_api.py
* admin/f13_runtime_guard.py
* schemas/f13_bridge_evidence_response.schema.json
* schemas/f13_bridge_check_policy_response.schema.json
* schemas/f13_bridge_explain_trace_response.schema.json
* gap_maps/F13_current_gap_map.md
* docs/feature_specs/F13_library_auto_intake_and_curation_v0.1.md

## 3. Contract Shape Summary

### retrieve-evidence response

The retrieve-evidence response shape is bounded by
schemas/f13_bridge_evidence_response.schema.json. Its required top-level fields
are result_status, evidence_items, hold_reason, feedback_candidate_required,
raw_text_included, internal_path_included, policy_result, and created_at.

The allowed result statuses are OK, HOLD, and DENIED. Returned evidence items
are pointer-only safe metadata with required evidence_id, bridge_trace_id,
safe_summary, pointer_uri, raw_text_policy, and rights_status fields. Optional
static fields include source_doc_kind and validation_shape_ids. The schema
requires raw_text_included to be false and internal_path_included to be false.

The current limited remediation for retrieve-evidence is accepted only within
the max-length contract scope: evidence_id is capped at 120 characters,
bridge_trace_id at 160 characters, and source_doc_kind at 120 characters.

### check-policy response

The check-policy response shape is bounded by
schemas/f13_bridge_check_policy_response.schema.json. Its required top-level
fields include result_status, bridge_trace_id, policy_result, hold_reason,
output_constraints, blocked_fields, role, evidence_depth, leak counters,
feedback_candidate_required, raw_text_included, internal_path_included, and
created_at.

The allowed result statuses are OK, HOLD, and DENIED. policy_result is PASS,
HOLD, or DENIED. The raw text, internal path, raw prompt, secret, and instructor
guide leak counters are fixed at zero by the schema. raw_text_included and
internal_path_included are fixed as false.

### explain-trace response

The explain-trace response shape is bounded by
schemas/f13_bridge_explain_trace_response.schema.json. Its required top-level
fields include result_status, request_id, bridge_trace_id, course_id, module_id,
binding_id, evidence_ids, policy_result, hold_reason, role, evidence_depth,
review_trace, audit_trace, leak counters, feedback_candidate_required,
feedback_candidate, visible_trace_summary, raw_text_included,
internal_path_included, and created_at.

feedback_candidate is null when the trace explanation is OK. Otherwise the
static schema permits a bounded BRIDGE_TRACE_REVIEW object with reason,
next_action, and optional bridge_trace_id. review_trace and audit_trace are
safe metadata shapes only; audit_trace keeps raw_export_allowed false.

### Bridge safe evidence projection

admin/f13_runtime_guard.py defines BRIDGE_EVIDENCE_ALLOWLIST_FIELDS for
evidence_id, bridge_trace_id, safe_summary, pointer_uri, raw_text_policy,
rights_status, source_doc_kind, and validation_shape_ids. The same source
normalizes rights_status and raw_text_policy, omits forbidden fields from the
projection, and applies field-specific max lengths for evidence_id,
bridge_trace_id, source_doc_kind, and validation_shape_ids.

The projection summary here is static documentation only. It does not prove
runtime behavior.

### HOLD and fail-closed posture visible from static surfaces

The static surfaces show HOLD or DENIED outcomes for missing or invalid
evidence, missing evidence_id, missing safe_summary, restricted rights,
unverified rights, denied or unverified raw text policy, forbidden field or
pattern detection, direct DB access attempts for Bridge-only Skillup requests,
role or evidence-depth policy mismatch, tenant or license boundary issues, and
missing bridge_trace_id or evidence_ids for trace explanation.

The feature spec states a no-DB Bridge boundary for Skillup-facing evidence
retrieval, policy checks, and trace explanation, and it keeps functional 200
behavior, runtime smoke, authenticated smoke, Track A approval, Beta approval,
F13 approval, and release approval outside this static gate.

## 4. Static Boundaries

* Static documentation only
* No runtime/server execution
* No HTTP/browser/healthcheck
* No DB/network
* No tests executed
* No schema changes
* No code changes

## 5. Gap Map Relationship

This file addresses the gap-map item:

* Gap: Shape documentation wrapper missing
* Source path: shapes/f13_bridge_runtime_contract_shape.md
* Classification: STATIC_DOC_GAP
* Static-only closable: YES

This file does not update gap_maps/F13_current_gap_map.md. Any gap-map status
change requires a later explicit packet.

## 6. NOT_EXECUTED / NOT_VERIFIED / NOT_GRANTED

* NOT_EXECUTED:
  tests, lint, build, integration, E2E, runtime/server,
  HTTP/browser/healthcheck, DB/network, deploy/release/tag/push
* NOT_VERIFIED:
  full regression, runtime behavior, HTTP behavior, DB/network behavior,
  production/release/deployment behavior, answer quality, Bridge health,
  Skillup MVP
* NOT_GRANTED:
  Track A PASS, Beta PASS, F13 PASS, release readiness, deployment readiness,
  production readiness, Bridge health PASS, answer quality PASS,
  Skillup MVP PASS

## 7. Quarantine Statement

reports/track_a/limited_skillup_beta_use_operation_runbook/raw_secret_leak_policy.md
remains QUARANTINE. It was not opened, hashed, summarized, inferred, copied,
deleted, inspected, printed, or used as a recovery source for this wrapper.

## 8. Future Gates

Any stronger claim requires a later explicit gate:

* explicit test packet
* explicit runtime packet
* explicit HTTP packet
* explicit DB/network packet
* explicit release/readiness packet
* explicit security-specific packet for any secret-bearing file handling

## 9. Final Static Doc Wrapper Status

* Status: DRAFT_STATIC_DOC_WRAPPER
* Completion claim: STATIC_DOC_WRAPPER_CREATED_WITH_LIMITS
* No PASS escalation.
