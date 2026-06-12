# F13 Bridge Runtime Contract Shape

## Purpose

This Markdown file is the human-readable static contract wrapper for
`shapes/f13_bridge_runtime_contract_shape.json`. The JSON shape remains the
canonical machine-readable Bridge/F13 runtime contract. This wrapper records
the Bridge Runtime MVP boundary and the verification status for later gates.

Artifact state: `DRAFT_PENDING_STATIC_VERIFICATION`

## Scope Boundary

The Bridge Runtime MVP boundary is limited to Skillup-facing evidence
retrieval, policy checking, and trace explanation. Bridge/F13 routes must
consume caller-provided evidence only.

Bridge/F13 routes must not query databases, warehouses, library indexes,
files, networks, runtime services, environment variables, or secret stores.
They must not expose raw evidence, internal paths, DB or DSN details, raw
prompts, secret-like values, or admin-only values in external responses.

## Endpoint / Function Map

| Endpoint | Function | Static contract |
|---|---|---|
| `POST /api/f13/bridge/retrieve-evidence` | `retrieve_bridge_evidence` | Accepts caller-provided evidence and returns safe pointer metadata or HOLD/DENIED. |
| `POST /api/f13/bridge/check-policy` | `check_bridge_policy` | Applies evidence, role, scope, license, and output constraints with fail-closed results. |
| `POST /api/f13/bridge/explain-trace` | `explain_bridge_trace` | Returns safe trace metadata and a bounded feedback candidate when trace explanation is not OK. |

## Evidence Requirement

`evidence_id` and `safe_summary` are required for OK Bridge evidence
decisions. Evidence returned across the Bridge boundary must be pointer-only
safe metadata and must keep `raw_text_included=false` and
`internal_path_included=false`.

Allowed safe evidence metadata:

- `evidence_id`
- `bridge_trace_id`
- `safe_summary`
- `pointer_uri`
- `raw_text_policy`
- `rights_status`
- optional `source_doc_kind`
- optional `validation_shape_ids`

## Trace Requirement

`bridge_trace_id` and safe evidence identifiers are required for OK trace
explanation. A missing or unsafe trace identifier, missing safe evidence IDs,
or forbidden trace content must return HOLD or DENIED.

The explain-trace response must expose top-level
`feedback_candidate_required` and `feedback_candidate` fields.
`feedback_candidate` is `null` when the trace explanation is OK. Otherwise it
must be a bounded safe review object without raw evidence, internal path,
secret-like values, or DB/DSN detail.

## Policy / Fail-Closed Requirement

The Bridge guard must return HOLD or DENIED for:

- unsafe or missing evidence
- restricted rights
- unverified rights
- denied or unverified raw text policy
- missing role, course, module, binding, tenant, organization, or cohort scope
- role/evidence-depth mismatch
- tenant, organization, cohort, or license scope mismatch
- missing or inactive license entitlement when required
- schema/model mismatch

## Raw Leak Blocking Requirement

The Bridge boundary must not return raw text, raw prompts, raw standard text,
internal paths, local paths, DB fields, DSN values, secret markers, token
markers, private keys, credential markers, admin-only output, audit-only output,
or raw instructor-guide values.

External response leak counters must remain zero where present:

- `raw_text_export_count`
- `internal_path_leak_count`
- `raw_prompt_output_count`
- `secret_leak_count`
- `instructor_guide_raw_leak_count`

## Schema References

- `schemas/f13_bridge_evidence_response.schema.json`
- `schemas/f13_bridge_check_policy_response.schema.json`
- `schemas/f13_bridge_explain_trace_response.schema.json`

## Test Candidate References

These are static future test candidates only. They are not executed by this
materialization packet.

- `admin/tests/test_f13_bridge_evidence_response_schema.py`
- `admin/tests/test_f13_bridge_check_policy_response_schema.py`
- `admin/tests/test_f13_bridge_explain_trace_response_schema.py`

## Explicit Status

| Item | Status |
|---|---|
| Runtime behavior | `NOT_VERIFIED` |
| Functional Bridge 200 behavior | `NOT_VERIFIED` |
| Tests | `NOT_EXECUTED` |
| Track A PASS | `NOT_GRANTED` |
| Beta PASS | `NOT_GRANTED` |
| F13 PASS | `NOT_GRANTED` |
| Release PASS | `NOT_GRANTED` |

## Prohibited Claims

- Do not claim runtime PASS.
- Do not claim Beta PASS.
- Do not claim F13 PASS.
- Do not claim Release PASS.

