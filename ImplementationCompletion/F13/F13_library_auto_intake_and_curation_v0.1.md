# F13 Library Auto Intake and Curation v0.1

## 1. Completion Artifact Identity

| Item | Value |
|---|---|
| Completion gate | ICD-G1 F13 Spec Materialized |
| Artifact path | ImplementationCompletion/F13/F13_library_auto_intake_and_curation_v0.1.md |
| Artifact type | F13 specification |
| Repository branch | track-a-07s-static-closure-proofpack |
| Materialization task | T-A1-07SOU_R2_ICD_G1_F13_SPEC_MATERIALIZATION_EXECUTION_ONLY |
| Source status | CANONICAL_COMPLETION_SPEC_CANDIDATE |

This file materializes the F13 specification at the completion evidence path required by ICD-G1.

## 2. Source Basis and Limits

This artifact is based on the committed development specification at:

- docs/feature_specs/F13_library_auto_intake_and_curation_v0.1.md

It also records the bounded Track A/F13 selected evidence already committed in:

- reports/track_a/QLIB_TA1_07SOJ_TRACK_A_P0_SELECTED_EVIDENCE_SNAPSHOT_20260603.md
- reports/track_a/proofpack/manifest.md
- reports/track_a/proofpack/gate_results.md
- reports/track_a/proofpack/test_results.md
- reports/track_a/proofpack/gap_map_reference.md
- reports/track_a/proofpack/release_board_reference.md
- reports/track_a/proofpack/rollback.md
- reports/track_a/proofpack/SHA256SUMS.txt

The ontology/semantic workflow guidebook path named in the approval packet was not present in this worktree during the preceding read-only closure packet. This artifact therefore does not quote, reconstruct, or infer unavailable guidebook text.

## 3. F13 Scope

F13 defines a no-DB Bridge boundary for library auto intake and curation surfaces used by Skillup-facing evidence retrieval, policy checks, trace explanation, HOLD feedback, feedback queue item materialization, course-library binding, release-board review, gap-map review, and ProofPack manifest review.

F13 behavior in this completion artifact is limited to bounded selected evidence and committed no-DB contract surfaces. It does not assert production DB behavior, production raw leak safety, full regression safety, deployment readiness, release readiness, F13 PASS, Track A PASS, or Beta PASS.

## 4. Bridge Boundary

Bridge/F13 routes and helpers must consume caller-provided evidence only.

Bridge/F13 routes and helpers must not query:

- databases;
- warehouses;
- library indexes;
- local files;
- external networks;
- runtime services.

Direct DB/internal access attempts from a Skillup path must be denied, blocked, or held without executing DB access. If the request requires review, the response must require or produce feedback.

## 5. Required Route Surface

The static F13 Bridge route surface includes:

- POST `/api/f13/bridge/retrieve-evidence`
- POST `/api/f13/bridge/check-policy`
- POST `/api/f13/bridge/explain-trace`
- POST `/api/f13/bridge/skillup/bridge-answer`

The Skillup Bridge answer route must delegate to the no-DB Skillup Bridge helper instead of duplicating direct DB logic.

## 6. Evidence Contract

Evidence returned across the Bridge boundary must contain safe pointer metadata only:

- evidence_id;
- bridge_trace_id;
- safe_summary;
- pointer_uri;
- raw_text_policy;
- rights_status;
- optional source_doc_kind;
- optional validation_shape_ids.

Bridge and Skillup responses must preserve:

- raw_text_included=false;
- internal_path_included=false;
- db_access_executed=false when exposed.

Responses must not expose:

- raw_text;
- raw_prompt;
- raw_query;
- full_source_text;
- internal_path;
- local filesystem path;
- secret-like value;
- token;
- key;
- credential;
- DSN;
- customer confidential text;
- paid standard raw text.

## 7. Result Status Contract

F13 Bridge result statuses are:

- OK: safe evidence was accepted and safe_summary may be used;
- HOLD: evidence, trace, schema, rights, or review information is missing or incomplete;
- DENIED: raw leak, restricted rights, direct DB/internal access, or safety boundary violation was detected.

HOLD and DENIED responses must preserve a safe hold reason or policy reason without echoing raw/internal values.

## 8. HOLD Feedback Contract

When Bridge returns or represents HOLD because evidence is missing:

- Skillup result status must be HOLD;
- feedback_candidate_required must be true or a feedback candidate must be present;
- feedback candidate surfaces must not contain raw text or internal paths;
- a feedback queue item may be materialized as a no-DB object;
- dedup_key must be stable for the same logical gap;
- no F13 PASS, Track A PASS, or Beta PASS field may be produced.

Feedback queue items must preserve safe traceability fields only.

## 9. Course Library Binding Contract

Course-library binding must bind a course module, lesson, or module reference to a safe library/evidence reference using safe identifiers only.

The binding output must include:

- binding_id;
- course_id;
- module_id or lesson_id;
- library_node_id or evidence_id;
- bridge_trace_id or trace_ref;
- binding_status;
- rights_status;
- raw_text_policy;
- created_at.

If evidence_id or library_node_id is missing, the binding must return HOLD or REVIEW_REQUIRED behavior and require or produce feedback. If rights_status is UNKNOWN or denied-like, Skillup use must be blocked or held.

## 10. Release Board, Gap Map, and ProofPack Contract

The Beta Release Board, F13 Gap Map, and F13 ProofPack Manifest helpers are no-DB, no-file-IO contract surfaces unless a later task explicitly approves generation or verification of files.

The committed ProofPack candidate artifacts record selected evidence and bounded SHA256 verification only. They do not record final approval and do not grant completion.

The Gap Map must keep open or review-required items explicit when any required F13 completion condition remains missing, not executed, or not verified.

## 11. Schema and Shape Policy

The F13 retrieve-evidence schema must expose safe response fields and reject raw/internal leak surfaces. The selected evidence refers to:

- schemas/f13_bridge_evidence_response.schema.json;
- schemas/f13/bridge_evidence_response.schema.json;
- shapes/f13_bridge_runtime_contract_shape.json;
- shapes/f13_bridge_runtime_contract_shape.md.

Schema materialization remains separately governed by ICD-G2 and is not closed by this ICD-G1 artifact alone.

## 12. Guard Contract

The guard must fail closed for:

- raw text markers;
- internal path markers;
- secret or token markers;
- DB or DSN markers;
- missing evidence identifiers;
- missing safe summaries;
- restricted rights;
- unverified rights or raw text policy;
- schema/model mismatch.

The guard must never return the offending value in an external response.

## 13. Evidence Recorded Before This Artifact

Committed selected evidence records:

- retrieve-evidence empty HOLD behavior;
- retrieve-evidence synthetic OK behavior;
- negative raw leak policy behavior;
- restricted rights policy behavior;
- Skillup Bridge HOLD feedback helper;
- Feedback Queue Item materialization;
- Skillup Bridge runtime route selected TestClient behavior;
- Skillup Bridge route bounded localhost diagnostic;
- course_library_binding selected behavior;
- Beta Release Board selected behavior;
- F13 ProofPack Manifest selected behavior;
- F13 Gap Map selected behavior;
- combined selected no-DB regression through 38 selected tests.

This evidence is selected-only and no-DB. It is not broad pytest, full regression, production DB verification, production raw leak verification, deployment approval, release approval, F13 PASS, Track A PASS, or Beta PASS.

## 14. Status Preservation

| Claim | Status |
|---|---|
| Direct Codex DB verification | NOT_EXECUTED / NOT_VERIFIED |
| DB-backed persistence | NOT_EXECUTED / NOT_VERIFIED |
| Production DB behavior | NOT_EXECUTED / NOT_VERIFIED |
| Production raw leak safety | NOT_VERIFIED |
| Full regression safety | NOT_VERIFIED |
| Deployment approval | NOT_GRANTED |
| Release approval | NOT_GRANTED |
| F13 PASS | NOT_GRANTED |
| Track A PASS | NOT_GRANTED |
| Beta PASS | NOT_GRANTED |

## 15. Deferred Completion Gates

This ICD-G1 artifact does not close:

- ICD-G2 Schema Materialized;
- ICD-G3 Gap Map Closed;
- ICD-G4 State Transition Enforced;
- ICD-G8 Feedback Loop Enforced for DB-backed persistence;
- ICD-G9 Migration Guard Enforced;
- ICD-G10 Quarantine/Rollback Enforced;
- ICD-G12 Final Approval Recorded.

Those gates require separate evidence closure or final approval tasks.
