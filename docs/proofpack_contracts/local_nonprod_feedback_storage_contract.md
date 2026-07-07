# Local Non-Production Feedback Storage Contract

TASK_ID=R9ZNW-380_BOUNDED_LOCAL_NONPROD_FEEDBACK_STORAGE_CONTRACT_SPEC_NO_PROD_DB_NO_DEPLOY_NO_RAW_TEXT
CONTRACT_STATUS=CANONICAL
CONTRACT_PURPOSE=DEFINE_LOCAL_NONPROD_FEEDBACK_STORAGE_BOUNDARY_BEFORE_EXECUTION

## Authority And Scope

OWNER_RESPONSIBILITY_AUTHORITY=DOCTOR_YOON
LEGAL_DOMAIN_SEMANTIC_FINAL_RESPONSIBILITY=DOCTOR_YOON
OWNER_DECISION_VALUES_CAPTURED_WITH_LIMITS=SUPPORTED
RIGHTS_LICENSE_LEGAL_APPROVAL=OWNER_RESPONSIBILITY_ACCEPTED_WITH_LIMITS
SEMANTIC_CORRECTNESS_BEYOND_SAFE_SUMMARIES=OWNER_RESPONSIBILITY_ACCEPTED_WITH_LIMITS_FOR_LOCAL_SAFE_SUMMARY_CHAIN_ONLY

This contract allows a future gate to implement or execute local non-production feedback storage only after separate execution approval. It does not itself execute storage behavior.

Allowed scope:

- LOCAL_NONPROD_ONLY=TRUE
- TASK_OWNED_BETA_EVIDENCE_ONLY=TRUE
- PRODUCTION_DB_ROOT_ALLOWED=FALSE
- PRODUCTION_LIBRARY_ROOT_ALLOWED=FALSE
- DEPLOY_ALLOWED=FALSE
- RAW_TEXT_ALLOWED=FALSE
- SECRET_CONFIG_DSN_ALLOWED=FALSE
- PUBLIC_API_SCHEMA_EXPANSION_ALLOWED=FALSE
- DURABLE_PRODUCTION_FEEDBACK_APPROVAL_GRANTED=FALSE

## Allowed Storage Target

Allowed target root:

```text
H:\tmp\<TASK_ID>\feedback\
```

Storage root requirements:

- The storage root must be explicit for each execution gate.
- The storage root must be outside production roots.
- The storage root must be task-owned and non-production.
- No default persistent production path is allowed.
- No automatic promotion to Library root is allowed.
- No automatic upload, sync, deploy, or public URL exposure is allowed.

Forbidden storage targets:

- production DB
- production Library root
- long-memory production root
- secret-like filenames or paths
- DSN/database URLs
- public API-exposed locations
- external network locations

## Allowed Feedback Fields

Only minimized metadata is allowed. A future implementation may use the existing tracked minimized queue shape when all flags below remain false.

Allowed core fields:

- event_id: opaque identifier only
- feedback_id: opaque identifier only
- origin_event_id: opaque identifier only
- created_at: timestamp
- route_name: route or module name, not URL payload
- status: HOLD, review, queued, duplicate, resolved, or rejected status
- reason_code: safe bounded reason code
- safe_summary: bounded summary only
- evidence_pointer: pointer only, not evidence body
- trace_pointer: pointer only, not trace payload
- trace_id: opaque identifier or null
- request_id: opaque identifier or null
- local_task_id: task id
- non_production_marker: required true marker
- dedup_key: safe bounded idempotency key
- evidence_required: boolean
- review_required: boolean
- evidence_count: count only
- warning_codes: safe bounded warning codes only
- raw_text_present: false
- raw_text_included: false
- secret_present: false
- internal_path_included: false
- production_db_used: false
- production_root_used: false
- db_access_executed: false

Allowed persistence metadata:

- contract_version
- persistence_mechanism
- retention_expires_at
- cleanup_required
- cleanup_evidence_pointer

## Prohibited Feedback Fields

The following fields and payload classes are prohibited:

- raw user query
- raw answer body
- raw paid standard text
- source text or clause text
- production raw body
- secret, token, key, cookie, header, or auth values
- DSN or database URL
- production DB row dump
- production Library root path
- full internal path dump
- external URL payload
- personal data beyond opaque local identifiers
- raw Bridge payload
- raw Bridge response
- raw evidence body
- raw source payload
- internal hostnames
- localhost URLs in stored records
- file URIs

Records containing prohibited fields must be rejected before write. Rejection must not echo the prohibited value.

## Retention And Cleanup

RETENTION_POLICY=SHORT_LOCAL_TASK_OWNED

Default retention for future execution:

- Target retention should be 72 hours or less unless a later approved gate narrows or changes it.
- Cleanup evidence is required before release-readiness advancement.
- No indefinite local storage is allowed.
- No background job is allowed by default.
- No automatic upload or sync is allowed.
- Cleanup failure must block release-readiness advancement.
- Cleanup must be bounded to the explicit task-owned local storage path.

## Redaction And Minimization

MINIMIZATION_POLICY=SAFE_SUMMARY_OR_POINTER_ONLY

Required behavior:

- Store safe summaries or pointers only.
- Do not store raw user query, raw answer, raw standard text, source text, or production body.
- Marker/raw-leak scan is required before accepting evidence.
- Unsafe markers force HOLD or rejection.
- Raw/internal/DB/production flags must remain false.
- Records with prohibited fields must be rejected before local write.
- Selected-route responses must not expose feedback queue internals.

## Marker And Raw Leak Scan Contract

A future execution gate must scan proposed stored records and captured evidence for unsafe markers before accepting storage evidence.

Required marker classes:

- raw_text
- raw_prompt
- raw_query
- raw_answer
- raw_source
- source_text
- source_uri_or_path
- full_answer
- full_source
- internal_path
- file_uri
- hostname
- secret
- token
- credential
- dsn
- api_key
- private_key
- service_account
- bridge_payload
- bridge_response
- evidence_items
- source_payload
- standard_text
- production_db
- production_library_root

Failure rule:

```text
MARKER_RAW_LEAK_SCAN_FAILURE=BLOCKS_RELEASE_READINESS_ADVANCEMENT
```

## Implementation Preconditions

Before any implementation or execution:

- CONTRACT_CANONICAL_BEFORE_IMPLEMENTATION=REQUIRED
- SEPARATE_EXECUTION_GATE=REQUIRED
- LOCAL_NONPROD_STORAGE_ONLY=REQUIRED
- PRODUCTION_STORAGE_REQUIRES_SEPARATE_HIGHER_RISK_GATE=REQUIRED
- PUBLIC_API_SCHEMA_EXPOSURE_REQUIRES_SEPARATE_SAFETY_BOARD=REQUIRED
- RAW_TEXT_APPROVAL_REQUIRED_FOR_RAW_TEXT=NOT_GRANTED
- PRODUCTION_DB_APPROVAL_REQUIRED_FOR_DB=NOT_GRANTED
- PRODUCTION_LIBRARY_ROOT_APPROVAL_REQUIRED_FOR_ROOT=NOT_GRANTED

## Claim Boundary

LOCAL_BETA_CHAIN_SELECTED_LOCAL_VALIDATION_CLOSED_WITH_LIMITS=SUPPORTED
LOCAL_ONLY_SAFE_SIDECAR_SEED_BRIDGE_SKILLUP_FEEDBACK_CHAIN_CLOSED_WITH_LIMITS=SUPPORTED
CANDIDATE_DOCS_CURRENT_ROUTE_BLOCKER=RESOLVED_WITH_PRESERVE_NO_COMMIT_DECISION
LOCAL_RUNTIME_BROWSER_EXECUTION=EXECUTED_WITH_LIMITS
BROWSER_PUBLIC_UI_BEHAVIOR=LOCAL_LOOPBACK_VALIDATED_WITH_LIMITS
RUNTIME_SERVER_BEHAVIOR=LOCAL_LOOPBACK_VALIDATED_WITH_LIMITS
SKILLUP_BRIDGE_RUNTIME_BEHAVIOR=LOCAL_LOOPBACK_VALIDATED_WITH_LIMITS
MARKER_RAW_LEAK_RUNTIME_SCAN=PASS_NO_UNSAFE_MARKERS_IN_CAPTURED_EVIDENCE
SHUTDOWN_CLEANUP=VERIFIED_LISTENER_GONE
DURABLE_FEEDBACK_STORAGE_DESIGN_REVIEW=APPROVED_WITH_LIMITS
LOCAL_NONPROD_FEEDBACK_STORAGE_CONTRACT_SPEC=CREATED_WITH_LIMITS

Still not verified:

```text
LOCAL_NONPROD_DURABLE_FEEDBACK_STORAGE_EXECUTION=NOT_VERIFIED
DURABLE_PRODUCTION_FEEDBACK_STORAGE=NOT_VERIFIED
PUBLIC_API_SCHEMA_EXPANSION_SAFETY=NOT_VERIFIED
FULL_RENDERED_BROWSER_PIXEL_BEHAVIOR=NOT_VERIFIED
PRODUCTION_DB_READINESS=NOT_VERIFIED
PRODUCTION_LIBRARY_ROOT_READINESS=NOT_VERIFIED
RAW_TEXT_LEGAL_SUFFICIENCY=NOT_VERIFIED
RELEASE_READINESS=NOT_VERIFIED
DEPLOYMENT_READINESS=NOT_VERIFIED
PRODUCTION_READINESS=NOT_VERIFIED
BETA_TRACK_A_F13_READINESS=NOT_VERIFIED
```

Still not granted:

```text
RAW_TEXT_APPROVAL=NOT_GRANTED
PRODUCTION_DB_OK_RETRIEVAL_PASS=NOT_GRANTED
PRODUCTION_LIBRARY_ROOT_PASS=NOT_GRANTED
DURABLE_FEEDBACK_STORAGE_APPROVAL=NOT_GRANTED
DEPLOY_APPROVAL=NOT_GRANTED
RELEASE_READY=NOT_GRANTED
DEPLOYMENT_READY=NOT_GRANTED
PRODUCTION_READY=NOT_GRANTED
BETA_PASS=NOT_GRANTED
TRACK_A_PASS=NOT_GRANTED
F13_PASS=NOT_GRANTED
CANONICAL_BRAND_APPROVED=NOT_GRANTED
CANDIDATE_DOCS_CANONICALIZED=NOT_GRANTED
```
