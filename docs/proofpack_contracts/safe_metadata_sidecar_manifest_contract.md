# Safe Metadata Sidecar Manifest Contract

Task scope: R9ZNW-343.

This contract defines the reviewable manifest boundary for Bridge-safe Library
metadata sidecars. It is a governance contract only. It does not approve
production placement, production DB writes, runtime wiring, browser validation,
deployment, or public schema expansion.

## Required Manifest Fields

- `manifest_version`
- `sidecar_id`
- `sidecar_kind`
- `created_at_utc`
- `created_by_task`
- `source_task_id`
- `source_proofpack_refs`
- `sidecar_sqlite_path`
- `sidecar_json_path`
- `sidecar_sha256`
- `record_count`
- `accepted_record_count`
- `hold_only_record_count`
- `rejected_record_count`
- `allowed_rights_statuses`
- `raw_text_policy_allowed_values`
- `summary_source_allowed_values`
- `semantic_summary_required`
- `raw_text_exposed_required_false`
- `production_path_exposed_required_false`
- `public_pointer_exposure_allowed`
- `skillup_direct_db_access_allowed`
- `production_db_write_allowed`
- `production_raw_text_read_allowed`
- `refresh_policy`
- `rollback_policy`
- `expiry_or_review_required_at`
- `approver_or_review_status`
- `artifact_state`

## Resolver Policy

The resolver accepts only an explicit manifest path from a caller. It must not
discover manifests through environment variables, config files, DSNs, network
locations, or production Library scanning.

The resolver must validate the sidecar hash before returning an internal Bridge
reference. Resolver output is for Bridge retrieval helpers only and must not be
projected into Skillup public/user-visible responses.

The resolver rejects:

- missing sidecar files;
- hash mismatches;
- secret-like manifest or sidecar filenames;
- manifests with `artifact_state` outside `APPROVED_SOURCE`, `PROOFPACKED`, or
  `CANONICAL_CANDIDATE_FOR_PLACEMENT`;
- manifests allowing public pointer exposure;
- manifests allowing Skillup direct DB access;
- manifests allowing production DB writes;
- manifests allowing production raw text reads;
- manifests that do not require raw text and production path exposure flags to
  remain false.

## Refresh Policy

Refresh is explicit and review-gated. Refresh must not be automatic.

Every refresh must:

- use approved safe metadata sources only;
- create a new `sidecar_id`;
- create a new sidecar hash;
- preserve the previous sidecar until a reviewed replacement is accepted;
- emit accepted, hold-only, and rejected counts;
- keep production DB writes disabled;
- keep production raw text reads disabled;
- keep public pointer exposure disabled.

Refresh failure must keep the prior approved sidecar valid for rollback by
reviewable manifest selection.

## Not Granted

This contract does not grant:

- `DB_BACKED_RETRIEVAL_PASS`;
- `PRODUCTION_DB_OK_RETRIEVAL_PASS`;
- `PRODUCTION_LIBRARY_ROOT_PASS`;
- `PUBLIC_API_PASS`;
- `FULL_E2E_PASS`;
- `BETA_PASS`;
- `TRACK_A_PASS`;
- `F13_PASS`;
- release, deployment, or production readiness.
