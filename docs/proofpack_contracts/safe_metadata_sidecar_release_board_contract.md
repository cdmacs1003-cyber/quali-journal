# Safe Metadata Sidecar Release-Board Draft Contract

Task scope: R9ZNW-344.

This contract defines a rehearsal-only release-board draft for safe metadata
sidecar placement. It does not approve live production-adjacent placement,
production Library root writes, production DB writes, runtime wiring, browser
validation, deployment, public schema expansion, or public response field
changes.

## Placement Candidate Policy

The placement candidate policy must define:

- `sidecar_root_candidate`
- `manifest_root_candidate`
- `ownership`
- `review_status_required`
- `allowed_artifact_states`
- `hash_validation_required`
- `rollback_requirement`
- `refresh_requirement`
- `expiry_or_review_date_requirement`
- `production_db_mutation_forbidden`
- `production_raw_text_read_forbidden`
- `public_pointer_exposure_forbidden`
- `skillup_direct_db_access_forbidden`
- `bridge_only_retrieval_required`

The policy is definition-only. R9ZNW-344 may write task-owned rehearsal
artifacts under `H:\tmp\R9ZNW-344`, but must not write to the production Library
root or any production-adjacent candidate path.

Allowed artifact states are limited to:

- `APPROVED_SOURCE`
- `PROOFPACKED`
- `CANONICAL_CANDIDATE_FOR_PLACEMENT`

## Release-Board Draft

The release-board draft must include:

- `release_board_id`
- `task_id`
- `source_commit`
- `sidecar_id`
- `sidecar_manifest_hash`
- `sidecar_sqlite_hash`
- `sidecar_json_hash`
- `resolver_validation`
- `bridge_retrieval_validation`
- `skillup_public_exposure_check`
- `rollback_plan`
- `refresh_plan`
- `review_status`
- `approval_required_before_live_use`
- `NOT_GRANTED claims preserved`
- `go_hold_cut_decision`
- `final_recommendation`

`approval_required_before_live_use` must remain true. `go_hold_cut_decision`
must remain HOLD for live use. The final recommendation must be rehearsal-only
and must not claim release, deployment, production, Track A, F13, Skillup MVP,
browser, full E2E, public API, production root, or production DB readiness.

## Not Granted

The release-board draft must preserve these claims as NOT_GRANTED:

- `PRODUCTION_ADJACENT_PLACEMENT_PASS`
- `DB_BACKED_RETRIEVAL_PASS`
- `PRODUCTION_DB_OK_RETRIEVAL_PASS`
- `PRODUCTION_LIBRARY_ROOT_PASS`
- `PUBLIC_API_PASS`
- `FULL_E2E_PASS`
- `BETA_PASS`
- `TRACK_A_PASS`
- `F13_PASS`
- `BROWSER_PASS`
- `SKILLUP_MVP_PASS`
- `RELEASE_READY`
- `DEPLOYMENT_READY`
- `PRODUCTION_READY`

## Validation

Validation must use task-owned rehearsal artifacts only. The resolver must
validate the rehearsal manifest and hashes, reject a tampered hash, Bridge must
retrieve through the validated rehearsal sidecar, and Skillup adapted public
output must omit manifest paths, sidecar paths, qlib pointers, production paths,
raw text, secrets, tokens, keys, and credentials.
