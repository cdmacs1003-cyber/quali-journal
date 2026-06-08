# Track A P0 Selected Test Evidence Index

Packet: T-A1-07SOU_R9ZBM_TRACK_A_P0_SELECTED_TEST_PROOFPACK_MATERIALIZATION_APPROVAL_PACKET
Evidence source: accepted terminal/session evidence from R9ZBK

| Evidence item | Path | Purpose | State |
|---|---|---|---|
| Selected test results | reports/track_a/p0_selected_test_proofpack/selected_test_results.md | Records accepted R9ZBK result, warnings, and non-execution boundaries | PROOFPACK_CANDIDATE |
| Selected test command | reports/track_a/p0_selected_test_proofpack/selected_test_command.txt | Records selected pytest command and execution mode from accepted evidence | PROOFPACK_CANDIDATE |
| Selected test scope | reports/track_a/p0_selected_test_proofpack/selected_test_scope.md | Maps Track A P0 axes to selected tracked test surfaces | PROOFPACK_CANDIDATE |
| Non-escalation register | reports/track_a/p0_selected_test_proofpack/non_escalation.md | Preserves NOT_GRANTED and NOT_VERIFIED boundaries | PROOFPACK_CANDIDATE |
| Rollback note | reports/track_a/p0_selected_test_proofpack/rollback_note.md | Describes future rollback handling without authorizing cleanup | PROOFPACK_CANDIDATE |
| SHA256 sums | reports/track_a/p0_selected_test_proofpack/SHA256SUMS | Records SHA256 hashes for non-recursive ProofPack artifacts | PROOFPACK_CANDIDATE |
| Manifest | reports/track_a/p0_selected_test_proofpack/proofpack_manifest.json | Records manifest metadata, artifact scope, and evidence boundaries | PROOFPACK_CANDIDATE |

## Evidence Limits

- R9ZBM materialized accepted session evidence only.
- R9ZBM did not rerun pytest.
- R9ZBM did not execute lint, build, runtime, HTTP, DB, external network, or release activity.
- R9ZBM did not perform git add, commit, tag, push, deployment, reset, restore, stash, checkout, or cleanup.
