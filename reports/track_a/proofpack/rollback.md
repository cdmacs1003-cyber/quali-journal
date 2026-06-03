# F13 ProofPack Candidate Rollback and Preservation Note

## 1. Current task rollback
This task creates uncommitted candidate files only.

Because deletion was not separately approved, this task does not delete generated files automatically.

If rollback is required, request a separate cleanup approval to remove exactly:

- reports/track_a/proofpack/manifest.md
- reports/track_a/proofpack/gate_results.md
- reports/track_a/proofpack/test_results.md
- reports/track_a/proofpack/selected_evidence_snapshot_reference.md
- reports/track_a/proofpack/gap_map_reference.md
- reports/track_a/proofpack/release_board_reference.md
- reports/track_a/proofpack/rollback.md
- reports/track_a/proofpack/SHA256SUMS.txt

## 2. Repository rollback after future commit
If a later commit-only task is approved and committed, rollback must use a separate approved revert or corrective commit.

## 3. Preserved limitations
- DB-backed persistence: NOT_EXECUTED / NOT_VERIFIED
- Production DB behavior: NOT_EXECUTED / NOT_VERIFIED
- Production raw leak safety: NOT_VERIFIED
- Full regression safety: NOT_VERIFIED
- Deployment approval: NOT_GRANTED
- Release approval: NOT_GRANTED
- F13 PASS: NOT_GRANTED
- Track A PASS: NOT_GRANTED
- Beta PASS: NOT_GRANTED
