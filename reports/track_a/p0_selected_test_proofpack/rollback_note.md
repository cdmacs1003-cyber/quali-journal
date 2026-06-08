# Rollback Note

R9ZBM is approved only to create the Track A P0 selected test ProofPack directory and exactly eight approved ProofPack files.

## If Rejected Before Commit

Rollback requires a separate explicit approval. The rollback scope should remove only these eight approved files and remove `reports/track_a/p0_selected_test_proofpack/` only if it is empty after those files are removed.

Approved R9ZBM does not authorize cleanup, reset, restore, stash, checkout, or deletion.

## If Committed Later

If this ProofPack is committed in a future packet and later rejected, rollback requires a separate git revert or corrective commit approval. No git write action is authorized by R9ZBM.

## Protected Boundaries

No rollback action may inspect secret-like contents, delete unrelated files, modify source code, run tests, start runtime, send HTTP requests, verify DB behavior, use external network access, tag, push, deploy, or release without separate approval.
