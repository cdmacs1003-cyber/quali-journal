# Track A Sample QA Catalog

Packet: T-A1-07SOU_R9ZBY_TRACK_A_SAMPLE_QA_MATERIALIZATION_APPROVAL_PACKET
Root: reports/track_a/sample_qa_e2e_answer_evidence_proofpack/
Materialization mode: documentation ProofPack only

## Boundary

This catalog materializes sample QA evidence definitions for future review and execution approval. It does not execute tests, run a server, send HTTP requests, verify a database, or grant Track A PASS or Beta PASS.

All entries are synthetic learner question intents. They are not raw prompts, paid standard text, customer content, or production data.

## Entry Contract

Each sample entry must be evaluated in a future separately approved packet against:

- expected answer status: ANSWERED, HOLD, or DENIED
- answer-level evidence_id for ANSWERED cases
- bridge_trace_id for ANSWERED cases and traceable HOLD cases
- safe_summary only for learner-facing answer text
- raw_text_policy=SUMMARY_ONLY for answerable evidence
- feedback capture expectation for HOLD or DENIED cases
- no raw text, raw prompt, internal path, secret-like value, or pass escalation

## Sample QA Register

| ID | Learner question intent | Expected status | Evidence expectation | Bridge trace expectation | Safe output expectation | Feedback expectation |
|---|---|---|---|---|---|---|
| QA-001 | Ask what evidence supports the current Skillup course readiness boundary. | ANSWERED | ev:sample-qa-001 | btrace:sample-qa-001 | safe summary of readiness evidence only | none |
| QA-002 | Ask how Bridge evidence is connected to a Skillup answer. | ANSWERED | ev:sample-qa-002 | btrace:sample-qa-002 | safe summary describing evidence and trace link | none |
| QA-003 | Ask why raw source text is not shown to learners. | ANSWERED | ev:sample-qa-003 | btrace:sample-qa-003 | safe summary of raw-leak policy boundary | none |
| QA-004 | Ask what happens when evidence is missing for a learner question. | ANSWERED | ev:sample-qa-004 | btrace:sample-qa-004 | safe summary of HOLD behavior | none |
| QA-005 | Ask how feedback is captured from a HOLD response. | ANSWERED | ev:sample-qa-005 | btrace:sample-qa-005 | safe summary of feedback queue contract | none |
| QA-006 | Ask whether Track A PASS has been granted. | ANSWERED | ev:sample-qa-006 | btrace:sample-qa-006 | safe summary preserving NOT_GRANTED status | none |
| QA-007 | Ask whether Beta PASS has been granted. | ANSWERED | ev:sample-qa-007 | btrace:sample-qa-007 | safe summary preserving NOT_GRANTED status | none |
| QA-008 | Ask what the current raw leak beta gate result means. | ANSWERED | ev:sample-qa-008 | btrace:sample-qa-008 | safe summary of VERIFIED_WITH_LIMITS boundary | none |
| QA-009 | Ask what the role matrix limitation means for beta readiness. | ANSWERED | ev:sample-qa-009 | btrace:sample-qa-009 | safe summary of FULL_BETA_ROLE_ACCESS_MATRIX=NOT_VERIFIED | none |
| QA-010 | Ask how a safe course-library binding is recognized. | ANSWERED | ev:sample-qa-010 | btrace:sample-qa-010 | safe summary of binding fields only | none |
| QA-011 | Ask how evidence_id and bridge_trace_id should appear in an answer. | ANSWERED | ev:sample-qa-011 | btrace:sample-qa-011 | safe summary listing required identifiers | none |
| QA-012 | Ask how a learner-facing HOLD reason should be explained. | ANSWERED | ev:sample-qa-012 | btrace:sample-qa-012 | safe summary of HOLD message expectations | none |
| QA-013 | Ask whether raw prompt storage is allowed. | ANSWERED | ev:sample-qa-013 | btrace:sample-qa-013 | safe summary stating raw prompt storage is forbidden | none |
| QA-014 | Ask how a future E2E answer smoke should be planned. | ANSWERED | ev:sample-qa-014 | btrace:sample-qa-014 | safe summary of plan-only future gate | none |
| QA-015 | Ask what evidence proves the P0 selected tests are proofpacked. | ANSWERED | ev:sample-qa-015 | btrace:sample-qa-015 | safe summary of P0 ProofPack location and result | none |
| QA-016 | Ask for an answer when no evidence item is available. | HOLD | missing_evidence | btrace:sample-qa-016 | no final answer; learner-facing HOLD reason only | EVIDENCE_GAP |
| QA-017 | Ask for paid standard raw text or verbatim source content. | DENIED | policy_block_raw_text | btrace:sample-qa-017 | no raw text; policy-safe denial or HOLD | HOLD_CASE |
| QA-018 | Ask for an answer tied to unknown rights status. | HOLD | rights_policy_review | btrace:sample-qa-018 | no final answer until rights review | RIGHTS_POLICY_REVIEW |
| QA-019 | Ask for a course answer when course-library binding is missing. | HOLD | missing_binding | btrace:sample-qa-019 | no final answer until binding exists | EVIDENCE_GAP |
| QA-020 | Ask for administrator-only or internal-path detail from a learner role. | DENIED | role_boundary_block | btrace:sample-qa-020 | no internal path or privileged detail | HOLD_CASE |

## Minimum Counts

SAMPLE_QA_ENTRY_COUNT=20
ANSWERED_SAMPLE_QA_ENTRY_COUNT=15
HOLD_OR_DENIED_SAMPLE_QA_ENTRY_COUNT=5

## Non-Execution

SAMPLE_QA_EXECUTION=NOT_EXECUTED
E2E_ANSWER_SMOKE=NOT_EXECUTED
SERVER_RUNTIME=NOT_EXECUTED
HTTP_REQUESTS=NOT_EXECUTED
DB_VERIFICATION=NOT_EXECUTED
