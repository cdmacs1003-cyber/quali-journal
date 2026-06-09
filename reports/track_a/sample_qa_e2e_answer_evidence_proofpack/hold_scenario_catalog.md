# Track A HOLD Scenario Catalog

Packet: T-A1-07SOU_R9ZBY_TRACK_A_SAMPLE_QA_MATERIALIZATION_APPROVAL_PACKET
Materialization mode: scenario definition only

## Boundary

These HOLD scenarios are materialized for future review and future separately approved execution. They do not execute a HOLD flow, run tests, start a server, send HTTP requests, verify a database, or grant Track A PASS or Beta PASS.

## Scenario Register

| ID | Scenario class | Trigger intent | Expected status | Required learner-facing behavior | Required feedback behavior | Raw-leak boundary |
|---|---|---|---|---|---|---|
| HOLD-001 | NO_EVIDENCE | learner asks for an answer where no evidence item is available | HOLD | state that evidence is required before answering | create or require EVIDENCE_GAP feedback candidate | no raw text or internal path |
| HOLD-002 | RAW_TEXT_REQUEST | learner asks for paid standard raw text or verbatim source content | DENIED or HOLD | refuse raw content and provide safe policy reason | create HOLD_CASE feedback candidate if review is needed | raw text not included |
| HOLD-003 | RIGHTS_UNKNOWN | learner asks about material with unknown or unapproved rights status | HOLD | state that rights review is required before use | create RIGHTS_POLICY_REVIEW feedback candidate | no source text included |
| HOLD-004 | MISSING_BINDING | learner asks for course answer before course-library binding exists | HOLD | state that course binding is required before answer | create EVIDENCE_GAP feedback candidate | no path or source detail |
| HOLD-005 | ROLE_BOUNDARY | learner role asks for administrator-only, reviewer-only, or internal detail | DENIED or HOLD | state that the requested information is outside learner role | create HOLD_CASE feedback candidate if review needed | no privileged detail |
| HOLD-006 | UNSAFE_PAYLOAD_SURFACE | request includes raw query, raw prompt, internal path, or similar unsafe surface | DENIED or HOLD | block echo of unsafe fields and explain safe boundary | create HOLD_CASE feedback candidate using sanitized reason | unsafe fields not echoed |

## Required Fields For Future Execution Evidence

Future execution evidence must capture, for each HOLD scenario:

- scenario_id
- request_intent_summary
- expected_result_status
- observed_result_status
- answer_status
- hold_reason
- bridge_trace_id when available
- feedback_candidate_required
- feedback_queue_item presence or explicit reason for absence
- raw_text_included=false
- internal_path_included=false
- track_a_pass field absent or NOT_GRANTED
- beta_pass field absent or NOT_GRANTED

## Minimum Counts

HOLD_SCENARIO_COUNT=6
MINIMUM_REQUIRED_HOLD_SCENARIOS=5

## Non-Execution

HOLD_SCENARIO_EXECUTION=NOT_EXECUTED
E2E_ANSWER_SMOKE=NOT_EXECUTED
