# Feedback Capture Contract

Packet: T-A1-07SOU_R9ZBY_TRACK_A_SAMPLE_QA_MATERIALIZATION_APPROVAL_PACKET

## Purpose

Define how future answer and HOLD evidence must prove that feedback can be captured without raw leak, database verification, or pass escalation in this packet.

## Required Feedback Behaviors

| Source condition | Expected feedback behavior |
|---|---|
| no evidence | EVIDENCE_GAP feedback candidate or queue item |
| raw text request | HOLD_CASE feedback candidate or queue item with sanitized reason |
| unknown rights | RIGHTS_POLICY_REVIEW feedback candidate or queue item |
| missing course binding | EVIDENCE_GAP feedback candidate or queue item |
| role boundary request | HOLD_CASE feedback candidate or queue item when review is needed |

## Required Feedback Fields For Future Execution

Future execution proof should capture:

- feedback_id
- feedback_type
- linked_evidence_id or missing_evidence marker
- bridge_trace_id when available
- origin_module=Skillup or equivalent safe module name
- result_status=HOLD for HOLD feedback
- dedup_key
- created_at or deterministic test timestamp
- raw_text_included=false
- internal_path_included=false

## Prohibited Feedback Surfaces

Feedback evidence must not include:

- raw_text
- raw_prompt
- raw_query
- raw_answer
- internal_path
- local_route
- secret-like values
- Track A PASS escalation
- Beta PASS escalation

## Non-Execution

FEEDBACK_CAPTURE_EXECUTION=NOT_EXECUTED
PERSISTENT_FEEDBACK_LOOP=NOT_VERIFIED
DB_VERIFICATION=NOT_EXECUTED
