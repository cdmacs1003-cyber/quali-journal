# Operation Stop Conditions

R9ZDF_PACKET=T-A1-07SOU_R9ZDF_LIMITED_SKILLUP_BETA_USE_OPERATION_RUNBOOK_MATERIALIZATION_PACKET_WITH_LIMITS

Future limited Skillup beta use operation must stop when any condition below is observed.

| Stop condition | Required response |
|---|---|
| Evidence-free answer found | REJECT or REVIEW_REQUIRED |
| Bridge bypass found | REJECT or REVIEW_REQUIRED |
| Raw text leak found | REJECT or REVIEW_REQUIRED |
| Internal path leak found | REJECT or REVIEW_REQUIRED |
| Raw prompt output leak found | REJECT or REVIEW_REQUIRED |
| Secret leak found | REJECT |
| Instructor-guide raw leak found | REJECT or REVIEW_REQUIRED |
| Role leak found | REJECT or REVIEW_REQUIRED |
| Unsafe answer without HOLD/DENIED found | REJECT or REVIEW_REQUIRED |
| Missing feedback or recovery path found | REVIEW_REQUIRED |
| Missing incident or rollback path found | REVIEW_REQUIRED |
| Runtime, HTTP, DB, or network needed without separate approval | REVIEW_REQUIRED_RUNTIME_OR_NETWORK_ASSUMPTION |
| Pass escalation found | REVIEW_REQUIRED_PASS_ESCALATION |

EVIDENCE_FREE_ANSWER=FORBIDDEN
BRIDGE_BYPASS=FORBIDDEN
STUDENT_INSTRUCTOR_REVIEWER_ADMIN_BOUNDARY=PRESERVED
SECRET_CONTENT_INSPECTION=FORBIDDEN

All stop-condition evidence must be preserved without exposing raw text, raw prompts, secrets, instructor-guide raw text, or unapproved internal paths.
