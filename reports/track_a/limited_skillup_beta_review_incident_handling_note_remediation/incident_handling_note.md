# Incident Handling Note

R9ZCW_PACKET=T-A1-07SOU_R9ZCW_LIMITED_SKILLUP_BETA_REVIEW_INCIDENT_HANDLING_NOTE_REMEDIATION_PACKET_WITH_LIMITS

INCIDENT_HANDLING_NOTE=AVAILABLE_WITH_LIMITS
INCIDENT_HANDLING_SCOPE=LOCAL_PROOFPACK_BASED_LIMITED_SKILLUP_BETA_EDUCATION_REVIEW_ONLY

## Scope

This note applies only to local proofpack-based limited Skillup beta education
review evidence. It is not live beta operation, runtime/server verification,
production DB verification, external network verification, deployment, release,
Track A PASS, Beta PASS, Release PASS, or Product PASS.

INCIDENT_HANDLING_LIVE_BETA_OPERATION=NOT_EXECUTED
INCIDENT_HANDLING_RUNTIME_SERVER=NOT_EXECUTED
INCIDENT_HANDLING_HTTP_NETWORK=NOT_EXECUTED
INCIDENT_HANDLING_PRODUCTION_DB_ACCESS=NOT_EXECUTED
INCIDENT_HANDLING_EXTERNAL_NETWORK=NOT_EXECUTED
INCIDENT_HANDLING_DEPLOYMENT_RELEASE=NOT_GRANTED

## Incident Trigger Table

| Trigger | Required result |
|---|---|
| raw text leak | INCIDENT_HANDLING_TRIGGER_RAW_LEAK=STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| internal path leak | STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| raw prompt output leak | STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| secret leak | INCIDENT_HANDLING_TRIGGER_SECRET_LEAK=STOP_AND_REJECT |
| instructor-guide raw leak | STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| evidence-free answer | INCIDENT_HANDLING_TRIGGER_EVIDENCE_FREE_ANSWER=STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| Bridge trace bypass | INCIDENT_HANDLING_TRIGGER_BRIDGE_BYPASS=STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| role access leak | INCIDENT_HANDLING_TRIGGER_ROLE_LEAK=STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| unsafe answer without HOLD/DENIED | INCIDENT_HANDLING_TRIGGER_UNSAFE_NO_HOLD=STOP_AND_REJECT_OR_REVIEW_REQUIRED |
| feedback or recovery path missing | STOP_AND_REVIEW_REQUIRED |
| rollback or incident handling missing | STOP_AND_REVIEW_REQUIRED |
| pass escalation | INCIDENT_HANDLING_TRIGGER_PASS_ESCALATION=STOP_AND_REVIEW_REQUIRED_PASS_ESCALATION |
| HTTP/runtime/DB/network needed without separate approval | STOP_AND_REVIEW_REQUIRED |

## Immediate Action Table

| Action | Required handling |
|---|---|
| stop current packet | Stop before further execution or artifact mutation. |
| preserve filename/path-level evidence only | INCIDENT_HANDLING_EVIDENCE_PRESERVATION=LOCAL_PROOFPACK_POINTERS_ONLY |
| secret-like filename found | INCIDENT_HANDLING_SECRET_LIKE_FILES=QUARANTINE_FILENAME_LEVEL_ONLY |
| secret content requested or exposed | INCIDENT_HANDLING_SECRET_CONTENT_INSPECTION=FORBIDDEN |
| raw or secret leak | Return REJECT or REVIEW_REQUIRED by severity. |
| evidence-free answer, Bridge bypass, role leak, unsafe no-HOLD answer | Return REJECT or REVIEW_REQUIRED by severity. |
| pass escalation | Return REVIEW_REQUIRED_PASS_ESCALATION. |
| remediation needed | INCIDENT_HANDLING_REMEDIATION_PATH=SEPARATE_APPROVED_PACKET_REQUIRED |
| owner decision required | INCIDENT_HANDLING_OWNER=HUMAN_REVIEW_REQUIRED_BEFORE_ANY_FURTHER_EXECUTION |
| all PASS/release/deployment tokens | Keep NOT_GRANTED unless separately approved by a later valid packet. |

## Evidence Preservation Rule

Incident evidence preservation is local proofpack pointers only. Do not copy
secret content. Do not export raw text. Do not expose internal paths beyond
already-approved proofpack path references. Do not access production DB. Do not
send HTTP or network requests.

INCIDENT_HANDLING_SECRET_CONTENT_INSPECTION=FORBIDDEN
INCIDENT_HANDLING_SECRET_LIKE_FILES=QUARANTINE_FILENAME_LEVEL_ONLY
INCIDENT_HANDLING_EVIDENCE_PRESERVATION=LOCAL_PROOFPACK_POINTERS_ONLY

## Escalation Rule

Any pass escalation beyond allowed bounded tokens returns
REVIEW_REQUIRED_PASS_ESCALATION.

TRACK_A_PASS=NOT_GRANTED
BETA_PASS=NOT_GRANTED
RELEASE_PASS=NOT_GRANTED
PRODUCT_PASS=NOT_GRANTED
DEPLOYMENT_RELEASE=NOT_GRANTED
