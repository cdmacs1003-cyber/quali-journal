# HOLD And DENIED Policy

R9ZDF_PACKET=T-A1-07SOU_R9ZDF_LIMITED_SKILLUP_BETA_USE_OPERATION_RUNBOOK_MATERIALIZATION_PACKET_WITH_LIMITS

Future limited Skillup beta use must prefer HOLD or DENIED over unsafe final answers when evidence, rights, binding, role, safety, or raw export risk is present.

MISSING_EVIDENCE_ACTION=HOLD_OR_DENIED
RIGHTS_RISK_ACTION=HOLD_OR_DENIED
ROLE_RISK_ACTION=HOLD_OR_DENIED
UNSAFE_ANSWER_RISK_ACTION=HOLD_OR_DENIED
RAW_EXPORT_RISK_ACTION=HOLD_OR_DENIED

## Required HOLD / DENIED Outcomes

| Condition | Required action |
|---|---|
| Missing evidence | HOLD or DENIED |
| Rights risk | HOLD or DENIED |
| Binding risk | HOLD or DENIED |
| Role risk | HOLD or DENIED |
| Unsafe answer risk | HOLD or DENIED |
| Raw export risk | HOLD or DENIED |
| Missing bridge trace for traceable answer | HOLD or DENIED |
| Feedback recovery path missing | HOLD or REVIEW_REQUIRED |

Unsafe answer without HOLD or DENIED is forbidden.

HOLD_GATE=PASS_WITH_LIMITS_CARRIED_FORWARD

## Stop Conditions

If a future operation finds unsafe answer without HOLD/DENIED, evidence-free answer, role-risk answer, rights-risk answer, or raw-export-risk answer, the operation must stop and return REVIEW_REQUIRED or REJECT according to severity.
