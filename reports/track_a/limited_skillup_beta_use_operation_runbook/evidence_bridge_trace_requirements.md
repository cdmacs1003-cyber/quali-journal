# Evidence And Bridge Trace Requirements

R9ZDF_PACKET=T-A1-07SOU_R9ZDF_LIMITED_SKILLUP_BETA_USE_OPERATION_RUNBOOK_MATERIALIZATION_PACKET_WITH_LIMITS

## Evidence Boundary

EVIDENCE_ID_REQUIRED=YES
EVIDENCE_FREE_ANSWER=FORBIDDEN

Answered cases in a future limited Skillup beta use operation must include a non-empty evidence_id or an equivalent evidence pointer. Evidence-free final answers are forbidden.

Evidence pointer requirements:
- evidence_id or equivalent pointer is required for ANSWERED cases
- pointer must reference approved local/proofpack or approved bounded evidence source
- no raw text export is allowed as evidence
- no internal path exposure is allowed beyond approved proofpack path references

## Bridge Trace Boundary

BRIDGE_TRACE_ID_REQUIRED=YES
BRIDGE_BYPASS=FORBIDDEN

Traceable answers must include bridge_trace_id or equivalent bridge trace evidence. Direct Skillup DB lookup outside the Bridge evidence boundary is not allowed.

Bridge trace requirements:
- bridge_trace_id or equivalent trace required when Bridge produced the answer
- no Bridge bypass
- no direct Skillup DB lookup outside Bridge evidence boundary
- missing trace must HOLD or DENIED unless explicitly documented as not traceable with a safe reason

## Stop Conditions

Evidence-free answer found: stop and return REJECT or REVIEW_REQUIRED.
Bridge bypass found: stop and return REJECT or REVIEW_REQUIRED.
Direct DB lookup needed without separate approval: stop and return REVIEW_REQUIRED_RUNTIME_OR_NETWORK_ASSUMPTION.

EVIDENCE_GATE=PASS_WITH_LIMITS_CARRIED_FORWARD
BRIDGE_TRACE_GATE=PASS_WITH_LIMITS_CARRIED_FORWARD
