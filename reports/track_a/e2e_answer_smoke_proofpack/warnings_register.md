# Warnings Register

Packet: T-A1-07SOU_R9ZCD_TRACK_A_E2E_ANSWER_SMOKE_PROOFPACK_MATERIALIZATION_APPROVAL_PACKET
Evidence source: R9ZCB terminal/session evidence

## Warning Summary

warnings=Starlette/Pydantic deprecation warnings only
bounded_e2e_answer_smoke_result=12 passed, 5 warnings

## Warning Classes

| Warning class | Source class | Handling |
|---|---|---|
| PendingDeprecationWarning | Starlette multipart import path | non-blocking for bounded local E2E answer smoke |
| PydanticDeprecatedSince20 / DeprecationWarning | Pydantic class-based config | non-blocking for bounded local E2E answer smoke |

## Handling

The warnings did not block the bounded local E2E answer smoke result.
The warnings should be tracked for future dependency hygiene.
R9ZCD does not claim that these warnings are fixed.
R9ZCD does not approve dependency changes.
R9ZCD does not approve lint, build, full regression, deployment, or release.
